"""NetCDF loader utilities for the Tapajos single-site tutorial case."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant


OBS_KEY_MAP = {
    "soil_moisture_obs": "soil_moisture",
    "soil_moisture": "soil_moisture",
    "net_photosynthesis_obs": "net_photosynthesis",
    "net_photosynthesis": "net_photosynthesis",
    "isoprene_emission_obs": "isoprene_emission",
    "isoprene_emission": "isoprene_emission",
}


@dataclass(frozen=True)
class LoaderConfig:
    case_root: Path
    forcing_path: Path
    obs_path: Path
    case_parameter_path: Path
    device: torch.device


class CaseNetCDFLoader:
    """Load processed case NetCDF assets and expose step-wise tensors."""

    def __init__(self, config: LoaderConfig) -> None:
        self.config = config
        self.device = config.device
        self.forcing_ds = xr.load_dataset(config.forcing_path)
        self.obs_ds = xr.load_dataset(config.obs_path)
        self.time_index = pd.DatetimeIndex(pd.to_datetime(self.forcing_ds["time"].values))
        self._observation_quality = self._load_observation_quality(config.case_root)

        with config.case_parameter_path.open("r", encoding="utf-8") as handle:
            self.case_parameters = yaml.safe_load(handle)
        if not isinstance(self.case_parameters, dict):
            raise ValueError("case_parameters.yaml must contain a mapping.")

        self._forcing_cache: Dict[str, torch.Tensor] = {}
        for var_name in self.forcing_ds.data_vars:
            values = np.asarray(self.forcing_ds[var_name].values, dtype=np.float32)
            self._forcing_cache[var_name] = torch.as_tensor(values, dtype=torch.float32, device=self.device)

        self._obs_cache: Dict[str, torch.Tensor] = {}
        for var_name in self.obs_ds.data_vars:
            values = np.asarray(self.obs_ds[var_name].values, dtype=np.float32)
            self._obs_cache[var_name] = torch.as_tensor(values, dtype=torch.float32, device=self.device)

        self._leaf_area_index = self._read_case_value("case.biosphere.leaf_area_index")
        self._reference_height = self._read_case_value("case.site.reference_height")
        self._canopy_height = self._read_case_value("case.site.canopy_height")
        self._initial_soil_moisture = self._read_case_value("case.soil.initial_moisture")
        self._initial_stomatal = self._read_case_value("case.biosphere.stomatal_conductance_initial")
        self._umol_to_nmol = self._read_case_value("case.conversions.umol_to_nmol")
        manager = get_global_manager()
        self._roughness_length_momentum = float(
            manager.get(
                "coupling.scales.biosphere_atmosphere.roughness_length_momentum",
                required=True,
            )
        )
        self._zero_displacement_height_fraction = float(
            manager.get(
                "coupling.scales.biosphere_atmosphere.zero_displacement_height_fraction",
                required=True,
            )
        )
        # References:
        # - Garratt (1992), canopy displacement in surface-layer similarity theory.
        # - Massman (1997), vegetation roughness and displacement parameterization.
        # Mathematical expression:
        # - d = f_d * h_c
        # Parameter meanings:
        # - d: zero-plane displacement height [m]
        # - f_d: ParameterManager-backed displacement fraction [dimensionless]
        # - h_c: case canopy height [m]
        self._displacement_height = (
            self._zero_displacement_height_fraction * self._canopy_height
        )
        self._isoprene_molecular_weight = float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.isoprene"
            )
        )
        # References:
        # - Guenther et al. (2012), MEGAN mass-flux emission outputs for isoprene.
        # - PTR-MS tower observations often reported as molar flux.
        # Mathematical expression:
        # - E_ug = E_nmol * (MW / umol_to_nmol)
        # Parameter meanings:
        # - MW: isoprene molecular weight [g mol-1], numerically equal to [ug umol-1]
        # - umol_to_nmol: conversion factor [nmol umol-1]
        self._isoprene_obs_nmol_to_ug = self._isoprene_molecular_weight / self._umol_to_nmol
        # References:
        # - Guenther et al. (2012), MEGAN species emissions are molar fluxes.
        # Mathematical expression:
        # - E_ug = E_mol * MW / c_ug_to_g
        # Parameter meanings:
        # - E_mol: modeled isoprene molar flux [mol m-2 s-1]
        # - MW: isoprene molecular weight [g mol-1]
        # - c_ug_to_g: micrograms-to-grams conversion [g ug-1]
        micrograms_to_grams = float(
            get_physical_constant(
                "relic.physical.constants.conversions.micrograms_to_grams"
            )
        )
        self._isoprene_model_mol_to_ug = (
            self._isoprene_molecular_weight / micrograms_to_grams
        )
        self._farquhar_vcmax_scaling_default = self._read_case_value("tutorial.case.farquhar.vcmax_scaling_default")
        self._farquhar_jmax_scaling_default = self._read_case_value("tutorial.case.farquhar.jmax_scaling_default")
        self._farquhar_leaf_water_potential_default = self._read_case_value(
            "tutorial.case.farquhar.leaf_water_potential_default"
        )
        self._penman_surface_resistance_default = self._read_case_value(
            "tutorial.case.penman.surface_resistance_default"
        )
        self._penman_canopy_conductance_default = self._read_case_value(
            "tutorial.case.penman.canopy_conductance_default"
        )
        self.par_effective_scale = torch.nn.Parameter(
            torch.tensor(
                self._read_case_value("tutorial.case.radiation.par_effective_scale"),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.initial_soil_moisture = torch.nn.Parameter(
            torch.tensor(
                self._read_case_value("tutorial.case.state_init.soil_moisture"),
                dtype=torch.float32,
                device=self.device,
            )
        )

    def _load_observation_quality(self, case_root: Path) -> Dict[str, Dict[str, Any]]:
        provenance_path = case_root / "data" / "processed" / "provenance.json"
        if not provenance_path.exists():
            return {}
        try:
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        processing = payload.get("processing")
        if not isinstance(processing, dict):
            return {}
        quality = processing.get("observation_quality")
        if not isinstance(quality, dict):
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        for key, value in quality.items():
            if not isinstance(value, dict):
                continue
            result[str(key)] = dict(value)
        return result

    def _read_case_value(self, key: str) -> float:
        raw = None
        parameter_mapping = self.case_parameters.get("parameters")
        if isinstance(parameter_mapping, dict):
            raw = parameter_mapping.get(key)
        if raw is None:
            raw = self.case_parameters.get(key)
        if not isinstance(raw, dict):
            raise KeyError(f"Missing case parameter: {key}")
        value = raw.get("value")
        if value is None:
            raise KeyError(f"Missing case parameter value: {key}")
        return float(value)

    @property
    def total_steps(self) -> int:
        return int(len(self.time_index))

    def get_time(self, step: int) -> pd.Timestamp:
        return pd.Timestamp(self.time_index[step])

    def _scalar_step(self, tensor_1d: torch.Tensor, step: int) -> torch.Tensor:
        return tensor_1d[step : step + 1].reshape(1, 1)

    def get_forcing_step(self, step: int) -> Dict[str, torch.Tensor]:
        forcing = {
            name: self._scalar_step(values, step)
            for name, values in self._forcing_cache.items()
        }

        # References:
        # - `par` is the copy of the scaled ppfd photon flux [umol m-2 s-1].
        # - Penman-Monteith and bucket hydrology require precipitation and pressure aliases.
        # Mathematical expression:
        # - par := ppfd
        # - precipitation := precipitation_flux
        # Parameter meanings:
        # - All aliases keep identical physical units to their source variables.
        if "ppfd" in forcing:
            scale = self.par_effective_scale.to(dtype=forcing["ppfd"].dtype, device=forcing["ppfd"].device)
            forcing["ppfd"] = forcing["ppfd"] * scale
            forcing["par"] = forcing["ppfd"]
        if "surface_pressure" in forcing:
            forcing["atmospheric_pressure"] = forcing["surface_pressure"]
        if "precipitation_flux" in forcing:
            forcing["precipitation"] = forcing["precipitation_flux"]
        if "co2_concentration" in forcing:
            forcing["co2_intercellular"] = forcing["co2_concentration"]
            forcing["atmospheric_co2_ppm"] = forcing["co2_concentration"]
        # o3_mixing_ratio [mol mol-1] and vapor_pressure_deficit [Pa] come from the forcing
        # file under their contract names; no copy is made.
        if "air_temperature" in forcing:
            forcing["leaf_temperature"] = forcing["air_temperature"]
            forcing["temperature"] = forcing["air_temperature"]
            forcing["surface_temperature"] = forcing["air_temperature"]

        if "leaf_area_index" not in forcing:
            forcing["leaf_area_index"] = torch.full((1, 1), self._leaf_area_index, dtype=torch.float32, device=self.device)
        forcing["reference_height"] = torch.full((1, 1), self._reference_height, dtype=torch.float32, device=self.device)
        forcing["canopy_height"] = torch.full((1, 1), self._canopy_height, dtype=torch.float32, device=self.device)
        forcing["roughness_length_momentum"] = torch.full(
            (1, 1),
            self._roughness_length_momentum,
            dtype=torch.float32,
            device=self.device,
        )
        forcing["displacement_height"] = torch.full(
            (1, 1),
            self._displacement_height,
            dtype=torch.float32,
            device=self.device,
        )
        forcing["vcmax_scaling"] = torch.full(
            (1, 1), self._farquhar_vcmax_scaling_default, dtype=torch.float32, device=self.device
        )
        forcing["jmax_scaling"] = torch.full(
            (1, 1), self._farquhar_jmax_scaling_default, dtype=torch.float32, device=self.device
        )
        forcing["leaf_water_potential"] = torch.full(
            (1, 1), self._farquhar_leaf_water_potential_default, dtype=torch.float32, device=self.device
        )
        forcing["surface_resistance"] = torch.full(
            (1, 1), self._penman_surface_resistance_default, dtype=torch.float32, device=self.device
        )
        forcing["canopy_conductance"] = torch.full(
            (1, 1), self._penman_canopy_conductance_default, dtype=torch.float32, device=self.device
        )

        return forcing

    def get_obs_step(self, step: int, obs_key: Optional[str]) -> Optional[torch.Tensor]:
        if obs_key is None:
            return None
        mapped = OBS_KEY_MAP.get(obs_key, obs_key)
        if mapped not in self._obs_cache:
            return None
        obs_value = self._scalar_step(self._obs_cache[mapped], step)
        if mapped == "isoprene_emission":
            scale = torch.tensor(
                self._isoprene_obs_nmol_to_ug,
                dtype=obs_value.dtype,
                device=obs_value.device,
            )
            return obs_value * scale
        return obs_value

    def convert_model_isoprene_to_observation_units(
        self,
        modeled_molar_flux: torch.Tensor,
    ) -> torch.Tensor:
        """Convert modeled isoprene from molar flux to ug m-2 s-1."""
        scale = torch.as_tensor(
            self._isoprene_model_mol_to_ug,
            dtype=modeled_molar_flux.dtype,
            device=modeled_molar_flux.device,
        )
        return modeled_molar_flux * scale

    def get_obs_quality_info(self, obs_key: Optional[str]) -> Dict[str, Any]:
        if obs_key is None:
            return {}
        mapped = OBS_KEY_MAP.get(obs_key, obs_key)
        payload = self._observation_quality.get(mapped)
        if not isinstance(payload, dict):
            return {}
        return payload

    def get_obs_quality_label(self, obs_key: Optional[str]) -> str:
        info = self.get_obs_quality_info(obs_key)
        value = info.get("class")
        if not isinstance(value, str) or len(value.strip()) == 0:
            return "unknown"
        return value.strip()

    def get_obs_weight_multiplier(self, obs_key: Optional[str]) -> float:
        info = self.get_obs_quality_info(obs_key)
        raw_value = info.get("recommended_weight_multiplier")
        if raw_value is None:
            return 1.0
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(value):
            return 1.0
        return max(0.0, value)

    def make_initial_state(self) -> Dict[str, torch.Tensor]:
        forcing0 = self.get_forcing_step(0)
        initial_state = dict(forcing0)
        # References:
        # - Rodriguez-Iturbe and Porporato (2004), sensitivity of transient soil moisture to initial condition.
        # Mathematical expression:
        # - theta(t=0) = theta_init
        # Parameter meanings:
        # - theta_init: initial volumetric soil moisture [m3 m-3]
        soil_init = self.initial_soil_moisture.to(dtype=torch.float32, device=self.device).reshape(1, 1)
        initial_state["soil_moisture"] = soil_init
        initial_state["stomatal_conductance"] = torch.full(
            (1, 1),
            self._initial_stomatal,
            dtype=torch.float32,
            device=self.device,
        )
        initial_state["canopy_storage"] = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        return initial_state
