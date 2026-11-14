"""Case-local wrappers around RELIC scientific functors.

These wrappers keep RELIC core untouched while declaring tutorial-specific
dependency topology and trainable hooks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from dataclasses import replace

import torch

from relic.core.state import UpdateItem
from relic.scientific.hydrosphere.runoff.surface import SurfaceRunoff
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
from relic.scientific.biosphere.bvoc.megan_model import MEGANModel
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.biosphere.stress.ozone_stress import OzoneStress
from relic.scientific.biosphere.stress.thermal_stress import ThermalStress


class CaseThermalStress(ThermalStress):
    """Thermal stress with case-local dependency declaration."""

    def get_dependencies(self) -> list[str]:
        # The point case supplies forcing from the StateManager: no static
        # provider; instance declarations are still merged by FunctorBase.
        return self._merge_declared_dependencies([])


class CaseOzoneStress(OzoneStress):
    """Ozone stress with case-local dependency declaration."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The point case has no chemistry provider and no Farquhar-stomatal
        # edge ahead of ozone: rebind the parent's default providers
        # (instance-level declarations) to the case topology.
        self.set_dependencies(["thermal_stress"])


class CasePenmanMonteithET(PenmanMonteithET):
    """Penman-Monteith ET with point-case dependency declarations."""

    def get_dependencies(self) -> dict[str, list[str]]:
        return self._merge_declared_dependencies(
            {
                "required": ["atmospheric_turbulence"],
                "lagged": ["bucket_soil_moisture", "richards_soil_moisture"],
            }
        )


class CaseBucketSoilMoisture(BucketSoilMoisture):
    """Bucket hydrology with case-local semi-explicit coupling topology.

    References:
    - Manabe (1969), single-layer bucket water balance.
    - Allen et al. (1998), FAO Penman-Monteith evapotranspiration.
    Mathematical expressions:
    - ET_n = PenmanMonteith(theta_n, meteorology_n).
    - theta_(n+1) = theta_n + (P_n - ET_n - Q_n - D_n) * dt / (rho_w * Z).
    Parameter meanings:
    - theta is volumetric soil moisture [m3 m-3].
    - P, ET, Q, and D are water mass fluxes [kg m-2 s-1].
    - rho_w is water density [kg m-3], and Z is soil depth [m].
    This wrapper changes only scheduling; the parent owns every calculation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metadata.input_variables = [
            "soil_moisture",
            "infiltration_flux",
            "evapotranspiration_flux",
        ]

    def get_dependencies(self) -> dict[str, list[str]]:
        return self._merge_declared_dependencies(
            {
                "required": ["surface_runoff", "penman_monteith_et"],
                "optional": [],
                "lagged": [],
            }
        )

    def forward(self, state, time: float, dt: float, **kwargs):
        # References:
        # - SCS (1972), precipitation partition into runoff and infiltration.
        # - Manabe (1969), single-layer bucket water balance.
        # Mathematical expressions:
        # - I_n = P_effective_n - Q_n.
        # - theta_(n+1) = theta_n + (I_n - ET_n - D_n) * dt / (rho_w * Z).
        # Parameter meanings:
        # - I is infiltration_flux [kg m-2 s-1].
        # - Q is surface runoff [kg m-2 s-1].
        # - ET is evapotranspiration [kg m-2 s-1].
        infiltration = state.get("infiltration_flux")
        if infiltration is None:
            raise ValueError(
                "CaseBucketSoilMoisture requires canonical infiltration_flux."
            )
        remapped_state = dict(state)
        remapped_state["precipitation_flux"] = infiltration
        return super().forward(remapped_state, time, dt, **kwargs)


class CaseFarquharPhotosynthesis(FarquharPhotosynthesis):
    """Farquhar photosynthesis with case-local dependency declaration."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metadata.output_variables = [
            key
            for key in list(getattr(self.metadata, "output_variables", []) or [])
            if key != "ozone_stress_factor"
        ]
        # Case topology: forcing and PPFD come from the StateManager, so the
        # parent's default providers (instance-level declarations) are rebound
        # to the case-local water-stress adapter.
        self.set_dependencies(["water_stress"])

    def forward(self, state, time: float, dt: float, **kwargs):
        # References:
        # - Farquhar et al. (1980), core photosynthesis formulation remains in parent class.
        # Mathematical expression:
        # - Wrapper post-process: drop duplicate ozone_stress_factor declaration.
        # Parameter meanings:
        # - No new scientific parameters introduced in this wrapper.
        outputs = super().forward(state, time, dt, **kwargs)
        outputs.pop("ozone_stress_factor", None)
        return outputs


class CaseSurfaceRunoff(SurfaceRunoff):
    """Surface runoff wrapper with case-local dependency and forcing-key mapping."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metadata.input_variables = ["precipitation_flux", "soil_moisture"]
        declared_outputs = list(getattr(self.metadata, "output_variables", []) or [])
        filtered = [key for key in declared_outputs if key != "surface_runoff_flux"]
        filtered.append("surface_runoff_flux_external")
        self.metadata.output_variables = filtered

    def get_dependencies(self) -> list[str]:
        # No static provider in the point case; the instance declaration
        # (set by the case composition) is merged by FunctorBase.
        return self._merge_declared_dependencies([])

    def forward(self, state, time: float, dt: float, **kwargs):
        # References:
        # - Horton (1933), infiltration-excess runoff partitioning.
        # - SCS (1972), runoff/infiltration separation under event forcing.
        # Mathematical expression:
        # - P_runoff_input := P_throughfall when throughfall is available.
        # - theta_surface := scalar(theta_case) only when theta_case has one value.
        # Parameter meanings:
        # - P_throughfall is precipitation_flux_effective after canopy interception [kg m-2 s-1].
        # - theta_case is the single tutorial grid-cell soil moisture [m3 m-3].
        effective_precip = state.get("precipitation_flux_effective")
        remapped_state = dict(state)
        soil_moisture = remapped_state.get("soil_moisture")
        if (
            isinstance(soil_moisture, torch.Tensor)
            and soil_moisture.ndim > 1
            and soil_moisture.numel() == 1
        ):
            remapped_state["soil_moisture"] = soil_moisture.reshape(())
        if effective_precip is not None:
            remapped_state["precipitation_flux"] = effective_precip
        outputs = super().forward(remapped_state, time, dt, **kwargs)
        # References:
        # - RELIC single-writer policy for governed overlap variables.
        # Mathematical expression:
        # - q_surface_external := q_surface_raw
        # Parameter meanings:
        # - q_surface_raw from SurfaceRunoff [kg m-2 s-1].
        raw_surface_flux = outputs.pop("surface_runoff_flux", None)
        if raw_surface_flux is not None:
            outputs["surface_runoff_flux_external"] = raw_surface_flux
        return outputs


class CaseMEGANModel(MEGANModel):
    """MEGAN wrapper with case-local trainable isoprene EF, light, and temperature parameters.

    The isoprene E_opt and C_T1 are isoprene Table 4 coefficients; T_opt and C_T2 are
    class independent (Guenther et al. 2012 Eq. 9 and Sect. 2.2), so the calibrated
    values act on the temperature activity of every compound class.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def case_isoprene_ef(self) -> Optional[torch.nn.Parameter]:
        value = getattr(self, "_case_isoprene_ef", None)
        return value if isinstance(value, torch.nn.Parameter) else None

    @property
    def case_alpha_light(self) -> Optional[torch.nn.Parameter]:
        return self.alpha_light if isinstance(self.alpha_light, torch.nn.Parameter) else None

    @property
    def case_cl_light(self) -> Optional[torch.nn.Parameter]:
        return self.cl_light if isinstance(self.cl_light, torch.nn.Parameter) else None

    @property
    def case_eopt_isoprene(self) -> Optional[torch.nn.Parameter]:
        return self.eopt_isoprene if isinstance(self.eopt_isoprene, torch.nn.Parameter) else None

    @property
    def case_topt(self) -> Optional[torch.nn.Parameter]:
        return self.topt if isinstance(self.topt, torch.nn.Parameter) else None

    @property
    def case_ct1_isoprene(self) -> Optional[torch.nn.Parameter]:
        return self.ct1_iso if isinstance(self.ct1_iso, torch.nn.Parameter) else None

    @property
    def case_ct2(self) -> Optional[torch.nn.Parameter]:
        return self.ct2 if isinstance(self.ct2, torch.nn.Parameter) else None

    def _promote_cached_tensor(self, attribute: str, value: object) -> torch.nn.Parameter:
        """Replace one parent runtime buffer with a trainable tensor."""
        cached = getattr(self, attribute)
        if not isinstance(cached, torch.Tensor):
            raise TypeError(f"MEGAN runtime tensor is unavailable: {attribute}")
        parameter = torch.nn.Parameter(
            torch.as_tensor(value, dtype=cached.dtype, device=cached.device)
            .detach()
            .clone()
            .reshape_as(cached)
        )
        setattr(self, attribute, parameter)
        return parameter

    def get_dependencies(self) -> list[str]:
        return self._merge_declared_dependencies(
            ["farquhar_photosynthesis", "water_stress"]
        )

    def initialize_case_trainables(self) -> Dict[str, torch.nn.Parameter]:
        isoprene_init = self.get_parameter_value(
            "default.isoprene", namespace="biosphere.bvoc", required=True
        )
        alpha_init = self.get_parameter_value(
            "alpha_light", namespace="biosphere.bvoc.megan", required=True
        )
        cl_init = self.get_parameter_value(
            "cl_light", namespace="biosphere.bvoc.megan", required=True
        )
        eopt_init = self.get_parameter_value(
            "eopt_isoprene", namespace="biosphere.bvoc.megan", required=True
        )
        topt_init = self.get_parameter_value(
            "topt", namespace="biosphere.bvoc.megan", required=True
        )
        ct1_init = self.get_parameter_value(
            "ct1_isoprene", namespace="biosphere.bvoc.megan", required=True
        )
        ct2_init = self.get_parameter_value(
            "ct2", namespace="biosphere.bvoc.megan", required=True
        )

        self._case_isoprene_ef = torch.nn.Parameter(
            torch.as_tensor(isoprene_init, dtype=torch.float32, device=self.device)
            .detach()
            .clone()
            .reshape(())
        )
        self.emission_factors["isoprene"] = self._case_isoprene_ef
        alpha = self._promote_cached_tensor("alpha_light", alpha_init)
        cl = self._promote_cached_tensor("cl_light", cl_init)
        eopt = self._promote_cached_tensor("eopt_isoprene", eopt_init)
        topt = self._promote_cached_tensor("topt", topt_init)
        ct1 = self._promote_cached_tensor("ct1_iso", ct1_init)
        ct2 = self._promote_cached_tensor("ct2", ct2_init)
        return {
            "biosphere.bvoc.default.isoprene": self._case_isoprene_ef,
            "biosphere.bvoc.megan.alpha_light": alpha,
            "biosphere.bvoc.megan.cl_light": cl,
            "biosphere.bvoc.megan.eopt_isoprene": eopt,
            "biosphere.bvoc.megan.topt": topt,
            "biosphere.bvoc.megan.ct1_isoprene": ct1,
            "biosphere.bvoc.megan.ct2": ct2,
        }

    def forward(self, state, time: float, dt: float, **kwargs: Any):
        # References:
        # - Guenther et al. (2012), MEGAN activity factors retain the mechanistic BVOC backbone.
        # - Reichstein et al. (2019), ML corrections should modify uncertain sub-processes without replacing the full physics chain.
        # Mathematical expression:
        # - E_iso,hybrid = E_iso,physics * f_LT,ML
        # Parameter meanings:
        # - E_iso,physics: mechanistic isoprene emission from MEGAN [ug m-2 s-1 or project-native units]
        # - f_LT,ML: multiplicative correction factor for combined light-temperature activity [-]
        outputs = super().forward(state, time, dt, **kwargs)
        correction = state.get("isoprene_activity_correction_ML")
        emission_update = outputs.get("isoprene_emission")
        emission = (
            emission_update.value
            if isinstance(emission_update, UpdateItem)
            else emission_update
        )
        if not isinstance(correction, torch.Tensor) or not isinstance(emission, torch.Tensor):
            return outputs

        correction_tensor = correction.to(device=emission.device, dtype=emission.dtype)
        if correction_tensor.shape != emission.shape:
            if correction_tensor.numel() == 1:
                correction_tensor = correction_tensor.reshape(1).expand_as(emission)
            elif correction_tensor.numel() == emission.numel():
                correction_tensor = correction_tensor.reshape_as(emission)
            else:
                correction_tensor = correction_tensor.mean().reshape(1).expand_as(emission)

        corrected_emission = emission * correction_tensor
        outputs["isoprene_emission"] = (
            replace(emission_update, value=corrected_emission)
            if isinstance(emission_update, UpdateItem)
            else corrected_emission
        )
        outputs["isoprene_activity_correction_applied"] = correction_tensor

        total_bvoc_update = outputs.get("total_bvoc_emission")
        total_bvoc = (
            total_bvoc_update.value
            if isinstance(total_bvoc_update, UpdateItem)
            else total_bvoc_update
        )
        if isinstance(total_bvoc, torch.Tensor):
            corrected_total_bvoc = total_bvoc - emission + corrected_emission
            outputs["total_bvoc_emission"] = (
                replace(total_bvoc_update, value=corrected_total_bvoc)
                if isinstance(total_bvoc_update, UpdateItem)
                else corrected_total_bvoc
            )

        total_carbon_update = outputs.get("total_bvoc_carbon_flux")
        total_carbon_flux = (
            total_carbon_update.value
            if isinstance(total_carbon_update, UpdateItem)
            else total_carbon_update
        )
        if isinstance(total_carbon_flux, torch.Tensor):
            carbon_fraction = self._get_carbon_fraction("isoprene")
            if not isinstance(carbon_fraction, torch.Tensor):
                carbon_fraction = torch.as_tensor(
                    carbon_fraction,
                    dtype=emission.dtype,
                    device=emission.device,
                )
            else:
                carbon_fraction = carbon_fraction.to(dtype=emission.dtype, device=emission.device)
            micrograms_to_grams = self.micrograms_to_grams
            if not isinstance(micrograms_to_grams, torch.Tensor):
                micrograms_to_grams = torch.as_tensor(
                    micrograms_to_grams,
                    dtype=emission.dtype,
                    device=emission.device,
                )
            else:
                micrograms_to_grams = micrograms_to_grams.to(
                    dtype=emission.dtype,
                    device=emission.device,
                )
            old_isoprene_carbon = emission * carbon_fraction * micrograms_to_grams
            new_isoprene_carbon = corrected_emission * carbon_fraction * micrograms_to_grams
            corrected_carbon_flux = (
                total_carbon_flux - old_isoprene_carbon + new_isoprene_carbon
            )
            outputs["total_bvoc_carbon_flux"] = (
                replace(total_carbon_update, value=corrected_carbon_flux)
                if isinstance(total_carbon_update, UpdateItem)
                else corrected_carbon_flux
            )

        return outputs
