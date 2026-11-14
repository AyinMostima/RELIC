"""Unit system and output conversion configuration."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from ...data.standards.unit_converter import ConversionContext, unit_converter
from .definitions import (
    STANDARD_PRESSURE as _STANDARD_PRESSURE,
    STANDARD_TEMPERATURE as _STANDARD_TEMPERATURE,
    UnitCategory,
)

logger = logging.getLogger(__name__)


class UnitSystem:
    """
    CF-1.8 compliant unit system for RELIC.

    All units follow the Climate and Forecast (CF) metadata conventions
    and International System of Units (SI) where applicable.

    References:
        CF Conventions: http://cfconventions.org/
        SI Units: https://www.bipm.org/en/measurement-units/
    """

    # Temperature (always Kelvin for internal calculations)
    TEMPERATURE = "K"  # Kelvin
    TEMPERATURE_DIFFERENCE = "K"  # Temperature differences

    # Concentration units
    MIXING_RATIO = "mol mol-1"  # Mole fraction (dimensionless)
    MASS_CONCENTRATION = "kg m-3"  # Mass per volume
    MOLAR_CONCENTRATION = "mol m-3"  # Moles per volume
    MASS_MIXING_RATIO = "kg kg-1"  # Mass fraction (dimensionless)

    # Flux units (per unit area and time)
    CARBON_FLUX = "kg C m-2 s-1"  # Carbon flux
    NITROGEN_FLUX = "kg N m-2 s-1"  # Nitrogen flux
    WATER_FLUX = "kg m-2 s-1"  # Water flux (equivalent to mm/s when density=1000)
    ENERGY_FLUX = "W m-2"  # Energy flux (J m-2 s-1)
    PHOTON_FLUX = "mol m-2 s-1"  # Photon flux (PAR, PPFD)

    # Mass flux (not per area)
    MASS_FLUX = "kg s-1"  # Mass flow rate

    # Time
    TIME = "s"  # Seconds (base unit)

    # Pressure
    PRESSURE = "Pa"  # Pascal
    VAPOR_PRESSURE = "Pa"  # Vapor pressure

    # Length
    LENGTH = "m"  # Meter
    HEIGHT = "m"  # Height/depth

    # Area and Volume
    AREA = "m2"  # Square meter
    VOLUME = "m3"  # Cubic meter

    # Mass
    MASS = "kg"  # Kilogram
    CARBON_MASS = "kg C"  # Carbon mass
    NITROGEN_MASS = "kg N"  # Nitrogen mass

    # Energy
    ENERGY = "J"  # Joule
    SPECIFIC_ENERGY = "J kg-1"  # Specific energy

    # Other derived units
    DENSITY = "kg m-3"  # Density
    VELOCITY = "m s-1"  # Velocity
    ACCELERATION = "m s-2"  # Acceleration
    DIFFUSIVITY = "m2 s-1"  # Diffusion coefficient
    CONDUCTIVITY = "W m-1 K-1"  # Thermal conductivity
    SPECIFIC_HEAT = "J kg-1 K-1"  # Specific heat capacity

    # Dimensionless
    FRACTION = "1"  # Dimensionless fraction (0-1)
    PERCENT = "%"  # Percentage (0-100)
    PPM = "ppm"  # Parts per million (1e-6)
    PPB = "ppb"  # Parts per billion (1e-9)

    # Standard conditions
    STANDARD_TEMPERATURE = _STANDARD_TEMPERATURE
    STANDARD_PRESSURE = _STANDARD_PRESSURE

    # Conversion factors are handled by data.standards.unit_converter.
    CONVERSIONS: Dict[Tuple[str, str], float] = {}

    _contract_unit_map: Optional[Dict[str, str]] = None
    _contract_unit_map_lower: Optional[Dict[str, str]] = None

    @classmethod
    def _load_unit_mappings_from_contracts(cls) -> Dict[str, str]:
        """Index the packaged contract units by variable name and alias.

        The contracts are read only through the single contract loader, which
        raises on a missing directory or a malformed contract; this index is a
        derived view of that registry.
        """
        if cls._contract_unit_map is not None:
            return cls._contract_unit_map
        from ...contracts.contract_loader import load_contract_registry

        registry = load_contract_registry()
        mappings: Dict[str, str] = {}
        for var in registry.list_variables():
            contract = registry.get_contract(var)
            if contract and contract.udunits:
                mappings[var] = contract.udunits
                for alias in contract.aliases:
                    if alias and alias not in mappings:
                        mappings[alias] = contract.udunits

        cls._contract_unit_map = mappings
        cls._contract_unit_map_lower = {k.lower(): v for k, v in mappings.items()}
        return cls._contract_unit_map

    @classmethod
    def get_contract_unit(cls, variable_name: str) -> Optional[str]:
        """Resolve an exact SCM unit without applying generic type fallbacks."""
        if not variable_name:
            return None
        mappings = cls._load_unit_mappings_from_contracts()
        if not mappings:
            return None
        if variable_name in mappings:
            return mappings[variable_name]
        if cls._contract_unit_map_lower is None:
            cls._contract_unit_map_lower = {k.lower(): v for k, v in mappings.items()}
        return cls._contract_unit_map_lower.get(variable_name.lower())

    @classmethod
    def validate_unit(
        cls,
        variable: str,
        unit: str,
        category: Optional[UnitCategory] = None,
    ) -> bool:
        """
        Validate if a unit is CF-1.8 compliant for a given variable.

        Args:
            variable: Variable name
            unit: Unit string to validate
            category: Optional unit category for validation

        Returns:
            True if unit is valid, False otherwise
        """
        _ = category  # preserved API argument
        standard_units = [
            cls.TEMPERATURE,
            cls.PRESSURE,
            cls.TIME,
            cls.LENGTH,
            cls.MASS,
            cls.ENERGY,
            cls.CARBON_FLUX,
            cls.WATER_FLUX,
            cls.ENERGY_FLUX,
            cls.MIXING_RATIO,
            cls.MASS_CONCENTRATION,
        ]

        if unit in standard_units:
            return True

        standard_unit = cls.get_standard_unit(variable)
        if standard_unit != "unknown" and unit == standard_unit:
            return True

        if standard_unit != "unknown":
            # can_convert answers False for a non-convertible pair; an error
            # raised by the converter is a failure, not an invalid unit.
            return bool(
                unit_converter.can_convert(
                    unit,
                    standard_unit,
                    variable_name="generic",
                )
            )

        return False

    @classmethod
    def convert(
        cls,
        value: Union[Tensor, float],
        from_unit: str,
        to_unit: str,
    ) -> Union[Tensor, float]:
        """
        Convert a value between units of a generic (variable-independent) quantity.

        This is not a second conversion implementation: it calls the single unit
        conversion owner, relic.data.standards.unit_converter.unit_converter.convert,
        with variable_name="generic". It therefore covers conversions that hold
        for any quantity (same-dimension UDUNITS ratios such as g C m-2 day-1 ->
        kg C m-2 s-1 or ppm -> mol mol-1, the affine temperature pairs, and the
        generic table entries). Conversions that depend on the variable, such as
        the liquid-water-equivalent rule between a depth (rate) and an areal mass
        (flux), for example mm day-1 -> kg m-2 s-1, are only defined for a
        declared liquid-water variable and must call
        unit_converter.convert(value, variable_name, source_units, target_units,
        ConversionContext()) with that variable name (lwe_*, precipitation_flux,
        water_flux, ...).

        Args:
            value: Value(s) to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value(s)

        Raises:
            ValueError: If the conversion is not defined for a generic quantity
                (the message names the variable-aware entry) or the converter
                rejects the value container
        """
        if from_unit == to_unit:
            return value

        # unit_converter.convert raises ValueError for an unsupported pair and
        # TypeError for an unsupported value container; other errors propagate.
        try:
            return unit_converter.convert(
                value,
                "generic",
                from_unit,
                to_unit,
                ConversionContext(),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"UnitSystem.convert cannot convert {from_unit} to {to_unit} as a "
                f"generic (variable-independent) quantity: {exc}. A conversion that "
                "depends on the variable, such as the liquid-water-equivalent rule "
                "(mm day-1 -> kg m-2 s-1), needs "
                "relic.data.standards.unit_converter.unit_converter.convert(value, "
                "variable_name, source_units, target_units, ConversionContext()) with "
                "the declared variable name (lwe_*, precipitation_flux, water_flux, ...)."
            ) from exc

    @classmethod
    def get_standard_unit(cls, variable_type: str) -> str:
        """
        Get the standard CF-1.8 unit for a variable type.

        Args:
            variable_type: Type of variable (e.g., "temperature", "carbon_flux")

        Returns:
            Standard unit string
        """
        contract_unit = cls.get_contract_unit(variable_type)
        if contract_unit:
            return contract_unit

        unit_map = {
            "temperature": cls.TEMPERATURE,
            "air_temperature": cls.TEMPERATURE,
            "soil_temperature": cls.TEMPERATURE,
            "pressure": cls.PRESSURE,
            "atmospheric_pressure": cls.PRESSURE,
            "carbon_flux": cls.CARBON_FLUX,
            "respiration": cls.CARBON_FLUX,
            "water_flux": cls.WATER_FLUX,
            "evapotranspiration": cls.WATER_FLUX,
            "precipitation": cls.WATER_FLUX,
            "energy_flux": cls.ENERGY_FLUX,
            "radiation": cls.ENERGY_FLUX,
            "sensible_heat": cls.ENERGY_FLUX,
            "latent_heat": cls.ENERGY_FLUX,
            "concentration": cls.MASS_CONCENTRATION,
            "mixing_ratio": cls.MIXING_RATIO,
            "time": cls.TIME,
            "timestep": cls.TIME,
        }

        return unit_map.get(variable_type.lower(), "unknown")

    @classmethod
    def ensure_tensor(
        cls,
        value: Union[Tensor, float],
        reference_tensor: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Ensure value is a tensor with correct device and dtype.

        Args:
            value: Input value
            reference_tensor: Reference tensor for device and dtype
            device: Target device
            dtype: Target dtype

        Returns:
            Tensor with correct properties
        """
        _ = cls
        from ..abstractions import ensure_tensor as core_ensure_tensor

        return core_ensure_tensor(
            value,
            reference_tensor=reference_tensor,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def celsius_to_kelvin(cls, celsius: Union[Tensor, float]) -> Union[Tensor, float]:
        """Convert Celsius to Kelvin."""
        return cls.convert(celsius, "degC", "K")

    @classmethod
    def kelvin_to_celsius(cls, kelvin: Union[Tensor, float]) -> Union[Tensor, float]:
        """Convert Kelvin to Celsius."""
        return cls.convert(kelvin, "K", "degC")


class OutputUnitConfig:
    """
    Configuration for output unit preferences.

    Allows users to specify desired output units for different variable types,
    enabling automatic conversion from internal standard units to user-preferred units.
    """

    def __init__(self, config: Optional[Dict[str, str]] = None):
        """
        Initialize output unit configuration.

        Args:
            config: Dictionary mapping variable types to desired output units
                   e.g., {"carbon_flux": "g C m-2 day-1", "temperature": "degC"}
        """
        self.default_config = {
            "temperature": "K",
            "carbon_flux": "g C m-2 day-1",
            "nitrogen_flux": "g N m-2 s-1",
            "water_flux": "kg m-2 s-1",
            "energy_flux": "W m-2",
            "concentration": "ppm",
            "pressure": "kPa",
            "n2o_emission": "mg N m-2 h-1",
            "ch4_emission": "mg C m-2 h-1",
            "ch4_emission_mass": "mg CH4 m-2 h-1",
            "co2_emission": "g C m-2 day-1",
        }

        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        self._conv_pair_supported: Dict[Tuple[str, str, str], bool] = {}

    def get_output_unit(self, variable_name: str, variable_type: Optional[str] = None) -> str:
        """
        Get the preferred output unit for a variable.

        Args:
            variable_name: Name of the variable
            variable_type: Type category of the variable

        Returns:
            Preferred output unit string
        """
        if variable_name in self.config:
            return self.config[variable_name]

        if variable_type and variable_type in self.config:
            return self.config[variable_type]

        lower_name = variable_name.lower()
        if "n2o" in lower_name:
            return self._get_config_value("n2o_emission", "kg N m-2 s-1")
        if "ch4" in lower_name:
            if "mass" in lower_name:
                return self._get_config_value("ch4_emission_mass", "kg CH4 m-2 s-1")
            return self._get_config_value("ch4_emission", "kg C m-2 s-1")
        if "co2" in lower_name or "respiration" in lower_name:
            return self._get_config_value("co2_emission", "kg C m-2 s-1")
        standard_unit = UnitSystem.get_standard_unit(variable_type or variable_name)
        config_key_by_standard_unit = {
            UnitSystem.TEMPERATURE: "temperature",
            UnitSystem.CARBON_FLUX: "carbon_flux",
            UnitSystem.NITROGEN_FLUX: "nitrogen_flux",
            UnitSystem.WATER_FLUX: "water_flux",
            UnitSystem.ENERGY_FLUX: "energy_flux",
            UnitSystem.PRESSURE: "pressure",
            UnitSystem.MIXING_RATIO: "concentration",
            UnitSystem.MASS_CONCENTRATION: "concentration",
        }
        config_key = config_key_by_standard_unit.get(standard_unit)
        if config_key is not None:
            return self._get_config_value(config_key, standard_unit)
        return standard_unit

    def convert_output(
        self,
        data: Dict[str, Any],
        internal_units: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Convert output data from internal units to user-preferred units.

        Args:
            data: Output data dictionary with values
            internal_units: Dictionary specifying the internal units of each variable

        Returns:
            Data dictionary with converted values and updated unit metadata
        """
        converted = data.copy()

        for var_name, value in data.items():
            if var_name.endswith("_units"):
                continue

            unit_key = f"{var_name}_units"
            declared_unit = internal_units.get(var_name)
            embedded_unit = data.get(unit_key)
            if (
                declared_unit is not None
                and embedded_unit is not None
                and declared_unit != embedded_unit
            ):
                raise ValueError(
                    "Conflicting internal units for "
                    f"{var_name}: {declared_unit} != {embedded_unit}"
                )
            internal_unit = declared_unit or embedded_unit
            output_unit = self.get_output_unit(var_name)
            if not internal_unit:
                if output_unit != "unknown":
                    raise ValueError(
                        f"Missing internal units for output variable '{var_name}'"
                    )
                continue

            if internal_unit != output_unit:
                pair = (var_name, internal_unit, output_unit)
                if pair not in self._conv_pair_supported:
                    self._conv_pair_supported[pair] = unit_converter.can_convert(
                        internal_unit,
                        output_unit,
                        variable_name=var_name,
                    )
                if not self._conv_pair_supported[pair]:
                    raise ValueError(
                        "Unsupported output conversion for "
                        f"{var_name}: {internal_unit} -> {output_unit}"
                    )
                converted[var_name] = unit_converter.convert(
                    value,
                    var_name,
                    internal_unit,
                    output_unit,
                    ConversionContext(),
                )
            converted[unit_key] = output_unit

        return converted

    def to_dict(self) -> Dict[str, str]:
        """Return configuration as dictionary."""
        return self.config.copy()

    def update(self, config: Dict[str, str]) -> None:
        """Update configuration with new settings."""
        self.config.update(config)

    def _get_config_value(self, key: str, default: str) -> str:
        """Return config value if set, otherwise default."""
        if key in self.config:
            return self.config[key]
        return default

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OutputUnitConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            OutputUnitConfig instance
        """
        import yaml

        with open(yaml_path, "r", encoding="utf-8-sig") as f:
            settings = yaml.safe_load(f)
        if settings and isinstance(settings, dict):
            output_units = settings.get("output_units", {})
        else:
            output_units = {}
        return cls(output_units)

    @classmethod
    def get_common_presets(cls) -> Dict[str, "OutputUnitConfig"]:
        """
        Get common preset configurations.

        Returns:
            Dictionary of preset configurations
        """
        presets = {
            "si_standard": cls(
                {
                    "temperature": "K",
                    "carbon_flux": "kg C m-2 s-1",
                    "nitrogen_flux": "kg N m-2 s-1",
                    "water_flux": "kg m-2 s-1",
                    "concentration": "mol mol-1",
                    "pressure": "Pa",
                }
            ),
            "field_measurements": cls(
                {
                    "temperature": "degC",
                    "carbon_flux": "g C m-2 day-1",
                    "nitrogen_flux": "g N m-2 s-1",
                    "water_flux": "kg m-2 s-1",
                    "concentration": "ppm",
                    "pressure": "kPa",
                }
            ),
            "eddy_covariance": cls(
                {
                    "temperature": "degC",
                    "carbon_flux": "umol C m-2 s-1",
                    "water_flux": "kg m-2 s-1",
                    "energy_flux": "W m-2",
                    "concentration": "ppm",
                    "pressure": "kPa",
                }
            ),
        }
        return presets


__all__ = [
    "OutputUnitConfig",
    "UnitSystem",
]
