#!/usr/bin/env python3
"""
L1 Unit Tests for Unit Conversion System

This module tests the unit conversion functionality across the RELIC framework,
including:
- Data I/O unit conversions
- Transport mechanism unit conversions
- Coupling adapter unit conversions
- Bridge class unit conversion methods

References:
- CF-1.8 Climate and Forecast Metadata Conventions
- UDUNITS-2 Unit Conversion System
- RELIC Unit Conversion Design Document

Author: Yuchong Long
Date: 2025-10-11
Version: 1.0.0
"""

import pytest
import torch
import numpy as np
import xarray as xr
from pathlib import Path

# Import RELIC modules
from relic.data.standards.unit_converter import UnitConverter, ConversionContext
from relic.data.standards.cf_conventions import CFConventions
from relic.core.functor import TransportMechanismFunctor
from relic.core.functor import FunctorMetadata
from relic.data.io.writer import DataWriter, WriteConfig, WriteResult
from relic.coupling.scales.temporal.core import CalendarError, ValidationError
from relic.parameters import ParameterCategory, ParameterSpec, get_global_manager
from relic.parameters.ops.unit_validation import should_skip_unit_validation
from relic.utils.physical_constants import get_physical_constant


class TestUnitConverter:
    """Test the core UnitConverter class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = UnitConverter()

    def test_soil_moisture_conversions(self):
        """
        Test soil moisture unit conversions.

        Mathematical Expression:
            M_mass = M_volume * rho_water * depth
            where M_mass [kg m-2], M_volume [m3 m-3], rho_water [kg m-3], depth [m]
        """
        # Test data
        volume_fraction = torch.tensor([0.3, 0.5, 0.7])

        # Create conversion context
        context = ConversionContext(soil_depth=1.0, water_density=1000.0)

        # Convert volumetric to mass-based
        mass_content = self.converter.convert(
            volume_fraction, 'soil_moisture_content', 'm3 m-3', 'kg m-2', context
        )

        # Expected values: 0.3*1000 = 300, 0.5*1000 = 500, 0.7*1000 = 700
        expected = torch.tensor([300.0, 500.0, 700.0])
        assert torch.allclose(mass_content, expected, rtol=1e-6)

        # Convert back to volumetric
        volume_recovered = self.converter.convert(
            mass_content, 'soil_moisture_content', 'kg m-2', 'm3 m-3', context
        )
        assert torch.allclose(volume_recovered, volume_fraction, rtol=1e-6)

    def test_conversion_context_keeps_state_fields_pointwise(self):
        """State context fields are used elementwise; nothing is averaged.

        Mathematical Expression:
            m_w[i] = theta[i] * L[i] * rho_w[i]
            theta = [0.2, 0.4], L = [1, 2] m, rho_w = [997, 999] kg m-3
            pointwise: [199.4, 799.2] kg m-2
            (a mean context L = 1.5, rho_w = 998 would give [299.4, 598.8])
        """
        soil_depth = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        water_density = torch.tensor([997.0, 999.0], dtype=torch.float64, requires_grad=True)
        temperature = torch.tensor([298.15, 303.15], dtype=torch.float64)

        context = self.converter.get_conversion_context(
            {
                "soil_layer_thickness": soil_depth,
                "water_density": water_density,
                "air_temperature": temperature,
            },
            layer_thickness_variable="soil_layer_thickness",
        )

        assert context.soil_depth is soil_depth
        assert context.water_density is water_density
        assert context.temperature is temperature

        mass = self.converter.convert(
            torch.tensor([0.2, 0.4], dtype=torch.float64),
            "soil_moisture_content",
            "m3 m-3",
            "kg m-2",
            context,
        )
        assert torch.allclose(mass, torch.tensor([199.4, 799.2], dtype=torch.float64), rtol=1e-12)

        mass.sum().backward()
        assert torch.allclose(soil_depth.grad, torch.tensor([0.2 * 997.0, 0.4 * 999.0], dtype=torch.float64))
        assert torch.allclose(water_density.grad, torch.tensor([0.2 * 1.0, 0.4 * 2.0], dtype=torch.float64))

    def test_conversion_context_takes_no_reduction_of_state_fields(self):
        import inspect

        source = inspect.getsource(UnitConverter.get_conversion_context)
        for reduction in (".mean(", ".item(", ".sum(", ".median("):
            assert reduction not in source

    def test_layer_thickness_comes_only_from_the_declaration_or_an_explicit_argument(self):
        import inspect

        parameters = inspect.signature(self.converter.get_conversion_context).parameters
        assert "parameters" not in parameters
        for name in ("layer_thickness_variable", "layer_thickness"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is None

    def test_root_zone_depth_is_not_substituted_for_the_layer_thickness(self):
        """
        theta of a 0.1 m surface layer must not be scaled by the root-zone depth.

        Mathematical Expression:
            m_w = theta * L * rho_w, theta = [0.25, 0.30] m3 m-3, rho_w = 1000 kg m-3
            declared L = 0.1 m          -> [25, 30] kg m-2
            former substitute L = z_root = [1.0, 0.6] m -> [250, 180] kg m-2
        """
        state = {
            "root_zone_depth": torch.tensor([1.0, 0.6], dtype=torch.float64),
            "soil_depth": torch.tensor([2.0, 2.0], dtype=torch.float64),
            "water_density": torch.tensor(1000.0, dtype=torch.float64),
        }
        theta = torch.tensor([0.25, 0.30], dtype=torch.float64)

        undeclared = self.converter.get_conversion_context(state)
        assert undeclared.soil_depth is None
        with pytest.raises(ValueError, match="Missing required conversion context: soil_depth"):
            self.converter.convert(theta, "soil_moisture_content", "m3 m-3", "kg m-2", undeclared)

        declared = self.converter.get_conversion_context(state, layer_thickness=0.1)
        mass = self.converter.convert(theta, "soil_moisture_content", "m3 m-3", "kg m-2", declared)
        assert torch.allclose(mass, torch.tensor([25.0, 30.0], dtype=torch.float64))

    def test_root_zone_mean_declares_the_root_zone_depth_as_its_thickness(self):
        root_zone_depth = torch.tensor([1.0, 0.6], dtype=torch.float64)

        context = self.converter.get_conversion_context(
            {"root_zone_depth": root_zone_depth},
            layer_thickness_variable="root_zone_depth",
        )

        assert context.soil_depth is root_zone_depth

    def test_explicit_layer_thickness_argument_is_kept_as_given(self):
        thickness = torch.tensor([0.1, 0.2], dtype=torch.float64)

        assert self.converter.get_conversion_context({}, layer_thickness=thickness).soil_depth is thickness
        assert self.converter.get_conversion_context({}, layer_thickness=0.1).soil_depth == 0.1

    @pytest.mark.parametrize(
        ("state", "kwargs", "message"),
        [
            ({"dz": torch.tensor([0.1])}, {"layer_thickness_variable": "dz", "layer_thickness": 0.1},
             "either layer_thickness_variable or layer_thickness"),
            ({}, {"layer_thickness_variable": "soil_layer_thickness"},
             "declared layer thickness variable 'soil_layer_thickness' is not in the state"),
            ({}, {"layer_thickness_variable": " "}, "layer_thickness_variable must be a non-empty"),
            ({}, {"layer_thickness": "0.1"}, "layer_thickness must be a tensor or a real number"),
            ({}, {"layer_thickness": True}, "layer_thickness must be a tensor or a real number"),
        ],
    )
    def test_invalid_layer_thickness_declarations_are_rejected(self, state, kwargs, message):
        with pytest.raises(ValueError, match=message):
            self.converter.get_conversion_context(state, **kwargs)

    def test_missing_depth_context_raises_where_the_conversion_needs_it(self):
        context = self.converter.get_conversion_context(
            {"water_density": torch.tensor([1000.0], dtype=torch.float64)}
        )

        assert context.soil_depth is None
        with pytest.raises(ValueError, match="Missing required conversion context: soil_depth"):
            self.converter.convert(
                torch.tensor([0.2], dtype=torch.float64),
                "soil_moisture_content",
                "m3 m-3",
                "kg m-2",
                context,
            )

    def test_water_density_context_is_state_derived(self):
        water_density = torch.tensor([997.0, 999.0])

        assert self.converter.get_conversion_context(
            {"water_density": water_density}
        ).water_density is water_density
        assert self.converter.get_conversion_context({}).water_density is None

    def test_thermodynamic_context_reads_only_the_canonical_state_names(self):
        # One quantity, one name: air_temperature and atmospheric_pressure from the
        # variable contracts. Aliases (air_pressure) are resolved by the ContractRegistry
        # before a state reaches the converter; no fallback order is kept here.
        temperature = torch.tensor([290.0, 300.0], dtype=torch.float64)
        pressure = torch.tensor([95000.0, 101325.0], dtype=torch.float64)

        canonical = self.converter.get_conversion_context(
            {"air_temperature": temperature, "atmospheric_pressure": pressure}
        )
        assert canonical.temperature is temperature
        assert canonical.pressure is pressure
        for state in (
            {"temperature": temperature, "pressure": pressure},
            {"air_pressure": pressure},
        ):
            context = self.converter.get_conversion_context(state)
            assert context.temperature is None and context.pressure is None, state

    def test_another_temperature_is_passed_explicitly(self):
        # A conversion that needs the leaf (or soil) temperature gets it from the caller.
        air = torch.tensor([290.0], dtype=torch.float64)
        leaf = torch.tensor([295.0], dtype=torch.float64)
        context = self.converter.get_conversion_context({"air_temperature": air}, temperature=leaf)
        assert context.temperature is leaf
        pressure = torch.tensor([90000.0], dtype=torch.float64)
        assert self.converter.get_conversion_context({}, pressure=pressure).pressure is pressure

    def test_mass_to_volumetric_conversion_does_not_silently_clip(self):
        """Unit conversion must preserve the mathematical value exactly."""
        context = ConversionContext(soil_depth=1.0, water_density=1000.0)

        result = self.converter.convert(
            torch.tensor([1100.0]),
            'soil_moisture_content',
            'kg m-2',
            'm3 m-3',
            context,
        )
        restored = self.converter.convert(
            result,
            'soil_moisture_content',
            'm3 m-3',
            'kg m-2',
            context,
        )

        assert torch.allclose(result, torch.tensor([1.1]), rtol=1e-6)
        assert torch.allclose(restored, torch.tensor([1100.0]), rtol=1e-6)

    def test_unit_conversion_lookup(self):
        """Test unit conversion capability detection via mapping lookup."""
        # Test supported conversion
        conversion_key = ('air_temperature', 'celsius', 'K')
        assert conversion_key in self.converter.UNIT_CONVERSIONS

        # Test unsupported conversion
        conversion_key = ('unknown_variable', 'invalid_unit', 'kg')
        assert conversion_key not in self.converter.UNIT_CONVERSIONS


# Units a conversion category holds (mass and amount forms of one element count as
# one quantity, because the element rule converts between them).
CATEGORY_UNITS = {
    "carbon_flux": ("kg C m-2 s-1", "mol C m-2 s-1"),
    "water_flux": ("kg m-2 s-1",),
    "soil_moisture_content": ("kg m-2",),
    "air_temperature": ("K",),
    "pressure": ("Pa",),
    "co2_concentration": ("mol mol-1", "mol m-3", "Pa"),
    "precipitation_flux": ("kg m-2 s-1",),
    "nitrogen_amount": ("g N m-2", "mol N m-2"),
    "nitrogen_flux": ("g N m-2 s-1", "mol N m-2 s-1"),
}


class TestUnitConverterAliasQuantityKind:
    """An alias never maps a variable to a category of another quantity kind."""

    def test_every_contracted_alias_has_the_quantity_of_its_category(self):
        from relic.contracts.contract_loader import load_contract_registry
        from relic.data.standards.dimensionality import dimensionality_from_unit

        registry = load_contract_registry()
        mismatches = []
        for alias, category in UnitConverter.VARIABLE_ALIASES.items():
            contract = registry.get_contract(alias)
            # A placeholder "dimensionless" contract unit is a defect of the contract.
            if contract is None or contract.udunits in ("dimensionless", "1", ""):
                continue
            allowed = {dimensionality_from_unit(unit) for unit in CATEGORY_UNITS[category]}
            if dimensionality_from_unit(contract.udunits) not in allowed:
                mismatches.append((alias, contract.udunits, category))
        assert mismatches == []

    @pytest.mark.parametrize(
        "name",
        ["soil_water_content", "volumetric_water_content", "soil_moisture_content_volumetric"],
    )
    def test_a_volumetric_fraction_is_not_the_areal_mass_content(self, name):
        converter = UnitConverter()
        context = ConversionContext(soil_depth=0.1)

        assert converter._normalize_variable_name(name) != "soil_moisture_content"
        with pytest.raises(ValueError, match="Unsupported conversion"):
            converter.convert(0.3, name, "m3 m-3", "kg m-2", context)

    @pytest.mark.parametrize("name", ["latent_heat_flux", "precipitation"])
    def test_energy_flux_and_ambiguous_amount_names_have_no_water_flux_category(self, name):
        assert UnitConverter()._normalize_variable_name(name) not in {"water_flux", "precipitation_flux"}


class TestElementMassAmountRule:
    """One rule converts element mass and element amount for any SI prefix."""

    def setup_method(self):
        self.converter = UnitConverter()

    @pytest.mark.parametrize(
        "element, symbol",
        [("carbon", "C"), ("nitrogen", "N"), ("phosphorus", "P")],
    )
    @pytest.mark.parametrize(
        "mass_unit, mass_to_gram, amount_unit, amount_to_mol, rest",
        [
            ("kg", 1000.0, "mol", 1.0, "m-2"),
            ("g", 1.0, "mol", 1.0, "m-2"),
            ("mg", 1.0e-3, "mmol", 1.0e-3, "m-2"),
            ("kg", 1000.0, "umol", 1.0e-6, "m-2 s-1"),
        ],
    )
    def test_mass_to_amount_and_back(self, element, symbol, mass_unit, mass_to_gram,
                                      amount_unit, amount_to_mol, rest):
        molar_mass = get_physical_constant(f"relic.physical.constants.molecular_weights.{element}")
        source = f"{mass_unit} {symbol} {rest}"
        target = f"{amount_unit} {symbol} {rest}"

        amount = self.converter.convert(1.0, f"{element}_pool", source, target)
        mass = self.converter.convert(amount, f"{element}_pool", target, source)

        assert amount == pytest.approx(mass_to_gram / molar_mass / amount_to_mol, rel=1e-12)
        assert mass == pytest.approx(1.0, rel=1e-12)

    def test_one_kilogram_of_carbon_per_square_metre(self):
        molar_mass = get_physical_constant("relic.physical.constants.molecular_weights.carbon")

        assert self.converter.convert(1.0, "soil_organic_carbon", "kg C m-2", "mol C m-2") == pytest.approx(
            1000.0 / molar_mass, rel=1e-12
        )
        assert self.converter.convert(1000.0 / molar_mass, "soil_organic_carbon", "mol C m-2", "kg C m-2") == (
            pytest.approx(1.0, rel=1e-12)
        )

    @pytest.mark.parametrize(
        "source, target",
        [("kg C m-2", "mol N m-2"), ("kg C m-2", "mol m-2"), ("kg C m-2", "mol C m-2 s-1")],
    )
    def test_mismatched_element_or_dimension_is_rejected(self, source, target):
        with pytest.raises(ValueError, match="Unsupported conversion"):
            self.converter.convert(1.0, "soil_organic_carbon", source, target)

    def test_element_labels_come_from_the_one_dimensionality_rule(self):
        import inspect

        # A "C" that does not follow a mass or amount unit is the coulomb, not a carbon label.
        with pytest.raises(ValueError, match="Unsupported conversion"):
            self.converter.convert(1.0, "soil_organic_carbon", "C kg m-2", "mol C m-2")
        source = inspect.getsource(UnitConverter._element_mass_amount_factor)
        assert "element_annotations_from_unit" in source
        assert "unit_without_element_annotations" in source

    def test_tonnes_of_carbon_per_hectare_to_amount(self):
        molar_mass = get_physical_constant("relic.physical.constants.molecular_weights.carbon")
        amount = self.converter.convert(1.0, "soil_organic_carbon", "t C ha-1", "mol C ha-1")
        assert amount == pytest.approx(1.0e6 / molar_mass, rel=1e-12)


class TestElementLabelOfTheStandardName:
    """An element label that the CF name states as its basis converts by the plain unit scale."""

    CARBON_NAME = "gross_primary_productivity_of_carbon"

    def setup_method(self):
        self.converter = UnitConverter()

    def test_the_basis_comes_from_the_cf_name(self):
        assert CFConventions.element_basis_of_standard_name(self.CARBON_NAME) == "C"
        assert CFConventions.element_basis_of_standard_name("precipitation_flux") is None

    def test_label_is_removed_when_the_name_states_the_element(self):
        assert self.converter.can_convert("kg C m-2 s-1", "kg m-2 s-1", self.CARBON_NAME)
        assert self.converter.convert(2.5, self.CARBON_NAME, "kg C m-2 s-1", "kg m-2 s-1") == pytest.approx(2.5)

    def test_label_is_added_in_the_reverse_direction(self):
        assert self.converter.can_convert("kg m-2 s-1", "kg C m-2 s-1", self.CARBON_NAME)
        assert self.converter.convert(2.5, self.CARBON_NAME, "kg m-2 s-1", "kg C m-2 s-1") == pytest.approx(2.5)

    def test_the_remaining_unit_scale_is_applied(self):
        converted = self.converter.convert(1.0, self.CARBON_NAME, "g C m-2 s-1", "kg m-2 s-1")
        assert converted == pytest.approx(1.0e-3, rel=1e-12)

    def test_an_element_other_than_the_stated_one_is_an_error(self):
        assert not self.converter.can_convert("kg N m-2 s-1", "kg m-2 s-1", self.CARBON_NAME)
        with pytest.raises(ValueError, match="contradicts the carbon basis"):
            self.converter.convert(1.0, self.CARBON_NAME, "kg N m-2 s-1", "kg m-2 s-1")

    @pytest.mark.parametrize("variable_name", ["precipitation_flux", "generic"])
    def test_a_name_without_an_element_basis_keeps_the_label(self, variable_name):
        assert not self.converter.can_convert("kg C m-2 s-1", "kg m-2 s-1", variable_name)
        with pytest.raises(ValueError, match="Unsupported conversion"):
            self.converter.convert(1.0, variable_name, "kg C m-2 s-1", "kg m-2 s-1")

    def test_a_different_dimension_is_not_a_label_removal(self):
        assert not self.converter.can_convert("kg C m-2", "kg m-2 s-1", self.CARBON_NAME)

class TestUnitConverterSoilMoisture:
    """Test soil moisture conversions using the standard UnitConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = UnitConverter()

    def test_bridge_soil_moisture_conversion(self):
        """Test soil moisture conversion."""
        volume_fraction = torch.tensor([0.2, 0.4, 0.6])
        depth = 1.5  # m
        context = ConversionContext(soil_depth=depth, water_density=1000.0)

        mass_content = self.converter.convert(
            volume_fraction, 'soil_moisture_content', 'm3 m-3', 'kg m-2', context
        )
        expected = volume_fraction * depth * 1000.0  # water density

        assert torch.allclose(mass_content, expected, rtol=1e-6)

    def test_bridge_conversion_methods(self):
        """Test forward/reverse soil moisture conversions."""
        volume_data = torch.tensor([0.1, 0.3, 0.5])
        mass_data = torch.tensor([100.0, 300.0, 500.0])
        depth = 1.0
        context = ConversionContext(soil_depth=depth, water_density=1000.0)

        mass_converted = self.converter.convert(
            volume_data, 'soil_moisture_content', 'm3 m-3', 'kg m-2', context
        )
        assert mass_converted.shape == volume_data.shape
        assert torch.all(mass_converted > 0)

        volume_converted = self.converter.convert(
            mass_converted, 'soil_moisture_content', 'kg m-2', 'm3 m-3', context
        )
        assert volume_converted.shape == mass_data.shape
        assert torch.all(volume_converted >= 0)
        assert torch.all(volume_converted <= 1.0)  # Volumetric water content should be <= 1


class TransportMechanismStub(TransportMechanismFunctor):
    """Minimal transport mechanism used by unit conversion tests."""

    TRANSPORT_TYPE = "diffusion"
    __test__ = False

    def forward(self, state, time, dt, **kwargs):
        """Minimal implementation for testing."""
        return {'test_flux': torch.tensor([1.0])}


def _build_transport_stub() -> TransportMechanismStub:
    metadata = FunctorMetadata(
        name="test_transport",
        description="Unit conversion test stub for transport mechanisms.",
        references=["Test stub reference for unit conversion checks."],
        input_variables=["test_flux_input"],
        output_variables=["test_flux"],
    )
    return TransportMechanismStub(
        name="test_transport",
        metadata=metadata,
    )


class TestTransportUnitConversions:
    """Test TransportMechanismFunctor unit conversion functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transport = _build_transport_stub()

    def test_flux_unit_conversion_carbon(self):
        """
        Test carbon flux unit conversions.

        Mathematical Expressions:
            F_C [kg C m-2 s-1] = F_C [g C m-2 day-1] * (1/1000) * (1/86400)
            F_C [kg C m-2 s-1] = F_C [mol C m-2 s-1] * 12.011 * 1e-3
        """
        # Test g C m-2 day-1 to kg C m-2 s-1 conversion; the target unit is the
        # per-second rate of the leaf_carbon contract unit (kg C m-2).
        flux_daily = torch.tensor([10.0, 20.0, 30.0])  # g C m-2 day-1
        flux_key = "carbon_flux"
        target_key = "leaf_carbon"

        converted_flux, conversion_info = self.transport._apply_flux_unit_conversion(
            flux_daily,
            flux_key,
            target_key,
            state={"units": {flux_key: "g C m-2 day-1"}},
        )

        # Expected: 10 g/d -> 10/1000/86400 = 1.157e-7 kg/s
        expected_factor = 1.0 / (1000.0 * 86400.0)
        expected = flux_daily * expected_factor

        assert torch.allclose(converted_flux, expected, rtol=1e-6)
        assert conversion_info['converted'] == True
        assert conversion_info['conversion_type'] == 'carbon_flux_g_day_to_kg_s'
        assert conversion_info['original_units'] == 'g C m-2 day-1'
        assert conversion_info['target_units'] == 'kg C m-2 s-1'

    def test_flux_unit_conversion_water(self):
        """
        A precipitation depth rate reaches a water-mass storage through the
        CF liquid-water-equivalent rule of the UnitConverter.

        Mathematical Expression:
            F_W [kg m-2 s-1] = P [mm day-1] * 1e-3 [m mm-1] * rho_w / t_day
        with rho_w = relic.physical.constants.water.density_liquid (SSOT) and
        t_day = relic.physical.constants.time.seconds_per_day; no state density.
        """
        # soil_water_storage declares kg m-2; precipitation_flux belongs to the
        # lwe family, so the depth rate converts with the SSOT water density.
        flux_mm = torch.tensor([5.0, 10.0, 15.0])  # mm day-1
        flux_key = "precipitation_flux"
        target_key = "soil_water_storage"

        converted_flux, conversion_info = self.transport._apply_flux_unit_conversion(
            flux_mm,
            flux_key,
            target_key,
            state={"units": {flux_key: "mm day-1"}},
        )

        expected_factor = _lwe_mm_day_to_kg_m2_s_factor()
        torch.testing.assert_close(converted_flux, flux_mm * expected_factor)
        assert conversion_info['converted'] is True
        assert conversion_info['conversion_type'] == (
            'precipitation_flux: mm day-1 -> kg m-2 s-1'
        )
        assert conversion_info['original_units'] == 'mm day-1'
        assert conversion_info['target_units'] == 'kg m-2 s-1'

    def test_lwe_water_flux_ignores_state_density_and_keeps_gradients(self):
        """
        The lwe density is the SSOT constant: a water_density field in the
        state does not enter the factor, which stays one scalar, and the
        converted flux remains differentiable with respect to the flux.

        Mathematical Expression:
            F_W,i = P_i * f,  f = 1e-3 * rho_w / t_day,  dF_W,i / dP_i = f
        """
        flux_mm = torch.tensor([5.0, 10.0], requires_grad=True)  # mm day-1
        state_density = torch.tensor([997.0, 1000.0], requires_grad=True)  # kg m-3

        converted_flux, conversion_info = self.transport._apply_flux_unit_conversion(
            flux_mm,
            "precipitation_flux",
            "soil_water_storage",
            state={
                "units": {"precipitation_flux": "mm day-1"},
                "water_density": state_density,
            },
        )

        expected_factor = _lwe_mm_day_to_kg_m2_s_factor()
        torch.testing.assert_close(converted_flux.detach(), flux_mm.detach() * expected_factor)
        factor = conversion_info["conversion_factor"]
        assert isinstance(factor, torch.Tensor)
        assert factor.numel() == 1
        assert float(factor) == pytest.approx(expected_factor)
        assert conversion_info["conversion_factor_min"] == pytest.approx(expected_factor)
        assert conversion_info["conversion_factor_max"] == pytest.approx(expected_factor)

        converted_flux.sum().backward()
        torch.testing.assert_close(
            flux_mm.grad, torch.full_like(flux_mm.detach(), expected_factor)
        )
        assert state_density.grad is None

    def test_depth_rate_into_a_content_state_raises(self):
        """
        soil_moisture declares m3 m-3, so a flux into it must be a content rate
        (m3 m-3 s-1). A water depth rate (m s-1) has no single UnitConverter
        rule to that rate, so the conversion raises instead of applying the
        depth rate to the content state.
        """
        with pytest.raises(ValueError, match="m s-1 -> m3 m-3 s-1"):
            self.transport._apply_flux_unit_conversion(
                torch.tensor([1.0e-6, 2.0e-6]),
                "infiltration_flux",
                "soil_moisture",
                state={
                    "units": {"infiltration_flux": "m s-1"},
                    "water_density": torch.tensor(1000.0),
                    "soil_depth": torch.tensor(0.5),
                },
            )

    def test_content_rate_into_a_mass_state_without_declared_thickness_raises(self):
        """A volumetric content rate into a mass storage (soil_water_storage,
        kg m-2) needs the layer thickness declared by the contract of the
        volumetric side, here the flux; a state entry named soil_depth is not a
        declaration, so the conversion raises."""
        with pytest.raises(ValueError, match="layer thickness"):
            self.transport._apply_flux_unit_conversion(
                torch.tensor([1.0e-6]),
                "infiltration_flux",
                "soil_water_storage",
                state={
                    "units": {"infiltration_flux": "m3 m-3 s-1"},
                    "water_density": torch.tensor(1000.0),
                    "soil_depth": torch.tensor(0.5),
                },
            )

    def test_mass_rate_into_soil_moisture_with_declared_thickness_absent_from_state_raises(self):
        """soil_moisture declares soil_depth as its layer thickness; a state
        without that field cannot be converted (no default, no substitute)."""
        with pytest.raises(ValueError, match="soil_depth"):
            self.transport._apply_flux_unit_conversion(
                torch.tensor([1.0e-3]),
                "infiltration_flux",
                "soil_moisture",
                state={
                    "units": {"infiltration_flux": "kg m-2 s-1"},
                    "water_density": torch.tensor(1000.0),
                    "root_zone_depth": torch.tensor(0.5),
                },
            )

    def test_soil_moisture_state_unit_is_not_a_flux_unit(self):
        """m3 m-3 is a soil-moisture state unit, not a flux unit."""
        with pytest.raises(ValueError, match="m3 m-3 -> m3 m-3 s-1"):
            self.transport._apply_flux_unit_conversion(
                torch.tensor([0.3]),
                "infiltration_flux",
                "soil_moisture",
                state={
                    "units": {"infiltration_flux": "m3 m-3"},
                    "water_density": torch.tensor(1000.0),
                    "soil_depth": torch.tensor(0.5),
                },
            )

    def test_soil_moisture_content_rate_uses_declared_thickness_per_cell(self):
        """
        The soil_moisture contract declares m3 m-3 and soil_depth as its layer
        thickness; a mass rate maps through the UnitConverter soil-moisture
        rule to the content rate with that thickness read pointwise.

        Mathematical Expression:
            d theta_i / dt = F_m,i / (L_i * rho_w),   L and rho_w constant over dt
        Numeric example: F_m = 1e-3 kg m-2 s-1, L = [0.5, 1.0] m,
            rho_w = 1000 kg m-3 -> [2e-6, 1e-6] m3 m-3 s-1.
        Gradients: d/dF_m = 1 / (L rho_w); d/dL = -F_m / (L^2 rho_w).
        """
        flux = torch.tensor([1.0e-3, 1.0e-3], requires_grad=True)
        thickness = torch.tensor([0.5, 1.0], requires_grad=True)
        converted_flux, conversion_info = self.transport._apply_flux_unit_conversion(
            flux,
            "infiltration_flux",
            "soil_moisture",
            state={
                "units": {"infiltration_flux": "kg m-2 s-1"},
                "water_density": torch.tensor(1000.0),
                "soil_depth": thickness,
            },
        )
        torch.testing.assert_close(converted_flux.detach(), torch.tensor([2.0e-6, 1.0e-6]))
        assert conversion_info["converted"] is True
        assert conversion_info["layer_thickness_variable"] == "soil_depth"
        assert conversion_info["original_units"] == "kg m-2 s-1"
        assert conversion_info["target_units"] == "m3 m-3 s-1"
        assert conversion_info["target_units_source"] == "variable_contract"
        torch.testing.assert_close(
            conversion_info["layer_thickness_used"], thickness.detach()
        )
        converted_flux.sum().backward()
        torch.testing.assert_close(
            flux.grad, 1.0 / (thickness.detach() * 1000.0)
        )
        torch.testing.assert_close(
            thickness.grad,
            -flux.detach() / (thickness.detach() ** 2 * 1000.0),
        )


def _lwe_mm_day_to_kg_m2_s_factor() -> float:
    """1e-3 [m mm-1] * rho_w [kg m-3] / t_day [s day-1] from the physical SSOT."""
    rho_w = float(get_physical_constant("relic.physical.constants.water.density_liquid"))
    return 1.0e-3 * rho_w / _seconds_per_day()


def _seconds_per_day() -> float:
    return float(get_physical_constant("relic.physical.constants.time.seconds_per_day"))


class TestDeclaredFluxUnitRouting:
    """
    The flux unit is the declared unit of the flux key and the target unit is
    the per-second rate of the target's variable-contract udunits, because the
    transport increment is flux * dt [s] in the target's units. A flux or
    target name never selects, skips, or defaults a conversion.
    """

    def setup_method(self):
        self.transport = _build_transport_stub()

    def test_neutral_flux_name_converts_to_the_target_contract_rate(self):
        # fast_som_c contract: g C m-2 -> target rate g C m-2 s-1.
        flux = torch.tensor([86.4, 172.8])
        converted, info = self.transport._apply_flux_unit_conversion(
            flux,
            "tracer_flux",
            "fast_som_c",
            state={"units": {"tracer_flux": "g C m-2 day-1"}},
        )
        torch.testing.assert_close(converted, flux / _seconds_per_day())
        assert info["converted"] is True
        assert info["original_units"] == "g C m-2 day-1"
        assert info["target_units"] == "g C m-2 s-1"
        assert info["target_units_source"] == "variable_contract"

    def test_carbon_named_flux_follows_the_target_contract_not_a_family_unit(self):
        # microbial_biomass_c contract: g C m-2; a carbon family unit
        # (kg C m-2 s-1) would be 1000 times too small for this state.
        flux = torch.tensor([10.0])
        converted, info = self.transport._apply_flux_unit_conversion(
            flux,
            "carbon_flux",
            "microbial_biomass_c",
            state={"units": {"carbon_flux": "g C m-2 day-1"}},
        )
        torch.testing.assert_close(converted, flux / _seconds_per_day())
        assert info["target_units"] == "g C m-2 s-1"

    def test_flux_declared_in_the_target_rate_is_returned_unconverted(self):
        flux = torch.tensor([1.0e-6])
        converted, info = self.transport._apply_flux_unit_conversion(
            flux,
            "tracer_flux",
            "fast_som_c",
            state={"units": {"tracer_flux": "g C m-2 s-1"}},
        )
        assert converted is flux
        assert info["converted"] is False
        assert info["original_units"] == "g C m-2 s-1"
        assert info["target_units"] == "g C m-2 s-1"

    @pytest.mark.parametrize("state", [None, {}, {"units": {}}])
    def test_undeclared_flux_units_raise_for_every_name(self, state):
        with pytest.raises(ValueError, match="declares no units"):
            self.transport._apply_flux_unit_conversion(
                torch.tensor([1.0]),
                "tracer_flux",
                "fast_som_c",
                state=state,
            )

    def test_target_without_variable_contract_raises(self):
        with pytest.raises(ValueError, match="no variable contract"):
            self.transport._apply_flux_unit_conversion(
                torch.tensor([1.0]),
                "tracer_flux",
                "unknown_target",
                state={"units": {"tracer_flux": "g C m-2 s-1"}},
            )

    def test_energy_flux_into_a_temperature_state_raises(self):
        # temperature contract: K -> rate K s-1; W m-2 needs a heat capacity.
        with pytest.raises(ValueError, match="K s-1"):
            self.transport._apply_flux_unit_conversion(
                torch.tensor([100.0]),
                "sensible_heat_flux",
                "temperature",
                state={"units": {"sensible_heat_flux": "W m-2"}},
            )

    def test_declared_target_unit_conflicting_with_the_contract_raises(self):
        with pytest.raises(ValueError, match="conflicts with"):
            self.transport._apply_flux_unit_conversion(
                torch.tensor([1.0e-3]),
                "infiltration_flux",
                "soil_moisture",
                state={
                    "units": {
                        "infiltration_flux": "kg m-2 s-1",
                        "soil_moisture": "kg m-2 s-1",
                    },
                    "water_density": torch.tensor(1000.0),
                    "soil_depth": torch.tensor(0.5),
                },
            )

    def test_finalize_without_state_raises_for_a_mapped_flux(self):
        with pytest.raises(ValueError, match="declares no units"):
            self.transport.finalize_transport_updates(
                {"tracer_flux": torch.tensor([1.0])},
                1.0,
                state=None,
                mapping={"tracer_flux": "fast_som_c"},
            )


class TestDataIOWriterConversions:
    """Test DataWriter unit conversion functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.writer = DataWriter()

    def test_cf_standard_units_mapping(self):
        """Official canonical units come only from the bundled v94 owner."""
        assert CFConventions.get_standard_units('air_temperature') == 'K'
        assert CFConventions.get_standard_units('air_pressure') == 'Pa'
        assert CFConventions.get_standard_units(
            'gross_primary_productivity_of_carbon'
        ) == 'kg m-2 s-1'
        assert CFConventions.get_standard_units('runoff_amount') == 'kg m-2'

    def test_unit_conversion_application(self):
        """Writer does not infer or own implicit CF unit conversions."""
        assert not hasattr(self.writer, '_get_cf_standard_units')
        assert not hasattr(self.writer, '_apply_unit_conversions')
        assert CFConventions.are_standard_name_units_equivalent(
            'air_temperature',
            'degree_Celsius',
        )
        assert not CFConventions.are_standard_name_units_equivalent(
            'air_temperature',
            'Pa',
        )

    def test_time_output_missing_cf_units_fails_closed(self, tmp_path):
        """Time output must not invent a reference epoch."""
        dataset = xr.Dataset(
            coords={
                "time": (
                    "time",
                    [0.0, 1.0],
                    {"calendar": "noleap"},
                )
            }
        )

        with pytest.raises(ValidationError, match="explicit CF time units"):
            self.writer.write_dataset(
                dataset,
                tmp_path / "missing_time_units.nc",
                cf_sync=False,
                auto_chunk=False,
                validate_output=False,
                strict_cf=False,
            )

    def test_time_output_missing_calendar_fails_closed(self):
        """Time output must carry one explicit calendar declaration."""
        dataset = xr.Dataset(
            coords={
                "time": (
                    "time",
                    [0.0, 1.0],
                    {"units": "days since 2000-01-01"},
                )
            }
        )
        result = WriteResult(output_path=Path("unused.nc"))

        with pytest.raises(CalendarError, match="explicit calendar"):
            self.writer._add_time_bounds(dataset, self.writer.config, result)

    def test_time_output_rejects_attr_encoding_conflict(self):
        """Attribute and encoding evidence must describe the same timeline."""
        dataset = xr.Dataset(
            coords={
                "time": (
                    "time",
                    [0.0, 1.0],
                    {
                        "units": "days since 2000-01-01",
                        "calendar": "noleap",
                    },
                )
            }
        )
        dataset.time.encoding["units"] = "hours since 2000-01-01"
        result = WriteResult(output_path=Path("unused.nc"))

        with pytest.raises(ValidationError, match="Conflicting CF time units"):
            self.writer._add_time_bounds(dataset, self.writer.config, result)

    def test_time_output_accepts_encoding_only_evidence_without_bounds(self):
        """Encoding-only CF evidence remains valid at the xarray boundary."""
        dataset = xr.Dataset(coords={"time": ("time", [0.0, 1.0])})
        dataset.time.encoding.update(
            {
                "units": "days since 2000-01-01",
                "calendar": "360_day",
            }
        )
        config = WriteConfig(add_bounds=False)
        result = WriteResult(output_path=Path("unused.nc"))

        processed = self.writer._add_time_bounds(dataset, config, result)

        assert processed.time.attrs["units"] == "days since 2000-01-01"
        assert processed.time.attrs["calendar"] == "360_day"
        assert "bounds" not in processed.time.attrs
        assert "time_bounds" not in processed.variables
        assert result.bounds_computed == []

    def test_time_output_preserves_declared_source_bounds(self):
        """Declared CF interval evidence must not be regenerated or renamed."""
        source_bounds = np.asarray([[0.0, 1.0], [1.0, 2.0]])
        dataset = xr.Dataset(
            coords={
                "time": (
                    "time",
                    [0.5, 1.5],
                    {
                        "units": "days since 2000-01-01",
                        "calendar": "noleap",
                        "bounds": "source_time_bounds",
                    },
                )
            }
        )
        dataset["source_time_bounds"] = (("time", "nv"), source_bounds)
        dataset["source_time_bounds"].attrs = {
            "long_name": "source interval evidence",
            "provenance": "fixture",
        }
        result = WriteResult(output_path=Path("unused.nc"))

        processed = self.writer._add_time_bounds(dataset, self.writer.config, result)

        assert processed.time.attrs["bounds"] == "source_time_bounds"
        assert processed["source_time_bounds"].dims == ("time", "nv")
        assert np.array_equal(processed["source_time_bounds"].values, source_bounds)
        assert processed["source_time_bounds"].attrs["provenance"] == "fixture"
        assert "time_bounds" not in processed.variables
        assert result.bounds_computed == []

    def test_time_output_rejects_singleton_without_interval_evidence(self):
        """One label cannot determine one support interval without evidence."""
        dataset = xr.Dataset(
            coords={
                "time": (
                    "time",
                    [0.0],
                    {
                        "units": "days since 2000-01-01",
                        "calendar": "noleap",
                    },
                )
            }
        )
        result = WriteResult(output_path=Path("unused.nc"))

        with pytest.raises(ValidationError, match="singleton time bounds"):
            self.writer._add_time_bounds(dataset, self.writer.config, result)

    def test_time_output_serializes_canonical_torch_bounds_to_numpy(self):
        """Torch owns bounds arithmetic; xarray receives CPU boundary arrays."""
        dataset = xr.Dataset(
            coords={
                "time": (
                    "time",
                    [0.5, 1.5],
                    {
                        "units": "days since 2000-01-01",
                        "calendar": "noleap",
                    },
                )
            }
        )
        result = WriteResult(output_path=Path("unused.nc"))

        processed = self.writer._add_time_bounds(dataset, self.writer.config, result)

        assert isinstance(processed.time.values, np.ndarray)
        assert isinstance(processed.time_bounds.values, np.ndarray)
        assert processed.time.attrs["calendar"] == "noleap"
        assert processed.time.attrs["bounds"] == "time_bounds"
        assert result.bounds_computed == ["time"]


class TestCouplingAdapterConversions:
    """Test coupling adapter unit conversion consistency."""

    def test_wrf_adapter_unit_conversion(self):
        """Test WRF adapter uses correct unit conversion interface."""
        # This would require the actual WRF adapter to be available
        # For now, test the expected interface signature
        from relic.coupling.external.atmospheric_models.wrf_adapter import WRFAdapter

        # Check that adapter can be instantiated (would need proper config)
        # This is more of an interface verification test
        assert WRFAdapter is not None

    def test_vic_adapter_unit_conversion(self):
        """Test VIC adapter unit conversion consistency."""
        # Similar to WRF test, verify interface consistency
        pass


class TestIntegrationScenarios:
    """Integration tests for unit conversion across components."""

    def test_end_to_end_carbon_flux_conversion(self):
        """
        Test carbon flux conversion through the full pipeline.

        Scenario: GPP measured in g C m-2 day-1 ->converted to kg C m-2 s-1 ->
        Transported ->Converted back for output
        """
        # Input: GPP in g C m-2 day-1
        gpp_input = torch.tensor([20.0, 15.0, 25.0])

        # Step 1: Transport mechanism conversion
        transport = _build_transport_stub()
        converted_flux, _ = transport._apply_flux_unit_conversion(
            gpp_input,
            "gross_photosynthesis_flux",
            "leaf_carbon",
            state={"units": {"gross_photosynthesis_flux": "g C m-2 day-1"}},
        )

        # Verify conversion to kg C m-2 s-1
        seconds_per_day = get_physical_constant(
            "relic.physical.constants.time.seconds_per_day"
        )
        expected_kg_per_s = gpp_input / (1000.0 * seconds_per_day)
        assert torch.allclose(converted_flux, expected_kg_per_s, rtol=1e-6)

        # Step 2: Verify CF compliance
        assert converted_flux.max() < 1.0  # Should be small in kg/s units

    def test_water_balance_conservation(self):
        """
        Test water balance conservation across unit conversions.

        Scenario: Precipitation in mm day-1 ->converted to kg m-2 s-1 ->
        Soil moisture update ->Output conversion back to mm day-1
        """
        # Input: Precipitation in mm day-1
        precip_input = torch.tensor([10.0, 5.0, 0.0])
        water_density = 997.0

        # Convert to kg m-2 s-1
        transport = _build_transport_stub()
        precip_kg_per_s, _ = transport._apply_flux_unit_conversion(
            precip_input,
            "precipitation_flux",
            "soil_water_storage",
            state={
                "units": {"precipitation_flux": "mm day-1"},
                "water_density": torch.tensor(water_density),
            },
        )

        # Convert back to mm day-1 for verification
        precip_recovered = UnitConverter().convert(
            precip_kg_per_s,
            "precipitation_flux",
            "kg m-2 s-1",
            "mm day-1",
            ConversionContext(water_density=water_density),
        )
        assert torch.allclose(precip_recovered, precip_input, rtol=1e-6)

    def test_energy_flux_is_not_a_temperature_increment(self):
        """
        Scenario: a sensible heat flux routed into a temperature state.

        The increment flux * dt must be in K, so the flux must be a K s-1 rate;
        W m-2 -> K s-1 needs a heat capacity that no declared conversion
        supplies, so the transport step raises instead of adding J m-2 to K.
        """
        transport = _build_transport_stub()
        with pytest.raises(ValueError, match="cal cm-2 day-1 -> K s-1"):
            transport._apply_flux_unit_conversion(
                torch.tensor([200.0, 150.0, 100.0]),
                "sensible_heat_flux",
                "air_temperature",
                state={"units": {"sensible_heat_flux": "cal cm-2 day-1"}},
            )


class TestErrorHandling:
    """Test error handling in unit conversions."""

    def test_non_finite_flux_values_keep_their_kind(self):
        """A linear unit factor maps NaN to NaN and +inf to +inf; the
        conversion does not mask or replace non-finite flux values."""
        transport = _build_transport_stub()

        invalid_flux = torch.tensor([float('nan'), float('inf')])
        converted_flux, conversion_info = transport._apply_flux_unit_conversion(
            invalid_flux,
            "tracer_flux",
            "fast_som_c",
            state={"units": {"tracer_flux": "g C m-2 day-1"}},
        )

        assert conversion_info['converted'] is True
        assert torch.isnan(converted_flux[0])
        assert torch.isposinf(converted_flux[1])

    def test_zero_flux_handling(self):
        """Test handling of zero flux values."""
        transport = _build_transport_stub()

        zero_flux = torch.tensor([0.0, 0.0, 0.0])
        converted_flux, conversion_info = transport._apply_flux_unit_conversion(
            zero_flux,
            "zero_flux",
            "fast_som_c",
            state={"units": {"zero_flux": "g C m-2 day-1"}},
        )

        # Zero flux should remain zero after conversion
        assert torch.allclose(converted_flux, zero_flux, rtol=1e-6)

    def test_negative_flux_handling(self):
        """Test handling of negative flux values."""
        transport = _build_transport_stub()

        negative_flux = torch.tensor([-5.0, -10.0, -2.0])
        converted_flux, conversion_info = transport._apply_flux_unit_conversion(
            negative_flux,
            "negative_flux",
            "leaf_carbon",
            state={"units": {"negative_flux": "g C m-2 day-1"}},
        )

        # Negative fluxes should preserve sign after conversion
        assert torch.all(converted_flux < 0)
        assert torch.allclose(torch.abs(converted_flux),
                             torch.abs(negative_flux) * 1.0/(1000.0*86400.0), rtol=1e-6)


def test_parameter_manager_unit_scan_returns_list():
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    issues = pm.scan_unit_parseability(strict=False)
    assert isinstance(issues, list)
    if issues:
        assert "key" in issues[0]
        assert "units" in issues[0]


@pytest.mark.parametrize("units", ["dimensionless", "1", ""])
def test_parameter_manager_unit_scan_does_not_skip_numeric_dimension_declarations(units):
    spec = ParameterSpec(
        key=f"test.unit_validation.{units or 'blank'}",
        value=0.5,
        units=units,
        dtype="float32",
        bounds=(0.0, 1.0),
        category=ParameterCategory.BIOPHYSICAL,
        description="Numeric parameter used to verify unit scan coverage.",
        references=["RELIC unit validation contract"],
        tags={"unit-validation"},
        dimension="temperature",
        provenance={"source": "test", "version": "1.0.0"},
        requires_grad=False,
    )

    assert should_skip_unit_validation(None, spec) is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
