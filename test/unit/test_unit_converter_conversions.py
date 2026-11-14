"""L1 Unit Test: Unit Conversion Accuracy.

Tests the accuracy of various unit conversions across different
physical quantities, ensuring compliance with CF-1.8 standards.

Mathematical References:
- Temperature: K = C + 273.15
- Length: 1 km = 1000 m, 1 m = 1000 mm
- Mass: 1 kg = 1000 g
- Time: 1 hour = 3600 s
- Pressure: 1 kPa = 1000 Pa
- Flux conversions with molecular weights (C: 12 g/mol)
"""
import pytest
import torch
import numpy as np
from relic.data.standards.unit_converter import (
    ConversionContext,
    UnitConverter,
    unit_converter,
)
from relic.core.units import OutputUnitConfig
from relic.core.units.validator import StateUnitValidator
from relic.utils.physical_constants import get_physical_constant


class TestTemperatureConversions:
    """Test temperature unit conversions.

    Reference:
    - T(K) = T(C) + 273.15
    - CF-1.8 standard: air_temperature in Kelvin
    """

    def test_celsius_to_kelvin_zero(self, unit_converter, conversion_context):
        """Test 0 Celsius = 273.15 Kelvin."""
        result = unit_converter.convert(
            0.0,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )
        assert abs(result - 273.15) < 1e-5, \
            f"0C should be 273.15K, got {result}"

    def test_celsius_to_kelvin_positive(self, unit_converter, conversion_context):
        """Test positive Celsius to Kelvin."""
        result = unit_converter.convert(
            25.0,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )
        assert abs(result - 298.15) < 1e-5, \
            f"25C should be 298.15K, got {result}"

    def test_celsius_to_kelvin_negative(self, unit_converter, conversion_context):
        """Test negative Celsius to Kelvin."""
        result = unit_converter.convert(
            -40.0,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )
        assert abs(result - 233.15) < 1e-5, \
            f"-40C should be 233.15K, got {result}"

    def test_kelvin_to_celsius_zero(self, unit_converter, conversion_context):
        """Test 273.15 Kelvin = 0 Celsius."""
        result = unit_converter.convert(
            273.15,
            'air_temperature',
            'K',
            'celsius',
            conversion_context
        )
        assert abs(result - 0.0) < 1e-5, \
            f"273.15K should be 0C, got {result}"

    def test_kelvin_to_celsius_positive(self, unit_converter, conversion_context):
        """Test Kelvin to positive Celsius."""
        result = unit_converter.convert(
            298.15,
            'air_temperature',
            'K',
            'celsius',
            conversion_context
        )
        assert abs(result - 25.0) < 1e-5, \
            f"298.15K should be 25C, got {result}"

    def test_temperature_array_conversion(self, unit_converter, conversion_context):
        """Test array of temperature values."""
        temps_c = np.array([0.0, 25.0, -40.0, 100.0])
        expected_k = np.array([273.15, 298.15, 233.15, 373.15])

        result = unit_converter.convert(
            temps_c,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )

        assert np.allclose(result, expected_k, atol=1e-5), \
            f"Array conversion failed: {result} != {expected_k}"


class TestLengthConversions:
    """Test length unit conversions.

    Reference:
    - 1 km = 1000 m
    - 1 m = 1000 mm
    - 1 m = 100 cm
    """

    def test_meters_to_kilometers(self, unit_converter, conversion_context):
        """Test 1000 m = 1 km."""
        result = unit_converter.convert(
            1000.0,
            'length',
            'm',
            'km',
            conversion_context
        )
        assert abs(result - 1.0) < 1e-6, \
            f"1000m should be 1km, got {result}"

    def test_kilometers_to_meters(self, unit_converter, conversion_context):
        """Test 1 km = 1000 m."""
        result = unit_converter.convert(
            1.0,
            'length',
            'km',
            'm',
            conversion_context
        )
        assert abs(result - 1000.0) < 1e-6, \
            f"1km should be 1000m, got {result}"

    def test_millimeters_to_meters(self, unit_converter, conversion_context):
        """Test 1000 mm = 1 m."""
        result = unit_converter.convert(
            1000.0,
            'length',
            'mm',
            'm',
            conversion_context
        )
        assert abs(result - 1.0) < 1e-6, \
            f"1000mm should be 1m, got {result}"

    def test_meters_to_millimeters(self, unit_converter, conversion_context):
        """Test 1 m = 1000 mm."""
        result = unit_converter.convert(
            1.0,
            'length',
            'm',
            'mm',
            conversion_context
        )
        assert abs(result - 1000.0) < 1e-6, \
            f"1m should be 1000mm, got {result}"


class TestMassConversions:
    """Test mass unit conversions.

    Reference:
    - 1 kg = 1000 g
    - 1 g = 1000 mg
    """

    def test_grams_to_kilograms(self, unit_converter, conversion_context):
        """Test 1000 g = 1 kg."""
        result = unit_converter.convert(
            1000.0,
            'mass',
            'g',
            'kg',
            conversion_context
        )
        assert abs(result - 1.0) < 1e-6, \
            f"1000g should be 1kg, got {result}"

    def test_kilograms_to_grams(self, unit_converter, conversion_context):
        """Test 1 kg = 1000 g."""
        result = unit_converter.convert(
            1.0,
            'mass',
            'kg',
            'g',
            conversion_context
        )
        assert abs(result - 1000.0) < 1e-6, \
            f"1kg should be 1000g, got {result}"


class TestTimeConversions:
    """Test time unit conversions.

    Reference:
    - 1 hour = 3600 seconds
    - 1 day = 86400 seconds
    """

    def test_seconds_to_hours(self, unit_converter, conversion_context):
        """Test 3600 s = 1 hour."""
        result = unit_converter.convert(
            3600.0,
            'time',
            's',
            'hour',
            conversion_context
        )
        assert abs(result - 1.0) < 1e-6, \
            f"3600s should be 1 hour, got {result}"

    def test_hours_to_seconds(self, unit_converter, conversion_context):
        """Test 1 hour = 3600 s."""
        result = unit_converter.convert(
            1.0,
            'time',
            'hour',
            's',
            conversion_context
        )
        assert abs(result - 3600.0) < 1e-6, \
            f"1 hour should be 3600s, got {result}"

    def test_days_to_seconds(self, unit_converter, conversion_context):
        """Test 1 day = 86400 s."""
        result = unit_converter.convert(
            1.0,
            'time',
            'day',
            's',
            conversion_context
        )
        assert abs(result - 86400.0) < 1e-6, \
            f"1 day should be 86400s, got {result}"


class TestPressureConversions:
    """Test pressure unit conversions.

    Reference:
    - 1 kPa = 1000 Pa
    - 1 atm = 101325 Pa
    """

    def test_pascals_to_kilopascals(self, unit_converter, conversion_context):
        """Test 1000 Pa = 1 kPa."""
        result = unit_converter.convert(
            1000.0,
            'pressure',
            'Pa',
            'kPa',
            conversion_context
        )
        assert abs(result - 1.0) < 1e-6, \
            f"1000Pa should be 1kPa, got {result}"

    def test_kilopascals_to_pascals(self, unit_converter, conversion_context):
        """Test 1 kPa = 1000 Pa."""
        result = unit_converter.convert(
            1.0,
            'pressure',
            'kPa',
            'Pa',
            conversion_context
        )
        assert abs(result - 1000.0) < 1e-6, \
            f"1kPa should be 1000Pa, got {result}"


class TestFluxConversions:
    """Test flux unit conversions.

    Reference:
    - Carbon flux: g C m-2 s-1 <-> umol C m-2 s-1
    - Molecular weight of C: 12 g/mol
    - 1 mol = 1e6 umol
    - Conversion: umol = g * (1e6 / 12)

    Mathematical expression:
    flux(umol/m2/s) = flux(g/m2/s) * (1e6 umol/mol) / (12 g/mol)
    """

    def test_carbon_flux_g_to_umol(self, unit_converter, conversion_context):
        """Test carbon flux conversion g C m-2 s-1 to umol C m-2 s-1.

        Using accurate carbon molar mass: M_C = 12.011 g/mol
        1 g C m-2 s-1 = 1e6/12.011 umol C m-2 s-1 = 83,257.0 umol C m-2 s-1
        """
        result = unit_converter.convert(
            1.0,
            'carbon_flux',
            'g m-2 s-1',
            'umol m-2 s-1',
            conversion_context
        )
        carbon_molar_mass = get_physical_constant(
            "relic.physical.constants.molecular_weights.carbon"
        )
        expected = 1e6 / carbon_molar_mass
        assert abs(result - expected) / expected < 1e-4, \
            f"1 g C m-2 s-1 should be {expected} umol C m-2 s-1, got {result}"

    def test_carbon_flux_umol_to_g(self, unit_converter, conversion_context):
        """Test carbon flux conversion umol C m-2 s-1 to g C m-2 s-1.

        Using accurate carbon molar mass: M_C = 12.011 g/mol
        83,257.0 umol C m-2 s-1 = 1.0 g C m-2 s-1
        """
        carbon_molar_mass = get_physical_constant(
            "relic.physical.constants.molecular_weights.carbon"
        )
        input_umol = 1e6 / carbon_molar_mass
        
        result = unit_converter.convert(
            input_umol,
            'carbon_flux',
            'umol m-2 s-1',
            'g m-2 s-1',
            conversion_context
        )
        assert abs(result - 1.0) < 1e-4, \
            f"{input_umol} umol C m-2 s-1 should be 1.0 g C m-2 s-1, got {result}"


class TestNitrogenConversions:
    """Test element-specific nitrogen mass and amount conversions."""

    def test_nitrogen_pool_kg_to_g_roundtrip(
        self,
        unit_converter,
        conversion_context,
    ):
        mass_kg = torch.tensor([0.001, 0.002])

        mass_g = unit_converter.convert(
            mass_kg,
            "leaf_nitrogen",
            "kg N m-2",
            "g N m-2",
            conversion_context,
        )
        restored = unit_converter.convert(
            mass_g,
            "leaf_nitrogen",
            "g N m-2",
            "kg N m-2",
            conversion_context,
        )

        assert torch.allclose(mass_g, torch.tensor([1.0, 2.0]))
        assert torch.allclose(restored, mass_kg)

    @pytest.mark.parametrize(
        "variable_name",
        [
            "fertilizer_n_flux",
            "leaf_litterfall_nitrogen",
            "froot_litterfall_nitrogen",
        ],
    )
    def test_nitrogen_flux_kg_to_g_roundtrip(
        self,
        variable_name,
        unit_converter,
        conversion_context,
    ):
        flux_kg = torch.tensor([1.0e-6, 2.0e-6])

        flux_g = unit_converter.convert(
            flux_kg,
            variable_name,
            "kg N m-2 s-1",
            "g N m-2 s-1",
            conversion_context,
        )
        restored = unit_converter.convert(
            flux_g,
            variable_name,
            "g N m-2 s-1",
            "kg N m-2 s-1",
            conversion_context,
        )

        assert torch.allclose(flux_g, torch.tensor([0.001, 0.002]))
        assert torch.allclose(restored, flux_kg)

    @pytest.mark.parametrize(
        "variable_name",
        [
            "organic_nitrogen",
            "microbial_nitrogen",
            "nh4_nitrogen",
            "no3_nitrogen",
        ],
    )
    def test_nitrogen_pool_mass_to_amount_roundtrip(
        self,
        variable_name,
        unit_converter,
        conversion_context,
    ):
        nitrogen_molar_mass = get_physical_constant(
            "relic.physical.constants.molecular_weights.nitrogen"
        )
        mass = torch.tensor([nitrogen_molar_mass, 2.0 * nitrogen_molar_mass])

        amount = unit_converter.convert(
            mass,
            variable_name,
            "g N m-2",
            "mol N m-2",
            conversion_context,
        )
        restored = unit_converter.convert(
            amount,
            variable_name,
            "mol N m-2",
            "g N m-2",
            conversion_context,
        )

        assert torch.allclose(amount, torch.tensor([1.0, 2.0]), atol=1e-6)
        assert torch.allclose(restored, mass, atol=1e-6)

    @pytest.mark.parametrize(
        "variable_name",
        [
            "mineralization_flux",
            "gross_mineralization_flux",
            "immobilization_flux",
            "microbial_death_flux",
            "nitrification_flux",
            "denitrification_flux",
        ],
    )
    def test_nitrogen_rate_mass_to_amount_roundtrip(
        self,
        variable_name,
        unit_converter,
        conversion_context,
    ):
        nitrogen_molar_mass = get_physical_constant(
            "relic.physical.constants.molecular_weights.nitrogen"
        )
        mass_rate = torch.tensor([nitrogen_molar_mass, 0.5 * nitrogen_molar_mass])

        amount_rate = unit_converter.convert(
            mass_rate,
            variable_name,
            "g N m-2 s-1",
            "mol N m-2 s-1",
            conversion_context,
        )
        restored = unit_converter.convert(
            amount_rate,
            variable_name,
            "mol N m-2 s-1",
            "g N m-2 s-1",
            conversion_context,
        )

        assert torch.allclose(amount_rate, torch.tensor([1.0, 0.5]), atol=1e-6)
        assert torch.allclose(restored, mass_rate, atol=1e-6)

    def test_nitrogen_label_converts_with_the_nitrogen_not_the_water_molar_mass(
        self,
        unit_converter,
        conversion_context,
    ):
        """The element token selects M_N; a generic name never selects water."""
        molar_mass = get_physical_constant("relic.physical.constants.molecular_weights.nitrogen")

        converted = unit_converter.convert(
            1.0,
            "generic",
            "g N m-2 s-1",
            "mol N m-2 s-1",
            conversion_context,
        )

        assert converted == pytest.approx(1.0 / molar_mass, rel=1e-12)

    def test_unknown_molar_flux_is_not_assumed_to_be_water(
        self,
        unit_converter,
        conversion_context,
    ):
        with pytest.raises(ValueError, match="Unsupported conversion"):
            unit_converter.convert(
                1.0,
                "unknown_species_flux",
                "mol m-2 s-1",
                "kg m-2 s-1",
                conversion_context,
            )


class TestElementMassScaleConversions:
    """Test exact mass-scale conversions used by elemental ledgers."""

    @pytest.mark.parametrize(
        "variable_name, source_units, target_units",
        [
            ("fast_som_c", "g C m-2", "kg C m-2"),
            ("available_phosphorus", "kg P m-2", "g P m-2"),
            (
                "fertilizer_p_flux",
                "kg P m-2 s-1",
                "g P m-2 s-1",
            ),
        ],
    )
    def test_element_mass_scale_roundtrip(
        self,
        variable_name,
        source_units,
        target_units,
        unit_converter,
        conversion_context,
    ):
        source = torch.tensor([1.0, 2.0])

        converted = unit_converter.convert(
            source,
            variable_name,
            source_units,
            target_units,
            conversion_context,
        )
        restored = unit_converter.convert(
            converted,
            variable_name,
            target_units,
            source_units,
            conversion_context,
        )

        expected = (
            torch.tensor([0.001, 0.002])
            if source_units.startswith("g ")
            else torch.tensor([1000.0, 2000.0])
        )
        assert torch.allclose(converted, expected)
        assert torch.allclose(restored, source)

    @pytest.mark.parametrize(
        "variable_name, mass_units, amount_units, molar_mass_key",
        [
            (
                "carbon_amount",
                "g C m-2",
                "mol C m-2",
                "relic.physical.constants.molecular_weights.carbon",
            ),
            (
                "phosphorus_amount",
                "g P m-2",
                "mol P m-2",
                "relic.physical.constants.molecular_weights.phosphorus",
            ),
            (
                "carbon_flux",
                "g C m-2 s-1",
                "mol C m-2 s-1",
                "relic.physical.constants.molecular_weights.carbon",
            ),
            (
                "phosphorus_flux",
                "g P m-2 s-1",
                "mol P m-2 s-1",
                "relic.physical.constants.molecular_weights.phosphorus",
            ),
        ],
    )
    def test_element_amount_mass_roundtrip(
        self,
        variable_name,
        mass_units,
        amount_units,
        molar_mass_key,
        unit_converter,
        conversion_context,
    ):
        amount = torch.tensor([1.0, 2.0])
        molar_mass = float(get_physical_constant(molar_mass_key))

        mass = unit_converter.convert(
            amount,
            variable_name,
            amount_units,
            mass_units,
            conversion_context,
        )
        restored = unit_converter.convert(
            mass,
            variable_name,
            mass_units,
            amount_units,
            conversion_context,
        )

        assert torch.allclose(
            mass,
            torch.tensor([molar_mass, 2.0 * molar_mass]),
        )
        assert torch.allclose(restored, amount)

    def test_element_budget_units_have_one_canonical_owner(self):
        assert UnitConverter.get_element_budget_units("carbon") == {
            "storage": "g C m-2",
            "flux": "g C m-2 s-1",
            "amount_variable": "carbon_amount",
            "flux_variable": "carbon_flux",
        }
        assert UnitConverter.get_element_budget_units("nitrogen")["storage"] == (
            "g N m-2"
        )
        assert UnitConverter.get_element_budget_units("phosphorus")["flux"] == (
            "g P m-2 s-1"
        )
        with pytest.raises(ValueError, match="Unsupported conservation element"):
            UnitConverter.get_element_budget_units("sulfur")

    def test_unknown_compound_cannot_use_elemental_amount_projection(
        self,
        unit_converter,
        conversion_context,
    ):
        with pytest.raises(ValueError, match="Unsupported conversion"):
            unit_converter.convert(
                1.0,
                "monoterpenes_emission",
                "mol monoterpene m-2 s-1",
                "g C m-2 s-1",
                conversion_context,
            )


class TestSpeciesSensitiveDispatch:
    """Require an explicit canonical variable for water-specific conversions."""

    def test_unknown_amount_flux_is_not_assumed_to_be_carbon(
        self,
        unit_converter,
        conversion_context,
    ):
        assert unit_converter.can_convert(
            "umol m-2 s-1",
            "kg C m-2 s-1",
            variable_name="unknown_species_flux",
        ) is False
        with pytest.raises(ValueError, match="Unsupported conversion"):
            unit_converter.convert(
                1.0,
                "unknown_species_flux",
                "umol m-2 s-1",
                "kg C m-2 s-1",
                conversion_context,
            )

    @pytest.mark.parametrize(
        "source_units,target_units,context",
        [
            (
                "m3 m-3",
                "kg m-2",
                ConversionContext(soil_depth=1.0, water_density=1000.0),
            ),
            (
                "kg m-2",
                "m3 m-3",
                ConversionContext(soil_depth=1.0, water_density=1000.0),
            ),
            (
                "mm day-1",
                "kg m-2 s-1",
                ConversionContext(water_density=1000.0),
            ),
            (
                "kg m-2 s-1",
                "mm day-1",
                ConversionContext(water_density=1000.0),
            ),
        ],
    )
    def test_generic_water_conversion_is_rejected(
        self,
        source_units,
        target_units,
        context,
        unit_converter,
    ):
        assert unit_converter.can_convert(
            source_units,
            target_units,
            variable_name="generic",
        ) is False
        with pytest.raises(ValueError, match="Unsupported conversion"):
            unit_converter.convert(
                1.0,
                "generic",
                source_units,
                target_units,
                context,
            )


class TestConversionPrecision:
    """Test numerical precision of conversions."""

    def test_roundtrip_temperature(self, unit_converter, conversion_context):
        """Test C -> K -> C roundtrip preserves value."""
        original = 25.0
        to_k = unit_converter.convert(
            original,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )
        back_to_c = unit_converter.convert(
            to_k,
            'air_temperature',
            'K',
            'celsius',
            conversion_context
        )
        assert abs(back_to_c - original) < 1e-10, \
            f"Roundtrip failed: {original} -> {to_k} -> {back_to_c}"

    def test_roundtrip_length(self, unit_converter, conversion_context):
        """Test m -> km -> m roundtrip preserves value."""
        original = 1234.5
        to_km = unit_converter.convert(
            original,
            'length',
            'm',
            'km',
            conversion_context
        )
        back_to_m = unit_converter.convert(
            to_km,
            'length',
            'km',
            'm',
            conversion_context
        )
        assert abs(back_to_m - original) < 1e-8, \
            f"Roundtrip failed: {original} -> {to_km} -> {back_to_m}"

    def test_relative_error_small_values(self, unit_converter, conversion_context):
        """Test relative error for small values is acceptable."""
        small_value = 1e-6
        result = unit_converter.convert(
            small_value,
            'length',
            'm',
            'km',
            conversion_context
        )
        expected = 1e-9
        relative_error = abs(result - expected) / expected
        assert relative_error < 1e-5, \
            f"Relative error too large: {relative_error}"


class TestSameDimensionCFScaleConversions:
    """UDUNITS-convertible, same-dimension pairs scale through the one owner.

    The CF standard-name validator accepts every unit that UDUNITS can convert
    to the canonical unit; the converter must reach that canonical unit for the
    same pairs instead of only for hand-listed element-annotated spellings.
    """

    @pytest.mark.parametrize(
        ("variable_name", "source_units", "target_units", "value", "expected"),
        [
            ("precipitation_flux", "kg m-2 day-1", "kg m-2 s-1", 8.64, 1.0e-4),
            (
                "gross_primary_productivity_of_carbon",
                "g m-2 day-1",
                "kg m-2 s-1",
                8.64,
                1.0e-7,
            ),
            ("generic", "kg m-2 s-1", "kg m-2 day-1", 1.0e-4, 8.64),
            ("air_pressure", "hPa", "kPa", 1013.25, 101.325),
        ],
    )
    def test_same_dimension_pairs_scale_exactly(
        self,
        variable_name,
        source_units,
        target_units,
        value,
        expected,
    ):
        converter = UnitConverter()

        assert converter.can_convert(source_units, target_units, variable_name) is True
        result = converter.convert(
            torch.tensor([value], dtype=torch.float64),
            variable_name,
            source_units,
            target_units,
            ConversionContext(),
        )

        torch.testing.assert_close(
            result,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1.0e-12,
            atol=0.0,
        )

    def test_same_dimension_scale_keeps_the_autograd_graph(self):
        values = torch.tensor([8.64, 17.28], dtype=torch.float64, requires_grad=True)

        converted = UnitConverter().convert(
            values,
            "precipitation_flux",
            "kg m-2 day-1",
            "kg m-2 s-1",
            ConversionContext(),
        )
        converted.sum().backward()

        torch.testing.assert_close(
            values.grad,
            torch.full_like(values, 1.0 / 86400.0),
            rtol=1.0e-12,
            atol=0.0,
        )

    @pytest.mark.parametrize(
        ("source", "target", "factor"),
        [
            ("kPa^0.5", "Pa^0.5", 1000.0 ** 0.5),
            ("kPa**0.5", "Pa**0.5", 1000.0 ** 0.5),
            ("kPa0.5", "Pa0.5", 1000.0 ** 0.5),
            ("kPa0.5 m", "kg0.5 m0.5 s-1", 1000.0 ** 0.5),
            ("K day-0.5", "K s-0.5", 86400.0 ** -0.5),
            ("m s-0.5", "m hour-0.5", 3600.0 ** 0.5),
            ("1e200 kPa^0.5", "1e200 Pa^0.5", 1000.0 ** 0.5),
            ("1e-200 kPa^0.5", "1e-200 Pa^0.5", 1000.0 ** 0.5),
            ("1e200 kPa^0.5", "2e200 Pa^0.5", 1000.0 ** 0.5 / 2),
        ],
    )
    def test_fractional_power_scales_round_trip_and_preserve_gradients(self, source, target, factor):
        converter = UnitConverter()
        values = torch.tensor([4.0, 8.0], dtype=torch.float64, requires_grad=True)
        assert converter.can_convert(source, target, "generic")
        converted = converter.convert(values, "generic", source, target)
        torch.testing.assert_close(converted, values * factor, rtol=1e-12, atol=0)
        restored = converter.convert(converted, "generic", target, source)
        torch.testing.assert_close(restored, values, rtol=1e-12, atol=0)
        converted.sum().backward()
        torch.testing.assert_close(values.grad, torch.full_like(values, factor), rtol=1e-12, atol=0)

    def test_fractional_power_of_affine_temperature_is_not_a_pure_scale(self):
        converter = UnitConverter()
        assert not converter.can_convert("degC0.5", "K0.5", "generic")
        with pytest.raises(ValueError, match="Unsupported conversion"):
            converter.convert(4.0, "generic", "degC0.5", "K0.5")

    @pytest.mark.parametrize(
        ("source_units", "target_units"),
        [
            ("g C m-2 s-1", "kg m-2 s-1"),  # element annotation is not a CF unit
            ("m s-1", "kg m-2 s-1"),        # different dimension needs density
            ("degF", "K"),                  # affine pairs need an explicit rule
            ("unknown_unit", "kg"),
        ],
    )
    def test_pairs_outside_the_scale_rule_stay_rejected(self, source_units, target_units):
        converter = UnitConverter()

        assert converter.can_convert(source_units, target_units, "generic") is False
        with pytest.raises(ValueError, match="Unsupported conversion|Cross-dimension"):
            converter.convert(
                1.0,
                "generic",
                source_units,
                target_units,
                ConversionContext(),
            )


class TestLiquidWaterEquivalentConversions:
    """CF liquid-water-equivalent rule: F = rho_w * d, rho_w from the physical-constant SSOT.

    CF standard-name table v94: "lwe" means liquid water equivalent, and
    lwe_thickness_of_X means "the vertical extent of a layer of liquid water having
    the same mass per unit area". The rule applies only to variables declared as
    water quantities, never to a unit pair alone.
    """

    @staticmethod
    def _rho_w():
        return get_physical_constant("relic.physical.constants.water.density_liquid")

    @pytest.mark.parametrize(
        "variable_name",
        [
            "precipitation_flux",
            "rainfall_flux",
            "evaporation_flux",
            "evapotranspiration_flux",
            "runoff_flux",
            "surface_runoff_flux",
            "infiltration_flux",
            "drainage_flux",
            "lwe_precipitation_rate",
            "lwe_water_evaporation_rate",
        ],
    )
    def test_five_mm_per_day_is_five_over_86400_kg_per_m2_s(self, unit_converter, variable_name):
        seconds_per_day = get_physical_constant("seconds_per_day")
        converted = unit_converter.convert(
            torch.tensor([5.0], dtype=torch.float64),
            variable_name,
            "mm day-1",
            "kg m-2 s-1",
        )

        expected = 5.0e-3 * self._rho_w() / seconds_per_day
        assert torch.allclose(converted, torch.tensor([expected], dtype=torch.float64), rtol=1e-12)
        assert expected == pytest.approx(5.0 / 86400.0, rel=1e-12)

    @pytest.mark.parametrize(
        ("source_units", "target_units", "value", "expected"),
        [
            ("m s-1", "kg m-2 s-1", 1.0e-6, 1.0e-3),
            ("mm s-1", "kg m-2 s-1", 2.0, 2.0),
            ("mm hr-1", "kg m-2 s-1", 3.6, 1.0e-3),
            ("kg m-2 s-1", "mm day-1", 1.0 / 86400.0, 1.0),
            ("kg m-2 s-1", "m s-1", 1.0e-3, 1.0e-6),
            ("kg m-2 day-1", "mm day-1", 4.0, 4.0),
        ],
    )
    def test_depth_rate_and_mass_flux_forms_convert_both_ways(
        self, unit_converter, source_units, target_units, value, expected
    ):
        converted = unit_converter.convert(
            torch.tensor([value], dtype=torch.float64),
            "runoff_flux",
            source_units,
            target_units,
        )

        assert torch.allclose(converted, torch.tensor([expected], dtype=torch.float64), rtol=1e-12)
        assert unit_converter.can_convert(source_units, target_units, variable_name="runoff_flux")

    def test_state_varying_density_is_not_used(self, unit_converter):
        context = ConversionContext(water_density=997.0)

        converted = unit_converter.convert(
            torch.tensor([5.0], dtype=torch.float64),
            "precipitation_flux",
            "mm day-1",
            "kg m-2 s-1",
            context,
        )

        assert converted.item() == pytest.approx(5.0e-3 * self._rho_w() / 86400.0, rel=1e-12)

    def test_conversion_keeps_the_autograd_graph(self, unit_converter):
        depth = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)

        flux = unit_converter.convert(depth, "infiltration_flux", "mm s-1", "kg m-2 s-1")
        flux.sum().backward()

        assert torch.allclose(depth.grad, torch.full((2,), 1.0e-3 * self._rho_w(), dtype=torch.float64))

    @pytest.mark.parametrize(
        "variable_name",
        ["generic", "external_model", "tracer_concentration_advective_boundary_flux", "wind_speed"],
    )
    def test_a_unit_pair_alone_does_not_declare_liquid_water(self, unit_converter, variable_name):
        assert not unit_converter.can_convert("mm day-1", "kg m-2 s-1", variable_name=variable_name)
        with pytest.raises(ValueError, match="Unsupported conversion"):
            unit_converter.convert(1.0, variable_name, "mm day-1", "kg m-2 s-1")

    @pytest.mark.parametrize(
        "variable_name",
        [
            "water_flux",
            "baseflow_flux",
            "interflow_flux",
            "capillary_rise_flux",
            "lateral_inflow",
            "groundwater_pumping",
            "snow_melt_flux",
            "rain_flux",
            "snowfall_flux",
            "sublimation_flux",
            "water_sublimation_flux",
        ],
    )
    def test_hydrology_water_variables_use_the_lwe_rule(self, unit_converter, variable_name):
        rate = torch.tensor([2.0, 0.5], dtype=torch.float64)

        flux = unit_converter.convert(rate, variable_name, "mm s-1", "kg m-2 s-1")
        back = unit_converter.convert(flux, variable_name, "kg m-2 s-1", "mm s-1")

        expected = rate * 1.0e-3 * self._rho_w()
        assert torch.allclose(flux, expected, rtol=1e-15, atol=0.0)
        assert torch.allclose(back, rate, rtol=1e-15, atol=0.0)

    def test_the_second_depth_rate_path_is_removed(self):
        for removed in (
            "depth_rate_mm_to_mass_flux",
            "mass_flux_to_depth_rate_mm",
            "mass_flux_to_velocity",
        ):
            assert not hasattr(UnitConverter, removed)

    def test_one_rule_family_replaces_the_context_density_depth_rules(self):
        for removed in (
            "external_precipitation_mm_to_flux",
            "external_precipitation_mm_hr_to_flux",
            "flux_to_precipitation_mm",
            "external_velocity_to_flux",
        ):
            assert not hasattr(UnitConverter, removed)
        depth_rate_rows = [
            key
            for key in UnitConverter.UNIT_CONVERSIONS
            if "mm day-1" in key[1:] or "mm hr-1" in key[1:] or key[1:] == ("m s-1", "kg m-2 s-1")
        ]
        assert depth_rate_rows == []


class TestOutputUnitConfig:
    """Verify that output conversion delegates to the canonical engine."""

    def test_converts_carbon_flux_and_updates_unit_metadata(self):
        config = OutputUnitConfig(
            {"gross_primary_productivity": "g C m-2 day-1"}
        )

        converted = config.convert_output(
            {"gross_primary_productivity": torch.tensor(1.0)},
            {"gross_primary_productivity": "kg C m-2 s-1"},
        )

        assert converted["gross_primary_productivity"].item() == pytest.approx(
            86_400_000.0
        )
        assert (
            converted["gross_primary_productivity_units"]
            == "g C m-2 day-1"
        )

    def test_rejects_unsupported_conversion_instead_of_silent_no_op(self):
        config = OutputUnitConfig(
            {"gross_primary_productivity": "unsupported_unit"}
        )

        with pytest.raises(ValueError, match="Unsupported output conversion"):
            config.convert_output(
                {"gross_primary_productivity": torch.tensor(1.0)},
                {"gross_primary_productivity": "kg C m-2 s-1"},
            )

    def test_rejects_missing_source_unit_instead_of_silent_pass_through(self):
        config = OutputUnitConfig(
            {"gross_primary_productivity": "g C m-2 day-1"}
        )

        with pytest.raises(ValueError, match="Missing internal units"):
            config.convert_output(
                {"gross_primary_productivity": torch.tensor(1.0)},
                {},
            )

    def test_rejects_conflicting_source_unit_metadata(self):
        config = OutputUnitConfig(
            {"gross_primary_productivity": "g C m-2 day-1"}
        )

        with pytest.raises(ValueError, match="Conflicting internal units"):
            config.convert_output(
                {
                    "gross_primary_productivity": torch.tensor(1.0),
                    "gross_primary_productivity_units": "W m-2",
                },
                {"gross_primary_productivity": "kg C m-2 s-1"},
            )

    def test_same_unit_path_publishes_explicit_unit_metadata(self):
        config = OutputUnitConfig(
            {"gross_primary_productivity": "kg C m-2 s-1"}
        )

        converted = config.convert_output(
            {"gross_primary_productivity": torch.tensor(1.0)},
            {"gross_primary_productivity": "kg C m-2 s-1"},
        )

        assert converted["gross_primary_productivity"].item() == 1.0
        assert (
            converted["gross_primary_productivity_units"]
            == "kg C m-2 s-1"
        )

    def test_default_and_retained_presets_only_expose_executable_units(self):
        representatives = {
            "temperature": ("air_temperature", "K"),
            "carbon_flux": (
                "gross_primary_productivity",
                "kg C m-2 s-1",
            ),
            "nitrogen_flux": ("mineralization_flux", "kg N m-2 s-1"),
            "water_flux": ("evapotranspiration_flux", "kg m-2 s-1"),
            "energy_flux": ("net_radiation", "W m-2"),
            "concentration": ("co2_concentration", "mol mol-1"),
            "pressure": ("atmospheric_pressure", "Pa"),
        }
        configurations = {
            "default": OutputUnitConfig(),
            **OutputUnitConfig.get_common_presets(),
        }

        assert "annual_budgets" not in configurations
        for config in configurations.values():
            for category, target_unit in config.config.items():
                if category not in representatives:
                    continue
                variable_name, source_unit = representatives[category]
                assert unit_converter.can_convert(
                    source_unit,
                    target_unit,
                    variable_name=variable_name,
                ), f"{category}: {source_unit} -> {target_unit}"


class TestStateUnitValidatorExpectedUnits:
    """Require input validation to honor the caller unit contract."""

    def test_rejects_caller_expected_unit_mismatch(self):
        validator = StateUnitValidator(strict=True)
        state = {
            "custom_flux": torch.tensor(1.0),
            "custom_flux_units": "W m-2",
        }

        with pytest.raises(
            ValueError,
            match="custom_flux has unit W m-2, expected kg m-2 s-1",
        ):
            validator.validate_input(
                state,
                {"custom_flux": "kg m-2 s-1"},
            )

    def test_rejects_missing_unit_metadata_for_expected_input(self):
        validator = StateUnitValidator(strict=True)

        with pytest.raises(
            ValueError,
            match="custom_flux missing unit metadata",
        ):
            validator.validate_input(
                {"custom_flux": torch.tensor(1.0)},
                {"custom_flux": "kg m-2 s-1"},
            )

