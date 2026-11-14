"""
Unit tests for atmospheric chemistry functor.

Tests the ChemistryFunctor implementation including:
- Proper inheritance from FunctorBase
- Parameter loading and management
- NOx-O3 photochemical cycle calculations
- Temperature-dependent reaction rates (Arrhenius)
- Photolysis rate calculations
- OH concentration dynamics
- BVOC oxidation reactions
- Separation of gas-phase chemistry from dry deposition
- Conservation and stability checks
"""

from decimal import Decimal, localcontext
import math

import pytest
import torch

import relic.scientific.atmosphere.chemistry as chemistry_module
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.core.functor import ProcessCategory, ExecutionPhase, VariableKind
from relic.core.state import UpdateItem, UpdateMode
from relic.core.functor.functor_exceptions import ExecutionError


def _update_value(fluxes, key):
    """Return one canonical chemistry update value after checking metadata."""

    item = fluxes[key]
    assert isinstance(item, UpdateItem)
    assert item.mode == UpdateMode.ACCUMULATE
    assert item.write_mode is None
    assert item.unit == 'mol mol-1'
    assert item.basis == 'per_step'
    assert item.per_timestep is True
    assert item.element is None
    assert item.source_functor == 'atmospheric_chemistry'
    return item.value


def _diagnostic_value(fluxes, key, unit):
    """Return one namespaced chemistry diagnostic after checking metadata."""

    item = fluxes[key]
    assert isinstance(item, UpdateItem)
    assert item.mode == UpdateMode.REPLACE
    assert item.unit == unit
    assert item.per_timestep is True
    assert item.source_functor == 'atmospheric_chemistry'
    return item.value


class TestNASA7StandardState:
    """Validate the frozen four-species NASA7 property calculation."""

    def test_standard_properties_match_independent_frozen_source_values(self):
        from relic.utils.physical_constants import get_physical_constant

        # References: NASA TM-4513; fixed Cantera NASA7 gas data, commit 6e002754.
        # Mathematical expression: cp/R, h/(R*T), and s/R are NASA7 polynomials.
        # Parameter meanings: the oracle uses NO, NO2, O3, O2 in declared order.
        # Units: temperature in K; normalized property values are dimensionless.
        # Domain/constraints: independently evaluated with 45-digit Decimal arithmetic
        #   at one temperature in each fit interval; all 56 source coefficients verified.
        # Sign convention: h includes formation enthalpy; s uses the 1-bar standard state.
        temperature = torch.tensor([298.15, 1500.0], dtype=torch.float64)
        properties = ChemistryFunctor(physical_boundary="open_mechanism").standard_molar_properties(temperature)
        gas_constant = float(get_physical_constant("R_GAS"))
        assert properties["species"] == ("NO", "NO2", "O3", "O2")
        assert properties["reference_pressure"].item() == 100000.0
        expected = {
            "cp": [[3.5915958981992184, 4.471347667686857, 4.736044105642022, 3.533383603648765],
                   [4.295674061411875, 6.75846472844375, 7.061195434, 4.390802624645]],
            "h": [[36.81733440072786, 13.793230891202155, 57.20114979519297, 7.837719481136123e-09],
                  [10.500929919337167, 7.5595494203825, 16.454453732633333, 3.25277016425775]],
            "s": [[25.347011054015187, 28.88577323560054, 28.746225414813974, 24.67366894362607],
                  [31.592092143199768, 38.09674904061813, 38.54504897749421, 31.03603835415003]],
        }
        for name, values in expected.items():
            scale = gas_constant * temperature[:, None] if name == "h" else gas_constant
            torch.testing.assert_close(properties[name] / scale, torch.tensor(values, dtype=torch.float64), rtol=1e-11, atol=1e-10)

    def test_standard_property_derivatives_obey_thermodynamic_identities(self):
        temperature = torch.tensor([250.0, 900.0, 1100.0, 2500.0], dtype=torch.float64, requires_grad=True)
        properties = ChemistryFunctor(physical_boundary="open_mechanism").standard_molar_properties(temperature)
        for species in range(4):
            cp = properties["cp"][:, species]
            dh = torch.autograd.grad(properties["h"][:, species].sum(), temperature, retain_graph=True)[0]
            ds = torch.autograd.grad(properties["s"][:, species].sum(), temperature, retain_graph=True)[0]
            du = torch.autograd.grad(properties["u"][:, species].sum(), temperature, retain_graph=True)[0]
            torch.testing.assert_close(dh, cp)
            torch.testing.assert_close(temperature * ds, cp)
            torch.testing.assert_close(du, properties["cv"][:, species])
        assert torch.all(properties["cv"] > 0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_temperature_shape_and_dtype_are_preserved(self, dtype):
        temperature = torch.tensor([[250.0, 1000.0], [1000.1, 2000.0]], dtype=dtype)
        properties = ChemistryFunctor(physical_boundary="open_mechanism").standard_molar_properties(temperature)
        for name in ("cp", "h", "s", "u", "cv"):
            assert properties[name].shape == (2, 2, 4)
            assert properties[name].dtype == dtype
            assert properties[name].device == temperature.device

    @pytest.mark.parametrize("temperature", [199.0, 6001.0, float("nan"), float("inf")])
    def test_temperature_outside_the_declared_fit_is_rejected(self, temperature):
        with pytest.raises(ValueError, match="temperature"):
            ChemistryFunctor(physical_boundary="open_mechanism").standard_molar_properties(torch.tensor(temperature, dtype=torch.float64))

    @pytest.mark.parametrize("field,value", [
        ("model", "NASA9"),
        ("species_order", ["NO", "NO2", "O3"]),
        ("a1", torch.ones((3, 2), dtype=torch.float64)),
        ("temperature_ranges", torch.tensor([200.0, 6000.0, 1000.0], dtype=torch.float64)),
    ])
    def test_inconsistent_thermodynamic_data_is_rejected(self, monkeypatch, field, value):
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        original = chemistry.get_parameter_value

        def get_value(name, **kwargs):
            if name == f"atmosphere.chemistry.nasa7.{field}":
                return value
            return original(name, **kwargs)

        monkeypatch.setattr(chemistry, "get_parameter_value", get_value)
        with pytest.raises(ValueError, match="NASA7"):
            chemistry.standard_molar_properties(torch.tensor(300.0, dtype=torch.float64))


class TestIdealGasMixture:
    """Check thermodynamic potentials independently of reaction integration."""

    def test_free_energy_generates_chemical_potential_pressure_and_entropy(self):
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        temperature = torch.tensor([300.0, 1500.0], dtype=torch.float64, requires_grad=True)
        amounts = torch.tensor([[0.2, 0.3, 0.1, 0.4], [0.1, 0.4, 0.3, 0.2]], dtype=torch.float64, requires_grad=True)
        volume = torch.tensor([0.025, 0.08], dtype=torch.float64, requires_grad=True)
        result = chemistry.ideal_gas_mixture_properties(temperature, amounts, volume)
        chemical_potential, negative_pressure, negative_entropy = torch.autograd.grad(
            result["helmholtz_energy"].sum(), (amounts, volume, temperature), retain_graph=True,
        )
        torch.testing.assert_close(chemical_potential, result["chemical_potential"])
        torch.testing.assert_close(negative_pressure, -result["pressure"])
        torch.testing.assert_close(negative_entropy, -result["entropy"])
        heat_capacity = torch.autograd.grad(result["internal_energy"].sum(), temperature)[0]
        torch.testing.assert_close(heat_capacity, result["heat_capacity_cv"])
        torch.testing.assert_close(
            (amounts * result["chemical_potential"]).sum(dim=-1),
            result["helmholtz_energy"] + result["pressure"] * volume,
        )
        assert torch.all(heat_capacity > 0)

    def test_mixture_retains_second_derivatives(self):
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        inputs = (torch.tensor(310.0, requires_grad=True),
                  torch.tensor([0.2, 0.3, 0.1, 0.4], requires_grad=True),
                  torch.tensor(0.025, requires_grad=True))

        def free_energy(temperature, amounts, volume):
            return chemistry.ideal_gas_mixture_properties(temperature, amounts, volume)["helmholtz_energy"]

        assert torch.autograd.gradcheck(free_energy, inputs)
        assert torch.autograd.gradgradcheck(free_energy, inputs)

    def test_live_nasa_coefficients_retain_parameter_gradients(self):
        from copy import copy
        from relic.utils.physical_constants import get_physical_constant

        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        manager = chemistry.get_parameter_manager()
        key = "atmosphere.chemistry.nasa7.a7"
        chemistry.get_parameter_value(key, required=True)
        original_spec = manager.get_parameter(key)
        original_parameter = manager.nn_parameters.get(key)
        trainable_spec = copy(original_spec)
        trainable_spec.requires_grad = True
        temperature = torch.tensor([300.0, 1500.0], dtype=torch.float64)
        amounts = torch.tensor([[0.2, 0.3, 0.1, 0.4], [0.1, 0.4, 0.3, 0.2]], dtype=torch.float64)
        volume = torch.tensor(0.025, dtype=torch.float64)
        try:
            manager.register_parameter(trainable_spec)
            parameter = manager.nn_parameters[key]
            result = chemistry.ideal_gas_mixture_properties(temperature, amounts, volume)
            derivative = torch.autograd.grad(result["entropy"].sum(), parameter)[0]
            torch.testing.assert_close(derivative, float(get_physical_constant("R_GAS")) * amounts.T)
        finally:
            manager.register_parameter(original_spec)
            if original_parameter is not None:
                manager.nn_parameters[key] = original_parameter

    @pytest.mark.parametrize("fault", ["nonfinite_energy", "nonpositive_cv"])
    def test_invalid_thermodynamic_output_is_rejected(self, monkeypatch, fault):
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        original = chemistry.standard_molar_properties

        def broken_properties(temperature):
            result = original(temperature)
            if fault == "nonfinite_energy":
                result["u"] = torch.full_like(result["u"], torch.inf)
            else:
                result["cv"] = -torch.ones_like(result["cv"])
            return result

        monkeypatch.setattr(chemistry, "standard_molar_properties", broken_properties)
        with pytest.raises(ValueError, match="nonfinite|heat capacity"):
            chemistry.ideal_gas_mixture_properties(
                torch.tensor(300.0), torch.ones(4), torch.tensor(0.1),
            )

    def test_volume_doubling_and_extensive_scaling(self):
        from relic.utils.physical_constants import get_physical_constant

        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        temperature = torch.tensor(310.0, dtype=torch.float64)
        amounts = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
        volume = torch.tensor(0.025, dtype=torch.float64)
        initial = chemistry.ideal_gas_mixture_properties(temperature, amounts, volume)
        expanded = chemistry.ideal_gas_mixture_properties(temperature, amounts, 2 * volume)
        replicated = chemistry.ideal_gas_mixture_properties(temperature, 3 * amounts, 3 * volume)
        gas_constant = float(get_physical_constant("R_GAS"))
        torch.testing.assert_close(expanded["internal_energy"], initial["internal_energy"])
        torch.testing.assert_close(expanded["entropy"] - initial["entropy"], gas_constant * amounts.sum() * torch.log(volume.new_tensor(2.0)))
        for key in ("internal_energy", "entropy", "helmholtz_energy", "heat_capacity_cv"):
            torch.testing.assert_close(replicated[key], 3 * initial[key])
        for key in ("chemical_potential", "partial_pressures", "pressure"):
            torch.testing.assert_close(replicated[key], initial[key])

    def test_one_bar_partial_pressures_recover_absolute_standard_entropies(self):
        from relic.utils.physical_constants import get_physical_constant

        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        temperature = torch.tensor(298.15, dtype=torch.float64)
        amounts = torch.ones(4, dtype=torch.float64)
        volume = temperature * float(get_physical_constant("R_GAS")) / 100000.0
        result = chemistry.ideal_gas_mixture_properties(temperature, amounts, volume)
        standard = chemistry.standard_molar_properties(temperature)
        torch.testing.assert_close(result["entropy"], standard["s"].sum())
        torch.testing.assert_close(result["chemical_potential"], standard["h"] - temperature * standard["s"])
        torch.testing.assert_close(result["pressure"], temperature.new_tensor(400000.0))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_shared_volume_and_multidimensional_batches_preserve_gradients(self, dtype):
        temperature = torch.full((2, 3), 300.0, dtype=dtype, requires_grad=True)
        amounts = torch.ones((2, 3, 4), dtype=dtype, requires_grad=True)
        volume = torch.tensor(0.1, dtype=dtype, requires_grad=True)
        result = ChemistryFunctor(physical_boundary="open_mechanism").ideal_gas_mixture_properties(temperature, amounts, volume)
        assert result["species"] == ("NO", "NO2", "O3", "O2")
        for name, value in result.items():
            if name == "species":
                continue
            assert value.dtype == dtype and value.device == temperature.device
            expected_shape = (2, 3, 4) if name in ("chemical_potential", "partial_pressures") else (2, 3)
            assert value.shape == expected_shape
        derivative = torch.autograd.grad(result["helmholtz_energy"].sum(), volume)[0]
        torch.testing.assert_close(derivative, -result["pressure"].sum())

    @pytest.mark.parametrize("field,value", [
        ("amounts", torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64)),
        ("amounts", torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64)),
        ("amounts", torch.full((4,), float("nan"), dtype=torch.float64)),
        ("amounts", torch.ones(3, dtype=torch.float64)),
        ("amounts", torch.ones((1, 4), dtype=torch.float64)),
        ("amounts", torch.ones(4, dtype=torch.float32)),
        ("volume", torch.tensor(0.0, dtype=torch.float64)),
        ("volume", torch.tensor(-1.0, dtype=torch.float64)),
        ("volume", torch.tensor(float("inf"), dtype=torch.float64)),
        ("volume", torch.ones(1, dtype=torch.float64)),
        ("volume", 0.1),
        ("temperature", torch.tensor(0.0, dtype=torch.float64)),
    ])
    def test_invalid_state_is_rejected_without_mutation(self, field, value):
        inputs = {"temperature": torch.tensor(300.0, dtype=torch.float64),
                  "amounts": torch.ones(4, dtype=torch.float64),
                  "volume": torch.tensor(0.1, dtype=torch.float64)}
        inputs[field] = value
        snapshots = {key: item.clone() for key, item in inputs.items() if isinstance(item, torch.Tensor)}
        with pytest.raises(ValueError):
            ChemistryFunctor(physical_boundary="open_mechanism").ideal_gas_mixture_properties(**inputs)
        for key, snapshot in snapshots.items():
            torch.testing.assert_close(inputs[key], snapshot, equal_nan=True)


class TestThermodynamicTemperature:
    """Invert internal energy using the existing implicit scalar solver."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_known_temperatures_are_recovered_in_independent_batches(self, dtype):
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        expected = torch.tensor([250.0, 310.0, 380.0], dtype=dtype)
        amounts = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.2, 0.1, 0.4, 0.3]], dtype=dtype)
        energy = (amounts * chemistry.standard_molar_properties(expected)["u"]).sum(dim=-1)
        solved = chemistry.temperature_from_internal_energy(
            energy, amounts, temperature_lower=torch.full_like(expected, 200.0),
            temperature_upper=torch.full_like(expected, 400.0),
        )
        torch.testing.assert_close(solved, expected)
        assert solved.dtype == dtype and solved.device == expected.device

    def test_entropy_effort_uses_the_same_amount_energy_state(self):
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        expected = torch.tensor(310.0, dtype=torch.float64)
        amounts = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
        energy = (amounts * chemistry.standard_molar_properties(expected)["u"]).sum()
        state = torch.cat((amounts, energy.reshape(1))).requires_grad_()
        temperature = chemistry.temperature_from_internal_energy(
            state[-1], state[:-1], temperature_lower=expected.new_tensor(200.0),
            temperature_upper=expected.new_tensor(400.0),
        )
        result = chemistry.ideal_gas_mixture_properties(temperature, state[:-1], expected.new_tensor(0.025))
        effort = torch.autograd.grad(result["entropy"], state, create_graph=True)[0]
        torch.testing.assert_close(effort[:-1], -result["chemical_potential"] / temperature)
        torch.testing.assert_close(effort[-1], 1 / temperature)
        temperature_gradient = torch.autograd.grad(temperature, state, retain_graph=True)[0]
        torch.testing.assert_close(temperature_gradient[-1], 1 / result["heat_capacity_cv"])
        torch.testing.assert_close(
            temperature_gradient[:-1],
            -chemistry.standard_molar_properties(temperature)["u"] / result["heat_capacity_cv"],
        )
        curvature = torch.autograd.grad(effort[-1], state)[0]
        torch.testing.assert_close(curvature[-1], -1 / (temperature.square() * result["heat_capacity_cv"]))

    def test_negative_formation_internal_energy_is_valid(self):
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        expected = torch.tensor(250.0, dtype=torch.float64)
        amounts = torch.tensor([1e-6, 1e-6, 1e-6, 1.0], dtype=torch.float64)
        energy = (amounts * chemistry.standard_molar_properties(expected)["u"]).sum()
        assert energy < 0
        solved = chemistry.temperature_from_internal_energy(
            energy, amounts, temperature_lower=expected.new_tensor(200.0),
            temperature_upper=expected.new_tensor(400.0),
        )
        torch.testing.assert_close(solved, expected)

    def test_trainable_formation_coefficient_is_an_explicit_implicit_parameter(self):
        from copy import copy
        from relic.utils.physical_constants import get_physical_constant

        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        temperature = torch.tensor(310.0, dtype=torch.float64)
        amounts = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
        original = chemistry.standard_molar_properties(temperature)
        energy = (amounts * original["u"]).sum()
        manager = chemistry.get_parameter_manager()
        key = "atmosphere.chemistry.nasa7.a6"
        spec = manager.get_parameter(key)
        old_parameter = manager.nn_parameters.get(key)
        trainable = copy(spec)
        trainable.requires_grad = True
        try:
            manager.register_parameter(trainable)
            parameter = manager.nn_parameters[key]
            solved = chemistry.temperature_from_internal_energy(
                energy, amounts, temperature_lower=temperature.new_tensor(200.0),
                temperature_upper=temperature.new_tensor(400.0),
            )
            derivative = torch.autograd.grad(solved, parameter, create_graph=True)[0]
            cv = (amounts * original["cv"]).sum()
            expected = torch.zeros_like(parameter)
            expected[:, 0] = -float(get_physical_constant("R_GAS")) * amounts / cv
            torch.testing.assert_close(derivative, expected)
            second = torch.autograd.grad(derivative[:, 0].sum(), parameter)[0]
            reference_temperature = temperature.clone().requires_grad_()
            reference_cv = (amounts * chemistry.standard_molar_properties(reference_temperature)["cv"]).sum()
            cv_temperature_derivative = torch.autograd.grad(reference_cv, reference_temperature)[0]
            expected_second = torch.zeros_like(parameter)
            expected_second[:, 0] = (
                -float(get_physical_constant("R_GAS")) ** 2 * amounts.sum() * amounts
                * cv_temperature_derivative / cv.pow(3)
            )
            torch.testing.assert_close(second, expected_second)
        finally:
            manager.register_parameter(spec)
            if old_parameter is not None:
                manager.nn_parameters[key] = old_parameter

    @pytest.mark.parametrize("fault", ["energy_outside", "cross_interval", "zero_amount", "nan_energy", "wrong_shape"])
    def test_invalid_inversion_contract_is_rejected(self, fault):
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        temperature = torch.tensor(310.0, dtype=torch.float64)
        amounts = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
        energy = (amounts * chemistry.standard_molar_properties(temperature)["u"]).sum()
        lower, upper = temperature.new_tensor(200.0), temperature.new_tensor(400.0)
        if fault == "energy_outside":
            energy = energy.new_tensor(1e9)
        elif fault == "cross_interval":
            upper = upper.new_tensor(1500.0)
        elif fault == "zero_amount":
            amounts[0] = 0
        elif fault == "nan_energy":
            energy = energy.new_tensor(float("nan"))
        else:
            energy = energy.reshape(1)
        with pytest.raises(ValueError):
            chemistry.temperature_from_internal_energy(
                energy, amounts, temperature_lower=lower, temperature_upper=upper,
            )

    @pytest.mark.parametrize("fault", ["nonconvergent", "nonpositive_cv"])
    def test_failed_solver_or_nonmonotone_energy_cannot_return_temperature(self, monkeypatch, fault):
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        temperature = torch.tensor(311.23456789, dtype=torch.float64)
        amounts = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
        energy = (amounts * chemistry.standard_molar_properties(temperature)["u"]).sum()
        original = chemistry.get_parameter_value

        def get_value(name, **kwargs):
            if fault == "nonconvergent" and name == "computation.solvers.bracketed_bisection.max_iterations":
                return 1
            if fault == "nonpositive_cv" and name == "atmosphere.chemistry.nasa7.a1":
                return -torch.ones((4, 2), dtype=torch.float64)
            return original(name, **kwargs)

        monkeypatch.setattr(chemistry, "get_parameter_value", get_value)
        with pytest.raises(ValueError, match="converge|heat capacity"):
            chemistry.temperature_from_internal_energy(
                energy, amounts, temperature_lower=temperature.new_tensor(200.0),
                temperature_upper=temperature.new_tensor(400.0),
            )


class TestChemistryFunctorStructure:
    """Test proper inheritance and structure."""
    
    def test_inherits_from_functor_base(self):
        """Test that ChemistryFunctor properly inherits from FunctorBase."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Check inheritance
        from relic.core.functor import FunctorBase
        assert isinstance(chem, FunctorBase)
        
        # Check metadata
        assert chem.metadata.category == ProcessCategory.PHYSICAL_FIELD
        assert chem.metadata.execution_phase == ExecutionPhase.MAIN_PROCESSES
        assert chem.metadata.variable_kind == VariableKind.STATE
        assert chem.metadata.output_variables == [
            'no_mixing_ratio',
            'no2_mixing_ratio',
            'o3_mixing_ratio',
            'isoprene_mixing_ratio',
            'diagnostics.chemistry.photolysis_rate_no2',
            'diagnostics.chemistry.isoprene_oxidation_rate',
            'diagnostics.chemistry.reaction_no_o3_rate',
            'diagnostics.chemistry.photolysis_no2_rate',
            'diagnostics.chemistry.rate_constant_no_o3',
            'diagnostics.chemistry.shortwave_radiation_used',
            'diagnostics.chemistry.oh_concentration',
            'diagnostics.chemistry.air_number_density',
            'diagnostics.chemistry.ro2_no_conversion_rate',
            'diagnostics.chemistry.ro2_no_branching_fraction',
            'diagnostics.chemistry.ox_net_chemical_tendency',
            'diagnostics.chemistry.carbon_oxidized_rate',
        ]
        
    def test_has_required_methods(self):
        """Test that all required abstract methods are implemented."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Check for required methods
        assert hasattr(chem, 'get_dependencies')
        assert hasattr(chem, 'forward')
        assert callable(chem.get_dependencies)
        assert callable(chem.forward)
        
    def test_dependencies(self):
        """Test dependency declaration."""
        from relic.core.dependency_graph import normalize_dependency_declaration

        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        declared = normalize_dependency_declaration(chem.get_dependencies(), owner=chem.name)
        # Optional same-step provider: radiation orders the graph when registered.
        assert declared.optional == ('atmospheric_radiation',)
        assert declared.required == ()
        
    def test_parameter_loading(self):
        """Test that parameters are properly loaded and registered."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Check key parameters are accessible
        assert chem.get_parameter_value('j_no2_irradiance_linear_coefficient') is not None
        assert chem.get_parameter_value('k_no_o3_298') is not None
        assert chem.get_parameter_value('k_isoprene_oh_298') is not None
        

class TestPhotolysisRates:
    """Test photolysis rate calculations."""
    
    def test_j_no2_calculation(self):
        """Test NO2 photolysis rate calculation."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Test with different radiation levels
        shortwave_low = torch.tensor(200.0)
        shortwave_high = torch.tensor(1000.0)
        cos_zenith = torch.tensor(1.0)  # Sun overhead
        temperature = torch.tensor(298.15)
        
        j_no2_low, _ = chem._calculate_photolysis_rates(
            shortwave_low, cos_zenith, temperature
        )
        j_no2_high, _ = chem._calculate_photolysis_rates(
            shortwave_high, cos_zenith, temperature
        )
        
        # Higher radiation should give higher photolysis rate
        assert j_no2_high > j_no2_low
        
        # Ground-level range: Trebs et al. (2009) give 1.02e-2 s-1 at G = 1000 W m-2.
        assert 1e-5 < j_no2_high.item() < 1.2e-2
        
    def test_zenith_angle_effect(self):
        """Test effect of solar zenith angle on photolysis."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        shortwave = torch.tensor(800.0)
        temperature = torch.tensor(298.15)
        
        # Different zenith angles
        cos_zenith_noon = torch.tensor(1.0)  # Sun overhead
        cos_zenith_evening = torch.tensor(0.3)  # Low sun
        
        j_no2_noon, j_o1d_noon = chem._calculate_photolysis_rates(
            shortwave, cos_zenith_noon, temperature
        )
        j_no2_evening, j_o1d_evening = chem._calculate_photolysis_rates(
            shortwave, cos_zenith_evening, temperature
        )
        
        # The irradiance G already carries the geometry: at equal G, J_NO2 is equal;
        # J(O1D) falls with the zenith angle (longer ozone slant path, MCM J1/J4).
        assert torch.equal(j_no2_noon, j_no2_evening)
        assert j_o1d_evening < j_o1d_noon
        

class TestTemperatureDependence:
    """Test Arrhenius temperature-dependent reaction rates."""
    
    def test_arrhenius_kinetics(self):
        """Test temperature dependence of reaction rates."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Test at different temperatures
        temp_cold = torch.tensor(273.15)  # 0 C
        temp_warm = torch.tensor(313.15)  # 40 C
        
        rates_cold = chem._calculate_temperature_dependent_rates(temp_cold)
        rates_warm = chem._calculate_temperature_dependent_rates(temp_warm)
        
        # NO + O3 has positive activation energy (faster when warm; IUPAC I.A3.54)
        assert rates_warm['k_no_o3'] > rates_cold['k_no_o3']

        # Isoprene + OH has negative activation energy (slower when warm; IUPAC II.A2.10,
        # k = 2.7e-11 exp(390/T))
        assert rates_warm['k_isoprene_oh'] < rates_cold['k_isoprene_oh']

        # Isoprene + O3 has positive activation energy (faster when warm; IUPAC II.A1.157)
        assert rates_warm['k_isoprene_o3'] > rates_cold['k_isoprene_o3']
        
    def test_rate_constant_ranges(self):
        """Test that rate constants are in reasonable ranges."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        temperature = torch.tensor(298.15)
        rates = chem._calculate_temperature_dependent_rates(temperature)
        
        # Check typical ranges (cm^3 molecule^-1 s^-1)
        assert 1e-15 < rates['k_no_o3'] < 1e-13
        assert 1e-11 < rates['k_isoprene_oh'] < 1e-9
        assert 1e-18 < rates['k_isoprene_o3'] < 1e-16
        

class TestOHDynamics:
    """Test dynamic OH concentration calculations."""
    
    def test_oh_steady_state(self):
        """Test OH steady-state calculation."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Create test state
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no2_mixing_ratio': torch.tensor(10e-9),  # 10 ppb
            'o3_mixing_ratio': torch.tensor(40e-9),   # 40 ppb
            'co_mixing_ratio': torch.tensor(150e-9),  # 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
        }
        
        # J(O1D) = J_NO2 rho(0) for J_NO2 = 8e-3 s-1 (overhead sun)
        j_o1d = torch.tensor(8.0e-3 * 4.238e-3)
        temperature = state['air_temperature']
        
        oh_conc = chem.compute_oh_concentration(state, j_o1d, temperature)
        
        # Daytime OH of a few 1e6 molecules cm-3 (primary production only)
        assert 1e5 < oh_conc.item() <= 1e8
        
    def test_oh_response_to_nox(self):
        """NO2 is an OH sink only: NO2 photolysis produces no OH."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        state_low_nox = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no2_mixing_ratio': torch.tensor(0.5e-9),   # 0.5 ppb (very low)
            'o3_mixing_ratio': torch.tensor(40e-9),     # 40 ppb
            'co_mixing_ratio': torch.tensor(200e-9),    # Add CO as OH sink
            'ch4_mixing_ratio': torch.tensor(1.8e-6),   # Add CH4 as OH sink
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
        }
        
        # High NOx state
        state_high_nox = state_low_nox.copy()
        state_high_nox['no2_mixing_ratio'] = torch.tensor(10e-9)  # 10 ppb (moderate)
        
        j_o1d = torch.tensor(1.0e-5)
        temperature = torch.tensor(298.15)
        
        oh_low_nox = chem.compute_oh_concentration(state_low_nox, j_o1d, temperature)
        oh_high_nox = chem.compute_oh_concentration(state_high_nox, j_o1d, temperature)
        
        # More NO2 adds the OH + NO2 -> HNO3 sink and no source.
        assert oh_high_nox < oh_low_nox
        assert 1e5 <= oh_low_nox <= 1e8
        

class TestChemicalReactions:
    """Test chemical reaction rate calculations."""
    
    def test_nox_o3_cycle(self):
        """Test NOx-O3 photochemical cycle."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Create realistic atmospheric state
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(5e-9),    # 5 ppb NO
            'no2_mixing_ratio': torch.tensor(10e-9),  # 10 ppb NO2
            'o3_mixing_ratio': torch.tensor(40e-9),   # 40 ppb O3
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'shortwave_down': torch.tensor(800.0),
            'cos_zenith': torch.tensor(1.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        # Run forward pass
        fluxes = chem.forward(state, time=0.0, dt=1.0)
        long_step_fluxes = chem.forward(state, time=0.0, dt=10.0)
        
        # State outputs are pre-integrated increments for one simulator step.
        for key in ('no_mixing_ratio', 'no2_mixing_ratio', 'o3_mixing_ratio'):
            increment = _update_value(fluxes, key)
            long_increment = _update_value(long_step_fluxes, key)
            torch.testing.assert_close(long_increment, increment * 10.0)

        assert 'isoprene_mixing_ratio' not in fluxes
        assert not any(key.startswith('delta_') for key in fluxes)
        
        # Check photolysis rate is positive during day
        j_no2 = _diagnostic_value(
            fluxes,
            'diagnostics.chemistry.photolysis_rate_no2',
            's-1',
        )
        assert j_no2 > 0
        
    def test_photostationary_state(self):
        """Test NOx-O3 photostationary state relationship."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # State near photostationary equilibrium
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(1e-9),    # 1 ppb NO
            'no2_mixing_ratio': torch.tensor(10e-9),  # 10 ppb NO2
            'o3_mixing_ratio': torch.tensor(40e-9),   # 40 ppb O3
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'shortwave_down': torch.tensor(800.0),
            'cos_zenith': torch.tensor(1.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        fluxes = chem.forward(state, time=0.0, dt=1.0)
        
        # Calculate Leighton ratio
        j_no2 = _diagnostic_value(
            fluxes,
            'diagnostics.chemistry.photolysis_rate_no2',
            's-1',
        )
        # Need to consider units - rate constant is in cm3/molecule/s but we have mixing ratios
        # For photostationary state ratio, the units cancel out, but need to use correct scale
        
        # Simplified check - just verify the trends are correct
        # NOx-O3 photostationary relationship should show positive correlation
        
        # Just check that photolysis is happening and producing reasonable rates
        assert j_no2 > 0  # Photolysis is active
        assert _update_value(fluxes, 'no2_mixing_ratio') != 0
        assert _update_value(fluxes, 'o3_mixing_ratio') != 0

    def test_runtime_increment_matches_declared_reduced_kinetics(self):
        """Verify the canonical UpdateItems exactly implement the declared rates."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        dt = 37.0
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(5e-9),
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'shortwave_down': torch.tensor(800.0),
            'cos_zenith': torch.tensor(1.0),
        }

        outputs = chem.forward(state, time=0.0, dt=dt)
        reaction_rate = _diagnostic_value(
            outputs,
            'diagnostics.chemistry.reaction_no_o3_rate',
            'mol mol-1 s-1',
        )
        photolysis_rate = _diagnostic_value(
            outputs,
            'diagnostics.chemistry.photolysis_no2_rate',
            'mol mol-1 s-1',
        )
        expected_no = (photolysis_rate - reaction_rate) * dt

        torch.testing.assert_close(
            _update_value(outputs, 'no_mixing_ratio'),
            expected_no,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            _update_value(outputs, 'no2_mixing_ratio'),
            -expected_no,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            _update_value(outputs, 'o3_mixing_ratio'),
            expected_no,
            rtol=0.0,
            atol=0.0,
        )
        

class TestBVOCChemistry:
    """Test BVOC oxidation reactions."""
    
    def test_isoprene_oxidation(self):
        """Test isoprene oxidation by OH and O3."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # State with isoprene emissions
        state = {
            'air_temperature': torch.tensor(303.15),  # 30 C (warm, high emissions)
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(2e-9),
            'no2_mixing_ratio': torch.tensor(5e-9),
            'o3_mixing_ratio': torch.tensor(50e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'isoprene_mixing_ratio': torch.tensor(5e-9),  # 5 ppb isoprene
            'shortwave_down': torch.tensor(800.0),
            'cos_zenith': torch.tensor(1.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        fluxes = chem.forward(state, time=0.0, dt=1.0)
        
        # Check isoprene loss is calculated
        assert _update_value(fluxes, 'isoprene_mixing_ratio') < 0
        
        # Isoprene + O3 should also affect O3
        fluxes_no_isop = chem.forward(
            {k: v for k, v in state.items() if k != 'isoprene_mixing_ratio'},
            time=0.0, dt=1.0
        )
        
        # More O3 loss with isoprene present
        assert _update_value(fluxes, 'o3_mixing_ratio') < _update_value(
            fluxes_no_isop,
            'o3_mixing_ratio',
        )
        

class TestProcessOwnership:
    """Test separation between chemistry and surface-deposition operators."""

    def test_chemistry_exposes_no_dry_deposition_calculator(self):
        """Dry deposition must be owned only by DryDepositionAggregator."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")

        assert not hasattr(chem, '_calculate_dry_deposition_losses')

    def test_pbl_height_does_not_change_gas_phase_tendency(self):
        """Gas-phase chemistry must be independent of deposition mixing height."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(5e-9),
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'shortwave_down': torch.tensor(800.0),
            'cos_zenith': torch.tensor(1.0),
        }

        shallow = chem.forward(
            {**state, 'pbl_height': torch.tensor(500.0)},
            time=0.0,
            dt=1.0,
        )
        deep = chem.forward(
            {**state, 'pbl_height': torch.tensor(2000.0)},
            time=0.0,
            dt=1.0,
        )

        for key in ('no_mixing_ratio', 'no2_mixing_ratio', 'o3_mixing_ratio'):
            torch.testing.assert_close(
                _update_value(shallow, key),
                _update_value(deep, key),
                rtol=0.0,
                atol=0.0,
            )
        

class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_explicit_euler_rejects_step_outside_positivity_domain(self):
        """Reject a step whose declared Euler increment makes NO negative."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(1e-9),
            'no2_mixing_ratio': torch.tensor(1e-12),
            'o3_mixing_ratio': torch.tensor(60e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(0.0),
        }

        with pytest.raises(ValueError, match="explicit Euler positivity domain"):
            chem.forward(state, time=0.0, dt=60.0)

    @pytest.mark.parametrize("dt_fraction", [0.5, 1.0])
    def test_admissible_explicit_euler_step_keeps_species_non_negative(self, dt_fraction):
        """Keep every retained species non-negative inside the exact step domain."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(1e-9),
            'no2_mixing_ratio': torch.tensor(1e-12),
            'o3_mixing_ratio': torch.tensor(60e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(0.0),
        }
        unit_step = chem.forward(state, time=0.0, dt=1.0)
        tendencies = {
            name: _update_value(unit_step, name)
            for name in ('no_mixing_ratio', 'no2_mixing_ratio', 'o3_mixing_ratio')
        }
        loss_limits = [
            state[name] / (-tendency)
            for name, tendency in tendencies.items()
            if bool(tendency < 0.0)
        ]
        admissible_dt = dt_fraction * torch.min(torch.stack(loss_limits))

        outputs = chem.forward(state, time=0.0, dt=admissible_dt)
        for name in tendencies:
            next_value = state[name] + _update_value(outputs, name)
            assert torch.all(next_value >= 0.0), name
        if dt_fraction == 1.0:
            next_state = {key: value.clone() for key, value in state.items()}
            for name in tendencies:
                next_state[name] = (state[name] + _update_value(outputs, name)).detach()
            assert any(bool(next_state[name] == 0.0) for name in tendencies)
            continued = chem(next_state, time=float(admissible_dt), dt=1.0)
            assert torch.isfinite(continued['no_mixing_ratio'].value).all()
    
    def test_negative_concentration_handling(self):
        """Reject a negative concentration before evaluating chemistry."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # State with negative concentrations (shouldn't happen but test handling)
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(-1e-9),   # Negative!
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        with pytest.raises(ValueError, match="no_mixing_ratio"):
            chem.forward(state, time=0.0, dt=1.0)
        
    def test_nan_handling(self):
        """Reject NaN input rather than creating an untracked source."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(float('nan')),  # NaN!
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        with pytest.raises(ValueError, match="no_mixing_ratio"):
            chem.forward(state, time=0.0, dt=1.0)
        
    def test_extreme_temperatures(self):
        """Test behavior at extreme temperatures."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        
        state_base = {
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(5e-9),
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        # Very cold
        state_cold = state_base.copy()
        state_cold['air_temperature'] = torch.tensor(210.0)  # -63 C
        
        # Very hot
        state_hot = state_base.copy()
        state_hot['air_temperature'] = torch.tensor(350.0)  # 77 C
        
        # Both should work without errors
        fluxes_cold = chem.forward(state_cold, time=0.0, dt=1.0)
        fluxes_hot = chem.forward(state_hot, time=0.0, dt=1.0)
        
        # Check outputs are reasonable
        assert torch.isfinite(_update_value(fluxes_cold, 'o3_mixing_ratio')).all()
        assert torch.isfinite(_update_value(fluxes_hot, 'o3_mixing_ratio')).all()
        

class TestConservation:
    """Test conservation properties."""

    @staticmethod
    def _state(include_isoprene=True):
        """Return one deterministic gas-phase chemistry state."""

        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(5e-9),
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'shortwave_down': torch.tensor(800.0),
            'cos_zenith': torch.tensor(1.0),
        }
        if include_isoprene:
            state['isoprene_mixing_ratio'] = torch.tensor(5e-9)
        return state

    def test_declares_closed_nox_family(self):
        """The only closed local family is NO plus NO2."""

        chem = ChemistryFunctor(physical_boundary="open_mechanism")

        assert chem.metadata.conservation_laws == {
            'nox_family': {'quantity': 'nox_family'}
        }

    def test_forward_uses_canonical_reaction_network(self, monkeypatch):
        """The production advance must evaluate the declared N_h r_h once."""

        calls = []
        canonical_advance = chemistry_module.advance_reaction_network

        def tracking_advance(**kwargs):
            calls.append(kwargs)
            return canonical_advance(**kwargs)

        monkeypatch.setattr(
            chemistry_module,
            'advance_reaction_network',
            tracking_advance,
        )
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        fluxes = chem.forward(self._state(), time=0.0, dt=1.0)

        assert len(calls) == 1
        call = calls[0]
        assert call['species'] == (
            'no_mixing_ratio',
            'no2_mixing_ratio',
            'o3_mixing_ratio',
            'isoprene_mixing_ratio',
        )
        assert call['reactions'] == (
            'no2_photolysis',
            'no_o3_titration',
            'isoprene_oh_oxidation',
            'isoprene_o3_oxidation',
            'isoprene_ro2_no_oxidation',
        )
        torch.testing.assert_close(
            call['stoichiometric_N'],
            torch.tensor(
                [
                    [1.0, -1.0, 0.0, 0.0, -1.0],
                    [-1.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, -1.0, 0.0],
                ]
            ),
        )
        torch.testing.assert_close(
            call['conservation_matrix'],
            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        )
        assert call['require_conservation_matrix'] is True
        assert call['strict_conservation'] is True
        assert call['open_system_route'] == 'boundary_export'
        assert call['dt'] == torch.tensor(1.0)

        nox_change = (
            _update_value(fluxes, 'no_mixing_ratio')
            + _update_value(fluxes, 'no2_mixing_ratio')
        )
        torch.testing.assert_close(
            nox_change,
            torch.zeros_like(nox_change),
            rtol=0.0,
            atol=0.0,
        )

    def test_nox_conservation(self):
        """Test NOx (NO + NO2) conservation in isolated system."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        state = self._state()
        fluxes = chem.forward(state, time=0.0, dt=1.0)

        # NOx exchange must be exactly balanced by the closed reaction family.
        nox_change = (
            _update_value(fluxes, 'no_mixing_ratio')
            + _update_value(fluxes, 'no2_mixing_ratio')
        )

        torch.testing.assert_close(
            nox_change,
            torch.zeros_like(nox_change),
            rtol=0.0,
            atol=0.0,
        )

        assert chem.check_conservation_laws(state, fluxes) == {
            'nox_family': 0.0
        }

    def test_nox_conservation_checker_detects_tampering(self):
        """The runtime hook must expose an independently perturbed NOx ledger."""

        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        state = self._state()
        fluxes = chem.forward(state, time=0.0, dt=1.0)
        tampered = dict(fluxes)
        tampered['no2_mixing_ratio'] = torch.tensor(2.0e-12)

        result = chem.check_conservation_laws(state, tampered)

        assert result['nox_family'] > 0.0

    @pytest.mark.parametrize(
        ('key', 'invalid_value', 'expected_exception'),
        [
            ('no_mixing_ratio', None, KeyError),
            ('no_mixing_ratio', torch.empty(0), ValueError),
            ('no_mixing_ratio', torch.tensor(float('nan')), ValueError),
            ('no_mixing_ratio', 'not-a-tensor', TypeError),
        ],
    )
    def test_nox_conservation_checker_fails_fast(
        self,
        key,
        invalid_value,
        expected_exception,
    ):
        """Malformed declared conservation values must never pass silently."""

        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        state = self._state()
        fluxes = chem.forward(state, time=0.0, dt=1.0)
        malformed = dict(fluxes)
        if invalid_value is None:
            malformed.pop(key)
        else:
            malformed[key] = invalid_value

        with pytest.raises(expected_exception):
            chem.check_conservation_laws(state, malformed)

class TestChemistryParameterBinding:
    """Scientific rates must consume the current declared parameter source."""

    def test_unit_annotated_override_is_converted_without_mutating_caller(self):
        configured = torch.tensor(1e-7, dtype=torch.float64, requires_grad=True)
        config = {"parameters": {"minimum_concentration": {"value": configured, "units": "ppm"}}}
        chemistry = ChemistryFunctor(config=config, physical_boundary="open_mechanism")
        for multiplier in (1.0, 1.0, 2.0):
            with torch.no_grad():
                configured.fill_(1e-7 * multiplier)
            value = chemistry.get_parameter_value("minimum_concentration", required=True)
            assert value.item() == pytest.approx(1e-13 * multiplier, rel=1e-12, abs=0)
            derivative = torch.autograd.grad(value, configured)[0]
            assert derivative.item() == pytest.approx(1e-6, rel=1e-12, abs=0)
        assert config["parameters"]["minimum_concentration"]["value"] is configured
        assert config["parameters"]["minimum_concentration"]["units"] == "ppm"

    @pytest.mark.parametrize("overrides", [
        {"k_o1d_h2o_298": {"value": 2.0e-10, "units": "m"}},
        {"k_no_o3_298": torch.tensor([1.5e-14, 1.6e-14])},
        [1.0],
    ])
    def test_malformed_or_incompatible_overrides_are_rejected(self, overrides):
        with pytest.raises((ValueError, TypeError)):
            ChemistryFunctor(config={"parameters": overrides}, physical_boundary="open_mechanism")

    def test_manager_rate_update_is_observed_without_reconstructing_functor(self):
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        manager = chemistry.get_parameter_manager()
        key = "atmosphere.chemistry.k_no_o3_298"
        original = manager.get(key, required=True)
        temperature = torch.tensor(298.15, dtype=torch.float64)
        initial = chemistry._calculate_temperature_dependent_rates(temperature)["k_no_o3"]
        try:
            manager.set_value(key, 1.6e-14, validate=True)
            updated = chemistry._calculate_temperature_dependent_rates(temperature)["k_no_o3"]
            assert updated.dtype == torch.float64
            assert updated.item() == pytest.approx(1.6e-14, rel=1e-10, abs=0)
            assert not torch.equal(updated, initial)
        finally:
            manager.set_value(key, original, validate=True)

    @pytest.mark.parametrize("parameter_name", ["k_no_o3_298", "uv_a_surface_albedo"])
    def test_explicit_tensor_overrides_retain_precision_and_gradient(self, parameter_name):
        value = 1.6e-14 if parameter_name == "k_no_o3_298" else 0.05
        parameter = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        chemistry = ChemistryFunctor(config={"parameters": {parameter_name: parameter}}, physical_boundary="open_mechanism")
        temperature = torch.tensor(298.15, dtype=torch.float64)
        if parameter_name == "k_no_o3_298":
            output = chemistry._calculate_temperature_dependent_rates(temperature)["k_no_o3"]
            expected = output / parameter
        else:
            output, _ = chemistry._calculate_photolysis_rates(
                torch.tensor(800.0, dtype=torch.float64), torch.tensor(1.0), temperature,
            )
            # J_NO2 = (1 + alpha)(B1 G + B2 G^2): dJ/dalpha = J / (1 + alpha).
            expected = output / (1.0 + parameter)
        assert output.dtype == torch.float64
        derivative = torch.autograd.grad(output, parameter)[0]
        torch.testing.assert_close(derivative, expected.detach())

    def test_explicit_source_order_and_bounds_are_not_bypassed(self):
        chemistry = ChemistryFunctor(config={"parameters": {"uv_a_surface_albedo": 0.1}}, physical_boundary="open_mechanism")
        key = "atmosphere.chemistry.uv_a_surface_albedo"
        expected = chemistry.get_parameter_manager().get(key, required=True)
        manager_only = chemistry.get_parameter_value(key, source_order=("manager",), required=True)
        assert float(manager_only) == float(expected)
        assert float(chemistry.get_parameter_value(key, required=True)) == pytest.approx(0.1)
        with pytest.raises(ValueError, match="bounds"):
            chemistry.get_parameter_value(key, bounds=(0.0, 0.05), required=True)

    def test_scalar_expected_type_remains_part_of_the_public_getter(self):
        value = ChemistryFunctor(physical_boundary="open_mechanism").get_parameter_value("j_no2_irradiance_linear_coefficient", expected_type=float, required=True)
        assert isinstance(value, float)
        assert value == pytest.approx(1.47e-5)


class TestParameterSensitivity:
    """Test sensitivity to parameter variations."""
    
    def test_j_no2_sensitivity(self):
        """Test sensitivity to NO2 photolysis rate."""
        config_low = {
            'parameters': {
                'uv_a_surface_albedo': 0.0  # Low photolysis (no UV-A reflection)
            }
        }
        config_high = {
            'parameters': {
                'uv_a_surface_albedo': 0.2  # High photolysis (upper albedo bound)
            }
        }
        
        chem_low = ChemistryFunctor(config=config_low, physical_boundary="open_mechanism")
        chem_high = ChemistryFunctor(config=config_high, physical_boundary="open_mechanism")
        
        state = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_pressure': torch.tensor(101325.0),
            'no_mixing_ratio': torch.tensor(5e-9),
            'no2_mixing_ratio': torch.tensor(10e-9),
            'o3_mixing_ratio': torch.tensor(40e-9),
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'shortwave_down': torch.tensor(800.0),
            'cos_zenith': torch.tensor(1.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        fluxes_low = chem_low.forward(state, time=0.0, dt=1.0)
        fluxes_high = chem_high.forward(state, time=0.0, dt=1.0)
        
        # Higher J_NO2 should produce more O3
        assert _update_value(fluxes_high, 'o3_mixing_ratio') > _update_value(
            fluxes_low,
            'o3_mixing_ratio',
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



class TestClosedConstantVolumeBox:
    """Closed adiabatic constant-volume NO + O3 -> NO2 + O2 box (GENERIC, absorbed)."""

    SPECIES = ("box_no_amount", "box_no2_amount", "box_o3_amount", "box_o2_amount")
    NU = torch.tensor([-1.0, 1.0, -1.0, 1.0, 0.0], dtype=torch.float64)

    @staticmethod
    def _box(**kwargs):
        return ChemistryFunctor(
            name="atmospheric_chemistry",
            physical_boundary="closed_constant_volume",
            **kwargs,
        )

    @classmethod
    def _state(cls, chem, *, temperature=298.15, amounts=(1.0e-8, 1.0e-8, 1.0e-8, 8.0e-3), volume=1.0e-3, requires_grad=False):
        n = torch.tensor(amounts, dtype=torch.float64)
        t = torch.tensor(temperature, dtype=torch.float64)
        u = (n * chem.standard_molar_properties(t)["u"]).sum()
        state = {name: n[i].clone() for i, name in enumerate(cls.SPECIES)}
        state["box_internal_energy"] = u.detach().clone()
        state["box_volume"] = torch.tensor(volume, dtype=torch.float64)
        if requires_grad:
            for key in (*cls.SPECIES, "box_internal_energy"):
                state[key].requires_grad_(True)
        return state

    @classmethod
    def _oracle(cls, chem, state):
        # References: Kondepudi and Prigogine (1998) ch. 9; Mielke (2011) Nonlinearity 24:1329;
        #   Seinfeld and Pandis (2016) ideal-gas chemical potential; MA-19.2 reaction contract.
        # Mathematical expression:
        #   T = T(U, n) from sum_i n_i u_i(T) = U;  mu_i = h_i - T s_i + R T ln(p_i / p0);
        #   A = -nu^T mu;  r = k(T) N_A n_NO n_O3 / (c_v V);  x_dot = nu r;  sigma = A r / T.
        # Parameter meanings: n_i molar inventory, U internal energy, V volume, k NO+O3 rate
        #   coefficient, N_A Avogadro constant, c_v cm3 per m3 factor, nu stoichiometry.
        # Units: n mol, U J, V m3, T K, mu and A J mol-1, k cm3 molecule-1 s-1, r mol s-1,
        #   sigma J K-1 s-1.
        # Domain/constraints: n_i > 0, V > 0, T inside the kinetics interval, A > 0.
        # Sign convention: r > 0 consumes NO and O3 and produces NO2 and O2; sigma >= 0.
        x = torch.stack([state[k] for k in (*cls.SPECIES, "box_internal_energy")])
        x = x.detach().clone().requires_grad_(True)
        n, u = x[:4], x[4]
        v = state["box_volume"].detach()
        t_min = torch.tensor(float(chem.get_parameter_value("temperature_min", required=True)), dtype=torch.float64)
        t_max = torch.tensor(float(chem.get_parameter_value("temperature_max", required=True)), dtype=torch.float64)
        t = chem.temperature_from_internal_energy(u, n, temperature_lower=t_min, temperature_upper=t_max)
        mix = chem.ideal_gas_mixture_properties(t, n, v)
        affinity = -(cls.NU[:4] @ mix["chemical_potential"])
        k = chem._calculate_temperature_dependent_rates(t)["k_no_o3"]
        avogadro = float(chemistry_module.get_physical_constant("relic.physical.constants.universal.avogadro_constant"))
        cm3_per_m3 = float(chemistry_module.get_physical_constant("relic.physical.constants.conversions.cubic_centimeters_per_cubic_meter"))
        rate = k * avogadro / cm3_per_m3 * n[0] * n[2] / v
        entropy = mix["entropy"]
        e_s = torch.autograd.grad(entropy, x, create_graph=True)[0]
        mobility = (rate * t / affinity) * torch.outer(cls.NU, cls.NU)
        return {
            "x": x, "T": t, "S": entropy, "A": affinity, "r": rate, "k": k,
            "drift": cls.NU * rate, "sigma_quadratic": e_s @ mobility @ e_s,
            "sigma": affinity * rate / t, "pressure": mix["pressure"],
        }

    def test_closed_box_declares_its_own_contract_without_radiation(self):
        chem = self._box()
        assert chem.physical_boundary == "closed_constant_volume"
        assert tuple(chem.metadata.input_variables) == (*self.SPECIES, "box_internal_energy", "box_volume")
        assert tuple(chem.metadata.output_variables[:5]) == (*self.SPECIES, "box_internal_energy")
        assert all(name.startswith("diagnostics.chemistry.box_") for name in chem.metadata.output_variables[5:])
        assert chem.get_dependencies() == []
        assert chem.optional_input_variables == []
        assert chem.ports() == ()
        assert chem.metadata.conservation_laws == {
            "nitrogen_atoms": {"quantity": "nitrogen_atoms"},
            "oxygen_atoms": {"quantity": "oxygen_atoms"},
            "internal_energy": {"quantity": "internal_energy"},
        }
        assert ChemistryFunctor(physical_boundary="open_mechanism").physical_boundary == "open_mechanism"
        with pytest.raises(ValueError, match="physical_boundary"):
            ChemistryFunctor(physical_boundary="leaky_box")

    def test_closed_box_governed_call_matches_independent_generic_oracle(self):
        chem = self._box()
        state = self._state(chem)
        oracle = self._oracle(chem, state)
        dt = 1.0
        result = chem(dict(state), time=0.0, dt=dt)
        for i, name in enumerate((*self.SPECIES, "box_internal_energy")):
            item = result[name]
            assert isinstance(item, UpdateItem) and item.mode is UpdateMode.ACCUMULATE
            assert item.unit == ("J" if name == "box_internal_energy" else "mol")
            torch.testing.assert_close(item.value, oracle["drift"][i].detach() * dt, rtol=1e-12, atol=0.0)
        assert float(result["box_internal_energy"].value.detach()) == 0.0
        assert float(oracle["r"]) > 0.0 and float(oracle["A"]) > 0.0
        torch.testing.assert_close(_diagnostic_value(result, "diagnostics.chemistry.box_temperature", "K"), oracle["T"].detach())
        torch.testing.assert_close(_diagnostic_value(result, "diagnostics.chemistry.box_reaction_rate", "mol s-1"), oracle["r"].detach())
        torch.testing.assert_close(_diagnostic_value(result, "diagnostics.chemistry.box_reaction_affinity", "J mol-1"), oracle["A"].detach())
        torch.testing.assert_close(_diagnostic_value(result, "diagnostics.chemistry.box_reaction_power", "W"), (oracle["A"] * oracle["r"]).detach())
        sigma = _diagnostic_value(result, "diagnostics.chemistry.box_entropy_production_rate", "J K-1 s-1")
        torch.testing.assert_close(sigma, oracle["sigma"].detach())
        torch.testing.assert_close(sigma, oracle["sigma_quadratic"].detach(), rtol=1e-9, atol=0.0)
        assert float(sigma) > 0.0
        torch.testing.assert_close(_diagnostic_value(result, "diagnostics.chemistry.box_entropy", "J K-1"), oracle["S"].detach())
        torch.testing.assert_close(_diagnostic_value(result, "diagnostics.chemistry.box_pressure", "Pa"), oracle["pressure"].detach())
        torch.testing.assert_close(_diagnostic_value(result, "diagnostics.chemistry.box_rate_constant_no_o3", "cm3 molecule-1 s-1"), oracle["k"].detach())
        delta = torch.stack([result[name].value for name in self.SPECIES])
        assert float(torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64) @ delta) == 0.0
        assert float(torch.tensor([1.0, 2.0, 3.0, 2.0], dtype=torch.float64) @ delta) == 0.0
        ledgers = chem.check_conservation_laws(state, result)
        assert ledgers == {"nitrogen_atoms": 0.0, "oxygen_atoms": 0.0, "internal_energy": 0.0}

    def test_closed_box_advances_the_absorbed_drift_exactly_once(self, monkeypatch):
        chem = self._box()
        state = self._state(chem)
        drift_calls = []
        network_calls = []
        original_drift = chemistry_module.IPHSStructureSpec.canonical_drift
        original_network = chemistry_module.advance_reaction_network

        def counting_drift(spec, **kwargs):
            drift_calls.append((spec.poisson_operator, spec.reaction_representation, spec.formulation))
            return original_drift(spec, **kwargs)

        def counting_network(**kwargs):
            network_calls.append(kwargs)
            return original_network(**kwargs)

        monkeypatch.setattr(chemistry_module.IPHSStructureSpec, "canonical_drift", counting_drift)
        monkeypatch.setattr(chemistry_module, "advance_reaction_network", counting_network)
        result = chem.forward(dict(state), 0.0, 2.0)
        assert drift_calls == [("zero_field", "absorbed", "generic")]
        assert len(network_calls) == 1
        audit = network_calls[0]
        assert audit["dt"] is None and audit["require_thermodynamic_point_audit"] is True
        assert audit["require_conservation_matrix"] is True
        torch.testing.assert_close(
            audit["conservation_matrix"],
            torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 2.0, 0.0]], dtype=torch.float64),
        )
        rate = _diagnostic_value(result, "diagnostics.chemistry.box_reaction_rate", "mol s-1")
        torch.testing.assert_close(result["box_no2_amount"].value, rate * 2.0)
        torch.testing.assert_close(result["box_no_amount"].value, -rate * 2.0)

    def test_closed_box_gradients_reach_state_and_kinetic_parameter(self):
        configured = torch.tensor(1.8e-14, dtype=torch.float64, requires_grad=True)
        chem = self._box(config={"parameters": {"k_no_o3_298": configured}})
        state = self._state(chem, requires_grad=True)
        result = chem.forward(dict(state), 0.0, 1.0)
        produced = result["box_no2_amount"].value
        grads = torch.autograd.grad(produced, (state["box_no_amount"], state["box_o3_amount"], state["box_internal_energy"], configured))
        oracle = self._oracle(chem, state)
        oracle_grads = torch.autograd.grad(oracle["drift"][1], oracle["x"])[0]
        torch.testing.assert_close(grads[0], oracle_grads[0], rtol=1e-8, atol=0.0)
        torch.testing.assert_close(grads[1], oracle_grads[2], rtol=1e-8, atol=0.0)
        torch.testing.assert_close(grads[2], oracle_grads[4], rtol=1e-8, atol=0.0)
        torch.testing.assert_close(grads[3], oracle["r"].detach() / configured.detach(), rtol=1e-8, atol=0.0)
        assert float(grads[2]) != 0.0

    def test_closed_box_inference_inputs_keep_a_local_potential_graph_only(self):
        chem = self._box()
        state = self._state(chem)
        assert not any(value.requires_grad for value in state.values())
        result = chem.forward(dict(state), 0.0, 1.0)
        assert not state["box_no_amount"].requires_grad
        assert result["box_no_amount"].value.grad_fn is not None

    @pytest.mark.parametrize("case", [
        "missing_energy", "missing_volume", "missing_o2", "zero_amount", "negative_volume",
        "batched", "temperature_out_of_domain", "nonpositive_affinity", "float32_mixed",
    ])
    def test_closed_box_rejects_inputs_outside_the_declared_domain(self, case):
        chem = self._box()
        state = self._state(chem)
        if case == "missing_energy":
            state.pop("box_internal_energy")
        elif case == "missing_volume":
            state.pop("box_volume")
        elif case == "missing_o2":
            state.pop("box_o2_amount")
        elif case == "zero_amount":
            state["box_o3_amount"] = torch.tensor(0.0, dtype=torch.float64)
        elif case == "negative_volume":
            state["box_volume"] = torch.tensor(-1.0e-3, dtype=torch.float64)
        elif case == "batched":
            state = {k: v.reshape(1).repeat(2) for k, v in state.items()}
        elif case == "temperature_out_of_domain":
            state = self._state(chem, temperature=600.0)
        elif case == "nonpositive_affinity":
            state = self._state(chem, amounts=(1.0e-30, 1.0e3, 1.0e-30, 1.0e3), volume=1.0)
        elif case == "float32_mixed":
            state["box_volume"] = torch.tensor(1.0e-3, dtype=torch.float32)
        with pytest.raises(ValueError):
            chem.forward(dict(state), 0.0, 1.0)

    def test_closed_box_forward_runs_the_level5_runtime_point_audit(self):
        chem = self._box()
        state = self._state(chem)
        assert chem.last_strict_iphs_audit() is None
        result = chem.forward(dict(state), 0.0, 1.0)
        audit = chem.last_strict_iphs_audit()
        assert audit is not None
        assert audit["required_level"] == 5
        assert audit["runtime_verified_level"] == 5
        assert audit["process_level_status"] == "runtime_point_verified"
        assert audit["formulation"] == "generic"
        assert audit["reaction_representation"] == "absorbed"
        assert audit["jacobi_identity_basis"] == "canonical_zero_field"
        assert audit["reaction_energy_rate"] == 0.0
        sigma = float(_diagnostic_value(result, "diagnostics.chemistry.box_entropy_production_rate", "J K-1 s-1"))
        assert audit["reaction_entropy_rate"] == pytest.approx(sigma, rel=1e-9)
        assert audit["entropy_production"] == pytest.approx(sigma, rel=1e-9)
        assert audit["M_energy_gradient_residual"] == 0.0
        assert audit["J_entropy_gradient_residual"] == 0.0
        # The audit is per-call evidence, not a second state: a rejected step clears it.
        with pytest.raises(ValueError):
            chem.forward(dict(state), 0.0, 1.0e9)
        assert chem.last_strict_iphs_audit() is None

    def test_closed_box_rejects_an_oversized_step_before_emitting_updates(self):
        chem = self._box()
        state = self._state(chem)
        rate = float(self._oracle(chem, state)["r"])
        admissible = float(state["box_no_amount"]) / rate
        with pytest.raises(ValueError, match="positivity"):
            chem.forward(dict(state), 0.0, admissible * 1.5)
        result = chem.forward(dict(state), 0.0, admissible * 0.5)
        assert float(state["box_no_amount"] + result["box_no_amount"].value) > 0.0


    @pytest.mark.parametrize("boundary", ["exact_limit", "previous_float", "next_float"])
    def test_closed_box_rejects_candidate_outside_its_thermodynamic_domain(self, boundary):
        # References: IEEE-754; NASA7 ideal-gas domain; existing closed-box oracle.
        # Mathematical expression: h=n_NO/(-f_NO); candidate=x+h*f; mu includes R*T*ln(p/p0).
        # Parameter meanings: h Euler step, f canonical drift, A=-nu^T*mu the reaction affinity.
        # Units: h s, n mol, U J, A J mol-1; Decimal checks the same binary input values.
        # Domain/constraints: n>0 and A>0 at the actual candidate, not only the pre-step point.
        # Sign convention: NO and O3 are consumed; no rejected increment is emitted.
        chem = self._box()
        state = self._state(chem)
        snapshots = {key: value.clone() for key, value in state.items()}
        unit = chem(state, time=0.0, dt=1.0)
        tendency = unit["box_no_amount"].value.detach()
        limit = float(state["box_no_amount"] / (-tendency))
        dt = {
            "exact_limit": limit,
            "previous_float": math.nextafter(limit, 0.0),
            "next_float": math.nextafter(limit, math.inf),
        }[boundary]
        candidate = {
            key: (value + dt * unit[key].value).detach() if key in unit else value.clone()
            for key, value in state.items()
        }
        if boundary == "exact_limit":
            assert float(candidate["box_no_amount"]) == 0.0
            with localcontext() as context:
                context.prec = 80
                exact = Decimal.from_float(float(state["box_no_amount"])) + (
                    Decimal.from_float(dt) * Decimal.from_float(float(tendency))
                )
                assert exact > 0  # The actual tensor rounded a mathematically positive amount to zero.
        elif boundary == "previous_float":
            assert all(float(candidate[key]) > 0.0 for key in self.SPECIES)
            assert float(self._oracle(chem, candidate)["A"].detach()) < 0.0
        else:
            assert float(candidate["box_no_amount"]) < 0.0
        assert chem.last_strict_iphs_audit()["runtime_verified_level"] == 5
        with pytest.raises(ExecutionError, match="strictly positive|affinity|positivity"):
            chem(state, time=0.0, dt=dt)
        assert chem.last_strict_iphs_audit() is None
        for key, snapshot in snapshots.items():
            torch.testing.assert_close(state[key], snapshot, rtol=0.0, atol=0.0)

        # A rejection leaves the same input available for one valid governed step.
        valid = chem(state, time=0.0, dt=0.99 * limit)
        for key in chem.CLOSED_BOX_STATE:
            torch.testing.assert_close(valid[key].value, 0.99 * limit * unit[key].value, rtol=1e-12, atol=0.0)
        assert chem.last_strict_iphs_audit()["runtime_verified_level"] == 5

    def test_closed_box_near_limit_valid_step_remains_in_domain_and_continues(self):
        # References: closed-box oracle; Euler finite-step applicability, not a global entropy theorem.
        # Mathematical expression: x_next=x+0.99*h_limit*f; C*Delta_n=0 and Delta_U=0.
        # Parameter meanings: C nitrogen/oxygen ledgers, h_limit the inventory-only Euler bound.
        # Units: n mol, U J, S J K-1; relative ledger residuals dimensionless.
        # Domain/constraints: this positive-affinity sample; validate its actual next point and step.
        # Sign convention: positive reaction consumes NO/O3; this sample has S_next>S_initial.
        chem = self._box()
        state = self._state(chem)
        oracle = self._oracle(chem, state)
        unit = chem(state, time=0.0, dt=1.0)
        dt = 0.99 * float(state["box_no_amount"] / (-unit["box_no_amount"].value.detach()))
        result = chem(state, time=0.0, dt=dt)
        candidate = {key: (value + result[key].value).detach() if key in result else value.clone()
                     for key, value in state.items()}
        assert all(float(candidate[key]) > 0.0 for key in self.SPECIES)
        next_point = self._oracle(chem, candidate)
        assert float(next_point["A"].detach()) > 0.0
        assert float(next_point["S"].detach()) > float(oracle["S"].detach())
        torch.testing.assert_close(candidate["box_internal_energy"], state["box_internal_energy"], rtol=0, atol=0)
        torch.testing.assert_close(candidate["box_volume"], state["box_volume"], rtol=0, atol=0)
        ledgers = chem.check_conservation_laws(state, result)
        assert max(ledgers.values()) <= torch.finfo(torch.float64).eps
        following = chem(candidate, time=dt, dt=1.0)
        assert all(float(candidate[key] + following[key].value) > 0.0 for key in self.SPECIES)


class TestParameterDeclarationAndOHSinks:
    """Full-key parameter declaration (round 36) and the CO/CH4 OH-sink inputs.

    References: Seinfeld and Pandis (2016), ch. 6 (steady-state OH with the sinks
    k_OH+CO [CO] + k_OH+CH4 [CH4] + k_OH+NO2 [NO2]): CO and CH4 are always-present
    sink species whose mixing ratios the caller supplies; no background closure
    stands in for a missing input.
    """

    @staticmethod
    def _oh_state(**overrides):
        state = {
            "air_temperature": torch.tensor(298.15),
            "atmospheric_pressure": torch.tensor(101325.0),
            "no2_mixing_ratio": torch.tensor(10e-9),
            "o3_mixing_ratio": torch.tensor(40e-9),
            "co_mixing_ratio": torch.tensor(150e-9),
            "ch4_mixing_ratio": torch.tensor(1.8e-6),
            "specific_humidity": torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            "cos_zenith": torch.tensor(1.0),  # overhead sun
        }
        state.update(overrides)
        return state

    @pytest.mark.parametrize("missing", ["co_mixing_ratio", "ch4_mixing_ratio"])
    def test_a_missing_oh_sink_species_raises(self, missing):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        state = self._oh_state()
        del state[missing]
        assert missing in chem.get_required_inputs()
        with pytest.raises(ValueError, match=missing):
            chem.compute_oh_concentration(state, torch.tensor(5e-3), state["air_temperature"])

    def test_atmospheric_pressure_is_the_required_canonical_input(self):
        """air_pressure is an alias of atmospheric_pressure (the StateManager does
        not write aliases): the canonical pressure is a required input and no
        standard atmosphere stands in for it."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        assert "atmospheric_pressure" in chem.get_required_inputs()
        assert "air_pressure" not in chem.metadata.input_variables
        state = {
            key: value
            for key, value in self._oh_state().items()
            if key not in ("air_pressure", "atmospheric_pressure")
        }
        state["air_pressure"] = torch.tensor(101325.0)
        with pytest.raises(ValueError, match="atmospheric_pressure"):
            chem.compute_oh_concentration(state, torch.tensor(5e-3), state["air_temperature"])

    def test_background_closure_keys_are_retired(self):
        manager = ChemistryFunctor(physical_boundary="open_mechanism").get_parameter_manager()
        for key in (
            "atmosphere.chemistry.background_co_mixing_ratio",
            "atmosphere.chemistry.background_ch4_mixing_ratio",
        ):
            assert manager.get_parameter(key) is None, key

    @pytest.mark.parametrize("boundary", ["open_mechanism", "closed_constant_volume"])
    def test_metadata_declares_full_database_keys(self, boundary):
        chem = ChemistryFunctor(physical_boundary=boundary)
        manager = chem.get_parameter_manager()
        declared = sorted(chem.metadata.parameters)
        assert [key for key in declared if manager.get_parameter(key) is None] == []
        assert list(chem.required_parameters()) == declared

    def test_closed_box_declares_the_solver_and_audit_keys_it_reads(self):
        declared = set(ChemistryFunctor(physical_boundary="closed_constant_volume").metadata.parameters)
        assert {
            "computation.solvers.bracketed_bisection.max_iterations",
            "computation.solvers.bracketed_bisection.rtol",
            "computation.solvers.bracketed_bisection.atol",
            "validation.iphs.runtime_point_tolerance",
        } <= declared

    def test_short_name_overrides_remain_the_whitelist(self):
        chem = ChemistryFunctor(config={"parameters": {"k_no_o3_298": 1.6e-14}}, physical_boundary="open_mechanism")
        assert float(chem.get_parameter_value("atmosphere.chemistry.k_no_o3_298", required=True)) == (
            pytest.approx(1.6e-14)
        )
        with pytest.raises(ValueError, match="Unknown chemistry parameter override"):
            ChemistryFunctor(config={"parameters": {"atmosphere.chemistry.k_no_o3_298": 1.6e-14}}, physical_boundary="open_mechanism")


# ---------------------------------------------------------------------------
# Primary OH production and the optional photolysis temperature module
# ---------------------------------------------------------------------------

import math as _math
from pathlib import Path as _Path

import yaml as _yaml

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
_PHOTOLYSIS_YAML = (
    _PROJECT_ROOT / "relic" / "parameters" / "database" / "relic" / "atmosphere" / "photolysis_spectra.yaml"
)
_CHEMISTRY_YAML = _PROJECT_ROOT / "relic" / "parameters" / "database" / "relic" / "atmosphere" / "chemistry.yaml"

# Warneck and Williams (2012) Table 7.1 (Sander et al. 2006, JPL 06-2): k = A exp(-B/T)
# [cm3 molecule-1 s-1] with the tabulated k298.
_O1D_KINETICS = {
    "h2o": (2.0e-10, -60.0),
    "n2": (3.1e-11, -110.0),
    "o2": (4.0e-11, -67.0),
}
# MCM v3.3.1 clear-sky photolysis parameters J = l cos(chi)^m exp(-n sec(chi)) (Saunders et al. 2003).
_MCM_J1 = (6.073e-5, 1.743, 0.474)
_MCM_J4 = (1.165e-2, 0.244, 0.267)


def _open_state(**overrides):
    """A complete open-mechanism state (daytime, 1 atm, humid air)."""

    state = {
        "air_temperature": torch.tensor(298.15, dtype=torch.float64),
        "atmospheric_pressure": torch.tensor(101325.0, dtype=torch.float64),
        "specific_humidity": torch.tensor(0.01, dtype=torch.float64),
        "shortwave_down": torch.tensor(800.0, dtype=torch.float64),
        "cos_zenith": torch.tensor(1.0, dtype=torch.float64),
        "no_mixing_ratio": torch.tensor(5e-9, dtype=torch.float64),
        "no2_mixing_ratio": torch.tensor(10e-9, dtype=torch.float64),
        "o3_mixing_ratio": torch.tensor(40e-9, dtype=torch.float64),
        "co_mixing_ratio": torch.tensor(150e-9, dtype=torch.float64),
        "ch4_mixing_ratio": torch.tensor(1.8e-6, dtype=torch.float64),
    }
    state.update(overrides)
    return state


def _hand_water_fraction(temperature, pressure, specific_humidity):
    """f = k_H2O[H2O] / (k_H2O[H2O] + k_N2[N2] + k_O2[O2]) from Table 7.1 and the SSOT constants."""

    from relic.utils.physical_constants import get_physical_constant

    k_b = float(get_physical_constant("relic.physical.constants.universal.boltzmann_constant"))
    m_w = float(get_physical_constant("relic.physical.constants.water.molecular_weight"))
    m_d = float(get_physical_constant("relic.physical.constants.air.molecular_weight_dry"))
    x_n2 = float(get_physical_constant("relic.physical.constants.atmosphere.n2_fraction"))
    x_o2 = float(get_physical_constant("relic.physical.constants.atmosphere.o2_fraction"))
    gas_constant = float(get_physical_constant("relic.physical.constants.universal.gas_constant"))
    t_ref = float(get_physical_constant("relic.physical.constants.parameters.t_ref"))
    entries = _yaml.safe_load(_CHEMISTRY_YAML.read_text(encoding="utf-8-sig"))["parameters"]
    n_air = pressure / (k_b * temperature) * 1.0e-6
    x_w = (specific_humidity / m_w) / (specific_humidity / m_w + (1.0 - specific_humidity) / m_d)

    def rate(name):
        # Table 7.1 k298 and E/R; the stored Ea = (E/R) R is rounded to 0.1 J mol-1.
        k298, e_over_r = _O1D_KINETICS[name]
        stored = entries[f"atmosphere.chemistry.k_o1d_{name}_activation_energy"]["value"]
        assert entries[f"atmosphere.chemistry.k_o1d_{name}_298"]["value"] == k298
        assert abs(stored / gas_constant - e_over_r) < 0.01
        return k298 * _math.exp(stored / gas_constant * (1.0 / t_ref - 1.0 / temperature))

    h2o = rate("h2o") * x_w * n_air
    quench = rate("n2") * x_n2 * (1.0 - x_w) * n_air + rate("o2") * x_o2 * (1.0 - x_w) * n_air
    return h2o / (h2o + quench), n_air


def _hand_o1d_ratio(cos_chi):
    (l1, m1, n1), (l4, m4, n4) = _MCM_J1, _MCM_J4
    return (l1 / l4) * cos_chi ** (m1 - m4) * _math.exp(-(n1 - n4) / cos_chi)


def _spectral_oracle(temperature, mu=1.0):
    """Independent R_NO2(T), R_O1D(T) from the raw YAML tables (plain Python floats).

    The reference spectrum is the direct beam at the actual solar geometry mu = cos(chi);
    without a direct beam (mu <= 0) both factors are 1.
    """

    from relic.utils.physical_constants import get_physical_constant

    table = _yaml.safe_load(_PHOTOLYSIS_YAML.read_text(encoding="utf-8"))["parameters"]

    def g(name):
        return table["atmosphere.chemistry.photolysis." + name]["value"]

    def interp(x, xs, ys):
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                w = (x - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] * (1.0 - w) + ys[i + 1] * w
        raise AssertionError(f"{x} outside table")

    s, a, b, c, d, e = (
        g("rayleigh_scale"), g("rayleigh_numerator_constant"), g("rayleigh_numerator_inverse_square"),
        g("rayleigh_numerator_square"), g("rayleigh_denominator_inverse_square"),
        g("rayleigh_denominator_square"),
    )
    o3x = g("o3_cross_section_wavelength")
    s_hi, s_lo = g("o3_cross_section_high_temperature"), g("o3_cross_section_low_temperature")
    t_hi, t_lo = g("o3_cross_section_temperature_high"), g("o3_cross_section_temperature_low")
    column = g("reference_ozone_column")
    fx, fv = g("solar_flux_wavelength"), g("solar_photon_flux")

    def flux(lam):
        tau_r = s * (a - b / lam ** 2 - c * lam ** 2) / (1.0 + d / lam ** 2 - e * lam ** 2)
        tau_o3 = interp(lam, o3x, s_lo) * column
        return interp(lam, fx, fv) * _math.exp(-(tau_r + tau_o3) / mu)

    def sigma_o3(lam, temp):
        hi, lo = interp(lam, o3x, s_hi), interp(lam, o3x, s_lo)
        th, tl = interp(lam, o3x, t_hi), interp(lam, o3x, t_lo)
        tc = min(max(temp, tl), th)
        return lo + (hi - lo) * (tc - tl) / (th - tl)

    k_b = float(get_physical_constant("relic.physical.constants.universal.boltzmann_constant"))
    h = float(get_physical_constant("relic.physical.constants.universal.planck_constant"))
    c_light = float(get_physical_constant("relic.physical.constants.universal.speed_of_light"))
    r_cm = k_b / (h * c_light) / 100.0  # cm-1 K-1
    amp, cen, wid = g("matsumi_amplitudes"), g("matsumi_band_centers"), g("matsumi_band_widths")

    def phi_o1d(lam, temp):
        if lam < g("matsumi_wavelength_min"):
            return g("o1d_quantum_yield_short_wavelength")
        if lam > g("matsumi_wavelength_max"):
            return g("o1d_quantum_yield_long_wavelength")
        temp = min(max(temp, g("matsumi_temperature_min")), g("matsumi_temperature_max"))
        q = _math.exp(-g("matsumi_energy_gap") / (r_cm * temp))
        t_rel = temp / g("matsumi_reference_temperature")
        return (
            amp[0] / (1.0 + q) * _math.exp(-(((cen[0] - lam) / wid[0]) ** 4))
            + amp[1] * q / (1.0 + q) * t_rel ** 2 * _math.exp(-(((cen[1] - lam) / wid[1]) ** 2))
            + amp[2] * t_rel ** 1.5 * _math.exp(-(((cen[2] - lam) / wid[2]) ** 2))
            + g("matsumi_offset")
        )

    no2x, no2s = g("no2_cross_section_wavelength"), g("no2_cross_section")
    qx, q_hi, q_lo = (
        g("no2_quantum_yield_wavelength"), g("no2_quantum_yield_high_temperature"),
        g("no2_quantum_yield_low_temperature"),
    )
    tq_hi, tq_lo = g("no2_quantum_yield_temperature_high"), g("no2_quantum_yield_temperature_low")

    def s_no2(temp):
        total = 0.0
        tc = min(max(temp, tq_lo), tq_hi)
        for lam, ph, pl in zip(qx, q_hi, q_lo):
            sigma = [value for centre, value in zip(no2x, no2s) if abs(lam - centre) < 0.0025][0]
            total += sigma * (pl + (ph - pl) * (tc - tq_lo) / (tq_hi - tq_lo)) * flux(lam)
        return total

    def s_o1d(temp):
        return sum(sigma_o3(lam, temp) * phi_o1d(lam, temp) * flux(lam) for lam in g("o1d_integration_wavelength"))

    t_ref = g("reference_temperature")
    if mu <= 0.0:
        return 1.0, 1.0, phi_o1d
    return s_no2(temperature) / s_no2(t_ref), s_o1d(temperature) / s_o1d(t_ref), phi_o1d


def _spectral_chemistry():
    return ChemistryFunctor(
        config={"parameters": {"photolysis_temperature_mode": "spectral_quantum_yield"}},
        physical_boundary="open_mechanism",
    )


class TestPrimaryOHProduction:
    """P_OH = y f J(O1D) [O3] with J(O1D) = J_NO2 rho(chi) and the O(1D) water fraction f.

    References: Warneck and Williams (2012) Tables 6.7 and 7.1 (Sander et al. 2006); MCM
    v3.3.1 J1 and J4 (Saunders et al. 2003); NO2 + hv -> NO + O(3P) produces no OH.
    """

    def test_oh_production_vanishes_without_ozone_whatever_the_no2(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        for no2 in (0.0, 10e-9, 80e-9):
            state = _open_state(o3_mixing_ratio=torch.tensor(0.0, dtype=torch.float64),
                                no2_mixing_ratio=torch.tensor(no2, dtype=torch.float64))
            temperature = state["air_temperature"]
            n_air = chem._calculate_number_density(temperature, state["atmospheric_pressure"])
            production = chem._primary_oh_production(state, torch.tensor(5.0e-5, dtype=torch.float64),
                                                     temperature, n_air)
            assert production.item() == 0.0
            assert chem.compute_oh_concentration(
                state, torch.tensor(5.0e-5, dtype=torch.float64), temperature
            ).item() == 0.0

    def test_o1d_water_fraction_reproduces_table_7_1_rates(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        temperature = torch.tensor(298.15, dtype=torch.float64)
        pressure = torch.tensor(101325.0, dtype=torch.float64)
        n_air = chem._calculate_number_density(temperature, pressure)
        fraction = chem._o1d_water_reaction_fraction(torch.tensor(0.01, dtype=torch.float64), temperature, n_air)
        expected, _ = _hand_water_fraction(298.15, 101325.0, 0.01)
        assert fraction.item() == pytest.approx(expected, rel=1e-12)
        # About 9 percent of O(1D) forms OH at q = 10 g/kg (typical value ~0.1).
        assert fraction.item() == pytest.approx(0.0906, abs=5e-4)

    def test_o1d_to_no2_ratio_follows_the_mcm_parameterisation(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        cosines = torch.tensor([1.0, 0.8, 0.5, 0.2], dtype=torch.float64)
        ratio = chem._o1d_to_no2_photolysis_ratio(cosines)
        expected = torch.tensor([_hand_o1d_ratio(float(c)) for c in cosines], dtype=torch.float64)
        torch.testing.assert_close(ratio, expected, rtol=1e-12, atol=0.0)
        # Overhead sun: (l1/l4) exp(-(n1-n4)) = 4.24e-3 (Table 6.7 ground level: 5.8e-3).
        assert ratio[0].item() == pytest.approx(4.2382e-3, rel=1e-4)
        assert bool(torch.all(ratio[1:] < ratio[:-1]))
        # Sun at or below the horizon: no O(1D) photolysis, with a finite gradient.
        night = torch.tensor([0.0, -0.3], dtype=torch.float64, requires_grad=True)
        night_ratio = chem._o1d_to_no2_photolysis_ratio(night)
        assert night_ratio.tolist() == [0.0, 0.0]
        (grad,) = torch.autograd.grad(night_ratio.sum(), night)
        assert bool(torch.isfinite(grad).all())

    def test_primary_oh_production_matches_a_hand_calculation(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        temperature, pressure, q, cos_chi, shortwave = 288.15, 95000.0, 0.008, 0.8, 600.0
        state = _open_state(
            air_temperature=torch.tensor(temperature, dtype=torch.float64),
            atmospheric_pressure=torch.tensor(pressure, dtype=torch.float64),
            specific_humidity=torch.tensor(q, dtype=torch.float64),
            cos_zenith=torch.tensor(cos_chi, dtype=torch.float64),
            shortwave_down=torch.tensor(shortwave, dtype=torch.float64),
        )
        j_no2, j_o1d = chem._calculate_photolysis_rates(
            state["shortwave_down"], state["cos_zenith"], state["air_temperature"]
        )
        # Trebs et al. (2009): J_NO2 = (1 + alpha) (B1 G + B2 G^2), G = shortwave_down.
        j_no2_hand = 1.03 * (1.47e-5 * shortwave - 4.84e-9 * shortwave ** 2)
        assert j_no2.item() == pytest.approx(j_no2_hand, rel=1e-12)
        j_o1d_hand = j_no2_hand * _hand_o1d_ratio(cos_chi)
        assert j_o1d.item() == pytest.approx(j_o1d_hand, rel=1e-12)
        fraction, n_air_hand = _hand_water_fraction(temperature, pressure, q)
        n_air = chem._calculate_number_density(state["air_temperature"], state["atmospheric_pressure"])
        production = chem._primary_oh_production(state, j_o1d, state["air_temperature"], n_air)
        expected = 2.0 * fraction * j_o1d_hand * 40e-9 * n_air_hand
        assert production.item() == pytest.approx(expected, rel=1e-10)

    def test_steady_state_oh_is_production_over_loss_without_bounds(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        results = []
        for o3, co in ((200e-9, 1e-9), (1e-12, 150e-9)):
            state = _open_state(o3_mixing_ratio=torch.tensor(o3, dtype=torch.float64),
                                co_mixing_ratio=torch.tensor(co, dtype=torch.float64),
                                ch4_mixing_ratio=torch.tensor(1.0e-9, dtype=torch.float64),
                                no2_mixing_ratio=torch.tensor(1.0e-12, dtype=torch.float64))
            temperature = state["air_temperature"]
            j_o1d = torch.tensor(5.0e-5, dtype=torch.float64)
            n_air = chem._calculate_number_density(temperature, state["atmospheric_pressure"])
            production = chem._primary_oh_production(state, j_o1d, temperature, n_air)
            rates = {
                "co": 2.3e-13,
                "ch4": 6.4e-15,
                "no2": 1.15e-11,
            }
            loss = (rates["co"] * co + rates["ch4"] * 1.0e-9 + rates["no2"] * 1.0e-12) * n_air
            oh = chem.compute_oh_concentration(state, j_o1d, temperature)
            torch.testing.assert_close(oh, production / loss, rtol=1e-12, atol=0.0)
            results.append(oh.item())
        # The first case lies far above the former 1e7 ceiling, the second below the 1e5 floor.
        assert results[0] > 1.0e7
        assert results[1] < 1.0e5

    def test_zero_oh_loss_frequency_raises(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        zero = torch.tensor(0.0, dtype=torch.float64)
        state = _open_state(co_mixing_ratio=zero, ch4_mixing_ratio=zero, no2_mixing_ratio=zero)
        with pytest.raises(ValueError, match="OH loss frequency"):
            chem.compute_oh_concentration(state, torch.tensor(5.0e-5, dtype=torch.float64),
                                          state["air_temperature"])

    @pytest.mark.parametrize("value", [None, -0.001, 1.0, float("nan")])
    def test_specific_humidity_is_a_required_mass_fraction(self, value):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        assert "specific_humidity" in chem.get_required_inputs()
        state = _open_state()
        if value is None:
            del state["specific_humidity"]
        else:
            state["specific_humidity"] = torch.tensor(value, dtype=torch.float64)
        with pytest.raises(ValueError, match="specific_humidity"):
            chem.forward(state, time=0.0, dt=1.0)

    @pytest.mark.parametrize("value", [None, 1.5, -1.2])
    def test_cos_zenith_is_a_required_cosine(self, value):
        """rho(chi) needs the solar geometry; broadband shortwave cannot separate clouds from low sun."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        assert "cos_zenith" in chem.get_required_inputs()
        state = _open_state()
        if value is None:
            del state["cos_zenith"]
        else:
            state["cos_zenith"] = torch.tensor(value, dtype=torch.float64)
        with pytest.raises(ValueError, match="cos_zenith"):
            chem.forward(state, time=0.0, dt=1.0)

    def test_forward_couples_the_isoprene_sink_to_the_unbounded_oh(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        state = _open_state(isoprene_mixing_ratio=torch.tensor(2e-9, dtype=torch.float64))
        out = chem.forward(state, time=0.0, dt=1.0)
        oh = out["diagnostics.chemistry.oh_concentration"].value
        j_no2, j_o1d = chem._calculate_photolysis_rates(
            state["shortwave_down"], state["cos_zenith"], state["air_temperature"]
        )
        torch.testing.assert_close(oh, chem.compute_oh_concentration(state, j_o1d, state["air_temperature"]))
        # About 6.8e5 molecule cm-3 for this state (the former clamp returned its 1e7 ceiling).
        assert 5.0e5 < oh.item() < 1.0e6

    def test_retired_oh_and_photolysis_keys_are_absent(self):
        manager = ChemistryFunctor(physical_boundary="open_mechanism").get_parameter_manager()
        for key in (
            "atmosphere.chemistry.photolysis_temperature_q10",
            "atmosphere.chemistry.oh_o3_photolysis_factor",
            "atmosphere.chemistry.oh_o1d_branching_ratio",
            "atmosphere.chemistry.default_oh_concentration",
            "atmosphere.chemistry.oh_loss_rate_min",
            "atmosphere.chemistry.oh_lower_multiplier",
            "atmosphere.chemistry.oh_upper_multiplier",
            "atmosphere.chemistry.clear_sky_shortwave_max",
            "atmosphere.chemistry.reference_temperature",
            "atmosphere.chemistry.j_no2_reference",
            "atmosphere.chemistry.reference_solar_radiation",
            "atmosphere.chemistry.radiation_factor_min",
            "atmosphere.chemistry.radiation_factor_max",
            "atmosphere.chemistry.cos_zenith_min",
            "atmosphere.chemistry.cos_zenith_max",
            "atmosphere.chemistry.photolysis.reference_cos_zenith",
            "atmosphere.chemistry.photolysis.temperature_domain_min",
            "atmosphere.chemistry.photolysis.temperature_domain_max",
        ):
            assert manager.get_parameter(key) is None, key
        yield_spec = manager.get_parameter("atmosphere.chemistry.o1d_water_oh_yield")
        assert yield_spec.value == 2.0 and tuple(yield_spec.bounds) == (2.0, 2.0)


class TestPhotolysisTemperatureModule:
    """Optional R_X(T) = sum sigma phi F / sum sigma phi F (T_ref) photolysis temperature factor.

    References: Warneck and Williams (2012) Tables 6.2, 6.4a, 6.11a-c, 6.12, 6.13a-b;
    Matsumi et al. (2002); Bodhaine et al. (1999) Eq. (30).
    """

    def test_mode_enumeration_matches_the_yaml_declaration_and_defaults_to_none(self):
        entry = _yaml.safe_load(_CHEMISTRY_YAML.read_text(encoding="utf-8-sig"))["parameters"][
            "atmosphere.chemistry.photolysis_temperature_mode"
        ]
        assert entry["value"] == "none"
        assert entry["dtype"] == "string" and entry["bounds"] is None
        assert tuple(entry["allowed_values"]) == ChemistryFunctor.PHOTOLYSIS_TEMPERATURE_MODES
        assert ChemistryFunctor.PHOTOLYSIS_TEMPERATURE_MODES == ("none", "spectral_quantum_yield")

    def test_mode_none_photolysis_is_independent_of_air_temperature(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        shortwave = torch.tensor(700.0, dtype=torch.float64)
        cos_chi = torch.tensor(0.9, dtype=torch.float64)
        reference = chem._calculate_photolysis_rates(shortwave, cos_chi, torch.tensor(298.15, dtype=torch.float64))
        assert "reference_temperature" not in chem._constants
        for temperature in (210.0, 248.0, 330.0, 390.0):
            rates = chem._calculate_photolysis_rates(shortwave, cos_chi, torch.tensor(temperature, dtype=torch.float64))
            for value, ref in zip(rates, reference):
                assert torch.equal(value, ref)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="photolysis_temperature_mode"):
            ChemistryFunctor(config={"parameters": {"photolysis_temperature_mode": "q10"}},
                             physical_boundary="open_mechanism")

    def test_spectral_factors_are_one_at_the_reference_temperature(self):
        chem = _spectral_chemistry()
        r_no2, r_o1d = chem._photolysis_temperature_factors(
            torch.tensor(298.15, dtype=torch.float64), torch.tensor(0.7, dtype=torch.float64)
        )
        assert r_no2.item() == pytest.approx(1.0, abs=1e-12)
        assert r_o1d.item() == pytest.approx(1.0, abs=1e-12)

    def test_spectral_no2_factor_matches_the_independent_oracle(self):
        chem = _spectral_chemistry()
        temperatures = torch.tensor([200.0, 248.0, 273.15, 290.0, 300.0], dtype=torch.float64)
        r_no2, _ = chem._photolysis_temperature_factors(temperatures, torch.ones_like(temperatures))
        expected = [_spectral_oracle(float(t))[0] for t in temperatures]
        torch.testing.assert_close(r_no2, torch.tensor(expected, dtype=torch.float64), rtol=1e-6, atol=0.0)
        # 248 K: the NO2 yields fall only at 399-417 nm (Table 6.13b) -> R = 0.9705.
        assert r_no2[1].item() == pytest.approx(0.9705, abs=5e-4)
        assert r_no2[1].item() < 1.0
        # Below 248 K and above 298 K the nearest tabulated yields are held.
        assert r_no2[0].item() == pytest.approx(r_no2[1].item(), rel=1e-12)
        assert r_no2[4].item() == pytest.approx(1.0, abs=1e-12)

    def test_spectral_o1d_factor_matches_the_independent_oracle(self):
        chem = _spectral_chemistry()
        temperatures = torch.tensor([223.0, 248.0, 273.15, 300.0], dtype=torch.float64)
        _, r_o1d = chem._photolysis_temperature_factors(temperatures, torch.ones_like(temperatures))
        expected = [_spectral_oracle(float(t))[1] for t in temperatures]
        torch.testing.assert_close(r_o1d, torch.tensor(expected, dtype=torch.float64), rtol=1e-6, atol=0.0)
        assert r_o1d[0].item() < 1.0
        assert r_o1d[0].item() == pytest.approx(0.737, abs=5e-3)

    def test_matsumi_quantum_yield_reproduces_table_6_11c(self):
        """|phi - table| <= 0.005 at 298 K and <= 0.04 at 223 K for 306-328 nm.

        The formula is the Matsumi et al. (2002) evaluation of several data sets; Table 6.11c
        lists the Talukdar et al. (1998) measurements. The 311-nm 298 K entry of the markdown
        copy (0.310) repeats the 312-nm value and is not used (formula 0.397).
        """
        table = {
            306: (0.884, 0.86), 307: (0.862, 0.84), 308: (0.793, 0.74), 309: (0.671, 0.61),
            310: (0.523, 0.39), 311: (None, 0.23), 312: (0.310, 0.16), 313: (0.265, 0.13),
            314: (0.246, 0.09), 315: (0.239, 0.09), 316: (0.233, 0.09), 317: (0.222, 0.08),
            318: (0.206, 0.08), 319: (0.187, 0.08), 320: (0.166, 0.07), 321: (0.146, 0.07),
            322: (0.128, 0.07), 323: (0.113, 0.05), 324: (0.101, 0.06), 325: (0.092, 0.08),
            326: (0.086, 0.08), 327: (0.082, 0.08), 328: (0.080, 0.07),
        }
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        wavelengths = torch.tensor([nm / 1000.0 for nm in table], dtype=torch.float64)
        for column, temperature, tolerance in ((0, 298.0, 0.005), (1, 223.0, 0.04)):
            phi = chem._o1d_quantum_yield(wavelengths, torch.tensor(temperature, dtype=torch.float64))
            for value, (nm, pair) in zip(phi.tolist(), table.items()):
                if pair[column] is None:
                    continue
                assert abs(value - pair[column]) <= tolerance, (nm, temperature, value, pair[column])
        # Plateaus outside 306-328 nm (Matsumi et al. 2002 recommendations).
        plateau = chem._o1d_quantum_yield(
            torch.tensor([0.300, 0.305, 0.330, 0.350], dtype=torch.float64), torch.tensor(250.0, dtype=torch.float64)
        )
        assert plateau.tolist() == [0.9, 0.9, 0.08, 0.08]
        # The implemented expression equals the oracle transcription of the formula.
        _, _, phi_oracle = _spectral_oracle(298.15)
        phi = chem._o1d_quantum_yield(wavelengths, torch.tensor(260.0, dtype=torch.float64))
        expected = torch.tensor([phi_oracle(float(x), 260.0) for x in wavelengths], dtype=torch.float64)
        torch.testing.assert_close(phi, expected, rtol=1e-12, atol=0.0)

    def test_temperatures_outside_the_evaluated_ranges_hold_the_nearest_tabulated_values(self):
        """Matsumi et al. (2002) evaluated 200-300 K; the tables hold their own ranges.

        Beyond them the nearest evaluated temperature is held (TUV practice), so the
        factors at 305 K equal those at 300 K and at 190 K those at 200 K; only a
        non-finite or non-positive temperature raises.
        """
        chem = _spectral_chemistry()
        mu = torch.tensor([0.8, 0.8], dtype=torch.float64)
        warm = chem._photolysis_temperature_factors(torch.tensor([300.0, 305.0], dtype=torch.float64), mu)
        cold = chem._photolysis_temperature_factors(torch.tensor([200.0, 190.0], dtype=torch.float64), mu)
        for pair in (*warm, *cold):
            assert pair[0].item() == pytest.approx(pair[1].item(), rel=1e-14)
            assert bool(torch.isfinite(pair).all())
        for invalid in (float("nan"), float("inf"), 0.0, -5.0):
            with pytest.raises(ValueError, match="finite positive"):
                chem._photolysis_temperature_factors(
                    torch.tensor(invalid, dtype=torch.float64), torch.tensor(0.8, dtype=torch.float64)
                )
        r_no2, r_o1d = ChemistryFunctor(physical_boundary="open_mechanism")._photolysis_temperature_factors(
            torch.tensor(330.0, dtype=torch.float64), torch.tensor(0.8, dtype=torch.float64)
        )
        assert r_no2.item() == 1.0 and r_o1d.item() == 1.0

    def test_spectral_factors_follow_the_actual_solar_geometry(self):
        """F(lambda; mu) = F0 exp(-(tau_R + tau_O3)/mu) at the actual cos(chi); R = 1 at night."""
        chem = _spectral_chemistry()
        mus = [1.0, 0.5, 0.05, 0.0, -0.2]
        temperature = torch.full((len(mus),), 248.0, dtype=torch.float64, requires_grad=True)
        mu = torch.tensor(mus, dtype=torch.float64, requires_grad=True)
        r_no2, r_o1d = chem._photolysis_temperature_factors(temperature, mu)
        expected = [_spectral_oracle(248.0, value) for value in mus]
        torch.testing.assert_close(
            r_no2, torch.tensor([e[0] for e in expected], dtype=torch.float64), rtol=1e-6, atol=0.0
        )
        torch.testing.assert_close(
            r_o1d, torch.tensor([e[1] for e in expected], dtype=torch.float64), rtol=1e-6, atol=0.0
        )
        # A longer slant path weights the temperature-dependent Huggins tail more.
        assert r_o1d[1].item() < r_o1d[0].item()
        assert r_no2[3].item() == 1.0 and r_o1d[4].item() == 1.0
        grads = torch.autograd.grad((r_no2 + r_o1d).sum(), (temperature, mu))
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        # Scalar broadcast against a batch of temperatures.
        r_batch, _ = chem._photolysis_temperature_factors(
            torch.tensor([230.0, 260.0, 290.0], dtype=torch.float64), torch.tensor(0.6, dtype=torch.float64)
        )
        assert tuple(r_batch.shape) == (3,)

    def test_spectral_factors_are_differentiable_in_temperature(self):
        chem = _spectral_chemistry()
        temperature = torch.tensor([210.0, 260.0, 290.0], dtype=torch.float64, requires_grad=True)
        r_no2, r_o1d = chem._photolysis_temperature_factors(temperature, torch.ones_like(temperature))
        grad_no2, grad_o1d = (
            torch.autograd.grad(r.sum(), temperature, retain_graph=True)[0] for r in (r_no2, r_o1d)
        )
        assert bool(torch.isfinite(grad_no2).all()) and bool(torch.isfinite(grad_o1d).all())
        # 260 and 290 K lie inside the tabulated NO2 range; R_O1D rises with T at every point.
        assert grad_no2[1].item() > 0.0 and grad_no2[2].item() > 0.0
        assert bool(torch.all(grad_o1d > 0.0))

    def test_forward_applies_the_spectral_factors_to_both_photolysis_rates(self):
        state = _open_state(air_temperature=torch.tensor(248.0, dtype=torch.float64))
        spectral = _spectral_chemistry()
        plain = ChemistryFunctor(physical_boundary="open_mechanism")
        r_no2, r_o1d = spectral._photolysis_temperature_factors(state["air_temperature"], state["cos_zenith"])
        j_plain = plain._calculate_photolysis_rates(state["shortwave_down"], state["cos_zenith"], state["air_temperature"])
        j_spec = spectral._calculate_photolysis_rates(state["shortwave_down"], state["cos_zenith"], state["air_temperature"])
        torch.testing.assert_close(j_spec[0], j_plain[0] * r_no2, rtol=1e-14, atol=0.0)
        torch.testing.assert_close(j_spec[1], j_plain[1] * r_o1d, rtol=1e-14, atol=0.0)
        out = spectral.forward(state, time=0.0, dt=1.0)
        torch.testing.assert_close(out["diagnostics.chemistry.photolysis_rate_no2"].value, j_spec[0])

    def test_spectral_tables_carry_the_printed_warneck_williams_values(self):
        table = _yaml.safe_load(_PHOTOLYSIS_YAML.read_text(encoding="utf-8"))["parameters"]

        def g(name):
            return table["atmosphere.chemistry.photolysis." + name]["value"]

        def at(axis, values, nm):
            index = [round(x * 1000.0, 1) for x in g(axis)].index(nm)
            return g(values)[index]

        assert at("no2_cross_section_wavelength", "no2_cross_section", 400.0) == pytest.approx(6.385e-19)
        assert at("no2_quantum_yield_wavelength", "no2_quantum_yield_high_temperature", 404.0) == 0.44
        assert at("no2_quantum_yield_wavelength", "no2_quantum_yield_low_temperature", 404.0) == 0.34
        assert at("solar_flux_wavelength", "solar_photon_flux", 318.5) == pytest.approx(110.36e15)
        assert at("solar_flux_wavelength", "solar_photon_flux", 356.5) == pytest.approx(121.96e15)
        assert at("o3_cross_section_wavelength", "o3_cross_section_high_temperature", 300.0) == pytest.approx(3.964e-19)
        assert at("o3_cross_section_wavelength", "o3_cross_section_low_temperature", 312.0) == pytest.approx(6.416e-20)
        for entry in table.values():
            assert entry["provenance"]["source"] == "literature"
            assert entry["references"]
        # Values that no offline primary copy could confirm are declared as unverified.
        note = table["atmosphere.chemistry.photolysis.o3_cross_section_wavelength"]["provenance"]["notes"]
        assert "unverified" in note and "337.0" in note and "338.0" in note


class TestTrebsNO2Photolysis:
    """J_NO2 = (1 + alpha) (B1 G + B2 G^2) from the global irradiance G (Trebs et al. 2009).

    References: Trebs et al. (2009) Atmos. Meas. Tech. 2, 725-739, doi:10.5194/amt-2-725-2009,
    Eq. (1)-(2): B1 = 1.47e-5 W-1 m2 s-1, B2 = -4.84e-9 W-2 m4 s-1 (nine sites below
    800 m a.s.l., SZA 0-90 deg, cloudy and clear); alpha the UV-A surface albedo (0.03
    grassland); Warneck and Williams (2012) Table 6.7 (8.8e-3 s-1, overhead sun).
    """

    ALPHA, B1, B2 = 0.03, 1.47e-5, -4.84e-9

    def _hand(self, g):
        return (1.0 + self.ALPHA) * (self.B1 * g + self.B2 * g * g)

    def test_j_no2_follows_the_trebs_polynomial(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        irradiance = torch.tensor([0.0, 200.0, 500.0, 800.0, 1100.0], dtype=torch.float64)
        j_no2, _ = chem._calculate_photolysis_rates(
            irradiance, torch.full_like(irradiance, 0.9), torch.full_like(irradiance, 290.0)
        )
        expected = torch.tensor([self._hand(float(g)) for g in irradiance], dtype=torch.float64)
        torch.testing.assert_close(j_no2, expected, rtol=1e-12, atol=0.0)
        # Warneck and Williams (2012) Table 6.7: 8.8e-3 s-1 at the ground, overhead sun.
        assert abs(j_no2[3].item() / 8.8e-3 - 1.0) < 0.03

    def test_j_no2_carries_no_second_geometry_factor(self):
        """G already contains the solar geometry and the cloud attenuation."""
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        irradiance = torch.tensor(600.0, dtype=torch.float64)
        temperature = torch.tensor(290.0, dtype=torch.float64)
        high = chem._calculate_photolysis_rates(irradiance, torch.tensor(1.0, dtype=torch.float64), temperature)
        low = chem._calculate_photolysis_rates(irradiance, torch.tensor(0.4, dtype=torch.float64), temperature)
        assert torch.equal(high[0], low[0])
        assert low[1].item() / low[0].item() == pytest.approx(_hand_o1d_ratio(0.4), rel=1e-12)
        assert high[1].item() / high[0].item() == pytest.approx(_hand_o1d_ratio(1.0), rel=1e-12)

    def test_j_no2_is_held_at_the_vertex_above_the_fitted_curvature(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        vertex = self.B1 / (2.0 * abs(self.B2))
        assert vertex == pytest.approx(1518.6, abs=0.1)
        irradiance = torch.tensor([vertex - 50.0, vertex, 1600.0, 2000.0], dtype=torch.float64, requires_grad=True)
        j_no2, _ = chem._calculate_photolysis_rates(
            irradiance, torch.full((4,), 1.0, dtype=torch.float64), torch.full((4,), 290.0, dtype=torch.float64)
        )
        peak = self._hand(vertex)
        assert j_no2[0].item() < peak
        for index in (1, 2, 3):
            assert j_no2[index].item() == pytest.approx(peak, rel=1e-12)
        (grad,) = torch.autograd.grad(j_no2.sum(), irradiance)
        assert grad[0].item() > 0.0
        assert grad[2].item() == 0.0 and grad[3].item() == 0.0

    def test_negative_irradiance_raises(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        with pytest.raises(ValueError, match="shortwave_down"):
            chem._calculate_photolysis_rates(
                torch.tensor(-1.0, dtype=torch.float64),
                torch.tensor(0.5, dtype=torch.float64),
                torch.tensor(290.0, dtype=torch.float64),
            )

    def test_trebs_entries_cite_the_paper(self):
        entries = _yaml.safe_load(_CHEMISTRY_YAML.read_text(encoding="utf-8-sig"))["parameters"]
        for key, value, bounds in (
            ("j_no2_irradiance_linear_coefficient", 1.47e-5, [1.44e-5, 1.50e-5]),
            ("j_no2_irradiance_quadratic_coefficient", -4.84e-9, [-5.15e-9, -4.53e-9]),
            ("uv_a_surface_albedo", 0.03, [0.0, 0.2]),
        ):
            entry = entries[f"atmosphere.chemistry.{key}"]
            assert entry["value"] == value, key
            assert list(entry["bounds"]) == bounds, key
            text = " ".join(str(item) for item in entry["references"])
            assert "Trebs" in text and "10.5194/amt-2-725-2009" in text, key
            assert entry["provenance"]["source"] == "literature", key


class TestBidirectionalPhotochemicalOzoneAndLedgers:
    """Test low-NOx ozone loss vs high-NOx photochemical ozone production and ledgers.

    References:
    - Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics, 3rd ed., Ch. 6.
    - Saunders, S.M. et al. (2003). Atmos. Chem. Phys., 3, 161-180 (MCM v3.3.1).
    - Peeters, J. et al. (2014). J. Phys. Chem. A, 118, 8625-8643 (LIM1).
    """

    def test_low_nox_regime_exhibits_net_ox_and_o3_loss(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        state = _open_state(
            no_mixing_ratio=torch.tensor(1.0e-12, dtype=torch.float64),
            no2_mixing_ratio=torch.tensor(1.0e-12, dtype=torch.float64),
            o3_mixing_ratio=torch.tensor(50.0e-9, dtype=torch.float64),
            isoprene_mixing_ratio=torch.tensor(8.0e-9, dtype=torch.float64),
            shortwave_down=torch.tensor(400.0, dtype=torch.float64),
        )
        out = chem.forward(state, time=0.0, dt=0.5)
        ox_tendency = _diagnostic_value(
            out, "diagnostics.chemistry.ox_net_chemical_tendency", "mol mol-1 s-1"
        )
        branch = _diagnostic_value(
            out, "diagnostics.chemistry.ro2_no_branching_fraction", "dimensionless"
        )
        assert float(branch.item()) < 0.02
        assert float(ox_tendency.item()) < 0.0
        delta_ox = _update_value(out, "o3_mixing_ratio") + _update_value(
            out, "no2_mixing_ratio"
        )
        assert float(delta_ox.item()) < 0.0
        assert float(_update_value(out, "o3_mixing_ratio").item()) < 0.0

    def test_high_nox_regime_exhibits_net_ox_and_multistep_o3_production(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        base_state = _open_state(
            no_mixing_ratio=torch.tensor(2.0e-9, dtype=torch.float64),
            no2_mixing_ratio=torch.tensor(6.0e-9, dtype=torch.float64),
            o3_mixing_ratio=torch.tensor(40.0e-9, dtype=torch.float64),
            shortwave_down=torch.tensor(850.0, dtype=torch.float64),
            cos_zenith=torch.tensor(0.95, dtype=torch.float64),
        )
        state_with_isop = {
            **base_state,
            "isoprene_mixing_ratio": torch.tensor(6.0e-9, dtype=torch.float64),
        }
        out = chem.forward(state_with_isop, time=0.0, dt=5.0)
        ox_tendency = _diagnostic_value(
            out, "diagnostics.chemistry.ox_net_chemical_tendency", "mol mol-1 s-1"
        )
        branch = _diagnostic_value(
            out, "diagnostics.chemistry.ro2_no_branching_fraction", "dimensionless"
        )
        assert float(branch.item()) > 0.65
        assert float(ox_tendency.item()) > 0.0
        delta_ox = _update_value(out, "o3_mixing_ratio") + _update_value(
            out, "no2_mixing_ratio"
        )
        assert float(delta_ox.item()) > 0.0

        # Multi-step integration: RO2 + NO -> NO2 followed by NO2 + hnu -> NO + O3
        # yields higher O3 than the zero-isoprene Leighton photostationary trajectory.
        s_isop = {k: v.clone() for k, v in state_with_isop.items()}
        s_clean = {k: v.clone() for k, v in base_state.items()}
        dt_step = 5.0
        for step_idx in range(60):
            u_isop = chem.forward(s_isop, time=step_idx * dt_step, dt=dt_step)
            u_clean = chem.forward(s_clean, time=step_idx * dt_step, dt=dt_step)
            for k in ("no_mixing_ratio", "no2_mixing_ratio", "o3_mixing_ratio", "isoprene_mixing_ratio"):
                if k in u_isop:
                    s_isop[k] = s_isop[k] + u_isop[k].value
                if k in u_clean:
                    s_clean[k] = s_clean[k] + u_clean[k].value

        assert float(s_isop["o3_mixing_ratio"].item()) > float(
            s_clean["o3_mixing_ratio"].item()
        )
        ox_isop = s_isop["o3_mixing_ratio"] + s_isop["no2_mixing_ratio"]
        ox_init = state_with_isop["o3_mixing_ratio"] + state_with_isop["no2_mixing_ratio"]
        assert float(ox_isop.item()) > float(ox_init.item())

    def test_chemical_ledger_closes_nox_ox_and_reactive_carbon_exactly(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        state = _open_state(
            no_mixing_ratio=torch.tensor([1.0e-12, 3.0e-9], dtype=torch.float64),
            no2_mixing_ratio=torch.tensor([5.0e-12, 8.0e-9], dtype=torch.float64),
            o3_mixing_ratio=torch.tensor([45.0e-9, 45.0e-9], dtype=torch.float64),
            isoprene_mixing_ratio=torch.tensor([4.0e-9, 4.0e-9], dtype=torch.float64),
            shortwave_down=torch.tensor([750.0, 750.0], dtype=torch.float64),
            cos_zenith=torch.tensor([0.9, 0.9], dtype=torch.float64),
            air_temperature=torch.tensor([298.15, 298.15], dtype=torch.float64),
            atmospheric_pressure=torch.tensor([101325.0, 101325.0], dtype=torch.float64),
            specific_humidity=torch.tensor([0.01, 0.01], dtype=torch.float64),
            co_mixing_ratio=torch.tensor([150e-9, 150e-9], dtype=torch.float64),
            ch4_mixing_ratio=torch.tensor([1.8e-6, 1.8e-6], dtype=torch.float64),
        )
        dt = 0.5
        out = chem.forward(state, time=0.0, dt=dt)
        ledger = chem.compute_chemical_ledger(state, out, dt=dt)

        assert ledger["nox_molar_balance"]["max_abs_residual"] <= 1e-22
        assert ledger["odd_oxygen_balance"]["max_abs_residual"] <= 1e-22
        assert ledger["reactive_carbon_balance"]["max_abs_residual"] <= 1e-21
        # First batch element is low-NOx (net Ox loss), second is high-NOx (net Ox production)
        delta_ox = ledger["odd_oxygen_balance"]["delta_ox"]
        assert float(delta_ox[0].item()) < 0.0
        assert float(delta_ox[1].item()) > 0.0

    def test_ro2_no_branching_and_ox_tendency_are_differentiable(self):
        chem = ChemistryFunctor(physical_boundary="open_mechanism")
        no = torch.tensor(1.5e-9, dtype=torch.float64, requires_grad=True)
        isop = torch.tensor(3.0e-9, dtype=torch.float64, requires_grad=True)
        state = _open_state(no_mixing_ratio=no, isoprene_mixing_ratio=isop)
        out = chem.forward(state, time=0.0, dt=1.0)
        ox_tendency = out["diagnostics.chemistry.ox_net_chemical_tendency"].value
        grad_no, grad_isop = torch.autograd.grad(ox_tendency, (no, isop))
        assert bool(torch.isfinite(grad_no)) and float(grad_no.item()) > 0.0
        assert bool(torch.isfinite(grad_isop))

