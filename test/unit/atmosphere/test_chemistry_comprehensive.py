"""
Comprehensive L1 Unit Tests for Atmospheric Chemistry Module

This test suite provides comprehensive coverage of the atmospheric chemistry
functor including NOx-O3 cycle, BVOC oxidation, photolysis calculations,
temperature-dependent kinetics, and chemical accuracy validation.

Test Coverage:
    - Basic functionality and initialization
    - Temperature response functions (Arrhenius)
    - Photolysis rate calculations
    - Chemical reaction rates and kinetics
    - NOx-O3 photostationary cycle
    - BVOC chemistry and OH calculations
    - Scientific accuracy validation
    - Edge cases and boundary conditions
    - Error handling and robustness
    - Parameter sensitivity analysis
    - Mass conservation verification

Scientific Validation:
    - Arrhenius temperature dependence: k(T) = k_298 * exp(Ea/R*(1/298-1/T))
    - Photostationary state: [O3] = J_NO2 * [NO2] / (k_NO_O3 * [NO])
    - Conservation laws: d[X]/dt = Production - Loss
    - Physical bounds: concentrations > 0, rates > 0
    - Literature values for rate constants and activation energies

References:
    - Seinfeld & Pandis (2016). Atmospheric Chemistry and Physics
    - Atkinson & Arey (2003). Atmospheric degradation of volatile organic compounds
    - IUPAC kinetic database for atmospheric chemistry
    - NASA/JPL kinetic evaluation data
"""

import torch
import logging

import pytest

# Import the module to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.core.state import UpdateItem
from test.utils.forward_call import call_forward


def _output_value(result, key):
    """Return the value of one declared canonical chemistry output."""

    item = result[key]
    assert isinstance(item, UpdateItem)
    return item.value


def _extract_activity_rate(result):
    """Extract a scalar-like activity metric from forward/call_forward output."""
    if not isinstance(result, dict):
        return result
    rate_value = result.get('rate')
    if isinstance(rate_value, torch.Tensor):
        return rate_value

    state_keys = ['no_mixing_ratio', 'no2_mixing_ratio', 'o3_mixing_ratio']
    state_values = [
        result[key].value
        for key in state_keys
        if isinstance(result.get(key), UpdateItem)
    ]
    if state_values:
        total_activity = torch.zeros_like(state_values[0])
        for value in state_values:
            total_activity = total_activity + torch.abs(value)
        return total_activity

    for value in result.values():
        if isinstance(value, torch.Tensor):
            return torch.abs(value)
    raise KeyError("No tensor activity metric found in output dictionary.")


class TestChemistryBasicFunctionality:
    """Test basic functionality and interface compliance for ChemistryFunctor."""
    
    @pytest.fixture
    def device(self):
        """Test device (CPU or CUDA if available)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_initialization_default_parameters(self, device):
        """Test chemistry functor initialization with default parameters."""
        chemistry = ChemistryFunctor(device=device, physical_boundary="open_mechanism")
        
        # Check basic attributes
        assert chemistry.name == "atmospheric_chemistry"
        assert hasattr(chemistry, 'metadata')
        assert hasattr(chemistry, 'config')
        
        # Verify parameters are loaded
        assert chemistry.get_parameter_value('uv_a_surface_albedo') is not None
        assert chemistry.get_parameter_value('k_no_o3_298') is not None
        assert chemistry.get_parameter_value('k_isoprene_oh_298') is not None
        
        # Test parameter bounds
        albedo = chemistry.get_parameter_value('uv_a_surface_albedo')
        assert 0.0 <= albedo <= 0.2
        
        k_no_o3 = chemistry.get_parameter_value('k_no_o3_298')
        assert 1e-15 <= k_no_o3 <= 1e-13
    
    def test_initialization_custom_config(self, device):
        """Test chemistry functor with custom configuration."""
        custom_config = {
            'parameters': {
                'uv_a_surface_albedo': {
                    'value': 0.05,
                    'units': 'dimensionless',
                    'description': 'Custom UV-A surface albedo'
                },
                'minimum_concentration': {
                    'value': 1e-4,
                    'units': 'ppb',
                    'description': 'Custom minimum concentration'
                }
            }
        }
        
        chemistry = ChemistryFunctor(config=custom_config, device=device, physical_boundary="open_mechanism")
        
        # Verify custom parameters
        assert chemistry.get_parameter_value('uv_a_surface_albedo') == 0.05
        assert float(chemistry.get_parameter_value('minimum_concentration')) == pytest.approx(1e-13, rel=1e-12, abs=0)
        
        # Verify other parameters still have defaults
        assert chemistry.get_parameter_value('k_no_o3_298') is not None
    
    def test_inherited_functor_keeps_the_chemistry_namespace(self):
        class DerivedChemistry(ChemistryFunctor):
            pass

        chemistry = DerivedChemistry(physical_boundary="open_mechanism")
        assert float(chemistry.get_parameter_value("uv_a_surface_albedo")) == pytest.approx(0.03)

    def test_enum_parameter_name_uses_the_parent_resolution_contract(self):
        from enum import Enum

        class ParameterName(Enum):
            J = "uv_a_surface_albedo"

        value = ChemistryFunctor(physical_boundary="open_mechanism").get_parameter_value(ParameterName.J, required=True)
        assert float(value) == pytest.approx(0.03)

    def test_override_metadata_cannot_replace_canonical_bounds(self):
        valid = ChemistryFunctor(config={"parameters": {"uv_a_surface_albedo": {"value": 0.03, "bounds": None}}}, physical_boundary="open_mechanism")
        assert float(valid.get_parameter_value("uv_a_surface_albedo")) == pytest.approx(0.03)
        with pytest.raises(ValueError, match="bounds"):
            ChemistryFunctor(config={"parameters": {"uv_a_surface_albedo": {"value": -1.0, "bounds": None}}}, physical_boundary="open_mechanism")

    def test_device_consistency(self, device):
        """Test device placement and tensor operations.

        The constructor device contract is owned by FunctorBase.to_device: the
        explicit device must update both the registered torch modules and the
        Functor ``device`` attribute that the parameter getter uses.
        """
        chemistry = ChemistryFunctor(device=device, physical_boundary="open_mechanism")

        # Constructor device contract: explicit device is the Functor device.
        assert chemistry.device == torch.device(device)
        # Parameter getter follows the Functor device, not the CPU placeholder.
        parameter = chemistry.get_parameter_value("k_no_o3_298", required=True)
        assert isinstance(parameter, torch.Tensor)
        assert parameter.device.type == device.type
        # Default construction keeps the pre-registration CPU placeholder.
        assert ChemistryFunctor(physical_boundary="open_mechanism").device == torch.device("cpu")

        # Create test data on specified device
        temperature = torch.tensor(298.15, device=device)

        # Test temperature-dependent rate calculation
        rate_constants = chemistry._calculate_temperature_dependent_rates(temperature)

        # Verify tensors are on correct device
        for rate_name, rate_value in rate_constants.items():
            assert rate_value.device.type == device.type  # Compare device types instead of exact devices
            assert rate_value.dtype == temperature.dtype

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA construction requires a CUDA device")
    @pytest.mark.parametrize("requested_device", ["cuda", torch.device("cuda:0")])
    def test_explicit_cuda_construction_updates_functor_device_and_getter(self, requested_device):
        """Explicit CUDA construction must migrate the Functor, not only nn.Module tensors."""
        configured = torch.tensor(1e-7, dtype=torch.float64, device="cuda", requires_grad=True)
        config = {"parameters": {"minimum_concentration": {"value": configured, "units": "ppm"}}}
        chemistry = ChemistryFunctor(config=config, device=requested_device, physical_boundary="open_mechanism")

        assert chemistry.device == torch.device(requested_device)
        assert chemistry.device.type == "cuda"

        # Getter returns tensors on the Functor device for plain YAML values.
        yaml_value = chemistry.get_parameter_value("k_no_o3_298", required=True)
        assert yaml_value.device.type == "cuda"

        # CUDA leaf override: ppm -> mol mol-1 conversion keeps device and gradient.
        value = chemistry.get_parameter_value("minimum_concentration", required=True)
        assert value.device.type == "cuda"
        assert value.item() == pytest.approx(1e-13, rel=1e-12, abs=0)
        derivative = torch.autograd.grad(value, configured)[0]
        assert derivative.device.type == "cuda"
        assert derivative.item() == pytest.approx(1e-6, rel=1e-12, abs=0)

        # Kinetics evaluated on CUDA inputs stay on CUDA with the input dtype.
        temperature = torch.tensor(298.15, dtype=torch.float64, device="cuda")
        rate_constants = chemistry._calculate_temperature_dependent_rates(temperature)
        for rate_value in rate_constants.values():
            assert rate_value.device.type == "cuda"
            assert rate_value.dtype == torch.float64
    
    def test_dependencies_interface(self, device):
        """Test dependencies interface compliance."""
        chemistry = ChemistryFunctor(device=device, physical_boundary="open_mechanism")
        
        from relic.core.dependency_graph import normalize_dependency_declaration

        declared = normalize_dependency_declaration(chemistry.get_dependencies(), owner=chemistry.name)
        # Chemistry orders after radiation when it is registered (optional provider of
        # shortwave_down and cos_zenith for photolysis).
        assert declared.optional == ('atmospheric_radiation',)
        assert declared.required == ()
    
    def test_functor_metadata(self, device):
        """Test functor metadata completeness."""
        chemistry = ChemistryFunctor(device=device, physical_boundary="open_mechanism")
        
        metadata = chemistry.metadata
        assert metadata.name == "atmospheric_chemistry"
        assert metadata.category is not None
        assert metadata.execution_phase is not None
        assert len(metadata.input_variables) > 0
        assert len(metadata.output_variables) > 0
        assert len(metadata.references) > 0
        assert metadata.conservation_laws == {
            "nox_family": {"quantity": "nox_family"}
        }


class TestTemperatureDependentKinetics:
    """Test temperature-dependent reaction rate calculations."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def chemistry(self, device):
        return ChemistryFunctor(device=device, physical_boundary="open_mechanism")
    
    def test_arrhenius_temperature_response(self, chemistry, device):
        """Test Arrhenius temperature dependence of rate constants."""
        # Test temperatures
        temperatures = torch.tensor([273.15, 298.15, 313.15], device=device)
        
        rate_constants = chemistry._calculate_temperature_dependent_rates(temperatures)
        
        # Verify all rate constants are computed
        expected_rates = ['k_no_o3', 'k_isoprene_oh', 'k_isoprene_o3']
        for rate_name in expected_rates:
            assert rate_name in rate_constants
            rates = rate_constants[rate_name]
            assert rates.shape == temperatures.shape
            assert torch.all(rates > 0)
        
        # Test Arrhenius behavior: rates should change with temperature
        k_no_o3 = rate_constants['k_no_o3']
        
        # For positive activation energy, rate should increase with temperature
        # k_no_o3 has positive Ea, so k(273K) < k(298K) < k(313K)
        assert k_no_o3[0] < k_no_o3[1] < k_no_o3[2]
    
    def test_temperature_bounds_validation(self, chemistry, device):
        """Test temperature validation and bounds checking."""
        # Test extreme temperatures
        extreme_temps = torch.tensor([200.0, 400.0], device=device)
        
        rate_constants = chemistry._calculate_temperature_dependent_rates(extreme_temps)
        
        # All rate constants should be positive and finite
        for rate_name, rates in rate_constants.items():
            assert torch.all(torch.isfinite(rates))
            assert torch.all(rates > 0)
    
    def test_rate_constant_magnitudes(self, chemistry, device):
        """Test that rate constants have realistic magnitudes."""
        temperature = torch.tensor(298.15, device=device)
        
        rate_constants = chemistry._calculate_temperature_dependent_rates(temperature)
        
        # Check realistic magnitudes (literature values)
        k_no_o3 = rate_constants['k_no_o3'].item()
        assert 5e-15 <= k_no_o3 <= 5e-14  # cm3 molecule-1 s-1
        k_isoprene_oh = rate_constants['k_isoprene_oh'].item()
        assert 5e-11 <= k_isoprene_oh <= 2e-10  # cm3 molecule-1 s-1
        k_isoprene_o3 = rate_constants['k_isoprene_o3'].item()
        assert 5e-18 <= k_isoprene_o3 <= 5e-17  # cm3 molecule-1 s-1

class TestPhotolysisCalculations:
    """Test photolysis rate calculations and light dependence."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def chemistry(self, device):
        return ChemistryFunctor(device=device, physical_boundary="open_mechanism")
    
    def test_photolysis_light_dependence(self, chemistry, device):
        """Test photolysis rate dependence on solar radiation."""
        shortwave_radiation = torch.tensor([0.0, 200.0, 600.0, 1000.0], device=device)
        cos_zenith = torch.ones_like(shortwave_radiation)
        temperature = torch.full_like(shortwave_radiation, 298.15)
        
        # Calculate photolysis rates
        j_no2_rates, _ = chemistry._calculate_photolysis_rates(shortwave_radiation, cos_zenith, temperature)
        
        assert j_no2_rates.shape == shortwave_radiation.shape
        
        # Photolysis should increase with radiation
        assert torch.all(j_no2_rates >= 0)
        assert j_no2_rates[0] < j_no2_rates[1] < j_no2_rates[2] < j_no2_rates[3]
        
        # Dark conditions should give near-zero photolysis
        assert j_no2_rates[0].item() < 1e-6
    
    def test_zenith_angle_correction(self, chemistry, device):
        """Test cosine zenith angle correction for photolysis."""
        shortwave = torch.tensor(800.0, device=device)
        cos_zenith_values = torch.tensor([0.0, 0.5, 0.87, 1.0], device=device)  # 90 deg, 60 deg, 30 deg, 0 deg
        
        j_no2_rates, j_o1d_rates = chemistry._calculate_photolysis_rates(
            shortwave, cos_zenith_values, torch.tensor(298.15, device=device)
        )
        
        # J_NO2 depends on the global irradiance only (it carries the geometry);
        # J(O1D) rises with cos(zenith angle) through the MCM J1/J4 ratio.
        assert torch.all(j_no2_rates >= 0)
        assert torch.all(j_no2_rates == j_no2_rates[0])
        assert j_o1d_rates[0].item() == 0.0
        assert j_o1d_rates[1] < j_o1d_rates[2] < j_o1d_rates[3]
    
    def test_photolysis_temperature_modes(self, chemistry, device):
        """Mode none: J independent of air temperature; spectral mode: evaluated yields."""
        shortwave = torch.tensor(600.0, device=device)
        cos_zenith = torch.tensor(1.0, device=device)
        cold, warm = torch.tensor(273.15, device=device), torch.tensor(298.15, device=device)
        
        j_cold = chemistry._calculate_photolysis_rates(shortwave, cos_zenith, cold)
        j_warm = chemistry._calculate_photolysis_rates(shortwave, cos_zenith, warm)
        for value_cold, value_warm in zip(j_cold, j_warm):
            assert torch.equal(value_cold, value_warm)
        
        spectral = ChemistryFunctor(
            device=device,
            physical_boundary="open_mechanism",
            config={"parameters": {"photolysis_temperature_mode": "spectral_quantum_yield"}},
        )
        s_cold = spectral._calculate_photolysis_rates(shortwave, cos_zenith, cold)
        s_warm = spectral._calculate_photolysis_rates(shortwave, cos_zenith, warm)
        # NO2 yields near threshold and the O3 Huggins cross sections and O(1D) yields
        # rise with temperature (Warneck and Williams 2012 Tables 6.11, 6.13b).
        assert s_warm[0] > s_cold[0]
        assert s_warm[1] > s_cold[1]
        torch.testing.assert_close(s_warm[0], j_warm[0])
    
    def test_photolysis_reference_conditions(self, chemistry, device):
        """Test photolysis under reference conditions."""
        # Reference conditions: 800 W/m2 radiation, overhead sun
        shortwave = torch.tensor(800.0, device=device)
        cos_zenith = torch.tensor(1.0, device=device)
        
        j_no2, _ = chemistry._calculate_photolysis_rates(shortwave, cos_zenith, torch.tensor(298.15, device=device))
        
        # Warneck and Williams (2012) Table 6.7: 8.8e-3 s-1 at the ground, overhead sun.
        j_ref = 8.8e-3
        tolerance = j_ref * 0.05  # 5% tolerance
        
        assert abs(j_no2.item() - j_ref) < tolerance


class TestChemicalReactionRates:
    """Test chemical reaction rate calculations and NOx-O3 chemistry."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def chemistry(self, device):
        return ChemistryFunctor(device=device, physical_boundary="open_mechanism")
    
    def test_nox_o3_reaction_rates(self, chemistry, device):
        """Test NO + O3 reaction rate calculation."""
        # Create test state
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(10e-9, device=device),  # 10 ppb NO
            'no2_mixing_ratio': torch.tensor(20e-9, device=device),  # 20 ppb NO2
            'o3_mixing_ratio': torch.tensor(50e-9, device=device),   # 50 ppb O3
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(600.0, device=device),
        }
        
        environment = {'shortwave_radiation': torch.tensor(600.0, device=device)}
        
        # Calculate reaction rate using compute_rate method
        rate_result = call_forward(chemistry, {**state, **environment})
        total_rate = _extract_activity_rate(rate_result)

        assert torch.isfinite(total_rate)
        # Net tendency can be signed (e.g., NO consumption in NO + O3 reaction),
        # so use magnitude to validate chemical activity.
        assert torch.abs(total_rate) > 0
    
    def test_closed_nox_reaction_family(self, chemistry, device):
        """Test the exact NO plus NO2 reaction-family identity."""
        # Create balanced test state
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(5e-9, device=device),
            'no2_mixing_ratio': torch.tensor(15e-9, device=device),
            'o3_mixing_ratio': torch.tensor(40e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(500.0, device=device)
        }
        
        # Calculate fluxes
        fluxes = chemistry.forward(state, time=43200.0, dt=60.0)  # Noon, 1-minute step
        
        # Extract NOx species changes
        delta_no = _output_value(fluxes, 'no_mixing_ratio')
        delta_no2 = _output_value(fluxes, 'no2_mixing_ratio')
        
        # Gas-phase interconversion preserves the NO plus NO2 family exactly.
        nox_change = delta_no + delta_no2

        torch.testing.assert_close(
            nox_change,
            torch.zeros_like(nox_change),
            rtol=0.0,
            atol=0.0,
        )
    
    def test_photostationary_relationship(self, chemistry, device):
        """Test photostationary state relationship for NOx-O3."""
        # Under steady state: J_NO2 * [NO2] approx k_NO_O3 * [NO] * [O3]
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(8e-9, device=device),
            'no2_mixing_ratio': torch.tensor(16e-9, device=device),
            'o3_mixing_ratio': torch.tensor(60e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'shortwave_down': torch.tensor(800.0, device=device),
            'cos_zenith': torch.tensor(0.8, device=device)
        }
        
        fluxes = chemistry.forward(state, time=43200.0, dt=60.0, return_diagnostics=True)
        
        # Extract rates
        j_no2 = _output_value(
            fluxes,
            'diagnostics.chemistry.photolysis_rate_no2',
        )
        
        # Calculate photostationary ratio
        # [O3] = J_NO2 * [NO2] / (k_NO_O3 * [NO])
        # Should be close to 1 for photostationary conditions
        
        # Get rate constant
        temp = state['air_temperature']
        rate_constants = chemistry._calculate_temperature_dependent_rates(temp)
        k_no_o3 = rate_constants['k_no_o3']
        
        # Calculate ratio (simplified, ignoring unit conversions for ratio test)
        numerator = j_no2 * state['no2_mixing_ratio']
        denominator = k_no_o3 * state['no_mixing_ratio'] * state['o3_mixing_ratio']
        
        # The ratio should be positive and of reasonable magnitude
        assert numerator > 0
        assert denominator > 0


class TestBVOCChemistry:
    """Test BVOC oxidation chemistry and OH calculations."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def chemistry(self, device):
        return ChemistryFunctor(device=device, physical_boundary="open_mechanism")
    
    def test_oh_concentration_calculation(self, chemistry, device):
        """Test dynamic OH concentration calculation using steady-state approximation."""
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no2_mixing_ratio': torch.tensor(15e-9, device=device),
            'o3_mixing_ratio': torch.tensor(45e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
        }
        
        # Calculate the O(1D) photolysis frequency (J_NO2 rho(chi) for overhead sun)
        _, j_o1d = chemistry._calculate_photolysis_rates(
            torch.tensor(700.0, device=device), state['cos_zenith'], state['air_temperature']
        )
        
        # Calculate OH concentration
        oh_conc = chemistry.compute_oh_concentration(state, j_o1d, state['air_temperature'])
        
        # OH concentration should be in realistic range (1e5-1e8 molecules cm-3)
        assert 1e5 <= oh_conc.item() <= 1e7
        assert torch.isfinite(oh_conc)
    
    def test_isoprene_oxidation_rates(self, chemistry, device):
        """Test isoprene oxidation by OH and O3."""
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(5e-9, device=device),
            'no2_mixing_ratio': torch.tensor(12e-9, device=device),
            'o3_mixing_ratio': torch.tensor(40e-9, device=device),
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'isoprene_mixing_ratio': torch.tensor(2e-9, device=device),  # 2 ppb isoprene
            'shortwave_down': torch.tensor(600.0, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device)
        }
        
        fluxes = chemistry.forward(state, time=43200.0, dt=60.0, return_diagnostics=True)
        
        # Should have isoprene oxidation output
        assert 'isoprene_mixing_ratio' in fluxes
        assert 'diagnostics.chemistry.isoprene_oxidation_rate' in fluxes
        
        delta_isoprene = _output_value(fluxes, 'isoprene_mixing_ratio')
        
        # Isoprene should be consumed (negative change)
        assert delta_isoprene < 0
        assert torch.isfinite(delta_isoprene)
    
    def test_bvoc_o3_consumption(self, chemistry, device):
        """Test O3 consumption by BVOC reactions."""
        # State with isoprene
        state_with_bvoc = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(5e-9, device=device),
            'no2_mixing_ratio': torch.tensor(12e-9, device=device),
            'o3_mixing_ratio': torch.tensor(50e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'isoprene_mixing_ratio': torch.tensor(3e-9, device=device),
            'shortwave_down': torch.tensor(600.0, device=device)
        }
        
        # State without isoprene
        state_no_bvoc = {k: v for k, v in state_with_bvoc.items() if k != 'isoprene_mixing_ratio'}
        
        fluxes_with = chemistry.forward(state_with_bvoc, time=43200.0, dt=60.0)
        fluxes_without = chemistry.forward(state_no_bvoc, time=43200.0, dt=60.0)
        
        delta_o3_with = _output_value(fluxes_with, 'o3_mixing_ratio')
        delta_o3_without = _output_value(fluxes_without, 'o3_mixing_ratio')
        
        # O3 should be consumed more when isoprene is present
        assert delta_o3_with < delta_o3_without

    def test_absent_isoprene_has_no_state_output_or_validation_warning(
        self,
        chemistry,
        device,
        caplog,
    ):
        """A missing optional BVOC state must not create a hidden zero update."""
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(5e-9, device=device),
            'no2_mixing_ratio': torch.tensor(12e-9, device=device),
            'o3_mixing_ratio': torch.tensor(40e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(600.0, device=device),
        }

        with caplog.at_level(logging.WARNING):
            outputs = chemistry(state, time=43200.0, dt=1.0)

        assert 'isoprene_mixing_ratio' not in outputs
        assert not any(
            "Output variable 'isoprene_mixing_ratio' not found" in record.message
            for record in caplog.records
        )


class TestScientificAccuracyValidation:
    """Test scientific accuracy against literature values and theory."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def chemistry(self, device):
        return ChemistryFunctor(device=device, physical_boundary="open_mechanism")
    
    def test_rate_constants_literature_values(self, chemistry, device):
        """Test rate constants against literature values at 298K."""
        temperature = torch.tensor(298.15, device=device)
        rate_constants = chemistry._calculate_temperature_dependent_rates(temperature)
        
        # k_NO_O3 at 298K should be ~1.8e-14 cm3 molecule-1 s-1
        k_no_o3 = rate_constants['k_no_o3'].item()
        expected_k_no_o3 = chemistry.get_parameter_value('k_no_o3_298')
        tolerance = expected_k_no_o3 * 0.1  # 10% tolerance
        assert abs(k_no_o3 - expected_k_no_o3) < tolerance
        
        # k_isoprene_OH at 298K should be ~1e-10 cm3 molecule-1 s-1
        k_isoprene_oh = rate_constants['k_isoprene_oh'].item()
        expected_k_isoprene_oh = chemistry.get_parameter_value('k_isoprene_oh_298')
        tolerance = expected_k_isoprene_oh * 0.2  # 20% tolerance
        assert abs(k_isoprene_oh - expected_k_isoprene_oh) < tolerance
    
    def test_photolysis_diurnal_cycle(self, chemistry, device):
        """Test realistic diurnal cycle of photolysis rates."""
        # Solar radiation values throughout day
        times_hours = [6, 9, 12, 15, 18]  # 6 AM to 6 PM
        radiation_values = [50, 400, 800, 600, 100]  # W m-2
        shortwave = torch.tensor(radiation_values, device=device)
        # Clear-sky cos(chi) of the same hours (solar noon at 12:00, equinox, 40 deg N)
        cos_zenith = torch.tensor([0.0, 0.54, 0.77, 0.54, 0.0], device=device)
        
        j_no2_rates, _ = chemistry._calculate_photolysis_rates(
            shortwave, cos_zenith, torch.full(shortwave.shape, 298.15, device=device)
        )
        
        # Should peak at noon
        max_idx = torch.argmax(j_no2_rates)
        assert max_idx == 2  # Noon (12:00)
        
        # Early morning and evening should be low
        assert j_no2_rates[0] < j_no2_rates[2] * 0.2  # 6 AM much less than noon
        assert j_no2_rates[4] < j_no2_rates[2] * 0.3  # 6 PM less than noon
    
    def test_temperature_q10_factors(self, chemistry, device):
        """Test Q10 temperature factors for reaction rates."""
        temps = torch.tensor([288.15, 298.15, 308.15], device=device)  # 15 C, 25 C, 35 C
        
        rate_constants = chemistry._calculate_temperature_dependent_rates(temps)
        k_no_o3 = rate_constants['k_no_o3']
        
        # Calculate Q10: ratio of rates for 10K temperature increase
        q10_1 = k_no_o3[1] / k_no_o3[0]  # 298K / 288K
        q10_2 = k_no_o3[2] / k_no_o3[1]  # 308K / 298K
        
        # For NO + O3 reaction (positive Ea), Q10 should be > 1
        assert q10_1 > 1.0
        assert q10_2 > 1.0
        
        # Q10 values should be consistent
        assert abs(q10_1 - q10_2) < 0.1  # Within 0.1 of each other
    
    def test_typical_atmospheric_conditions(self, chemistry, device):
        """Test chemistry under typical atmospheric conditions."""
        # Typical urban atmospheric conditions
        state = {
            'air_temperature': torch.tensor(295.15, device=device),  # 22 C
            'no_mixing_ratio': torch.tensor(20e-9, device=device),  # 20 ppb
            'no2_mixing_ratio': torch.tensor(30e-9, device=device), # 30 ppb  
            'o3_mixing_ratio': torch.tensor(60e-9, device=device),  # 60 ppb
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(500.0, device=device),
            'relative_humidity': torch.tensor(0.6, device=device)
        }
        
        # The reduced explicit solver requires a step inside the instantaneous
        # non-negativity domain; 30 s is below the 46.7 s bound for this state.
        fluxes = chemistry.forward(state, time=39600.0, dt=30.0)
        
        # All outputs should be finite and physically reasonable
        for flux_name, flux_value in fluxes.items():
            assert isinstance(flux_value, UpdateItem)
            assert torch.isfinite(flux_value.value), f"{flux_name} is not finite"
        
        # Photolysis rate should be reasonable for daytime
        j_no2 = _output_value(
            fluxes,
            'diagnostics.chemistry.photolysis_rate_no2',
        )
        assert 1e-4 <= j_no2.item() <= 2e-2  # s-1

class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases, boundary conditions, and robustness."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture  
    def chemistry(self, device):
        return ChemistryFunctor(device=device, physical_boundary="open_mechanism")
    
    def test_zero_radiation_conditions(self, chemistry, device):
        """Test chemistry under dark conditions."""
        state = {
            'air_temperature': torch.tensor(285.15, device=device),
            'no_mixing_ratio': torch.tensor(15e-9, device=device),
            'no2_mixing_ratio': torch.tensor(25e-9, device=device),
            'o3_mixing_ratio': torch.tensor(45e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(0.0, device=device)  # No radiation
        }
        
        # Dark NO titration gives a 58.6 s positivity limit for this state.
        fluxes = chemistry.forward(state, time=0.0, dt=30.0)
        
        # Photolysis should be near zero
        j_no2 = _output_value(
            fluxes,
            'diagnostics.chemistry.photolysis_rate_no2',
        )
        assert j_no2.item() < 1e-6
        
        # Chemistry should still work (thermal reactions)
        assert torch.isfinite(_output_value(fluxes, 'no_mixing_ratio'))
        assert torch.isfinite(_output_value(fluxes, 'no2_mixing_ratio'))
        assert torch.isfinite(_output_value(fluxes, 'o3_mixing_ratio'))
    
    def test_very_low_concentrations(self, chemistry, device):
        """Test chemistry with very low species concentrations."""
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(1e-12, device=device),   # 0.001 ppb
            'no2_mixing_ratio': torch.tensor(1e-12, device=device),  # 0.001 ppb
            'o3_mixing_ratio': torch.tensor(1e-11, device=device),   # 0.01 ppb
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(400.0, device=device)
        }
        
        # Should not crash with minimum concentration handling
        fluxes = chemistry.forward(state, time=43200.0, dt=60.0)
        
        # All outputs should be finite
        for flux_value in fluxes.values():
            assert isinstance(flux_value, UpdateItem)
            assert torch.isfinite(flux_value.value)
    
    def test_extreme_temperature_conditions(self, chemistry, device):
        """Test chemistry at temperature extremes."""
        extreme_temps = [250.0, 350.0]  # -23 C to 77 C
        
        for temp in extreme_temps:
            state = {
                'air_temperature': torch.tensor(temp, device=device),
                'no_mixing_ratio': torch.tensor(10e-9, device=device),
                'no2_mixing_ratio': torch.tensor(20e-9, device=device),
                'o3_mixing_ratio': torch.tensor(40e-9, device=device),
                'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
                'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
                'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
                'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
                'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
                'shortwave_down': torch.tensor(500.0, device=device)
            }
            
            # At 350 K the explicit Euler positivity bound of NO is about 53 s.
            fluxes = chemistry.forward(state, time=43200.0, dt=30.0)
            
            # Should produce finite results despite extreme conditions
            assert torch.isfinite(_output_value(fluxes, 'no_mixing_ratio'))
            assert torch.isfinite(_output_value(fluxes, 'no2_mixing_ratio'))
            assert torch.isfinite(_output_value(fluxes, 'o3_mixing_ratio'))
    
    def test_missing_required_shortwave_input(self, chemistry, device):
        """Reject chemistry input without the canonical radiation proxy."""
        # Minimal required inputs only
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(10e-9, device=device),
            'no2_mixing_ratio': torch.tensor(20e-9, device=device),
            'o3_mixing_ratio': torch.tensor(40e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            # No radiation data provided
        }
        
        with pytest.raises(ValueError, match="shortwave_down is required"):
            chemistry.forward(state, time=43200.0, dt=60.0)


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness to invalid inputs."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def chemistry(self, device):
        return ChemistryFunctor(device=device, physical_boundary="open_mechanism")
    
    def test_missing_required_inputs(self, chemistry, device):
        """Test error handling for missing required inputs."""
        # Missing air_temperature
        state_missing_temp = {
            'no_mixing_ratio': torch.tensor(10e-9, device=device),
            'no2_mixing_ratio': torch.tensor(20e-9, device=device),
            'o3_mixing_ratio': torch.tensor(40e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
        }
        
        with pytest.raises(ValueError, match="air_temperature is required"):
            chemistry.forward(state_missing_temp, time=0.0, dt=60.0)
        
        # Missing NO mixing ratio
        state_missing_no = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no2_mixing_ratio': torch.tensor(20e-9, device=device),
            'o3_mixing_ratio': torch.tensor(40e-9, device=device)
        }
        
        with pytest.raises(ValueError, match="no_mixing_ratio is required"):
            chemistry.forward(state_missing_no, time=0.0, dt=60.0)
    
    def test_negative_concentration_handling(self, chemistry, device):
        """Reject negative concentrations without hidden state clipping."""
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(-5e-9, device=device),    # Negative
            'no2_mixing_ratio': torch.tensor(20e-9, device=device),
            'o3_mixing_ratio': torch.tensor(-10e-9, device=device),   # Negative
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(500.0, device=device)
        }
        
        with pytest.raises(ValueError, match="no_mixing_ratio"):
            chemistry.forward(state, time=43200.0, dt=60.0)
    
    def test_nan_input_handling(self, chemistry, device):
        """Test handling of NaN inputs."""
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(float('nan'), device=device),
            'no2_mixing_ratio': torch.tensor(20e-9, device=device),
            'o3_mixing_ratio': torch.tensor(40e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(500.0, device=device)
        }
        
        with pytest.raises(ValueError, match="no_mixing_ratio"):
            chemistry.forward(state, time=43200.0, dt=60.0)
    
    def test_inf_input_handling(self, chemistry, device):
        """Test handling of infinite inputs."""
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(float('inf'), device=device),
            'no2_mixing_ratio': torch.tensor(20e-9, device=device),
            'o3_mixing_ratio': torch.tensor(40e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(500.0, device=device)
        }
        
        with pytest.raises(ValueError, match="no_mixing_ratio"):
            chemistry.forward(state, time=43200.0, dt=60.0)


class TestParameterSensitivity:
    """Test parameter sensitivity and configuration robustness."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_photolysis_parameter_sensitivity(self, device):
        """Test sensitivity to photolysis parameters."""
        base_config = {'parameters': {'uv_a_surface_albedo': {'value': 0.0}}}
        high_config = {'parameters': {'uv_a_surface_albedo': {'value': 0.2}}}
        
        chemistry_base = ChemistryFunctor(config=base_config, device=device, physical_boundary="open_mechanism")
        chemistry_high = ChemistryFunctor(config=high_config, device=device, physical_boundary="open_mechanism")
        
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(10e-9, device=device),
            'no2_mixing_ratio': torch.tensor(20e-9, device=device),
            'o3_mixing_ratio': torch.tensor(40e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(800.0, device=device)
        }
        
        fluxes_base = chemistry_base.forward(state, time=43200.0, dt=60.0)
        fluxes_high = chemistry_high.forward(state, time=43200.0, dt=60.0)
        
        # Higher j_NO2 should lead to higher photolysis rates
        assert (
            _output_value(
                fluxes_high,
                'diagnostics.chemistry.photolysis_rate_no2',
            )
            > _output_value(
                fluxes_base,
                'diagnostics.chemistry.photolysis_rate_no2',
            )
        )
        
        # Should affect NOx chemistry
        delta_no_diff = abs(
            _output_value(fluxes_high, 'no_mixing_ratio')
            - _output_value(fluxes_base, 'no_mixing_ratio')
        )
        assert delta_no_diff > 1e-12  # Should be sensitive to parameter change
    
    def test_reaction_rate_sensitivity(self, device):
        """Test sensitivity to reaction rate parameters."""
        base_config = {'parameters': {'k_no_o3_298': {'value': 1.8e-14}}}
        high_config = {'parameters': {'k_no_o3_298': {'value': 2.0e-14}}}
        
        chemistry_base = ChemistryFunctor(config=base_config, device=device, physical_boundary="open_mechanism")
        chemistry_high = ChemistryFunctor(config=high_config, device=device, physical_boundary="open_mechanism")
        
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(15e-9, device=device),
            'no2_mixing_ratio': torch.tensor(20e-9, device=device),
            'o3_mixing_ratio': torch.tensor(50e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(600.0, device=device)
        }
        
        rate_base_result = call_forward(chemistry_base, {**state, **{'shortwave_radiation': torch.tensor(600.0, device=device)}})
        rate_high_result = call_forward(chemistry_high, {**state, **{'shortwave_radiation': torch.tensor(600.0, device=device)}})
        rate_base = _extract_activity_rate(rate_base_result)
        rate_high = _extract_activity_rate(rate_high_result)
        
        # Check that rates are different (sensitivity test)
        # Use relative difference to handle very small numbers
        relative_diff = torch.abs(rate_high - rate_base) / (torch.abs(rate_base) + 1e-20)
        
        assert relative_diff > 1e-6, (
            f"Expected parameter sensitivity, got base={rate_base}, "
            f"high={rate_high}, rel_diff={relative_diff}"
        )


# Parametrized tests for comprehensive coverage
class TestParametrizedConditions:
    """Parametrized tests for comprehensive condition coverage."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def chemistry(self, device):
        return ChemistryFunctor(device=device, physical_boundary="open_mechanism")
    
    @pytest.mark.parametrize("temperature", [283.15, 298.15, 313.15])
    def test_temperature_sweep(self, chemistry, device, temperature):
        """Test chemistry across temperature range."""
        state = {
            'air_temperature': torch.tensor(temperature, device=device),
            'no_mixing_ratio': torch.tensor(10e-9, device=device),
            'no2_mixing_ratio': torch.tensor(20e-9, device=device),
            'o3_mixing_ratio': torch.tensor(40e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(600.0, device=device)
        }
        
        fluxes = chemistry.forward(state, time=43200.0, dt=60.0)
        
        # All outputs should be finite
        assert torch.isfinite(_output_value(fluxes, 'no_mixing_ratio'))
        assert torch.isfinite(_output_value(fluxes, 'no2_mixing_ratio'))
        assert torch.isfinite(_output_value(fluxes, 'o3_mixing_ratio'))
        assert torch.isfinite(_output_value(
            fluxes,
            'diagnostics.chemistry.photolysis_rate_no2',
        ))
    
    @pytest.mark.parametrize("radiation", [0, 200, 600, 1000])
    def test_radiation_sweep(self, chemistry, device, radiation):
        """Test chemistry across radiation range."""
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(10e-9, device=device),
            'no2_mixing_ratio': torch.tensor(20e-9, device=device), 
            'o3_mixing_ratio': torch.tensor(40e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(float(radiation), device=device)
        }
        
        fluxes = chemistry.forward(state, time=43200.0, dt=30.0)
        
        # Photolysis should scale with radiation
        j_no2 = _output_value(
            fluxes,
            'diagnostics.chemistry.photolysis_rate_no2',
        )
        if radiation == 0:
            assert j_no2.item() < 1e-6
        else:
            assert j_no2 > 0
        
        # All outputs should be finite
        assert torch.isfinite(_output_value(fluxes, 'no_mixing_ratio'))
        assert torch.isfinite(_output_value(fluxes, 'no2_mixing_ratio'))
        assert torch.isfinite(_output_value(fluxes, 'o3_mixing_ratio'))
    
    @pytest.mark.parametrize("nox_level", [5e-9, 20e-9, 50e-9])  # 5, 20, 50 ppb
    def test_nox_concentration_sweep(self, chemistry, device, nox_level):
        """Test chemistry across NOx concentration range."""
        state = {
            'air_temperature': torch.tensor(298.15, device=device),
            'no_mixing_ratio': torch.tensor(nox_level * 0.4, device=device),   # 40% NO
            'no2_mixing_ratio': torch.tensor(nox_level * 0.6, device=device),  # 60% NO2  
            'o3_mixing_ratio': torch.tensor(60e-9, device=device),
            'atmospheric_pressure': torch.tensor(101325.0, device=device),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9, device=device),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6, device=device),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01, device=device),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0, device=device),  # overhead sun
            'shortwave_down': torch.tensor(600.0, device=device)
        }
        
        fluxes = chemistry.forward(state, time=43200.0, dt=30.0)
        
        # Reaction rates should scale with concentrations
        rate_result = call_forward(chemistry, {**state, **{'shortwave_radiation': torch.tensor(600.0, device=device)}})
        total_rate = _extract_activity_rate(rate_result)
        assert torch.abs(total_rate) > 0
        
        # All outputs should be finite and reasonable
        assert torch.isfinite(_output_value(fluxes, 'no_mixing_ratio'))
        assert torch.isfinite(_output_value(fluxes, 'no2_mixing_ratio'))
        assert torch.isfinite(_output_value(fluxes, 'o3_mixing_ratio'))


class TestIsopreneChemicalLifetime:
    """Isoprene chemical lifetime tau = 1 / (k_OH[OH] + k_O3[O3] + k_NO3[NO3]) [s].

    Hand computation at T = 298.15 K, where k(T_ref) = k_298: k_OH = 1.0e-10 (IUPAC 2006
    II.A2.10 k298), k_O3 = 1.27e-17 (IUPAC 2006 II.A1.157 k298) and
    k_NO3 = 3.15e-12 exp(-450/298.15) (MCM v3.3.1), all in cm3 molecule-1 s-1.
    """

    K_OH_298 = 1.0e-10
    K_O3_298 = 1.27e-17
    NO3_PRE_EXPONENTIAL = 3.15e-12
    NO3_E_OVER_R = 450.0

    @pytest.fixture
    def chemistry(self):
        return ChemistryFunctor(device=torch.device("cpu"), physical_boundary="open_mechanism")

    def _hand_tau(self, oh, o3, no3):
        import math

        k_no3 = self.NO3_PRE_EXPONENTIAL * math.exp(-self.NO3_E_OVER_R / 298.15)
        return 1.0 / (self.K_OH_298 * oh + self.K_O3_298 * o3 + k_no3 * no3)

    def test_lifetime_matches_hand_computation_in_number_density(self, chemistry):
        temperature = torch.tensor(298.15, dtype=torch.float64)
        oh, o3, no3 = 2.0e6, 1.0e12, 1.0e8
        tau = chemistry.compute_isoprene_chemical_lifetime(
            temperature,
            oh=torch.tensor(oh, dtype=torch.float64),
            o3=torch.tensor(o3, dtype=torch.float64),
            no3=torch.tensor(no3, dtype=torch.float64),
            concentration_basis="number_density",
        )
        expected = self._hand_tau(oh, o3, no3)
        assert abs(float(tau) / expected - 1.0) < 1.0e-6
        # About 59 min: 2.0e-4 (OH) + 1.27e-5 (O3) + 6.96e-5 (NO3) s-1.
        assert 3540.0 < float(tau) < 3545.0

    def test_mixing_ratio_basis_uses_the_ideal_gas_number_density(self, chemistry):
        from relic.utils.physical_constants import get_physical_constant

        temperature = torch.tensor([298.15], dtype=torch.float64)
        pressure = torch.tensor([101325.0], dtype=torch.float64)
        boltzmann = float(get_physical_constant("relic.physical.constants.universal.boltzmann_constant"))
        cm3_per_m3 = float(
            get_physical_constant("relic.physical.constants.conversions.cubic_centimeters_per_cubic_meter")
        )
        air = 101325.0 / (boltzmann * 298.15) / cm3_per_m3
        oh, o3, no3 = 2.0e6, 1.0e12, 1.0e8
        tau = chemistry.compute_isoprene_chemical_lifetime(
            temperature,
            oh=torch.tensor([oh / air], dtype=torch.float64),
            o3=torch.tensor([o3 / air], dtype=torch.float64),
            no3=torch.tensor([no3 / air], dtype=torch.float64),
            concentration_basis="mixing_ratio",
            atmospheric_pressure=pressure,
        )
        assert tau.shape == temperature.shape
        assert abs(float(tau[0]) / self._hand_tau(oh, o3, no3) - 1.0) < 1.0e-6
        tau_density = chemistry.compute_isoprene_chemical_lifetime(
            temperature,
            oh=torch.tensor([oh], dtype=torch.float64),
            o3=torch.tensor([o3], dtype=torch.float64),
            no3=torch.tensor([no3], dtype=torch.float64),
            concentration_basis="number_density",
        )
        assert abs(float(tau[0] / tau_density[0]) - 1.0) < 1.0e-12

    def test_lifetime_temperature_dependence_follows_each_activation_energy(self, chemistry):
        temperature = torch.tensor([273.15, 298.15, 313.15], dtype=torch.float64)
        zero = torch.zeros(3, dtype=torch.float64)
        oxidant = torch.full((3,), 1.0e8, dtype=torch.float64)
        tau_no3 = chemistry.compute_isoprene_chemical_lifetime(
            temperature, oh=zero, o3=zero, no3=oxidant, concentration_basis="number_density"
        )
        tau_oh = chemistry.compute_isoprene_chemical_lifetime(
            temperature, oh=oxidant * 1.0e-2, o3=zero, no3=zero, concentration_basis="number_density"
        )
        # E/R = +450 K (NO3): faster loss and shorter lifetime when warmer.
        assert bool(torch.all(tau_no3[1:] < tau_no3[:-1]))
        # E/R = -390 K (OH): slower loss and longer lifetime when warmer.
        assert bool(torch.all(tau_oh[1:] > tau_oh[:-1]))

    def test_lifetime_is_differentiable_in_the_oxidant_densities(self, chemistry):
        oh = torch.tensor(2.0e6, dtype=torch.float64, requires_grad=True)
        tau = chemistry.compute_isoprene_chemical_lifetime(
            torch.tensor(298.15, dtype=torch.float64),
            oh=oh,
            o3=torch.tensor(1.0e12, dtype=torch.float64),
            no3=torch.tensor(1.0e8, dtype=torch.float64),
            concentration_basis="number_density",
        )
        tau.backward()
        # d tau / d[OH] = -k_OH tau^2.
        expected = -self.K_OH_298 * float(tau.detach()) ** 2
        assert abs(float(oh.grad) / expected - 1.0) < 1.0e-6

    def test_iupac_2015_sensitivity_is_a_config_override(self):
        import math

        k_iupac_2015 = 2.95e-12 * math.exp(-450.0 / 298.15)
        sensitivity = ChemistryFunctor(
            device=torch.device("cpu"),
            physical_boundary="open_mechanism",
            config={"parameters": {"k_isoprene_no3_298": {"value": k_iupac_2015, "units": "cm3 molecule-1 s-1"}}},
        )
        tau = sensitivity.compute_isoprene_chemical_lifetime(
            torch.tensor(298.15, dtype=torch.float64),
            oh=torch.tensor(0.0, dtype=torch.float64),
            o3=torch.tensor(0.0, dtype=torch.float64),
            no3=torch.tensor(1.0e8, dtype=torch.float64),
            concentration_basis="number_density",
        )
        assert abs(float(tau) * k_iupac_2015 * 1.0e8 - 1.0) < 1.0e-6

    def test_invalid_inputs_fail_fast_without_a_floor(self, chemistry):
        temperature = torch.tensor(298.15, dtype=torch.float64)
        zero = torch.tensor(0.0, dtype=torch.float64)
        one = torch.tensor(1.0e6, dtype=torch.float64)
        with pytest.raises(ValueError, match="positive"):
            chemistry.compute_isoprene_chemical_lifetime(
                temperature, oh=zero, o3=zero, no3=zero, concentration_basis="number_density"
            )
        with pytest.raises(ValueError):
            chemistry.compute_isoprene_chemical_lifetime(
                temperature, oh=-one, o3=zero, no3=one, concentration_basis="number_density"
            )
        with pytest.raises(ValueError):
            chemistry.compute_isoprene_chemical_lifetime(
                temperature, oh=one, o3=zero, no3=zero, concentration_basis="mixing_ratio"
            )
        with pytest.raises(ValueError):
            chemistry.compute_isoprene_chemical_lifetime(
                temperature,
                oh=one,
                o3=zero,
                no3=zero,
                concentration_basis="number_density",
                atmospheric_pressure=torch.tensor(101325.0, dtype=torch.float64),
            )
        with pytest.raises(ValueError):
            chemistry.compute_isoprene_chemical_lifetime(
                temperature, oh=one, o3=zero, no3=zero, concentration_basis="ppb"
            )
        with pytest.raises(ValueError):
            chemistry.compute_isoprene_chemical_lifetime(
                torch.tensor(150.0, dtype=torch.float64),
                oh=one,
                o3=zero,
                no3=zero,
                concentration_basis="number_density",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

