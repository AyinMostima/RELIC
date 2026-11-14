"""L1 tests for the canonical sesquiterpene synthesis closure.

Coverage includes the BiologicalProcess bridge contract, ParameterManager-backed
response parameters, explicit UpdateItem semantics, molar port conversion,
namespaced diagnostics, open-port routing, SCM provenance, and device behavior.

Scientific checks use the reduced MEGAN-style activity expression documented in
the module theory. The tests do not claim full MEGAN2.1 compound speciation.
"""

import inspect

import torch
import pytest
import warnings

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from relic.parameters import get_global_manager
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter
from relic.scientific.biosphere.bvoc.synthesis.sesquiterpene import (
    SesquiterpeneSynthesis,
    SesquiterpeneStressModel
)
from relic.utils.physical_constants import get_physical_constant

_PARAM_MANAGER = get_global_manager()


def _get_parameter_scalar(key: str, namespace: str, units: str) -> float:
    """Fetch a scalar parameter from the global manager for test expectations."""
    value = _PARAM_MANAGER.get(
        key,
        namespace=namespace,
        units=units,
    )
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


STRESS_THRESHOLD = _get_parameter_scalar(
    "stress_threshold",
    "biosphere.bvoc.sesquiterpene",
    "dimensionless",
)
STRESS_AMPLIFICATION = _get_parameter_scalar(
    "stress_amplification",
    "biosphere.bvoc.sesquiterpene",
    "dimensionless",
)


def _synthesis_state(temperature=303.15, leaf_area_index=1.0):
    """Build the two mandatory drivers for the reduced SQT closure."""
    return {
        "temperature": temperature,
        "leaf_area_index": leaf_area_index,
    }


class TestSesquiterpeneSynthesisL1:
    """L1 unit tests for the canonical SesquiterpeneSynthesis closure."""
    
    def test_basic_instantiation(self):
        """Test L1.1: Basic instantiation with default parameters."""
        model = SesquiterpeneSynthesis()

        assert isinstance(model, SesquiterpeneSynthesis)
        assert model.name == "sesquiterpene_synthesis"

        # Check pre-registration device placeholder from the device policy.
        assert model.device.type == "cpu"

        # Check BiologicalProcess inheritance
        from relic.core.abstractions import BiologicalProcess
        assert isinstance(model, BiologicalProcess)

    def test_device_default_uses_explicit_none_check(self):
        """Test that device defaulting follows the device policy style."""
        source_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..', '..', '..', '..',
            'relic', 'scientific', 'biosphere', 'bvoc', 'synthesis', 'sesquiterpene.py',
        ))
        with open(source_path, 'r', encoding='utf-8') as handle:
            content = handle.read()

        assert "device or torch.device" not in content
    
    def test_custom_configuration_instantiation(self):
        """Test L1.2: Instantiation with custom configuration."""
        config = {
            'base_emission': 0.6,
            'temperature_sensitivity': 0.2
        }
        
        model = SesquiterpeneSynthesis(
            name="custom_sesquiterpene",
            config=config,
            device=torch.device('cpu')
        )
        
        assert model.name == "custom_sesquiterpene"
        assert model.config == config
        assert model.device == torch.device('cpu')
    
    def test_parameter_initialization(self):
        """Test L1.3: Parameter initialization has one response owner."""
        model = SesquiterpeneSynthesis()

        # Check retained module parameters match their canonical database values.
        assert model.base_emission_factor.item() == pytest.approx(0.45)
        # T_s of the shared Eq. (11) law is resolved by the response helper; the
        # retired duplicate reference temperature is not cached as a buffer.
        assert not hasattr(model, "temp_ref")

        # The shared response integrator owns the MEGAN temperature coefficient.
        response_beta = _get_parameter_scalar(
            "beta_sesquiterpenes",
            "biosphere.bvoc.megan",
            "K-1",
        )
        assert response_beta == pytest.approx(0.17)
        assert not hasattr(model, "temperature_sensitivity")

        # Check retained buffers are on the configured device.
        assert model.base_emission_factor.device == model.device
        assert model.seconds_per_hour.item() == pytest.approx(
            get_physical_constant("seconds_per_hour")
        )
        expected_carbon_fraction = _get_parameter_scalar(
            "carbon_fraction.sesquiterpenes",
            "biosphere.bvoc",
            "dimensionless",
        )
        assert model.carbon_fraction_sesquiterpenes.item() == pytest.approx(
            expected_carbon_fraction
        )
        assert not hasattr(model, "default_leaf_temperature")

    def test_illegal_emission_factor_is_rejected_instead_of_floored(self):
        """With validated drivers every activity factor is non-negative, so a
        non-negative floor could only mask an illegal emission factor; a negative or
        non-finite factor raises and the floor parameter is retired."""
        model = SesquiterpeneSynthesis()
        assert "emission_factor_min" not in model.metadata.parameters
        assert _PARAM_MANAGER.get(
            "biosphere.bvoc.sesquiterpene.emission_factor_min", required=False
        ) is None
        for illegal in (-0.1, float("nan"), float("inf")):
            model.base_emission_factor.fill_(illegal)
            with pytest.raises(ValueError, match="emission factor"):
                model._compute_rate_core({"temperature": torch.tensor(303.15)})

    def test_temperature_function_is_the_executed_megan21_law(self):
        """temperature_function returns the executed Guenther et al. (2012) Eq. (7)
        MEGAN2.1 temperature activity; no BiologicalProcess temperature_response or
        ph_dependence label is passed (12.16), and none of the generic bridge response
        parameters are declared."""
        from pathlib import Path

        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )
        from relic.scientific.biosphere.bvoc.synthesis import sesquiterpene

        source = inspect.getsource(sesquiterpene)
        assert "temperature_response=" not in source
        assert "ph_dependence=" not in source
        model = SesquiterpeneSynthesis()
        temperatures = torch.tensor([283.15, 303.15, 318.15], dtype=torch.float32)
        executed = model._compute_rate_core({"temperature": temperatures})
        assert torch.allclose(
            model.temperature_function(temperatures),
            executed["gamma_temperature"],
            rtol=1e-6,
            atol=0.0,
        )
        generic = {
            "q10",
            "activation_energy",
            "temperature_sigma",
            "ph_optimum",
            "ph_width",
            "moisture_half_saturation",
            "moisture_linear_min",
            "moisture_linear_max",
            "moisture_optimum",
            "moisture_width",
        }
        assert not (generic & set(model.metadata.parameters))
        certificate = load_iphs_equivalence_certificate(
            Path(__file__).resolve().parents[4]
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_synthesis_sesquiterpene_sesquiterpene_synthesis.yaml"
        )
        assert "temperature_response" not in certificate.interface_map
        declared = str(certificate.interface_map["temperature_function"])
        assert "Eq. (7)" in declared and "Table 4" in declared
    
    def test_dependencies_and_provides(self):
        """Test L1.4: Model dependencies and provided variables."""
        model = SesquiterpeneSynthesis()

        expected_outputs = {
            "sesquiterpenes_emission",
            "bvoc_carbon_allocation_flux",
            "diagnostics.bvoc.sesquiterpene.emission_rate_molar",
            "diagnostics.bvoc.sesquiterpene.gamma_temperature",
            "diagnostics.bvoc.sesquiterpene.gamma_light",
            "diagnostics.bvoc.sesquiterpene.gamma_stress",
            "diagnostics.bvoc.sesquiterpene.gamma_total",
        }
        from relic.core.dependency_graph import normalize_dependency_declaration

        deps = normalize_dependency_declaration(model.get_dependencies(), owner=model.name)
        assert set(deps.optional) == {"water_stress", "thermal_stress", "nutrient_stress"}
        assert not deps.required
        assert set(model.get_provides()) == expected_outputs
        assert set(model.metadata.output_variables) == expected_outputs
        assert model.metadata.input_variables == [
            "temperature",
            "leaf_area_index",
            "ppfd",
            "soil_moisture",
            "water_stress_factor",
            "temperature_stress_factor",
            "nutrient_stress_factor",
        ]
        assert set(model.optional_input_variables) == {
            "ppfd",
            "soil_moisture",
            "water_stress_factor",
            "temperature_stress_factor",
            "nutrient_stress_factor",
        }
        assert "sesquiterpene_emission_mechanistic" not in expected_outputs
        assert "bvoc_sesquiterpene_carbon_loss" not in expected_outputs
        assert "emission_rate_molar" not in expected_outputs
    
    def test_device_specification(self):
        """Test L1.5: Device specification and tensor placement."""
        # Test CPU device
        cpu_model = SesquiterpeneSynthesis(device=torch.device('cpu'))
        assert cpu_model.device == torch.device('cpu')
        assert cpu_model.base_emission_factor.device == torch.device('cpu')

        # Test CUDA device if available
        if torch.cuda.is_available():
            cuda_model = SesquiterpeneSynthesis(device=torch.device('cuda'))
            assert cuda_model.device.type == torch.device('cuda').type
            assert cuda_model.base_emission_factor.device.type == torch.device('cuda').type

    def test_references_are_registered_bibliographic_records(self):
        """Metadata, certificate and module references name registered works: Duhl et
        al. (2008) carries its registered title, the unregistered Jardine et al. (2011)
        entry is retired, and the module header cites the registered Dudareva et al.
        (2006) and Copolovici et al. (2012) records."""
        from pathlib import Path

        import relic.scientific.biosphere.bvoc.synthesis.sesquiterpene as module
        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )

        path = (
            Path(__file__).resolve().parents[4]
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_synthesis_sesquiterpene_sesquiterpene_synthesis.yaml"
        )
        certificate = load_iphs_equivalence_certificate(path)
        declared = list(SesquiterpeneSynthesis().metadata.references)
        duhl = "Duhl et al. (2008). Sesquiterpene emissions from vegetation: a review."
        for references in (declared, list(certificate.references)):
            assert duhl in references
            assert not any("Jardine" in reference for reference in references)
        header = module.__doc__
        for retired in (
            "Critical Reviews in Plant Sciences 22",
            "Plant Physiology 160",
            "Quercus ilex",
        ):
            assert retired not in header, retired
        assert "Critical Reviews in Plant Sciences 25, 417-440" in header
        assert "Journal of Plant Physiology 169, 664-672" in header

    def test_production_light_activity_is_the_megan21_ldf_blend(self):
        """gamma_L of the production rate is Guenther et al. (2012) Eq. (3) with
        the registered sesquiterpene LDF and the shared Eq. (4) law; the retired
        weak-light tanh curve confined gamma_L to [0.8, 1.2]."""
        import math

        from relic.parameters import get_global_manager

        manager = get_global_manager()
        alpha = float(manager.get("biosphere.bvoc.megan.alpha_light", units="m2 s umol-1", required=True))
        cl = float(manager.get("biosphere.bvoc.megan.cl_light", units="dimensionless", required=True))
        ldf = float(
            manager.get("biosphere.bvoc.megan.ldf_sesquiterpenes", units="dimensionless", required=True)
        )
        model = SesquiterpeneSynthesis()
        for ppfd in (0.0, 200.0, 1000.0, 2000.0):
            result = model._compute_rate_core(
                {"temperature": torch.tensor(303.15), "ppfd": torch.tensor(ppfd)}
            )
            expected = (1.0 - ldf) + ldf * alpha * cl * ppfd / math.sqrt(1.0 + (alpha * ppfd) ** 2)
            assert result["gamma_light"].item() == pytest.approx(expected, rel=1e-5)

    def test_production_stress_activity_has_no_temperature_default_or_floor(self):
        """With a soil-moisture driver only, gamma_stress is the relative soil
        moisture: the parabolic temperature-stress default and the 0.1 floor are
        retired, and an out-of-range driver is rejected."""
        model = SesquiterpeneSynthesis()
        for temperature in (298.15, 313.15):
            result = model._compute_rate_core(
                {"temperature": torch.tensor(temperature), "soil_moisture": torch.tensor(1.0)}
            )
            assert result["gamma_stress"].item() == pytest.approx(1.0)
        dry = model._compute_rate_core(
            {"temperature": torch.tensor(303.15), "soil_moisture": torch.tensor(0.05)}
        )
        assert dry["gamma_stress"].item() == pytest.approx(0.05)
        with pytest.raises(ValueError, match="soil_moisture"):
            model._compute_rate_core(
                {"temperature": torch.tensor(303.15), "soil_moisture": torch.tensor(1.5)}
            )

    def test_production_temperature_activity_has_no_clamp(self):
        """gamma_T is the Guenther et al. (2012) Eq. (7) blend of Eq. (11) and Eq. (8)
        with the Table 4 'Other Sesquiterpenes' row; it holds where the retired
        [0.1, 10] clamp used to bind (below about 16.5 degC and above about 43.6 degC)."""
        import math

        from relic.parameters import get_global_manager
        from relic.utils.physical_constants import get_physical_constant

        manager = get_global_manager()

        def value(key, units):
            return float(manager.get(key, units=units, required=True))

        beta = value("biosphere.bvoc.megan.beta_sesquiterpenes", "K-1")
        ldf = value("biosphere.bvoc.megan.ldf_sesquiterpenes", "dimensionless")
        t_standard = value("biosphere.bvoc.t_standard", "K")
        eopt = value("biosphere.bvoc.megan.eopt_sesquiterpenes", "dimensionless")
        ct1 = value("biosphere.bvoc.megan.ct1_sesquiterpenes", "J mol-1")
        topt = value("biosphere.bvoc.megan.topt", "K")
        ct2 = value("biosphere.bvoc.megan.ct2", "J mol-1")
        r_gas = float(get_physical_constant("gas_constant"))
        model = SesquiterpeneSynthesis()
        for temperature in (273.15, 283.15, 318.15, 323.15):
            x = (1.0 / topt - 1.0 / temperature) / r_gas
            light_dependent = (
                eopt * ct2 * math.exp(ct1 * x) / (ct2 - ct1 * (1.0 - math.exp(ct2 * x)))
            )
            expected = (1.0 - ldf) * math.exp(beta * (temperature - t_standard)) + (
                ldf * light_dependent
            )
            result = model._compute_rate_core({"temperature": torch.tensor(temperature)})
            assert result["gamma_temperature"].item() == pytest.approx(expected, rel=1e-4)

    def test_production_temperature_activity_is_the_shared_eq7_law(self):
        """gamma_T is Guenther et al. (2012) Eq. (7) evaluated by the shared
        megan_model law with the ParameterManager T_s of its Eq. (11) term. The retired
        normalization by a second reference temperature (with its eps floor) made T_s
        cancel, so a calibrated T_s must now enter the light-independent term."""
        from relic.parameters import get_global_manager
        from relic.scientific.biosphere.bvoc.megan_model import (
            megan_temperature_activity,
        )
        from relic.utils.physical_constants import get_physical_constant

        base = get_global_manager()
        calibrated_t_standard = 300.15

        class _CalibratedStandardTemperature:
            """Pass-through manager that calibrates only biosphere.bvoc.t_standard."""

            def __init__(self, manager):
                self._manager = manager

            def get(self, key, *args, **kwargs):
                if key == "biosphere.bvoc.t_standard":
                    return calibrated_t_standard
                return self._manager.get(key, *args, **kwargs)

        def value(key, units):
            return torch.tensor(
                float(base.get(key, units=units, required=True)), dtype=torch.float32
            )

        model = SesquiterpeneSynthesis()
        model.response_integrator.parameter_manager = _CalibratedStandardTemperature(base)
        for temperature in (calibrated_t_standard, 283.15, 318.15):
            result = model._compute_rate_core({"temperature": torch.tensor(temperature)})
            expected = megan_temperature_activity(
                torch.tensor(temperature, dtype=torch.float32),
                value("biosphere.bvoc.megan.ldf_sesquiterpenes", "dimensionless"),
                value("biosphere.bvoc.megan.beta_sesquiterpenes", "K-1"),
                torch.tensor(calibrated_t_standard, dtype=torch.float32),
                value("biosphere.bvoc.megan.eopt_sesquiterpenes", "dimensionless"),
                value("biosphere.bvoc.megan.topt", "K"),
                value("biosphere.bvoc.megan.ct1_sesquiterpenes", "J mol-1"),
                value("biosphere.bvoc.megan.ct2", "J mol-1"),
                torch.tensor(float(get_physical_constant("gas_constant")), dtype=torch.float32),
            )
            assert result["gamma_temperature"].item() == pytest.approx(expected.item(), rel=1e-6)

    def test_metadata_and_certificate_declare_the_executed_response_parameters(self):
        """The executed Eq. (3), (4) and (11) parameters are declared; the retired
        interval rescaling and the duplicate reference temperatures of the helper
        and of the retired gamma_T normalization are not."""
        from pathlib import Path

        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )

        declared = set(SesquiterpeneSynthesis().metadata.parameters)
        executed = {
            "biosphere.bvoc.megan.alpha_light",
            "biosphere.bvoc.megan.cl_light",
            "biosphere.bvoc.megan.ldf_sesquiterpenes",
            "biosphere.bvoc.megan.beta_sesquiterpenes",
            "biosphere.bvoc.t_standard",
            "biosphere.bvoc.megan.eopt_sesquiterpenes",
            "biosphere.bvoc.megan.ct1_sesquiterpenes",
            "biosphere.bvoc.megan.topt",
            "biosphere.bvoc.megan.ct2",
            "relic.physical.constants.universal.gas_constant",
        }
        retired = {
            "biosphere.bvoc.megan.tm_reference",
            "biosphere.bvoc.q10.temperature_interval",
            "biosphere.constants.temperature.megan_reference",
        }
        assert executed <= declared
        assert not (retired & declared)
        path = (
            Path(__file__).resolve().parents[4]
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_synthesis_sesquiterpene_sesquiterpene_synthesis.yaml"
        )
        certificate = load_iphs_equivalence_certificate(path)
        assert certificate.validate()["passed"] is True
        keys = set(certificate.parameter_keys)
        assert executed <= keys
        assert not (retired & keys)


class TestSesquiterpeneSynthesisComputeRateL1:
    """L1 unit tests for sesquiterpene emission rate computation."""
    
    @pytest.fixture
    def model(self):
        """Create sesquiterpene model for testing."""
        return SesquiterpeneSynthesis()
    
    def test_compute_rate_basic(self, model):
        """Test L1.1: Basic compute_rate functionality."""
        state = {}
        environment = {'temperature': torch.tensor(298.15)}  # 25掳C
        
        result = model._compute_rate_core({**state, **environment})
        
        # Check output structure
        assert isinstance(result, dict)
        assert 'rate' in result
        
        rate = result['rate']
        assert isinstance(rate, torch.Tensor)
        assert torch.isfinite(rate).all()
        assert rate >= 0.0  # Should be non-negative
    
    def test_compute_rate_temperature_response(self, model):
        """Test L1.2: Temperature response in compute_rate."""
        state = {}
        temperatures = [283.15, 298.15, 313.15, 328.15]  # 10, 25, 40, 55掳C
        emission_rates = []
        
        for temp in temperatures:
            environment = {'temperature': torch.tensor(temp)}
            result = model._compute_rate_core({**state, **environment})
            emission_rates.append(result['rate'].item())
        
        # Should increase exponentially with temperature
        for i in range(len(emission_rates) - 1):
            assert emission_rates[i+1] > emission_rates[i]
        
        # Check exponential behavior
        # At reference temperature (30掳C), should be close to base emission factor
        temp_ref = 303.15
        environment_ref = {'temperature': torch.tensor(temp_ref)}
        result_ref = model._compute_rate_core({**state, **environment_ref})
        
        expected_ref = model.base_emission_factor.item()
        assert result_ref['rate'].item() == pytest.approx(expected_ref, rel=0.01)
        
        # Calculate Q10 from temperature response
        temp_1 = 298.15  # 25掳C
        temp_2 = 308.15  # 35掳C
        
        env_1 = {'temperature': torch.tensor(temp_1)}
        env_2 = {'temperature': torch.tensor(temp_2)}
        
        rate_1 = model._compute_rate_core({**state, **env_1})['rate'].item()
        rate_2 = model._compute_rate_core({**state, **env_2})['rate'].item()
        
        q10 = rate_2 / rate_1
        # The Guenther et al. (2012) Eq. (7) activity of the shared response helper sets
        # the 25 to 35 degC ratio (the Eq. (11) term alone would give exp(10 beta)).
        from relic.scientific.biosphere.bvoc.responses import BVOCTemperatureResponse

        gamma = BVOCTemperatureResponse.get_compound_temperature_response(
            "sesquiterpenes", torch.tensor([temp_1, temp_2], dtype=torch.float32)
        )
        expected_q10 = (gamma[1] / gamma[0]).item()
        assert q10 == pytest.approx(expected_q10, rel=0.01)
        response_beta = _get_parameter_scalar(
            "beta_sesquiterpenes",
            "biosphere.bvoc.megan",
            "K-1",
        )
        assert q10 < torch.exp(torch.tensor(response_beta) * 10.0).item()
    
    def test_compute_rate_requires_temperature(self, model):
        """The declared temperature driver must not have a hidden default."""

        with pytest.raises(ValueError, match="Required input 'temperature'"):
            model._compute_rate_core({})
    
    def test_compute_rate_tensor_types(self, model):
        """Test L1.4: Different tensor input types."""
        state = {}
        
        # Test with float
        environment_float = {'temperature': 298.15}
        result_float = model._compute_rate_core({**state, **environment_float})
        
        # Test with tensor
        environment_tensor = {'temperature': torch.tensor(298.15)}
        result_tensor = model._compute_rate_core({**state, **environment_tensor})
        
        # Results should be equivalent
        assert torch.allclose(result_float['rate'], result_tensor['rate'])
        
        # Test with tensor on different device
        if model.device != torch.device('cpu'):
            temp_cpu = torch.tensor(298.15, device='cpu')
            environment_cpu = {'temperature': temp_cpu}
            result_cpu = model._compute_rate_core({**state, **environment_cpu})
            
            # Should still work (tensor should be moved to model device)
            assert isinstance(result_cpu['rate'], torch.Tensor)
            assert result_cpu['rate'].device == model.device
    
    def test_compute_rate_boundary_conditions(self, model):
        """Test L1.5: Boundary conditions and edge cases."""
        state = {}
        
        # Very low temperature
        environment_cold = {'temperature': torch.tensor(250.15)}  # -23掳C
        result_cold = model._compute_rate_core({**state, **environment_cold})
        
        assert result_cold['rate'] >= 0.0
        assert torch.isfinite(result_cold['rate']).all()
        
        # Very high temperature
        environment_hot = {'temperature': torch.tensor(350.15)}  # 77掳C
        result_hot = model._compute_rate_core({**state, **environment_hot})
        
        assert result_hot['rate'] >= 0.0
        assert torch.isfinite(result_hot['rate']).all()
        
        # Hot should be much higher than cold due to exponential response
        assert result_hot['rate'] > result_cold['rate']


class TestSesquiterpeneSynthesisForwardL1:
    """L1 unit tests for sesquiterpene synthesis forward pass."""
    
    @pytest.fixture
    def model(self):
        """Create sesquiterpene model for testing."""
        return SesquiterpeneSynthesis()
    
    def test_forward_basic(self, model):
        """Test L1.1: Basic forward pass functionality."""
        state = _synthesis_state(temperature=298.15)
        time = 0.0
        dt = 3600.0  # 1 hour
        
        result = model.forward(state, time, dt)
        
        expected_outputs = set(model.metadata.output_variables)
        assert set(result) == expected_outputs
        assert "sesquiterpene_emission_mechanistic" not in result
        assert "emission_rate_molar" not in result

        expected_units = {
            "sesquiterpenes_emission": "mol m-2 s-1",
            "bvoc_carbon_allocation_flux": "kg C m-2 s-1",
            "diagnostics.bvoc.sesquiterpene.emission_rate_molar": "umol m-2 s-1",
            "diagnostics.bvoc.sesquiterpene.gamma_temperature": "1",
            "diagnostics.bvoc.sesquiterpene.gamma_light": "1",
            "diagnostics.bvoc.sesquiterpene.gamma_stress": "1",
            "diagnostics.bvoc.sesquiterpene.gamma_total": "1",
        }
        for key, item in result.items():
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.source_functor == model.name
            assert item.unit == expected_units[key]
            assert torch.isfinite(item.value).all()

        emission = result["sesquiterpenes_emission"].value
        emission_molar = result[
            "diagnostics.bvoc.sesquiterpene.emission_rate_molar"
        ].value
        gamma_temperature = result[
            "diagnostics.bvoc.sesquiterpene.gamma_temperature"
        ].value
        gamma_light = result["diagnostics.bvoc.sesquiterpene.gamma_light"].value
        gamma_stress = result["diagnostics.bvoc.sesquiterpene.gamma_stress"].value
        gamma_total = result["diagnostics.bvoc.sesquiterpene.gamma_total"].value
        expected_mass_flux = (
            model.base_emission_factor
            * gamma_total
            * state["leaf_area_index"]
            * model.foliar_density_factor
            / model.seconds_per_hour
        )
        expected_umol_flux = expected_mass_flux / model.sesquiterpene_mw
        expected_mol_flux = unit_converter.convert(
            expected_umol_flux,
            variable_name="sesquiterpenes_emission",
            source_units="umol m-2 s-1",
            target_units="mol m-2 s-1",
        )
        expected_carbon_allocation = unit_converter.convert(
            expected_mol_flux
            * model.sesquiterpene_mw
            * model.carbon_fraction_sesquiterpenes,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        )
        assert torch.allclose(emission, expected_mol_flux)
        assert torch.allclose(emission_molar, emission * 1.0e6)
        assert torch.allclose(
            result["bvoc_carbon_allocation_flux"].value,
            expected_carbon_allocation,
        )
        assert torch.all(result["bvoc_carbon_allocation_flux"].value >= 0)
        assert torch.allclose(
            gamma_total,
            gamma_temperature * gamma_light * gamma_stress,
        )

    @staticmethod
    def _replace_update_value(result, key, value):
        """Replace one UpdateItem value while preserving its runtime contract."""
        item = result[key]
        result[key] = UpdateItem(
            value=value,
            mode=item.mode,
            unit=item.unit,
            per_timestep=item.per_timestep,
            source_functor=item.source_functor,
        )

    def test_open_carbon_boundary_conservation(self, model):
        """The emitted molar port must map exactly to its carbon debit."""
        result = model.forward(_synthesis_state(), time=0.0, dt=60.0)

        assert set(model.metadata.conservation_laws) == {"carbon_boundary"}
        allocation = result["bvoc_carbon_allocation_flux"].value
        rounding_bound = (
            torch.finfo(allocation.dtype).eps
            * torch.max(torch.abs(allocation)).item()
        )
        assert model.check_conservation_laws({}, result)[
            "carbon_boundary"
        ] <= rounding_bound

        tampered = dict(result)
        self._replace_update_value(
            tampered,
            "bvoc_carbon_allocation_flux",
            allocation + allocation.new_tensor(2.0e-9),
        )
        assert model.check_conservation_laws({}, tampered)[
            "carbon_boundary"
        ] == pytest.approx(2.0e-9, rel=1.0e-6, abs=1.0e-15)

    @pytest.mark.parametrize(
        ("key", "invalid_value", "error_type", "message"),
        [
            (
                "sesquiterpenes_emission",
                1.0,
                TypeError,
                "must be a torch.Tensor",
            ),
            (
                "sesquiterpenes_emission",
                torch.empty(0),
                ValueError,
                "must be non-empty",
            ),
            (
                "sesquiterpenes_emission",
                torch.tensor(float("nan")),
                ValueError,
                "must be finite",
            ),
            (
                "bvoc_carbon_allocation_flux",
                torch.tensor(float("inf")),
                ValueError,
                "must be finite",
            ),
            (
                "sesquiterpenes_emission",
                torch.tensor(1, dtype=torch.int64),
                TypeError,
                "must be floating",
            ),
        ],
    )
    def test_carbon_boundary_rejects_invalid_outputs(
        self,
        model,
        key,
        invalid_value,
        error_type,
        message,
    ):
        """Invalid conservation evidence must fail before residual evaluation."""
        result = model.forward(_synthesis_state(), time=0.0, dt=60.0)
        self._replace_update_value(result, key, invalid_value)

        with pytest.raises(error_type, match=message):
            model.check_conservation_laws({}, result)

    @pytest.mark.parametrize(
        "missing_key",
        ["sesquiterpenes_emission", "bvoc_carbon_allocation_flux"],
    )
    def test_carbon_boundary_requires_both_open_ports(
        self,
        model,
        missing_key,
    ):
        """Both sides of the open carbon mapping are mandatory."""
        result = model.forward(_synthesis_state(), time=0.0, dt=60.0)
        result.pop(missing_key)

        with pytest.raises(KeyError, match=missing_key):
            model.check_conservation_laws({}, result)

    def test_carbon_boundary_requires_matching_shapes(self, model):
        """The molar and carbon ports must have identical tensor shapes."""
        result = model.forward(_synthesis_state(), time=0.0, dt=60.0)
        allocation = result["bvoc_carbon_allocation_flux"].value
        self._replace_update_value(
            result,
            "bvoc_carbon_allocation_flux",
            allocation.reshape(1).expand(2).clone(),
        )

        with pytest.raises(ValueError, match="identical shapes"):
            model.check_conservation_laws({}, result)
    
    def test_forward_temperature_dependency(self, model):
        """Test L1.2: Temperature dependency in forward pass."""
        temperatures = [283.15, 298.15, 313.15]  # 10, 25, 40掳C
        time = 0.0
        dt = 3600.0
        emissions = []
        
        for temp in temperatures:
            state = _synthesis_state(temperature=temp)
            result = model.forward(state, time, dt)
            emissions.append(result['sesquiterpenes_emission'].value.item())
        
        # Should increase with temperature
        for i in range(len(emissions) - 1):
            assert emissions[i+1] > emissions[i]
        
        # Check exponential relationship
        # Higher temperature difference should give larger emission ratio
        ratio_1_2 = emissions[1] / emissions[0]  # 25掳C / 10掳C
        ratio_2_3 = emissions[2] / emissions[1]  # 40掳C / 25掳C
        
        # Both ratios should be > 1, and similar for equal temperature differences
        assert ratio_1_2 > 1.0
        assert ratio_2_3 > 1.0
    
    def test_forward_tensor_extraction(self, model):
        """Test L1.3: Tensor extraction in forward pass."""
        # Test with tensor input
        state_tensor = _synthesis_state(temperature=torch.tensor(298.15))
        result_tensor = model.forward(state_tensor, 0.0, 3600.0)
        
        # Test with float input
        state_float = _synthesis_state(temperature=298.15)
        result_float = model.forward(state_float, 0.0, 3600.0)
        
        # Results should be equivalent
        assert torch.allclose(
            result_tensor['sesquiterpenes_emission'].value,
            result_float['sesquiterpenes_emission'].value,
        )
    
    def test_forward_missing_temperature(self, model):
        """Required temperature must fail fast instead of using a default."""

        with pytest.raises(ValueError, match="Required input 'temperature'"):
            model.forward({"leaf_area_index": 1.0}, 0.0, 3600.0)

    def test_forward_requires_leaf_area_index(self, model):
        """The leaf-mass to ground-area bridge cannot use a hidden LAI."""
        with pytest.raises(ValueError, match="Required input 'leaf_area_index'"):
            model.forward({"temperature": 303.15}, 0.0, 3600.0)

    @pytest.mark.parametrize(
        "invalid_lai",
        [-1.0, float("nan"), float("inf")],
    )
    def test_forward_rejects_invalid_leaf_area_index(self, model, invalid_lai):
        """LAI must be finite and non-negative before area conversion."""
        with pytest.raises(ValueError, match="leaf_area_index"):
            model.forward(_synthesis_state(leaf_area_index=invalid_lai), 0.0, 3600.0)

    def test_zero_leaf_area_index_closes_all_material_fluxes(self, model):
        """Zero canonical LAI area must produce zero species and carbon flux."""
        result = model.forward(_synthesis_state(leaf_area_index=0.0), 0.0, 3600.0)
        for key in (
            "sesquiterpenes_emission",
            "bvoc_carbon_allocation_flux",
            "diagnostics.bvoc.sesquiterpene.emission_rate_molar",
        ):
            assert torch.count_nonzero(result[key].value) == 0

    def test_leaf_area_index_scales_reduced_bridge_linearly(self, model):
        """With responses fixed, RELIC's reduced mass-area bridge is linear in LAI."""
        lai_one = model.forward(_synthesis_state(leaf_area_index=1.0), 0.0, 3600.0)
        lai_two = model.forward(_synthesis_state(leaf_area_index=2.0), 0.0, 3600.0)
        for key in (
            "sesquiterpenes_emission",
            "bvoc_carbon_allocation_flux",
            "diagnostics.bvoc.sesquiterpene.emission_rate_molar",
        ):
            assert torch.allclose(lai_two[key].value, 2.0 * lai_one[key].value)

    @pytest.mark.parametrize("invalid_temperature", [float("nan"), float("inf")])
    def test_forward_rejects_nonfinite_temperature(self, model, invalid_temperature):
        """Non-finite temperature must not enter the emission response."""

        with pytest.raises(ValueError, match="temperature.*finite"):
            model.forward(
                _synthesis_state(temperature=invalid_temperature), 0.0, 3600.0
            )
    
    def test_forward_consistency_with_compute_rate(self, model):
        """Test L1.5: Consistency between forward and compute_rate methods."""
        reference_temperature = _get_parameter_scalar("t_standard", "biosphere.bvoc", "K")
        state = _synthesis_state(temperature=reference_temperature)
        environment = {'temperature': torch.tensor(reference_temperature)}

        # Forward method
        forward_result = model.forward(state, 0.0, 3600.0)
        forward_emission = forward_result['sesquiterpenes_emission'].value
        
        # Compute_rate method
        rate_result = model._compute_rate_core(environment)
        rate_emission = rate_result['rate']
        
        expected_area_mass_flux = (
            rate_emission
            * state["leaf_area_index"]
            * model.foliar_density_factor
            / model.seconds_per_hour
        )
        expected_molar_flux = (
            unit_converter.convert(
                expected_area_mass_flux / model.sesquiterpene_mw,
                variable_name="sesquiterpenes_emission",
                source_units="umol m-2 s-1",
                target_units="mol m-2 s-1",
            )
        )
        assert torch.allclose(forward_emission, expected_molar_flux, rtol=0.01)

    def test_canonical_call_routes_open_carbon_port_without_warning(self, model):
        """The declared atmospheric port is not a closed local carbon ledger."""

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = model(_synthesis_state(temperature=torch.tensor(303.15)), time=0.0, dt=60.0)
            second = model(_synthesis_state(temperature=torch.tensor(303.15)), time=60.0, dt=60.0)

        conservation_warnings = [
            warning
            for warning in caught
            if "Conservation law 'carbon' violated" in str(warning.message)
        ]
        assert conservation_warnings == []
        assert torch.allclose(
            first["sesquiterpenes_emission"].value,
            second["sesquiterpenes_emission"].value,
        )
        assert first["sesquiterpenes_emission"].source_functor == model.name

    def test_simulator_routes_emission_as_open_port_with_scm_provenance(self):
        """The real simulator must not classify the emission flux as carbon storage."""
        simulator = RELICSimulator({
            "time_step": 60.0,
            "max_time": 120.0,
            "device": "cpu",
            "enable_auto_scale": True,
        })
        simulator.initialize_state({
            "temperature": torch.tensor(303.15),
            "leaf_area_index": torch.tensor(1.0),
            "sesquiterpenes_emission": torch.tensor(0.0),
            "bvoc_carbon_allocation_flux": torch.tensor(0.0),
        })
        model = SesquiterpeneSynthesis()
        simulator.register_functor(model)

        carbon_ledger = simulator.state_manager.get_conservation_ledger().get(
            "carbon", []
        )
        assert "sesquiterpenes_emission" not in carbon_ledger
        assert "bvoc_carbon_allocation_flux" not in carbon_ledger

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulator.step(dt=60.0)
            first_value = simulator.state_manager.get_state(
                "sesquiterpenes_emission"
            ).clone()
            first_carbon_allocation = simulator.state_manager.get_state(
                "bvoc_carbon_allocation_flux"
            ).clone()
            simulator.step(dt=60.0)
            second_value = simulator.state_manager.get_state(
                "sesquiterpenes_emission"
            ).clone()
            second_carbon_allocation = simulator.state_manager.get_state(
                "bvoc_carbon_allocation_flux"
            ).clone()

        assert torch.allclose(first_value, second_value)
        assert torch.allclose(first_carbon_allocation, second_carbon_allocation)
        assert first_value.item() > 0.0
        assert first_carbon_allocation.item() > 0.0
        conservation_warnings = [
            warning
            for warning in caught
            if "Conservation violation" in str(warning.message)
        ]
        assert conservation_warnings == []

        provenance = simulator.get_variable_provenance(
            "sesquiterpenes_emission"
        )
        assert provenance
        assert {record["functor"] for record in provenance} == {
            "sesquiterpene_synthesis"
        }
        final_writers = [
            record for record in provenance if record["role"] == "final_writer"
        ]
        assert final_writers == [{
            "functor": "sesquiterpene_synthesis",
            "mode": "UpdateMode.REPLACE",
            "step": None,
            "role": "final_writer",
        }]
        carbon_provenance = simulator.get_variable_provenance(
            "bvoc_carbon_allocation_flux"
        )
        assert {record["functor"] for record in carbon_provenance} == {
            "sesquiterpene_synthesis"
        }
        carbon_final_writers = [
            record
            for record in carbon_provenance
            if record["role"] == "final_writer"
        ]
        assert carbon_final_writers == [{
            "functor": "sesquiterpene_synthesis",
            "mode": "UpdateMode.REPLACE",
            "step": None,
            "role": "final_writer",
        }]
        assert simulator.get_effective_device(model.name).type == "cpu"


class TestSesquiterpeneStressModelL1:
    """L1 unit tests for SesquiterpeneStressModel."""

    def test_stress_model_certificate_has_no_stray_carriage_return(self):
        """Every carriage return of the SesquiterpeneStressModel certificate belongs to a
        CRLF line ending, the file's main line ending; the stray CR of the former
        CR CR LF sequence is normalised."""
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[4]
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_synthesis_sesquiterpene_sesquiterpene_stress_model.yaml"
        )
        raw = path.read_bytes()
        crlf = raw.count(b"\r\n")
        assert crlf > raw.count(b"\n") - crlf
        assert raw.replace(b"\r\n", b"").count(b"\r") == 0
    
    def test_stress_model_instantiation(self):
        """Test L1.1: Stress model instantiation."""
        stress_model = SesquiterpeneStressModel()
        
        assert isinstance(stress_model, SesquiterpeneStressModel)
        assert isinstance(stress_model, torch.nn.Module)
        
        # Check parameters
        assert hasattr(stress_model, 'stress_threshold')
        assert hasattr(stress_model, 'stress_amplification')

        assert stress_model.stress_threshold.item() == pytest.approx(STRESS_THRESHOLD)
        assert stress_model.stress_amplification.item() == pytest.approx(STRESS_AMPLIFICATION)
    
    def test_stress_model_custom_config(self):
        """Test L1.2: Stress model with custom configuration."""
        config = {
            'stress_threshold': 0.3,
            'stress_amplification': 5.0
        }
        
        # Note: Current implementation doesn't use config
        # This test is for future consistency
        stress_model = SesquiterpeneStressModel(config=config)
        
        # Should instantiate successfully
        assert isinstance(stress_model, SesquiterpeneStressModel)

    def test_forward_contract_rejects_compatibility_entrypoints(self):
        """Both functors expose only state-dict forward and explicit constructor inputs."""
        with pytest.raises(TypeError):
            SesquiterpeneSynthesis(compatibility_mode=True)
        with pytest.raises(TypeError):
            SesquiterpeneStressModel(compatibility_mode=True)

        synthesis = SesquiterpeneSynthesis()
        stress_model = SesquiterpeneStressModel()
        with pytest.raises(TypeError):
            synthesis.forward(
                {"temperature": torch.tensor(303.15)},
                time=0.0,
                dt=1.0,
                compatibility_mode=True,
            )
        with pytest.raises(TypeError):
            stress_model.forward(
                {"stress_level": torch.tensor(0.8)},
                time=0.0,
                dt=1.0,
                compatibility_mode=True,
            )
        with pytest.raises(TypeError):
            stress_model.forward(torch.tensor(0.8), time=0.0, dt=1.0)
    
    def test_stress_response_basic(self):
        """Test L1.3: Basic stress response calculation."""
        stress_model = SesquiterpeneStressModel()
        
        # Test different stress levels
        stress_levels = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        result = stress_model.forward(
            {"stress_level": stress_levels},
            time=0.0,
            dt=1.0,
        )
        response = result["stress_response_factor"].value
        
        # Check output structure
        assert set(result) == {"stress_response_factor"}
        assert isinstance(response, torch.Tensor)
        assert response.shape == stress_levels.shape
        
        # Check value range
        assert torch.all(response >= 1.0)  # Minimum response
        max_response = 1.0 + stress_model.stress_amplification.item()
        assert torch.all(response <= max_response)  # Maximum response
    
    def test_stress_response_threshold_behavior(self):
        """Test L1.4: Stress response threshold behavior."""
        stress_model = SesquiterpeneStressModel()
        threshold = stress_model.stress_threshold.item()
        amplification = stress_model.stress_amplification.item()
        epsilon = min(0.1, max(1e-3, (1.0 - threshold) / 2))

        # Below threshold
        stress_low = torch.tensor([
            max(0.0, threshold - 2 * epsilon),
            max(0.0, threshold - epsilon),
            max(0.0, threshold - 0.25 * epsilon),
        ])
        response_low = stress_model.forward(
            {"stress_level": stress_low}, time=0.0, dt=1.0
        )["stress_response_factor"].value

        # Should all be 1.0 (no amplification)
        assert torch.allclose(response_low, torch.ones_like(stress_low))

        # At threshold
        stress_threshold = torch.tensor(threshold)
        response_threshold = stress_model.forward(
            {"stress_level": stress_threshold}, time=0.0, dt=1.0
        )["stress_response_factor"].value
        assert response_threshold.item() == pytest.approx(1.0)

        # Above threshold
        stress_high = torch.tensor([
            min(1.0, threshold + epsilon),
            min(1.0, threshold + 2 * epsilon),
            1.0
        ])
        response_high = stress_model.forward(
            {"stress_level": stress_high}, time=0.0, dt=1.0
        )["stress_response_factor"].value

        # Should be amplified
        assert torch.all(response_high > 1.0)

        # Check amplification calculation
        # For stress = 1.0: response = 1.0 + (1.0 - threshold) * amplification
        stress_max = torch.tensor(1.0)
        response_max = stress_model.forward(
            {"stress_level": stress_max}, time=0.0, dt=1.0
        )["stress_response_factor"].value
        expected_max = 1.0 + (1.0 - threshold) * amplification
        assert response_max.item() == pytest.approx(expected_max)
    
    def test_stress_response_edge_cases(self):
        """Test L1.5: Stress response edge cases."""
        stress_model = SesquiterpeneStressModel()
        
        # Zero stress
        stress_zero = torch.tensor(0.0)
        response_zero = stress_model.forward(
            {"stress_level": stress_zero}, time=0.0, dt=1.0
        )["stress_response_factor"].value
        assert response_zero.item() == pytest.approx(1.0)
        
        # Exactly at threshold
        threshold = stress_model.stress_threshold.item()
        stress_exact = torch.tensor(threshold)
        response_exact = stress_model.forward(
            {"stress_level": stress_exact}, time=0.0, dt=1.0
        )["stress_response_factor"].value
        assert response_exact.item() == pytest.approx(1.0)
        
        # Very high stress (should be clamped)
        stress_extreme = torch.tensor(10.0)  # Way above threshold
        response_extreme = stress_model.forward(
            {"stress_level": stress_extreme}, time=0.0, dt=1.0
        )["stress_response_factor"].value
        
        max_response = 1.0 + stress_model.stress_amplification.item()
        assert response_extreme.item() <= max_response + 1e-6  # Allow for numerical precision
    
    def test_stress_response_multiple_values(self):
        """Test L1.6: Stress response with multiple values."""
        stress_model = SesquiterpeneStressModel()

        threshold = stress_model.stress_threshold.item()
        delta = min(0.1, max(1e-3, threshold / 2))
        # Mixed stress levels relative to the configured threshold
        stress_mixed = torch.tensor([
            max(0.0, threshold - 2 * delta),
            max(0.0, threshold - delta),
            min(1.0, threshold + delta),
            min(1.0, threshold + 2 * delta),
        ])
        response_mixed = stress_model.forward(
            {"stress_level": stress_mixed}, time=0.0, dt=1.0
        )["stress_response_factor"].value

        # First two should be 1.0 (below threshold)
        assert torch.allclose(response_mixed[:2], torch.ones(2))

        # Last two should be > 1.0 (above threshold)
        assert torch.all(response_mixed[2:] > 1.0)

        # Should increase with stress level above threshold
        assert response_mixed[3] > response_mixed[2]

    def test_forward_returns_the_contract_replacement_item(self):
        """The algebraic factor is one REPLACE item in the SCM unit "1"."""
        stress_model = SesquiterpeneStressModel()
        result = stress_model.forward(
            {"stress_level": torch.tensor([STRESS_THRESHOLD + 0.1])},
            time=0.0,
            dt=60.0,
        )

        item = result["stress_response_factor"]
        assert isinstance(item, UpdateItem)
        assert item.mode == UpdateMode.REPLACE
        assert item.unit == "1"
        assert item.per_timestep is False
        assert item.source_functor == stress_model.name

    def test_stress_factor_declares_no_conserved_quantity(self):
        """A dimensionless multiplier advances no carbon or other reservoir."""
        stress_model = SesquiterpeneStressModel()
        assert not stress_model.metadata.conservation_laws

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stress_model(
                {"stress_level": torch.tensor([STRESS_THRESHOLD + 0.1])},
                time=0.0,
                dt=60.0,
            )
        assert [w for w in caught if "Conservation law" in str(w.message)] == []

    def test_governed_default_step_replaces_the_factor(self):
        """A default-configuration simulator commits the algebraic factor as a
        replacement: the committed value stays f(S) and is not summed over steps."""
        stress_level = STRESS_THRESHOLD + 0.1
        expected = 1.0 + (stress_level - STRESS_THRESHOLD) * STRESS_AMPLIFICATION
        stress_model = SesquiterpeneStressModel()
        simulator = RELICSimulator({
            "time_step": 60.0,
            "max_time": 600.0,
            "device": "cpu",
            "state_units": {"stress_level": "1"},
        })
        simulator.initialize_state({"stress_level": torch.tensor([stress_level])})
        simulator.register_functor(stress_model, execution_group="fast")

        for _ in range(3):
            simulator.step(dt=60.0)
            committed = simulator.state_manager.get_state("stress_response_factor")
            assert committed.reshape(-1)[0].item() == pytest.approx(expected, rel=1e-6)

    EXECUTED_STRESS_LAW = "min(max(1 + max(S - S_th, 0) * A, 1), 1 + A)"

    def test_metadata_declares_only_the_executed_threshold_law(self):
        """The executed law reads one aggregated stress driver and two ParameterManager
        parameters (docs/source/theory/bvoc_emissions.md); no multi-stress weighting,
        recovery state or bridge response parameter is executed, so none is declared."""
        stress_model = SesquiterpeneStressModel()
        metadata = stress_model.metadata

        assert set(metadata.parameters) == {
            "biosphere.bvoc.sesquiterpene.stress_threshold",
            "biosphere.bvoc.sesquiterpene.stress_amplification",
        }
        for full_key in metadata.parameters:
            assert _PARAM_MANAGER.get_parameter(full_key) is not None, full_key
        # FunctorMetadata replaces the declared expression by the canonical IPHS
        # template, whose parameter-access line lists the declared parameters.
        assert (
            "metadata.parameters keys: biosphere.bvoc.sesquiterpene.stress_amplification, "
            "biosphere.bvoc.sesquiterpene.stress_threshold"
            in metadata.mathematical_expression
        )
        assert "multi-stress" not in metadata.description
        declared_source = inspect.getsource(SesquiterpeneStressModel.__init__)
        assert self.EXECUTED_STRESS_LAW in declared_source
        for retired in ("sum_i", "dR/dt", "w_i", "tau_recovery", "water_stress_weight"):
            assert retired not in declared_source, retired

    def test_certificate_states_the_executed_threshold_law(self):
        """The level-1 certificate carries the executed law and exactly the parameters
        the model reads."""
        from pathlib import Path

        import yaml

        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )

        path = (
            Path(__file__).resolve().parents[4]
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_synthesis_sesquiterpene_sesquiterpene_stress_model.yaml"
        )
        certificate = load_iphs_equivalence_certificate(path)
        assert certificate.validate()["passed"] is True
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        stress_model = SesquiterpeneStressModel()
        assert set(certificate.parameter_keys) == set(stress_model.metadata.parameters)
        for text in (
            payload["legacy_formula"],
            certificate.literature_formula,
            certificate.legacy_expression,
        ):
            assert self.EXECUTED_STRESS_LAW in text
            assert "sum_i" not in text

    VERIFIED_STRESS_REFERENCES = (
        "Copolovici et al. (2012). Journal of Plant Physiology 169:664-672",
        "Loreto & Schnitzler (2010). Trends in Plant Science 15:154-166",
        "Niinemets et al. (2013). Frontiers in Plant Science 4:262",
    )

    def test_references_are_registered_stress_dose_response_records(self):
        """Declared and certified references are registered records reporting
        stress-severity dependent emissions; the unregistered Plant Physiology 160
        and Plant, Cell & Environment 36 citations are retired."""
        from pathlib import Path

        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )

        path = (
            Path(__file__).resolve().parents[4]
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_synthesis_sesquiterpene_sesquiterpene_stress_model.yaml"
        )
        certificate = load_iphs_equivalence_certificate(path)
        declared = list(SesquiterpeneStressModel().metadata.references)
        for references in (declared, list(certificate.references)):
            for reference in self.VERIFIED_STRESS_REFERENCES:
                assert reference in references
            for retired in ("Plant Physiology 160", "Plant, Cell & Environment 36"):
                assert not any(retired in reference for reference in references), retired


class TestSesquiterpeneBiologicalProcessInterfaceL1:
    """L1 unit tests for BiologicalProcess interface consistency."""
    
    @pytest.fixture
    def model(self):
        """Create sesquiterpene model for testing."""
        return SesquiterpeneSynthesis()
    
    def test_biological_process_inheritance(self, model):
        """Test L1.1: BiologicalProcess inheritance."""
        from relic.core.abstractions import BiologicalProcess
        assert isinstance(model, BiologicalProcess)
        
        # Check required attributes
        assert hasattr(model, 'process_name')
        
        # Check required methods
        assert hasattr(model, 'forward')
        assert callable(model.forward)
        assert not hasattr(model, 'compute_rate')
    
    def test_temperature_response_parameter_has_single_owner(self, model):
        """The delegated response coefficient must not be duplicated as a buffer."""
        response_beta = _get_parameter_scalar(
            "beta_sesquiterpenes",
            "biosphere.bvoc.megan",
            "K-1",
        )

        assert response_beta == pytest.approx(0.17)
        assert not hasattr(model, "temperature_sensitivity")
    
    def test_process_attributes(self, model):
        """Test L1.3: Process attribute validation."""
        # Check process name
        assert isinstance(model.process_name, str)
        assert len(model.process_name) > 0


class TestSesquiterpeneModuleBoundaryConditionsL1:
    """L1 unit tests for boundary conditions and robustness."""
    
    @pytest.fixture
    def model(self):
        """Create sesquiterpene model for testing."""
        return SesquiterpeneSynthesis()
    
    def test_extreme_temperature_handling(self, model):
        """Test L1.1: Extreme temperature handling."""
        # Very low temperature
        state_cold = _synthesis_state(temperature=200.15)  # -73 C
        result_cold = model.forward(state_cold, 0.0, 3600.0)
        
        cold_emission = result_cold['sesquiterpenes_emission'].value
        assert torch.isfinite(cold_emission).all()
        assert cold_emission >= 0.0
        
        # Very high temperature
        state_hot = _synthesis_state(temperature=400.15)  # 127 C
        result_hot = model.forward(state_hot, 0.0, 3600.0)
        
        hot_emission = result_hot['sesquiterpenes_emission'].value
        assert torch.isfinite(hot_emission).all()
        assert hot_emission >= 0.0
        
        # Extreme difference should be reflected in emissions
        assert hot_emission > cold_emission
    
    def test_tensor_device_consistency(self, model):
        """Test L1.2: Tensor device consistency."""
        # Input on different device than model
        if torch.cuda.is_available() and model.device != torch.device('cuda'):
            # This might not work if model is on CUDA and input is on CPU
            # But _extract_tensor should handle device transfer
            pass
        
        # Test with CPU tensors
        state_cpu = _synthesis_state(
            temperature=torch.tensor(298.15, device='cpu')
        )
        result = model.forward(state_cpu, 0.0, 3600.0)
        
        # Result should be on model's device
        assert result['sesquiterpenes_emission'].value.device == model.device
    
    def test_numerical_stability(self, model):
        """Test L1.3: Numerical stability."""
        # Test with very small temperature differences
        temp_base = 298.15
        temp_close = 298.150001  # Very close temperature
        
        state_base = _synthesis_state(temperature=temp_base)
        state_close = _synthesis_state(temperature=temp_close)
        
        result_base = model.forward(state_base, 0.0, 3600.0)
        result_close = model.forward(state_close, 0.0, 3600.0)
        
        # Should be numerically stable (small input change -> small output change)
        emission_base = result_base['sesquiterpenes_emission'].value
        emission_close = result_close['sesquiterpenes_emission'].value
        
        relative_diff = torch.abs(emission_close - emission_base) / (emission_base + 1e-8)
        assert relative_diff < 0.01  # Less than 1% difference
    
    def test_memory_efficiency(self, model):
        """Test L1.4: Memory efficiency with multiple calls."""
        # Multiple forward passes should not accumulate memory
        state = _synthesis_state(temperature=298.15)
        
        results = []
        for i in range(100):
            result = model.forward(state, 0.0, 3600.0)
            results.append(result['sesquiterpenes_emission'].value.item())
        
        # All results should be the same
        assert all(abs(r - results[0]) < 1e-6 for r in results)
        
        # No memory issues
        assert len(results) == 100


class TestSesquiterpeneModuleExportsL1:
    """L1 unit tests for module exports."""
    
    def test_module_exports(self):
        """Test L1.1: Module __all__ exports."""
        from relic.scientific.biosphere.bvoc.synthesis import sesquiterpene
        
        # Check that __all__ is defined
        assert hasattr(sesquiterpene, '__all__')
        
        expected_exports = [
            'SesquiterpeneSynthesis',
            'SesquiterpeneStressModel'
        ]
        
        for item in expected_exports:
            assert item in sesquiterpene.__all__
            assert hasattr(sesquiterpene, item)
    
    def test_class_availability(self):
        """Test L1.2: Direct class imports."""
        # Test that classes can be imported directly
        try:
            from relic.scientific.biosphere.bvoc.synthesis.sesquiterpene import SesquiterpeneSynthesis
            from relic.scientific.biosphere.bvoc.synthesis.sesquiterpene import SesquiterpeneStressModel
            
            # Should be able to instantiate
            synthesis = SesquiterpeneSynthesis()
            stress = SesquiterpeneStressModel()
            
            assert isinstance(synthesis, SesquiterpeneSynthesis)
            assert isinstance(stress, SesquiterpeneStressModel)
            
        except ImportError as e:
            pytest.fail(f"Failed to import classes: {e}")
    
    def test_functor_lifecycle_surface(self):
        """Test L1.3: The canonical Functor lifecycle surface is explicit."""
        model = SesquiterpeneSynthesis()

        # Check the configured Functor lifecycle state.
        assert hasattr(model, 'config')
        assert hasattr(model, 'device')

        # The public lifecycle is forward-only; no compute-rate alias is exposed.
        assert callable(model.forward)
        assert not hasattr(model, 'compute_rate')
        assert callable(model.get_dependencies)
        assert callable(model.get_provides)


class TestDeclaredStateInputsOnly:
    """Sesquiterpene inputs come only from state; environment is not an input channel."""

    def test_synthesis_reads_its_inputs_only_from_state(self):
        model = SesquiterpeneSynthesis()
        state = {
            "temperature": torch.tensor([303.15]),
            "leaf_area_index": torch.tensor([2.0]),
            "ppfd": torch.tensor([800.0]),
        }
        with pytest.raises(TypeError):
            model.forward(state, 0.0, 60.0, environment={"ppfd": torch.tensor([0.0])})
        with pytest.raises(TypeError):
            model._compute_rate_core(state, {"ppfd": torch.tensor([0.0])})
        core = model._compute_rate_core(state)
        assert torch.isfinite(core["rate"]).all()

    def test_stress_model_reads_its_input_only_from_state(self):
        model = SesquiterpeneStressModel()
        with pytest.raises(TypeError):
            model.forward({"stress_level": torch.tensor([0.9])}, 0.0, 60.0, environment={})


if __name__ == "__main__":
    pytest.main([__file__])


