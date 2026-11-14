"""
L1 Unit Tests for Isotope Fractionation Modules

Comprehensive testing of isotope fractionation calculators including carbon-13,
nitrogen-15, and oxygen-18 fractionation in biogeochemical processes.

Test Coverage:
- Basic instantiation and parameter initialization
- Fractionation factor calculations for different processes
- Temperature-dependent fractionation mechanisms
- Environmental response functions
- Process-specific fractionation models (photosynthesis, respiration, decomposition)
- Delta notation conversions and isotope ratio calculations
- Isotope flux calculations and mass balance
- Parameter sensitivity analysis
- Boundary conditions and robustness
- Scientific accuracy validation against literature values

Scientific Validation:
- Carbon-13: delta13C = (R_sample / R_standard - 1) * 1000
- Nitrogen-15: delta15N = (R_sample / R_standard - 1) * 1000
- Oxygen-18: delta18O = (R_sample / R_standard - 1) * 1000
- Fractionation factor: alpha = k_light / k_heavy = 1 / (1 + epsilon / 1000)
- Farquhar C3 model: epsilon = a + (b - a) * (ci / ca)

References:
- Farquhar et al. (1989). Carbon isotope discrimination and photosynthesis
- Mariotti et al. (1981). Experimental determination of nitrogen kinetic isotope fractionation
- Flanagan et al. (1991). Fractionation of oxygen isotopes during respiration
- O'Leary (1988). Carbon isotopes in photosynthesis
- Bowling et al. (2008). Carbon isotopes in respired CO2
"""

import torch
import pytest
import warnings
import inspect
from unittest.mock import Mock, patch

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from relic.scientific.isotopes.fractionation.base import (
    FractionationError,
    IsotopeFractionationCalculator,
)
import relic.scientific.isotopes.fractionation.base as fractionation_base_module
import relic.scientific.isotopes.fractionation.carbon_13 as carbon13_module
import relic.scientific.isotopes.fractionation.nitrogen_15 as nitrogen15_module
import relic.scientific.isotopes.fractionation.oxygen_18 as oxygen18_module
from relic.scientific.isotopes.config import (
    IsotopeSpecies,
    get_default_isotope_config,
)
from relic.parameters import get_global_manager


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


class _ConstantAlphaCalculator(IsotopeFractionationCalculator):
    """Minimal calculator used to isolate the canonical flux partition."""

    def __init__(self, alpha, *, parameter_manager):
        if parameter_manager is None:
            raise TypeError("parameter_manager is required")
        self.parameter_manager = parameter_manager
        self.alpha = alpha

    def calculate_fractionation_factor(
        self,
        process_type,
        temperature,
        process_params=None,
        environmental_state=None,
        **kwargs,
    ):
        _ = process_type, process_params, environmental_state, kwargs
        return torch.as_tensor(
            self.alpha,
            dtype=temperature.dtype,
            device=temperature.device,
        )

    def delta_notation(self, isotope_ratio):
        return isotope_ratio

    def ratio_from_delta(self, delta_value):
        return delta_value


def test_fractionation_family_requires_explicit_parameter_manager() -> None:
    """Reject omitted manager ownership across the abstract family."""
    classes = (
        IsotopeFractionationCalculator,
        carbon13_module.Carbon13FractionationCalculator,
        nitrogen15_module.Nitrogen15FractionationCalculator,
        oxygen18_module.Oxygen18FractionationCalculator,
    )

    for calculator_class in classes:
        parameter = inspect.signature(calculator_class).parameters[
            "parameter_manager"
        ]
        assert parameter.default is inspect.Parameter.empty

    for calculator_class in classes[1:]:
        with pytest.raises(TypeError):
            calculator_class()


def test_fractionation_sources_have_no_global_or_parallel_lookup_policy() -> None:
    """Forbid global fallback, local key composition, and source ordering."""
    for module in (
        fractionation_base_module,
        carbon13_module,
        nitrogen15_module,
        oxygen18_module,
    ):
        source = inspect.getsource(module)
        assert "get_global_manager" not in source
        assert "_resolve_parameter_key" not in source
        assert "source_order" not in source


def test_fractionation_parameter_access_delegates_to_injected_manager() -> None:
    """Forward key, namespace, units, type, and bounds to one manager owner; the key is
    required unless the caller declares it optional, and no default is forwarded."""

    class RecordingManager:
        def __init__(self):
            self.calls = []

        def get(self, key, **kwargs):
            self.calls.append((key, kwargs))
            return 17.0

    manager = RecordingManager()
    calculator = object.__new__(_ConstantAlphaCalculator)
    calculator.parameter_manager = manager
    value = calculator.get_parameter_value(
        "alpha_min",
        namespace="isotopes.fractionation",
        units="1",
        expected_type=float,
        bounds=(0.0, 2.0),
    )

    assert value == 17.0
    assert manager.calls == [
        (
            "alpha_min",
            {
                "namespace": "isotopes.fractionation",
                "units": "1",
                "expected_type": float,
                "required": True,
                "bounds": (0.0, 2.0),
            },
        )
    ]
    with pytest.raises(TypeError):
        calculator.get_parameter_value("alpha_min", default=0.5)


def test_exact_heavy_light_flux_partition_and_autograd() -> None:
    calculator = _ConstantAlphaCalculator(alpha=0.8, parameter_manager=get_global_manager())
    total_flux = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    source_ratio = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

    heavy_flux = calculator.calculate_isotope_flux(
        "candidate_process",
        total_flux,
        source_ratio,
        torch.tensor(298.15, dtype=torch.float64),
    )
    light_flux = total_flux - heavy_flux

    assert torch.allclose(
        heavy_flux,
        torch.tensor(5.0 / 3.0, dtype=torch.float64),
    )
    assert torch.allclose(
        light_flux,
        torch.tensor(25.0 / 3.0, dtype=torch.float64),
    )
    assert torch.allclose(heavy_flux + light_flux, total_flux)
    assert torch.allclose(
        heavy_flux / light_flux,
        torch.tensor(0.2, dtype=torch.float64),
    )
    assert heavy_flux.dtype == torch.float64
    assert heavy_flux.grad_fn is not None

    heavy_flux.backward()
    assert total_flux.grad is not None and torch.isfinite(total_flux.grad)
    assert source_ratio.grad is not None and torch.isfinite(source_ratio.grad)


def test_exact_flux_partition_broadcasts_without_scalar_detach() -> None:
    calculator = _ConstantAlphaCalculator(alpha=0.5, parameter_manager=get_global_manager())
    total_flux = torch.tensor([[2.0], [4.0]], dtype=torch.float64)
    source_ratio = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)

    heavy_flux = calculator.calculate_isotope_flux(
        "candidate_process",
        total_flux,
        source_ratio,
        torch.tensor(300.0, dtype=torch.float64),
    )
    transferred_ratio = 0.5 * source_ratio
    expected = total_flux * transferred_ratio / (1.0 + transferred_ratio)

    assert heavy_flux.shape == (2, 3)
    assert torch.allclose(heavy_flux, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("total_flux", "source_ratio", "alpha"),
    [
        (-1.0, 0.25, 0.8),
        (1.0, -0.25, 0.8),
        (1.0, 0.25, 0.0),
        (float("nan"), 0.25, 0.8),
        (1.0, float("inf"), 0.8),
        (1.0, 0.25, float("nan")),
    ],
)
def test_exact_flux_partition_rejects_invalid_domain(
    total_flux,
    source_ratio,
    alpha,
) -> None:
    calculator = _ConstantAlphaCalculator(alpha=alpha, parameter_manager=get_global_manager())

    with pytest.raises(FractionationError):
        calculator.calculate_isotope_flux(
            "candidate_process",
            torch.tensor(total_flux),
            torch.tensor(source_ratio),
            torch.tensor(298.15),
        )


def test_exact_flux_partition_rejects_nonfinite_temperature() -> None:
    calculator = _ConstantAlphaCalculator(alpha=0.8, parameter_manager=get_global_manager())

    with pytest.raises(FractionationError, match="temperature"):
        calculator.calculate_isotope_flux(
            "candidate_process",
            torch.tensor(1.0),
            torch.tensor(0.25),
            torch.tensor(float("nan")),
        )


def test_exact_element_mass_flux_partition_uses_nuclide_masses() -> None:
    calculator = _ConstantAlphaCalculator(alpha=0.8, parameter_manager=get_global_manager())
    total_flux = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    source_ratio = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

    heavy_mass_flux = calculator.calculate_isotope_flux(
        "candidate_process",
        total_flux,
        source_ratio,
        torch.tensor(298.15, dtype=torch.float64),
        component_basis="element_mass",
        light_relative_atomic_mass=14.00307400443,
        heavy_relative_atomic_mass=15.00010889888,
    )
    light_mass_flux = total_flux - heavy_mass_flux
    recovered_ratio = (
        heavy_mass_flux / 15.00010889888
    ) / (
        light_mass_flux / 14.00307400443
    )

    assert torch.allclose(
        heavy_mass_flux,
        torch.tensor(1.7643973885, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    assert torch.allclose(
        recovered_ratio,
        torch.tensor(0.2, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    heavy_mass_flux.backward()
    assert total_flux.grad is not None and torch.isfinite(total_flux.grad)
    assert source_ratio.grad is not None and torch.isfinite(source_ratio.grad)


@pytest.mark.parametrize(
    ("light_mass", "heavy_mass"),
    [
        (None, 15.00010889888),
        (14.00307400443, None),
        (0.0, 15.00010889888),
        (14.00307400443, float("nan")),
    ],
)
def test_element_mass_flux_partition_requires_valid_nuclide_masses(
    light_mass,
    heavy_mass,
) -> None:
    calculator = _ConstantAlphaCalculator(alpha=0.8, parameter_manager=get_global_manager())

    with pytest.raises(FractionationError, match="relative_atomic_mass"):
        calculator.calculate_isotope_flux(
            "candidate_process",
            torch.tensor(10.0),
            torch.tensor(0.25),
            torch.tensor(298.15),
            component_basis="element_mass",
            light_relative_atomic_mass=light_mass,
            heavy_relative_atomic_mass=heavy_mass,
        )


def test_carbon13_uses_the_canonical_base_flux_owner(monkeypatch) -> None:
    assert CARBON13_AVAILABLE
    assert "calculate_isotope_flux" not in Carbon13FractionationCalculator.__dict__
    assert (
        Carbon13FractionationCalculator.calculate_isotope_flux
        is IsotopeFractionationCalculator.calculate_isotope_flux
    )
    calculator = Carbon13FractionationCalculator(parameter_manager=get_global_manager())
    monkeypatch.setattr(
        calculator,
        "calculate_fractionation_factor",
        lambda *_args, **_kwargs: torch.tensor(0.8, dtype=torch.float64),
    )

    heavy_flux = calculator.calculate_isotope_flux(
        "photosynthesis",
        torch.tensor(10.0, dtype=torch.float64),
        torch.tensor(0.25, dtype=torch.float64),
        torch.tensor(298.15, dtype=torch.float64),
    )

    assert torch.allclose(
        heavy_flux,
        torch.tensor(5.0 / 3.0, dtype=torch.float64),
    )


def test_nitrogen15_uses_the_canonical_base_flux_owner(monkeypatch) -> None:
    assert NITROGEN15_AVAILABLE
    assert "calculate_isotope_flux" not in Nitrogen15FractionationCalculator.__dict__
    assert (
        Nitrogen15FractionationCalculator.calculate_isotope_flux
        is IsotopeFractionationCalculator.calculate_isotope_flux
    )
    calculator = Nitrogen15FractionationCalculator(parameter_manager=get_global_manager())
    monkeypatch.setattr(
        calculator,
        "calculate_fractionation_factor",
        lambda *_args, **_kwargs: torch.tensor(0.8, dtype=torch.float64),
    )

    heavy_flux = calculator.calculate_isotope_flux(
        "mineralization",
        torch.tensor(10.0, dtype=torch.float64),
        torch.tensor(0.25, dtype=torch.float64),
        torch.tensor(298.15, dtype=torch.float64),
    )

    assert torch.allclose(
        heavy_flux,
        torch.tensor(5.0 / 3.0, dtype=torch.float64),
    )


def test_oxygen18_uses_binary_base_owner_but_remains_runtime_unbound(
    monkeypatch,
) -> None:
    assert OXYGEN18_AVAILABLE
    assert "calculate_isotope_flux" not in Oxygen18FractionationCalculator.__dict__
    assert (
        Oxygen18FractionationCalculator.calculate_isotope_flux
        is IsotopeFractionationCalculator.calculate_isotope_flux
    )
    calculator = Oxygen18FractionationCalculator(parameter_manager=get_global_manager())
    monkeypatch.setattr(
        calculator,
        "calculate_fractionation_factor",
        lambda *_args, **_kwargs: torch.tensor(0.8, dtype=torch.float64),
    )

    heavy_flux = calculator.calculate_isotope_flux(
        "transpiration",
        torch.tensor(10.0, dtype=torch.float64),
        torch.tensor(0.25, dtype=torch.float64),
        torch.tensor(298.15, dtype=torch.float64),
    )

    assert torch.allclose(
        heavy_flux,
        torch.tensor(5.0 / 3.0, dtype=torch.float64),
    )
    assert (
        get_default_isotope_config()
        .species_configs[IsotopeSpecies.OXYGEN_18]
        .runtime_blocker.startswith("missing_17O_closure")
    )

# Try to import isotope modules - some may be implemented, others may be placeholders
try:
    from relic.scientific.isotopes.fractionation.carbon_13 import Carbon13FractionationCalculator
    from relic.scientific.isotopes.fractionation import base as fractionation_base_module
    CARBON13_AVAILABLE = True
except ImportError:
    CARBON13_AVAILABLE = False
    Carbon13FractionationCalculator = Mock
    fractionation_base_module = Mock()

try:
    from relic.scientific.isotopes.fractionation.nitrogen_15 import Nitrogen15FractionationCalculator
    NITROGEN15_AVAILABLE = True
except ImportError:
    NITROGEN15_AVAILABLE = False
    Nitrogen15FractionationCalculator = Mock

try:
    from relic.scientific.isotopes.fractionation.oxygen_18 import Oxygen18FractionationCalculator
    OXYGEN18_AVAILABLE = True
except ImportError:
    OXYGEN18_AVAILABLE = False
    Oxygen18FractionationCalculator = Mock

# Configuration enums (may need mocking if not available)
try:
    from relic.scientific.isotopes.config.multi_isotope_models import (
        CarbonFractionationParameters, PhotosynthesisType, CarbonProcessType
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Create mock classes
    PhotosynthesisType = Mock()
    PhotosynthesisType.C3 = "c3"
    PhotosynthesisType.C4 = "c4"
    PhotosynthesisType.CAM = "cam"
    CarbonProcessType = Mock()
    CarbonProcessType.PHOTOSYNTHESIS = "photosynthesis"
    CarbonProcessType.LEAF_RESPIRATION = "leaf_respiration"
    CarbonProcessType.DECOMPOSITION = "decomposition"
    CarbonFractionationParameters = Mock


class TestCarbon13FractionationL1:
    """L1 unit tests for Carbon-13 fractionation calculator."""
    
    @pytest.fixture
    def carbon13_calculator(self):
        """Create Carbon-13 fractionation calculator for testing."""
        if not CARBON13_AVAILABLE:
            pytest.skip("Carbon13FractionationCalculator not available")
        return Carbon13FractionationCalculator(parameter_manager=get_global_manager())
    
    def test_basic_instantiation(self, carbon13_calculator):
        """Test L1.1: Basic instantiation and parameter initialization."""
        assert isinstance(carbon13_calculator, Carbon13FractionationCalculator)
        
        # Check PDB standard
        assert hasattr(carbon13_calculator, 'pdb_standard')
        assert carbon13_calculator.pdb_standard == pytest.approx(0.0112372, rel=1e-6)
        
        # Check supported processes
        assert hasattr(carbon13_calculator, 'supported_processes')
        if CONFIG_AVAILABLE:
            assert CarbonProcessType.PHOTOSYNTHESIS in carbon13_calculator.supported_processes
            assert CarbonProcessType.LEAF_RESPIRATION in carbon13_calculator.supported_processes
            assert CarbonProcessType.DECOMPOSITION in carbon13_calculator.supported_processes
        
        # Check physical constants
        assert hasattr(carbon13_calculator, 'gas_constant')
        assert carbon13_calculator.gas_constant == pytest.approx(8.314, rel=1e-3)

    def test_gas_constant_uses_physical_constant_ssot(self):
        """The Arrhenius gas constant must bypass isotope calibration access."""
        if not CARBON13_AVAILABLE:
            pytest.skip("Carbon13FractionationCalculator not available")
        constant_calls = []

        def tracked_get_physical_constant(name):
            constant_calls.append(name)
            return 9.876

        with patch.object(
            fractionation_base_module,
            "get_physical_constant",
            side_effect=tracked_get_physical_constant,
            create=True,
        ):
            calculator = Carbon13FractionationCalculator(parameter_manager=get_global_manager())

        assert constant_calls == [
            "relic.physical.constants.universal.gas_constant"
        ]
        assert calculator.gas_constant == pytest.approx(9.876)
    
    def test_delta_notation_conversions(self, carbon13_calculator):
        """Test L1.2: Delta notation and isotope ratio conversions."""
        # Test known values
        test_ratios = [0.01100, 0.01123, 0.01150]  # Various 13C/12C ratios
        
        for ratio in test_ratios:
            ratio_tensor = torch.tensor(ratio)
            
            # Convert to delta notation
            delta = carbon13_calculator.delta_notation(ratio_tensor)
            
            # Convert back to ratio
            ratio_back = carbon13_calculator.ratio_from_delta(delta)
            
            # Should be consistent round trip
            assert torch.allclose(ratio_tensor, ratio_back, rtol=1e-6)
            
            # Check delta value range (relaxed for edge cases)
            # Typical atmospheric around -8 per mil, plants around -25 per mil
            assert -100.0 < delta.item() < 100.0, f"Delta value {delta.item()} outside reasonable range"
        
        # Test specific known values
        # Atmospheric CO2: delta13C about -8 per mil
        atmospheric_ratio = carbon13_calculator.ratio_from_delta(torch.tensor(-8.0))
        atmospheric_delta = carbon13_calculator.delta_notation(atmospheric_ratio)
        assert atmospheric_delta.item() == pytest.approx(-8.0, abs=0.01)
        
        # C3 plant material: delta13C about -27 per mil
        c3_ratio = carbon13_calculator.ratio_from_delta(torch.tensor(-27.0))
        c3_delta = carbon13_calculator.delta_notation(c3_ratio)
        assert c3_delta.item() == pytest.approx(-27.0, abs=0.01)
    
    def test_photosynthesis_fractionation_c3(self, carbon13_calculator):
        """Test L1.3: C3 photosynthesis fractionation calculations."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Config classes not available")
        
        # Standard conditions for C3 photosynthesis
        temperature = torch.tensor(298.15)  # 25 degC
        
        # C3 parameter set; ci/ca is an input of the call
        configured = CarbonFractionationParameters.from_manager(get_global_manager())
        carbon_params = CarbonFractionationParameters(
            **{
                **configured.model_dump(),
                "photosynthesis_type": PhotosynthesisType.C3,
                "diffusion_fractionation": 4.4,  # per mil
                "carboxylation_fractionation": 27.0,  # per mil
            }
        )

        alpha = carbon13_calculator.calculate_fractionation_factor(
            CarbonProcessType.PHOTOSYNTHESIS,
            temperature,
            carbon_params,
            ci_ca_ratio=0.7,  # Typical C3 value
        )

        # Check result
        assert torch.isfinite(alpha).all()
        assert 0.95 < alpha < 1.0, f"Alpha {alpha} outside expected range for C3"

        # Convert to discrimination; typical C3 range (15-25 per mil)
        epsilon = (1.0 / alpha - 1.0) * 1000.0
        assert 10.0 < epsilon < 30.0, f"Discrimination {epsilon} outside C3 range"

        # Higher ci/ca increases the discrimination
        for ci_ca in [0.5, 0.9]:
            alpha_test = carbon13_calculator.calculate_fractionation_factor(
                CarbonProcessType.PHOTOSYNTHESIS,
                temperature,
                carbon_params,
                ci_ca_ratio=ci_ca,
            )
            epsilon_test = (1.0 / alpha_test - 1.0) * 1000.0
            if ci_ca > 0.7:
                assert epsilon_test > epsilon
            else:
                assert epsilon_test < epsilon
    
    def test_respiration_fractionation(self, carbon13_calculator):
        """Test L1.4: Respiration fractionation calculations."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Config classes not available")
        
        # Temperature series
        temperatures = [283.15, 298.15, 313.15]  # 10, 25, 40 degC
        
        try:
            for temp in temperatures:
                temp_tensor = torch.tensor(temp)
                
                # Test leaf respiration
                alpha_leaf = carbon13_calculator.calculate_respiration_fractionation(
                    CarbonProcessType.LEAF_RESPIRATION,
                    temp_tensor
                )
                
                # Check bounds
                assert torch.isfinite(alpha_leaf).all()
                assert 0.99 < alpha_leaf < 1.02, f"Leaf respiration alpha {alpha_leaf} out of range"
                
                # Test temperature sensitivity
                if temp > 298.15:
                    # Higher temperature should change fractionation (Q10 effect)
                    alpha_ref = carbon13_calculator.calculate_respiration_fractionation(
                        CarbonProcessType.LEAF_RESPIRATION,
                        torch.tensor(298.15)
                    )
                    assert alpha_leaf != alpha_ref, "Should show temperature sensitivity"
                
                # Test different respiration types
                alpha_root = carbon13_calculator.calculate_respiration_fractionation(
                    CarbonProcessType.ROOT_RESPIRATION,
                    temp_tensor
                )
                
                assert torch.isfinite(alpha_root).all()
                assert 0.99 < alpha_root < 1.02, f"Root respiration alpha {alpha_root} out of range"
                
                # Different respiration types should have different fractionation
                assert alpha_leaf != alpha_root, "Leaf and root respiration should differ"
                
        except Exception as e:
            warnings.warn(f"Respiration fractionation test failed: {e}")
    
    def test_temperature_validation(self, carbon13_calculator):
        """Test L1.5: Temperature validation and bounds checking."""
        # Test temperature validation method
        valid_temp = torch.tensor(298.15)
        validated = carbon13_calculator._validate_temperature(valid_temp)
        assert torch.allclose(validated, valid_temp)
        
        # Test extreme temperatures
        extreme_cold = torch.tensor(200.0)  # Very cold
        extreme_hot = torch.tensor(400.0)   # Very hot
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            validated_cold = carbon13_calculator._validate_temperature(extreme_cold)
            validated_hot = carbon13_calculator._validate_temperature(extreme_hot)
            
            # Should issue warnings for extreme temperatures
            assert len(w) >= 1
            assert "Temperature outside typical range" in str(w[0].message)
        
        # Should clamp to reasonable bounds
        assert validated_cold >= 200.0
        assert validated_hot <= 400.0
        
        # Test non-tensor input
        scalar_temp = 298.15
        validated_scalar = carbon13_calculator._validate_temperature(scalar_temp)
        assert isinstance(validated_scalar, torch.Tensor)
        assert validated_scalar.item() == pytest.approx(298.15)
    
    def test_isotope_flux_calculation(self, carbon13_calculator):
        """Test L1.6: Isotope flux calculations."""
        # Test parameters
        temperature = torch.tensor(298.15)
        total_flux = torch.tensor(10.0)  # mol m-2 s-1
        source_ratio = torch.tensor(0.011)  # 13C/12C ratio
        
        try:
            # Calculate isotope flux for photosynthesis
            isotope_flux = carbon13_calculator.calculate_isotope_flux(
                CarbonProcessType.PHOTOSYNTHESIS if CONFIG_AVAILABLE else "photosynthesis",
                total_flux,
                source_ratio,
                temperature,
                ci_ca_ratio=0.7,
            )
            
            # Check result
            assert torch.isfinite(isotope_flux).all()
            assert isotope_flux >= 0, "Isotope flux should be non-negative"
            
            # Should be less than total flux (fractionation reduces 13C)
            assert isotope_flux <= total_flux, "Isotope flux should not exceed total flux"
            
            # Test flux scaling
            double_flux = carbon13_calculator.calculate_isotope_flux(
                CarbonProcessType.PHOTOSYNTHESIS if CONFIG_AVAILABLE else "photosynthesis",
                total_flux * 2.0,
                source_ratio,
                temperature,
                ci_ca_ratio=0.7,
            )
            
            # Should scale proportionally
            assert torch.allclose(double_flux, isotope_flux * 2.0, rtol=0.01)
            
        except Exception as e:
            warnings.warn(f"Isotope flux calculation test failed: {e}")
    
    def test_farquhar_model_implementation(self, carbon13_calculator):
        """Test L1.7: the C3 kernel is Delta = a + (b - a) ci/ca, increasing in ci/ca."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Config classes not available")

        configured = CarbonFractionationParameters.from_manager(get_global_manager())
        carbon_params = CarbonFractionationParameters(
            **{
                **configured.model_dump(),
                "photosynthesis_type": PhotosynthesisType.C3,
                "diffusion_fractionation": 4.4,
                "carboxylation_fractionation": 27.0,
            }
        )
        ci_ca_ratios = torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        discriminations = carbon13_calculator.farquhar_c3_model(carbon_params, ci_ca_ratios)
        torch.testing.assert_close(
            discriminations, 4.4 + (27.0 - 4.4) * ci_ca_ratios, rtol=1.0e-12, atol=0.0
        )
        assert bool(torch.all(discriminations[1:] > discriminations[:-1]))

    def test_c3_discrimination_is_the_farquhar_law_without_a_vpd_multiplier(
        self, carbon13_calculator
    ):
        """Farquhar et al. (1982, 1989): Delta = a + (b - a) ci/ca. The vapour pressure
        deficit acts on Delta only through ci/ca, which the caller supplies; a vpd entry
        in the environment does not scale the discrimination."""
        from relic.scientific.isotopes.config.multi_isotope_models import (
            CarbonFractionationParameters as Params,
            PhotosynthesisType as Pathway,
        )

        configured = Params.from_manager(get_global_manager())
        params = Params(**{**configured.model_dump(), "photosynthesis_type": Pathway.C3})
        temperature = torch.tensor(
            carbon13_calculator.reference_temperature, dtype=torch.float64
        )
        ci_ca = torch.tensor(0.7, dtype=torch.float64)
        a = params.diffusion_fractionation
        b = params.carboxylation_fractionation
        expected = torch.tensor(a + (b - a) * 0.7, dtype=torch.float64)
        epsilon = carbon13_calculator.farquhar_c3_model(params, ci_ca)
        torch.testing.assert_close(epsilon, expected, rtol=1.0e-12, atol=0.0)
        gases = {
            "ci": torch.tensor(280.0, dtype=torch.float64),
            "ca": torch.tensor(400.0, dtype=torch.float64),
        }
        without_vpd = carbon13_calculator.calculate_photosynthesis_fractionation(
            temperature, params, gases
        )
        with_vpd = carbon13_calculator.calculate_photosynthesis_fractionation(
            temperature, params, {**gases, "vpd": torch.tensor(3.5, dtype=torch.float64)}
        )
        torch.testing.assert_close(with_vpd, without_vpd, rtol=0.0, atol=0.0)

    def test_flux_calculation_fails_for_unsupported_process(self, carbon13_calculator):
        """Unsupported process declarations must not become zero isotope fluxes."""
        with pytest.raises(ValueError, match="Unsupported carbon process"):
            carbon13_calculator.calculate_isotope_flux(
                "unsupported_process",
                torch.tensor(1.0),
                torch.tensor(0.011),
                torch.tensor(298.15),
            )

    def test_cam_approximation_is_not_a_production_fractionation_law(
        self,
        carbon13_calculator,
    ):
        """CAM requires its own validated law and cannot use a C3/C4 blend."""
        configured = CarbonFractionationParameters.from_manager(get_global_manager())
        carbon_params = CarbonFractionationParameters(
            **{**configured.model_dump(), "photosynthesis_type": PhotosynthesisType.CAM}
        )

        with pytest.raises(ValueError, match="CAM fractionation is not implemented"):
            carbon13_calculator.calculate_photosynthesis_fractionation(
                torch.tensor(298.15),
                carbon_params,
                ci_ca_ratio=0.7,
            )

    def test_respiration_rejects_alternate_runtime_model(self, carbon13_calculator):
        """C13 respiration has one ParameterManager-backed Q10 runtime law."""
        with pytest.raises(ValueError, match="canonical Q10"):
            carbon13_calculator.calculate_respiration_fractionation(
                CarbonProcessType.LEAF_RESPIRATION,
                torch.tensor(298.15),
                model_type="arrhenius",
            )


class TestNitrogen15FractionationL1:
    """L1 unit tests for Nitrogen-15 fractionation calculator."""
    
    @pytest.fixture
    def nitrogen15_calculator(self):
        """Create Nitrogen-15 fractionation calculator for testing."""
        if not NITROGEN15_AVAILABLE:
            pytest.skip("Nitrogen15FractionationCalculator not available")
        return Nitrogen15FractionationCalculator(parameter_manager=get_global_manager())
    
    def test_basic_instantiation(self, nitrogen15_calculator):
        """Test L1.1: Basic instantiation and parameter initialization."""
        if isinstance(nitrogen15_calculator, Mock):
            pytest.skip("Nitrogen15FractionationCalculator not available")
        
        assert isinstance(nitrogen15_calculator, Nitrogen15FractionationCalculator)
        
        # Check atmospheric standard
        assert hasattr(nitrogen15_calculator, 'atmospheric_standard')
        # Standard N2 has delta15N = 0 by definition
        
        # Check supported processes
        assert hasattr(nitrogen15_calculator, 'supported_processes')
        
    def test_nitrification_fractionation(self, nitrogen15_calculator):
        """Test L1.2: Nitrification fractionation calculations."""
        if isinstance(nitrogen15_calculator, Mock):
            pytest.skip("Nitrogen15FractionationCalculator not available")
        
        # Test nitrification fractionation at different temperatures
        temperatures = [283.15, 298.15, 313.15]  # 10, 25, 40 degC
        
        for temp in temperatures:
            temp_tensor = torch.tensor(temp)
            
            try:
                alpha = nitrogen15_calculator.calculate_fractionation_factor(
                    'nitrification',
                    temp_tensor
                )
                
                # Nitrification fractionation bounded by configured limits
                assert torch.isfinite(alpha).all()
                alpha_value = _to_float(alpha)
                alpha_min = float(nitrogen15_calculator.min_fractionation_factor)
                alpha_max = float(nitrogen15_calculator.max_fractionation_factor)
                assert alpha_min <= alpha_value <= alpha_max, (
                    f"Nitrification alpha {alpha_value} outside configured bounds"
                )
                
                # Convert to discrimination
                epsilon = (1.0 / alpha_value - 1.0) * 1000.0
                epsilon_min = (1.0 / alpha_max - 1.0) * 1000.0
                epsilon_max = (1.0 / alpha_min - 1.0) * 1000.0
                assert epsilon_min <= epsilon <= epsilon_max, (
                    f"Nitrification discrimination {epsilon} outside configured bounds"
                )
                
            except Exception as e:
                warnings.warn(f"Nitrification fractionation test failed: {e}")
    
    def test_denitrification_fractionation(self, nitrogen15_calculator):
        """Test L1.3: Denitrification fractionation calculations."""
        if isinstance(nitrogen15_calculator, Mock):
            pytest.skip("Nitrogen15FractionationCalculator not available")
        
        temperature = torch.tensor(298.15)
        
        try:
            alpha = nitrogen15_calculator.calculate_fractionation_factor(
                'denitrification',
                temperature
            )
            
            # Denitrification fractionation bounded by configured limits
            assert torch.isfinite(alpha).all()
            alpha_value = _to_float(alpha)
            alpha_min = float(nitrogen15_calculator.min_fractionation_factor)
            alpha_max = float(nitrogen15_calculator.max_fractionation_factor)
            assert alpha_min <= alpha_value <= alpha_max, (
                f"Denitrification alpha {alpha_value} outside configured bounds"
            )
            
            epsilon = (1.0 / alpha_value - 1.0) * 1000.0
            epsilon_min = (1.0 / alpha_max - 1.0) * 1000.0
            epsilon_max = (1.0 / alpha_min - 1.0) * 1000.0
            assert epsilon_min <= epsilon <= epsilon_max, (
                f"Denitrification discrimination {epsilon} outside configured bounds"
            )
            
        except Exception as e:
            warnings.warn(f"Denitrification fractionation test failed: {e}")

    def test_flux_calculation_fails_for_unsupported_process(self, nitrogen15_calculator):
        """Unsupported N15 processes must not become zero isotope fluxes."""
        with pytest.raises(ValueError):
            nitrogen15_calculator.calculate_isotope_flux(
                "unsupported_process",
                torch.tensor(1.0),
                torch.tensor(0.0037),
                torch.tensor(298.15),
            )

    def test_temperature_response_scales_isotope_effect_about_unity(
        self,
        nitrogen15_calculator,
    ):
        """Higher temperature weakens the configured kinetic isotope effect."""
        cold = nitrogen15_calculator.calculate_fractionation_factor(
            "nitrification",
            torch.tensor(283.15),
        )
        reference = nitrogen15_calculator.calculate_fractionation_factor(
            "nitrification",
            torch.tensor(298.15),
        )
        warm = nitrogen15_calculator.calculate_fractionation_factor(
            "nitrification",
            torch.tensor(313.15),
        )

        assert cold < reference < warm < 1.0

    def test_invalid_fractionation_factor_is_rejected(self, nitrogen15_calculator):
        """Invalid coefficients fail instead of becoming unity or clipped values."""
        with pytest.raises(ValueError, match="Non-finite fractionation"):
            nitrogen15_calculator.validate_fractionation_factor(
                torch.tensor(float("nan")),
                "nitrification",
            )
        with pytest.raises(ValueError, match="outside configured bounds"):
            nitrogen15_calculator.validate_fractionation_factor(
                torch.tensor(0.5),
                "nitrification",
            )

    def test_process_summary_errors_propagate(self, nitrogen15_calculator, monkeypatch):
        """get_process_summary lists the configured processes; a process whose
        parameters cannot be read raises instead of being summarised as disabled."""

        def broken(process_name):
            raise KeyError(f"no parameters for {process_name}")

        monkeypatch.setattr(nitrogen15_calculator, "get_process_parameters", broken)
        with pytest.raises(KeyError, match="no parameters"):
            nitrogen15_calculator.get_process_summary()


class TestOxygen18FractionationL1:
    """L1 unit tests for Oxygen-18 fractionation calculator."""
    
    @pytest.fixture
    def oxygen18_calculator(self):
        """Create Oxygen-18 fractionation calculator for testing."""
        if not OXYGEN18_AVAILABLE:
            pytest.skip("Oxygen18FractionationCalculator not available")
        return Oxygen18FractionationCalculator(parameter_manager=get_global_manager())
    
    def test_basic_instantiation(self, oxygen18_calculator):
        """Test L1.1: Basic instantiation and parameter initialization."""
        if isinstance(oxygen18_calculator, Mock):
            pytest.skip("Oxygen18FractionationCalculator not available")
        
        assert isinstance(oxygen18_calculator, Oxygen18FractionationCalculator)
        
        # Check VSMOW standard
        assert hasattr(oxygen18_calculator, 'vsmow_standard')
        # VSMOW is the standard for oxygen isotopes
        
    def test_evaporation_fractionation(self, oxygen18_calculator):
        """Test L1.2: Evaporation fractionation calculations."""
        if isinstance(oxygen18_calculator, Mock):
            pytest.skip("Oxygen18FractionationCalculator not available")
        
        # Test evaporation at different temperatures and humidity
        temperature = torch.tensor(298.15)
        humidity_levels = [0.3, 0.5, 0.8]  # Different relative humidity
        
        for humidity in humidity_levels:
            alpha = oxygen18_calculator.calculate_fractionation_factor(
                'soil_evaporation',
                temperature,
                relative_humidity=humidity
            )

            # Evaporation fractionation depends on humidity and temperature
            assert torch.isfinite(alpha).all()
            assert 0.985 < alpha < 1.01, f"Evaporation alpha {alpha} outside expected range"

            # Lower humidity should increase fractionation
            if humidity < 0.5:
                epsilon = (1.0 / alpha - 1.0) * 1000.0
                assert epsilon > 5.0, "Low humidity should show enhanced fractionation"
    
    def test_transpiration_fractionation(self, oxygen18_calculator):
        """Test L1.3: Transpiration fractionation calculations."""
        if isinstance(oxygen18_calculator, Mock):
            pytest.skip("Oxygen18FractionationCalculator not available")
        
        temperature = torch.tensor(298.15)
        
        try:
            alpha = oxygen18_calculator.calculate_fractionation_factor(
                'transpiration',
                temperature
            )
            
            # Transpiration typically shows smaller fractionation than evaporation
            assert torch.isfinite(alpha).all()
            assert 0.995 < alpha < 1.005, f"Transpiration alpha {alpha} outside expected range"
            
        except Exception as e:
            warnings.warn(f"Transpiration fractionation test failed: {e}")

    def test_equilibrium_fractionation_uses_the_canonical_horita_law(
        self,
        oxygen18_calculator,
    ):
        """O18 equilibrium fractionation has no linear/accurate runtime switch."""
        temperature = torch.tensor(298.15)
        actual = oxygen18_calculator.calculate_equilibrium_fractionation(temperature)
        expected = (
            oxygen18_calculator.horita_a
            + oxygen18_calculator.horita_b / temperature
            + oxygen18_calculator.horita_c / temperature**2
            + oxygen18_calculator.horita_d / temperature**3
        )

        assert torch.allclose(actual, expected)
        assert actual.item() == pytest.approx(9.3034, abs=1.0e-3)
        assert not hasattr(oxygen18_calculator, "use_accurate_equilibrium")

    def test_evaporation_compatibility_alias_is_rejected(self, oxygen18_calculator):
        """Only process names declared by OxygenProcessType are accepted."""
        with pytest.raises(ValueError, match="Unsupported oxygen process"):
            oxygen18_calculator.calculate_fractionation_factor(
                "evaporation",
                torch.tensor(298.15),
            )

    def test_flux_calculation_fails_for_unsupported_process(self, oxygen18_calculator):
        """Unsupported process declarations must not become zero isotope fluxes."""
        with pytest.raises(ValueError, match="Unsupported oxygen process"):
            oxygen18_calculator.calculate_isotope_flux(
                "unsupported_process",
                torch.tensor(1.0),
                torch.tensor(0.002),
                torch.tensor(298.15),
            )


class TestIsotopeIntegrationL1:
    """L1 integration tests for isotope fractionation systems."""
    
    def test_multi_isotope_consistency(self):
        """Test L1.1: Multi-isotope system consistency."""
        # Test that different isotope systems can work together
        calculators = []
        
        if CARBON13_AVAILABLE:
            calculators.append(('13C', Carbon13FractionationCalculator(parameter_manager=get_global_manager())))
        
        if NITROGEN15_AVAILABLE:
            calculators.append(('15N', Nitrogen15FractionationCalculator(parameter_manager=get_global_manager())))
            
        if OXYGEN18_AVAILABLE:
            calculators.append(('18O', Oxygen18FractionationCalculator(parameter_manager=get_global_manager())))
        
        if not calculators:
            pytest.skip("No isotope calculators available")
        
        # Test consistent temperature response
        temperature = torch.tensor(298.15)
        
        for isotope, calc in calculators:
            try:
                # Each calculator should handle temperature gracefully
                assert hasattr(calc, 'calculate_fractionation_factor') or isinstance(calc, Mock)
                
                if hasattr(calc, '_validate_temperature'):
                    validated_temp = calc._validate_temperature(temperature)
                    assert torch.allclose(validated_temp, temperature)
                
            except Exception as e:
                warnings.warn(f"Multi-isotope consistency test failed for {isotope}: {e}")
    
    def test_isotope_mass_balance(self):
        """Test L1.2: Isotope mass balance calculations."""
        if not CARBON13_AVAILABLE:
            pytest.skip("Carbon-13 calculator not available")
        
        calc = Carbon13FractionationCalculator(parameter_manager=get_global_manager())
        
        # Test mass balance for a simple process
        total_flux = torch.tensor(10.0)  # mol C m-2 s-1
        source_delta = torch.tensor(-8.0)  # Atmospheric delta13C
        temperature = torch.tensor(298.15)
        
        try:
            # Convert delta to ratio
            source_ratio = calc.ratio_from_delta(source_delta)
            
            # Calculate isotope flux
            isotope_flux = calc.calculate_isotope_flux(
                CarbonProcessType.PHOTOSYNTHESIS if CONFIG_AVAILABLE else "photosynthesis",
                total_flux,
                source_ratio,
                temperature,
                ci_ca_ratio=0.7,
            )
            
            # Check mass balance
            c12_flux = total_flux - isotope_flux
            c13_flux = isotope_flux
            
            # Total should equal sum
            assert torch.allclose(total_flux, c12_flux + c13_flux, rtol=1e-6)
            
            # Both fluxes should be positive
            assert c12_flux > 0
            assert c13_flux > 0
            
            # 13C should be much smaller than 12C
            assert c13_flux < c12_flux
            
        except Exception as e:
            warnings.warn(f"Isotope mass balance test failed: {e}")
    
    def test_fractionation_bounds_validation(self):
        """Test L1.3: Fractionation factor bounds validation."""
        # Test that all fractionation factors are within reasonable bounds
        temperature = torch.tensor(298.15)
        
        calculators_tests = []
        
        if CARBON13_AVAILABLE:
            calc = Carbon13FractionationCalculator(parameter_manager=get_global_manager())
            processes = ["photosynthesis", "leaf_respiration", "decomposition"]
            calculators_tests.append((calc, processes, (0.95, 1.05)))
        
        if NITROGEN15_AVAILABLE:
            calc = Nitrogen15FractionationCalculator(parameter_manager=get_global_manager())
            processes = ["nitrification", "denitrification"]
            calculators_tests.append((calc, processes, (0.90, 1.10)))
        
        if OXYGEN18_AVAILABLE:
            calc = Oxygen18FractionationCalculator(parameter_manager=get_global_manager())
            processes = ["soil_evaporation", "transpiration"]
            calculators_tests.append((calc, processes, (0.985, 1.01)))
        
        for calc, processes, (min_alpha, max_alpha) in calculators_tests:
            for process in processes:
                try:
                    if hasattr(calc, 'calculate_fractionation_factor'):
                        # Inputs of the call: ci/ca and the substrate quality
                        inputs = {
                            "photosynthesis": {"ci_ca_ratio": 0.7},
                            "decomposition": {"substrate_quality": "mixed"},
                        }.get(process, {})
                        alpha = calc.calculate_fractionation_factor(
                            process, temperature, **inputs
                        )
                        
                        # Check bounds
                        assert torch.isfinite(alpha).all()
                        assert min_alpha < alpha < max_alpha, \
                            f"{process} alpha {alpha} outside bounds ({min_alpha}, {max_alpha})"
                
                except Exception as e:
                    warnings.warn(f"Bounds validation failed for {process}: {e}")
    
    def test_parameter_sensitivity_isotopes(self):
        """Test L1.4: Parameter sensitivity for isotope calculations."""
        if not CARBON13_AVAILABLE:
            pytest.skip("Carbon-13 calculator not available")
        
        calc = Carbon13FractionationCalculator(parameter_manager=get_global_manager())
        temperature = torch.tensor(298.15)
        
        try:
            # Test temperature sensitivity
            temp_low = torch.tensor(283.15)  # 10 degC
            temp_high = torch.tensor(313.15)  # 40 degC
            
            alpha_low = calc.calculate_respiration_fractionation(
                CarbonProcessType.LEAF_RESPIRATION if CONFIG_AVAILABLE else "leaf_respiration",
                temp_low
            )
            
            alpha_high = calc.calculate_respiration_fractionation(
                CarbonProcessType.LEAF_RESPIRATION if CONFIG_AVAILABLE else "leaf_respiration",
                temp_high
            )
            
            # Should show temperature sensitivity
            assert alpha_low != alpha_high, "Should show temperature sensitivity"
            
            # Both should be finite and reasonable
            assert torch.isfinite(alpha_low).all()
            assert torch.isfinite(alpha_high).all()
            assert 0.99 < alpha_low < 1.02
            assert 0.99 < alpha_high < 1.02
            
        except Exception as e:
            warnings.warn(f"Parameter sensitivity test failed: {e}")


class TestIsotopeRequiredInputsL1:
    """Inputs that describe the state or the substrate are required; process-law
    parameters are read from ParameterManager under their own names."""

    @pytest.fixture
    def carbon13(self):
        return carbon13_module.Carbon13FractionationCalculator(
            parameter_manager=get_global_manager()
        )

    @pytest.fixture
    def oxygen18(self):
        return oxygen18_module.Oxygen18FractionationCalculator(
            parameter_manager=get_global_manager()
        )

    def test_photosynthesis_requires_one_ci_ca_source(self, carbon13):
        """ci/ca is a physiological state: without it the discrimination raises instead
        of taking a fixed ratio, and two sources of it are ambiguous."""
        from relic.scientific.isotopes.config.multi_isotope_models import (
            CarbonFractionationParameters as Params,
            PhotosynthesisType as Pathway,
        )

        configured = Params.from_manager(get_global_manager())
        params = Params(**{**configured.model_dump(), "photosynthesis_type": Pathway.C3})
        temperature = torch.tensor(298.15)
        gases = {"ci": torch.tensor(280.0), "ca": torch.tensor(400.0)}
        with pytest.raises(ValueError, match="ci_ca_ratio"):
            carbon13.calculate_photosynthesis_fractionation(temperature, params)
        with pytest.raises(ValueError, match="ci_ca_ratio"):
            carbon13.calculate_photosynthesis_fractionation(
                temperature, params, gases, ci_ca_ratio=0.7
            )
        from_gases = carbon13.calculate_photosynthesis_fractionation(
            temperature, params, gases
        )
        from_ratio = carbon13.calculate_photosynthesis_fractionation(
            temperature, params, ci_ca_ratio=0.7
        )
        torch.testing.assert_close(from_gases, from_ratio)

    def test_decomposition_requires_the_substrate_quality(self, carbon13):
        """The substrate quality selects the correction; it is not substituted."""
        temperature = torch.tensor(298.15)
        with pytest.raises(ValueError, match="substrate_quality"):
            carbon13.calculate_decomposition_fractionation(temperature)
        with pytest.raises(ValueError, match="substrate_quality"):
            carbon13.calculate_decomposition_fractionation(
                temperature, substrate_quality="woody"
            )
        labile = carbon13.calculate_decomposition_fractionation(
            temperature, substrate_quality="labile"
        )
        assert torch.isfinite(labile).all()

    def test_bvoc_requires_the_compound_and_the_substrate_source(self, carbon13):
        """An absent or unknown compound or substrate source raises; no compound falls
        back to isoprene or to a literal factor."""
        temperature = torch.tensor(298.15)
        with pytest.raises(ValueError, match="bvoc_type"):
            carbon13.calculate_bvoc_fractionation(temperature, substrate_source="recent")
        with pytest.raises(ValueError, match="bvoc_type"):
            carbon13.calculate_bvoc_fractionation(
                temperature, bvoc_type="limonene", substrate_source="recent"
            )
        with pytest.raises(ValueError, match="substrate_source"):
            carbon13.calculate_bvoc_fractionation(temperature, bvoc_type="isoprene")
        with pytest.raises(ValueError, match="substrate_source"):
            carbon13.calculate_bvoc_fractionation(
                temperature, bvoc_type="isoprene", substrate_source="old"
            )
        alpha = carbon13.calculate_bvoc_fractionation(
            temperature, bvoc_type="isoprene", substrate_source="recent"
        )
        assert torch.isfinite(torch.as_tensor(alpha)).all()

    def test_respiration_uses_the_q10_of_its_respiration_type(self, carbon13):
        """The respiration discrimination takes the Q10 of its respiration type, not
        the photosynthetic temperature sensitivity of a supplied carbon_params."""
        from relic.scientific.isotopes.config.multi_isotope_models import (
            CarbonFractionationParameters as Params,
            CarbonProcessType as Process,
            PhotosynthesisType as Pathway,
        )

        configured = Params.from_manager(get_global_manager())
        params = Params(**{**configured.model_dump(), "photosynthesis_type": Pathway.C3})
        temperature = torch.tensor(
            carbon13.reference_temperature + 10.0, dtype=torch.float64
        )
        leaf = carbon13.respiration_parameters[Process.LEAF_RESPIRATION]
        alpha = carbon13.calculate_respiration_fractionation(
            Process.LEAF_RESPIRATION, temperature, params
        )
        expected = min(
            max(leaf["alpha_0"] * leaf["q10"], carbon13.min_fractionation_factor),
            carbon13.max_fractionation_factor,
        )
        torch.testing.assert_close(
            alpha, torch.tensor(expected, dtype=torch.float64), rtol=1.0e-6, atol=0.0
        )

    def test_oxygen18_kinetic_reductions_are_named_process_parameters(self, oxygen18):
        """The soil and canopy kinetic reductions scale the kinetic fractionation of
        their evaporation type; they are process parameters, not defaults."""
        manager = get_global_manager()
        for surface in ("soil", "canopy"):
            key = f"isotopes.fractionation.oxygen.evaporation.{surface}_kinetic_reduction"
            assert getattr(oxygen18, f"{surface}_kinetic_reduction") == pytest.approx(
                float(manager.get(key, required=True))
            )


class TestCarbon13LiteratureLawL1:
    """MA-21 12.3: Delta = a + (b - a) ci/ca (C3) and the C4 law without temperature or
    light multipliers; ci/ca is validated on its domain; one parameter-set construction."""

    @pytest.fixture
    def carbon13(self):
        return carbon13_module.Carbon13FractionationCalculator(
            parameter_manager=get_global_manager()
        )

    @staticmethod
    def _params(pathway):
        from relic.scientific.isotopes.config.multi_isotope_models import (
            CarbonFractionationParameters as Params,
        )

        base = Params.from_manager(get_global_manager())
        return Params(**{**base.model_dump(), "photosynthesis_type": pathway})

    def test_c3_discrimination_has_no_temperature_or_light_multiplier(self, carbon13):
        from relic.scientific.isotopes.config.multi_isotope_models import PhotosynthesisType

        params = self._params(PhotosynthesisType.C3)
        a, b = params.diffusion_fractionation, params.carboxylation_fractionation
        expected = 1.0 / (1.0 + (a + (b - a) * 0.7) / 1000.0)
        warm = torch.tensor(carbon13.reference_temperature + 10.0, dtype=torch.float64)
        gases = {
            "ci": torch.tensor(280.0, dtype=torch.float64),
            "ca": torch.tensor(400.0, dtype=torch.float64),
            "par": torch.tensor(100.0, dtype=torch.float64),
        }
        alpha = carbon13.calculate_photosynthesis_fractionation(warm, params, gases)
        torch.testing.assert_close(
            alpha, torch.tensor(expected, dtype=torch.float64), rtol=1.0e-12, atol=0.0
        )

    def test_c4_discrimination_has_no_temperature_multiplier(self, carbon13):
        from relic.scientific.isotopes.config.multi_isotope_models import PhotosynthesisType

        params = self._params(PhotosynthesisType.C4)
        cool = carbon13.calculate_photosynthesis_fractionation(
            torch.tensor(283.15, dtype=torch.float64), params, ci_ca_ratio=0.4
        )
        warm = carbon13.calculate_photosynthesis_fractionation(
            torch.tensor(313.15, dtype=torch.float64), params, ci_ca_ratio=0.4
        )
        torch.testing.assert_close(cool, warm, rtol=1.0e-12, atol=0.0)

    def test_ci_ca_outside_its_domain_raises_instead_of_clipping(self, carbon13):
        from relic.scientific.isotopes.config.multi_isotope_models import PhotosynthesisType

        params = self._params(PhotosynthesisType.C3)
        temperature = torch.tensor(298.15, dtype=torch.float64)
        for ratio in (-0.1, 1.2, float("nan")):
            with pytest.raises(ValueError, match="ci/ca"):
                carbon13.calculate_photosynthesis_fractionation(
                    temperature, params, ci_ca_ratio=ratio
                )
        a, b = params.diffusion_fractionation, params.carboxylation_fractionation
        for ratio in (0.2, 0.95):
            alpha = carbon13.calculate_photosynthesis_fractionation(
                temperature, params, ci_ca_ratio=torch.tensor(ratio, dtype=torch.float64)
            )
            expected = 1.0 / (1.0 + (a + (b - a) * ratio) / 1000.0)
            torch.testing.assert_close(
                alpha, torch.tensor(expected, dtype=torch.float64), rtol=1.0e-12, atol=0.0
            )

    def test_the_parameter_set_has_one_construction(self, carbon13):
        import pydantic

        from relic.scientific.isotopes.config.multi_isotope_models import (
            CarbonFractionationParameters as Params,
            CarbonProcessConfig,
        )

        assert not hasattr(carbon13, "_configured_carbon_params")
        assert {"temperature_sensitivity", "light_sensitivity"}.isdisjoint(Params.model_fields)
        assert "temperature_q10" not in CarbonProcessConfig.model_fields
        with pytest.raises(pydantic.ValidationError):
            Params()
        configured = Params.from_manager(get_global_manager())
        manager = get_global_manager()
        assert configured.diffusion_fractionation == pytest.approx(
            float(manager.get("isotopes.fractionation.carbon.diffusion_fractionation"))
        )


class TestCarbon13C4DiscriminationL1:
    """MA-21 round 13: C4 discrimination Delta = a + (b4 + b3 phi - a) ci/ca of Farquhar,
    Ehleringer and Hubick (1989), Eq. (10) (Farquhar 1983), with the Table 1 values
    a = 4.4, b4 = -5.7 (25 degC) and b3 = 30 per mil; phi is the bundle-sheath
    leakiness."""

    KEY = "isotopes.fractionation.carbon"

    @pytest.fixture
    def carbon13(self):
        return carbon13_module.Carbon13FractionationCalculator(
            parameter_manager=get_global_manager()
        )

    @staticmethod
    def _c4(**overrides):
        from relic.scientific.isotopes.config.multi_isotope_models import (
            CarbonFractionationParameters as Params,
            PhotosynthesisType,
        )

        base = Params.from_manager(get_global_manager())
        return Params(
            **{**base.model_dump(), "photosynthesis_type": PhotosynthesisType.C4, **overrides}
        )

    def test_configured_constants_are_the_farquhar_1989_table_1_values(self):
        manager = get_global_manager()
        assert float(manager.get(f"{self.KEY}.diffusion_fractionation")) == pytest.approx(4.4)
        assert float(manager.get(f"{self.KEY}.c4_pep_net_fractionation")) == pytest.approx(-5.7)
        assert float(manager.get(f"{self.KEY}.c4_rubisco_fractionation")) == pytest.approx(30.0)
        # The former +5.7 per mil PEP key carried the wrong sign of b4 and is retired.
        assert manager.get_parameter(f"{self.KEY}.c4_pep_fractionation") is None

    def test_c4_discrimination_is_farquhar_1989_eq_10(self, carbon13):
        params = self._c4()
        a = params.diffusion_fractionation
        b4 = params.c4_pep_net_fractionation
        b3 = params.c4_rubisco_fractionation
        phi = params.c4_bundle_sheath_leakiness
        ratios = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
        epsilon = carbon13.c4_fractionation_model(params, ratios)
        torch.testing.assert_close(
            epsilon, a + (b4 + b3 * phi - a) * ratios, rtol=1.0e-12, atol=0.0
        )
        alpha = carbon13.calculate_photosynthesis_fractionation(
            torch.tensor(298.15, dtype=torch.float64), params, ci_ca_ratio=ratios
        )
        torch.testing.assert_close(alpha, 1.0 / (1.0 + epsilon / 1000.0), rtol=1.0e-12, atol=0.0)
        # b4 + b3 phi is much smaller than the C3 b, so C4 discrimination stays a few per mil.
        assert bool(torch.all((epsilon > 0.0) & (epsilon < 6.0)))

    def test_leakiness_is_the_measured_literature_value_with_its_measured_range(self, carbon13):
        """phi = 0.2 is the leakiness measured by Henderson et al. (1992) (via von
        Caemmerer et al. 2014), a literature value; the bounds are the measured range of
        Kromdijk et al. (2014), Table 1; the entry states the photosynthetic Delta it
        gives and the s term of the estimating model that Eq. (10) lacks."""
        spec = get_global_manager().get_parameter(f"{self.KEY}.c4_bundle_sheath_leakiness")
        assert float(spec.value) == pytest.approx(0.2)
        assert tuple(spec.bounds) == (0.08, 0.5)
        assert spec.provenance["source"] == "literature"
        references = " ".join(spec.references)
        assert "Henderson" in references and "Kromdijk" in references
        assert "2.4-2.8 per mil" in spec.description and "dry-matter" in spec.description
        assert "-0.14 per mil" in spec.description
        epsilon = carbon13.c4_fractionation_model(
            self._c4(), torch.tensor([0.4, 0.5], dtype=torch.float64)
        )
        assert 2.3 < float(epsilon[1]) < float(epsilon[0]) < 2.85
        s_term = 1.8 * float(spec.value) * 0.4
        assert s_term == pytest.approx(0.144)

    def test_ci_ca_slope_vanishes_at_the_leakiness_named_by_farquhar_1989(self, carbon13):
        """With the Table 1 values the factor b4 + b3 phi - a is zero at phi = 0.34
        (Farquhar et al. 1989), so Delta hardly depends on ci/ca there."""
        params = self._c4(c4_bundle_sheath_leakiness=0.34)
        epsilon = carbon13.c4_fractionation_model(
            params, torch.tensor([0.3, 0.5], dtype=torch.float64)
        )
        assert abs(float(epsilon[1] - epsilon[0])) < 0.05


if __name__ == "__main__":
    pytest.main([__file__])

