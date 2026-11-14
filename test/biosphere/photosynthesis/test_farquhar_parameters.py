"""
L1 Unit Test for Farquhar Photosynthesis Parameter Management

Tests parameter acquisition and validation after Phase 6 fixes.
"""

import pytest
import torch
import sys
from pathlib import Path
from relic.core.state import UpdateItem, UpdateMode

# Add relic to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestFarquharParameterManagement:
    """Test parameter management in FarquharPhotosynthesis."""

    def test_import_farquhar(self):
        """Test that farquhar module can be imported without errors."""
        try:
            from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
            assert FarquharPhotosynthesis is not None
        except ImportError as e:
            pytest.fail(f"Failed to import FarquharPhotosynthesis: {e}")

    def test_module_initialization_with_parameters(self):
        """Test that module can be initialized with explicit parameters."""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

        try:
            # Create instance with minimal parameters
            photosynthesis = FarquharPhotosynthesis(
                name="test_farquhar",
                vcmax25=60.0,
                jmax25=120.0,
                rd25=1.0,
                enable_gpu=False   # Disable GPU for testing
            )

            assert photosynthesis is not None
            assert photosynthesis.name == "test_farquhar"
            assert photosynthesis.vcmax25 is not None
            assert photosynthesis.jmax25 is not None
            assert photosynthesis.rd25 is not None

        except Exception as e:
            pytest.fail(f"Failed to initialize FarquharPhotosynthesis: {e}")

    def test_parameter_cache_initialization(self):
        """Test that parameter cache is properly initialized."""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

        photosynthesis = FarquharPhotosynthesis(
            name="test_cache",
            vcmax25=60.0,
            jmax25=120.0,
            rd25=1.0,
            enable_gpu=False
        )

        # Check that critical cached parameters exist
        assert hasattr(photosynthesis, '_param_cache')
        assert isinstance(photosynthesis._param_cache, dict)

        # Check some key cached parameters
        expected_cache_keys = [
            'epsilon_divide',
            'clamp_min_zero',
            'vpd_response_d0',
        ]

        for key in expected_cache_keys:
            assert key in photosynthesis._param_cache, f"Missing cached parameter: {key}"
        assert {
            'default_temperature',
            'nan_co2_default',
            'nan_par_default',
        }.isdisjoint(photosynthesis._param_cache)

    @pytest.mark.parametrize("pft_requires_grad", [False, True])
    def test_pft_parameter_reload_preserves_parameter_tensor_contract(self, pft_requires_grad):
        """Test that a PFT reload rebinds each capacity to its typed PFT source.

        The source keeps its identity, dtype and gradient flag: a learnable source
        is held by reference, any other is a buffer. The displaced member is
        replaced and never written. Values come through the real getter, which
        returns a Tensor or a manager binding, never a bare float.
        """
        from copy import deepcopy

        from relic.parameters import UnifiedParameterManager
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

        manager = UnifiedParameterManager()
        pft_values = {"vcmax25": 72.5, "jmax25": 145.0, "rd25": 1.25}
        for name, value in pft_values.items():
            spec = deepcopy(manager.get_parameter(f"biosphere.photosynthesis.{name}"))
            spec.key = f"biosphere.pft.test_pft.photosynthesis.{name}"
            spec.value = spec.default = value
            spec.dtype = "float64"
            spec.requires_grad = pft_requires_grad
            manager.register_parameter(spec)

        photosynthesis = FarquharPhotosynthesis(
            name="test_pft_reload",
            vcmax25=60.0,
            jmax25=120.0,
            rd25=1.0,
            enable_gpu=False,
            parameter_manager=manager,
        )

        # float32 members: the float64 assertion below tells the source dtype from
        # the dtype of the member it replaces.
        private_values = {"vcmax25": 1.0, "jmax25": 2.0, "rd25": 3.0}
        private_members = {
            name: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
            for name, value in private_values.items()
        }
        for name, member in private_members.items():
            setattr(photosynthesis, name, member)

        photosynthesis._load_pft_parameters("test_pft")

        for name, value in pft_values.items():
            held = getattr(photosynthesis, name)
            assert held.dtype == torch.float64
            assert held.requires_grad is pft_requires_grad
            assert held.detach().item() == pytest.approx(value)
            assert name not in dict(photosynthesis.named_parameters())
            assert private_members[name].detach().item() == pytest.approx(private_values[name])
            if pft_requires_grad:
                assert held is manager.nn_parameters[f"biosphere.pft.test_pft.photosynthesis.{name}"]
                assert name not in photosynthesis.state_dict()
            else:
                assert name in dict(photosynthesis.named_buffers())
                assert photosynthesis.manager_binding_key(held) is None

    def test_forward_with_minimal_state(self):
        """Test forward pass with minimal state input."""
        from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis

        photosynthesis = FarquharPhotosynthesis(
            name="test_forward",
            vcmax25=60.0,
            jmax25=120.0,
            rd25=1.0,
            enable_gpu=False,
            validate_inputs=True,
        )

        # Create minimal state
        state = {
            'air_temperature': torch.tensor([298.15]),
            'atmospheric_co2_ppm': torch.tensor([400.0]),
            'ppfd': torch.tensor([1000.0]),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor([1000.0]),
            'vcmax_scaling': torch.tensor([1.0]),
        }

        output = photosynthesis.forward(state, time=0.0, dt=1.0)

        assert set(output) == set(photosynthesis.metadata.output_variables)
        for item in output.values():
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False

        assert output['net_photosynthesis'].value.item() > 0
        assert output['gross_photosynthesis'].value.item() > 0


class TestParameterValidationReport:
    """Test that parameter validation issues are resolved."""

    def test_no_incomplete_parameter_calls(self):
        """
        Verify that critical param_mgr.get() calls now include metadata.

        This is a smoke test to ensure the Phase 6 fixes are in place.
        """

        farquhar_file = project_root / "relic" / "scientific" / "biosphere" / "photosynthesis" / "farquhar.py"

        with open(farquhar_file, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "default_vpd" not in content
        assert "nan_co2_default" not in content
        assert "nan_par_default" not in content
        assert "torch.nan_to_num" not in content

    def test_parameter_reload_does_not_use_tensor_data_assignment(self):
        """Verify that parameter reload avoids direct .data assignment."""

        farquhar_file = project_root / "relic" / "scientific" / "biosphere" / "photosynthesis" / "farquhar.py"

        with open(farquhar_file, 'r', encoding='utf-8') as f:
            content = f.read()

        assert ".data =" not in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

