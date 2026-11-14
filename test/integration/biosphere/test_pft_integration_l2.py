"""
L2 Integration Tests for PFT Module

Tests integration of the PFT module with other biosphere components
and the unified parameter system. Verifies data flow between modules
and parameter consistency across the system.
"""

import pytest
import torch

from relic.scientific.biosphere.pft.pft_registry import PFTRegistry, PFTRegistryError
from relic.scientific.biosphere.pft.pft_types import PFTDefinition
from relic.parameters import get_global_manager
from relic.core.state import StateManager


class TestPFTConfigurationIntegration:
    """Test typed PFT configuration integration with system architecture."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()
        self.registry = PFTRegistry()
        self.device = torch.device("cpu")

    def test_pft_typed_definition_initialization(self):
        """Test typed PFT initialization from the registry."""
        pft_name = "broadleaf_deciduous_temperate"
        pft = self.registry.get_pft(pft_name)
        assert isinstance(pft, PFTDefinition)
        assert pft.name == pft_name
        assert pft.photosynthesis is not None
        assert pft.respiration is not None
        assert pft.allocation is not None

    def test_pft_typed_definition_completeness(self):
        """Test all required typed PFT parameters against their SSOT keys."""
        pft_name = "broadleaf_deciduous_temperate"
        pft = self.registry.get_pft(pft_name)
        namespace = f"biosphere.pft.{pft_name}"
        required = {
            f"{namespace}.photosynthesis.vcmax25": pft.photosynthesis.vcmax25,
            f"{namespace}.photosynthesis.jmax25": pft.photosynthesis.jmax25,
            f"{namespace}.photosynthesis.rd25": pft.photosynthesis.rd25,
            f"{namespace}.respiration.maintenance.rm_leaf_25": pft.respiration.maint_resp_leaf,
            f"{namespace}.allocation.f_leaf": pft.allocation.f_leaf,
            f"{namespace}.phenology.gdd_min": pft.phenology.gdd_min,
            f"{namespace}.bvoc.ef_isoprene": pft.bvoc.ef_isoprene,
            f"{namespace}.stress.water_stress_threshold": pft.stress.water_stress_threshold,
        }
        for key, typed_value in required.items():
            assert typed_value is not None, f"Missing typed value for {key}"
            assert typed_value == pytest.approx(
                self.param_manager.get(key), rel=1e-6, abs=1e-12
            )
            assert self.param_manager.get_parameter(key).units

    def test_pft_values_materialize_on_selected_device(self):
        """Test explicit device materialization outside simulator state."""
        pft_name = "needleleaf_evergreen_temperate"
        pft = self.registry.get_pft(pft_name)
        values = torch.as_tensor(
            [pft.photosynthesis.vcmax25, pft.allocation.f_leaf],
            dtype=torch.float32,
            device=self.device,
        )
        assert values.device == self.device
        assert values.dtype == torch.float32
        assert torch.all(torch.isfinite(values))

    def test_pft_allocation_fraction_constraint(self):
        """Test the static allocation-fraction configuration constraint."""
        pft_name = "c3_grass_temperate"
        allocation = self.registry.get_pft(pft_name).allocation
        fractions = torch.as_tensor(
            [allocation.f_leaf, allocation.f_stem, allocation.f_root],
            dtype=torch.float64,
        )
        assert torch.all(fractions >= 0)
        assert torch.allclose(fractions.sum(), torch.tensor(1.0, dtype=torch.float64))


class TestPFTRegistryIntegration:
    """Test PFT Registry integration with parameter system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()

    def test_registry_initialization(self):
        """Test that PFTRegistry initializes with all available PFTs."""
        registry = PFTRegistry()

        # Check registry is initialized
        assert registry is not None
        assert hasattr(registry, '_pft_definitions')

    def test_list_available_pfts(self):
        """Test listing all available PFT types."""
        registry = PFTRegistry()
        pft_list = registry.list_pfts()

        # Should have 15 PFT types
        assert len(pft_list) >= 15, f"Should have at least 15 PFTs, got {len(pft_list)}"

        # Check expected PFTs are present
        expected_pfts = [
            'broadleaf_deciduous_temperate',
            'broadleaf_evergreen_tropical',
            'needleleaf_evergreen_temperate',
            'c3_grass_temperate',
            'c4_grass_temperate'
        ]

        for expected_pft in expected_pfts:
            assert expected_pft in pft_list, f"Missing expected PFT: {expected_pft}"

    def test_get_pft_definition(self):
        """Test retrieving a specific PFT definition."""
        registry = PFTRegistry()

        # Get a specific PFT
        pft_name = "broadleaf_deciduous_temperate"
        pft_def = registry.get_pft(pft_name)

        # Check definition is complete
        assert pft_def is not None, f"Failed to get PFT: {pft_name}"
        assert pft_def.name == pft_name
        assert pft_def.photosynthesis is not None
        assert pft_def.phenology is not None
        assert pft_def.allocation is not None

        # Check parameters are loaded
        assert pft_def.photosynthesis.vcmax25 > 0
        assert pft_def.allocation.f_leaf > 0

    def test_registry_caching(self):
        """Test that PFT definitions are properly cached."""
        registry = PFTRegistry()

        # Get same PFT twice
        pft_name = "c3_grass_temperate"
        pft_def1 = registry.get_pft(pft_name)
        pft_def2 = registry.get_pft(pft_name)

        # Should return the same cached instance
        assert pft_def1 is pft_def2, "Should return cached PFT definition"

    def test_invalid_pft_handling(self):
        """Test handling of invalid PFT requests."""
        registry = PFTRegistry()

        # Try to get non-existent PFT
        with pytest.raises(PFTRegistryError):
            registry.get_pft("nonexistent_pft_type")


class TestPFTParameterConsistency:
    """Test parameter consistency across different system components."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()
        self.device = torch.device("cpu")

    def test_parameter_consistency_across_access_methods(self):
        """Test that parameters are consistent when accessed different ways."""
        pft_name = "broadleaf_evergreen_tropical"
        namespace = f"biosphere.pft.{pft_name}"

        vcmax_direct = self.param_manager.get(f"{namespace}.photosynthesis.vcmax25")
        registry = PFTRegistry()
        pft_def = registry.get_pft(pft_name)
        vcmax_registry = pft_def.photosynthesis.vcmax25

        assert vcmax_registry == pytest.approx(vcmax_direct, rel=1e-6, abs=1e-12)
        assert self.param_manager.get_parameter(
            f"{namespace}.photosynthesis.vcmax25"
        ).units == "umol CO2 m-2 s-1"

    def test_parameter_updates_propagation(self):
        """Test that parameter updates propagate correctly."""
        pft_name = "shrub_deciduous_temperate"
        namespace = f"biosphere.pft.{pft_name}"

        key = f"{namespace}.photosynthesis.vcmax25"
        original_value = self.param_manager.get(key)
        new_value = original_value * 1.1
        try:
            self.param_manager.set_value(key, new_value, notify=False)
            fresh_definition = PFTDefinition(
                name=pft_name,
                parameter_manager=self.param_manager,
            )
            assert fresh_definition.photosynthesis.vcmax25 == pytest.approx(
                new_value, rel=1e-6, abs=1e-12
            )
        finally:
            self.param_manager.set_value(key, original_value, notify=False)

    def test_multi_pft_simulation(self):
        """Test running multiple PFTs simultaneously."""
        pft_names = [
            "broadleaf_deciduous_temperate",
            "needleleaf_evergreen_temperate",
            "c3_grass_temperate"
        ]

        registry = PFTRegistry()
        definitions = [registry.get_pft(name) for name in pft_names]

        # Check each PFT has distinct parameters
        vcmax_values = [pft.photosynthesis.vcmax25 for pft in definitions]
        assert len(set(vcmax_values)) > 1, "Different PFTs should have different Vcmax25 values"

        # Check allocation fractions are valid for all PFTs
        for i, pft in enumerate(definitions):
            f_total = (
                pft.allocation.f_leaf
                + pft.allocation.f_stem
                + pft.allocation.f_root
            )
            assert abs(f_total - 1.0) < 0.01, \
                f"PFT {pft_names[i]} allocation fractions don't sum to 1.0: {f_total}"


class TestPFTStateManagerIntegration:
    """Test PFT integration with StateManager for transactional updates."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()
        self.registry = PFTRegistry()
        self.device = torch.device("cpu")
        self.state_manager = StateManager()

    def test_pft_parameters_remain_outside_state_manager(self):
        """Test that static PFT configuration is not emitted as simulator state."""
        # Initialize state manager with vegetation pools
        initial_state = {
            'leaf_carbon': torch.tensor(1.0, device=self.device),
            'root_carbon': torch.tensor(0.5, device=self.device),
            'stem_carbon': torch.tensor(2.0, device=self.device)
        }
        self.state_manager.initialize_state(initial_state)

        pft_name = "broadleaf_deciduous_temperate"
        pft = self.registry.get_pft(pft_name)
        current_state = self.state_manager.get_state()
        assert set(current_state) == set(initial_state)
        assert 'pft_vcmax25' not in current_state
        assert pft.photosynthesis.vcmax25 > 0
        assert pft.allocation.f_leaf > 0

    def test_pft_allocation_with_conservation(self):
        """Test that PFT allocation parameters maintain conservation in state updates."""
        # Initialize state
        initial_carbon = torch.tensor(10.0, device=self.device)
        initial_state = {
            'total_carbon': initial_carbon,
            'leaf_carbon': torch.tensor(0.0, device=self.device),
            'root_carbon': torch.tensor(0.0, device=self.device),
            'stem_carbon': torch.tensor(0.0, device=self.device)
        }
        self.state_manager.initialize_state(initial_state)

        pft_name = "c4_grass_temperate"
        pft = self.registry.get_pft(pft_name)
        fractions = torch.as_tensor(
            [pft.allocation.f_leaf, pft.allocation.f_root, pft.allocation.f_stem],
            dtype=initial_carbon.dtype,
            device=self.device,
        )

        # Calculate allocation increments (deltas) - IMPORTANT for conservation
        # We're moving carbon FROM total_carbon TO leaf/root/stem
        leaf_allocation = initial_carbon * fractions[0]
        root_allocation = initial_carbon * fractions[1]
        stem_allocation = initial_carbon * fractions[2]
        
        allocation_updates = {
            'leaf_carbon': leaf_allocation,  # Increment (was 0, now adding this)
            'root_carbon': root_allocation,  # Increment (was 0, now adding this)
            'stem_carbon': stem_allocation,  # Increment (was 0, now adding this)
            'total_carbon': -(leaf_allocation + root_allocation + stem_allocation)  # Decrement (removing allocated carbon)
        }

        # Apply updates with conservation checking disabled for this test
        # (or ensure increments sum to zero)
        self.state_manager.apply_updates(allocation_updates, check_conservation=False)

        # Check conservation
        state = self.state_manager.get_state()
        total_allocated = (state['leaf_carbon'] + state['root_carbon'] +
                          state['stem_carbon'] + state['total_carbon'])

        assert torch.allclose(total_allocated, initial_carbon, rtol=1e-5), \
            f"Carbon not conserved: initial={initial_carbon.item()}, final={total_allocated.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

