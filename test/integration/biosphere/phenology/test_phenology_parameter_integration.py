"""L2 Integration Tests for Phenology Parameter System.

Tests integration between phenology modules and parameter management system.
"""

import pytest
import torch
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch

from relic.parameters import (
    UnifiedParameterManager,
    ParameterSpec,
)
from relic.parameters.types import ParameterCategory
from relic.scientific.biosphere.phenology.base import (
    PhenologyBase,
    PhenologyOutput,
    PhenologyPhase,
)
from relic.scientific.biosphere.phenology.parameters import (
    PhenologyParameters,
    PhenologyParameterSets,
)
from relic.scientific.biosphere.phenology.models import (
    StaticPhenology as StaticPhenologyFunctor,
    GDDPhenology as GDDPhenologyFunctor,
)


ROOT = Path(__file__).resolve().parents[4]
CANONICAL_SNAPSHOT = (
    ROOT / "relic" / "parameters" / "database" / "calibrated" / "saved_parameters.yaml"
)


@pytest.fixture(autouse=True)
def canonical_parameter_snapshot_guard():
    """Ensure phenology integration tests never persist into the source database."""
    snapshot_before = CANONICAL_SNAPSHOT.read_bytes()
    mtime_before = CANONICAL_SNAPSHOT.stat().st_mtime_ns
    yield
    assert CANONICAL_SNAPSHOT.read_bytes() == snapshot_before
    assert CANONICAL_SNAPSHOT.stat().st_mtime_ns == mtime_before


def _complete_test_spec(
    *,
    key: str,
    value: float,
    units: str,
    category: ParameterCategory,
    dimension: str,
) -> ParameterSpec:
    """Build one persistence-valid ParameterSpec for an isolated test database."""
    return ParameterSpec(
        key=key,
        value=value,
        units=units,
        dtype="float32",
        bounds=(-float("inf"), float("inf")),
        category=category,
        description=f"Isolated phenology integration parameter for {key}.",
        references=["test://phenology-parameter-integration"],
        tags={"test", "phenology"},
        dimension=dimension,
        provenance={
            "source": "test_fixture",
            "created_at": None,
            "schema_version": "1.0.0",
        },
        requires_grad=False,
    )


class _ParameterTestPhenology(PhenologyBase):
    """Concrete test harness for the abstract common phenology contract."""

    def compute_phenological_state(self, state, *, dt):
        del dt
        reference = torch.as_tensor(state["day_of_year"], dtype=torch.float32)
        zero = torch.zeros_like(reference)
        return PhenologyOutput(
            phenology_phase=PhenologyPhase.DORMANCY,
            growth_factor=zero,
            leaf_area_index=zero,
            senescence_rate=zero,
        )


class TestPhenologyParameterIntegration:
    """Test integration between phenology modules and parameter system."""
    
    @pytest.fixture
    def temp_param_db(self, fresh_global_parameter_manager):
        """Create temporary parameter database (on a fresh parameter manager)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "params"
            db_path.mkdir()
            
            # Create phenology parameters file
            phenology_file = db_path / "phenology.yaml"
            phenology_params = {
                'schema_version': '1.0.0',
                'tier': 'relic',
                'namespace': 'biosphere.phenology',
                'description': 'Test phenology parameters',
                'metadata': {
                    'version': '1.0.0',
                    'description': 'Test phenology parameters'
                },
                'parameters': {
                    'morphology': {
                        'min_lai': {'value': 0.1, 'units': '1', 'dtype': 'float32'},
                        'max_lai': {'value': 6.0, 'units': '1', 'dtype': 'float32'}
                    },
                    'static': {
                        'transition_steepness': {'value': 0.2, 'units': 'day-1', 'dtype': 'float32'},
                        'sigmoid_steepness': {'value': 4.0, 'units': 'dimensionless', 'dtype': 'float32'}
                    },
                    'gdd': {
                        'base_temp': {'value': 278.15, 'units': 'K', 'dtype': 'float32'}
                    }
                }
            }
            
            with open(phenology_file, 'w') as f:
                yaml.dump(phenology_params, f)
                
            yield db_path
            
    def test_phenology_base_parameter_loading(self, temp_param_db):
        """Test PhenologyBase loading parameters from manager."""
        # Initialize parameter manager with temp database
        manager = UnifiedParameterManager()
        manager.database.add_source(temp_param_db)
        manager.database.load_all()
        
        with patch('relic.parameters.manager.get_global_manager', return_value=manager):
            # Create phenology base instance
            phenology = _ParameterTestPhenology(name="test_phenology")
            
            # Check parameter loading
            min_lai = phenology.get_parameter_value(
                'morphology.min_lai',
                namespace='biosphere.phenology',
            )
            assert min_lai == 0.1

            max_lai = phenology.get_parameter_value(
                'morphology.max_lai',
                namespace='biosphere.phenology',
            )
            if torch.is_tensor(max_lai):
                max_lai_value = float(max_lai.detach().item())
            else:
                max_lai_value = float(max_lai)
            assert max_lai_value == 6.0
            
    def test_phenology_parameters_defaults(self, temp_param_db):
        """Test explicit value-object construction from manager-owned values."""
        values = PhenologyParameterSets.deciduous_tree_gsi().model_dump()
        manager = UnifiedParameterManager()
        manager.database.add_source(temp_param_db)
        manager.database.load_all()

        values.update(
            min_lai=manager.get("biosphere.phenology.morphology.min_lai"),
            max_lai=manager.get("biosphere.phenology.morphology.max_lai"),
        )
        with patch(
            "relic.scientific.biosphere.phenology.parameters.get_global_manager",
            side_effect=AssertionError("Value object accessed ParameterManager"),
        ):
            params = PhenologyParameters(**values)

        assert params.min_lai == 0.1
        assert params.max_lai == 6.0
            
    def test_static_phenology_functor_integration(self, temp_param_db):
        """Test StaticPhenologyFunctor parameter integration."""
        manager = UnifiedParameterManager()
        manager.database.add_source(temp_param_db)
        manager.database.load_all()
        
        with patch('relic.parameters.manager.get_global_manager', return_value=manager):
            functor = StaticPhenologyFunctor(name="static_phenology")

            # Test parameter access through functor
            steepness = functor.get_parameter_value('static.transition_steepness')
            if torch.is_tensor(steepness):
                steepness_value = float(steepness.detach().item())
            else:
                steepness_value = float(steepness)
            assert steepness_value == pytest.approx(0.2)
            
            # Test compute with loaded parameters
            state = {
                'leaf_carbon': torch.tensor(0.5),
                'froot_carbon': torch.tensor(0.3),
                'leaf_nitrogen': torch.tensor(0.02),
                'froot_nitrogen': torch.tensor(0.01),
            }
            
            environment = {
                'air_temperature': torch.tensor(293.15),
                'day_of_year': torch.tensor(150.0)
            }
            
            updates = functor(
                {**state, **environment},
                time=0.0,
                dt=1.0,
            )
            result = {key: update.value for key, update in updates.items()}
            assert 'phenological_factor' in result
            
    def test_gdd_phenology_functor_integration(self, temp_param_db):
        """Test GDDPhenologyFunctor parameter integration."""
        manager = UnifiedParameterManager()
        manager.database.add_source(temp_param_db)
        manager.database.load_all()
        
        with patch('relic.parameters.manager.get_global_manager', return_value=manager):
            functor = GDDPhenologyFunctor(name="gdd_phenology")
            
            # Test parameter access
            base_temp = functor.get_parameter_value('gdd.base_temp')
            assert base_temp == 278.15
            
    def test_parameter_override(self, temp_param_db, monkeypatch):
        """Test parameter override functionality."""
        manager = UnifiedParameterManager()
        monkeypatch.setattr(manager.database, "set_parameter", lambda spec, tier: None)
        manager.database.add_source(temp_param_db)
        manager.database.load_all()
        
        # Override a parameter
        spec = _complete_test_spec(
            key='biosphere.phenology.morphology.max_lai',
            value=10.0,
            units='dimensionless',
            category=ParameterCategory.BIOLOGICAL_PARAMETER,
            dimension='dimensionless',
        )
        manager.set_parameter(spec, tier="calibrated")
        
        with patch('relic.parameters.manager.get_global_manager', return_value=manager):
            phenology = _ParameterTestPhenology(name="test_override")
            max_lai = phenology.get_parameter_value(
                'morphology.max_lai',
                namespace='biosphere.phenology',
            )
            assert max_lai == 10.0
            
    def test_missing_parameter_handling(self):
        """Test handling of missing parameters."""
        manager = UnifiedParameterManager()
        
        with patch('relic.parameters.manager.get_global_manager', return_value=manager):
            phenology = _ParameterTestPhenology(name="test_missing")

            # Optional missing values remain absent without a hidden fallback.
            missing_param = phenology.get_parameter_value(
                'nonexistent.param',
                required=False,
                namespace='biosphere.phenology',
            )
            assert missing_param is None
            
            # Required parameter should raise error
            with pytest.raises(ValueError, match="Required parameter"):
                phenology.get_parameter_value(
                    'nonexistent.required',
                    required=True,
                    namespace='biosphere.phenology',
                )
                
    def test_parameter_audit(self, temp_param_db):
        """Test parameter audit functionality for phenology module."""
        manager = UnifiedParameterManager()
        manager.database.add_source(temp_param_db)
        manager.database.load_all()
        
        with patch('relic.parameters.manager.get_global_manager', return_value=manager):
            # Create phenology instances to trigger parameter requests
            phenology = _ParameterTestPhenology(name="test_audit")
            phenology.get_parameter_value(
                'morphology.min_lai',
                namespace='biosphere.phenology',
            )
            phenology.get_parameter_value(
                'missing.param',
                required=False,
                namespace='biosphere.phenology',
            )
            
            # Run audit
            audit_result = manager.run_registry_audit()
            
            # Check audit results
            assert audit_result.observed_total > 0
            missing = manager.get_missing_requests()
            
            # Should have recorded missing parameter request
            if 'biosphere.phenology.missing.param' in missing:
                assert missing['biosphere.phenology.missing.param'] >= 1
                
    def test_parameter_category_mapping(self, monkeypatch):
        """Test proper category mapping for phenology parameters."""
        manager = UnifiedParameterManager()
        monkeypatch.setattr(manager.database, "set_parameter", lambda spec, tier: None)
        
        # Add parameter with specific category
        spec = _complete_test_spec(
            key='biosphere.phenology.test.biological',
            value=1.0,
            units='dimensionless',
            category=ParameterCategory.BIOLOGICAL_PARAMETER,
            dimension='dimensionless',
        )
        manager.set_parameter(spec, tier="calibrated")
        
        # Retrieve and verify category
        param = manager.get_parameter('biosphere.phenology.test.biological')
        assert param is not None
        assert param.category == ParameterCategory.BIOLOGICAL_PARAMETER
        
    def test_concurrent_access(self, temp_param_db):
        """Test thread-safe concurrent parameter access."""
        import threading
        import time
        
        manager = UnifiedParameterManager()
        manager.database.add_source(temp_param_db)
        manager.database.load_all()
        
        results = []
        errors = []
        
        def access_parameters():
            try:
                phenology = _ParameterTestPhenology(
                    name=f"test_{threading.current_thread().ident}",
                    parameter_manager=manager,
                )
                for _ in range(10):
                    value = phenology.get_parameter_value(
                        'morphology.min_lai',
                        namespace='biosphere.phenology',
                    )
                    results.append(value)
                    time.sleep(0.001)  # Small delay to increase chance of concurrency
            except Exception as e:
                errors.append(e)
                
        # Create multiple threads
        threads = [threading.Thread(target=access_parameters) for _ in range(5)]
        
        # Start all threads
        for t in threads:
            t.start()
            
        # Wait for completion
        for t in threads:
            t.join()
            
        # Check results
        assert len(errors) == 0
        assert all(r == 0.1 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


