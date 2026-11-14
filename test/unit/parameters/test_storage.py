"""
L1 Unit Tests for Parameter Storage Component

Tests for the ParameterStorage class and related functionality.
"""

import pytest
import torch
from datetime import datetime
import threading

from relic.parameters.storage import (
    ParameterStorage,
    ParameterSnapshot,
    extract_namespace_hierarchy,
    merge_namespaces
)


class TestParameterStorage:
    """Test ParameterStorage functionality."""

    @pytest.mark.parametrize("empty", [False, True])
    def test_snapshot_can_capture_resolved_values_without_mutating_storage(self, empty):
        storage = ParameterStorage()
        storage.set("test.source", 1.0)
        resolved = {} if empty else {"resolved.parameter": torch.tensor(0.75)}
        snapshot = storage.snapshot("Resolved runtime values", parameters=resolved)
        assert set(snapshot.parameters) == set(resolved)
        assert storage.get_all() == {"test.source": 1.0}
        assert storage.get_statistics()["total_snapshots"] == 1
        assert snapshot.metadata["parameter_count"] == len(resolved)
        assert snapshot.metadata["namespace_count"] == (0 if empty else 1)
        if not empty:
            resolved["resolved.parameter"].fill_(9.0)
            assert snapshot.parameters["resolved.parameter"].item() == 0.75

    @pytest.mark.parametrize("empty", [False, True])
    def test_snapshot_keeps_stored_values_and_records_extra_metadata_without_aliasing(self, empty):
        """The stored values are captured as they are; extra metadata (for example the
        manager's live learnable values) is recorded apart and deep-copied."""
        storage = ParameterStorage()
        storage.set("test.source", 1.0)
        extra = {} if empty else {"nn_values": {"resolved.parameter": torch.tensor(0.75)}}
        snapshot = storage.snapshot("Stored values", metadata=extra)
        assert snapshot.parameters == {"test.source": 1.0}
        assert type(snapshot.parameters["test.source"]) is float
        assert storage.get_all() == {"test.source": 1.0}
        assert storage.get_statistics()["total_snapshots"] == 1
        assert snapshot.metadata["parameter_count"] == 1
        assert snapshot.metadata["namespace_count"] == 1
        if not empty:
            extra["nn_values"]["resolved.parameter"].fill_(9.0)
            assert snapshot.metadata["nn_values"]["resolved.parameter"].item() == 0.75

    def test_snapshot_tensor_projection_cannot_mutate_saved_value(self):
        storage = ParameterStorage()
        storage.set("test.source", torch.tensor([1.0, 2.0]))
        snapshot = storage.snapshot()
        exposed = snapshot.parameters
        exposed["test.source"].fill_(9.0)
        assert torch.equal(snapshot.parameters["test.source"], torch.tensor([1.0, 2.0]))

    @pytest.mark.parametrize("error", [ValueError, KeyError, RuntimeError])
    def test_restore_raises_a_callback_failure_and_keeps_the_previous_values(self, error):
        """A restore callback that rejects the snapshot fails the restore with its own
        error, and the storage keeps the values it held before the restore."""
        storage = ParameterStorage()
        storage.set("test.kept", 1.0)
        candidate = storage.snapshot("candidate", parameters={"test.kept": 2.0, "test.new": 3.0})

        def reject(snapshot):
            raise error("callback rejected the snapshot")

        storage.register_restore_callback(reject)
        with pytest.raises(error, match="callback rejected the snapshot"):
            storage.restore(candidate)
        assert storage.get_all() == {"test.kept": 1.0}

    def test_restored_storage_does_not_alias_a_reusable_snapshot(self):
        storage = ParameterStorage()
        storage.set("test.source", torch.tensor([1.0, 2.0]))
        snapshot = storage.snapshot()
        storage.restore(snapshot)
        storage.get("test.source").fill_(9.0)
        assert torch.equal(snapshot.parameters["test.source"], torch.tensor([1.0, 2.0]))
        storage.restore(snapshot)
        assert torch.equal(storage.get("test.source"), torch.tensor([1.0, 2.0]))
    
    @pytest.fixture
    def storage(self):
        """Create a fresh parameter storage instance."""
        return ParameterStorage()
    
    @pytest.fixture
    def sample_parameters(self):
        """Create sample parameters for testing."""
        return {
            "biosphere.photosynthesis.vcmax25": 60.0,
            "biosphere.photosynthesis.jmax25": 120.0,
            "atmosphere.radiation.albedo": 0.3,
            "pedosphere.decomposition.k_fast": 0.5,
            "pedosphere.decomposition.k_slow": 0.05,
            "coupling.tolerance.water_balance": 1e-6
        }
    
    def test_storage_initialization(self, storage):
        """Test storage initialization."""
        assert len(storage.get_all()) == 0
        assert storage.list() == []
        
        stats = storage.get_statistics()
        assert stats["total_parameters"] == 0
        assert stats["total_namespaces"] == 0
        assert stats["total_snapshots"] == 0
    
    def test_basic_crud_operations(self, storage):
        """Test Create, Read, Update, Delete operations."""
        # Create
        storage.set("test.param1", 42.0)
        assert storage.exists("test.param1")
        assert storage.get("test.param1") == 42.0
        
        # Read with default
        assert storage.get("nonexistent", default=10.0) == 10.0
        
        # Update
        storage.set("test.param1", 84.0)
        assert storage.get("test.param1") == 84.0
        
        # Delete
        assert storage.delete("test.param1") == True
        assert not storage.exists("test.param1")
        assert storage.delete("test.param1") == False  # Already deleted
    
    def test_bulk_operations(self, storage, sample_parameters):
        """Test bulk update operations."""
        # Bulk update
        storage.update(sample_parameters)
        
        # Verify all parameters were added
        assert len(storage.get_all()) == len(sample_parameters)
        for key, value in sample_parameters.items():
            assert storage.get(key) == value
        
        # Clear all
        storage.clear()
        assert len(storage.get_all()) == 0
    
    def test_namespace_listing_returns_every_key_below_the_namespace(self):
        """One rule for every namespace: all keys under '<namespace>.', never a sibling prefix."""
        storage = ParameterStorage()
        for key in ("a.b.c", "a.b.d.e", "a.bc.f", "a.x"):
            storage.set(key, 1.0)

        assert storage.list(namespace="a.b") == ["a.b.c", "a.b.d.e"]
        assert storage.list(namespace="a") == ["a.b.c", "a.b.d.e", "a.bc.f", "a.x"]
        assert storage.list(namespace="a.b.d") == ["a.b.d.e"]
        assert storage.list(namespace="a.b.c") == []
        storage.delete("a.b.c")
        assert storage.list(namespace="a.b") == ["a.b.d.e"]
        storage.delete("a.b.d.e")
        assert storage.list(namespace="a.b") == []
        assert storage.list(namespace="a") == ["a.bc.f", "a.x"]

    def test_namespace_operations(self, storage, sample_parameters):
        """Test namespace-based operations."""
        storage.update(sample_parameters)
        
        # List by namespace
        photosynthesis_params = storage.list(namespace="biosphere.photosynthesis")
        assert len(photosynthesis_params) == 2
        assert all("photosynthesis" in p for p in photosynthesis_params)
        
        decomposition_params = storage.list(namespace="pedosphere.decomposition")
        assert len(decomposition_params) == 2
        assert all("decomposition" in p for p in decomposition_params)
        
        # Non-existent namespace
        empty_params = storage.list(namespace="nonexistent.namespace")
        assert empty_params == []
    
    def test_pattern_matching(self, storage, sample_parameters):
        """Test pattern-based listing."""
        storage.update(sample_parameters)
        
        # Wildcard pattern matching
        vcmax_params = storage.list(pattern="*vcmax*")
        assert len(vcmax_params) == 1
        assert "vcmax25" in vcmax_params[0]
        
        k_params = storage.list(pattern="*k_*")
        assert len(k_params) == 2
        assert all("k_" in p for p in k_params)
        
        # Combined namespace and pattern
        biosphere_25_params = storage.list(
            namespace="biosphere",
            pattern="*25"
        )
        assert len(biosphere_25_params) == 2
        assert all("25" in p for p in biosphere_25_params)
    
    def test_tensor_storage(self, storage):
        """Test storing and retrieving PyTorch tensors."""
        # Store tensor
        tensor_val = torch.tensor([1.0, 2.0, 3.0])
        storage.set("tensor.param", tensor_val)
        
        # Retrieve tensor
        retrieved = storage.get("tensor.param")
        assert isinstance(retrieved, torch.Tensor)
        assert torch.allclose(retrieved, tensor_val)
        
        # Store gradient-enabled tensor
        grad_tensor = torch.tensor([1.0], requires_grad=True)
        storage.set("grad.param", grad_tensor)
        retrieved_grad = storage.get("grad.param")
        assert retrieved_grad.requires_grad
    
    def test_snapshot_operations(self, storage, sample_parameters):
        """Test snapshot creation and restoration."""
        # Add initial parameters
        storage.update(sample_parameters)
        initial_count = len(storage.get_all())
        
        # Create snapshot
        snapshot1 = storage.snapshot("Initial state")
        assert isinstance(snapshot1, ParameterSnapshot)
        assert snapshot1.description == "Initial state"
        assert len(snapshot1.parameters) == initial_count
        
        # Modify parameters
        storage.set("new.param", 100.0)
        storage.set("biosphere.photosynthesis.vcmax25", 70.0)
        storage.delete("atmosphere.radiation.albedo")
        
        # Create second snapshot
        snapshot2 = storage.snapshot("Modified state")
        
        # Restore first snapshot
        storage.restore(snapshot1)
        assert len(storage.get_all()) == initial_count
        assert storage.get("biosphere.photosynthesis.vcmax25") == 60.0
        assert storage.exists("atmosphere.radiation.albedo")
        assert not storage.exists("new.param")
    
    def test_diff_operations(self, storage, sample_parameters):
        """Test diff between snapshots and current state."""
        # Initial state
        storage.update(sample_parameters)
        snapshot1 = storage.snapshot("Initial")
        
        # Modify parameters
        storage.set("biosphere.photosynthesis.vcmax25", 70.0)  # Changed
        storage.set("new.param", 100.0)  # Added
        storage.delete("atmosphere.radiation.albedo")  # Removed
        
        # Diff with current state
        diff = storage.diff(snapshot1, None)
        
        # Check changed parameter
        assert "biosphere.photosynthesis.vcmax25" in diff
        assert diff["biosphere.photosynthesis.vcmax25"] == (60.0, 70.0)
        
        # Check added parameter
        assert "new.param" in diff
        assert diff["new.param"][0] is None
        assert diff["new.param"][1] == 100.0
        
        # Check removed parameter
        assert "atmosphere.radiation.albedo" in diff
        assert diff["atmosphere.radiation.albedo"][0] == 0.3
        assert diff["atmosphere.radiation.albedo"][1] is None
        
        # No changes should return empty diff
        snapshot2 = storage.snapshot("Current")
        diff_same = storage.diff(snapshot2, snapshot2)
        assert len(diff_same) == 0
    
    def test_thread_safety(self, storage):
        """Test thread-safe operations."""
        results = []
        errors = []
        
        def writer_thread(thread_id):
            """Write parameters from thread."""
            try:
                for i in range(100):
                    key = f"thread{thread_id}.param{i}"
                    storage.set(key, float(i * thread_id))
            except Exception as e:
                errors.append(e)
        
        def reader_thread(thread_id):
            """Read parameters from thread."""
            try:
                for i in range(100):
                    key = f"thread{thread_id % 3}.param{i}"
                    value = storage.get(key, default=-1)
                    results.append(value)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            writer = threading.Thread(target=writer_thread, args=(i,))
            reader = threading.Thread(target=reader_thread, args=(i,))
            threads.extend([writer, reader])
        
        # Run all threads
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0
        
        # Verify parameters were written
        all_params = storage.get_all()
        assert len(all_params) == 500  # 5 threads * 100 params each
    
    def test_statistics_tracking(self, storage):
        """Test access statistics tracking."""
        # Access parameters multiple times
        storage.set("freq.param1", 1.0)
        storage.set("freq.param2", 2.0)
        
        for _ in range(10):
            storage.get("freq.param1")
        for _ in range(5):
            storage.get("freq.param2")
        storage.get("freq.param3", default=0)  # Non-existent
        
        # Check statistics
        stats = storage.get_statistics()
        most_accessed = dict(stats["most_accessed"])
        
        assert most_accessed["freq.param1"] == 10
        assert most_accessed["freq.param2"] == 5
        assert most_accessed.get("freq.param3", 1) == 1
    
    def test_snapshot_immutability(self, storage):
        """Test that snapshots are immutable."""
        storage.set("test.param", 10.0)
        snapshot = storage.snapshot("Test snapshot")
        
        # Try to modify snapshot's parameters (should raise TypeError due to MappingProxyType)
        snapshot_params = snapshot.parameters  
        with pytest.raises(TypeError):
            snapshot_params["test.param"] = 20.0  # MappingProxyType prevents modification
        
        # Original snapshot should be unchanged
        assert snapshot.parameters["test.param"] == 10.0
        
        # Storage should also be unchanged
        assert storage.get("test.param") == 10.0
    
    def test_value_equality_comparison(self, storage):
        """Test _values_equal method with various types."""
        # Test with regular values
        assert storage._values_equal(1.0, 1.0)
        assert not storage._values_equal(1.0, 2.0)
        
        # Test with None
        assert not storage._values_equal(None, 1.0)
        assert not storage._values_equal(1.0, None)
        assert storage._values_equal(None, None) == True  # Both None are equal
        
        # Test with tensors
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([1.0, 2.0])
        t3 = torch.tensor([1.0, 3.0])
        assert storage._values_equal(t1, t2)
        assert not storage._values_equal(t1, t3)
        
        # Test with NaN
        nan1 = float('nan')
        nan2 = float('nan')
        assert storage._values_equal(nan1, nan2)
        assert not storage._values_equal(nan1, 1.0)
    
    def test_namespace_extraction(self):
        """Test namespace extraction from keys."""
        assert ParameterStorage._extract_namespace("a.b.c") == "a.b"
        assert ParameterStorage._extract_namespace("single") is None
        assert ParameterStorage._extract_namespace("two.parts") == "two"
        assert ParameterStorage._extract_namespace("deep.nested.param.name") == "deep.nested.param"


class TestHelperFunctions:
    """Test helper functions for namespace operations."""
    
    def test_extract_namespace_hierarchy(self):
        """Test namespace hierarchy extraction."""
        hierarchy = extract_namespace_hierarchy("a.b.c.d")
        assert hierarchy == ['a', 'a.b', 'a.b.c']
        
        hierarchy_single = extract_namespace_hierarchy("single")
        assert hierarchy_single == []
        
        hierarchy_two = extract_namespace_hierarchy("two.parts")
        assert hierarchy_two == ['two']
    
    def test_merge_namespaces(self):
        """Test namespace merging with precedence."""
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"b": 20, "c": 30, "d": 40}
        
        # Second takes precedence (default)
        merged_second = merge_namespaces(params1, params2)
        assert merged_second == {"a": 1, "b": 20, "c": 30, "d": 40}
        
        # First takes precedence
        merged_first = merge_namespaces(params1, params2, precedence="first")
        assert merged_first == {"a": 1, "b": 2, "c": 3, "d": 40}
        
        # Empty dicts
        assert merge_namespaces({}, params1) == params1
        assert merge_namespaces(params1, {}) == params1
        assert merge_namespaces({}, {}) == {}


class TestParameterSnapshot:
    """Test ParameterSnapshot dataclass."""
    
    def test_snapshot_creation(self):
        """Test creating a snapshot."""
        params = {"param1": 1.0, "param2": 2.0}
        metadata = {"version": "1.0.0"}
        
        snapshot = ParameterSnapshot(
            timestamp=datetime.now(),
            _parameters=params,
            metadata=metadata,
            description="Test snapshot"
        )
        
        assert snapshot.description == "Test snapshot"
        assert snapshot.parameters == params
        assert snapshot.metadata == metadata
        assert isinstance(snapshot.timestamp, datetime)
    
    def test_snapshot_immutability_via_deepcopy(self):
        """Test that snapshot performs deep copy in post_init."""
        original_params = {"param1": [1, 2, 3]}
        original_meta = {"list": [1, 2, 3]}
        
        snapshot = ParameterSnapshot(
            timestamp=datetime.now(),
            _parameters=original_params,
            metadata=original_meta,
            description="Deep copy test"
        )
        
        # Modify original dicts
        original_params["param1"].append(4)
        original_meta["list"].append(4)
        
        # Snapshot should be unchanged
        assert snapshot.parameters["param1"] == [1, 2, 3]
        assert snapshot.metadata["list"] == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

