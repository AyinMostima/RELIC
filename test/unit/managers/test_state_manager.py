"""
Unit tests for StateManager core behavior.

Focuses on StateManager API usage without historical manager consistency.
"""

import pytest
import torch
import sys
import os

# Add relic to path
sys.path.insert(0, os.path.abspath('.'))

from relic.core.state import (
    StateManager, 
    StateVariableSpec, 
    StateVariableType
)
from relic.core.state.conservation import ConservationSummary


class TestStateManagerV2Migration:
    """Test StateManager core behavior."""

    @pytest.mark.parametrize("initializer", ["initialize", "initialize_state"])
    def test_initialization_rejects_invalid_state_atomically(self, initializer):
        """Invalid initial values must not mutate any existing runtime ledger."""
        spec = StateVariableSpec(
            name="bounded_state",
            units="kg C m-2",
            variable_type=StateVariableType.SCALAR,
            bounds=(0.0, 100.0),
            conserved=True,
        )
        manager = StateManager(
            schema={"bounded_state": spec},
            autoload_contract_registry=False,
        )
        manager.set_state({"bounded_state": torch.tensor(25.0)})
        manager._history.append({"bounded_state": torch.tensor(20.0)})
        manager._current_diagnostics["sentinel"] = "preserve"
        manager._state_reset_counter = 3
        manager.set_conservation_ledger({"carbon": ["bounded_state"]})

        with pytest.raises(ValueError, match="bounded_state.*out of bounds"):
            getattr(manager, initializer)({"bounded_state": torch.tensor(150.0)})

        assert manager.get_state()["bounded_state"].item() == 25.0
        assert manager._history[0]["bounded_state"].item() == 20.0
        assert manager._current_diagnostics == {"sentinel": "preserve"}
        assert manager._state_reset_counter == 3
        assert manager._conservation_ledger == {"carbon": ["bounded_state"]}

    def test_transaction_savepoint_restores_append_only_ledgers_without_cloning_history(
        self,
    ):
        manager = StateManager(schema={}, autoload_contract_registry=False)
        manager.initialize_state({"pool": torch.tensor(1.0)})
        original_history_entry = {"pool": torch.tensor(0.5)}
        original_summary = ConservationSummary(total_violations=0, tolerance_value=1e-6)
        manager._history.append(original_history_entry)
        manager._conservation_history.append(original_summary)
        manager._current_diagnostics["status"] = "before"

        savepoint = manager.create_transaction_savepoint()

        assert savepoint["history_owner"] is manager._history
        assert savepoint["conservation_history_owner"] is manager._conservation_history
        assert savepoint["history_length"] == 1
        assert savepoint["conservation_history_length"] == 1

        manager.set_state("pool", torch.tensor(9.0), validate=False)
        manager._history.append({"pool": torch.tensor(9.0)})
        manager._conservation_history.append(ConservationSummary(total_violations=0, tolerance_value=1e-6))
        manager._current_diagnostics["status"] = "after"

        manager.restore_transaction_savepoint(savepoint)

        assert manager.get_state("pool").item() == pytest.approx(1.0)
        assert manager._history is savepoint["history_owner"]
        assert manager._history == [original_history_entry]
        assert manager._history[0] is original_history_entry
        assert manager._conservation_history == [original_summary]
        assert manager._conservation_history[0] is original_summary
        assert manager._current_diagnostics == {"status": "before"}

    def test_transaction_savepoint_rejects_foreign_payload(self):
        manager = StateManager(schema={}, autoload_contract_registry=False)

        with pytest.raises(ValueError, match="transaction savepoint"):
            manager.restore_transaction_savepoint({})

    def test_bounded_history_enforces_o1_memory_and_supports_rollback(self):
        """Configured max_history_length must cap history and conservation ledgers to O(1)."""
        spec = StateVariableSpec(
            name="carbon_pool",
            units="kg C m-2",
            variable_type=StateVariableType.SCALAR,
            bounds=(0.0, 10000.0),
            conserved=False,
        )
        manager = StateManager(
            schema={"carbon_pool": spec},
            autoload_contract_registry=False,
            max_history_length=4,
        )
        assert manager.max_history_length == 4
        manager.initialize_state({"carbon_pool": torch.tensor(0.0)})

        for step in range(25):
            manager.record_history_snapshot()
            manager.apply_updates(
                {"carbon_pool": torch.tensor(1.0)},
                dt=1800.0,
                check_conservation=True,
            )

        assert len(manager._history) == 4
        assert len(manager.conservation_history) == 4
        assert manager.get_state("carbon_pool").item() == pytest.approx(25.0)
        assert [snap["carbon_pool"].item() for snap in manager._history] == [
            21.0,
            22.0,
            23.0,
            24.0,
        ]

        restored = manager.rollback(steps=2)
        assert restored["carbon_pool"].item() == pytest.approx(23.0)
        assert manager.get_state("carbon_pool").item() == pytest.approx(23.0)
        assert len(manager._history) == 2

        manager.configure(max_history_length=1)
        assert manager.max_history_length == 1
        assert len(manager._history) == 1
        assert len(manager.conservation_history) == 1

        for invalid in (-1, True, "4", 2.5):
            with pytest.raises(ValueError, match="max_history_length"):
                manager.configure(max_history_length=invalid)

    def test_transaction_savepoint_restores_evicted_window_when_history_is_bounded(self):
        """Transaction savepoints must remain exact and O(1) when max_history_length evicts oldest entries."""
        manager = StateManager(
            schema={},
            autoload_contract_registry=False,
            max_history_length=2,
        )
        manager.initialize_state({"pool": torch.tensor(1.0)})
        first_entry = manager.record_history_snapshot()
        manager.apply_updates({"pool": torch.tensor(1.0)}, check_conservation=True)
        second_entry = manager.record_history_snapshot()
        manager.apply_updates({"pool": torch.tensor(1.0)}, check_conservation=True)

        assert len(manager._history) == 2
        assert len(manager._conservation_history) == 2
        pre_conservation_entries = list(manager._conservation_history)

        savepoint = manager.create_transaction_savepoint()
        history_owner = manager._history
        conservation_owner = manager._conservation_history

        manager.record_history_snapshot()
        manager.apply_updates({"pool": torch.tensor(5.0)}, check_conservation=True)
        assert len(manager._history) == 2
        assert manager._history[0] is second_entry

        manager.restore_transaction_savepoint(savepoint)

        assert manager.get_state("pool").item() == pytest.approx(3.0)
        assert manager._history is history_owner
        assert manager._conservation_history is conservation_owner
        assert manager._history == [first_entry, second_entry]
        assert manager._history[0] is first_entry
        assert manager._history[1] is second_entry
        assert manager._conservation_history == pre_conservation_entries


    def test_initialize_state_enforces_bounds_by_declaration_not_name(self):
        """A signed flux declares no storage bounds and keeps its sign; a
        declared bound is enforced whatever the variable is called."""
        signed = StateVariableSpec(
            name="signed_flux",
            units="kg m-2 s-1",
            variable_type=StateVariableType.SCALAR,
            bounds=None,
            conserved=False,
        )
        manager = StateManager(
            schema={"signed_flux": signed},
            autoload_contract_registry=False,
        )

        manager.initialize_state({"signed_flux": torch.tensor(-0.25)})

        assert manager.get_state()["signed_flux"].item() == pytest.approx(-0.25)
        assert manager.get_validation_errors() == []

        bounded = StateVariableSpec(
            name="signed_flux",
            units="kg m-2 s-1",
            variable_type=StateVariableType.SCALAR,
            bounds=(0.0, 1.0),
            conserved=False,
        )
        bounded_manager = StateManager(
            schema={"signed_flux": bounded},
            autoload_contract_registry=False,
        )
        with pytest.raises(ValueError, match="signed_flux out of bounds"):
            bounded_manager.initialize_state({"signed_flux": torch.tensor(-0.25)})

    def test_initialize_state_preserves_configured_conservation_ledger(self):
        """Runtime reinitialization must not erase the configured conservation ledger."""
        manager = StateManager(schema={}, autoload_contract_registry=False)
        manager.set_conservation_ledger({"carbon": ["carbon_pool"]})

        manager.initialize_state({"carbon_pool": torch.tensor(1.0)})

        assert manager.get_conservation_ledger() == {"carbon": ["carbon_pool"]}

    def test_physical_constraint_manager_is_read_only_owner_path(self):
        """The architecture-facing property must expose the injected owner exactly."""
        owner = object()
        manager = StateManager(
            schema={},
            autoload_contract_registry=False,
            physical_constraint_manager=owner,
        )

        assert manager.physical_constraint_manager is owner
        with pytest.raises(AttributeError):
            manager.physical_constraint_manager = object()

    def test_physical_constraint_manager_is_owned_per_state_manager(self):
        """StateManager instances must not share a hidden module singleton."""
        from relic.core.managers.base import ManagerState

        first = StateManager(schema={}, autoload_contract_registry=False)
        second = StateManager(schema={}, autoload_contract_registry=False)

        first_owner = first.physical_constraint_manager
        second_owner = second.physical_constraint_manager

        assert first_owner is first.physical_constraint_manager
        assert second_owner is second.physical_constraint_manager
        assert first_owner is not second_owner
        assert first_owner.state == ManagerState.READY
        assert second_owner.state == ManagerState.READY

    def test_core_api_surface(self):
        """Test that StateManager exposes core API methods."""
        manager = StateManager()

        assert hasattr(manager, "initialize")
        assert hasattr(manager, "configure")
        assert hasattr(manager, "validate")
        assert hasattr(manager, "register_variable")
        assert hasattr(manager, "set_state")
        assert hasattr(manager, "get_state")
        assert hasattr(manager, "apply_updates")

        print("[OK] StateManager core API surface available")
    def test_lifecycle_management(self):
        """Test StateManager lifecycle without historical manager state."""
        manager = StateManager()

        manager.initialize()
        config = {'conservation_tolerance': 1e-6}
        manager.configure(config)
        assert manager.conservation_tolerance == 1e-6

        errors = manager.validate()
        assert errors == [] or errors is None
        assert hasattr(manager, 'schema')
        assert isinstance(manager.schema, dict)
        assert len(manager.schema) > 0

        print("[OK] Lifecycle management working correctly")

    def test_state_variable_management(self):
        """Test core state variable functionality remains intact."""
        manager = StateManager()
        manager.initialize()
        manager.validate()
        
        # Test registering a new variable
        spec = StateVariableSpec(
            name="test_variable",
            units="dimensionless",
            variable_type=StateVariableType.SCALAR,
            bounds=(0.0, 100.0),
            default_value=10.0,
            description="Test variable"
        )
        manager.register_variable(spec)
        
        # Check variable is registered
        assert "test_variable" in manager.schema
        assert "test_variable" in manager.state_vars
        
        # Test setting and getting state (include required time variable)
        new_value = torch.tensor(50.0)
        manager.set_state({
            "test_variable": new_value,
            "time": torch.tensor(0.0)  # Required variable
        })
        state = manager.get_state(["test_variable"])
        assert torch.allclose(state["test_variable"], new_value)
        
        print("[OK] State variable management intact")
    
    def test_conservation_checking(self):
        """Test conservation checking functionality."""
        manager = StateManager()
        manager.initialize()
        manager.validate()

        for name in ("root_carbon", "soil_carbon"):
            manager.register_variable(
                StateVariableSpec(
                    name=name,
                    units="kg C m-2",
                    conserved=True,
                    required=False,
                    variable_type=StateVariableType.SCALAR,
                )
            )

        # Create some carbon pool updates
        updates = {
            'leaf_carbon': torch.tensor(10.0),
            'root_carbon': torch.tensor(-5.0),
            'soil_carbon': torch.tensor(-5.0)
        }
        
        # Check conservation
        violations = manager._check_conservation(updates)
        
        # Balanced updates should not trigger violations; if reported, they must be near zero
        assert all(abs(value) < 1e-6 for value in violations.values())
        
        print("[OK] Conservation checking functional")
    
    def test_validation_system(self):
        """Test state validation system."""
        manager = StateManager()
        manager.initialize()
        manager.validate()
        
        # Test with valid state (include required time variable)
        valid_state = {
            'air_temperature': torch.tensor(298.15),  # 25 C
            'soil_moisture': torch.tensor(0.3),
            'time': torch.tensor(0.0)  # Required variable
        }
        errors = manager.validate_state(valid_state)
        assert len(errors) == 0
        
        # Test with invalid state (negative temperature)
        invalid_state = {
            'air_temperature': torch.tensor(-10.0)  # Invalid negative Kelvin
        }
        errors = manager.validate_state(invalid_state)
        assert len(errors) > 0
        assert any('below minimum bound' in e for e in errors)
        
        print("[OK] Validation system working")
    
    def test_explicit_state_owners_do_not_share_mutations(self):
        """Independent StateManager owners must have isolated ledgers."""
        manager1 = StateManager(device=torch.device("cpu"))
        manager2 = StateManager(device=torch.device("cpu"))
        manager1.initialize_state({"temperature": torch.tensor(280.0)})
        manager2.initialize_state({"temperature": torch.tensor(290.0)})

        manager1.set_state("temperature", torch.tensor(281.0))

        assert manager1 is not manager2
        assert torch.equal(manager1.get_state("temperature"), torch.tensor(281.0))
        assert torch.equal(manager2.get_state("temperature"), torch.tensor(290.0))

    def test_device_awareness(self):
        """Test device-aware functionality."""
        # Test with CPU (always available)
        device = torch.device("cpu")
        manager = StateManager(device=device)
        manager.initialize()
        manager.validate()
        
        # All tensors should be on correct device
        for var_name, value in manager.state_vars.items():
            if isinstance(value, torch.Tensor):
                assert value.device == device
        
        print("[OK] Device awareness working")

def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("StateManager V2 Migration Tests")
    print("="*60 + "\n")
    
    test_suite = TestStateManagerV2Migration()
    
    tests = [
        test_suite.test_inheritance_from_base_manager,
        test_suite.test_lifecycle_management,
        test_suite.test_state_variable_management,
        test_suite.test_conservation_checking,
        test_suite.test_validation_system,
        test_suite.test_global_singleton_access,
        test_suite.test_registry_integration,
        test_suite.test_factory_creation,
        test_suite.test_device_awareness,
        test_suite.test_backward_consistency
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        test_name = test.__name__
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_name}: {e}")
            failed += 1
        except Exception as e:
            if "[SKIP]" in str(e):
                skipped += 1
            else:
                print(f"[ERROR] {test_name}: {e}")
                failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)


