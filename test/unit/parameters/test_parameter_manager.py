"""
L1 Unit Tests for RELIC Parameter Management System
Tests the core functionality of UnifiedParameterManager, ParameterSpec, and related components.
"""

import pytest
import torch
import tempfile
import yaml
from pathlib import Path

# Import parameter management system
from relic.parameters import (
    UnifiedParameterManager,
    ParameterSpec,
    ParameterCategory,
    get_global_manager,
    reset_global_manager,
)
from relic.parameters.spec import has_complete_parameter_metadata
from relic.parameters.serialization import export
from relic.parameters.validators.range_validator import RangeValidator, PhysicalConsistencyValidator
from relic.parameters.validators.unit_validator import UnitValidator


class TestParameterSpec:
    """Test ParameterSpec data model and validation."""

    @pytest.mark.parametrize("projection", ["to_dict", "model_dump", "mapping"])
    @pytest.mark.parametrize("field", ["default", "allowed_values"])
    def test_parameter_spec_roundtrip_preserves_default_and_categorical_contract(self, projection, field):
        spec = ParameterSpec(
            key="test.roundtrip_contract", value="wet", default="dry", dtype="str",
            allowed_values=["dry", "wet"], units="dimensionless", bounds=(-float("inf"), float("inf")), category="model_parameter",
        )
        payload = dict(spec) if projection == "mapping" else getattr(spec, projection)()
        restored = ParameterSpec.from_dict(payload)
        assert getattr(restored, field) == getattr(spec, field)
        assert restored.validate_value("wet") == []
        assert restored.validate_value("unknown")

    def test_parameter_spec_from_dict_keeps_explicit_default(self):
        spec = ParameterSpec.from_dict({
            "key": "test.explicit_default", "value": 2.0, "default": 0.0,
            "units": "1", "bounds": [-10.0, 10.0], "dtype": "float32", "category": "model_parameter",
        })
        assert spec.default == 0.0

    @pytest.mark.parametrize("field", ["value", "default"])
    @pytest.mark.parametrize("shape", [(), (1,), (2,)])
    def test_parameter_spec_tensor_roundtrip_preserves_dtype_shape_and_gradient_flag(self, field, shape):
        source = torch.full(shape, 1.0e-50, dtype=torch.float64, requires_grad=True)
        spec = ParameterSpec(key="test.tensor_contract", value=3.0, default=1.0, dtype="float64", units="dimensionless", bounds=(-float("inf"), float("inf")), category="model_parameter")
        setattr(spec, field, source)
        restored = ParameterSpec.from_dict(spec.to_dict())
        result = getattr(restored, field)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == source.dtype
        assert result.shape == source.shape
        assert result.requires_grad is True
        assert torch.equal(result, source)

    @pytest.mark.parametrize(
        "missing_field",
        ["key", "units", "dtype", "bounds", "category", "references", "provenance"],
    )
    def test_complete_parameter_metadata_requires_every_scientific_field(
        self,
        missing_field,
    ):
        payload = {
            "key": "test.strict_tolerance",
            "value": 1.0e-8,
            "category": "model_parameter",
            "units": "1",
            "dtype": "float64",
            "bounds": (0.0, 1.0e-6),
            "references": ("unit-test",),
            "provenance": {"source": "unit_test"},
        }
        payload.pop(missing_field)

        assert has_complete_parameter_metadata(payload) is False

    def test_complete_parameter_metadata_can_require_runtime_value(self):
        metadata_only = {
            "key": "test.strict_tolerance",
            "category": "model_parameter",
            "units": "1",
            "dtype": "float64",
            "bounds": (0.0, 1.0e-6),
            "references": ("unit-test",),
            "provenance": {"source": "unit_test"},
        }

        assert has_complete_parameter_metadata(metadata_only) is True
        assert (
            has_complete_parameter_metadata(metadata_only, require_value=True)
            is False
        )

        spec = ParameterSpec(value=1.0e-8, **metadata_only)
        assert has_complete_parameter_metadata(spec, require_value=True) is True

    def test_parameter_spec_creation(self):
        """Test creating a valid ParameterSpec."""
        spec = ParameterSpec(
            key="test.parameter",
            value=25.0,
            units="celsius",
            dtype="float32",
            bounds=(0.0, 100.0),
            category=ParameterCategory.BIOPHYSICAL,
            description="Test temperature parameter",
            references=["Test et al. 2023"],
            tags={'temperature', 'test'},
            dimension="temperature",
            provenance={'source': 'test', 'version': '1.0.0'},
            requires_grad=False
        )

        assert spec.key == "test.parameter"
        assert spec.value == 25.0
        assert spec.units == "celsius"
        assert spec.dtype == "float32"
        assert spec.bounds == (0.0, 100.0)
        assert spec.category == ParameterCategory.BIOPHYSICAL
        assert "Test et al. 2023" in spec.references
        assert 'temperature' in spec.tags
        assert spec.dimension == "temperature"
        assert spec.provenance['source'] == 'test'
        assert not spec.requires_grad

    def test_parameter_spec_normalizes_direct_string_category(self):
        """Direct construction must store the canonical category enum."""
        spec = ParameterSpec(
            key="test.direct_category",
            value=0.5,
            units="dimensionless",
            dtype="float32",
            bounds=(0.0, 1.0),
            category="biophysical",
            description="Direct category normalization test",
            references=["RELIC parameter category normalization test"],
            tags={"test"},
            dimension="dimensionless",
            provenance={"source": "test", "version": "1.0.0"},
            requires_grad=False,
        )

        assert spec.category is ParameterCategory.BIOPHYSICAL
        assert spec.to_dict()["category"] == ParameterCategory.BIOPHYSICAL.value

    def test_parameter_spec_rejects_non_string_category(self):
        """Direct construction must reject non-string category objects."""
        with pytest.raises(TypeError, match="Parameter category"):
            ParameterSpec(
                key="test.invalid_category_type",
                value=0.5,
                category=object(), units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf")),
            )

    def test_parameter_spec_validation(self):
        """Test ParameterSpec field validation."""
        # Test invalid bounds (min > max)
        with pytest.raises(ValueError, match="bounds"):
            ParameterSpec(
                key="invalid.bounds",
                value=50.0,
                bounds=(100.0, 0.0),  # Invalid: min > max

                category=ParameterCategory.BIOPHYSICAL,
                description="Invalid bounds test", units="dimensionless", dtype="float32"
            )

        # Note: Value bounds checking is performed by validators at runtime,
        # not during ParameterSpec construction

        # Test invalid key format
        with pytest.raises(ValueError, match="key"):
            ParameterSpec(
                key="",  # Empty key
                value=25.0,
                category=ParameterCategory.BIOPHYSICAL,
                description="Empty key test", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            )

    def test_parameter_spec_tensor_conversion(self):
        """Test conversion to/from PyTorch tensors."""
        spec = ParameterSpec(
            key="tensor.test",
            value=42.0,
            dtype="float32",
            category=ParameterCategory.BIOPHYSICAL,
            description="Tensor conversion test", units="dimensionless", bounds=(-float("inf"), float("inf"))
        )

        # Convert to tensor
        tensor = spec.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.item() == 42.0

        # Convert from tensor
        new_tensor = torch.tensor(84.0, dtype=torch.float64)
        spec.from_tensor(new_tensor)
        assert spec.value == 84.0
        assert spec.dtype == "float64"  # Should update dtype

    @pytest.mark.parametrize("stored_requires_grad", [False, True])
    def test_to_tensor_never_mutates_the_stored_value(self, stored_requires_grad):
        """A shallow copy shares the stored tensor; converting it must not change that tensor."""
        from copy import copy

        stored = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=stored_requires_grad)
        spec = ParameterSpec(
            key="tensor.alias_probe",
            value=stored,
            dtype="float64",
            requires_grad=stored_requires_grad,
            category=ParameterCategory.BIOPHYSICAL,
            description="Aliasing probe", units="dimensionless", bounds=(-float("inf"), float("inf")),
        )
        trainable = copy(spec)
        trainable.requires_grad = not stored_requires_grad

        converted = trainable.to_tensor()

        assert converted.requires_grad is (not stored_requires_grad)
        assert stored.requires_grad is stored_requires_grad
        assert spec.value is stored
        torch.testing.assert_close(converted.detach(), stored.detach())
        # The conversion keeps the stored storage (no data copy on the same device).
        assert converted.data_ptr() == stored.data_ptr()

    def test_to_tensor_of_a_trainable_leaf_returns_the_leaf_itself(self):
        stored = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        spec = ParameterSpec(
            key="tensor.leaf_probe",
            value=stored,
            dtype="float64",
            requires_grad=True,
            category=ParameterCategory.BIOPHYSICAL,
            description="Trainable leaf probe", units="dimensionless", bounds=(-float("inf"), float("inf")),
        )
        assert spec.to_tensor() is stored

    def test_parameter_spec_serialization(self):
        """Test JSON serialization/deserialization."""
        spec = ParameterSpec(
            key="serialization.test",
            value=3.14159,
            units="radians",
            category=ParameterCategory.COMPUTATIONAL,
            description="Serialization test",
            tags={'pi', 'math'}, dtype="float32", bounds=(-float("inf"), float("inf"))
        )

        # Serialize to dict
        spec_dict = spec.model_dump()
        assert spec_dict['key'] == "serialization.test"
        assert spec_dict['value'] == 3.14159
        assert spec_dict['units'] == "radians"

        # Deserialize from dict
        restored_spec = ParameterSpec(**spec_dict)
        assert restored_spec.key == spec.key
        assert restored_spec.value == spec.value
        assert restored_spec.units == spec.units
        assert restored_spec.tags == spec.tags


def test_set_parameter_takes_the_tier_from_the_caller():
    """The tier decides precedence; no call writes to a default tier."""
    import inspect

    from relic.parameters.manager import UnifiedParameterManager

    parameter = inspect.signature(UnifiedParameterManager.set_parameter).parameters["tier"]
    assert parameter.default is inspect.Parameter.empty


class TestUnifiedParameterManager:
    """Test UnifiedParameterManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create a fresh parameter manager for testing."""
        mgr = UnifiedParameterManager()
        # Clear any existing parameters for clean test state
        mgr.clear_parameters()
        return mgr

    @pytest.fixture
    def sample_spec(self):
        """Create a sample parameter specification."""
        return ParameterSpec(
            key="sample.parameter",
            value=10.0,
            units="meters",
            category=ParameterCategory.PHYSICAL,
            description="Sample parameter for testing", dtype="float32", bounds=(-float("inf"), float("inf"))
        )

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.get_all_parameters()) == 0
        assert manager.device.type in ['cpu', 'cuda']  # Should have valid device

        # Test statistics
        stats = manager.get_statistics()
        assert stats['total_parameters'] == 0
        assert stats['by_category'] == {}
        assert stats['with_gradients'] == 0

    def test_parameter_registration(self, manager, sample_spec):
        """Test parameter registration."""
        # Register parameter
        manager.register_parameter(sample_spec)

        # Verify registration
        assert manager.has_parameter(sample_spec.key)
        assert len(manager.get_all_parameters()) == 1

        # Get parameter value
        value = manager.get(sample_spec.key)
        assert value == sample_spec.value

        # Test duplicate registration (should update)
        updated_spec = ParameterSpec(
            key=sample_spec.key,
            value=20.0,  # Different value
            units="meters",
            category=ParameterCategory.PHYSICAL,
            description="Updated parameter", dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(updated_spec)
        assert manager.get(sample_spec.key) == 20.0

    def test_parameter_metadata_uses_one_public_entrypoint(self, manager, sample_spec):
        """Expose metadata only through get_parameter()."""
        manager.register_parameter(sample_spec)

        assert manager.get_parameter(sample_spec.key) is not None
        assert manager.get_parameter("parameter", namespace="sample") is not None
        assert not hasattr(manager, "get_spec")
        assert not hasattr(manager, "get_parameter_spec")
        assert not hasattr(manager, "clear")
        assert not hasattr(manager, "load_database")
        assert not hasattr(manager, "get_tensor")
        assert not hasattr(manager, "get_as_tensor")
        assert not hasattr(manager, "invalidate_cache")

    def test_parameter_operations(self, manager):
        """Test parameter get/set operations."""
        # Test getting non-existent parameter with default
        value = manager.get("nonexistent.param", default=42.0)
        assert value == 42.0

        # Test setting parameter value
        spec = ParameterSpec(
            key="operations.test",
            value=100.0,
            category=ParameterCategory.COMPUTATIONAL,
            description="Operations test", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(spec)

        # Set new value
        manager.set_value("operations.test", 200.0)
        assert manager.get("operations.test") == 200.0

        # Test setting with validation
        spec_with_bounds = ParameterSpec(
            key="bounded.test",
            value=50.0,
            bounds=(0.0, 100.0),
            category=ParameterCategory.BIOPHYSICAL,
            description="Bounded parameter test", units="dimensionless", dtype="float32"
        )
        manager.register_parameter(spec_with_bounds)

        # Valid set
        manager.set_value("bounded.test", 75.0)
        assert manager.get("bounded.test") == 75.0

        # Invalid set should be clamped rather than raising
        with pytest.warns(UserWarning):
            manager.set_value("bounded.test", 150.0, validate=True)
        assert manager.get("bounded.test") == 100.0

    @pytest.mark.parametrize(
        ("bounds", "requested", "violated_side"),
        [
            # float32(0.1) = 0.10000000149011612 lies above the float64 bound 0.1
            pytest.param((0.001, 0.1), 3.83, 1, id="upper_0.1_float32_form_above"),
            # float32(0.3) = 0.30000001192092896 lies above the float64 bound 0.3
            pytest.param((0.0, 0.3), 5.0, 1, id="upper_0.3_float32_form_above"),
            # float32(1.0e-6) = 9.999999974752427e-07 lies below the float64 bound 1.0e-6
            pytest.param((1.0e-6, 1.0), 0.0, 0, id="lower_1e-6_float32_form_below"),
            # the same out-of-bounds request as a float32 tensor
            pytest.param((0.001, 0.1), torch.tensor(3.83, dtype=torch.float32), 1, id="upper_0.1_float32_tensor"),
            # controls: 2.0 is exact in float32; float32(0.7) = 0.699999988079071 lies inside
            pytest.param((-2.0, 2.0), 2.5, 1, id="upper_2.0_float32_exact"),
            pytest.param((0.0, 0.7), 5.0, 1, id="upper_0.7_float32_form_inside"),
        ],
    )
    def test_out_of_bounds_write_clamps_for_any_float32_form_of_the_bound(
        self, manager, bounds, requested, violated_side
    ):
        """WARN_AND_CLAMP must not depend on how float32 rounds the violated bound.

        The stored value is the float32 value nearest to the violated bound that
        lies inside the declared interval: it is a float32 value, it satisfies
        the interval exactly, and its float32 neighbour on the side of the
        violated bound lies outside. The write warns once and the clamp event
        records the stored value.
        """
        key = "clamp_probe.float32_bound"
        manager.register_parameter(
            ParameterSpec(
                key=key,
                value=0.5 * (bounds[0] + bounds[1]),
                units="1",
                dtype="float32",
                bounds=bounds,
                category=ParameterCategory.COMPUTATIONAL,
                description="Out-of-bounds write on a float32 parameter",
            )
        )

        with pytest.warns(UserWarning, match="clamped") as record:
            manager.set_value(key, requested, validate=True, notify=False)

        assert len([item for item in record if "clamped" in str(item.message)]) == 1
        stored = manager.storage.get(key)
        if isinstance(requested, torch.Tensor):
            assert isinstance(stored, torch.Tensor) and stored.dtype == requested.dtype
        else:
            assert isinstance(stored, float)
        stored_value = float(stored)
        lower, upper = bounds
        as_float32 = torch.tensor(stored_value, dtype=torch.float32)
        assert float(as_float32) == stored_value
        assert lower <= stored_value <= upper
        outward = torch.tensor(float("inf") if violated_side == 1 else float("-inf"), dtype=torch.float32)
        assert not lower <= float(torch.nextafter(as_float32, outward)) <= upper
        assert manager.get_parameter(key).validate_value(stored) == []
        events = manager.get_clamp_events()
        assert len(events) == 1
        assert events[0]["clamped_value"] == stored_value

    def test_out_of_bounds_write_keeps_the_bound_when_no_float32_value_lies_inside(self, manager):
        """An interval that holds no float32 value clamps a scalar to the declared bound.

        float32(0.1) = 0.10000000149011612 and its lower neighbour
        0.09999999403953552 both lie outside [0.1, 0.1]; the declared bound is
        the only admissible value.
        """
        key = "clamp_probe.float32_fixed"
        manager.register_parameter(
            ParameterSpec(
                key=key,
                value=0.1,
                units="1",
                dtype="float32",
                bounds=(0.1, 0.1),
                category=ParameterCategory.COMPUTATIONAL,
                description="Out-of-bounds write on a fixed float32 parameter",
            )
        )

        with pytest.warns(UserWarning, match="clamped"):
            manager.set_value(key, 0.2, validate=True, notify=False)

        assert manager.storage.get(key) == 0.1
        events = manager.get_clamp_events()
        assert len(events) == 1
        assert events[0]["clamped_value"] == 0.1

    @pytest.mark.parametrize(
        ("bounds", "requested"),
        [
            pytest.param((0.001, 0.1), float("nan"), id="nan_float"),
            pytest.param((0.001, 0.1), torch.tensor(float("nan"), dtype=torch.float32), id="nan_float32_tensor"),
            pytest.param((-float("inf"), float("inf")), float("nan"), id="nan_float_open_bounds"),
            pytest.param((0.001, 0.1), float("inf"), id="pos_inf_float"),
            pytest.param((0.001, 0.1), -float("inf"), id="neg_inf_float"),
            pytest.param((0.001, 0.1), torch.tensor(float("inf"), dtype=torch.float32), id="pos_inf_float32_tensor"),
            pytest.param((0.001, 0.1), torch.tensor(-float("inf"), dtype=torch.float32), id="neg_inf_float32_tensor"),
        ],
    )
    def test_non_finite_write_is_rejected_and_never_clamped(self, manager, recwarn, bounds, requested):
        """A non-finite request is rejected under WARN_AND_CLAMP, never clamped.

        NaN fails every order comparison, so it has no nearest admissible value,
        and an infinite request reports an upstream overflow rather than a value
        near a bound; the bounds clamp accepts neither as clampable.
        set_value raises ValueError, the stored value is unchanged, and no clamp
        warning or clamp event is emitted.
        """
        key = "clamp_probe.non_finite"
        manager.register_parameter(
            ParameterSpec(
                key=key,
                value=0.05,
                units="1",
                dtype="float32",
                bounds=bounds,
                category=ParameterCategory.COMPUTATIONAL,
                description="Non-finite write on a float32 parameter",
            )
        )

        with pytest.raises(ValueError, match="clamp_probe.non_finite"):
            manager.set_value(key, requested, validate=True, notify=False)

        assert manager.storage.get(key) == 0.05
        assert not [item for item in recwarn if "clamped" in str(item.message)]
        assert manager.get_clamp_events() == []

    @pytest.mark.parametrize(
        ("bounds", "value"),
        [
            pytest.param((0.001, 0.1), float("nan"), id="nan_float"),
            pytest.param((-float("inf"), float("inf")), float("nan"), id="nan_float_open_bounds"),
            pytest.param((0.001, 0.1), torch.tensor(float("nan"), dtype=torch.float32), id="nan_float32_tensor"),
            pytest.param((0.001, 0.1), float("inf"), id="pos_inf_float"),
            pytest.param((0.001, 0.1), -float("inf"), id="neg_inf_float"),
            pytest.param((-float("inf"), float("inf")), float("inf"), id="pos_inf_float_open_bounds"),
            pytest.param((-float("inf"), float("inf")), -float("inf"), id="neg_inf_float_open_bounds"),
            pytest.param((0.001, 0.1), torch.tensor(float("inf"), dtype=torch.float32), id="pos_inf_float32_tensor"),
        ],
    )
    def test_non_finite_registration_is_rejected(self, manager, bounds, value):
        """register_parameter rejects a non-finite value at the default validation level.

        A specification whose value is NaN or +/-inf is rejected under the
        default level (STANDARD), as a validated set_value rejects such a
        write: NaN has no position relative to the declared interval and an
        infinite value reports an upstream overflow, with finite or open
        bounds alike. register_parameter raises ValueError naming the key and
        nothing is stored under it. The key contains none of the substrings
        "nan", "inf" or "type", so the rejection does not depend on the key.
        """
        key = "clamp_probe.reg_non_finite"
        spec = ParameterSpec(
            key=key,
            value=value,
            units="1",
            dtype="float32",
            bounds=bounds,
            category=ParameterCategory.COMPUTATIONAL,
            description="Non-finite registration on a float32 parameter",
        )

        with pytest.raises(ValueError, match="clamp_probe.reg_non_finite"):
            manager.register_parameter(spec)

        assert not manager.has_parameter(key)
        assert manager.storage.get(key) is None

    def test_restore_synchronizes_specs_storage_and_caches(self, manager):
        """Restore must make every parameter owner expose the snapshotted value."""
        spec = ParameterSpec(
            key="snapshot.restore.test",
            value=1.0,
            units="1",
            bounds=(0.0, 3.0),
            category=ParameterCategory.COMPUTATIONAL,
            description="Snapshot restore consistency test", dtype="float32",
        )
        manager.register_parameter(spec)
        snapshot = manager.snapshot()

        manager.set_value(spec.key, 2.0)
        assert manager.get(spec.key, expected_type=float) == 2.0

        manager.restore(snapshot)

        assert manager.get(spec.key, expected_type=float) == 1.0
        assert manager.storage.get(spec.key) == 1.0
        restored_spec = manager.get_parameter(spec.key)
        assert restored_spec is not None
        assert restored_spec.value == 1.0

    @pytest.mark.parametrize("snapshot_kind", ["full", "values"])
    def test_snapshot_captures_optimizer_value_without_mutating_live_owners(self, manager, snapshot_kind):
        spec = ParameterSpec(key="test.optimized_snapshot", value=1.0, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        manager.register_parameter(spec)
        parameter = manager.nn_parameters[spec.key]
        optimizer = torch.optim.SGD([parameter], lr=0.25)
        parameter.sum().backward()
        optimizer.step()
        captured = manager.snapshot() if snapshot_kind == "full" else manager.create_snapshot()
        # The spec and stored values are captured exactly (a float stays a float);
        # the live optimizer value of the learnable binding is captured apart.
        if snapshot_kind == "full":
            spec_value = ParameterSpec.from_dict(captured["parameters"][spec.key]).value
            stored_value = captured["storage_snapshot"].parameters[spec.key]
            live_value = captured["nn_values"][spec.key]
        else:
            spec_value = stored_value = captured.parameters[spec.key]
            live_value = captured.metadata["nn_values"][spec.key]
        assert type(spec_value) is float and spec_value == 1.0
        assert type(stored_value) is float and stored_value == 1.0
        assert isinstance(live_value, torch.Tensor) and live_value.item() == 0.75
        assert manager.parameters[spec.key].value == 1.0
        assert manager.storage.get(spec.key) == 1.0
        assert manager.nn_parameters[spec.key] is parameter
        assert optimizer.param_groups[0]["params"][0] is parameter

    @pytest.mark.parametrize("snapshot_kind", ["full", "values"])
    def test_restoring_optimizer_snapshot_preserves_parameter_identity_and_next_step(self, manager, snapshot_kind):
        spec = ParameterSpec(key="test.optimizer_restore", value=1.0, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        manager.register_parameter(spec)
        parameter = manager.nn_parameters[spec.key]
        optimizer = torch.optim.SGD([parameter], lr=0.25)
        captured = manager.snapshot() if snapshot_kind == "full" else manager.create_snapshot()
        parameter.sum().backward()
        optimizer.step()
        if snapshot_kind == "full":
            manager.restore(captured)
        else:
            manager.restore_snapshot(captured)
        assert manager.nn_parameters[spec.key] is parameter
        assert parameter.item() == 1.0
        optimizer.zero_grad()
        parameter.sum().backward()
        optimizer.step()
        assert manager.get(spec.key, expected_type=torch.Tensor).item() == 0.75

    def test_full_snapshot_restores_frozen_flag_without_replacing_parameter(self, manager):
        spec = ParameterSpec(key="test.frozen_snapshot", value=1.0, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        manager.register_parameter(spec)
        parameter = manager.nn_parameters[spec.key]
        manager.freeze([spec.key])
        captured = manager.snapshot()
        manager.unfreeze([spec.key])
        manager.restore(captured)
        assert manager.nn_parameters[spec.key] is parameter
        assert parameter.requires_grad is False

    @pytest.mark.parametrize("invalid", ["dtype", "shape", "flags"])
    def test_incompatible_optimizer_restore_rejects_before_mutating_owners(self, manager, invalid):
        spec = ParameterSpec(
            key="test.incompatible_snapshot", value=torch.tensor([1.0], dtype=torch.float32),
            units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter",
        )
        manager.register_parameter(spec)
        parameter = manager.nn_parameters[spec.key]
        captured = manager.snapshot()
        if invalid == "dtype":
            captured["parameters"][spec.key]["dtype"] = "float64"
        elif invalid == "shape":
            captured["parameters"][spec.key]["value"].update(data=[1.0, 2.0], shape=[2])
        else:
            captured["nn_requires_grad"][spec.key] = "false"
        with pytest.raises(ValueError, match="binding|flags|conflicting values"):
            manager.restore(captured)
        assert manager.parameters[spec.key] is spec
        assert manager.nn_parameters[spec.key] is parameter
        assert parameter.item() == 1.0
        assert manager.storage.get(spec.key).item() == 1.0

    def test_diff_of_current_optimizer_snapshot_is_empty_without_new_history(self, manager):
        spec = ParameterSpec(key="test.live_diff", value=1.0, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        manager.register_parameter(spec)
        parameter = manager.nn_parameters[spec.key]
        optimizer = torch.optim.SGD([parameter], lr=0.25)
        parameter.sum().backward()
        optimizer.step()
        captured = manager.create_snapshot()
        history_count = manager.storage.get_statistics()["total_snapshots"]
        assert manager.get_diff(captured) == {}
        assert manager.storage.get_statistics()["total_snapshots"] == history_count

    @pytest.mark.parametrize("snapshot_kind", ["full", "values"])
    def test_diffs_report_live_optimizer_change_against_earlier_snapshot(self, manager, snapshot_kind):
        """A learnable value moved by an optimizer after the snapshot is a change in both
        diff APIs, although its declared spec value and its stored value stay unchanged."""
        spec = ParameterSpec(key="test.live_change_diff", value=1.0, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        manager.register_parameter(spec)
        parameter = manager.nn_parameters[spec.key]
        optimizer = torch.optim.SGD([parameter], lr=0.25)
        captured = manager.snapshot() if snapshot_kind == "full" else manager.create_snapshot()
        parameter.sum().backward()
        optimizer.step()
        if snapshot_kind == "full":
            changed = manager.diff(captured)
            assert changed["summary"]["total_changes"] == 1
            old, new = changed["modified"][spec.key]["old"], changed["modified"][spec.key]["new"]
        else:
            old, new = manager.get_diff(captured)[spec.key]
        assert float(old) == 1.0 and float(new) == 0.75
        assert manager.parameters[spec.key].value == 1.0

    def test_full_restore_rejects_conflicting_spec_and_storage_values(self, manager):
        spec = ParameterSpec(key="test.snapshot_consistency", value=1.0, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        manager.register_parameter(spec)
        parameter = manager.nn_parameters[spec.key]
        captured = manager.snapshot()
        captured["parameters"][spec.key]["value"] = 2.0
        with pytest.raises(ValueError, match="snapshot.*value|value.*snapshot"):
            manager.restore(captured)
        assert manager.parameters[spec.key] is spec
        assert manager.nn_parameters[spec.key] is parameter
        assert parameter.item() == 1.0

    @pytest.mark.parametrize("snapshot_kind", ["full", "values"])
    def test_snapshot_restore_keeps_every_value_type_dtype_and_spec(self, manager, snapshot_kind):
        """snapshot -> set_value -> restore gives back each value with its type and dtype,
        and spec content equal to the snapshotted spec; a learnable float stays a float."""
        specs = [
            ParameterSpec(key="snapshot.types.float", value=1.5, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key="snapshot.types.int", value=3, units="1", dtype="int64", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key="snapshot.types.bool", value=True, units="1", dtype="bool", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key="snapshot.types.str", value="linear", units="1", dtype="str", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key="snapshot.types.list", value=[1, 2], units="1", dtype="list", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key="snapshot.types.learnable", value=0.5, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
        ]
        for spec in specs:
            manager.register_parameter(spec)
        before = {
            spec.key: (manager.parameters[spec.key].to_dict(), manager.storage.get(spec.key))
            for spec in specs
        }
        live_before = manager.nn_parameters["snapshot.types.learnable"].detach().clone()
        captured = manager.snapshot() if snapshot_kind == "full" else manager.create_snapshot()

        changed = {
            "snapshot.types.float": 2.5,
            "snapshot.types.int": 4,
            "snapshot.types.bool": False,
            "snapshot.types.str": "cubic",
            "snapshot.types.list": [3, 4, 5],
            "snapshot.types.learnable": 0.75,
        }
        for key, value in changed.items():
            manager.set_value(key, value, validate=False)
        if snapshot_kind == "full":
            manager.restore(captured)
        else:
            manager.restore_snapshot(captured)

        for key, (spec_dict_before, stored_before) in before.items():
            spec_after = manager.parameters[key]
            stored_after = manager.storage.get(key)
            assert type(stored_after) is type(stored_before), key
            assert stored_after == stored_before, key
            assert type(spec_after.value) is type(spec_dict_before["value"]), key
            assert spec_after.to_dict() == spec_dict_before, key
        restored_live = manager.nn_parameters["snapshot.types.learnable"]
        assert restored_live.dtype == live_before.dtype
        assert torch.equal(restored_live.detach(), live_before)

    def test_bulk_database_load_keeps_every_owner_in_step(self, manager, monkeypatch):
        """Loading the whole database (here through debug_report_units on an empty
        manager) gives every specification its stored value and every learnable
        specification its live binding."""
        declared = ParameterSpec(key="bulk.declared", value=2.0, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        learnable = ParameterSpec(key="bulk.learnable", value=0.5, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        monkeypatch.setattr(
            manager.database, "load_all", lambda: {declared.key: declared, learnable.key: learnable}
        )
        assert not manager.parameters
        manager.debug_report_units()
        assert manager.parameters[declared.key].value == 2.0
        assert manager.storage.get(declared.key) == 2.0
        assert manager.storage.get(learnable.key) == 0.5
        assert manager.nn_parameters[learnable.key].item() == 0.5

    @pytest.mark.parametrize("snapshot_kind", ["full", "values"])
    def test_snapshot_restore_keeps_a_learnable_spec_whose_binding_is_not_created_yet(
        self, manager, snapshot_kind
    ):
        """A learnable specification taken from the database gets its live binding on
        first access; a snapshot taken before that restores it without one, and a
        value set afterwards comes back with its type."""
        key = "pedosphere.transport.nitrate_distribution_coefficient"
        database_spec = manager.database.get_parameter(key)
        assert database_spec is not None and database_spec.requires_grad
        manager.storage.restore(
            manager.storage.snapshot("lazy", parameters={key: database_spec.value})
        )
        assert key in manager.parameters and key not in manager.nn_parameters
        specs_before = {name: spec.to_dict() for name, spec in manager.parameters.items()}
        stored_before = manager.storage.get_all()
        captured = manager.snapshot() if snapshot_kind == "full" else manager.create_snapshot()

        manager.set_value(key, database_spec.value * 2, validate=False)
        manager.get_parameter(key)
        assert key in manager.nn_parameters
        if snapshot_kind == "full":
            manager.restore(captured)
            assert {name: spec.to_dict() for name, spec in manager.parameters.items()} == specs_before
            assert manager.storage.get_all() == stored_before
            assert key not in manager.nn_parameters
        else:
            manager.restore_snapshot(captured)
        restored = manager.parameters[key]
        assert type(restored.value) is type(database_spec.value)
        assert restored.value == database_spec.value

    def test_storage_restore_rejects_a_value_without_a_spec_and_changes_nothing(self, manager):
        """A stored value without a specification has no units, bounds or dtype: the
        restore is rejected instead of inventing a dimensionless specification, and
        neither the specifications nor the stored values change."""
        manager.register_parameter(ParameterSpec(key="restore.known", value=1.0, units="m", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))
        candidate = manager.storage.snapshot(
            "candidate", parameters={"restore.known": 2.0, "restore.orphan": 5.0}
        )
        specs_before = {key: spec.to_dict() for key, spec in manager.parameters.items()}
        with pytest.raises(KeyError, match="restore.orphan"):
            manager.storage.restore(candidate)
        assert {key: spec.to_dict() for key, spec in manager.parameters.items()} == specs_before
        assert manager.storage.get("restore.known") == 1.0
        assert not manager.storage.exists("restore.orphan")

    def test_storage_restore_takes_the_database_spec_of_a_value_whose_spec_is_not_loaded(
        self, manager
    ):
        """A restored value whose specification is only in the database gets that
        specification (its real units), not an invented dimensionless one."""
        key = "pedosphere.transport.nitrate_distribution_coefficient"
        database_spec = manager.database.get_parameter(key)
        assert database_spec is not None and database_spec.units != "dimensionless"
        assert key not in manager.parameters
        manager.storage.restore(
            manager.storage.snapshot("candidate", parameters={key: database_spec.value})
        )
        assert manager.parameters[key].to_dict() == database_spec.to_dict()

    def test_storage_restore_keeps_live_bindings_with_the_restored_values(self, manager):
        """After a storage restore every learnable binding holds its restored value
        (same Parameter object), and a parameter registered after the snapshot loses
        its binding together with its specification."""
        spec = ParameterSpec(key="restore.learnable", value=1.0, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        manager.register_parameter(spec)
        binding = manager.nn_parameters[spec.key]
        captured = manager.storage.snapshot("before")
        manager.set_value(spec.key, 0.5, validate=False)
        later = ParameterSpec(key="restore.later", value=2.0, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        manager.register_parameter(later)
        manager.storage.restore(captured)
        assert manager.nn_parameters[spec.key] is binding
        assert binding.item() == 1.0
        assert later.key not in manager.parameters
        assert later.key not in manager.nn_parameters

    def test_set_parameter_writes_one_parameter_and_keeps_its_owners_in_step(
        self, manager, monkeypatch
    ):
        """set_parameter persists one specification and sets that parameter's spec,
        stored value and live binding (the existing Parameter object is kept); the
        manager does not take the other database specifications."""
        from dataclasses import replace

        persisted = []
        monkeypatch.setattr(
            manager.database, "set_parameter", lambda spec, tier: persisted.append((spec.key, tier))
        )
        spec = ParameterSpec(key="set.one", value=0.5, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        keys_before = set(manager.parameters)
        manager.set_parameter(spec, tier="calibrated")
        assert persisted == [("set.one", "calibrated")]
        assert set(manager.parameters) == keys_before | {"set.one"}
        assert manager.parameters["set.one"].value == 0.5
        assert manager.storage.get("set.one") == 0.5
        binding = manager.nn_parameters["set.one"]
        assert binding.item() == 0.5

        manager.set_parameter(replace(spec, value=0.75), tier="calibrated")
        assert manager.nn_parameters["set.one"] is binding
        assert binding.item() == 0.75
        assert manager.storage.get("set.one") == 0.75

        manager.set_parameter(replace(spec, value=0.8, requires_grad=False), tier="calibrated")
        assert "set.one" not in manager.nn_parameters
        assert manager.storage.get("set.one") == 0.8

    def test_set_parameter_rejects_a_learnable_spec_it_cannot_bind_before_any_write(
        self, manager, monkeypatch
    ):
        """A learnable specification whose value has no tensor form is rejected before
        anything is persisted or changed."""
        persisted = []
        monkeypatch.setattr(
            manager.database, "set_parameter", lambda spec, tier: persisted.append(spec.key)
        )
        bad = ParameterSpec(key="set.bad", value="linear", units="1", dtype="str", requires_grad=True, bounds=(-float("inf"), float("inf")), category="model_parameter")
        with pytest.raises(TypeError):
            manager.set_parameter(bad, tier="calibrated")
        assert persisted == []
        assert "set.bad" not in manager.parameters
        assert not manager.storage.exists("set.bad")

    def test_set_value_on_an_unknown_key_raises_and_registers_nothing(self, manager):
        """A misspelled key is not silently turned into a new parameter."""
        with pytest.raises(KeyError, match="register_parameter"):
            manager.set_value("set.unknown.key", 1.0)
        assert manager.get_parameter("set.unknown.key") is None
        assert not manager.storage.exists("set.unknown.key")

    def test_restored_spec_metadata_does_not_alias_reusable_snapshot(self, manager):
        spec = ParameterSpec(key="test.snapshot_metadata", value=1.0, provenance={"source": "original"}, units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        manager.register_parameter(spec)
        captured = manager.snapshot()
        manager.restore(captured)
        manager.parameters[spec.key].provenance["source"] = "changed"
        assert captured["parameters"][spec.key]["provenance"]["source"] == "original"
        manager.restore(captured)
        assert manager.parameters[spec.key].provenance["source"] == "original"

    def test_parameter_validation(self, manager):
        """Test parameter validation functionality."""
        # Register valid parameter
        valid_spec = ParameterSpec(
            key="temp.air",
            value=298.15,  # 25掳C in Kelvin
            units="K",
            bounds=(200.0, 350.0),  # Reasonable temperature range
            category=ParameterCategory.ENVIRONMENTAL,
            description="Air temperature", dtype="float32"
        )
        manager.register_parameter(valid_spec)

        # Register parameter without bounds, then set invalid value
        invalid_spec = ParameterSpec(
            key="invalid.temp",
            value=250.0,  # Start with valid value
            units="K",
            category=ParameterCategory.ENVIRONMENTAL,
            description="Invalid temperature for testing", dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(invalid_spec)

        # Create a parameter with valid value but then change bounds to make it invalid
        # This simulates a parameter that becomes invalid due to configuration changes
        manager.parameters["invalid.temp"].bounds = (200.0, 350.0)
        # Set invalid value directly (bypassing bounds check)
        manager.parameters["invalid.temp"].value = 100.0

        # Validate all parameters
        errors = manager.validate_parameters()

        # Should find error with invalid.temp
        assert len(errors) > 0
        error_keys = [err['parameter'] for err in errors]
        assert "invalid.temp" in error_keys

        # Validate specific parameters
        specific_errors = manager.validate_parameters(["temp.air"])
        assert len(specific_errors) == 0  # Should be valid

    def test_parameter_validation_detects_invalid_units(self, manager):
        """Test that validate_parameters uses full validator (units)."""
        bad_unit_spec = ParameterSpec(
            key="unit.bad",
            value=1.0,
            units="invalid_unit",
            bounds=(0.0, 2.0),
            category=ParameterCategory.PHYSICAL,
            description="Invalid unit for validation test", dtype="float32",
        )
        manager.register_parameter(bad_unit_spec)

        errors = manager.validate_parameters(["unit.bad"])
        assert len(errors) > 0
        assert any("unit" in err["error"].lower() for err in errors)

    def test_batch_operations(self, manager):
        """Test batch parameter operations."""
        # Create multiple parameters
        specs = [
            ParameterSpec(
                key=f"batch.param.{i}",
                value=float(i * 10),
                category=ParameterCategory.COMPUTATIONAL,
                description=f"Batch parameter {i}", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
            for i in range(5)
        ]

        # Batch register
        manager.register_parameters(specs)
        assert len(manager.get_all_parameters()) == 5

        # Batch get
        keys = [f"batch.param.{i}" for i in range(5)]
        values = manager.get_batch(keys)
        expected_values = [float(i * 10) for i in range(5)]
        assert values == expected_values

        # Batch set
        new_values = [float(i * 20) for i in range(5)]
        updates = dict(zip(keys, new_values))
        manager.set_batch(updates)

        # Verify batch set
        updated_values = manager.get_batch(keys)
        assert updated_values == new_values

    def test_set_batch_rolls_back_all_owners_when_one_update_fails(self, manager):
        """A failed batch must leave values, tensors, events, and callbacks unchanged."""
        specs = [
            ParameterSpec(
                key="batch.atomic.first",
                value=1.0,
                units="1",
                bounds=(0.0, 10.0),
                requires_grad=True,
                category=ParameterCategory.LEARNABLE,
                description="First atomic batch parameter", dtype="float32",
            ),
            ParameterSpec(
                key="batch.atomic.second",
                value=2.0,
                units="1",
                bounds=(0.0, 10.0),
                requires_grad=True,
                category=ParameterCategory.LEARNABLE,
                description="Second atomic batch parameter", dtype="float32",
            ),
        ]
        manager.register_parameters(specs)
        first_tensor = manager.nn_parameters["batch.atomic.first"]
        second_tensor = manager.nn_parameters["batch.atomic.second"]
        manager.change_history.clear()
        clamp_events_before = manager.get_clamp_events()
        callback_events = []

        def callback(key, old_value, new_value):
            callback_events.append((key, old_value, new_value))

        manager.add_change_callback(callback)
        previous_enforce_bounds_error = manager.enforce_bounds_error
        manager.enforce_bounds_error = True
        try:
            with pytest.raises(ValueError, match="bounds"):
                manager.set_batch(
                    {
                        "batch.atomic.first": 3.0,
                        "batch.atomic.second": 99.0,
                    }
                )
        finally:
            manager.enforce_bounds_error = previous_enforce_bounds_error
            manager.remove_change_callback(callback)

        assert manager.get_batch(
            ["batch.atomic.first", "batch.atomic.second"]
        ) == [1.0, 2.0]
        assert manager.storage.get("batch.atomic.first") == 1.0
        assert manager.storage.get("batch.atomic.second") == 2.0
        assert manager.get_parameter("batch.atomic.first").value == 1.0
        assert manager.get_parameter("batch.atomic.second").value == 2.0
        assert manager.nn_parameters["batch.atomic.first"] is first_tensor
        assert manager.nn_parameters["batch.atomic.second"] is second_tensor
        assert first_tensor.item() == pytest.approx(1.0)
        assert second_tensor.item() == pytest.approx(2.0)
        assert callback_events == []
        assert manager.change_history == []
        assert manager.get_clamp_events() == clamp_events_before

    def test_set_batch_notifies_only_after_the_full_batch_is_committed(self, manager):
        """Every callback must observe the final state of the complete batch."""
        manager.register_parameters(
            [
                ParameterSpec(
                    key="batch.notify.first",
                    value=1.0,
                    units="1",
                    category=ParameterCategory.COMPUTATIONAL,
                    description="First notification batch parameter", dtype="float32", bounds=(-float("inf"), float("inf")),
                ),
                ParameterSpec(
                    key="batch.notify.second",
                    value=2.0,
                    units="1",
                    category=ParameterCategory.COMPUTATIONAL,
                    description="Second notification batch parameter", dtype="float32", bounds=(-float("inf"), float("inf")),
                ),
            ]
        )
        observed_states = []

        def callback(key, _old_value, _new_value):
            observed_states.append(
                (
                    key,
                    manager.get("batch.notify.first"),
                    manager.get("batch.notify.second"),
                )
            )

        manager.add_change_callback(callback)
        try:
            manager.set_batch(
                {
                    "batch.notify.first": 10.0,
                    "batch.notify.second": 20.0,
                }
            )
        finally:
            manager.remove_change_callback(callback)

        assert observed_states == [
            ("batch.notify.first", 10.0, 20.0),
            ("batch.notify.second", 10.0, 20.0),
        ]

    def test_set_batch_notifies_every_key_before_raising_callback_failures(self, manager):
        """A callback failure on one key does not hide the notification of the others."""
        manager.register_parameters(
            [
                ParameterSpec(key="batch.fail.first", value=1.0, units="1",
                              category=ParameterCategory.COMPUTATIONAL, description="first", dtype="float32", bounds=(-float("inf"), float("inf"))),
                ParameterSpec(key="batch.fail.second", value=2.0, units="1",
                              category=ParameterCategory.COMPUTATIONAL, description="second", dtype="float32", bounds=(-float("inf"), float("inf"))),
            ]
        )
        notified = []

        def callback(key, _old_value, _new_value):
            notified.append(key)
            if key == "batch.fail.first":
                raise RuntimeError("cache refresh failed for first")

        manager.add_change_callback(callback)
        try:
            with pytest.raises(ExceptionGroup) as error:
                manager.set_batch({"batch.fail.first": 10.0, "batch.fail.second": 20.0})
        finally:
            manager.remove_change_callback(callback)

        assert notified == ["batch.fail.first", "batch.fail.second"]
        assert manager.get("batch.fail.first") == 10.0
        assert manager.get("batch.fail.second") == 20.0
        assert "cache refresh failed for first" in repr(error.value)

    def test_adopting_a_learnable_spec_without_tensor_form_raises_before_any_write(self, manager):
        """A learnable database entry that cannot be bound is malformed; no owner changes."""
        from relic.parameters.ops.bootstrap import adopt_database_specs

        numeric = ParameterSpec(key="adopt.numeric", value=1.0, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        textual = ParameterSpec(key="adopt.textual", value="wet", units="1", dtype="str", requires_grad=True, bounds=(-float("inf"), float("inf")), category="model_parameter")
        with pytest.raises(TypeError, match="adopt.textual"):
            adopt_database_specs(manager, {"adopt.numeric": numeric, "adopt.textual": textual})
        assert "adopt.numeric" not in manager.parameters
        assert not manager.storage.exists("adopt.numeric")
        assert "adopt.numeric" not in manager.nn_parameters

    @pytest.mark.parametrize("declared", ["dimensionless", "1"])
    def test_set_value_never_replaces_a_declared_dimensionless_unit(self, manager, declared):
        manager.register_parameter(
            ParameterSpec(key="units.fraction", value=0.5, units=declared, bounds=(0.0, 10.0), dtype="float32", category="model_parameter")
        )
        with pytest.raises(ValueError, match="units.fraction"):
            manager.set_value("units.fraction", 2.0, units="m")
        assert manager.get_parameter("units.fraction").units == declared
        assert manager.get("units.fraction") == 0.5

        manager.set_value("units.fraction", 0.25, units="1" if declared == "dimensionless" else "dimensionless")
        assert manager.get("units.fraction") == 0.25
        assert manager.get_parameter("units.fraction").units == declared

    _VALIDATE_PROBES = (
        ("indep.length", {}),
        ("indep.length", {"units": "m"}),
        ("indep.length", {"expected_type": float}),
        ("indep.weight", {}),
        ("indep.label", {}),
        ("indep.missing", {"default": 7.0}),
    )

    @staticmethod
    def _register_validate_probes(manager, mode):
        from relic.parameters.types import UnitValidationMode

        manager.unit_validation_mode = UnitValidationMode(mode)
        manager.register_parameters([
            ParameterSpec(key="indep.length", value=250.0, units="cm", bounds=(0.0, 1000.0), dtype="float32", category="model_parameter"),
            ParameterSpec(key="indep.weight", value=1.5, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key="indep.label", value="wet", units="1", dtype="str", bounds=(-float("inf"), float("inf")), category="model_parameter"),
        ])

    def test_the_read_path_has_no_validate_argument(self, manager):
        """Reading never had a validate effect (the value was independent of it);
        no function of the get chain declares it and get() rejects it."""
        import inspect

        from relic.parameters import manager as manager_module
        from relic.parameters.ops import access_impl, accessors, manager_registry, unit_validation

        chain = (
            manager.get,
            manager._get_impl,
            manager._convert_parameter_units,
            manager_module.get,
            manager_registry.get,
            accessors.get_value,
            access_impl.get_impl,
            unit_validation.convert_parameter_units,
        )
        for function in chain:
            parameters = inspect.signature(function).parameters
            assert "validate" not in parameters, function.__qualname__
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(manager.get).parameters.values()
        )

        self._register_validate_probes(manager, "off")
        with pytest.raises(TypeError, match="validate"):
            manager.get("indep.length", validate=True)
        with pytest.raises(TypeError, match="unit"):
            manager.get("indep.length", unit="m")
        for key, kwargs in self._VALIDATE_PROBES:
            manager.clear_cache()
            manager.get(key, **kwargs)

    def test_expected_float_widens_an_integer_value_only(self, manager):
        """An int value reads as float when float is expected (lossless); bool and
        narrowing float -> int are type errors."""
        manager.register_parameters([
            ParameterSpec(key="widen.start_day", value=120, units="day", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key="widen.fraction", value=0.5, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key="widen.flag", value=True, units="1", dtype="bool", bounds=(-float("inf"), float("inf")), category="model_parameter"),
        ])
        value = manager.get("widen.start_day", expected_type=float)
        assert isinstance(value, float) and value == 120.0
        with pytest.raises(TypeError, match="widen.fraction"):
            manager.get("widen.fraction", expected_type=int)
        with pytest.raises(TypeError, match="widen.flag"):
            manager.get("widen.flag", expected_type=float)

    def test_a_failing_direct_storage_read_is_not_retried_elsewhere(self, manager, monkeypatch):
        """The direct storage path raises its failure instead of silently taking the slow path."""
        self._register_validate_probes(manager, "off")

        def broken_exists(key):
            raise RuntimeError("storage index broke")

        monkeypatch.setattr(manager.storage, "exists", broken_exists)
        with pytest.raises(RuntimeError, match="storage index broke"):
            manager.get("indep.length")

    def test_a_failing_namespace_load_is_not_read_as_absence(self, manager, monkeypatch):
        """A namespace-tree lookup whose load fails raises instead of returning None."""
        self._register_validate_probes(manager, "off")
        # Once the database is enumerated the single-key lookup loads nothing;
        # the load that fails below is the one of the namespace rule.
        manager.database.load_all()

        def broken_load():
            raise RuntimeError("namespace index broke")

        monkeypatch.setattr(manager.database, "load_all", broken_load)
        with pytest.raises(RuntimeError, match="namespace index broke"):
            manager.get("indep.absent_namespace_probe")

    @pytest.mark.parametrize("mode", ["off", "check-only"])
    def test_every_successful_read_is_recorded_in_the_access_history(self, manager, mode):
        """The access record (restart contract) does not depend on the read path."""
        self._register_validate_probes(manager, mode)
        for key, kwargs in self._VALIDATE_PROBES[:5]:
            manager.clear_cache()
            manager._access_history.clear()
            manager.get(key, **kwargs)
            assert key in manager._access_history, (mode, key, kwargs)

    def test_gradient_parameters(self, manager):
        """Test gradient-enabled parameters."""
        spec = ParameterSpec(
            key="grad.param",
            value=5.0,
            requires_grad=True,
            category=ParameterCategory.LEARNABLE,
            description="Gradient parameter test", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(spec)

        # Retrieve the manager-owned trainable tensor through the canonical API.
        tensor = manager.get("grad.param", expected_type=torch.Tensor)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.requires_grad
        assert tensor is manager.nn_parameters[spec.key]

        # Test gradient computation
        loss = tensor.pow(2).sum()  # Simple loss: x^2
        loss.backward()

        # Check gradient exists
        assert tensor.grad is not None
        assert tensor.grad.item() == 2 * 5.0  # Gradient of x^2 is 2x
        assert manager.nn_parameters[spec.key].grad is tensor.grad

    def test_parameter_categories(self, manager):
        """Test parameter categorization and filtering."""
        # Register parameters in different categories
        categories_specs = [
            (ParameterCategory.BIOPHYSICAL, "biophysical.param"),
            (ParameterCategory.COMPUTATIONAL, "computational.param"),
            (ParameterCategory.ENVIRONMENTAL, "environmental.param"),
            (ParameterCategory.CALIBRATION, "calibration.param")
        ]

        for category, key in categories_specs:
            spec = ParameterSpec(
                key=key,
                value=1.0,
                category=category,
                description=f"Parameter in {category} category", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
            manager.register_parameter(spec)

        # Test filtering by category
        biophysical_params = manager.get_parameters_by_category(ParameterCategory.BIOPHYSICAL)
        assert len(biophysical_params) == 1
        assert biophysical_params[0].key == "biophysical.param"

        # Test statistics by category
        stats = manager.get_statistics()
        assert stats['by_category'][ParameterCategory.BIOPHYSICAL] == 1
        assert stats['by_category'][ParameterCategory.COMPUTATIONAL] == 1


class TestParameterValidators:
    """Test parameter validation classes."""

    def test_range_validator(self):
        """Test RangeValidator functionality."""
        validator = RangeValidator()

        # Test valid parameter within bounds
        valid_spec = ParameterSpec(
            key="valid.range",
            value=50.0,
            bounds=(0.0, 100.0),
            category=ParameterCategory.BIOPHYSICAL,
            description="Valid range test", units="dimensionless", dtype="float32"
        )
        result = validator.validate(valid_spec)
        assert result.is_valid
        assert len(result.errors) == 0

        # Test invalid parameter outside bounds
        # Create spec without bounds first, then add bounds for validation
        invalid_spec = ParameterSpec(
            key="invalid.range",
            value=150.0,  # Will be outside bounds
            category=ParameterCategory.BIOPHYSICAL,
            description="Invalid range test", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        # Add bounds that make the value invalid
        invalid_spec.bounds = (0.0, 100.0)

        result = validator.validate(invalid_spec)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "bounds" in result.errors[0].lower() or "maximum" in result.errors[0].lower()

    def test_physical_consistency_validator(self):
        """Test PhysicalConsistencyValidator for inter-parameter relationships."""
        validator = PhysicalConsistencyValidator()

        # Test valid water-related parameters
        specs = [
            ParameterSpec(
                key="water.field_capacity",
                value=0.3,  # 30% volumetric water content
                units="m3/m3",
                bounds=(0.0, 1.0),
                category=ParameterCategory.PHYSICAL,
                description="Soil field capacity", dtype="float32"
            ),
            ParameterSpec(
                key="water.wilting_point",
                value=0.1,  # 10% volumetric water content
                units="m3/m3",
                bounds=(0.0, 1.0),
                category=ParameterCategory.PHYSICAL,
                description="Soil wilting point", dtype="float32"
            )
        ]

        # Should pass - field capacity > wilting point
        result = validator.validate_water_parameters(specs)
        assert result.is_valid

        # Test invalid relationship
        specs[1].value = 0.4  # Wilting point > field capacity (invalid)
        result = validator.validate_water_parameters(specs)
        assert not result.is_valid
        assert "field capacity" in result.errors[0].lower()

    def test_unit_validator(self):
        """Test UnitValidator for CF-1.8 compliance."""
        validator = UnitValidator()

        # Test valid CF-1.8 units
        valid_spec = ParameterSpec(
            key="custom.thermal_state",
            value=298.15,
            units="K",  # Valid CF-1.8 unit
            dimension="temperature",
            category=ParameterCategory.ENVIRONMENTAL,
            description="Air temperature", dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        result = validator.validate(valid_spec)
        assert result.is_valid

        # Test invalid units
        invalid_spec = ParameterSpec(
            key="custom.ambiguous_angle",
            value=25.0,
            units="degrees",  # Invalid/ambiguous unit
            dimension="temperature",
            category=ParameterCategory.ENVIRONMENTAL,
            description="Temperature with invalid units", dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        result = validator.validate(invalid_spec)
        assert not result.is_valid
        assert "inconsistent" in result.errors[0].lower()

        misleading_key_spec = ParameterSpec(
            key="temperature.spatial_resolution",
            value=1000.0,
            units="m",
            dimension="length",
            category=ParameterCategory.PHYSICAL,
            description="Spatial resolution with a misleading key token", dtype="float32", bounds=(-float("inf"), float("inf")),
        )
        result = validator.validate(misleading_key_spec)
        assert result.is_valid


class TestGlobalManager:
    """Test global parameter manager singleton."""

    def test_reset_replaces_the_canonical_parameter_singleton(self):
        """Reset must replace the sole ParameterManager singleton."""
        manager1 = get_global_manager()

        reset_global_manager()
        manager2 = get_global_manager()
        assert manager2 is not manager1
        assert manager2 is get_global_manager()

    @pytest.mark.parametrize(
        ("variable", "value"),
        [("RELIC_BOUNDS_POLICY", "clamp_quietly"), ("RELIC_UNIT_VALIDATION_MODE", "lenient")],
    )
    def test_invalid_policy_environment_variable_is_rejected(self, monkeypatch, variable, value):
        """An invalid policy is a configuration error, not a request for the default policy."""
        monkeypatch.setenv(variable, value)
        with pytest.raises(ValueError, match=variable):
            UnifiedParameterManager()

    def test_parameter_count_failure_propagates(self):
        class _BrokenDatabase:
            def __len__(self):
                raise RuntimeError("database index unreadable")

            def load_all(self):
                return {}

        manager = UnifiedParameterManager()
        manager.database = _BrokenDatabase()
        with pytest.raises(RuntimeError, match="database index unreadable"):
            len(manager)

    def test_construction_returns_an_independent_manager(self):
        """get_global_manager() is the only global entry point; construction is private."""
        global_manager = get_global_manager()
        private = UnifiedParameterManager()
        other = UnifiedParameterManager()

        assert private is not global_manager
        assert other is not private
        assert not hasattr(UnifiedParameterManager, "_instance")
        private.register_parameter(ParameterSpec(
            key="private.only_probe",
            value=1.0,
            units="dimensionless",
            category=ParameterCategory.BIOPHYSICAL,
            description="Private manager probe", dtype="float32", bounds=(-float("inf"), float("inf")),
        ))
        assert private.get_parameter("private.only_probe") is not None
        assert global_manager.get_parameter("private.only_probe") is None
        assert other.get_parameter("private.only_probe") is None
        assert get_global_manager() is global_manager

    def test_global_manager_singleton(self):
        """Test that get_global_manager returns singleton instance."""
        manager1 = get_global_manager()
        manager2 = get_global_manager()

        assert manager1 is manager2  # Should be same instance
        assert isinstance(manager1, UnifiedParameterManager)

    def test_global_manager_persistence(self):
        """Test that global manager maintains state between calls."""
        manager = get_global_manager()
        snapshot = manager.snapshot()

        try:
            # Add parameter
            spec = ParameterSpec(
                key="global.test.param",
                value=42.0,
                category=ParameterCategory.COMPUTATIONAL,
                description="Global manager persistence test", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
            manager.register_parameter(spec)

            # Get manager again and check parameter persists
            manager2 = get_global_manager()
            assert manager2.has_parameter("global.test.param")
            assert manager2.get("global.test.param") == 42.0
        finally:
            # Preserve parameters registered before this test.
            manager.restore(snapshot)

    def test_independent_manager_leaves_the_process_variable_mapper_alone(self, monkeypatch, tmp_path):
        """Only the process-wide manager refreshes the process-wide variable mapper."""
        from relic.core.variable_mapper import variable_name_mapper

        calls = []
        monkeypatch.setattr(
            variable_name_mapper,
            "refresh_from_parameter_manager",
            lambda manager=None: calls.append(manager),
        )

        UnifiedParameterManager(database_path=tmp_path / "db")
        UnifiedParameterManager()

        assert calls == []

    def test_global_manager_refreshes_the_variable_mapper_once(self, monkeypatch):
        from relic.core.variable_mapper import variable_name_mapper
        from relic.parameters.ops import manager_registry

        calls = []
        monkeypatch.setattr(manager_registry, "_global_manager", None)
        monkeypatch.setattr(
            variable_name_mapper,
            "refresh_from_parameter_manager",
            lambda manager=None: calls.append(manager),
        )

        manager = get_global_manager()
        assert get_global_manager() is manager

        assert calls == [manager]

    def test_variable_mapper_refresh_failure_propagates(self, monkeypatch):
        """A failed alias refresh is not hidden and leaves no half-built global manager."""
        from relic.core.variable_mapper import variable_name_mapper
        from relic.parameters.ops import manager_registry

        def _fail(manager=None):
            raise KeyError("variable catalog probe missing")

        monkeypatch.setattr(manager_registry, "_global_manager", None)
        monkeypatch.setattr(variable_name_mapper, "refresh_from_parameter_manager", _fail)

        with pytest.raises(KeyError, match="variable catalog probe missing"):
            get_global_manager()
        assert manager_registry._global_manager is None


class TestParameterDatabase:
    """Test parameter database loading and saving."""

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    @pytest.mark.parametrize("kind", ["tensor", "category"])
    def test_public_single_file_load_preserves_complete_spec(self, tmp_path, file_format, kind):
        key = "test.single_file_contract"
        if kind == "tensor":
            spec = ParameterSpec(
                key=key, value=torch.tensor([1.0e-50], dtype=torch.float64),
                default=torch.tensor(2.0e-50, dtype=torch.float64), dtype="float64",
                provenance={"source": "file_roundtrip"}, units="dimensionless", bounds=(-float("inf"), float("inf")), category="model_parameter",
            )
        else:
            spec = ParameterSpec(
                key=key, value="wet", default="dry", dtype="str",
                allowed_values=["dry", "wet"], provenance={"source": "file_roundtrip"}, units="dimensionless", bounds=(-float("inf"), float("inf")), category="model_parameter",
            )
        path = tmp_path / ("parameters." + file_format)
        export({key: spec}, format=file_format, output_path=path)
        manager = UnifiedParameterManager()
        manager.clear_parameters()
        assert manager.load_from_database(db_path=str(path)) == 1
        restored = manager.get_parameter(key)
        assert restored.dtype == spec.dtype
        assert restored.provenance == spec.provenance
        if kind == "tensor":
            assert torch.equal(restored.value, spec.value)
            assert torch.equal(restored.default, spec.default)
            assert restored.default.shape == ()
            assert restored.value.dtype == torch.float64
            assert torch.equal(manager.get(key, expected_type=torch.Tensor), spec.value)
        else:
            assert restored.default == "dry"
            assert restored.allowed_values == ["dry", "wet"]
            assert restored.validate_value("unknown")

    def test_database_loading(self):
        """Test loading parameters from database files."""
        manager = UnifiedParameterManager()
        manager.clear_parameters()  # Clear any existing parameters

        # Create temporary database directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir)
            relic_dir = db_path / "relic"
            relic_dir.mkdir()

            # Create test parameter file
            test_params = {
                "photosynthesis.vcmax25": {
                    "value": 62.5,
                    "units": "umol CO2 m-2 s-1",
                    "dtype": "float32",
                    "bounds": [10.0, 200.0],
                    "category": "biophysical",
                    "description": "Maximum carboxylation rate at 25掳C",
                    "references": ["Farquhar et al. 1980"]
                }
            }

            param_file = relic_dir / "photosynthesis.yaml"
            with open(param_file, 'w') as f:
                yaml.dump(test_params, f)

            # Load from database
            loaded_count = manager.load_from_database(
                db_path=str(db_path),
                source_priority=["relic"]
            )

            assert loaded_count > 0
            assert manager.has_parameter("photosynthesis.vcmax25")

            # Verify parameter properties
            spec = manager.get_parameter("photosynthesis.vcmax25")
            assert spec.value == 62.5
            assert spec.units == "umol CO2 m-2 s-1"
            assert spec.bounds == (10.0, 200.0)
            assert "Farquhar" in spec.references[0]

    def test_database_saving(self):
        """Test saving parameters to database files."""
        manager = UnifiedParameterManager()
        manager.clear_parameters()  # Clear any existing parameters

        # Add test parameters
        specs = [
            ParameterSpec(
                key="save.test.param1",
                value=1.0,
                units="dimensionless",
                category=ParameterCategory.COMPUTATIONAL,
                description="Save test parameter 1", dtype="float32", bounds=(-float("inf"), float("inf"))
            ),
            ParameterSpec(
                key="save.test.param2",
                value=2.0,
                units="m/s",
                category=ParameterCategory.PHYSICAL,
                description="Save test parameter 2", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
        ]

        for spec in specs:
            manager.register_parameter(spec)

        # Save to database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir)
            saved_count = manager.save_to_database(
                db_path=str(db_path),
                format="yaml"
            )

            assert saved_count == 2

            # Verify files were created
            saved_files = list(db_path.glob("**/*.yaml"))
            assert len(saved_files) > 0

            # Load and verify content
            for file_path in saved_files:
                with open(file_path) as f:
                    data = yaml.safe_load(f)
                    assert len(data) > 0  # Should contain parameters



class TestParameterSpecRequiredMetadata:
    """units, dtype, bounds and category have no universal value: ParameterSpec requires them."""

    def test_units_dtype_and_bounds_have_no_defaults(self):
        import dataclasses

        fields = ParameterSpec.__dataclass_fields__
        for name in ("units", "dtype", "bounds", "category"):
            assert fields[name].default is dataclasses.MISSING, name
        with pytest.raises(TypeError, match="category"):
            ParameterSpec(key="test.no_category", value=1.0, units="m", dtype="float32", bounds=(0.0, 2.0))
        with pytest.raises(TypeError, match="units"):
            ParameterSpec(key="test.no_units", value=1.0, dtype="float32", bounds=(0.0, 2.0), category="model_parameter")
        with pytest.raises(TypeError, match="dtype"):
            ParameterSpec(key="test.no_dtype", value=1.0, units="m", bounds=(0.0, 2.0), category="model_parameter")
        with pytest.raises(TypeError, match="bounds"):
            ParameterSpec(key="test.no_bounds", value=1.0, units="m", dtype="float32", category="model_parameter")

    def test_unbounded_is_declared_explicitly(self):
        inf = float("inf")
        spec = ParameterSpec(key="test.unbounded", value=1.0, units="m", dtype="float32", bounds=(-inf, inf), category="model_parameter")
        assert spec.bounds == (-inf, inf)

    def test_from_dict_requires_a_declared_dtype_and_category(self):
        payload = {"key": "test.partial", "value": 1.0, "units": "m", "bounds": [0.0, 2.0]}
        # No dtype is inferred from the value type.
        with pytest.raises(TypeError, match="dtype"):
            ParameterSpec.from_dict({**payload, "category": "model_parameter"})
        with pytest.raises(TypeError, match="category"):
            ParameterSpec.from_dict({**payload, "dtype": "float32"})

    def test_from_dict_requires_units_and_bounds(self):
        with pytest.raises(TypeError, match="units"):
            ParameterSpec.from_dict({"key": "test.partial", "value": 1.0, "dtype": "float32", "bounds": [0.0, 2.0]})
        with pytest.raises(TypeError, match="bounds"):
            ParameterSpec.from_dict({"key": "test.partial", "value": 1.0, "dtype": "float32", "units": "m"})


class TestBoundsNoneOnlyForNonNumericDtype:
    """bounds None states that no interval applies; only a non-numeric dtype may state it.

    A numeric dtype declares a (lower, upper) interval and writes an unbounded
    side as -inf or inf. ParameterSpec, has_complete_parameter_metadata, the
    structural audit and the YAML schema read the same rule.
    """

    @staticmethod
    def _metadata(dtype, bounds, value=1.0):
        return {
            "key": "test.bounds_rule",
            "value": value,
            "units": "1",
            "dtype": dtype,
            "bounds": bounds,
            "category": "model_parameter",
            "references": ("unit-test",),
            "provenance": {"source": "unit_test"},
        }

    @pytest.mark.parametrize(
        "dtype", ["float32", "float64", "int32", "int64", "int", "float", "double", "torch.float32"],
    )
    def test_numeric_dtype_rejects_none_bounds(self, dtype):
        with pytest.raises(ValueError, match="bounds"):
            ParameterSpec(**self._metadata(dtype, None))

    @pytest.mark.parametrize("bounds", [(0.0, None), (None, 1.0), (None, None)])
    def test_numeric_dtype_rejects_an_undeclared_side(self, bounds):
        with pytest.raises(ValueError, match="inf"):
            ParameterSpec(**self._metadata("float64", bounds))

    @pytest.mark.parametrize(
        "bounds",
        [("1.0e7", 5.0e7), (0.0, "1e-6"), ("-inf", "inf"), ([0.0, "1e-3"], [1.0, 1.0])],
    )
    def test_numeric_dtype_rejects_text_endpoints(self, bounds):
        """PyYAML (YAML 1.1) loads 1.0e7 and 1e-6 as text; a numeric interval states
        numbers, and a text endpoint is rejected instead of being converted."""
        with pytest.raises(ValueError, match="text"):
            ParameterSpec(**self._metadata("float32", bounds))
        assert has_complete_parameter_metadata(self._metadata("float32", bounds)) is False

    @pytest.mark.parametrize("field", ["value", "default"])
    def test_numeric_dtype_rejects_a_text_value(self, field):
        payload = {**self._metadata("float64", (0.0, 1.0)), field: "1e-6"}
        with pytest.raises(ValueError, match="text"):
            ParameterSpec(**payload)
        with pytest.raises(ValueError, match="text"):
            ParameterSpec.from_dict(payload)

    def test_non_numeric_dtype_keeps_text_bounds_and_value(self):
        spec = ParameterSpec(**self._metadata("str", ("C3", "C4"), "C3"))
        assert spec.bounds == ("C3", "C4")
        assert spec.value == "C3"

    def test_unrecognized_dtype_is_not_exempt(self):
        with pytest.raises(ValueError, match="bounds"):
            ParameterSpec(**self._metadata("quantity", None))

    @pytest.mark.parametrize(
        ("dtype", "value"),
        [
            ("str", "horton"),
            ("string", "C3"),
            ("bool", True),
            ("boolean", False),
            ("list", [1, 2]),
            ("list[float]", [0.1, 0.2]),
            ("tuple", (1, 2)),
            ("dict", {"a": 1}),
            ("mapping", {"a": 1}),
            ("object", {"a": 1}),
            ("enum", "linear"),
        ],
    )
    def test_non_numeric_dtype_keeps_none_bounds(self, dtype, value):
        spec = ParameterSpec(**self._metadata(dtype, None, value))
        assert spec.bounds is None
        assert has_complete_parameter_metadata(spec) is True

    def test_explicit_unbounded_interval_is_complete(self):
        inf = float("inf")
        spec = ParameterSpec(**self._metadata("float64", (-inf, inf)))
        assert spec.bounds == (-inf, inf)
        assert has_complete_parameter_metadata(spec) is True
        assert has_complete_parameter_metadata(self._metadata("float32", [0.0, inf])) is True

    @pytest.mark.parametrize(
        ("dtype", "bounds"),
        [("float64", None), ("int32", None), ("float32", (0.0, None)), ("quantity", None), ("float32", ())],
    )
    def test_complete_metadata_rejects_undeclared_numeric_bounds(self, dtype, bounds):
        assert has_complete_parameter_metadata(self._metadata(dtype, bounds)) is False

    @pytest.mark.parametrize("dtype", ["string", "str", "bool", "list", "dict", "mapping", "enum"])
    def test_complete_metadata_accepts_none_for_non_numeric_dtype(self, dtype):
        assert has_complete_parameter_metadata(self._metadata(dtype, None)) is True

    def test_complete_metadata_requires_the_bounds_key(self):
        payload = self._metadata("string", None)
        payload.pop("bounds")
        assert has_complete_parameter_metadata(payload) is False

    def test_the_rule_has_one_owner(self):
        from relic.parameters.spec import parameter_bounds_error

        inf = float("inf")
        assert parameter_bounds_error(None, "string") is None
        assert parameter_bounds_error((-inf, inf), "float32") is None
        assert parameter_bounds_error(("C3", "C4"), "str") is None
        assert "numeric" in parameter_bounds_error(None, "float32")
        assert "inf" in parameter_bounds_error((0.0, None), "float64")
        assert "pair" in parameter_bounds_error((0.0,), "float64")
        assert "text" in parameter_bounds_error(("1.0e7", 5.0e7), "float32")


class TestRemovedRegisterEntry:
    """The keyword register() entry had no caller; register_parameter(spec) is the entry."""

    def test_manager_has_no_register_entry(self):
        import relic.parameters.ops.mutation as mutation

        # ABCMeta.register (virtual subclassing) is not a parameter entry; no class of the
        # manager defines a register() method.
        assert all("register" not in vars(cls) for cls in UnifiedParameterManager.__mro__)
        assert not hasattr(mutation, "register")
        assert hasattr(UnifiedParameterManager, "register_parameter")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
