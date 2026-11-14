import time

import torch
import pytest

from relic.parameters import (
    get_global_manager,
    reset_global_manager,
    ParameterSpec,
    UnitValidationMode,
    UnifiedParameterManager,
)
from relic.core.functor import BiologicalProcessFunctor, FunctorMetadata, ProcessCategory
from relic.utils.physical_constants import get_physical_constant

# Get constants via the unified physical_constants interface
WATER_DENSITY = get_physical_constant('water_density')


def hydrate_physical_constants(parameter_manager=None):
    """Bridge function to ensure physical constants are loaded via ParameterManager."""
    # Constants are now loaded via get_physical_constant which uses ParameterManager internally
    pass


def test_default_unit_validation_mode_is_explicit_check_only(monkeypatch):
    monkeypatch.delenv("RELIC_UNIT_VALIDATION_MODE", raising=False)
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    assert pm.unit_validation_mode == UnitValidationMode.CHECK_ONLY


def test_lazy_manager_fetch_single_namespace_fast():
    start = time.time()
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    seconds_per_day = pm.get(
        "relic.physical.constants.time.seconds_per_day",
        units="s",
        expected_type=float,
    )
    assert seconds_per_day is not None
    assert float(seconds_per_day) > 0.0
    assert getattr(pm, "_preloaded", False) is False
    assert time.time() - start < 10.0


def test_lazy_namespace_lookup_loads_complete_database_namespace():
    """Namespace queries must not stop at the first lazily cached parameter."""
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    pm.get(
        "coupling.scales.temporal.calendars.gregorian.days_per_year",
        units="days",
        required=True,
    )

    parameters = pm.get_namespace_parameters(
        namespace="coupling.scales.temporal.calendars",
        include_subnamespaces=True,
        return_specs=False,
    )

    assert "coupling.scales.temporal.calendars.gregorian.days_per_year" in parameters
    assert "coupling.scales.temporal.calendars.noleap.days_per_year" in parameters
    assert "coupling.scales.temporal.calendars.360_day.days_per_year" in parameters


def _namespace_tree_leaves(tree, namespace):
    """Flatten a namespace tree into {full key: value}."""
    leaves = {}
    for name, value in tree.items():
        key = f"{namespace}.{name}"
        if isinstance(value, dict):
            leaves.update(_namespace_tree_leaves(value, key))
        else:
            leaves[key] = value
    return leaves


def test_lazy_namespace_tree_holds_the_complete_database_namespace():
    """manager.get("<namespace>") follows the membership rule of the namespace loader.

    The tree does not depend on which keys were loaded on demand before it is
    read, and a later full preload leaves it as it is.
    """
    namespace = "hydrosphere.evapotranspiration"
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    pm.get(f"{namespace}.sub_freezing_phase", required=True)
    held_on_demand = pm.storage.list(namespace=namespace)

    before = _namespace_tree_leaves(pm.get(namespace), namespace)
    pm._preload_parameters(warmup_cache=False)
    after = _namespace_tree_leaves(pm.get(namespace), namespace)

    complete = pm.get_namespace_parameters(namespace=namespace)
    assert len(held_on_demand) < len(complete)
    assert before == complete
    assert after == complete


@pytest.mark.parametrize("preload_all", [False, True])
def test_namespace_tree_holds_every_database_key_at_its_dotted_path(preload_all):
    """manager.get("<domain>") reaches every database key of the domain.

    The tree has one entry per dotted path segment, so it holds a key only at
    its own path: a key that named a parameter and the namespace of another
    key could not be held as a value and as a sub-tree at once.
    """
    pm = UnifiedParameterManager(preload_all=preload_all, warmup_cache=False)
    database_keys = sorted(pm.database.load_all())

    unreachable = []
    for domain in sorted({key.split(".", 1)[0] for key in database_keys}):
        tree = pm.get(domain)
        for key in database_keys:
            if not key.startswith(f"{domain}."):
                continue
            node = tree
            for part in key.split(".")[1:]:
                if not isinstance(node, dict) or part not in node:
                    unreachable.append(key)
                    break
                node = node[part]

    assert unreachable == []


def test_namespace_tree_is_built_again_at_each_read(tmp_path):
    """A tree read belongs to its caller and is never served again.

    A parameter source added after a read (the extension route,
    database.add_source) is part of the next read, and emptying a returned
    tree changes nothing in the manager.
    """
    import yaml

    namespace = "coupling.scales.temporal.calendars"
    added = f"{namespace}.c9_probe.days_per_year"
    source = tmp_path / "calendar_probe.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema_version": "2.0.0",
                "tier": "relic",
                "namespace": f"{namespace}.c9_probe",
                "description": "Namespace tree probe source",
                "parameters": {
                    added: {
                        "value": 365.0,
                        "units": "days",
                        "dtype": "float64",
                        "bounds": [0.0, 1000.0],
                        "category": "computational",
                        "description": "Namespace tree probe",
                        "references": ["RELIC namespace tree test"],
                        "tags": ["probe"],
                        "requires_grad": False,
                        "provenance": {"source": "test"},
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    first = pm.get(namespace)
    assert added not in _namespace_tree_leaves(first, namespace)

    pm.database.add_source(source)
    assert pm.get(added, required=True) == 365.0
    first.clear()

    again = _namespace_tree_leaves(pm.get(namespace), namespace)
    assert again == pm.get_namespace_parameters(namespace=namespace)
    assert again[added] == 365.0


def _register_probe_parameters(pm, values):
    """Register run-time keys {key: value}; a mapping or list value has dtype dict or list and no bounds."""
    for key, value in values.items():
        if isinstance(value, dict):
            dtype, bounds = "dict", None
        elif isinstance(value, list):
            dtype, bounds = "list", None
        else:
            dtype, bounds = "float32", (-float("inf"), float("inf"))
        pm.register_parameter(ParameterSpec(key=key, value=value, units="1", dtype=dtype, bounds=bounds, category="model_parameter"))


@pytest.mark.parametrize("child", ["zz9.a.b", "zz9.a.b.c"])
def test_namespace_tree_rejects_a_key_below_a_parameter_key(child):
    """A key whose namespace is another parameter key has no place in the tree.

    Registration accepts both keys (the listing rule reads such keys, see
    test_namespace_parameters_are_the_descendants_of_the_namespace); the tree
    read raises a ValueError that names both keys, and the single-key reads
    keep their values.
    """
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    _register_probe_parameters(pm, {"zz9.a": 1.0, child: 2.0})

    with pytest.raises(ValueError) as excinfo:
        pm.get("zz9")

    assert "'zz9.a'" in str(excinfo.value)
    assert f"'{child}'" in str(excinfo.value)
    assert (pm.get("zz9.a"), pm.get(child)) == (1.0, 2.0)


def test_namespace_tree_never_writes_into_a_stored_mapping_value():
    """A mapping-valued parameter key with a key below it is not a sub-tree.

    The tree read raises a ValueError that names both keys and leaves the
    stored mapping as it was registered.
    """
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    _register_probe_parameters(pm, {"zz9.m": {"k": 1}, "zz9.m.b": 2.0})

    with pytest.raises(ValueError) as excinfo:
        pm.get("zz9")

    assert "'zz9.m'" in str(excinfo.value)
    assert "'zz9.m.b'" in str(excinfo.value)
    assert pm.storage.get("zz9.m") == {"k": 1}
    assert pm.get("zz9.m") == {"k": 1}


def test_namespace_tree_builder_rejects_a_key_at_the_place_of_a_sub_tree():
    """The conflict does not depend on the member order.

    namespace_keys returns sorted keys, so a tree read meets a parameter key
    before the keys below it. Members in the other order (the sub-tree first)
    raise the same ValueError instead of replacing the sub-tree by the value.
    """
    from relic.parameters.ops.accessors import _build_namespace_tree

    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    _register_probe_parameters(pm, {"zz9.a": 1.0, "zz9.a.b": 2.0})

    with pytest.raises(ValueError) as excinfo:
        _build_namespace_tree(manager=pm, full_key="zz9", child_keys=["zz9.a.b", "zz9.a"])

    assert "'zz9.a'" in str(excinfo.value)
    assert "'zz9.a.b'" in str(excinfo.value)


class _ProbeReader(BiologicalProcessFunctor):
    """A Functor that reads its parameters from an injected manager."""

    def __init__(self, manager):
        super().__init__(
            name="probe_reader",
            config={},
            metadata=FunctorMetadata(
                name="probe_reader",
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                description="Parameter read probe",
                references=["RELIC architecture overview Section 6.1"],
                input_variables=["x"],
                output_variables=["y"],
            ),
            process_name="probe_reader",
            parameter_manager=manager,
        )

    def forward(self, state, time, dt, **kwargs):
        return {}

    def get_dependencies(self):
        return []


def _read_value(pm, read, key):
    """One value read of key through the named read path of the manager."""
    if read == "get":
        return pm.get(key)
    if read == "query_cache":
        pm.get(key)
        hits = pm._parameter_cache.get_stats()["hits"]
        value = pm.get(key)
        assert pm._parameter_cache.get_stats()["hits"] == hits + 1
        return value
    if read == "units":
        return pm.get(key, units="1")
    if read == "direct_storage":
        pm.set_unit_validation_mode(UnitValidationMode.OFF)
        return pm.get(key)
    if read == "namespace_parameters":
        return pm.get_namespace_parameters(namespace=key.rsplit(".", 1)[0])[key]
    if read == "namespace_tree":
        namespace, name = key.rsplit(".", 1)
        return pm.get(namespace)[name]
    if read == "functor":
        return _ProbeReader(pm).get_parameter_value(key)
    raise AssertionError(f"unknown read path {read}")


@pytest.mark.parametrize(
    "read",
    ["get", "query_cache", "units", "direct_storage", "namespace_parameters", "namespace_tree", "functor"],
)
def test_a_value_read_hands_out_a_copy_of_a_stored_container(read):
    """A mapping or list value leaves the manager as a copy owned by the caller.

    Every value read (manager.get on the slow path, from the query-result
    cache, with units and on the direct storage path; get_namespace_parameters;
    the namespace tree; FunctorBase.get_parameter_value) hands out a copy at
    every depth. Changing it changes neither the stored value nor a later read,
    and runs no change callback: the stored value changes through set_value.
    """
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    _register_probe_parameters(pm, {"zz9.m": {"k": 1, "inner": {"j": [1, 2]}}, "zz9.l": [[1.0, 2.0], [3.0]]})
    changes = []
    pm.add_change_callback(lambda key, old, new: changes.append(key))

    mapping = _read_value(pm, read, "zz9.m")
    sequence = _read_value(pm, read, "zz9.l")
    assert mapping is not pm.storage.get("zz9.m")
    assert sequence is not pm.storage.get("zz9.l")
    mapping["k"] = 5
    mapping["inner"]["j"].append(3)
    sequence[0].append(9.0)
    sequence.append([4.0])

    assert pm.storage.get("zz9.m") == {"k": 1, "inner": {"j": [1, 2]}}
    assert pm.storage.get("zz9.l") == [[1.0, 2.0], [3.0]]
    assert _read_value(pm, read, "zz9.m") == {"k": 1, "inner": {"j": [1, 2]}}
    assert _read_value(pm, read, "zz9.l") == [[1.0, 2.0], [3.0]]
    assert changes == []


def test_a_changed_copy_written_back_is_a_change_of_the_stored_value():
    """Read, change and write back with set_value: the write is a change.

    The change callbacks receive the stored value before and after the write,
    and the next read returns the written value.
    """
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    _register_probe_parameters(pm, {"zz9.m": {"k": 1}})
    changes = []
    pm.add_change_callback(lambda key, old, new: changes.append((key, old, new)))

    value = pm.get("zz9.m")
    value["k"] = 5
    pm.set_value("zz9.m", value)

    assert changes == [("zz9.m", {"k": 1}, {"k": 5})]
    assert pm.get("zz9.m") == {"k": 5}


def test_a_changed_read_of_a_database_list_leaves_the_database_value():
    """A database list read and changed by the caller is not the database's own value.

    The manager takes the database specification on demand; a change to the
    returned list reaches neither the stored value nor the database cache, so
    a read after clear_parameters() returns the declared value again.
    """
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    key = "optimization.uncertainty.percentiles"
    declared = list(pm.database.get_parameter(key).value)

    value = pm.get(key, required=True)
    value[0] = -1.0
    value.append(99.0)

    assert pm.database.get_parameter(key).value == declared
    pm.clear_parameters()
    assert pm.get(key, required=True) == declared


def test_reads_hand_out_tensor_values_and_learnable_bindings_by_reference():
    """The copy covers Python containers only (Sections 4.2 and 6.1).

    A tensor value and a learnable binding leave the manager as the objects
    the manager holds; a tensor inside a mapping keeps its identity in the
    copied mapping.
    """
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    leaf = torch.tensor([3.0])
    pm.register_parameter(ParameterSpec(key="zz9.t", value=torch.tensor([1.0, 2.0]), units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))
    pm.register_parameter(ParameterSpec(key="zz9.theta", value=2.0, units="1", dtype="float32", bounds=(0.0, 10.0), category="calibration_parameter", requires_grad=True))
    pm.register_parameter(ParameterSpec(key="zz9.held", value={"leaf": leaf}, units="1", dtype="dict", bounds=None, category="model_parameter"))

    assert pm.get("zz9.t") is pm.storage.get("zz9.t")
    assert pm.get("zz9.theta", expected_type=torch.Tensor) is pm.nn_parameters["zz9.theta"]
    held = pm.get("zz9.held")
    assert held is not pm.storage.get("zz9.held")
    assert held["leaf"] is leaf


def test_the_specification_lookup_is_the_metadata_the_manager_holds():
    """get_parameter(key) returns the specification object the manager holds.

    It is the metadata lookup and is not copied; its value field is the stored
    value itself, which callers do not change. The value read is manager.get.
    """
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    _register_probe_parameters(pm, {"zz9.m": {"k": 1}})

    spec = pm.get_parameter("zz9.m")
    assert spec is pm.get_parameter("zz9.m")
    assert spec.value is pm.storage.get("zz9.m")
    assert pm.get("zz9.m") == spec.value
    assert pm.get("zz9.m") is not spec.value


def test_a_written_container_is_held_as_the_managers_own_copy():
    """set_value takes a copy of a mapping or list value.

    A caller that changes its container after the write changes neither the
    stored value nor the specification value; the stored value changes
    through the next set_value only.
    """
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    _register_probe_parameters(pm, {"zz9.m": {"k": 1}, "zz9.l": [1.0]})
    mapping = {"k": 2, "inner": [1]}
    sequence = [2.0, 3.0]

    pm.set_value("zz9.m", mapping)
    pm.set_value("zz9.l", sequence)
    mapping["k"] = 3
    mapping["inner"].append(2)
    sequence.append(4.0)

    assert pm.storage.get("zz9.m") == {"k": 2, "inner": [1]}
    assert pm.get_parameter("zz9.m").value == {"k": 2, "inner": [1]}
    assert pm.storage.get("zz9.l") == [2.0, 3.0]
    assert pm.get("zz9.l") == [2.0, 3.0]


def test_namespace_lookup_raises_when_the_database_cannot_be_enumerated(monkeypatch):
    """An unreadable database fails the namespace query instead of returning a partial namespace."""
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)

    def _unreadable():
        raise OSError("parameter database unreadable")

    monkeypatch.setattr(pm.database, "load_all", _unreadable)

    with pytest.raises(OSError, match="parameter database unreadable"):
        pm.get_namespace_parameters(namespace="coupling.scales.temporal.calendars")


def test_parameter_listing_rejects_a_storage_result_that_is_not_a_list():
    """A storage listing outside its List[str] contract fails instead of listing nothing."""
    from types import SimpleNamespace

    from relic.parameters.ops.namespaces import list_parameters

    manager = SimpleNamespace(
        storage=SimpleNamespace(list=lambda namespace=None, pattern=None: None),
        parameters={"soil.decay_rate": SimpleNamespace(category=None)},
    )

    with pytest.raises(TypeError, match="storage.list"):
        list_parameters(manager, namespace="soil")


def test_parameter_listing_applies_the_storage_filters_to_registered_specs():
    """Registered keys follow ParameterStorage.list: namespace selects keys below
    '<namespace>.', pattern is an fnmatch wildcard pattern."""
    from types import SimpleNamespace

    from relic.parameters.ops.namespaces import list_parameters

    registered = {
        key: SimpleNamespace(category=None)
        for key in ("soil.decay_rate", "soil.bulk_density", "soilwater.field_capacity", "air.decay_rate")
    }
    manager = SimpleNamespace(
        storage=SimpleNamespace(list=lambda namespace=None, pattern=None: []),
        parameters=registered,
    )

    assert list_parameters(manager, pattern="*decay*") == ["air.decay_rate", "soil.decay_rate"]
    assert list_parameters(manager, namespace="soil") == ["soil.bulk_density", "soil.decay_rate"]
    assert list_parameters(manager, namespace="soil", pattern="*density") == ["soil.bulk_density"]


def test_manager_listing_pattern_selects_only_matching_registered_parameters():
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    for key in ("c9.pattern.alpha", "c9.pattern.beta", "c9.patternx.gamma"):
        pm.register_parameter(ParameterSpec(key=key, value=1.0, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"))

    assert pm.list(pattern="c9.pattern.a*") == ["c9.pattern.alpha"]
    assert pm.list(namespace="c9.pattern") == ["c9.pattern.alpha", "c9.pattern.beta"]


def test_namespace_parameters_are_the_descendants_of_the_namespace(monkeypatch):
    """As ParameterStorage.list: the keys below '<namespace>.', not the namespace key itself
    and not a sibling prefix."""
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    specs = {
        key: ParameterSpec(key=key, value=1.0, units="1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        for key in ("c9.ns", "c9.ns.child", "c9.nsx.other")
    }
    for spec in specs.values():
        pm.register_parameter(spec)
    monkeypatch.setattr(pm.database, "load_all", lambda: dict(specs))

    assert sorted(pm.get_namespace_parameters(namespace="c9.ns")) == ["c9.ns.child"]


def test_physical_constant_bridge_uses_parameter_manager():
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    hydrate_physical_constants(parameter_manager=pm)
    rho_water = pm.get(
        "relic.physical.constants.water.density_liquid",
        units="kg m-3",
        expected_type=float,
    )
    assert abs(float(WATER_DENSITY) - float(rho_water)) < 1e-6


def test_preload_registers_trainable_parameters():
    reset_global_manager()
    pm = get_global_manager(preload_all=True, warmup_cache=False)
    trainable_keys = [
        key for key, spec in pm.parameters.items() if getattr(spec, "requires_grad", False)
    ]
    assert trainable_keys, "Expected learnable parameters in database"
    for key in trainable_keys[:5]:
        assert key in pm.nn_parameters
        param = pm.nn_parameters[key]
        assert isinstance(param, torch.nn.Parameter)
        assert param.requires_grad


def test_unit_validation_check_only_records_issues():
    reset_global_manager()
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    pm.set_unit_validation_mode(UnitValidationMode.CHECK_ONLY)
    spec = ParameterSpec(
        key="validation.unit_test.bad_units",
        value=1.0,
        units="bad_unit",
        bounds=(0.0, 2.0), dtype="float32", category="model_parameter",
    )
    pm.register_parameter(spec)
    issues = pm.get_unit_validation_issues()
    assert any(issue.get("key") == spec.key for issue in issues)


def test_unit_validation_strict_raises():
    reset_global_manager()
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    pm.set_unit_validation_mode(UnitValidationMode.STRICT)
    spec = ParameterSpec(
        key="validation.unit_test.strict_units",
        value=1.0,
        units="bad_unit",
        bounds=(0.0, 2.0), dtype="float32", category="model_parameter",
    )
    with pytest.raises(ValueError):
        pm.register_parameter(spec)


def test_unit_validation_strict_rechecks_direct_storage_hits():
    reset_global_manager()
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    spec = ParameterSpec(
        key="validation.unit_test.direct_storage_bad_units",
        value=1.0,
        units="bad_unit",
        bounds=(0.0, 2.0), dtype="float32", category="model_parameter",
    )
    pm.register_parameter(spec)
    pm.set_unit_validation_mode(UnitValidationMode.STRICT)

    with pytest.raises(ValueError, match="Unit validation failed"):
        pm.get(spec.key)


def test_late_preload_keeps_what_the_manager_already_holds():
    """A preload after first use completes the manager and replaces nothing it holds.

    A written value and a learnable value moved in place (as an optimizer
    moves it) stay as they are for the specification, the stored value, the
    binding and every query form; a key the manager did not hold is adopted.
    """
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    database = pm.database.load_all()
    key = "hydrosphere.evapotranspiration.sub_freezing_phase"
    spec = pm.get_parameter(key)
    written = next(value for value in spec.allowed_values if value != spec.value)
    pm.set_value(key, written)
    assert pm.get(key) == written

    learnable = next(
        name for name, entry in sorted(database.items())
        if entry.requires_grad and isinstance(entry.value, float)
    )
    pm.get_parameter(learnable)
    binding = pm.nn_parameters[learnable]
    with torch.no_grad():
        binding.add_(0.125)
    moved = binding.detach().clone()
    absent = next(name for name in sorted(database) if name not in pm.parameters)

    pm._preload_parameters(warmup_cache=False)

    assert pm._preloaded is True
    assert absent in pm.parameters and pm.storage.exists(absent)
    assert (
        pm.get_parameter(key).value,
        pm.storage.get(key),
        pm.get(key),
        pm.get(key, units=spec.units),
    ) == (written, written, written, written)
    assert pm.nn_parameters[learnable] is binding
    assert torch.equal(binding.detach(), moved)


def test_initialize_keeps_what_a_manager_preloaded_at_construction_holds():
    """initialize() loads nothing into a manager that was preloaded at construction.

    Such a manager holds the whole database when its constructor returns. A
    value written and a learnable value moved in place (as an optimizer moves
    it) before initialize() stay as they are for the specification, the stored
    value, the binding and every query form, and no key is added.
    """
    pm = UnifiedParameterManager(preload_all=True, warmup_cache=False)
    assert pm._preloaded is True
    assert set(pm.parameters) == set(pm.database.load_all())
    held = set(pm.parameters)
    key = "hydrosphere.evapotranspiration.sub_freezing_phase"
    spec = pm.get_parameter(key)
    written = next(value for value in spec.allowed_values if value != spec.value)
    pm.set_value(key, written)

    learnable = next(
        name for name, entry in sorted(pm.parameters.items())
        if entry.requires_grad and isinstance(entry.value, float)
    )
    binding = pm.nn_parameters[learnable]
    with torch.no_grad():
        binding.add_(0.125)
    moved = binding.detach().clone()

    pm.initialize()

    assert set(pm.parameters) == held
    assert (
        pm.get_parameter(key).value,
        pm.storage.get(key),
        pm.get(key),
        pm.get(key, units=spec.units),
    ) == (written, written, written, written)
    assert pm.nn_parameters[learnable] is binding
    assert torch.equal(binding.detach(), moved)


def test_database_switch_leaves_no_query_result_of_the_previous_database(tmp_path):
    """A database switch replaces the written value for every query form.

    A value written and read once (raw and with units, which fills the query
    cache) is replaced by the value of the new database: after
    configure({"database_path": ...}) the specification, the stored value and
    both query forms agree on it.
    """
    import shutil

    pm = UnifiedParameterManager(preload_all=True, warmup_cache=False)
    pm.initialize()
    key = "hydrosphere.evapotranspiration.sub_freezing_phase"
    spec = pm.get_parameter(key)
    database_value = spec.value
    written = next(value for value in spec.allowed_values if value != database_value)
    pm.set_value(key, written)
    assert (pm.get(key), pm.get(key, units=spec.units)) == (written, written)

    database_copy = tmp_path / "database"
    shutil.copytree(pm.database_path, database_copy)
    pm.configure({"database_path": str(database_copy)})

    assert pm.database_path == database_copy
    assert (
        pm.get_parameter(key).value,
        pm.storage.get(key),
        pm.get(key),
        pm.get(key, units=spec.units),
    ) == (database_value, database_value, database_value, database_value)


@pytest.mark.parametrize("mode", [UnitValidationMode.CHECK_ONLY, UnitValidationMode.STRICT])
def test_database_switch_unit_checks_every_specification_it_adopts(tmp_path, mode):
    """A database switch unit-checks each specification it adopts.

    A unit check recorded for the specification a manager holds under a key
    does not cover the specification of the new database that replaces it:
    after configure({"database_path": ...}) an unknown unit in the new
    database is recorded as an issue (check-only) or fails the switch naming
    the key (strict). The repository database is copied, never edited.
    """
    import shutil

    from relic.core.managers.exceptions import ManagerInitializationError

    key = "hydrosphere.soil_moisture.alpha_vg"
    unknown_units = "relic_probe_unknown_unit"
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    pm.set_unit_validation_mode(mode)
    pm.initialize()
    assert pm.get_parameter(key).units == "m-1"

    database_copy = tmp_path / "database"
    shutil.copytree(pm.database_path, database_copy)
    source = database_copy / "relic" / "hydrosphere" / "soil_moisture.yaml"
    data = source.read_bytes()
    units_at = data.index(b"units: m-1", data.index(f"{key}:".encode()))
    source.write_bytes(
        data[:units_at]
        + f"units: {unknown_units}".encode()
        + data[units_at + len(b"units: m-1"):]
    )
    held_issues = len(pm.get_unit_validation_issues())

    if mode == UnitValidationMode.STRICT:
        with pytest.raises(ManagerInitializationError) as excinfo:
            pm.configure({"database_path": str(database_copy)})
        chain, error = [], excinfo.value
        while error is not None:
            chain.append(error)
            error = error.__cause__ or error.__context__
        assert any(
            isinstance(error, ValueError) and f"Unit validation failed for {key}" in str(error)
            for error in chain
        )
        return

    pm.configure({"database_path": str(database_copy)})

    assert pm.get_parameter(key).units == unknown_units
    switch_issues = [
        issue for issue in pm.get_unit_validation_issues()[held_issues:] if issue["key"] == key
    ]
    assert [issue["units"] for issue in switch_issues] == [unknown_units]


def test_rejected_database_switch_keeps_the_database_the_manager_holds(tmp_path):
    """A database switch that fails leaves the manager on its previous database.

    Under the strict unit mode an unknown unit in the new database fails the
    switch before any specification is adopted. The manager then still names
    the previous database, and after the recovery (reset() and initialize())
    a key it did not hold is loaded from there, not from the rejected
    database. The repository database is copied, never edited.
    """
    import re
    import shutil

    from relic.core.managers.exceptions import ManagerInitializationError

    rejected_key = "hydrosphere.soil_moisture.alpha_vg"
    unheld_key = "hydrosphere.soil_moisture.field_capacity"
    pm = UnifiedParameterManager(preload_all=False, warmup_cache=False)
    pm.set_unit_validation_mode(UnitValidationMode.STRICT)
    pm.initialize()
    pm.get_parameter(rejected_key)
    held_path = pm.database_path
    held_value = pm.database.get_parameter(unheld_key).value
    assert unheld_key not in pm.parameters

    database_copy = tmp_path / "database"
    shutil.copytree(held_path, database_copy)
    source = database_copy / "relic" / "hydrosphere" / "soil_moisture.yaml"
    data = source.read_bytes()
    units_at = data.index(b"units: m-1", data.index(f"  {rejected_key}:".encode()))
    data = data[:units_at] + b"units: relic_probe_unknown_unit" + data[units_at + len(b"units: m-1"):]
    value_line = re.compile(rb"value: [^\r\n]*")
    value_match = value_line.search(data, data.index(f"  {unheld_key}:".encode()))
    source.write_bytes(data[:value_match.start()] + b"value: 0.45" + data[value_match.end():])
    assert held_value != 0.45

    with pytest.raises(ManagerInitializationError):
        pm.configure({"database_path": str(database_copy)})

    assert (pm.database_path, pm.database.base_path) == (held_path, held_path)
    pm.reset()
    pm.initialize()
    assert pm.get_parameter(unheld_key).value == held_value
    assert pm.get_parameter(rejected_key).units == "m-1"


def test_value_written_before_the_first_process_import_reaches_the_process():
    """In a fresh interpreter a process reads the options written before its import.

    The first import of relic.scientific registers the solvers, whose
    construction requests the full preload of the process-wide manager; the
    values written before that import are what the process reads through
    get_parameter_value.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    probe = r'''
import importlib
import json

from relic.parameters import get_global_manager

manager = get_global_manager()
written = {
    "hydrosphere.evapotranspiration.sub_freezing_phase": "ice",
    "hydrosphere.numerical.temperature_bounds": [223.15, 323.15],
}
database = {key: manager.get_parameter(key).value for key in written}
for key, value in written.items():
    manager.set_value(key, value, units=manager.get_parameter(key).units)
module = importlib.import_module("relic.scientific.hydrosphere.evapotranspiration.priestley_taylor")
process = module.PriestleyTaylorET()
seen = {}
for key in written:
    namespace, name = key.rsplit(".", 1)
    value = process.get_parameter_value(name, namespace=namespace, required=True)
    seen[key] = value.tolist() if hasattr(value, "tolist") else value
print(json.dumps({"database": database, "written": written, "process": seen}))
'''
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads([line for line in completed.stdout.splitlines() if line.strip()][-1])
    assert all(result["database"][key] != value for key, value in result["written"].items())
    assert result["process"] == result["written"]
