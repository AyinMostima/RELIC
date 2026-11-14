from types import SimpleNamespace

from relic.core.functor import FunctorBase, FunctorMetadata, ProcessCategory


class _AliasFunctor(FunctorBase):
    def __init__(self) -> None:
        metadata = FunctorMetadata(
            name="alias_functor",
            category=ProcessCategory.SCALE_COUPLING,
            description="Alias mapping test functor",
            references=["alias-test"],
            input_variables=["diagnostics.alias_test.input"],
            output_variables=["diagnostics.alias_test.output"],
        )
        super().__init__(
            name="alias_functor",
            config={"parameters": {}, "dependencies": []},
            metadata=metadata,
        )

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {}


def test_process_category_canonical_value():
    category = ProcessCategory("scale_coupling")
    assert category is ProcessCategory.SCALE_COUPLING


def test_parameter_lookup_canonical_namespace():
    alias_functor = _AliasFunctor()
    full_key, query_name, namespace = alias_functor._resolve_parameter_lookup(
        "standard_temperature",
        "relic.physical.constants.temperature",
    )
    assert full_key == "relic.physical.constants.temperature.standard_temperature"
    assert query_name == "standard_temperature"
    assert namespace == "relic.physical.constants.temperature"


def test_namespace_canonical_value():
    alias_functor = _AliasFunctor()
    full_key, query_name, namespace = alias_functor._resolve_parameter_lookup(
        "gas_constant",
        "relic.physical.constants.universal",
    )
    assert namespace == "relic.physical.constants.universal"
    assert query_name == "gas_constant"
    assert full_key == "relic.physical.constants.universal.gas_constant"


def test_parameter_lookup_uses_a_namespace_declared_by_the_database():
    """The dotted prefix is an absolute namespace when the manager's parameter
    database declares it (namespaces.is_declared_namespace), not merely because
    its root directory exists."""
    from relic.parameters import get_global_manager

    alias_functor = _AliasFunctor()
    database = get_global_manager().database
    alias_functor.get_parameter_manager = lambda: SimpleNamespace(
        parameters={}, database=database
    )
    full_key, query_name, namespace = alias_functor._resolve_parameter_lookup(
        "isotopes.heartbeat.interval_seconds",
        None,
    )
    assert namespace == "isotopes.heartbeat"
    assert query_name == "interval_seconds"
    assert full_key == "isotopes.heartbeat.interval_seconds"
