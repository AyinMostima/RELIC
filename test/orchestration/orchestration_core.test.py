import json
import tempfile
from pathlib import Path

import pytest
import torch

import relic.orchestration.compiler as compiler_module
import relic.orchestration.validator as validator_module
from relic.core.provenance import ProvenanceRecorder
from relic.data.standards.unit_converter import ConversionContext, UnitConverter
from relic.orchestration.adapters import (
    AdapterRegistry,
    AreaBasisTransformAdapter,
    ExecutableAdapterFunctor,
    SupportAggregationAdapter,
    SurfaceFluxToVolumeSourceAdapter,
    TemporalResampleAdapter,
)
from relic.orchestration.compiler import (
    compile_graph,
    plan_to_dict,
    verify_execution_plan_integrity,
    write_run_bundle,
)
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.port_spec import (
    PORT_SUPPORT_FIELDS,
    NodeDefinition,
    PortSpec,
)
from relic.orchestration.ssot_catalog import SSOTCatalog, SSOTEntry
from relic.orchestration.validator import can_connect, validate_graph
from relic.coupling.scales.types import SpatialQuantityBasis
from relic.parameters import get_global_manager
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
from relic.utils.physical_constants import get_physical_constant
from test.fixtures.dummy_functor import (
    DummyConfiguredOutputFunctor,
    DummyIncrementFunctor,
)

_BUCKET_CLASS_TYPE = (
    "Functor:relic.scientific.hydrosphere.soil_moisture.bucket."
    "BucketSoilMoisture"
)
_MEGAN_CLASS_TYPE = (
    "Functor:relic.scientific.biosphere.bvoc.megan_model.MEGANModel"
)
_AGGREGATE_WRITER_A_TYPE = (
    "Functor:test.fixtures.dummy_functor."
    "DummyConfiguredOutputFunctor.aggregate_a"
)
_AGGREGATE_WRITER_B_TYPE = (
    "Functor:test.fixtures.dummy_functor."
    "DummyConfiguredOutputFunctor.aggregate_b"
)


@pytest.fixture
def project_tmp_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(
        dir=project_root,
        prefix=".orchestration-core-",
    ) as tmp_dir:
        yield Path(tmp_dir)


def _build_catalog_and_registry():
    catalog = SSOTCatalog.from_parameter_manager()
    return catalog, NodeRegistry(ssot_catalog=catalog)


def _register_definition(registry, class_type, *, inputs=(), outputs=()):
    registry.register_definition(
        class_type,
        registry.build_from_node_spec(
            class_type,
            input_names=inputs,
            output_names=outputs,
        ),
    )


def _register_bucket_soil_moisture(registry):
    registry.register_functor_instance(BucketSoilMoisture())


def test_registered_type_metadata_is_independent_from_prototype_containers():
    registry = NodeRegistry(SSOTCatalog())
    functor = DummyIncrementFunctor()
    functor.metadata.parameters["nested_contract"] = {"bounds": [0.0, 1.0]}
    definition = registry.register_functor_instance(functor)
    functor.metadata.parameters["nested_contract"]["bounds"][0] = 9.0
    assert definition.metadata["parameters"]["nested_contract"]["bounds"] == [0.0, 1.0]


@pytest.mark.parametrize("field_name", ["metadata", "shape"])
@pytest.mark.parametrize("mutate_source", [False, True])
def test_node_definition_projection_is_independent(field_name, mutate_source):
    definition = NodeDefinition(
        "Functor:projection",
        inputs={"x": PortSpec("x", "x", units="1", shape={"dims": ["cell"]})},
        metadata={"nested": {"values": ["cell"]}},
    )
    projection = definition.to_dict()
    if field_name == "metadata":
        source = definition.metadata["nested"]["values"]
        copied = projection["metadata"]["nested"]["values"]
    else:
        source = definition.inputs["x"].shape["dims"]
        copied = projection["inputs"]["x"]["shape"]["dims"]
    (source if mutate_source else copied).append("layer")
    assert (copied if mutate_source else source) == ["cell"]


@pytest.mark.parametrize("invalid", ["exception", "non_mapping"])
def test_failed_metadata_capture_keeps_the_registered_definition(monkeypatch, invalid):
    registry = NodeRegistry(SSOTCatalog())
    key = "Functor:atomic_metadata"
    previous = registry.register_functor_instance(DummyIncrementFunctor(), key)
    candidate = DummyIncrementFunctor()

    def invalid_metadata():
        if invalid == "exception":
            raise RuntimeError("metadata construction failed")
        return []

    monkeypatch.setattr(candidate.metadata, "to_dict", invalid_metadata)
    with pytest.raises((RuntimeError, TypeError), match="metadata"):
        registry.register_functor_instance(candidate, key)
    assert registry.get_definition(key) is previous


def _inconsistent_link_graph():
    return GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source_a": {
                    "class_type": "State:SourceA",
                    "outputs": ["air_temperature"],
                },
                "source_b": {
                    "class_type": "State:SourceB",
                    "outputs": ["air_temperature"],
                },
                "sink": {
                    "class_type": "Functor:Sink",
                    "inputs": {
                        "air_temperature": {
                            "link": ["source_b", "air_temperature"],
                        }
                    },
                },
            },
            "links": [
                {
                    "from": {"node": "source_a", "port": "air_temperature"},
                    "to": {"node": "sink", "port": "air_temperature"},
                }
            ],
        }
    )


def test_ssot_catalog_loads():
    catalog = SSOTCatalog.from_parameter_manager()
    entry = catalog.get("air_temperature")
    assert entry is not None
    assert entry.units == "K"


def test_node_registry_ports_use_ssot_units():
    catalog, registry = _build_catalog_and_registry()
    definition = registry.build_from_node_spec(
        "Functor:dummy", input_names=["air_temperature"], output_names=["precipitation_flux"]
    )
    assert definition.inputs["air_temperature"].units == "K"
    assert definition.outputs["precipitation_flux"].units == "kg m-2 s-1"


def test_validate_graph_rejects_inconsistent_link_declarations_strict():
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, "Functor:Sink", inputs=["air_temperature"])

    report = validate_graph(_inconsistent_link_graph(), registry, strict=True)

    assert any(issue.code == "link_declaration_mismatch" for issue in report.errors)


def test_validate_graph_reuses_canonical_unit_converter(monkeypatch):
    """The default validation path must not construct a second runtime owner."""
    _, registry = _build_catalog_and_registry()

    def reject_second_owner():
        raise AssertionError("validate_graph must reuse the canonical unit converter")

    monkeypatch.setattr(validator_module, "UnitConverter", reject_second_owner)
    report = validator_module.validate_graph(
        _inconsistent_link_graph(),
        registry,
        strict=False,
    )

    assert report is not None


def test_validate_graph_warns_for_inconsistent_link_declarations_non_strict():
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, "Functor:Sink", inputs=["air_temperature"])

    report = validate_graph(_inconsistent_link_graph(), registry, strict=False)

    assert not any(
        issue.code == "link_declaration_mismatch" for issue in report.errors
    )
    assert any(
        issue.code == "link_declaration_mismatch" for issue in report.warnings
    )


def test_validate_graph_rejects_undeclared_source_output():
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, "Functor:Sink", inputs=["air_temperature"])
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {"class_type": "State:Source", "outputs": []},
                "sink": {
                    "class_type": "Functor:Sink",
                    "inputs": {
                        "air_temperature": {
                            "link": ["source", "air_temperature"],
                        }
                    },
                },
            },
            "links": [
                {
                    "from": {"node": "source", "port": "air_temperature"},
                    "to": {"node": "sink", "port": "air_temperature"},
                }
            ],
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(issue.code == "undeclared_source_output" for issue in report.errors)


def test_validate_graph_rejects_undeclared_target_input():
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, "Functor:Sink", inputs=["air_temperature"])
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": "State:Source",
                    "outputs": ["air_temperature"],
                },
                "sink": {"class_type": "Functor:Sink", "inputs": {}},
            },
            "links": [
                {
                    "from": {"node": "source", "port": "air_temperature"},
                    "to": {"node": "sink", "port": "air_temperature"},
                }
            ],
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(issue.code == "undeclared_target_input" for issue in report.errors)


def test_validate_graph_requires_registered_functor_definition_in_strict_mode():
    _, registry = _build_catalog_and_registry()
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "sink": {
                    "class_type": "Functor:Sink",
                    "outputs": ["air_temperature"],
                }
            },
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(
        issue.code == "unregistered_functor_definition"
        for issue in report.errors
    )


def test_validate_graph_does_not_synthesize_missing_registered_ports():
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, "Functor:Sink")
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": "State:Source",
                    "outputs": ["air_temperature"],
                },
                "sink": {
                    "class_type": "Functor:Sink",
                    "inputs": {
                        "air_temperature": {
                            "link": ["source", "air_temperature"],
                        }
                    },
                },
            },
            "links": [
                {
                    "from": {"node": "source", "port": "air_temperature"},
                    "to": {"node": "sink", "port": "air_temperature"},
                }
            ],
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(issue.code == "missing_port" for issue in report.errors)


def test_graph_validation_and_compile():
    payload = {
        "schema_version": "1.0.0",
        "nodes": {
            "0": {
                "class_type": "State:ExternalForcing",
                "outputs": ["precipitation_flux"],
            },
            "1": {
                    "class_type": "Functor:relic.scientific.hydrosphere.soil_moisture.bucket.BucketSoilMoisture",
                    "inputs": {
                        "precipitation_flux": {"link": ["0", "precipitation_flux"]},
                        "soil_moisture": {"state": "soil_moisture"},
                        "evapotranspiration_flux": {
                            "state": "evapotranspiration_flux"
                        },
                    },
                "outputs": ["soil_moisture"],
            },
        },
        "links": [
            {
                "from": {"node": "0", "port": "precipitation_flux"},
                "to": {"node": "1", "port": "precipitation_flux"},
            }
        ],
        "metadata": {},
    }
    graph = GraphIR.from_dict(payload)
    _, registry = _build_catalog_and_registry()
    _register_bucket_soil_moisture(registry)
    report = validate_graph(graph, registry, strict=True)
    assert report.errors == []
    plan = compile_graph(graph, registry, strict=True)
    assert [step.node_id for step in plan.steps] == ["1"]
    assert plan.steps[0].inputs["precipitation_flux"] == {
        "state": "precipitation_flux"
    }
    assert plan.report.metadata["state_boundary_lowering"] == {
        "lowered_nodes": [
            {
                "node_id": "0",
                "class_type": "State:ExternalForcing",
                "state_keys": ["precipitation_flux"],
            }
        ],
        "lowered_links": [
            {
                "source_node": "0",
                "source_port": "precipitation_flux",
                "target_node": "1",
                "target_port": "precipitation_flux",
                "state_key": "precipitation_flux",
            }
        ],
        "required_initial_state_keys": [
            "evapotranspiration_flux",
            "precipitation_flux",
            "soil_moisture",
        ],
        "executable_node_ids": ["1"],
    }


def test_graph_ir_owns_required_initial_state_identity_query():
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "forcing": {
                    "class_type": "State:ExternalForcing",
                    "outputs": ["air_temperature"],
                },
                "process": {
                    "class_type": "Functor:Process",
                    "inputs": {
                        "air_temperature": {
                            "link": ["forcing", "air_temperature"]
                        },
                        "soil_carbon": {"state": "soil_carbon"},
                    },
                    "outputs": ["carbon_flux"],
                },
            },
        }
    )

    assert graph.required_initial_state_keys() == [
        "air_temperature",
        "soil_carbon",
    ]


@pytest.mark.parametrize(
    ("node_payload", "error_match"),
    [
        (
            {
                "class_type": "State:ExternalForcing",
                "outputs": [],
            },
            "must declare at least one output",
        ),
        (
            {
                "class_type": "State:ExternalForcing",
                "outputs": [" air_temperature "],
            },
            "must not contain leading or trailing whitespace",
        ),
        (
            {
                "class_type": "Functor:test.Example",
                "inputs": {
                    "soil_carbon": {"state": " soil_carbon "},
                },
                "outputs": ["carbon_flux"],
            },
            "must not contain leading or trailing whitespace",
        ),
    ],
)
def test_graph_ir_rejects_ambiguous_required_state_identity(
    node_payload,
    error_match,
):
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {"node": node_payload},
        }
    )

    with pytest.raises(ValueError, match=error_match):
        graph.required_initial_state_keys()


def test_compile_reuses_graph_ir_required_initial_state_identity_owner(monkeypatch):
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "forcing": {
                    "class_type": "State:ExternalForcing",
                    "outputs": ["air_temperature"],
                }
            },
        }
    )
    monkeypatch.setattr(
        GraphIR,
        "required_initial_state_keys",
        lambda self: ["canonical_owner_key"],
    )

    plan = compile_graph(graph, NodeRegistry(SSOTCatalog()), strict=True)

    assert plan.report.metadata["state_boundary_lowering"][
        "required_initial_state_keys"
    ] == ["canonical_owner_key"]


@pytest.mark.parametrize("outputs", [[], ["air_temperature", "air_temperature"]])
def test_external_boundary_lowerer_does_not_repeat_graph_ir_identity_validation(
    monkeypatch,
    outputs,
):
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "forcing": {
                    "class_type": "State:ExternalForcing",
                    "outputs": outputs,
                }
            },
        }
    )
    monkeypatch.setattr(graph, "required_initial_state_keys", lambda: [])

    executable_graph, metadata = (
        compiler_module._lower_external_forcing_boundaries(graph)
    )

    assert executable_graph.nodes == {}
    assert metadata["lowered_nodes"][0]["state_keys"] == sorted(outputs)


def test_compile_rejects_external_forcing_port_alias():
    payload = {
        "schema_version": "1.0.0",
        "nodes": {
            "0": {
                "class_type": "State:ExternalForcing",
                "outputs": ["forcing"],
            },
            "1": {
                "class_type": _BUCKET_CLASS_TYPE,
                "inputs": {
                    "precipitation_flux": {"link": ["0", "forcing"]},
                    "soil_moisture": {"state": "soil_moisture"},
                    "evapotranspiration_flux": {
                        "state": "evapotranspiration_flux"
                    },
                },
                "outputs": ["soil_moisture"],
            },
        },
        "links": [
            {
                "from": {"node": "0", "port": "forcing"},
                "to": {"node": "1", "port": "precipitation_flux"},
            }
        ],
    }
    graph = GraphIR.from_dict(payload)
    _, registry = _build_catalog_and_registry()
    _register_bucket_soil_moisture(registry)

    with pytest.raises(ValueError, match="same-named target input"):
        compile_graph(graph, registry, strict=False)


def test_compile_rejects_external_forcing_runtime_config():
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "forcing": {
                    "class_type": "State:ExternalForcing",
                    "outputs": ["precipitation_flux"],
                    "config": {"resolver": "duplicate_runtime"},
                }
            },
        }
    )

    with pytest.raises(ValueError, match="must not define inputs or config"):
        compile_graph(graph, NodeRegistry(SSOTCatalog()), strict=True)


def test_compile_rejects_unlowered_state_node_kind():
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "identity": {
                    "class_type": "State:Identity",
                    "outputs": ["x"],
                }
            },
        }
    )

    with pytest.raises(ValueError, match="Unsupported executable state node"):
        compile_graph(graph, NodeRegistry(SSOTCatalog()), strict=True)


def test_compile_rejects_unlinked_external_forcing_reference():
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "forcing": NodeSpec(
                class_type="State:ExternalForcing",
                outputs=["precipitation_flux"],
            ),
            "bucket": NodeSpec(
                class_type=_BUCKET_CLASS_TYPE,
                inputs={
                    "precipitation_flux": {
                        "link": ["forcing", "precipitation_flux"]
                    },
                    "soil_moisture": {"state": "soil_moisture"},
                    "evapotranspiration_flux": {
                        "state": "evapotranspiration_flux"
                    },
                },
                outputs=["soil_moisture"],
            ),
        },
        links=[],
    )
    _, registry = _build_catalog_and_registry()
    _register_bucket_soil_moisture(registry)

    with pytest.raises(ValueError, match="still references lowered boundary"):
        compile_graph(graph, registry, strict=False)


def test_compile_rejects_mutually_exclusive_output_closures():
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "first_order_decomposition": {
                    "class_type": _BUCKET_CLASS_TYPE,
                    "outputs": ["carbon_decomposition_rate"],
                },
                "pools_matrix": {
                    "class_type": _MEGAN_CLASS_TYPE,
                    "outputs": ["carbon_decomposition_rate"],
                },
            },
            "links": [],
            "metadata": {},
        }
    )
    _, registry = _build_catalog_and_registry()
    _register_definition(
        registry,
        _BUCKET_CLASS_TYPE,
        outputs=["carbon_decomposition_rate"],
    )
    _register_definition(
        registry,
        _MEGAN_CLASS_TYPE,
        outputs=["carbon_decomposition_rate"],
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(
        issue.code == "mutually_exclusive_output_overlap"
        for issue in report.errors
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        compile_graph(graph, registry, strict=True)


def test_compile_accepts_aggregated_output_writers():
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "mineralization": {
                    "class_type": _AGGREGATE_WRITER_A_TYPE,
                    "outputs": ["nh4_nitrogen"],
                    "config": {"output_name": "nh4_nitrogen"},
                },
                "nitrification": {
                    "class_type": _AGGREGATE_WRITER_B_TYPE,
                    "outputs": ["nh4_nitrogen"],
                    "config": {"output_name": "nh4_nitrogen"},
                },
            },
            "links": [],
            "metadata": {},
        }
    )
    _, registry = _build_catalog_and_registry()
    registry.register_functor_instance(
        DummyConfiguredOutputFunctor(
            config={"output_name": "nh4_nitrogen"}
        ),
        node_type_id=_AGGREGATE_WRITER_A_TYPE,
    )
    registry.register_functor_instance(
        DummyConfiguredOutputFunctor(
            config={"output_name": "nh4_nitrogen"}
        ),
        node_type_id=_AGGREGATE_WRITER_B_TYPE,
    )

    report = validate_graph(graph, registry, strict=True)
    plan = compile_graph(graph, registry, strict=True)

    assert report.errors == []
    assert [step.node_id for step in plan.steps] == ["mineralization", "nitrification"]


def test_compile_rejects_registered_unresolvable_functor_class_in_strict_mode():
    class_type = "Functor:not_a_real_module.MissingFunctor"
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "missing": {
                    "class_type": class_type,
                    "outputs": [],
                }
            },
        }
    )
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, class_type)

    with pytest.raises(ValueError, match="could not be resolved"):
        compile_graph(graph, registry, strict=True)


def test_compile_allows_unresolvable_functor_class_in_non_strict_mode():
    class_type = "Functor:not_a_real_module.MissingFunctor"
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "missing": {
                    "class_type": class_type,
                    "outputs": [],
                }
            },
        }
    )
    _, registry = _build_catalog_and_registry()

    plan = compile_graph(graph, registry, strict=False)

    assert [step.class_type for step in plan.steps] == [class_type]


def test_compiler_seal_is_deterministic_and_binds_graph_config():
    class_type = "Functor:not_a_real_module.SealedFunctor"
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, class_type)

    def _graph(gain: torch.Tensor) -> GraphIR:
        return GraphIR.from_dict(
            {
                "schema_version": "1.0.0",
                "nodes": {
                    "sealed": {
                        "class_type": class_type,
                        "outputs": [],
                        "config": {"gain": gain},
                    }
                },
            }
        )

    first = compile_graph(
        _graph(torch.tensor([1.0], dtype=torch.float64)),
        registry,
        strict=False,
    )
    repeated = compile_graph(
        _graph(torch.tensor([1.0], dtype=torch.float64)),
        registry,
        strict=False,
    )
    changed = compile_graph(
        _graph(torch.tensor([2.0], dtype=torch.float64)),
        registry,
        strict=False,
    )

    first_seal = plan_to_dict(first)["report"]["compiler_seal"]
    repeated_seal = plan_to_dict(repeated)["report"]["compiler_seal"]
    changed_seal = plan_to_dict(changed)["report"]["compiler_seal"]
    assert first_seal == repeated_seal
    assert first_seal["source_graph_sha256"] != changed_seal["source_graph_sha256"]
    assert first_seal["overall_sha256"] != changed_seal["overall_sha256"]
    assert verify_execution_plan_integrity(first, registry)["phase"] == "final"


def test_compiler_seal_rejects_plan_and_referenced_registry_drift():
    class_type = "Functor:not_a_real_module.SealedFunctor"
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, class_type)
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "sealed": {
                    "class_type": class_type,
                    "outputs": [],
                    "config": {"gain": torch.tensor(1.0)},
                }
            },
        }
    )

    plan = compile_graph(graph, registry, strict=False)
    registry.register_definition(
        "Functor:unrelated",
        registry.build_from_node_spec(
            "Functor:unrelated",
            input_names=(),
            output_names=(),
        ),
    )
    assert verify_execution_plan_integrity(plan, registry)["phase"] == "final"

    plan.steps[0].config["gain"] = torch.tensor(2.0)
    with pytest.raises(ValueError, match="plan content"):
        verify_execution_plan_integrity(plan, registry)

    plan = compile_graph(graph, registry, strict=False)
    definition = registry.get_definition(class_type)
    assert definition is not None
    definition.metadata["contract_revision"] = "drifted"
    with pytest.raises(ValueError, match="registry contract"):
        verify_execution_plan_integrity(plan, registry)


def test_compiler_seal_records_missing_definition_without_fail_open_binding():
    class_type = "Functor:not_a_real_module.MissingFunctor"
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "missing": {
                    "class_type": class_type,
                    "outputs": [],
                }
            },
        }
    )
    _, registry = _build_catalog_and_registry()

    plan = compile_graph(graph, registry, strict=False)
    seal = plan_to_dict(plan)["report"]["compiler_seal"]

    assert seal["referenced_node_types"] == [class_type]
    assert seal["registry_bound_node_types"] == []
    assert plan.report.metadata["compile_run_witness"] == {
        "schema_version": "1.0.0",
        "document_kind": "compile_run_witness",
        "passed": False,
        "execution_performed": False,
        "skipped": True,
        "reason": "strict_compilation_required",
    }
    assert verify_execution_plan_integrity(plan, registry)["phase"] == "final"

    _register_definition(registry, class_type)
    with pytest.raises(ValueError, match="registry contract"):
        verify_execution_plan_integrity(plan, registry)


def test_compiler_seal_binds_report_witness_and_compiler_context():
    class_type = "Functor:not_a_real_module.SealedFunctor"
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, class_type)
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "sealed": {
                    "class_type": class_type,
                    "outputs": [],
                }
            },
        }
    )

    default_plan = compile_graph(graph, registry, strict=False)
    contextual_plan = compile_graph(
        graph,
        registry,
        strict=False,
        context=ConversionContext(allow_identity=True),
    )
    default_seal = plan_to_dict(default_plan)["report"]["compiler_seal"]
    contextual_seal = plan_to_dict(contextual_plan)["report"]["compiler_seal"]
    assert (
        default_seal["compiler_options_sha256"]
        != contextual_seal["compiler_options_sha256"]
    )
    assert default_seal["overall_sha256"] != contextual_seal["overall_sha256"]

    default_plan.report.metadata["compile_run_witness"]["reason"] = "tampered"
    with pytest.raises(ValueError, match="plan content"):
        verify_execution_plan_integrity(default_plan, registry)


def test_compiler_seal_distinguishes_source_and_executable_graphs():
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "forcing": {
                    "class_type": "State:ExternalForcing",
                    "outputs": ["precipitation_flux"],
                },
                "bucket": {
                    "class_type": _BUCKET_CLASS_TYPE,
                    "inputs": {
                        "precipitation_flux": {
                            "link": ["forcing", "precipitation_flux"]
                        },
                        "soil_moisture": {"state": "soil_moisture"},
                        "evapotranspiration_flux": {
                            "state": "evapotranspiration_flux"
                        },
                    },
                    "outputs": ["soil_moisture"],
                },
            },
        }
    )
    _, registry = _build_catalog_and_registry()
    _register_bucket_soil_moisture(registry)

    plan = compile_graph(graph, registry, strict=True)
    seal = plan_to_dict(plan)["report"]["compiler_seal"]

    assert seal["source_graph_sha256"] != seal["executable_graph_sha256"]
    assert seal["source_graph_sha256"] == seal["transformed_graph_sha256"]
    assert seal["source_node_types"] == sorted(
        ["State:ExternalForcing", _BUCKET_CLASS_TYPE]
    )
    assert seal["referenced_node_types"] == sorted(
        ["State:ExternalForcing", _BUCKET_CLASS_TYPE]
    )
    assert seal["phase"] == "final"
    assert verify_execution_plan_integrity(plan, registry)["phase"] == "final"


def test_compiler_seal_rejects_non_leaf_tensor_config():
    class_type = "Functor:not_a_real_module.SealedFunctor"
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, class_type)
    leaf = torch.tensor(1.0, requires_grad=True)
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "sealed": {
                    "class_type": class_type,
                    "outputs": [],
                    "config": {"gain": leaf * 2.0},
                }
            },
        }
    )

    with pytest.raises(ValueError, match="non-leaf tensor"):
        compile_graph(graph, registry, strict=False)


def test_compiler_seal_rejects_unsealed_plan_and_registry_identity_drift():
    class_type = "Functor:not_a_real_module.SealedFunctor"
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, class_type)
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "sealed": {
                    "class_type": class_type,
                    "outputs": [],
                }
            },
        }
    )

    plan = compile_graph(graph, registry, strict=False)
    unsealed = type(plan)(steps=plan.steps, report=plan.report)
    with pytest.raises(ValueError, match="no canonical compiler seal"):
        verify_execution_plan_integrity(unsealed, registry)

    registry.unregister_definition(class_type)
    with pytest.raises(ValueError, match="registry contract"):
        verify_execution_plan_integrity(plan, registry)

    mismatched = registry.build_from_node_spec(
        "Functor:not_a_real_module.OtherFunctor",
        input_names=(),
        output_names=(),
    )
    registry.register_definition(class_type, mismatched)
    with pytest.raises(ValueError, match="lookup key"):
        verify_execution_plan_integrity(plan, registry)


def test_compiler_seal_binds_leaf_tensor_identity_without_numpy():
    class_type = "Functor:not_a_real_module.SealedFunctor"
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, class_type)

    def _source_digest(value: torch.Tensor) -> str:
        graph = GraphIR.from_dict(
            {
                "schema_version": "1.0.0",
                "nodes": {
                    "sealed": {
                        "class_type": class_type,
                        "outputs": [],
                        "config": {"gain": value},
                    }
                },
            }
        )
        plan = compile_graph(graph, registry, strict=False)
        return plan_to_dict(plan)["report"]["compiler_seal"][
            "source_graph_sha256"
        ]

    digests = {
        _source_digest(torch.tensor([1.0], dtype=torch.float64)),
        _source_digest(torch.tensor([1.0], dtype=torch.float32)),
        _source_digest(torch.tensor(1.0, dtype=torch.float64)),
        _source_digest(
            torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        ),
        _source_digest(torch.tensor([True], dtype=torch.bool)),
        _source_digest(torch.tensor([1.0], dtype=torch.bfloat16)),
    }
    assert len(digests) == 6


def test_compile_report_includes_unified_audit_summary():
    payload = {
        "schema_version": "1.0.0",
        "nodes": {
            "0": {
                "class_type": "State:ExternalForcing",
                "outputs": ["precipitation_flux"],
            },
            "1": {
                    "class_type": "Functor:relic.scientific.hydrosphere.soil_moisture.bucket.BucketSoilMoisture",
                    "inputs": {
                        "precipitation_flux": {"link": ["0", "precipitation_flux"]},
                        "soil_moisture": {"state": "soil_moisture"},
                        "evapotranspiration_flux": {
                            "state": "evapotranspiration_flux"
                        },
                    },
                "outputs": ["soil_moisture"],
            },
        },
        "links": [
            {
                "from": {"node": "0", "port": "precipitation_flux"},
                "to": {"node": "1", "port": "precipitation_flux"},
            }
        ],
        "metadata": {},
        "assertion": {
            "phi": "precipitation_flux >= 0",
            "op": "0",
            "scope": {"scale": "grid_cell"},
        },
    }
    graph = GraphIR.from_dict(payload)
    _, registry = _build_catalog_and_registry()
    _register_bucket_soil_moisture(registry)
    plan = compile_graph(graph, registry, strict=True, include_topology=True)
    audit = plan.report.metadata.get("audit_summary")
    assert isinstance(audit, dict)
    assert audit["assertion_ok"] is True
    assert audit["conservation_audit"]["enabled"] is True
    assert audit["conservation_audit"]["conservation_vector_count"] >= 0
    topology = plan.report.metadata["topology_signature"]
    assert topology["n_nodes"] == 1
    assert topology["n_links"] == 0
    assert topology["node_ids"] == ["1"]


def test_validate_graph_rejects_dimension_mismatch_against_ssot_in_strict_mode():
    catalog = SSOTCatalog.from_parameter_manager()
    registry = NodeRegistry(ssot_catalog=catalog)
    registry.register_definition(
        "State:BadSource",
        NodeDefinition(
            node_type_id="State:BadSource",
            inputs={},
            outputs={
                "air_temperature": PortSpec(
                    name="air_temperature",
                    standard_name="air_temperature",
                    units="W m-2",
                )
            },
        ),
    )
    registry.register_definition(
        "Functor:BadSink",
        NodeDefinition(
            node_type_id="Functor:BadSink",
            inputs={
                "air_temperature": PortSpec(
                    name="air_temperature",
                    standard_name="air_temperature",
                    units="W m-2",
                )
            },
            outputs={},
        ),
    )

    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "0": {"class_type": "State:BadSource", "outputs": ["air_temperature"]},
                "1": {
                    "class_type": "Functor:BadSink",
                    "inputs": {"air_temperature": {"link": ["0", "air_temperature"]}},
                    "outputs": [],
                },
            },
            "links": [
                {
                    "from": {"node": "0", "port": "air_temperature"},
                    "to": {"node": "1", "port": "air_temperature"},
                }
            ],
            "metadata": {},
        }
    )
    report = validate_graph(graph, registry, strict=True, ssot_catalog=catalog)
    assert any(issue.code == "dimension_inconsistent" for issue in report.errors)
    assert not any(issue.code == "dimension_inconsistent" for issue in report.warnings)

    non_strict_report = validate_graph(
        graph,
        registry,
        strict=False,
        ssot_catalog=catalog,
    )
    assert not any(
        issue.code == "dimension_inconsistent" for issue in non_strict_report.errors
    )
    assert any(
        issue.code == "dimension_inconsistent" for issue in non_strict_report.warnings
    )


def test_run_bundle_written(project_tmp_path):
    payload = {
        "schema_version": "1.0.0",
        "nodes": {
            "0": {
                "class_type": "State:ExternalForcing",
                "outputs": ["precipitation_flux"],
            },
            "1": {
                    "class_type": "Functor:relic.scientific.hydrosphere.soil_moisture.bucket.BucketSoilMoisture",
                    "inputs": {
                        "precipitation_flux": {"link": ["0", "precipitation_flux"]},
                        "soil_moisture": {"state": "soil_moisture"},
                        "evapotranspiration_flux": {
                            "state": "evapotranspiration_flux"
                        },
                    },
                "outputs": ["soil_moisture"],
            },
        },
        "links": [
            {
                "from": {"node": "0", "port": "precipitation_flux"},
                "to": {"node": "1", "port": "precipitation_flux"},
            }
        ],
        "metadata": {},
    }
    graph = GraphIR.from_dict(payload)
    _, registry = _build_catalog_and_registry()
    _register_bucket_soil_moisture(registry)
    plan = compile_graph(graph, registry, strict=True)
    output_dir = project_tmp_path / "run_bundle"
    output_dir.mkdir(parents=True, exist_ok=True)
    forcing_path = project_tmp_path / "forcing.csv"
    forcing_path.write_text("time,value\n0,1\n", encoding="utf-8")
    paths = write_run_bundle(
        graph,
        plan,
        output_dir,
        run_id="g3_PROFILE-001",
        input_files=[forcing_path],
        metadata={
            "publication": {
                "gate": "G3",
                "profile_id": "PROFILE-001",
            }
        },
    )

    assert set(paths) == {
        "workflow",
        "execution_plan",
        "compile_report",
        "runtime_log",
        "env_snapshot",
        "git_status",
        "inputs_manifest",
        "prov",
        "manifest",
        "checksums",
    }
    assert all(Path(path).is_file() for path in paths.values())
    serialized_plan = plan_to_dict(plan)
    assert set(serialized_plan) == {"steps", "report"}
    assert "compiler_seal" not in serialized_plan
    assert "compiler_seal" not in serialized_plan["report"]["metadata"]
    assert json.loads(
        Path(paths["execution_plan"]).read_text(encoding="utf-8")
    ) == serialized_plan["steps"]
    compile_report = json.loads(
        Path(paths["compile_report"]).read_text(encoding="utf-8")
    )
    assert compile_report == serialized_plan["report"]
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["run_id"] == "g3_PROFILE-001"
    assert manifest["metadata"]["schema_version"] == graph.schema_version
    assert manifest["metadata"]["graph_metadata"] == graph.metadata
    assert manifest["metadata"]["publication"] == {
        "gate": "G3",
        "profile_id": "PROFILE-001",
    }
    assert Path(paths["inputs_manifest"]).read_text(encoding="utf-8").endswith(
        f"  {forcing_path.resolve()}\n"
    )
    assert ProvenanceRecorder(output_dir=output_dir).verify_run_bundle() == {
        "checksums_verified": 9,
        "file_count": 10,
        "run_id": "g3_PROFILE-001",
        "schema": "relic.run_bundle",
        "schema_version": "1.0.0",
        "valid": True,
    }


def test_run_bundle_rejects_plan_or_source_graph_drift_before_writing(
    project_tmp_path,
):
    class_type = "Functor:not_a_real_module.SealedFunctor"
    _, registry = _build_catalog_and_registry()
    _register_definition(registry, class_type)

    def _graph() -> GraphIR:
        return GraphIR.from_dict(
            {
                "schema_version": "1.0.0",
                "nodes": {
                    "sealed": {
                        "class_type": class_type,
                        "outputs": [],
                        "config": {"gain": torch.tensor(1.0)},
                    }
                },
            }
        )

    graph = _graph()
    plan = compile_graph(graph, registry, strict=False)
    plan.steps[0].config["gain"] = torch.tensor(2.0)
    plan_drift_dir = project_tmp_path / "plan_drift_bundle"
    with pytest.raises(ValueError, match="plan content"):
        write_run_bundle(graph, plan, plan_drift_dir)
    assert not plan_drift_dir.exists()

    graph = _graph()
    plan = compile_graph(graph, registry, strict=False)
    graph.metadata["tampered"] = True
    graph_drift_dir = project_tmp_path / "graph_drift_bundle"
    with pytest.raises(ValueError, match="source graph"):
        write_run_bundle(graph, plan, graph_drift_dir)
    assert not graph_drift_dir.exists()


def test_can_connect_unit_adapter_suggested():
    source = PortSpec(
        name="generic_length",
        standard_name="generic_length",
        units="m",
    )
    target = PortSpec(
        name="generic_length",
        standard_name="generic_length",
        units="cm",
    )
    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=True)
    assert decision.ok is True
    assert decision.requires_adapter is True
    assert decision.adapter_type == "unit_cast"


def test_can_connect_missing_units_fails_strict():
    source = PortSpec(name="air_temperature", standard_name="air_temperature", units=None)
    target = PortSpec(name="air_temperature", standard_name="air_temperature", units="K")
    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=True)
    assert decision.ok is False
    assert decision.errors


def test_can_connect_rejects_matching_unknown_units_in_strict_mode():
    source = PortSpec(
        name="unknown_quantity",
        standard_name="unknown_quantity",
        units="banana",
    )
    target = PortSpec(
        name="unknown_quantity",
        standard_name="unknown_quantity",
        units="banana",
    )

    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=True)

    assert decision.ok is False
    assert any(issue.code == "unsupported_units" for issue in decision.errors)


@pytest.mark.parametrize(
    ("standard_name", "units", "expected_code"),
    [
        ("unknown_quantity", None, "unsupported_port_units"),
        ("unknown_quantity", "banana", "unsupported_port_units"),
        ("air_temperature", "W m-2", "dimension_inconsistent"),
        ("air_temperature", "degC", "unit_scale_or_offset_unbound"),
    ],
)
def test_validate_graph_rejects_invalid_units_on_unlinked_declared_port(
    standard_name,
    units,
    expected_code,
):
    catalog = SSOTCatalog.from_parameter_manager()
    registry = NodeRegistry(catalog)
    registry.register_definition(
        "State:InvalidUnitSource",
        NodeDefinition(
            node_type_id="State:InvalidUnitSource",
            inputs={},
            outputs={
                "unknown_quantity": PortSpec(
                    name="unknown_quantity",
                    standard_name=standard_name,
                    units=units,
                )
            },
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": "State:InvalidUnitSource",
                    "outputs": ["unknown_quantity"],
                }
            },
            "links": [],
        }
    )

    report = validate_graph(graph, registry, strict=True, ssot_catalog=catalog)

    assert any(issue.code == expected_code for issue in report.errors)


def test_validate_graph_allows_bound_executable_unit_cast_port_units():
    catalog = SSOTCatalog.from_parameter_manager()
    registry = NodeRegistry(catalog)
    registry.register_definition(
        "State:ExternalCelsius",
        NodeDefinition(
            node_type_id="State:ExternalCelsius",
            inputs={},
            outputs={
                "air_temperature_celsius": PortSpec(
                    name="air_temperature_celsius",
                    standard_name="air_temperature",
                    units="degC",
                )
            },
        ),
    )
    adapter_registry = AdapterRegistry()
    adapter = ExecutableAdapterFunctor(
        name="air_temperature_unit_cast",
        adapter_type="unit_cast",
        source_node="external_forcing",
        source_port="air_temperature_celsius",
        target_port="air_temperature",
        config={
            "variable_name": "air_temperature",
            "source_units": "degC",
            "target_units": "K",
        },
    )
    definition = adapter_registry.register_executable_node(
        node_registry=registry,
        node_type_id="Functor:adapter.unit_cast.air_temperature",
        functor=adapter,
        source_port=PortSpec(
            name="air_temperature_celsius",
            standard_name="air_temperature",
            units="degC",
        ),
        target_port=PortSpec(
            name="air_temperature",
            standard_name="air_temperature",
            units="K",
            semantics="diagnostic",
            write_mode="derived",
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "external_forcing": {
                    "class_type": "State:ExternalCelsius",
                    "outputs": ["air_temperature_celsius"],
                },
                "adapter": {
                    "class_type": "Functor:adapter.unit_cast.air_temperature",
                    "inputs": {
                        "air_temperature_celsius": {
                            "link": ["external_forcing", "air_temperature_celsius"]
                        }
                    },
                    "outputs": ["air_temperature"],
                    "config": definition.metadata["adapter_contract"],
                }
            },
            "links": [
                {
                    "from": {
                        "node": "external_forcing",
                        "port": "air_temperature_celsius",
                    },
                    "to": {
                        "node": "adapter",
                        "port": "air_temperature_celsius",
                    },
                }
            ],
        }
    )

    report = validate_graph(graph, registry, strict=True, ssot_catalog=catalog)

    assert not any(
        issue.code == "unit_scale_or_offset_unbound" for issue in report.errors
    )


def test_validate_graph_rejects_forged_unit_cast_runtime_metadata():
    catalog = SSOTCatalog.from_parameter_manager()
    registry = NodeRegistry(catalog)
    registry.register_definition(
        "State:ExternalCelsius",
        NodeDefinition(
            node_type_id="State:ExternalCelsius",
            inputs={},
            outputs={
                "air_temperature_celsius": PortSpec(
                    name="air_temperature_celsius",
                    standard_name="air_temperature",
                    units="degC",
                )
            },
        ),
    )
    contract = {
        "adapter_type": "unit_cast",
        "source_node": "external_forcing",
        "source_port": "air_temperature_celsius",
        "target_port": "air_temperature",
        "variable_name": "air_temperature",
        "source_units": "degC",
        "target_units": "K",
    }
    registry.register_definition(
        "Operator:ForgedUnitCast",
        NodeDefinition(
            node_type_id="Operator:ForgedUnitCast",
            inputs={
                "air_temperature_celsius": PortSpec(
                    name="air_temperature_celsius",
                    standard_name="air_temperature",
                    units="degC",
                )
            },
            outputs={
                "air_temperature": PortSpec(
                    name="air_temperature",
                    standard_name="air_temperature",
                    units="K",
                )
            },
            metadata={
                "runtime_class": "test.fake.ForgedUnitCast",
                "adapter_contract": contract,
            },
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "external_forcing": {
                    "class_type": "State:ExternalCelsius",
                    "outputs": ["air_temperature_celsius"],
                },
                "adapter": {
                    "class_type": "Operator:ForgedUnitCast",
                    "inputs": {
                        "air_temperature_celsius": {
                            "link": [
                                "external_forcing",
                                "air_temperature_celsius",
                            ]
                        }
                    },
                    "outputs": ["air_temperature"],
                    "config": contract,
                },
            },
            "links": [
                {
                    "from": {
                        "node": "external_forcing",
                        "port": "air_temperature_celsius",
                    },
                    "to": {
                        "node": "adapter",
                        "port": "air_temperature_celsius",
                    },
                }
            ],
        }
    )

    report = validate_graph(graph, registry, strict=True, ssot_catalog=catalog)

    assert any(
        issue.code == "unit_scale_or_offset_unbound" for issue in report.errors
    )


def test_validate_graph_rejects_unexecutable_runtime_bound_unit_cast_contract():
    # A liquid-water depth rate has no molar-flux rule (mm day-1 -> kg m-2 s-1 is
    # executable through the CF liquid-water-equivalent rule), so this cast is not.
    catalog = SSOTCatalog.from_parameter_manager()
    registry = NodeRegistry(catalog)
    registry.register_definition(
        "State:ExternalDailyPrecipitation",
        NodeDefinition(
            node_type_id="State:ExternalDailyPrecipitation",
            inputs={},
            outputs={
                "precipitation_flux": PortSpec(
                    name="precipitation_flux",
                    standard_name="precipitation_flux",
                    units="mm day-1",
                )
            },
        ),
    )
    contract = {
        "adapter_type": "unit_cast",
        "source_node": "external_forcing",
        "source_port": "precipitation_flux",
        "target_port": "precipitation_flux_si",
        "variable_name": "precipitation_flux",
        "source_units": "mm day-1",
        "target_units": "mol m-2 s-1",
    }
    registry.register_definition(
        "Functor:adapter.unit_cast.precipitation_flux",
        NodeDefinition(
            node_type_id="Functor:adapter.unit_cast.precipitation_flux",
            inputs={
                "precipitation_flux": PortSpec(
                    name="precipitation_flux",
                    standard_name="precipitation_flux",
                    units="mm day-1",
                )
            },
            outputs={
                "precipitation_flux_si": PortSpec(
                    name="precipitation_flux_si",
                    standard_name="precipitation_flux",
                    units="mol m-2 s-1",
                    semantics="diagnostic",
                    write_mode="derived",
                )
            },
            metadata={
                "runtime_class": (
                    "relic.orchestration.adapters.ExecutableAdapterFunctor"
                ),
                "adapter_contract": contract,
            },
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "external_forcing": {
                    "class_type": "State:ExternalDailyPrecipitation",
                    "outputs": ["precipitation_flux"],
                },
                "adapter": {
                    "class_type": "Functor:adapter.unit_cast.precipitation_flux",
                    "inputs": {
                        "precipitation_flux": {
                            "link": ["external_forcing", "precipitation_flux"]
                        }
                    },
                    "outputs": ["precipitation_flux_si"],
                    "config": contract,
                },
            },
            "links": [
                {
                    "from": {
                        "node": "external_forcing",
                        "port": "precipitation_flux",
                    },
                    "to": {"node": "adapter", "port": "precipitation_flux"},
                }
            ],
        }
    )

    report = validate_graph(graph, registry, strict=True, ssot_catalog=catalog)

    assert any(issue.code == "unit_cast_not_executable" for issue in report.errors)


def test_adapter_registry_rejects_unexecutable_unit_cast_contract():
    # A liquid-water depth rate has no molar-flux rule (mm day-1 -> kg m-2 s-1 is
    # executable through the CF liquid-water-equivalent rule), so this cast is not.
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    adapter = ExecutableAdapterFunctor(
        name="precipitation_unit_cast",
        adapter_type="unit_cast",
        source_node="external_forcing",
        source_port="precipitation_flux",
        target_port="precipitation_flux_si",
        config={
            "variable_name": "precipitation_flux",
            "source_units": "mm day-1",
            "target_units": "mol m-2 s-1",
        },
    )

    with pytest.raises(ValueError, match="not executable"):
        AdapterRegistry().register_executable_node(
            node_registry=registry,
            node_type_id="Functor:adapter.unit_cast.precipitation_flux",
            functor=adapter,
            source_port=PortSpec(
                name="precipitation_flux",
                standard_name="precipitation_flux",
                units="mm day-1",
            ),
            target_port=PortSpec(
                name="precipitation_flux_si",
                standard_name="precipitation_flux",
                units="mol m-2 s-1",
                semantics="diagnostic",
                write_mode="derived",
            ),
        )


def test_can_connect_quantity_mismatch():
    source = PortSpec(
        name="energy_flux",
        standard_name="surface_downwelling_shortwave_flux_in_air",
        units="W m-2",
        quantity="energy",
    )
    target = PortSpec(
        name="water_flux",
        standard_name="precipitation_flux",
        units="kg m-2 s-1",
        quantity="mass",
    )
    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=True)
    assert decision.ok is False
    assert any(issue.code == "quantity_mismatch" for issue in decision.errors)


def test_can_connect_element_mismatch():
    source = PortSpec(
        name="carbon_flux",
        standard_name="gross_primary_productivity_of_carbon",
        units="kg C m-2 s-1",
        element="c",
    )
    target = PortSpec(
        name="nitrogen_flux",
        standard_name="nitrogen_flux",
        units="kg N m-2 s-1",
        element="n",
    )
    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=True)
    assert decision.ok is False
    assert any(issue.code == "element_mismatch" for issue in decision.errors)


def test_can_connect_cf_standard_name_mismatch():
    source = PortSpec(
        name="air_temperature",
        standard_name="air_temperature",
        cf_standard_name="air_temperature",
        units="K",
    )
    target = PortSpec(
        name="surface_temperature",
        standard_name="air_temperature",
        cf_standard_name="surface_temperature",
        units="K",
    )
    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=True)
    assert decision.ok is False
    assert any(issue.code == "cf_standard_name_mismatch" for issue in decision.errors)


def _identity_port(
    local_name,
    cf_name=None,
    *,
    units="K",
):
    return PortSpec(
        name=str(local_name),
        standard_name=local_name,
        cf_standard_name=cf_name,
        units=units,
    )


@pytest.mark.parametrize(
    (
        "source_local",
        "source_cf",
        "target_local",
        "target_cf",
        "expected_code",
    ),
    [
        ("temperature", None, "temperature", None, None),
        ("source_temperature", None, "target_temperature", None, "standard_name_mismatch"),
        ("source_temperature", "air_temperature", "target_temperature", "air_temperature", None),
        ("temperature", "air_temperature", "temperature", "surface_temperature", "cf_standard_name_mismatch"),
        ("source_temperature", "air_temperature", "target_temperature", "surface_temperature", "cf_standard_name_mismatch"),
        ("temperature", "air_temperature", "temperature", None, None),
        ("temperature", None, "temperature", "air_temperature", None),
        ("source_temperature", "air_temperature", "target_temperature", None, "standard_name_mismatch"),
        ("source_temperature", None, "target_temperature", "air_temperature", "standard_name_mismatch"),
        ("source_local", "target_local", "target_local", None, "standard_name_mismatch"),
    ],
    ids=(
        "both-none-same-local",
        "both-none-different-local",
        "same-explicit-cf-different-local",
        "different-explicit-cf-same-local",
        "different-explicit-cf-different-local",
        "source-cf-only-same-local",
        "target-cf-only-same-local",
        "source-cf-only-different-local",
        "target-cf-only-different-local",
        "source-cf-must-not-cross-match-target-local",
    ),
)
def test_can_connect_local_and_optional_cf_identity_truth_table(
    source_local,
    source_cf,
    target_local,
    target_cf,
    expected_code,
):
    source = _identity_port(source_local, source_cf)
    target = _identity_port(target_local, target_cf)

    decision = can_connect(
        source,
        target,
        unit_converter=UnitConverter(),
        strict=True,
    )

    if expected_code is None:
        assert decision.ok is True
        assert decision.errors == []
        assert decision.warnings == []
    else:
        assert decision.ok is False
        assert [issue.code for issue in decision.errors] == [expected_code]
        issue = decision.errors[0]
        if expected_code == "standard_name_mismatch":
            assert issue.got == source_local
            assert issue.expected == target_local
            assert "CF" not in issue.message
        else:
            assert issue.got == source_cf
            assert issue.expected == target_cf
            assert "CF standard name" in issue.message


def test_can_connect_compares_explicit_cf_names_exactly():
    source = _identity_port("temperature", "air_temperature")
    target = _identity_port("temperature", "Air_Temperature")

    decision = can_connect(
        source,
        target,
        unit_converter=UnitConverter(),
        strict=True,
    )

    assert decision.ok is False
    assert [issue.code for issue in decision.errors] == [
        "cf_standard_name_mismatch"
    ]


@pytest.mark.parametrize("invalid_name", [None, "", "   ", " local", "local ", True, 7])
def test_can_connect_rejects_malformed_local_standard_names(invalid_name):
    source = _identity_port(invalid_name)
    target = _identity_port("local")

    with pytest.raises((TypeError, ValueError), match="standard_name"):
        can_connect(source, target, unit_converter=UnitConverter(), strict=True)


@pytest.mark.parametrize("invalid_cf", ["", "   ", " air_temperature", "air_temperature ", True, 7])
def test_can_connect_rejects_malformed_explicit_cf_names(invalid_cf):
    source = _identity_port("temperature", invalid_cf)
    target = _identity_port("temperature")

    with pytest.raises((TypeError, ValueError), match="cf_standard_name"):
        can_connect(source, target, unit_converter=UnitConverter(), strict=True)


@pytest.mark.parametrize(
    ("source", "target", "expected_code"),
    [
        (
            _identity_port("source_temperature"),
            _identity_port("target_temperature"),
            "standard_name_mismatch",
        ),
        (
            _identity_port("temperature", "air_temperature"),
            _identity_port("temperature", "surface_temperature"),
            "cf_standard_name_mismatch",
        ),
    ],
)
def test_nonstrict_identity_warning_makes_semantic_audit_fail(
    source,
    target,
    expected_code,
):
    decision = can_connect(
        source,
        target,
        unit_converter=UnitConverter(),
        strict=False,
    )

    assert decision.ok is True
    assert decision.errors == []
    assert [issue.code for issue in decision.warnings] == [expected_code]
    audit = validator_module.build_audit_summary(decision)
    assert audit["syntax_ok"] is True
    assert audit["semantic_ok"] is False


def test_same_explicit_cf_does_not_bypass_unit_gate():
    source = _identity_port("source_temperature", "air_temperature", units="K")
    target = _identity_port("target_temperature", "air_temperature", units="Pa")

    decision = can_connect(
        source,
        target,
        unit_converter=UnitConverter(),
        strict=True,
    )

    assert decision.ok is False
    assert any(issue.code == "unit_inconsistent" for issue in decision.errors)


def test_can_connect_rejects_structural_form_mismatch_in_strict_mode():
    source = PortSpec(
        name="tracer_flux",
        standard_name="tracer_flux",
        units="kg s-1",
        form_degree=2,
        primality="primal",
        extensivity="integral",
        complex_id="atmosphere_grid",
    )
    target = PortSpec(
        name="tracer_flux",
        standard_name="tracer_flux",
        units="kg s-1",
        form_degree=1,
        primality="dual",
        extensivity="extensive",
        complex_id="land_grid",
    )

    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=True)

    assert decision.ok is False
    assert any(issue.code == "form_degree_mismatch" for issue in decision.errors)


def test_can_connect_rejects_partial_structural_metadata_in_strict_mode():
    source = PortSpec(
        name="water_flux",
        standard_name="water_flux",
        units="kg s-1",
        form_degree=1,
        primality="primal",
        extensivity="integral",
        complex_id="soil_column",
    )
    target = PortSpec(
        name="water_flux",
        standard_name="water_flux",
        units="kg s-1",
    )

    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=True)

    assert decision.ok is False
    assert any(issue.code == "missing_structural_port_metadata" for issue in decision.errors)


def test_can_connect_rejects_context_self_attested_chain_map_for_structural_mismatch():
    source = PortSpec(
        name="tracer_storage",
        standard_name="tracer_storage",
        units="kg",
        form_degree=3,
        primality="primal",
        extensivity="extensive",
        complex_id="atmosphere_grid",
    )
    target = PortSpec(
        name="tracer_storage",
        standard_name="tracer_storage",
        units="kg",
        form_degree=2,
        primality="primal",
        extensivity="extensive",
        complex_id="land_surface",
    )
    context = {
        "structural_chain_map": {
            "passed": True,
            "source_complex_id": "atmosphere_grid",
            "target_complex_id": "land_surface",
        }
    }

    decision = can_connect(
        source,
        target,
        unit_converter=UnitConverter(),
        context=context,
        strict=True,
    )

    assert decision.ok is False
    assert decision.requires_adapter is False
    assert any(issue.code == "form_degree_mismatch" for issue in decision.errors)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("shape", {"dimensions": ["cell"]}),
        ("spatial_kind", "grid_cell"),
        ("area_basis", "ground_area"),
        ("temporal_basis", "interval_mean"),
        ("vertical_basis", "layer_integral"),
    ],
)
def test_can_connect_rejects_one_sided_support_metadata(field_name, value):
    source = PortSpec(
        name="isoprene_emission",
        standard_name="isoprene_emission",
        units="kg m-2 s-1",
        **{field_name: value},
    )
    target = PortSpec(
        name="isoprene_emission",
        standard_name="isoprene_emission",
        units="kg m-2 s-1",
    )

    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=True)

    assert decision.ok is False
    assert decision.requires_adapter is False
    assert any(
        issue.code == "missing_port_support_metadata"
        for issue in decision.errors
    )


def test_can_connect_distinguishes_explicit_null_from_undeclared_support():
    source = PortSpec(
        name="scalar_diagnostic",
        standard_name="scalar_diagnostic",
        units="1",
        declared_support_fields=frozenset({"area_basis"}),
    )
    target = PortSpec(
        name="scalar_diagnostic",
        standard_name="scalar_diagnostic",
        units="1",
    )

    decision = can_connect(
        source,
        target,
        unit_converter=UnitConverter(),
        strict=True,
    )

    assert decision.ok is False
    assert any(
        issue.code == "missing_port_support_metadata"
        for issue in decision.errors
    )


def test_can_connect_rejects_explicit_null_against_non_null_support():
    source = PortSpec(
        name="scalar_diagnostic",
        standard_name="scalar_diagnostic",
        units="1",
        declared_support_fields=frozenset({"area_basis"}),
    )
    target = PortSpec(
        name="scalar_diagnostic",
        standard_name="scalar_diagnostic",
        units="1",
        area_basis="ground_area",
    )

    decision = can_connect(
        source,
        target,
        unit_converter=UnitConverter(),
        strict=True,
    )

    assert decision.ok is False
    assert any(
        issue.code == "area_basis_mismatch" for issue in decision.errors
    )


def test_can_connect_accepts_matching_explicit_null_support():
    source = PortSpec(
        name="scalar_diagnostic",
        standard_name="scalar_diagnostic",
        units="1",
        declared_support_fields=frozenset({"area_basis"}),
    )
    target = PortSpec(
        name="scalar_diagnostic",
        standard_name="scalar_diagnostic",
        units="1",
        declared_support_fields=frozenset({"area_basis"}),
    )

    decision = can_connect(
        source,
        target,
        unit_converter=UnitConverter(),
        strict=True,
    )

    assert decision.ok is True
    assert decision.errors == []


def _single_output_support_graph(
    port: PortSpec,
    *,
    publication: bool,
) -> tuple[GraphIR, NodeRegistry]:
    registry = NodeRegistry(SSOTCatalog())
    registry.register_definition(
        "State:SupportSource",
        NodeDefinition(
            node_type_id="State:SupportSource",
            outputs={port.name: port},
        ),
    )
    metadata = {"profile_id": "PROFILE-TEST"} if publication else {}
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": "State:SupportSource",
                    "outputs": [port.name],
                }
            },
            "metadata": metadata,
        }
    )
    return graph, registry


def test_publication_profile_rejects_unlinked_supportless_output():
    graph, registry = _single_output_support_graph(
        PortSpec(
            name="supportless_output",
            standard_name="supportless_output",
        ),
        publication=True,
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(
        issue.code == "publication_port_support_incomplete"
        for issue in report.errors
    )


def test_non_publication_graph_preserves_supportless_output_compatibility():
    graph, registry = _single_output_support_graph(
        PortSpec(
            name="supportless_output",
            standard_name="supportless_output",
        ),
        publication=False,
    )

    report = validate_graph(graph, registry, strict=True)

    assert not any(
        issue.code.startswith("publication_port_")
        for issue in report.errors
    )


@pytest.mark.parametrize(
    ("shape", "expected_error"),
    [
        (None, "publication_port_shape_invalid"),
        ({}, "publication_port_shape_invalid"),
        ({"dimensions": ["", "cell"]}, "publication_port_shape_invalid"),
        ({"dimensions": []}, None),
        ({"dimensions": ["cell"]}, None),
    ],
)
def test_publication_profile_validates_declared_port_shape(
    shape,
    expected_error,
):
    graph, registry = _single_output_support_graph(
        PortSpec(
            name="declared_output",
            standard_name="declared_output",
            shape=shape,
            declared_support_fields=frozenset(PORT_SUPPORT_FIELDS),
        ),
        publication=True,
    )

    report = validate_graph(graph, registry, strict=True)
    shape_errors = [
        issue
        for issue in report.errors
        if issue.code == "publication_port_shape_invalid"
    ]

    if expected_error is None:
        assert shape_errors == []
    else:
        assert [issue.code for issue in shape_errors] == [expected_error]


@pytest.mark.parametrize(
    ("field_name", "source_value", "target_value"),
    [
        (
            "shape",
            {"dimensions": ["cell"]},
            {"dimensions": ["canopy_layer", "cell"]},
        ),
        ("spatial_kind", "grid_cell", "footprint"),
        ("area_basis", "ground_area", "leaf_area"),
        ("temporal_basis", "instantaneous", "interval_mean"),
        ("vertical_basis", "layer_integral", "height_level"),
    ],
)
def test_can_connect_rejects_support_mismatch(
    field_name,
    source_value,
    target_value,
):
    source = PortSpec(
        name="isoprene_emission",
        standard_name="isoprene_emission",
        units="kg m-2 s-1",
        **{field_name: source_value},
    )
    target = PortSpec(
        name="isoprene_emission",
        standard_name="isoprene_emission",
        units="kg m-2 s-1",
        **{field_name: target_value},
    )

    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=True)

    assert decision.ok is False
    assert decision.requires_adapter is False
    assert any(
        issue.code == f"{field_name}_mismatch" for issue in decision.errors
    )


def test_can_connect_warns_for_support_mismatch_non_strict():
    source = PortSpec(
        name="isoprene_emission",
        standard_name="isoprene_emission",
        units="kg m-2 s-1",
        area_basis="ground_area",
    )
    target = PortSpec(
        name="isoprene_emission",
        standard_name="isoprene_emission",
        units="kg m-2 s-1",
        area_basis="leaf_area",
    )

    decision = can_connect(source, target, unit_converter=UnitConverter(), strict=False)

    assert decision.ok is True
    assert decision.requires_adapter is False
    assert any(issue.code == "area_basis_mismatch" for issue in decision.warnings)


def test_temporal_resample_adapter_computes_weighted_mean_and_integral():
    adapter = TemporalResampleAdapter()
    values = torch.tensor([1.0, 3.0])
    durations = torch.tensor([1.0, 3.0])

    mean_result = adapter.apply(
        values,
        method="interval_mean",
        sample_durations=durations,
        axis=0,
    )
    integral_result = adapter.apply(
        values,
        method="interval_integral",
        sample_durations=durations,
        axis=0,
    )

    assert torch.allclose(mean_result.value, torch.tensor(2.5))
    assert torch.allclose(integral_result.value, torch.tensor(10.0))
    assert mean_result.metadata["method"] == "interval_mean"
    assert integral_result.metadata["method"] == "interval_integral"


def _register_current_step_temporal_adapter(
    *,
    config_overrides=None,
    source_dimensions=None,
    target_dimensions=None,
    source_temporal_basis="instantaneous",
    target_temporal_basis="interval_mean",
):
    registry = NodeRegistry(SSOTCatalog())
    config = {
        "sampling_mode": "current_step",
        "method": "interval_mean",
        "sample_durations_state": "observation_window_duration",
    }
    config.update(config_overrides or {})
    adapter = ExecutableAdapterFunctor(
        name="current_step_temporal_adapter",
        adapter_type="temporal_resample",
        source_node="source",
        source_port="instantaneous_flux",
        target_port="interval_flux",
        config=config,
    )
    definition = AdapterRegistry().register_executable_node(
        node_registry=registry,
        node_type_id="Functor:adapter.temporal.current_step",
        functor=adapter,
        source_port=PortSpec(
            name="instantaneous_flux",
            standard_name="instantaneous_flux",
            units="mol m-2 s-1",
            shape={"dimensions": source_dimensions or ["cell"]},
            spatial_kind="grid_cell",
            area_basis="ground_area",
            temporal_basis=source_temporal_basis,
            vertical_basis="none",
        ),
        target_port=PortSpec(
            name="interval_flux",
            standard_name="interval_flux",
            units="mol m-2 s-1",
            shape={"dimensions": target_dimensions or ["cell"]},
            semantics="diagnostic",
            write_mode="derived",
            spatial_kind="grid_cell",
            area_basis="ground_area",
            temporal_basis=target_temporal_basis,
            vertical_basis="none",
        ),
        context_ports={
            "observation_window_duration": PortSpec(
                name="observation_window_duration",
                standard_name="observation_window_duration",
                units="s",
                shape={"dimensions": []},
                semantics="forcing",
            )
        },
    )
    adapter.bind_node_definition(definition)
    return adapter, definition


def test_current_step_temporal_adapter_is_identity_with_dt_guard_and_gradient():
    adapter, definition = _register_current_step_temporal_adapter()
    values = torch.tensor(
        [1.25, 3.5],
        dtype=torch.float64,
        requires_grad=True,
    )

    outputs = adapter.forward(
        state={"observation_window_duration": torch.tensor(1800.0)},
        time=0.0,
        dt=1800.0,
        subprocess_outputs={
            "source": {"instantaneous_flux": values},
        },
    )

    interval_flux = outputs["interval_flux"].value
    torch.testing.assert_close(interval_flux, values, rtol=0.0, atol=0.0)
    assert definition.inputs["instantaneous_flux"].shape == {
        "dimensions": ["cell"]
    }
    assert definition.outputs["interval_flux"].shape == {
        "dimensions": ["cell"]
    }
    assert adapter.last_adapter_metadata["sampling_mode"] == "current_step"
    assert adapter.last_adapter_metadata["method"] == "interval_mean"
    interval_flux.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


def test_current_step_temporal_adapter_rejects_window_dt_mismatch():
    adapter, _ = _register_current_step_temporal_adapter()

    with pytest.raises(ValueError, match="must equal the runtime dt"):
        adapter.forward(
            state={"observation_window_duration": torch.tensor(900.0)},
            time=0.0,
            dt=1800.0,
            subprocess_outputs={
                "source": {
                    "instantaneous_flux": torch.tensor([1.0, 2.0]),
                }
            },
        )


@pytest.mark.parametrize(
    ("config_overrides", "source_temporal_basis", "target_temporal_basis", "match"),
    [
        ({"method": "interval_integral"}, "instantaneous", "interval_mean", "interval_mean"),
        ({"axis": 0}, "instantaneous", "interval_mean", "must not declare axis"),
        ({}, "interval_mean", "interval_mean", "instantaneous source"),
        ({}, "instantaneous", "instantaneous", "interval_mean target"),
    ],
)
def test_current_step_temporal_adapter_rejects_unsupported_contracts(
    config_overrides,
    source_temporal_basis,
    target_temporal_basis,
    match,
):
    with pytest.raises(ValueError, match=match):
        _register_current_step_temporal_adapter(
            config_overrides=config_overrides,
            source_temporal_basis=source_temporal_basis,
            target_temporal_basis=target_temporal_basis,
        )


def test_current_step_temporal_adapter_requires_same_shape():
    with pytest.raises(ValueError, match="identical source and target shapes"):
        _register_current_step_temporal_adapter(target_dimensions=["site"])


def test_executable_temporal_adapter_validates_named_axis_reduction():
    def register(
        source_dimensions: list[str],
        target_dimensions: list[str],
    ) -> NodeDefinition:
        registry = NodeRegistry(SSOTCatalog())
        adapter = ExecutableAdapterFunctor(
            name="temporal_adapter",
            adapter_type="temporal_resample",
            source_node="source",
            source_port="instantaneous_flux",
            target_port="interval_flux",
            config={"method": "interval_mean", "axis": 0},
        )
        return AdapterRegistry().register_executable_node(
            node_registry=registry,
            node_type_id="Functor:adapter.temporal.test",
            functor=adapter,
            source_port=PortSpec(
                name="instantaneous_flux",
                standard_name="instantaneous_flux",
                units="mol m-2 s-1",
                shape={"dimensions": source_dimensions},
            ),
            target_port=PortSpec(
                name="interval_flux",
                standard_name="interval_flux",
                units="mol m-2 s-1",
                shape={"dimensions": target_dimensions},
            ),
        )

    definition = register(["sample", "cell"], ["cell"])
    assert definition.inputs["instantaneous_flux"].shape == {
        "dimensions": ["sample", "cell"]
    }
    assert definition.outputs["interval_flux"].shape == {
        "dimensions": ["cell"]
    }

    with pytest.raises(ValueError, match="named temporal axis"):
        register(["cell"], [])
    with pytest.raises(ValueError, match="remove exactly"):
        register(["sample", "cell"], ["sample", "cell"])


def test_temporal_then_support_adapter_preserves_shape_and_gradients():
    values = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    durations = torch.tensor(
        [1.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = torch.tensor(
        [0.25, 0.75],
        dtype=torch.float64,
        requires_grad=True,
    )

    temporal = TemporalResampleAdapter().apply(
        values,
        method="interval_mean",
        sample_durations=durations,
        axis=0,
    )
    support = SupportAggregationAdapter().apply(
        temporal.value,
        weights=weights,
        strategy="area_weighted_mean",
        variable_name="isoprene_emission",
        source_scale="grid_cell",
        target_scale="footprint",
        quantity_basis="area_density",
        source_port_spec=_support_port("grid_flux"),
        target_port_spec=_support_port(
            "footprint_flux", dimensions=(), spatial_kind="footprint"
        ),
        conservation_tolerance=_support_tolerance(),
    )

    torch.testing.assert_close(
        temporal.value,
        torch.tensor([2.5, 3.5], dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        support.value,
        torch.tensor(3.25, dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    support.value.backward()
    torch.testing.assert_close(
        values.grad,
        torch.tensor(
            [[0.0625, 0.1875], [0.1875, 0.5625]],
            dtype=torch.float64,
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        durations.grad,
        torch.tensor([-0.375, 0.125], dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        weights.grad,
        torch.tensor([-0.75, 0.25], dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "sample_durations",
    [
        torch.tensor([-1.0, 1.0]),
        torch.tensor([0.0, 0.0]),
    ],
)
def test_temporal_resample_adapter_rejects_invalid_durations(sample_durations):
    with pytest.raises(ValueError, match="sample_durations"):
        TemporalResampleAdapter().apply(
            torch.tensor([1.0, 2.0]),
            method="interval_mean",
            sample_durations=sample_durations,
            axis=0,
        )


def test_area_basis_transform_adapter_round_trips_with_lai():
    adapter = AreaBasisTransformAdapter()
    leaf_flux = torch.tensor([2.0, 4.0])
    leaf_area_index = torch.tensor([3.0, 2.0])

    ground_result = adapter.apply(
        leaf_flux,
        source_area_basis="leaf_area",
        target_area_basis="ground_area",
        leaf_area_index=leaf_area_index,
    )
    leaf_result = adapter.apply(
        ground_result.value,
        source_area_basis="ground_area",
        target_area_basis="leaf_area",
        leaf_area_index=leaf_area_index,
    )

    assert torch.allclose(ground_result.value, torch.tensor([6.0, 8.0]))
    assert torch.allclose(leaf_result.value, leaf_flux)


def test_surface_flux_to_volume_source_closes_mass_ledger_and_gradient():
    """The lower-cell source must be the sole conservative surface injection."""
    adapter = SurfaceFluxToVolumeSourceAdapter()
    flux = torch.tensor(
        [2.0e-6, 3.0e-6],
        dtype=torch.float64,
        requires_grad=True,
    )
    cell_area = torch.tensor([100.0, 200.0], dtype=torch.float64)
    layer_volumes = torch.tensor(
        [[1000.0, 1200.0, 1400.0], [800.0, 900.0, 1000.0]],
        dtype=torch.float64,
    )
    constant_key = (
        "relic.physical.constants.molecular_weights.isoprene"
    )

    result = adapter.apply(
        flux,
        cell_area=cell_area,
        layer_volumes=layer_volumes,
        vertical_axis=1,
        molar_mass_g_per_mol_constant_key=constant_key,
    )

    molar_mass_kg_per_mol = (
        float(get_physical_constant(constant_key)) * 1.0e-3
    )
    expected_lower = (
        flux * molar_mass_kg_per_mol * cell_area / layer_volumes[:, 0]
    )
    expected = torch.zeros_like(layer_volumes)
    expected[:, 0] = expected_lower
    torch.testing.assert_close(result.value, expected)
    assert result.metadata["injected_layer_index"] == 0
    assert result.metadata["molar_mass_constant_key"] == constant_key
    torch.testing.assert_close(
        torch.tensor(result.metadata["domain_mass_rate_residual"]),
        torch.tensor(0.0),
        rtol=0.0,
        atol=1.0e-18,
    )
    assert len(result.metadata["column_mass_rate_residuals"]) == 2

    result.value.sum().backward()
    torch.testing.assert_close(
        flux.grad,
        molar_mass_kg_per_mol * cell_area / layer_volumes[:, 0],
    )


def test_surface_flux_to_volume_source_rejects_empty_vertical_grid():
    adapter = SurfaceFluxToVolumeSourceAdapter()

    with pytest.raises(ValueError, match="at least one vertical layer"):
        adapter.apply(
            torch.tensor(2.0e-6),
            cell_area=torch.tensor(100.0),
            layer_volumes=torch.empty(0),
            vertical_axis=0,
            molar_mass_g_per_mol_constant_key=(
                "relic.physical.constants.molecular_weights.isoprene"
            ),
        )


def test_surface_flux_to_volume_source_rejects_negative_emission():
    """The emission adapter must not silently reinterpret uptake as a source."""
    adapter = SurfaceFluxToVolumeSourceAdapter()

    with pytest.raises(ValueError, match="non-negative"):
        adapter.apply(
            torch.tensor([-1.0e-6]),
            cell_area=torch.tensor([100.0]),
            layer_volumes=torch.tensor([[1000.0, 1000.0]]),
            vertical_axis=1,
            molar_mass_g_per_mol_constant_key=(
                "relic.physical.constants.molecular_weights.isoprene"
            ),
        )


def test_executable_surface_flux_adapter_binds_declared_support():
    """Registry lowering must bind area and volume context without a second owner."""
    registry = NodeRegistry(SSOTCatalog())
    constant_key = (
        "relic.physical.constants.molecular_weights.isoprene"
    )
    adapter = ExecutableAdapterFunctor(
        name="surface_flux_to_source",
        adapter_type="surface_flux_to_volume_source",
        source_node="temporal_flux",
        source_port="isoprene_interval_flux",
        target_port="source_term",
        config={
            "cell_area_state": "cell_area",
            "layer_volumes_state": "layer_volumes",
            "vertical_axis": 1,
            "molar_mass_g_per_mol_constant_key": constant_key,
        },
    )
    definition = AdapterRegistry().register_executable_node(
        node_registry=registry,
        node_type_id="Functor:adapter.surface_flux_to_volume_source.test",
        functor=adapter,
        source_port=PortSpec(
            name="isoprene_interval_flux",
            standard_name="surface_upward_mole_flux_of_isoprene",
            units="mol m-2 s-1",
            shape={"dimensions": ["column"]},
            spatial_kind="grid_cell",
            area_basis="ground_area",
            temporal_basis="interval_mean",
            vertical_basis="none",
        ),
        target_port=PortSpec(
            name="source_term",
            standard_name="isoprene_mass_source_tendency",
            units="kg m-3 s-1",
            shape={"dimensions": ["column", "layer"]},
            semantics="diagnostic",
            write_mode="derived",
            spatial_kind="vertical_column",
            area_basis="grid_area",
            temporal_basis="interval_mean",
            vertical_basis="layer_integral",
        ),
        context_ports={
            "cell_area": PortSpec(
                name="cell_area",
                standard_name="cell_area",
                units="m2",
                shape={"dimensions": ["column"]},
                semantics="forcing",
            ),
            "layer_volumes": PortSpec(
                name="layer_volumes",
                standard_name="grid_cell_volume",
                units="m3",
                shape={"dimensions": ["column", "layer"]},
                semantics="forcing",
            ),
        },
    )
    adapter.bind_node_definition(definition)

    outputs = adapter.forward(
        state={
            "cell_area": torch.tensor([100.0, 200.0]),
            "layer_volumes": torch.tensor(
                [[1000.0, 1000.0], [800.0, 800.0]]
            ),
        },
        time=0.0,
        dt=60.0,
        subprocess_outputs={
            "temporal_flux": {
                "isoprene_interval_flux": torch.tensor([2.0e-6, 3.0e-6])
            }
        },
    )

    source = outputs["source_term"]
    assert source.write_mode.value == "derived"
    assert source.unit == "kg m-3 s-1"
    assert source.value.shape == (2, 2)
    torch.testing.assert_close(source.value[:, 1], torch.zeros(2))


def _support_port(
    name,
    area_basis="ground_area",
    dimensions=("cell",),
    spatial_kind="grid_cell",
    units="mol m-2 s-1",
):
    """A bound port; its support declares the area measure (area_basis) of the value."""
    return PortSpec(
        name=name,
        standard_name=name,
        units=units,
        spatial_kind=spatial_kind,
        area_basis=area_basis,
        temporal_basis="interval_mean",
        vertical_basis="none",
        shape={"dimensions": list(dimensions)},
    )


def _support_tolerance() -> float:
    return float(
        get_global_manager().get(
            "coupling.scales.defaults.conservation_tolerance", required=True
        )
    )


def _apply_support(
    values,
    weights,
    *,
    basis="area_density",
    source_basis="ground_area",
    target_basis="ground_area",
    **kwargs,
):
    return SupportAggregationAdapter().apply(
        values,
        weights=weights,
        variable_name="isoprene_emission",
        source_scale="grid_cell",
        target_scale="footprint",
        quantity_basis=basis,
        source_port_spec=_support_port("grid_flux", area_basis=source_basis),
        target_port_spec=_support_port(
            "footprint_flux",
            area_basis=target_basis,
            dimensions=(),
            spatial_kind="footprint",
        ),
        conservation_tolerance=_support_tolerance(),
        **kwargs,
    )


def test_support_aggregation_reduces_on_the_given_basis_with_its_ledger():
    """An area density keeps its basis: the support-weighted mean, audited by its ledger."""
    values = torch.tensor([1.0, 3.0], dtype=torch.float64)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64)

    result = _apply_support(values, weights, strategy="area_weighted_mean")

    # (1 * 1 + 3 * 3) / 4 = 2.5 mol m-2 s-1; ledger sum_i w_i x_i = 10 = 2.5 * 4.
    assert torch.allclose(result.value, torch.tensor(2.5, dtype=torch.float64))
    assert result.metadata["quantity_basis"] == "area_density"
    assert result.metadata["area_basis"] == "ground_area"
    assert result.metadata["strategy"] == "area_weighted_mean"
    assert result.metadata["conservation_audited_variables"] == [
        "isoprene_emission"
    ]
    ledger = result.metadata["conservation_audit"]["isoprene_emission"]
    assert ledger["source_total"] == pytest.approx(10.0)
    assert ledger["target_total"] == pytest.approx(10.0)
    assert result.metadata["relative_error"] < 1.0e-15
    assert result.metadata["conservation_enforced"] is False

    enforced = _apply_support(values, weights, enforce_conservation=True)
    assert torch.allclose(enforced.value, result.value)
    assert enforced.metadata["conservation_enforced"] is True
    assert enforced.metadata["strategy"] is None


@pytest.mark.parametrize("strategy", ["area_weighted_sum", "nearest", "mode"])
def test_support_aggregation_rejects_a_strategy_outside_the_basis(strategy):
    with pytest.raises(ValueError, match="conflicts with the declared basis"):
        _apply_support(
            torch.tensor([1.0, 3.0]),
            torch.tensor([1.0, 3.0]),
            strategy=strategy,
        )


def test_support_aggregation_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        _apply_support(
            torch.tensor([1.0, 3.0]),
            torch.tensor([1.0, 3.0]),
            strategy="median",
        )


@pytest.mark.parametrize(
    ("basis", "source_basis", "target_basis", "match"),
    [
        ("cell_total", "ground_area", "ground_area", "cell total"),
        ("cell_total", "leaf_area", "leaf_area", "cell total"),
        ("area_density", None, None, "no area_basis"),
        ("area_density", "ground_area", "leaf_area", "area_basis"),
    ],
)
def test_support_aggregation_rejects_a_measure_that_contradicts_the_basis(
    basis,
    source_basis,
    target_basis,
    match,
):
    """area_basis is only the area measure; it cannot contradict the SSOT basis."""
    weights = None if basis == "cell_total" else torch.tensor([1.0, 3.0])
    with pytest.raises(ValueError, match=match):
        _apply_support(
            torch.tensor([1.0, 3.0]),
            weights,
            basis=basis,
            source_basis=source_basis,
            target_basis=target_basis,
        )


def test_a_grid_area_measure_is_a_valid_area_density_measure():
    result = _apply_support(
        torch.tensor([1.0, 3.0], dtype=torch.float64),
        torch.tensor([1.0, 3.0], dtype=torch.float64),
        source_basis="grid_area",
        target_basis="grid_area",
    )
    assert torch.allclose(result.value, torch.tensor(2.5, dtype=torch.float64))
    assert result.metadata["area_basis"] == "grid_area"


def test_support_aggregation_requires_basis_ports_tolerance_and_weights():
    adapter = SupportAggregationAdapter()
    values = torch.tensor([1.0, 3.0])
    ports = dict(
        source_port_spec=_support_port("grid_flux"),
        target_port_spec=_support_port(
            "footprint_flux", dimensions=(), spatial_kind="footprint"
        ),
    )
    with pytest.raises(ValueError, match="quantity_basis"):
        adapter.apply(values, weights=values, conservation_tolerance=1.0e-8, **ports)
    with pytest.raises(ValueError, match="source_port_spec"):
        adapter.apply(
            values,
            weights=values,
            quantity_basis="area_density",
            conservation_tolerance=1.0e-8,
        )
    with pytest.raises(ValueError, match="conservation_tolerance"):
        adapter.apply(values, weights=values, quantity_basis="area_density", **ports)
    with pytest.raises(ValueError, match="weights"):
        _apply_support(values, None)
    with pytest.raises(ValueError, match="cell total"):
        _apply_support(
            values,
            values,
            basis="cell_total",
            source_basis="grid_area",
            target_basis="grid_area",
        )


def test_support_aggregation_ledger_is_that_of_the_returned_value(monkeypatch):
    """A wrong reduction raises when enforced and reports its residual otherwise."""
    import relic.coupling.scales.alias_utils as alias_utils

    original = alias_utils._reduce_on_basis
    monkeypatch.setattr(
        alias_utils,
        "_reduce_on_basis",
        lambda *args, **kwargs: 2.0 * original(*args, **kwargs),
    )
    values = torch.tensor([1.0, 3.0], dtype=torch.float64)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="violates conservation"):
        _apply_support(values, weights, enforce_conservation=True)
    reported = _apply_support(values, weights)
    assert torch.allclose(reported.value, torch.tensor(5.0, dtype=torch.float64))
    assert reported.metadata["relative_error"] == pytest.approx(1.0)
    ledger = reported.metadata["conservation_audit"]["isoprene_emission"]
    assert ledger["is_conserved"] is False


def _support_catalog(entries) -> SSOTCatalog:
    """An SSOT catalog declaring the spatial_basis (or none) of each support port variable."""
    catalog = SSOTCatalog()
    for name, (basis, units) in entries.items():
        catalog.add_entry(
            SSOTEntry(
                name=name,
                standard_name=name,
                units=units,
                spatial_basis=(
                    None if basis is None else SpatialQuantityBasis.parse(basis)
                ),
            )
        )
    return catalog


def _support_weights_port() -> PortSpec:
    return PortSpec(
        name="footprint_weights",
        standard_name="footprint_weights",
        units="1",
        spatial_kind="footprint",
        area_basis="grid_area",
        temporal_basis="instantaneous",
        vertical_basis="none",
        shape={"dimensions": ["cell"]},
    )


def _register_support(
    catalog,
    *,
    area_basis="ground_area",
    units="mol m-2 s-1",
    weighted=True,
    config=None,
    bind=True,
):
    adapter_config = {"variable_name": "q", **(config or {})}
    if weighted:
        adapter_config["weights_state"] = "footprint_weights"
    adapter = ExecutableAdapterFunctor(
        name="support_adapter",
        adapter_type="support_aggregation",
        source_node="source",
        source_port="grid_q",
        target_port="footprint_q",
        config=adapter_config,
    )
    definition = AdapterRegistry().register_executable_node(
        node_registry=NodeRegistry(catalog),
        node_type_id="Functor:adapter.support.ssot_basis",
        functor=adapter,
        source_port=_support_port("grid_q", area_basis=area_basis, units=units),
        target_port=_support_port(
            "footprint_q",
            area_basis=area_basis,
            dimensions=(),
            spatial_kind="footprint",
            units=units,
        ),
        context_ports=(
            {"footprint_weights": _support_weights_port()} if weighted else {}
        ),
    )
    if bind:
        adapter.bind_node_definition(definition)
    return adapter, definition


def _run_support(adapter, values, weights=None):
    state = {} if weights is None else {"footprint_weights": weights}
    return adapter.forward(
        state=state,
        time=0.0,
        dt=1.0,
        subprocess_outputs={"source": {"grid_q": values}},
    )["footprint_q"].value


@pytest.mark.parametrize(
    ("basis", "area_basis", "units", "values", "weights", "expected"),
    [
        # area density: (1 * 1 + 3 * 3) / 4 = 2.5 on the ground-area measure.
        ("area_density", "ground_area", "mol m-2 s-1", [1.0, 3.0], [1.0, 3.0], 2.5),
        # intensive: support-weighted mean (280 * 1 + 290 * 3) / 4 = 287.5 K.
        ("intensive", "ground_area", "K", [280.0, 290.0], [1.0, 3.0], 287.5),
        # cell total: the amounts of the support cells are summed, 4 + 8 = 12 mol.
        ("cell_total", "grid_area", "mol", [4.0, 8.0], None, 12.0),
        ("cell_total", None, "mol", [4.0, 8.0], None, 12.0),
        # categorical: class 7 holds 3 of the 4 support weight units.
        ("categorical", "grid_area", "1", [3.0, 7.0], [1.0, 3.0], 7.0),
    ],
)
def test_executable_support_adapter_takes_the_basis_from_the_ssot_catalog(
    basis,
    area_basis,
    units,
    values,
    weights,
    expected,
):
    catalog = _support_catalog({"grid_q": (basis, units), "footprint_q": (basis, units)})
    adapter, definition = _register_support(
        catalog,
        area_basis=area_basis,
        units=units,
        weighted=weights is not None,
    )
    assert definition.metadata["support_quantity_basis"] == basis
    result = _run_support(
        adapter,
        torch.tensor(values, dtype=torch.float64),
        None if weights is None else torch.tensor(weights, dtype=torch.float64),
    )
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
    metadata = adapter.last_adapter_metadata
    assert metadata["quantity_basis"] == basis
    ledger = metadata["conservation_audit"]
    expected_audited = ["q"] if basis in {"area_density", "cell_total"} else []
    assert metadata["conservation_audited_variables"] == expected_audited
    if expected_audited:
        assert ledger["q"]["tolerance"] == pytest.approx(_support_tolerance())


def test_the_port_name_or_its_standard_name_finds_the_ssot_entry():
    """A renamed port (a copied PortSpec) keeps the standard_name of its SSOT variable."""
    catalog = _support_catalog({"grid_q": ("area_density", "mol m-2 s-1"), "footprint_q": ("area_density", "mol m-2 s-1")})
    adapter = ExecutableAdapterFunctor(
        name="support_adapter_renamed",
        adapter_type="support_aggregation",
        source_node="source",
        source_port="grid_q",
        target_port="footprint_q_copy",
        config={"weights_state": "footprint_weights", "variable_name": "q"},
    )
    target = _support_port("footprint_q", dimensions=(), spatial_kind="footprint")
    target.name = "footprint_q_copy"
    definition = AdapterRegistry().register_executable_node(
        node_registry=NodeRegistry(catalog),
        node_type_id="Functor:adapter.support.renamed",
        functor=adapter,
        source_port=_support_port("grid_q"),
        target_port=target,
        context_ports={"footprint_weights": _support_weights_port()},
    )
    assert definition.metadata["support_quantity_basis"] == "area_density"


@pytest.mark.parametrize(
    ("entries", "area_basis", "weighted", "match"),
    [
        # a variable without an SSOT entry, or an entry without spatial_basis, raises
        ({"footprint_q": ("area_density", "mol m-2 s-1")}, "ground_area", True, "SSOT"),
        (
            {"grid_q": (None, "mol m-2 s-1"), "footprint_q": ("area_density", "mol m-2 s-1")},
            "ground_area",
            True,
            "spatial_basis",
        ),
        # the reduction keeps the basis
        (
            {"grid_q": ("area_density", "mol m-2 s-1"), "footprint_q": ("cell_total", "mol")},
            "grid_area",
            True,
            "same spatial_basis",
        ),
        # a cell total has no leaf or ground area measure
        ({"grid_q": ("cell_total", "mol"), "footprint_q": ("cell_total", "mol")}, "ground_area", False, "cell total"),
        # a cell total is summed: support weights do not apply
        ({"grid_q": ("cell_total", "mol"), "footprint_q": ("cell_total", "mol")}, "grid_area", True, "weights"),
        # an area density needs its support weights
        (
            {"grid_q": ("area_density", "mol m-2 s-1"), "footprint_q": ("area_density", "mol m-2 s-1")},
            "ground_area",
            False,
            "weights_state",
        ),
    ],
)
def test_executable_support_registration_rejects_an_undeclared_or_contradicting_basis(
    entries,
    area_basis,
    weighted,
    match,
):
    with pytest.raises(ValueError, match=match):
        _register_support(
            _support_catalog(entries),
            area_basis=area_basis,
            weighted=weighted,
            bind=False,
        )


def test_executable_support_adapter_uses_bound_ports_and_parameter_tolerance():
    catalog = _support_catalog({"grid_q": ("area_density", "mol m-2 s-1"), "footprint_q": ("area_density", "mol m-2 s-1")})
    adapter, _ = _register_support(catalog, config={"strategy": "area_weighted_mean"})
    result = _run_support(
        adapter,
        torch.tensor([1.0, 3.0], dtype=torch.float64),
        torch.tensor([0.25, 0.75], dtype=torch.float64),
    )
    torch.testing.assert_close(result, torch.tensor(2.5, dtype=torch.float64))
    ledger = adapter.last_adapter_metadata["conservation_audit"]["q"]
    assert ledger["tolerance"] == pytest.approx(_support_tolerance())
    # The bound ports, the basis and the governed tolerance are not configuration keys.
    for key in ("source_port_spec", "quantity_basis", "conservation_tolerance"):
        assert key not in adapter.contract_config()


def test_a_definition_without_the_ssot_basis_cannot_be_bound():
    catalog = _support_catalog({"grid_q": ("area_density", "mol m-2 s-1"), "footprint_q": ("area_density", "mol m-2 s-1")})
    adapter, definition = _register_support(catalog, bind=False)
    definition.metadata.pop("support_quantity_basis")
    with pytest.raises(ValueError, match="support_quantity_basis"):
        adapter.bind_node_definition(definition)


@pytest.mark.parametrize(
    "key",
    ["conservation_tolerance", "quantity_basis", "source_port_spec", "target_port_spec"],
)
def test_executable_support_adapter_rejects_config_duplicating_governed_inputs(key):
    with pytest.raises(ValueError, match=key):
        ExecutableAdapterFunctor(
            name="support_adapter_duplicate",
            adapter_type="support_aggregation",
            source_node="source",
            source_port="grid_flux",
            target_port="footprint_flux",
            config={key: 1.0e-3},
        )


def test_executable_support_adapter_requires_bound_ports():
    adapter = ExecutableAdapterFunctor(
        name="support_adapter_unbound",
        adapter_type="support_aggregation",
        source_node="source",
        source_port="grid_flux",
        target_port="footprint_flux",
        config={},
    )
    with pytest.raises(RuntimeError, match="NodeDefinition"):
        adapter.forward(
            state={},
            time=0.0,
            dt=1.0,
            subprocess_outputs={
                "source": {"grid_flux": torch.tensor([1.0, 3.0])}
            },
        )


def test_adapter_registry_exposes_all_explicit_graph_adapter_families():
    registry = AdapterRegistry()

    assert registry.get("unit_cast") is not None
    assert registry.get("temporal_resample") is not None
    assert registry.get("semantic_adapter") is registry.get("flux_to_increment")
    assert registry.get("area_basis_transform") is not None
    assert registry.get("surface_flux_to_volume_source") is not None
    assert registry.get("support_aggregation") is not None


def test_strict_explicit_adapter_policy_rejects_unlowered_direct_mismatch():
    registry = NodeRegistry(SSOTCatalog())
    source_type = "State:LengthMetres"
    target_type = "State:LengthCentimetres"
    registry.register_definition(
        source_type,
        NodeDefinition(
            node_type_id=source_type,
            outputs={
                "length": PortSpec(
                    name="length",
                    standard_name="generic_length",
                    units="m",
                )
            },
        ),
    )
    registry.register_definition(
        target_type,
        NodeDefinition(
            node_type_id=target_type,
            inputs={
                "length": PortSpec(
                    name="length",
                    standard_name="generic_length",
                    units="cm",
                )
            },
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": source_type,
                    "inputs": {},
                    "outputs": ["length"],
                    "config": {},
                },
                "target": {
                    "class_type": target_type,
                    "inputs": {"length": {"link": ["source", "length"]}},
                    "outputs": [],
                    "config": {},
                },
            },
            "links": [
                {
                    "from": {"node": "source", "port": "length"},
                    "to": {"node": "target", "port": "length"},
                }
            ],
            "metadata": {"adapters": "explicit"},
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(
        issue.code == "unlowered_explicit_adapter_requirement"
        for issue in report.errors
    )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"execution_group": None},
        {"execution_group": "fast"},
        {"execution_group": "medium"},
        {"execution_group": "slow"},
    ],
)
def test_strict_validation_allows_supported_or_omitted_execution_group(config):
    registry = NodeRegistry(SSOTCatalog())
    class_type = "State:ExecutionGroupSource"
    _register_definition(registry, class_type, outputs=("x",))
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": class_type,
                    "inputs": {},
                    "outputs": ["x"],
                    "config": config,
                }
            },
            "metadata": {},
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert not any(
        issue.code == "invalid_execution_group"
        for issue in report.errors + report.warnings
    )


@pytest.mark.parametrize(
    ("execution_group", "expected_got"),
    [("fas", "'fas'"), (1, "1")],
)
def test_strict_validation_rejects_invalid_execution_group(
    execution_group,
    expected_got,
):
    registry = NodeRegistry(SSOTCatalog())
    class_type = "State:InvalidExecutionGroupSource"
    _register_definition(registry, class_type, outputs=("x",))
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "bad": {
                    "class_type": class_type,
                    "inputs": {},
                    "outputs": ["x"],
                    "config": {"execution_group": execution_group},
                }
            },
            "metadata": {},
        }
    )

    report = validate_graph(graph, registry, strict=True)
    issues = [
        issue
        for issue in report.errors
        if issue.code == "invalid_execution_group"
    ]

    assert len(issues) == 1
    assert not any(
        issue.code == "invalid_execution_group"
        for issue in report.warnings
    )
    assert issues[0].node_id == "bad"
    assert issues[0].expected == "fast, medium, slow, or omitted/None"
    assert issues[0].got == expected_got


def test_non_strict_validation_rejects_invalid_execution_group_without_rewriting():
    """Validation is never more permissive than runtime registration, which raises."""
    registry = NodeRegistry(SSOTCatalog())
    class_type = "State:NonStrictExecutionGroupSource"
    _register_definition(registry, class_type, outputs=("x",))
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "bad": {
                    "class_type": class_type,
                    "inputs": {},
                    "outputs": ["x"],
                    "config": {"execution_group": "fas"},
                }
            },
            "metadata": {},
        }
    )

    report = validate_graph(graph, registry, strict=False)

    assert (
        len(
            [
                issue
                for issue in report.errors
                if issue.code == "invalid_execution_group"
            ]
        )
        == 1
    )
    assert not any(
        issue.code == "invalid_execution_group"
        for issue in report.warnings
    )
    assert graph.nodes["bad"].config["execution_group"] == "fas"


@pytest.mark.parametrize("strict", [True, False])
def test_graph_validation_reports_the_runtime_execution_group_message(strict):
    """The validator and register_functor share one predicate and one message."""
    from relic.core.simulator import RELICSimulator

    with pytest.raises(ValueError) as runtime_error:
        RELICSimulator({"time_step": 60.0}).register_functor(
            DummyIncrementFunctor(), execution_group="fas"
        )
    runtime_message = str(runtime_error.value)
    prefix = "Invalid execution_group 'fas' from "
    suffix = "; expected one of ['fast', 'medium', 'slow']."
    assert runtime_message.startswith(prefix) and runtime_message.endswith(suffix)

    registry = NodeRegistry(SSOTCatalog())
    class_type = "State:RuntimeMessageExecutionGroupSource"
    _register_definition(registry, class_type, outputs=("x",))
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "bad": {
                    "class_type": class_type,
                    "inputs": {},
                    "outputs": ["x"],
                    "config": {"execution_group": "fas"},
                }
            },
            "metadata": {},
        }
    )

    report = validate_graph(graph, registry, strict=strict)
    [issue] = [issue for issue in report.errors if issue.code == "invalid_execution_group"]

    assert issue.message == prefix + "config.execution_group of node 'bad'" + suffix


def test_strict_compile_rejects_invalid_execution_group():
    registry = NodeRegistry(SSOTCatalog())
    class_type = "Functor:test.fixtures.dummy_functor.DummyIncrementFunctor"
    registry.register_functor_instance(
        DummyIncrementFunctor(),
        node_type_id=class_type,
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "bad": {
                    "class_type": class_type,
                    "inputs": {},
                    "outputs": ["x"],
                    "config": {"execution_group": "fas"},
                }
            },
            "metadata": {},
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid execution_group 'fas' from config\.execution_group of node 'bad'",
    ):
        compile_graph(graph, registry, strict=True)


def test_strict_validation_rejects_raw_cross_execution_group_dependency():
    registry = NodeRegistry(SSOTCatalog())
    source_type = "State:InstantaneousFlux"
    target_type = "State:IntervalConsumer"
    source_port = PortSpec(
        name="instantaneous_flux",
        standard_name="instantaneous_flux",
        units="mol m-2 s-1",
        semantics="flux",
        shape={"dimensions": ["cell"]},
        direction="out",
    )
    target_port = PortSpec(
        name="instantaneous_flux",
        standard_name="instantaneous_flux",
        units="mol m-2 s-1",
        semantics="flux",
        shape={"dimensions": ["cell"]},
        direction="in",
    )
    registry.register_definition(
        source_type,
        NodeDefinition(
            node_type_id=source_type,
            outputs={"instantaneous_flux": source_port},
        ),
    )
    registry.register_definition(
        target_type,
        NodeDefinition(
            node_type_id=target_type,
            inputs={"instantaneous_flux": target_port},
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": source_type,
                    "inputs": {},
                    "outputs": ["instantaneous_flux"],
                    "config": {"execution_group": "fast"},
                },
                "consumer": {
                    "class_type": target_type,
                    "inputs": {
                        "instantaneous_flux": {
                            "link": ["source", "instantaneous_flux"],
                        }
                    },
                    "outputs": [],
                    "config": {"execution_group": "medium"},
                },
            },
            "links": [
                {
                    "from": {
                        "node": "source",
                        "port": "instantaneous_flux",
                    },
                    "to": {
                        "node": "consumer",
                        "port": "instantaneous_flux",
                    },
                }
            ],
            "metadata": {"adapters": "explicit"},
        }
    )

    report = validate_graph(graph, registry, strict=True)

    issues = [
        issue
        for issue in report.errors
        if issue.code == "cross_execution_group_raw_dependency"
    ]
    assert len(issues) == 1
    assert "crosses execution groups" in issues[0].message
    assert issues[0].expected == (
        "same execution_group or registered temporal_resample adapter"
    )
    assert issues[0].got == "fast -> medium"
    with pytest.raises(ValueError, match="crosses execution groups"):
        compile_graph(graph, registry, strict=True)


def test_strict_validation_allows_registered_temporal_adapter_at_group_boundary():
    registry = NodeRegistry(SSOTCatalog())
    source_type = "State:WindowedFlux"
    adapter_type = "Functor:adapter.temporal.execution_group_boundary"
    source_port = PortSpec(
        name="instantaneous_flux",
        standard_name="instantaneous_flux",
        units="mol m-2 s-1",
        semantics="flux",
        shape={"dimensions": ["sample", "cell"]},
        direction="out",
    )
    registry.register_definition(
        source_type,
        NodeDefinition(
            node_type_id=source_type,
            outputs={"instantaneous_flux": source_port},
        ),
    )
    adapter = ExecutableAdapterFunctor(
        name="temporal_boundary",
        adapter_type="temporal_resample",
        source_node="source",
        source_port="instantaneous_flux",
        target_port="interval_flux",
        config={"method": "interval_mean", "axis": 0},
    )
    AdapterRegistry().register_executable_node(
        node_registry=registry,
        node_type_id=adapter_type,
        functor=adapter,
        source_port=source_port,
        target_port=PortSpec(
            name="interval_flux",
            standard_name="interval_flux",
            units="mol m-2 s-1",
            semantics="flux",
            shape={"dimensions": ["cell"]},
            direction="out",
            write_mode="mutable",
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": source_type,
                    "inputs": {},
                    "outputs": ["instantaneous_flux"],
                    "config": {"execution_group": "fast"},
                },
                "temporal": {
                    "class_type": adapter_type,
                    "inputs": {
                        "instantaneous_flux": {
                            "link": ["source", "instantaneous_flux"],
                        }
                    },
                    "outputs": ["interval_flux"],
                    "config": {
                        **adapter.contract_config(),
                        "execution_group": "medium",
                    },
                },
            },
            "links": [
                {
                    "from": {
                        "node": "source",
                        "port": "instantaneous_flux",
                    },
                    "to": {
                        "node": "temporal",
                        "port": "instantaneous_flux",
                    },
                }
            ],
            "metadata": {"adapters": "explicit"},
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert not any(
        issue.code == "cross_execution_group_raw_dependency"
        for issue in report.errors
    )


def test_strict_validation_rejects_adapter_registry_config_drift():
    registry = NodeRegistry(SSOTCatalog())
    source_type = "State:MassFlux"
    source_port = PortSpec(
        name="mass_flux",
        standard_name="mass_flux",
        units="kg s-1",
        semantics="flux",
    )
    registry.register_definition(
        source_type,
        NodeDefinition(
            node_type_id=source_type,
            outputs={"mass_flux": source_port},
        ),
    )
    registered_config = {
        "adapter_type": "flux_to_increment",
        "source_node": "source",
        "source_port": "mass_flux",
        "target_port": "mass_increment",
    }
    adapter = ExecutableAdapterFunctor(
        name="mass_flux_adapter",
        **registered_config,
    )
    adapter_type = "Functor:adapter.flux_to_increment.mass"
    AdapterRegistry().register_executable_node(
        node_registry=registry,
        node_type_id=adapter_type,
        functor=adapter,
        source_port=source_port,
        target_port=PortSpec(
            name="mass_increment",
            standard_name="mass_increment",
            units="kg",
            semantics="increment",
            write_mode="mutable",
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "source": {
                    "class_type": source_type,
                    "inputs": {},
                    "outputs": ["mass_flux"],
                    "config": {},
                },
                "adapter": {
                    "class_type": adapter_type,
                    "inputs": {
                        "mass_flux": {
                            "link": ["source", "mass_flux"],
                        }
                    },
                    "outputs": ["mass_increment"],
                    "config": {
                        **registered_config,
                        "source_node": "forged_source",
                    },
                },
            },
            "links": [
                {
                    "from": {"node": "source", "port": "mass_flux"},
                    "to": {"node": "adapter", "port": "mass_flux"},
                }
            ],
            "metadata": {"adapters": "explicit"},
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(
        issue.code == "adapter_contract_mismatch"
        for issue in report.errors
    )


def test_strict_validation_rejects_registered_node_config_mismatch():
    registry = NodeRegistry(SSOTCatalog())
    class_type = "Functor:test.configured_node"
    registry.register_definition(
        class_type,
        NodeDefinition(
            node_type_id=class_type,
            metadata={
                "config_contract": {
                    "enabled_compounds": ["isoprene"],
                    "emission_pathway": "standard",
                }
            },
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "configured": {
                    "class_type": class_type,
                    "inputs": {},
                    "outputs": [],
                    "config": {
                        "enabled_compounds": ["isoprene"],
                        "emission_pathway": "hybrid",
                    },
                }
            },
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(
        issue.code == "node_config_contract_mismatch"
        for issue in report.errors
    )

    graph.nodes["configured"].config["emission_pathway"] = "standard"
    matching_report = validate_graph(graph, registry, strict=True)
    assert not any(
        issue.code == "node_config_contract_mismatch"
        for issue in matching_report.errors
    )


def test_strict_validation_requires_registered_input_binding():
    registry = NodeRegistry(SSOTCatalog())
    class_type = "Functor:test.required_input"
    registry.register_definition(
        class_type,
        NodeDefinition(
            node_type_id=class_type,
            inputs={
                "forcing": PortSpec(
                    name="forcing",
                    standard_name="forcing",
                    units="K",
                    required=True,
                )
            },
            outputs={},
            metadata={
                "runtime_class": (
                    "test.fixtures.dummy_functor.DummyIncrementFunctor"
                )
            },
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "consumer": {
                    "class_type": class_type,
                    "inputs": {},
                    "outputs": [],
                    "config": {},
                }
            },
            "links": [],
            "metadata": {},
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(
        issue.code == "missing_required_input_binding"
        for issue in report.errors
    )


def test_strict_validation_accepts_same_name_state_input_binding():
    registry = NodeRegistry(SSOTCatalog())
    class_type = "Functor:test.state_input"
    registry.register_definition(
        class_type,
        NodeDefinition(
            node_type_id=class_type,
            inputs={
                "forcing": PortSpec(
                    name="forcing",
                    standard_name="forcing",
                    units="K",
                    required=True,
                )
            },
            outputs={},
            metadata={
                "runtime_class": (
                    "test.fixtures.dummy_functor.DummyIncrementFunctor"
                )
            },
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "consumer": {
                    "class_type": class_type,
                    "inputs": {"forcing": {"state": "forcing"}},
                    "outputs": [],
                    "config": {},
                }
            },
            "links": [],
            "metadata": {},
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert not any(
        issue.code
        in {
            "missing_required_input_binding",
            "unsupported_input_binding",
            "state_input_alias_not_executable",
        }
        for issue in report.errors
    )


def test_strict_validation_rejects_state_input_alias():
    registry = NodeRegistry(SSOTCatalog())
    class_type = "Functor:test.state_alias"
    registry.register_definition(
        class_type,
        NodeDefinition(
            node_type_id=class_type,
            inputs={
                "forcing": PortSpec(
                    name="forcing",
                    standard_name="forcing",
                    units="K",
                    required=True,
                )
            },
            outputs={},
            metadata={
                "runtime_class": (
                    "test.fixtures.dummy_functor.DummyIncrementFunctor"
                )
            },
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "consumer": {
                    "class_type": class_type,
                    "inputs": {"forcing": {"state": "other_name"}},
                    "outputs": [],
                    "config": {},
                }
            },
            "links": [],
            "metadata": {},
        }
    )

    report = validate_graph(graph, registry, strict=True)

    assert any(
        issue.code == "state_input_alias_not_executable"
        for issue in report.errors
    )


def test_executable_adapter_requires_explicit_context_input_ports():
    registry = NodeRegistry(
        _support_catalog(
            {
                "grid_flux": ("area_density", "mol m-2 s-1"),
                "footprint_flux": ("area_density", "mol m-2 s-1"),
            }
        )
    )
    adapter_registry = AdapterRegistry()
    config = {
        "adapter_type": "support_aggregation",
        "source_node": "source",
        "source_port": "grid_flux",
        "target_port": "footprint_flux",
        "weights_state": "footprint_weights",
        "strategy": "area_weighted_mean",
    }
    adapter = ExecutableAdapterFunctor(
        name="support_adapter",
        **{
            key: config[key]
            for key in (
                "adapter_type",
                "source_node",
                "source_port",
                "target_port",
            )
        },
        config={
            "weights_state": config["weights_state"],
            "strategy": config["strategy"],
        },
    )
    source_port = PortSpec(
        name="grid_flux",
        standard_name="isoprene_emission",
        units="mol m-2 s-1",
        semantics="flux",
        spatial_kind="grid_cell",
        area_basis="ground_area",
        temporal_basis="interval_mean",
        vertical_basis="none",
        shape={"dimensions": ["cell"]},
        write_mode="immutable",
        direction="out",
    )
    target_port = PortSpec(
        name="footprint_flux",
        standard_name="isoprene_emission",
        units="mol m-2 s-1",
        semantics="diagnostic",
        write_mode="derived",
        spatial_kind="footprint",
        area_basis="ground_area",
        temporal_basis="interval_mean",
        vertical_basis="none",
        shape={"dimensions": []},
    )

    with pytest.raises(ValueError, match="context PortSpec"):
        adapter_registry.register_executable_node(
            node_registry=registry,
            node_type_id="Functor:adapter.support.missing_context",
            functor=adapter,
            source_port=source_port,
            target_port=target_port,
        )

    definition = adapter_registry.register_executable_node(
        node_registry=registry,
        node_type_id="Functor:adapter.support.complete",
        functor=adapter,
        source_port=source_port,
        target_port=target_port,
        context_ports={
            "footprint_weights": PortSpec(
                name="footprint_weights",
                standard_name="footprint_weights",
                units="dimensionless",
                semantics="parameter",
                spatial_kind="footprint",
                area_basis="grid_area",
                temporal_basis="instantaneous",
                vertical_basis="none",
                shape={"dimensions": ["cell"]},
                write_mode="immutable",
                direction="out",
            )
        },
    )

    assert set(definition.inputs) == {"grid_flux", "footprint_weights"}
    assert definition.inputs["grid_flux"].direction == "in"
    assert definition.inputs["grid_flux"].write_mode is None
    assert definition.inputs["footprint_weights"].direction == "in"
    assert definition.inputs["footprint_weights"].write_mode is None
    assert definition.inputs["footprint_weights"].required is True
    assert definition.outputs["footprint_flux"].direction == "out"
    assert definition.outputs["footprint_flux"].write_mode == "derived"
    assert source_port.direction == "out"
    assert source_port.write_mode == "immutable"

    with pytest.raises(ValueError, match="weights shape"):
        adapter_registry.register_executable_node(
            node_registry=registry,
            node_type_id="Functor:adapter.support.shape_mismatch",
            functor=adapter,
            source_port=source_port,
            target_port=target_port,
            context_ports={
                "footprint_weights": PortSpec(
                    name="footprint_weights",
                    standard_name="footprint_weights",
                    units="dimensionless",
                    shape={"dimensions": ["sample"]},
                )
            },
        )


class _ConverterWithBrokenInfrastructure:
    """UnitConverter stand-in whose conversion fails for a non-context reason."""

    def can_convert(self, source_units, target_units, variable_name):
        return True

    def convert(self, value, *, variable_name, source_units, target_units, context):
        raise RuntimeError("parameter manager unavailable for molecular weight")


class _ConverterNeedingContext(_ConverterWithBrokenInfrastructure):
    """UnitConverter stand-in that rejects the trial conversion for missing context."""

    def convert(self, value, *, variable_name, source_units, target_units, context):
        raise ValueError("soil depth context is required for this conversion")


def _unit_cast_ports():
    source = PortSpec(name="flux_out", standard_name="soil_water_flux", units="kg m-2 s-1")
    target = PortSpec(name="flux_in", standard_name="soil_water_flux", units="mm s-1")
    return source, target


def test_unit_compatibility_records_missing_conversion_context():
    source, target = _unit_cast_ports()
    decision = validator_module.ConnectionDecision(ok=True)

    accepted = validator_module._validate_units_compatibility(
        decision,
        source=source,
        target=target,
        unit_converter=_ConverterNeedingContext(),
        context=None,
        strict=True,
    )

    assert accepted is False
    assert [issue.code for issue in decision.errors] == ["missing_context"]
    assert "soil depth" in decision.errors[0].message


def test_unit_compatibility_propagates_converter_infrastructure_failure():
    source, target = _unit_cast_ports()
    decision = validator_module.ConnectionDecision(ok=True)

    with pytest.raises(RuntimeError, match="parameter manager unavailable"):
        validator_module._validate_units_compatibility(
            decision,
            source=source,
            target=target,
            unit_converter=_ConverterWithBrokenInfrastructure(),
            context=None,
            strict=True,
        )


def test_validate_graph_propagates_ssot_catalog_load_failure(monkeypatch):
    """Without an explicit catalog, a conflicting SSOT catalog is an error, not a skipped check."""

    def _conflicting_catalog(*_args, **_kwargs):
        raise ValueError("conflicting SSOT declarations for 'soil_temperature'")

    monkeypatch.setattr(validator_module.SSOTCatalog, "from_parameter_manager", _conflicting_catalog)
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={"n0": NodeSpec(class_type="State:ExternalForcing", outputs=["forcing"])},
    )

    with pytest.raises(ValueError, match="conflicting SSOT declarations"):
        validate_graph(graph, NodeRegistry(SSOTCatalog()))


def _runtime_bound_unit_cast_case():
    """A bound, executable unit_cast adapter node (mm day-1 -> kg m-2 s-1)."""
    registry = NodeRegistry(SSOTCatalog())
    registry.register_definition(
        "State:ExternalDailyPrecipitation",
        NodeDefinition(
            node_type_id="State:ExternalDailyPrecipitation",
            inputs={},
            outputs={
                "precipitation_flux": PortSpec(
                    name="precipitation_flux",
                    standard_name="precipitation_flux",
                    units="mm day-1",
                )
            },
        ),
    )
    contract = {
        "adapter_type": "unit_cast",
        "source_node": "external_forcing",
        "source_port": "precipitation_flux",
        "target_port": "precipitation_flux_si",
        "variable_name": "precipitation_flux",
        "source_units": "mm day-1",
        "target_units": "kg m-2 s-1",
    }
    registry.register_definition(
        "Functor:adapter.unit_cast.precipitation_flux",
        NodeDefinition(
            node_type_id="Functor:adapter.unit_cast.precipitation_flux",
            inputs={
                "precipitation_flux": PortSpec(
                    name="precipitation_flux",
                    standard_name="precipitation_flux",
                    units="mm day-1",
                )
            },
            outputs={
                "precipitation_flux_si": PortSpec(
                    name="precipitation_flux_si",
                    standard_name="precipitation_flux",
                    units="kg m-2 s-1",
                    semantics="diagnostic",
                    write_mode="derived",
                )
            },
            metadata={
                "runtime_class": "relic.orchestration.adapters.ExecutableAdapterFunctor",
                "adapter_contract": contract,
            },
        ),
    )
    graph = GraphIR.from_dict(
        {
            "schema_version": "1.0.0",
            "nodes": {
                "external_forcing": {
                    "class_type": "State:ExternalDailyPrecipitation",
                    "outputs": ["precipitation_flux"],
                },
                "adapter": {
                    "class_type": "Functor:adapter.unit_cast.precipitation_flux",
                    "inputs": {"precipitation_flux": {"link": ["external_forcing", "precipitation_flux"]}},
                    "outputs": ["precipitation_flux_si"],
                    "config": contract,
                },
            },
            "links": [
                {
                    "from": {"node": "external_forcing", "port": "precipitation_flux"},
                    "to": {"node": "adapter", "port": "precipitation_flux"},
                }
            ],
        }
    )
    return graph, registry


def test_bound_unit_cast_reports_a_context_rejection_as_not_executable():
    graph, registry = _runtime_bound_unit_cast_case()

    _, issues = validator_module._bound_unit_cast_endpoints(
        graph, registry, _ConverterNeedingContext()
    )

    assert [issue.code for issue in issues] == ["unit_cast_not_executable"]


def test_bound_unit_cast_propagates_converter_infrastructure_failure():
    graph, registry = _runtime_bound_unit_cast_case()

    with pytest.raises(RuntimeError, match="parameter manager unavailable"):
        validator_module._bound_unit_cast_endpoints(
            graph, registry, _ConverterWithBrokenInfrastructure()
        )


@pytest.mark.parametrize("invalid_mode", ["unknown", "", None, 1, {}, ["state"]])
def test_executable_adapter_rejects_undeclared_source_modes(invalid_mode):
    with pytest.raises(ValueError, match="source_mode"):
        _register_current_step_temporal_adapter(
            config_overrides={"source_mode": invalid_mode}
        )


def test_executable_adapter_default_contract_does_not_acquire_source_mode():
    adapter, definition = _register_current_step_temporal_adapter()
    expected = {
        "adapter_type": "temporal_resample",
        "source_node": "source",
        "source_port": "instantaneous_flux",
        "target_port": "interval_flux",
        "sampling_mode": "current_step",
        "method": "interval_mean",
        "sample_durations_state": "observation_window_duration",
    }
    assert adapter.contract_config() == expected
    assert definition.metadata["adapter_contract"] == expected
    assert "source_mode" not in adapter.adapter_config


def test_executable_state_source_reads_only_declared_port_with_gradient():
    from relic.core.state.types import UpdateItem, UpdateMode

    adapter, definition = _register_current_step_temporal_adapter(
        config_overrides={"source_mode": "state"}
    )
    flux = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    result = adapter.forward(
        state={
            "instantaneous_flux": UpdateItem(flux, mode=UpdateMode.REPLACE),
            "observation_window_duration": torch.tensor(30.0),
            "unrelated_state": torch.tensor([-1000.0]),
        },
        time=0.0,
        dt=30.0,
        subprocess_outputs={},
    )
    torch.testing.assert_close(result["interval_flux"].value, flux, rtol=0.0, atol=0.0)
    assert set(result) == {"interval_flux"}
    assert definition.metadata["adapter_contract"]["source_mode"] == "state"
    result["interval_flux"].value.sum().backward()
    torch.testing.assert_close(flux.grad, torch.ones_like(flux))


def test_executable_state_source_missing_port_does_not_read_upstream():
    adapter, _ = _register_current_step_temporal_adapter(
        config_overrides={"source_mode": "state"}
    )
    with pytest.raises(KeyError, match="instantaneous_flux"):
        adapter.forward(
            state={"observation_window_duration": torch.tensor(30.0)},
            time=0.0,
            dt=30.0,
            subprocess_outputs={"source": {"instantaneous_flux": torch.tensor([8.0])}},
        )


def test_default_adapter_missing_upstream_does_not_fallback_to_state():
    adapter, _ = _register_current_step_temporal_adapter()
    with pytest.raises(RuntimeError, match="upstream output"):
        adapter.forward(
            state={
                "instantaneous_flux": torch.tensor([8.0]),
                "observation_window_duration": torch.tensor(30.0),
            },
            time=0.0,
            dt=30.0,
            subprocess_outputs={},
        )


def test_state_source_selector_never_enters_the_mathematical_adapter(monkeypatch):
    from relic.orchestration.adapters import AdapterResult

    adapter, _ = _register_current_step_temporal_adapter(
        config_overrides={"source_mode": "state"}
    )
    observed = []

    def apply(value, **kwargs):
        assert "source_mode" not in kwargs
        observed.append(value)
        return AdapterResult(value=value, metadata={})

    monkeypatch.setattr(adapter._adapter, "apply", apply)
    state_flux = torch.tensor([2.0])
    result = adapter.forward(
        state={
            "instantaneous_flux": state_flux,
            "observation_window_duration": torch.tensor(30.0),
        },
        time=0.0,
        dt=30.0,
        subprocess_outputs={"source": {"instantaneous_flux": torch.tensor([99.0])}},
    )
    assert observed == [state_flux]
    assert result["interval_flux"].value is state_flux


def test_state_source_keeps_distinct_source_and_target_contract():
    with pytest.raises(ValueError, match="source_port and target_port must differ"):
        ExecutableAdapterFunctor(
            adapter_type="temporal_resample",
            source_node="source",
            source_port="isoprene_emission",
            target_port="isoprene_emission",
            config={"source_mode": "state"},
        )


def _state_source_column_case():
    from scripts.publication import profile_registry
    from relic.orchestration.graph_io import load_graph

    manifest = profile_registry.load_profile_004_synthetic_fixture_manifest()
    registry, adapters = profile_registry.build_profile_004_publication_registries()
    old = registry.get_definition(profile_registry._PROFILE_004_TEMPORAL_CLASS_TYPE)
    state_adapter = ExecutableAdapterFunctor(
        name="temporal_resample_adapter",
        adapter_type="temporal_resample",
        source_node="synthetic_forcing",
        source_port="isoprene_emission",
        target_port="isoprene_emission_interval_mean",
        config={
            "source_mode": "state",
            "sampling_mode": "current_step",
            "method": "interval_mean",
            "sample_durations_state": "observation_window_duration",
        },
    )
    type_id = "Functor:adapter.temporal.state_source.test"
    definition = adapters.register_executable_node(
        node_registry=registry,
        node_type_id=type_id,
        functor=state_adapter,
        source_port=old.inputs["isoprene_emission"],
        target_port=old.outputs["isoprene_emission_interval_mean"],
        context_ports={
            "observation_window_duration": old.inputs["observation_window_duration"]
        },
    )
    root = Path(__file__).resolve().parents[2]
    original = load_graph(root / manifest["profile"]["graph_path"]).to_dict()
    nodes = {
        name: original["nodes"][name]
        for name in (
            "synthetic_forcing", "temporal_resample_adapter",
            "surface_flux_to_volume_source_adapter", "fickian_diffusion",
        )
    }
    nodes["synthetic_forcing"] = {
        "class_type": "State:ExternalForcing",
        "inputs": {},
        "outputs": ["isoprene_emission"],
        "config": {},
    }
    temporal = nodes["temporal_resample_adapter"]
    temporal["class_type"] = type_id
    temporal["config"] = {**definition.metadata["adapter_contract"], "execution_group": "fast"}
    temporal["inputs"]["isoprene_emission"] = {
        "link": ["synthetic_forcing", "isoprene_emission"]
    }
    temporal["inputs"]["observation_window_duration"] = {
        "state": "observation_window_duration"
    }
    surface = nodes["surface_flux_to_volume_source_adapter"]
    for name in ("cell_area", "layer_volumes"):
        surface["inputs"][name] = {"state": name}
    graph = GraphIR.from_dict({"schema_version": "1.0.0", "nodes": nodes})
    initial = profile_registry.build_profile_004_initial_state()
    initial["isoprene_emission"] = torch.tensor([0.0])
    return graph, registry, adapters, initial, manifest


def test_state_boundary_adapter_graph_advances_diffusion_once_with_mass_ledger():
    import importlib.util
    from relic.orchestration.executor import GraphExecutor

    graph, registry, adapters, initial, manifest = _state_source_column_case()
    plan = compile_graph(graph, registry, adapter_registry=adapters, auto_adapt=False, strict=True)
    snapshots = [
        {"isoprene_emission": torch.tensor([value])}
        for value in (1.0e-6, 0.0, 2.0e-6)
    ]
    assert all(set(snapshot) == {"isoprene_emission"} for snapshot in snapshots)
    executor = GraphExecutor(
        plan, registry=registry, cache=None,
        simulator_config=manifest["profile"]["simulator_config"],
    )
    dt = float(manifest["profile"]["base_dt_seconds"])
    final = executor.execute_forcing_sequence(initial, snapshots, dt=dt)
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "canonical_column_reference_state_source",
        root / "test/system/publication_profiles/acp_observation_twins.test.py",
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    oracle = manifest["oracle"]
    series = executor.time_series
    expected, _ = reference._finite_volume_column_oracle(
        initial["tracer_concentration"], series["source_term"],
        diffusivity_m2_s=float(oracle["diffusivity_m2_s"]),
        spacing_m=float(oracle["grid_spacing_m"]), dt_seconds=dt,
    )
    torch.testing.assert_close(final["tracer_concentration"], expected, rtol=0.0, atol=0.0)
    assert executor.execution_summary["step_count"] == len(snapshots)
    assert executor.execution_summary["current_time"] == len(snapshots) * dt
    assert all(len(series[name]) == len(snapshots) for name in (
        "isoprene_emission_interval_mean", "source_term", "tracer_concentration"
    ))
    assert all(torch.count_nonzero(source[..., 1:]) == 0 for source in series["source_term"])
    assert torch.count_nonzero(series["source_term"][1]) == 0
    # References: Patankar (1980); Crank (1975); original PROFILE-004 mass oracle.
    # Mathematical expression: M_final - M_initial = sum_n F_n M_iso A dt.
    # Parameter meanings: M is column tracer mass; F is the prescribed molar flux;
    #   M_iso is isoprene molar mass; A is column area; dt is the accepted step.
    # Units: M [kg], F [mol m-2 s-1], M_iso [kg mol-1], A [m2], dt [s].
    # Domain/constraints: one positive-volume column, zero-Neumann boundary,
    #   no chemistry or other source; all prescribed source records are retained.
    # Sign convention: F positive into the lowest layer increases total mass.
    volume = initial["layer_volumes"]
    molar_mass = float(get_physical_constant(
        "relic.physical.constants.molecular_weights.isoprene"
    )) * 1.0e-3
    expected_injection = sum(
        float(snapshot["isoprene_emission"].double().sum())
        * molar_mass * float(initial["cell_area"].double().sum()) * dt
        for snapshot in snapshots
    )
    measured_storage = float((
        (final["tracer_concentration"].double() - initial["tracer_concentration"].double())
        * volume.double()
    ).sum())
    assert measured_storage == pytest.approx(
        expected_injection, rel=float(oracle["mass_relative_tolerance"]), abs=0.0
    )
    assert executor._simulator.get_forcing_cursor()["consumed"] == len(snapshots)
    assert final["tracer_concentration"].shape == initial["tracer_concentration"].shape


def test_state_boundary_adapter_does_not_weaken_mutually_exclusive_source_writers():
    graph, registry, _, _, manifest = _state_source_column_case()
    from relic.orchestration.graph_io import load_graph

    root = Path(__file__).resolve().parents[2]
    original = load_graph(root / manifest["profile"]["graph_path"])
    megan = original.nodes["megan_bvoc_model"]
    megan.inputs = {name: {"state": name} for name in megan.inputs}
    graph.nodes["megan_bvoc_model"] = megan
    report = validate_graph(graph, registry, strict=True, ssot_catalog=registry.ssot_catalog)
    assert any(issue.code == "mutually_exclusive_output_overlap" for issue in report.errors)