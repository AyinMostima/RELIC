import json
import shutil
import uuid
from pathlib import Path

from relic.computation.solvers import IntegrationPlan, IntegrationStage
from relic.core.provenance import ProvenanceRecorder
from relic.orchestration.compiler import compile_graph, plan_to_dict, write_run_bundle
from relic.orchestration.graph_ir import GraphIR
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture


def test_run_bundle_system_l3():
    integration_plan = IntegrationPlan(
        plan_id="run_bundle_metadata_plan",
        scheme="single",
        stages=(
            IntegrationStage(
                stage_id="bucket_step",
                operator_kind="water_balance",
                solver_name="implicit_midpoint",
                runtime_owner="solver_registry", structure_kind="iphs", dt_fraction=1.0, coupling_mode="direct",
            ),
        ),
    ).to_dict()
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
        "metadata": {"integration_plan": integration_plan},
    }
    graph = GraphIR.from_dict(payload)
    registry = NodeRegistry(ssot_catalog=SSOTCatalog.from_parameter_manager())
    registry.register_functor_instance(BucketSoilMoisture())
    plan = compile_graph(graph, registry, strict=True)

    base_dir = Path(__file__).resolve().parent / "_tmp_run_bundle_l3"
    output_dir = base_dir / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = write_run_bundle(graph, plan, output_dir)

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
    for key in paths:
        assert Path(paths[key]).exists()

    workflow = json.loads(Path(paths["workflow"]).read_text(encoding="utf-8"))
    assert workflow["metadata"]["integration_plan"] == integration_plan
    assert workflow["nodes"]["0"]["class_type"] == "State:ExternalForcing"
    execution_plan = json.loads(
        Path(paths["execution_plan"]).read_text(encoding="utf-8")
    )
    assert execution_plan == plan_to_dict(plan)["steps"]
    report = json.loads(Path(paths["compile_report"]).read_text(encoding="utf-8"))
    assert report["audit"]["integration_plan_audit"]["passed"] is True
    assert report["audit"]["integration_plan_audit"]["solver_execution_count"] == 0
    lowering = report["metadata"]["state_boundary_lowering"]
    assert lowering["required_initial_state_keys"] == [
        "evapotranspiration_flux",
        "precipitation_flux",
        "soil_moisture",
    ]
    assert lowering["executable_node_ids"] == ["1"]
    receipt = ProvenanceRecorder(output_dir=output_dir).verify_run_bundle()
    assert receipt["file_count"] == 10
    assert receipt["checksums_verified"] == 9
    assert receipt["valid"] is True

    shutil.rmtree(output_dir, ignore_errors=True)
