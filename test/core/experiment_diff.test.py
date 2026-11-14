import pytest
import torch

from relic.core.simulator.main import RELICSimulator
from relic.parameters import ParameterSpec, UnifiedParameterManager
from relic.parameters.serialization import serialize_tensor
from relic.parameters.ops.snapshots import (
    compute_parameter_diff,
    generate_experiment_diff_report,
)


def test_parameter_diff_numeric() -> None:
    baseline = {"parameters": {"a": {"value": 1.0}}}
    current = {"parameters": {"a": {"value": 2.5}}}

    diff = compute_parameter_diff(current, baseline)
    assert diff["total_changes"] == 1
    assert len(diff["changed_parameters"]) == 1
    assert diff["changed_parameters"][0]["delta"] == 1.5


def _tensor_snapshot(value):
    spec = ParameterSpec(key="test.tensor_diff", value=value, units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
    return {"parameters": {spec.key: spec.to_dict()}}


def test_parameter_diff_decodes_scalar_tensor_wire():
    baseline = _tensor_snapshot(torch.tensor(1.0))
    current = _tensor_snapshot(torch.tensor(2.0))
    item = compute_parameter_diff(current, baseline)["changed_parameters"][0]
    assert item["kind"] == "numeric"
    assert item["baseline"] == 1.0
    assert item["current"] == 2.0
    assert item["delta"] == 1.0
    assert item["relative_change"] == 1.0
    report = generate_experiment_diff_report(current, baseline, {}, {})
    assert "delta=1.0" in report
    assert "'type': 'tensor'" not in report


@pytest.mark.parametrize("field", ["dtype", "shape", "requires_grad"])
def test_parameter_diff_retains_tensor_contract_changes_at_equal_values(field):
    baseline_value = torch.tensor(1.0, dtype=torch.float32)
    if field == "dtype":
        current_value = baseline_value.to(torch.float64)
    elif field == "shape":
        current_value = baseline_value.reshape(1)
    else:
        current_value = baseline_value.clone().requires_grad_(True)
    baseline = _tensor_snapshot(baseline_value)
    current = _tensor_snapshot(current_value)
    diff = compute_parameter_diff(current, baseline)
    assert diff["total_changes"] == 1
    assert field in diff["changed_parameters"][0]["contract_changes"]
    assert field in generate_experiment_diff_report(current, baseline, {}, {})


def test_parameter_diff_compares_tensor_arrays_without_ambiguous_truth_values():
    baseline = _tensor_snapshot(torch.tensor([1.0, 2.0]))
    assert compute_parameter_diff(baseline, baseline)["total_changes"] == 0
    current = _tensor_snapshot(torch.tensor([1.0, 3.0]))
    item = compute_parameter_diff(current, baseline)["changed_parameters"][0]
    assert item["kind"] == "tensor"
    assert torch.equal(item["current"], torch.tensor([1.0, 3.0]))


@pytest.mark.parametrize("baseline_form", ["snapshot", "json_wire"])
def test_parameter_diff_compares_live_learnable_values_of_manager_snapshots(baseline_form):
    """Manager snapshots keep the declared value and the live learnable value apart;
    the experiment diff compares the live value (the value manager.get returns)."""
    manager = UnifiedParameterManager()
    manager.clear_parameters()
    spec = ParameterSpec(key="test.live_experiment_diff", value=1.0, units="1", requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
    manager.register_parameter(spec)
    baseline = manager.snapshot()
    if baseline_form == "json_wire":
        # A JSON baseline carries its live values in the Tensor wire format.
        baseline = {
            "parameters": baseline["parameters"],
            "nn_values": {key: serialize_tensor(value) for key, value in baseline["nn_values"].items()},
        }
    with torch.no_grad():
        manager.nn_parameters[spec.key].fill_(0.75)
    diff = compute_parameter_diff(manager.snapshot(), baseline)
    assert diff["total_changes"] == 1
    item = diff["changed_parameters"][0]
    assert item["key"] == spec.key and item["kind"] == "numeric"
    assert item["baseline"] == 1.0 and item["current"] == 0.75
    assert item["delta"] == -0.25
    assert compute_parameter_diff(baseline, baseline)["total_changes"] == 0


def test_parameter_diff_categorical() -> None:
    baseline = {"parameters": {"mode": {"value": "linear"}}}
    current = {"parameters": {"mode": {"value": "exponential"}}}

    diff = compute_parameter_diff(current, baseline)
    assert diff["total_changes"] == 1
    assert diff["changed_parameters"][0]["kind"] == "categorical"


def test_diff_report_generation() -> None:
    baseline_snapshot = {"parameters": {"a": {"value": 1.0}, "b": {"value": "x"}}}
    current_snapshot = {"parameters": {"a": {"value": 2.0}, "b": {"value": "y"}, "c": {"value": 3.0}}}
    baseline_config = {"solver": {"mode": "lie", "dt": 60.0}}
    current_config = {"solver": {"mode": "strang", "dt": 60.0}}

    report = generate_experiment_diff_report(
        current_snapshot=current_snapshot,
        baseline_snapshot=baseline_snapshot,
        current_config=current_config,
        baseline_config=baseline_config,
    )

    assert "Experiment Differential Report" in report
    assert "Changed Parameters" in report
    assert "Configuration Differences" in report


def test_no_baseline_graceful() -> None:
    simulator = RELICSimulator(config={"time_step": 60.0, "max_time": 120.0})
    assert simulator.get_experiment_diff() is None
