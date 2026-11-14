"""Request-schema contracts: scientific quantities and method choices carry no silent defaults.

Boundary defaults are kept only for pure software switches (format, device,
compression, benchmark workload); sample sizes, run lengths, tolerances,
method and sampling choices must be stated by the caller.
"""

import json

import pytest
from pydantic import ValidationError

from relic.interfaces.api import schemas
from relic.interfaces.api.schemas import (
    CalibrationRequest,
    MarkedJumpRequest,
    ParameterBound,
    SDERequest,
    SensitivityRequest,
    UncertaintyRequest,
    WeatherGenerationRequest,
)

pytestmark = pytest.mark.unit

_BOUND = ParameterBound(min=0.0, max=1.0)

_COMPLETE = {
    CalibrationRequest: {
        "parameters": ["rate"],
        "parameter_bounds": {"rate": _BOUND},
        "method": "differential_evolution",
        "max_iterations": 200,
        "tolerance": 1e-8,
    },
    UncertaintyRequest: {
        "method": "ensemble",
        "uncertain_parameters": {"rate": {}},
        "n_samples": 64,
        "sampling_strategy": "sobol",
        "confidence_levels": [0.9],
    },
    WeatherGenerationRequest: {
        "duration_days": 30,
        "n_realizations": 4,
        "variables": ["temperature"],
    },
    SDERequest: {
        "process": "ou",
        "initial_value": 0.0,
        "dt_seconds": 60.0,
        "n_steps": 2,
        "n_realizations": 8,
        "seed": 1,
    },
    MarkedJumpRequest: {
        "process": "jumps",
        "dt_seconds": 60.0,
        "n_steps": 2,
        "n_realizations": 8,
        "seed": 1,
    },
}

_REQUIRED = [
    (CalibrationRequest, "method"),
    (CalibrationRequest, "max_iterations"),
    (CalibrationRequest, "tolerance"),
    (UncertaintyRequest, "n_samples"),
    (UncertaintyRequest, "sampling_strategy"),
    (UncertaintyRequest, "confidence_levels"),
    (WeatherGenerationRequest, "duration_days"),
    (WeatherGenerationRequest, "n_realizations"),
    (WeatherGenerationRequest, "variables"),
    (SDERequest, "n_realizations"),
    (MarkedJumpRequest, "n_realizations"),
]


@pytest.mark.parametrize(("model", "field"), _REQUIRED, ids=lambda v: getattr(v, "__name__", v))
def test_scientific_and_method_fields_are_required(model, field):
    payload = dict(_COMPLETE[model])
    payload.pop(field)

    with pytest.raises(ValidationError, match=field):
        model(**payload)

    assert model.model_fields[field].is_required()


@pytest.mark.parametrize("model", list(_COMPLETE), ids=lambda m: m.__name__)
def test_complete_requests_are_accepted_as_given(model):
    request = model(**_COMPLETE[model])

    for key, value in _COMPLETE[model].items():
        assert getattr(request, key) == value


def test_weather_output_format_keeps_its_boundary_default():
    request = WeatherGenerationRequest(**_COMPLETE[WeatherGenerationRequest])

    assert request.output_format == "netcdf4"


@pytest.mark.parametrize(
    ("method", "missing", "extra"),
    [
        ("morris", "n_trajectories", {"n_levels": 4}),
        ("morris", "n_levels", {"n_trajectories": 10}),
        ("sobol", "n_samples", {}),
    ],
)
def test_sensitivity_method_parameters_are_required_per_method(method, missing, extra):
    with pytest.raises(ValidationError, match=missing):
        SensitivityRequest(method=method, parameters={"rate": _BOUND}, **extra)


def test_sensitivity_requests_state_their_method_parameters():
    morris = SensitivityRequest(
        method="morris", parameters={"rate": _BOUND}, n_trajectories=10, n_levels=6
    )
    sobol = SensitivityRequest(method="sobol", parameters={"rate": _BOUND}, n_samples=256)

    assert (morris.n_trajectories, morris.n_levels) == (10, 6)
    assert sobol.n_samples == 256
    assert SensitivityRequest.model_fields["n_levels"].default is None


def test_removed_request_models_are_gone():
    """DDPCheckRequest only echoed its fields; FileInfo was never served by a route."""
    assert not hasattr(schemas, "DDPCheckRequest")
    assert not hasattr(schemas, "FileInfo")


def test_sensitivity_request_has_no_second_order_switch():
    """SobolConfig computes first-order and total indices only (saltelli estimator)."""
    assert "calc_second_order" not in SensitivityRequest.model_fields
    assert not any("estimator" in name for name in SensitivityRequest.model_fields)


# Request bodies (and the ParameterBound component nested in them); response models are not listed.
_REQUEST_MODEL_NAMES = [
    "DataReadRequest",
    "DataWriteRequest",
    "DataConvertRequest",
    "ComputeHealthRequest",
    "ComputeBenchmarkRequest",
    "ParameterBound",
    "CalibrationRequest",
    "SensitivityRequest",
    "UncertaintyRequest",
    "WeatherGenerationRequest",
    "SDERequest",
    "MarkedJumpRequest",
    "ConfigValidationRequest",
]


@pytest.mark.parametrize("name", _REQUEST_MODEL_NAMES)
def test_request_model_rejects_an_unknown_field(name):
    """A misspelled or removed field is an error, never silently dropped."""
    model = getattr(schemas, name)

    with pytest.raises(ValidationError) as exc_info:
        model.model_validate({"unexpected_field": 1})

    assert any(
        error["type"] == "extra_forbidden" and error["loc"] == ("unexpected_field",)
        for error in exc_info.value.errors()
    )


def test_every_request_model_is_listed_and_forbids_extra_fields():
    from pydantic import BaseModel

    request_models = {
        name for name, obj in vars(schemas).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == schemas.__name__
        and (name.endswith("Request") or name == "ParameterBound")
    }
    assert request_models == set(_REQUEST_MODEL_NAMES)
    for name in request_models:
        assert getattr(schemas, name).model_config.get("extra") == "forbid", name


def test_removed_second_order_switch_is_rejected_not_ignored():
    payload = {"method": "sobol", "parameters": {"rate": {"min": 0.0, "max": 1.0}}, "n_samples": 64}
    SensitivityRequest.model_validate(payload)

    with pytest.raises(ValidationError, match="calc_second_order"):
        SensitivityRequest.model_validate({**payload, "calc_second_order": True})


def test_unknown_request_field_answers_http_422():
    from fastapi.testclient import TestClient

    from relic.interfaces.api.server import create_app

    app = create_app(
        enable_cors=False,
        enable_docs=False,
        config={"security": {"api_auth": {"enabled": False}}},
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/compute/benchmark", json={"target": "transport", "n_run": 3})

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert "n_run" in json.dumps(body)
