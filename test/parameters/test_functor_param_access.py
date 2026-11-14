# -*- coding: utf-8 -*-
"""
Unit tests for Functor parameter access through UnifiedParameterManager.

ASCII-only comments and prints.
"""
import re

import pytest
import torch

from relic.core.functor import FunctorBase, FunctorMetadata, ProcessCategory, ExecutionPhase
from relic.core.device_policy import is_cuda_available
from relic.parameters import get_global_manager, ParameterSpec, ParameterCategory


class ParamTestFunctor(FunctorBase):
    """
    Minimal functor that requests a parameter via get_parameter_value.

    References: None. Pure configuration path test.
    Formula: Not applicable.
    Parameters: alpha (dimensionless).
    """
    def __init__(self, parameter_manager=None, device=None):
        meta = FunctorMetadata(
            name="test_functor",
            description="Param access test functor",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            input_variables=["diagnostics.parameter_test.input"],
            output_variables=["diagnostics.parameter_test.output"],
            mathematical_expression="None",
            references=["None"]
        )
        super().__init__(name="test_functor", config={}, metadata=meta,
                         parameter_manager=parameter_manager, device=device)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        return {}


def test_a_missing_parameter_is_not_supplied_or_registered_by_the_caller():
    """No caller default and no auto-registration: the YAML database owns values."""
    pm = get_global_manager()

    f = ParamTestFunctor()
    full_key = f"{f.metadata.category.to_namespace_prefix()}.alpha"
    assert f.get_parameter_value(name="alpha", units="dimensionless", bounds=(0.0, 10.0)) is None
    assert pm.get_parameter(full_key) is None
    with pytest.raises(ValueError, match="alpha"):
        f.get_parameter_value(name="alpha", required=True)
    for removed in ({"default": 2.5}, {"register_if_missing": True}):
        with pytest.raises(TypeError):
            f.get_parameter_value(name="alpha", **removed)


def test_functor_parameter_returns_trainable_tensor():
    pm = get_global_manager()
    functor = ParamTestFunctor()
    full_key = f"{functor.metadata.category.to_namespace_prefix()}.trainable_alpha"

    spec = ParameterSpec(
        key=full_key,
        value=1.25,
        units="dimensionless",
        category=ParameterCategory.CALIBRATION,
        requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")),
    )
    pm.register_parameter(spec)

    value = functor.get_parameter_value(
        name="trainable_alpha",
        units="dimensionless",
        validate=True,
        required=True,
    )

    assert isinstance(value, torch.Tensor)
    assert value.requires_grad
    assert full_key in pm.nn_parameters
    assert value is pm.nn_parameters[full_key]


def test_functor_does_not_eagerly_copy_manager_namespace_parameters():
    functor = ParamTestFunctor()

    # ParameterManager is the SSOT. Functor construction must not duplicate an
    # entire manager namespace into local param_* parameters.
    duplicate_names = [
        name for name, _ in functor.named_parameters() if name.startswith("param_")
    ]

    assert duplicate_names == []
    assert functor.parameter_values == {}


def test_functor_reuses_single_manager_parameter_for_autograd():
    pm = get_global_manager()
    full_key = "biosphere.single_instance_alpha_20260716"
    spec = ParameterSpec(
        key=full_key,
        value=1.5,
        units="dimensionless",
        category=ParameterCategory.CALIBRATION,
        requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf")),
    )
    pm.register_parameter(spec)

    functor = ParamTestFunctor()
    second = ParamTestFunctor()
    for holder in (functor, second):
        holder.single_instance_alpha = holder.get_parameter_value(
            name="single_instance_alpha_20260716",
            units="dimensionless",
            validate=True,
            required=True,
        )

    # One theta, owned by the ParameterManager (item 74, option A): the
    # trainable interface returns it, each Functor attribute is that object,
    # held by reference and not registered as a Functor parameter.
    theta = pm.get_trainable_parameters()[full_key]
    assert theta is pm.nn_parameters[full_key]
    for holder in (functor, second):
        assert holder.single_instance_alpha is theta
        assert holder.held_manager_parameters() == {full_key: theta}
        assert dict(holder.named_parameters()) == {}

    # Gradients of both holders accumulate in the one theta:
    # d/dtheta (theta^2 + theta^2) = 4 * theta = 6.0 at theta = 1.5.
    loss = functor.single_instance_alpha.square() + second.single_instance_alpha.square()
    loss.backward()
    assert theta.grad is not None
    assert theta.grad.item() == pytest.approx(6.0)


@pytest.mark.skipif(not is_cuda_available(), reason="Cross-device getter requires a CUDA device")
@pytest.mark.parametrize("manager_device,holder_device", [("cuda", "cpu"), ("cpu", "cuda")])
def test_getter_keeps_own_binding_identity_before_registration(manager_device, holder_device):
    manager = get_global_manager()
    key = "biosphere.cross_device_binding"
    manager.register_parameter(ParameterSpec(
        key=key, value=1.5, units="1", dtype="float32", bounds=(0.0, 10.0),
        category=ParameterCategory.CALIBRATION, requires_grad=True,
    ))
    manager.to(manager_device)
    binding = manager.nn_parameters[key]
    functor = ParamTestFunctor(parameter_manager=manager, device=holder_device)
    assert functor.device.type != binding.device.type
    functor.alpha = functor.get_parameter_value("cross_device_binding", required=True)
    assert functor.alpha is binding
    assert functor.manager_binding_key(functor.alpha) == key
    assert functor.held_manager_parameters() == {key: binding}
    assert dict(functor.named_parameters()) == {}
    assert "alpha" not in functor.state_dict()
    assert binding.device.type == manager_device
    functor.alpha.square().backward()
    assert binding.grad.item() == pytest.approx(3.0)


def test_getter_keeps_the_runtime_dtype_of_an_own_manager_binding():
    manager = get_global_manager()
    key = "biosphere.manager_runtime_dtype"
    manager.register_parameter(ParameterSpec(
        key=key, value=1.5, units="1", dtype="float32", bounds=(0.0, 10.0),
        category=ParameterCategory.CALIBRATION, requires_grad=True,
    ))
    manager.to("cpu", dtype=torch.float64)
    binding = manager.nn_parameters[key]
    functor = ParamTestFunctor(parameter_manager=manager)
    functor.alpha = functor.get_parameter_value("manager_runtime_dtype", required=True)
    assert functor.alpha is binding
    assert binding.dtype == torch.float64
    assert manager.get_parameter(key).dtype == "float32"
    functor.alpha.square().backward()
    assert binding.grad.dtype == torch.float64
    assert binding.grad.item() == pytest.approx(3.0)


@pytest.mark.skipif(not is_cuda_available(), reason="Getter registration and restart require CUDA")
@pytest.mark.parametrize("target_device", ["cpu", "cuda"])
def test_getter_binding_survives_registration_and_manager_checkpoint(tmp_path, target_device):
    from relic.core.simulator import RELICSimulator
    from relic.core.simulator.checkpoint import _decode_checkpoint_value
    from test.fixtures.dummy_functor import DummyIncrementFunctor

    manager = get_global_manager()
    key = "biosphere.registered_getter_binding"
    manager.register_parameter(ParameterSpec(
        key=key, value=1.5, units="1", dtype="float32", bounds=(0.0, 10.0),
        category=ParameterCategory.CALIBRATION, requires_grad=True,
    ))
    manager.to("cuda" if target_device == "cpu" else "cpu")
    binding = manager.nn_parameters[key]
    functor = DummyIncrementFunctor(name="getter_binding")
    functor.parameter_manager = manager
    functor.alpha = functor.get_parameter_value("registered_getter_binding", required=True)
    functor.alpha.square().backward()
    simulator = RELICSimulator({
        "device": target_device, "conservation_checks": False, "strict_dependencies": False,
        "adaptive_dt_enabled": False, "state_units": {"x": "1"},
    })
    simulator.initialize_state({"x": torch.tensor(1.0)})
    simulator.register_functor(functor, execution_group="fast")
    assert functor.alpha is binding
    assert binding.device.type == target_device
    assert binding.grad.device.type == target_device
    assert functor.held_manager_parameters() == {key: binding}
    with torch.no_grad():
        binding.fill_(3.25)
    path = tmp_path / "getter_binding.pt"
    receipt = simulator.create_checkpoint(path)
    assert receipt["format_version"] == 3
    entry = torch.load(path, map_location="cpu", weights_only=True)["functors"][functor.name]
    saved = _decode_checkpoint_value(entry["parameter_bindings"], path="getter_binding")
    assert set(saved) == {key}
    assert saved[key].item() == pytest.approx(3.25)
    assert "alpha" not in _decode_checkpoint_value(entry["state_dict"], path="getter_state")
    with torch.no_grad():
        binding.fill_(9.0)
    simulator.restore_checkpoint(path)
    assert functor.alpha is manager.nn_parameters[key] is binding
    assert binding.item() == pytest.approx(3.25)
    assert binding.grad.item() == pytest.approx(3.0)


@pytest.mark.skipif(not is_cuda_available(), reason="Private candidate transfer requires CUDA")
def test_cross_device_private_config_candidate_is_not_an_own_binding():
    manager = get_global_manager()
    key = "biosphere.private_device_candidate"
    manager.register_parameter(ParameterSpec(
        key=key, value=1.5, units="1", dtype="float32", bounds=(0.0, 10.0),
        category=ParameterCategory.CALIBRATION, requires_grad=True,
    ))
    manager.to("cuda")
    candidate = torch.nn.Parameter(torch.tensor(0.75, device="cuda"))
    functor = ParamTestFunctor(parameter_manager=manager)
    functor._original_config["private_device_candidate"] = candidate
    functor.alpha = functor.get_parameter_value(
        "private_device_candidate", source_order=("config", "manager"), required=True,
    )
    assert functor.alpha.device.type == "cpu"
    assert functor.alpha.item() == pytest.approx(0.75)
    assert functor.manager_binding_key(functor.alpha) is None
    assert functor.held_manager_parameters() == {}
    functor.alpha.square().backward()
    assert candidate.grad.item() == pytest.approx(1.5)
    assert manager.nn_parameters[key].item() == pytest.approx(1.5)
    assert manager.nn_parameters[key].grad is None


@pytest.mark.skipif(not is_cuda_available(), reason="Unit-conversion view transfer requires CUDA")
def test_cross_device_unit_conversion_is_a_view_not_an_own_binding():
    manager = get_global_manager()
    key = "biosphere.converted_device_mass"
    manager.register_parameter(ParameterSpec(
        key=key, value=1.5, units="kg", dtype="float32", bounds=(0.0, 10.0),
        category=ParameterCategory.CALIBRATION, requires_grad=True,
    ))
    manager.to("cuda")
    binding = manager.nn_parameters[key]
    functor = ParamTestFunctor(parameter_manager=manager)
    functor.mass_g = functor.get_parameter_value("converted_device_mass", units="g", required=True)
    assert functor.mass_g.device.type == "cpu"
    assert functor.mass_g.item() == pytest.approx(1500.0)
    assert functor.manager_binding_key(functor.mass_g) is None
    assert functor.held_manager_parameters() == {}
    functor.mass_g.backward()
    assert binding.grad.item() == pytest.approx(1000.0)
    assert binding.item() == pytest.approx(1.5)
    assert binding.device.type == "cuda"


def test_equal_private_parameter_from_config_is_not_an_own_binding():
    manager = get_global_manager()
    key = "biosphere.equal_private_parameter"
    manager.register_parameter(ParameterSpec(
        key=key, value=1.5, units="1", dtype="float32", bounds=(0.0, 10.0),
        category=ParameterCategory.CALIBRATION, requires_grad=True,
    ))
    candidate = torch.nn.Parameter(manager.nn_parameters[key].detach().clone())
    functor = ParamTestFunctor(parameter_manager=manager)
    functor._original_config["equal_private_parameter"] = candidate
    functor.alpha = functor.get_parameter_value(
        "equal_private_parameter", source_order=("config", "manager"), required=True,
    )
    assert functor.alpha is candidate
    assert functor.manager_binding_key(functor.alpha) is None
    assert functor.held_manager_parameters() == {}
    assert dict(functor.named_parameters()) == {"alpha": candidate}


@pytest.mark.parametrize("failure_type", [OSError, ValueError, KeyError])
@pytest.mark.parametrize("trainable", [False, True])
def test_manager_read_failure_does_not_select_a_different_parameter_source(monkeypatch, failure_type, trainable):
    manager = get_global_manager()
    functor = ParamTestFunctor()
    key = "biosphere.resolution_failure_parameter"
    manager.register_parameter(ParameterSpec(
        key=key, value=2.0, units="1", bounds=(0.0, 10.0),
        category=ParameterCategory.CALIBRATION, requires_grad=trainable, dtype="float32",
    ))
    functor._original_config["resolution_failure_parameter"] = 3.0

    def fail_read(*args, **kwargs):
        raise failure_type("manager read failed")

    monkeypatch.setattr(manager, "get", fail_read)
    with pytest.raises(failure_type, match="manager read failed"):
        functor.get_parameter_value("resolution_failure_parameter", required=True)


@pytest.mark.parametrize("writer", ["config", "set_parameter_value"])
def test_missing_manager_value_uses_the_next_declared_source(writer):
    """The config is the one instance-level source; set_parameter_value writes it."""
    functor = ParamTestFunctor()
    name = "missing_resolution_parameter"
    assert get_global_manager().get_parameter("biosphere." + name) is None
    if writer == "config":
        functor._original_config[name] = 3.0
    else:
        functor.set_parameter_value(name, 3.0)
        assert functor._original_config["parameters"][name] == 3.0
    value = functor.get_parameter_value(name, required=True)
    assert value.item() == 3.0


def test_config_first_resolution_does_not_read_manager_value(monkeypatch):
    functor = ParamTestFunctor()
    functor._original_config["explicit_config_parameter"] = 3.0

    def forbidden_read(*args, **kwargs):
        pytest.fail("config-first resolution must not read the manager value")

    monkeypatch.setattr(get_global_manager(), "get", forbidden_read)
    value = functor.get_parameter_value(
        "explicit_config_parameter", source_order=("config", "manager"), required=True,
    )
    assert value.item() == 3.0


def test_config_source_units_rebuild_conversion_graph_and_observe_current_tensor():
    manager = get_global_manager()
    manager.register_parameter(ParameterSpec(
        key="biosphere.source_fraction", value=0.5, units="mol mol-1", dtype="float64",
        bounds=(0.0, 1.0), category=ParameterCategory.CALIBRATION,
    ))
    functor = ParamTestFunctor()
    source = torch.tensor([1e-4, 2e-4], dtype=torch.float64, requires_grad=True)
    descriptor = {"value": source, "units": "ppb"}
    functor._original_config["parameters"] = {"source_fraction": descriptor}
    for multiplier in (1.0, 2.0):
        with torch.no_grad():
            source.copy_(torch.tensor([1e-4, 2e-4], dtype=torch.float64) * multiplier)
        source.grad = None
        value = functor.get_parameter_value("source_fraction", source_order=("config", "manager"), required=True)
        torch.testing.assert_close(value, source * 1e-9, rtol=1e-12, atol=0)
        value.sum().backward()
        torch.testing.assert_close(source.grad, torch.full_like(source, 1e-9), rtol=1e-12, atol=0)
    assert descriptor["value"] is source and descriptor["units"] == "ppb"
    assert manager.get("biosphere.source_fraction", required=True) == 0.5


@pytest.mark.parametrize("validate", [True, False])
def test_config_source_and_requested_units_are_each_applied_once(validate):
    manager = get_global_manager()
    manager.register_parameter(ParameterSpec(
        key="biosphere.source_mass", value=0.5, units="kg", dtype="float64",
        bounds=(0.0, 1.0), category=ParameterCategory.CALIBRATION,
    ))
    functor = ParamTestFunctor()
    functor._original_config["source_mass"] = {"value": 250.0, "units": "g"}
    assert functor.get_parameter_value("source_mass", source_order=("config",), required=True, validate=validate).item() == 0.25
    assert functor.get_parameter_value("source_mass", units="g", source_order=("config",), required=True, validate=validate).item() == 250.0


def test_config_dimensionless_source_alias_is_explicitly_authorized():
    get_global_manager().register_parameter(ParameterSpec(
        key="biosphere.source_alias", value=0.5, units="dimensionless",
        bounds=(0.0, 1.0), category=ParameterCategory.CALIBRATION, dtype="float32",
    ))
    functor = ParamTestFunctor()
    functor._original_config["source_alias"] = {"value": 0.25, "units": "1"}
    assert functor.get_parameter_value("source_alias", source_order=("config",), required=True).item() == 0.25


@pytest.mark.parametrize("source_units", ["m", "", None, ["kg"]])
def test_invalid_config_unit_does_not_override_manager_source(source_units):
    get_global_manager().register_parameter(ParameterSpec(
        key="biosphere.invalid_source_unit", value=0.5, units="kg", dtype="float64",
        bounds=(0.0, 1.0), category=ParameterCategory.CALIBRATION,
    ))
    functor = ParamTestFunctor()
    functor._original_config["invalid_source_unit"] = {"value": 0.25, "units": source_units}
    with pytest.raises(ValueError):
        functor.get_parameter_value("invalid_source_unit", source_order=("config",), required=True)
    assert functor.get_parameter_value("invalid_source_unit", source_order=("manager",), required=True).item() == 0.5


@pytest.mark.parametrize("trainable", [False, True])
def test_bounded_matrix_parameter_keeps_shape_dtype_and_parameter_identity(trainable):
    manager = get_global_manager()
    functor = ParamTestFunctor()
    name = f"bounded_matrix_parameter_{int(trainable)}"
    key = "biosphere." + name
    matrix = torch.tensor([[0.2, 0.8], [0.5, 1.0]], dtype=torch.float64)
    manager.register_parameter(ParameterSpec(
        key=key, value=matrix, units="1", dtype="float64", bounds=(0.0, 1.0),
        category=ParameterCategory.CALIBRATION, requires_grad=trainable,
    ))
    value = functor.get_parameter_value(name, required=True)
    torch.testing.assert_close(value, matrix)
    if trainable:
        assert value is manager.nn_parameters[key]
        value.square().sum().backward()
        torch.testing.assert_close(value.grad, 2.0 * matrix)


@pytest.mark.parametrize("invalid", [
    torch.tensor([[0.2, 0.4], [0.6, 1.01]], dtype=torch.float64),
    torch.tensor([[0.2, -0.01]], dtype=torch.float64),
    torch.tensor([[0.2, float("nan")]], dtype=torch.float64),
])
def test_config_matrix_checks_every_component_against_bounds(invalid):
    functor = ParamTestFunctor()
    functor._original_config["invalid_config_matrix"] = invalid
    with pytest.raises(ValueError, match="outside bounds"):
        functor.get_parameter_value(
            "invalid_config_matrix", source_order=("config",),
            bounds=(0.0, 1.0), required=True,
        )


def test_dimensionless_alias_conversion_does_not_skip_matrix_bounds():
    # The spec interval is open on both sides, so the only bound in force is
    # the caller's explicit (0, 1) interval checked below.
    manager = get_global_manager()
    manager.register_parameter(ParameterSpec(
        key="biosphere.dimensionless_matrix_alias", value=0.5, units="1",
        bounds=(float("-inf"), float("inf")), category=ParameterCategory.CALIBRATION,
        dtype="float32",
    ))
    functor = ParamTestFunctor()
    functor._original_config["dimensionless_matrix_alias"] = torch.tensor([0.2, 2.0])
    with pytest.raises(ValueError, match="outside bounds"):
        functor.get_parameter_value(
            "dimensionless_matrix_alias", units="dimensionless", bounds=(0.0, 1.0),
            source_order=("config",), required=True,
        )


def test_unit_alias_preserves_float64_bound_endpoints_and_config_gradient():
    manager = get_global_manager()
    manager.register_parameter(ParameterSpec(
        key="biosphere.exact_alias_endpoints", value=0.5, units="1", dtype="float64",
        bounds=(0.1, 0.9), category=ParameterCategory.CALIBRATION,
    ))
    functor = ParamTestFunctor()
    configured = torch.tensor([0.1, 0.9], dtype=torch.float64, requires_grad=True)
    functor._original_config["exact_alias_endpoints"] = configured
    value = functor.get_parameter_value(
        "exact_alias_endpoints", units="dimensionless", source_order=("config",), required=True,
    )
    torch.testing.assert_close(value, configured, rtol=0, atol=0)
    value.sum().backward()
    torch.testing.assert_close(configured.grad, torch.ones_like(configured))


def test_config_value_cannot_strip_a_physical_unit_by_requesting_dimensionless():
    manager = get_global_manager()
    manager.register_parameter(ParameterSpec(
        key="biosphere.mass_parameter_override", value=0.5, units="kg",
        bounds=(0.0, float("inf")), category=ParameterCategory.CALIBRATION, dtype="float32",
    ))
    functor = ParamTestFunctor()
    functor._original_config["mass_parameter_override"] = 0.3
    with pytest.raises(ValueError, match="convert|conversion"):
        functor.get_parameter_value(
            "mass_parameter_override", units="dimensionless", source_order=("config",), required=True,
        )


# hydrosphere.numerical.temperature_bounds [K] is a list-valued parameter that declares one
# [minimum, maximum] pair per component: [[183.15, 273.15], [323.15, 373.15]].
_COMPONENTWISE_KEY = "hydrosphere.numerical.temperature_bounds"


def _read_temperature_domain(functor, source_order=("manager", "config")):
    return functor.get_parameter_value(
        "temperature_bounds", namespace="hydrosphere.numerical",
        source_order=source_order, required=True,
    )


@pytest.mark.parametrize(("candidate", "message"), [
    ([180.0, 323.15], "component 0 below minimum: 180.0 < 183.15"),
    ((233.15, 400.0), "component 1 above maximum: 400.0 > 373.15"),
    ([273.15], "has length 1 but its bounds declare 2 components"),
    (["273.15", 323.15], "component 0 must be a real number, got str '273.15'"),
    ([float("nan"), 323.15], "component 0 is not finite: nan"),
    (torch.tensor([273.15, 323.15]),
     "takes a list or tuple of 2 numbers (one per declared bounds pair), got Tensor"),
    (300.0, "takes a list or tuple of 2 numbers (one per declared bounds pair), got float"),
])
def test_config_candidate_is_checked_against_componentwise_bounds(candidate, message):
    functor = ParamTestFunctor()
    functor._original_config["temperature_bounds"] = candidate
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        _read_temperature_domain(functor, source_order=("config", "manager"))
    assert "Parameter temperature_bounds" in str(error.value)


@pytest.mark.parametrize("candidate", [[223.15, 373.15], (250.0, 350.0)])
def test_config_list_inside_its_componentwise_bounds_is_returned_as_given(candidate):
    functor = ParamTestFunctor()
    functor._original_config["temperature_bounds"] = candidate
    value = _read_temperature_domain(functor, source_order=("config", "manager"))
    assert value == candidate and type(value) is type(candidate)


def test_manager_list_value_is_checked_against_componentwise_bounds_at_the_functor_read():
    manager = get_global_manager()
    functor = ParamTestFunctor()
    assert _read_temperature_domain(functor) == manager.get(_COMPONENTWISE_KEY)
    # An unvalidated write is not a way around the declared pairs: the read rejects it.
    manager.set_value(_COMPONENTWISE_KEY, [180.0, 323.15], units="K", validate=False)
    with pytest.raises(ValueError, match=re.escape("component 0 below minimum: 180.0 < 183.15")):
        _read_temperature_domain(functor)


def test_flat_bounds_keep_their_previous_reading_for_a_config_list():
    # Unchanged: a flat (lower, upper) interval is not applied to the components of a list.
    functor = ParamTestFunctor()
    functor._original_config["flat_bounded_list"] = [5.0, -3.0]
    assert functor.get_parameter_value(
        "flat_bounded_list", bounds=(0.0, 1.0), source_order=("config",), required=True,
    ) == [5.0, -3.0]

