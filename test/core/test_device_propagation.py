import types

import pytest
import torch

from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorBase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    ScaleCouplingFunctor,
    TransportMechanismFunctor,
)
from relic.core.simulator import registration as registration_ops
from relic.utils.tensor_ops import to_tensor


def _expected_default_device() -> torch.device:
    return torch.device("cpu")


def _test_metadata(
    name: str,
    category: ProcessCategory,
    output_variables: list[str] | None = None,
) -> FunctorMetadata:
    """Build complete metadata for a device-policy test fixture."""
    return FunctorMetadata(
        name=name,
        description=f"Device-policy test fixture for {name}",
        category=category,
        mathematical_expression="device_out = device_policy(device_in)",
        references=["RELIC device-policy architecture test fixture"],
        input_variables=["test_device_input"],
        output_variables=list(output_variables or []),
    )


class _BaseDummyFunctor(FunctorBase):
    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {}


class _DummyBiological(BiologicalProcessFunctor):
    def forward(self, state, time, dt, **kwargs):
        return {"bio": torch.tensor(0.0, device=self.device)}


class _DummyTransport(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def forward(self, state, time, dt, **kwargs):
        return {"transport_flux": torch.tensor(0.0, device=self.device)}


class _DummyPhysical(PhysicalFieldFunctor):

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )
    def forward(self, state, time, dt, **kwargs):
        return {"field_value": torch.tensor(0.0, device=self.device)}


class _DummyScale(ScaleCouplingFunctor):
    def forward(self, state, time, dt, **kwargs):
        return {"scale_value": torch.tensor(0.0, device=self.device)}


class _TaggedSubmodule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")


def _build_bridge_functors(device=None):
    return [
        _DummyBiological(
            name="dummy_bio",
            config={},
            metadata=_test_metadata(
                "dummy_bio",
                ProcessCategory.BIOLOGICAL_PROCESS,
                ["bio"],
            ),
            device=device,
        ),
        _DummyTransport(
            name="dummy_transport",
            config={},
            metadata=_test_metadata(
                "dummy_transport",
                ProcessCategory.TRANSPORT_MECHANISM,
                ["transport_flux"],
            ),
            device=device,
        ),
        _DummyPhysical(
            name="dummy_physical",
            config={},
            metadata=_test_metadata(
                "dummy_physical",
                ProcessCategory.PHYSICAL_FIELD,
                ["field_value"],
            ),
            device=device,
        ),
        _DummyScale(
            name="dummy_scale",
            config={},
            metadata=_test_metadata(
                "dummy_scale",
                ProcessCategory.SCALE_COUPLING,
                ["scale_value"],
            ),
            device=device,
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        ),
    ]


def test_functor_base_sets_cpu_placeholder_device_by_default() -> None:
    functor = _BaseDummyFunctor(
        name="dummy_base",
        config={},
        metadata=_test_metadata(
            "dummy_base",
            ProcessCategory.BIOLOGICAL_PROCESS,
        ),
    )
    assert functor.device.type == _expected_default_device().type


def test_bridge_defaults_follow_cpu_placeholder_device() -> None:
    expected = _expected_default_device().type
    for functor in _build_bridge_functors():
        assert functor.device.type == expected


def test_explicit_cpu_device_is_preserved() -> None:
    explicit_cpu = torch.device("cpu")
    base_functor = _BaseDummyFunctor(
        name="dummy_base_cpu",
        config={},
        metadata=_test_metadata(
            "dummy_base_cpu",
            ProcessCategory.BIOLOGICAL_PROCESS,
        ),
        device=explicit_cpu,
    )
    assert base_functor.device.type == "cpu"

    for functor in _build_bridge_functors(device=explicit_cpu):
        assert functor.device.type == "cpu"


def test_register_migration_updates_module_device_attributes(monkeypatch) -> None:
    target_device = torch.device("cuda")
    simulator = types.SimpleNamespace(device=target_device, _device_overrides={})
    functor = _BaseDummyFunctor(
        name="dummy_migrate",
        config={},
        metadata=_test_metadata(
            "dummy_migrate",
            ProcessCategory.BIOLOGICAL_PROCESS,
        ),
    )
    functor.child = _TaggedSubmodule()

    def _fake_to_device(device):
        functor.device = torch.device(device)
        return functor

    monkeypatch.setattr(functor, "to_device", _fake_to_device, raising=False)
    monkeypatch.setattr(
        registration_ops,
        "_migrate_non_registered_tensor_attributes",
        lambda *_args, **_kwargs: None,
    )

    registration_ops._maybe_migrate_functor_device(simulator, functor)

    assert functor.device.type == "cuda"
    assert functor.child.device.type == "cuda"


def test_register_migration_honors_per_functor_override(monkeypatch) -> None:
    simulator = types.SimpleNamespace(
        device=torch.device("cpu"),
        _device_overrides={"dummy_override": torch.device("cuda")},
    )
    functor = _BaseDummyFunctor(
        name="dummy_override",
        config={},
        metadata=_test_metadata(
            "dummy_override",
            ProcessCategory.BIOLOGICAL_PROCESS,
        ),
    )
    functor.child = _TaggedSubmodule()

    def _fake_to_device(device):
        functor.device = torch.device(device)
        return functor

    monkeypatch.setattr(functor, "to_device", _fake_to_device, raising=False)
    monkeypatch.setattr(
        registration_ops,
        "_migrate_non_registered_tensor_attributes",
        lambda *_args, **_kwargs: None,
    )

    registration_ops._maybe_migrate_functor_device(simulator, functor)

    assert functor.device.type == "cuda"
    assert functor.child.device.type == "cuda"


class _BindingStandIn:
    """A ParameterManager stand-in holding one learnable binding (item 74)."""

    def __init__(self, value: float = 2.0) -> None:
        self.nn_parameters = {
            "test.theta": torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
        }

    def get_parameter(self, key):
        return None

    def get(self, key, **kwargs):
        return None


class _BindingHolder(_BaseDummyFunctor):
    """Holds the stand-in's learnable binding by reference."""

    def __init__(self, manager: _BindingStandIn, name: str) -> None:
        super().__init__(
            name=name,
            config={},
            metadata=_test_metadata(name, ProcessCategory.BIOLOGICAL_PROCESS),
            parameter_manager=manager,
        )
        self.theta = manager.nn_parameters["test.theta"]


_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="relocation across devices needs a CUDA device"
)


@_needs_cuda
def test_registration_relocates_a_held_binding_once() -> None:
    """The one ParameterManager binding follows the device policy of its holders:
    the manager moves it in place once (identity kept); a later holder on the
    same device moves nothing."""
    manager = _BindingStandIn()
    binding = manager.nn_parameters["test.theta"]
    first = _BindingHolder(manager, "holder_a")
    second = _BindingHolder(manager, "holder_b")
    simulator = types.SimpleNamespace(
        device=torch.device("cuda"), _device_overrides={}, functors={"holder_a": first},
    )
    registration_ops._maybe_migrate_functor_device(simulator, first)
    assert binding.device.type == "cuda"
    assert first.theta is binding
    assert manager.nn_parameters["test.theta"] is binding
    storage = binding.data_ptr()
    simulator.functors["holder_b"] = second
    registration_ops._maybe_migrate_functor_device(simulator, second)
    assert binding.data_ptr() == storage
    assert second.theta is binding


@_needs_cuda
def test_holders_of_one_binding_on_different_devices_are_rejected() -> None:
    """Two registered Functors holding one binding on different devices raise,
    naming the parameter, both Functors and their devices; nothing moves."""
    manager = _BindingStandIn()
    binding = manager.nn_parameters["test.theta"]
    first = _BindingHolder(manager, "holder_a")
    second = _BindingHolder(manager, "holder_b")
    simulator = types.SimpleNamespace(
        device=torch.device("cuda"),
        _device_overrides={"holder_b": torch.device("cpu")},
        functors={"holder_a": first},
    )
    registration_ops._maybe_migrate_functor_device(simulator, first)
    simulator.functors["holder_b"] = second
    with pytest.raises(ValueError, match=r"test\.theta.*holder_a.*cuda.*holder_b.*cpu"):
        registration_ops._maybe_migrate_functor_device(simulator, second)
    assert binding.device.type == "cuda"
    assert second.device.type == "cpu"


@_needs_cuda
def test_failed_migration_returns_the_binding_to_its_device(monkeypatch) -> None:
    """A registration migration that fails after moving the binding moves it back."""
    manager = _BindingStandIn()
    binding = manager.nn_parameters["test.theta"]
    holder = _BindingHolder(manager, "holder_a")
    simulator = types.SimpleNamespace(
        device=torch.device("cuda"), _device_overrides={}, functors={"holder_a": holder},
    )

    def _failing_to_device(device):
        raise RuntimeError("device move failed")

    monkeypatch.setattr(holder, "to_device", _failing_to_device, raising=False)
    with pytest.raises(RuntimeError, match="device move failed"):
        registration_ops._maybe_migrate_functor_device(simulator, holder)
    assert binding.device.type == "cpu"
    assert holder.theta is binding


def test_to_tensor_uses_cpu_placeholder_when_device_unspecified() -> None:
    tensor = to_tensor([1.0, 2.0], device=None)
    assert tensor.device.type == _expected_default_device().type
    assert torch.allclose(
        tensor.to(device=torch.device("cpu")),
        torch.tensor([1.0, 2.0], dtype=torch.float32),
    )


@pytest.mark.parametrize("prepare_device", [False, True])
@pytest.mark.parametrize("coordinated", [False, True])
def test_registration_witness_defers_only_device_preparation(monkeypatch, prepare_device, coordinated):
    from relic.core.simulator import RELICSimulator

    simulator = RELICSimulator({"device": "cpu"})
    functor = _DummyPhysical(name="witness_field", config={},
                             metadata=_test_metadata("witness_field", ProcessCategory.PHYSICAL_FIELD, ["field_value"]))
    calls, votes = [], []
    monkeypatch.setattr(registration_ops, "_maybe_migrate_functor_device", lambda *args: calls.append("prepare"))
    monkeypatch.setattr(simulator, "_coordination_active", lambda: coordinated)

    def coordinate(operation, *, prepare, apply, compensate):
        votes.append(prepare())
        apply()

    monkeypatch.setattr(simulator, "_run_coordinated", coordinate)
    simulator.register_functor(functor, execution_group="fast", _prepare_device=prepare_device)
    assert calls == (["prepare"] if prepare_device else [])
    assert simulator.functors[functor.name] is functor
    assert simulator._groups["fast"] == [functor.name]
    receipt = simulator.get_lifecycle_receipts()[-1]
    expected = "device_migration:" if prepare_device else "device_preparation_deferred:"
    assert any(action.startswith(expected) for action in receipt.completed_actions)
    if coordinated:
        assert votes[0]["prepare_device"] is prepare_device


def test_registration_witness_retains_output_writer_rejection():
    from relic.core.simulator import RELICSimulator

    simulator = RELICSimulator({"device": "cpu", "strict_dependencies": True})
    for name in ("first_witness", "second_witness"):
        functor = _DummyPhysical(name=name, config={},
                                 metadata=_test_metadata(name, ProcessCategory.PHYSICAL_FIELD, ["field_value"]))
        if name == "first_witness":
            simulator.register_functor(functor, _prepare_device=False)
        else:
            with pytest.raises(ValueError, match="field_value"):
                simulator.register_functor(functor, _prepare_device=False)
    assert list(simulator.functors) == ["first_witness"]
