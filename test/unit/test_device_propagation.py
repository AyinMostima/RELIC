import types

import pytest
import torch

from relic.core import device_policy
from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorBase,
    PhysicalFieldFunctor,
    ScaleCouplingFunctor,
    TransportMechanismFunctor,
)
from relic.core.simulator import registration as registration_ops
from relic.core.simulator.main import (
    RELICSimulator,
    _resolve_runtime_device,
    _resolve_single_device,
)
from relic.utils.tensor_ops import to_tensor

EXPECTED_PRE_REG_DEVICE = torch.device("cpu")


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
        _DummyBiological(name="dummy_bio", config={}, device=device),
        _DummyTransport(name="dummy_transport", config={}, device=device),
        _DummyPhysical(name="dummy_physical", config={}, device=device),
        _DummyScale(name="dummy_scale", config={}, device=device, source_scale="fine", target_scale="coarse", coupling_type="spatial"),
    ]


def test_functor_base_default_is_cpu_placeholder() -> None:
    functor = _BaseDummyFunctor(name="dummy_base", config={})
    assert functor.device == EXPECTED_PRE_REG_DEVICE


def test_functor_base_respects_explicit_device() -> None:
    explicit_device = torch.device("cpu")
    functor = _BaseDummyFunctor(name="dummy_base_cpu", config={}, device=explicit_device)
    assert functor.device == explicit_device


def test_bridge_defaults_use_cpu_placeholder() -> None:
    for functor in _build_bridge_functors():
        assert functor.device == EXPECTED_PRE_REG_DEVICE


def test_to_tensor_default_uses_cpu_placeholder() -> None:
    tensor = to_tensor([1.0, 2.0], device=None)
    assert tensor.device == EXPECTED_PRE_REG_DEVICE


def test_device_policy_auto_respects_require_gpu(monkeypatch) -> None:
    monkeypatch.setattr(device_policy, "is_cuda_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        device_policy.resolve_device(explicit_device="auto", require_gpu=True)


def test_simulator_device_helpers_reuse_cuda_index_validation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(device_policy, "is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="out of range"):
        _resolve_runtime_device({"device": "cuda:99"})
    with pytest.raises(RuntimeError, match="out of range"):
        _resolve_single_device("cuda:99")


def test_registration_migrates_to_simulator_device(monkeypatch) -> None:
    target_device = torch.device("cuda")
    simulator = types.SimpleNamespace(device=target_device, _device_overrides={})
    functor = _BaseDummyFunctor(name="dummy_migrate", config={})
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


def test_registration_honors_functor_override(monkeypatch) -> None:
    simulator = types.SimpleNamespace(
        device=torch.device("cpu"),
        _device_overrides={"dummy_override": torch.device("cuda")},
    )
    functor = _BaseDummyFunctor(name="dummy_override", config={})

    def _fake_to_device(device):
        functor.device = torch.device(device)
        return functor

    monkeypatch.setattr(functor, "to_device", _fake_to_device, raising=False)

    registration_ops._maybe_migrate_functor_device(simulator, functor)

    assert functor.device.type == "cuda"


def test_simulator_effective_device_prioritizes_override() -> None:
    simulator = RELICSimulator(
        config={
            "device": "auto",
            "device_overrides": {"dummy_bio": "cpu"},
        }
    )

    assert simulator.get_effective_device("dummy_bio") == torch.device("cpu")
    assert simulator.get_effective_device("unknown_functor") == simulator.device


@pytest.mark.parametrize("key", ["device_overrides", "module_device_overrides"])
def test_invalid_device_override_is_rejected_with_its_cause(key) -> None:
    """An override the device policy cannot resolve is a configuration error, not a warning."""
    with pytest.raises(RuntimeError, match="dummy_target") as error:
        RELICSimulator(config={"device": "cpu", key: {"dummy_target": "no_such_device:7"}})
    assert error.value.__cause__ is not None


def test_simulator_supports_nested_device_config() -> None:
    simulator = RELICSimulator(
        config={
            "simulator": {
                "device": "cpu",
                "device_overrides": {"dummy_bio": "cpu"},
            }
        }
    )

    assert simulator.device.type == "cpu"
    assert simulator.get_effective_device("dummy_bio") == torch.device("cpu")


def test_simulator_consumes_nested_computation_config() -> None:
    simulator = RELICSimulator(
        config={
            "computation": {
                "device": "cpu",
                "enable_profiling": True,
                "device_overrides": {"dummy_bio": "cpu"},
            }
        }
    )

    assert simulator.device.type == "cpu"
    assert simulator.enable_profiling is True
    assert simulator.get_effective_device("dummy_bio") == torch.device("cpu")


def test_simulator_synchronizes_device_policy_overrides(monkeypatch) -> None:
    captured: dict = {}

    def _fake_set_default_device(device, **kwargs):
        captured["default_device"] = torch.device(device)
        captured["default_kwargs"] = kwargs

    def _fake_set_device_overrides(overrides, **kwargs):
        captured["overrides"] = dict(overrides)
        captured["override_kwargs"] = kwargs

    monkeypatch.setattr(device_policy, "set_default_device", _fake_set_default_device)
    monkeypatch.setattr(device_policy, "set_device_overrides", _fake_set_device_overrides)

    RELICSimulator(
        config={
            "simulator": {
                "device": "cpu",
                "device_overrides": {"dummy_bio": "cpu"},
                "module_device_overrides": {"dummy_module": "cpu"},
            }
        }
    )

    assert captured["default_device"] == torch.device("cpu")
    assert captured["override_kwargs"]["clear_existing"] is True
    assert captured["overrides"]["dummy_bio"] == torch.device("cpu")
    assert captured["overrides"]["dummy_module"] == torch.device("cpu")


def test_cuda_probe_failure_is_not_reported_as_missing_cuda(monkeypatch) -> None:
    """A broken torch CUDA probe raises instead of selecting the CPU fallback."""

    def _broken_probe() -> bool:
        raise RuntimeError("cuda probe broke")

    monkeypatch.setattr(torch.cuda, "is_available", _broken_probe)

    with pytest.raises(RuntimeError, match="cuda probe broke"):
        device_policy.is_cuda_available()


def test_unparseable_device_override_is_rejected_with_its_key(monkeypatch) -> None:
    monkeypatch.setattr(device_policy, "_RUNTIME_OVERRIDES", {})

    with pytest.raises(RuntimeError, match="relic.dummy.module"):
        device_policy.set_device_overrides({"relic.dummy.module": "not_a_device"})
    assert device_policy.get_device_overrides() == {}


def test_required_gpu_override_is_not_dropped_when_cuda_is_missing(monkeypatch) -> None:
    """require_gpu must not fall through to the CPU default via a dropped override."""
    monkeypatch.setattr(device_policy, "is_cuda_available", lambda: False)

    with pytest.raises(RuntimeError, match="relic.dummy.gpu_module"):
        device_policy.resolve_device(
            module_name="relic.dummy.gpu_module",
            config={"device_overrides": {"relic.dummy.gpu_module": "cuda"}},
            require_gpu=True,
        )


def test_non_mapping_config_device_overrides_are_rejected() -> None:
    with pytest.raises(TypeError, match="device_overrides"):
        device_policy.resolve_device(
            module_name="relic.dummy.module",
            config={"device_overrides": ["cuda"]},
        )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"relic.dummy.env_module": "cpu"', "not valid JSON"),
        ('["cuda"]', "JSON object"),
    ],
)
def test_invalid_env_device_overrides_are_rejected(monkeypatch, raw, message) -> None:
    monkeypatch.setenv("RELIC_DEVICE_OVERRIDES_JSON", raw)

    with pytest.raises(ValueError, match=message):
        device_policy.resolve_device(module_name="relic.dummy.env_module")


def test_runtime_override_is_checked_against_each_request_require_gpu(monkeypatch) -> None:
    """set_device_overrides keeps the request; require_gpu is applied per resolution."""
    monkeypatch.setattr(device_policy, "_RUNTIME_OVERRIDES", {})
    monkeypatch.setattr(device_policy, "is_cuda_available", lambda: False)

    device_policy.set_device_overrides({"relic.dummy.runtime_module": "cuda"})

    assert device_policy.resolve_device(module_name="relic.dummy.runtime_module") == torch.device("cpu")
    assert device_policy.get_device_overrides() == {"relic.dummy.runtime_module": torch.device("cpu")}
    with pytest.raises(RuntimeError, match="relic.dummy.runtime_module"):
        device_policy.resolve_device(module_name="relic.dummy.runtime_module", require_gpu=True)


def test_unrelated_unavailable_override_does_not_fail_another_module(monkeypatch) -> None:
    monkeypatch.setattr(device_policy, "is_cuda_available", lambda: False)

    resolved = device_policy.resolve_device(
        module_name="relic.dummy.cpu_module",
        config={
            "device_overrides": {
                "relic.dummy.cpu_module": "cpu",
                "relic.dummy.gpu_module": "cuda",
            }
        },
        require_gpu=True,
    )

    assert resolved == torch.device("cpu")


def test_env_override_cache_does_not_hide_a_require_gpu_failure(monkeypatch) -> None:
    """A parse cached without require_gpu cannot answer a require_gpu request."""
    monkeypatch.setattr(device_policy, "is_cuda_available", lambda: False)
    monkeypatch.setenv(
        "RELIC_DEVICE_OVERRIDES_JSON", '{"relic.dummy.cached_module": "cuda"}'
    )

    fallback = device_policy.resolve_device(module_name="relic.dummy.cached_module")
    assert fallback == torch.device("cpu")
    with pytest.raises(RuntimeError, match="relic.dummy.cached_module"):
        device_policy.resolve_device(
            module_name="relic.dummy.cached_module",
            require_gpu=True,
        )
