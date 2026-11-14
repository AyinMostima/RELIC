# Devices and Distributed Execution

RELIC centralizes Central Processing Unit (CPU) and Graphics Processing Unit
(GPU) resolution in `relic.core.device_policy`. Scientific modules receive
their final device from simulator registration and must not perform independent
Compute Unified Device Architecture (CUDA) detection.

## Device policy

```text
resolve_device(
    explicit_device=None,
    *,
    module_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    default="cpu",
    distributed: bool = False,
    require_gpu: bool = False,
) -> torch.device
```

Resolution order is:

1. `explicit_device`;
2. module override in `config`;
3. process-wide runtime override or environment override;
4. global device in `config`;
5. process-wide default;
6. `RELIC_DEFAULT_DEVICE`;
7. distributed GPU-first fallback or the supplied default.

Accepted device requests are `auto`, `cpu`, `cuda`, `cuda:N`, and
`torch.device` values. `auto` and tier-7 distributed default resolution prefer
CUDA when available and fall back to CPU when CUDA is unavailable.
`require_gpu=True` turns any CPU fallback or unavailable CUDA request into
`RuntimeError`. An explicit `enable_distributed: true` setting that cannot be
satisfied (unavailable distributed runtime, configuration validation errors,
process-group setup failure, or a GPU `local_rank` without a visible CUDA
device) raises immediately without silent fallback.

```python
from relic.core.device_policy import resolve_device

device = resolve_device(
    config={
        "device": "auto",
        "module_device_overrides": {
            "relic.scientific.biosphere": "cpu",
        },
    },
    module_name="relic.scientific.biosphere",
)
print(device)
```

### Process-wide policy

```python
from relic.core.device_policy import (
    get_device_overrides,
    set_default_device,
    set_device_overrides,
)

set_default_device("auto")
set_device_overrides(
    {"relic.scientific.biosphere": "cpu"},
    clear_existing=True,
)
print(get_device_overrides())
set_device_overrides({}, clear_existing=True)
```

Environment default:

```powershell
$env:RELIC_DEFAULT_DEVICE = "auto"
python -c "from relic.core.device_policy import resolve_device; print(resolve_device())"
Remove-Item Env:RELIC_DEFAULT_DEVICE
```

Use explicit configuration for reproducible runs. Process-wide setters and
environment variables are most useful for launch policy and diagnostics.

## Simulator device configuration

```python
from relic.core.simulator import RELICSimulator

simulator = RELICSimulator(
    {
        "device": "auto",
        "require_gpu": False,
        "device_overrides": {
            "farquhar_photosynthesis": "cuda",
            "management_coordinator": "cpu",
        },
        "module_device_overrides": {
            "relic.scientific.pedosphere": "cpu",
        },
    }
)
print(simulator.get_effective_device("management_coordinator"))
```

| Key | Meaning |
| --- | --- |
| `device` | Simulator-wide default. |
| `require_gpu` | Reject CPU fallback for the configured request. |
| `device_overrides` | Per-Functor requests keyed by registered Functor name. |
| `module_device_overrides` | Per-module requests used by central policy resolution. |

The lifecycle is:

```text
Functor construction on CPU placeholder
  -> simulator registration
  -> effective device resolution
  -> module and tensor migration
  -> execution
```

Query the registered device and runtime device summary:

```python
from relic.core.simulator import RELICSimulator

simulator = RELICSimulator(
    {
        "device": "auto",
        "device_overrides": {"management_coordinator": "cpu"},
    }
)
device = simulator.get_effective_device("management_coordinator")
info = simulator.get_device_info()
print(device, info["distributed_enabled"])
```

`get_effective_device(functor_name)` returns the per-Functor override when
present, otherwise the simulator device. `get_device_info()` reports device
type and index, CUDA availability, distributed status, rank and world size
when initialized, and GPU memory information when applicable.

## Forbidden device patterns

Scientific modules must not:

- call `torch.cuda.is_available()` directly;
- hardcode `torch.device("cuda")`;
- select a device independently in `forward()`;
- call the simulator's private runtime-resolution methods;
- create a second global device policy.

The audit command is:

```powershell
rg "torch.cuda.is_available\(" relic
```

The canonical occurrence belongs in `relic\core\device_policy.py`.

## Distributed configuration

```text
DistributedConfig(
    backend: str = "gloo",
    master_addr: str = "localhost",
    master_port: int = 29500,
    world_size: int = 1,
    rank: Optional[int] = None,
    timeout_minutes: int = 30,
    init_method: Optional[str] = None,
    local_rank: Optional[int] = None,
    use_cuda: bool = True,
)
```

`DistributedConfig` is the single owner of these launch defaults (software
and deployment settings); `ProcessGroupManager`, `init_distributed`, and
`setup_distributed_simulation` take a `DistributedConfig` or overrides of its
fields and declare no defaults of their own.

| Parameter | Meaning |
| --- | --- |
| `backend` | PyTorch distributed backend. Use `gloo` on Windows. |
| `master_addr`, `master_port` | Rendezvous host and port. |
| `world_size` | Total number of participating processes. |
| `rank` | Global process rank. |
| `timeout_minutes` | Process-group initialization timeout. |
| `init_method` | Optional explicit PyTorch initialization method. |
| `local_rank` | Device-local rank. |
| `use_cuda` | Prefer CUDA when supported. |

## `ProcessGroupManager`

```text
ProcessGroupManager(config: Optional[DistributedConfig] = None)
```

Without a configuration the manager starts from `DistributedConfig()`;
`initialize()` restarts every cycle from a copy of the construction
configuration.

| Method | Purpose |
| --- | --- |
| `configure(config)` | Update process-group settings from a mapping. |
| `validate()` | Return configuration and backend errors. |
| `init_process_group(config)` | Initialize the PyTorch process group. |
| `destroy_process_group()` | Destroy the active group. |
| `barrier()` | Synchronize ranks. |
| `all_reduce(tensor, op=None)` | Reduce a tensor across ranks. |
| `broadcast(tensor, src=0)` | Broadcast a tensor from one rank. |
| `get_rank()` | Return the current rank. |
| `get_world_size()` | Return the active world size. |
| `cleanup()` | Release manager resources. |

Invalid backend availability, world sizes below one, and ports outside
`1024..65535` are reported by `validate()`.

## Functional distributed API

```python
from relic.computation.distributed.launcher import (
    barrier,
    cleanup_distributed,
    distributed_context,
    distributed_healthcheck,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
)
```

```text
init_distributed(**overrides) -> bool
```

`overrides` are `DistributedConfig` fields; the others keep the
`DistributedConfig` defaults, and an unknown name raises `TypeError`.

Use the context manager when initialization and cleanup belong to the same
scope:

```python
from relic.computation.distributed.launcher import (
    distributed_context,
    get_rank,
    get_world_size,
)

with distributed_context(
    backend="gloo",
    world_size=1,
    rank=0,
) as initialized:
    if initialized:
        print(get_rank(), get_world_size())
```

On Windows, Gloo is the supported default. NVIDIA Collective Communications
Library (NCCL) is CUDA-oriented and is not the normal Windows backend. Each multi-process launch must supply consistent
address, port, world size, and unique ranks.

### Health inspection

```powershell
python -c "from relic.computation.distributed.launcher import distributed_healthcheck; print(distributed_healthcheck())"
python -m relic.interfaces.cli.main compute ddp-check
```

`distributed_healthcheck()` reports PyTorch and distributed availability,
initialization status, available backends, current distributed state,
warnings, and recommendations. A backend being available does not mean a
process group is initialized.

## Simulator distributed toggle

```python
import torch

from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.core.state.types import StateVariableSpec

manager = StateManager(
    {"x": StateVariableSpec(name="x", units="dimensionless")}
)
manager.initialize({"x": torch.tensor(1.0)})
simulator = RELICSimulator({"time_step": 60.0})
simulator.initialize(manager, DependencyGraphResolver())

with simulator.enable_distributed_execution(True):
    simulator.step(60.0)
print(simulator.step_count, simulator.enable_distributed)
```

This context manager temporarily changes the simulator's distributed
execution flag. It does not initialize a PyTorch process group, assign ranks,
or launch worker processes. Call `init_distributed()` or use
`distributed_context()` before relying on collectives.

## Provider retirement and rank coordination

Provider retirement is coordinated with device completion through the existing
`relic.core.device_policy.DeviceFence` owner. A retired provider view remains
retained until its readers have left and recorded device work has completed.
Returning from a Python scheduling call alone does not establish completion
of CUDA work. Lifecycle status and receipts distinguish waiting for readers,
waiting for a device fence, disposal success, and disposal failure; supported
disposal failures can be retried through the same lifecycle owner.

With a real multi-rank process group and distributed execution enabled,
registration and retirement use
`relic.computation.distributed.launcher.coordinate_transaction`. Every rank
prepares and votes on epoch, contract, binding, device, and random number generator (RNG) identity before
application. Failed application triggers compensation and records its outcome.
A missing peer, timeout, or divergent decision is observable; this protocol
does not promise atomicity during partitions or tolerance of Byzantine ranks.
It also does not provide a coordinated multi-rank checkpoint format.

## Related documentation

- [Architecture](../architecture.md)
- [Configuration and Parameters](configuration_and_parameters.md)
- [State, Functors, and Simulation](state_functors_and_simulation.md)
- [Core Runtime API](../api/core_runtime.md)
- [Orchestration](../orchestration.md)
- [Theory: Numerics](../theory/numerics.md)
