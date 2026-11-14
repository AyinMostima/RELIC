"""CR-A-04 T0: regression digests of fast-group splitting pinned before the change.

The digests below were computed on 2026-10-19 from the RELIC_PROJECT working tree
(HEAD cf3dba3a plus the uncommitted edits present then) before CR-A-04 step 1
(scheduling.yaml) and step 2 (simulator configuration validation and splitting
admissibility). They pin the executed fast-group behaviour of the cases that stay
admissible after those steps; a changed digest means a changed trajectory and must
be explained, never re-pinned silently.

Cases (design 20261019_cr_a04_design_01, section 4, T0):
- rotation Lie and Strang: the non-autonomous rotation problem of
  test_time_space_enhancements.py (exact shear sub-flows);
- N/C chain: the governed pedosphere carbon/nitrogen chain of
  budget_ledger_governed_multistep.test.py under Lie;
- snow pack + bucket under Lie: snow_pack releases rain and melt, the bucket
  receives them in the same step (SCM flux contracts snow_melt_flux, rain_flux);
  the run includes the step in which the snowpack is depleted;
- C7-like Strang: Priestley-Taylor ET (no persistent state) + bucket, Lie and
  Strang give bitwise identical states (the C7 = C0 mechanism of CMP-O02);
- adaptive Lie: the rotation problem through run() with the adaptive step
  controller (rejections and step growth);
- split restart: Strang rotation restarted from a durable checkpoint in the middle
  of the run continues bitwise.

References: Strang (1968) SIAM J. Numer. Anal. 5(3):506-517; Hairer, Lubich and
Wanner (2006) Geometric Numerical Integration, 2nd ed., Sect. II.5; Anderson (1976)
NOAA Tech. Rep. NWS 19 and Hock (2003) J. Hydrol. 282:104-115 (snow pack);
Priestley and Taylor (1972) Mon. Weather Rev. 100:81-92 (ET).
Mathematical expression: digest = SHA-256 over (name, shape, float64 bytes) of each
recorded tensor, names in sorted order.
Parameter meanings: the recorded tensors are the per-step trajectories of the named
state variables (or final states where stated).
Units: as declared by the owning Functors (rotation dimensionless; SWE mm; soil
moisture m3 m-3; fluxes kg m-2 s-1; soil pools g m-2; vegetation pools kg m-2).
Domain/constraints: CPU, float64 test default dtype (test/conftest.py).
Sign convention: as declared by the owning Functors.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from relic.core.simulator import RELICSimulator

_TEST_ROOT = Path(__file__).resolve().parents[1]


def _load_test_module(relative: str, alias: str):
    """Load an existing test module by path to reuse its fixtures (no copies).

    The module is entered in sys.modules so that inspect finds the source of its
    classes (the provider kernel identity of a durable checkpoint needs it).
    """
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, _TEST_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_TIME_SPACE = _load_test_module("unit/test_time_space_enhancements.py", "_cr_a04_time_space_fixtures")
_BUDGET = _load_test_module(
    "integration/pedosphere/budget_ledger_governed_multistep.test.py", "_cr_a04_budget_fixtures"
)


def _digest(records: Dict[str, torch.Tensor]) -> str:
    """SHA-256 over (name, shape, float64 bytes) of every record, sorted by name."""
    digest = hashlib.sha256()
    for name in sorted(records):
        tensor = torch.as_tensor(records[name]).detach().to(device="cpu", dtype=torch.float64).contiguous()
        digest.update(name.encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _stack(rows: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack per-step records into one trajectory tensor per name."""
    return {
        name: torch.stack([torch.as_tensor(row[name], dtype=torch.float64).reshape(-1) for row in rows])
        for name in rows[0]
    }


# --------------------------------------------------------------------------
# Rotation problem (fixture reused from test_time_space_enhancements.py).
# --------------------------------------------------------------------------

_ROTATION_STEPS = 16
_ROTATION_DT = 0.125


def _rotation_simulator(splitting_mode: str, **extra) -> RELICSimulator:
    config = {
        "time_step": _ROTATION_DT,
        "splitting_mode": splitting_mode,
        "device": "cpu",
        "enable_auto_scale": False,
        **extra,
    }
    simulator = _TIME_SPACE._declared_simulator(config)
    centre = _TIME_SPACE._ROTATION_CENTRE
    simulator.initialize_state(
        {
            "osc_u": torch.tensor(centre + 1.0, dtype=torch.float64),
            "osc_v": torch.tensor(centre, dtype=torch.float64),
        }
    )
    simulator.register_functor(_TIME_SPACE.ShearSubFlowFunctor("shear_u", "osc_u", "osc_v", 1.0, True))
    simulator.register_functor(_TIME_SPACE.ShearSubFlowFunctor("shear_v", "osc_v", "osc_u", -1.0, True))
    assert list(simulator._order_fast) == ["shear_u", "shear_v"]
    return simulator


def _rotation_row(simulator: RELICSimulator) -> Dict[str, torch.Tensor]:
    state = simulator.get_state()
    return {name: torch.as_tensor(state[name]).clone() for name in ("osc_u", "osc_v")}


def _rotation_trajectory(splitting_mode: str) -> Dict[str, torch.Tensor]:
    simulator = _rotation_simulator(splitting_mode)
    rows = []
    for _ in range(_ROTATION_STEPS):
        simulator.step(_ROTATION_DT)
        rows.append(_rotation_row(simulator))
    assert simulator.current_time == pytest.approx(_ROTATION_STEPS * _ROTATION_DT)
    return _stack(rows)


# --------------------------------------------------------------------------
# Governed N/C chain (fixture reused from budget_ledger_governed_multistep.test.py).
# --------------------------------------------------------------------------

_NC_STEPS = 12
_NC_KEYS = _BUDGET.SOIL_C + _BUDGET.VEG_C + _BUDGET.SOIL_N + _BUDGET.VEG_N + (_BUDGET.PLANT_N,)


def _nc_chain_trajectory() -> Dict[str, torch.Tensor]:
    simulator = _BUDGET._build(_BUDGET._initial_state())
    rows = []
    for _ in range(_NC_STEPS):
        simulator.step(dt=_BUDGET.DT)
        rows.append({key: simulator.state_manager.get_state(key).clone() for key in _NC_KEYS})
    return _stack(rows)


# --------------------------------------------------------------------------
# Snow pack + bucket, both fast, Lie (flux contracts snow_melt_flux, rain_flux).
# --------------------------------------------------------------------------

_SNOW_STEPS = 24
_SNOW_DT = 3600.0
_SNOW_KEYS = (
    "snow_water_equivalent",
    "snow_melt_flux",
    "rain_flux",
    "soil_moisture",
    "surface_runoff_flux",
    "drainage_flux",
)


def _snow_bucket_simulator(splitting_mode: str) -> RELICSimulator:
    from relic.scientific.hydrosphere.snow.snowpack import SnowPack
    from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture

    simulator = RELICSimulator(
        {
            "time_step": _SNOW_DT,
            "device": "cpu",
            "splitting_mode": splitting_mode,
            "state_units": {"sublimation_flux": "kg m-2 s-1"},
        }
    )
    simulator.initialize_state(
        {
            "precipitation_flux": torch.tensor([3.0e-5]),
            "air_temperature": torch.tensor([278.15]),
            "snow_water_equivalent": torch.tensor([10.0]),
            "sublimation_flux": torch.tensor([0.0]),
            "soil_moisture": torch.tensor([0.25]),
            "evapotranspiration_flux": torch.tensor([0.0]),
            "rain_flux": torch.tensor([0.0]),
            "snow_melt_flux": torch.tensor([0.0]),
        }
    )
    simulator.register_functor(SnowPack(), execution_group="fast")
    simulator.register_functor(
        BucketSoilMoisture(config={"water_input_source": "rain_flux_plus_snow_melt_flux"}),
        execution_group="fast",
    )
    return simulator


def _snow_bucket_trajectory() -> Dict[str, torch.Tensor]:
    simulator = _snow_bucket_simulator("lie")
    assert list(simulator._order_fast) == ["snow_pack", "bucket_soil_moisture"]
    rows = []
    for _ in range(_SNOW_STEPS):
        simulator.step(dt=_SNOW_DT)
        state = simulator.state_manager.get_state()
        rows.append({key: state[key].clone() for key in _SNOW_KEYS})
    return _stack(rows)


# --------------------------------------------------------------------------
# C7-like: Priestley-Taylor ET (no persistent state) + bucket.
# --------------------------------------------------------------------------

_C7_STEPS = 12
_C7_DT = 1800.0
_C7_KEYS = (
    "soil_moisture",
    "evapotranspiration_flux",
    "latent_heat_flux",
    "surface_runoff_flux",
    "drainage_flux",
)


def _c7_like_trajectory(splitting_mode: str) -> Dict[str, torch.Tensor]:
    from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import PriestleyTaylorET
    from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture

    simulator = RELICSimulator({"time_step": _C7_DT, "device": "cpu", "splitting_mode": splitting_mode})
    simulator.initialize_state(
        {
            "net_radiation": torch.tensor([400.0]),
            "air_temperature": torch.tensor([298.15]),
            "atmospheric_pressure": torch.tensor([101325.0]),
            "soil_heat_flux": torch.tensor([40.0]),
            "soil_moisture": torch.tensor([0.25]),
            "precipitation_flux": torch.tensor([1.0e-5]),
            "evapotranspiration_flux": torch.tensor([0.0]),
        }
    )
    simulator.register_functor(PriestleyTaylorET(), execution_group="fast")
    simulator.register_functor(BucketSoilMoisture(), execution_group="fast")
    assert list(simulator._order_fast) == ["priestley_taylor_et", "bucket_soil_moisture"]
    rows = []
    for _ in range(_C7_STEPS):
        simulator.step(dt=_C7_DT)
        state = simulator.state_manager.get_state()
        rows.append({key: state[key].clone() for key in _C7_KEYS})
    return _stack(rows)


# --------------------------------------------------------------------------
# Adaptive Lie: rotation through run() with the adaptive step controller.
# --------------------------------------------------------------------------

_ADAPTIVE_END = 2.0
# adaptive_dt_min = 0.01 s lies below the attempts, so an attempt that fails the
# committed-change metric (Err ~ 40 at dt = 0.25 s with the database atol/rtol) is
# rolled back and retried shorter; the database dt_min (1 s) would accept every
# attempt of this dimensionless problem without a retry.
_ADAPTIVE_DT_MIN = 0.01


def _adaptive_lie_record() -> Dict[str, torch.Tensor]:
    simulator = _rotation_simulator(
        "lie", adaptive_dt_enabled=True, time_step=0.25, adaptive_dt_min=_ADAPTIVE_DT_MIN
    )
    simulator.run(start_time=0.0, end_time=_ADAPTIVE_END)
    assert simulator.current_time == pytest.approx(_ADAPTIVE_END)
    record = _rotation_row(simulator)
    record["step_count"] = torch.tensor(float(simulator.step_count))
    record["current_time"] = torch.tensor(float(simulator.current_time))
    record["adaptive_next_dt"] = torch.tensor(float(simulator._adaptive_next_dt))
    return record


# --------------------------------------------------------------------------
# Split restart: Strang rotation restarted from a mid-run durable checkpoint.
# --------------------------------------------------------------------------


def _split_restart_trajectories(tmp_path: Path):
    half = _ROTATION_STEPS // 2
    source = _rotation_simulator("strang")
    for _ in range(half):
        source.step(_ROTATION_DT)
    checkpoint = tmp_path / "strang_rotation_midrun.ckpt"
    source.create_checkpoint(checkpoint)
    uninterrupted = []
    for _ in range(half):
        source.step(_ROTATION_DT)
        uninterrupted.append(_rotation_row(source))

    resumed = _rotation_simulator("strang")
    resumed.restore_checkpoint(checkpoint)
    restarted = []
    for _ in range(half):
        resumed.step(_ROTATION_DT)
        restarted.append(_rotation_row(resumed))
    return _stack(uninterrupted), _stack(restarted)


# Pinned digests (computed before CR-A-04 step 1; see the module docstring).
_PINNED = {
    "rotation_lie": "cf1a1391e9bd5931a5e8f471bca6cc174c3d5c26c84acc7e3ff13dd8f92b1760",
    "rotation_strang": "2664f1e4eba77ff0be4703dd0172294654ac08e8f52ec0f100a291de6361bcb9",
    "nc_chain_lie": "5e95e7e0512730a736814309ff062d0b23ca216858e7757e4a63ea846fd3e778",
    "snow_bucket_lie": "26fc574b61116524e21985213a236f0b978902052f090aa6a5b00191b9666fbd",
    "c7_like": "907f175ac452ae8d043dca06197493f49da1ca934ab5d00566641a956e081719",
    "adaptive_lie": "2b89028d65ce68166bd8eb44e86a624010ff679d0be3f0a82a2d15ea90867875",
    "split_restart_strang": "5428c0594dd4a9ec3c75dd7046fae5ef5a563b051650bcfbbc071ec7948e499c",
}


def _assert_pinned(name: str, records: Dict[str, torch.Tensor]) -> None:
    observed = _digest(records)
    assert observed == _PINNED[name], (
        f"T0 digest of '{name}' changed: {observed} != {_PINNED[name]}; "
        f"records: { {key: value.tolist() for key, value in records.items()} }"
    )


@pytest.mark.parametrize("splitting_mode", ["lie", "strang"])
def test_rotation_trajectory_digest_is_pinned(splitting_mode):
    _assert_pinned(f"rotation_{splitting_mode}", _rotation_trajectory(splitting_mode))


def test_governed_nc_chain_lie_trajectory_digest_is_pinned():
    _assert_pinned("nc_chain_lie", _nc_chain_trajectory())


def test_snow_pack_bucket_lie_trajectory_digest_is_pinned():
    records = _snow_bucket_trajectory()
    swe = records["snow_water_equivalent"].reshape(-1)
    # The run covers the depletion step: SWE positive before, zero after.
    assert float(swe[0]) > 0.0 and float(swe[-1]) == 0.0
    _assert_pinned("snow_bucket_lie", records)


def test_c7_like_strang_equals_lie_bitwise_and_is_pinned():
    lie = _c7_like_trajectory("lie")
    strang = _c7_like_trajectory("strang")
    for key in _C7_KEYS:
        assert torch.equal(lie[key], strang[key]), key
    _assert_pinned("c7_like", strang)


def test_adaptive_lie_run_digest_is_pinned():
    record = _adaptive_lie_record()
    # The controller rejected or grew steps: the run is not the fixed 0.25 s grid.
    assert float(record["step_count"]) != _ADAPTIVE_END / 0.25
    _assert_pinned("adaptive_lie", record)


def test_strang_split_restart_continues_bitwise_and_is_pinned(tmp_path):
    uninterrupted, restarted = _split_restart_trajectories(tmp_path)
    for key in uninterrupted:
        assert torch.equal(uninterrupted[key], restarted[key]), key
    _assert_pinned("split_restart_strang", restarted)
