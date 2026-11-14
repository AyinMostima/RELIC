"""CR-A-04 T8: restart in the middle of a consumer block with interval_mean exchanges.

Design 20261019_cr_a04_design_01, sections 2, 3 and T8: the running exchange mean
m_e is a StateManager variable (state and schema) of its owner exchange:<id>, so a
durable checkpoint taken while a consumer block is incomplete carries it, and the
restarted run continues bitwise. The checkpoint format stays version 3 and its
fields are unchanged (the means are state, not new runtime fields); the schema
entries are part of the scientific contract digest, so the restoring simulator
declares the same means whether its Functors were registered before or after its
state was initialized.

References: Gear and Wells (1984) BIT 24(4):484-502; RELIC exact-restart contract
(checkpoint owner, format version 3).
Mathematical expression: m_e <- (m_e tau + vbar h) / (tau + h), tau = t_k - t_last;
digests and comparisons are bitwise (torch.equal).
Parameter meanings: see splitting_multirate_exchanges.test.py (T7 fixtures reused).
Units: fixture variables dimensionless; t, h, tau [s].
Domain/constraints: CPU, float64 test default dtype; medium interval 4 fast steps,
slow interval 8 fast steps; the checkpoint is taken after 6 steps (2 of 4 medium
steps and 6 of 8 slow steps done).
Sign convention: none.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

import torch

from relic.core.simulator import checkpoint as checkpoint_module

_TEST_ROOT = Path(__file__).resolve().parents[1]


def _load_test_module(relative: str, alias: str):
    """Load an existing test module by path to reuse its fixtures (no copies)."""
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, _TEST_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_T7 = _load_test_module("core/splitting_multirate_exchanges.test.py", "_cr_a04_t7_multirate_exchanges")

_EXCHANGES = (_T7._MEDIUM_MEAN, _T7._SLOW_MEAN)
_MEANS = (_T7._mean_name("ramp_medium"), _T7._mean_name("ramp_slow"))
_BEFORE, _AFTER = 6, 10


def _payload(path: Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _restart(tmp_path: Path, *, register_first: bool):
    writer_log: List[tuple] = []
    writer = _T7._ramp_simulator(_EXCHANGES, log=writer_log)
    _T7._run(writer, _BEFORE)
    path = tmp_path / "multirate_mid_block.ckpt"
    writer.create_checkpoint(path)
    del writer_log[:]
    uninterrupted = _T7._run(writer, _AFTER, extra=_MEANS)

    resumed_log: List[tuple] = []
    resumed = _T7._ramp_simulator(_EXCHANGES, log=resumed_log, register_first=register_first)
    resumed.restore_checkpoint(path)
    restarted = _T7._run(resumed, _AFTER, extra=_MEANS)
    return path, writer_log, uninterrupted, resumed_log, restarted


def test_t8_mid_block_restart_with_interval_mean_is_bitwise(tmp_path):
    _, writer_log, uninterrupted, resumed_log, restarted = _restart(tmp_path, register_first=False)
    _T7._assert_rows_equal(uninterrupted, restarted)
    _T7._assert_logs_equal(writer_log, resumed_log)
    # The restarted consumers completed the blocks begun before the checkpoint.
    assert _T7._seen(resumed_log, "medium_reader", "ramp")[0] == (240.0, 240.0, 6.5)
    assert _T7._seen(resumed_log, "slow_reader", "ramp")[0] == (0.0, 480.0, 4.5)


def test_t8_restart_into_a_simulator_registered_before_its_state_is_bitwise(tmp_path):
    _, writer_log, uninterrupted, resumed_log, restarted = _restart(tmp_path, register_first=True)
    _T7._assert_rows_equal(uninterrupted, restarted)
    _T7._assert_logs_equal(writer_log, resumed_log)


def test_t8_checkpoint_carries_the_means_in_state_and_schema_and_stays_version_3(tmp_path):
    simulator = _T7._ramp_simulator(_EXCHANGES)
    _T7._run(simulator, _BEFORE)
    path = tmp_path / "with_exchanges.ckpt"
    simulator.create_checkpoint(path)
    payload = _payload(path)
    assert payload["format"] == checkpoint_module.CHECKPOINT_FORMAT
    assert payload["format_version"] == checkpoint_module.CHECKPOINT_FORMAT_VERSION == 3
    state_manager = checkpoint_module._decode_checkpoint_value(payload["state_manager"], path="state_manager")
    saved_state = state_manager["state"]
    # 2 of 4 medium steps: mean(5, 6); 6 of 8 slow steps: mean(1, ..., 6).
    assert float(saved_state[_MEANS[0]]) == 5.5 and float(saved_state[_MEANS[1]]) == 3.5
    assert all(saved_state[name].dtype == torch.float64 for name in _MEANS)
    topology = checkpoint_module._decode_checkpoint_value(payload["topology"], path="topology")
    schema = topology["scientific_contract"]["state_manager"]["schema"]
    assert set(_MEANS) <= set(schema)

    plain = _T7._ramp_simulator()
    _T7._run(plain, _BEFORE)
    plain_path = tmp_path / "without_exchanges.ckpt"
    plain.create_checkpoint(plain_path)
    plain_payload = _payload(plain_path)
    # Fields are additive: the same body and runtime fields; the means are state.
    assert set(payload) == set(plain_payload)
    assert set(payload["runtime"]) == set(plain_payload["runtime"])
    plain_state = checkpoint_module._decode_checkpoint_value(plain_payload["state_manager"], path="state_manager")
    assert set(saved_state) - set(plain_state["state"]) == set(_MEANS)
    runtime = {
        name: checkpoint_module._decode_checkpoint_value(value, path=f"runtime.{name}")
        for name, value in payload["runtime"].items()
    }
    assert runtime["_multirate_exchanges"] == tuple(dict(item) for item in _EXCHANGES)
