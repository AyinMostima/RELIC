"""Coordinated prepare/decision/apply registry transactions (E02).

The pure decision is verified in-process; the protocol itself is verified
with two real Gloo ranks (spawned processes) driving real RELICSimulator
registrations, with failures injected on one rank: prepare failure, apply
failure after a commit decision, an exiting rank, and an RNG identity
mismatch. Every rank's state, epoch, reduction result, device, logs, and
cleanup are read back from the ledger the workers return.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import time
from typing import Any

import pytest
import torch

from relic.computation.distributed.launcher import (
    CoordinationOutcome,
    coordinate_transaction,
    decide_coordination,
)
from test.fixtures.distributed_coordination_worker import run_coordination_worker


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_real_workers(scenario: str, *, expected_exit_codes: dict[int, int] | None = None) -> list[dict[str, Any]]:
    world_size = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_local_port()
    processes = [
        context.Process(target=run_coordination_worker, args=(rank, world_size, port, scenario, result_queue))
        for rank in range(world_size)
    ]
    records: list[dict[str, Any]] = []
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 120.0
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
        if any(process.is_alive() for process in processes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(5.0)
            pytest.fail("Real Gloo coordination worker exceeded the 120 second deadline")
        expected = {0: 0, 1: 0, **(expected_exit_codes or {})}
        assert {rank: process.exitcode for rank, process in enumerate(processes)} == expected
        reporting = [rank for rank, code in expected.items() if code == 0]
        records = [result_queue.get(timeout=5.0) for _ in reporting]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5.0)
            process.close()
        result_queue.close()
        result_queue.join_thread()
    return sorted(records, key=lambda record: record["rank"])


def _vote(rank, ok=True, identity=None, error=None):
    return {"rank": rank, "ok": ok, "identity": identity, "error": error}


class TestDecision:
    def test_commit_requires_every_rank_ok_and_identical_identity(self):
        identity = {"epoch": 1, "contract": "c", "binding": "b", "device": "cpu", "rng": 7}
        assert decide_coordination([_vote(0, identity=identity), _vote(1, identity=dict(identity))]) == ("commit", [])

    def test_abort_reasons_are_deterministic_and_name_the_rank(self):
        good = {"epoch": 1, "contract": "c", "binding": "b", "device": "cpu", "rng": 7}
        decision, reasons = decide_coordination([
            _vote(1, ok=False, error="RuntimeError: boom"),
            _vote(0, identity=good),
        ])
        assert decision == "abort"
        assert reasons == [
            "rank 1 prepare failed: RuntimeError: boom",
            "binding mismatch: rank 0='b', rank 1=<absent>",
            "contract mismatch: rank 0='c', rank 1=<absent>",
            "device mismatch: rank 0='cpu', rank 1=<absent>",
            "epoch mismatch: rank 0=1, rank 1=<absent>",
            "rng mismatch: rank 0=7, rank 1=<absent>",
        ]
        # An identity field present as None differs from an absent one.
        decision, reasons = decide_coordination([
            _vote(0, identity={"epoch": None}), _vote(1, identity={}),
        ])
        assert decision == "abort"
        assert reasons == ["epoch mismatch: rank 0=None, rank 1=<absent>"]
        decision, reasons = decide_coordination([
            _vote(0, identity=good), _vote(1, identity={**good, "rng": 8, "epoch": 2}),
        ])
        assert decision == "abort"
        assert reasons == ["epoch mismatch: rank 0=1, rank 1=2", "rng mismatch: rank 0=7, rank 1=8"]

    @pytest.mark.parametrize("missing", ["rank", "ok", "identity", "error"])
    def test_a_vote_missing_a_key_raises(self, missing):
        """Votes are built by coordinate_transaction; a missing key is an internal error."""
        vote = _vote(0, identity={"epoch": 1})
        del vote[missing]
        with pytest.raises(KeyError, match=missing):
            decide_coordination([vote])

    def test_incomplete_world_aborts(self):
        assert decide_coordination([_vote(0, identity={"epoch": 1})])[0] == "commit"
        decision, reasons = decide_coordination([_vote(1, identity={"epoch": 1})])
        assert decision == "abort" and reasons == ["incomplete votes: ranks [1]"]
        assert decide_coordination([])[0] == "abort"


class TestSingleProcessProtocol:
    def test_single_process_runs_every_phase_locally(self):
        applied = []
        outcome = coordinate_transaction("t", prepare=lambda: {"epoch": 1}, apply=lambda: applied.append(1))
        assert isinstance(outcome, CoordinationOutcome)
        assert outcome.status == "committed" and outcome.decision == "commit"
        assert outcome.world_size == 1 and outcome.rank == 0 and outcome.applied is True
        assert outcome.decisions == ["commit"] and applied == [1]
        assert outcome.as_dict()["schema_version"] == 1

    def test_prepare_failure_aborts_and_rolls_back_prepare(self):
        rolled = []
        outcome = coordinate_transaction(
            "t",
            prepare=lambda: (_ for _ in ()).throw(ValueError("bad prepare")),
            apply=lambda: pytest.fail("apply must not run"),
            rollback_prepare=lambda: rolled.append(1),
        )
        assert outcome.status == "aborted" and outcome.decision == "abort"
        assert outcome.reasons == ["rank 0 prepare failed: ValueError: bad prepare"]
        assert rolled == [1] and outcome.applied is False

    def test_apply_failure_without_any_applied_rank_is_apply_failed(self):
        outcome = coordinate_transaction(
            "t",
            prepare=lambda: {"epoch": 1},
            apply=lambda: (_ for _ in ()).throw(RuntimeError("bad apply")),
            compensate=lambda: pytest.fail("nothing to compensate"),
        )
        assert outcome.status == "apply_failed" and outcome.decision == "commit"
        assert outcome.apply_results == [{"rank": 0, "ok": False, "error": "RuntimeError: bad apply"}]
        assert outcome.compensated is None


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.slow
class TestRealTwoRankCoordination:
    def test_commit_reduces_identically_and_keeps_the_checkpoint_rejection(self):
        records = _run_real_workers("commit")
        assert [record["rank"] for record in records] == [0, 1]
        assert all(record["error"] is None for record in records), [r["traceback"] for r in records]
        assert all(record["coordination_active"] is True for record in records)
        assert all(record["device"] == "cpu" for record in records)
        assert all(record["rng"] == 7 for record in records)
        for record in records:
            register, unregister = record["outcomes"]
            assert register["operation"] == "register_functor" and register["status"] == "committed"
            assert register["decision"] == "commit" and register["decisions"] == ["commit", "commit"]
            assert [vote["ok"] for vote in register["votes"]] == [True, True]
            assert register["votes"][0]["identity"] == register["votes"][1]["identity"]
            assert register["votes"][0]["identity"]["rng"] == 7
            assert [item["ok"] for item in register["apply_results"]] == [True, True]
            assert unregister["operation"] == "unregister_functor" and unregister["status"] == "committed"
            uids = [vote["identity"]["local_provider_uid"] for vote in unregister["votes"]]
            assert all(uid is not None for uid in uids) and uids[0] != uids[1]
            assert record["raised"] is None
            assert record["epochs"] == [1, 2, 3]
            assert record["providers_after"] == []
            # Balance of the reduction: mean of (1, 2) on every rank, bitwise equal.
            assert record["step_value"] == pytest.approx(1.5)
            assert "single-process" in record["checkpoint_error"]
            assert record["owner_initialized_after_cleanup"] is False
            assert record["torch_initialized_after_cleanup"] is False
        assert records[0]["step_digest"] == records[1]["step_digest"]
        assert records[0]["outcomes"][0]["transaction_id"] != records[1]["outcomes"][0]["transaction_id"]

    def test_prepare_failure_on_one_rank_aborts_everywhere_without_mutation(self):
        records = _run_real_workers("prepare_failure")
        assert all(record["error"] is None for record in records), [r["traceback"] for r in records]
        for record in records:
            (outcome,) = record["outcomes"]
            assert outcome["status"] == "aborted" and outcome["decision"] == "abort"
            assert outcome["decisions"] == ["abort", "abort"]
            assert outcome["votes"][1]["ok"] is False
            assert "injected prepare failure" in outcome["votes"][1]["error"]
            assert outcome["reasons"][0] == "rank 1 prepare failed: RuntimeError: injected prepare failure"
            assert outcome["applied"] is False and outcome["apply_results"] == []
            assert "ended 'aborted'" in record["raised"]
            assert record["epochs"] == [1, 1]
            assert record["providers_after"] == []
        assert any("prepare failed on rank 1" in line for line in records[1]["logs"])

    def test_apply_failure_after_commit_is_compensated_and_the_divergence_is_caught(self):
        records = _run_real_workers("apply_failure")
        assert all(record["error"] is None for record in records), [r["traceback"] for r in records]
        first_0, second_0 = records[0]["outcomes"]
        first_1, second_1 = records[1]["outcomes"]
        for outcome in (first_0, first_1):
            assert outcome["decision"] == "commit" and outcome["status"] == "compensated"
            assert [item["ok"] for item in outcome["apply_results"]] == [True, False]
            assert "injected apply failure" in outcome["apply_results"][1]["error"]
            assert outcome["reasons"] == ["apply failed on ranks [1]"]
        assert first_0["applied"] is True and first_0["compensated"] is True
        assert first_1["applied"] is False and first_1["compensated"] is None
        assert first_0["compensation_results"] == [
            {"rank": 0, "applied": True, "ok": True, "error": None},
            {"rank": 1, "applied": False, "ok": None, "error": None},
        ]
        # Rank 0 registered then retired (two epochs); rank 1 never moved.
        assert records[0]["epochs"][:2] == [1, 3] and records[1]["epochs"][:2] == [1, 1]
        assert all("ended 'compensated'" in record["raised"] or "ended 'aborted'" in record["raised"] for record in records)
        # The second transaction sees the epoch divergence and aborts on both ranks.
        for outcome in (second_0, second_1):
            assert outcome["status"] == "aborted" and outcome["decisions"] == ["abort", "abort"]
            assert outcome["reasons"] == ["epoch mismatch: rank 0=3, rank 1=1"]
        assert records[0]["providers_after"] == [] and records[1]["providers_after"] == []
        assert any("apply failed on rank 1" in line for line in records[1]["logs"])

    def test_exiting_rank_is_observed_as_a_partition_with_no_local_mutation(self):
        records = _run_real_workers("rank_exit", expected_exit_codes={1: 3})
        assert [record["rank"] for record in records] == [0]
        record = records[0]
        assert record["error"] is None, record["traceback"]
        (outcome,) = record["outcomes"]
        assert outcome["status"] == "partition" and outcome["phase"] == "vote_gather"
        assert outcome["decision"] == "undecided" and outcome["applied"] is False
        assert outcome["error"] is not None
        assert "ended 'partition'" in record["raised"]
        assert record["epochs"] == [1, 1] and record["providers_after"] == []
        assert any("contact lost during vote_gather on rank 0" in line for line in record["logs"])
        assert record["owner_initialized_after_cleanup"] is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA device required; GPU part unverified")
    def test_device_identity_mismatch_between_ranks_aborts(self):
        records = _run_real_workers("device_mismatch")
        assert all(record["error"] is None for record in records), [r["traceback"] for r in records]
        assert [record["device"] for record in records] == ["cuda:0", "cpu"]
        for record in records:
            (outcome,) = record["outcomes"]
            assert outcome["status"] == "aborted"
            assert outcome["reasons"] == ["device mismatch: rank 0='cuda:0', rank 1='cpu'"]
            assert outcome["applied"] is False
            assert record["epochs"] == [1, 1] and record["providers_after"] == []

    def test_rng_identity_mismatch_aborts_before_any_apply(self):
        records = _run_real_workers("identity_mismatch")
        assert all(record["error"] is None for record in records), [r["traceback"] for r in records]
        assert [record["rng"] for record in records] == [7, 8]
        for record in records:
            (outcome,) = record["outcomes"]
            assert outcome["status"] == "aborted"
            assert outcome["reasons"] == ["rng mismatch: rank 0=7, rank 1=8"]
            assert outcome["applied"] is False
            assert record["epochs"] == [1, 1] and record["providers_after"] == []

    def test_device_preparation_mode_mismatch_aborts_before_any_apply(self):
        records = _run_real_workers("registration_mode_mismatch")
        assert all(record["error"] is None for record in records), [r["traceback"] for r in records]
        assert [record["prepare_device"] for record in records] == [True, False]
        for record in records:
            (outcome,) = record["outcomes"]
            assert outcome["status"] == "aborted"
            assert outcome["reasons"] == ["prepare_device mismatch: rank 0=True, rank 1=False"]
            assert outcome["applied"] is False
            assert record["epochs"] == [1, 1] and record["providers_after"] == []
