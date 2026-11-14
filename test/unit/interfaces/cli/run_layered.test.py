"""L1 tests for run_layered CLI module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest
import torch

from relic.interfaces.cli import run_layered


class _DummySimulator:
    """Minimal simulator stub for CLI execution tests."""

    def __init__(self, config):
        self.config = config
        self.initialized_state = None

    def initialize_state(self, state):
        self.initialized_state = state

    def run(self, start_time=0.0, end_time=None, dt=None):
        return {
            "start_time": start_time,
            "end_time": end_time,
            "dt": dt,
            "has_initial_state": self.initialized_state is not None,
        }


def test_setup_parser_smoke() -> None:
    parser = argparse.ArgumentParser()
    run_layered.setup_parser(parser)
    args = parser.parse_args(
        ["--config", "cfg.json", "--output", "out.json", "--start-time", "0"]
    )
    assert args.config == "cfg.json"
    assert args.start_time == 0.0
    assert args.output == "out.json"
    assert args.format == "json"


def test_execute_success_with_valid_paths(monkeypatch) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="run-layered-ok-"))
    try:
        root = temp_dir / "sandbox"
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("RELIC_API_DATA_ROOT", str(root))
        monkeypatch.setattr(run_layered, "RELICSimulator", _DummySimulator)

        config_path = root / "config.json"
        state_path = root / "state.json"
        output_path = root / "results.json"
        config_path.write_text(
            json.dumps({"simulation": {"time_step": 3600.0, "max_time": 7200.0}}),
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps({"soil_moisture": [0.3], "air_temperature": 293.15}),
            encoding="utf-8",
        )

        args = argparse.Namespace(
            config=str(config_path),
            initial_state=str(state_path),
            output=str(output_path),
            format="json",
            start_time=0.0,
            end_time=7200.0,
            dt=3600.0,
        )
        ok = run_layered.execute(args, {})
        assert ok is True
        assert output_path.exists()

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["status"] == "ok"
        assert payload["result_type"] == "dict"
        assert payload["results"]["has_initial_state"] is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class _TensorResultSimulator(_DummySimulator):
    """Simulator stub whose run returns multi-element tensor results."""

    def run(self, start_time=0.0, end_time=None, dt=None):
        return {
            "soil_moisture": torch.tensor([0.25, 0.5]),
            "air_temperature": torch.tensor(293.15, dtype=torch.float64),
            "layer_depth": np.array([[0.1, 0.3], [0.6, 1.0]]),
        }


def test_execute_writes_tensor_results_as_values(monkeypatch) -> None:
    """Array results are written as their values, never as a string repr."""
    temp_dir = Path(tempfile.mkdtemp(prefix="run-layered-tensor-"))
    try:
        root = temp_dir / "sandbox"
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("RELIC_API_DATA_ROOT", str(root))
        monkeypatch.setattr(run_layered, "RELICSimulator", _TensorResultSimulator)

        config_path = root / "config.json"
        output_path = root / "results.json"
        config_path.write_text(json.dumps({"time_step": 3600.0}), encoding="utf-8")

        args = argparse.Namespace(
            config=str(config_path),
            initial_state=None,
            output=str(output_path),
            format="json",
            start_time=0.0,
            end_time=3600.0,
            dt=3600.0,
        )
        assert run_layered.execute(args, {}) is True

        results = json.loads(output_path.read_text(encoding="utf-8"))["results"]
        assert results["soil_moisture"] == [0.25, 0.5]
        assert results["air_temperature"] == pytest.approx(293.15)
        assert results["layer_depth"] == [[0.1, 0.3], [0.6, 1.0]]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_execute_blocks_path_traversal(monkeypatch) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="run-layered-block-"))
    try:
        root = temp_dir / "sandbox"
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("RELIC_API_DATA_ROOT", str(root))
        monkeypatch.setattr(run_layered, "RELICSimulator", _DummySimulator)

        config_path = root / "config.json"
        config_path.write_text(json.dumps({"time_step": 3600.0}), encoding="utf-8")

        outside_output = root.parent / "escape.json"
        args = argparse.Namespace(
            config=str(config_path),
            initial_state=None,
            output=str(outside_output),
            format="json",
            start_time=0.0,
            end_time=3600.0,
            dt=3600.0,
        )
        ok = run_layered.execute(args, {})
        assert ok is False
        assert not outside_output.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class _OpaqueResult:
    """Result object without a JSON representation."""


class _OpaqueResultSimulator(_DummySimulator):
    def run(self, start_time=0.0, end_time=None, dt=None):
        return {"diagnostics": _OpaqueResult()}


def test_execute_rejects_results_without_json_representation(monkeypatch) -> None:
    """An object with no JSON form is an error, never written as its str()."""
    temp_dir = Path(tempfile.mkdtemp(prefix="run-layered-opaque-"))
    try:
        root = temp_dir / "sandbox"
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("RELIC_API_DATA_ROOT", str(root))
        monkeypatch.setattr(run_layered, "RELICSimulator", _OpaqueResultSimulator)

        config_path = root / "config.json"
        output_path = root / "results.json"
        config_path.write_text(json.dumps({"time_step": 3600.0}), encoding="utf-8")

        args = argparse.Namespace(
            config=str(config_path),
            initial_state=None,
            output=str(output_path),
            format="json",
            start_time=0.0,
            end_time=3600.0,
            dt=3600.0,
        )

        assert run_layered.execute(args, {}) is False
        assert not output_path.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
