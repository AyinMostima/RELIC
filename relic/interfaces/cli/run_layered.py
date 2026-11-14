"""
Executable CLI entry for layered RELIC simulation runs.

This module is intentionally lightweight and production-usable:
- Parses CLI arguments
- Validates file paths against traversal attacks
- Loads JSON/YAML configuration and optional initial state
- Runs RELICSimulator
- Persists structured outputs with deterministic exit codes
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from relic.core.simulator import RELICSimulator
from relic.interfaces.api.schemas import validate_safe_local_path
from relic.interfaces.cli.utils import print_error, print_success, save_output

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _load_mapping(path: Path) -> Dict[str, Any]:
    """Load a JSON/YAML file and return a mapping."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configuration files.")
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(payload).__name__}.")
    return payload


def _coerce_state_value(raw: Any) -> Any:
    """Convert scalar/list state payloads to tensors while preserving existing tensors."""
    if isinstance(raw, torch.Tensor):
        return raw
    if isinstance(raw, (int, float)):
        return torch.tensor(float(raw), dtype=torch.float32)
    if isinstance(raw, (list, tuple)):
        return torch.tensor(raw, dtype=torch.float32)
    return raw


def _coerce_initial_state(state_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize initial state payload for simulator initialization."""
    normalized: Dict[str, Any] = {}
    for key, value in state_payload.items():
        normalized[key] = _coerce_state_value(value)
    return normalized


def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for layered execution."""
    parser.add_argument(
        "--config",
        required=True,
        help="Path to simulation config file (.json/.yaml/.yml).",
    )
    parser.add_argument(
        "--initial-state",
        default=None,
        help="Optional path to initial state mapping (.json/.yaml/.yml).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path for run summary (json/yaml/text via --format).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "txt"],
        default="json",
        help="Output serialization format.",
    )
    parser.add_argument("--start-time", type=float, required=True, help="Simulation start time [s].")
    parser.add_argument("--end-time", type=float, default=None, help="Simulation end time [s].")
    parser.add_argument("--dt", type=float, default=None, help="Simulation time step [s].")


def run_layered_simulation(
    *,
    config_path: str,
    output_path: str,
    output_format: str = "json",
    initial_state_path: Optional[str] = None,
    start_time: float,
    end_time: Optional[float] = None,
    dt: Optional[float] = None,
) -> Dict[str, Any]:
    """Run layered simulation and persist a structured summary."""
    safe_config = Path(validate_safe_local_path(config_path, must_exist=True))
    safe_output = Path(validate_safe_local_path(output_path, must_exist=False))
    safe_state: Optional[Path] = None
    if initial_state_path:
        safe_state = Path(validate_safe_local_path(initial_state_path, must_exist=True))

    config_payload = _load_mapping(safe_config)
    simulator_config = config_payload.get("simulation", config_payload)
    if not isinstance(simulator_config, dict):
        raise ValueError("Simulation configuration must be a mapping.")

    simulator = RELICSimulator(config=simulator_config)

    state_initialized = False
    if safe_state is not None:
        initial_state_payload = _load_mapping(safe_state)
        simulator.initialize_state(_coerce_initial_state(initial_state_payload))
        state_initialized = True
    elif isinstance(config_payload.get("initial_state"), dict):
        simulator.initialize_state(_coerce_initial_state(config_payload["initial_state"]))
        state_initialized = True

    if not state_initialized:
        simulator.initialize_state({})

    results = simulator.run(start_time=start_time, end_time=end_time, dt=dt)
    output = {
        "status": "ok",
        "config_path": str(safe_config),
        "initial_state_path": str(safe_state) if safe_state is not None else None,
        "output_path": str(safe_output),
        "start_time": float(start_time),
        "end_time": float(end_time) if end_time is not None else None,
        "dt": float(dt) if dt is not None else None,
        "result_type": type(results).__name__,
        "results": results,
    }
    save_output(output, str(safe_output), output_format=output_format)
    return output


def execute(args: Any, _: Optional[Dict[str, Any]] = None) -> bool:
    """Execute layered run command."""
    try:
        run_layered_simulation(
            config_path=args.config,
            output_path=args.output,
            output_format=args.format,
            initial_state_path=args.initial_state,
            start_time=args.start_time,
            end_time=args.end_time,
            dt=args.dt,
        )
        print_success("Layered simulation completed.")
        return True
    except (
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        logger.exception("Layered simulation failed")
        print_error(f"Layered simulation failed: {exc}")
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Standalone CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="run_layered",
        description="Run layered RELIC simulations with validated inputs.",
    )
    setup_parser(parser)
    args = parser.parse_args(argv)
    return 0 if execute(args, {}) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
