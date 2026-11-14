"""Main entry for Tapajos single-site BVOC tutorial case."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relic.core import device_policy
from relic.core.state import UpdateItem
from relic.core.simulator.main import RELICSimulator
from relic.parameters import get_global_manager
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor

from tutorial.case.functors.case_relic_wrappers import (
    CaseBucketSoilMoisture,
    CaseFarquharPhotosynthesis,
    CaseMEGANModel,
    CaseOzoneStress,
    CasePenmanMonteithET,
    CaseSurfaceRunoff,
    CaseThermalStress,
)
from tutorial.case.functors.irrigation_precip_adapter import PrecipThroughfallAdapter
from tutorial.case.functors.ml_surrogates import (
    CaseMEGANDiurnalActivitySurrogate,
    CaseIsopreneResidualSurrogate,
    CaseOzoneStressSurrogate,
    CaseSoilMoistureSurrogate,
    CaseStomatalSurrogate,
    CaseWaterStressSurrogate,
)
from tutorial.case.functors.stomatal_proxy import StomatalProxyFunctor
from tutorial.case.functors.stress_key_adapter import StressKeyAdapter
from tutorial.case.utils.diagnostics_writer import DiagnosticsWriter
from tutorial.case.utils.netcdf_loader import CaseNetCDFLoader, LoaderConfig
from tutorial.case.utils.optimization_loop import OptimizationLoopEngine


INSTANTANEOUS_STATE_KEYS: Tuple[str, ...] = (
    "evapotranspiration_flux",
    "drainage_flux",
    "precipitation_flux_effective",
    "surface_runoff_flux",
    "surface_runoff_flux_external",
    "infiltration_flux",
    "runoff_coefficient",
    "temperature_stress_factor",
    "ozone_stress_factor",
    "water_stress_factor",
    "temperature_stress",
    "ozone_stress",
    "water_stress",
    "gross_photosynthesis",
    "net_photosynthesis",
    "gross_primary_productivity",
    "electron_transport_rate",
    "isoprene_emission",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tapajos single-site BVOC tutorial case.")
    parser.add_argument(
        "--case-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Case root directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulate", "optimize"],
        default="simulate",
        help="Execution mode.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Optional day limit for simulate mode.",
    )
    parser.add_argument(
        "--use-mock-obs",
        action="store_true",
        help="Use generated mock observation file instead of real_obs.nc.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id override.",
    )
    parser.add_argument(
        "--opt-phases",
        type=str,
        default=None,
        help="Comma-separated optimization phases to run (for example: phase1,phase2).",
    )
    parser.add_argument(
        "--init-params",
        type=Path,
        default=None,
        help="Optional YAML file of parameter overrides for trainable initialization.",
    )
    parser.add_argument(
        "--init-from-run",
        type=str,
        default=None,
        help="Optional previous run-id used to initialize trainables from saved outputs.",
    )
    parser.add_argument(
        "--init-from-phase",
        type=str,
        default=None,
        help="Optional phase checkpoint name from --init-from-run, for example phase3.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous run's latest checkpoint. Pass the run-id of the run to resume.",
    )
    parser.add_argument(
        "--no-ml-surrogate",
        action="store_true",
        help="Disable tutorial-case ML surrogate registration and derived ML diagnostics.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def load_optional_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return load_yaml(path)


def _resolve_case_runtime_device(sim_cfg: Dict[str, object]) -> torch.device:
    simulation = sim_cfg.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("simulation.yaml must contain a simulation mapping.")
    requested_device = simulation.get("device", "auto")
    require_gpu = bool(simulation.get("require_gpu", False))
    return device_policy.resolve_device(
        explicit_device=requested_device,
        default="auto",
        require_gpu=require_gpu,
    )


def _move_surrogate_aux_tensors(functor: object, device: torch.device) -> None:
    for attr_name in ("_input_mean", "_input_scale", "_output_mean", "_output_scale"):
        attr_value = getattr(functor, attr_name, None)
        if isinstance(attr_value, torch.Tensor):
            setattr(functor, attr_name, attr_value.to(device))


@contextmanager
def _temporary_surrogate_training_device(functor: object, device: torch.device):
    original_device = torch.device(getattr(functor, "device", torch.device("cpu")))
    target_device = torch.device(device)
    if target_device == original_device or getattr(functor, "model", None) is None:
        yield original_device
        return

    if hasattr(functor, "to_device") and callable(getattr(functor, "to_device")):
        functor.to_device(target_device)
    else:
        if getattr(functor, "model", None) is not None:
            functor.model.to(target_device)
        setattr(functor, "device", target_device)
    _move_surrogate_aux_tensors(functor, target_device)

    try:
        yield target_device
    finally:
        if hasattr(functor, "to_device") and callable(getattr(functor, "to_device")):
            functor.to_device(original_device)
        else:
            if getattr(functor, "model", None) is not None:
                functor.model.to(original_device)
            setattr(functor, "device", original_device)
        _move_surrogate_aux_tensors(functor, original_device)


def _resolve_ml_training_device_for_functor(
    *,
    sim_cfg: Dict[str, object],
    functor_name: str,
    functor: object,
    sample_count: int,
    epochs: int,
) -> torch.device:
    current_device = torch.device(getattr(functor, "device", torch.device("cpu")))
    runtime_settings = _resolve_ml_runtime_settings(sim_cfg)
    acceleration_cfg = runtime_settings.get("training_acceleration", {})
    if not isinstance(acceleration_cfg, dict) or not bool(acceleration_cfg.get("enabled", False)):
        return current_device

    eligible_functors = acceleration_cfg.get("eligible_functors", [])
    if isinstance(eligible_functors, (list, tuple, set)):
        eligible_names = {str(value) for value in eligible_functors}
        if eligible_names and functor_name not in eligible_names:
            return current_device

    min_workload = int(acceleration_cfg.get("min_workload", 20000) or 0)
    workload = max(1, int(sample_count)) * max(1, int(epochs))
    if workload < min_workload:
        return current_device

    requested_device = acceleration_cfg.get("device", "auto")
    require_gpu = bool(acceleration_cfg.get("require_gpu", False))
    resolved_device = device_policy.resolve_device(
        explicit_device=requested_device,
        default=str(current_device),
        require_gpu=require_gpu,
    )
    if resolved_device.type != "cuda":
        return current_device
    return resolved_device


def read_case_bounds(case_cfg: Dict[str, object], key: str) -> Tuple[Optional[float], Optional[float]]:
    parameters = case_cfg.get("parameters")
    if isinstance(parameters, dict):
        entry = parameters.get(key)
    else:
        entry = case_cfg.get(key)
    if not isinstance(entry, dict):
        return (None, None)
    bounds = entry.get("bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return (None, None)
    lower = None if bounds[0] is None else float(bounds[0])
    upper = None if bounds[1] is None else float(bounds[1])
    return (lower, upper)


def read_case_value(case_cfg: Dict[str, object], key: str) -> Optional[float]:
    parameters = case_cfg.get("parameters")
    entry = None
    if isinstance(parameters, dict):
        entry = parameters.get(key)
    if entry is None:
        entry = case_cfg.get(key)
    if not isinstance(entry, dict):
        return None
    value = entry.get("value")
    if value is None:
        return None
    return float(value)


def apply_case_trainable_initial_values(
    trainable: Dict[str, torch.nn.Parameter],
    case_cfg: Dict[str, object],
) -> None:
    with torch.no_grad():
        for key, param in trainable.items():
            case_value = read_case_value(case_cfg, key)
            if case_value is None:
                continue
            param.copy_(torch.as_tensor(case_value, dtype=param.dtype, device=param.device))


def ensure_processed_data(case_root: Path, use_mock_obs: bool) -> Path:
    data_dir = case_root / "data"
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    forcing_path = processed / "site_forcing.nc"
    obs_path = processed / "real_obs.nc"
    subprocess.run(
        [sys.executable, str(data_dir / "prepare_netcdf.py"), "--case-root", str(case_root)],
        check=True,
    )

    if use_mock_obs:
        mock_obs_path = processed / "mock_obs.nc"
        if not mock_obs_path.exists():
            subprocess.run(
                [sys.executable, str(data_dir / "generate_mock_data.py"), "--case-root", str(case_root)],
                check=True,
            )
        return mock_obs_path

    return obs_path


def register_case_parameters(case_parameter_path: Path) -> None:
    manager = get_global_manager()
    coupling_defaults = (
        PROJECT_ROOT / "relic" / "parameters" / "database" / "relic" / "coupling" / "defaults.yaml"
    )
    coupling_scale_vars = (
        PROJECT_ROOT / "relic" / "parameters" / "database" / "relic" / "coupling" / "scale_variables.yaml"
    )
    if coupling_defaults.exists():
        manager.database.add_source(coupling_defaults)
    if coupling_scale_vars.exists():
        manager.database.add_source(coupling_scale_vars)
    manager.database.add_source(case_parameter_path)
    # Force a database refresh so newly added sources override already-cached defaults.
    manager.load_from_database()


def _resolve_case_relative_path(case_root: Path, raw_path: object) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate
    return (case_root / candidate).resolve()


def _resolve_ml_runtime_settings(sim_cfg: Dict[str, object]) -> Dict[str, object]:
    simulation = sim_cfg.get("simulation")
    if not isinstance(simulation, dict):
        return {}
    ml_settings = simulation.get("ml_surrogate", {})
    if not isinstance(ml_settings, dict):
        return {}
    return dict(ml_settings)


def _resolve_monitoring_settings(sim_cfg: Dict[str, object]) -> Dict[str, object]:
    simulation = sim_cfg.get("simulation")
    if not isinstance(simulation, dict):
        return {}
    monitoring = simulation.get("monitoring", {})
    if not isinstance(monitoring, dict):
        monitoring = {}
    resolved = dict(monitoring)
    resolved.setdefault("enabled", True)
    resolved.setdefault("log_interval_steps", 24)
    resolved.setdefault("checkpoint_interval_steps", 24)
    return resolved


def _create_progress_bar(
    *,
    enabled: bool,
    total: int,
    desc: str,
    position: int = 0,
    leave: bool = False,
):
    if not enabled or tqdm is None:
        return None
    return tqdm(
        total=max(0, int(total)),
        desc=str(desc),
        position=int(position),
        leave=leave,
        dynamic_ncols=True,
    )


def _close_progress_bar(progress_bar: object) -> None:
    if progress_bar is not None and hasattr(progress_bar, "close"):
        progress_bar.close()


def _update_progress_bar(
    progress_bar: object,
    *,
    advance: int = 1,
    postfix: Optional[Dict[str, object]] = None,
) -> None:
    if progress_bar is None:
        return
    if hasattr(progress_bar, "update"):
        progress_bar.update(int(advance))
    if postfix and hasattr(progress_bar, "set_postfix"):
        progress_bar.set_postfix(postfix)


def _get_ml_component_config(ml_cfg: Optional[Dict[str, object]], component_name: str) -> Dict[str, object]:
    if not isinstance(ml_cfg, dict):
        return {}
    payload = ml_cfg.get("ml_surrogates", {})
    if not isinstance(payload, dict):
        return {}
    component = payload.get(component_name, {})
    if not isinstance(component, dict):
        return {}
    return copy.deepcopy(component)


def _clone_state_mapping(state: Dict[str, object]) -> Dict[str, object]:
    cloned: Dict[str, object] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _split_state_flux_pairs(
    states: List[Dict[str, object]],
    fluxes: List[Dict[str, object]],
    validation_fraction: float,
) -> Tuple[
    Tuple[List[Dict[str, object]], List[Dict[str, object]]],
    Optional[Tuple[List[Dict[str, object]], List[Dict[str, object]]]],
]:
    if len(states) != len(fluxes):
        raise ValueError("State and flux sequences must have identical lengths.")
    if len(states) < 4 or validation_fraction <= 0.0:
        return (states, fluxes), None
    split_index = max(1, int(round(len(states) * (1.0 - validation_fraction))))
    split_index = min(split_index, len(states) - 1)
    training = (states[:split_index], fluxes[:split_index])
    validation = (states[split_index:], fluxes[split_index:])
    if len(validation[0]) == 0:
        return training, None
    return training, validation


def _suffix_outputs(
    outputs: Dict[str, object],
    keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    suffixed: Dict[str, torch.Tensor] = {}
    for key in keys:
        value = _unwrap_output_value(outputs.get(key))
        if isinstance(value, torch.Tensor):
            suffixed[f"{key}_ML"] = value
    return suffixed


def _unwrap_output_value(value: object) -> object:
    """Return the payload of a canonical state update for diagnostics."""
    return value.value if isinstance(value, UpdateItem) else value


def _merge_runtime_diagnostics_for_logging(
    *,
    simulator: RELICSimulator,
    state_for_log: Dict[str, object],
    step_outputs: Optional[Dict[str, object]] = None,
) -> List[str]:
    """Expose runtime diagnostics to the case artifact writer without state mutation."""
    diagnostic_outputs: Dict[str, object] = {}
    if isinstance(step_outputs, dict):
        diagnostic_outputs.update(
            {
                str(key): _unwrap_output_value(value)
                for key, value in step_outputs.items()
                if str(key).startswith("diagnostics.")
            }
        )
    else:
        current_diagnostics = getattr(
            getattr(simulator, "state_manager", None),
            "_current_diagnostics",
            {},
        )
        if isinstance(current_diagnostics, dict):
            diagnostic_outputs.update(
                {
                    f"diagnostics.{key}": value
                    for key, value in current_diagnostics.items()
                }
            )

    state_for_log.update(diagnostic_outputs)
    flat_diagnostic_aliases = {
        "runoff_coefficient": "surface_runoff.runoff_coefficient",
    }
    for target_key, diagnostic_suffix in flat_diagnostic_aliases.items():
        for diagnostic_key, value in diagnostic_outputs.items():
            if diagnostic_key.endswith(diagnostic_suffix):
                state_for_log[target_key] = value
                break
    return sorted(diagnostic_outputs)


def _get_registered_functor(
    simulator: RELICSimulator,
    name: str,
) -> Optional[torch.nn.Module]:
    functors = simulator.functors
    if name in functors:
        return functors[name]
    aux_functors = getattr(simulator, "_case_aux_functors", None)
    if isinstance(aux_functors, dict) and name in aux_functors:
        return aux_functors[name]
    return None


def _register_case_aux_functor(
    simulator: RELICSimulator,
    *,
    name: str,
    functor: torch.nn.Module,
) -> None:
    aux_functors = getattr(simulator, "_case_aux_functors", None)
    if not isinstance(aux_functors, dict):
        aux_functors = {}
        setattr(simulator, "_case_aux_functors", aux_functors)
    target_device = simulator.device
    if hasattr(simulator, "get_effective_device") and callable(getattr(simulator, "get_effective_device")):
        target_device = torch.device(simulator.get_effective_device(name))
    if hasattr(functor, "to_device") and callable(getattr(functor, "to_device")):
        functor.to_device(target_device)
    else:
        if hasattr(functor, "to") and callable(getattr(functor, "to")):
            functor.to(target_device)
        setattr(functor, "device", target_device)
    _move_surrogate_aux_tensors(functor, target_device)
    aux_functors[name] = functor


def _is_valid_observation_tensor(value: Optional[torch.Tensor]) -> bool:
    return isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())


def _monitor_scalar(state: Dict[str, object], key: str) -> str:
    value = state.get(key)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return "nan"
        return f"{float(value.detach().reshape(-1)[0].cpu().item()):.6g}"
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return "nan"


def _collect_ml_surrogate_functors(simulator: RELICSimulator) -> Dict[str, object]:
    ml_functor_types = (
        CaseSoilMoistureSurrogate,
        CaseWaterStressSurrogate,
        CaseStomatalSurrogate,
        CaseOzoneStressSurrogate,
        CaseMEGANDiurnalActivitySurrogate,
        CaseIsopreneResidualSurrogate,
    )
    collected: Dict[str, object] = {}
    for name, functor in simulator.functors.items():
        if isinstance(functor, ml_functor_types):
            collected[name] = functor
    aux_functors = getattr(simulator, "_case_aux_functors", None)
    if isinstance(aux_functors, dict):
        for name, functor in aux_functors.items():
            if isinstance(functor, ml_functor_types):
                collected[name] = functor
    return collected


def _simulator_has_ml_path(simulator: RELICSimulator) -> bool:
    return bool(_collect_ml_surrogate_functors(simulator))


def _persist_ml_surrogate_checkpoints(
    *,
    simulator: RELICSimulator,
    diagnostics: DiagnosticsWriter,
    log_fn: Callable[[str], None],
) -> Dict[str, str]:
    checkpoint_sources: Dict[str, Path] = {}
    for functor_name, functor in _collect_ml_surrogate_functors(simulator).items():
        model_path = getattr(functor, "model_path", None)
        if model_path is None:
            continue
        if hasattr(functor, "save_model"):
            functor.save_model(Path(model_path))
        checkpoint_sources[functor_name] = Path(model_path)
    manifest = diagnostics.archive_ml_surrogate_checkpoints(checkpoint_sources)
    if manifest:
        log_fn(
            "[INFO] archived ML surrogate checkpoints: "
            + ", ".join(f"{name}={path}" for name, path in sorted(manifest.items()))
        )
    return manifest


@dataclass
class ResumeContext:
    run_id: str
    run_dir: Path
    checkpoint_dir: Path
    checkpoint_path: Path
    checkpoint_payload: Dict[str, object]
    resume_phase: str
    resume_epoch: int
    resume_epoch_loss: Optional[float]
    resume_optimizer_state: Optional[Dict[str, object]]
    completed_phases: List[str]
    de_warmstart_payloads: Dict[str, Dict[str, object]]


def _load_de_warmstart_payloads(checkpoint_dir: Path) -> Dict[str, Dict[str, object]]:
    payloads: Dict[str, Dict[str, object]] = {}
    if not checkpoint_dir.exists():
        return payloads
    for checkpoint_path in sorted(checkpoint_dir.glob("de_*_warmstart.pt")):
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, dict):
            continue
        phase_name = payload.get("phase_name")
        if not isinstance(phase_name, str) or not phase_name:
            continue
        payloads[phase_name] = payload
    return payloads


def _resolve_resume_context(
    *,
    output_root: Path,
    run_id: str,
) -> ResumeContext:
    run_dir = _resolve_saved_run_dir(output_root, run_id)
    if run_dir is None:
        raise FileNotFoundError(f"resume run directory not found for run-id: {run_id}")

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_path = OptimizationLoopEngine.find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"no optimization checkpoint found under {checkpoint_dir}")

    checkpoint_payload = OptimizationLoopEngine.load_checkpoint(checkpoint_path)
    phase_name = checkpoint_payload.get("phase_name")
    epoch_value = checkpoint_payload.get("epoch")
    if not isinstance(phase_name, str) or not isinstance(epoch_value, int):
        raise ValueError(f"invalid resume checkpoint payload: {checkpoint_path}")

    completed_phases_payload = checkpoint_payload.get("completed_phases", [])
    if not isinstance(completed_phases_payload, list):
        completed_phases_payload = []

    checkpoint_epoch_loss = checkpoint_payload.get("epoch_loss")
    resume_epoch_loss = None
    if checkpoint_epoch_loss is not None:
        resume_epoch_loss = float(checkpoint_epoch_loss)

    resume_optimizer_state = checkpoint_payload.get("optimizer_state_dict")
    if resume_optimizer_state is not None and not isinstance(resume_optimizer_state, dict):
        raise ValueError(f"invalid optimizer_state_dict in {checkpoint_path}")

    return ResumeContext(
        run_id=run_id,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=checkpoint_payload,
        resume_phase=phase_name,
        resume_epoch=int(epoch_value) + 1,
        resume_epoch_loss=resume_epoch_loss,
        resume_optimizer_state=resume_optimizer_state,
        completed_phases=[str(value) for value in completed_phases_payload],
        de_warmstart_payloads=_load_de_warmstart_payloads(checkpoint_dir),
    )


def _restore_surrogate_checkpoint(
    *,
    functor: object,
    checkpoint_path: Path,
) -> None:
    model_path = getattr(functor, "model_path", None)
    if model_path is None:
        raise RuntimeError(f"ML surrogate {getattr(functor, 'name', 'unknown')} has no model_path")
    if not hasattr(functor, "_load_model") or not callable(getattr(functor, "_load_model")):
        raise RuntimeError(f"ML surrogate {getattr(functor, 'name', 'unknown')} cannot load checkpoints")
    source_path = Path(checkpoint_path)
    if not source_path.exists():
        raise FileNotFoundError(f"ML surrogate checkpoint not found: {source_path}")

    original_model_path = Path(model_path)
    setattr(functor, "model_path", source_path)
    try:
        functor._load_model()
    finally:
        setattr(functor, "model_path", original_model_path)
    if getattr(functor, "model", None) is None or not bool(getattr(functor, "is_trained", False)):
        raise RuntimeError(
            f"Failed to restore trained state for ML surrogate {getattr(functor, 'name', 'unknown')}"
        )
    functor.model.to(getattr(functor, "device", torch.device("cpu")))
    functor.model.eval()


def _restore_ml_surrogate_checkpoints_from_run(
    *,
    simulator: RELICSimulator,
    run_dir: Path,
    log_fn: Callable[[str], None],
) -> Dict[str, str]:
    ml_functors = _collect_ml_surrogate_functors(simulator)
    if not ml_functors:
        return {}

    manifest_path = run_dir / "checkpoints" / "ml_surrogates" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"strict resume requires ML surrogate manifest at {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid ML surrogate manifest: {manifest_path}")

    restored_manifest: Dict[str, str] = {}
    for functor_name, functor in ml_functors.items():
        source_value = payload.get(functor_name)
        if not isinstance(source_value, str) or not source_value:
            raise KeyError(f"missing ML surrogate checkpoint entry for {functor_name} in {manifest_path}")
        source_path = Path(source_value)
        if not source_path.is_absolute():
            source_path = (manifest_path.parent / source_path).resolve()
        _restore_surrogate_checkpoint(functor=functor, checkpoint_path=source_path)
        restored_manifest[functor_name] = str(source_path.resolve())

    _set_ml_surrogate_mode(simulator, training=False)
    log_fn(
        "[RESUME] restored ML surrogate checkpoints: "
        + ", ".join(f"{name}={path}" for name, path in sorted(restored_manifest.items()))
    )
    return restored_manifest


def _apply_resume_de_warmstart(
    *,
    phase_name: str,
    payload: Dict[str, object],
    trainable: Dict[str, torch.nn.Parameter],
    checkpoint_dir: Path,
    log_fn: Callable[[str], None],
) -> Dict[str, object]:
    optimized_parameters = payload.get("optimized_parameters", {})
    if not isinstance(optimized_parameters, dict):
        raise ValueError(f"invalid DE warm-start payload for {phase_name}: optimized_parameters missing")
    with torch.no_grad():
        for key, value in optimized_parameters.items():
            if key not in trainable:
                continue
            param = trainable[key]
            param.copy_(torch.as_tensor(float(value), dtype=param.dtype, device=param.device))

    local_checkpoint_path = checkpoint_dir / f"de_{phase_name}_warmstart.pt"
    checkpoint_payload = dict(payload)
    torch.save(checkpoint_payload, local_checkpoint_path)
    log_fn(f"[RESUME] restored DE warm-start for {phase_name} from saved checkpoint.")

    return {
        "optimized_parameters": {
            str(key): float(value)
            for key, value in optimized_parameters.items()
        },
        "objective_value": float(payload.get("objective_value", float("nan"))),
        "initial_objective_value": float(payload.get("initial_objective_value", float("nan"))),
        "iteration_count": int(payload.get("iteration_count", 0)),
        "success": bool(payload.get("success", True)),
        "termination_reason": str(payload.get("termination_reason", "restored_from_checkpoint")),
        "convergence_history": [
            float(value) for value in payload.get("convergence_history", [])
        ],
        "checkpoint_path": str(local_checkpoint_path.resolve()),
    }


def _train_surrogate_with_progress(
    *,
    functor: object,
    functor_name: str,
    train_pairs: Tuple[List[Dict[str, object]], List[Dict[str, object]]],
    validation_pairs: Optional[Tuple[List[Dict[str, object]], List[Dict[str, object]]]],
    epochs_value: Optional[int],
    show_progress: bool,
) -> Dict[str, object]:
    training_cfg = getattr(functor, "training_config", {})
    target_epochs = int(
        epochs_value
        if epochs_value not in (None, 0)
        else (training_cfg.get("epochs", 100) if isinstance(training_cfg, dict) else 100)
    )
    progress_bar = _create_progress_bar(
        enabled=show_progress,
        total=target_epochs,
        desc=f"train {functor_name}",
        position=0,
        leave=False,
    )
    progress_state = {"last_epoch": 0}
    original_log = getattr(functor, "_log_training_progress", None)

    def _progress_hook(
        *,
        epoch: int,
        epochs: int,
        avg_train_loss: float,
        avg_val_loss: Optional[float],
        verbose: bool,
    ) -> None:
        del verbose
        current_epoch = int(epoch) + 1
        if current_epoch > progress_state["last_epoch"]:
            _update_progress_bar(
                progress_bar,
                advance=current_epoch - progress_state["last_epoch"],
                postfix={
                    "train": f"{float(avg_train_loss):.4g}",
                    "val": "na" if avg_val_loss is None else f"{float(avg_val_loss):.4g}",
                },
            )
            progress_state["last_epoch"] = current_epoch

    if original_log is not None:
        setattr(functor, "_log_training_progress", _progress_hook)
    try:
        result = functor.train(
            training_data=train_pairs,
            validation_data=validation_pairs,
            epochs=target_epochs,
            verbose=False,
            validate_units=False,
        )
    finally:
        if original_log is not None:
            setattr(functor, "_log_training_progress", original_log)
    final_epoch = int(result.get("epochs_trained") or 0)
    if final_epoch > progress_state["last_epoch"]:
        _update_progress_bar(
            progress_bar,
            advance=final_epoch - progress_state["last_epoch"],
        )
    _close_progress_bar(progress_bar)
    return result


def _observation_quality_scale(loader: CaseNetCDFLoader, obs_key: str) -> float:
    quality_label = loader.get_obs_quality_label(obs_key)
    if quality_label == "real_observation":
        return 1.0
    if quality_label == "real_observation_sparse":
        return 0.7
    if quality_label == "proxy_external_reanalysis":
        return 0.35
    if quality_label == "proxy_climatology_template":
        return 0.25
    return 0.5


def _convert_prediction_to_observation_units(
    loader: CaseNetCDFLoader,
    obs_key: str,
    predicted: torch.Tensor,
) -> torch.Tensor:
    if obs_key == "isoprene_emission":
        return loader.convert_model_isoprene_to_observation_units(predicted)
    return predicted


def _encode_bounded_factor_target(
    target: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Encode a bounded factor as the unconstrained target of a logistic link."""
    if upper <= lower:
        raise ValueError("Bounded factor upper limit must exceed lower limit.")
    # References:
    # - Standard logistic link for bounded regression.
    # Mathematical expression:
    # - z = log((f - f_min) / (f_max - f))
    # Parameter meanings:
    # - f: bounded multiplicative correction factor [-]
    # - z: unconstrained network target [-]
    lower_tensor = torch.as_tensor(lower, dtype=target.dtype, device=target.device)
    upper_tensor = torch.as_tensor(upper, dtype=target.dtype, device=target.device)
    fraction = (target - lower_tensor) / (upper_tensor - lower_tensor)
    epsilon = torch.finfo(target.dtype).eps
    fraction = torch.clamp(fraction, min=epsilon, max=1.0 - epsilon)
    return torch.log(fraction) - torch.log1p(-fraction)


def _build_scalar_like_tensor(
    reference_tensor: Optional[torch.Tensor],
    value: float,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if isinstance(reference_tensor, torch.Tensor):
        return torch.full_like(reference_tensor, float(value))
    return torch.tensor(
        [[float(value)]],
        dtype=dtype or torch.float32,
        device=device or torch.device("cpu"),
    )


def _augment_state_with_diurnal_features(
    state: Dict[str, object],
    *,
    time_seconds: float,
    reference_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    hour_fraction = (float(time_seconds) / 3600.0) % 24.0
    angle = 2.0 * math.pi * hour_fraction / 24.0
    enriched_state = dict(state)
    enriched_state["hour_sin"] = _build_scalar_like_tensor(reference_tensor, math.sin(angle))
    enriched_state["hour_cos"] = _build_scalar_like_tensor(reference_tensor, math.cos(angle))
    return enriched_state


def _set_ml_surrogate_mode(simulator: RELICSimulator, *, training: bool) -> None:
    surrogate_types = (
        CaseSoilMoistureSurrogate,
        CaseWaterStressSurrogate,
        CaseStomatalSurrogate,
        CaseOzoneStressSurrogate,
        CaseMEGANDiurnalActivitySurrogate,
        CaseIsopreneResidualSurrogate,
    )
    for functor_name in (
        "ml_soil_moisture",
        "ml_water_stress",
        "ml_stomatal",
        "ml_ozone_stress",
        "ml_diurnal_activity",
        "ml_isoprene_residual",
    ):
        functor = _get_registered_functor(simulator, functor_name)
        if not isinstance(functor, surrogate_types):
            continue
        if functor.model is None:
            continue
        if training:
            functor.model.train()
        else:
            functor.model.eval()


def _reset_ml_surrogate_model(functor: object) -> None:
    if not hasattr(functor, "_create_model"):
        return
    if not callable(getattr(functor, "_create_model")):
        return
    functor._create_model()
    if getattr(functor, "model", None) is None:
        raise RuntimeError(f"Failed to rebuild ML surrogate model for {getattr(functor, 'name', 'unknown')}")
    functor.model.to(functor.device)
    functor.is_trained = False
    functor.best_model_state = None
    functor.epochs_trained = 0
    functor.training_history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }


def _collect_ml_training_samples(
    *,
    sim_cfg: Dict[str, object],
    ml_cfg: Optional[Dict[str, object]],
    loader: CaseNetCDFLoader,
    steps: int,
    dt_seconds: float,
    log_fn: Callable[[str], None],
    show_progress: bool = False,
) -> Dict[str, object]:
    physics_only = build_simulator(sim_cfg, ml_cfg=ml_cfg, ml_enabled_override=False)
    physics_only.initialize_state(loader.make_initial_state())

    soil_states: List[Dict[str, object]] = []
    soil_targets: List[Dict[str, object]] = []
    water_stress_states: List[Dict[str, object]] = []
    water_stress_targets: List[Dict[str, object]] = []
    stomatal_states: List[Dict[str, object]] = []
    stomatal_targets: List[Dict[str, object]] = []
    ozone_states: List[Dict[str, object]] = []
    ozone_targets: List[Dict[str, object]] = []
    diurnal_activity_states: List[Dict[str, object]] = []
    diurnal_activity_targets: List[Dict[str, object]] = []
    isoprene_residual_states: List[Dict[str, object]] = []
    isoprene_residual_targets: List[Dict[str, object]] = []
    fine_tune_samples: List[Dict[str, object]] = []
    manager = get_global_manager()
    diurnal_component_cfg = _get_ml_component_config(ml_cfg, "diurnal_activity")
    diurnal_training_cfg = diurnal_component_cfg.get("training", {})
    if not isinstance(diurnal_training_cfg, dict):
        diurnal_training_cfg = {}
    diurnal_par_threshold = float(diurnal_training_cfg.get("daytime_par_threshold", 10.0))
    diurnal_factor_min = float(
        manager.get("tutorial.case.ml_surrogate.diurnal_activity.output_min", required=True)
    )
    diurnal_factor_max = float(
        manager.get("tutorial.case.ml_surrogate.diurnal_activity.output_max", required=True)
    )
    residual_factor_min = float(
        manager.get("tutorial.case.ml_surrogate.isoprene_residual.output_min", required=True)
    )
    residual_factor_max = float(
        manager.get("tutorial.case.ml_surrogate.isoprene_residual.output_max", required=True)
    )

    log_fn("[INFO] collecting physics-only samples for ML surrogate training")
    sampling_bar = _create_progress_bar(
        enabled=show_progress,
        total=steps,
        desc="physics sampling",
        position=0,
        leave=False,
    )
    try:
        for step in range(steps):
            reset_instantaneous_states(physics_only.state_manager)
            forcing = loader.get_forcing_step(step)
            physics_only.state_manager.set_state(forcing, validate=False)
            physics_only.step(dt_seconds)
            current_state = physics_only.state_manager.get_state(clone=True)
            _update_progress_bar(
                sampling_bar,
                advance=1,
                postfix={
                    "step": f"{step + 1}/{steps}",
                    "soil": _monitor_scalar(current_state, "soil_moisture"),
                    "iso": _monitor_scalar(current_state, "isoprene_emission"),
                },
            )
            if "soil_moisture" not in current_state:
                raise KeyError("Physics-only training pass missing soil_moisture.")
            if "water_stress_factor" not in current_state:
                raise KeyError("Physics-only training pass missing water_stress_factor.")
            if "stomatal_conductance" not in current_state:
                raise KeyError("Physics-only training pass missing stomatal_conductance.")
            if "ozone_stress_factor" not in current_state:
                raise KeyError("Physics-only training pass missing ozone_stress_factor.")

            soil_obs = loader.get_obs_step(step, "soil_moisture")
            net_photo_obs = loader.get_obs_step(step, "net_photosynthesis")
            isoprene_obs = loader.get_obs_step(step, "isoprene_emission")

            soil_target_value = current_state["soil_moisture"].detach().clone()
            if _is_valid_observation_tensor(soil_obs):
                soil_target_value = soil_obs.detach().clone().reshape_as(soil_target_value)
            soil_state = _clone_state_mapping(current_state)
            soil_states.append(soil_state)
            soil_targets.append({"soil_moisture_ML": soil_target_value})

            water_stress_state = _clone_state_mapping(current_state)
            water_stress_state["soil_moisture_proxy"] = soil_target_value.detach().clone()
            water_stress_state["soil_moisture_ML"] = soil_target_value.detach().clone()
            water_stress_states.append(water_stress_state)
            water_stress_targets.append(
                {
                    "water_stress_factor_ML": current_state["water_stress_factor"].detach().clone(),
                }
            )

            stomatal_state = _clone_state_mapping(current_state)
            stomatal_state["soil_moisture_proxy"] = soil_target_value.detach().clone()
            stomatal_state["soil_moisture_ML"] = soil_target_value.detach().clone()
            stomatal_state["net_photosynthesis"] = current_state["net_photosynthesis"].detach().clone()
            stomatal_states.append(stomatal_state)
            stomatal_targets.append(
                {
                    "stomatal_conductance_ML": current_state["stomatal_conductance"].detach().clone()
                }
            )

            ozone_state = _clone_state_mapping(current_state)
            ozone_state["stomatal_conductance_proxy"] = current_state["stomatal_conductance"].detach().clone()
            ozone_state["stomatal_conductance_ML"] = current_state["stomatal_conductance"].detach().clone()
            ozone_states.append(ozone_state)
            ozone_targets.append(
                {
                    "ozone_stress_factor_ML": current_state["ozone_stress_factor"].detach().clone(),
                }
            )

            residual_reference = current_state["isoprene_emission"].detach().clone()
            diurnal_state = _clone_state_mapping(current_state)
            diurnal_state = _augment_state_with_diurnal_features(
                diurnal_state,
                time_seconds=float(physics_only.current_time),
                reference_tensor=residual_reference,
            )
            physics_isoprene_obs_units = _convert_prediction_to_observation_units(
                loader,
                "isoprene_emission",
                residual_reference,
            )
            eps = torch.as_tensor(
                1.0e-6,
                dtype=physics_isoprene_obs_units.dtype,
                device=physics_isoprene_obs_units.device,
            )
            ppfd_value = current_state.get("ppfd")
            is_daytime_sample = False
            if isinstance(ppfd_value, torch.Tensor):
                is_daytime_sample = bool(torch.any(ppfd_value > diurnal_par_threshold))

            diurnal_target = torch.ones_like(residual_reference)
            if is_daytime_sample and _is_valid_observation_tensor(isoprene_obs):
                diurnal_target = isoprene_obs.detach().clone().reshape_as(
                    physics_isoprene_obs_units
                ) / torch.clamp(physics_isoprene_obs_units, min=eps)
            diurnal_target = torch.clamp(
                diurnal_target,
                min=torch.as_tensor(
                    diurnal_factor_min,
                    dtype=diurnal_target.dtype,
                    device=diurnal_target.device,
                ),
                max=torch.as_tensor(
                    diurnal_factor_max,
                    dtype=diurnal_target.dtype,
                    device=diurnal_target.device,
                ),
            )
            diurnal_activity_states.append(diurnal_state)
            encoded_diurnal_target = _encode_bounded_factor_target(
                diurnal_target,
                lower=diurnal_factor_min,
                upper=diurnal_factor_max,
            )
            diurnal_activity_targets.append(
                {
                    "isoprene_activity_correction_ML": (
                        encoded_diurnal_target.detach().clone()
                    )
                }
            )

            residual_state = _clone_state_mapping(diurnal_state)
            residual_reference_obs_units = physics_isoprene_obs_units * diurnal_target
            residual_target = torch.ones_like(residual_reference)
            if _is_valid_observation_tensor(isoprene_obs):
                residual_target = isoprene_obs.detach().clone().reshape_as(
                    residual_reference_obs_units
                ) / torch.clamp(residual_reference_obs_units, min=eps)
            residual_target = torch.clamp(
                residual_target,
                min=torch.as_tensor(
                    residual_factor_min,
                    dtype=residual_target.dtype,
                    device=residual_target.device,
                ),
                max=torch.as_tensor(
                    residual_factor_max,
                    dtype=residual_target.dtype,
                    device=residual_target.device,
                ),
            )
            isoprene_residual_states.append(residual_state)
            encoded_residual_target = _encode_bounded_factor_target(
                residual_target,
                lower=residual_factor_min,
                upper=residual_factor_max,
            )
            isoprene_residual_targets.append(
                {
                    "isoprene_correction_factor_ML": (
                        encoded_residual_target.detach().clone()
                    )
                }
            )

            fine_tune_samples.append(
                {
                    "step": step,
                    "time_seconds": float(physics_only.current_time),
                    "state": _clone_state_mapping(current_state),
                    "observations": {
                        "soil_moisture": None if soil_obs is None else soil_obs.detach().clone(),
                        "net_photosynthesis": None if net_photo_obs is None else net_photo_obs.detach().clone(),
                        "isoprene_emission": None if isoprene_obs is None else isoprene_obs.detach().clone(),
                    },
                    "physics_reference": {
                        "soil_moisture": current_state["soil_moisture"].detach().clone(),
                        "water_stress_factor": current_state["water_stress_factor"].detach().clone(),
                        "stomatal_conductance": current_state["stomatal_conductance"].detach().clone(),
                        "ozone_stress_factor": current_state["ozone_stress_factor"].detach().clone(),
                    },
                }
            )
    finally:
        _close_progress_bar(sampling_bar)

    return {
        "soil_states": soil_states,
        "soil_targets": soil_targets,
        "water_stress_states": water_stress_states,
        "water_stress_targets": water_stress_targets,
        "stomatal_states": stomatal_states,
        "stomatal_targets": stomatal_targets,
        "ozone_states": ozone_states,
        "ozone_targets": ozone_targets,
        "diurnal_activity_states": diurnal_activity_states,
        "diurnal_activity_targets": diurnal_activity_targets,
        "isoprene_residual_states": isoprene_residual_states,
        "isoprene_residual_targets": isoprene_residual_targets,
        "fine_tune_samples": fine_tune_samples,
    }


def train_ml_surrogates(
    *,
    simulator: RELICSimulator,
    sim_cfg: Dict[str, object],
    ml_cfg: Optional[Dict[str, object]],
    loader: CaseNetCDFLoader,
    steps: int,
    dt_seconds: float,
    log_fn: Callable[[str], None],
    show_progress: bool = False,
    training_payload: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, object]]:
    if training_payload is None:
        training_payload = _collect_ml_training_samples(
            sim_cfg=sim_cfg,
            ml_cfg=ml_cfg,
            loader=loader,
            steps=steps,
            dt_seconds=dt_seconds,
            log_fn=log_fn,
            show_progress=show_progress,
        )

    runtime_settings = _resolve_ml_runtime_settings(sim_cfg)
    epochs_override = runtime_settings.get("training_epochs")
    epochs_value = None if epochs_override in (None, "", 0) else int(epochs_override)

    results: Dict[str, Dict[str, object]] = {}
    surrogate_map = {
        "ml_soil_moisture": (
            training_payload["soil_states"],
            training_payload["soil_targets"],
        ),
        "ml_water_stress": (
            training_payload["water_stress_states"],
            training_payload["water_stress_targets"],
        ),
        "ml_stomatal": (
            training_payload["stomatal_states"],
            training_payload["stomatal_targets"],
        ),
        "ml_ozone_stress": (
            training_payload["ozone_states"],
            training_payload["ozone_targets"],
        ),
        "ml_diurnal_activity": (
            training_payload["diurnal_activity_states"],
            training_payload["diurnal_activity_targets"],
        ),
        "ml_isoprene_residual": (
            training_payload["isoprene_residual_states"],
            training_payload["isoprene_residual_targets"],
        ),
    }
    for functor_name, raw_pairs in surrogate_map.items():
        functor = _get_registered_functor(simulator, functor_name)
        if not isinstance(
            functor,
            (
                CaseSoilMoistureSurrogate,
                CaseWaterStressSurrogate,
                CaseStomatalSurrogate,
                CaseOzoneStressSurrogate,
                CaseMEGANDiurnalActivitySurrogate,
                CaseIsopreneResidualSurrogate,
            ),
        ):
            continue
        _reset_ml_surrogate_model(functor)
        validation_fraction = float(functor.training_config.get("validation_split", 0.2))
        train_pairs, validation_pairs = _split_state_flux_pairs(
            raw_pairs[0],
            raw_pairs[1],
            validation_fraction=validation_fraction,
        )
        target_epochs = int(
            epochs_value
            if epochs_value not in (None, 0)
            else (
                functor.training_config.get("epochs", 100)
                if isinstance(functor.training_config, dict)
                else 100
            )
        )
        training_device = _resolve_ml_training_device_for_functor(
            sim_cfg=sim_cfg,
            functor_name=functor_name,
            functor=functor,
            sample_count=len(train_pairs[0]),
            epochs=target_epochs,
        )
        log_fn(
            f"[INFO] training {functor_name} with {len(train_pairs[0])} samples "
            f"on device={training_device}"
        )
        with _temporary_surrogate_training_device(functor, training_device):
            result = _train_surrogate_with_progress(
                functor=functor,
                functor_name=functor_name,
                train_pairs=train_pairs,
                validation_pairs=validation_pairs,
                epochs_value=epochs_value,
                show_progress=show_progress,
            )
        result["training_device"] = str(training_device)
        result["runtime_device"] = str(getattr(functor, "device", torch.device("cpu")))
        results[functor_name] = result
        final_train_loss = result.get("final_train_loss")
        final_val_loss = result.get("final_val_loss")
        log_fn(
            f"[INFO] trained {functor_name}: "
            f"epochs={result.get('epochs_trained')} "
            f"train_loss={final_train_loss} "
            f"val_loss={final_val_loss}"
        )
    return results


def fine_tune_ml_surrogates_with_observations(
    *,
    simulator: RELICSimulator,
    sim_cfg: Dict[str, object],
    loader: CaseNetCDFLoader,
    fine_tune_samples: Sequence[Dict[str, object]],
    dt_seconds: float,
    log_fn: Callable[[str], None],
    show_progress: bool = False,
) -> Dict[str, object]:
    soil_surrogate = _get_registered_functor(simulator, "ml_soil_moisture")
    water_surrogate = _get_registered_functor(simulator, "ml_water_stress")
    stomatal_surrogate = _get_registered_functor(simulator, "ml_stomatal")
    soil_is_trainable = isinstance(soil_surrogate, CaseSoilMoistureSurrogate) and soil_surrogate.model is not None
    water_is_trainable = (
        isinstance(water_surrogate, CaseWaterStressSurrogate) and water_surrogate.model is not None
    )
    if not isinstance(stomatal_surrogate, CaseStomatalSurrogate):
        return {}
    ozone_surrogate = _get_registered_functor(simulator, "ml_ozone_stress")
    residual_surrogate = _get_registered_functor(simulator, "ml_isoprene_residual")
    if stomatal_surrogate.model is None:
        return {}

    runtime_settings = _resolve_ml_runtime_settings(sim_cfg)
    guidance_cfg = runtime_settings.get("observation_guidance", {})
    if not isinstance(guidance_cfg, dict) or not bool(guidance_cfg.get("enabled", False)):
        return {}

    epochs = int(guidance_cfg.get("epochs", 0) or 0)
    if epochs <= 0:
        return {}

    learning_rate = float(guidance_cfg.get("learning_rate", 5.0e-4))
    gradient_clip_norm = float(guidance_cfg.get("gradient_clip_norm", 0.0) or 0.0)
    teacher_regularization = float(guidance_cfg.get("teacher_regularization", 0.0) or 0.0)
    residual_smoothness = float(guidance_cfg.get("residual_smoothness", 0.0) or 0.0)
    isoprene_loss_mode = str(guidance_cfg.get("isoprene_loss_mode", "linear_mse")).strip().lower()
    loss_weights_cfg = guidance_cfg.get("loss_weights", {})
    if not isinstance(loss_weights_cfg, dict):
        loss_weights_cfg = {}

    trainable_parameters: List[torch.nn.Parameter] = []
    residual_is_trainable = (
        isinstance(residual_surrogate, CaseIsopreneResidualSurrogate)
        and residual_surrogate.model is not None
    )
    if residual_is_trainable:
        trainable_parameters.extend(list(residual_surrogate.model.parameters()))
    else:
        if soil_is_trainable:
            trainable_parameters.extend(list(soil_surrogate.model.parameters()))
        if water_is_trainable:
            trainable_parameters.extend(list(water_surrogate.model.parameters()))
        trainable_parameters.extend(list(stomatal_surrogate.model.parameters()))
    ozone_is_trainable = isinstance(ozone_surrogate, CaseOzoneStressSurrogate) and ozone_surrogate.model is not None
    if ozone_is_trainable and not residual_is_trainable:
        trainable_parameters.extend(list(ozone_surrogate.model.parameters()))
    if not trainable_parameters:
        return {}
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    mse_loss = torch.nn.MSELoss()
    _set_ml_surrogate_mode(simulator, training=True)

    summary_history: List[float] = []
    component_history: List[Dict[str, float]] = []
    fine_tune_bar = _create_progress_bar(
        enabled=show_progress,
        total=epochs,
        desc="observation fine-tune",
        position=0,
        leave=False,
    )
    try:
        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            epoch_step_count = 0
            component_sums = {
                "soil_moisture": 0.0,
                "net_photosynthesis": 0.0,
                "isoprene_emission": 0.0,
                "smooth_isoprene_residual": 0.0,
                "teacher_water": 0.0,
                "teacher_stomatal": 0.0,
                "teacher_ozone": 0.0,
                "teacher_isoprene_residual": 0.0,
            }
            previous_correction_prediction: Optional[torch.Tensor] = None

            for sample in fine_tune_samples:
                state = sample.get("state")
                observations = sample.get("observations")
                physics_reference = sample.get("physics_reference")
                time_seconds = sample.get("time_seconds")
                if not isinstance(state, dict) or not isinstance(observations, dict):
                    continue
                if not isinstance(physics_reference, dict):
                    physics_reference = {}
                if not isinstance(time_seconds, (int, float)):
                    continue

                ml_outputs = compute_ml_path_diagnostics(
                    simulator=simulator,
                    state=state,
                    time_seconds=float(time_seconds),
                    dt_seconds=dt_seconds,
                )

                loss_terms: List[torch.Tensor] = []
                for obs_key, ml_key in (
                    ("soil_moisture", "soil_moisture_ML"),
                    ("net_photosynthesis", "net_photosynthesis_ML"),
                    ("isoprene_emission", "isoprene_emission_ML"),
                ):
                    observed = observations.get(obs_key)
                    predicted = ml_outputs.get(ml_key)
                    if not _is_valid_observation_tensor(observed):
                        continue
                    if not isinstance(predicted, torch.Tensor):
                        continue
                    comparable_predicted = _convert_prediction_to_observation_units(
                        loader,
                        obs_key,
                        predicted,
                    )
                    base_weight = float(loss_weights_cfg.get(obs_key, 1.0))
                    quality_weight = loader.get_obs_weight_multiplier(obs_key) * _observation_quality_scale(
                        loader, obs_key
                    )
                    total_weight = base_weight * quality_weight
                    if total_weight <= 0.0:
                        continue
                    target_tensor = observed.to(
                        dtype=comparable_predicted.dtype,
                        device=comparable_predicted.device,
                    ).reshape_as(comparable_predicted)
                    if obs_key == "isoprene_emission" and isoprene_loss_mode == "log1p_mse":
                        # References:
                        # - Guenther et al. (2012), isoprene emissions span orders of magnitude across diurnal forcing.
                        # - Log-space residual fitting reduces dominance of daytime peaks in skewed emission targets.
                        # Mathematical expression:
                        # - L_iso = MSE(log(1 + E_pred), log(1 + E_obs))
                        # Parameter meanings:
                        # - E_pred: predicted isoprene emission in observation units [ug m-2 s-1]
                        # - E_obs: observed isoprene emission in observation units [ug m-2 s-1]
                        comparable_predicted = torch.clamp(comparable_predicted, min=0.0)
                        target_tensor = torch.clamp(target_tensor, min=0.0)
                        component = mse_loss(
                            torch.log1p(comparable_predicted),
                            torch.log1p(target_tensor),
                        )
                    else:
                        component = mse_loss(comparable_predicted, target_tensor)
                    weighted_component = component * total_weight
                    loss_terms.append(weighted_component)
                    component_sums[obs_key] += float(weighted_component.detach().item())

                if teacher_regularization > 0.0:
                    water_teacher_target = physics_reference.get("water_stress_factor")
                    predicted_water = ml_outputs.get("water_stress_factor_ML")
                    if (
                        water_is_trainable
                        and _is_valid_observation_tensor(water_teacher_target)
                        and isinstance(predicted_water, torch.Tensor)
                    ):
                        water_teacher_tensor = water_teacher_target.to(
                            dtype=predicted_water.dtype,
                            device=predicted_water.device,
                        ).reshape_as(predicted_water)
                        water_teacher_component = mse_loss(
                            predicted_water,
                            water_teacher_tensor,
                        ) * teacher_regularization
                        loss_terms.append(water_teacher_component)
                        component_sums["teacher_water"] += float(
                            water_teacher_component.detach().item()
                        )

                    teacher_target = physics_reference.get("stomatal_conductance")
                    predicted_stomatal = ml_outputs.get("stomatal_conductance_ML")
                    if _is_valid_observation_tensor(teacher_target) and isinstance(predicted_stomatal, torch.Tensor):
                        teacher_tensor = teacher_target.to(
                            dtype=predicted_stomatal.dtype,
                            device=predicted_stomatal.device,
                        ).reshape_as(predicted_stomatal)
                        teacher_component = mse_loss(
                            predicted_stomatal,
                            teacher_tensor,
                        ) * teacher_regularization
                        loss_terms.append(teacher_component)
                        component_sums["teacher_stomatal"] += float(teacher_component.detach().item())

                    ozone_teacher_target = physics_reference.get("ozone_stress_factor")
                    predicted_ozone = ml_outputs.get("ozone_stress_factor_ML")
                    if (
                        ozone_is_trainable
                        and not residual_is_trainable
                        and _is_valid_observation_tensor(ozone_teacher_target)
                        and isinstance(predicted_ozone, torch.Tensor)
                    ):
                        ozone_teacher_tensor = ozone_teacher_target.to(
                            dtype=predicted_ozone.dtype,
                            device=predicted_ozone.device,
                        ).reshape_as(predicted_ozone)
                        ozone_teacher_component = mse_loss(
                            predicted_ozone,
                            ozone_teacher_tensor,
                        ) * teacher_regularization
                        loss_terms.append(ozone_teacher_component)
                        component_sums["teacher_ozone"] += float(
                            ozone_teacher_component.detach().item()
                        )

                    predicted_correction = ml_outputs.get("isoprene_correction_factor_ML")
                    if residual_is_trainable and isinstance(predicted_correction, torch.Tensor):
                        if (
                            residual_smoothness > 0.0
                            and previous_correction_prediction is not None
                        ):
                            smoothness_component = mse_loss(
                                predicted_correction,
                                previous_correction_prediction.to(
                                    dtype=predicted_correction.dtype,
                                    device=predicted_correction.device,
                                ).reshape_as(predicted_correction),
                            ) * residual_smoothness
                            loss_terms.append(smoothness_component)
                            component_sums["smooth_isoprene_residual"] += float(
                                smoothness_component.detach().item()
                            )
                        correction_teacher = mse_loss(
                            predicted_correction,
                            torch.ones_like(predicted_correction),
                        ) * teacher_regularization
                        loss_terms.append(correction_teacher)
                        component_sums["teacher_isoprene_residual"] += float(
                            correction_teacher.detach().item()
                        )
                        previous_correction_prediction = predicted_correction.detach()

                if not loss_terms:
                    continue

                optimizer.zero_grad(set_to_none=True)
                total_loss = torch.stack(loss_terms).sum()
                total_loss.backward()
                if gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, gradient_clip_norm)
                optimizer.step()

                epoch_loss_sum += float(total_loss.detach().item())
                epoch_step_count += 1

            if epoch_step_count == 0:
                log_fn("[INFO] observation guidance skipped: no valid observed samples available")
                break

            avg_epoch_loss = epoch_loss_sum / float(epoch_step_count)
            summary_history.append(avg_epoch_loss)
            component_history.append(
                {
                    key: value / float(epoch_step_count)
                    for key, value in component_sums.items()
                }
            )
            _update_progress_bar(
                fine_tune_bar,
                advance=1,
                postfix={
                    "epoch": f"{epoch + 1}/{epochs}",
                    "loss": f"{avg_epoch_loss:.4g}",
                    "iso": f"{component_history[-1]['isoprene_emission']:.4g}",
                },
            )
            if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
                component_text = ", ".join(
                    f"{key}={component_history[-1][key]:.6f}"
                    for key in (
                        "soil_moisture",
                        "net_photosynthesis",
                        "isoprene_emission",
                        "smooth_isoprene_residual",
                        "teacher_water",
                        "teacher_stomatal",
                        "teacher_ozone",
                        "teacher_isoprene_residual",
                    )
                )
                log_fn(
                    f"[INFO] observation-guided fine-tuning epoch {epoch + 1}/{epochs}: "
                    f"loss={avg_epoch_loss:.6f}, {component_text}"
                )
    finally:
        _close_progress_bar(fine_tune_bar)

    _set_ml_surrogate_mode(simulator, training=False)
    if not summary_history:
        return {}
    return {
        "epochs": len(summary_history),
        "learning_rate": learning_rate,
        "final_loss": summary_history[-1],
        "loss_history": summary_history,
        "component_history": component_history,
    }


def prepare_ml_surrogates(
    *,
    simulator: RELICSimulator,
    sim_cfg: Dict[str, object],
    ml_cfg: Optional[Dict[str, object]],
    loader: CaseNetCDFLoader,
    steps: int,
    dt_seconds: float,
    log_fn: Callable[[str], None],
    show_progress: bool = False,
) -> Dict[str, Dict[str, object]]:
    ml_functors = sorted(_collect_ml_surrogate_functors(simulator).keys())
    if not ml_functors:
        return {}
    runtime_settings = _resolve_ml_runtime_settings(sim_cfg)
    should_train = bool(runtime_settings.get("train_from_physics", True))
    missing_models = [
        name
        for name in ml_functors
        if not bool(getattr(_get_registered_functor(simulator, name), "is_trained", False))
    ]
    if missing_models and not should_train:
        log_fn(
            "[INFO] ML surrogate checkpoint missing or untrained; enabling physics-derived training"
        )
        should_train = True
    guidance_cfg = runtime_settings.get("observation_guidance", {})
    guidance_enabled = isinstance(guidance_cfg, dict) and bool(
        guidance_cfg.get("enabled", False)
    )
    training_payload: Optional[Dict[str, object]] = None
    if should_train or guidance_enabled:
        training_payload = _collect_ml_training_samples(
            sim_cfg=sim_cfg,
            ml_cfg=ml_cfg,
            loader=loader,
            steps=steps,
            dt_seconds=dt_seconds,
            log_fn=log_fn,
            show_progress=show_progress,
        )
    summary: Dict[str, Dict[str, object]] = {}
    if should_train:
        summary.update(
            train_ml_surrogates(
                simulator=simulator,
                sim_cfg=sim_cfg,
                ml_cfg=ml_cfg,
                loader=loader,
                steps=steps,
                dt_seconds=dt_seconds,
                log_fn=log_fn,
                show_progress=show_progress,
                training_payload=training_payload,
            )
        )
    else:
        log_fn("[INFO] using existing ML surrogate checkpoints without retraining")

    fine_tune_summary: Dict[str, object] = {}
    if guidance_enabled:
        if training_payload is None:
            raise RuntimeError("ML observation guidance requires a training payload")
        fine_tune_samples = training_payload.get("fine_tune_samples", [])
        if not isinstance(fine_tune_samples, list):
            raise TypeError("fine_tune_samples must be a list")
        fine_tune_summary = fine_tune_ml_surrogates_with_observations(
            simulator=simulator,
            sim_cfg=sim_cfg,
            loader=loader,
            fine_tune_samples=fine_tune_samples,
            dt_seconds=dt_seconds,
            log_fn=log_fn,
            show_progress=show_progress,
        )
    if fine_tune_summary:
        summary["observation_guidance"] = fine_tune_summary

    _set_ml_surrogate_mode(simulator, training=False)
    return summary


def compute_ml_path_diagnostics(
    *,
    simulator: RELICSimulator,
    state: Dict[str, object],
    time_seconds: float,
    dt_seconds: float,
) -> Dict[str, torch.Tensor]:
    ml_functor_types = (
        CaseSoilMoistureSurrogate,
        CaseWaterStressSurrogate,
        CaseStomatalSurrogate,
        CaseOzoneStressSurrogate,
        CaseMEGANDiurnalActivitySurrogate,
        CaseIsopreneResidualSurrogate,
    )
    has_registered_ml_functor = any(
        isinstance(_get_registered_functor(simulator, functor_name), ml_functor_types)
        for functor_name in (
            "ml_soil_moisture",
            "ml_water_stress",
            "ml_stomatal",
            "ml_ozone_stress",
            "ml_diurnal_activity",
            "ml_isoprene_residual",
        )
    )
    if not has_registered_ml_functor:
        return {}

    derived: Dict[str, torch.Tensor] = {}

    soil_surrogate = _get_registered_functor(simulator, "ml_soil_moisture")
    if isinstance(soil_surrogate, CaseSoilMoistureSurrogate):
        # References:
        # - Razavi et al. (2012), surrogate emulation for hydrologic responses.
        # - Rodriguez-Iturbe and Porporato (2004), physically admissible soil-moisture range.
        # Mathematical expression:
        # - theta_ML = clip(NN(T_air, PPFD, VPD, P, Rn, U, LAI), theta_min, theta_max)
        # Parameter meanings:
        # - theta_min/theta_max: admissible soil moisture bounds [m3 m-3]
        soil_outputs = soil_surrogate.forward(state, time_seconds, dt_seconds)
        soil_moisture_ml = soil_outputs.get("soil_moisture_ML")
        if isinstance(soil_moisture_ml, torch.Tensor):
            derived["soil_moisture_ML"] = soil_moisture_ml
    else:
        soil_moisture_ml = state.get("soil_moisture_ML")
        if isinstance(soil_moisture_ml, torch.Tensor):
            derived["soil_moisture_ML"] = soil_moisture_ml

    water_surrogate = _get_registered_functor(simulator, "ml_water_stress")
    if isinstance(water_surrogate, CaseWaterStressSurrogate):
        # References:
        # - Egea et al. (2011), nonlinear water-stress effects on canopy conductance.
        # - Verhoef and Egea (2014), hydraulic parameter uncertainty under soil drying.
        # Mathematical expression:
        # - f_W_ML = clip(NN(theta_proxy, T_air, VPD, LAI), f_W_min, f_W_max)
        # Parameter meanings:
        # - theta_proxy: ML or physics soil moisture proxy [m3 m-3]
        # - f_W_min/f_W_max: admissible water stress factor bounds [-]
        water_state = dict(state)
        if "soil_moisture_ML" in derived:
            water_state["soil_moisture_proxy"] = derived["soil_moisture_ML"]
            water_state["soil_moisture_ML"] = derived["soil_moisture_ML"]
        elif "soil_moisture" in state and isinstance(state["soil_moisture"], torch.Tensor):
            water_state["soil_moisture_proxy"] = state["soil_moisture"]
        water_outputs = water_surrogate.forward(water_state, time_seconds, dt_seconds)
        water_stress_factor_ml = water_outputs.get("water_stress_factor_ML")
        if isinstance(water_stress_factor_ml, torch.Tensor):
            derived["water_stress_factor_ML"] = water_stress_factor_ml
            derived["water_stress_ML"] = water_stress_factor_ml
    else:
        stress_adapter = _get_registered_functor(simulator, "water_stress")
        if isinstance(stress_adapter, StressKeyAdapter) and (
            "soil_moisture_ML" in derived
            or ("soil_moisture" in state and isinstance(state["soil_moisture"], torch.Tensor))
        ):
            # References:
            # - Berry and Bjorkman (1980), stress-factor normalization.
            # - Manabe (1969), bucket-style soil-moisture stress proxy.
            # Mathematical expression:
            # - f_W,ML = clip((theta - theta_wp) / (theta_fc - theta_wp), fmin, fmax)
            # Parameter meanings:
            # - theta: ML or physics soil moisture used by the fallback stress path [m3 m-3]
            # - theta_wp: wilting point [m3 m-3]
            # - theta_fc: field capacity [m3 m-3]
            stress_state = dict(state)
            if "soil_moisture_ML" in derived:
                stress_state["soil_moisture"] = derived["soil_moisture_ML"]
            stress_outputs = stress_adapter.forward(stress_state, time_seconds, dt_seconds)
            derived.update(
                _suffix_outputs(
                    stress_outputs,
                    ["water_stress", "water_stress_factor"],
                )
            )

    farquhar = _get_registered_functor(simulator, "farquhar_photosynthesis")
    if isinstance(farquhar, CaseFarquharPhotosynthesis) and (
        "soil_moisture_ML" in derived
        or "water_stress_ML" in derived
        or "water_stress_factor_ML" in derived
    ):
        # References:
        # - Farquhar et al. (1980), biochemical photosynthesis core.
        # Mathematical expression:
        # - A_n,ML = FvCB(T, CO2, PAR, f_T, f_O3, f_W,ML)
        # Parameter meanings:
        # - f_W,ML: surrogate-informed water stress factor [-]
        farquhar_state = dict(state)
        if "water_stress_ML" in derived or "water_stress_factor_ML" in derived:
            # FarquharPhotosynthesis prioritizes leaf_water_potential over water_stress
            # when both are present. Remove the hydraulic proxy here so the explicit
            # surrogate water stress factor controls the ML comparison path.
            farquhar_state.pop("leaf_water_potential", None)
        if "water_stress_ML" in derived:
            farquhar_state["water_stress"] = derived["water_stress_ML"]
        if "water_stress_factor_ML" in derived:
            farquhar_state["water_stress_factor"] = derived["water_stress_factor_ML"]
        farquhar_outputs = farquhar.forward(farquhar_state, time_seconds, dt_seconds)
        derived.update(
            _suffix_outputs(
                farquhar_outputs,
                [
                    "gross_photosynthesis",
                    "net_photosynthesis",
                    "gross_primary_productivity",
                    "electron_transport_rate",
                ],
            )
        )

    stomatal = _get_registered_functor(simulator, "ml_stomatal")
    if isinstance(stomatal, CaseStomatalSurrogate):
        # References:
        # - Ball et al. (1987), assimilation-conductance empirical linkage.
        # - Leuning (1995), VPD-sensitive stomatal closure response.
        # Mathematical expression:
        # - g_s,ML = NN(T_air, VPD, PPFD, CO2, A_n, theta_proxy)
        # Parameter meanings:
        # - A_n: net photosynthesis used as stomatal predictor [umol m-2 s-1]
        # - theta_proxy: soil moisture proxy used by the stomatal surrogate [m3 m-3]
        stomatal_state = dict(state)
        if "net_photosynthesis_ML" in derived:
            stomatal_state["net_photosynthesis"] = derived["net_photosynthesis_ML"]
        if "soil_moisture_ML" in derived:
            stomatal_state["soil_moisture_proxy"] = derived["soil_moisture_ML"]
            stomatal_state["soil_moisture_ML"] = derived["soil_moisture_ML"]
        elif "soil_moisture" in state and isinstance(state["soil_moisture"], torch.Tensor):
            stomatal_state["soil_moisture_proxy"] = state["soil_moisture"]
        stomatal_outputs = stomatal.forward(stomatal_state, time_seconds, dt_seconds)
        if "stomatal_conductance_ML" in stomatal_outputs:
            derived["stomatal_conductance_ML"] = stomatal_outputs[
                "stomatal_conductance_ML"
            ]

    ozone_surrogate = _get_registered_functor(simulator, "ml_ozone_stress")
    if isinstance(ozone_surrogate, CaseOzoneStressSurrogate):
        # References:
        # - Lombardozzi et al. (2015), ozone damage scaling to vegetation function.
        # - Sitch et al. (2007), ozone forcing uncertainty in land-atmosphere coupling.
        # Mathematical expression:
        # - f_O3_ML = clip(NN(g_s_proxy, T_air, VPD, O3), f_O3_min, f_O3_max)
        # Parameter meanings:
        # - g_s_proxy: ML or physics stomatal conductance proxy [mol m-2 s-1]
        # - f_O3_min/f_O3_max: admissible ozone stress factor bounds [-]
        ozone_state = dict(state)
        if "stomatal_conductance_ML" in derived:
            ozone_state["stomatal_conductance_proxy"] = derived["stomatal_conductance_ML"]
            ozone_state["stomatal_conductance_ML"] = derived["stomatal_conductance_ML"]
        elif "stomatal_conductance" in state and isinstance(state["stomatal_conductance"], torch.Tensor):
            ozone_state["stomatal_conductance_proxy"] = state["stomatal_conductance"]
        ozone_outputs = ozone_surrogate.forward(ozone_state, time_seconds, dt_seconds)
        ozone_stress_factor_ml = ozone_outputs.get("ozone_stress_factor_ML")
        if isinstance(ozone_stress_factor_ml, torch.Tensor):
            derived["ozone_stress_factor_ML"] = ozone_stress_factor_ml
    else:
        ozone = _get_registered_functor(simulator, "ozone_stress")
        if isinstance(ozone, CaseOzoneStress) and (
            "stomatal_conductance_ML" in derived
            or ("stomatal_conductance" in state and isinstance(state["stomatal_conductance"], torch.Tensor))
        ):
            ozone_state = dict(state)
            if "stomatal_conductance_ML" in derived:
                ozone_state["stomatal_conductance"] = derived["stomatal_conductance_ML"]
            ozone_outputs = ozone.forward(ozone_state, time_seconds, dt_seconds)
            derived.update(
                _suffix_outputs(
                    ozone_outputs,
                    [
                        "ozone_stress_factor",
                        "ozone_uptake",
                        "antioxidant_depletion",
                    ],
                )
            )

    diurnal_surrogate = _get_registered_functor(simulator, "ml_diurnal_activity")
    if isinstance(diurnal_surrogate, CaseMEGANDiurnalActivitySurrogate):
        # References:
        # - Guenther et al. (2012), isoprene daytime flux is controlled by light-temperature activity factors.
        # - Reichstein et al. (2019), process-level ML corrections should preserve mechanistic structure.
        # Mathematical expression:
        # - f_LT,ML = clip(NN(PPFD, T_air, LAI, A_n, f_W, f_O3, sin_h, cos_h), f_min, f_max)
        # Parameter meanings:
        # - f_LT,ML: multiplicative correction factor for MEGAN light-temperature activity [-]
        diurnal_reference = None
        for candidate_key in (
            "isoprene_emission_ML",
            "isoprene_emission",
            "net_photosynthesis_ML",
            "net_photosynthesis",
        ):
            candidate_value = derived.get(candidate_key, state.get(candidate_key))
            if isinstance(candidate_value, torch.Tensor):
                diurnal_reference = candidate_value
                break
        diurnal_state = dict(state)
        if "net_photosynthesis_ML" in derived:
            diurnal_state["net_photosynthesis"] = derived["net_photosynthesis_ML"]
        if "water_stress_factor_ML" in derived:
            diurnal_state["water_stress_factor"] = derived["water_stress_factor_ML"]
        if "ozone_stress_factor_ML" in derived:
            diurnal_state["ozone_stress_factor"] = derived["ozone_stress_factor_ML"]
        diurnal_state = _augment_state_with_diurnal_features(
            diurnal_state,
            time_seconds=time_seconds,
            reference_tensor=diurnal_reference,
        )
        diurnal_outputs = diurnal_surrogate.forward(diurnal_state, time_seconds, dt_seconds)
        diurnal_correction = diurnal_outputs.get("isoprene_activity_correction_ML")
        if isinstance(diurnal_correction, torch.Tensor):
            derived["isoprene_activity_correction_ML"] = diurnal_correction

    megan = _get_registered_functor(simulator, "megan_bvoc_model")
    if isinstance(megan, CaseMEGANModel) and (
        "net_photosynthesis_ML" in derived
        or "water_stress_ML" in derived
        or "stomatal_conductance_ML" in derived
        or "ozone_stress_factor_ML" in derived
        or "isoprene_activity_correction_ML" in derived
    ):
        megan_state = dict(state)
        if "net_photosynthesis_ML" in derived:
            megan_state["net_photosynthesis"] = derived["net_photosynthesis_ML"]
        if "gpp_ML" in derived:
            megan_state["gpp"] = derived["gpp_ML"]
        if "water_stress_ML" in derived:
            megan_state["water_stress"] = derived["water_stress_ML"]
        if "water_stress_factor_ML" in derived:
            megan_state["water_stress_factor"] = derived["water_stress_factor_ML"]
        if "stomatal_conductance_ML" in derived:
            megan_state["stomatal_conductance"] = derived["stomatal_conductance_ML"]
        if "ozone_stress_factor_ML" in derived:
            megan_state["ozone_stress_factor"] = derived["ozone_stress_factor_ML"]
        if "isoprene_activity_correction_ML" in derived:
            megan_state["isoprene_activity_correction_ML"] = derived["isoprene_activity_correction_ML"]
        # References: RELIC MEGANModel input contract (relic/scientific/biosphere/bvoc/megan_model.py,
        #   canonical inputs must share one floating dtype since 2026-10-15; no implicit promotion).
        # Mathematical expression: x_k <- cast(x_k, dtype(T_air)) for every floating tensor x_k.
        # Parameter meanings: T_air is the forcing air temperature of this step; x_k are the
        #   MEGAN state entries, including the ML-path substitutes of this diagnostic branch.
        # Units: unchanged by the cast; only the floating storage type is aligned.
        # Domain/constraints: the ML surrogates may hold the torch default dtype; the forcing
        #   dtype is the case precision, which MEGAN used before the strict rule as well.
        forcing_dtype = megan_state["air_temperature"].dtype
        megan_state = {
            key: (value.to(dtype=forcing_dtype)
                  if isinstance(value, torch.Tensor) and value.is_floating_point() else value)
            for key, value in megan_state.items()
        }
        megan_outputs = megan.forward(megan_state, time_seconds, dt_seconds)
        derived.update(_suffix_outputs(megan_outputs, list(megan_outputs.keys())))

    isoprene_residual = _get_registered_functor(simulator, "ml_isoprene_residual")
    if isinstance(isoprene_residual, CaseIsopreneResidualSurrogate) and "isoprene_emission_ML" in derived:
        # References:
        # - Guenther et al. (2012), MEGAN remains the mechanistic isoprene backbone.
        # - Residual correction acts multiplicatively on the mechanistic estimate only.
        # Mathematical expression:
        # - E_iso_hybrid = E_iso_chain * f_iso,residual
        # Parameter meanings:
        # - E_iso_chain: isoprene emission from the MEGAN-based ML chain [mol m-2 s-1]
        # - f_iso,residual: bounded residual correction factor predicted by the MLP [-]
        chain_isoprene = derived["isoprene_emission_ML"]
        residual_state = dict(state)
        residual_state.update(derived)
        residual_state["isoprene_emission"] = chain_isoprene
        if "water_stress_factor_ML" in derived:
            residual_state["water_stress_factor"] = derived["water_stress_factor_ML"]
        if "ozone_stress_factor_ML" in derived:
            residual_state["ozone_stress_factor"] = derived["ozone_stress_factor_ML"]
        if "stomatal_conductance_ML" in derived:
            residual_state["stomatal_conductance"] = derived["stomatal_conductance_ML"]
        if "net_photosynthesis_ML" in derived:
            residual_state["net_photosynthesis"] = derived["net_photosynthesis_ML"]
        residual_state = _augment_state_with_diurnal_features(
            residual_state,
            time_seconds=time_seconds,
            reference_tensor=chain_isoprene,
        )
        residual_outputs = isoprene_residual.forward(residual_state, time_seconds, dt_seconds)
        correction_factor = residual_outputs.get("isoprene_correction_factor_ML")
        if isinstance(correction_factor, torch.Tensor):
            derived["isoprene_correction_factor_ML"] = correction_factor
            derived["isoprene_emission_chain_ML"] = chain_isoprene
            corrected_isoprene = chain_isoprene * correction_factor
            derived["isoprene_emission_ML"] = corrected_isoprene
            if "total_bvoc_emission_ML" in derived:
                derived["total_bvoc_emission_ML"] = (
                    derived["total_bvoc_emission_ML"] - chain_isoprene + corrected_isoprene
                )
            if "total_bvoc_carbon_flux_ML" in derived and isinstance(megan, CaseMEGANModel):
                carbon_fraction = megan._get_carbon_fraction("isoprene")
                if not isinstance(carbon_fraction, torch.Tensor):
                    carbon_fraction = torch.as_tensor(
                        carbon_fraction,
                        dtype=chain_isoprene.dtype,
                        device=chain_isoprene.device,
                    )
                else:
                    carbon_fraction = carbon_fraction.to(
                        dtype=chain_isoprene.dtype,
                        device=chain_isoprene.device,
                    )
                micrograms_to_grams = megan.micrograms_to_grams
                if not isinstance(micrograms_to_grams, torch.Tensor):
                    micrograms_to_grams = torch.as_tensor(
                        micrograms_to_grams,
                        dtype=chain_isoprene.dtype,
                        device=chain_isoprene.device,
                    )
                else:
                    micrograms_to_grams = micrograms_to_grams.to(
                        dtype=chain_isoprene.dtype,
                        device=chain_isoprene.device,
                    )
                old_isoprene_carbon = chain_isoprene * carbon_fraction * micrograms_to_grams
                new_isoprene_carbon = corrected_isoprene * carbon_fraction * micrograms_to_grams
                derived["total_bvoc_carbon_flux_ML"] = (
                    derived["total_bvoc_carbon_flux_ML"] - old_isoprene_carbon + new_isoprene_carbon
                )

    return derived


def build_simulator(
    sim_cfg: Dict[str, object],
    ml_cfg: Optional[Dict[str, object]] = None,
    ml_enabled_override: Optional[bool] = None,
) -> RELICSimulator:
    simulation = sim_cfg["simulation"]
    if not isinstance(simulation, dict):
        raise ValueError("simulation.yaml must contain a simulation mapping.")

    dt_seconds = float(simulation["dt_seconds"])
    n_steps = int(simulation["force_steps"])
    case_root = Path(__file__).resolve().parent
    irf_specs_path = _resolve_case_relative_path(
        case_root,
        simulation.get("irf_specs_path", "config/irf_specs.yaml"),
    )
    cfg = {
        "time_step": dt_seconds,
        "max_time": dt_seconds * n_steps,
        "device": simulation.get("device", "auto"),
        "require_gpu": bool(simulation.get("require_gpu", False)),
        "strict_dependencies": bool(simulation.get("strict_dependencies", False)),
        "strict_scale_detection": bool(simulation.get("strict_scale_detection", True)),
        "strict_conservation_enabled": bool(simulation.get("strict_conservation_enabled", False)),
        "conservation_checks": False,
        "enable_auto_scale": bool(simulation.get("enable_auto_scale", True)),
        "irf_specs_path": str(irf_specs_path),
        "irf_strict": bool(simulation.get("irf_strict", True)),
        "allow_bucket_surface_coexistence": bool(
            simulation.get("allow_bucket_surface_coexistence", True)
        ),
        # Units of case-level inputs declared by simulation.yaml (no name inference).
        "state_units": copy.deepcopy(simulation["state_units"]),
    }
    for override_key in ("device_overrides", "module_device_overrides"):
        override_mapping = simulation.get(override_key)
        if isinstance(override_mapping, dict):
            cfg[override_key] = copy.deepcopy(override_mapping)
    simulator = RELICSimulator(config=cfg)

    turbulence = TurbulenceFunctor(name="atmospheric_turbulence")
    # The case forcing supplies surface_resistance (tutorial.case.penman.surface_resistance_default)
    # and canopy_conductance, so Penman-Monteith reads its canopy resistance from state.
    penman = CasePenmanMonteithET(
        name="penman_monteith_et",
        config={"canopy_resistance_source": "state"},
    )
    precip_adapter = PrecipThroughfallAdapter(name="precip_adapter")
    surface_runoff = CaseSurfaceRunoff(name="surface_runoff")
    bucket = CaseBucketSoilMoisture(name="bucket_soil_moisture")
    thermal = CaseThermalStress(name="thermal_stress")
    ozone = CaseOzoneStress(name="ozone_stress")
    stress_adapter = StressKeyAdapter(name="water_stress")
    farquhar = CaseFarquharPhotosynthesis(
        name="farquhar_photosynthesis",
        validate_inputs=False,
    )
    stomatal_proxy = StomatalProxyFunctor(name="stomatal_proxy")
    megan = CaseMEGANModel(name="megan_bvoc_model")

    # Case topology: surface runoff consumes the throughfall adapter output.
    # Declared through the dependency API and read back by get_dependencies;
    # the adapter and Farquhar keep the declarations their classes make.
    surface_runoff.add_dependency("precip_adapter")

    for functor in [
        turbulence,
        penman,
        precip_adapter,
        surface_runoff,
        bucket,
        thermal,
        ozone,
        stress_adapter,
        farquhar,
        stomatal_proxy,
        megan,
    ]:
        if hasattr(functor, "metadata") and getattr(functor.metadata, "conservation_laws", None):
            functor.metadata.conservation_laws = []

    ordered = [
        turbulence,
        penman,
        precip_adapter,
        surface_runoff,
        bucket,
        thermal,
        ozone,
        stress_adapter,
        farquhar,
        stomatal_proxy,
        megan,
    ]
    for functor in ordered:
        simulator.register_functor(functor, execution_group="fast")

    ml_runtime = _resolve_ml_runtime_settings(sim_cfg)
    ml_enabled = bool(ml_runtime.get("enabled", True))
    if ml_enabled_override is not None:
        ml_enabled = bool(ml_enabled_override)
    if ml_enabled:
        manager = get_global_manager()
        model_dir = case_root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        enable_soil_surrogate = bool(ml_runtime.get("soil_moisture", True))
        enable_water_surrogate = bool(ml_runtime.get("water_stress", True))
        enable_stomatal_surrogate = bool(ml_runtime.get("stomatal", True))
        enable_ozone_surrogate = bool(ml_runtime.get("ozone_stress", True))
        enable_diurnal_activity = bool(ml_runtime.get("diurnal_activity", True))
        enable_isoprene_residual = bool(ml_runtime.get("isoprene_residual", True))

        if enable_soil_surrogate:
            soil_cfg = _get_ml_component_config(ml_cfg, "soil_moisture")
            soil_model_path = manager.get(
                "tutorial.case.ml_surrogate.soil_moisture.model_path",
                required=True,
            )
            soil_cfg["model_path"] = str(
                _resolve_case_relative_path(case_root, soil_model_path)
            )
            soil_surrogate = CaseSoilMoistureSurrogate(
                name="ml_soil_moisture",
                model_dir=model_dir,
                config=soil_cfg,
            )
            _register_case_aux_functor(
                simulator,
                name="ml_soil_moisture",
                functor=soil_surrogate,
            )

        if enable_water_surrogate:
            water_cfg = _get_ml_component_config(ml_cfg, "water_stress")
            water_model_path = manager.get(
                "tutorial.case.ml_surrogate.water_stress.model_path",
                required=True,
            )
            water_cfg["model_path"] = str(
                _resolve_case_relative_path(case_root, water_model_path)
            )
            if enable_soil_surrogate:
                water_cfg["dependencies"] = ["ml_soil_moisture"]
                water_cfg["input_mapping"] = {"soil_moisture_proxy": "soil_moisture_ML"}
            else:
                water_cfg["dependencies"] = ["bucket_soil_moisture"]
                water_cfg["input_mapping"] = {"soil_moisture_proxy": "soil_moisture"}
            water_surrogate = CaseWaterStressSurrogate(
                name="ml_water_stress",
                model_dir=model_dir,
                config=water_cfg,
            )
            _register_case_aux_functor(
                simulator,
                name="ml_water_stress",
                functor=water_surrogate,
            )

        if enable_stomatal_surrogate:
            stomatal_cfg = _get_ml_component_config(ml_cfg, "stomatal")
            stomatal_model_path = manager.get(
                "tutorial.case.ml_surrogate.stomatal.model_path",
                required=True,
            )
            stomatal_cfg["model_path"] = str(
                _resolve_case_relative_path(case_root, stomatal_model_path)
            )
            stomatal_cfg["dependencies"] = ["farquhar_photosynthesis"]
            if enable_soil_surrogate:
                stomatal_cfg["dependencies"].append("ml_soil_moisture")
                stomatal_cfg["input_mapping"] = {"soil_moisture_proxy": "soil_moisture_ML"}
            else:
                stomatal_cfg["input_mapping"] = {"soil_moisture_proxy": "soil_moisture"}
            stomatal_surrogate = CaseStomatalSurrogate(
                name="ml_stomatal",
                model_dir=model_dir,
                config=stomatal_cfg,
            )
            _register_case_aux_functor(
                simulator,
                name="ml_stomatal",
                functor=stomatal_surrogate,
            )

        if enable_ozone_surrogate:
            ozone_cfg = _get_ml_component_config(ml_cfg, "ozone_stress")
            ozone_model_path = manager.get(
                "tutorial.case.ml_surrogate.ozone_stress.model_path",
                required=True,
            )
            ozone_cfg["model_path"] = str(
                _resolve_case_relative_path(case_root, ozone_model_path)
            )
            if enable_stomatal_surrogate:
                ozone_cfg["dependencies"] = ["ml_stomatal"]
                ozone_cfg["input_mapping"] = {
                    "stomatal_conductance_proxy": "stomatal_conductance_ML"
                }
            else:
                ozone_cfg["dependencies"] = ["stomatal_proxy"]
                ozone_cfg["input_mapping"] = {
                    "stomatal_conductance_proxy": "stomatal_conductance"
                }
            ozone_surrogate = CaseOzoneStressSurrogate(
                name="ml_ozone_stress",
                model_dir=model_dir,
                config=ozone_cfg,
            )
            _register_case_aux_functor(
                simulator,
                name="ml_ozone_stress",
                functor=ozone_surrogate,
            )

        if enable_diurnal_activity:
            diurnal_cfg = _get_ml_component_config(ml_cfg, "diurnal_activity")
            diurnal_model_path = manager.get(
                "tutorial.case.ml_surrogate.diurnal_activity.model_path",
                required=True,
            )
            diurnal_cfg["model_path"] = str(
                _resolve_case_relative_path(case_root, diurnal_model_path)
            )
            diurnal_cfg["dependencies"] = ["farquhar_photosynthesis", "megan_bvoc_model"]
            diurnal_surrogate = CaseMEGANDiurnalActivitySurrogate(
                name="ml_diurnal_activity",
                model_dir=model_dir,
                config=diurnal_cfg,
            )
            _register_case_aux_functor(
                simulator,
                name="ml_diurnal_activity",
                functor=diurnal_surrogate,
            )

        if enable_isoprene_residual:
            residual_cfg = _get_ml_component_config(ml_cfg, "isoprene_residual")
            residual_model_path = manager.get(
                "tutorial.case.ml_surrogate.isoprene_residual.model_path",
                required=True,
            )
            residual_cfg["model_path"] = str(
                _resolve_case_relative_path(case_root, residual_model_path)
            )
            residual_cfg["dependencies"] = ["megan_bvoc_model"]
            residual_surrogate = CaseIsopreneResidualSurrogate(
                name="ml_isoprene_residual",
                model_dir=model_dir,
                config=residual_cfg,
            )
            _register_case_aux_functor(
                simulator,
                name="ml_isoprene_residual",
                functor=residual_surrogate,
            )

    return simulator


def initialize_trainable_map(
    simulator: RELICSimulator,
    loader: CaseNetCDFLoader,
) -> Dict[str, torch.nn.Parameter]:
    bucket = simulator.functors["bucket_soil_moisture"]
    farquhar = simulator.functors["farquhar_photosynthesis"]
    megan = simulator.functors["megan_bvoc_model"]
    if not isinstance(megan, CaseMEGANModel):
        raise TypeError("Expected megan_bvoc_model to be CaseMEGANModel in tutorial case.")

    if not isinstance(bucket.field_capacity, torch.nn.Parameter):
        bucket.field_capacity = torch.nn.Parameter(bucket.field_capacity.clone().detach())
    if not isinstance(bucket.saturation, torch.nn.Parameter):
        bucket.saturation = torch.nn.Parameter(bucket.saturation.clone().detach())
    if not isinstance(bucket.wilting_point, torch.nn.Parameter):
        bucket.wilting_point = torch.nn.Parameter(bucket.wilting_point.clone().detach())
    if not isinstance(bucket.k_sat, torch.nn.Parameter):
        bucket.k_sat = torch.nn.Parameter(bucket.k_sat.clone().detach())
    if not isinstance(bucket.drainage_exponent, torch.nn.Parameter):
        bucket.drainage_exponent = torch.nn.Parameter(bucket.drainage_exponent.clone().detach())

    megan_trainables = megan.initialize_case_trainables()
    trainable = {
        "hydrosphere.soil_moisture.field_capacity": bucket.field_capacity,
        "hydrosphere.soil_moisture.porosity": bucket.saturation,
        "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": bucket.k_sat,
        "hydrosphere.soil_moisture.drainage_exponent": bucket.drainage_exponent,
        "tutorial.case.state_init.soil_moisture": loader.initial_soil_moisture,
        "biosphere.photosynthesis.vcmax25": farquhar.vcmax25,
        "biosphere.photosynthesis.jmax25": farquhar.jmax25,
        "biosphere.photosynthesis.rd25": farquhar.rd25,
        "tutorial.case.radiation.par_effective_scale": loader.par_effective_scale,
    }
    trainable.update(megan_trainables)
    return trainable


def apply_init_param_overrides(
    trainable: Dict[str, torch.nn.Parameter],
    init_param_yaml: Optional[Path],
) -> None:
    if init_param_yaml is None:
        return
    init_path = init_param_yaml.resolve()
    if not init_path.exists():
        raise FileNotFoundError(f"init parameter file not found: {init_path}")
    with init_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"init parameter YAML must be mapping: {init_path}")
    with torch.no_grad():
        for key, value in payload.items():
            if key not in trainable:
                continue
            param = trainable[key]
            scalar = float(value)
            param.copy_(torch.tensor(scalar, dtype=param.dtype, device=param.device))


def _coerce_saved_trainable_values(payload: Dict[str, object]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            values[str(key)] = to_float_scalar(value)
        except (TypeError, ValueError):
            continue
    return values


def _resolve_saved_run_dir(output_root: Path, run_id: str) -> Optional[Path]:
    candidate_raw = Path(str(run_id))
    candidates: List[Path] = []
    if candidate_raw.is_absolute():
        candidates.append(candidate_raw)
    else:
        candidates.extend(
            [
                output_root / run_id,
                output_root.parent / run_id,
                output_root.parent / "runs" / run_id,
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _load_saved_trainables_from_legacy_layout(
    *,
    output_root: Path,
    phase_name: Optional[str],
) -> Dict[str, float]:
    legacy_root = output_root.parent
    if phase_name:
        legacy_checkpoint_candidates: List[Path] = []
        for checkpoint_dir in [
            legacy_root / "checkpoints",
            legacy_root / "optimization" / "checkpoints",
        ]:
            if checkpoint_dir.exists():
                legacy_checkpoint_candidates.extend(
                    sorted(checkpoint_dir.glob(f"checkpoint_{phase_name}_epoch*.pt"))
                )
        if not legacy_checkpoint_candidates:
            raise FileNotFoundError(
                f"phase source parameters are missing: no checkpoint_{phase_name}_epoch*.pt found under "
                f"{legacy_root}. Re-run optimize with checkpoint output preserved, or provide --init-params."
            )
        checkpoint_payload = OptimizationLoopEngine.load_checkpoint(legacy_checkpoint_candidates[-1])
        trainable_payload = checkpoint_payload.get("trainable_params", {})
        if not isinstance(trainable_payload, dict):
            raise ValueError(f"invalid trainable_params payload in {legacy_checkpoint_candidates[-1]}")
        return _coerce_saved_trainable_values(trainable_payload)

    for params_path in [
        legacy_root / "optimization" / "optimized_parameters.yaml",
        legacy_root / "optimization" / "best_params.yaml",
    ]:
        if not params_path.exists():
            continue
        with params_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"optimized parameter YAML must be mapping: {params_path}")
        return _coerce_saved_trainable_values(payload)

    raise FileNotFoundError(
        f"saved optimized parameters are missing under {legacy_root / 'optimization'}. "
        "Provide a valid --init-from-run source or use --init-params."
    )


def load_saved_trainables_from_run(
    *,
    output_root: Path,
    run_id: str,
    phase_name: Optional[str] = None,
) -> Dict[str, float]:
    run_dir = _resolve_saved_run_dir(output_root, run_id)
    if run_dir is None:
        return _load_saved_trainables_from_legacy_layout(
            output_root=output_root,
            phase_name=phase_name,
        )

    if phase_name:
        checkpoint_dir = run_dir / "checkpoints"
        candidates = sorted(checkpoint_dir.glob(f"checkpoint_{phase_name}_epoch*.pt"))
        if not candidates:
            return _load_saved_trainables_from_legacy_layout(
                output_root=output_root,
                phase_name=phase_name,
            )
        checkpoint_payload = OptimizationLoopEngine.load_checkpoint(candidates[-1])
        trainable_payload = checkpoint_payload.get("trainable_params", {})
        if not isinstance(trainable_payload, dict):
            raise ValueError(f"invalid trainable_params payload in {candidates[-1]}")
        return _coerce_saved_trainable_values(trainable_payload)

    params_path = run_dir / "optimization" / "optimized_parameters.yaml"
    if not params_path.exists():
        params_path = run_dir / "optimization" / "best_params.yaml"
    if not params_path.exists():
        raise FileNotFoundError(f"optimized parameter file not found under {run_dir / 'optimization'}")
    with params_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"optimized parameter YAML must be mapping: {params_path}")
    return _coerce_saved_trainable_values(payload)


def apply_saved_run_overrides(
    trainable: Dict[str, torch.nn.Parameter],
    *,
    output_root: Path,
    run_id: Optional[str],
    phase_name: Optional[str],
) -> Optional[Dict[str, str]]:
    if run_id is None:
        return None
    loaded_values = load_saved_trainables_from_run(
        output_root=output_root,
        run_id=run_id,
        phase_name=phase_name,
    )
    with torch.no_grad():
        for key, value in loaded_values.items():
            if key not in trainable:
                continue
            param = trainable[key]
            param.copy_(torch.tensor(float(value), dtype=param.dtype, device=param.device))
    return {
        "run_id": run_id,
        "phase_name": phase_name or "final",
        "source": "checkpoint" if phase_name else "optimized_parameters",
    }


def to_float_scalar(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)  # type: ignore[arg-type]


def run_case_preflight_checks(
    *,
    manager: object,
    case_cfg: Dict[str, object],
    loader: CaseNetCDFLoader,
    log_fn: Callable[[str], None],
) -> None:
    override_keys = [
        "biosphere.stress.ozone.ozone_detox_capacity",
        "biosphere.stress.ozone.detox_capacity",
    ]
    for key in override_keys:
        expected = read_case_value(case_cfg, key)
        if expected is None:
            continue
        actual = to_float_scalar(manager.get(key, required=True))
        if not math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=0.0):
            raise RuntimeError(
                f"Parameter override mismatch for '{key}': expected={expected:.6e}, actual={actual:.6e}"
            )
        log_fn(f"[OK] parameter_override_active key={key} value={actual:.6e}")

    obs_keys = [
        "soil_moisture_obs",
        "net_photosynthesis_obs",
        "isoprene_emission_obs",
    ]
    for obs_key in obs_keys:
        quality_label = loader.get_obs_quality_label(obs_key)
        weight_multiplier = loader.get_obs_weight_multiplier(obs_key)
        quality_info = loader.get_obs_quality_info(obs_key)
        reason = quality_info.get("reason", "")
        log_fn(
            f"[INFO] observation_quality obs={obs_key} class={quality_label} "
            f"weight_multiplier={weight_multiplier:.3f} reason={reason}"
        )


def reset_instantaneous_states(state_manager: object) -> None:
    state = state_manager.get_state(clone=False)
    zero_updates: Dict[str, torch.Tensor] = {}
    for key in INSTANTANEOUS_STATE_KEYS:
        value = state.get(key)
        if isinstance(value, torch.Tensor):
            zero_updates[key] = torch.zeros_like(value)
    if zero_updates:
        state_manager.set_state(zero_updates, validate=False)


def run_forward_pass(
    *,
    simulator: RELICSimulator,
    loader: CaseNetCDFLoader,
    diagnostics: DiagnosticsWriter,
    steps: int,
    dt_seconds: float,
    reset_state: callable,
    epoch_label: int = 0,
    enable_grad: bool = True,
    phase_name: str = "simulate",
    log_fn: Optional[Callable[[str], None]] = None,
    monitoring_cfg: Optional[Dict[str, object]] = None,
    show_progress: bool = False,
) -> None:
    reset_state()
    has_ml_path = _simulator_has_ml_path(simulator)
    resolved_monitoring = dict(monitoring_cfg or {})
    monitoring_enabled = bool(resolved_monitoring.get("enabled", True))
    log_interval_steps = max(1, int(resolved_monitoring.get("log_interval_steps", 24)))
    checkpoint_interval_steps = max(
        1,
        int(resolved_monitoring.get("checkpoint_interval_steps", 24)),
    )
    physics_bar = _create_progress_bar(
        enabled=show_progress,
        total=steps,
        desc=f"{phase_name} physics",
        position=0,
        leave=False,
    )
    ml_bar = _create_progress_bar(
        enabled=show_progress and has_ml_path,
        total=steps,
        desc=f"{phase_name} ML",
        position=1,
        leave=False,
    )
    grad_context = nullcontext() if enable_grad else torch.no_grad()
    try:
        with grad_context:
            for step in range(steps):
                reset_instantaneous_states(simulator.state_manager)
                forcing = loader.get_forcing_step(step)
                simulator.state_manager.set_state(forcing, validate=False)
                step_outputs = simulator.step(dt_seconds)
                state = simulator.state_manager.get_state(clone=False)
                ml_outputs = (
                    compute_ml_path_diagnostics(
                        simulator=simulator,
                        state=state,
                        time_seconds=float(simulator.current_time),
                        dt_seconds=dt_seconds,
                    )
                    if has_ml_path
                    else {}
                )
                state_for_log = dict(state)
                state_for_log.update(ml_outputs)
                diagnostic_keys = _merge_runtime_diagnostics_for_logging(
                    simulator=simulator,
                    state_for_log=state_for_log,
                    step_outputs=step_outputs,
                )
                extra_keys = sorted(set(ml_outputs).union(diagnostic_keys))
                diagnostics.log_step(
                    epoch=epoch_label,
                    step=step,
                    time=loader.get_time(step),
                    state=state_for_log,
                    phase_name=phase_name,
                    extra_keys=extra_keys or None,
                )
                _update_progress_bar(
                    physics_bar,
                    advance=1,
                    postfix={
                        "soil": _monitor_scalar(state_for_log, "soil_moisture"),
                        "iso": _monitor_scalar(state_for_log, "isoprene_emission"),
                    },
                )
                if has_ml_path:
                    _update_progress_bar(
                        ml_bar,
                        advance=1,
                        postfix={
                            "soil_ML": _monitor_scalar(state_for_log, "soil_moisture_ML"),
                            "iso_ML": _monitor_scalar(state_for_log, "isoprene_emission_ML"),
                        },
                    )
                should_log = monitoring_enabled and (
                    step == 0 or (step + 1) % log_interval_steps == 0 or step == steps - 1
                )
                if should_log and log_fn is not None:
                    message = (
                        f"[MONITOR] phase={phase_name} step={step + 1}/{steps} "
                        f"soil={_monitor_scalar(state_for_log, 'soil_moisture')} "
                        f"gpp={_monitor_scalar(state_for_log, 'gross_primary_productivity')} "
                        f"iso={_monitor_scalar(state_for_log, 'isoprene_emission')}"
                    )
                    if has_ml_path:
                        message = (
                            f"{message} "
                            f"soil_ML={_monitor_scalar(state_for_log, 'soil_moisture_ML')} "
                            f"gpp_ML={_monitor_scalar(state_for_log, 'gross_primary_productivity_ML')} "
                            f"iso_ML={_monitor_scalar(state_for_log, 'isoprene_emission_ML')}"
                        )
                    log_fn(message)
                should_checkpoint = monitoring_enabled and (
                    step == 0 or (step + 1) % checkpoint_interval_steps == 0 or step == steps - 1
                )
                if should_checkpoint:
                    checkpoint_path = diagnostics.write_simulation_checkpoint(
                        epoch=epoch_label,
                        step=step,
                        time=loader.get_time(step),
                        state=state_for_log,
                        phase_name=phase_name,
                    )
                    if log_fn is not None:
                        log_fn(f"[CHECKPOINT] {phase_name} step={step + 1}/{steps}: {checkpoint_path.name}")
    finally:
        _close_progress_bar(physics_bar)
        _close_progress_bar(ml_bar)


def main() -> None:
    args = parse_args()
    case_root = args.case_root.resolve()
    config_dir = case_root / "config"
    data_processed = case_root / "data" / "processed"

    simulation_yaml = config_dir / "simulation.yaml"
    optimization_yaml = config_dir / "optimization.yaml"
    ml_surrogate_yaml = config_dir / "ml_surrogate.yaml"
    case_parameter_yaml = config_dir / "case_parameters.yaml"
    simulation_cfg = load_yaml(simulation_yaml)
    optimization_cfg = load_yaml(optimization_yaml)
    ml_surrogate_cfg = load_optional_yaml(ml_surrogate_yaml)
    case_cfg = load_yaml(case_parameter_yaml)

    obs_path = ensure_processed_data(case_root, use_mock_obs=args.use_mock_obs)
    forcing_path = data_processed / "site_forcing.nc"
    metadata_path = data_processed / "site_metadata.json"
    provenance_path = data_processed / ("provenance_mock.json" if args.use_mock_obs else "provenance.json")

    register_case_parameters(case_parameter_yaml)
    manager = get_global_manager()

    simulation = simulation_cfg["simulation"]
    if not isinstance(simulation, dict):
        raise ValueError("simulation section missing.")
    if args.no_ml_surrogate:
        simulation.setdefault("ml_surrogate", {})
        if not isinstance(simulation["ml_surrogate"], dict):
            simulation["ml_surrogate"] = {}
        simulation["ml_surrogate"]["enabled"] = False
    monitoring_cfg = _resolve_monitoring_settings(simulation_cfg)
    dt_seconds = float(simulation["dt_seconds"])
    output_root = case_root / str(simulation.get("output_dir", "output")) / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    diagnostics = DiagnosticsWriter(output_root / run_id)
    def log_info(message: str) -> None:
        print(message, flush=True)
        diagnostics.append_log(message)

    log_info(f"[INFO] run_id={run_id} mode={args.mode}")
    diagnostics.snapshot_inputs(
        [
            simulation_yaml,
            optimization_yaml,
            ml_surrogate_yaml,
            case_parameter_yaml,
            forcing_path,
            obs_path,
            metadata_path,
            provenance_path,
        ]
    )

    runtime_device = _resolve_case_runtime_device(simulation_cfg)
    loader = CaseNetCDFLoader(
        LoaderConfig(
            case_root=case_root,
            forcing_path=forcing_path,
            obs_path=obs_path,
            case_parameter_path=case_parameter_yaml,
            device=runtime_device,
        )
    )
    log_info(f"[INFO] resolved_runtime_device={runtime_device}")
    run_case_preflight_checks(
        manager=manager,
        case_cfg=case_cfg,
        loader=loader,
        log_fn=log_info,
    )

    simulator = build_simulator(simulation_cfg, ml_cfg=ml_surrogate_cfg)
    log_info(f"[INFO] simulator_device={simulator.device} loader_device={loader.device}")
    trainable = initialize_trainable_map(simulator, loader)
    apply_case_trainable_initial_values(trainable, case_cfg)
    saved_run_source = apply_saved_run_overrides(
        trainable,
        output_root=output_root,
        run_id=args.init_from_run,
        phase_name=args.init_from_phase,
    )
    apply_init_param_overrides(trainable, args.init_params)

    initial_state = loader.make_initial_state()
    simulator.initialize_state(initial_state)

    def reset_state() -> None:
        simulator.current_time = 0.0
        simulator.step_count = 0
        simulator.state_manager.reset()
        simulator.state_manager.set_state(loader.make_initial_state(), validate=False)

    start_time = time.perf_counter()

    if args.mode == "simulate":
        diagnostics.reset_step_logs()
        diagnostics.write_initial_trainable_parameters(
            {
                key: float(param.detach().reshape(-1)[0].tolist())
                for key, param in trainable.items()
            }
        )
        total_steps = loader.total_steps
        if args.days is not None:
            total_steps = min(total_steps, int(args.days * 24))
        ml_training_summary = prepare_ml_surrogates(
            simulator=simulator,
            sim_cfg=simulation_cfg,
            ml_cfg=ml_surrogate_cfg,
            loader=loader,
            steps=total_steps,
            dt_seconds=dt_seconds,
            log_fn=log_info,
            show_progress=not args.no_progress,
        )
        ml_checkpoint_manifest = _persist_ml_surrogate_checkpoints(
            simulator=simulator,
            diagnostics=diagnostics,
            log_fn=log_info,
        )
        run_forward_pass(
            simulator=simulator,
            loader=loader,
            diagnostics=diagnostics,
            steps=total_steps,
            dt_seconds=dt_seconds,
            reset_state=reset_state,
            epoch_label=0,
            enable_grad=False,
            phase_name="simulate",
            log_fn=log_info,
            monitoring_cfg=monitoring_cfg,
            show_progress=not args.no_progress,
        )
        monitor_outputs = diagnostics.write_process_monitor()
        diagnostics.write_timeseries()
        diagnostics.write_ml_training_summary(ml_training_summary)
        diagnostics.write_optimized_parameters(
            {
                key: float(param.detach().reshape(-1)[0].tolist())
                for key, param in trainable.items()
            }
        )
        elapsed = time.perf_counter() - start_time
        summary = {
            "run_id": run_id,
            "mode": "simulate",
            "steps": total_steps,
            "elapsed_seconds": elapsed,
            "obs_file": obs_path.name,
            "ml_surrogate_enabled": bool(
                _resolve_ml_runtime_settings(simulation_cfg).get("enabled", True)
            ),
            "ml_training_summary": ml_training_summary,
            "ml_checkpoint_manifest": ml_checkpoint_manifest,
            "monitor_outputs": {name: str(path.resolve()) for name, path in monitor_outputs.items()},
            "init_from_run": None if saved_run_source is None else saved_run_source["run_id"],
            "init_from_phase": None if saved_run_source is None else saved_run_source["phase_name"],
            "init_param_file": None if args.init_params is None else str(args.init_params.resolve()),
        }
        diagnostics.write_run_summary(summary)
        if saved_run_source is not None:
            log_info(
                f"[INFO] initialized simulate run from {saved_run_source['source']} "
                f"run={saved_run_source['run_id']} phase={saved_run_source['phase_name']}"
            )
        log_info(f"[OK] simulation completed in {elapsed:.2f} s")
        log_info(f"[OK] run output: {diagnostics.layout.run_dir}")
        return

    optimization = optimization_cfg.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("optimization section missing.")

    resume_context: Optional[ResumeContext] = None
    if args.resume:
        resume_output_root = case_root / str(simulation.get("output_dir", "output")) / "runs"
        resume_context = _resolve_resume_context(
            output_root=resume_output_root,
            run_id=args.resume,
        )
        log_info(f"[RESUME] loading checkpoint: {resume_context.checkpoint_path.name}")
        log_info(
            f"[RESUME] resolved source run={resume_context.run_id} "
            f"phase={resume_context.resume_phase} epoch={resume_context.resume_epoch}"
        )

    ml_resume_manifest: Dict[str, str] = {}
    if resume_context is not None:
        ml_resume_manifest = _restore_ml_surrogate_checkpoints_from_run(
            simulator=simulator,
            run_dir=resume_context.run_dir,
            log_fn=log_info,
        )
        ml_training_summary = {
            "resume_restore": {
                "source_run": resume_context.run_id,
                "checkpoint_path": str(resume_context.checkpoint_path.resolve()),
                "ml_surrogate_manifest": ml_resume_manifest,
            }
        }
    else:
        ml_training_summary = prepare_ml_surrogates(
            simulator=simulator,
            sim_cfg=simulation_cfg,
            ml_cfg=ml_surrogate_cfg,
            loader=loader,
            steps=loader.total_steps,
            dt_seconds=dt_seconds,
            log_fn=log_info,
            show_progress=not args.no_progress,
        )
    ml_checkpoint_manifest = _persist_ml_surrogate_checkpoints(
        simulator=simulator,
        diagnostics=diagnostics,
        log_fn=log_info,
    )

    engine = OptimizationLoopEngine(
        simulator=simulator,
        loader=loader,
        dt_seconds=dt_seconds,
        gradient_clip_norm=float(optimization.get("gradient_clip_norm", 1.0)),
        show_progress=not args.no_progress,
        checkpoint_dir=diagnostics.layout.checkpoint_dir,
        log_fn=log_info,
        progress_position=0,
    )
    diagnostics.reset_step_logs()

    all_loss_rows: Dict[str, List[Dict[str, object]]] = {}
    all_grad_rows: List[Dict[str, object]] = []
    phase_final_losses: Dict[str, float] = {}
    phase_final_window_losses: Dict[str, float] = {}
    de_phase_summaries: Dict[str, Dict[str, object]] = {}
    has_ml_path = _simulator_has_ml_path(simulator)

    all_param_bounds = {
        key: read_case_bounds(case_cfg, key)
        for key in trainable.keys()
    }

    phase_ids = [
        key
        for key, value in optimization.items()
        if isinstance(key, str) and key.lower().startswith("phase") and isinstance(value, dict)
    ]
    phase_ids = sorted(
        phase_ids,
        key=lambda name: (
            int("".join(ch for ch in name if ch.isdigit()) or "9999"),
            name,
        ),
    )
    if args.opt_phases:
        requested = [item.strip() for item in args.opt_phases.split(",") if item.strip()]
        phase_ids = [phase for phase in phase_ids if phase in requested]
    if not phase_ids:
        raise ValueError("No phase* configuration found under optimization.")

    resume_phase: Optional[str] = None
    resume_epoch: int = 0
    resume_epoch_loss: Optional[float] = None
    resume_optimizer_state = None
    completed_phases: List[str] = []
    if resume_context is not None:
        ckpt = resume_context.checkpoint_payload
        ckpt_params = ckpt.get("trainable_params", {})
        if not isinstance(ckpt_params, dict):
            raise ValueError(f"invalid trainable_params payload in {resume_context.checkpoint_path}")
        with torch.no_grad():
            for key, value in ckpt_params.items():
                if key in trainable:
                    trainable[key].copy_(value.to(trainable[key].device))
        completed_phases = list(resume_context.completed_phases)
        resume_phase = resume_context.resume_phase
        resume_epoch = resume_context.resume_epoch
        resume_epoch_loss = resume_context.resume_epoch_loss
        resume_optimizer_state = resume_context.resume_optimizer_state
        log_info(
            f"[RESUME] restored params, completed_phases={completed_phases}, "
            f"resume_phase={resume_phase}, resume_epoch={resume_epoch}"
        )

    diagnostics.write_initial_trainable_parameters(
        {
            key: float(param.detach().reshape(-1)[0].tolist())
            for key, param in trainable.items()
        }
    )

    for phase_id in phase_ids:
        # Skip completed phases when resuming.
        if phase_id in completed_phases:
            log_info(f"[SKIP] {phase_id}: already completed (from checkpoint)")
            continue

        phase_cfg = optimization.get(phase_id)
        if not isinstance(phase_cfg, dict):
            continue
        phase_cfg_local = dict(phase_cfg)
        if "convergence_threshold" not in phase_cfg_local and "convergence_threshold" in optimization:
            phase_cfg_local["convergence_threshold"] = optimization["convergence_threshold"]
        if "early_stopping" not in phase_cfg_local and isinstance(optimization.get("early_stopping"), dict):
            phase_cfg_local["early_stopping"] = optimization["early_stopping"]

        # Determine start_epoch and optimizer state for resume.
        current_start_epoch = 0
        current_optimizer_state = None
        if resume_phase == phase_id and resume_epoch > 0:
            current_start_epoch = resume_epoch
            current_optimizer_state = resume_optimizer_state
            log_info(f"[RESUME] {phase_id}: resuming from epoch {current_start_epoch}")
        phase_epochs = int(phase_cfg_local.get("epochs", 0))
        if current_start_epoch >= phase_epochs:
            if phase_id not in completed_phases:
                completed_phases.append(phase_id)
            reused_loss = float("nan")
            if resume_phase == phase_id and resume_epoch_loss is not None:
                reused_loss = float(resume_epoch_loss)
            all_loss_rows[phase_id] = []
            phase_final_losses[phase_id] = reused_loss
            phase_final_window_losses[phase_id] = float("nan")
            log_info(
                f"[SKIP] {phase_id}: start_epoch({current_start_epoch}) >= epochs({phase_epochs}), "
                "treat phase as completed."
            )
            continue

        log_info(f"[INFO] start {phase_id}: {phase_cfg_local.get('name', phase_id)}")
        ml_epoch_bar_holder: Dict[str, object] = {"bar": None, "epoch": None}
        de_preopt_cfg_raw = optimization.get("de_preoptimization", {})
        de_phase_cfg: Dict[str, object] = {}
        de_enabled = False
        if isinstance(de_preopt_cfg_raw, dict):
            de_phase_cfg = {
                key: value
                for key, value in de_preopt_cfg_raw.items()
                if key != "phases"
            }
            phase_overrides = de_preopt_cfg_raw.get("phases", {})
            if isinstance(phase_overrides, dict):
                phase_specific_cfg = phase_overrides.get(phase_id, {})
                if isinstance(phase_specific_cfg, dict):
                    de_phase_cfg.update(phase_specific_cfg)
            de_enabled = bool(de_preopt_cfg_raw.get("enabled", False)) and bool(
                de_phase_cfg.get("enabled", False)
            )
        de_pre_step = lambda _epoch, _step: reset_instantaneous_states(simulator.state_manager)

        if de_enabled and current_start_epoch == 0:
            de_summary: Optional[Dict[str, object]] = None
            if resume_context is not None and phase_id in resume_context.de_warmstart_payloads:
                de_summary = _apply_resume_de_warmstart(
                    phase_name=phase_id,
                    payload=resume_context.de_warmstart_payloads[phase_id],
                    trainable=trainable,
                    checkpoint_dir=diagnostics.layout.checkpoint_dir,
                    log_fn=log_info,
                )
            else:
                de_result = engine.run_de_preoptimization(
                    phase_name=phase_id,
                    phase_cfg=phase_cfg_local,
                    de_cfg=de_phase_cfg,
                    trainable_params=trainable,
                    param_bounds=all_param_bounds,
                    reset_hook=reset_state,
                    pre_step_hook=de_pre_step,
                )
                if de_result is not None:
                    de_summary = {
                        "optimized_parameters": de_result.optimized_parameters,
                        "objective_value": de_result.objective_value,
                        "initial_objective_value": de_result.initial_objective_value,
                        "iteration_count": de_result.iteration_count,
                        "success": de_result.success,
                        "termination_reason": de_result.termination_reason,
                        "convergence_history": de_result.convergence_history,
                        "checkpoint_path": None
                        if de_result.checkpoint_path is None
                        else str(de_result.checkpoint_path.resolve()),
                    }
            if de_summary is not None:
                de_phase_summaries[phase_id] = de_summary
                de_summary_path = diagnostics.layout.optimization_dir / f"{phase_id}_de_summary.json"
                de_summary_path.write_text(json.dumps(de_summary, indent=2), encoding="utf-8")
        elif de_enabled and current_start_epoch > 0:
            log_info(
                f"[INFO] phase={phase_id} DE warm-start skipped because resume starts at epoch {current_start_epoch}."
            )

        def pre_step(epoch: int, step: int) -> None:
            reset_instantaneous_states(simulator.state_manager)
            if (
                has_ml_path
                and
                not args.no_progress
                and step == 0
                and ml_epoch_bar_holder.get("epoch") != epoch
            ):
                _close_progress_bar(ml_epoch_bar_holder.get("bar"))
                ml_epoch_bar_holder["bar"] = _create_progress_bar(
                    enabled=True,
                    total=loader.total_steps,
                    desc=f"{phase_id} ML epoch {epoch + 1}/{phase_epochs}",
                    position=1,
                    leave=False,
                )
                ml_epoch_bar_holder["epoch"] = epoch

        def post_step(epoch: int, step: int, state: Dict[str, torch.Tensor]) -> None:
            ml_outputs = (
                compute_ml_path_diagnostics(
                    simulator=simulator,
                    state=state,
                    time_seconds=float(simulator.current_time),
                    dt_seconds=dt_seconds,
                )
                if has_ml_path
                else {}
            )
            state_for_log = dict(state)
            state_for_log.update(ml_outputs)
            diagnostic_keys = _merge_runtime_diagnostics_for_logging(
                simulator=simulator,
                state_for_log=state_for_log,
            )
            extra_keys = sorted(set(ml_outputs).union(diagnostic_keys))
            diagnostics.log_step(
                epoch=epoch,
                step=step,
                time=loader.get_time(step),
                state=state_for_log,
                phase_name=phase_id,
                extra_keys=extra_keys or None,
            )
            if has_ml_path:
                _update_progress_bar(
                    ml_epoch_bar_holder.get("bar"),
                    advance=1,
                    postfix={
                        "soil_ML": _monitor_scalar(state_for_log, "soil_moisture_ML"),
                        "iso_ML": _monitor_scalar(state_for_log, "isoprene_emission_ML"),
                    },
                )

        outputs = engine.run_phase(
            phase_name=phase_id,
            phase_cfg=phase_cfg_local,
            trainable_params=trainable,
            param_bounds=all_param_bounds,
            reset_hook=reset_state,
            pre_step_hook=pre_step,
            post_step_hook=post_step,
            start_epoch=current_start_epoch,
            resume_optimizer_state=current_optimizer_state,
            completed_phases=completed_phases,
        )
        _close_progress_bar(ml_epoch_bar_holder.get("bar"))
        completed_phases.append(phase_id)
        all_loss_rows[phase_id] = outputs.loss_rows
        all_grad_rows.extend(outputs.grad_rows)
        phase_final_losses[phase_id] = outputs.final_epoch_loss
        if outputs.loss_rows:
            phase_final_window_losses[phase_id] = float(outputs.loss_rows[-1]["loss"])
        else:
            phase_final_window_losses[phase_id] = float("nan")
        diagnostics.write_phase_loss(phase_id, outputs.loss_rows)
        diagnostics.write_gradient_norms(all_grad_rows)
        checkpoint_values = {
            key: float(param.detach().reshape(-1)[0].tolist())
            for key, param in trainable.items()
        }
        diagnostics.write_optimized_parameters(checkpoint_values)
        log_info(f"[INFO] done {phase_id}: final_epoch_loss={outputs.final_epoch_loss:.6f}")

    # Final forward pass with optimized parameters for clean final timeseries.
    run_forward_pass(
        simulator=simulator,
        loader=loader,
        diagnostics=diagnostics,
        steps=loader.total_steps,
        dt_seconds=dt_seconds,
        reset_state=reset_state,
        epoch_label=-1,
        enable_grad=False,
        phase_name="final_forward",
        log_fn=log_info,
        monitoring_cfg=monitoring_cfg,
        show_progress=not args.no_progress,
    )
    monitor_outputs = diagnostics.write_process_monitor()
    diagnostics.write_timeseries()
    diagnostics.write_ml_training_summary(ml_training_summary)

    optimized_values = {
        key: float(param.detach().reshape(-1)[0].tolist())
        for key, param in trainable.items()
    }
    diagnostics.write_optimized_parameters(optimized_values)

    elapsed = time.perf_counter() - start_time
    final_phase_id = phase_ids[-1]
    summary = {
        "run_id": run_id,
        "mode": "optimize",
        "steps": loader.total_steps,
        "elapsed_seconds": elapsed,
        "obs_file": obs_path.name,
        "phases_run": phase_ids,
        "phase_final_losses": phase_final_losses,
        "phase_final_window_losses": phase_final_window_losses,
        "final_phase": final_phase_id,
        "final_loss": phase_final_losses.get(final_phase_id),
        "final_window_loss": phase_final_window_losses.get(final_phase_id),
        "gradient_records": len(all_grad_rows),
        "phase_id_map": diagnostics.get_phase_id_map(),
        "ml_surrogate_enabled": bool(
            _resolve_ml_runtime_settings(simulation_cfg).get("enabled", True)
        ),
        "init_from_run": None if saved_run_source is None else saved_run_source["run_id"],
        "init_from_phase": None if saved_run_source is None else saved_run_source["phase_name"],
        "init_param_file": None if args.init_params is None else str(args.init_params.resolve()),
        "resume_source": None
        if resume_context is None
        else {
            "run_id": resume_context.run_id,
            "checkpoint_path": str(resume_context.checkpoint_path.resolve()),
            "resume_phase": resume_context.resume_phase,
            "resume_epoch": resume_context.resume_epoch,
            "completed_phases": resume_context.completed_phases,
            "ml_surrogate_manifest": ml_resume_manifest,
            "de_warmstart_phases": sorted(resume_context.de_warmstart_payloads.keys()),
        },
        "de_preoptimization": de_phase_summaries,
        "ml_training_summary": ml_training_summary,
        "ml_checkpoint_manifest": ml_checkpoint_manifest,
        "monitor_outputs": {name: str(path.resolve()) for name, path in monitor_outputs.items()},
    }
    diagnostics.write_run_summary(summary)

    log_info(f"[OK] optimization completed in {elapsed:.2f} s")
    log_info(f"[OK] run output: {diagnostics.layout.run_dir}")


if __name__ == "__main__":
    main()
