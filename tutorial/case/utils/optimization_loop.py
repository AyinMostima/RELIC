"""Reusable optimization engine for multi-phase truncated BPTT in tutorial case.

Supports epoch-level checkpoint saving and resume for fault-tolerant training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from relic.parameters.calibration.calibrator import (
    CalibrationConfig,
    CalibrationMethod,
    ParameterBounds,
    ParameterCalibrator,
)
from relic.parameters.calibration.hooks import (
    CalibrationCallback,
    CalibrationEvent,
    CalibrationEventData,
    CalibrationHooks,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PreStepHook = Callable[[int, int], None]
PostStepHook = Callable[[int, int, Dict[str, torch.Tensor]], None]
ResetHook = Callable[[], None]


@dataclass
class PhaseOutputs:
    loss_rows: List[Dict[str, object]]
    grad_rows: List[Dict[str, object]]
    final_epoch_loss: float


@dataclass
class DEPhaseOutputs:
    optimized_parameters: Dict[str, float]
    objective_value: float
    initial_objective_value: float
    convergence_history: List[float]
    iteration_count: int
    success: bool
    termination_reason: str
    checkpoint_path: Optional[Path] = None


class _DEProgressCallback(CalibrationCallback):
    """Log and render DE generation progress for tutorial optimization."""

    def __init__(
        self,
        *,
        phase_name: str,
        max_iterations: int,
        log_fn: Optional[Callable[[str], None]],
        progress_bar: Optional[object],
    ) -> None:
        self.phase_name = str(phase_name)
        self.max_iterations = int(max_iterations)
        self.log_fn = log_fn
        self.progress_bar = progress_bar
        self._seen_iterations = 0

    @property
    def supported_events(self) -> List[CalibrationEvent]:
        return [CalibrationEvent.ITERATION_COMPLETE]

    def __call__(self, event_data: CalibrationEventData) -> Optional[Dict[str, Any]]:
        if event_data.event != CalibrationEvent.ITERATION_COMPLETE:
            return None

        iteration = self._seen_iterations + 1
        self._seen_iterations = iteration
        objective_value = event_data.objective_value

        if self.progress_bar is not None:
            self.progress_bar.update(1)
            if objective_value is not None and hasattr(self.progress_bar, "set_postfix"):
                self.progress_bar.set_postfix({"objective": f"{float(objective_value):.4g}"})

        if self.log_fn is not None:
            objective_repr = "nan" if objective_value is None else f"{float(objective_value):.6f}"
            self.log_fn(
                f"[DE] phase={self.phase_name} iteration={iteration}/{self.max_iterations} "
                f"objective={objective_repr}"
            )
        return None


class OptimizationLoopEngine:
    """Windowed BPTT executor with finite-gradient checks and clipping."""

    def __init__(
        self,
        *,
        simulator: object,
        loader: object,
        dt_seconds: float,
        gradient_clip_norm: float,
        show_progress: bool = True,
        checkpoint_dir: Optional[Path] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        progress_position: int = 0,
    ) -> None:
        self.simulator = simulator
        self.loader = loader
        self.dt_seconds = float(dt_seconds)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.show_progress = bool(show_progress)
        self.checkpoint_dir = checkpoint_dir
        self.log_fn = log_fn
        self.progress_position = int(progress_position)
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _emit_message(self, message: str) -> None:
        if self.log_fn is not None:
            self.log_fn(str(message))
            return
        print(str(message), flush=True)

    def detach_states(self, keys: Sequence[str]) -> None:
        state_vars = getattr(self.simulator.state_manager, "state_vars", {})
        for key in keys:
            value = state_vars.get(key)
            if isinstance(value, torch.Tensor):
                state_vars[key] = value.detach()

    def _safe_float(self, value: torch.Tensor) -> float:
        return float(value.detach().reshape(-1)[0].tolist())

    def _compute_correlation_penalty(
        self,
        *,
        pred_valid: torch.Tensor,
        obs_valid: torch.Tensor,
        obj_target_key: str,
        phase_cfg: Dict[str, object],
    ) -> Optional[torch.Tensor]:
        corr_cfg_raw = phase_cfg.get("correlation_loss")
        if not isinstance(corr_cfg_raw, dict) or not bool(corr_cfg_raw.get("enabled", False)):
            return None
        apply_targets = corr_cfg_raw.get("target_state_keys", [])
        if isinstance(apply_targets, list) and len(apply_targets) > 0:
            if obj_target_key not in [str(v) for v in apply_targets]:
                return None
        min_count = int(corr_cfg_raw.get("min_count", 4))
        positive_only = bool(corr_cfg_raw.get("positive_only", False))
        threshold = float(corr_cfg_raw.get("threshold", 0.0))
        corr_weight = float(corr_cfg_raw.get("weight", 1.0))
        eps = float(corr_cfg_raw.get("epsilon", 1.0e-8))

        pred_corr = pred_valid
        obs_corr = obs_valid
        if positive_only:
            pos_mask = obs_valid > threshold
            if not bool(pos_mask.any()):
                return None
            pred_corr = pred_valid[pos_mask]
            obs_corr = obs_valid[pos_mask]

        if int(pred_corr.numel()) < min_count:
            return None

        pred_centered = pred_corr - torch.mean(pred_corr)
        obs_centered = obs_corr - torch.mean(obs_corr)
        covariance = torch.sum(pred_centered * obs_centered)
        pred_norm = torch.sqrt(torch.sum(pred_centered * pred_centered) + eps)
        obs_norm = torch.sqrt(torch.sum(obs_centered * obs_centered) + eps)
        corr = covariance / (pred_norm * obs_norm + eps)
        corr = torch.clamp(corr, min=-1.0, max=1.0)
        return (1.0 - corr) * corr_weight

    def _compute_bias_penalty(
        self,
        *,
        pred_valid: torch.Tensor,
        obs_valid: torch.Tensor,
        obj_target_key: str,
        phase_cfg: Dict[str, object],
    ) -> Optional[torch.Tensor]:
        bias_cfg_raw = phase_cfg.get("bias_penalty")
        if not isinstance(bias_cfg_raw, dict) or not bool(bias_cfg_raw.get("enabled", False)):
            return None
        apply_targets = bias_cfg_raw.get("target_state_keys", [])
        if isinstance(apply_targets, list) and len(apply_targets) > 0:
            if obj_target_key not in [str(v) for v in apply_targets]:
                return None
        positive_only = bool(bias_cfg_raw.get("positive_only", False))
        threshold = float(bias_cfg_raw.get("threshold", 0.0))
        min_count = int(bias_cfg_raw.get("min_count", 4))
        weight = float(bias_cfg_raw.get("weight", 1.0))

        pred_bias = pred_valid
        obs_bias = obs_valid
        if positive_only:
            mask = obs_valid > threshold
            if not bool(mask.any()):
                return None
            pred_bias = pred_valid[mask]
            obs_bias = obs_valid[mask]
        if int(pred_bias.numel()) < min_count:
            return None
        bias = torch.mean(pred_bias - obs_bias)
        return (bias * bias) * weight

    def _resolve_daytime_mask(
        self,
        *,
        pred: torch.Tensor,
        forcing: Optional[Dict[str, object]],
        phase_cfg: Dict[str, object],
    ) -> Optional[torch.Tensor]:
        """Build a boolean daytime mask aligned with the prediction tensor.

        References:
            - Guenther et al. (2012), isoprene emissions are strongly light-driven during daytime.
            - Pacifico et al. (2011), daytime tropical BVOC peaks dominate high-flux periods.

        Mathematical expression:
            - I_day = 1(PAR > PAR_threshold)

        Parameter meanings:
            - PAR: photosynthetically active radiation or PPFD [umol m-2 s-1]
            - PAR_threshold: threshold separating day from night [umol m-2 s-1]
        """
        day_cfg_raw = phase_cfg.get("daytime_loss")
        if not isinstance(day_cfg_raw, dict) or not bool(day_cfg_raw.get("enabled", False)):
            return None
        if not isinstance(forcing, dict):
            return None
        par_key = str(day_cfg_raw.get("par_key", "ppfd"))
        if par_key not in forcing:
            return None
        threshold = float(day_cfg_raw.get("par_threshold", 10.0))
        par_value = forcing[par_key]
        if isinstance(par_value, torch.Tensor):
            par_tensor = par_value.to(device=pred.device, dtype=pred.dtype)
        else:
            par_tensor = torch.as_tensor(par_value, dtype=pred.dtype, device=pred.device)
        day_tensor = par_tensor > threshold
        if day_tensor.dim() == 0:
            return torch.full_like(pred, bool(day_tensor.item()), dtype=torch.bool)
        if day_tensor.shape == pred.shape:
            return day_tensor.to(dtype=torch.bool)
        if day_tensor.numel() == pred.numel():
            return day_tensor.reshape(pred.shape).to(dtype=torch.bool)
        if day_tensor.numel() == 1:
            return torch.full_like(
                pred,
                bool(day_tensor.reshape(-1)[0].item()),
                dtype=torch.bool,
            )
        scalar_day = bool(day_tensor.reshape(-1).any().item())
        return torch.full_like(pred, scalar_day, dtype=torch.bool)

    def _compute_daytime_penalty(
        self,
        *,
        pred_valid: torch.Tensor,
        obs_valid: torch.Tensor,
        is_daytime: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
        obj_target_key: str,
        phase_cfg: Dict[str, object],
    ) -> Optional[torch.Tensor]:
        """Compute an asymmetric daytime penalty for underpredicted BVOC peaks.

        References:
            - Guenther et al. (2012), tropical isoprene fluxes peak during high-light daytime windows.
            - This work, daytime-weighted penalty for tropical site calibration.

        Mathematical expression:
            - L_day = w_pos * MSE(pred_day, obs_day)
                    + w_under * mean(relu(obs_day - pred_day)^2)

        Parameter meanings:
            - w_pos: extra daytime MSE multiplier [-]
            - w_under: asymmetric penalty on pred < obs during daytime [-]
            - pred_day, obs_day: valid daytime predictions and observations
        """
        day_cfg_raw = phase_cfg.get("daytime_loss")
        if not isinstance(day_cfg_raw, dict) or not bool(day_cfg_raw.get("enabled", False)):
            return None
        apply_targets = day_cfg_raw.get("target_state_keys", [])
        if isinstance(apply_targets, list) and len(apply_targets) > 0:
            if obj_target_key not in [str(v) for v in apply_targets]:
                return None
        if is_daytime is None:
            return None
        day_mask = is_daytime.to(device=pred_valid.device, dtype=torch.bool)
        if day_mask.dim() == 0:
            day_mask = torch.full_like(pred_valid, bool(day_mask.item()), dtype=torch.bool)
        elif day_mask.shape == valid_mask.shape:
            day_mask = day_mask[valid_mask]
        elif day_mask.numel() == valid_mask.numel():
            day_mask = day_mask.reshape(valid_mask.shape)[valid_mask]
        elif day_mask.numel() == 1:
            day_mask = torch.full_like(
                pred_valid,
                bool(day_mask.reshape(-1)[0].item()),
                dtype=torch.bool,
            )
        elif day_mask.numel() == pred_valid.numel():
            day_mask = day_mask.reshape(pred_valid.shape)
        else:
            raise ValueError(
                f"Daytime mask shape mismatch: got {tuple(day_mask.shape)}, "
                f"valid_mask={tuple(valid_mask.shape)}, pred_valid={tuple(pred_valid.shape)}"
            )

        if not bool(day_mask.any()):
            return None

        pred_day = pred_valid[day_mask]
        obs_day = obs_valid[day_mask]
        positive_weight = float(day_cfg_raw.get("positive_weight", 2.0))
        underprediction_weight = float(day_cfg_raw.get("underprediction_penalty_weight", 1.0))
        day_mse = F.mse_loss(pred_day, obs_day) * positive_weight
        underprediction = torch.relu(obs_day - pred_day)
        underprediction_penalty = torch.mean(underprediction * underprediction) * underprediction_weight
        return day_mse + underprediction_penalty

    def _apply_auxiliary_penalties(
        self,
        *,
        base_loss: torch.Tensor,
        pred_valid: torch.Tensor,
        obs_valid: torch.Tensor,
        valid_mask: torch.Tensor,
        is_daytime: Optional[torch.Tensor],
        obj_target_key: str,
        phase_cfg: Dict[str, object],
        weight: float,
    ) -> torch.Tensor:
        corr_penalty = self._compute_correlation_penalty(
            pred_valid=pred_valid,
            obs_valid=obs_valid,
            obj_target_key=obj_target_key,
            phase_cfg=phase_cfg,
        )
        if corr_penalty is not None:
            base_loss = base_loss + corr_penalty * float(weight)
        bias_penalty = self._compute_bias_penalty(
            pred_valid=pred_valid,
            obs_valid=obs_valid,
            obj_target_key=obj_target_key,
            phase_cfg=phase_cfg,
        )
        if bias_penalty is not None:
            base_loss = base_loss + bias_penalty * float(weight)
        daytime_penalty = self._compute_daytime_penalty(
            pred_valid=pred_valid,
            obs_valid=obs_valid,
            is_daytime=is_daytime,
            valid_mask=valid_mask,
            obj_target_key=obj_target_key,
            phase_cfg=phase_cfg,
        )
        if daytime_penalty is not None:
            base_loss = base_loss + daytime_penalty * float(weight)
        return base_loss

    def _compute_weighted_loss(
        self,
        *,
        pred: torch.Tensor,
        obs: torch.Tensor,
        weight: float,
        obj_target_key: str,
        phase_cfg: Dict[str, object],
        is_daytime: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        valid = torch.isfinite(obs)
        if not bool(valid.any()):
            return None
        pred_valid = pred[valid]
        obs_valid = obs[valid]
        base_loss: Optional[torch.Tensor] = None

        zero_cfg_raw = phase_cfg.get("zero_inflation")
        if isinstance(zero_cfg_raw, dict) and bool(zero_cfg_raw.get("enabled", False)):
            apply_targets = zero_cfg_raw.get("target_state_keys", [])
            if isinstance(apply_targets, list) and len(apply_targets) > 0:
                if obj_target_key not in [str(v) for v in apply_targets]:
                    base_loss = F.mse_loss(pred_valid, obs_valid) * float(weight)
                    return self._apply_auxiliary_penalties(
                        base_loss=base_loss,
                        pred_valid=pred_valid,
                        obs_valid=obs_valid,
                        valid_mask=valid,
                        is_daytime=is_daytime,
                        obj_target_key=obj_target_key,
                        phase_cfg=phase_cfg,
                        weight=weight,
                    )
            threshold = float(zero_cfg_raw.get("threshold", 0.0))
            zero_weight = float(zero_cfg_raw.get("zero_weight", 1.0))
            positive_weight = float(zero_cfg_raw.get("positive_weight", 1.0))
            false_positive_weight = float(zero_cfg_raw.get("false_positive_weight", 0.0))
            use_log1p = bool(zero_cfg_raw.get("log1p_positive", False))

            zero_mask = obs_valid <= threshold
            pos_mask = obs_valid > threshold
            total_loss: Optional[torch.Tensor] = None

            if bool(zero_mask.any()):
                pred_zero = pred_valid[zero_mask]
                zero_target = torch.zeros_like(pred_zero)
                zero_loss = F.mse_loss(pred_zero, zero_target) * zero_weight
                total_loss = zero_loss if total_loss is None else (total_loss + zero_loss)
                if false_positive_weight > 0.0:
                    fp = torch.relu(pred_zero - threshold)
                    fp_penalty = torch.mean(fp * fp) * false_positive_weight
                    total_loss = fp_penalty if total_loss is None else (total_loss + fp_penalty)

            if bool(pos_mask.any()):
                pred_pos = pred_valid[pos_mask]
                obs_pos = obs_valid[pos_mask]
                if use_log1p:
                    pred_term = torch.log1p(torch.relu(pred_pos))
                    obs_term = torch.log1p(torch.relu(obs_pos))
                    pos_loss = F.mse_loss(pred_term, obs_term) * positive_weight
                else:
                    pos_loss = F.mse_loss(pred_pos, obs_pos) * positive_weight
                total_loss = pos_loss if total_loss is None else (total_loss + pos_loss)

            if total_loss is None:
                return None
            base_loss = total_loss * float(weight)
            return self._apply_auxiliary_penalties(
                base_loss=base_loss,
                pred_valid=pred_valid,
                obs_valid=obs_valid,
                valid_mask=valid,
                is_daytime=is_daytime,
                obj_target_key=obj_target_key,
                phase_cfg=phase_cfg,
                weight=weight,
            )

        base_loss = F.mse_loss(pred_valid, obs_valid) * float(weight)
        return self._apply_auxiliary_penalties(
            base_loss=base_loss,
            pred_valid=pred_valid,
            obs_valid=obs_valid,
            valid_mask=valid,
            is_daytime=is_daytime,
            obj_target_key=obj_target_key,
            phase_cfg=phase_cfg,
            weight=weight,
        )

    def _save_checkpoint(
        self,
        phase_name: str,
        epoch: int,
        trainable_params: Dict[str, torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        loss_rows: List[Dict[str, object]],
        grad_rows: List[Dict[str, object]],
        epoch_loss: float,
        completed_phases: Optional[List[str]] = None,
    ) -> Path:
        """Save an epoch-level checkpoint to disk.

        References:
            - PyTorch checkpoint best practices:
              https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

        Mathematical expression:
            - No computation; purely serialization.

        Parameter meanings:
            - phase_name: current optimization phase identifier
            - epoch: 0-based epoch index within the current phase
            - trainable_params: dict of name -> nn.Parameter being optimized
            - optimizer: Adam optimizer whose state_dict is saved
            - loss_rows / grad_rows: accumulated log records
            - epoch_loss: scalar loss for this epoch
            - completed_phases: list of phase names already finished
        """
        if self.checkpoint_dir is None:
            raise RuntimeError("Cannot save checkpoint: checkpoint_dir is None")
        payload: Dict[str, Any] = {
            "phase_name": phase_name,
            "epoch": epoch,
            "completed_phases": completed_phases or [],
            "trainable_params": {
                key: param.detach().clone().cpu() for key, param in trainable_params.items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_rows": loss_rows,
            "grad_rows": grad_rows,
            "epoch_loss": epoch_loss,
        }
        path = self.checkpoint_dir / f"checkpoint_{phase_name}_epoch{epoch:04d}.pt"
        torch.save(payload, path)
        return path

    @staticmethod
    def load_checkpoint(path: Path) -> Dict[str, Any]:
        """Load a checkpoint from disk.

        Returns:
            Dict with keys: phase_name, epoch, completed_phases,
            trainable_params, optimizer_state_dict, loss_rows, grad_rows,
            epoch_loss.
        """
        return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
        """Find the most recent checkpoint file in a directory.

        Checkpoint files are named checkpoint_<phase>_epoch<NNNN>.pt and
        are sorted lexicographically so that the last file by sort order
        corresponds to the latest training progress.
        """
        if not checkpoint_dir.exists():
            return None
        candidates = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        return candidates[-1] if candidates else None

    def _resolve_objective_runtime_specs(
        self,
        *,
        phase_name: str,
        phase_cfg: Dict[str, object],
    ) -> List[Tuple[str, str, float]]:
        target_key = str(phase_cfg["target_state_key"])
        obs_key_cfg = phase_cfg.get("obs_key")
        obs_key = None if obs_key_cfg is None else str(obs_key_cfg)
        primary_weight = float(phase_cfg.get("primary_weight", 1.0))
        auxiliary_targets_cfg = phase_cfg.get("auxiliary_targets", [])
        auxiliary_targets: List[Tuple[str, Optional[str], float]] = []
        if isinstance(auxiliary_targets_cfg, list):
            for item in auxiliary_targets_cfg:
                if not isinstance(item, dict):
                    continue
                aux_target = item.get("target_state_key")
                if aux_target is None:
                    continue
                aux_obs = item.get("obs_key")
                aux_weight = float(item.get("weight", 1.0))
                auxiliary_targets.append(
                    (
                        str(aux_target),
                        None if aux_obs is None else str(aux_obs),
                        aux_weight,
                    )
                )
        objective_specs: List[Tuple[str, Optional[str], float]] = [
            (target_key, obs_key, primary_weight),
            *auxiliary_targets,
        ]
        objective_runtime_specs: List[Tuple[str, str, float]] = []
        for obj_target_key, obj_obs_key, obj_weight in objective_specs:
            if obj_weight <= 0.0:
                continue
            lookup_obs_key = obj_obs_key if obj_obs_key is not None else f"{obj_target_key}_obs"
            quality_multiplier = 1.0
            quality_label = "unknown"
            if hasattr(self.loader, "get_obs_weight_multiplier"):
                try:
                    quality_multiplier = float(self.loader.get_obs_weight_multiplier(lookup_obs_key))
                except (TypeError, ValueError):
                    quality_multiplier = 1.0
            if not math.isfinite(quality_multiplier):
                quality_multiplier = 1.0
            if hasattr(self.loader, "get_obs_quality_label"):
                try:
                    quality_label = str(self.loader.get_obs_quality_label(lookup_obs_key))
                except (TypeError, ValueError):
                    quality_label = "unknown"
            effective_weight = float(obj_weight) * max(0.0, quality_multiplier)
            if effective_weight <= 0.0:
                if self.show_progress:
                    self._emit_message(
                        f"  [WARN] phase={phase_name} objective={obj_target_key} obs={lookup_obs_key} "
                        f"skipped by observation quality class={quality_label}"
                    )
                continue
            if self.show_progress and quality_multiplier != 1.0:
                self._emit_message(
                    f"  [INFO] phase={phase_name} objective={obj_target_key} obs={lookup_obs_key} "
                    f"quality_class={quality_label} multiplier={quality_multiplier:.3f}"
                )
            objective_runtime_specs.append((obj_target_key, lookup_obs_key, effective_weight))
        return objective_runtime_specs

    def _evaluate_phase_objective(
        self,
        *,
        phase_cfg: Dict[str, object],
        objective_runtime_specs: List[Tuple[str, str, float]],
        reset_hook: ResetHook,
        pre_step_hook: Optional[PreStepHook] = None,
        post_step_hook: Optional[PostStepHook] = None,
        steps_limit: Optional[int] = None,
    ) -> float:
        total_steps = int(self.loader.total_steps)
        eval_steps = total_steps if steps_limit is None else min(total_steps, max(1, int(steps_limit)))
        reset_hook()
        total_loss = 0.0
        valid_loss_count = 0

        with torch.no_grad():
            for step in range(eval_steps):
                if pre_step_hook is not None:
                    pre_step_hook(0, step)

                forcing = self.loader.get_forcing_step(step)
                self.simulator.state_manager.set_state(forcing, validate=False)
                self.simulator.step(self.dt_seconds)
                state = self.simulator.state_manager.get_state(clone=False)

                if post_step_hook is not None:
                    post_step_hook(0, step, state)

                for obj_target_key, lookup_obs_key, obj_weight in objective_runtime_specs:
                    pred = state.get(obj_target_key)
                    if not isinstance(pred, torch.Tensor):
                        raise KeyError(f"Target state '{obj_target_key}' not available in state manager.")
                    obs = self.loader.get_obs_step(step, lookup_obs_key)
                    if obs is None:
                        continue
                    is_daytime = self._resolve_daytime_mask(
                        pred=pred,
                        forcing=forcing,
                        phase_cfg=phase_cfg,
                    )
                    step_loss = self._compute_weighted_loss(
                        pred=pred,
                        obs=obs,
                        weight=float(obj_weight),
                        obj_target_key=obj_target_key,
                        phase_cfg=phase_cfg,
                        is_daytime=is_daytime,
                    )
                    if step_loss is None:
                        continue
                    loss_value = float(step_loss.detach().item())
                    if not math.isfinite(loss_value):
                        return float("inf")
                    total_loss += loss_value
                    valid_loss_count += 1

        if valid_loss_count == 0:
            return float("inf")
        return float(total_loss)

    def _save_de_checkpoint(
        self,
        *,
        phase_name: str,
        payload: Dict[str, Any],
    ) -> Optional[Path]:
        if self.checkpoint_dir is None:
            return None
        path = self.checkpoint_dir / f"de_{phase_name}_warmstart.pt"
        torch.save(payload, path)
        return path

    def run_de_preoptimization(
        self,
        *,
        phase_name: str,
        phase_cfg: Dict[str, object],
        de_cfg: Dict[str, object],
        trainable_params: Dict[str, torch.nn.Parameter],
        param_bounds: Dict[str, Tuple[Optional[float], Optional[float]]],
        reset_hook: ResetHook,
        pre_step_hook: Optional[PreStepHook] = None,
        post_step_hook: Optional[PostStepHook] = None,
    ) -> Optional[DEPhaseOutputs]:
        selected_keys = [str(k) for k in phase_cfg["trainable_keys"] if str(k) in trainable_params]
        if not selected_keys:
            return None

        objective_runtime_specs = self._resolve_objective_runtime_specs(
            phase_name=phase_name,
            phase_cfg=phase_cfg,
        )
        if not objective_runtime_specs:
            if self.show_progress:
                self._emit_message(
                    f"  [WARN] phase={phase_name} has no valid objectives for DE preoptimization; skip."
                )
            return None

        bounds: Dict[str, ParameterBounds] = {}
        initial_parameters: Dict[str, float] = {}
        for key in selected_keys:
            lower, upper = param_bounds.get(key, (None, None))
            if lower is None or upper is None:
                raise ValueError(f"DE preoptimization requires explicit bounds for parameter '{key}'.")
            initial_value = float(trainable_params[key].detach().reshape(-1)[0].item())
            bounds[key] = ParameterBounds(
                lower=float(lower),
                upper=float(upper),
                scale="linear",
                initial=float(initial_value),
            )
            initial_parameters[key] = initial_value

        steps_limit = int(de_cfg.get("objective_steps", int(self.loader.total_steps)))
        max_iterations = int(de_cfg.get("max_iterations", 10))
        population_size_factor = max(1, int(de_cfg.get("population_size_factor", 8)))
        tolerance = float(de_cfg.get("tolerance", 1.0e-3))
        seed = de_cfg.get("seed", None)
        strategy = str(de_cfg.get("strategy", "best1bin"))
        mutation = de_cfg.get("mutation_factor", (0.5, 1.0))
        crossover_probability = float(de_cfg.get("crossover_probability", 0.8))
        requested_parallel = bool(de_cfg.get("parallel", False))
        requested_workers = de_cfg.get("n_workers", None)
        if requested_parallel and self.show_progress:
            self._emit_message(
                f"  [WARN] phase={phase_name} DE parallel workers disabled for tutorial case because "
                "objective evaluations mutate shared simulator state."
            )

        progress_bar = None
        hooks = CalibrationHooks()
        if self.show_progress and tqdm is not None:
            progress_bar = tqdm(
                total=max_iterations,
                desc=f"{phase_name} DE",
                position=max(self.progress_position + 2, 2),
                leave=False,
                dynamic_ncols=True,
            )
        hooks.register_callback(
            _DEProgressCallback(
                phase_name=phase_name,
                max_iterations=max_iterations,
                log_fn=self.log_fn,
                progress_bar=progress_bar,
            )
        )

        def objective(parameters: Dict[str, float], observations: Any) -> float:
            # RELIC declares no divergence error type that could be scored as an
            # infinite objective: an error of a candidate simulation propagates.
            del observations
            with torch.no_grad():
                for key, value in parameters.items():
                    if key not in trainable_params:
                        continue
                    param = trainable_params[key]
                    param.copy_(torch.as_tensor(float(value), dtype=param.dtype, device=param.device))
            return self._evaluate_phase_objective(
                phase_cfg=phase_cfg,
                objective_runtime_specs=objective_runtime_specs,
                reset_hook=reset_hook,
                pre_step_hook=pre_step_hook,
                post_step_hook=post_step_hook,
                steps_limit=steps_limit,
            )

        initial_objective_value = objective(initial_parameters, None)
        if self.show_progress:
            self._emit_message(
                f"  [INFO] phase={phase_name} DE start objective={initial_objective_value:.6f} "
                f"steps={min(int(self.loader.total_steps), steps_limit)}"
            )

        calibrator = ParameterCalibrator(hooks=hooks)
        config = CalibrationConfig(
            method=CalibrationMethod.DIFFERENTIAL_EVOLUTION,
            target_parameters=selected_keys,
            parameter_bounds=bounds,
            max_iterations=max_iterations,
            tolerance=tolerance,
            population_size=population_size_factor * len(selected_keys),
            random_seed=None if seed is None else int(seed),
            parallel=False,
            n_workers=None if requested_workers is None else int(requested_workers),
            method_kwargs={
                "strategy": strategy,
                "mutation": mutation,
                "recombination": crossover_probability,
                "polish": False,
            },
            verbose=False,
        )

        try:
            result = calibrator.calibrate(
                config=config,
                objective_function=objective,
                observations=torch.zeros(1, dtype=torch.float32),
                initial_parameters=initial_parameters,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        with torch.no_grad():
            for key, value in result.optimized_parameters.items():
                if key not in trainable_params:
                    continue
                param = trainable_params[key]
                param.copy_(torch.as_tensor(float(value), dtype=param.dtype, device=param.device))

        checkpoint_payload = {
            "kind": "de_warm_start",
            "phase_name": str(phase_name),
            "selected_keys": selected_keys,
            "initial_parameters": initial_parameters,
            "optimized_parameters": {
                key: float(value)
                for key, value in result.optimized_parameters.items()
            },
            "initial_objective_value": float(initial_objective_value),
            "objective_value": float(result.objective_value),
            "iteration_count": int(result.iteration_count),
            "success": bool(result.success),
            "termination_reason": str(result.termination_reason),
            "convergence_history": [float(v) for v in result.convergence_history],
        }
        checkpoint_path = self._save_de_checkpoint(
            phase_name=phase_name,
            payload=checkpoint_payload,
        )
        if checkpoint_path is not None and self.show_progress:
            self._emit_message(f"  [DE_CKPT] saved {checkpoint_path.name}")

        if self.show_progress:
            self._emit_message(
                f"  [INFO] phase={phase_name} DE done objective={float(result.objective_value):.6f} "
                f"iterations={int(result.iteration_count)} success={bool(result.success)}"
            )

        return DEPhaseOutputs(
            optimized_parameters={
                key: float(value)
                for key, value in result.optimized_parameters.items()
            },
            objective_value=float(result.objective_value),
            initial_objective_value=float(initial_objective_value),
            convergence_history=[float(v) for v in result.convergence_history],
            iteration_count=int(result.iteration_count),
            success=bool(result.success),
            termination_reason=str(result.termination_reason),
            checkpoint_path=checkpoint_path,
        )

    def run_phase(
        self,
        *,
        phase_name: str,
        phase_cfg: Dict[str, object],
        trainable_params: Dict[str, torch.nn.Parameter],
        param_bounds: Dict[str, Tuple[Optional[float], Optional[float]]],
        reset_hook: ResetHook,
        pre_step_hook: Optional[PreStepHook] = None,
        post_step_hook: Optional[PostStepHook] = None,
        start_epoch: int = 0,
        resume_optimizer_state: Optional[Dict[str, Any]] = None,
        completed_phases: Optional[List[str]] = None,
    ) -> PhaseOutputs:
        selected_keys = [str(k) for k in phase_cfg["trainable_keys"]]
        selected_params = [trainable_params[k] for k in selected_keys if k in trainable_params]
        if not selected_params:
            raise ValueError(f"No trainable parameters resolved for phase {phase_name}: {selected_keys}")

        for key, param in trainable_params.items():
            param.requires_grad_(key in selected_keys)

        optimizer = torch.optim.Adam(selected_params, lr=float(phase_cfg["lr"]))
        if resume_optimizer_state is not None:
            try:
                optimizer.load_state_dict(resume_optimizer_state)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to restore optimizer state for phase {phase_name}"
                ) from exc
        epochs = int(phase_cfg["epochs"])
        window_hours = int(phase_cfg["window_hours"])
        objective_runtime_specs = self._resolve_objective_runtime_specs(
            phase_name=phase_name,
            phase_cfg=phase_cfg,
        )

        if not objective_runtime_specs:
            if self.show_progress:
                self._emit_message(f"  [WARN] phase={phase_name} has no valid objectives after quality screening; skip phase.")
            return PhaseOutputs(loss_rows=[], grad_rows=[], final_epoch_loss=float("nan"))

        window_steps = max(1, int(round(window_hours * 3600.0 / self.dt_seconds)))
        detach_keys = [str(v) for v in phase_cfg.get("detach_state_keys", [])]
        convergence_threshold_raw = phase_cfg.get("convergence_threshold")
        convergence_threshold = None if convergence_threshold_raw is None else float(convergence_threshold_raw)
        early_cfg_raw = phase_cfg.get("early_stopping")
        early_cfg = early_cfg_raw if isinstance(early_cfg_raw, dict) else {}
        early_enabled = bool(early_cfg.get("enabled", False))
        early_mode = str(early_cfg.get("mode", "min")).lower()
        if early_mode not in {"min", "max"}:
            early_mode = "min"
        early_min_delta = float(early_cfg.get("min_delta", 0.0))
        early_patience = max(1, int(early_cfg.get("patience", 3)))
        early_warmup = max(0, int(early_cfg.get("warmup_epochs", 0)))
        early_restore_best = bool(early_cfg.get("restore_best", True))
        best_metric: Optional[float] = None
        best_epoch: Optional[int] = None
        epochs_without_improve = 0
        best_param_snapshot: Dict[str, torch.Tensor] = {}

        loss_rows: List[Dict[str, object]] = []
        grad_rows: List[Dict[str, object]] = []
        final_epoch_loss = float("nan")
        total_steps = int(self.loader.total_steps)
        previous_epoch_loss: Optional[float] = None
        last_completed_epoch = max(start_epoch - 1, -1)

        for epoch in range(start_epoch, epochs):
            reset_hook()
            window_loss: Optional[torch.Tensor] = None
            window_start = 0
            window_id = 0
            epoch_loss_value = 0.0

            step_iter: Iterable[int]
            if self.show_progress and tqdm is not None:
                step_iter = tqdm(
                    range(total_steps),
                    total=total_steps,
                    desc=f"{phase_name} epoch {epoch + 1}/{epochs}",
                    position=self.progress_position,
                    leave=False,
                    dynamic_ncols=True,
                )
            else:
                step_iter = range(total_steps)

            for step in step_iter:
                if pre_step_hook is not None:
                    pre_step_hook(epoch, step)

                forcing = self.loader.get_forcing_step(step)
                self.simulator.state_manager.set_state(forcing, validate=False)
                self.simulator.step(self.dt_seconds)
                state = self.simulator.state_manager.get_state(clone=False)

                if post_step_hook is not None:
                    post_step_hook(epoch, step, state)

                for obj_target_key, lookup_obs_key, obj_weight in objective_runtime_specs:
                    pred = state.get(obj_target_key)
                    if not isinstance(pred, torch.Tensor):
                        raise KeyError(f"Target state '{obj_target_key}' not available in state manager.")
                    obs = self.loader.get_obs_step(step, lookup_obs_key)
                    if obs is None:
                        continue
                    is_daytime = self._resolve_daytime_mask(
                        pred=pred,
                        forcing=forcing,
                        phase_cfg=phase_cfg,
                    )
                    step_loss = self._compute_weighted_loss(
                        pred=pred,
                        obs=obs,
                        weight=float(obj_weight),
                        obj_target_key=obj_target_key,
                        phase_cfg=phase_cfg,
                        is_daytime=is_daytime,
                    )
                    if step_loss is not None:
                        window_loss = step_loss if window_loss is None else (window_loss + step_loss)

                boundary = ((step + 1) % window_steps == 0) or (step + 1 == total_steps)
                if not boundary:
                    continue

                if window_loss is not None and bool(window_loss.requires_grad):
                    optimizer.zero_grad()
                    window_loss.backward()

                    for key in selected_keys:
                        param = trainable_params[key]
                        if param.grad is None:
                            grad_value = 0.0
                        else:
                            if not bool(torch.isfinite(param.grad).all()):
                                raise RuntimeError(f"Non-finite gradient in phase {phase_name}, param {key}")
                            grad_value = self._safe_float(torch.norm(param.grad.detach()))
                        grad_rows.append(
                            {
                                "phase": phase_name,
                                "epoch": epoch,
                                "window": window_id,
                                "param_key": key,
                                "grad_norm": grad_value,
                            }
                        )

                    torch.nn.utils.clip_grad_norm_(selected_params, max_norm=self.gradient_clip_norm)
                    optimizer.step()

                    with torch.no_grad():
                        for key in selected_keys:
                            lower, upper = param_bounds.get(key, (None, None))
                            if lower is None and upper is None:
                                continue
                            p = trainable_params[key]
                            min_value = -torch.inf if lower is None else torch.tensor(lower, dtype=p.dtype, device=p.device)
                            max_value = torch.inf if upper is None else torch.tensor(upper, dtype=p.dtype, device=p.device)
                            p.copy_(torch.clamp(p, min=min_value, max=max_value))

                    loss_value = self._safe_float(window_loss.detach())
                    epoch_loss_value += loss_value
                    loss_rows.append(
                        {
                            "phase": phase_name,
                            "epoch": epoch,
                            "window": window_id,
                            "step_start": window_start,
                            "step_end": step,
                            "loss": loss_value,
                        }
                    )
                    if self.show_progress and tqdm is not None and hasattr(step_iter, "set_postfix"):
                        step_iter.set_postfix({"window_loss": f"{loss_value:.4f}", "window": window_id})

                # Enforce strict truncated BPTT at each window boundary:
                # detach all persisted state tensors to prevent cross-window graph reuse.
                state_vars = getattr(self.simulator.state_manager, "state_vars", {})
                boundary_keys = list(state_vars.keys())
                for key in detach_keys:
                    if key not in boundary_keys:
                        boundary_keys.append(key)
                self.detach_states(boundary_keys)
                window_loss = None
                window_id += 1
                window_start = step + 1

            if self.show_progress and tqdm is not None and hasattr(step_iter, "close"):
                step_iter.close()

            final_epoch_loss = epoch_loss_value
            last_completed_epoch = epoch

            # Save epoch-level checkpoint for fault-tolerant resume.
            if self.checkpoint_dir is not None:
                ckpt_path = self._save_checkpoint(
                    phase_name=phase_name,
                    epoch=epoch,
                    trainable_params=trainable_params,
                    optimizer=optimizer,
                    loss_rows=loss_rows,
                    grad_rows=grad_rows,
                    epoch_loss=epoch_loss_value,
                    completed_phases=completed_phases,
                )
                if self.show_progress:
                    self._emit_message(f"  [CKPT] saved {ckpt_path.name}")

            if convergence_threshold is not None and previous_epoch_loss is not None:
                if abs(previous_epoch_loss - epoch_loss_value) < convergence_threshold:
                    break
            previous_epoch_loss = epoch_loss_value

            if early_enabled:
                metric = float(epoch_loss_value)
                improved = False
                if best_metric is None:
                    improved = True
                elif early_mode == "min":
                    improved = metric < (best_metric - early_min_delta)
                else:
                    improved = metric > (best_metric + early_min_delta)

                if improved:
                    best_metric = metric
                    best_epoch = epoch
                    epochs_without_improve = 0
                    if early_restore_best:
                        with torch.no_grad():
                            best_param_snapshot = {
                                key: trainable_params[key].detach().clone() for key in selected_keys
                            }
                elif epoch >= early_warmup:
                    epochs_without_improve += 1

                should_stop = epoch >= early_warmup and epochs_without_improve >= early_patience
                if should_stop:
                    if early_restore_best and best_param_snapshot:
                        with torch.no_grad():
                            for key, value in best_param_snapshot.items():
                                trainable_params[key].copy_(value.to(trainable_params[key].device))
                    if best_metric is not None:
                        final_epoch_loss = float(best_metric)
                    if self.show_progress:
                        self._emit_message(
                            f"  [EARLY_STOP] phase={phase_name} epoch={epoch} "
                            f"best_epoch={best_epoch} best_metric={best_metric}"
                        )
                    break

        if early_enabled and early_restore_best and best_param_snapshot:
            with torch.no_grad():
                for key, value in best_param_snapshot.items():
                    trainable_params[key].copy_(value.to(trainable_params[key].device))
            if best_metric is not None:
                final_epoch_loss = float(best_metric)

        if self.checkpoint_dir is not None and last_completed_epoch >= 0:
            finalized_completed_phases = list(completed_phases or [])
            if phase_name not in finalized_completed_phases:
                finalized_completed_phases.append(phase_name)
            self._save_checkpoint(
                phase_name=phase_name,
                epoch=last_completed_epoch,
                trainable_params=trainable_params,
                optimizer=optimizer,
                loss_rows=loss_rows,
                grad_rows=grad_rows,
                epoch_loss=final_epoch_loss,
                completed_phases=finalized_completed_phases,
            )

        return PhaseOutputs(loss_rows=loss_rows, grad_rows=grad_rows, final_epoch_loss=final_epoch_loss)
