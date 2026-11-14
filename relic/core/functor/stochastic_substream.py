"""Functor-owned random-number substream with replayable extra state.

References:
    - PyTorch ``torch.nn.Module.get_extra_state``/``set_extra_state`` contract.
    - Revised RELIC foundations: accepted-step transaction snapshot of Functor
      ``state_dict`` (including extra state) and checkpoint restore-and-replay.
    - Kloeden and Platen (1992), ch. 10; Gillespie (1977): each accepted step
      consumes a deterministic number of draws from one private stream.

A stochastic Functor owns exactly one ``torch.Generator``. Its byte state and
the number of emitted steps are part of the module extra state, so the
simulator's accepted-step snapshot rolls a rejected step back to the same
stream position and a durable checkpoint restores the exact continuation.
Global Python/NumPy/Torch RNG state is never touched by this mixin; that
state stays with ``relic.core.simulator.checkpoint``.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor

Device = Union[str, torch.device]
_PATH_TIME_RTOL = 1e-9


class FunctorSubstreamMixin:
    """Own a seeded ``torch.Generator`` and expose its position as extra state.

    Subclasses declare ``EXTRA_STATE_FORMAT`` (a unique format tag) and
    ``SUBSTREAM_LABEL`` (used in error messages), call
    ``_initialize_substream(seed)`` after ``FunctorBase.__init__`` has set
    ``self.device``, and increment ``self.draw_count`` once per emitted step.
    The mixin must precede the bridge class in the MRO so that ``to_device``
    and the extra-state hooks override the ``FunctorBase`` defaults.

    Mathematical expression:
        A stream is the sequence (u_1, u_2, ...) of the generator; the pair
        (generator_state, draw_count) identifies the pending index so that a
        restored Functor emits the same u_{n+1} as the uninterrupted run.
    Parameter meanings:
        seed: non-negative integer below 2**63 that initializes the stream.
        draw_count: number of emitted steps since initialization.
    Units: dimensionless.
    Domain/constraints:
        The stream can change device type only before the first draw; a
        cross-device migration after draws is [N] and rejected.
    Sign convention: not applicable.
    """

    EXTRA_STATE_FORMAT: str = "relic.functor.substream"
    EXTRA_STATE_VERSION: int = 2
    SUBSTREAM_LABEL: str = "Stochastic Functor"

    seed: int
    draw_count: int
    generator: torch.Generator
    wiener_pending: List[Dict[str, Any]]
    _trial_draw_spec: Optional[Tuple[Tuple[int, ...], torch.dtype, torch.device, float, float]]

    # ------------------------------------------------------------------
    # Seed and stream construction
    # ------------------------------------------------------------------
    @classmethod
    def validate_substream_seed(cls, seed: Any) -> int:
        """Return the validated integer seed or fail closed."""
        if seed is None:
            raise ValueError(
                f"{cls.SUBSTREAM_LABEL} config requires an integer 'seed' for the "
                "Functor-owned random substream."
            )
        if (
            isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or int(seed) < 0
            or int(seed) >= 2**63
        ):
            raise ValueError(
                f"{cls.SUBSTREAM_LABEL} seed must be a non-negative integer below 2**63."
            )
        return int(seed)

    def _initialize_substream(self, seed: Any) -> None:
        self.seed = self.validate_substream_seed(seed)
        self.draw_count = 0
        self.generator = self._new_generator(getattr(self, "device", "cpu"))
        self.wiener_pending = []
        self._trial_draw_spec = None

    # ------------------------------------------------------------------
    # Brownian tree: path-consistent Wiener increments under step changes
    # ------------------------------------------------------------------
    @staticmethod
    def _same_time(left: float, right: float, scale: float) -> bool:
        return abs(left - right) <= _PATH_TIME_RTOL * max(abs(scale), 1.0)

    def draw_wiener_increment(self, time: float, width: float, *, like: Tensor) -> Tensor:
        """Draw Delta W over [time, time + width] consistently with the path.

        References:
            Gaines and Lyons (1997), Variable step size control in the
            numerical solution of SDEs, SIAM J. Appl. Math. 57 (Brownian
            tree); Karlin and Taylor (1981), A Second Course in Stochastic
            Processes, ch. 9 (Brownian bridge); Kloeden and Platen (1992)
            ch. 10.
        Mathematical expression:
            fresh:  Delta W(h) = sqrt(h) xi;
            split of a pending Delta W(H), h < H:
                Delta W(h) = (h/H) Delta W(H) + sqrt(h (H - h) / H) xi,
                Delta W(H - h) = Delta W(H) - Delta W(h) stays pending;
            merge of a pending Delta W(H), H < h:
                Delta W(h) = Delta W(H) + Delta W'(h - H) recursively.
            xi ~ N(0, I) i.i.d. from the Functor-owned substream, so
            E[Delta W(h) | Delta W(H)] = (h/H) Delta W(H) and
            Var[Delta W(h) | Delta W(H)] = h (H - h) / H.
        Parameter meanings:
            time is the interval start, width the requested width h, H the
            width of a pending increment starting at time, like the tensor
            whose shape, dtype, and device the increment takes.
        Units:
            time and width in the caller's time unit; Delta W in the square
            root of that unit.
        Domain/constraints:
            width finite and positive; a pending increment is consumed only
            when its start coincides with time (relative tolerance 1e-9);
            a pending increment overlapping the request without sharing its
            start is a broken path and fails closed.
        Sign convention:
            not applicable.
        """
        width_value = float(width)
        if not math.isfinite(width_value) or width_value <= 0.0:
            raise ValueError(f"{self.SUBSTREAM_LABEL} Wiener increment width must be finite and positive.")
        if not isinstance(like, Tensor):
            raise TypeError("like must be a torch.Tensor")
        self._trial_draw_spec = (
            tuple(like.shape), like.dtype, like.device, float(time), width_value,
        )
        return self._draw_on_path(float(time), width_value, like)

    def _pending_index_at(self, time: float, width: float) -> Optional[int]:
        for index, entry in enumerate(self.wiener_pending):
            start = float(entry["time"])
            if self._same_time(start, time, width):
                return index
            end = start + float(entry["width"])
            overlaps = start < time + width and time < end
            if overlaps and not self._same_time(start, time, width):
                raise RuntimeError(
                    f"{self.SUBSTREAM_LABEL} Wiener path is inconsistent: a pending "
                    f"increment on [{start}, {end}] overlaps the request at {time} "
                    f"with width {width} without sharing its start."
                )
        return None

    def _fresh_normal(self, like: Tensor) -> Tensor:
        return torch.empty(like.shape, dtype=like.dtype, device=like.device).normal_(
            generator=self.generator,
        )

    def _draw_on_path(self, time: float, width: float, like: Tensor) -> Tensor:
        index = self._pending_index_at(time, width)
        if index is None:
            return self._fresh_normal(like) * math.sqrt(width)
        entry = self.wiener_pending.pop(index)
        pending_width = float(entry["width"])
        pending = entry["increment"].to(dtype=like.dtype, device=like.device)
        if tuple(pending.shape) != tuple(like.shape):
            raise RuntimeError(
                f"{self.SUBSTREAM_LABEL} pending Wiener increment shape {tuple(pending.shape)} "
                f"does not match the state shape {tuple(like.shape)}."
            )
        if self._same_time(pending_width, width, width):
            return pending
        if pending_width > width:
            fraction = width / pending_width
            conditional_std = math.sqrt(width * (pending_width - width) / pending_width)
            partial = fraction * pending + conditional_std * self._fresh_normal(like)
            self.wiener_pending.append({
                "time": time + width,
                "width": pending_width - width,
                "increment": (pending - partial).detach().clone(),
            })
            return partial
        return pending + self._draw_on_path(time + pending_width, width - pending_width, like)

    def record_rejected_trial_path(self) -> None:
        """Pin the rejected trial's increment on the path after a rewind.

        The simulator restores this Functor's extra state (generator position
        and pending ledger) to the pre-trial snapshot before calling this
        hook. Replaying the trial draw from that position reproduces the same
        Delta W(h); pinning it as pending lets the shorter retry refine it by
        the bridge instead of redrawing a different path.
        """
        spec = self._trial_draw_spec
        if spec is None:
            return
        shape, dtype, device, time, width = spec
        like = torch.empty(shape, dtype=dtype, device=device)
        increment = self._draw_on_path(time, width, like)
        self.wiener_pending.append({
            "time": time, "width": width, "increment": increment.detach().clone(),
        })
        self._trial_draw_spec = None

    def _new_generator(self, device: Device) -> torch.Generator:
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(self.seed)
        return generator

    def _require_substream_device(self, tensor: Tensor) -> None:
        """Reject advancing a state whose device type differs from the stream."""
        if torch.device(self.generator.device).type != tensor.device.type:
            raise ValueError(
                f"The {self.SUBSTREAM_LABEL} substream lives on the Functor device; "
                "migrate the Functor before advancing a state on another device."
            )

    # ------------------------------------------------------------------
    # Device migration
    # ------------------------------------------------------------------
    def to_device(self, device: Device) -> Any:
        """Migrate the Functor; the substream can only move before any draw."""
        target = torch.device(device)
        if torch.device(self.generator.device).type != target.type:
            if self.draw_count != 0:
                raise ValueError(
                    f"{self.SUBSTREAM_LABEL} substream migration across device types "
                    "after draws is [N]; checkpoint the run and restart on the "
                    "target device instead."
                )
            super().to_device(target)  # type: ignore[misc]
            self.generator = self._new_generator(target)
            return self
        super().to_device(target)  # type: ignore[misc]
        return self

    # ------------------------------------------------------------------
    # Extra state (rollback and restart)
    # ------------------------------------------------------------------
    def get_extra_state(self) -> Dict[str, Any]:
        """Return the replay-critical stream position for rollback and restart."""
        return {
            "format": self.EXTRA_STATE_FORMAT,
            "version": self.EXTRA_STATE_VERSION,
            "seed": self.seed,
            "draw_count": self.draw_count,
            "generator_state": self.generator.get_state().clone(),
            "wiener_pending": [
                {
                    "time": float(entry["time"]),
                    "width": float(entry["width"]),
                    "increment": entry["increment"].detach().cpu().clone(),
                }
                for entry in self.wiener_pending
            ],
        }

    def set_extra_state(self, state: Any) -> None:
        label = self.SUBSTREAM_LABEL
        if not isinstance(state, Mapping):
            raise TypeError(f"{label} extra state must be a mapping.")
        expected = {"format", "version", "seed", "draw_count", "generator_state", "wiener_pending"}
        if set(state) != expected:
            raise ValueError(f"{label} extra-state keys do not match the schema.")
        if state["format"] != self.EXTRA_STATE_FORMAT:
            raise ValueError(f"{label} extra-state format is unsupported.")
        if state["version"] != self.EXTRA_STATE_VERSION:
            raise ValueError(f"{label} extra-state version is unsupported.")
        seed = state["seed"]
        draw_count = state["draw_count"]
        generator_state = state["generator_state"]
        if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
            raise ValueError(f"{label} extra-state seed must be a non-negative integer.")
        if (
            isinstance(draw_count, bool)
            or not isinstance(draw_count, Integral)
            or int(draw_count) < 0
        ):
            raise ValueError(f"{label} extra-state draw_count must be a non-negative integer.")
        if not isinstance(generator_state, Tensor) or generator_state.dtype != torch.uint8:
            raise TypeError(f"{label} generator_state must be a uint8 tensor.")
        pending_raw = state["wiener_pending"]
        if not isinstance(pending_raw, list):
            raise TypeError(f"{label} wiener_pending must be a list.")
        pending: List[Dict[str, Any]] = []
        for entry in pending_raw:
            if not isinstance(entry, Mapping) or set(entry) != {"time", "width", "increment"}:
                raise ValueError(f"{label} wiener_pending entries must carry time, width, increment.")
            entry_width = entry["width"]
            entry_time = entry["time"]
            if (
                isinstance(entry_width, bool) or not isinstance(entry_width, (int, float))
                or not math.isfinite(float(entry_width)) or float(entry_width) <= 0.0
            ):
                raise ValueError(f"{label} wiener_pending width must be finite and positive.")
            if isinstance(entry_time, bool) or not isinstance(entry_time, (int, float)) or not math.isfinite(float(entry_time)):
                raise ValueError(f"{label} wiener_pending time must be finite.")
            if not isinstance(entry["increment"], Tensor):
                raise TypeError(f"{label} wiener_pending increment must be a tensor.")
            pending.append({
                "time": float(entry_time), "width": float(entry_width),
                "increment": entry["increment"].detach().clone(),
            })
        self.seed = int(seed)
        self.generator.set_state(generator_state.detach().cpu().clone())
        self.draw_count = int(draw_count)
        self.wiener_pending = pending


__all__ = ["FunctorSubstreamMixin"]
