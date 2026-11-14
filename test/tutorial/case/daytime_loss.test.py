from __future__ import annotations

import torch

from tutorial.case.utils.optimization_loop import OptimizationLoopEngine


def _build_engine() -> OptimizationLoopEngine:
    return OptimizationLoopEngine(
        simulator=object(),
        loader=object(),
        dt_seconds=3600.0,
        gradient_clip_norm=1.0,
        show_progress=False,
        checkpoint_dir=None,
    )


def _daytime_phase_cfg() -> dict[str, object]:
    return {
        "daytime_loss": {
            "enabled": True,
            "par_key": "ppfd",
            "par_threshold": 10.0,
            "target_state_keys": ["isoprene_emission"],
            "positive_weight": 3.0,
            "underprediction_penalty_weight": 2.0,
        }
    }


def test_compute_daytime_penalty_matches_expected_asymmetric_form() -> None:
    engine = _build_engine()
    pred = torch.tensor([0.0, 0.0, 10.0, 10.0], dtype=torch.float32)
    obs = torch.tensor([0.0, 0.0, 20.0, 20.0], dtype=torch.float32)
    is_daytime = torch.tensor([False, False, True, True])
    valid_mask = torch.isfinite(obs)

    penalty = engine._compute_daytime_penalty(
        pred_valid=pred,
        obs_valid=obs,
        is_daytime=is_daytime,
        valid_mask=valid_mask,
        obj_target_key="isoprene_emission",
        phase_cfg=_daytime_phase_cfg(),
    )

    assert penalty is not None
    assert torch.isclose(penalty, torch.tensor(500.0, dtype=torch.float32))


def test_compute_weighted_loss_adds_daytime_penalty_on_top_of_base_loss() -> None:
    engine = _build_engine()
    pred = torch.tensor([0.0, 0.0, 10.0, 10.0], dtype=torch.float32)
    obs = torch.tensor([0.0, 0.0, 20.0, 20.0], dtype=torch.float32)
    is_daytime = torch.tensor([False, False, True, True])

    total_loss = engine._compute_weighted_loss(
        pred=pred,
        obs=obs,
        weight=1.0,
        obj_target_key="isoprene_emission",
        phase_cfg=_daytime_phase_cfg(),
        is_daytime=is_daytime,
    )

    assert total_loss is not None
    assert torch.isclose(total_loss, torch.tensor(550.0, dtype=torch.float32))


def test_resolve_daytime_mask_uses_forcing_par_threshold_and_broadcasts() -> None:
    engine = _build_engine()
    pred = torch.tensor([1.0, 2.0], dtype=torch.float32)

    day_mask = engine._resolve_daytime_mask(
        pred=pred,
        forcing={"ppfd": torch.tensor(50.0)},
        phase_cfg=_daytime_phase_cfg(),
    )
    night_mask = engine._resolve_daytime_mask(
        pred=pred,
        forcing={"ppfd": torch.tensor(0.0)},
        phase_cfg=_daytime_phase_cfg(),
    )

    assert day_mask is not None
    assert night_mask is not None
    assert torch.equal(day_mask, torch.tensor([True, True]))
    assert torch.equal(night_mask, torch.tensor([False, False]))
