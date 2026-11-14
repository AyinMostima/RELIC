from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from relic.computation.operators import apply_non_negative_constraint
from relic.scientific.atmosphere.radiation_energy_conservation import (
    RadiationEnergyConservationValidator,
)
from relic.scientific.biosphere.bvoc.synthesis.sesquiterpene import (
    SesquiterpeneStressModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "analysis_results" / "runtime_scientific_audit_checks.json"


def check_forward_signatures() -> dict:
    validator_sig = str(inspect.signature(RadiationEnergyConservationValidator.forward))
    sesqui_sig = str(inspect.signature(SesquiterpeneStressModel.forward))
    return {
        "radiation_validator_forward_signature": validator_sig,
        "sesquiterpene_stress_forward_signature": sesqui_sig,
    }


def check_simulator_style_call_failure() -> dict:
    validator = RadiationEnergyConservationValidator(config={})
    state = {"shortwave_down": torch.tensor(100.0)}
    error = None
    try:
        # Simulator style: functor(state, time, dt)
        _ = validator(state, 0.0, torch.tensor(1.0))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {"simulator_style_call_error": error}


def check_deposition_sign_clipping() -> dict:
    trial = torch.tensor([-1.0e-5, -2.0e-6, 3.0e-7], dtype=torch.float32)
    constrained = apply_non_negative_constraint(trial, method="zero", small_value=None)
    return {
        "input_rate": trial.tolist(),
        "after_non_negative_constraint": constrained.tolist(),
    }


def main() -> None:
    result = {
        "forward_signatures": check_forward_signatures(),
        "validator_call": check_simulator_style_call_failure(),
        "deposition_sign_test": check_deposition_sign_clipping(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] report={OUT_PATH}")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
