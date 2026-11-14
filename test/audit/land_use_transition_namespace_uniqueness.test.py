"""Audit gate for land-use transition namespace uniqueness."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LAND_USE_YAML = REPO_ROOT / "relic" / "parameters" / "database" / "relic" / "anthroposphere" / "land_use.yaml"


def test_land_use_transition_prefix_is_unique() -> None:
    with LAND_USE_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "land_use.yaml must parse to dict"

    params = data.get("parameters")
    assert isinstance(params, dict), "land_use.yaml missing parameters mapping"
    keys = set(params.keys())

    forbidden = sorted(k for k in keys if k.startswith("anthroposphere.land_use.transition."))
    assert not forbidden, (
        "Found deprecated duplicate transition.* keys. Only transition_rates.* is allowed:\n"
        + "\n".join(forbidden)
    )

    required = {
        "anthroposphere.land_use.transition_rates.deforestation_rate",
        "anthroposphere.land_use.transition_rates.afforestation_rate",
        "anthroposphere.land_use.transition_rates.urbanization_rate",
        "anthroposphere.land_use.transition_rates.agricultural_expansion_rate",
        "anthroposphere.land_use.transition_rates.abandonment_rate",
    }
    missing = sorted(required - keys)
    assert not missing, "Missing transition_rates keys:\n" + "\n".join(missing)
