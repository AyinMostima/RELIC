"""Shared fixtures for minimal reproducible tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        return {}
    return data


@pytest.fixture
def dt_limit_cfg() -> Dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "dt_limit_cfg.yaml"
    return _load_yaml(path)


@pytest.fixture
def update_mode_cfg() -> Dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "update_mode_cfg.yaml"
    return _load_yaml(path)
