"""
Unit tests for the conservation tolerance source and soil water conversions.
"""

import importlib.util
from pathlib import Path

import torch

import relic
from relic.coupling.scales.soil_water_conversions import (
    matric_potential_to_volumetric,
    volumetric_to_matric_potential,
)
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace


def test_no_conservation_config_wrapper_or_second_tolerance_resolver_remains():
    """The one-key wrapper relic.coupling.scales.conservation_config is deleted: StateManager
    reads validation.conservation.conservation_tolerance itself (see
    test_state_manager_conservation.TestConservationToleranceOwner), and no relic module
    names the former resolvers."""
    assert importlib.util.find_spec("relic.coupling.scales.conservation_config") is None
    relic_root = Path(relic.__file__).parent
    offenders = sorted(
        f"{path.relative_to(relic_root)}:{name}"
        for path in relic_root.rglob("*.py")
        for name in ("get_conservation_tolerance", "get_conservation_config", "ConservationConfig")
        if name in path.read_text(encoding="utf-8", errors="replace")
    )
    assert offenders == []


def test_resolve_namespace_handles_enum():
    resolved = resolve_namespace(ParameterNamespace.COUPLING_WATER_CARBON)
    assert resolved == "coupling.scales.water_carbon"


def test_van_genuchten_roundtrip_loam():
    psi_original = torch.tensor([-33000.0])  # Pa
    theta = matric_potential_to_volumetric(psi_original, theta_sat=0.43, alpha=3.6e-6, n=1.56)
    psi_recovered = volumetric_to_matric_potential(theta, theta_sat=0.43, alpha=3.6e-6, n=1.56)
    assert torch.allclose(psi_original, psi_recovered, rtol=1e-3, atol=1e-2)

