"""
Governance check for intentional evapotranspiration overlap.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from relic.core.functor import ConflictResolutionPolicy
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
    PenmanMonteithET,
)
from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import (
    PriestleyTaylorET,
)


_BRIDGES = {
    "PhysicalFieldFunctor",
    "BiologicalProcessFunctor",
    "TransportMechanismFunctor",
    "ScaleCouplingFunctor",
}

_CONTRACT_ROOT = Path("relic/contracts/variables/hydrosphere")


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _collect_output_writers(variable_name: str) -> set[tuple[str, str]]:
    writers: set[tuple[str, str]] = set()
    roots = [Path("relic/scientific"), Path("relic/coupling")]
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue

            for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
                bases = {_base_name(b) for b in cls.bases}
                if not (bases & _BRIDGES):
                    continue
                for node in ast.walk(cls):
                    if not isinstance(node, ast.Call):
                        continue
                    fn = node.func.id if isinstance(node.func, ast.Name) else (
                        node.func.attr if isinstance(node.func, ast.Attribute) else None
                    )
                    if fn != "FunctorMetadata":
                        continue
                    for kw in node.keywords:
                        if kw.arg != "output_variables" or not isinstance(kw.value, ast.List):
                            continue
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Constant) and elt.value == variable_name:
                                writers.add((str(path).replace("\\", "/"), cls.name))
                    break
    return writers


def _load_contract(filename: str) -> dict:
    return yaml.safe_load((_CONTRACT_ROOT / filename).read_text(encoding="utf-8"))


def _writer_names(contract: dict) -> set[str]:
    return {
        writer["name"]
        for writer in contract["writers"]["tendencies"]
    }


def test_evapotranspiration_overlap_is_explicitly_governed() -> None:
    """
    References:
    - docs/source/architecture_overview.md (overlap governance and conflict policy).
    - FAO-56 (Allen et al., 1998), ET formulation families.

    Mathematical expression:
    - ET = f_pm(...) or ET = f_pt(...), with exclusive model selection.

    Parameter meanings:
    - conflict_resolution_policy=EXCLUSIVE means ET alternatives must not be merged by accumulation.
    """
    writers = _collect_output_writers("evapotranspiration_flux")
    expected = {
        (
            "relic/scientific/hydrosphere/evapotranspiration/penman_monteith.py",
            "PenmanMonteithET",
        ),
        (
            "relic/scientific/hydrosphere/evapotranspiration/priestley_taylor.py",
            "PriestleyTaylorET",
        ),
    }
    assert writers == expected

    pm = PenmanMonteithET(config={"canopy_resistance_source": "state"})
    pt = PriestleyTaylorET()
    assert pm.metadata.conflict_resolution_policy == ConflictResolutionPolicy.EXCLUSIVE
    assert pt.metadata.conflict_resolution_policy == ConflictResolutionPolicy.EXCLUSIVE


def test_evapotranspiration_flux_contracts_share_the_exclusive_writer_set() -> None:
    """The RELIC ET models are the runtime writers; the external adapters that map a field to the
    variable are its alternative writers (adapter_writers). Registration admits one writer."""
    expected_writers = {"penman_monteith_et", "priestley_taylor_et"}

    for filename, variable, unit, adapters in (
        ("evapotranspiration_flux.yaml", "evapotranspiration_flux", "kg m-2 s-1", {"VICAdapter", "SWATAdapter"}),
        ("latent_heat_flux.yaml", "latent_heat_flux", "W m-2", {"WRFAdapter", "CESMAdapter", "CMIP6Adapter"}),
    ):
        contract = _load_contract(filename)
        assert contract["variable"] == variable
        assert contract["kind"] == "flux"
        assert contract["dimension"]["udunits"] == unit
        assert contract["owner"] is None
        assert contract["incrementable"] is False
        assert _writer_names(contract) == expected_writers | adapters
        assert set(contract["flux_contract"]["allowed_in"]) == expected_writers | adapters
        assert set(contract["provenance"]["runtime_writers"]) == expected_writers
        assert set(contract["provenance"]["adapter_writers"]) == adapters
        assert contract["provenance"]["runtime_role"].startswith(
            "mutually_exclusive_"
        )

    diagnostics = _load_contract("diagnostics_evapotranspiration.yaml")
    assert diagnostics["owner"] is None
    assert _writer_names(diagnostics) == expected_writers
    assert set(diagnostics["provenance"]["runtime_writers"]) == expected_writers
