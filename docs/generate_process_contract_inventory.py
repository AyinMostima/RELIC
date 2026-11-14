"""Generate the G00 process contract inventory from RELIC SSOT sources.

The page is a derived working view (not a second classification owner): it
reads the IPHS certificates under ``docs/source/theory/iphs_certificates``
and the live functor inventory scan (``test/audit/collect_functor_inventory``)
and renders, for every scientific process, the computation form actually
used, the applicability of the additional G04-G12 contracts, the source and
theory-document status, the parameter and test premises, and the remaining
gap (which is always owned by one task card of the repair plan). The
certificate scope is read from the live main-certificate path and the
certificate field ``constructor_selection``.

Usage:
    python docs/generate_process_contract_inventory.py           # write
    python docs/generate_process_contract_inventory.py --check   # verify
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_ROOT = REPO_ROOT / "docs" / "source" / "theory" / "iphs_certificates"
OUTPUT_PATH = REPO_ROOT / "docs" / "source" / "theory" / "process_contract_inventory.md"

_PDE_PATTERN = re.compile(r"partial|\\partial|nabla|grad|div\b|\u2202|\\frac\{\\partial", re.IGNORECASE)
_ODE_PATTERN = re.compile(r"\bd\s*\[?[A-Za-z_]+\]?\s*/\s*dt|dC/dt|dB_i/dt|dSWE/dt|dM/d|dot\b", re.IGNORECASE)


def _text(*values: Any) -> str:
    return " ".join(str(value) for value in values if value)


def computation_form(cert: Mapping[str, Any]) -> str:
    """Deterministic working view of the computation form a certificate describes.

    References: the certificate fields ``canonical_role``, ``iphs_level``,
    ``iphs_level_evidence.classification``, ``literature_formula`` and
    ``iphs_level_evidence.canonical_operator`` are the only inputs; the
    rules below are the whole classifier so a reviewer can audit every row.
    """
    role = str(cert.get("canonical_role", ""))
    level = int(cert.get("iphs_level") or 0)
    evidence = cert.get("iphs_level_evidence") or {}
    classification = _text(evidence.get("classification"))
    formula = _text(cert.get("literature_formula"), cert.get("canonical_operator"), evidence.get("canonical_operator"))
    lowered = classification.lower()
    if level >= 4 or lowered.startswith("strict"):
        return "strict IPHS runtime point (claimed Level 4/5)"
    if "dae" in lowered or "descriptor" in lowered:
        return "algebraic constraint (index-1 descriptor DAE)"
    if "ale" in lowered.split() or "moving-domain" in lowered or "moving domain" in lowered:
        return "moving-mesh conservative FVM (ALE)"
    if "continuous-sde" in lowered or "sde" in lowered.split() or "ornstein" in lowered:
        return "continuous stochastic process (SDE)"
    if "marked-jump" in lowered or "poisson" in lowered:
        return "marked jump process (marked Poisson)"
    if "mobile-immobile" in lowered or "memory" in lowered:
        return "history-dependent exchange (memory kernel)"
    if "topology" in lowered or "hybrid" in lowered:
        return "hybrid topology event detector"
    if role == "observation":
        return "algebraic diagnostic (read-only observation)"
    if role == "port":
        return "managed boundary exchange (scheduled event / port)"
    if role == "scale_coupling":
        return "conservative interface mapping (scale coupling)"
    if role == "transport":
        if _PDE_PATTERN.search(formula):
            return "spatial transport PDE (FVM/FDM/DEC discretization)"
        return "storage-discharge / resistance transport ODE"
    if role == "reaction":
        if _ODE_PATTERN.search(formula):
            return "pool-transfer kinetics ODE"
        return "empirical rate law"
    if role == "field":
        if _PDE_PATTERN.search(formula):
            return "field PDE"
        return "algebraic field / forcing law"
    return "unclassified (certificate lacks canonical_role)"


def applicability(cert: Mapping[str, Any], form: str) -> Dict[str, str]:
    """Which additional contracts apply to this process (working view)."""
    level = int(cert.get("iphs_level") or 0)
    role = str(cert.get("canonical_role", ""))
    strict = level >= 4
    return {
        "G04 budget": "applies" if role in {"reaction", "transport", "port", "scale_coupling", "field"} else "n/a (read-only diagnostic)",
        "G05 strict dPH/GENERIC/reaction": "applies (declared)" if strict else "n/a (no thermodynamic claim)",
        "G06 SDE": "applies" if "SDE" in form else "n/a",
        "G07 event/jump": "applies" if ("jump" in form or role == "port" or "event" in form) else "n/a",
        "G08 memory/ALE": "applies" if ("memory" in form or "ALE" in form) else "n/a",
        "G09 DAE/topology": "applies" if ("DAE" in form or "topology" in form) else "n/a",
        "G10 numerics": "applies",
        "G12 MAP/THRESHOLD/EXERGY": (
            "EXERGY candidate" if strict else ("MAP candidate" if role == "scale_coupling" else ("THRESHOLD candidate" if role == "reaction" and "ODE" in form else "n/a"))
        ),
    }


def load_certificates() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(CERTIFICATE_ROOT.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        data["_certificate_file"] = path.name
        rows.append(data)
    return rows


def _ensure_repository_import_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def load_live_inventory() -> List[Dict[str, Any]]:
    _ensure_repository_import_path()
    from test.audit.collect_functor_inventory import scan  # audit owner is the inventory scanner

    return list(scan())


def constructor_selection(cert: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the certificate-owned ``constructor_selection`` field.

    The certificate owner (``IPHSEquivalenceCertificate.from_mapping``) parses the
    field; a malformed value raises there. A non-empty mapping marks a
    constructor-scoped runtime-point certificate; this generator adds no
    file-name or level heuristic of its own.
    """
    _ensure_repository_import_path()
    from relic.computation.operators.iphs_certificate import IPHSEquivalenceCertificate

    return dict(IPHSEquivalenceCertificate.from_mapping(cert).constructor_selection)


def _format_constructor_selection(selection: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in selection.items())


def build_rows(certificates: Sequence[Mapping[str, Any]], live: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    live_by_process = {str(row.get("process_id")): row for row in live}
    rows: List[Dict[str, Any]] = []
    for cert in certificates:
        process_id = str(cert.get("process_id", ""))
        live_row = live_by_process.get(process_id, {})
        main_certificate = str(live_row.get("equivalence_certificate_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
        selection = constructor_selection(cert)
        certificate_file = str(cert["_certificate_file"])
        if main_certificate and certificate_file == main_certificate:
            if selection:
                raise ValueError(
                    f"{certificate_file} is the main process certificate of {process_id} "
                    "and cannot declare constructor_selection"
                )
            certificate_scope = "main_process"
        elif selection:
            if not main_certificate:
                raise ValueError(
                    f"constructor-scoped certificate {certificate_file} has no main process "
                    f"certificate for {process_id} in the live inventory"
                )
            certificate_scope = "runtime_point"
        else:
            certificate_scope = "certificate_only"
        form = computation_form(cert)
        references = list(cert.get("references") or [])
        tests = list(cert.get("validation_tests") or cert.get("baseline_tests") or [])
        parameters = list(cert.get("parameter_keys") or [])
        theory_status = str(cert.get("theory_document_status", ""))
        implementation = str(live_row.get("implementation_status", "certificate only"))
        if certificate_scope == "main_process":
            runtime_evidence = str(live_row.get("runtime_evidence_status", ""))
            state_advancement = str(live_row.get("state_advancement_status", ""))
        else:
            implementation = "constructor-scoped declaration" if certificate_scope == "runtime_point" else "certificate only"
            runtime_evidence = "executed_governed_witness_required"
            state_advancement = "constructor_scoped_witness_required"
        gaps: List[str] = []
        if implementation == "not_implemented_fail_closed":
            gaps.append("G11-PUBLIC: fail-closed public entry")
        if not references:
            gaps.append("G00: no primary source recorded")
        if theory_status != "updated":
            gaps.append("G00: theory page not updated")
        if not tests:
            gaps.append("G10: no validation test recorded")
        if int(cert.get("iphs_level") or 0) == 0 and not (cert.get("iphs_level_evidence") or {}).get("classification"):
            gaps.append("G11-PUBLIC: Level 0 without classification")
        rows.append({
            "process_id": process_id,
            "module": str(cert.get("module_name", "")),
            "certificate": certificate_file,
            "certificate_scope": certificate_scope,
            "constructor_selection": selection,
            "role": str(cert.get("canonical_role", "")),
            "level": int(cert.get("iphs_level") or 0),
            "form": form,
            "applicability": applicability(cert, form),
            "references": len(references),
            "first_reference": references[0] if references else "",
            "theory_document": str(cert.get("theory_document", "")),
            "theory_status": theory_status,
            "parameters": len(parameters),
            "tests": tests,
            "implementation": implementation,
            "runtime_evidence": runtime_evidence,
            "state_advancement": state_advancement,
            "gaps": gaps,
        })
    return rows


def render(rows: Sequence[Mapping[str, Any]], live: Sequence[Mapping[str, Any]]) -> str:
    implemented = [row for row in live if row.get("process_status") == "process" and row.get("implementation_status") == "implemented"]
    fail_closed = [row for row in live if row.get("implementation_status") == "not_implemented_fail_closed"]
    lines: List[str] = []
    lines.append("# Process Contract Inventory (G00 working view)")
    lines.append("")
    lines.append("<!-- Generated by docs/generate_process_contract_inventory.py from the IPHS certificates and the live functor inventory scan. Do not edit by hand. -->")
    lines.append("")
    lines.append("This page is a derived view of existing metadata. The certificates in")
    lines.append("`docs/source/theory/iphs_certificates/` and `test/audit/collect_functor_inventory.py`")
    lines.append("remain the owners; the computation form and applicability columns are computed")
    lines.append("by the auditable rules in the generator and add no second classification registry.")
    lines.append("A process without a strict thermodynamic claim is complete when its own mature")
    lines.append("contract (source, units, support, time, budget, tests) holds; missing IPHS")
    lines.append("objects are not defects for it.")
    lines.append("")
    lines.append("Main-process rows use the live inventory evidence for the exact main-certificate path.")
    lines.append("A certificate is a constructor-scoped runtime point only when it declares the")
    lines.append("certificate field `constructor_selection` (constructor keyword -> JSON scalar);")
    lines.append("each process class keeps exactly one main certificate, which declares no selection.")
    lines.append("Runtime-point certificates carry their own declared Level 4/5 claim and require an")
    lines.append("executed governed witness from the named acceptance tests. Their presence does not")
    lines.append("promote the main process level or inherit its runtime/state-advance evidence.")
    lines.append("This metadata page does not ingest test-run results.")
    lines.append("")
    lines.append(f"- Certificates: {len(rows)}; live implemented processes: {len(implemented)}; fail-closed public entries: {len(fail_closed)}.")
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["form"]] = counts.get(row["form"], 0) + 1
    lines.append("- Computation forms: " + "; ".join(f"{name} = {count}" for name, count in sorted(counts.items())) + ".")
    lines.append("")
    lines.append("## Inventory")
    lines.append("")
    lines.append("| Module | Certificate scope | Role | Declared level | Computation form | Source (first reference) | Theory page | Parameters | Acceptance tests | Runtime evidence | Gap -> task card |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        tests = "<br>".join(str(t).replace("\\", "/") for t in row["tests"]) or "-"
        gaps = "<br>".join(row["gaps"]) or "none"
        theory = f"{row['theory_document'].replace(chr(92), '/')} ({row['theory_status']})" if row["theory_document"] else "-"
        first_ref = str(row["first_reference"]).replace("|", "/")
        scope = row["certificate_scope"]
        if row["constructor_selection"]:
            scope = f"{scope} (`{_format_constructor_selection(row['constructor_selection'])}`)"
        lines.append(
            f"| `{row['module']}` (`{row['certificate']}`) | {scope} | {row['role']} | {row['level']} | {row['form']} | "
            f"{first_ref} (+{max(0, row['references'] - 1)}) | {theory} | {row['parameters']} | {tests} | "
            f"{row['runtime_evidence'] or '-'} / {row['state_advancement'] or '-'} | {gaps} |"
        )
    lines.append("")
    lines.append("## Applicability of additional contracts")
    lines.append("")
    keys = list(rows[0]["applicability"].keys()) if rows else []
    lines.append("| Module | " + " | ".join(keys) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in keys) + " |")
    for row in rows:
        lines.append(f"| `{row['module']}` | " + " | ".join(row["applicability"][key] for key in keys) + " |")
    lines.append("")
    lines.append("## Remaining gaps by task card")
    lines.append("")
    by_card: Dict[str, List[str]] = {}
    for row in rows:
        for gap in row["gaps"]:
            card = gap.split(":")[0]
            by_card.setdefault(card, []).append(f"`{row['module']}`: {gap.split(': ', 1)[1]}")
    if not by_card:
        lines.append("- none")
    for card in sorted(by_card):
        lines.append(f"- {card}: " + "; ".join(by_card[card]))
    lines.append("")
    return "\n".join(lines) + "\n"


def generate() -> str:
    certificates = load_certificates()
    live = load_live_inventory()
    return render(build_rows(certificates, live), live)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="Verify the generated page is current.")
    args = parser.parse_args()
    content = generate()
    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        if current.replace("\r\n", "\n") != content:
            print(f"[STALE] {OUTPUT_PATH} differs from the generated content")
            return 1
        print(f"[OK] {OUTPUT_PATH} is current")
        return 0
    OUTPUT_PATH.write_text(content, encoding="utf-8", newline="\n")
    print(f"[OK] wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
