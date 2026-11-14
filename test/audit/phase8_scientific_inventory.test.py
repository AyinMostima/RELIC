import importlib
import inspect
import re
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path

import yaml

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from test.audit import collect_functor_inventory as inventory_collector
from test.audit.collect_functor_inventory import CERTIFICATE_ROOT, PROJECT_ROOT, scan


FORBIDDEN_COMPLETED_RUNTIME_MARKERS = (
    "legacy_mode",
    "use_legacy",
    "enable_legacy",
    "legacy_runtime",
    "fallback_formula",
    "fallback_process",
    "legacy_or_unmigrated",
    "zero_flux_fallback",
    "enable_simplified_mode",
    "_compute_simplified_solution",
    "richards_compute_mode",
    "compute_mode",
    "_compute_flux_core",
    "subprocess_execution_mode",
    "internal_compatibility",
    "compatibility_return_tensor",
    "enable_storage_dynamics",
    "legacy_shared",
)

FAIL_CLOSED_LEVEL_ZERO_PROCESSES = {
    "relic.scientific.anthroposphere.emissions.coupling_to_atmosphere."
    "AtmosphereCoupling": "transport",
    "relic.scientific.anthroposphere.interface_mapping."
    "AnthroposphereInterfaceMapping": "scale_coupling",
    "relic.scientific.anthroposphere.inventory.EmissionInventory": (
        "scale_coupling"
    ),
    "relic.scientific.isotopes.functor.IsotopeHeartbeatFunctor": "scale_coupling",
}


def test_abstract_functor_classes_have_no_process_certificates() -> None:
    """Equivalence certificates belong only to executable process classes."""
    abstract_certificates = []
    for certificate_path in sorted(CERTIFICATE_ROOT.glob("*.yaml")):
        payload = yaml.safe_load(certificate_path.read_text(encoding="utf-8"))
        process_id = str(payload.get("process_id", ""))
        module_name, class_name = process_id.rsplit(".", maxsplit=1)
        process_class = getattr(importlib.import_module(module_name), class_name)
        if inspect.isclass(process_class) and inspect.isabstract(process_class):
            abstract_certificates.append(certificate_path.name)

    assert not abstract_certificates


def test_inventory_dependencies_come_from_the_runtime_declaration_owner():
    """The dependency column is the Functor's own get_dependencies(), not a literal.

    Every Functor row equals the declaration of a minimally constructed
    instance (list form ``a,b``; structured form ``category=a|b;...``), so
    overrides routed through the merge owner, structured declarations and
    instance-level defaults are all visible; unresolved source expressions
    such as ``self.<attribute>`` never appear.
    """
    from relic.core.dependency_graph import normalize_dependency_declaration
    from test.utils.functor_minimal_construction import (
        NOT_MINIMALLY_CONSTRUCTIBLE,
        construct_minimal,
    )

    rows = {
        f"{row['process_id'].rsplit('.', 1)[0]}:{row['class_name']}": row
        for row in scan()
        if row.get("class_name") != "<PARSE_ERROR>"
    }
    expected = {
        "relic.scientific.biosphere.photosynthesis.farquhar:FarquharPhotosynthesis": (
            "environmental_forcing,atmospheric_radiation"
        ),
        "relic.scientific.hydrosphere.evapotranspiration.penman_monteith:PenmanMonteithET": (
            "lagged=bucket_soil_moisture|richards_soil_moisture"
        ),
        "relic.scientific.hydrosphere.runoff.surface_ponding:HortonianPondingEventFunctor": (
            "stochastic_rainfall"
        ),
        "relic.scientific.pedosphere.methane.budget_coupler:MethaneBudgetCoupler": (
            "pedosphere_methane_production,pedosphere_methane_oxidation,"
            "pedosphere_methane_transport"
        ),
    }
    for class_path, text in expected.items():
        assert rows[class_path]["dependencies"] == text, class_path

    for class_path, row in rows.items():
        assert "self." not in row["dependencies"], class_path
        if class_path in NOT_MINIMALLY_CONSTRUCTIBLE:
            assert row["dependencies"] == "", class_path
            continue
        module_name, class_name = class_path.split(":")
        process_class = getattr(importlib.import_module(module_name), class_name)
        if not inspect.isclass(process_class) or not hasattr(process_class, "get_dependencies"):
            assert row["dependencies"] == "", class_path
            continue
        runtime = normalize_dependency_declaration(
            construct_minimal(class_path).get_dependencies(), owner=class_name
        ).canonical()
        names = {name for category in runtime.values() for name in category}
        column = row["dependencies"].replace(";", ",").replace("|", ",")
        listed = {
            item.split("=", 1)[-1] for item in column.split(",") if item
        }
        assert listed == names, class_path


def test_phase8_inventory_marks_exact_fail_closed_level_zero_processes():
    rows = {
        row["process_id"]: row
        for row in scan()
        if row.get("implementation_status") == "not_implemented_fail_closed"
    }

    assert set(rows) == set(FAIL_CLOSED_LEVEL_ZERO_PROCESSES)
    for process_id, row in rows.items():
        assert row["canonical_role"] == FAIL_CLOSED_LEVEL_ZERO_PROCESSES[process_id]
        assert row["process_status"] == "process"
        assert row["iphs_level"] == "0"
        assert row["iphs_level_name"] == "ordinary_empirical"
        assert row["iphs_level_source"] == "certificate"
        assert row["equivalence_certificate_status"] == "invalid"
        assert row["runtime_verified_level"] == "0"
        assert row["runtime_evidence_status"] == "descriptive_only"
        assert row["runtime_replacement_status"] == "not_implemented_fail_closed"
        assert row["completion_status"] == "not_implemented_fail_closed"


def test_phase8_certificate_test_evidence_is_default_collection_reachable(
    pytestconfig,
    tmp_path,
):
    """Verify every certificate test reference exists and pytest can collect it."""
    test_root = (PROJECT_ROOT / "test").resolve()
    python_file_patterns = tuple(pytestconfig.getini("python_files"))
    structural_failures = []
    referenced_targets = []
    referenced_files = set()

    for certificate_path in sorted(CERTIFICATE_ROOT.glob("*.yaml")):
        payload = yaml.safe_load(certificate_path.read_text(encoding="utf-8-sig"))
        for field_name in ("baseline_tests", "validation_tests"):
            targets = payload.get(field_name)
            if not isinstance(targets, list) or not targets:
                structural_failures.append(
                    f"{certificate_path.name}:{field_name}: no test targets"
                )
                continue
            for target in targets:
                if not isinstance(target, str) or not target.strip():
                    structural_failures.append(
                        f"{certificate_path.name}:{field_name}: invalid target {target!r}"
                    )
                    continue
                file_part = target.partition("::")[0]
                target_path = (PROJECT_ROOT / Path(file_part)).resolve()
                try:
                    target_path.relative_to(test_root)
                except ValueError:
                    structural_failures.append(
                        f"{certificate_path.name}:{field_name}:{target}: "
                        "target is outside test"
                    )
                    continue
                if not target_path.is_file():
                    structural_failures.append(
                        f"{certificate_path.name}:{field_name}:{target}: "
                        "target file does not exist"
                    )
                    continue
                if not any(
                    fnmatchcase(target_path.name, pattern)
                    for pattern in python_file_patterns
                ):
                    structural_failures.append(
                        f"{certificate_path.name}:{field_name}:{target}: "
                        "file is not reachable through pytest python_files"
                    )
                    continue
                referenced_targets.append(
                    (certificate_path.name, field_name, target.replace("\\", "/"))
                )
                referenced_files.add(file_part)

    assert not structural_failures, "\n".join(structural_failures)

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "--color=no",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(tmp_path / "certificate_collection"),
        *sorted(referenced_files),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    node_ids = {
        line.strip().replace("\\", "/")
        for line in completed.stdout.splitlines()
        if line.strip().startswith("test") and "::" in line
    }
    unreachable = []
    for certificate_name, field_name, target in referenced_targets:
        file_target, separator, _ = target.partition("::")
        if separator:
            matched = any(
                node_id == target
                or node_id.startswith(f"{target}::")
                or node_id.startswith(f"{target}[")
                for node_id in node_ids
            )
        else:
            matched = any(
                node_id.startswith(f"{file_target}::")
                for node_id in node_ids
            )
        if not matched:
            unreachable.append(f"{certificate_name}:{field_name}:{target}")

    assert not unreachable, "\n".join(unreachable)


def test_phase8_inventory_assigns_roles_and_theory_paths():
    rows = scan()
    process_rows = [row for row in rows if row["process_status"] == "process"]
    implemented_rows = [
        row for row in process_rows if row["implementation_status"] == "implemented"
    ]
    unavailable_rows = [
        row
        for row in process_rows
        if row["implementation_status"] == "not_implemented_fail_closed"
    ]

    assert process_rows
    assert len(implemented_rows) + len(unavailable_rows) == len(process_rows)
    assert all(row["canonical_role"] != "unknown" for row in process_rows)
    assert all(row["theory_document_path"] for row in process_rows)
    assert all(
        row["input_output_baseline_status"] == "captured"
        for row in implemented_rows
    )
    assert all(
        row["input_output_baseline_status"] == "not_applicable_runtime_unbound"
        for row in unavailable_rows
    )


def test_phase8_audit_envelope_binds_publication_source_fingerprint(monkeypatch):
    fingerprint = {
        "schema_version": "1.0.0",
        "git_head": "a" * 40,
        "fingerprint_sha256": "B" * 64,
        "file_count": 12,
        "excluded_prefixes": [".git/"],
        "excluded_files": ["secret.txt"],
    }
    binding = {
        "fingerprint": fingerprint,
        "command": ["-m", "test.audit.collect_functor_inventory"],
        "owner": r"scripts\publication\02_validate_g1_evidence.py",
    }
    monkeypatch.setattr(
        inventory_collector,
        "_build_publication_source_fingerprint",
        lambda: fingerprint,
        raising=False,
    )
    monkeypatch.setattr(
        inventory_collector,
        "_build_publication_provenance_binding",
        lambda: binding,
        raising=False,
    )

    rows = scan()
    envelope = inventory_collector.build_phase8_audit_envelope(rows)

    assert envelope["schema_version"] == "2.0.0"
    assert set(envelope) == {"schema_version", "provenance", "summary", "rows"}
    assert set(envelope["provenance"]) == {
        "generated_at_utc",
        "producer",
        "command",
        "return_code",
        "source_fingerprint_owner",
        "source_fingerprint_schema_version",
        "source_fingerprint_sha256",
        "git_head",
        "source_file_count",
        "excluded_prefixes",
        "excluded_files",
        "dirty_tree_bound",
    }
    assert envelope["provenance"]["producer"] == (
        "test.audit.collect_functor_inventory"
    )
    assert envelope["provenance"]["source_fingerprint_owner"] == (
        "scripts\\publication\\02_validate_g1_evidence.py"
    )
    assert envelope["provenance"]["git_head"] == "a" * 40
    assert envelope["provenance"]["source_fingerprint_sha256"] == "B" * 64
    assert envelope["provenance"]["command"] == [
        "-m",
        "test.audit.collect_functor_inventory",
    ]
    assert type(envelope["provenance"]["return_code"]) is int
    assert envelope["provenance"]["return_code"] == 0
    assert "command_return_code" not in envelope["provenance"]
    assert envelope["summary"]["row_count"] == len(rows)
    assert envelope["summary"]["process_count"] == 92
    assert envelope["summary"]["support_only_count"] == 0
    assert envelope["summary"]["levels"] == {
        "level_0": 4,
        "level_1": 17,
        "level_2": 39,
        "level_3": 32,
        "level_4": 0,
        "level_5": 0,
    }
    assert envelope["rows"] == rows


def test_phase8_inventory_reports_honest_non_strict_classification():
    process_rows = [row for row in scan() if row["process_status"] == "process"]
    implemented_rows = [
        row for row in process_rows if row["implementation_status"] == "implemented"
    ]
    unavailable_rows = [
        row
        for row in process_rows
        if row["implementation_status"] == "not_implemented_fail_closed"
    ]

    assert process_rows
    assert all(row["equivalence_certificate_path"].endswith(".yaml") for row in process_rows)
    assert all(
        row["equivalence_certificate_status"] == "valid"
        for row in implemented_rows
    )
    assert all(
        row["equivalence_certificate_status"] == "invalid"
        for row in unavailable_rows
    )
    assert all(
        row["runtime_replacement_status"] == "canonical_wrapper_declared"
        for row in implemented_rows
    )
    for row in implemented_rows:
        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT / Path(row["equivalence_certificate_path"])
        )
        assert certificate.runtime_replacement_map.get("runtime_uses_iphs") is False
        assert certificate.runtime_replacement_map.get("legacy_removed") is False
    assert all(
        row["composition_contract_status"]
        in {"ports_declared", "composition_map_declared", "metadata_declared"}
        for row in implemented_rows
    )
    assert all(
        row["overflow_routing_status"] == "routing_map_declared"
        for row in implemented_rows
    )
    assert all(row["state_advancement_status"] == "complete" for row in implemented_rows)
    assert all(
        row["state_advancement_specificity_status"] == "specific"
        for row in implemented_rows
    )
    assert all(
        row["source_state_advancement_status"] == "complete"
        for row in implemented_rows
    )
    incomplete = [
        (
            row["process_id"],
            row["state_advancement_specificity_status"],
            row["source_state_advancement_status"],
            row["composition_contract_status"],
            row["overflow_routing_status"],
            row["completion_status"],
        )
        for row in implemented_rows
        if row["completion_status"] != "classified_non_strict"
    ]
    assert not incomplete


def test_phase8_inventory_exports_machine_checkable_iphs_levels():
    process_rows = [row for row in scan() if row["process_status"] == "process"]

    assert process_rows
    for row in process_rows:
        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT / Path(row["equivalence_certificate_path"])
        )
        runtime_audit = certificate.validate()
        assert row["iphs_level"]
        assert row["iphs_level_source"] == "certificate"
        if row["implementation_status"] == "not_implemented_fail_closed":
            assert row["iphs_level"] == "0"
            assert row["iphs_level_status"] == "invalid"
            assert runtime_audit["passed"] is False
        else:
            assert row["iphs_level_status"] == "valid"
            assert int(row["iphs_level"]) >= 1
        assert int(row["runtime_verified_level"]) == int(
            runtime_audit["runtime_verified_level"]
        )
        assert row["runtime_evidence_status"] == runtime_audit["runtime_evidence_status"]
        assert int(row["runtime_evidence_issue_count"]) == int(
            runtime_audit["runtime_evidence_issue_count"]
        )


def test_phase8_structured_levels_have_runtime_structural_hooks():
    process_rows = [row for row in scan() if row["process_status"] == "process"]

    assert process_rows
    dynamic_rows = [row for row in process_rows if row["canonical_role"] != "observation"]
    missing_evidence = [
        (
            row["process_id"],
            row["canonical_role"],
            row["iphs_level"],
            row["iphs_structural_hooks"],
        )
        for row in dynamic_rows
        if int(row["iphs_level"]) >= 3 and not row["iphs_structural_hooks"]
    ]
    assert not missing_evidence


def test_phase8_lower_levels_do_not_retain_strict_iphs_evidence_claims():
    process_rows = [
        row
        for row in scan()
        if row["process_status"] == "process"
    ]
    dynamic_rows = [
        row
        for row in process_rows
        if row["canonical_role"] != "observation"
        and row["implementation_status"] == "implemented"
    ]

    assert dynamic_rows
    failures = []
    strict_keys = {
        "storage_functional",
        "effort_definition",
        "dissipation_audit",
        "port_power_audit",
        "energy_entropy_pair",
        "generic_degeneracy",
        "entropy_production",
    }
    strict_metadata_markers = (
        "e_E = delta E_h / delta x_h",
        "e_S = delta S_h / delta x_h",
        "J_h e_S = 0",
        "M_h e_E = 0",
        "M_h = M_h.T >= 0",
        "sigma_h = e_S.T M_h e_S >= 0",
    )
    for row in process_rows:
        if int(row["iphs_level"]) >= 5:
            continue
        expression = row["metadata_math"] or ""
        retained_markers = [
            marker for marker in strict_metadata_markers if marker in expression
        ]
        if retained_markers:
            failures.append((row["process_id"], retained_markers))

    for row in dynamic_rows:
        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT / Path(row["equivalence_certificate_path"])
        )
        evidence = certificate.iphs_level_evidence
        retained = sorted(strict_keys.intersection(evidence))
        if retained:
            failures.append((row["process_id"], retained))
        expected_limit = "level_3_limit" if int(row["iphs_level"]) == 3 else "level_2_limit"
        if not evidence.get(expected_limit):
            failures.append((row["process_id"], expected_limit))
        runtime_audit = certificate.validate()
        assert row["runtime_evidence_status"] == runtime_audit["runtime_evidence_status"]
        assert int(row["runtime_verified_level"]) == int(
            runtime_audit["runtime_verified_level"]
        )
        assert int(row["runtime_evidence_issue_count"]) == int(
            runtime_audit["runtime_evidence_issue_count"]
        )

    assert not failures


def test_phase8_observation_processes_have_explicit_diagnostic_ceiling():
    observation_rows = [
        row
        for row in scan()
        if row["process_status"] == "process"
        and row["canonical_role"] == "observation"
        and row["implementation_status"] == "implemented"
    ]

    assert observation_rows
    for row in observation_rows:
        assert int(row["iphs_level"]) == 1
        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT / Path(row["equivalence_certificate_path"])
        )
        evidence = certificate.iphs_level_evidence
        assert evidence.get("classification")
        assert evidence.get("compatible_basis")
        assert "diagnostic" in str(evidence).lower() or "observation" in str(evidence).lower()


def test_phase8_inventory_roles_match_certificates():
    process_rows = [row for row in scan() if row["process_status"] == "process"]

    assert process_rows
    mismatches = []
    for row in process_rows:
        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT / Path(row["equivalence_certificate_path"])
        )
        if certificate.process_id != row["process_id"]:
            mismatches.append(
                (
                    row["process_id"],
                    "process_id",
                    certificate.process_id,
                )
            )
        if certificate.canonical_role != row["canonical_role"]:
            mismatches.append(
                (
                    row["process_id"],
                    row["canonical_role"],
                    certificate.canonical_role,
                )
            )

    assert not mismatches


def test_phase8_certificate_interfaces_match_functor_metadata():
    process_rows = [row for row in scan() if row["process_status"] == "process"]

    mismatches = [
        (row["process_id"], row["certificate_interface_consistency"])
        for row in process_rows
        if row["certificate_interface_consistency"] != "matched"
    ]
    for row in process_rows:
        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT / Path(row["equivalence_certificate_path"])
        )
        for field_name, expected in (
            ("legacy_inputs", certificate.input_variables),
            ("iphs_inputs", certificate.input_variables),
            ("legacy_outputs", certificate.output_variables),
            ("iphs_outputs", certificate.output_variables),
        ):
            declared = certificate.interface_map.get(field_name)
            if declared is not None and set(declared) != set(expected):
                mismatches.append(
                    (
                        row["process_id"],
                        field_name,
                        sorted(set(declared) - set(expected)),
                        sorted(set(expected) - set(declared)),
                    )
                )
    assert not mismatches


def test_phase8_certificate_theory_paths_match_inventory_owner():
    process_rows = [row for row in scan() if row["process_status"] == "process"]

    mismatches = [
        (row["process_id"], row["certificate_theory_consistency"])
        for row in process_rows
        if row["certificate_theory_consistency"] != "matched"
    ]
    assert not mismatches


def test_phase8_inventory_classifies_inherited_phenology_models():
    rows = {
        row["class_name"]: row
        for row in scan()
        if row["file_path"].endswith("biosphere\\phenology\\models.py")
    }

    for class_name in ("StaticPhenology", "GDDPhenology", "GSIPhenology", "CropPhenology"):
        assert rows[class_name]["abstraction"] == "BiologicalProcess"
        assert rows[class_name]["canonical_role"] == "reaction"


def test_phase8_inventory_classifies_management_as_ports():
    rows = {
        row["class_name"]: row
        for row in scan()
        if "\\anthroposphere\\management\\" in row["file_path"]
    }

    assert rows["IrrigationManagement"]["canonical_role"] == "port"
    assert rows["HarvestManagement"]["canonical_role"] == "port"
    assert rows["TillageManagement"]["canonical_role"] == "port"


def test_phase8_inventory_classifies_management_reactions_and_coordinator():
    rows = {
        row["class_name"]: row
        for row in scan()
        if "\\anthroposphere\\management\\" in row["file_path"]
    }

    assert rows["FertilizationManagement"]["canonical_role"] == "port"
    assert rows["FireManagement"]["canonical_role"] == "reaction"
    assert rows["GrazingManagement"]["canonical_role"] == "reaction"
    assert rows["ManagementCoordinator"]["canonical_role"] == "scale_coupling"


def test_phase8_inventory_classifies_structural_growth_as_reaction_runtime():
    row = next(
        row
        for row in scan()
        if row["process_id"].endswith("biosphere.growth.structural_growth.StructuralGrowth")
    )

    assert row["canonical_role"] == "reaction"
    assert row["source_state_advancement_status"] == "complete"


def test_phase8_metadata_math_is_canonical_iphs_only():
    process_rows = [row for row in scan() if row["process_status"] == "process"]

    assert process_rows
    for row in process_rows:
        expression = row["metadata_math"]
        if not expression:
            continue
        assert "Canonical IPHS:" in expression
        assert "Base law:" not in expression
        assert "IPHS equivalent:" not in expression
        if row["canonical_role"] == "reaction":
            assert "N_h r_h" in expression
        if row["canonical_role"] == "port":
            assert not re.search(r"N_h\s+r_(?!h\b)[A-Za-z0-9]+", expression)


def test_phase8_certificates_use_standard_reaction_vector_symbol():
    certificate_dir = PROJECT_ROOT / "docs" / "source" / "theory" / "iphs_certificates"
    pattern = re.compile(r"N_h\s+r_(?!h\b)[A-Za-z0-9]+")
    failures = []

    for certificate_path in sorted(certificate_dir.glob("*.yaml")):
        text = certificate_path.read_text(encoding="utf-8-sig")
        if pattern.search(text):
            failures.append(str(certificate_path.relative_to(PROJECT_ROOT)))

    assert not failures


def test_phase8_inventory_applies_state_advancement_lock_to_soil_water_pilot():
    rows = {
        row["process_id"]: row
        for row in scan()
        if row["process_id"].endswith(
            "pedosphere.transport.SoilWaterTransportFunctor.SoilWaterTransportFunctor"
        )
    }
    row = next(iter(rows.values()))

    assert row["canonical_role"] == "transport"
    assert row["runtime_replacement_status"] == "canonical_wrapper_declared"
    assert row["state_advancement_status"]
    if row["state_advancement_status"] == "complete":
        assert row["equivalence_certificate_status"] == "valid"
        assert row["completion_status"] == "classified_non_strict"
    else:
        assert row["completion_status"] == "pending"


def test_phase8_inventory_requires_state_advancement_for_classification_completion():
    process_rows = [row for row in scan() if row["process_status"] == "process"]
    implemented_rows = [
        row for row in process_rows if row["implementation_status"] == "implemented"
    ]

    assert process_rows
    for row in implemented_rows:
        assert row["completion_status"] == "classified_non_strict"
        assert row["state_advancement_status"] == "complete"
        assert row["state_advancement_specificity_status"] == "specific"
        assert row["source_state_advancement_status"] == "complete"


def test_phase8_reaction_modules_use_explicit_reaction_network_runtime():
    reaction_rows = [
        row
        for row in scan()
        if row["process_status"] == "process" and row["canonical_role"] == "reaction"
    ]

    assert reaction_rows
    failures = [
        (row["process_id"], row["source_state_advancement_status"])
        for row in reaction_rows
        if row["source_state_advancement_status"] != "complete"
    ]
    assert not failures


def test_scientific_reaction_network_calls_declare_conservation_or_open_route():
    scientific_root = PROJECT_ROOT / "relic" / "scientific"
    failures = []

    for source_path in sorted(scientific_root.rglob("*.py")):
        source_text = source_path.read_text(encoding="utf-8-sig")
        start = 0
        while True:
            index = source_text.find("advance_reaction_network(", start)
            if index < 0:
                break
            depth = 0
            end = index
            for position in range(index, len(source_text)):
                character = source_text[position]
                if character == "(":
                    depth += 1
                elif character == ")":
                    depth -= 1
                    if depth == 0:
                        end = position + 1
                        break
            call_source = source_text[index:end]
            if (
                "conservation_matrix" not in call_source
                and "open_system_route" not in call_source
            ):
                rel_path = source_path.relative_to(PROJECT_ROOT)
                failures.append(f"{rel_path}:{source_text.count(chr(10), 0, index) + 1}")
            start = max(end, index + 1)

    assert not failures


def test_phase8_strict_completed_modules_have_unique_canonical_runtime():
    complete_rows = [
        row for row in scan() if row["completion_status"] == "strict_complete"
    ]

    for row in complete_rows:
        assert row["runtime_evidence_status"] == "verified"
        assert int(row["runtime_verified_level"]) >= 4
        source_path = PROJECT_ROOT / Path(row["file_path"])
        source_text = source_path.read_text(encoding="utf-8").lower()
        hits = [
            marker
            for marker in FORBIDDEN_COMPLETED_RUNTIME_MARKERS
            if marker in source_text
        ]
        assert not hits, (
            f"{row['process_id']} exposes non-canonical runtime markers: "
            + ", ".join(hits)
        )

        certificate = load_iphs_equivalence_certificate(
            PROJECT_ROOT / Path(row["equivalence_certificate_path"])
        )
        runtime_map = certificate.runtime_replacement_map
        assert runtime_map.get("runtime_uses_iphs") is True
        assert runtime_map.get("legacy_removed") is True
        state_advancement_map = certificate.state_advancement_map
        assert state_advancement_map.get("state_advancer")
        assert state_advancement_map.get("runtime_usage")
        assert row["state_advancement_specificity_status"] == "specific"
        assert row["source_state_advancement_status"] == "complete"


def test_scientific_runtime_has_no_known_parallel_process_entrypoints():
    """Prevent previously removed legacy selectors and second flux runtimes."""
    scientific_root = PROJECT_ROOT / "relic" / "scientific"
    failures = []

    for source_path in sorted(scientific_root.rglob("*.py")):
        source_text = source_path.read_text(encoding="utf-8-sig").lower()
        hits = [
            marker
            for marker in FORBIDDEN_COMPLETED_RUNTIME_MARKERS
            if marker in source_text
        ]
        if hits:
            failures.append(
                f"{source_path.relative_to(PROJECT_ROOT)}: {', '.join(hits)}"
            )

    assert not failures, "\n".join(failures)
