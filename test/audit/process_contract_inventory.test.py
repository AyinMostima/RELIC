"""G00 process contract inventory: generator is current, complete, and auditable."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "docs" / "generate_process_contract_inventory.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("relic_process_contract_inventory", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_page_is_current():
    generator = _load_generator()
    current = generator.OUTPUT_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert current == generator.generate()


def test_every_implemented_process_has_exactly_one_non_strict_certificate_row():
    generator = _load_generator()
    certificates = generator.load_certificates()
    live = generator.load_live_inventory()
    rows = generator.build_rows(certificates, live)
    implemented = {
        str(row["process_id"])
        for row in live
        if row.get("process_status") == "process" and row.get("implementation_status") == "implemented"
    }
    covered = {row["process_id"] for row in rows}
    assert implemented <= covered, sorted(implemented - covered)
    # A strict runtime point is an additional certificate for a process that
    # also keeps its ordinary certificate; nothing else is duplicated.
    duplicates = {}
    for row in rows:
        duplicates.setdefault(row["process_id"], []).append(row["form"])
    for process_id, forms in duplicates.items():
        if len(forms) > 1:
            assert sum("strict" in form for form in forms) == len(forms) - 1, (process_id, forms)


def test_classifier_is_deterministic_and_only_strict_certificates_claim_level_4_or_5():
    generator = _load_generator()
    certificates = generator.load_certificates()
    for cert in certificates:
        form = generator.computation_form(cert)
        assert form == generator.computation_form(cert)
        level = int(cert.get("iphs_level") or 0)
        if level >= 4:
            assert form.startswith("strict IPHS runtime point")
            assert generator.applicability(cert, form)["G05 strict dPH/GENERIC/reaction"] == "applies (declared)"
        else:
            assert not form.startswith("strict IPHS runtime point")
            assert generator.applicability(cert, form)["G05 strict dPH/GENERIC/reaction"].startswith("n/a")


def test_known_special_processes_are_classified_by_their_actual_form():
    generator = _load_generator()
    by_module = {}
    for cert in generator.load_certificates():
        by_module.setdefault(cert.get("module_name"), []).append(generator.computation_form(cert))
    assert "algebraic constraint (index-1 descriptor DAE)" in by_module["EquilibriumSorptionFunctor"]
    assert "moving-mesh conservative FVM (ALE)" in by_module["MovingDomainDiffusionFunctor"]
    assert "continuous stochastic process (SDE)" in by_module["OrnsteinUhlenbeckTemperatureFunctor"]
    assert "marked jump process (marked Poisson)" in by_module["MarkedPoissonRainfallFunctor"]
    assert "hybrid topology event detector" in by_module["HortonianPondingEventFunctor"]
    assert "history-dependent exchange (memory kernel)" in by_module["TwoRegionMassTransferFunctor"]
    assert "spatial transport PDE (FVM/FDM/DEC discretization)" in by_module["SoluteAdvectionDiffusion1D"]
    assert "pool-transfer kinetics ODE" in by_module["FirstOrderDecomposition"]
    assert "algebraic diagnostic (read-only observation)" in by_module["WaterStress"]


def test_additional_runtime_point_does_not_inherit_main_certificate_evidence():
    generator = _load_generator()
    common = {"process_id": "transport.fixture", "module_name": "FixtureTransport",
              "canonical_role": "transport", "references": ["A bounded fixture"],
              "validation_tests": ["test/transport/test_diffusion_l1.py"],
              "theory_document_status": "updated"}
    certificates = [dict(common, _certificate_file="fixture.yaml", iphs_level=3),
                    dict(common, _certificate_file="fixture_point.yaml", iphs_level=4,
                         constructor_selection={"strict_runtime_point": True}),
                    dict(common, _certificate_file="fixture_strict_unscoped.yaml", iphs_level=4)]
    live = [{"process_id": "transport.fixture",
             "equivalence_certificate_path": "docs/source/theory/iphs_certificates/fixture.yaml",
             "implementation_status": "implemented", "runtime_evidence_status": "descriptive_only",
             "state_advancement_status": "complete"}]
    main, point, unscoped = generator.build_rows(certificates, live)
    assert main["certificate_scope"] == "main_process"
    assert main["constructor_selection"] == {}
    assert main["level"] == 3
    assert main["runtime_evidence"] == "descriptive_only"
    assert main["state_advancement"] == "complete"
    assert point["certificate_scope"] == "runtime_point"
    assert point["constructor_selection"] == {"strict_runtime_point": True}
    assert point["runtime_evidence"] == "executed_governed_witness_required"
    assert point["state_advancement"] == "constructor_scoped_witness_required"
    assert point["implementation"] == "constructor-scoped declaration"
    # The certificate field, not a level number or a file name, marks a runtime point.
    assert unscoped["certificate_scope"] == "certificate_only"


def test_constructor_selection_scope_is_fail_closed():
    import pytest

    generator = _load_generator()
    common = {"process_id": "transport.fixture", "module_name": "FixtureTransport",
              "canonical_role": "transport", "references": ["A bounded fixture"],
              "validation_tests": ["test/transport/test_diffusion_l1.py"],
              "theory_document_status": "updated"}
    live = [{"process_id": "transport.fixture",
             "equivalence_certificate_path": "docs/source/theory/iphs_certificates/fixture.yaml",
             "implementation_status": "implemented", "runtime_evidence_status": "descriptive_only",
             "state_advancement_status": "complete"}]
    scoped_main = [dict(common, _certificate_file="fixture.yaml", iphs_level=4,
                        constructor_selection={"strict_runtime_point": True})]
    with pytest.raises(ValueError, match="main process certificate"):
        generator.build_rows(scoped_main, live)
    orphan = [dict(common, process_id="transport.orphan", _certificate_file="orphan_point.yaml",
                   iphs_level=4, constructor_selection={"strict_runtime_point": True})]
    with pytest.raises(ValueError, match="no main process certificate"):
        generator.build_rows(orphan, live)
    malformed = [dict(common, _certificate_file="fixture_point.yaml", iphs_level=4,
                      constructor_selection="strict_runtime_point=true")]
    with pytest.raises(ValueError, match="constructor_selection must be a mapping"):
        generator.build_rows(malformed, live)


def test_actual_runtime_point_rows_remain_separate_from_main_process_levels():
    generator = _load_generator()
    rows = generator.build_rows(generator.load_certificates(), generator.load_live_inventory())
    point_rows = [row for row in rows if row["certificate_scope"] == "runtime_point"]
    assert {row["certificate"] for row in point_rows} == {
        "atmosphere_chemistry_chemistry_functor_closed_constant_volume.yaml",
        "transport_diffusion_fickian_diffusion_strict_runtime_point.yaml",
    }
    selections = {row["certificate"]: row["constructor_selection"] for row in point_rows}
    assert selections == {
        "atmosphere_chemistry_chemistry_functor_closed_constant_volume.yaml":
            {"physical_boundary": "closed_constant_volume"},
        "transport_diffusion_fickian_diffusion_strict_runtime_point.yaml":
            {"strict_runtime_point": True, "spatial_dims": 1},
    }
    for point in point_rows:
        main = next(row for row in rows if row["process_id"] == point["process_id"]
                    and row["certificate_scope"] == "main_process")
        assert main["level"] == 3
        assert main["constructor_selection"] == {}
        assert point["level"] in {4, 5}
        assert "witness_required" in point["runtime_evidence"]
    # Each process class keeps exactly one main certificate row.
    main_rows = [row["process_id"] for row in rows if row["certificate_scope"] == "main_process"]
    assert len(main_rows) == len(set(main_rows))
