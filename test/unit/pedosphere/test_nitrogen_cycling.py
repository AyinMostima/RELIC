"""Unit tests for the NitrogenCycling composition diagnostic."""

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from relic.scientific.pedosphere.nutrient_cycling.nitrogen import NitrogenCycling
from relic.core.state import UpdateItem, UpdateMode


def _declared_subprocess_outputs():
    """Return deterministic outputs from separately scheduled dependencies."""

    return {
        "mineralization": {
            "organic_nitrogen": torch.tensor(-1.0),
            "nh4_nitrogen": torch.tensor(1.0),
            "mineralization_flux": torch.tensor(0.2),
        },
        "nitrification": {
            "nh4_nitrogen": torch.tensor(-0.5),
            "no3_nitrogen": torch.tensor(0.5),
            "nitrification_flux": torch.tensor(0.3),
        },
        "denitrification": {
            "no3_nitrogen": torch.tensor(-0.4),
            "denitrification_flux": torch.tensor(0.1),
        },
        "nitrogen_leaching": {
            "no3_nitrogen": torch.tensor(-0.2),
            "nitrogen_leaching_flux": torch.tensor(0.05),
        },
        "root_nutrient_uptake": {
            "no3_nitrogen": torch.tensor(-0.1),
            "nh4_nitrogen": torch.tensor(-0.05),
            "nitrogen_uptake_rate": torch.tensor(0.1),
            "no3_uptake_by_layer": torch.tensor([0.01, 0.02]),
        },
    }


def test_coordinator_reports_ledger_without_rewriting_subprocess_outputs():
    """Dependency outputs must be observed once and never emitted again."""
    cycling = NitrogenCycling(
        config={},
        enable_mineralization=True,
        enable_nitrification=True,
        enable_denitrification=True,
        enable_leaching=True,
        enable_root_uptake=True,
    )

    state = {
        "nh4_nitrogen": torch.tensor(4.45),
        "no3_nitrogen": torch.tensor(0.8),
    }
    output = cycling.forward(
        state=state,
        time=0.0,
        dt=10.0,
        subprocess_outputs=_declared_subprocess_outputs(),
    )

    assert torch.allclose(
        output["diagnostics.nitrogen_cycling.expected_delta_nh4_nitrogen"],
        torch.tensor(0.45),
    )
    assert torch.allclose(
        output["diagnostics.nitrogen_cycling.expected_delta_no3_nitrogen"],
        torch.tensor(-0.2),
    )
    forbidden = {
        "organic_nitrogen",
        "nh4_nitrogen",
        "no3_nitrogen",
        "microbial_nitrogen",
        "available_phosphorus",
        "plant_n_storage",
        "plant_p_storage",
        "mineralization_flux",
        "nitrification_flux",
        "denitrification_flux",
        "nitrogen_leaching_flux",
        "nitrogen_uptake_rate",
        "phosphorus_uptake_rate",
        "no3_uptake_by_layer",
        "nh4_uptake_by_layer",
        "po4_uptake_by_layer",
    }
    assert forbidden.isdisjoint(output)


def test_dependencies_are_declared_without_embedded_functors():
    """Ensure composition uses the dependency graph, not direct Functor calls."""
    cycling = NitrogenCycling(
        config={},
        enable_mineralization=True,
        enable_nitrification=True,
        enable_denitrification=True,
        enable_leaching=True,
        enable_root_uptake=True,
    )

    assert cycling.get_dependencies() == [
        "mineralization",
        "nitrification",
        "denitrification",
        "nitrogen_leaching",
        "root_nutrient_uptake",
    ]
    for name in ("mineralization", "nitrification", "denitrification", "leaching", "root_uptake"):
        assert not hasattr(cycling, name)
    assert ".forward(" not in inspect.getsource(NitrogenCycling.forward)


def test_certificate_binds_fertilizer_conversion_to_physical_constant_ssot():
    """The equivalence certificate must name the canonical conversion owner."""
    path = Path(
        "docs/source/theory/iphs_certificates/"
        "pedosphere_nutrient_cycling_nitrogen_nitrogen_cycling.yaml"
    )
    certificate = yaml.safe_load(path.read_text(encoding="utf-8"))
    key = "relic.physical.constants.conversions.kg_to_g"
    assert certificate["parameter_keys"] == [key]
    assert key in certificate["parameter_map"]
    binding = certificate["parameter_management_map"][key]["source"]
    assert "get_physical_constant" in binding
    assert "UnifiedParameterManager.get" in binding
    no_legacy_mode = certificate["runtime_replacement_map"]["no_legacy_mode"]
    assert "No selectable legacy runtime branch" in no_legacy_mode


def test_declared_outputs_are_the_only_composition_path():
    """Ensure the canonical path consumes scheduler-provided dependency outputs."""
    cycling = NitrogenCycling(
        config={},
        enable_mineralization=True,
        enable_nitrification=True,
        enable_denitrification=True,
        enable_leaching=True,
        enable_root_uptake=True,
    )

    output = cycling.forward(
        state={"nh4_nitrogen": torch.tensor(2.0), "no3_nitrogen": torch.tensor(1.0)},
        time=0.0,
        dt=10.0,
        subprocess_outputs={
            "mineralization": {"nh4_nitrogen": torch.tensor(0.2)},
            "nitrification": {
                "nh4_nitrogen": torch.tensor(-0.1),
                "no3_nitrogen": torch.tensor(0.1),
            },
            "denitrification": {"no3_nitrogen": torch.tensor(-0.05)},
            "nitrogen_leaching": {"no3_nitrogen": torch.tensor(-0.02)},
            "root_nutrient_uptake": {"no3_nitrogen": torch.tensor(-0.03)},
        },
    )

    assert torch.allclose(
        output["diagnostics.nitrogen_cycling.expected_delta_nh4_nitrogen"],
        torch.tensor(0.1),
    )
    assert torch.allclose(
        output["diagnostics.nitrogen_cycling.expected_delta_no3_nitrogen"],
        torch.tensor(0.0),
    )
    assert "nh4_nitrogen" not in output
    assert "no3_nitrogen" not in output


def test_fertilizer_port_is_the_only_state_write_and_converts_kg_to_g():
    """The external fertilizer port contributes one NH4 increment in pool units."""

    cycling = NitrogenCycling(
        config={},
        enable_mineralization=False,
        enable_nitrification=False,
        enable_denitrification=False,
        enable_leaching=False,
        enable_root_uptake=False,
    )
    output = cycling.forward(
        state={"fertilizer_n_flux": torch.tensor(1.0e-6)},
        time=0.0,
        dt=10.0,
        subprocess_outputs={},
    )

    fertilizer_update = output["nh4_nitrogen"]
    assert isinstance(fertilizer_update, UpdateItem)
    assert fertilizer_update.mode is UpdateMode.ACCUMULATE
    assert fertilizer_update.unit == "g N m-2"
    assert fertilizer_update.basis == "per_step"
    assert fertilizer_update.per_timestep is True
    assert fertilizer_update.element == "N"
    assert fertilizer_update.source_functor == cycling.name
    assert torch.allclose(fertilizer_update.value, torch.tensor(0.01))
    assert torch.allclose(
        output["diagnostics.nitrogen_cycling.external_fertilizer_n_delta"],
        torch.tensor(0.01),
    )
    assert torch.equal(
        output["diagnostics.nitrogen_cycling.external_urine_n_delta"],
        torch.tensor(0.0),
    )
    for key, value in output.items():
        if key.startswith("diagnostics."):
            assert getattr(value, "element", None) is None
    state_keys = {key for key in output if not key.startswith("diagnostics.")}
    assert state_keys == {"nh4_nitrogen"}


def test_fertilizer_and_urine_ports_are_combined_once_in_nh4() -> None:
    """The two explicit external mineral-N ports contribute exactly once."""
    cycling = NitrogenCycling(
        config={"dependencies": ["grazing_management"]},
        enable_mineralization=False,
        enable_nitrification=False,
        enable_denitrification=False,
        enable_leaching=False,
        enable_root_uptake=False,
    )
    state = {
        "fertilizer_n_flux": torch.tensor(1.0e-6),
        "urine_n_flux": torch.tensor(2.0e-6),
    }

    output = cycling.forward(
        state=state,
        time=0.0,
        dt=10.0,
        subprocess_outputs={},
    )

    torch.testing.assert_close(
        output["nh4_nitrogen"].value,
        torch.tensor(0.03),
    )
    torch.testing.assert_close(
        output["diagnostics.nitrogen_cycling.external_fertilizer_n_delta"],
        torch.tensor(0.01),
    )
    torch.testing.assert_close(
        output["diagnostics.nitrogen_cycling.external_urine_n_delta"],
        torch.tensor(0.02),
    )
    assert cycling.check_conservation_laws(state, output)["nitrogen"] == 0.0


def test_urine_port_conservation_checker_detects_tampered_nh4_update() -> None:
    """The open-port checker must recompute rather than self-attest closure."""
    cycling = NitrogenCycling(
        config={"dependencies": ["grazing_management"]},
        enable_mineralization=False,
        enable_nitrification=False,
        enable_denitrification=False,
        enable_leaching=False,
        enable_root_uptake=False,
    )
    state = {"urine_n_flux": torch.tensor(2.0e-6)}
    output = cycling.forward(
        state=state,
        time=0.0,
        dt=10.0,
        subprocess_outputs={},
    )
    item = output["nh4_nitrogen"]
    output["nh4_nitrogen"] = UpdateItem(
        value=item.value + 0.1,
        mode=item.mode,
        unit=item.unit,
        basis=item.basis,
        per_timestep=item.per_timestep,
        element=item.element,
        source_functor=item.source_functor,
    )

    assert cycling.check_conservation_laws(state, output)["nitrogen"] > 0.0


@pytest.mark.parametrize(
    ("key", "value", "error_type"),
    [
        ("urine_n_flux", torch.tensor(-1.0e-6), ValueError),
        ("urine_n_flux", torch.tensor(float("nan")), ValueError),
        ("urine_n_flux", 1.0e-6, TypeError),
        ("fertilizer_n_flux", torch.tensor(1), TypeError),
    ],
)
def test_external_nitrogen_ports_fail_fast(
    key: str,
    value: object,
    error_type: type[Exception],
) -> None:
    cycling = NitrogenCycling(
        config=(
            {"dependencies": ["grazing_management"]}
            if key == "urine_n_flux"
            else {}
        ),
        enable_mineralization=False,
        enable_nitrification=False,
        enable_denitrification=False,
        enable_leaching=False,
        enable_root_uptake=False,
    )

    with pytest.raises(error_type, match=key):
        cycling.forward(
            state={key: value},
            time=0.0,
            dt=10.0,
            subprocess_outputs={},
        )


def test_urine_port_requires_declared_grazing_dependency() -> None:
    cycling = NitrogenCycling(
        config={},
        enable_mineralization=False,
        enable_nitrification=False,
        enable_denitrification=False,
        enable_leaching=False,
        enable_root_uptake=False,
    )

    with pytest.raises(
        ValueError,
        match="requires the grazing_management dependency",
    ):
        cycling.forward(
            state={"urine_n_flux": torch.tensor(1.0e-6)},
            time=0.0,
            dt=10.0,
            subprocess_outputs={},
        )


def test_selected_grazing_dependency_requires_urine_port() -> None:
    cycling = NitrogenCycling(
        config={"dependencies": ["grazing_management"]},
        enable_mineralization=False,
        enable_nitrification=False,
        enable_denitrification=False,
        enable_leaching=False,
        enable_root_uptake=False,
    )

    with pytest.raises(KeyError, match="urine_n_flux"):
        cycling.forward(
            state={},
            time=0.0,
            dt=10.0,
            subprocess_outputs={},
        )


def test_missing_declared_dependency_output_fails_fast():
    """Ensure missing dependency output cannot silently skip a process."""
    cycling = NitrogenCycling(
        config={},
        enable_mineralization=True,
        enable_nitrification=False,
        enable_denitrification=False,
        enable_leaching=False,
        enable_root_uptake=False,
    )

    with pytest.raises(KeyError, match="mineralization"):
        cycling.forward(state={}, time=0.0, dt=60.0)


def test_non_mapping_dependency_output_fails_fast():
    """Ensure malformed dependency interconnections are rejected."""
    cycling = NitrogenCycling(
        config={},
        enable_mineralization=True,
        enable_nitrification=True,
        enable_denitrification=False,
        enable_leaching=False,
        enable_root_uptake=False,
    )

    with pytest.raises(TypeError, match="nitrification"):
        cycling.forward(
            state={},
            time=0.0,
            dt=60.0,
            subprocess_outputs={"mineralization": {}, "nitrification": 1.0},
        )


def test_nonfinite_dependency_output_fails_fast():
    """Ensure non-finite dependency rates are never repaired silently."""
    cycling = NitrogenCycling(
        config={},
        enable_mineralization=True,
        enable_nitrification=False,
        enable_denitrification=False,
        enable_leaching=False,
        enable_root_uptake=False,
    )

    with pytest.raises(ValueError, match="non-finite"):
        cycling.forward(
            state={"nh4_nitrogen": torch.tensor(1.0)},
            time=0.0,
            dt=60.0,
            subprocess_outputs={
                "mineralization": {"mineralization_flux": torch.tensor(float("nan"))}
            },
        )


def test_metadata_declares_nitrogen_conservation_and_execution_phase():
    """Ensure metadata contract fields are explicitly populated."""
    cycling = NitrogenCycling(
        config={},
        enable_mineralization=False,
        enable_nitrification=False,
        enable_denitrification=False,
        enable_leaching=False,
        enable_root_uptake=False,
    )
    assert cycling.metadata.version is not None
    assert cycling.metadata.execution_phase is not None
    assert "nitrogen" in cycling.metadata.conservation_laws
    assert "no3_nitrogen" not in cycling.metadata.output_variables
    assert "mineralization_flux" not in cycling.metadata.output_variables
    assert "nh4_nitrogen" in cycling.metadata.output_variables
    assert set(cycling.metadata.input_variables) == {
        "fertilizer_n_flux",
        "urine_n_flux",
    }
    assert set(cycling.optional_input_variables) == {
        "fertilizer_n_flux",
        "urine_n_flux",
    }
    assert (
        "diagnostics.nitrogen_cycling.external_urine_n_delta"
        in cycling.metadata.output_variables
    )


def test_declared_nitrogen_stack_has_no_conservation_or_nonfinite_warnings(caplog):
    """Run declared dependency outputs and ensure finite canonical aggregation."""
    cycling = NitrogenCycling(
        config={},
        enable_mineralization=True,
        enable_nitrification=True,
        enable_denitrification=True,
        enable_leaching=True,
        enable_root_uptake=True,
    )

    state = {
        "organic_nitrogen": torch.tensor(80.0),
        "nh4_nitrogen": torch.tensor(4.0),
        "no3_nitrogen": torch.tensor(8.0),
        "microbial_nitrogen": torch.tensor(1.5),
        "available_phosphorus": torch.tensor(0.8),
        "plant_n_storage": torch.tensor(2.0),
        "plant_p_storage": torch.tensor(0.4),
        "soil_temperature": torch.tensor(20.0),
        "soil_moisture": torch.tensor(0.35),
        "soil_ph": torch.tensor(6.8),
        "carbon_decomposition_rate": torch.tensor(0.02),
        "microbial_biomass": torch.tensor(3.0),
        "organic_carbon": torch.tensor(180.0),
        "labile_carbon": torch.tensor(12.0),
        "drainage_flux": torch.tensor(1.0e-4),
        "plant_n_demand": torch.tensor(0.001),
        "plant_p_demand": torch.tensor(0.0002),
        "soil_depth": torch.tensor(0.3),
        "root_density": torch.tensor(1.0),
        "soil_water_content": torch.tensor(0.3),
    }
    dt = torch.tensor(3600.0)

    with caplog.at_level("WARNING"):
        for step in range(3):
            output = cycling.forward(
                state=state,
                time=float(step) * float(dt.item()),
                dt=dt,
                subprocess_outputs=_declared_subprocess_outputs(),
            )
            for key, value in output.items():
                if torch.is_tensor(value):
                    assert torch.isfinite(value).all(), f"{key} contains non-finite values"
                if key in state and torch.is_tensor(value) and torch.isfinite(value).all():
                    state[key] = state[key] + value

    noisy_logs = [
        record.getMessage()
        for record in caplog.records
        if (
            "Conservation violation" in record.getMessage()
            or "contains NaN" in record.getMessage()
            or "contains Inf" in record.getMessage()
        )
    ]
    assert not noisy_logs, f"Unexpected warning logs: {noisy_logs}"
