"""System evidence for the two bounded ACP observation twins."""

from __future__ import annotations

import hashlib
import importlib.util
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from relic.core.simulator.checkpoint import CHECKPOINT_FORMAT_VERSION
from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_io import load_graph
from relic.orchestration.validator import validate_graph
from relic.utils.physical_constants import get_physical_constant

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / "scripts" / "publication" / "profile_registry.py"
SPEC = importlib.util.spec_from_file_location(
    "publication_profile_registry_system",
    REGISTRY_PATH,
)
assert SPEC is not None and SPEC.loader is not None
REGISTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REGISTRY)

TOWER_FLUX = (
    "diagnostics.observation.tower_flux.tower_equivalent_flux"
)
WEIGHTED_FLUX = (
    "diagnostics.observation.tower_flux."
    "footprint_weighted_isoprene_flux"
)
RAW_PPBV = (
    "diagnostics.observation.tower_concentration.raw_ppbv"
)
WINDOW_COVERAGE = (
    "diagnostics.observation.tower_concentration.window_coverage"
)
WINDOW_COMPLETE = (
    "diagnostics.observation.tower_concentration.window_complete"
)
QC_MASK = "diagnostics.observation.tower_concentration.mask"
QC_FLAG = "diagnostics.observation.tower_concentration.flag"
QC_WEIGHT = "diagnostics.observation.tower_concentration.weight"
BOUNDARY_FLUX = "tracer_concentration_diffusive_boundary_flux"


def _compile_acp_profile(profile_id: str) -> dict[str, Any]:
    if profile_id == "PROFILE-003":
        manifest = REGISTRY.load_g3_synthetic_fixture_manifest()
        profile = manifest["profiles"][profile_id]
        node_registry, adapter_registry = (
            REGISTRY.build_g2_publication_registries(profile_id)
        )
    elif profile_id == "PROFILE-004":
        manifest = REGISTRY.load_profile_004_synthetic_fixture_manifest()
        profile = manifest["profile"]
        node_registry, adapter_registry = (
            REGISTRY.build_profile_004_publication_registries()
        )
    else:
        raise ValueError(f"Unsupported ACP twin profile: {profile_id}")

    graph_path = ROOT / profile["graph_path"]
    assert hashlib.sha256(graph_path.read_bytes()).hexdigest() == profile[
        "graph_sha256"
    ]
    graph = load_graph(graph_path, validate_schema=True)
    validation = validate_graph(
        graph,
        node_registry,
        strict=True,
        ssot_catalog=node_registry.ssot_catalog,
    )
    assert validation.errors == []
    assert validation.warnings == []
    plan = compile_graph(
        graph,
        node_registry,
        adapter_registry=adapter_registry,
        auto_adapt=False,
        strict=True,
    )
    return {
        "manifest": manifest,
        "profile": profile,
        "graph": graph,
        "registry": node_registry,
        "adapter_registry": adapter_registry,
        "plan": plan,
    }


@pytest.fixture(scope="module")
def compiled_profiles() -> dict[str, dict[str, Any]]:
    return {
        profile_id: _compile_acp_profile(profile_id)
        for profile_id in ("PROFILE-003", "PROFILE-004")
    }


@pytest.fixture(scope="module")
def profile_003_run(
    compiled_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    compiled = compiled_profiles["PROFILE-003"]
    profile = compiled["profile"]
    initial_state = REGISTRY.build_g3_initial_state("PROFILE-003")
    initial_snapshot = {
        name: value.clone() for name, value in initial_state.items()
    }
    simulator_config = {
        **profile["simulator_config"],
        "device": "cpu",
    }
    executor = GraphExecutor(
        compiled["plan"],
        registry=compiled["registry"],
        cache=None,
        simulator_config=simulator_config,
    )
    final_state = executor.execute(
        initial_state,
        n_steps=int(profile["base_n_steps"]),
        dt=float(profile["base_dt_seconds"]),
    )
    return {
        "initial": initial_snapshot,
        "final": final_state,
        "executor": executor,
    }


@pytest.fixture(scope="module")
def profile_004_run(
    compiled_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    compiled = compiled_profiles["PROFILE-004"]
    profile = compiled["profile"]
    initial_state = REGISTRY.build_profile_004_initial_state()
    initial_snapshot = {
        name: value.clone() for name, value in initial_state.items()
    }
    executor = GraphExecutor(
        compiled["plan"],
        registry=compiled["registry"],
        cache=None,
        simulator_config=profile["simulator_config"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final_state = executor.execute(
            initial_state,
            n_steps=int(profile["base_n_steps"]),
            dt=float(profile["base_dt_seconds"]),
        )
    return {
        "initial": initial_snapshot,
        "final": final_state,
        "executor": executor,
        "warnings": [str(item.message) for item in caught],
    }


def _weighted_flux_oracle(
    cell_flux: torch.Tensor,
    normalized_weights: torch.Tensor,
) -> torch.Tensor:
    """Return the independent footprint-weighted flux.

    References:
        Kljun et al. (2015), GMD 8, 3695-3713.
    Mathematical expression:
        F_tower = sum_i(alpha_i * F_i), sum_i(alpha_i) = 1.
    Parameter meanings:
        F_i is cell flux [mol m-2 s-1] and alpha_i is the normalized
        footprint weight [1].
    """

    return torch.sum(cell_flux * normalized_weights)


def _finite_volume_column_oracle(
    initial_concentration: torch.Tensor,
    source_sequence: list[torch.Tensor],
    *,
    diffusivity_m2_s: float,
    spacing_m: float,
    dt_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance an independent zero-flux finite-volume column.

    References:
        Patankar (1980), Numerical Heat Transfer and Fluid Flow.
        Crank (1975), The Mathematics of Diffusion.
    Mathematical expression:
        G_(i+1/2) = K(c_(i+1)-c_i)/dz;
        c_i^(n+1) = c_i^n + dt[(G_(i+1/2)-G_(i-1/2))/dz + S_i].
        Both exterior face fluxes are zero.
    Parameter meanings:
        c is concentration [kg m-3], K is diffusivity [m2 s-1], dz is
        layer spacing [m], S is source [kg m-3 s-1], and dt is time [s].
    """

    concentration = initial_concentration.clone()
    diffusivity = torch.as_tensor(
        diffusivity_m2_s,
        dtype=concentration.dtype,
        device=concentration.device,
    )
    trajectory = []
    for source in source_sequence:
        interior_flux = (
            diffusivity
            * (concentration[..., 1:] - concentration[..., :-1])
            / spacing_m
        )
        right_flux = torch.cat(
            [interior_flux, torch.zeros_like(concentration[..., -1:])],
            dim=-1,
        )
        left_flux = torch.cat(
            [torch.zeros_like(concentration[..., :1]), interior_flux],
            dim=-1,
        )
        diffusion_rate = (right_flux - left_flux) / spacing_m
        concentration = concentration + dt_seconds * (
            diffusion_rate + source.to(concentration)
        )
        trajectory.append(concentration)
    return concentration, torch.stack(trajectory)


def _point_window_oracle(
    concentration_trajectory: torch.Tensor,
    *,
    layer_centers_m: torch.Tensor,
    measurement_heights_m: torch.Tensor,
    temperature_k: torch.Tensor,
    pressure_pa: torch.Tensor,
    dt_seconds: float,
) -> torch.Tensor:
    """Map a trajectory to an independent point-linear ppbv window.

    References:
        Seinfeld and Pandis (2016), Atmospheric Chemistry and Physics.
        CF Metadata Conventions 1.8.
    Mathematical expression:
        chi_h = 1e9(W_h rho)R(W_h T)/(M_iso(W_h p));
        chi_bar = sum_n(chi_h,n dt_n)/sum_n(dt_n).
    Parameter meanings:
        W_h is the point-linear support [1], rho is concentration
        [kg m-3], R is [J mol-1 K-1], T is [K], M_iso is [kg mol-1],
        p is wet-air pressure [Pa], and dt is interval duration [s].
    """

    layer_count = int(layer_centers_m.numel())
    rows = []
    for height in measurement_heights_m:
        right = int(torch.searchsorted(layer_centers_m, height).item())
        left = right - 1
        fraction = (
            (height - layer_centers_m[left])
            / (layer_centers_m[right] - layer_centers_m[left])
        )
        row = torch.zeros(
            layer_count,
            dtype=layer_centers_m.dtype,
            device=layer_centers_m.device,
        )
        row[left] = 1.0 - fraction
        row[right] = fraction
        rows.append(row)
    support = torch.stack(rows)

    profile = concentration_trajectory[:, 0, :]
    temperature_profile = temperature_k.reshape(-1)
    pressure_profile = pressure_pa.reshape(-1)
    if temperature_profile.numel() == 1:
        temperature_profile = temperature_profile.expand(layer_count)
    if pressure_profile.numel() == 1:
        pressure_profile = pressure_profile.expand(layer_count)
    supported_concentration = profile @ support.T
    supported_temperature = support @ temperature_profile
    supported_pressure = support @ pressure_profile
    gas_constant = torch.as_tensor(
        get_physical_constant("gas_constant"),
        dtype=profile.dtype,
        device=profile.device,
    )
    molar_mass = (
        torch.as_tensor(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.isoprene"
            ),
            dtype=profile.dtype,
            device=profile.device,
        )
        * 1.0e-3
    )
    instantaneous = (
        1.0e9
        * supported_concentration
        * gas_constant
        * supported_temperature
        / (molar_mass * supported_pressure)
    )
    numerator = torch.zeros_like(instantaneous[0])
    for value in instantaneous:
        numerator = numerator + value * dt_seconds
    return numerator / (instantaneous.shape[0] * dt_seconds)


def _assert_series_bitwise_equal(
    actual: dict[str, list[Any]],
    expected: dict[str, list[Any]],
) -> None:
    assert set(actual) == set(expected)
    for name, expected_values in expected.items():
        actual_values = actual[name]
        assert len(actual_values) == len(expected_values)
        for actual_value, expected_value in zip(
            actual_values,
            expected_values,
            strict=True,
        ):
            if isinstance(expected_value, torch.Tensor):
                assert isinstance(actual_value, torch.Tensor)
                assert torch.equal(
                    actual_value.detach().cpu(),
                    expected_value.detach().cpu(),
                ), name
            else:
                assert actual_value == expected_value


def test_acp_twins_use_distinct_frozen_strict_graphs(
    compiled_profiles: dict[str, dict[str, Any]],
) -> None:
    flux = compiled_profiles["PROFILE-003"]
    concentration = compiled_profiles["PROFILE-004"]

    assert flux["profile"]["graph_sha256"] != concentration["profile"][
        "graph_sha256"
    ]
    assert "tower_flux_support_operator" in flux["graph"].nodes
    assert "fickian_diffusion" not in flux["graph"].nodes
    assert "tower_concentration_profile_operator" in concentration[
        "graph"
    ].nodes
    assert "support_aggregation_adapter" not in concentration["graph"].nodes
    assert [
        step.node_id for step in flux["plan"].steps
    ] == [
        "atmospheric_radiation",
        "megan_bvoc_model",
        "temporal_resample_adapter",
        "support_aggregation_adapter",
        "tower_flux_support_operator",
    ]
    assert [
        step.node_id for step in concentration["plan"].steps
    ] == [
        "atmospheric_radiation",
        "megan_bvoc_model",
        "temporal_resample_adapter",
        "surface_flux_to_volume_source_adapter",
        "fickian_diffusion",
        "tower_concentration_profile_operator",
        "post_observation_quality_operator",
    ]


def test_acp_flux_twin_matches_independent_weighted_oracle(
    compiled_profiles: dict[str, dict[str, Any]],
    profile_003_run: dict[str, Any],
) -> None:
    initial = profile_003_run["initial"]
    final_state = profile_003_run["final"]
    executor = profile_003_run["executor"]
    emissions = final_state["isoprene_emission"]
    expected_flux = _weighted_flux_oracle(
        emissions,
        initial["footprint_weights"],
    )
    interval_mean = executor.time_series[
        "isoprene_emission_interval_mean"
    ][0]
    actual_weighted = executor.time_series[WEIGHTED_FLUX][0]
    actual_tower = executor.time_series[TOWER_FLUX][0]

    assert torch.equal(interval_mean, emissions)
    assert not torch.equal(emissions[0], emissions[1])
    assert not torch.equal(expected_flux, torch.mean(emissions))
    assert torch.equal(actual_weighted, expected_flux)
    assert torch.equal(actual_tower, expected_flux)
    assert torch.equal(
        executor.time_series[
            "diagnostics.observation.tower_flux.support_integral"
        ][0],
        torch.ones_like(expected_flux),
    )
    assert torch.equal(
        executor.time_series[
            "diagnostics.observation.tower_flux.support_area"
        ][0],
        initial["footprint_cell_area"].sum(),
    )
    assert torch.equal(
        executor.time_series[
            "diagnostics.observation.tower_flux."
            "support_extrapolation_flag"
        ][0],
        torch.zeros_like(expected_flux),
    )
    assert torch.equal(
        executor.time_series[
            "diagnostics.observation.tower_flux.weight_sum"
        ][0],
        torch.ones_like(expected_flux),
    )
    assert torch.equal(
        executor.time_series[
            "diagnostics.observation.tower_flux.coverage_fraction"
        ][0],
        torch.ones_like(expected_flux),
    )
    graph = compiled_profiles["PROFILE-003"]["graph"]
    registry = compiled_profiles["PROFILE-003"]["registry"]
    definition = registry.get_definition(
        graph.nodes["tower_flux_support_operator"].class_type
    )
    assert definition is not None
    assert definition.outputs[TOWER_FLUX].units == "mol m-2 s-1"


def test_acp_concentration_twin_closes_numerics_and_observation(
    compiled_profiles: dict[str, dict[str, Any]],
    profile_004_run: dict[str, Any],
) -> None:
    compiled = compiled_profiles["PROFILE-004"]
    manifest = compiled["manifest"]
    profile = compiled["profile"]
    oracle = manifest["oracle"]
    initial = profile_004_run["initial"]
    final_state = profile_004_run["final"]
    executor = profile_004_run["executor"]
    series = executor.time_series
    dt_seconds = float(profile["base_dt_seconds"])

    assert profile_004_run["warnings"] == []
    assert executor.execution_summary["device"] == "cpu"
    assert executor.execution_summary["current_time"] == 1800.0
    assert executor.execution_summary["step_count"] == 60
    assert all(len(values) == 60 for values in series.values())

    expected_final = torch.as_tensor(
        oracle["expected_final_tracer_kg_m3"],
        dtype=torch.float32,
    )
    assert torch.equal(final_state["tracer_concentration"], expected_final)
    independent_final, concentration_trajectory = (
        _finite_volume_column_oracle(
            initial["tracer_concentration"],
            series["source_term"],
            diffusivity_m2_s=float(oracle["diffusivity_m2_s"]),
            spacing_m=float(oracle["grid_spacing_m"]),
            dt_seconds=dt_seconds,
        )
    )
    assert torch.equal(independent_final, expected_final)

    layer_volumes = initial["layer_volumes"]
    molar_mass_kg_per_mol = (
        torch.as_tensor(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.isoprene"
            ),
            dtype=layer_volumes.dtype,
        )
        * 1.0e-3
    )
    for interval_flux, source in zip(
        series["isoprene_emission_interval_mean"],
        series["source_term"],
        strict=True,
    ):
        volume_source_rate = torch.sum(
            source.to(layer_volumes) * layer_volumes
        )
        surface_source_rate = torch.sum(
            interval_flux.to(layer_volumes)
            * molar_mass_kg_per_mol
            * initial["cell_area"]
        )
        torch.testing.assert_close(
            volume_source_rate,
            surface_source_rate,
            rtol=1.0e-6,
            atol=0.0,
        )
        assert torch.count_nonzero(source) == 1

    initial_mass = torch.sum(
        initial["tracer_concentration"] * layer_volumes
    )
    source_mass = torch.stack(
        [
            torch.sum(source.to(layer_volumes) * layer_volumes)
            * dt_seconds
            for source in series["source_term"]
        ]
    ).sum()
    final_mass = torch.sum(
        final_state["tracer_concentration"] * layer_volumes
    )
    boundary_flux = torch.stack(series[BOUNDARY_FLUX])
    boundary_export_mass = (
        boundary_flux.sum()
        * initial["cell_area"].reshape(-1)[0]
        * dt_seconds
    )
    ledger_residual = (
        final_mass
        - initial_mass
        - source_mass
        + boundary_export_mass
    )
    assert boundary_flux.shape[-1] == 2
    assert torch.count_nonzero(boundary_flux) == 0
    assert float(initial_mass.detach()) == oracle[
        "expected_initial_mass_kg"
    ]
    assert float(final_mass.detach()) == oracle["expected_final_mass_kg"]
    assert float(source_mass.detach()) == pytest.approx(
        oracle["expected_cumulative_source_mass_kg"],
        rel=float(oracle["mass_relative_tolerance"]),
    )
    assert (
        float(
            (torch.abs(ledger_residual) / source_mass).detach()
        )
        <= float(oracle["mass_relative_tolerance"])
    )
    assert max(
        float(value.detach())
        for value in series["diagnostics.conservation_error"]
    ) <= float(oracle["mass_relative_tolerance"])
    for value in series["diagnostics.diffusion_stability"]:
        torch.testing.assert_close(
            value,
            torch.tensor(
                oracle["stability_number"],
                dtype=value.dtype,
                device=value.device,
            ),
        )

    expected_raw = torch.as_tensor(
        oracle["expected_final_raw_ppbv"],
        dtype=torch.float32,
    )
    actual_raw = series[RAW_PPBV][-1]
    assert torch.equal(actual_raw, expected_raw)
    independent_raw = _point_window_oracle(
        concentration_trajectory,
        layer_centers_m=initial["layer_center_height"],
        measurement_heights_m=torch.as_tensor(
            oracle["measurement_heights_m"],
            dtype=torch.float32,
        ),
        temperature_k=initial["air_temperature"],
        pressure_pa=initial["atmospheric_pressure"],
        dt_seconds=dt_seconds,
    )
    torch.testing.assert_close(
        actual_raw,
        independent_raw,
        rtol=float(oracle["observation_relative_tolerance"]),
        atol=float(oracle["observation_absolute_tolerance_ppbv"]),
    )

    torch.testing.assert_close(
        series[WINDOW_COVERAGE][0],
        torch.full_like(series[WINDOW_COVERAGE][0], 1.0 / 60.0),
    )
    assert torch.equal(
        series[WINDOW_COVERAGE][-1],
        torch.ones_like(series[WINDOW_COVERAGE][-1]),
    )
    assert torch.count_nonzero(series[WINDOW_COMPLETE][0]) == 0
    assert torch.count_nonzero(series[WINDOW_COMPLETE][29]) == 0
    assert torch.count_nonzero(series[WINDOW_COMPLETE][30]) == 0
    assert torch.equal(
        series[WINDOW_COMPLETE][-1],
        torch.ones_like(series[WINDOW_COMPLETE][-1]),
    )
    assert torch.count_nonzero(series[QC_MASK][0]) == 0
    assert torch.equal(
        series[QC_FLAG][0],
        torch.full_like(series[QC_FLAG][0], 8),
    )
    assert torch.count_nonzero(series[QC_WEIGHT][0]) == 0
    assert torch.equal(
        series[QC_MASK][-1],
        torch.ones_like(series[QC_MASK][-1]),
    )
    assert torch.count_nonzero(series[QC_FLAG][-1]) == 0
    assert torch.equal(
        series[QC_WEIGHT][-1],
        torch.ones_like(series[QC_WEIGHT][-1]),
    )
    for support_sum in series[
        "diagnostics.observation.tower_concentration.support_weight_sum"
    ]:
        assert torch.equal(support_sum, torch.ones_like(support_sum))
    assert "source_term" not in final_state
    assert RAW_PPBV not in final_state
    assert not any(
        name.startswith("diagnostics.") for name in final_state
    )

    graph = compiled["graph"]
    registry = compiled["registry"]
    diffusion = registry.get_definition(
        graph.nodes["fickian_diffusion"].class_type
    )
    concentration_operator = registry.get_definition(
        graph.nodes["tower_concentration_profile_operator"].class_type
    )
    assert diffusion is not None
    assert concentration_operator is not None
    surface_source = registry.get_definition(
        graph.nodes[
            "surface_flux_to_volume_source_adapter"
        ].class_type
    )
    assert surface_source is not None
    assert surface_source.outputs["source_term"].units == "kg m-3 s-1"
    assert diffusion.outputs["tracer_concentration"].units == "kg m-3"
    assert diffusion.outputs[BOUNDARY_FLUX].units == "kg m-2 s-1"
    assert concentration_operator.outputs[RAW_PPBV].units == "ppbv"


def test_profile_004_partial_window_checkpoint_is_bitwise_exact(
    tmp_path: Path,
    compiled_profiles: dict[str, dict[str, Any]],
    profile_004_run: dict[str, Any],
) -> None:
    compiled = compiled_profiles["PROFILE-004"]
    profile = compiled["profile"]
    initial_state = REGISTRY.build_profile_004_initial_state()
    first_leg = GraphExecutor(
        compiled["plan"],
        registry=compiled["registry"],
        cache=None,
        simulator_config=profile["simulator_config"],
    )
    first_leg.execute(
        initial_state,
        n_steps=30,
        dt=float(profile["base_dt_seconds"]),
    )
    torch.testing.assert_close(
        first_leg.time_series[WINDOW_COVERAGE][-1],
        torch.full_like(
            first_leg.time_series[WINDOW_COVERAGE][-1],
            0.5,
        ),
    )
    assert torch.count_nonzero(
        first_leg.time_series[WINDOW_COMPLETE][-1]
    ) == 0

    checkpoint_path = tmp_path / "profile_004_partial_window.pt"
    receipt = first_leg.create_checkpoint(checkpoint_path)
    assert receipt["path"] == str(checkpoint_path.resolve())
    assert receipt["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert receipt["size_bytes"] > 0
    assert len(receipt["sha256"]) == 64
    assert len(receipt["payload_sha256"]) == 64
    restarted = GraphExecutor(
        compiled["plan"],
        registry=compiled["registry"],
        cache=None,
        simulator_config=profile["simulator_config"],
    )
    restarted_state = restarted.execute_from_checkpoint(
        REGISTRY.build_profile_004_initial_state(),
        checkpoint_path,
        n_steps=30,
        dt=float(profile["base_dt_seconds"]),
    )

    uninterrupted_state = profile_004_run["final"]
    assert set(restarted_state) == set(uninterrupted_state)
    for name, expected in uninterrupted_state.items():
        assert torch.equal(
            restarted_state[name].detach().cpu(),
            expected.detach().cpu(),
        ), name
    _assert_series_bitwise_equal(
        restarted.time_series,
        profile_004_run["executor"].time_series,
    )
    assert restarted.execution_summary["current_time"] == 1800.0
    assert restarted.execution_summary["step_count"] == 60
    assert (
        restarted.execution_summary["execution_order"]
        == profile_004_run["executor"].execution_summary["execution_order"]
    )


def test_profile_004_full_graph_preserves_storage_and_source_gradients(
    compiled_profiles: dict[str, dict[str, Any]],
) -> None:
    compiled = compiled_profiles["PROFILE-004"]
    profile = compiled["profile"]
    initial_state = REGISTRY.build_profile_004_initial_state()
    initial_concentration = initial_state[
        "tracer_concentration"
    ].requires_grad_(True)
    leaf_area_index = initial_state["leaf_area_index"].requires_grad_(True)
    executor = GraphExecutor(
        compiled["plan"],
        registry=compiled["registry"],
        cache=None,
        simulator_config=profile["simulator_config"],
    )
    final_state = executor.execute(
        initial_state,
        n_steps=int(profile["base_n_steps"]),
        dt=float(profile["base_dt_seconds"]),
    )
    storage_gradient, source_gradient = torch.autograd.grad(
        final_state["tracer_concentration"].sum(),
        (initial_concentration, leaf_area_index),
    )

    assert torch.all(torch.isfinite(storage_gradient))
    assert torch.all(torch.isfinite(source_gradient))
    torch.testing.assert_close(
        storage_gradient,
        torch.ones_like(storage_gradient),
    )
    assert torch.all(source_gradient > 0.0)


@pytest.fixture(scope="module")
def paper02_goal_wrapper():
    """Load the existing research wrapper; production oracles remain above."""
    import os
    import sys

    paper_root = Path(os.environ.get("RELIC_PAPER_ROOT", ROOT.parent / "RELICPAPER"))
    analysis = paper_root / "paper02_structure_preserving_scales" / "analysis"
    path = analysis / "run_bounded_manufactured_acp.py"
    sys.path.insert(0, str(analysis))
    try:
        spec = importlib.util.spec_from_file_location("paper02_goal_wrapper", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(analysis))
    return module


def test_paper02_trace_excludes_a_failed_governed_forcing_step(
    compiled_profiles, paper02_goal_wrapper,
):
    from relic.core.functor.functor_exceptions import ExecutionError
    initial = REGISTRY.build_profile_004_initial_state()
    observer = paper02_goal_wrapper.AcceptedColumnTrace()
    invalid = torch.full_like(initial["observation.tower_concentration.instrument_valid"], 2.0)
    with pytest.raises(ExecutionError, match="Instrument validity"):
        paper02_goal_wrapper.execute_goal_column(
            compiled_profiles["PROFILE-004"], initial, dt_seconds=30.0,
            step_count=2, observer=observer,
            forcing_sequence=[{}, {"observation.tower_concentration.instrument_valid": invalid}],
        )
    assert len(observer.records) == 1
    assert observer.records[0]["transaction_open"] is False
    assert observer.records[0]["post"].shape == (1, 6)
    assert torch.equal(observer.records[0]["pre"], initial["tracer_concentration"])


def test_paper02_actual_goal_is_enclosed_by_complete_equation_residual(
    compiled_profiles, paper02_goal_wrapper,
):
    initial = REGISTRY.build_profile_004_initial_state()
    case = paper02_goal_wrapper.execute_goal_column(
        compiled_profiles["PROFILE-004"], initial,
        dt_seconds=30.0, step_count=60,
    )
    assert case["accepted_states"].shape == (61, 6)
    assert torch.equal(case["accepted_states"][-1], case["final"]["tracer_concentration"][0])
    result = paper02_goal_wrapper.evaluate_column_goal(case, compiled_profiles["PROFILE-004"])
    from fractions import Fraction
    serialized_error = max(abs(Fraction.from_float(result["actual_goal_ppbv"]) - Fraction.from_float(value)) for value in result["reference_goal_interval_ppbv"])
    assert Fraction.from_float(result["measured_goal_error_upper_ppbv"]) >= serialized_error
    assert result["reference_goal_interval_ppbv"][1] - result["reference_goal_interval_ppbv"][0] < 1.0e-10
    assert result["actual_goal_ppbv"] > 0.0
    assert result["measured_goal_error_upper_ppbv"] <= result["total_goal_error_bound_ppbv"]
    assert result["residual_integral_upper_kg_m3"] > 0.0
    assert result["endpoint_quadrature_bound_ppbv"] > 0.0
    assert result["reference_scope"] == "fixed_given_source_semidiscrete_transport_not_MEGAN_law_validation"


def test_paper02_residual_does_not_omit_the_source_or_accept_incomplete_trace(
    compiled_profiles, paper02_goal_wrapper,
):
    initial = REGISTRY.build_profile_004_initial_state()
    case = paper02_goal_wrapper.execute_goal_column(
        compiled_profiles["PROFILE-004"], initial,
        dt_seconds=30.0, step_count=3,
    )
    result = paper02_goal_wrapper.evaluate_column_goal(case, compiled_profiles["PROFILE-004"])
    without_source = paper02_goal_wrapper.evaluate_column_goal(
        case, compiled_profiles["PROFILE-004"], reference_source=torch.zeros(6, dtype=torch.float64),
    )
    assert without_source["residual_integral_upper_kg_m3"] > result["residual_integral_upper_kg_m3"]
    incomplete = dict(case, accepted_states=case["accepted_states"][:-1])
    with pytest.raises(ValueError, match="complete"):
        paper02_goal_wrapper.evaluate_column_goal(incomplete, compiled_profiles["PROFILE-004"])


def test_paper02_direction_requires_strict_separation_of_complete_bounds(paper02_goal_wrapper):
    assert paper02_goal_wrapper.goal_direction(0.5, 0.25, 0.25) == "undetermined"
    assert paper02_goal_wrapper.goal_direction(0.51, 0.25, 0.25) == "positive"
    assert paper02_goal_wrapper.goal_direction(-0.51, 0.25, 0.25) == "negative"
    with pytest.raises(ValueError):
        paper02_goal_wrapper.goal_direction(float("nan"), 0.25, 0.25)


@pytest.fixture(scope="module")
def paper02_temperature_coarse_result(compiled_profiles, paper02_goal_wrapper):
    """Exercise three original columns without requiring external saved run artifacts."""
    return paper02_goal_wrapper.qualify_temperature_coarse_pair(
        compiled_profiles["PROFILE-004"], temperature_offset_k=5.0,
        dt_seconds=30.0, step_count=60,
    )


def test_paper02_same_extended_coarse_state_has_certified_different_future(
    paper02_temperature_coarse_result,
):
    result = paper02_temperature_coarse_result
    assert result["unique_original_column_executions"] == 3
    assert result["accepted_steps"] == 180
    assert result["initial_coarse_state_exactly_equal"]
    assert result["complete_forcing_explicit_and_held"]
    assert result["source_and_transport_independent_checks_passed"]
    assert result["heterogeneous_pair"]["M1_fibre_condition_violated_on_this_pair"]
    assert result["heterogeneous_pair"]["M2_state_norm_half_separation_lower"] > 0
    assert result["heterogeneous_pair"]["M2_inventory_half_separation_lower_kg"] > 0
    assert result["homogeneous_control"]["finite_diagonal_control_exactly_equal"]
    assert not result["all_state_closure_or_ecological_validation"]
    from fractions import Fraction
    pair = result["heterogeneous_pair"]
    q = lambda value: Fraction.from_float(float(value))
    shortfalls = []
    for suffix in ("A", "B"):
        values = pair["actual_coarse_concentration_" + suffix + "_kg_m3"]
        intervals = pair["coarse_reference_" + suffix + "_intervals_kg_m3"]
        bounds = pair["coarse_" + suffix + "_component_error_bounds_kg_m3"]
        for value, endpoints, bound in zip(values, intervals, bounds):
            actual_error = max(abs(Fraction.from_float(value) - Fraction.from_float(endpoint)) for endpoint in endpoints)
            if Fraction.from_float(bound) < actual_error:
                shortfalls.append("component_" + suffix)
        volumes = [2 * q(value) for value in result["cases"]["uniform"]["initial"]["layer_volumes"][0]]
        inventory_endpoints = [sum(volume * q(endpoint[index]) for volume, endpoint in zip(volumes, intervals)) for index in (0, 1)]
        inventory_value = q(pair["actual_coarse_inventory_" + suffix + "_kg"])
        inventory_need = max(abs(inventory_value - endpoint) for endpoint in inventory_endpoints)
        if q(pair["coarse_" + suffix + "_complete_inventory_error_bound_kg"]) < inventory_need:
            shortfalls.append("inventory_" + suffix)
    molar_mass = q(float(get_physical_constant("relic.physical.constants.molecular_weights.isoprene"))) / 1000
    for key, case in result["cases"].items():
        factor = molar_mass * q(case["initial"]["cell_area"][0]) / q(case["initial"]["layer_volumes"][0, 0])
        point = q(case["source_reference_flux_mol_m2_s"] * float(molar_mass) * float(case["initial"]["cell_area"][0]) / float(case["initial"]["layer_volumes"][0, 0]))
        needed = max(abs(point - q(endpoint) * factor) for endpoint in case["source_reference_interval_mol_m2_s"])
        if q(case["injected_source_L2_uncertainty_kg_m3_s"]) < needed:
            shortfalls.append("source_" + key)
        assert q(case["matrix_reference_source_error_inflation_kg_m3"]) >= (
            q(case["dt_seconds"]) * case["step_count"] * q(case["injected_source_L2_uncertainty_kg_m3_s"])
        )
    assert not shortfalls, shortfalls
    gap = sum(volume * abs(q(left) - q(right)) for volume, left, right in zip(
        volumes, pair["actual_coarse_concentration_A_kg_m3"], pair["actual_coarse_concentration_B_kg_m3"]))
    state_limit = (gap - q(pair["coarse_A_complete_concentration_error_bound_kg"])
                   - q(pair["coarse_B_complete_concentration_error_bound_kg"])
                   - q(pair["coarse_concentration_aggregation_rounding_bound_kg"])) / (2 * q(result["inventory_scale_kg"]))
    assert q(pair["M2_state_norm_half_separation_lower"]) <= state_limit
    inventory_limit = (abs(q(pair["actual_coarse_inventory_A_kg"]) - q(pair["actual_coarse_inventory_B_kg"]))
                       - q(pair["coarse_A_complete_inventory_error_bound_kg"])
                       - q(pair["coarse_B_complete_inventory_error_bound_kg"])
                       - q(pair["coarse_inventory_aggregation_rounding_bound_kg"])) / 2
    assert q(pair["M2_inventory_half_separation_lower_kg"]) <= inventory_limit
    for case in result["cases"].values():
        assert case["accepted_states"].shape == (61, 6)
        assert case["source_reference_relative_error"] < 1.0e-6
        assert case["source_interval_width_mol_m2_s"] < 1.0e-20
        assert case["matrix_reference_source_error_inflation_kg_m3"] >= 0


def test_paper02_coarse_contract_rejects_a_hidden_forcing_difference(
    paper02_temperature_coarse_result, paper02_goal_wrapper,
):
    result = paper02_temperature_coarse_result
    bad = {key: dict(value) for key, value in result["cases"].items()}
    bad["minus"]["initial"] = dict(bad["minus"]["initial"])
    bad["minus"]["initial"]["incoming_shortwave_potential"] = (
        bad["minus"]["initial"]["incoming_shortwave_potential"] * 0.5
    )
    with pytest.raises(ValueError, match="held forcing"):
        paper02_goal_wrapper.evaluate_temperature_coarse_pair(bad)
