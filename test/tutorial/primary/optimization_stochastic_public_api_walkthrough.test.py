"""Contract tests for the NB09 optimization public-API tutorial."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "tutorial"
    / "primary"
    / "09_optimization_stochastic_computation_public_api_walkthrough.ipynb"
)


def _notebook_sources() -> tuple[list[str], str]:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    sources = [cell.source for cell in notebook.cells]
    return sources, "\n".join(sources)


def test_nb09_uses_live_optimization_public_api() -> None:
    sources, all_source = _notebook_sources()

    assert "relic.optimization.manager" not in all_source
    assert "OptimizationManager" not in all_source
    assert "get_optimization_manager" not in all_source
    assert "from relic.parameters import get_global_manager" in all_source
    assert "enable_jit=" not in all_source

    inventory_cell = next(
        source for source in sources if "optimizer_inventory =" in source
    )
    assert "component_available" in inventory_cell
    assert "runtime_ready" in inventory_cell
    assert 'meta.get("available")' not in inventory_cell


def test_nb09_injects_parameter_manager_into_calibrator() -> None:
    _, all_source = _notebook_sources()

    assert "ParameterCalibrator(\n    parameter_manager=get_global_manager()," in all_source
    assert "optimization_component_available_count" in all_source
    assert "optimization_runtime_ready_count" in all_source


def test_nb09_uses_current_bounded_computation_compile_api() -> None:
    _, all_source = _notebook_sources()

    assert "apply_compile" not in all_source
    assert "from relic.computation.jit import maybe_compile" in all_source
    assert "compiled_module = maybe_compile(" in all_source
    assert "mode='compile'" in all_source
    assert "backend='eager'" in all_source
    assert "'scientific_equivalence': False" in all_source
    assert "non_scientific_single_function_current_environment" in all_source
    assert "single_batch_current_environment_allclose" in all_source
    assert "BucketSoilMoisture" not in all_source


def test_nb09_uses_canonical_farquhar_ports_and_update_items() -> None:
    _, all_source = _notebook_sources()

    for key in (
        "air_temperature",
        "atmospheric_co2_ppm",
        "ppfd",
        "vapor_pressure_deficit",
        "water_stress_factor",
        "temperature_stress_factor",
        "ozone_stress_factor",
    ):
        assert f"'{key}':" in all_source
    for legacy_key in (
        "'temperature':",
        "'co2_concentration':",
        "'par':",
        "'vpd':",
        "'water_stress':",
        "'temperature_stress':",
        "'ozone_stress':",
    ):
        assert legacy_key not in all_source
    assert "torch.as_tensor(update.value" in all_source
    assert "['gpp']" not in all_source
    assert "['gross_photosynthesis']" in all_source


def test_nb09_consumes_scalar_ensemble_member_inputs() -> None:
    _, all_source = _notebook_sources()

    assert "inputs['vcmax25'][0]" not in all_source
    assert "inputs['jmax25'][0]" not in all_source
    assert "inputs['temperature_offset'][0]" not in all_source
    assert "torch.as_tensor(inputs['vcmax25']).item()" in all_source
    assert "torch.as_tensor(inputs['jmax25']).item()" in all_source


def test_nb09_declares_explicit_sde_operators() -> None:
    _, all_source = _notebook_sources()

    assert "process_type=" not in all_source
    assert "set_process_parameters" not in all_source
    assert "def decomposition_drift(" in all_source
    assert "def decomposition_diffusion(" in all_source
    assert "drift_func=decomposition_drift" in all_source
    assert "diffusion_func=decomposition_diffusion" in all_source
    assert "# Units:" in all_source
    assert "# Sign convention:" in all_source
    assert "# Open-system route:" in all_source
