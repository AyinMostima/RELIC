"""L1 canonical contract tests for soil-vegetation coupling."""

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling
from relic.parameters import get_global_manager


CANONICAL_OUTPUTS = {
    "root_uptake_profile",
    "effective_transpiration",
    "diagnostics.soil_vegetation.feddes_stress_factor",
    "diagnostics.soil_vegetation.root_zone_moisture",
    "diagnostics.soil_vegetation.root_distribution",
    "diagnostics.soil_vegetation.water_balance_residual",
}

EXPECTED_UNITS = {
    "root_uptake_profile": "kg m-3 s-1",
    "effective_transpiration": "kg m-2 s-1",
    "diagnostics.soil_vegetation.feddes_stress_factor": "dimensionless",
    "diagnostics.soil_vegetation.root_zone_moisture": "m3 m-3",
    "diagnostics.soil_vegetation.root_distribution": "m-1",
    "diagnostics.soil_vegetation.water_balance_residual": "kg m-2 s-1",
}


def _build_coupling() -> SoilVegetationCoupling:
    return SoilVegetationCoupling(
        name="soil_vegetation_contract",
        config={},
        parameter_manager=get_global_manager(),
    )


def _canonical_state() -> dict[str, torch.Tensor]:
    return {
        "soil_moisture_profile": torch.tensor([0.22, 0.20], dtype=torch.float32),
        "soil_matric_potential": torch.tensor([-1.2e5, -1.4e5], dtype=torch.float32),
        "potential_transpiration": torch.tensor(8.0e-4, dtype=torch.float32),
        "soil_depth_layers": torch.tensor([0.3, 1.0], dtype=torch.float32),
    }


def test_soil_vegetation_requires_canonical_inputs_without_defaults() -> None:
    coupling = _build_coupling()
    state = _canonical_state()

    for required_key in (
        "soil_moisture_profile",
        "soil_matric_potential",
        "potential_transpiration",
        "soil_depth_layers",
    ):
        incomplete = dict(state)
        incomplete.pop(required_key)
        with pytest.raises(ValueError, match=required_key):
            coupling.forward(incomplete, time=0.0, dt=900.0)


def test_soil_vegetation_runtime_accepts_absent_optional_root_density() -> None:
    coupling = _build_coupling()

    outputs = coupling(_canonical_state(), time=0.0, dt=900.0)

    assert set(outputs) == CANONICAL_OUTPUTS
    assert coupling.optional_input_variables == {"root_density_profile"}


def test_soil_vegetation_rejects_old_input_aliases() -> None:
    coupling = _build_coupling()
    state = _canonical_state()

    scalar_alias = dict(state)
    scalar_alias["soil_moisture"] = scalar_alias.pop("soil_moisture_profile").mean()
    with pytest.raises(ValueError, match="soil_moisture_profile"):
        coupling.forward(scalar_alias, time=0.0, dt=900.0)

    transpiration_alias = dict(state)
    transpiration_alias["transpiration_rate"] = transpiration_alias.pop(
        "potential_transpiration"
    )
    with pytest.raises(ValueError, match="potential_transpiration"):
        coupling.forward(transpiration_alias, time=0.0, dt=900.0)


def test_soil_vegetation_outputs_are_exact_explicit_updates() -> None:
    coupling = _build_coupling()
    outputs = coupling.forward(_canonical_state(), time=0.0, dt=900.0)

    assert set(coupling.metadata.output_variables) == CANONICAL_OUTPUTS
    assert set(outputs) == CANONICAL_OUTPUTS
    for key, update in outputs.items():
        assert isinstance(update, UpdateItem), key
        assert update.mode == UpdateMode.REPLACE
        assert update.per_timestep is False
        assert update.unit == EXPECTED_UNITS[key]
        assert update.source_functor == coupling.name

    for old_key in (
        "water_stress_factor",
        "soil_moisture_change",
        "root_zone_moisture",
        "root_distribution",
        "water_uptake",
        "nutrient_uptake",
    ):
        assert old_key not in outputs


def test_soil_vegetation_water_ledger_closes() -> None:
    coupling = _build_coupling()
    state = _canonical_state()
    outputs = coupling.forward(state, time=0.0, dt=900.0)

    depth = state["soil_depth_layers"]
    thickness = torch.diff(torch.cat([depth.new_zeros(1), depth]))
    uptake = outputs["root_uptake_profile"].value
    transpiration = outputs["effective_transpiration"].value
    residual = outputs[
        "diagnostics.soil_vegetation.water_balance_residual"
    ].value

    assert torch.allclose(torch.sum(uptake * thickness), transpiration, atol=1.0e-10)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-10)


def test_optimal_profile_meets_potential_transpiration_demand() -> None:
    """Normalized Feddes uptake must meet demand when every layer is optimal."""

    coupling = _build_coupling()
    state = _canonical_state()
    optimum_dry, optimum_wet = coupling.stress_params.optimal_range
    state["soil_matric_potential"] = torch.full_like(
        state["soil_matric_potential"],
        0.5 * (optimum_dry + optimum_wet),
    )

    outputs = coupling.forward(state, time=0.0, dt=900.0)

    assert torch.allclose(
        outputs["effective_transpiration"].value,
        state["potential_transpiration"],
        rtol=1.0e-6,
        atol=1.0e-12,
    )


def test_soil_vegetation_feddes_four_threshold_response() -> None:
    coupling = _build_coupling()
    wilt = coupling.stress_params.wilting_point
    optimum_dry, optimum_wet = coupling.stress_params.optimal_range
    anaerobic = coupling.stress_params.anaerobic_point
    potentials = torch.tensor(
        [
            wilt - 1.0,
            wilt,
            0.5 * (wilt + optimum_dry),
            optimum_dry,
            0.5 * (optimum_dry + optimum_wet),
            optimum_wet,
            0.5 * (optimum_wet + anaerobic),
            anaerobic,
            anaerobic + 1.0,
        ],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
        dtype=torch.float64,
    )

    actual = coupling.compute_water_stress_factor(potentials)

    assert torch.allclose(actual, expected, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("soil_moisture_profile", torch.tensor([0.22, float("nan")])),
        ("soil_matric_potential", torch.tensor([-1.2e5, float("inf")])),
        ("potential_transpiration", torch.tensor(float("nan"))),
        ("soil_depth_layers", torch.tensor([0.3, float("inf")])),
    ],
)
def test_soil_vegetation_rejects_nonfinite_inputs(
    key: str,
    bad_value: torch.Tensor,
) -> None:
    coupling = _build_coupling()
    state = _canonical_state()
    state[key] = bad_value

    with pytest.raises(ValueError, match="finite"):
        coupling.forward(state, time=0.0, dt=900.0)


def test_soil_vegetation_rejects_profile_length_mismatch() -> None:
    coupling = _build_coupling()
    state = _canonical_state()
    state["soil_matric_potential"] = torch.tensor([-1.2e5])

    with pytest.raises(ValueError, match="same number of layers"):
        coupling.forward(state, time=0.0, dt=900.0)


def test_soil_depth_layers_has_one_documented_meaning():
    """Every reader takes soil_depth_layers as n layer-bottom depths with the top at 0
    (layer thickness = diff(depths, prepend=0)); the catalog entry and the docstrings state
    the same meaning, not layer thicknesses and not n + 1 boundaries."""
    import inspect
    from pathlib import Path

    import yaml

    from relic.validation.framework.rules import coupling_contracts

    root = Path(__file__).resolve().parents[3]
    catalog = yaml.safe_load(
        (root / "relic" / "parameters" / "database" / "relic" / "coupling" / "scale_variables.yaml")
        .read_text(encoding="utf-8-sig")
    )
    entry = catalog["parameters"]["coupling.scales.defaults.variables"]["value"]["soil_depth_layers"]
    description = str(entry["description"]).lower()
    assert "bottom depth" in description
    assert not description.startswith("soil layer thicknesses")
    for source in (
        inspect.getsource(SoilVegetationCoupling),
        inspect.getsource(coupling_contracts.validate_soil_vegetation_coupling),
    ):
        assert "Soil layer thicknesses" not in source
        assert "layer boundaries" not in source
        assert "layer_depth_i" not in source
