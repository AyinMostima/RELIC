import pytest
import torch

import relic.scientific.pedosphere.methane.oxidation as oxidation_module
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific.pedosphere.methane import MethaneOxidationFunctor


CANONICAL_OUTPUTS = {
    "ch4_oxidation_rate",
    "ch4_storage_tendency_oxid",
    "diagnostics.methane_oxidation.kinetic_rate",
    "diagnostics.methane_oxidation.storage_limiter_residual",
}

REMOVED_OUTPUTS = {
    "ch4_oxidation_imaple_rate",
    "ch4_oxidation_dndc_rate",
    "diagnostics.methane_oxidation.imaple_rate",
    "diagnostics.methane_oxidation.calibrated_rate",
    "diagnostics.methane_oxidation.dndc_rate",
    "diagnostics.methane_oxidation.reaction_affinity",
    "diagnostics.methane_oxidation.entropy_production",
    "diagnostics.methane_oxidation.exergy_storage",
    "diagnostics.methane_oxidation.exergy_dissipation",
    "diagnostics.methane_oxidation.port_power",
    "diagnostics.methane_oxidation.boundary_residual",
    "diagnostics.methane_oxidation.generic_j_entropy_residual",
    "diagnostics.methane_oxidation.generic_m_energy_residual",
}

REMOVED_PARAMETERS = (
    "pedosphere.methane.oxidation.base_rate",
    "pedosphere.methane.oxidation.calibrated.maximum_rate",
    "pedosphere.methane.oxidation.calibrated.half_saturation_o2",
    "pedosphere.methane.oxidation.calibrated.eh_threshold",
    "pedosphere.methane.oxidation.calibrated.eh_slope",
)


def canonical_state():
    return {
        "ch4_soil_storage": torch.tensor(0.01, dtype=torch.float64),
        "soil_temperature": torch.tensor(285.15, dtype=torch.float64),
        "soil_o2_concentration": torch.tensor(0.02, dtype=torch.float64),
        "porosity": torch.tensor(0.5, dtype=torch.float64),
        "water_filled_pore_space": torch.tensor(1.0, dtype=torch.float64),
    }


def _riley_kinetic_rate(oxidation, state):
    # References: Riley et al. (2011), doi:10.5194/bg-8-1925-2011, Eq. 7 and
    #   Table 1; CLM4.5 Technical Note (Oleson et al. 2013), Eq. 19.11 and
    #   Table 19.1 (base temperature 12 C); Riley et al. (2011) Sect. 2.7 (Q10 1.9).
    # Mathematical expression: R = Rmax * C/(K_CH4 + C) * O2/(K_O2 + O2)
    #   * Q10^((T - T_base)/dT10); C = M / (M_C * z * phi * WFPS); O = R * z * M_C.
    # Parameter meanings: Rmax maximum oxidation rate; K_CH4, K_O2 half-saturation
    #   concentrations on the water-volume basis; z layer depth; phi porosity.
    # Units: R, Rmax [mol m-3 s-1]; C, O2, K [mol m-3 water]; z [m]; O [g C m-2 s-1].
    # Domain/constraints: saturated (inundated) layer, so the moisture factor is 1.
    # Sign convention: O is a positive methane-carbon sink.
    as_float = lambda value: float(value)
    carbon = oxidation.carbon_molar_mass
    depth = as_float(oxidation.layer_depth)
    water_volume = depth * state["porosity"].item() * state["water_filled_pore_space"].item()
    methane = state["ch4_soil_storage"].item() / carbon / water_volume
    oxygen = state["soil_o2_concentration"].item()
    temperature_factor = as_float(oxidation.q10) ** (
        (state["soil_temperature"].item() - as_float(oxidation.temperature_base))
        / oxidation.q10_interval
    )
    rate = (
        as_float(oxidation.maximum_rate)
        * methane
        / (as_float(oxidation.half_saturation_ch4) + methane)
        * oxygen
        / (as_float(oxidation.half_saturation_o2) + oxygen)
        * temperature_factor
    )
    return rate * depth * carbon


def test_oxidation_uses_physical_constants_ssot(monkeypatch):
    values = {
        "relic.physical.constants.temperature.q10_interval": 10.0,
        "relic.physical.constants.molecular_weights.carbon": 12.011,
    }
    calls = []

    def fake_get_physical_constant(name):
        calls.append(name)
        return values[name]

    monkeypatch.setattr(
        oxidation_module,
        "get_physical_constant",
        fake_get_physical_constant,
        raising=False,
    )

    oxidation = MethaneOxidationFunctor()

    assert calls == list(values)
    assert oxidation.q10_interval == pytest.approx(10.0)
    assert oxidation.carbon_molar_mass == pytest.approx(12.011)
    assert all(
        not key.startswith("relic.physical.constants.")
        for key in oxidation.metadata.parameters
    )


def test_oxidation_outputs_use_exact_canonical_contracts():
    oxidation = MethaneOxidationFunctor()
    updates = oxidation.forward(canonical_state(), time=0.0, dt=torch.tensor(1800.0))

    assert set(oxidation.metadata.output_variables) == CANONICAL_OUTPUTS
    assert set(updates) == CANONICAL_OUTPUTS
    assert not (set(updates) & REMOVED_OUTPUTS)
    for update in updates.values():
        assert isinstance(update, UpdateItem)
        assert update.mode == UpdateMode.REPLACE
        assert update.per_timestep is False
        assert update.unit == "g C m-2 s-1"
        assert update.source_functor == oxidation.name
    assert updates["ch4_oxidation_rate"].element == "C"
    assert updates["ch4_storage_tendency_oxid"].element == "C"


def test_oxidation_is_one_governed_riley_2011_path():
    # Round 51.2: the k_base branch, the calibrated Vmax branch, the logistic Eh
    # gate and the gating weight are removed; one CLM4Me rate law remains.
    oxidation = MethaneOxidationFunctor()
    metadata = oxidation.metadata
    manager = get_global_manager()

    assert set(metadata.input_variables) == set(canonical_state())
    assert "redox_potential" not in metadata.input_variables
    assert "methane_gating_weight" not in metadata.input_variables
    assert oxidation.get_dependencies() == []
    for key in REMOVED_PARAMETERS:
        assert key not in metadata.parameters, key
        assert manager.get(key, required=False) is None, key
    for name in ("calibrated_max_rate", "calibrated_eh_threshold", "calibrated_eh_slope", "kg_to_g"):
        assert not hasattr(oxidation, name), name
    references = " ".join(metadata.references)
    assert "Riley et al. (2011)" in references
    assert "CLM4.5 Technical Note" in references
    assert "Watson" not in references


@pytest.mark.parametrize(
    "key,expected_reference",
    [
        ("pedosphere.methane.oxidation.maximum_rate", "Riley et al. (2011)"),
        ("pedosphere.methane.oxidation.half_saturation_ch4", "Riley et al. (2011)"),
        ("pedosphere.methane.oxidation.half_saturation_o2", "Riley et al. (2011)"),
        ("pedosphere.methane.oxidation.q10", "Riley et al. (2011)"),
        ("pedosphere.methane.oxidation.temperature_base", "Oleson et al. (2013)"),
    ],
)
def test_oxidation_parameters_cite_the_checked_primary_source(key, expected_reference):
    spec = get_global_manager().get_parameter(key)

    assert spec.provenance["source"] == "literature"
    assert spec.provenance.get("literature_check")
    assert any(ref.startswith(expected_reference) for ref in spec.references)
    assert spec.validate_value(spec.value) == []


def test_oxidation_parameters_hold_the_riley_2011_baseline_values():
    oxidation = MethaneOxidationFunctor()

    assert float(oxidation.maximum_rate) == pytest.approx(1.25e-5)
    assert float(oxidation.half_saturation_ch4) == pytest.approx(5.0e-3)
    assert float(oxidation.half_saturation_o2) == pytest.approx(2.0e-2)
    assert float(oxidation.q10) == pytest.approx(1.9)
    assert float(oxidation.temperature_base) == pytest.approx(285.15)


@pytest.mark.parametrize("missing_key", sorted(canonical_state()))
def test_oxidation_fails_fast_when_required_input_is_missing(missing_key):
    oxidation = MethaneOxidationFunctor()
    state = canonical_state()
    state.pop(missing_key)

    with pytest.raises(KeyError, match=missing_key):
        oxidation.forward(state, time=0.0, dt=torch.tensor(1800.0))


def test_oxidation_rejects_non_tensor_input():
    oxidation = MethaneOxidationFunctor()
    state = canonical_state()
    state["soil_temperature"] = 285.15

    with pytest.raises(TypeError, match="soil_temperature"):
        oxidation.forward(state, time=0.0, dt=torch.tensor(1800.0))


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("ch4_soil_storage", -1.0),
        ("ch4_soil_storage", float("nan")),
        ("soil_temperature", float("nan")),
        ("soil_o2_concentration", -0.1),
        ("soil_o2_concentration", float("inf")),
        ("porosity", 0.0),
        ("porosity", 1.1),
        ("water_filled_pore_space", 0.0),
        ("water_filled_pore_space", 1.1),
    ],
)
def test_oxidation_rejects_invalid_scientific_inputs(key, bad_value):
    oxidation = MethaneOxidationFunctor()
    state = canonical_state()
    state[key] = torch.tensor(bad_value, dtype=torch.float64)

    with pytest.raises(ValueError, match=key):
        oxidation.forward(state, time=0.0, dt=torch.tensor(1800.0))


@pytest.mark.parametrize("bad_dt", [0.0, -1.0])
def test_oxidation_rejects_invalid_timestep(bad_dt):
    oxidation = MethaneOxidationFunctor()

    with pytest.raises(ValueError):
        oxidation.forward(canonical_state(), time=0.0, dt=torch.tensor(bad_dt))


@pytest.mark.parametrize(
    "storage,temperature,oxygen,wfps",
    [
        (0.01, 285.15, 0.02, 1.0),
        (0.002, 278.15, 0.005, 0.9),
        (0.05, 298.15, 0.1, 1.0),
    ],
)
def test_kinetic_rate_matches_riley_2011_equation_7(storage, temperature, oxygen, wfps):
    oxidation = MethaneOxidationFunctor()
    state = canonical_state()
    state["ch4_soil_storage"] = torch.tensor(storage, dtype=torch.float64)
    state["soil_temperature"] = torch.tensor(temperature, dtype=torch.float64)
    state["soil_o2_concentration"] = torch.tensor(oxygen, dtype=torch.float64)
    state["water_filled_pore_space"] = torch.tensor(wfps, dtype=torch.float64)

    updates = oxidation.forward(state, time=0.0, dt=torch.tensor(60.0))

    expected = _riley_kinetic_rate(oxidation, state)
    assert updates["diagnostics.methane_oxidation.kinetic_rate"].value.item() == (
        pytest.approx(expected, rel=1.0e-6)
    )
    assert updates["ch4_oxidation_rate"].value.item() == pytest.approx(
        min(expected, storage / 60.0), rel=1.0e-6
    )


def test_temperature_factor_is_one_at_the_clm_base_temperature():
    oxidation = MethaneOxidationFunctor()
    state = canonical_state()
    at_base = oxidation.forward(state, time=0.0, dt=torch.tensor(60.0))
    state["soil_temperature"] = torch.tensor(295.15, dtype=torch.float64)
    warmer = oxidation.forward(state, time=0.0, dt=torch.tensor(60.0))

    ratio = (
        warmer["diagnostics.methane_oxidation.kinetic_rate"].value.item()
        / at_base["diagnostics.methane_oxidation.kinetic_rate"].value.item()
    )
    assert ratio == pytest.approx(1.9, rel=1.0e-6)


def test_oxidation_requires_oxygen():
    oxidation = MethaneOxidationFunctor()
    state = canonical_state()
    state["soil_o2_concentration"] = torch.tensor(0.0, dtype=torch.float64)

    updates = oxidation.forward(state, time=0.0, dt=torch.tensor(1800.0))

    assert updates["ch4_oxidation_rate"].value.item() == 0.0


def test_storage_limiter_is_explicit_and_preserves_local_rate_closure():
    # Riley et al. (2011) Sect. 2.1.7: a demand larger than the available CH4 is
    # scaled to the available amount; the unmet demand is reported, not lost carbon.
    oxidation = MethaneOxidationFunctor()
    state = canonical_state()
    state["ch4_soil_storage"] = torch.tensor(2.0e-3, dtype=torch.float64)
    dt = torch.tensor(1.0e9, dtype=torch.float64)

    updates = oxidation.forward(state, time=0.0, dt=dt)

    rate = updates["ch4_oxidation_rate"].value
    tendency = updates["ch4_storage_tendency_oxid"].value
    kinetic = updates["diagnostics.methane_oxidation.kinetic_rate"].value
    residual = updates["diagnostics.methane_oxidation.storage_limiter_residual"].value
    assert rate.item() == pytest.approx(2.0e-12, rel=1.0e-6, abs=1.0e-20)
    assert residual.item() > 0.0
    torch.testing.assert_close(kinetic - rate, residual)
    torch.testing.assert_close(rate + tendency, torch.zeros_like(rate))


def test_conservation_check_covers_every_grid_element():
    oxidation = MethaneOxidationFunctor()
    state = {key: torch.stack((value, value)) for key, value in canonical_state().items()}
    updates = oxidation.forward(state, time=0.0, dt=torch.tensor(1800.0))
    assert oxidation.check_conservation_laws(state, updates)["mass"] == 0.0

    tendency = updates["ch4_storage_tendency_oxid"]
    tampered_value = tendency.value.clone()
    tampered_value[1] += 1.0
    tampered = dict(updates)
    tampered["ch4_storage_tendency_oxid"] = UpdateItem(
        value=tampered_value,
        mode=tendency.mode,
        unit=tendency.unit,
        per_timestep=tendency.per_timestep,
        element=tendency.element,
        source_functor=tendency.source_functor,
    )
    assert oxidation.check_conservation_laws(state, tampered)["mass"] == pytest.approx(1.0)
