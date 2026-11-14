"""
Regression tests for BVOC functor-level conservation reporting.
"""

import pytest
import torch

from relic.scientific.biosphere.bvoc.synthesis.isoprene import IsopreneSynthesis
from relic.scientific.biosphere.bvoc.synthesis.monoterpene import MonoterpeneSynthesis
from relic.scientific.biosphere.bvoc.synthesis.sesquiterpene import (
    SesquiterpeneSynthesis,
)


def _monoterpene_state():
    """Return one deterministic stored-monoterpene state."""

    return {
        "air_temperature": torch.tensor(303.15),
        "gross_photosynthesis": torch.tensor(20.0),
        "monoterpene_pool_concentration": torch.tensor(50.0),
    }


def test_isoprene_open_port_reports_carbon_mapping_residual():
    model = IsopreneSynthesis()
    state = {
        "air_temperature": torch.tensor(298.15),
        "gross_photosynthesis": torch.tensor(20.0),
        "electron_transport_rate": torch.tensor(150.0),
        "atmospheric_co2_ppm": torch.tensor(300.0),
        "leaf_area_index": torch.tensor(1.0),
    }
    updates = model.forward(state, time=0.0, dt=3600.0)

    residuals = model.check_conservation_laws(state, updates)
    assert residuals["carbon_boundary"] == pytest.approx(0.0, abs=1.0e-15)

    perturbed_updates = dict(updates)
    carbon_allocation = updates["bvoc_carbon_allocation_flux"].value
    perturbed_updates["bvoc_carbon_allocation_flux"] = (
        carbon_allocation + torch.full_like(carbon_allocation, 2.0e-9)
    )
    perturbed = model.check_conservation_laws(state, perturbed_updates)
    assert perturbed["carbon_boundary"] == pytest.approx(2.0e-9, rel=1.0e-5)


def test_sesquiterpene_open_port_reports_carbon_mapping_residual():
    model = SesquiterpeneSynthesis()
    state = {
        "temperature": torch.tensor(303.15),
        "leaf_area_index": torch.tensor(1.0),
    }
    updates = model.forward(state, time=0.0, dt=60.0)

    residuals = model.check_conservation_laws(state, updates)
    assert residuals["carbon_boundary"] == pytest.approx(0.0, abs=1.0e-15)

    perturbed_updates = dict(updates)
    carbon_allocation = updates["bvoc_carbon_allocation_flux"].value
    perturbed_updates["bvoc_carbon_allocation_flux"] = (
        carbon_allocation + torch.full_like(carbon_allocation, 2.0e-9)
    )
    perturbed = model.check_conservation_laws(state, perturbed_updates)
    assert perturbed["carbon_boundary"] == pytest.approx(2.0e-9, rel=1.0e-5)


def test_monoterpene_open_port_reports_storage_carbon_residual():
    model = MonoterpeneSynthesis()
    state = _monoterpene_state()
    updates = model.forward(state, time=0.0, dt=60.0)

    assert model.metadata.conservation_laws == {
        "carbon_boundary": {"quantity": "carbon_boundary"}
    }
    residuals = model.check_conservation_laws(state, updates)
    assert residuals["carbon_boundary"] == pytest.approx(0.0, abs=1.0e-15)

    perturbed_updates = dict(updates)
    carbon_allocation = updates["bvoc_carbon_allocation_flux"].value
    perturbed_updates["bvoc_carbon_allocation_flux"] = (
        carbon_allocation + torch.full_like(carbon_allocation, 2.0e-9)
    )
    perturbed = model.check_conservation_laws(state, perturbed_updates)
    assert perturbed["carbon_boundary"] == pytest.approx(2.0e-9, rel=1.0e-5)


@pytest.mark.parametrize(
    "missing_key",
    [
        "bvoc_carbon_allocation_flux",
        "monoterpenes_emission",
        "diagnostics.bvoc.monoterpene.degradation_rate",
        "diagnostics.bvoc.monoterpene.storage_change",
        "diagnostics.bvoc.monoterpene.pool_boundary_residual_rate",
    ],
)
def test_monoterpene_carbon_checker_requires_complete_ledger(missing_key):
    model = MonoterpeneSynthesis()
    state = _monoterpene_state()
    updates = model.forward(state, time=0.0, dt=60.0)
    malformed = dict(updates)
    malformed.pop(missing_key)

    with pytest.raises(KeyError, match=missing_key):
        model.check_conservation_laws(state, malformed)


@pytest.mark.parametrize(
    ("invalid_value", "expected_exception"),
    [
        ("not-a-tensor", TypeError),
        (torch.empty(0), ValueError),
        (torch.tensor(float("nan")), ValueError),
    ],
)
def test_monoterpene_carbon_checker_rejects_invalid_values(
    invalid_value,
    expected_exception,
):
    model = MonoterpeneSynthesis()
    state = _monoterpene_state()
    updates = model.forward(state, time=0.0, dt=60.0)
    malformed = dict(updates)
    malformed["bvoc_carbon_allocation_flux"] = invalid_value

    with pytest.raises(expected_exception):
        model.check_conservation_laws(state, malformed)


def test_monoterpene_carbon_checker_rejects_shape_mismatch():
    model = MonoterpeneSynthesis()
    state = _monoterpene_state()
    updates = model.forward(state, time=0.0, dt=60.0)
    malformed = dict(updates)
    malformed["bvoc_carbon_allocation_flux"] = torch.ones(2)

    with pytest.raises(ValueError, match="identical shapes"):
        model.check_conservation_laws(state, malformed)


def test_monoterpene_diagnostic_requires_the_canonical_carbon_port():
    model = MonoterpeneSynthesis()
    with pytest.raises(KeyError, match="bvoc_carbon_allocation_flux"):
        model.diagnose_conservation(
            {},
            {"monoterpenes_emission": torch.tensor(1.0e-10)},
        )


def test_bvoc_open_ports_require_their_declared_local_evidence():
    isoprene_model = IsopreneSynthesis()
    monoterpene_model = MonoterpeneSynthesis()

    with pytest.raises(KeyError, match="isoprene_emission"):
        isoprene_model.check_conservation_laws({}, {})
    with pytest.raises(KeyError, match="bvoc_carbon_allocation_flux"):
        monoterpene_model.check_conservation_laws({}, {})
