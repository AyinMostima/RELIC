import torch

from pathlib import Path

import yaml

from relic.contracts.contract_loader import ContractRegistry
from relic.scientific.pedosphere.methane import (
    MethaneBudgetCoupler,
    MethaneOxidationFunctor,
    MethaneProductionFunctor,
    MethaneScaleCouplingFunctor,
    MethaneTransportFunctor,
)
from relic.core.state import UpdateItem, UpdateMode


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def apply_updates(state, updates):
    for key, value in updates.items():
        mode = UpdateMode.ACCUMULATE
        raw_value = value
        if isinstance(value, UpdateItem):
            raw_value = value.value
            mode = value.mode
        tensor = raw_value if isinstance(raw_value, torch.Tensor) else torch.as_tensor(float(raw_value))
        target_dtype = state[key].dtype if key in state else tensor.dtype
        target_device = state[key].device if key in state else tensor.device
        tensor = tensor.to(device=target_device, dtype=target_dtype)
        if key in state and mode != UpdateMode.REPLACE:
            state[key] = state[key] + tensor
        else:
            state[key] = tensor.clone()
    return state


def run_chain(state, dt):
    dt_tensor = torch.tensor(float(dt))
    functors = [
        MethaneScaleCouplingFunctor(),
        MethaneProductionFunctor(),
        MethaneOxidationFunctor(),
        MethaneTransportFunctor(),
        MethaneBudgetCoupler(),
    ]
    for functor in functors:
        updates = functor(state, time=0.0, dt=dt_tensor)
        apply_updates(state, updates)
    return state


def test_mass_balance_residual_within_tolerance():
    initial_state = {
        "soil_temperature": torch.tensor(294.0),
        "soil_ph": torch.tensor(6.8),
        "heterotrophic_respiration": torch.tensor(6.0),
        "redox_potential": torch.tensor(-180.0),
        "soil_co2_concentration": torch.tensor(0.2),
        "microbial_activity_index": torch.tensor(0.9),
        "ch4_soil_storage": torch.tensor(8.0),
        "land_surface.wetland_fraction": torch.tensor(0.8),
        "water_filled_pore_space": torch.tensor(0.85),
        "methane_data_score": torch.tensor(0.8),
        "root_length_density": torch.tensor(0.2),
        "soil_o2_concentration": torch.tensor(0.015),
        "porosity": torch.tensor(0.45),
        "atmospheric_ch4_concentration": torch.tensor(0.0),
    }

    storage_before = initial_state["ch4_soil_storage"].clone()
    run_chain(initial_state, dt=1800.0)

    residual = initial_state[
        "diagnostics.methane_budget.mass_balance_residual"
    ].item()
    emission = initial_state["ch4_atmospheric_flux"].item()
    storage_delta = initial_state["ch4_soil_storage"] - storage_before

    assert abs(residual) < 1e-6
    assert emission >= 0.0
    torch.testing.assert_close(
        storage_delta,
        initial_state["diagnostics.methane_budget.storage_tendency_net"] * 1800.0,
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert torch.isfinite(storage_delta).all()
    for removed_name in (
        "ch4_emission",
        "ch4_emission_mass",
        "ch4_mass_balance_residual",
        "ch4_storage_tendency_net",
        "ch4_soil_storage_delta",
    ):
        assert removed_name not in initial_state


def test_methane_production_public_ports_have_canonical_scm_contracts():
    contract_dir = PROJECT_ROOT / "relic" / "contracts" / "variables" / "pedosphere"
    expected_variables = {
        "ch4_production_rate": {
            "runtime_role": "methane_production_rate",
            "accumulates": "carbon.total",
            "allowed_in": ["pedosphere_methane_production"],
            "guardian": "pedosphere_methane_budget",
        },
        "ch4_storage_tendency_prod": {
            "runtime_role": "methane_storage_production_tendency",
            "accumulates": None,
            "allowed_in": [],
            "guardian": None,
        },
    }

    for variable, expectation in expected_variables.items():
        contract_path = contract_dir / f"{variable}.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        writer = contract["writers"]["tendencies"]

        assert contract["variable"] == variable
        assert contract["kind"] == "flux"
        assert contract["dimension"]["udunits"] == "g C m-2 s-1"
        assert contract["owner"] == "pedosphere_methane_production"
        assert contract["incrementable"] is False
        assert contract["accumulates"] == expectation["accumulates"]
        assert writer == [
            {
                "name": "pedosphere_methane_production",
                "module": "relic.scientific.pedosphere.methane.production",
                "provides": [variable],
                "unit": "g C m-2 s-1",
            }
        ]
        assert contract["flux_contract"]["allowed_in"] == expectation["allowed_in"]
        assert contract["flux_contract"]["allowed_out"] == []
        assert contract["flux_contract"]["budget_check"]["guardian"] == (
            expectation["guardian"]
        )
        assert contract["provenance"]["runtime_role"] == expectation["runtime_role"]
        assert contract["provenance"]["runtime_writer"] == (
            "pedosphere_methane_production"
        )

    storage_contract = yaml.safe_load(
        (contract_dir / "ch4_storage_tendency_prod.yaml").read_text(encoding="utf-8")
    )
    assert storage_contract["provenance"]["realized_state_owner"] == (
        "pedosphere_methane_budget"
    )


def test_methane_scale_outputs_have_canonical_diagnostic_scm_contracts():
    contract_dir = PROJECT_ROOT / "relic" / "contracts" / "variables" / "pedosphere"
    expected = {
        "methane_gating_weight": {
            "bounds": (0.0, 1.0),
            "runtime_role": "instantaneous_methane_pathway_gating_control",
        },
        "methane_gating_score": {
            "bounds": (0.0, 3.0),
            "runtime_role": (
                "instantaneous_methane_pathway_gating_score_observation"
            ),
        },
    }

    for variable, expectation in expected.items():
        contract = yaml.safe_load(
            (contract_dir / f"{variable}.yaml").read_text(encoding="utf-8")
        )

        assert contract["variable"] == variable
        assert contract["kind"] == "diagnostic"
        assert contract["dimension"]["udunits"] == "dimensionless"
        assert contract["owner"] == "pedosphere_methane_scale"
        assert contract["aliases"] == []
        assert contract["incrementable"] is False
        assert contract["accumulates"] is None
        assert (
            contract["numerics"]["lower_bound"],
            contract["numerics"]["upper_bound"],
        ) == expectation["bounds"]
        assert contract["writers"]["increments"] == []
        assert contract["writers"]["tendencies"] == [
            {
                "name": "pedosphere_methane_scale",
                "module": "relic.scientific.pedosphere.methane.scales",
                "provides": [variable],
                "unit": "dimensionless",
            }
        ]
        assert contract["flux_contract"]["allowed_in"] == []
        assert contract["flux_contract"]["allowed_out"] == []
        assert contract["flux_contract"]["budget_check"]["guardian"] is None
        assert contract["provenance"]["runtime_role"] == expectation["runtime_role"]
        assert contract["provenance"]["runtime_writer"] == (
            "pedosphere_methane_scale"
        )


def test_methane_oxidation_public_ports_have_canonical_scm_contracts():
    contract_dir = PROJECT_ROOT / "relic" / "contracts" / "variables" / "pedosphere"
    expected_variables = {
        "ch4_oxidation_rate": {
            "runtime_role": "methane_oxidation_export_rate",
            "lower_bound": 0.0,
            "upper_bound": None,
        },
        "ch4_storage_tendency_oxid": {
            "runtime_role": "methane_storage_oxidation_tendency",
            "lower_bound": None,
            "upper_bound": 0.0,
        },
    }

    for variable, expectation in expected_variables.items():
        contract_path = contract_dir / f"{variable}.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

        assert contract["variable"] == variable
        assert contract["kind"] == "flux"
        assert contract["dimension"]["udunits"] == "g C m-2 s-1"
        assert contract["owner"] == "pedosphere_methane_oxidation"
        assert contract["incrementable"] is False
        assert contract["accumulates"] == "carbon.total"
        assert contract["numerics"]["lower_bound"] == expectation["lower_bound"]
        assert contract["numerics"]["upper_bound"] == expectation["upper_bound"]
        assert contract["writers"]["tendencies"] == [
            {
                "name": "pedosphere_methane_oxidation",
                "module": "relic.scientific.pedosphere.methane.oxidation",
                "provides": [variable],
                "unit": "g C m-2 s-1",
            }
        ]
        assert contract["flux_contract"]["allowed_in"] == []
        assert contract["flux_contract"]["allowed_out"] == [
            "pedosphere_methane_oxidation"
        ]
        assert contract["flux_contract"]["budget_check"]["guardian"] == (
            "pedosphere_methane_budget"
        )
        assert contract["provenance"]["runtime_role"] == expectation["runtime_role"]
        assert contract["provenance"]["runtime_writer"] == (
            "pedosphere_methane_oxidation"
        )


def test_methane_transport_public_ports_have_canonical_scm_contracts():
    contract_dir = PROJECT_ROOT / "relic" / "contracts" / "variables" / "pedosphere"
    expected_variables = {
        "ch4_transport_loss_rate": {
            "runtime_role": "signed_requested_methane_transport_loss",
            "sign_convention": "positive_outflow_negative_inflow",
        },
        "ch4_storage_tendency_trans": {
            "runtime_role": "methane_storage_transport_tendency",
            "sign_convention": "negative_outflow_positive_inflow",
        },
    }

    for variable, expectation in expected_variables.items():
        contract_path = contract_dir / f"{variable}.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

        assert contract["variable"] == variable
        assert contract["kind"] == "flux"
        assert contract["dimension"]["udunits"] == "g C m-2 s-1"
        assert contract["owner"] == "pedosphere_methane_transport"
        assert contract["incrementable"] is False
        assert contract["accumulates"] is None
        assert contract["numerics"]["lower_bound"] is None
        assert contract["numerics"]["upper_bound"] is None
        assert contract["writers"]["increments"] == []
        assert contract["writers"]["tendencies"] == [
            {
                "name": "pedosphere_methane_transport",
                "module": "relic.scientific.pedosphere.methane.transport",
                "provides": [variable],
                "unit": "g C m-2 s-1",
            }
        ]
        assert contract["flux_contract"]["allowed_in"] == []
        assert contract["flux_contract"]["allowed_out"] == []
        assert contract["flux_contract"]["budget_check"]["guardian"] is None
        assert contract["provenance"]["runtime_role"] == expectation["runtime_role"]
        assert contract["provenance"]["runtime_writer"] == (
            "pedosphere_methane_transport"
        )
        assert contract["provenance"]["sign_convention"] == (
            expectation["sign_convention"]
        )
        assert contract["provenance"]["realized_boundary_owner"] == (
            "pedosphere_methane_budget"
        )


def test_methane_transport_inputs_have_external_canonical_scm_contracts():
    contract_root = PROJECT_ROOT / "relic" / "contracts" / "variables"
    expectations = {
        contract_root / "pedosphere" / "water_filled_pore_space.yaml": {
            "variable": "water_filled_pore_space",
            "unit": "1",
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "runtime_role": "external_water_filled_pore_space",
        },
        contract_root / "pedosphere" / "porosity.yaml": {
            "variable": "porosity",
            "unit": "m3 m-3",
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "runtime_role": "external_soil_porosity",
        },
        contract_root / "atmosphere" / "atmospheric_ch4_concentration.yaml": {
            "variable": "atmospheric_ch4_concentration",
            "unit": "g C m-3",
            "lower_bound": 0.0,
            "upper_bound": None,
            "runtime_role": "external_atmospheric_ch4_concentration",
        },
    }
    registry = ContractRegistry(strict=True)

    for contract_path, expectation in expectations.items():
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        registry.load_contract_file(contract_path)

        assert contract["variable"] == expectation["variable"]
        assert contract["kind"] == "forcing"
        assert contract["dimension"]["udunits"] == expectation["unit"]
        assert contract["owner"] is None
        assert contract["aliases"] == []
        assert contract["incrementable"] is False
        assert contract["accumulates"] is None
        assert contract["numerics"]["lower_bound"] == expectation["lower_bound"]
        assert contract["numerics"]["upper_bound"] == expectation["upper_bound"]
        assert contract["writers"] == {"increments": [], "tendencies": []}
        assert contract["flux_contract"]["allowed_in"] == []
        assert contract["flux_contract"]["allowed_out"] == []
        assert contract["provenance"]["runtime_role"] == expectation["runtime_role"]
        assert contract["provenance"]["runtime_writer"] is None
        assert not registry.can_write(
            expectation["variable"], "pedosphere_methane_transport"
        )


def test_methane_budget_public_ports_have_canonical_scm_contracts():
    contract_dir = PROJECT_ROOT / "relic" / "contracts" / "variables" / "pedosphere"

    atmospheric_contract = yaml.safe_load(
        (contract_dir / "ch4_atmospheric_flux.yaml").read_text(encoding="utf-8")
    )
    assert atmospheric_contract["variable"] == "ch4_atmospheric_flux"
    assert atmospheric_contract["kind"] == "flux"
    assert atmospheric_contract["dimension"]["udunits"] == "g C m-2 s-1"
    assert atmospheric_contract["owner"] == "pedosphere_methane_budget"
    assert atmospheric_contract["incrementable"] is False
    assert atmospheric_contract["accumulates"] == "carbon.total"
    assert atmospheric_contract["writers"]["increments"] == []
    assert atmospheric_contract["writers"]["tendencies"] == [
        {
            "name": "pedosphere_methane_budget",
            "module": "relic.scientific.pedosphere.methane.budget_coupler",
            "provides": ["ch4_atmospheric_flux"],
            "unit": "g C m-2 s-1",
        }
    ]
    assert atmospheric_contract["flux_contract"]["allowed_in"] == []
    assert atmospheric_contract["flux_contract"]["allowed_out"] == [
        "pedosphere_methane_budget"
    ]
    assert atmospheric_contract["flux_contract"]["budget_check"]["guardian"] == (
        "pedosphere_methane_budget"
    )
    assert atmospheric_contract["provenance"]["runtime_writer"] == (
        "pedosphere_methane_budget"
    )

    storage_contract = yaml.safe_load(
        (contract_dir / "ch4_soil_storage.yaml").read_text(encoding="utf-8")
    )
    assert storage_contract["variable"] == "ch4_soil_storage"
    assert storage_contract["kind"] == "state"
    assert storage_contract["dimension"]["udunits"] == "g C m-2"
    assert storage_contract["owner"] == "pedosphere_methane_budget"
    assert storage_contract["incrementable"] is True
    assert storage_contract["accumulates"] == "carbon.total"
    assert storage_contract["numerics"]["lower_bound"] == 0.0
    assert storage_contract["writers"]["increments"] == [
        {
            "name": "pedosphere_methane_budget",
            "module": "relic.scientific.pedosphere.methane.budget_coupler",
            "provides": ["ch4_soil_storage"],
            "unit": "g C m-2",
        }
    ]
    assert storage_contract["writers"]["tendencies"] == []
    assert storage_contract["flux_contract"]["allowed_in"] == [
        "pedosphere_methane_budget"
    ]
    assert storage_contract["flux_contract"]["allowed_out"] == [
        "pedosphere_methane_budget"
    ]
    assert storage_contract["flux_contract"]["budget_check"]["guardian"] == (
        "pedosphere_methane_budget"
    )
    assert storage_contract["provenance"]["runtime_writer"] == (
        "pedosphere_methane_budget"
    )

