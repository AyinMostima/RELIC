import pytest
import torch

from relic.core.state import StateWriteMode, UpdateItem, UpdateMode
from relic.coupling.observation.quality import PostObservationQualityOperator
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.validation.framework.rules.observation_diagnostic import (
    validate_observation_diagnostic_payload,
)


def _config() -> dict:
    return {
        "instrument_contract_parameter_prefix": (
            "coupling.observation.tower_concentration."
            "profile_004_synthetic_qc"
        )
    }


def test_qc_loads_versioned_instrument_contract_from_parameter_prefix():
    prefix = (
        "coupling.observation.tower_concentration."
        "profile_004_synthetic_qc"
    )
    operator = PostObservationQualityOperator(
        config={"instrument_contract_parameter_prefix": prefix}
    )

    assert set(operator.metadata.parameters) == {
        f"{prefix}.detection_limit_ppbv",
        f"{prefix}.valid_min_ppbv",
        f"{prefix}.valid_max_ppbv",
        f"{prefix}.base_weight",
    }
    torch.testing.assert_close(
        operator.detection_limit_ppbv,
        torch.full((3,), 0.001, dtype=torch.float64),
    )
    torch.testing.assert_close(
        operator.valid_min_ppbv,
        torch.zeros(3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        operator.valid_max_ppbv,
        torch.full((3,), 1000.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        operator.base_weight,
        torch.ones(3, dtype=torch.float64),
    )


def _state(
    raw: torch.Tensor,
    *,
    instrument_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    del raw
    return {
        PostObservationQualityOperator.INSTRUMENT_VALID_INPUT: (
            instrument_valid
            if instrument_valid is not None
            else torch.ones(3, dtype=torch.float64)
        ),
    }


def _dependency_outputs(
    raw: torch.Tensor,
    *,
    complete: torch.Tensor | None = None,
) -> dict[str, dict[str, UpdateItem]]:
    completion = (
        complete
        if complete is not None
        else torch.ones(3, dtype=raw.dtype)
    )
    return {
        "tower_concentration_profile_operator": {
            PostObservationQualityOperator.RAW_PPBV: UpdateItem(
                value=raw,
                mode=UpdateMode.REPLACE,
                write_mode=StateWriteMode.DERIVED,
                unit="ppbv",
                per_timestep=False,
            ),
            PostObservationQualityOperator.WINDOW_COMPLETE_INPUT: UpdateItem(
                value=completion,
                mode=UpdateMode.REPLACE,
                write_mode=StateWriteMode.DERIVED,
                unit="1",
                per_timestep=False,
            ),
        }
    }


def test_qc_separates_raw_mask_flag_and_weight_without_gap_fill():
    operator = PostObservationQualityOperator(config=_config())
    raw = torch.tensor(
        [1.0, 0.0001, 1200.0],
        dtype=torch.float64,
    )
    original = raw.clone()

    outputs = operator.forward(
        state=_state(raw),
        time=0.0,
        dt=1800.0,
        subprocess_outputs=_dependency_outputs(raw),
    )

    assert set(outputs) == {
        operator.MASK,
        operator.FLAG,
        operator.WEIGHT,
    }
    torch.testing.assert_close(raw, original, rtol=0.0, atol=0.0)
    assert operator.RAW_PPBV not in outputs
    assert torch.equal(
        outputs[operator.MASK].value,
        torch.tensor([True, False, False]),
    )
    assert torch.equal(
        outputs[operator.FLAG].value,
        torch.tensor([0, 2, 4], dtype=torch.int64),
    )
    torch.testing.assert_close(
        outputs[operator.WEIGHT].value,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )
    assert all(
        update.write_mode == StateWriteMode.DERIVED
        for update in outputs.values()
    )


def test_qc_flags_nonfinite_incomplete_and_instrument_invalid_independently():
    operator = PostObservationQualityOperator(config=_config())
    raw = torch.tensor(
        [float("nan"), 1.0, 1.0],
        dtype=torch.float64,
    )

    outputs = operator.forward(
        state=_state(
            raw,
            instrument_valid=torch.tensor([1.0, 0.0, 0.0]),
        ),
        time=0.0,
        dt=1800.0,
        subprocess_outputs=_dependency_outputs(
            raw,
            complete=torch.tensor([1.0, 0.0, 1.0]),
        ),
    )

    assert torch.equal(
        outputs[operator.FLAG].value,
        torch.tensor([1, 24, 16], dtype=torch.int64),
    )
    assert torch.equal(
        outputs[operator.MASK].value,
        torch.tensor([False, False, False]),
    )
    torch.testing.assert_close(
        outputs[operator.WEIGHT].value,
        torch.zeros(3, dtype=torch.float64),
    )
    assert torch.isnan(raw[0])


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({}, "instrument_contract"),
        (
            {
                "instrument_contract": {
                    "detection_limit_ppbv": [0.2],
                    "valid_min_ppbv": [0.0, 0.0, 0.0],
                    "valid_max_ppbv": [10.0, 10.0, 10.0],
                    "base_weight": [1.0, 1.0, 1.0],
                }
            },
            "no inline",
        ),
        ({"instrument_contract_parameter_prefix": "  "}, "non-empty"),
    ],
)
def test_qc_requires_versioned_instrument_parameter_prefix(config, match):
    with pytest.raises((ValueError, KeyError), match=match):
        PostObservationQualityOperator(config=config)


def test_qc_diagnostic_contract_freezes_post_observation_scope():
    operator = PostObservationQualityOperator(config=_config())
    diagnostic = operator.diagnostic_operator()

    assert diagnostic.writes_state is False
    assert diagnostic.metadata["raw_modified"] is False
    assert diagnostic.metadata["gap_filling"] is False
    assert diagnostic.metadata["flag_masks"] == {
        "nonfinite": 1,
        "below_detection": 2,
        "outside_valid_range": 4,
        "incomplete_window": 8,
        "instrument_invalid": 16,
    }
    assert validate_observation_diagnostic_payload(
        [],
        [diagnostic],
    ).passed


def test_qc_outputs_have_canonical_variable_contracts():
    operator = PostObservationQualityOperator(config=_config())
    catalog = SSOTCatalog.from_parameter_manager()

    for output_name in operator.metadata.output_variables:
        entry = catalog.get(output_name)
        assert entry is not None, output_name
        assert entry.write_mode == "derived"
        assert entry.shape == {"dimensions": ["height"]}


def test_qc_requires_only_external_instrument_state():
    operator = PostObservationQualityOperator(config=_config())

    assert operator.get_required_inputs() == [
        operator.INSTRUMENT_VALID_INPUT
    ]
    assert operator.optional_input_variables == {
        operator.RAW_PPBV,
        operator.WINDOW_COMPLETE_INPUT,
    }


def test_qc_fails_closed_without_declared_observation_dependency():
    operator = PostObservationQualityOperator(config=_config())
    raw = torch.ones(3, dtype=torch.float64)

    with pytest.raises(
        RuntimeError,
        match="tower_concentration_profile_operator",
    ):
        operator.forward(
            state=_state(raw),
            time=0.0,
            dt=1800.0,
            subprocess_outputs={},
        )
