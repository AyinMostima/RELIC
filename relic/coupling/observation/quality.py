"""Non-destructive quality diagnostics applied after observation operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from ...computation.operators.dec.forms import DiagnosticSpec
from ...core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    VariableKind,
)
from ...core.state import StateWriteMode, UpdateItem, UpdateMode
from .tower_concentration import TowerConcentrationProfileOperator


class PostObservationQualityOperator(ScaleCouplingFunctor):
    """Classify raw concentration observations without modifying them."""

    RAW_PPBV = TowerConcentrationProfileOperator.RAW_PPBV
    WINDOW_COMPLETE_INPUT = (
        TowerConcentrationProfileOperator.WINDOW_COMPLETE
    )
    INSTRUMENT_VALID_INPUT = (
        "observation.tower_concentration.instrument_valid"
    )
    MASK = "diagnostics.observation.tower_concentration.mask"
    FLAG = "diagnostics.observation.tower_concentration.flag"
    WEIGHT = "diagnostics.observation.tower_concentration.weight"
    OBSERVATION_DEPENDENCY = "tower_concentration_profile_operator"

    FLAG_MASKS = {
        "nonfinite": 1,
        "below_detection": 2,
        "outside_valid_range": 4,
        "incomplete_window": 8,
        "instrument_invalid": 16,
    }
    _HEIGHT_COUNT = 3
    _CONTRACT_FIELDS = (
        "detection_limit_ppbv",
        "valid_min_ppbv",
        "valid_max_ppbv",
        "base_weight",
    )

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        resolved_config = dict(config or {})
        # The FunctorBase-owned dependency declaration is not an instrument
        # threshold; every other key besides the contract prefix is rejected.
        if set(resolved_config) - {"dependencies"} != {
            "instrument_contract_parameter_prefix"
        }:
            raise ValueError(
                "PostObservationQualityOperator requires an "
                "instrument_contract_parameter_prefix and no inline "
                "instrument thresholds."
            )
        parameter_prefix = resolved_config[
            "instrument_contract_parameter_prefix"
        ]
        if (
            not isinstance(parameter_prefix, str)
            or not parameter_prefix.strip()
        ):
            raise ValueError(
                "instrument_contract_parameter_prefix must be a "
                "non-empty string."
            )
        parameter_prefix = parameter_prefix.strip()
        parameter_keys = {
            field: f"{parameter_prefix}.{field}"
            for field in self._CONTRACT_FIELDS
        }

        functor_name = name or "post_observation_quality_operator"
        metadata = FunctorMetadata(
            name=functor_name,
            version="1.0.0",
            description=(
                "Non-destructive post-observation mask, bit flag, and "
                "weight classification."
            ),
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.POSTPROCESSING,
            mathematical_expression=(
                "mask_i = finite_i and detected_i and range_i and "
                "window_i and instrument_i; weight_i = base_i * mask_i"
            ),
            references=[
                "CF Metadata Conventions 1.8, quality flag attributes.",
                (
                    "WMO (2021), Guide to Instruments and Methods of "
                    "Observation, WMO-No. 8."
                ),
            ],
            parameters={
                key: (
                    "Versioned post-observation synthetic instrument "
                    f"contract field '{field}'"
                )
                for field, key in parameter_keys.items()
            },
            input_variables=[
                self.RAW_PPBV,
                self.WINDOW_COMPLETE_INPUT,
                self.INSTRUMENT_VALID_INPUT,
            ],
            output_variables=[self.MASK, self.FLAG, self.WEIGHT],
            conservation_laws=[],
            spatial_scale=SpatialScale.POINT,
            variable_kind=VariableKind.DIAGNOSTIC,
            coupling_interfaces=["post_observation_quality"],
        )
        super().__init__(
            name=functor_name,
            config=resolved_config,
            metadata=metadata,
            source_scale="tower_height",
            target_scale="tower_height",
            coupling_type="hierarchical",
            conservation_properties=[],
        )
        self.optional_input_variables = {
            self.RAW_PPBV,
            self.WINDOW_COMPLETE_INPUT,
        }
        contract = {
            field: self._contract_vector(
                self.get_parameter_value(key, required=True),
                field=field,
                parameter_key=key,
            )
            for field, key in parameter_keys.items()
        }
        self._validate_instrument_contract(contract)
        for field, value in contract.items():
            self.register_buffer(field, value)

    @classmethod
    def _contract_vector(
        cls,
        value: Any,
        *,
        field: str,
        parameter_key: str,
    ) -> Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if tensor.ndim != 1 or tensor.numel() != cls._HEIGHT_COUNT:
            raise ValueError(
                f"Instrument parameter '{parameter_key}' for '{field}' "
                "must contain three values."
            )
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(
                f"Instrument parameter '{parameter_key}' for '{field}' "
                "must be finite."
            )
        return tensor

    @staticmethod
    def _validate_instrument_contract(
        contract: Mapping[str, Tensor],
    ) -> None:
        detection = contract["detection_limit_ppbv"]
        valid_min = contract["valid_min_ppbv"]
        valid_max = contract["valid_max_ppbv"]
        base_weight = contract["base_weight"]
        if bool(torch.any(detection < 0)) or bool(torch.any(valid_min < 0)):
            raise ValueError(
                "Detection limits and valid minima must be non-negative."
            )
        if bool(torch.any(valid_max <= valid_min)):
            raise ValueError(
                "Each valid maximum must exceed its valid minimum."
            )
        if bool(torch.any(detection < valid_min)) or bool(
            torch.any(detection > valid_max)
        ):
            raise ValueError(
                "Each detection limit must lie inside the valid range."
            )
        if bool(torch.any(base_weight < 0)):
            raise ValueError("Base observation weights must be non-negative.")

    def diagnostic_operator(self) -> DiagnosticSpec:
        """Return the read-only post-observation quality contract."""

        return DiagnosticSpec(
            name="tower_concentration_post_observation_quality",
            input_state_vars=tuple(self.metadata.input_variables),
            output_units="1",
            diagnostic_kind="post_observation_quality",
            writes_state=False,
            metadata={
                "outputs": tuple(self.metadata.output_variables),
                "flag_masks": dict(self.FLAG_MASKS),
                "raw_modified": False,
                "gap_filling": False,
                "physical_observation_owner": self.OBSERVATION_DEPENDENCY,
            },
        )

    @classmethod
    def _require_height_vector(
        cls,
        state: Mapping[str, Any],
        key: str,
        *,
        allow_nonfinite: bool,
    ) -> Tensor:
        if key not in state:
            raise KeyError(f"Required quality input '{key}' is missing.")
        value = state[key]
        if isinstance(value, UpdateItem):
            value = value.value
        if not isinstance(value, Tensor):
            raise TypeError(f"Quality input '{key}' must be a torch.Tensor.")
        if value.ndim != 1 or value.numel() != cls._HEIGHT_COUNT:
            raise ValueError(
                f"Quality input '{key}' must contain three values."
            )
        if not allow_nonfinite and not bool(
            torch.all(torch.isfinite(value))
        ):
            raise ValueError(f"Quality input '{key}' must be finite.")
        return value

    @staticmethod
    def _require_binary(values: Tensor, label: str) -> Tensor:
        zero = torch.zeros((), dtype=values.dtype, device=values.device)
        one = torch.ones((), dtype=values.dtype, device=values.device)
        binary = torch.logical_or(values == zero, values == one)
        if not bool(torch.all(binary)):
            raise ValueError(f"{label} must contain only zero or one.")
        return values == one

    @classmethod
    def _require_observation_dependency(
        cls,
        subprocess_outputs: Mapping[str, Mapping[str, Any]] | None,
    ) -> Mapping[str, Any]:
        if not isinstance(subprocess_outputs, Mapping):
            raise RuntimeError(
                "Post-observation QC requires declared dependency outputs "
                f"from '{cls.OBSERVATION_DEPENDENCY}'."
            )
        dependency = subprocess_outputs.get(cls.OBSERVATION_DEPENDENCY)
        if not isinstance(dependency, Mapping):
            raise RuntimeError(
                "Post-observation QC requires declared dependency outputs "
                f"from '{cls.OBSERVATION_DEPENDENCY}'."
            )
        return dependency

    @staticmethod
    def _derived(
        *,
        value: Tensor,
        source_functor: str,
    ) -> UpdateItem:
        return UpdateItem(
            value=value,
            mode=UpdateMode.REPLACE,
            write_mode=StateWriteMode.DERIVED,
            unit="1",
            per_timestep=False,
            source_functor=source_functor,
        )

    def forward(
        self,
        state: dict[str, Tensor],
        time: float,
        dt: Any,
        subprocess_outputs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, UpdateItem]:
        """
        Classify three raw ppbv observations without changing physical values.

        References:
            CF Metadata Conventions 1.8; WMO-No. 8 (2021).
        Mathematical Expression:
            q_i = finite_i and x_i >= LOD_i and
            min_i <= x_i <= max_i and complete_i and instrument_i;
            w_i = w_base,i q_i.
        Parameters:
            x is raw concentration [ppbv]; LOD is the instrument detection
            limit [ppbv]; min/max are the site-valid range [ppbv];
            complete and instrument are binary gates [1]; w_base is the
            externally declared base weight [1].
        """
        # References: CF Metadata Conventions 1.8 (flag_masks); WMO-No. 8 (2021);
        #   module docstring; DiagnosticSpec from diagnostic_operator().
        # Mathematical expression: flag_i = OR of bits {nonfinite 1, x_i < LOD_i 2,
        #   x_i outside [min_i, max_i] 4, not complete_i 8, not instrument_i 16};
        #   q_i = (flag_i == 0); w_i = w_base,i q_i.
        # Parameter meanings: x_i raw window-mean concentration at height i (from the
        #   tower_concentration_profile_operator dependency); LOD_i detection limit;
        #   min_i, max_i valid range; complete_i, instrument_i gates; w_base,i base weight.
        # Units: x, LOD, min, max [ppbv]; complete, instrument, mask, flag, weight [1];
        #   outputs are REPLACE UpdateItems, StateWriteMode.DERIVED, unit "1",
        #   per_timestep=False; raw ppbv values are never modified.
        # Domain/constraints: subprocess_outputs must hold the dependency mapping; raw,
        #   complete, instrument each exactly three values; gates finite and binary;
        #   x may be non-finite (flagged); contract: LOD in [min, max], min >= 0, w >= 0.
        # Sign convention: not applicable: outputs are boolean masks, integer bit
        #   flags, and non-negative weights; no flux or gradient is formed.
        del time, dt, kwargs
        observation_outputs = self._require_observation_dependency(
            subprocess_outputs
        )
        raw = self._require_height_vector(
            observation_outputs,
            self.RAW_PPBV,
            allow_nonfinite=True,
        )
        complete_values = self._require_height_vector(
            observation_outputs,
            self.WINDOW_COMPLETE_INPUT,
            allow_nonfinite=False,
        ).to(dtype=raw.dtype, device=raw.device)
        instrument_values = self._require_height_vector(
            state,
            self.INSTRUMENT_VALID_INPUT,
            allow_nonfinite=False,
        ).to(dtype=raw.dtype, device=raw.device)
        complete = self._require_binary(
            complete_values,
            "Observation-window completion",
        )
        instrument_valid = self._require_binary(
            instrument_values,
            "Instrument validity",
        )
        detection = self.detection_limit_ppbv.to(
            dtype=raw.dtype,
            device=raw.device,
        )
        valid_min = self.valid_min_ppbv.to(
            dtype=raw.dtype,
            device=raw.device,
        )
        valid_max = self.valid_max_ppbv.to(
            dtype=raw.dtype,
            device=raw.device,
        )
        base_weight = self.base_weight.to(
            dtype=raw.dtype,
            device=raw.device,
        )

        finite = torch.isfinite(raw)
        below_detection = torch.logical_and(finite, raw < detection)
        outside_range = torch.logical_and(
            finite,
            torch.logical_or(raw < valid_min, raw > valid_max),
        )
        flags = torch.zeros(raw.shape, dtype=torch.int64, device=raw.device)
        flags = flags | torch.where(
            ~finite,
            self.FLAG_MASKS["nonfinite"],
            0,
        )
        flags = flags | torch.where(
            below_detection,
            self.FLAG_MASKS["below_detection"],
            0,
        )
        flags = flags | torch.where(
            outside_range,
            self.FLAG_MASKS["outside_valid_range"],
            0,
        )
        flags = flags | torch.where(
            ~complete,
            self.FLAG_MASKS["incomplete_window"],
            0,
        )
        flags = flags | torch.where(
            ~instrument_valid,
            self.FLAG_MASKS["instrument_invalid"],
            0,
        )
        mask = flags == 0
        weight = torch.where(mask, base_weight, torch.zeros_like(base_weight))
        return {
            self.MASK: self._derived(
                value=mask,
                source_functor=self.name,
            ),
            self.FLAG: self._derived(
                value=flags,
                source_functor=self.name,
            ),
            self.WEIGHT: self._derived(
                value=weight,
                source_functor=self.name,
            ),
        }


__all__ = ["PostObservationQualityOperator"]
