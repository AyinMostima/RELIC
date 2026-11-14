"""Dependency-injected phenology observation aggregation.

Scientific phenology equations are owned by ``StaticPhenology``,
``GDDPhenology``, ``GSIPhenology``, and ``CropPhenology`` in ``models.py``.
This module does not implement or select equations at runtime. Graph
configuration selects one canonical owner, and the existing simulator injects
that owner's output through ``subprocess_outputs``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from relic.core.functor import (
    BiologicalProcessFunctor,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from relic.core.state import UpdateItem, UpdateMode


class PhenologyFunctor(BiologicalProcessFunctor):
    """Project one canonical phenology owner's outputs into diagnostics."""

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"

    _CANONICAL_MODEL_OWNERS = frozenset(
        {
            "static_phenology",
            "gdd_phenology",
            "gsi_phenology",
            "crop_phenology",
        }
    )
    _OBSERVATION_PROJECTION = {
        "phenological_factor": "diagnostics.phenology_state",
        "phenology_phase": "diagnostics.growth_stage",
        "leaf_area_index": "diagnostics.leaf_area_index",
        "phenology_senescence_factor": (
            "diagnostics.phenology.senescence_factor"
        ),
    }
    _OBSERVATION_UNITS = {
        "diagnostics.phenology_state": "dimensionless",
        "diagnostics.growth_stage": "dimensionless",
        "diagnostics.leaf_area_index": "m2 m-2",
        "diagnostics.phenology.senescence_factor": "dimensionless",
    }

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize a compile-time-bound phenology observation aggregator.

        ``config.canonical_model_owner`` must name one already registered
        canonical phenology Functor. Selection is immutable after construction.
        """

        if not isinstance(config, dict) or not config.get("canonical_model_owner"):
            raise ValueError(
                "config.canonical_model_owner is required for PhenologyFunctor."
            )
        canonical_model_owner = str(config["canonical_model_owner"]).strip()
        if canonical_model_owner not in self._CANONICAL_MODEL_OWNERS:
            allowed = ", ".join(sorted(self._CANONICAL_MODEL_OWNERS))
            raise ValueError(
                "Unsupported canonical phenology model owner "
                f"'{canonical_model_owner}'; expected one of: {allowed}."
            )

        metadata = FunctorMetadata(
            name="biosphere_phenology_functor",
            version="2.0.0",
            description=(
                "Parameter-free observation aggregator for one compile-time-selected "
                "canonical phenology model owner."
            ),
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="z_h = P_owner y_h",
            references=[
                "White et al. (1997) Global Biogeochemical Cycles 11:217-234",
                "Jolly et al. (2005) Global Change Biology 11:619-632",
                "Richardson et al. (2013) Agricultural and Forest Meteorology 169:156-173",
            ],
            parameters={},
            input_variables=[
                "leaf_area_index",
                "phenological_factor",
                "phenology_senescence_factor",
                "phenology_phase",
            ],
            output_variables=[
                "diagnostics.phenology_state",
                "diagnostics.growth_stage",
                "diagnostics.leaf_area_index",
                "diagnostics.phenology.senescence_factor",
            ],
            conservation_laws=[],
            temporal_scale=TemporalScale.DAY,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
            coupling_interfaces=["biosphere.phenology.canonical_model_owner"],
        )
        super().__init__(
            name="biosphere_phenology_functor",
            config=dict(config),
            metadata=metadata,
            device=device,
        )
        self.canonical_model_owner = canonical_model_owner

    def get_dependencies(self) -> dict[str, list[str]]:
        """Declare the single compile-time-selected canonical owner."""

        return self._merge_declared_dependencies({"required": [self.canonical_model_owner], "optional": []})

    @staticmethod
    def _validate_scalar_control(value: Tensor | float, name: str) -> None:
        """Require a finite scalar control value and positive ``dt``."""

        tensor = torch.as_tensor(value)
        if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be a finite scalar.")
        if name == "dt" and not bool(tensor > 0):
            raise ValueError("dt must be a positive finite scalar.")

    def _required_observation(
        self,
        owner_output: Mapping[str, Any],
        observation_name: str,
    ) -> Tensor:
        """Return one finite canonical-owner observation without fallback."""

        if observation_name not in owner_output:
            raise KeyError(
                f"required observation '{observation_name}' is missing from "
                f"dependency output '{self.canonical_model_owner}'"
            )
        value = owner_output[observation_name]
        if isinstance(value, UpdateItem):
            value = value.value
        if isinstance(value, Tensor):
            tensor = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        else:
            raise TypeError(
                f"required observation '{observation_name}' from "
                f"'{self.canonical_model_owner}' must be numeric"
            )
        if tensor.numel() == 0:
            raise ValueError(
                f"required observation '{observation_name}' from "
                f"'{self.canonical_model_owner}' must not be empty"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"required observation '{observation_name}' from "
                f"'{self.canonical_model_owner}' contains non-finite values"
            )
        return tensor

    def forward(
        self,
        state: Mapping[str, Any],
        time: Tensor | float,
        dt: Tensor | float,
        subprocess_outputs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, UpdateItem]:
        """Project dependency-injected canonical observations.

        References:
          - White et al. (1997), Jolly et al. (2005), and Richardson et al.
            (2013) define the observations evaluated by canonical model owners.
        Mathematical expression:
          z_h = P_owner y_h, with no prognostic state advancement.
        Parameter meanings:
          y_h: selected owner output; P_owner: parameter-free key projection;
          z_h: diagnostic observation vector; dt: positive step duration [s].
        Units:
          leaf_area_index [m2 m-2]; all other projected controls
          [dimensionless]; dt [s].
        Open system route:
          observation only; no matter transfer or prognostic state advance.
        """

        # References: White et al. (1997); Jolly et al. (2005).
        # Mathematical expression: z_h = P_owner y_h.
        # Parameter meanings: P_owner is a key projection and dt uses seconds.
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping.")
        self._validate_scalar_control(time, "time")
        self._validate_scalar_control(dt, "dt")
        if not isinstance(subprocess_outputs, Mapping):
            raise TypeError("subprocess_outputs must be a mapping.")
        if self.canonical_model_owner not in subprocess_outputs:
            raise KeyError(
                "declared dependency output "
                f"'{self.canonical_model_owner}' is required"
            )
        owner_output = subprocess_outputs[self.canonical_model_owner]
        if not isinstance(owner_output, Mapping):
            raise TypeError(
                f"dependency output '{self.canonical_model_owner}' must be a mapping"
            )

        return {
            diagnostic_name: UpdateItem(
                value=self._required_observation(
                    owner_output,
                    observation_name,
                ),
                mode=UpdateMode.REPLACE,
                unit=self._OBSERVATION_UNITS[diagnostic_name],
                per_timestep=True,
                source_functor=self.name,
            )
            for observation_name, diagnostic_name in self._OBSERVATION_PROJECTION.items()
        }

    def __repr__(self) -> str:
        return (
            "PhenologyFunctor("
            f"canonical_model_owner='{self.canonical_model_owner}', "
            f"device={self.device})"
        )


__all__ = ["PhenologyFunctor"]
