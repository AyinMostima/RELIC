"""Methane budget coupler enforcing mass conservation."""

import logging
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.core.abstractions import StateDict
from relic.scientific.pedosphere.api_contract import PedosphereContractMixin
from relic.utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)


class MethaneBudgetCoupler(PedosphereContractMixin, ScaleCouplingFunctor):
    """Aggregate methane production, oxidation, and transport into a conserved budget.

    Scientific references:
        - Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 (Eq.33)
        - Walter et al. (2001) doi:10.1029/2001JD900165

    Governing equation (ASCII form):
        F_requested = F_transport
        dM_raw = dt * (T_prod + T_oxid + T_trans)
        dM = max(dM_raw, -M)
        R_limit = (dM - dM_raw) / dt
        F_atm = F_requested - R_limit
        Residual = P - O - F_atm - dM/dt
        |Residual| <= max(abs_tol, rel_tol * max(P, O, F_atm))

    The storage limiter is a RELIC operator-splitting policy that preserves
    oxidation before reducing requested atmospheric transport. It is not
    presented as an additional literature methane-process law.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(
        self,
        name: str = "pedosphere_methane_budget",
        device: Optional[torch.device] = None,
    ) -> None:
        parameter_descriptions = {
            "pedosphere.methane.conservation.relative_tolerance": "Relative tolerance for methane mass balance [-]",
            "pedosphere.methane.conservation.absolute_tolerance": "Absolute tolerance for methane mass balance [g C m-2 s-1]",
            "pedosphere.numerical.minimum_timestep": "Minimum timestep for methane calculations [s]",
        }

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Enforce methane conservation across production, oxidation, and transport",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression=(
                "dM=max(dt*(T_prod+T_oxid+T_trans),-M); "
                "R_limit=(dM-dM_raw)/dt; F_atm=F_raw-R_limit; "
                "Residual=P-O-F_atm-dM/dt"
            ),
            references=[
                "Yue et al. (2024) doi:10.5194/gmd-17-4621-2024",
                "Walter et al. (2001) doi:10.1029/2001JD900165",
            ],
            parameters=parameter_descriptions,
            input_variables=[
                "ch4_production_rate",
                "ch4_oxidation_rate",
                "ch4_transport_loss_rate",
                "ch4_storage_tendency_prod",
                "ch4_storage_tendency_oxid",
                "ch4_storage_tendency_trans",
                "ch4_soil_storage",
            ],
            output_variables=[
                "ch4_atmospheric_flux",
                "ch4_soil_storage",
                "diagnostics.methane_budget.atmospheric_ch4_mass_flux",
                "diagnostics.methane_budget.mass_balance_residual",
                "diagnostics.methane_budget.storage_tendency_net",
                "diagnostics.methane_budget.transport_limiter_residual",
            ],
            conservation_laws=["mass"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
        )
        # The coupler joins the pedosphere soil CH4 ledger with the atmospheric
        # emission boundary (compartment coupling, as the interface mappers).
        super().__init__(
            name=name,
            metadata=metadata,
            device=device,
            source_scale="pedosphere",
            target_scale="atmosphere",
            coupling_type="spatial",
        )

        self.relative_tolerance = self.get_parameter_value(
            "relative_tolerance",
            namespace="pedosphere.methane.conservation", required=True)
        self.absolute_tolerance = self.get_parameter_value(
            "absolute_tolerance",
            namespace="pedosphere.methane.conservation", required=True)
        methane_molar_mass = float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.ch4"
            )
        )
        carbon_molar_mass = float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.carbon"
            )
        )
        self.dt_minimum = self.get_parameter_value(
            "minimum_timestep",
            namespace="pedosphere.numerical", required=True)
        self.c_to_ch4_mass_ratio = methane_molar_mass / carbon_molar_mass

        self.add_dependency("pedosphere_methane_production")
        self.add_dependency("pedosphere_methane_oxidation")
        self.add_dependency("pedosphere_methane_transport")

    @staticmethod
    def _scalar_from_value(value: Any) -> float:
        """Convert numeric values to Python float without tensor.item/cpu APIs."""
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, Tensor):
            detached = value.detach().reshape(-1)
            if detached.numel() == 0:
                raise ValueError("Cannot convert empty tensor to scalar.")
            return detached[:1].to(dtype=torch.float64).tolist()[0]
        return float(value)

    def forward(
        # References: Yue et al. (2024); Walter et al. (2001).
        # Mathematical expression: dM = max(dt (T_prod + T_oxid + T_trans), -M);
        #   R_limit = (dM - dM_raw)/dt; F_atm = F_raw - R_limit; Residual = P - O - F_atm - dM/dt.
        # Parameter meanings: M areal methane carbon storage; T_prod, T_oxid, T_trans the
        #   production, oxidation and transport tendencies proposed by their owners; F_raw and
        #   F_atm the requested and availability-limited atmospheric flux; R_limit the routed
        #   limiter residual.
        # Units: M, dM [g C m-2]; tendencies, P, O, F, R_limit, Residual [g C m-2 s-1]; the
        #   atmospheric port is also reported as [g CH4 m-2 s-1].
        # Domain/constraints: the coupler alone advances methane storage exactly once per step;
        #   storage cannot become negative; the limiter residual is routed, not discarded.
        # Sign convention: production positive into storage; oxidation and export positive out
        #   of storage; F_atm positive to the atmosphere.
        self,
        state: StateDict,
        time: float,
        dt: Tensor | float,
        **kwargs: Any,
    ) -> Dict[str, UpdateItem]:
        dt_value = (
            dt.to(self.device)
            if isinstance(dt, Tensor)
            else torch.as_tensor(dt, device=self.device)
        )
        if dt_value.numel() != 1 or not bool(torch.all(torch.isfinite(dt_value))):
            raise ValueError("Methane budget dt must be one finite scalar.")
        if bool(dt_value <= 0.0) or bool(dt_value < self.dt_minimum):
            raise ValueError(
                "Methane budget dt must be positive and meet the configured minimum."
            )

        production = self._require_state_tensor(
            state, "ch4_production_rate", lower_bound=0.0
        )
        oxidation = self._require_state_tensor(
            state, "ch4_oxidation_rate", lower_bound=0.0
        )
        transport_loss = self._require_state_tensor(
            state, "ch4_transport_loss_rate"
        )
        tendency_prod = self._require_state_tensor(
            state, "ch4_storage_tendency_prod"
        )
        tendency_oxid = self._require_state_tensor(
            state, "ch4_storage_tendency_oxid"
        )
        tendency_trans = self._require_state_tensor(
            state, "ch4_storage_tendency_trans"
        )
        storage = self._require_state_tensor(
            state, "ch4_soil_storage", lower_bound=0.0
        )
        reference = production
        dt_value = dt_value.to(device=reference.device, dtype=reference.dtype)
        production = production.to(device=reference.device, dtype=reference.dtype)
        oxidation = oxidation.to(device=reference.device, dtype=reference.dtype)
        transport_loss = transport_loss.to(
            device=reference.device, dtype=reference.dtype
        )
        tendency_prod = tendency_prod.to(device=reference.device, dtype=reference.dtype)
        tendency_oxid = tendency_oxid.to(device=reference.device, dtype=reference.dtype)
        tendency_trans = tendency_trans.to(device=reference.device, dtype=reference.dtype)
        storage = storage.to(device=reference.device, dtype=reference.dtype)

        # References: Yue et al. (2024), doi:10.5194/gmd-17-4621-2024, Eq. 33.
        # Equation: dM_raw=dt*(T_prod+T_oxid+T_trans); dM=max(dM_raw,-M).
        # Parameters: process tendencies [g C m-2 s-1], M [g C m-2], dt [s].
        # Open-system route: denied transport is reported as an explicit rate residual.
        raw_storage_tendency = tendency_prod + tendency_oxid + tendency_trans
        raw_storage_delta = raw_storage_tendency * dt_value
        storage_delta = torch.maximum(raw_storage_delta, -storage)
        transport_limiter_rate = (storage_delta - raw_storage_delta) / dt_value
        storage_tendency = storage_delta / dt_value

        requested_transport_rate = transport_loss

        # References: Yue et al. (2024), Eq. 33; RELIC discrete carbon ledger.
        # Equation: F_atm=F_transport-R_limit, with 0<=R_limit<=max(F_transport,0).
        # Parameters: F_transport, R_limit [g C m-2 s-1].
        # Open-system route: R_limit is the denied atmospheric transport demand.
        positive_requested_transport_rate = torch.maximum(
            requested_transport_rate,
            torch.zeros_like(requested_transport_rate),
        )
        absolute_tolerance = self.absolute_tolerance.to(
            device=reference.device, dtype=reference.dtype
        )
        if bool(
            torch.any(
                transport_limiter_rate
                > positive_requested_transport_rate + absolute_tolerance
            )
        ):
            raise ValueError(
                "Methane budget storage limiter cannot be routed entirely to "
                "the atmospheric transport port."
            )
        emission_rate = requested_transport_rate - transport_limiter_rate

        # References: NIST Chemistry WebBook; Biogeochemistry (2010) Section 7 soil methane emissions.
        # Equation: F_CH4 = F_CH4_C * (M_CH4 / M_C)
        # Parameters: F_CH4_C, M_CH4, M_C
        c_to_ch4_mass_ratio = torch.as_tensor(
            self.c_to_ch4_mass_ratio,
            device=reference.device,
            dtype=reference.dtype,
        )
        emission_mass_rate = emission_rate * c_to_ch4_mass_ratio
        # References: Yue et al. (2024) doi:10.5194/gmd-17-4621-2024
        # Equation: Residual = P - O - F_atm - dM/dt
        # Parameters: P, O, F_atm, dM/dt
        residual = production - oxidation - emission_rate - storage_tendency

        # References: Yue et al. (2024)
        # Equation: |Residual| <= max(abs_tol, rel_tol * max(P, O, F_atm))
        # Parameters: abs_tol, rel_tol, P, O, F_atm
        relative_tolerance = self.relative_tolerance.to(
            device=reference.device, dtype=reference.dtype
        )
        tolerance = absolute_tolerance + relative_tolerance * torch.maximum(
            torch.maximum(torch.abs(production), torch.abs(oxidation)),
            torch.abs(emission_rate),
        )

        # References: RELIC methane architecture policy
        # Equation: no post-hoc correction; residual is diagnostic only
        # Parameters: residual, tolerance
        violation = torch.abs(residual) > tolerance
        if torch.any(violation):
            logger.warning(
                "Methane mass balance residual exceeds tolerance without correction. "
                "max_abs_residual=%s max_tolerance=%s",
                self._scalar_from_value(torch.max(torch.abs(residual))),
                self._scalar_from_value(torch.max(tolerance)),
            )

        updates: Dict[str, UpdateItem] = {}
        updates["ch4_atmospheric_flux"] = UpdateItem(
            value=emission_rate,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            element="C",
            source_functor=self.name,
        )
        updates[
            "diagnostics.methane_budget.atmospheric_ch4_mass_flux"
        ] = UpdateItem(
            value=emission_mass_rate,
            mode=UpdateMode.REPLACE,
            unit="g CH4 m-2 s-1",
            per_timestep=False,
            source_functor=self.name,
        )
        updates["diagnostics.methane_budget.mass_balance_residual"] = UpdateItem(
            value=residual,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            source_functor=self.name,
        )
        updates["diagnostics.methane_budget.storage_tendency_net"] = UpdateItem(
            value=storage_tendency,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            source_functor=self.name,
        )
        updates["ch4_soil_storage"] = UpdateItem(
            value=storage_delta,
            mode=UpdateMode.ACCUMULATE,
            unit="g C m-2",
            per_timestep=True,
            element="C",
            source_functor=self.name,
        )
        updates[
            "diagnostics.methane_budget.transport_limiter_residual"
        ] = UpdateItem(
            value=transport_limiter_rate,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            source_functor=self.name,
        )

        self._last_updates = updates
        return updates

    def _require_state_tensor(
        self,
        state: StateDict,
        key: str,
        *,
        lower_bound: Optional[float] = None,
    ) -> Tensor:
        """Return one required finite tensor and enforce its declared domain."""
        if key not in state:
            raise KeyError(f"Required methane budget input '{key}' is missing.")
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"Methane budget input '{key}' must be a torch.Tensor.")
        tensor = value.to(self.device)
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(
                f"Methane budget input '{key}' must contain only finite values."
            )
        if lower_bound is not None and bool(torch.any(tensor < lower_bound)):
            raise ValueError(
                f"Methane budget input '{key}' must be >= {lower_bound}."
            )
        return tensor

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Report the explicit methane budget residual produced by the coupler.

        References:
            - Yue et al. (2024) doi:10.5194/gmd-17-4621-2024 (Eq.33)
            - Walter et al. (2001) doi:10.1029/2001JD900165
        Mathematical expression:
            violation["mass"] = |Residual|
        Parameter meanings:
            - diagnostics.methane_budget.mass_balance_residual: signed methane closure residual [g C m-2 s-1]
        """
        if "mass" not in self.metadata.conservation_laws:
            return {}

        residual = output_updates.get(
            "diagnostics.methane_budget.mass_balance_residual"
        )
        if residual is None:
            return {}

        if hasattr(residual, "value"):
            residual = residual.value
        residual_tensor = (
            residual
            if isinstance(residual, Tensor)
            else torch.as_tensor(residual)
        )
        return {
            "mass": float(
                torch.max(torch.abs(residual_tensor)).detach().cpu().item()
            )
        }


__all__ = ["MethaneBudgetCoupler"]

