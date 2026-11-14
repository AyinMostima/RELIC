"""Nitrogen dependency-ledger diagnostic and external mineral-N port router.

Dependency Functors own and submit their state increments directly. This
module observes those outputs without copying them and routes only the external
fertilizer-N and dependency-selected grazing urine-N ports to the ammonium
state ledger.
"""

from typing import Dict, Optional, List
import torch
from torch import Tensor
import logging

from ....core.functor import (
    BiologicalProcessFunctor,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from ....core.abstractions import StateDict, FluxDict
from ..api_contract import PedosphereContractMixin
from ....core.state import UpdateItem, UpdateMode
from ....computation.operators.iphs import advance_reaction_network
from ....utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)


class NitrogenCycling(PedosphereContractMixin, BiologicalProcessFunctor):
    """
    Observe the composed mineral-N ledger without becoming a second writer.

    The diagnostic projection evaluates ``d_dot = N_h r_h`` from dependency
    increments. The sole state output is the aggregate external mineral-N port
    contribution
    ``Delta_NH4 = c_kg_to_g * (F_fert + F_urine) * dt``.

    Grazing urine is represented as a non-strict reduced-order instantaneous
    urea-hydrolysis aggregate. No mechanistic urea pool or hydrolysis rate is
    inferred by this observer.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        enable_mineralization: bool = True,
        enable_nitrification: bool = True,
        enable_denitrification: bool = True,
        enable_leaching: bool = True,
        enable_root_uptake: bool = True,
    ):
        """
        Initialize unified nitrogen cycling functor.
        
        Parameters:
            config : Optional[Dict]
                Configuration dictionary
            device : Optional[torch.device]
                Computation device
            enable_mineralization : bool
                Enable mineralization process
            enable_nitrification : bool
                Enable nitrification process
            enable_denitrification : bool
                Enable denitrification process
            enable_leaching : bool
                Enable nitrogen leaching
            enable_root_uptake : bool
                Enable root nitrogen uptake
        """
        metadata = FunctorMetadata(
            name="NitrogenCycling",
            version="1.1.0",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description=(
                "Read-only nitrogen dependency ledger with explicit "
                "fertilizer and dependency-selected grazing urine inputs"
            ),
            references=[
                "Parton, W.J. et al. (1987). Soil Sci. Soc. Am. J., 51(5), 1173-1179.",
                "Davidson, E.A. et al. (2012). Global Biogeochem. Cycles, 26(2).",
                "Li, C. et al. (1992). J. Geophys. Res., 97(D9), 9759-9776.",
            ],
            mathematical_expression=(
                "Canonical IPHS: d_dot = N_h r_h for dependency diagnostics; "
                "Delta_NH4_ext = c_kg_to_g "
                "(F_fert + F_urine) dt"
            ),
            parameters={
                "relic.physical.constants.conversions.kg_to_g": (
                    "Fertilizer flux conversion [g kg-1]"
                ),
            },
            input_variables=["fertilizer_n_flux", "urine_n_flux"],
            output_variables=[
                "nh4_nitrogen",
                "diagnostics.nitrogen_cycling.expected_delta_nh4_nitrogen",
                "diagnostics.nitrogen_cycling.expected_delta_no3_nitrogen",
                "diagnostics.nitrogen_cycling.expected_delta_mineral_nitrogen",
                "diagnostics.nitrogen_cycling.external_fertilizer_n_delta",
                "diagnostics.nitrogen_cycling.external_urine_n_delta",
            ],
            conservation_laws=["nitrogen"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        
        super().__init__(
            name="NitrogenCycling",
            config=config or {},
            metadata=metadata
        )
        
        self.device = device if device is not None else getattr(
            self,
            "device",
            torch.device("cpu"),
        )
        self.register_buffer(
            "kilograms_to_grams",
            torch.as_tensor(
                get_physical_constant(
                    "relic.physical.constants.conversions.kg_to_g"
                ),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self.optional_input_variables = ["fertilizer_n_flux", "urine_n_flux"]
        self._enabled_subprocesses = tuple(
            name
            for name, enabled in (
                ("mineralization", enable_mineralization),
                ("nitrification", enable_nitrification),
                ("denitrification", enable_denitrification),
                ("nitrogen_leaching", enable_leaching),
                ("root_nutrient_uptake", enable_root_uptake),
            )
            if enabled
        )
        self._grazing_urine_selected = (
            "grazing_management" in self.get_dependencies()
        )
        logger.info(
            "NitrogenCycling initialized with declared dependencies: %s",
            ", ".join(self._enabled_subprocesses) or "none",
        )
    
    def forward(
        # References: Parton et al. (1987); Galloway et al. (2004); Li et al. (1992).
        # Mathematical expression: DeltaN_dep=sum(declared_subprocess_pool_increments);
        #   DeltaNH4_external=kg_to_g*(F_fertilizer+F_urine)*dt.
        # Parameter meanings: Injected dependency outputs provide diagnostic increments;
        #   selected fertilizer/urine ports provide external mineral nitrogen.
        # Units: dependency/external increments [g N m-2]; input ports [kg N m-2 s-1];
        #   kg_to_g [g kg-1]; dt [s].
        # Domain/constraints: Declared subprocess outputs are required; no child Functor
        #   is called and dependency increments are not re-applied. Urine requires its
        #   selected grazing dependency; rapid urea hydrolysis is a reduced boundary assumption.
        # Sign convention: Dependency increments appear only in diagnostics; the sole
        #   stock proposal is positive external NH4 input, integrated exactly once.
        self,
        state: StateDict,
        time: float,
        dt: Tensor | float,
        subprocess_outputs: Optional[Dict[str, Dict[str, object]]] = None,
        **kwargs,
    ) -> FluxDict:
        """Audit dependency increments and route external mineral N once."""

        dt_tensor = self._ensure_tensor(dt)
        if dt_tensor.numel() != 1 or not bool(torch.isfinite(dt_tensor).all()):
            raise ValueError("dt must be one finite scalar")
        if bool((dt_tensor <= 0).any()):
            raise ValueError("dt must be positive")

        def _sanitize_tensor(key_name: str, tensor_value: Tensor) -> Tensor:
            if not torch.is_floating_point(tensor_value):
                return tensor_value
            if not bool(torch.isfinite(tensor_value).all()):
                raise ValueError(f"{self.name}: non-finite values detected in '{key_name}'")
            return tensor_value

        def _to_tensor(value: object, key_name: str = "") -> Optional[Tensor]:
            if isinstance(value, UpdateItem):
                value = value.value
            if torch.is_tensor(value):
                return _sanitize_tensor(key_name, value)
            if isinstance(value, (int, float)):
                scalar = torch.tensor(
                    float(value),
                    device=dt_tensor.device,
                    dtype=dt_tensor.dtype,
                )
                return _sanitize_tensor(key_name, scalar)
            return None

        def _external_flux(key_name: str) -> Optional[Tensor]:
            if key_name not in state:
                return None
            value = state[key_name]
            if not torch.is_tensor(value):
                raise TypeError(f"{key_name} must be a floating-point Tensor")
            if value.numel() == 0:
                raise ValueError(f"{key_name} must be non-empty")
            if not torch.is_floating_point(value):
                raise TypeError(f"{key_name} must be a floating-point Tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key_name} must contain only finite values")
            if bool((value < 0).any()):
                raise ValueError(f"{key_name} must be non-negative")
            return value

        if subprocess_outputs is None:
            subprocess_outputs = {}
        if not isinstance(subprocess_outputs, dict):
            raise TypeError("subprocess_outputs must be a mapping when provided.")

        subprocess_deltas: Dict[str, Tensor] = {}
        forbidden_delta_keys = {"delta_nh4_nitrogen", "delta_no3_nitrogen"}
        for process_name in self._enabled_subprocesses:
            if process_name not in subprocess_outputs:
                raise KeyError(
                    f"declared dependency output '{process_name}' is required"
                )
            process_output = subprocess_outputs[process_name]
            if not isinstance(process_output, dict):
                raise TypeError(
                    f"declared dependency output '{process_name}' must be a mapping"
                )
            noncanonical = forbidden_delta_keys.intersection(process_output)
            if noncanonical:
                raise KeyError(
                    "non-canonical dependency outputs are forbidden: "
                    + ", ".join(sorted(noncanonical))
                )
            for output_key, output_value in process_output.items():
                tensor_value = _to_tensor(
                    output_value,
                    key_name=f"{process_name}.{output_key}",
                )
                if tensor_value is None or output_key not in {
                    "nh4_nitrogen",
                    "no3_nitrogen",
                }:
                    continue
                existing = subprocess_deltas.get(output_key)
                subprocess_deltas[output_key] = (
                    tensor_value if existing is None else existing + tensor_value
                )

        fertilizer_flux = _external_flux("fertilizer_n_flux")
        urine_flux = _external_flux("urine_n_flux")
        if self._grazing_urine_selected and urine_flux is None:
            raise KeyError(
                "urine_n_flux is required when the grazing_management "
                "dependency is selected"
            )
        if not self._grazing_urine_selected and urine_flux is not None:
            raise ValueError(
                "urine_n_flux requires the grazing_management dependency"
            )

        default_prototype = next(
            (
                value
                for value in (fertilizer_flux, urine_flux)
                if value is not None
            ),
            dt_tensor,
        )
        prototype = next(iter(subprocess_deltas.values()), default_prototype)
        dt_network = dt_tensor.to(device=prototype.device, dtype=prototype.dtype)
        zero = torch.zeros_like(prototype)
        nh4_delta = subprocess_deltas.get("nh4_nitrogen", zero)
        no3_delta = subprocess_deltas.get("no3_nitrogen", zero)
        nh4_delta, no3_delta = torch.broadcast_tensors(nh4_delta, no3_delta)

        # References:
        #   - Parton et al. (1987), soil nitrogen transformation bookkeeping.
        #   - Feinberg (2019), reaction-network incidence representation.
        #   - Strang (1968); architecture_overview.md Section 4.4.1 (dependency
        #     outputs of a fast-group composition).
        # Mathematical expression:
        #   d = N_h r_h dt with r_h = Delta_obs / dt and N_h = I, so
        #   d = (Delta_NH4, Delta_NO3) = Delta_obs: dt cancels and the expected
        #   deltas equal the observed dependency increments exactly.
        # Parameter meanings:
        #   N_h: mineral-pool diagnostic incidence map (identity) [1].
        #   Delta_obs: summed NH4 and NO3 pool increments of the enabled
        #     dependencies in subprocess_outputs, i.e. the outputs of each
        #     provider's most recent sub-execution before this call.
        #   r_h: Delta_obs / dt; it is the providers' mean rate only when their
        #     interval equals this call's interval (lie mode).
        #   dt: the interval of this call.
        # Units: Delta_obs and d [g N m-2]; r_h [g N m-2 s-1]; dt [s]; N_h [1].
        # Domain/constraints: no interval rescaling is applied. The increments
        #   cover the providers' own interval: [t, t + h] in lie mode and the
        #   first half-step [t, t + h/2] in strang mode, also when this Functor
        #   runs its second half-step over [t + h/2, t + h]. Under strang these
        #   diagnostics are therefore not a whole-step budget; the whole-step
        #   budget is the ledger StateManager commits. The network uses the
        #   diagnostic_residual route and advances no state.
        # Sign convention: positive Delta_obs and d increase the NH4 and NO3 pools.
        diagnostic_network = advance_reaction_network(
            name="nitrogen_cycling_dependency_diagnostic",
            species=("nh4_nitrogen", "no3_nitrogen"),
            reactions=("nh4_dependency_projection", "no3_dependency_projection"),
            stoichiometric_N=torch.eye(
                2,
                device=prototype.device,
                dtype=prototype.dtype,
            ),
            rates=(nh4_delta / dt_network, no3_delta / dt_network),
            dt=dt_network,
            open_system_route="diagnostic_residual",
        )
        expected_nh4_delta = diagnostic_network.state_delta[0]
        expected_no3_delta = diagnostic_network.state_delta[1]

        updates: Dict[str, Tensor | UpdateItem] = {
            "diagnostics.nitrogen_cycling.expected_delta_nh4_nitrogen": (
                expected_nh4_delta
            ),
            "diagnostics.nitrogen_cycling.expected_delta_no3_nitrogen": (
                expected_no3_delta
            ),
            "diagnostics.nitrogen_cycling.expected_delta_mineral_nitrogen": (
                expected_nh4_delta + expected_no3_delta
            ),
        }

        fertilizer_delta = torch.zeros_like(expected_nh4_delta)
        urine_delta = torch.zeros_like(expected_nh4_delta)
        external_delta_present = (
            fertilizer_flux is not None or urine_flux is not None
        )
        if fertilizer_flux is not None:
            fertilizer_flux = fertilizer_flux.to(
                device=expected_nh4_delta.device,
                dtype=expected_nh4_delta.dtype,
            )
            conversion = self.kilograms_to_grams.to(
                device=fertilizer_flux.device,
                dtype=fertilizer_flux.dtype,
            )
            # References:
            #   - Galloway et al. (2004), anthropogenic reactive-N inputs.
            #   - BIPM SI Brochure, mass-unit conversion.
            # Mathematical expression:
            #   Delta_NH4_fert = c_kg_to_g * F_fert * dt.
            # Parameter meanings:
            #   F_fert: fertilizer N input [kg N m-2 s-1].
            #   c_kg_to_g: physical-constants SSOT conversion [g kg-1].
            #   dt: model time step [s].
            fertilizer_delta = fertilizer_flux * conversion * dt_network
        if urine_flux is not None:
            urine_flux = urine_flux.to(
                device=expected_nh4_delta.device,
                dtype=expected_nh4_delta.dtype,
            )
            conversion = self.kilograms_to_grams.to(
                device=urine_flux.device,
                dtype=urine_flux.dtype,
            )
            # References:
            #   - Li et al. (1992), DNDC soil nitrogen transformations.
            # Mathematical expression:
            #   Delta_NH4_urine = c_kg_to_g * F_urine * dt.
            # Parameter meanings:
            #   F_urine: grazing urine N input [kg N m-2 s-1].
            #   c_kg_to_g: physical-constants SSOT conversion [g kg-1].
            #   dt: model time step [s].
            # Open-system route:
            #   This non-strict aggregate treats rapid urea hydrolysis as an
            #   instantaneous NH4 boundary input; it is not a urea mechanism.
            urine_delta = urine_flux * conversion * dt_network

        if external_delta_present:
            fertilizer_delta, urine_delta = torch.broadcast_tensors(
                fertilizer_delta,
                urine_delta,
            )
            updates["nh4_nitrogen"] = UpdateItem(
                value=fertilizer_delta + urine_delta,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                basis="per_step",
                per_timestep=True,
                element="N",
                source_functor=self.name,
            )
        updates[
            "diagnostics.nitrogen_cycling.external_fertilizer_n_delta"
        ] = fertilizer_delta
        updates[
            "diagnostics.nitrogen_cycling.external_urine_n_delta"
        ] = urine_delta
        return updates

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Return the residual of the declared external mineral-N port."""
        # References: Galloway et al. (2004); Li et al. (1992), external nitrogen accounting.
        # Mathematical expression: residual=max(abs(DeltaNH4_actual-DeltaN_fertilizer-DeltaN_urine)).
        # Parameter meanings: External diagnostics are the actual step amounts;
        #   DeltaNH4_actual is this aggregate's own governed NH4 UpdateItem.
        # Units: all three amounts and residual [g N m-2].
        # Domain/constraints: Checks the external input owner only; validates NH4 mode,
        #   unit, element, per_timestep and source_functor, not subprocess internal budgets.
        # Sign convention: External input and NH4 increase are positive and cancel in the residual.

        def _diagnostic_tensor(key: str) -> Tensor:
            value = output_updates.get(key)
            if not torch.is_tensor(value):
                raise TypeError(f"{key} must be a Tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key} must contain only finite values")
            return value

        fertilizer_delta = _diagnostic_tensor(
            "diagnostics.nitrogen_cycling.external_fertilizer_n_delta"
        )
        urine_delta = _diagnostic_tensor(
            "diagnostics.nitrogen_cycling.external_urine_n_delta"
        )
        fertilizer_delta, urine_delta = torch.broadcast_tensors(
            fertilizer_delta,
            urine_delta,
        )
        expected_delta = fertilizer_delta + urine_delta

        update = output_updates.get("nh4_nitrogen")
        if update is None:
            actual_delta = torch.zeros_like(expected_delta)
        elif isinstance(update, UpdateItem):
            if (
                update.mode is not UpdateMode.ACCUMULATE
                or update.per_timestep is not True
                or update.unit != "g N m-2"
                or update.element != "N"
                or update.source_functor != self.name
            ):
                raise ValueError(
                    "nh4_nitrogen external-port update contract is invalid"
                )
            actual_delta = update.value
        else:
            raise TypeError("nh4_nitrogen must be an UpdateItem")

        actual_delta, expected_delta = torch.broadcast_tensors(
            actual_delta,
            expected_delta,
        )
        residual = float(
            torch.max(torch.abs(actual_delta - expected_delta)).detach().cpu()
        )
        return {
            law: residual
            for law in self.metadata.conservation_laws
        }
    
    def get_dependencies(self) -> List[str]:
        """Return the enabled subprocess owners plus instance-declared names.

        Composition config (for example ``grazing_management``) is merged with
        the enabled subprocess owners whose outputs ``forward`` requires; it
        no longer replaces them.
        """
        return self._merge_declared_dependencies(list(self._enabled_subprocesses))
