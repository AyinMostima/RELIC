"""
Hierarchical Aggregator for Multi-Scale Operations

Implements conservation-aware aggregation and disaggregation between
hierarchical scales with support for different variable types and
aggregation strategies.

Mathematical Expression:
    Every variable carries one explicit SpatialQuantityBasis (relic.coupling.scales.types);
    extensive fields are converted to their area-density representation x_i (from a cell
    total X_i: x_i = X_i / A_i) and back after the kernel.
    Extensive aggregation (area density): Y_j = sum_{i in j} x_i * A_i / A_j
    Intensive aggregation (mean): phi_j = sum_{i in j} phi_i * A_i / sum_{i in j} A_i
    Conservation check: |sum_j Y_j * A_j - sum_i x_i * A_i| <= tolerance

References:
    - Jones, P.W. (1999). Conservative remapping: First- and second-order
    - Ullrich, P.A. et al. (2009). Conservative transport schemes
"""

import logging
import re
from typing import Dict, List, Mapping, Optional, Any, Tuple, Union
import torch
from relic.core import device_policy
from torch import Tensor

from .hierarchy_index import HierarchyIndex
from .weight_ops import WeightOps
from ....core.abstractions import HierarchicalScalingBase
from ....core.functor import ScaleCouplingFunctor, FunctorMetadata
from ....core.functor import ProcessCategory, ExecutionPhase, SpatialScale, VariableKind
from ...core.conservation_checker import coarsest_floating_dtype, require_relative_tolerance
from ..conservation_enforcer import ConservationEnforcer
from ..variable_catalog import find_catalog_entry
from ..types import SpatialQuantityBasis, convert_spatial_basis, resolve_spatial_basis
from ....parameters import UnifiedParameterManager
from ....parameters.namespaces import ParameterNamespace

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_SCALE = "fine"
DEFAULT_TARGET_SCALE = "coarse"
DEFAULTS_NS = ParameterNamespace.COUPLING_DEFAULTS.value


class HierarchicalAggregator(HierarchicalScalingBase, ScaleCouplingFunctor):
    """
    Hierarchical aggregator for multi-scale operations with enhanced parameter management.

    Performs aggregation and disaggregation between hierarchy levels with
    conservation enforcement, parameter integration, and support for different variable types.

    Parameters
    ----------
    index : HierarchyIndex, optional
        Hierarchical mapping between scales.
    device : torch.device, optional
        Target computation device.
    hierarchy_levels : List[str], optional
        Ordered levels from fine to coarse (dimensionless aliases).
    weight_type : str, optional
        Aggregation weight scheme ('area','sum','mean'), dimensionless selector.
    rule : str, optional
        Scaling rule ('conservative','mean', etc.), dimensionless selector.
    topology : Dict, optional
        Hierarchy topology definition.
    parameter_manager : Optional[UnifiedParameterManager]
        Parameter manager for validation and conversion.

    Returns
    -------
    Dict[str, Tensor]
        Aggregated/disaggregated outputs keyed by variable name; extensive variables
        conserve totals within tolerance, intensive variables averaged.

    Units
    -----
    Areas in m^2 for weights; weights dimensionless; variables retain input units.

    Mathematical Expression
    -----------------------
    Each variable declares one SpatialQuantityBasis (``variable_metadata`` or the YAML
    catalogs; an undeclared variable and a bare ``extensive`` raise). Extensive fields
    are handled in their area-density representation ``x_i`` (``x_i = X_i / A_i`` for a
    cell total) and returned in their input basis.
    Extensive: ``Y_j = sum_{i in child(j)} x_i * A_i / A_j``.
    Intensive: ``phi_j = (sum_{i in child(j)} phi_i * A_i) / (sum_{i in child(j)} A_i)``.
    Conservation: ``|sum_j Y_j * A_j - sum_i x_i * A_i| <= tolerance``.

    References
    ----------
    Jones, P.W. (1999). Conservative remapping: first- and second-order.
    Ullrich, P.A. et al. (2009). Conservative transport schemes.
    """
    
    def __init__(
        self,
        index: Optional[HierarchyIndex] = None,
        device: torch.device = None,
        hierarchy_levels: Optional[List[str]] = None,
        *,
        weight_type: str,
        rule: str,
        topology: Optional[Dict] = None,
        parameter_manager: Optional[UnifiedParameterManager] = None,
    ):
        """Initialize hierarchical aggregator.

        ``weight_type`` (the weighting rule, e.g. "area") and ``rule`` (the scaling rule, e.g.
        "conservative") select the aggregation method and are required; there is no default
        and no second spelling of either.
        """

        if index is None and topology is None:
            raise ValueError(
                "HierarchicalAggregator requires an explicit index or topology."
            )

        # Handle device parameter
        if device is None:
            device = getattr(
                index,
                "device",
                device_policy.resolve_device(default="auto"),
            )

        if index is None:
            inferred_levels = list(hierarchy_levels) if hierarchy_levels else [DEFAULT_SOURCE_SCALE, DEFAULT_TARGET_SCALE]
            if len(inferred_levels) < 2:
                inferred_levels = [DEFAULT_SOURCE_SCALE, DEFAULT_TARGET_SCALE]
            source_guess = inferred_levels[0]
            target_guess = inferred_levels[1]
            n_source_units = int(topology.get("n_source_units", 1))
            index = HierarchyIndex(
                source_scale=source_guess,
                target_scale=target_guess,
                device=device,
                n_source_units=max(1, n_source_units),
                topology=topology,
                parameter_manager=parameter_manager,
            )

        validated_topology = None
        if topology is not None:
            validated_topology = self._validate_topology(topology, "init")
            if hasattr(index, "update_topology"):
                index.update_topology(validated_topology)

        def _normalize_scale(value: Any, default: str, field: str) -> str:
            """Normalize scale names coming from hierarchy indexes or mocks."""
            if isinstance(value, str):
                return value
            raise TypeError(f"{field} must be str, got {type(value)}")

        # Extract source and target scales from index as strings
        raw_source_scale = getattr(index, "source_scale", DEFAULT_SOURCE_SCALE)
        raw_target_scale = getattr(index, "target_scale", DEFAULT_TARGET_SCALE)
        source_scale = _normalize_scale(raw_source_scale, DEFAULT_SOURCE_SCALE, "source_scale")
        target_scale = _normalize_scale(raw_target_scale, DEFAULT_TARGET_SCALE, "target_scale")

        # Extract hierarchy levels from index if not provided, ensure string list
        raw_levels = hierarchy_levels if hierarchy_levels is not None else getattr(
            index, "levels", [DEFAULT_SOURCE_SCALE, DEFAULT_TARGET_SCALE]
        )
        if not isinstance(raw_levels, (list, tuple)) or not all(isinstance(level, str) for level in raw_levels):
            raise TypeError("hierarchy_levels must be a list of strings.")
        hierarchy_levels = list(raw_levels)

        self.hierarchy_levels = hierarchy_levels
        self.weight_type = weight_type
        self.rule = rule

        raw_name = f"HierarchicalAggregator_{source_scale}_to_{target_scale}"
        aggregator_name = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
        metadata = FunctorMetadata(
            name=aggregator_name,
            version="1.0.0",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            description="Hierarchical aggregator enforcing conservation across scale levels",
            references=[
                "Jones, P.W. (1999). Conservative remapping schemes.",
                "O'Neill, R.V. et al. (1986). A hierarchical concept of ecosystems."
            ],
            mathematical_expression=(
                "Y_j = sum_{i in j} X_i * w_ij for extensive variables; "
                "epsilon = |sum_j Y_j A_j - sum_i X_i A_i| / |sum_i X_i A_i|"
            ),
            parameters={
                "w_ij": "Hierarchy weight from child i to parent j [dimensionless]",
                "A_i": "Area or mass weight of child element i [m2 or kg]",
                "epsilon": "Relative conservation error across hierarchy [dimensionless]",
            },
            input_variables=["source_hierarchy_fields"],
            output_variables=["target_hierarchy_fields"],
            optional_output_variables=["target_hierarchy_fields"],
            conservation_laws=["mass", "energy"] if str(rule).lower() == "conservative" else [],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )

        ScaleCouplingFunctor.__init__(
            self,
            name=aggregator_name,
            metadata=metadata,
            source_scale=source_scale,
            target_scale=target_scale,
            coupling_type="hierarchical",
            device=device,
            parameter_manager=parameter_manager,
        )
        HierarchicalScalingBase.__init__(
            self,
            source_scale=source_scale,
            target_scale=target_scale,
            hierarchy_levels=hierarchy_levels,
            weight_type=weight_type,
            rule=rule,
            device=device,
            name=aggregator_name,
            metadata=metadata,
        )

        # The logical input port names the call-time field set (``data``/``variables``); it is
        # not a literal state key, so FunctorBase.validate_state must not require it.
        self.optional_input_variables = ("source_hierarchy_fields",)

        # Initialize parameter manager through FunctorBase injection chain
        self.parameter_manager = self.get_parameter_manager()

        self.index = index
        self.device = device

        # Tolerance of this operator's own ledger checks: the one scale-operator default
        # coupling.scales.defaults.conservation_tolerance. The ScaleManager facade replaces it per
        # call with the ledger tolerance of the call's mode (ScaleManager.ledger_tolerance).
        tolerance = self.get_parameter_value(
            "conservation_tolerance",
            namespace=DEFAULTS_NS,
            units="dimensionless",
            validate=True,
            required=True,
        )
        if isinstance(tolerance, torch.Tensor):
            tolerance = float(tolerance.detach().cpu().item())
        self.conservation_tolerance = require_relative_tolerance(
            tolerance, f"{DEFAULTS_NS}.conservation_tolerance"
        )
        # Catalog resolutions only (per-call metadata is never cached).
        self._variable_metadata_cache: Dict[str, Tuple[Dict[str, Any], str]] = {}
        self.epsilon = self._resolve_numerical_epsilon()

        # Initialize conservation enforcer eagerly to avoid lazy-state bugs; each check passes
        # self.conservation_tolerance (the facade sets it to the tolerance of the call's mode).
        self._conservation_enforcer = ConservationEnforcer(
            track_history=False,
            device=self.device,
        )
        
        # Aggregation method mapping
        self.method_map = {
            'area_weighted_mean': self._area_weighted_mean,
            'area_weighted_sum': self._area_weighted_sum,
            'flow_conserving': self._flow_conserving,
            'conservative': self._conservative_aggregate,
            'mass_weighted': self._mass_weighted,
            'volume_weighted': self._volume_weighted,
            'nearest': self._nearest_neighbor,
            'bilinear': self._bilinear_interpolate,
            'harmonic_mean': self._harmonic_mean,
            'arithmetic_mean': self._area_weighted_mean,
            'bounded_area_mean': self._bounded_area_mean,
            'mode': self._area_weighted_mode,
            'circular_mean': self._circular_mean,
            'component_wise_mean': self._component_wise_mean,
            'geometric_mean': self._geometric_mean,
            'effective_roughness_length': self._effective_roughness_length,
        }
        
        logger.info(f"HierarchicalAggregator initialized for {index.source_scale} -> {index.target_scale}")
    
    
    def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
        """
        Forward computation following Functor interface.

        Args:
            state: Current state variables
            time: Current simulation time [s]
            dt: Time step [s]
            **kwargs: Additional parameters

        Returns:
            Dict of state variable increments for StateManager
        """
        # References: Jones (1999) and Ullrich et al. (2009) as cited in the module docstring;
        #   RELIC ScaleCoupling contract (relic/core/functor/scale_coupling_bridge.py).
        # Mathematical expression: dispatches to self.aggregate or self.disaggregate with the
        #   explicit SpatialQuantityBasis of each variable. Default "conservative" aggregation:
        #   extensive fields in their area-density representation x_i,
        #   Y_j = sum_i x_i A_i M_ij / A_j per parent with the membership shares M_ij and no global
        #   rescale (_flow_conserving; orphan amounts are reported); intensive
        #   Y_j = sum_i X_i A_i W_ij / sum_i A_i W_ij (_area_weighted_mean); other strategies via
        #   method_map. Disaggregation broadcasts X_i = Y_j (intensive) or closes
        #   sum_{i in j} x_i A_i = Y_j A_j (extensive, partition).
        # Parameter meanings: X_i source value; Y_j target value; A_i, A_j areas; W_ij membership
        #   weight; outputs keyed "<var>_<target_scale>" or "<var>_<source_scale>" plus
        #   "_hierarchical_diagnostics".
        # Units: variables retain input units and basis; areas share one unit (m^2).
        # Domain/constraints: data None logs a warning and returns {}; a Tensor input is the
        #   variable "default_var" (other requested names raise); an unknown operation is inferred
        #   from shape[0] == n_source_units; the basis comes from variable_metadata or the YAML
        #   catalogs (an undeclared variable and a bare "extensive" raise);
        #   extensive variables record ConservationEnforcer relative error; failures raise RuntimeError.
        # Sign convention: signed values are preserved; area and membership weights are nonnegative.
        # Extract operation and data from state or kwargs
        if 'operation' in kwargs:
            operation = kwargs['operation']
        elif 'operation' in state:
            operation = state['operation']
        else:
            operation = 'aggregate'
        if 'data' in kwargs:
            data = kwargs['data']
        elif 'data' in state:
            data = state['data']
        else:
            data = None
        if 'variables' in kwargs:
            variables = kwargs['variables']
        elif 'variables' in state:
            variables = state['variables']
        else:
            variables = ['default_var']
        if 'strategy' in kwargs:
            strategy = kwargs['strategy']
        elif 'strategy' in state:
            strategy = state['strategy']
        else:
            strategy = None
        if 'mode' in kwargs:
            mode = kwargs['mode']
        elif 'mode' in state:
            mode = state['mode']
        else:
            mode = 'conservative'
        if 'variable_metadata' in kwargs:
            variable_metadata = kwargs['variable_metadata']
        elif 'variable_metadata' in state:
            variable_metadata = state['variable_metadata']
        else:
            variable_metadata = None

        if data is None:
            logger.warning("No data provided for hierarchical aggregation")
            return {}

        # Convert single tensor to dict format if needed
        if isinstance(data, Tensor):
            data_dict = {'default_var': data}
            if 'variables' not in kwargs and 'variables' not in state:
                variables = ['default_var']
            elif list(variables) != ['default_var']:
                raise ValueError(
                    "A tensor 'data' input is the variable 'default_var'; pass data as a mapping to "
                    f"aggregate named variables {list(variables)}."
                )
        else:
            data_dict = data

        try:
            # Perform the appropriate hierarchical operation
            if operation == 'aggregate':
                result_data, audit_info = self.aggregate(
                    data=data_dict,
                    variables=variables,
                    strategy=strategy,
                    mode=mode,
                    variable_metadata=variable_metadata,
                )
            elif operation == 'disaggregate':
                result_data, audit_info = self.disaggregate(
                    data=data_dict,
                    variables=variables,
                    strategy=strategy,
                    mode=mode,
                    variable_metadata=variable_metadata,
                )
            else:
                # Auto-detect based on index and grid sizes
                if hasattr(self.index, 'n_target_units') and hasattr(self.index, 'n_source_units'):
                    if 'default_var' in data_dict and data_dict['default_var'].shape[0] == self.index.n_source_units:
                        operation = 'aggregate'
                        result_data, audit_info = self.aggregate(
                            data=data_dict,
                            variables=variables,
                            strategy=strategy,
                            mode=mode,
                            variable_metadata=variable_metadata,
                        )
                    else:
                        operation = 'disaggregate'
                        result_data, audit_info = self.disaggregate(
                            data=data_dict,
                            variables=variables,
                            strategy=strategy,
                            mode=mode,
                            variable_metadata=variable_metadata,
                        )
                else:
                    # Default to aggregation
                    operation = 'aggregate'
                    result_data, audit_info = self.aggregate(
                        data=data_dict,
                        variables=variables,
                        strategy=strategy,
                        mode=mode,
                        variable_metadata=variable_metadata,
                    )

            # Prepare comprehensive diagnostics
            diagnostics = {
                'operation': operation,
                'variables': variables,
                'strategy': strategy,
                'mode': mode,
                'source_scale': getattr(self.index, 'source_scale', 'unknown'),
                'target_scale': getattr(self.index, 'target_scale', 'unknown'),
                'n_source_units': getattr(self.index, 'n_source_units', 0),
                'n_target_units': getattr(self.index, 'n_target_units', 0),
                'weight_type': self.weight_type,
                'rule': self.rule
            }

            # Add audit information
            diagnostics.update(audit_info)

            # Log hierarchical operation
            logger.debug(
                f"Hierarchical {operation}: {list(data_dict.keys())} "
                f"from {diagnostics['source_scale']} to {diagnostics['target_scale']} "
                f"using {diagnostics.get('method_used', 'unknown')} method"
            )

            # Return state increments in simulator format
            # The result should be increments that can be applied to state variables
            state_updates = {}

            if isinstance(data, Tensor):
                # Single tensor input - create appropriate state variable name
                if operation == 'aggregate':
                    var_name = f"{getattr(self.index, 'target_scale', 'coarse')}_data"
                else:
                    var_name = f"{getattr(self.index, 'source_scale', 'fine')}_data"

                if 'default_var' not in result_data:
                    raise RuntimeError(
                        "Hierarchical operation produced no result for the tensor input 'default_var'."
                    )
                state_updates[var_name] = result_data['default_var']
            else:
                # Multiple variables - preserve naming but add scale suffix
                for var_name, var_data in result_data.items():
                    if operation == 'aggregate':
                        scaled_name = f"{var_name}_{getattr(self.index, 'target_scale', 'coarse')}"
                    else:
                        scaled_name = f"{var_name}_{getattr(self.index, 'source_scale', 'fine')}"
                    state_updates[scaled_name] = var_data

            # Add diagnostics to a special field that won't be applied to state
            state_updates['_hierarchical_diagnostics'] = diagnostics

            return state_updates

        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            error_msg = f"Hierarchical aggregation operation failed: {exc}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from exc

    @staticmethod
    def _validate_topology(topology: Dict[str, Any], operation: str) -> Dict[str, Any]:
        """Validate required topology keys and shapes before use."""
        required_keys = {"n_source_units", "n_target_units", "parent_map", "source_areas", "target_areas"}
        missing = required_keys - set(topology.keys())
        if missing:
            raise ValueError(
                f"Incomplete topology for {operation}: missing keys {missing}. "
                "Ensure coupling.scales.hierarchy parameters provide parent_map/source_areas/target_areas."
            )
        n_src = int(topology["n_source_units"])
        n_tgt = int(topology["n_target_units"])
        parent_map = topology["parent_map"]
        source_areas = topology["source_areas"]
        target_areas = topology["target_areas"]
        if len(source_areas) != n_src:
            raise ValueError(f"source_areas length {len(source_areas)} does not match n_source_units {n_src}")
        if len(target_areas) != n_tgt:
            raise ValueError(f"target_areas length {len(target_areas)} does not match n_target_units {n_tgt}")
        for child, parent in parent_map.items():
            if child >= n_src or child < 0:
                raise ValueError(f"parent_map index {child} out of range for n_source_units={n_src}")
            if parent >= n_tgt or parent < 0:
                raise ValueError(f"parent_map target {parent} out of range for n_target_units={n_tgt}")
        return topology

    def _extract_tensor(self, obj, name):
        """Validate tensor type."""
        if isinstance(obj, torch.Tensor):
            return obj
        try:
            tensor = torch.as_tensor(obj, device=self.device)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001
            raise ValueError(f"{name} must be convertible to tensor, got {type(obj)}: {exc}") from exc
        if tensor.numel() == 0:
            raise ValueError(f"{name} must contain data")
        return tensor

    def aggregate(
        self,
        data: Dict[str, Tensor],
        variables: List[str],
        strategy: Optional[Union[str, Mapping[str, str]]] = None,
        mode: str = "conservative",
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """
        Aggregate data from fine to coarse scale.

        Parameters:
            data : Dict[str, Tensor]
                Input data to aggregate
            variables : List[str]
                Variable names
            strategy : Optional[Union[str, Mapping[str, str]]]
                Aggregation strategy for every variable, or one strategy per variable
                (a mapping must name every aggregated variable); None auto-selects each
                variable's method from the mode and its declared basis
            mode : str
                Aggregation mode
            variable_metadata : Optional[Dict[str, Dict[str, Any]]]
                Per-variable metadata with ``spatial_basis`` (or ``variable_type``
                intensive/categorical); variables without an entry use the YAML catalogs,
                and a variable declared in neither raises ``ValueError``.

        Returns:
            Tuple of aggregated data (same basis as the input) and audit information
            including ``quantity_basis`` and ``output_basis`` per variable.
        """
        # Handle Enum values - extract value if it's an Enum
        if hasattr(strategy, 'value'):
            strategy = strategy.value
        if hasattr(mode, 'value'):
            mode = mode.value

        aggregated_data = {}
        audit_info = {
            'method_used': {},
            'conservation_errors': {},
            'quantity_basis': {},
            'output_basis': {},
            'warnings': [],
            'n_variables': len(variables)
        }

        max_conservation_error = 0.0

        for var_name in variables:
            if var_name not in data:
                warning_msg = f"Variable {var_name} not found in data"
                logger.warning(warning_msg)
                audit_info['warnings'].append(warning_msg)
                continue

            var_data = data[var_name]
            basis = self._variable_basis(var_name, variable_metadata)

            # Keep input dtype in strict mode to stay numerically consistent with
            # source/target area tensors used by conservation checks.
            if mode == 'strict_conservative':
                var_data = var_data.to(dtype=var_data.dtype)

            # Select aggregation method: an explicit strategy (one for the call or one per
            # variable) must be a known method; only an absent strategy is auto-selected from
            # the mode and the declared basis.
            if isinstance(strategy, Mapping):
                if var_name not in strategy:
                    raise ValueError(
                        f"The per-variable strategy mapping gives no strategy for {var_name!r}."
                    )
                var_strategy = strategy[var_name]
            else:
                var_strategy = strategy
            if hasattr(var_strategy, 'value'):
                var_strategy = var_strategy.value
            if var_strategy is not None:
                if var_strategy not in self.method_map:
                    raise ValueError(
                        f"Unknown aggregation strategy {var_strategy!r} for {var_name!r}; accepted "
                        f"strategies are {sorted(self.method_map)}."
                    )
                method_name = var_strategy
                method = self.method_map[var_strategy]
            else:
                method, method_name = self._auto_select_method(mode, basis)

            # Perform aggregation
            try:
                aggregated_var = self._apply_aggregation_method(method, method_name, var_data, var_name, basis)
                aggregated_data[var_name] = aggregated_var
                audit_info['method_used'][var_name] = method_name

                # Check conservation for extensive variables on the covered (non-orphan) support
                # and report the orphan amount instead of redistributing it.
                if basis.is_extensive:
                    conservation_error, uncovered = self._audit_extensive_aggregation(
                        var_data, aggregated_var, var_name, basis
                    )
                    audit_info['conservation_errors'][var_name] = conservation_error
                    audit_info.setdefault('uncovered_source_amount', {})[var_name] = uncovered
                    max_conservation_error = max(max_conservation_error, abs(conservation_error))

            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                error_msg = f"Aggregation failed for {var_name}: {exc}"
                logger.error(error_msg)
                audit_info['warnings'].append(error_msg)
                if mode == 'strict_conservative':
                    if hasattr(self.index, '_temp_matrix'):
                        delattr(self.index, '_temp_matrix')
                    if hasattr(self.index, '_temp_source_areas'):
                        delattr(self.index, '_temp_source_areas')
                    if hasattr(self.index, '_temp_target_areas'):
                        delattr(self.index, '_temp_target_areas')
                raise RuntimeError(error_msg) from exc
            audit_info['quantity_basis'][var_name] = basis.value
            audit_info['output_basis'][var_name] = basis.value

        # Clean up temporary float64 variables if they were created
        if mode == 'strict_conservative':
            if hasattr(self.index, '_temp_matrix'):
                delattr(self.index, '_temp_matrix')
            if hasattr(self.index, '_temp_source_areas'):
                delattr(self.index, '_temp_source_areas')
            if hasattr(self.index, '_temp_target_areas'):
                delattr(self.index, '_temp_target_areas')

        audit_info['max_conservation_error'] = max_conservation_error
        hierarchy_matrix = self._extract_tensor(self.index.get_hierarchy_matrix(), "hierarchy_matrix")
        orphans = (hierarchy_matrix > 0).sum(dim=1) == 0
        audit_info['orphan_source_units'] = torch.nonzero(orphans).reshape(-1).tolist()

        return aggregated_data, audit_info

    def _audit_extensive_aggregation(
        self,
        source_data: Tensor,
        target_data: Tensor,
        var_name: str,
        basis: SpatialQuantityBasis,
    ) -> Tuple[float, Any]:
        """Covered-support conservation error and orphan (uncovered) amount of one aggregation."""
        # References: docs/source/theory/scale_coupling.md sections 2.3 and 2.9 (covered-support
        #   ledgers, uncovered amounts reported rather than routed).
        # Mathematical expression: X_(b,i) = convert(source, A_i, basis -> cell_total) for every
        #   batch index b of the leading axes; U_b = sum_{i orphan} X_(b,i); epsilon = max_b
        #   epsilon_b, the largest ConservationEnforcer batch relative error between the source
        #   with orphan children zeroed and the target, both in their area-density representation.
        # Parameter meanings: X child amount; U_b uncovered (orphan) amount of batch b; A_i child
        #   area; b batch index (fields are [*batch, n_source]).
        # Units: U in the amount unit; epsilon [1].
        # Domain/constraints: extensive bases only; orphan children are those without a parent;
        #   U is one float for a field without batch axes, else nested lists shaped like the batch
        #   axes (amounts of different batches are never added).
        # Sign convention: U_b is signed like the field.
        _mapping, orphans = self.index.membership_mapping(device=source_data.device)
        source_areas = self._extract_tensor(self.index.get_source_areas(), "source_areas").to(
            device=source_data.device
        )
        totals = convert_spatial_basis(
            source_data.to(dtype=torch.float64), source_areas.to(dtype=torch.float64), basis,
            SpatialQuantityBasis.CELL_TOTAL,
        )
        orphan_mask = orphans.to(device=source_data.device)
        uncovered = totals[..., orphan_mask].sum(dim=-1).detach().cpu().tolist()
        covered_source = torch.where(orphan_mask, torch.zeros_like(source_data), source_data)
        error = self._check_extensive_conservation(covered_source, target_data, var_name, basis)
        return error, uncovered

    def _apply_aggregation_method(
        self,
        method: Any,
        method_name: str,
        data: Tensor,
        var_name: str,
        basis: SpatialQuantityBasis,
    ) -> Tensor:
        """Run one aggregation kernel on its internal representation of the declared basis."""
        # References: docs/source/theory/scale_coupling.md sections 2.3.2 and 2.9; the shared
        #   convert_spatial_basis owner in relic/coupling/scales/types.py.
        # Mathematical expression: intensive and categorical fields are passed unchanged. An
        #   extensive field is converted to the kernel representation r: r = cell_total for
        #   area_weighted_sum (Y_j = sum_i X_i H_ij) and r = area_density for every other kernel
        #   (conservative/flow_conserving Y_j = sum_{i in j} x_i A_i / A_j, and the value-level
        #   kernels, which only average densities); X_i = x_i A_i and x_i = X_i / A_i with the
        #   source areas A_i on input, and the same conversion with the target areas A_j on output,
        #   so the result carries the input basis.
        # Parameter meanings: x_i/X_i source density/total; Y_j target value; H_ij hierarchy
        #   weight; A_i, A_j source and target areas.
        # Units: A [m2]; totals in the amount unit; densities in amount m-2.
        # Domain/constraints: areas finite and positive (checked by convert_spatial_basis);
        #   "conservative" dispatches to the area-density flow-conserving kernel for extensive
        #   fields and to the area-weighted mean otherwise.
        # Sign convention: values keep their sign; areas are positive.
        if basis is SpatialQuantityBasis.CATEGORICAL and method_name not in ('mode', 'nearest'):
            # Class codes have no arithmetic: only the area-weighted mode or a nearest member.
            raise ValueError(
                f"Aggregation method {method_name!r} averages values; variable {var_name!r} is "
                "categorical and accepts only 'mode' or 'nearest'."
            )
        if not basis.is_extensive:
            if method_name in ('area_weighted_sum', 'flow_conserving'):
                raise ValueError(
                    f"Aggregation method {method_name!r} is defined for amounts only (cell_total or "
                    f"area_density); variable {var_name!r} is {basis.value}."
                )
            if method_name == 'conservative':
                return method(data, var_name, extensive=False)
            return method(data, var_name)
        internal = (
            SpatialQuantityBasis.CELL_TOTAL
            if method_name == 'area_weighted_sum'
            else SpatialQuantityBasis.AREA_DENSITY
        )
        source_areas = self._extract_tensor(self.index.get_source_areas(), "source_areas").to(device=data.device)
        target_areas = self._extract_tensor(self.index.get_target_areas(), "target_areas").to(device=data.device)
        internal_data = convert_spatial_basis(data, source_areas, basis, internal)
        if method_name == 'conservative':
            internal_result = method(internal_data, var_name, extensive=True)
        else:
            internal_result = method(internal_data, var_name)
        return convert_spatial_basis(internal_result, target_areas, internal, basis)

    def _check_extensive_conservation(
        self,
        source_data: Tensor,
        target_data: Tensor,
        var_name: str,
        basis: SpatialQuantityBasis,
        reverse: bool = False,
    ) -> float:
        """Compare area-weighted totals of an extensive field in its area-density representation."""
        # References: docs/source/theory/scale_coupling.md section 2.9 (ConservationEnforcer ledger).
        # Mathematical expression: x = convert(X, A, basis -> area_density) on both sides, then
        #   epsilon = |sum_j y_j A_j - sum_i x_i A_i| / |sum_i x_i A_i| via ConservationEnforcer.
        # Parameter meanings: X source field, Y target field in the declared basis; x, y their
        #   area densities; A_i, A_j supports of the source and target side (swapped for reverse).
        # Units: A [m2]; x, y [amount m-2]; epsilon [1].
        # Domain/constraints: areas finite and positive; only called for extensive bases.
        # Sign convention: signed amounts; epsilon is a non-negative relative error.
        source_areas = self._extract_tensor(self.index.get_source_areas(), "source_areas")
        target_areas = self._extract_tensor(self.index.get_target_areas(), "target_areas")
        source_support, target_support = (target_areas, source_areas) if reverse else (source_areas, target_areas)
        source_density = convert_spatial_basis(
            source_data, source_support.to(device=source_data.device), basis, SpatialQuantityBasis.AREA_DENSITY
        )
        target_density = convert_spatial_basis(
            target_data, target_support.to(device=target_data.device), basis, SpatialQuantityBasis.AREA_DENSITY
        )
        return self._check_conservation(source_density, target_density, var_name, reverse=reverse)

    def disaggregate(
        self,
        data: Dict[str, Tensor],
        variables: List[str],
        strategy: Optional[Union[str, Mapping[str, str]]] = None,
        mode: str = "conservative",
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """
        Disaggregate data from coarse to fine scale.

        Parameters:
            data : Dict[str, Tensor]
                Input data to disaggregate
            variables : List[str]
                Variable names
            strategy : Optional[Union[str, Mapping[str, str]]]
                Accepted for symmetry with ``aggregate``; amounts are always split area-
                proportionally and intensive values broadcast
            mode : str
                Disaggregation mode
            variable_metadata : Optional[Dict[str, Dict[str, Any]]]
                Per-variable metadata with the explicit ``spatial_basis`` (see ``aggregate``).

        Returns:
            Tuple of disaggregated data (same basis as the input) and audit information
            including ``quantity_basis`` and ``output_basis`` per variable.
        """
        # Handle Enum values - extract value if it's an Enum
        if hasattr(strategy, 'value'):
            strategy = strategy.value
        if hasattr(mode, 'value'):
            mode = mode.value

        disaggregated_data = {}
        audit_info = {
            'method_used': {},
            'conservation_errors': {},
            'quantity_basis': {},
            'output_basis': {},
            'warnings': [],
            'n_variables': len(variables)
        }

        max_conservation_error = 0.0

        for var_name in variables:
            if var_name not in data:
                warning_msg = f"Variable {var_name} not found in data"
                logger.warning(warning_msg)
                audit_info['warnings'].append(warning_msg)
                continue

            var_data = data[var_name]
            basis = self._variable_basis(var_name, variable_metadata)
            if basis is SpatialQuantityBasis.CATEGORICAL:
                raise ValueError(
                    f"Categorical variable {var_name!r} cannot be disaggregated: a parent class code "
                    "does not determine the class codes of its children."
                )

            # Area-proportional disaggregation on the area-density representation of an
            # extensive field; intensive fields are broadcast.
            if basis.is_extensive:
                parent_areas = self._extract_tensor(self.index.get_target_areas(), "target_areas").to(
                    device=var_data.device
                )
                child_areas = self._extract_tensor(self.index.get_source_areas(), "source_areas").to(
                    device=var_data.device
                )
                parent_density = convert_spatial_basis(
                    var_data, parent_areas, basis, SpatialQuantityBasis.AREA_DENSITY
                )
                child_density = self._area_proportional_disaggregate(parent_density, var_name, extensive=True)
                disaggregated_var = convert_spatial_basis(
                    child_density, child_areas, SpatialQuantityBasis.AREA_DENSITY, basis
                )
            else:
                disaggregated_var = self._area_proportional_disaggregate(var_data, var_name, extensive=False)
            disaggregated_data[var_name] = disaggregated_var
            audit_info['method_used'][var_name] = 'area_proportional'
            audit_info['quantity_basis'][var_name] = basis.value
            audit_info['output_basis'][var_name] = basis.value

            # Check conservation for extensive variables
            if basis.is_extensive:
                conservation_error = self._check_extensive_conservation(
                    var_data, disaggregated_var, var_name, basis, reverse=True
                )
                audit_info['conservation_errors'][var_name] = conservation_error
                max_conservation_error = max(max_conservation_error, abs(conservation_error))

        audit_info['max_conservation_error'] = max_conservation_error

        return disaggregated_data, audit_info

    # Implementation of ScaleCoupling abstract methods
    def upscale(
        self,
        fine_data: Tensor,
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tensor:
        """
        Upscale from fine to coarse scale using hierarchical aggregation.

        Parameters:
            fine_data : Tensor
                Fine-scale data to upscale (the variable ``default_var``)
            variable_metadata : Optional[Dict[str, Dict[str, Any]]]
                ``{"default_var": {"spatial_basis": ...}}``; without it ``default_var`` is
                undeclared and the call raises ``ValueError``.

        Returns:
            Tensor: Upscaled coarse data
        """
        # Convert single tensor to dict format for aggregate method
        data_dict = {"default_var": fine_data}
        variables = ["default_var"]

        # Use aggregate method with conservative strategy
        aggregated_data, _ = self.aggregate(
            data=data_dict,
            variables=variables,
            strategy="conservative",
            variable_metadata=variable_metadata,
        )

        return aggregated_data["default_var"]

    def downscale(
        self,
        coarse_data: Tensor,
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tensor:
        """
        Downscale from coarse to fine scale using hierarchical disaggregation.

        Parameters:
            coarse_data : Tensor
                Coarse-scale data to downscale (the variable ``default_var``)
            variable_metadata : Optional[Dict[str, Dict[str, Any]]]
                Optional ``{"default_var": {"spatial_basis": ...}}`` (see ``upscale``).

        Returns:
            Tensor: Downscaled fine data
        """
        # Convert single tensor to dict format for disaggregate method
        data_dict = {"default_var": coarse_data}
        variables = ["default_var"]

        # Use disaggregate method
        disaggregated_data, _ = self.disaggregate(
            data=data_dict,
            variables=variables,
            variable_metadata=variable_metadata,
        )

        return disaggregated_data["default_var"]

    def _auto_select_method(self, mode: str, basis: SpatialQuantityBasis) -> Tuple[callable, str]:
        """
        Aggregation method of one variable from the mode and its declared spatial basis.

        categorical -> mode; strict_conservative and conservative -> conservative; balanced ->
        area_weighted_sum (amounts) or area_weighted_mean (intensive); fast_approx -> bilinear,
        which needs the declared source/target coordinates of the topology and raises without
        them (it is an interpolation, not a conservative remap). Other modes raise.
        """
        if basis is SpatialQuantityBasis.CATEGORICAL:
            # Class codes are never averaged or interpolated: the area-weighted mode applies.
            return self.method_map['mode'], 'mode'
        if mode in ("strict_conservative", "conservative"):
            return self.method_map['conservative'], 'conservative'
        if mode == "balanced":
            # Amounts are summed over the partition; intensive fields are area-weighted means.
            name = 'area_weighted_sum' if basis.is_extensive else 'area_weighted_mean'
            return self.method_map[name], name
        if mode == "fast_approx":
            return self.method_map['bilinear'], 'bilinear'
        raise ValueError(
            f"Unknown aggregation mode {mode!r}; accepted modes are strict_conservative, "
            "conservative, balanced and fast_approx."
        )
    
    def _area_fractions(self, dtype: torch.dtype, device: torch.device, var_name: str) -> Tensor:
        """Area fractions f_ij of every child in every parent from the validated membership."""
        # References: Jones (1999) Mon. Weather Rev. 127:2204-2210 (area-weighted cell means);
        #   docs/source/theory/scale_coupling.md section 2.9 (partition memberships).
        # Mathematical expression: f_ij = A_i M_ij / sum_k A_k M_kj with M from
        #   HierarchyIndex.membership_mapping (1 for a single parent, the declared share for a
        #   child in several parents whose shares sum to one, 0 for an orphan).
        # Parameter meanings: A_i child area; M_ij membership share; f_ij area fraction of child i
        #   in parent j (sum_i f_ij = 1 for every parent).
        # Units: A [m2]; M and f [1].
        # Domain/constraints: shared children whose shares do not sum to one raise (in
        #   membership_mapping); a parent without member area raises (no placeholder value).
        # Sign convention: A, M, f >= 0.
        source_areas = self._extract_tensor(self.index.get_source_areas(), "source_areas").to(
            dtype=dtype, device=device
        ).reshape(-1)
        mapping, _orphans = self.index.membership_mapping(dtype=dtype, device=device)
        weights = source_areas.unsqueeze(1) * mapping
        parent_area = weights.sum(dim=0)
        if bool((parent_area <= 0).any()):
            offenders = torch.nonzero(parent_area <= 0).reshape(-1).tolist()
            raise ValueError(
                f"Parent unit(s) {offenders[:8]} have no member area; the parent value of "
                f"{var_name!r} is undefined there."
            )
        return weights / parent_area

    def _area_weighted_mean(self, data: Tensor, var_name: str) -> Tensor:
        """Area-weighted mean over the members of each parent (intensive values, densities)."""
        # References: Jones (1999) Mon. Weather Rev. 127:2204-2210 (area-weighted cell means).
        # Mathematical expression: y_j = sum_i f_ij x_i with f from _area_fractions.
        # Parameter meanings: x_i child value; y_j parent value; f_ij area fraction.
        # Units: y carries the units of x; f [1].
        # Domain/constraints: data [..., n_source]; orphans carry zero fraction.
        # Sign convention: values keep their sign.
        return torch.matmul(data, self._area_fractions(data.dtype, data.device, var_name))

    def _area_weighted_sum(self, data: Tensor, var_name: str) -> Tensor:
        """Partition sum of cell totals (amounts) over the members of each parent."""
        # References: docs/source/theory/scale_coupling.md sections 2.3.2 and 2.9; RHESSys-style
        #   hierarchical accumulation of amounts (Tague and Band, 2004).
        # Mathematical expression: Y_j = sum_i X_i M_ij with M = HierarchyIndex.membership_mapping
        #   (M_ij = 1 for the single parent of child i, the stored share for a child in several
        #   parents with sum_j M_ij = 1, a zero row for an orphan).
        # Parameter meanings: X_i amount of child i; Y_j amount of parent j; M_ij share of child i
        #   in parent j. The stored normalisation of H (binary or column-normalised 1/n_j) does not
        #   enter, so a column-normalised index no longer returns an average of member totals.
        # Units: X and Y in the amount unit (for example kg); M dimensionless.
        # Domain/constraints: data are cell totals (the caller converts area densities with the
        #   source areas first); orphan amounts are excluded and reported by aggregate().
        # Sign convention: amounts keep their sign; M_ij >= 0.
        mapping, _orphans = self.index.membership_mapping(dtype=data.dtype, device=data.device)
        if data.dim() == 1:
            return torch.matmul(data.unsqueeze(0), mapping).squeeze(0)
        return torch.matmul(data, mapping)

    def _flow_conserving(self, data: Tensor, var_name: str) -> Tensor:
        """Per-parent conservative aggregation of an area density (no global rescale)."""
        # References: Jones (1999) first-order conservative remapping on a partition;
        #   docs/source/theory/scale_coupling.md section 2.9.
        # Mathematical expression: y_j = sum_i x_i A_i M_ij / A_j, so the per-parent ledger
        #   y_j A_j = sum_i x_i A_i M_ij holds exactly with the declared parent area A_j, and the
        #   global ledger sum_j y_j A_j = sum_{i not orphan} x_i A_i holds on the covered support.
        # Parameter meanings: x_i area density of child i; A_i child area; A_j declared parent area;
        #   M_ij membership share from HierarchyIndex.membership_mapping; y_j parent density.
        # Units: A [m2]; x and y in amount m-2.
        # Domain/constraints: a parent with members must have A_j > 0 (raises otherwise, no
        #   fallback to the member area); parents without members receive zero; orphan amounts
        #   are not redistributed (reported by aggregate()); there is no global rescale factor.
        # Sign convention: densities keep their sign; areas and shares are non-negative.
        source_areas = self._extract_tensor(self.index.get_source_areas(), "source_areas")
        target_areas = self._extract_tensor(self.index.get_target_areas(), "target_areas")
        work_dtype = torch.promote_types(
            torch.promote_types(data.dtype, source_areas.dtype),
            target_areas.dtype,
        )
        mapping, _orphans = self.index.membership_mapping(dtype=work_dtype, device=data.device)
        source_areas_work = source_areas.to(device=data.device, dtype=work_dtype).reshape(-1)
        target_areas_work = target_areas.to(device=data.device, dtype=work_dtype).reshape(-1)
        has_members = mapping.sum(dim=0) > 0
        invalid_parent = has_members & ~(target_areas_work > 0)
        if bool(invalid_parent.any()):
            offenders = torch.nonzero(invalid_parent).reshape(-1).tolist()
            raise ValueError(
                f"Parent unit(s) {offenders[:8]} have members but no positive declared area; the "
                "parent density y_j = sum_i x_i A_i M_ij / A_j is undefined."
            )
        safe_target_areas = torch.where(has_members, target_areas_work, torch.ones_like(target_areas_work))
        amounts = data.to(dtype=work_dtype) * source_areas_work
        if amounts.dim() == 1:
            parent_amounts = torch.matmul(amounts.unsqueeze(0), mapping).squeeze(0)
        else:
            parent_amounts = torch.matmul(amounts, mapping)
        return parent_amounts / safe_target_areas
    
    def _conservative_aggregate(self, data: Tensor, var_name: str, *, extensive: bool) -> Tensor:
        """Conservative aggregation using index weights.

        ``extensive`` comes from the variable's declared SpatialQuantityBasis; extensive data
        arrive in their area-density representation (see _apply_aggregation_method).
        """
        # Use float64 for conservative mode
        target_dtype = torch.float64
        data = data.to(dtype=target_dtype)

        # For conservative aggregation, use the same logic as flow_conserving
        # but ensure float64 precision
        if extensive:
            # Use the same conservation logic as _flow_conserving for extensive variables
            result = self._flow_conserving(data, var_name)
        else:
            # For intensive variables, use area-weighted mean
            result = self._area_weighted_mean(data, var_name)
        
        # Ensure result is also float64
        if hasattr(result, 'dtype') and result.dtype != target_dtype:
            result = result.to(dtype=target_dtype)
        
        return result
    
    def _mass_weighted(self, data: Tensor, var_name: str) -> Tensor:
        """Mass-weighted aggregation using area weights."""
        return self._area_weighted_mean(data, var_name)
    
    def _volume_weighted(self, data: Tensor, var_name: str) -> Tensor:
        """Volume-weighted aggregation using area weights."""
        return self._area_weighted_mean(data, var_name)
    
    def _nearest_neighbor(self, data: Tensor, var_name: str) -> Tensor:
        """Nearest neighbor aggregation."""
        # Simple: take first value in each group
        hierarchy_matrix = self.index.get_hierarchy_matrix()
        
        # Use unified tensor extraction logic
        hierarchy_matrix = self._extract_tensor(hierarchy_matrix, "hierarchy_matrix")
        
        # Find first non-zero index for each target unit
        result = torch.zeros(self.index.n_target_units, device=self.device)
        
        for j in range(self.index.n_target_units):
            # Find source units belonging to target j
            source_mask = hierarchy_matrix[:, j] > 0
            if source_mask.any():
                # Take first value
                first_idx = torch.where(source_mask)[0][0]
                result[j] = data[first_idx]
        
        return result

    def _resolve_bilinear_coordinates(self) -> Tuple[Tensor, Tensor]:
        """Return the node coordinates declared by the hierarchy index (its topology keys
        source_coordinates and target_coordinates); no other name is read."""
        source_coords = self.index.source_coordinates
        target_coords = self.index.target_coordinates
        if source_coords is None or target_coords is None:
            raise ValueError(
                "Bilinear interpolation requires the declared source/target coordinates "
                "(topology source_coordinates/target_coordinates); they are not inferred."
            )

        source_tensor = self._extract_tensor(source_coords, "source_coordinates").to(
            device=self.device, dtype=torch.float64
        )
        target_tensor = self._extract_tensor(target_coords, "target_coordinates").to(
            device=self.device, dtype=torch.float64
        )
        if source_tensor.dim() != 2 or source_tensor.shape[1] != 2:
            raise ValueError(f"source_coordinates must be [n_source, 2], got {tuple(source_tensor.shape)}")
        if target_tensor.dim() != 2 or target_tensor.shape[1] != 2:
            raise ValueError(f"target_coordinates must be [n_target, 2], got {tuple(target_tensor.shape)}")
        if source_tensor.shape[0] != int(self.index.n_source_units):
            raise ValueError(
                f"source_coordinates length {source_tensor.shape[0]} "
                f"does not match n_source_units={self.index.n_source_units}"
            )
        if target_tensor.shape[0] != int(self.index.n_target_units):
            raise ValueError(
                f"target_coordinates length {target_tensor.shape[0]} "
                f"does not match n_target_units={self.index.n_target_units}"
            )
        return source_tensor, target_tensor

    @staticmethod
    def _find_bracketing_values(sorted_unique: Tensor, value: float) -> Tuple[float, float]:
        """Lower and upper node coordinates around ``value`` (inside [min, max] of the nodes)."""
        lower = float(sorted_unique[sorted_unique <= value][-1].item())
        upper = float(sorted_unique[sorted_unique >= value][0].item())
        return lower, upper

    def _bilinear_interpolate(self, data: Tensor, var_name: str) -> Tensor:
        """
        Bilinear interpolation based on source/target coordinates.

        References:
            - Press et al. (2007). Numerical Recipes, bilinear interpolation.
            - Jones, P.W. (1999). Conservative remapping context for scale coupling.

        Mathematical Expression:
            f(x, y) = (1-tx)(1-ty)f00 + tx(1-ty)f10 + (1-tx)ty f01 + tx ty f11
            tx = (x - x0) / (x1 - x0), ty = (y - y0) / (y1 - y0)

        Parameter Meanings:
            x0, x1, y0, y1 : bracketing source coordinates
            f00, f10, f01, f11 : source values at four corners
            tx, ty : normalized interpolation coordinates in [0, 1]
        """
        if data.dim() == 1:
            if data.shape[0] != int(self.index.n_source_units):
                raise ValueError(
                    f"Bilinear interpolation expects n_source={self.index.n_source_units}, got {data.shape[0]}"
                )
            data_work = data.unsqueeze(1)
            squeeze_output = True
        elif data.dim() == 2:
            if data.shape[0] != int(self.index.n_source_units):
                raise ValueError(
                    f"Bilinear interpolation expects first dim n_source={self.index.n_source_units}, got {data.shape[0]}"
                )
            data_work = data
            squeeze_output = False
        else:
            raise ValueError(
                f"Bilinear interpolation supports data dims 1 or 2, got shape {tuple(data.shape)}"
            )

        source_coords, target_coords = self._resolve_bilinear_coordinates()
        source_coords = source_coords.to(device=data_work.device, dtype=torch.float64)
        target_coords = target_coords.to(device=data_work.device, dtype=torch.float64)
        unique_x = torch.sort(torch.unique(source_coords[:, 0])).values
        unique_y = torch.sort(torch.unique(source_coords[:, 1])).values

        # Corners are looked up by their exact coordinates: the brackets are values of the same
        # float64 tensor, so no rounding (whose size would depend on the coordinate unit) is needed.
        coord_lookup: Dict[Tuple[float, float], int] = {}
        for idx in range(source_coords.shape[0]):
            key = (float(source_coords[idx, 0].item()), float(source_coords[idx, 1].item()))
            if key in coord_lookup:
                raise ValueError(
                    f"Source nodes {coord_lookup[key]} and {idx} share the coordinates {key}; a bilinear "
                    "corner must be one declared node."
                )
            coord_lookup[key] = idx
        x_range = (float(unique_x[0].item()), float(unique_x[-1].item()))
        y_range = (float(unique_y[0].item()), float(unique_y[-1].item()))

        result = torch.zeros(
            (target_coords.shape[0], data_work.shape[1]),
            dtype=data_work.dtype,
            device=data_work.device,
        )

        for target_idx in range(target_coords.shape[0]):
            x_target = float(target_coords[target_idx, 0].item())
            y_target = float(target_coords[target_idx, 1].item())
            if not (x_range[0] <= x_target <= x_range[1] and y_range[0] <= y_target <= y_range[1]):
                raise ValueError(
                    f"Target {target_idx} at ({x_target}, {y_target}) lies outside the source nodes "
                    f"[{x_range[0]}, {x_range[1]}] x [{y_range[0]}, {y_range[1]}]; bilinear interpolation "
                    "does not extrapolate (no boundary value is substituted)."
                )
            x0, x1 = self._find_bracketing_values(unique_x, x_target)
            y0, y1 = self._find_bracketing_values(unique_y, y_target)

            # Corner ordering: (x0,y0), (x1,y0), (x0,y1), (x1,y1)
            corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
            corner_indices: List[int] = []
            for x_corner, y_corner in corners:
                idx = coord_lookup.get((x_corner, y_corner))
                if idx is None:
                    raise ValueError(
                        f"Bilinear corner ({x_corner}, {y_corner}) of target {target_idx} is not a "
                        "declared source node: the source coordinates must form a rectilinear grid "
                        "around every target (no nearest-node substitute)."
                    )
                corner_indices.append(idx)

            f00 = data_work[corner_indices[0]]
            f10 = data_work[corner_indices[1]]
            f01 = data_work[corner_indices[2]]
            f11 = data_work[corner_indices[3]]

            # References: Press et al. (2007) Numerical Recipes, 3rd ed., section 3.6 (bilinear
            #   interpolation on a rectangular grid).
            # Mathematical expression: tx = (x - x0) / (x1 - x0) if |x1 - x0| > eps * max(|x0|, |x1|),
            #   else tx = 0 (the bracket has collapsed to one node); ty likewise with y0, y1, y.
            # Parameter meanings: x0, x1 (y0, y1) the bracketing source coordinates, x (y) the target
            #   coordinate, eps = coupling.scales.defaults.epsilon, a relative threshold.
            # Units: x0, x1, x and y0, y1, y in the coordinate unit of the topology (for example
            #   degree or m); eps, tx and ty [1].
            # Domain/constraints: targets inside the node range only (outside raises above), one
            #   node per coordinate pair (duplicates raise above); the bracket gives x0 <= x <= x1,
            #   so tx lies in [0, 1] (clamped below); the threshold scales with the coordinates, so
            #   the weights do not depend on the coordinate unit.
            # Sign convention: tx and ty are fractions measured from the lower bracket.
            denom_x = x1 - x0
            denom_y = y1 - y0
            collapsed_x = abs(denom_x) <= self.epsilon * max(abs(x0), abs(x1))
            collapsed_y = abs(denom_y) <= self.epsilon * max(abs(y0), abs(y1))
            tx = 0.0 if collapsed_x else (x_target - x0) / denom_x
            ty = 0.0 if collapsed_y else (y_target - y0) / denom_y
            tx = max(0.0, min(1.0, tx))
            ty = max(0.0, min(1.0, ty))

            w00 = (1.0 - tx) * (1.0 - ty)
            w10 = tx * (1.0 - ty)
            w01 = (1.0 - tx) * ty
            w11 = tx * ty
            result[target_idx] = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11

        if squeeze_output:
            return result.squeeze(1)
        return result
    
    def _harmonic_mean(self, data: Tensor, var_name: str) -> Tensor:
        """Area-weighted harmonic mean for resistance-like variables (parallel paths)."""
        # References: Monteith and Unsworth (2013) Principles of Environmental Physics
        #   (resistances in parallel); Baldocchi et al. (1987) Boundary-Layer Meteorol. 40:123-136.
        # Mathematical expression: 1 / y_j = sum_i f_ij / x_i, so y_j lies between the smallest
        #   and the largest member value.
        # Parameter meanings: x_i child resistance; y_j parent resistance; f_ij area fraction.
        # Units: x and y share units (for example s m-1); f [1].
        # Domain/constraints: member values must be positive (raises otherwise, no clamping);
        #   orphans ignored.
        # Sign convention: resistances are positive.
        fractions = self._area_fractions(data.dtype, data.device, var_name)
        members = fractions.sum(dim=1) > 0
        if bool((data[..., members] <= 0).any()):
            raise ValueError(f"Harmonic mean of {var_name!r} needs positive member values.")
        safe = torch.where(members, data, torch.ones_like(data))
        return 1.0 / torch.matmul(1.0 / safe, fractions)

    def _bounded_area_mean(self, data: Tensor, var_name: str) -> Tensor:
        """Area-weighted mean of a fraction whose values must lie in [0, 1]."""
        # References: Pielke (2002) Mesoscale Meteorological Modeling (fractional cover
        #   aggregation); Jones (1999) area-weighted cell means.
        # Mathematical expression: y_j = sum_i f_ij x_i with 0 <= x_i <= 1, hence 0 <= y_j <= 1.
        # Parameter meanings: x_i child fraction; y_j parent fraction; f_ij area fraction.
        # Units: x, y, f [1].
        # Domain/constraints: member values outside [0, 1] raise (no clipping); orphans ignored.
        # Sign convention: fractions are non-negative.
        fractions = self._area_fractions(data.dtype, data.device, var_name)
        members = fractions.sum(dim=1) > 0
        member_values = data[..., members]
        if bool((member_values < 0).any()) or bool((member_values > 1).any()):
            raise ValueError(f"Bounded area mean of {var_name!r} needs member values in [0, 1].")
        return torch.matmul(data, fractions)

    def _area_weighted_mode(self, data: Tensor, var_name: str) -> Tensor:
        """Area-weighted mode of a categorical field (AggregationMethod.MODE, strategy ``mode``)."""
        # References: Friedl & Brodley (1997) IEEE Trans. Geosci. Remote Sens. 35:716-724
        #   (majority-area land cover aggregation); CF Conventions 1.8 section 7.3 (cell_methods
        #   "area: mode", the mode with respect to area).
        # Mathematical expression: S_cj = sum_i A_i M_ij [x_i = c];
        #   mode_j = min { c : S_cj = max_c' S_c'j } (the largest area share, ties broken to the
        #   smallest class code).
        # Parameter meanings: x_i class code of child i; A_i child area; M_ij membership share from
        #   HierarchyIndex.membership_mapping (1 for a single parent, the declared share for a
        #   child in several parents, 0 for an orphan); S_cj area of class c in parent j.
        # Units: A_i and S_cj [m2]; x_i and mode_j are dimensionless class codes.
        # Domain/constraints: one-dimensional data [n_source]; every parent must have member area
        #   (raises otherwise, no placeholder class); the cell count of a class does not enter.
        # Sign convention: areas and shares are non-negative; codes are returned unchanged.
        if data.dim() != 1:
            raise ValueError(
                f"Area-weighted mode of {var_name!r} expects one class code per source unit, got "
                f"shape {tuple(data.shape)}."
            )
        source_areas = self._extract_tensor(self.index.get_source_areas(), "source_areas").to(
            device=data.device, dtype=torch.float64
        ).reshape(-1)
        mapping, _orphans = self.index.membership_mapping(dtype=torch.float64, device=data.device)
        shares = source_areas.unsqueeze(1) * mapping
        parent_area = shares.sum(dim=0)
        if bool((parent_area <= 0).any()):
            offenders = torch.nonzero(parent_area <= 0).reshape(-1).tolist()
            raise ValueError(
                f"Parent unit(s) {offenders[:8]} have no member area; the mode of {var_name!r} is "
                "undefined there."
            )
        return WeightOps.weighted_mode(data, shares)

    def _circular_mean(self, data: Tensor, var_name: str) -> Tensor:
        """Area-weighted circular mean for angles (wind direction, slope aspect) in radians."""
        # References: Mardia and Jupp (2000) Directional Statistics, chapter 2; Jammalamadaka and
        #   SenGupta (2001) Topics in Circular Statistics.
        # Mathematical expression: y_j = atan2(sum_i f_ij sin x_i, sum_i f_ij cos x_i).
        # Parameter meanings: x_i child angle; y_j parent mean angle; f_ij area fraction.
        # Units: x, y [rad]; f [1].
        # Domain/constraints: y in (-pi, pi]; orphans ignored.
        # Sign convention: counter-clockwise positive angles.
        fractions = self._area_fractions(data.dtype, data.device, var_name)
        return torch.atan2(torch.matmul(torch.sin(data), fractions), torch.matmul(torch.cos(data), fractions))

    def _component_wise_mean(self, data: Tensor, var_name: str) -> Tensor:
        """Area-weighted mean of each vector component; data [n_source, n_components]."""
        # References: Pielke (2002) Mesoscale Meteorological Modeling (momentum aggregation).
        # Mathematical expression: V_jc = sum_i f_ij V_ic for every component c.
        # Parameter meanings: V_ic component c of child i; V_jc of parent j; f_ij area fraction.
        # Units: V keeps its units (for example m s-1); f [1].
        # Domain/constraints: 1-D data are a scalar field (area-weighted mean); 2-D data carry
        #   the components in the last dimension; orphans ignored.
        # Sign convention: components keep their sign.
        if data.dim() == 1:
            return self._area_weighted_mean(data, var_name)
        if data.dim() != 2:
            raise NotImplementedError(
                "component_wise_mean expects data shaped as [n_source, n_components]"
            )
        fractions = self._area_fractions(data.dtype, data.device, var_name)
        return torch.matmul(fractions.transpose(0, 1), data)

    def _geometric_mean(self, data: Tensor, var_name: str) -> Tensor:
        """Area-weighted geometric mean for log-normally distributed variables."""
        # References: Limpert et al. (2001) BioScience 51:341-352 (log-normal distributions);
        #   Aitchison and Brown (1957) The Lognormal Distribution.
        # Mathematical expression: y_j = exp(sum_i f_ij ln x_i).
        # Parameter meanings: x_i child value; y_j parent value; f_ij area fraction.
        # Units: y carries the units of x; f [1].
        # Domain/constraints: member values must be positive (raises otherwise, no clamping);
        #   orphans ignored.
        # Sign convention: values are positive.
        fractions = self._area_fractions(data.dtype, data.device, var_name)
        members = fractions.sum(dim=1) > 0
        if bool((data[..., members] <= 0).any()):
            raise ValueError(f"Geometric mean of {var_name!r} needs positive member values.")
        safe = torch.where(members, data, torch.ones_like(data))
        return torch.exp(torch.matmul(torch.log(safe), fractions))

    def _effective_roughness_length(self, data: Tensor, var_name: str) -> Tensor:
        """Effective roughness length of each parent from linearly additive drag coefficients."""
        # References: Byun, D.W. et al. (1999) Chapter 12 MCIP, CMAQ Science Documentation,
        #   EPA/600/R-99/030, Eq. 12-48 (C_D = u*^2 / U^2 = sum_j f_j C_Dj at the blending height)
        #   and Eq. 12-37' (the resulting average roughness length); z_r = 10 m as used in MCIP.
        # Mathematical expression: with the neutral drag coefficient C_Di = [kappa / ln(z_r / z0_i)]^2,
        #   sum_i f_ij C_Di = [kappa / ln(z_r / z0_j)]^2 gives
        #   z0_j = z_r exp(-(sum_i f_ij / ln(z_r / z0_i)^2)^(-1/2)); kappa cancels.
        # Parameter meanings: z0_i roughness length of child i; z0_j effective roughness length of
        #   parent j; z_r blending height (coupling.scales.defaults.roughness_blending_height);
        #   f_ij = A_i M_ij / sum_k A_k M_kj area fraction of child i in parent j, with M the
        #   membership mapping of HierarchyIndex.membership_mapping and A_i the child areas.
        # Units: z0_i, z0_j, z_r [m]; A_i [m2]; f_ij and M_ij [1].
        # Domain/constraints: 0 < z0_i < z_r for every member child (raises otherwise, no clipping);
        #   every parent must have members (raises otherwise); orphan children carry zero weight;
        #   data are [..., n_source] with the children in the last dimension.
        # Sign convention: all lengths and fractions are positive; z0_j lies between the smallest
        #   and the largest member roughness length.
        z_r = float(
            self.get_parameter_value(
                "roughness_blending_height",
                namespace=DEFAULTS_NS,
                units="m",
                validate=True,
                required=True,
            )
        )
        source_areas = self._extract_tensor(self.index.get_source_areas(), "source_areas")
        work_dtype = torch.promote_types(data.dtype, source_areas.dtype)
        mapping, orphans = self.index.membership_mapping(dtype=work_dtype, device=data.device)
        areas = source_areas.to(device=data.device, dtype=work_dtype).reshape(-1)
        area_weights = areas.unsqueeze(1) * mapping
        parent_area = area_weights.sum(dim=0)
        if bool((parent_area <= 0).any()):
            offenders = torch.nonzero(parent_area <= 0).reshape(-1).tolist()
            raise ValueError(
                f"Parent unit(s) {offenders[:8]} have no member area; the effective roughness "
                f"length of {var_name!r} is undefined there."
            )
        fractions = area_weights / parent_area

        z0 = data.to(dtype=work_dtype)
        members = ~orphans.to(device=data.device)
        member_z0 = z0[..., members]
        if bool((member_z0 <= 0).any()) or bool((member_z0 >= z_r).any()):
            raise ValueError(
                f"Roughness lengths of {var_name!r} must satisfy 0 < z0 < z_r = {z_r} m for the "
                "drag-coefficient blending; got values outside that open interval."
            )
        # Orphan children have zero fractions; a neutral placeholder keeps their log finite.
        z0_safe = torch.where(members, z0, torch.full_like(z0, 0.5 * z_r))
        inverse_log_sq = torch.log(z_r / z0_safe) ** -2
        blended = torch.matmul(inverse_log_sq, fractions)
        return z_r * torch.exp(-torch.rsqrt(blended))

    def _area_proportional_disaggregate(self, data: Tensor, var_name: str, *, extensive: bool) -> Tensor:
        """Disaggregate parent values to member children with area-proportional closure."""
        # References: O'Neill et al. (1986) hierarchy theory as cited by the class;
        #   docs/source/theory/scale_coupling.md section 2.9.
        # Mathematical expression: intensive or categorical X_i = Y_j (membership broadcast, or a
        #   convex combination sum_j H_ij Y_j for a child in several parents with sum_j H_ij = 1);
        #   extensive per-area density X_i = Y_j A_j / sum_{k in j} A_k, which closes
        #   sum_{i in j} X_i A_i = Y_j A_j exactly even when the declared parent area A_j
        #   differs from the member area sum.
        # Parameter meanings: Y_j parent value; X_i child value; A_i child area; A_j declared
        #   parent area; membership {k in j} is hierarchy_matrix[k, j] > 0; ``extensive`` comes
        #   from the declared SpatialQuantityBasis and the data are area densities then.
        # Units: A [m2]; X and Y share the input units (densities per m2 for extensive
        #   variables, as the conservation enforcer weights them by area).
        # Domain/constraints: the extensive split requires a strict partition (every child in
        #   exactly one parent, else ValueError); a parent carrying non-zero extensive data must
        #   have members with positive total area; parents without members contribute nothing.
        # Sign convention: values keep their sign; the closure factor A_j / sum_k A_k is positive.
        if not extensive:
            return self.index.disaggregate_weights(data)
        membership = self.index.partition_membership().to(device=data.device, dtype=data.dtype)
        source_areas = self._extract_tensor(self.index.get_source_areas(), "source_areas").to(
            device=data.device, dtype=data.dtype
        )
        target_areas = self._extract_tensor(self.index.get_target_areas(), "target_areas").to(
            device=data.device, dtype=data.dtype
        )
        member_area = torch.matmul(source_areas.reshape(1, -1), membership).reshape(-1)
        has_members = member_area > 0
        parent_values = data.reshape(-1, data.shape[-1]) if data.dim() > 1 else data.reshape(1, -1)
        orphaned = (~has_members).unsqueeze(0) & (parent_values != 0)
        if bool(orphaned.any()):
            offenders = torch.nonzero(orphaned.any(dim=0)).reshape(-1).tolist()
            raise ValueError(
                f"Extensive variable {var_name} carries data on parent unit(s) {offenders[:8]} "
                "that have no member area; area-proportional disaggregation is undefined there."
            )
        safe_member_area = torch.where(has_members, member_area, torch.ones_like(member_area))
        closure = torch.where(has_members, target_areas / safe_member_area, torch.zeros_like(member_area))
        return self.index.disaggregate_weights(data * closure)
    
    def _resolve_numerical_epsilon(self) -> float:
        """Return coupling.scales.defaults.epsilon, the numerical threshold of the scale operators
        (here the relative size below which a bilinear bracket counts as one node); a missing or
        invalid value raises from the parameter system."""
        value = self.get_parameter_value(
            "epsilon",
            namespace=DEFAULTS_NS,
            units="dimensionless",
            validate=True,
            required=True,
        )
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        return float(value)

    def _get_variable_metadata(
        self,
        var_name: str,
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Return ``(metadata, source)`` for one variable; nothing is inferred from its name.

        Order: the call's ``variable_metadata`` (source ``"call"``), then the YAML variable
        catalogs ``coupling.scales.<scale>.variables`` and ``coupling.scales.defaults.variables``
        (source = namespace). Without either the variable is undeclared and a ValueError is
        raised; no default metadata is applied.
        """
        if variable_metadata:
            for candidate in (var_name, var_name.lower()):
                if candidate in variable_metadata:
                    entry = variable_metadata[candidate]
                    if not isinstance(entry, dict):
                        raise TypeError(
                            f"variable_metadata[{candidate!r}] must be a mapping, got {type(entry).__name__}."
                        )
                    return dict(entry), "call"

        if self.parameter_manager is None:
            self.parameter_manager = self.get_parameter_manager()

        key = var_name.lower()
        if key in self._variable_metadata_cache:
            return self._variable_metadata_cache[key]

        namespaces = []
        source_scale = getattr(self.index, "source_scale", None)
        target_scale = getattr(self.index, "target_scale", None)
        for scale in (source_scale, target_scale):
            if scale:
                namespaces.append(f"coupling.scales.{scale}")
        namespaces.append(DEFAULTS_NS)

        def _read_catalog(namespace: str) -> Optional[Dict[str, Any]]:
            return self.get_parameter_value(
                "variables",
                namespace=namespace,
                validate=True,
                expected_type=dict,
                required=False,
            )

        found = find_catalog_entry(var_name, namespaces, _read_catalog)
        if found is None:
            raise ValueError(
                f"Variable '{var_name}' is not declared: pass variable_metadata with spatial_basis or "
                f"add it to one of the variable catalogs {namespaces} (<namespace>.variables); no "
                "default metadata is applied."
            )
        self._variable_metadata_cache[key] = found
        return found

    def _variable_basis(
        self,
        var_name: str,
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> SpatialQuantityBasis:
        """Explicit SpatialQuantityBasis of one variable (a bare 'extensive' raises as ambiguous)."""
        metadata, _source = self._get_variable_metadata(var_name, variable_metadata)
        return resolve_spatial_basis(var_name, metadata)

    def _check_conservation(
        self,
        source_data: Tensor,
        target_data: Tensor,
        var_name: str,
        reverse: bool = False
    ) -> float:
        """Check conservation using centralized conservation enforcer."""
        if self._conservation_enforcer is None:
            raise RuntimeError("ConservationEnforcer must be initialized during __init__.")
        
        if reverse:
            source_weights = self.index.get_target_areas()
            target_weights = self.index.get_source_areas()
        else:
            source_weights = self.index.get_source_areas()
            target_weights = self.index.get_target_areas()

        source_weights = self._extract_tensor(source_weights, "source_weights")
        target_weights = self._extract_tensor(target_weights, "target_weights")

        # The shared acceptance rule (conservation_checker.ledger_acceptance): the rounding bound of
        # the coarsest floating dtype of the field, the result and the areas admits the rounding of
        # the reduction, so a float32 field is judged on its own rounding, never on a float64 scale.
        # The kernels map [..., n_source] to [..., n_target]: every leading axis is a batch with its
        # own ledger (batch_ndim = ndim - 1), and the returned error is the largest batch error.
        if source_data.dim() < 1:
            raise ValueError(f"The ledger of {var_name!r} needs a field with its support on the last axis.")
        diagnostics = self._conservation_enforcer.check_conservation(
            source_data=source_data,
            target_data=target_data,
            property_name=var_name,
            source_weights=source_weights,
            target_weights=target_weights,
            rounding_dtype=coarsest_floating_dtype(
                source_data.dtype,
                target_data.dtype,
                torch.as_tensor(source_weights).dtype,
                torch.as_tensor(target_weights).dtype,
            ),
            tolerance=self.conservation_tolerance,
            batch_ndim=source_data.dim() - 1,
        )

        return diagnostics.relative_error
