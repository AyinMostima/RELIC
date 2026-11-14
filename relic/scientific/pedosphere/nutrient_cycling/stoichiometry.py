"""
Stoichiometry Module for RELIC Nutrient Cycling

This module implements stoichiometric constraints and coupling between
carbon and nitrogen cycles, ensuring mass balance and realistic C:N ratios.

Scientific Basis:
-----------------
Stoichiometric constraints ensure that C and N cycling are coupled through
microbial biomass and decomposition processes, following Redfield-type ratios.

References:
- Manzoni, S. & Porporato, A. (2009). Substrate-decomposer stoichiometry
- McGroddy, M.E. et al. (2004). Scaling of C:N:P stoichiometry in forests
- Cleveland, C.C. & Liptzin, D. (2007). C:N:P stoichiometry in soil

Mathematical Expression:
-----------------------
Stoichiometric constraint:
C_flux / N_flux = (C/N)_target +/- flexibility

Mass balance:
SigmaC_in - SigmaC_out = DeltaC_pools
SigmaN_in - SigmaN_out = DeltaN_pools
"""

from typing import Dict, List, Optional, Tuple
import torch

from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.abstractions import StateDict, FluxDict

from relic.core.functor import ScaleCouplingFunctor
from relic.scientific.pedosphere.api_contract import PedosphereContractMixin


class StoichiometryConstraint(PedosphereContractMixin, ScaleCouplingFunctor):
    """
    Stoichiometric constraint functor for C-N coupling.
    
    References:
    - Manzoni & Porporato (2009) theoretical framework
    - McGroddy et al. (2004) forest C:N:P scaling
    - Cleveland & Liptzin (2007) soil stoichiometry
    
    Mathematical Expression:
    -----------------------
    Constraint enforcement:
    If |C_flux/N_flux - CN_target| > tolerance:
        Apply correction to balance fluxes
    
    Mass balance verification:
    Error = |SigmaInputs - SigmaOutputs - DeltaPools| / |SigmaInputs|
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    def get_dependencies(self) -> List[str]:
        """
        Get list of dependencies for this functor.

        Nitrogen dependencies enforce post-processing order. Carbon coupling is
        supplied through ``diagnostics.carbon_fluxes`` so the compiled graph
        can select one decomposition closure without simultaneous dependencies.

        Returns:
            List[str]: List of dependency names
        """
        dependencies = [
            "mineralization", "nitrification", "denitrification",
            "nitrogen_leaching", "root_nutrient_uptake"
        ]

        return self._merge_declared_dependencies(dependencies)

    
    def __init__(self, name: str = "stoichiometry_constraint", config: Optional[Dict] = None):
        if config is None:
            config = {}

        # Prepare metadata first to pass to base class
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Stoichiometric constraints for C-N coupling",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.POSTPROCESSING,
            mathematical_expression="C/N balance with mass conservation",
            references=[
                "Manzoni, S. & Porporato, A. (2009). Substrate-decomposer stoichiometry",
                "Cleveland, C.C. & Liptzin, D. (2007). C:N:P stoichiometry in soil",
                "Theory doc: docs/source/theory/stoichiometry_constraint.md"
            ],
            parameters={
                "pedosphere.stoichiometry.microbial_biomass_cn": (
                    "C:N ratio of microbial biomass [dimensionless]"
                ),
                "pedosphere.stoichiometry.metabolic_litter_cn": (
                    "C:N ratio of metabolic litter [dimensionless]"
                ),
                "pedosphere.stoichiometry.structural_litter_cn": (
                    "C:N ratio of structural litter [dimensionless]"
                ),
                "pedosphere.stoichiometry.fast_som_cn": (
                    "C:N ratio of the fast soil organic matter pool [dimensionless]"
                ),
                "pedosphere.stoichiometry.slow_som_cn": (
                    "C:N ratio of the slow soil organic matter pool [dimensionless]"
                ),
                "pedosphere.stoichiometry.passive_som_cn": (
                    "C:N ratio of the passive soil organic matter pool [dimensionless]"
                ),
                "pedosphere.numerical.stoichiometric_flexibility": (
                    "Allowed relative deviation from the target C:N ratios [dimensionless]"
                ),
                "pedosphere.numerical.stoichiometric_correction_strength": (
                    "Strength of the stoichiometric correction (0 none, 1 full) [dimensionless]"
                ),
            },
            input_variables=[
                "diagnostics.carbon_fluxes",
                "diagnostics.nitrogen_fluxes",
                "diagnostics.carbon_inputs",
                "diagnostics.nitrogen_inputs",
                "diagnostics.carbon_outputs",
                "diagnostics.nitrogen_outputs",
            ],
            output_variables=[
                "diagnostics.stoichiometry_mass_balance_error_carbon",
                "diagnostics.stoichiometry_mass_balance_error_nitrogen",
                "diagnostics.stoichiometry_mass_balance_error_total",
                "diagnostics.stoichiometry_constraint_violations_count",
            ],
            # A carbon or nitrogen budget with an absent input or output term is not
            # evaluable, so its error diagnostic is emitted only when all its terms exist.
            optional_output_variables=[
                "diagnostics.stoichiometry_mass_balance_error_carbon",
                "diagnostics.stoichiometry_mass_balance_error_nitrogen",
                "diagnostics.stoichiometry_mass_balance_error_total",
            ],
            conservation_laws=["carbon", "nitrogen", "mass"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["carbon_cycle", "nitrogen_cycle"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        
        config = config or {}
        config_for_parameters = dict(config)
        parameter_overrides = dict(config_for_parameters.get("parameters", {}))
        for direct_key in ("stoichiometric_flexibility",):
            if direct_key in config:
                parameter_overrides[direct_key] = config[direct_key]
        if parameter_overrides:
            config_for_parameters["parameters"] = parameter_overrides

        # Initialize base class with proper parameters
        super().__init__(
            name=name,
            config=config_for_parameters,
            metadata=metadata,
            source_scale="process",
            target_scale="ecosystem",
            coupling_type="hierarchical",
            conservation_properties=["carbon", "nitrogen", "mass"],
            device=None
        )
        
        self.metadata = metadata
        self._config_raw = dict(config) if config else {}
        self._execution_count = 0
        self.optional_input_variables = [
            "diagnostics.carbon_fluxes",
            "diagnostics.nitrogen_fluxes",
            "diagnostics.carbon_inputs",
            "diagnostics.nitrogen_inputs",
            "diagnostics.carbon_outputs",
            "diagnostics.nitrogen_outputs",
        ]
        
        # Target C:N ratios
        self.target_ratios = {
            'metabolic_litter': self.get_parameter_value(
                'metabolic_litter_cn',
                namespace='pedosphere.stoichiometry', required=True),
            'structural_litter': self.get_parameter_value(
                'structural_litter_cn',
                namespace='pedosphere.stoichiometry', required=True),
            'fast_som': self.get_parameter_value(
                'fast_som_cn',
                namespace='pedosphere.stoichiometry', required=True),
            'slow_som': self.get_parameter_value(
                'slow_som_cn',
                namespace='pedosphere.stoichiometry', required=True),
            'passive_som': self.get_parameter_value(
                'passive_som_cn',
                namespace='pedosphere.stoichiometry', required=True),
            'microbial': self.get_parameter_value(
                'microbial_biomass_cn',
                namespace='pedosphere.stoichiometry', required=True),
        }
        
        # Flexibility parameters
        self.flexibility = self.get_parameter_value(
            'stoichiometric_flexibility',
            namespace='pedosphere.numerical', required=True)
        self.correction_strength = self.get_parameter_value(
            'stoichiometric_correction_strength',
            namespace='pedosphere.numerical', required=True)

        # Tracking
        self.constraint_violations = 0
        self.mass_balance_errors = []

    def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
        """
        Upscale constraint information (average fine-scale values to coarse scale).

        For stoichiometric ratios, upscaling typically involves averaging
        because ratios are intensive properties.
        """
        if fine_data.dim() == 0:
            return fine_data
        if fine_data.dim() == 1:
            return fine_data.mean()
        return fine_data.mean(dim=-1, keepdim=True)
    
    def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
        """
        Downscale constraint information (not typically used for stoichiometry).
        """
        return coarse_data.repeat_interleave(2, dim=-1) if coarse_data.dim() > 0 else coarse_data
    

    
# get_parameter_value removed - using BiologicalProcess implementation
    
    def forward(
        # References: Manzoni and Porporato (2009); Cleveland and Liptzin (2007);
        #   docs/source/theory/stoichiometry_constraint.md.
        # Mathematical expression: for each pool with paired delta_<pool>_c and delta_<pool>_n
        #   fluxes, corrected fluxes hold the target ratio (C:N)_target and a violation indicator
        #   is raised where the pair departs from it; mass balance
        #   err_C = sum(C_in) - sum(C_out) - sum(delta C), err_N likewise, err_total combines them.
        # Parameter meanings: (C:N)_target the pool stoichiometric ratio from ParameterManager;
        #   delta fluxes the reported carbon and nitrogen pool tendencies; inputs and outputs the
        #   declared boundary terms; dt the step.
        # Units: carbon terms [g C m-2 s-1]; nitrogen terms [g N m-2 s-1]; ratios [g C g N-1];
        #   violation indicators [1]; balance errors in the flux units times dt.
        # Domain/constraints: required flux dictionaries are supplied via state diagnostics or
        #   kwargs; ratios are positive and finite; all outputs are diagnostics and no state is
        #   advanced by this Functor.
        # Sign convention: inputs positive into the system; outputs and losses positive out.
        self,
        state: StateDict,
        time: float,
        dt: torch.Tensor | float,
        **kwargs,
    ) -> FluxDict:
        """
        Apply stoichiometric constraints and check mass balance.
        
        Parameters:
        -----------
        state : StateDict
            Current system state
        time : float
            Current time
        dt : float
            Time step
        **kwargs : Dict[str, Any]
            Optional overrides:
            - carbon_fluxes, nitrogen_fluxes
            - carbon_inputs, nitrogen_inputs
            - carbon_outputs, nitrogen_outputs
        Returns:
        --------
        FluxDict
            Corrected fluxes and constraint diagnostics
        """
        # References: Manzoni and Porporato 2009; Cleveland and Liptzin 2007
        # Equation: target CN => N_target = C_flux / target_cn; apply correction with flexibility
        # Parameters: target_cn, stoichiometric_flexibility, correction_strength, mass_balance_tolerance
        # Extract fluxes from diagnostics namespace unless overrides are provided
        c_fluxes = kwargs.get("carbon_fluxes")
        if c_fluxes is None:
            c_fluxes = state.get("diagnostics.carbon_fluxes")
        if c_fluxes is None:
            raise KeyError(
                "Missing required input 'diagnostics.carbon_fluxes'. "
                "Provide it in state or via kwargs['carbon_fluxes']."
            )
        if not isinstance(c_fluxes, dict):
            raise TypeError(
                f"'diagnostics.carbon_fluxes' must be dict, got {type(c_fluxes)}"
            )

        n_fluxes = kwargs.get("nitrogen_fluxes")
        if n_fluxes is None:
            n_fluxes = state.get("diagnostics.nitrogen_fluxes")
        if n_fluxes is None:
            raise KeyError(
                "Missing required input 'diagnostics.nitrogen_fluxes'. "
                "Provide it in state or via kwargs['nitrogen_fluxes']."
            )
        if not isinstance(n_fluxes, dict):
            raise TypeError(
                f"'diagnostics.nitrogen_fluxes' must be dict, got {type(n_fluxes)}"
            )

        carbon_inputs = kwargs.get("carbon_inputs")
        if carbon_inputs is None:
            carbon_inputs = state.get("diagnostics.carbon_inputs")

        nitrogen_inputs = kwargs.get("nitrogen_inputs")
        if nitrogen_inputs is None:
            nitrogen_inputs = state.get("diagnostics.nitrogen_inputs")

        carbon_outputs = kwargs.get("carbon_outputs")
        if carbon_outputs is None:
            carbon_outputs = state.get("diagnostics.carbon_outputs")

        nitrogen_outputs = kwargs.get("nitrogen_outputs")
        if nitrogen_outputs is None:
            nitrogen_outputs = state.get("diagnostics.nitrogen_outputs")
        
        # Initialize corrections
        corrected_c_fluxes = {}
        corrected_n_fluxes = {}
        violations = {}
        
        # Check and correct each pool
        for pool in self.target_ratios.keys():
            c_flux_key = f'delta_{pool}_c'
            n_flux_key = f'delta_{pool}_n'
            
            if c_flux_key in c_fluxes and n_flux_key in n_fluxes:
                c_flux = c_fluxes[c_flux_key]
                n_flux = n_fluxes[n_flux_key]
                target_cn = self.target_ratios[pool]
                
                # Calculate stoichiometric correction
                corrected_fluxes = self._apply_stoichiometric_correction(
                    c_flux, n_flux, target_cn, pool
                )
                
                corrected_c_fluxes[c_flux_key] = corrected_fluxes['carbon']
                corrected_n_fluxes[n_flux_key] = corrected_fluxes['nitrogen']
                violations[f'{pool}_violation'] = corrected_fluxes['violation']
        
        # Check the carbon and nitrogen budgets; a budget with an absent term is not
        # evaluable (its term is never taken as zero or as a default).
        mass_balance = self._check_mass_balance(
            corrected_c_fluxes,
            corrected_n_fluxes,
            carbon_inputs,
            nitrogen_inputs,
            carbon_outputs,
            nitrogen_outputs,
            dt,
        )
        
        # Update tracking
        def _has_violation(value: object) -> bool:
            if isinstance(value, torch.Tensor):
                return bool(torch.any(value > 0))
            return bool(value)

        if any(_has_violation(value) for value in violations.values()):
            self.constraint_violations += 1
        if 'total_error' in mass_balance:
            self.mass_balance_errors.append(mass_balance['total_error'])
        
        output: Dict[str, torch.Tensor] = {}
        for key, value in corrected_c_fluxes.items():
            output[f"diagnostics.stoichiometry_corrected_{key}"] = value
        for key, value in corrected_n_fluxes.items():
            output[f"diagnostics.stoichiometry_corrected_{key}"] = value
        for key, value in violations.items():
            output[f"diagnostics.stoichiometry_{key}"] = value

        for element in ('carbon', 'nitrogen', 'total'):
            if f'{element}_error' in mass_balance:
                output[f"diagnostics.stoichiometry_mass_balance_error_{element}"] = (
                    mass_balance[f'{element}_error']
                )
        output["diagnostics.stoichiometry_constraint_violations_count"] = torch.as_tensor(
            self.constraint_violations
        )

        return output
    
    def _apply_stoichiometric_correction(self,
                                       carbon_flux: torch.Tensor,
                                       nitrogen_flux: torch.Tensor,
                                       target_cn: float,
                                       pool_name: str) -> Dict[str, torch.Tensor]:
        """
        Apply stoichiometric correction to maintain target C:N ratio.
        
        Parameters:
        -----------
        carbon_flux : torch.Tensor
            Carbon flux [g C m^-2 s^-1]
        nitrogen_flux : torch.Tensor
            Nitrogen flux [g N m^-2 s^-1]
        target_cn : float
            Target C:N ratio
        pool_name : str
            Pool identifier
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Corrected fluxes and violation indicator
        """
        # References: Manzoni and Porporato (2009); Cleveland and Liptzin (2007).
        # Mathematical expression: v = 1 iff |dC - CN dN| > phi CN |dN|, which equals
        #   |dC/dN - CN| / CN > phi for dN != 0 and, for dN = 0, v = 1 iff dC != 0.
        # Parameter meanings: dC and dN the carbon and nitrogen changes of the pool,
        #   CN the target C:N ratio, phi the stoichiometric flexibility.
        # Units: dC [g C m-2], dN [g N m-2]; CN [g C g-1 N]; phi and v [1].
        # Domain/constraints: CN > 0 and phi >= 0; the test is division-free, so no
        #   substituted nitrogen change stands in for a zero or tiny dN (a pure carbon
        #   change violates any finite target, two zero changes do not).
        # Sign convention: dC and dN are signed pool changes (gain positive).
        target_tensor = torch.as_tensor(
            target_cn, dtype=carbon_flux.dtype, device=carbon_flux.device
        )
        needs_correction = torch.abs(carbon_flux - target_tensor * nitrogen_flux) > (
            self.flexibility * target_tensor * torch.abs(nitrogen_flux)
        )
        
        if torch.any(needs_correction):
            # References:
            #   - Manzoni and Porporato (2009) Global Change Biology, doi:10.1111/j.1365-2486.2008.01716.x
            # Mathematical expression:
            #   CN_flux = C_flux / N_flux
            #   if CN_flux > CN_target then adjust N_flux, else adjust C_flux
            # Parameter meanings:
            #   correction_strength: relaxation coefficient for stoichiometric correction [dimensionless]
            #   target_cn: target C:N ratio for the pool [dimensionless]
            # Apply element-wise correction to avoid ambiguous tensor truth-value branching.
            c_magnitude = torch.abs(carbon_flux)
            n_magnitude = torch.abs(nitrogen_flux)
            correction_factor = torch.as_tensor(
                self.correction_strength,
                dtype=carbon_flux.dtype,
                device=carbon_flux.device,
            )

            carbon_excess_mask = torch.logical_and(
                needs_correction,
                c_magnitude > (n_magnitude * target_cn),
            )
            nitrogen_excess_mask = torch.logical_and(
                needs_correction,
                torch.logical_not(carbon_excess_mask),
            )

            corrected_n = carbon_flux / target_cn
            corrected_c = nitrogen_flux * target_cn

            nitrogen_corrected = torch.where(
                carbon_excess_mask,
                nitrogen_flux + correction_factor * (corrected_n - nitrogen_flux),
                nitrogen_flux,
            )
            carbon_corrected = torch.where(
                nitrogen_excess_mask,
                carbon_flux + correction_factor * (corrected_c - carbon_flux),
                carbon_flux,
            )
        else:
            carbon_corrected = carbon_flux
            nitrogen_corrected = nitrogen_flux
        
        return {
            'carbon': carbon_corrected,
            'nitrogen': nitrogen_corrected,
            'violation': needs_correction.float()
        }
    
    def _check_mass_balance(self,
                           carbon_fluxes: Dict[str, torch.Tensor],
                           nitrogen_fluxes: Dict[str, torch.Tensor],
                           carbon_inputs: torch.Tensor,
                           nitrogen_inputs: torch.Tensor,
                           carbon_outputs: torch.Tensor,
                           nitrogen_outputs: torch.Tensor,
                           dt: float) -> Dict[str, torch.Tensor]:
        """
        Check mass balance for carbon and nitrogen.
        
        Parameters:
        -----------
        carbon_fluxes : Dict[str, torch.Tensor]
            Carbon flux changes
        nitrogen_fluxes : Dict[str, torch.Tensor] 
            Nitrogen flux changes
        dt : float
            Time step
        carbon_inputs : torch.Tensor
            External carbon inputs [g C m-2 s-1]
        nitrogen_inputs : torch.Tensor
            External nitrogen inputs [g N m-2 s-1]
        carbon_outputs : torch.Tensor
            External carbon outputs [g C m-2 s-1]
        nitrogen_outputs : torch.Tensor
            External nitrogen outputs [g N m-2 s-1]
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Mass balance errors
        """
        def _ensure_tensor(value, template):
            template_tensor = torch.as_tensor(template)
            if isinstance(value, torch.Tensor):
                return value.to(dtype=template_tensor.dtype, device=template_tensor.device)
            return torch.as_tensor(
                value,
                dtype=template_tensor.dtype,
                device=template_tensor.device,
            )

        # References: mass balance identity of the C and N pools (Manzoni and Porporato
        #   2009); RELIC budget rule (architecture_overview Section 20.3: a budget with a
        #   missing term is not evaluable).
        # Mathematical expression: Delta_X = sum_pools delta_X; e_X = |Delta_X - (I_X - O_X) dt|;
        #   rho_X = e_X / S_X with the gross budget activity
        #   S_X = |Delta_X| + |I_X dt| + |O_X dt| (the scale of ledger_acceptance,
        #   relic.validation.cross_domain_coupling_ops); total = rho_C + rho_N.
        # Parameter meanings: I_X and O_X the external inputs and outputs of element X,
        #   delta_X the corrected pool changes, dt the step.
        # Units: I_X, O_X [g X m-2 s-1]; Delta_X, e_X and S_X [g X m-2]; dt [s]; rho_X [1].
        # Domain/constraints: element X is evaluated only when both I_X and O_X are
        #   supplied; the total only when both elements are; an empty set of pool pairs
        #   has the exact change 0. e_X <= S_X (triangle inequality), so rho_X is in
        #   [0, 1]; S_X = 0 means every term is zero and then e_X = 0, reported as
        #   rho_X = 0 exactly (no additive epsilon).
        # Sign convention: inputs positive into the pools, outputs positive out.
        balance: Dict[str, torch.Tensor] = {}
        for element, pool_changes, inputs, outputs in (
            ('carbon', carbon_fluxes, carbon_inputs, carbon_outputs),
            ('nitrogen', nitrogen_fluxes, nitrogen_inputs, nitrogen_outputs),
        ):
            if inputs is None or outputs is None:
                continue
            inputs = torch.as_tensor(inputs)
            outputs = _ensure_tensor(outputs, inputs)
            total_change = sum(pool_changes.values()) if pool_changes else inputs.new_zeros(())
            total_change = _ensure_tensor(total_change, inputs)
            error = torch.abs(total_change - (inputs - outputs) * dt)
            scale = torch.abs(total_change) + torch.abs(inputs * dt) + torch.abs(outputs * dt)
            active = scale > 0
            # The unit denominator is only evaluated where S_X = 0, where the selected
            # value is the exact 0 (it keeps the discarded branch finite for autograd).
            balance[f'{element}_error'] = error
            balance[f'{element}_relative_error'] = torch.where(
                active,
                error / torch.where(active, scale, torch.ones_like(scale)),
                torch.zeros_like(error),
            )
        if 'carbon_relative_error' in balance and 'nitrogen_relative_error' in balance:
            balance['total_error'] = (
                balance['carbon_relative_error'] + balance['nitrogen_relative_error']
            )
        return balance
    
    def resolution_converter(self, data: torch.Tensor, source_res: Tuple[float, float], target_res: Tuple[float, float]) -> torch.Tensor:
        """
        Resolution converter for stoichiometry constraints.
        
        For stoichiometry, resolution conversion is typically identity because
        stoichiometric ratios are intensive properties independent of spatial resolution.
        
        Parameters:
        -----------
        data : torch.Tensor
            Data to convert
        source_res : Tuple[float, float]
            Source resolution (dx, dy) [m]
        target_res : Tuple[float, float]
            Target resolution (dx, dy) [m]
            
        Returns:
        --------
        torch.Tensor
            Converted data (identity for stoichiometry)
            
        Mathematical Expression:
        -----------------------
        C:N ratios are intensive properties:
        (C/N)_fine = (C/N)_coarse (conservation of stoichiometry)
        """
        # Stoichiometric ratios are intensive properties
        # They don't change with spatial resolution
        return data.clone()
    
    def get_stoichiometry_statistics(self) -> Dict[str, float]:
        """
        Get stoichiometry statistics.
        
        Returns:
        --------
        Dict[str, float]
            Statistics on constraint enforcement
        """
        return {
            'constraint_violations': self.constraint_violations,
            'execution_count': self._execution_count,
            'violation_frequency': self.constraint_violations / max(1, self._execution_count),
            'mean_mass_balance_error': sum(self.mass_balance_errors) / max(1, len(self.mass_balance_errors)),
            'max_mass_balance_error': max(self.mass_balance_errors) if self.mass_balance_errors else 0.0,
            **{f'target_cn_{pool}': ratio for pool, ratio in self.target_ratios.items()}
        }


__all__ = ['StoichiometryConstraint']
