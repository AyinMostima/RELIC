'\nRadiation Energy Conservation Validator for RELIC\n\nThis module provides comprehensive energy conservation validation for radiation\ntransfer processes, with integration to global feature flags and detailed\nreporting capabilities.\n\nScientific Basis:\n----------------\nEnergy conservation in radiation transfer requires:\n1. Total energy balance: E_in = E_absorbed + E_reflected + E_transmitted  \n2. Component-wise conservation: SW_in = SW_absorbed + SW_reflected\n3. Canopy layer conservation: Sigma_layers[APAR_i] <= PAR_total\n4. Longwave conservation: LW_net = LW_down - LW_up\n\nMathematical Expressions:\n-----------------------\n1. Global energy balance:\n   E_balance = |E_in - (E_absorbed + E_reflected + E_transmitted)|\n\n2. Canopy layer conservation:\n   Absorption excess = max(Sigma_layers[APAR_i] - PAR_total, 0); full closure requires transmitted PAR\n\n3. Albedo consistency:\n   alpha_effective = E_reflected / E_in (must be 0 <= alpha <= 1)\n\n4. Emissivity consistency:  \n   LW_up = epsilon * sigma * T^4 + (1 - epsilon) * LW_down (total emitted plus reflected)\n\nReferences:\n-----------\n- Campbell, G.S. & Norman, J.M. (1998). Environmental Biophysics\n- Monteith, J.L. & Unsworth, M.H. (2013). Principles of Environmental Physics\n- Bonan, G.B. (2019). Climate Change and Terrestrial Ecosystem Modeling\n- Energy conservation principles in atmospheric physics\n'
import torch
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from ...core.abstractions import StateDict
from ...core.feature_flags import feature_manager
from ...core.functor import PhysicalFieldFunctor
from ...core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from ...utils.physical_constants import get_physical_constant
logger = logging.getLogger(__name__)

class ConservationViolationType(Enum):
    """Types of energy conservation violations."""
    MINOR = 'minor'
    MODERATE = 'moderate'
    SEVERE = 'severe'

@dataclass
class EnergyConservationReport:
    """Detailed energy conservation analysis report."""
    timestamp: float
    total_energy_in: float
    total_energy_out: float
    energy_balance_error: float
    relative_error: float
    violation_type: ConservationViolationType
    shortwave_balance: Dict[str, float] = field(default_factory=dict)
    longwave_balance: Dict[str, float] = field(default_factory=dict)
    canopy_balance: Dict[str, float] = field(default_factory=dict)
    albedo_valid: bool = True
    emissivity_valid: bool = True
    temperature_valid: bool = True
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

class RadiationEnergyConservationValidator(PhysicalFieldFunctor):
    """
    Audit the declared atmospheric radiation energy balance.

    The Functor reads declared shortwave and longwave flux components and
    emits a diagnostic report. It does not advance prognostic state or prove
    whole-system surface energy closure.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    # Declared inputs read by forward from the state (the one input channel).
    REQUIRED_FLUX_INPUTS = ('shortwave_down', 'shortwave_net', 'longwave_down', 'longwave_up', 'longwave_net')
    OPTIONAL_FLUX_INPUTS = ('net_radiation', 'shortwave_reflected')
    OPTIONAL_PROPERTY_INPUTS = ('surface_albedo', 'surface_emissivity', 'surface_temperature', 'air_temperature')

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(self, name: str='radiation_conservation_audit', config: Optional[Dict]=None, parameter_manager: Optional[Any]=None):
        """
        Initialize energy conservation validator.
        
        Parameters:
        -----------
        config : Dict
            Configuration parameters for conservation checking
        """
        config = config or {}
        injected_manager = parameter_manager or (config.get('parameter_manager') if isinstance(config, dict) else None)
        parameter_overrides = self._normalize_override_block(config.get('parameters'))
        metadata = FunctorMetadata(name=name, version='1.0.0', description='Energy conservation audit for atmospheric radiation fluxes', category=ProcessCategory.PHYSICAL_FIELD, execution_phase=ExecutionPhase.VALIDATION, mathematical_expression='E_balance = |E_in - (SW_net + LW_net + SW_reflected + LW_up)| / max(E_in, eps)', references=['Campbell & Norman (1998) Environmental Biophysics', 'Monteith & Unsworth (2013) Principles of Environmental Physics', 'Bonan (2019) Climate Change and Terrestrial Ecosystem Modeling', 'Nicodemus (1965), doi:10.1364/AO.4.000767', 'Noah-MP Technical Note, Eqs. 3.6.330 and 3.6.416-417'], parameters={'validation.conservation.radiation.*': 'Radiation energy-budget tolerances, physical bounds and check switches [units per key]'}, input_variables=[*self.REQUIRED_FLUX_INPUTS, *self.OPTIONAL_FLUX_INPUTS, *self.OPTIONAL_PROPERTY_INPUTS], output_variables=['diagnostics.energy_conservation_report'], conservation_laws=['energy'], spatial_dimensions=0, temporal_scale=TemporalScale.SECOND, coupling_interfaces=['atmospheric_radiation'], spatial_scale=SpatialScale.POINT, variable_kind=VariableKind.DIAGNOSTIC)
        # Parameter overrides and the injected manager are consumed above; the
        # FunctorBase-owned dependency declaration is forwarded, never dropped.
        base_config = {key: config[key] for key in ('dependencies',) if key in config}
        super().__init__(name=name, config=base_config, metadata=metadata)
        self.optional_input_variables = [*self.OPTIONAL_FLUX_INPUTS, *self.OPTIONAL_PROPERTY_INPUTS]
        self._parameter_manager = injected_manager
        self._parameter_overrides = parameter_overrides
        self._parameters_loaded = False
        self.diagnostics: Dict[str, Any] = {}
        self.total_checks = 0
        self.violation_counts = {ConservationViolationType.MINOR: 0, ConservationViolationType.MODERATE: 0, ConservationViolationType.SEVERE: 0}
        self.conservation_history: List[EnergyConservationReport] = []
        self.last_report: Optional[EnergyConservationReport] = None

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Execute conservation audit within Functor lifecycle.

        Parameters
        ----------
        state : StateDict
            State dictionary containing radiation fluxes.
        time : float
            Current simulation time [s].
        dt : float
            Time step [s] (unused, kept for interface completeness).
        **kwargs : Any
            Not read: the state is the one input channel (declared inputs only). Direct
            audits with explicit fluxes, state properties or canopy outputs call
            validate_energy_conservation.

        Returns
        -------
        Dict[str, torch.Tensor]
            Diagnostic output containing ``diagnostics.energy_conservation_report``.
        """
        # References: Campbell and Norman (1998), surface radiation balance.
        # Mathematical expression: r = SW_down + LW_down - R_net - SW_reflected - LW_up.
        # Parameter meanings: R_net is the signed total net radiative flux.
        # Units: fluxes and r [W m-2]; the emitted relative residual [1].
        # Domain/constraints: finite scalar observations; incomplete ledgers retain an explicit scope.
        # Sign convention: downwelling and outgoing magnitudes are positive; R_net may be negative.
        # The state is the one input channel: only declared inputs are read, the
        # optional ones when present; no keyword channel supplies inputs.
        radiation_fluxes = {
            key: state[key]
            for key in (*self.REQUIRED_FLUX_INPUTS, *self.OPTIONAL_FLUX_INPUTS)
            if key in state
        }
        if not radiation_fluxes:
            raise ValueError('radiation_fluxes are required for conservation auditing.')
        radiation_state = {key: state[key] for key in self.OPTIONAL_PROPERTY_INPUTS if key in state}
        report = self.validate_energy_conservation(radiation_fluxes=radiation_fluxes, radiation_state=radiation_state)
        self.last_report = report
        self.diagnostics['energy_conservation'] = {'relative_error': report.relative_error, 'violation_type': report.violation_type.value}
        diagnostic_value = torch.as_tensor(
            float(report.relative_error),
            device=self.device,
            dtype=torch.float32,
        )
        return {"diagnostics.energy_conservation_report": diagnostic_value}

    def _apply_parameters(self, params: Dict[str, Any]) -> None:
        """Apply loaded parameter values to instance attributes."""
        self.minor_tolerance = params['minor_tolerance']
        self.moderate_tolerance = params['moderate_tolerance']
        self.severe_tolerance = params['severe_tolerance']
        self.relative_minor = params['relative_minor']
        self.relative_moderate = params['relative_moderate']
        self.relative_severe = params['relative_severe']
        self.min_albedo = params['min_albedo']
        self.max_albedo = params['max_albedo']
        self.min_emissivity = params['min_emissivity']
        self.max_emissivity = params['max_emissivity']
        self.min_temperature = params['min_temperature']
        self.max_temperature = params['max_temperature']
        self.canopy_conservation_tolerance = params['canopy_conservation_tolerance']
        self.enable_canopy_layer_check = params['enable_canopy_layer_check']

    def _ensure_parameters_loaded(self) -> None:
        """Lazy-load parameters to avoid hard dependency during initialization."""
        if getattr(self, '_parameters_loaded', False):
            return
        params = self._load_parameters(self._parameter_overrides)
        self._apply_parameters(params)
        self._parameters_loaded = True

    @staticmethod
    def _normalize_override_block(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize parameter override structures."""
        if not raw or not isinstance(raw, dict):
            return {}
        normalized: Dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, dict) and 'value' in value:
                normalized[key] = value['value']
            else:
                normalized[key] = value
        return normalized

    @staticmethod
    def _cast_parameter_value(value: Any, caster: Callable[[Any], Any]) -> Any:
        """Cast parameter values while handling tensors."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return caster(value.detach().reshape(-1)[0].tolist())
            raise ValueError('Multi-element tensor parameters are not supported for conservation validator.')
        return caster(value)

    def _fetch_parameter(self, key: str, units: Optional[str], caster: Callable[[Any], Any]) -> Any:
        """Fetch a parameter from the FunctorBase parameter access chain."""
        expected_type = bool if caster is bool else float
        value = self.get_parameter_value(
            key,
            units=units,
            expected_type=expected_type,
            validate=True,
            source_order=("manager",),
        )
        if value is None:
            raise ValueError(f"Required conservation parameter '{key}' not found in parameter database.")
        return self._cast_parameter_value(value, caster)

    def _load_parameters(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Load radiation conservation parameters from the parameter manager."""
        specs: List[Tuple[str, str, Optional[str], Callable[[Any], Any]]] = [('minor_tolerance', 'validation.conservation.radiation.minor_tolerance', 'W m-2', float), ('moderate_tolerance', 'validation.conservation.radiation.moderate_tolerance', 'W m-2', float), ('severe_tolerance', 'validation.conservation.radiation.severe_tolerance', 'W m-2', float), ('relative_minor', 'validation.conservation.radiation.relative_minor', 'dimensionless', float), ('relative_moderate', 'validation.conservation.radiation.relative_moderate', 'dimensionless', float), ('relative_severe', 'validation.conservation.radiation.relative_severe', 'dimensionless', float), ('min_albedo', 'validation.conservation.radiation.min_albedo', 'dimensionless', float), ('max_albedo', 'validation.conservation.radiation.max_albedo', 'dimensionless', float), ('min_emissivity', 'validation.conservation.radiation.min_emissivity', 'dimensionless', float), ('max_emissivity', 'validation.conservation.radiation.max_emissivity', 'dimensionless', float), ('min_temperature', 'validation.conservation.radiation.min_temperature', 'K', float), ('max_temperature', 'validation.conservation.radiation.max_temperature', 'K', float), ('canopy_conservation_tolerance', 'validation.conservation.radiation.canopy_conservation_tolerance', 'dimensionless', float), ('enable_canopy_layer_check', 'validation.conservation.radiation.enable_canopy_layer_check', 'dimensionless', bool)]
        params: Dict[str, Any] = {}
        for attr, key, units, caster in specs:
            if attr in overrides:
                params[attr] = self._cast_parameter_value(overrides[attr], caster)
            else:
                params[attr] = self._fetch_parameter(key, units, caster)
        return params

    def validate_energy_conservation(self, radiation_fluxes: Dict[str, torch.Tensor], radiation_state: Optional[Dict[str, torch.Tensor]]=None, canopy_outputs: Optional[Dict[str, torch.Tensor]]=None) -> EnergyConservationReport:
        """
        Perform comprehensive energy conservation validation.
        
        Parameters:
        -----------
        radiation_fluxes : Dict[str, torch.Tensor]
            All radiation flux components
        radiation_state : Optional[Dict[str, torch.Tensor]]
            Current radiation state variables
        canopy_outputs : Optional[Dict[str, torch.Tensor]]
            Canopy radiation outputs (if enabled)
            
        Returns:
        --------
        EnergyConservationReport
            Detailed conservation analysis
        """
        # References: Campbell and Norman (1998); Monteith and Unsworth (2013);
        #   Noah-MP Technical Note, Eqs. 3.6.416-417.
        # Mathematical expression: aliases denote the same observed flux, never additional terms.
        # Parameter meanings: albedo maps downwelling shortwave to its reflected component.
        # Units: fluxes [W m-2], albedo/emissivity [1], temperatures [K].
        # Domain/constraints: finite scalar observations and matching quantities;
        #   emitted-only surface_emission cannot denote total outgoing longwave.
        # Sign convention: alias resolution preserves the canonical component orientation.
        if 'surface_emission' in radiation_fluxes:
            raise ValueError(
                'surface_emission denotes emitted-only longwave; supply '
                'longwave_up as total emitted plus reflected outgoing longwave.'
            )
        fluxes = {}
        for key, value in radiation_fluxes.items():
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                raise ValueError(f'Radiation observation {key} must be a finite scalar.')
            scalar = float(value)
            if not math.isfinite(scalar):
                raise ValueError(f'Radiation observation {key} must be finite.')
            fluxes[key] = scalar
        aliases = {
            'incoming_shortwave': 'shortwave_down',
            'solar_radiation': 'shortwave_down',
        }
        for alias, canonical in aliases.items():
            if alias in fluxes:
                if canonical in fluxes and fluxes[canonical] != fluxes[alias]:
                    raise ValueError(f'Conflicting radiation observations for {canonical}: {alias}.')
                fluxes[canonical] = fluxes.pop(alias)
        for key in ('surface_albedo', 'surface_emissivity', 'surface_temperature', 'air_temperature'):
            if radiation_state is not None and key in radiation_state:
                value = radiation_state[key]
                if isinstance(value, torch.Tensor) and value.numel() != 1:
                    raise ValueError(f'Radiation state {key} must be a finite scalar.')
                scalar = float(value)
                if not math.isfinite(scalar):
                    raise ValueError(f'Radiation state {key} must be finite.')
                if key in fluxes and fluxes[key] != scalar:
                    raise ValueError(f'Conflicting radiation observations for {key}.')
                fluxes[key] = scalar
        reflected_source = 'not_supplied'
        if 'shortwave_reflected' in fluxes:
            reflected_source = 'shortwave_reflected'
        elif 'shortwave_down' in fluxes and 'surface_albedo' in fluxes:
            fluxes['shortwave_reflected'] = fluxes['shortwave_down'] * fluxes['surface_albedo']
            reflected_source = 'surface_albedo'
        self._ensure_parameters_loaded()
        self.total_checks += 1
        import time
        timestamp = time.time()
        report = EnergyConservationReport(timestamp=timestamp, total_energy_in=0.0, total_energy_out=0.0, energy_balance_error=0.0, relative_error=0.0, violation_type=ConservationViolationType.MINOR)
        report.diagnostics['reflected_flux_source'] = reflected_source
        self._check_global_energy_balance(fluxes, report)
        self._check_shortwave_conservation(fluxes, report)
        self._check_longwave_conservation(fluxes, report)
        if canopy_outputs and self.enable_canopy_layer_check:
            self._check_canopy_conservation(canopy_outputs, report)
        self._check_physical_constraints(fluxes, radiation_state, report)
        self._classify_violation_severity(report)
        self.violation_counts[report.violation_type] += 1
        self.conservation_history.append(report)
        self._log_conservation_results(report)
        return report

    def _check_global_energy_balance(self, radiation_fluxes: Dict[str, torch.Tensor], report: EnergyConservationReport) -> None:
        """Audit one radiative ledger, with explicit coverage for partial observations."""
        # References: Campbell and Norman (1998), surface radiation balance.
        # Mathematical expression: SW_down + LW_down = R_net + SW_reflected + LW_up;
        #   R_net = SW_net + LW_net is a redundant identity, not another outgoing term.
        # Parameter meanings: down/up are directional magnitudes; net is signed absorption.
        # Units: all ledger terms and absolute errors [W m-2]; relative errors [1].
        # Domain/constraints: only supplied components enter; a missing band is not certified.
        # Sign convention: positive net warms the surface; negative net radiative cooling is retained.
        input_keys = [key for key in ('shortwave_down', 'longwave_down') if key in radiation_fluxes]
        report.diagnostics['radiation_balance_scope'] = 'incomplete'
        report.diagnostics['global_balance_checked'] = False
        if not input_keys:
            report.warnings.append('No energy input fluxes found for global balance check')
            return
        output_keys = (
            ['net_radiation'] if 'net_radiation' in radiation_fluxes
            else [key for key in ('shortwave_net', 'longwave_net') if key in radiation_fluxes]
        )
        if not output_keys:
            report.warnings.append('No energy output fluxes found for global balance check')
            return
        output_keys += [key for key in ('shortwave_reflected', 'longwave_up') if key in radiation_fluxes]
        full_ledger = (
            len(input_keys) == 2
            and 'shortwave_reflected' in radiation_fluxes
            and 'longwave_up' in radiation_fluxes
            and ('net_radiation' in radiation_fluxes or
                 all(key in radiation_fluxes for key in ('shortwave_net', 'longwave_net')))
        )
        scope = 'full_radiative_flux' if full_ledger else 'partial_radiative_flux'
        if input_keys == ['shortwave_down'] and 'longwave_net' not in radiation_fluxes:
            scope = 'shortwave_supplied_components'
        report.diagnostics['radiation_balance_scope'] = scope
        report.diagnostics['global_balance_checked'] = True
        if not full_ledger:
            report.warnings.append(f'Radiation balance has limited {scope} coverage; missing components are not certified.')
        total_energy_in = sum(radiation_fluxes[key] for key in input_keys)
        total_energy_out = sum(radiation_fluxes[key] for key in output_keys)
        energy_balance_error = abs(total_energy_in - total_energy_out)
        relative_error = energy_balance_error / max(abs(total_energy_in), 1e-10)
        report.total_energy_in = total_energy_in
        report.total_energy_out = total_energy_out
        report.energy_balance_error = energy_balance_error
        report.relative_error = relative_error
        report.diagnostics['energy_inputs_count'] = len(input_keys)
        report.diagnostics['energy_outputs_count'] = len(output_keys)
        report.diagnostics['energy_balance_closure'] = total_energy_out / max(total_energy_in, 1e-10)
        if all(key in radiation_fluxes for key in ('net_radiation', 'shortwave_net', 'longwave_net')):
            residual = radiation_fluxes['net_radiation'] - radiation_fluxes['shortwave_net'] - radiation_fluxes['longwave_net']
            report.diagnostics['net_component_residual'] = residual
            relative = abs(residual) / max(abs(total_energy_in), 1e-10)
            if abs(residual) > self.moderate_tolerance or relative > self.relative_moderate:
                report.violations.append(f'Net radiation component inconsistency: {residual:.6g} W m-2')

    def _check_shortwave_conservation(self, radiation_fluxes: Dict[str, torch.Tensor], report: EnergyConservationReport) -> None:
        """Check shortwave radiation conservation: SW_in = SW_absorbed + SW_reflected."""
        # References: Campbell and Norman (1998), shortwave surface partition.
        # Mathematical expression: r_SW = SW_down - SW_net - SW_reflected.
        # Parameter meanings: reflected flux is observed or derived from supplied albedo.
        # Units: shortwave components and r_SW [W m-2]; albedo [1].
        # Domain/constraints: components must be supplied; missing terms produce a coverage warning.
        # Sign convention: incoming, absorbed, and reflected shortwave magnitudes are positive.
        sw_in = radiation_fluxes.get('shortwave_down', None)
        sw_net = radiation_fluxes.get('shortwave_net', None)
        sw_reflected = radiation_fluxes.get('shortwave_reflected', None)
        surface_albedo = radiation_fluxes.get('surface_albedo', None)
        if sw_in is None:
            report.warnings.append('Shortwave input not available for conservation check')
            return
        sw_in_val = float(sw_in)
        if sw_net is not None and sw_reflected is not None:
            sw_absorbed = float(sw_net)
            sw_refl_val = float(sw_reflected)
        elif sw_net is not None and surface_albedo is not None:
            sw_absorbed = float(sw_net)
            sw_refl_val = float(surface_albedo) * sw_in_val
        elif surface_albedo is not None:
            albedo_val = float(surface_albedo)
            sw_absorbed = (1.0 - albedo_val) * sw_in_val
            sw_refl_val = albedo_val * sw_in_val
        else:
            report.warnings.append('Insufficient shortwave components for conservation check')
            return
        sw_out_total = sw_absorbed + sw_refl_val
        sw_error = abs(sw_in_val - sw_out_total)
        sw_relative_error = sw_error / max(abs(sw_in_val), 1e-10)
        report.shortwave_balance = {'sw_in': sw_in_val, 'sw_absorbed': sw_absorbed, 'sw_reflected': sw_refl_val, 'sw_error': sw_error, 'sw_relative_error': sw_relative_error}
        if sw_error > self.moderate_tolerance:
            report.violations.append(f'Shortwave conservation violation: {sw_error:.2f} W/m^2')
        elif sw_relative_error > self.relative_moderate:
            report.violations.append(f'Shortwave relative error: {sw_relative_error:.1%}')

    def _check_longwave_conservation(self, radiation_fluxes: Dict[str, torch.Tensor], report: EnergyConservationReport) -> None:
        """Check longwave radiation conservation."""
        # References:
        #   Monteith and Unsworth (2013), Stefan-Boltzmann radiation;
        #   Nicodemus (1965), doi:10.1364/AO.4.000767;
        #   Noah-MP Technical Note, Eqs. 3.6.330 and 3.6.416-417.
        # Mathematical expression:
        #   r_LW=LW_net-(LW_down-LW_up); when total LW_up is not observed,
        #   LW_up=eps*sigma*T^4+(1-eps)*LW_down and
        #   LW_net_expected=eps*LW_down-eps*sigma*T^4.
        # Parameter meanings:
        #   eps is supplied LW surface emissivity/absorptivity; T is supplied
        #   surface temperature; sigma is the physical SSOT constant;
        #   LW_down is incident and LW_up total emitted plus reflected LW.
        # Units:
        #   LW components and r_LW [W m-2]; T [K]; eps [1]; sigma [W m-2 K-4].
        # Domain/constraints:
        #   finite scalar observations; explicit total LW_up takes precedence;
        #   property inference assumes an opaque reciprocal grey surface and
        #   supplied eps/T, retains partial coverage and existing screening bounds.
        # Sign convention:
        #   incident and signed net LW are inward-positive; total outgoing LW
        #   is outward-positive; negative net LW is retained radiative cooling.
        lw_down = radiation_fluxes.get('longwave_down', None)
        lw_up = radiation_fluxes.get('longwave_up', None)
        lw_net = radiation_fluxes.get('longwave_net', None)
        surface_emissivity = radiation_fluxes.get('surface_emissivity', None)
        surface_temperature = radiation_fluxes.get('surface_temperature', None)
        if lw_down is None:
            report.warnings.append('Longwave input not available for conservation check')
            return
        lw_down_val = float(lw_down)
        if lw_net is not None and lw_up is not None:
            lw_net_val = float(lw_net)
            lw_up_val = float(lw_up)
            expected_net = lw_down_val - lw_up_val
            lw_error = abs(lw_net_val - expected_net)
        elif lw_net is not None and surface_emissivity is not None and (surface_temperature is not None):
            lw_net_val = float(lw_net)
            emissivity = float(surface_emissivity)
            temperature = float(surface_temperature)
            stefan_boltzmann = get_physical_constant('stefan_boltzmann_constant')
            lw_up_val = (
                emissivity * stefan_boltzmann * temperature ** 4
                + (1.0 - emissivity) * lw_down_val
            )
            expected_net = lw_down_val - lw_up_val
            lw_error = abs(lw_net_val - expected_net)
        else:
            report.warnings.append('Insufficient longwave components for conservation check')
            return
        lw_relative_error = lw_error / max(abs(lw_down_val), 1e-10)
        report.longwave_balance = {'lw_down': lw_down_val, 'lw_up': lw_up_val, 'lw_net': lw_net_val, 'lw_error': lw_error, 'lw_relative_error': lw_relative_error}
        if lw_error > self.moderate_tolerance:
            report.violations.append(f'Longwave conservation violation: {lw_error:.2f} W/m^2')

    def _check_canopy_conservation(self, canopy_outputs: Dict[str, torch.Tensor], report: EnergyConservationReport) -> None:
        """Check absorbed PAR and certify full closure only with a transmitted port."""
        # References: Campbell and Norman (1998), Beer-Lambert canopy attenuation.
        # Mathematical expression: A=sum_i APAR_i <= P; A=fPAR*P;
        #   A+T=P only when the transmitted flux T is supplied.
        # Parameter meanings: P is incident PAR, A layer absorption, and T transmitted PAR.
        # Units: P, A, T and absolute residuals [W m-2]; fPAR and relative residuals [1].
        # Domain/constraints: finite nonnegative incident and layer fluxes; no inferred missing port.
        # Sign convention: absorption and transmission are positive partitions of incident energy.
        # Incident PAR energy [W m-2] is par_total; 'par' is a photon flux
        # density [umol photons m-2 s-1] and is not read as energy.
        par_total = canopy_outputs.get('par_total', None)
        apar_layer = canopy_outputs.get('apar_layer', None)
        fpar = canopy_outputs.get('fpar', None)
        if par_total is None:
            report.warnings.append('PAR total not available for canopy conservation check')
            return
        par_total_val = float(par_total)
        if not math.isfinite(par_total_val) or par_total_val < 0.0:
            raise ValueError('Incident canopy PAR must be finite and non-negative.')
        if apar_layer is not None:
            if isinstance(apar_layer, torch.Tensor):
                if not bool(torch.isfinite(apar_layer).all()) or bool((apar_layer < 0.0).any()):
                    raise ValueError('Canopy layer absorption must be finite and non-negative.')
                apar_sum = float(torch.sum(apar_layer))
            else:
                apar_sum = float(apar_layer)
                if not math.isfinite(apar_sum) or apar_sum < 0.0:
                    raise ValueError('Canopy layer absorption must be finite and non-negative.')
            canopy_error = max(apar_sum - par_total_val, 0.0)
            scope = 'absorption_only'
            transmitted = canopy_outputs.get('par_transmitted')
            if transmitted is not None:
                transmitted = float(transmitted)
                if not math.isfinite(transmitted) or transmitted < 0.0:
                    raise ValueError('Transmitted canopy PAR must be finite and non-negative.')
                canopy_error = abs(apar_sum + transmitted - par_total_val)
                scope = 'absorbed_and_transmitted'
            else:
                report.warnings.append('Canopy audit has limited absorption-only coverage; transmitted PAR is not supplied.')
            canopy_relative_error = canopy_error / max(par_total_val, 1e-10)
            report.canopy_balance = {'par_total': par_total_val, 'apar_sum': apar_sum, 'canopy_error': canopy_error, 'canopy_relative_error': canopy_relative_error, 'n_layers': len(apar_layer) if hasattr(apar_layer, '__len__') else 1}
            report.diagnostics['canopy_balance_scope'] = scope
            if transmitted is not None:
                report.canopy_balance['par_transmitted'] = transmitted
            if canopy_relative_error > self.canopy_conservation_tolerance:
                report.violations.append(f'Canopy conservation violation: {canopy_relative_error:.1%} relative error')
        if fpar is not None:
            fpar_val = float(fpar)
            if not math.isfinite(fpar_val):
                raise ValueError('Canopy fPAR must be finite.')
            if fpar_val < 0.0 or fpar_val > 1.0:
                report.violations.append(f'fPAR out of physical bounds: {fpar_val:.3f}')
            report.canopy_balance['fpar'] = fpar_val
            if apar_layer is not None:
                fpar_error = abs(apar_sum - fpar_val * par_total_val)
                report.canopy_balance['fpar_error'] = fpar_error
                if fpar_error / max(par_total_val, 1e-10) > self.canopy_conservation_tolerance:
                    report.violations.append(f'Canopy fPAR does not match layer absorption: {fpar_error:.6g} W m-2')

    def _check_physical_constraints(self, radiation_fluxes: Dict[str, torch.Tensor], radiation_state: Optional[Dict[str, torch.Tensor]], report: EnergyConservationReport) -> None:
        """Check physical constraints on radiation variables."""
        if 'surface_albedo' in radiation_fluxes:
            albedo = float(radiation_fluxes['surface_albedo'])
            if not self.min_albedo <= albedo <= self.max_albedo:
                report.albedo_valid = False
                report.violations.append(f'Albedo out of bounds: {albedo:.3f}')
        if 'surface_emissivity' in radiation_fluxes:
            emissivity = float(radiation_fluxes['surface_emissivity'])
            if not self.min_emissivity <= emissivity <= self.max_emissivity:
                report.emissivity_valid = False
                report.violations.append(f'Emissivity out of bounds: {emissivity:.3f}')
        temp_keys = ['surface_temperature', 'skin_temperature', 'air_temperature']
        for temp_key in temp_keys:
            if temp_key in radiation_fluxes:
                temperature = float(radiation_fluxes[temp_key])
                if not self.min_temperature <= temperature <= self.max_temperature:
                    report.temperature_valid = False
                    report.violations.append(f'Temperature out of bounds: {temperature:.1f} K')
                break
        incoming_keys = ['shortwave_down', 'longwave_down', 'solar_radiation']
        for key in incoming_keys:
            if key in radiation_fluxes:
                flux_val = float(radiation_fluxes[key])
                if flux_val < -1.0:
                    report.violations.append(f'Negative incoming radiation: {key} = {flux_val:.2f}')

    def _classify_violation_severity(self, report: EnergyConservationReport) -> None:
        """Classify violation severity based on errors and constraint violations."""
        if report.energy_balance_error > self.severe_tolerance:
            report.violation_type = ConservationViolationType.SEVERE
        elif report.energy_balance_error > self.moderate_tolerance:
            report.violation_type = ConservationViolationType.MODERATE
        elif report.energy_balance_error > self.minor_tolerance:
            report.violation_type = ConservationViolationType.MINOR
        if report.relative_error > self.relative_severe:
            report.violation_type = ConservationViolationType.SEVERE
        elif report.relative_error > self.relative_moderate and report.violation_type == ConservationViolationType.MINOR:
            report.violation_type = ConservationViolationType.MODERATE
        if not (report.albedo_valid and report.emissivity_valid and report.temperature_valid):
            if report.violation_type == ConservationViolationType.MINOR:
                report.violation_type = ConservationViolationType.MODERATE

    def _log_conservation_results(self, report: EnergyConservationReport) -> None:
        """Log conservation results based on violation severity."""
        if not feature_manager.is_enabled('canopy_layers_enabled'):
            return
        if report.violation_type == ConservationViolationType.SEVERE:
            logger.error(f'SEVERE energy conservation violation: {report.energy_balance_error:.2f} W/m^2 ({report.relative_error:.1%} relative)')
            for violation in report.violations:
                logger.error(f'  - {violation}')
        elif report.violation_type == ConservationViolationType.MODERATE:
            logger.warning(f'MODERATE energy conservation violation: {report.energy_balance_error:.2f} W/m^2 ({report.relative_error:.1%} relative)')
            for violation in report.violations:
                logger.warning(f'  - {violation}')
        elif report.violations:
            logger.debug(f'Minor energy conservation issues: {len(report.violations)} items')
            for violation in report.violations:
                logger.debug(f'  - {violation}')
        for warning in report.warnings:
            logger.warning(f'Energy conservation warning: {warning}')

    def get_conservation_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """Get energy conservation validation statistics."""
        if self.total_checks == 0:
            return {'total_checks': 0, 'message': 'No conservation checks performed'}
        violation_rates = {}
        for violation_type, count in self.violation_counts.items():
            violation_rates[f'{violation_type.value}_rate'] = count / self.total_checks
        recent_reports = self.conservation_history[-10:]
        if recent_reports:
            avg_balance_error = sum((r.energy_balance_error for r in recent_reports)) / len(recent_reports)
            avg_relative_error = sum((r.relative_error for r in recent_reports)) / len(recent_reports)
        else:
            avg_balance_error = 0.0
            avg_relative_error = 0.0
        return {'total_checks': self.total_checks, 'violation_counts': dict(self.violation_counts), 'violation_rates': violation_rates, 'average_balance_error': avg_balance_error, 'average_relative_error': avg_relative_error, 'recent_reports_count': len(recent_reports), 'conservation_history_length': len(self.conservation_history)}

    def generate_conservation_summary(self) -> str:
        """Generate a human-readable conservation summary."""
        stats = self.get_conservation_statistics()
        if stats['total_checks'] == 0:
            return 'No energy conservation checks have been performed.'
        summary = 'Energy Conservation Summary:\n'
        summary += f"  Total Checks: {stats['total_checks']}\n"
        summary += '  Violation Rates:\n'
        for violation_type in ConservationViolationType:
            rate = stats['violation_rates'].get(f'{violation_type.value}_rate', 0.0)
            summary += f'    - {violation_type.value.title()}: {rate:.1%}\n'
        summary += f"  Average Balance Error: {stats['average_balance_error']:.3f} W/m^2\n"
        summary += f"  Average Relative Error: {stats['average_relative_error']:.1%}\n"
        return summary

def create_energy_conservation_validator(config: Optional[Dict]=None) -> RadiationEnergyConservationValidator:
    """
    Factory function to create radiation energy conservation validator.
    
    Parameters:
    -----------
    config : Optional[Dict]
        Configuration for conservation tolerances and settings
        
    Returns:
    --------
    RadiationEnergyConservationValidator
        Configured validator instance
    """
    return RadiationEnergyConservationValidator(name='radiation_conservation_audit', config=config)
__all__ = ['RadiationEnergyConservationValidator', 'EnergyConservationReport', 'ConservationViolationType', 'create_energy_conservation_validator']
