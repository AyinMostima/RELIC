"""Phenomenological stored-pool monoterpene closure for RELIC.

The literature supports separating synthesis from release when vegetation stores
monoterpenes and supports a strong temperature response of stored-pool release.
The exact GPP saturation, Q10 synthesis response, finite-capacity limiter, and
first-order degradation used here are ParameterManager-backed calibratable
closures; they are not claimed as universal equations from one source.

References
----------
- Tingey, D. T., et al. (1980). Influence of light and temperature on
  monoterpene emission rates from slash pine. Plant Physiology 65, 797-801.
- Keeling, C. I. and Bohlmann, J. (2006). Genes, enzymes and chemicals of
  terpenoid diversity in the constitutive and induced defence of conifers
  against insects and pathogens. New Phytologist 170, 657-675.
- Niinemets, U. and Reichstein, M. (2002). A model analysis of the effects of
  nonspecific monoterpenoid storage in leaf tissues on emission kinetics.
  Global Biogeochemical Cycles 16, 1110.
- Niinemets, U., et al. (2014). Bidirectional exchange of biogenic volatiles
  with vegetation: emission sources, reactions, breakdown and deposition.
  Plant, Cell & Environment 37, 1790-1809.

Mathematical expressions
------------------------
- F_alloc = f_C f_seasonal GPP
- J_sat = V_max(T) F_alloc / (K_GPP + F_alloc)
- J_syn = J_sat (1 - P / P_max)
- J_emit = k_25 exp(beta (T - T_25)) P
- dP/dt = J_syn - J_emit - J_deg + J_boundary
- P_next = clip(P + dt (J_syn - J_emit - J_deg), 0, P_max)
- F_C,alloc = convert(J_syn 10^-6 MW_mono f_C, g C m-2 s-1,
  kg C m-2 s-1)

Parameter meanings and units
----------------------------
- GPP is gross photosynthesis and F_alloc is its allocated driver
  [umol CO2 m-2 s-1].
- P and P_max are the pool and capacity [umol m-2].
- V_max and K_GPP are phenomenological synthesis coefficients
  [umol m-2 s-1].
- k_25 and k_deg are first-order rate coefficients [s-1].
- beta is the release-temperature slope [K-1].
- T_25 is the physical-constant SSOT reference temperature [K].

The process is open. ``monoterpenes_emission`` is its atmosphere-facing molar
port. ``bvoc_carbon_allocation_flux`` is the positive plant-carbon allocation
into newly synthesized monoterpene and is therefore based on J_syn rather than
release from stored material. Pool clipping and degradation are exposed
explicitly, while their end-to-end plant-carbon routing remains a central
cross-module integration obligation.
"""
from typing import Any, Dict, Optional, Union
import torch
from torch import Tensor
import torch.nn as nn
from relic.core.functor import BiologicalProcessFunctor, TemporalScale
from relic.core.state import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter
from relic.utils.unified_response_functions import q10_temperature_response, michaelis_menten_response
from relic.computation.operators.iphs import advance_reaction_network
from relic.utils.physical_constants import get_physical_constant

class MonoterpenePoolModel(nn.Module):
    """
    Monoterpene Storage Pool Dynamics Model.
    
    Models the synthesis, storage, and emission of monoterpenes from specialized
    storage structures (resin ducts, oil glands, etc.) in plants.
    
    Pool Dynamics:
    - Synthesis from photosynthetic precursors (GPP, IPP)
    - Storage in specialized structures with finite capacity
    - Temperature-dependent emission to atmosphere
    - Degradation and metabolic turnover
    
    References
    ----------
    - Tingey et al. (1980): Temperature dependence of monoterpene emissions
    - Keeling & Bohlmann (2006): Terpene synthase mechanisms
    """

    def __init__(
        self,
        *,
        v_max_mts: Union[Tensor, float],
        k_gpp: Union[Tensor, float],
        pool_capacity: Union[Tensor, float],
        pool_initial: Union[Tensor, float],
        k_emit_25: Union[Tensor, float],
        beta_emission: Union[Tensor, float],
        k_degradation: Union[Tensor, float],
        q10_synthase: Union[Tensor, float],
        device: Optional[torch.device] = None,
    ):
        """
        Initialize MonoterpenePoolModel with validated parameter access.
        
        Parameters
        ----------
        v_max_mts, k_gpp, pool_capacity, pool_initial : Tensor | float
            Parent-resolved synthesis and storage parameters.
        k_emit_25, beta_emission, k_degradation, q10_synthase : Tensor | float
            Parent-resolved release, degradation, and temperature parameters.
        device : torch.device, optional
            Target device for tensors
        """
        super().__init__()
        self.device = device or (
            torch.device("cpu")
        )
        # References: Tingey et al. (1980); Niinemets and Reichstein (2002);
        #   ParameterManager binding ownership,
        #   docs/source/architecture_overview.md Sections 4.2 and 4.4.2.
        # Mathematical expression: J_syn = V_max q^((T - T_25)/10) F / (K + F)
        #   (1 - P / P_max); J_emit = k_25 exp(beta (T - T_25)) P;
        #   J_deg = k_deg P.
        # Parameter meanings: V_max (v_max_mts), q (q10_synthase), k_25
        #   (k_emit_25), beta (beta_emission) and k_deg (k_degradation) are the
        #   kinetic coefficients a parent Functor may hand over as learnable
        #   ParameterManager bindings; F is the allocated carbon flux, K its
        #   half-saturation, P and P_max the pool and its capacity.
        # Units: V_max, F, K, J_syn, J_emit, J_deg [umol m-2 s-1]; q [1];
        #   k_25, k_deg [s-1]; beta [K-1]; T, T_25 [K]; P, P_max [umol m-2].
        # Domain/constraints: a coefficient handed over as an nn.Parameter is the
        #   parent's binding. It is kept by reference in _held_coefficients and
        #   never registered here, so it is no state of this helper and module
        #   conversions of the helper cannot convert it; it is read at every use
        #   on the device of this helper, so the law follows a changed binding
        #   and the gradient of a call reaches it. Any other value is a float32
        #   buffer of this helper.
        # Sign convention: none (kinetic coefficients; forward sets the signs of
        #   the rates).
        self._held_coefficients: Dict[str, nn.Parameter] = {}
        for coefficient_name, coefficient_value in (
            ("v_max_mts", v_max_mts),
            ("k_emit_25", k_emit_25),
            ("beta_emission", beta_emission),
            ("k_degradation", k_degradation),
            ("q10_synthase", q10_synthase),
        ):
            if isinstance(coefficient_value, nn.Parameter):
                self._held_coefficients[coefficient_name] = coefficient_value
            else:
                self.register_buffer(
                    coefficient_name,
                    self._ensure_tensor(coefficient_value).detach().clone(),
                )
        self.register_buffer(
            "k_gpp",
            self._ensure_tensor(k_gpp).detach().clone(),
        )
        self.register_buffer(
            "pool_capacity",
            self._ensure_tensor(pool_capacity).detach().clone(),
        )
        self.register_buffer(
            "pool_initial",
            self._ensure_tensor(pool_initial).detach().clone(),
        )
        self.register_buffer(
            'temp_ref',
            torch.tensor(
                float(get_physical_constant('reference_temperature')),
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def __getattr__(self, name: str) -> Any:
        held = self.__dict__.get("_held_coefficients")
        if held is not None and name in held:
            # Tensor.to returns the binding itself when its device and dtype
            # match, otherwise a differentiable converted copy of it.
            return self._ensure_tensor(held[name])
        return super().__getattr__(name)

    def _ensure_tensor(self, value: Union[Tensor, float]) -> Tensor:
        """Cast helper ensuring tensors live on the configured device."""
        if isinstance(value, Tensor):
            tensor = value.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.tensor(float(value), dtype=torch.float32, device=self.device)
        return tensor

    @staticmethod
    def _scalar_from_value(value: Union[Tensor, float]) -> float:
        """Convert numeric values to Python scalar without direct extraction APIs."""
        if isinstance(value, Tensor):
            detached = value.detach().reshape(-1)
            if detached.numel() == 0:
                raise ValueError("Cannot convert empty tensor to scalar.")
            return detached[:1].to(dtype=torch.float64).tolist()[0]
        return float(value)

    def forward(self, gpp_availability: Union[Tensor, float], temperature: Union[Tensor, float], dt: float=1.0, *, pool_state: Union[Tensor, float]) -> Dict[str, Tensor]:
        """
        Update monoterpene pool dynamics using caller-managed pool state.
        
        Parameters
        ----------
        gpp_availability : Tensor | float
            Available photosynthetic carbon flux [umol m-2 s-1]
        temperature : Tensor | float
            Temperature [K]
        dt : float, optional
            Time step [s], default 1.0
        pool_state : Tensor | float
            Current StateManager-owned pool concentration [umol m-2].
            
        Returns
        -------
        Dict[str, Tensor]
            Pool dynamics diagnostics and updated pool concentration.
        """
        # References: Tingey et al. (1980); Niinemets and Reichstein (2002).
        # Mathematical expression: J_syn = V_max(T) F_alloc
        # / (K_GPP + F_alloc) * (1 - P/P_max);
        # J_emit = k_25 exp(beta (T - T_25)) P;
        # P_next = clip(P + dt * (J_syn - J_emit - J_deg), 0, P_max).
        # Parameter meanings: GPP is the photosynthesis driver, P is stored
        # monoterpene, P_max is capacity, and all response coefficients are
        # ParameterManager values. T_25 comes from the physical-constant SSOT.
        gpp_tensor = self._ensure_tensor(gpp_availability)
        temp_tensor = self._ensure_tensor(temperature)
        current_pool = self._ensure_tensor(pool_state)
        dt_tensor = torch.as_tensor(dt, dtype=torch.float32, device=self.device)
        for name, value in (
            ("gpp_availability", gpp_tensor),
            ("temperature", temp_tensor),
            ("pool_state", current_pool),
            ("pool_capacity", self.pool_capacity),
        ):
            if value.numel() == 0 or not torch.all(torch.isfinite(value)):
                raise ValueError(f"{name} must be non-empty and finite")
        if torch.any(gpp_tensor < 0):
            raise ValueError("gpp_availability must be non-negative")
        if torch.any(temp_tensor <= 0):
            raise ValueError("temperature must be positive Kelvin")
        if torch.any(self.pool_capacity <= 0):
            raise ValueError("pool_capacity must be positive")
        if torch.any((current_pool < 0) | (current_pool > self.pool_capacity)):
            raise ValueError("pool_state must lie within [0, pool_capacity]")
        if dt_tensor.numel() != 1 or not torch.all(torch.isfinite(dt_tensor)):
            raise ValueError("dt must be one finite scalar")
        if torch.any(dt_tensor <= 0):
            raise ValueError("dt must be positive")

        v_max_mts_temp = q10_temperature_response(self.v_max_mts, temp_tensor, self.q10_synthase, self.temp_ref)
        synthesis_rate = michaelis_menten_response(gpp_tensor, v_max_mts_temp, self.k_gpp)
        capacity_factor = 1.0 - current_pool / self.pool_capacity
        synthesis_rate = synthesis_rate * capacity_factor
        k_emit_temp = self.k_emit_25 * torch.exp(self.beta_emission * (temp_tensor - self.temp_ref))
        emission_rate = k_emit_temp * current_pool
        degradation_rate = self.k_degradation * current_pool
        raw_tendency = synthesis_rate - emission_rate - degradation_rate
        raw_next_pool = current_pool + raw_tendency * dt_tensor
        bounded_pool = torch.minimum(
            torch.maximum(raw_next_pool, torch.zeros_like(raw_next_pool)),
            self.pool_capacity,
        )
        boundary_residual_rate = (bounded_pool - raw_next_pool) / dt_tensor
        actual_tendency = raw_tendency + boundary_residual_rate
        pool_delta = actual_tendency * dt_tensor
        return {
            'synthesis_rate': synthesis_rate,
            'emission_rate': emission_rate,
            'degradation_rate': degradation_rate,
            'pool_concentration': bounded_pool,
            'pool_delta': pool_delta,
            'pool_tendency': actual_tendency,
            'pool_boundary_residual_rate': boundary_residual_rate,
            'pool_capacity_utilization': bounded_pool / self.pool_capacity,
            'temperature_emission_factor': k_emit_temp / self.k_emit_25,
        }

    def reset_pool(self, initial_concentration: Optional[Tensor]=None):
        """Return a validated initial tensor for external StateManager setup."""
        value = self.pool_initial if initial_concentration is None else self._ensure_tensor(initial_concentration)
        if value.numel() == 0 or not torch.all(torch.isfinite(value)):
            raise ValueError("Initial monoterpene pool must be non-empty and finite")
        if torch.any((value < 0) | (value > self.pool_capacity)):
            raise ValueError("Initial monoterpene pool must lie within [0, pool_capacity]")
        return value.clone()

class MonoterpeneSynthesis(BiologicalProcessFunctor):
    """
    Phenomenological Stored-Pool Monoterpene Synthesis and Emission Model.
    
    Integrates monoterpene biosynthesis with storage pool dynamics to provide
    a calibratable monoterpene emission closure. It models the decoupling
    between synthesis and emission that characterizes stored-pool emissions.
    
    Features
    --------
    - Phenomenological synthesis response to photosynthetic carbon availability
    - Dynamic storage pool with finite capacity
    - Temperature-dependent emission from stored pools
    - Degradation and metabolic turnover
    - Integration with photosynthetic carbon availability
    
    Architecture
    ------------
    - Uses MonoterpenePoolModel for storage dynamics
    - Couples synthesis to photosynthetic carbon flux
    - Temperature-dependent emission kinetics
    - Pool capacity and saturation effects
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    POOL_STATE_KEY = 'monoterpene_pool_concentration'

    def __init__(self, name: str='mechanistic_monoterpene_synthesis', config: Optional[Dict[str, Any]]=None, device: Optional[torch.device]=None):
        """
        Initialize mechanistic monoterpene synthesis model.
        
        Parameters
        ----------
        name : str, optional
            Model name
        config : Dict[str, Any], optional
            Configuration parameters
        device : torch.device, optional
            Computation device
        """
        canonical_name = 'mechanistic_monoterpene_synthesis'
        if name != canonical_name:
            raise ValueError(
                f"MonoterpeneSynthesis runtime name must be '{canonical_name}'."
            )
        from relic.core.functor import FunctorMetadata, ProcessCategory, ExecutionPhase, SpatialScale, VariableKind

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Phenomenological monoterpene synthesis and emission closure with storage pool dynamics",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            parameters={
                'biosphere.bvoc.monoterpene.v_max_mts': {
                    'description': (
                        'Maximum rate in the calibratable monoterpene synthesis response'
                    ),
                    'units': 'umol m-2 s-1',
                },
                'biosphere.bvoc.monoterpene.k_gpp': {
                    'description': (
                        'Allocated photosynthetic-carbon half-saturation scale in the '
                        'calibratable synthesis response'
                    ),
                    'units': 'umol m-2 s-1',
                },
                'biosphere.bvoc.monoterpene.q10_synthase': {
                    'description': 'Q10 coefficient for the calibratable synthesis response',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.monoterpene.pool.capacity': {
                    'description': 'Maximum monoterpene storage pool capacity',
                    'units': 'umol m-2',
                },
                'biosphere.bvoc.monoterpene.pool_initial': {
                    'description': 'Initial monoterpene pool concentration',
                    'units': 'umol m-2',
                },
                'biosphere.bvoc.monoterpene.pool.k_degradation': {
                    'description': 'First-order degradation rate constant',
                    'units': 's-1',
                },
                'biosphere.bvoc.monoterpene.pool.k_emit_25': {
                    'description': 'Emission rate constant at 25 degC (298.15 K) reference',
                    'units': 's-1',
                },
                'biosphere.bvoc.monoterpene.pool.beta_emission': {
                    'description': 'Temperature sensitivity coefficient for emission',
                    'units': 'K-1',
                },
                'biosphere.bvoc.photosynthetic_coupling.f_carbon_monoterpenes': {
                    'description': 'Fraction of GPP allocated to monoterpene synthesis',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.molecular_weights.monoterpenes': {
                    'description': 'Average molecular weight of monoterpenes (C10H16)',
                    'units': 'g mol-1',
                },
                'biosphere.bvoc.carbon_fraction.monoterpenes': {
                    'description': (
                        'Carbon mass fraction of monoterpenes (stoichiometric fraction of '
                        'compound mass that is carbon)'
                    ),
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.monoterpene.enable_seasonal_variation': {
                    'description': 'Enable seasonal variation in synthesis rates',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.monoterpene.seasonal.amplitude': {
                    'description': 'Amplitude of seasonal modulation (peak deviation from unity)',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.monoterpene.seasonal.minimum_factor': {
                    'description': 'Minimum clamp applied to seasonal modulation factor',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.monoterpene.seasonal.peak_day': {
                    'description': (
                        'Day-of-year when seasonal peak occurs (Northern Hemisphere default)'
                    ),
                    'units': 'day',
                },
            },
            input_variables=[
                "air_temperature",
                "gross_photosynthesis",
                "monoterpene_pool_concentration",
            ],
            output_variables=[
                "monoterpenes_emission",
                "monoterpene_pool_concentration",
                "diagnostics.bvoc.monoterpene.emission_rate_molar",
                "diagnostics.bvoc.monoterpene.emission_rate_mass",
                "diagnostics.bvoc.monoterpene.pool_tendency",
                "diagnostics.bvoc.monoterpene.pool_delta",
                "diagnostics.bvoc.monoterpene.pool_boundary_residual_rate",
                "diagnostics.bvoc.monoterpene.storage_change",
                "diagnostics.bvoc.monoterpene.synthesis_rate",
                "diagnostics.bvoc.monoterpene.pool_capacity_utilization",
                "diagnostics.bvoc.monoterpene.temperature_emission_factor",
                "diagnostics.bvoc.monoterpene.carbon_flux_allocated",
                "diagnostics.bvoc.monoterpene.degradation_rate",
                "diagnostics.bvoc.monoterpene.seasonal_factor",
                "bvoc_carbon_allocation_flux",
            ],
            conservation_laws=["carbon_boundary"],
            coupling_interfaces=["photosynthesis_carbon_allocation", "bvoc_emission_flux"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            mathematical_expression="Storage Pool Dynamics:\n            d[Pool]/dt = J_synthesis - J_emission - J_degradation + J_boundary\n            \n            Phenomenological Monoterpene Synthesis Response:\n            F_alloc = f_carbon * f_seasonal * GPP\n            J_sat = V_max_MTS(T) * F_alloc / (K_GPP + F_alloc)\n            V_max_MTS(T) = V_max_MTS_25 * Q10^((T - T_25)/10)\n            \n            Capacity Limitation:\n            J_synthesis = J_sat * (1 - [Pool]/[Pool_max])\n            \n            Temperature-Dependent Emission:\n            J_emission = k_emit(T) * [Pool]\n            k_emit(T) = k_emit_25 * exp(beta * (T - T_25))\n            T_25 = 298.15 K from the physical-constant SSOT\n            \n            Degradation:\n            J_degradation = k_degradation * [Pool]\n            \n            Seasonal Factor (if enabled):\n            f_seasonal = 1 + A * sin(2*pi*t/year - phase_offset)",
            references=[
                "Tingey, D.T., et al. (1980). Influence of light and temperature on monoterpene emission rates from slash pine. Plant Physiology 65, 797-801.",
                "Keeling, C.I. & Bohlmann, J. (2006). Genes, enzymes and chemicals of terpenoid diversity in the constitutive and induced defence of conifers against insects and pathogens. New Phytologist 170, 657-675.",
                "Dudareva, N., et al. (2006). Plant volatiles: recent advances and future perspectives. Critical Reviews in Plant Sciences 25, 417-440.",
                "Niinemets, U. & Reichstein, M. (2002). A model analysis of the effects of nonspecific monoterpenoid storage in leaf tissues on emission kinetics. Global Biogeochemical Cycles 16, 1110.",
            ],
        )
        super().__init__(name=name, config=config or {}, metadata=metadata, device=device)
        self.process_name = name
        self._executions = 0
        self._provides = []
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        # The learnable coefficients are the ParameterManager bindings, held by
        # reference (see _held_coefficient); the Functor owns no module
        # parameter of its own.
        self.f_carbon_monoterpenes = self._held_coefficient(
            "biosphere.bvoc.photosynthetic_coupling.f_carbon_monoterpenes",
            units="dimensionless",
        )
        self.seasonal_amplitude = self._held_coefficient(
            "biosphere.bvoc.monoterpene.seasonal.amplitude",
            units="dimensionless",
        )
        self.seasonal_peak_day = self._held_coefficient(
            "biosphere.bvoc.monoterpene.seasonal.peak_day",
            units="day",
        )
        monoterpene_mw_value = self.get_parameter_value(
            "biosphere.bvoc.molecular_weights.monoterpenes",
            units="g mol-1",
            required=True,
        )
        self.register_buffer(
            "monoterpene_mw",
            torch.as_tensor(
                monoterpene_mw_value, dtype=torch.float32, device=self.device
            ).detach().clone(),
        )
        carbon_fraction_value = self.get_parameter_value(
            "biosphere.bvoc.carbon_fraction.monoterpenes",
            units="dimensionless",
            required=True,
        )
        self.register_buffer(
            "carbon_fraction_monoterpenes",
            torch.as_tensor(
                carbon_fraction_value, dtype=torch.float32, device=self.device
            ).detach().clone(),
        )
        seasonal_minimum_value = self.get_parameter_value(
            "biosphere.bvoc.monoterpene.seasonal.minimum_factor",
            units="dimensionless",
            required=True,
        )
        self.register_buffer(
            "seasonal_min_factor",
            torch.as_tensor(
                seasonal_minimum_value, dtype=torch.float32, device=self.device
            ).detach().clone(),
        )
        self.register_buffer(
            "seasonal_period_seconds",
            torch.tensor(
                float(get_physical_constant("seconds_per_year")),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self.register_buffer(
            "seconds_per_day",
            torch.tensor(
                float(get_physical_constant("seconds_per_day")),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        # The pool helper is no Functor, so a binding kept only inside it would
        # be invisible to held_manager_parameters(). Its learnable kinetic
        # coefficients are resolved and held here under their ParameterManager
        # keys and handed to the helper unchanged.
        self._pool_coefficients: Dict[str, Tensor] = {
            key: self._held_coefficient(key, units=units)
            for key, units in (
                ("biosphere.bvoc.monoterpene.v_max_mts", "umol m-2 s-1"),
                ("biosphere.bvoc.monoterpene.pool.k_emit_25", "s-1"),
                ("biosphere.bvoc.monoterpene.pool.beta_emission", "K-1"),
                ("biosphere.bvoc.monoterpene.pool.k_degradation", "s-1"),
                ("biosphere.bvoc.monoterpene.q10_synthase", "dimensionless"),
            )
        }
        self.pool_model = MonoterpenePoolModel(
            v_max_mts=self._pool_coefficients[
                "biosphere.bvoc.monoterpene.v_max_mts"
            ],
            k_gpp=self.get_parameter_value(
                "biosphere.bvoc.monoterpene.k_gpp",
                units="umol m-2 s-1",
                required=True,
                validate=True,
            ),
            pool_capacity=self.get_parameter_value(
                "biosphere.bvoc.monoterpene.pool.capacity",
                units="umol m-2",
                required=True,
                validate=True,
            ),
            pool_initial=self.get_parameter_value(
                "biosphere.bvoc.monoterpene.pool_initial",
                units="umol m-2",
                required=True,
                validate=True,
            ),
            k_emit_25=self._pool_coefficients[
                "biosphere.bvoc.monoterpene.pool.k_emit_25"
            ],
            beta_emission=self._pool_coefficients[
                "biosphere.bvoc.monoterpene.pool.beta_emission"
            ],
            k_degradation=self._pool_coefficients[
                "biosphere.bvoc.monoterpene.pool.k_degradation"
            ],
            q10_synthase=self._pool_coefficients[
                "biosphere.bvoc.monoterpene.q10_synthase"
            ],
            device=self.device,
        ).to(self.device)
        self._init_coupling_parameters()

    def _held_coefficient(self, key: str, units: str) -> Tensor:
        """Resolve one ParameterManager coefficient in the form this Functor holds."""
        # References: Tingey et al. (1980); Niinemets and Reichstein (2002)
        #   (the coefficients of the stored-pool closure); ParameterManager
        #   binding ownership, docs/source/architecture_overview.md Sections
        #   4.2, 4.4.2 and 4.4.3.
        # Mathematical expression: theta_held = theta_manager (the same
        #   object) for a learnable key; otherwise theta_held =
        #   float32(theta_manager).
        # Parameter meanings: theta_manager is the value the ParameterManager
        #   returns for the key; theta_held the value this Functor keeps.
        # Units: the declared units of the key (passed as units).
        # Domain/constraints: a learnable coefficient is held by reference with
        #   the dtype and device of its owner; registration relocates it to the
        #   device of this Functor and a durable checkpoint saves it in
        #   parameter_bindings. A private copy would take the gradient of a call
        #   away from the binding the calibration path optimizes, would not
        #   follow a changed value and would hide the binding from the
        #   registration and checkpoint owners. Every other value is a float32
        #   tensor on the device of this Functor. Every use aligns the held
        #   value with _ensure_tensor.
        # Sign convention: none (a parameter value is passed through unchanged).
        value = self.get_parameter_value(
            key,
            units=units,
            required=True,
            validate=True,
        )
        if self.manager_binding_key(value) is None:
            value = torch.as_tensor(
                value, dtype=torch.float32, device=self.device
            ).detach().clone()
        return value

    def _init_coupling_parameters(self) -> None:
        """Load the non-differentiable seasonal feature selector."""
        self.enable_seasonal_variation = self._load_seasonal_flag()

    def _as_scalar(self, value: Union[Tensor, float]) -> float:
        if isinstance(value, Tensor):
            detached = value.detach()
            if detached.numel() == 1:
                return detached.reshape(-1).to(dtype=torch.float64).tolist()[0]
            return torch.mean(detached).reshape(-1).to(dtype=torch.float64).tolist()[0]
        return float(value)

    def _load_seasonal_flag(self) -> bool:
        enable_seasonal = self.get_parameter_value(
            "biosphere.bvoc.monoterpene.enable_seasonal_variation",
            units="dimensionless",
            required=True,
        )
        return bool(self._as_scalar(enable_seasonal))

    @property
    def seasonal_phase_offset(self) -> Tensor:
        """Return the differentiable seasonal peak phase in radians."""
        return (
            2.0
            * torch.pi
            * self._ensure_tensor(self.seasonal_peak_day, dtype=torch.float32)
            * self.seconds_per_day
            / self.seasonal_period_seconds
        )

    def get_dependencies(self) -> list:
        """Declare the canonical gross-photosynthesis provider."""
        return self._merge_declared_dependencies(["farquhar_photosynthesis"])

    def get_provides(self) -> list:
        """Get provided output variables.

        Returns
        -------
        list
            Output variable names with units:
            - monoterpenes_emission: mol m-2 s-1 (SYSTEM STANDARD)
            - diagnostics.bvoc.monoterpene.emission_rate_molar: umol m-2 s-1
            - diagnostics.bvoc.monoterpene.emission_rate_mass: ug m-2 s-1
            - monoterpene_pool_concentration: umol m-2 (full state)
            - bvoc_carbon_allocation_flux: kg C m-2 s-1 (plant allocation)
        """
        return list(self.metadata.output_variables)

    def forward(
        self, state: Dict[str, Union[Tensor, float]], time: float, dt: float, **kwargs
    ) -> Dict[str, UpdateItem]:
        """
        Forward pass through the phenomenological stored-pool closure.

        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state containing:
            - air_temperature: Grid-cell emitting-foliage temperature proxy [K]
            - gross_photosynthesis: GPP [umol CO2 m-2 s-1]
            - monoterpene_pool_concentration: StateManager-owned pool [umol m-2]
        time : float
            Current time [s]
        dt : float
            Time step [s]

        Returns
        -------
        Dict[str, UpdateItem]
            Canonical monoterpene state, ports, and diagnostics.
        """
        # References: Tingey et al. (1980); Niinemets and Reichstein (2002).
        # Mathematical expression: dP/dt = J_syn - J_emit - J_deg + J_boundary;
        # F_emit,mol = convert(J_emit, umol m-2 s-1, mol m-2 s-1).
        # Parameter meanings: air_temperature is the grid-cell emitting-foliage
        # proxy, gross_photosynthesis drives the calibratable synthesis response,
        # and monoterpene_pool_concentration is the StateManager-owned pool.
        local_state = dict(state)
        temperature = self._extract_tensor(local_state, "air_temperature")
        gpp = self._extract_tensor(local_state, "gross_photosynthesis")
        for name, value in (("air_temperature", temperature), ("gross_photosynthesis", gpp)):
            if value.numel() == 0 or not torch.all(torch.isfinite(value)):
                raise ValueError(f"{name} must be non-empty and finite")
        if torch.any(temperature <= 0):
            raise ValueError("air_temperature must be positive Kelvin")
        if torch.any(gpp < 0):
            raise ValueError("gross_photosynthesis must be non-negative")
        pool_state = self._resolve_pool_state(local_state)
        try:
            temperature, gpp, pool_state = torch.broadcast_tensors(
                temperature, gpp, pool_state
            )
        except RuntimeError as exc:
            raise ValueError(
                "air_temperature, gross_photosynthesis, and "
                "monoterpene_pool_concentration must be broadcast-compatible"
            ) from exc
        if self.enable_seasonal_variation:
            seasonal_factor = self._compute_seasonal_factor(time)
        else:
            seasonal_factor = torch.ones_like(temperature, dtype=torch.float32, device=self.device)
        f_carbon_monoterpenes = self._ensure_tensor(
            self.f_carbon_monoterpenes, dtype=torch.float32
        )
        carbon_flux_available = gpp * f_carbon_monoterpenes * seasonal_factor
        pool_results = self.pool_model(
            carbon_flux_available, temperature, dt, pool_state=pool_state
        )
        emission_rate_molar = pool_results["emission_rate"]
        synthesis_rate = pool_results["synthesis_rate"]
        pool_concentration = pool_results["pool_concentration"]
        emission_rate_molar = emission_rate_molar.to(dtype=torch.float32)
        # References: SI Brochure, 9th Edition; Niinemets et al. (2014).
        # Mathematical expression: E_mol = convert(E_umol, umol m-2 s-1, mol m-2 s-1).
        # Parameter meanings: E_umol is the pool emission rate and E_mol is the
        # canonical atmosphere-facing flux. The central UnitConverter owns the
        # exact SI-prefix conversion.
        emission_rate_mol = unit_converter.convert(
            emission_rate_molar,
            variable_name="monoterpenes_emission",
            source_units="umol m-2 s-1",
            target_units="mol m-2 s-1",
        ).to(dtype=torch.float32)
        # Unit identity: umol m-2 s-1 * g mol-1 = ug m-2 s-1.
        emission_rate_mass = (emission_rate_molar * self.monoterpene_mw).to(
            dtype=torch.float32
        )
        carbon_fraction = self.carbon_fraction_monoterpenes.to(dtype=torch.float32)
        # References: Niinemets et al. (2014); Keeling and Bohlmann (2006).
        # Mathematical expression:
        # F_C,alloc = convert(J_syn * MW_mono * f_C, g C m-2 s-1,
        #                     kg C m-2 s-1).
        # Parameter meanings: J_syn is synthesis [mol m-2 s-1], MW_mono is
        # molecular weight [g mol-1], and f_C is carbon mass fraction [1].
        # Units and route: F_C,alloc is a positive plant-carbon allocation
        # [kg C m-2 s-1]. Stored-pool emission remains a separate atmosphere
        # port and does not create a second plant-carbon debit.
        synthesis_rate_mol = unit_converter.convert(
            synthesis_rate,
            variable_name="molar_flux",
            source_units="umol m-2 s-1",
            target_units="mol m-2 s-1",
        )
        carbon_allocation_flux_g = (
            synthesis_rate_mol * self.monoterpene_mw * carbon_fraction
        )
        carbon_allocation_flux = unit_converter.convert(
            carbon_allocation_flux_g,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        ).to(dtype=torch.float32)
        pool_tendency = pool_results["pool_tendency"]
        pool_delta = pool_results["pool_delta"]
        boundary_residual_rate = pool_results["pool_boundary_residual_rate"]
        pool_tendency_mol = unit_converter.convert(
            pool_tendency,
            variable_name="monoterpene_pool_tendency",
            source_units="umol m-2 s-1",
            target_units="mol m-2 s-1",
        )
        storage_change_g = pool_tendency_mol * self.monoterpene_mw * carbon_fraction
        monoterpene_storage_change = unit_converter.convert(
            storage_change_g,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        ).to(dtype=torch.float32)
        pool_utilization = pool_results["pool_capacity_utilization"]
        temp_factor = pool_results["temperature_emission_factor"]
        # References: Tingey et al. (1980); Keeling and Bohlmann (2006); Grmela and Ottinger (1997).
        # Mathematical expression: x_dot = N_h r_h for synthesis, emission,
        # storage-pool tendency, routed boundary residual, degradation, and
        # plant-carbon allocation cochains.
        # Parameter meanings: N_h is the monoterpene typed-output map; r_h contains
        # ParameterManager-backed synthase, pool, emission, degradation, and carbon rates.
        reaction_rates = torch.broadcast_tensors(
            emission_rate_mol,
            emission_rate_mass,
            synthesis_rate,
            carbon_flux_available,
            emission_rate_molar,
            pool_results["degradation_rate"],
            carbon_allocation_flux,
            pool_tendency,
            monoterpene_storage_change,
            boundary_residual_rate,
        )
        monoterpene_network = advance_reaction_network(
            name=f"{self.name}.monoterpene_synthesis_reaction_network",
            species=(
                "monoterpenes_emission",
                "diagnostics.bvoc.monoterpene.emission_rate_mass",
                "diagnostics.bvoc.monoterpene.synthesis_rate",
                "diagnostics.bvoc.monoterpene.carbon_flux_allocated",
                "diagnostics.bvoc.monoterpene.emission_rate_molar",
                "diagnostics.bvoc.monoterpene.degradation_rate",
                "bvoc_carbon_allocation_flux",
                "diagnostics.bvoc.monoterpene.pool_tendency",
                "diagnostics.bvoc.monoterpene.storage_change",
                "diagnostics.bvoc.monoterpene.pool_boundary_residual_rate",
            ),
            reactions=(
                "monoterpene_molar_emission",
                "monoterpene_mass_emission",
                "monoterpene_synthesis",
                "allocated_carbon_flux",
                "monoterpene_umol_emission",
                "monoterpene_degradation",
                "monoterpene_carbon_allocation",
                "monoterpene_pool_tendency",
                "monoterpene_storage_change",
                "pool_boundary_residual",
            ),
            stoichiometric_N=torch.eye(
                10, device=emission_rate_molar.device, dtype=emission_rate_molar.dtype
            ),
            rates=reaction_rates,
            open_system_route="port",
        )
        values_and_units = {
            "monoterpenes_emission": (
                monoterpene_network.species_rate("monoterpenes_emission"),
                "mol m-2 s-1",
            ),
            self.POOL_STATE_KEY: (pool_concentration, "umol m-2"),
            "bvoc_carbon_allocation_flux": (
                monoterpene_network.species_rate("bvoc_carbon_allocation_flux"),
                "kg C m-2 s-1",
            ),
            "diagnostics.bvoc.monoterpene.emission_rate_molar": (
                monoterpene_network.species_rate(
                    "diagnostics.bvoc.monoterpene.emission_rate_molar"
                ),
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.monoterpene.emission_rate_mass": (
                monoterpene_network.species_rate(
                    "diagnostics.bvoc.monoterpene.emission_rate_mass"
                ),
                "ug m-2 s-1",
            ),
            "diagnostics.bvoc.monoterpene.pool_tendency": (
                monoterpene_network.species_rate(
                    "diagnostics.bvoc.monoterpene.pool_tendency"
                ),
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.monoterpene.pool_delta": (pool_delta, "umol m-2"),
            "diagnostics.bvoc.monoterpene.pool_boundary_residual_rate": (
                monoterpene_network.species_rate(
                    "diagnostics.bvoc.monoterpene.pool_boundary_residual_rate"
                ),
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.monoterpene.storage_change": (
                monoterpene_network.species_rate(
                    "diagnostics.bvoc.monoterpene.storage_change"
                ),
                "kg C m-2 s-1",
            ),
            "diagnostics.bvoc.monoterpene.synthesis_rate": (
                monoterpene_network.species_rate(
                    "diagnostics.bvoc.monoterpene.synthesis_rate"
                ),
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.monoterpene.pool_capacity_utilization": (
                pool_utilization,
                "1",
            ),
            "diagnostics.bvoc.monoterpene.temperature_emission_factor": (
                temp_factor,
                "1",
            ),
            "diagnostics.bvoc.monoterpene.carbon_flux_allocated": (
                monoterpene_network.species_rate(
                    "diagnostics.bvoc.monoterpene.carbon_flux_allocated"
                ),
                "umol CO2 m-2 s-1",
            ),
            "diagnostics.bvoc.monoterpene.degradation_rate": (
                monoterpene_network.species_rate(
                    "diagnostics.bvoc.monoterpene.degradation_rate"
                ),
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.monoterpene.seasonal_factor": (
                seasonal_factor,
                "1",
            ),
        }
        return {
            key: UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=False,
                source_functor=self.name,
            )
            for key, (value, unit) in values_and_units.items()
        }

    def diagnose_conservation(
        self,
        state: Dict[str, Union[Tensor, float]],
        updates: Dict[str, Union[Tensor, UpdateItem]],
    ) -> Dict[str, Tensor]:
        """Provide plant-carbon allocation and atmosphere-port diagnostics.

        References:
            - Tingey et al. (1980) monoterpene emission temperature dependence
            - Keeling & Bohlmann (2006) terpene synthase mechanisms
        Mathematical expressions:
            m_C,g = J_syn * MW_mono * f_C
            m_C,kg = convert(m_C,g, g C m-2 s-1, kg C m-2 s-1)
            where J_syn is monoterpene synthesis [mol m-2 s-1],
                  MW_mono is molecular weight [g mol-1], and
                  f_C is the ParameterManager carbon mass fraction [1].
        Parameters:
            monoterpene_mw: molecular weight of monoterpenes from parameter DB [g mol-1]
        """
        del state
        emission_update = updates.get("monoterpenes_emission")
        if emission_update is None:
            return {}
        emission_mol = (
            emission_update.value
            if isinstance(emission_update, UpdateItem)
            else emission_update
        )
        emission_mol = emission_mol.to(device=self.device, dtype=torch.float32)
        carbon_allocation_update = updates.get("bvoc_carbon_allocation_flux")
        if carbon_allocation_update is None:
            raise KeyError(
                "Missing canonical output 'bvoc_carbon_allocation_flux'."
            )
        carbon_allocation = (
            carbon_allocation_update.value
            if isinstance(carbon_allocation_update, UpdateItem)
            else carbon_allocation_update
        )
        return {
            "bvoc_carbon_allocation_flux": carbon_allocation,
            "bvoc_monoterpene_emission_mol": emission_mol,
        }

    def check_conservation_laws(
        self,
        input_state: Dict[str, Tensor],
        output_updates: Dict[str, Union[Tensor, UpdateItem]],
    ) -> Dict[str, float]:
        """
        Audit the open monoterpene synthesis-storage carbon ledger.

        References:
            - Niinemets and Reichstein (2002), stored-pool emission kinetics
            - Niinemets et al. (2014), BVOC formation, storage, and release
        Mathematical expression:
            R_C = F_alloc,C - F_emit,C - F_deg,C + F_boundary,C
                  - F_storage,C.
        Parameter meanings:
            F_alloc,C is positive plant-carbon allocation, F_emit,C is
            atmosphere-facing emission, F_deg,C is pool degradation,
            F_boundary,C is the capacity or non-negativity correction, and
            F_storage,C is the resulting stored-pool tendency. All terms are
            converted to kg C m-2 s-1 through the central UnitConverter.
        Open-system route:
            Plant carbon enters through bvoc_carbon_allocation_flux and emitted
            monoterpenes leave through the existing atmosphere coupling port.
        """
        # References: Tingey et al. (1980); Niinemets and Reichstein (2002); declared pool ledger.
        # Mathematical expression: r=F_C_alloc-F_C_emit-F_C_deg+F_C_boundary-F_C_storage;
        #   report=max(abs(r)).
        # Parameter meanings: All molar terms are converted by the canonical unit owner using the
        #   same compound MW and carbon fraction.
        # Units: Every residual term [kg C m-2 s-1].
        # Domain/constraints: Present finite same-shape ports; the current implementation uses
        #   float32. The clipping term is a disclosed numerical residual, not evidence of a real
        #   external carbon source.
        # Sign convention: Synthesis is positive input, emission/degradation exports positive,
        #   storage tendency signed, and boundary correction retains its actual sign.

        del input_state

        def require_tensor(key: str) -> Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing conservation output '{key}'")
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Monoterpene conservation output '{key}' must be a torch.Tensor."
                )
            if value.numel() == 0:
                raise ValueError(
                    f"Monoterpene conservation output '{key}' must be non-empty."
                )
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"Monoterpene conservation output '{key}' must be floating."
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    f"Monoterpene conservation output '{key}' must be finite."
                )
            return value.to(device=self.device, dtype=torch.float32)

        allocation_carbon = require_tensor("bvoc_carbon_allocation_flux")
        emission_mol = require_tensor("monoterpenes_emission")
        degradation_umol = require_tensor(
            "diagnostics.bvoc.monoterpene.degradation_rate"
        )
        storage_carbon = require_tensor(
            "diagnostics.bvoc.monoterpene.storage_change"
        )
        boundary_umol = require_tensor(
            "diagnostics.bvoc.monoterpene.pool_boundary_residual_rate"
        )
        ledger_values = (
            allocation_carbon,
            emission_mol,
            degradation_umol,
            storage_carbon,
            boundary_umol,
        )
        if len({tuple(value.shape) for value in ledger_values}) != 1:
            raise ValueError(
                "Monoterpene conservation outputs must have identical shapes."
            )

        carbon_fraction = self.carbon_fraction_monoterpenes.to(
            device=self.device,
            dtype=torch.float32,
        )
        molecular_weight = self.monoterpene_mw.to(
            device=self.device,
            dtype=torch.float32,
        )

        def molar_carbon_to_kg(molar_flux: Tensor) -> Tensor:
            carbon_g = molar_flux * molecular_weight * carbon_fraction
            return unit_converter.convert(
                carbon_g,
                variable_name="carbon_flux",
                source_units="g C m-2 s-1",
                target_units="kg C m-2 s-1",
            ).to(dtype=torch.float32)

        degradation_mol = unit_converter.convert(
            degradation_umol,
            variable_name="molar_flux",
            source_units="umol m-2 s-1",
            target_units="mol m-2 s-1",
        )
        boundary_mol = unit_converter.convert(
            boundary_umol,
            variable_name="molar_flux",
            source_units="umol m-2 s-1",
            target_units="mol m-2 s-1",
        )
        residual = (
            allocation_carbon
            - molar_carbon_to_kg(emission_mol)
            - molar_carbon_to_kg(degradation_mol)
            + molar_carbon_to_kg(boundary_mol)
            - storage_carbon
        )
        return {
            "carbon_boundary": float(torch.max(torch.abs(residual)).detach().item())
        }

    def _compute_seasonal_factor(self, time: float) -> Tensor:
        """
        Compute seasonal variation factor for monoterpene synthesis.

        Seasonal modulation uses ParameterManager response parameters and
        physical-constant time bases:
        - seasonal_period_seconds (s), physical-constant SSOT
        - seasonal_phase_offset (rad)
        - seasonal_amplitude (dimensionless)
        - seasonal_min_factor (dimensionless clamp)
        """
        # References: Existing ParameterManager seasonal modulation; Niinemets and Reichstein
        #   (2002) model context.
        # Mathematical expression: f_season=max(f_min,1+a*sin(2*pi*t/period-phase_offset)).
        # Parameter meanings: a is seasonal amplitude; phase_offset the fitted phase; period is the
        #   physical-constant time basis.
        # Units: t,period [s]; phase [rad]; f_season,a [1].
        # Domain/constraints: A calibrated periodic multiplier, not a calendar-aware phenology
        #   model; the current implementation converts time to float32.
        # Sign convention: The lower-bounded nonnegative multiplier scales the allocated synthesis
        #   driver.
        time_tensor = torch.as_tensor(time, dtype=torch.float32, device=self.device)
        phase = 2.0 * torch.pi / self.seasonal_period_seconds * time_tensor
        seasonal_amplitude = self._ensure_tensor(
            self.seasonal_amplitude, dtype=torch.float32
        )
        seasonal_factor = 1.0 + seasonal_amplitude * torch.sin(
            phase - self.seasonal_phase_offset
        )
        return torch.clamp(seasonal_factor, min=self.seasonal_min_factor)

    def _resolve_pool_state(self, state: Dict[str, Union[Tensor, float]]) -> Tensor:
        """
        Resolve the required canonical monoterpene pool state.

        Returns
        -------
        Tensor
            Pool concentration tensor on the functor device.
        """
        if self.POOL_STATE_KEY not in state:
            raise KeyError(f"Missing required state '{self.POOL_STATE_KEY}'.")
        pool_value = state[self.POOL_STATE_KEY]
        if not isinstance(pool_value, Tensor):
            raise TypeError(f"{self.POOL_STATE_KEY} must be a torch.Tensor")
        if not torch.is_floating_point(pool_value):
            raise TypeError(f"{self.POOL_STATE_KEY} must be a floating torch.Tensor")
        resolved = pool_value.to(device=self.device, dtype=torch.float32)
        if resolved.numel() == 0 or not torch.all(torch.isfinite(resolved)):
            raise ValueError(f"{self.POOL_STATE_KEY} must be non-empty and finite")
        if torch.any((resolved < 0) | (resolved > self.pool_model.pool_capacity)):
            raise ValueError(f"{self.POOL_STATE_KEY} must lie within [0, pool_capacity]")
        return resolved

    def _extract_tensor(
        self, state: Dict[str, Union[Tensor, float]], key: str, default: Optional[float] = None
    ) -> Tensor:
        """Extract one required floating Tensor without scalar fallback."""
        if key in state:
            value = state[key]
        elif default is not None:
            value = default
        else:
            raise KeyError(f"Missing required state '{key}'.")
        if not isinstance(value, Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        if not torch.is_floating_point(value):
            raise TypeError(f"{key} must be a floating torch.Tensor")
        return value.to(device=self.device, dtype=torch.float32)

    def reset_pools(self, initial_concentration: Optional[Tensor] = None) -> Tensor:
        """Return sanitized pool concentration for external state managers."""
        return self.pool_model.reset_pool(initial_concentration)

    def get_pool_diagnostics(self, state: Dict[str, Union[Tensor, float]]) -> Dict[str, Any]:
        """Get detailed pool diagnostic information."""
        pool_state_tensor = self._resolve_pool_state(state)
        pool_capacity = self.pool_model.pool_capacity
        capacity_utilization = self._as_scalar(pool_state_tensor / pool_capacity)
        if capacity_utilization > 0.9:
            status = "full"
        elif capacity_utilization < 0.1:
            status = "empty"
        else:
            status = "normal"
        return {
            "current_pool_concentration": self._as_scalar(pool_state_tensor),
            "pool_capacity": self._as_scalar(pool_capacity),
            "capacity_utilization": capacity_utilization,
            "pool_status": status,
        }


__all__ = ["MonoterpenePoolModel", "MonoterpeneSynthesis"]
