"""
Unified Farquhar Photosynthesis Model for RELIC

This module provides the authoritative implementation of the Farquhar-von Caemmerer-Berry
biochemical model of C3 photosynthesis. It serves
as the single source of photosynthetic calculations for the RELIC framework.

References
----------
- Farquhar, G.D., von Caemmerer, S., Berry, J.A. (1980). A biochemical model of 
  photosynthetic CO2 assimilation in leaves of C3 species. Planta 149, 78-90
- Bernacchi, C.J., et al. (2001). Improved temperature response functions for models 
  of Rubisco-limited photosynthesis. Plant, Cell & Environment 24, 253-259
- Medlyn, B.E., et al. (2002). Temperature response of parameters of a biochemically 
  based model of photosynthesis. Plant, Cell & Environment 25, 1205-1216
- De Pury, D.G.G. and Farquhar, G.D. (1997). Simple scaling of photosynthesis from 
  leaves to canopies without the errors of big-leaf models. Plant, Cell and Environment 20, 537-557
- Wullschleger, S.D. (1993). Biochemical limitations to carbon assimilation in C3 
  plants - A retrospective analysis of the A/Ci curves from 109 species. 
  Journal of Experimental Botany 44, 907-920

Mathematical Expressions
-------------------------
1. RuBisCO-limited rate: Ac = Vcmax * (Ci - Gamma_star) / (Ci + Kc * (1 + O/Ko))
2. Light-limited rate: Aj = J * (Ci - Gamma_star) / (4 * (Ci + 2*Gamma_star))
3. Net photosynthesis: A = min(Ac, Aj) - Rd
4. Electron transport: J = (alpha*I + Jmax - sqrt((alpha*I + Jmax)^2 - 4*theta*alpha*I*Jmax)) / (2*theta)
5. Temperature response: f(T) = f25 * exp(Ea/R * (1/T_ref - 1/T)) / (1 + exp((S*T - Hd)/(R*T)))

Units
-----
- A, Ac, Aj, Rd: umol CO2 m-2 s-1
- Vcmax: umol CO2 m^-2 s^-1
- Jmax, J: umol e- m^-2 s^-1
- Ci, Ca, Gamma_star: umol mol^-1 (ppm)
- Kc: umol mol^-1 (ppm)
- Ko: mmol mol^-1
- I: umol photons m^-2 s^-1
- Temperature: K
- O2: mmol mol^-1 (assumed 210)
"""

from typing import Dict, Literal, Optional, Union, Any, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

# References: Farquhar et al. (1980); Leuning (1995); Sellers et al. (1996).
# Mathematical expression: A = A_pot * s_v * f_w * f_T * f_O3; an absent optional
#   modifier s_v, f_w, f_T or f_O3 takes the identity value 1.
# Parameter meanings: s_v the Vcmax capacity scaling; f_w, f_T, f_O3 the water,
#   temperature and ozone stress factors (1 = no limitation).
# Units: all modifiers are dimensionless.
# Domain/constraints: the identity of a multiplicative modifier, not a physical
#   parameter; a supplied stress factor is validated in [0, 1].
# Sign convention: none (multiplicative factors).
_ABSENT_MODIFIER_IDENTITY = 1.0

# Import from core framework
from relic.core.functor import BiologicalProcessFunctor
from relic.core.abstractions import StateDict, FluxDict
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.unified_response_functions import (
    arrhenius_temperature_response,
    beer_lambert_effective_leaf_area,
    peaked_temperature_response,
    q10_temperature_response,
    light_response_rectangular_hyperbola,
    validate_positive_values
)
from relic.utils.physical_constants import get_physical_constant
from relic.computation.operators.iphs import advance_reaction_network
from relic.scientific.atmosphere import daylight_quadrature

class FarquharPhotosynthesis(BiologicalProcessFunctor):
    """
    Unified Farquhar-von Caemmerer-Berry C3 Photosynthesis Model.
    
    This is the authoritative implementation for photosynthesis calculations.
    Supports differentiable CPU/GPU computation through one eager runtime path.
    
    Features
    --------
    - Complete C3 biochemical model with temperature dependencies
    - One differentiable eager tensor path on the simulator-assigned device
    - Comprehensive parameter validation
    - Uncertainty quantification support
    - Integration with unified response function library
    
    Architecture
    ------------
    - Inherits from FunctorBase for framework integration
    - Implements BiologicalProcess abstraction
    - Uses PyTorch nn.Module for automatic differentiation
    - Modular design with separate methods for each limitation
    
    Runtime Contract
    ----------------
    - Emits one molar gross-assimilation port and one molar net-assimilation port.
    - Converts gross assimilation to the canonical carbon-mass GPP port using
      the ParameterManager physical-constant SSOT.
    - Uses the existing BiologicalProcessFunctor and StateManager lifecycle.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    ENVIRONMENT_SOURCES = ("canonical_providers", "external_forcing")
    # Declared temporal supports; config["temporal_support"] selects one, and an absent
    # key keeps "instantaneous" (the law evaluated once on the supplied inputs):
    #   "daylight_gauss": daily step only (dt = seconds_per_day). The law is evaluated at
    #     n Gauss instants of the daylight period and once for the night and averaged
    #     with the sample weights of relic.scientific.atmosphere.daylight_quadrature
    #     (Goudriaan 1986; Goudriaan and van Laar 1994): ppfd is the 24-hour mean
    #     redistributed by the solar elevation, air_temperature follows the SWAT+
    #     Eq. 1:1.3.1 course of the daily mean, maximum and minimum, and the vapour
    #     pressure deficit is e0(T_k) - e_a with the daily actual vapour pressure, which
    #     replaces the vapor_pressure_deficit input. config["daylight_gauss_nodes"] (3 or
    #     5) and config["daylight_shortwave_shape"] select the rule. Outputs on the
    #     existing ports are 24-hour means; net_photosynthesis_daylight_samples carries
    #     the per-sample net assimilation for a daylight_gauss Medlyn conductance.
    TEMPORAL_SUPPORTS = ("instantaneous", daylight_quadrature.DAYLIGHT_GAUSS)
    DAYLIGHT_INPUTS = ("latitude", "day_of_year", "air_temperature_max", "air_temperature_min",
                       "actual_vapor_pressure")
    DAYLIGHT_SAMPLE_OUTPUTS = ("net_photosynthesis_daylight_samples",)

    def __init__(
        self,
        name: str = "farquhar_photosynthesis",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        enable_gpu: bool = True,
        validate_inputs: bool = True,
        pft_name: Optional[str] = None,
        pathway: Optional[str] = None,
        # Allow direct parameter specification for testing
        vcmax25: Optional[float] = None,
        jmax25: Optional[float] = None,
        rd25: float | None = None,
        parameter_manager: Any | None = None,
        *,
        environment_source: Literal[
            "canonical_providers", "external_forcing"
        ] = "canonical_providers",
    ):
        """
        Initialize unified Farquhar photosynthesis model.
        
        Parameters
        ----------
        name : str, optional
            Functor name for identification, default "farquhar_photosynthesis"
        config : Dict[str, Any], optional
            Configuration parameters, uses defaults if None
        device : torch.device, optional
            Optional pre-registration device override; CPU is the placeholder if None
        enable_gpu : bool, optional
            Enable GPU computation if available, default True  
        validate_inputs : bool, optional
            Enable input validation for safety, default True
        pft_name : Optional[str], optional
            Plant Functional Type name for parameter loading, default None
            If provided, loads PFT-specific photosynthesis parameters
        pathway : Optional[str], optional
            Photosynthetic pathway override ("C3" or "C4"), default None
        parameter_manager : Optional[Any], optional
            Existing parameter owner injected before biochemical initialization.
        environment_source : str, optional
            GraphIR selection field, not a parameter: the provider of
            vapor_pressure_deficit and ppfd. "canonical_providers" declares
            environmental_forcing and atmospheric_radiation as dependencies;
            "external_forcing" declares neither, and both arrive as forcing
            such as observed tower series. The law in forward is the same.
        """
        if environment_source not in self.ENVIRONMENT_SOURCES:
            raise ValueError(
                f"environment_source must be one of {self.ENVIRONMENT_SOURCES}, "
                f"got {environment_source!r}."
            )
        # Fix config to avoid None issues
        conf = dict(config) if config else {}

        direct_overrides = {}
        if vcmax25 is not None:
            direct_overrides["vcmax25"] = vcmax25
        if jmax25 is not None:
            direct_overrides["jmax25"] = jmax25
        if rd25 is not None:
            direct_overrides["rd25"] = rd25
        if direct_overrides:
            # Instance overrides use the existing config-first getter path.
            # Copy the nested mapping so the caller's configuration stays intact.
            parameters = dict(conf.get("parameters", {}))
            parameters.update(direct_overrides)
            conf["parameters"] = parameters
        if pathway is not None:
            conf['pathway'] = pathway
        
        # Create metadata for FunctorBase
        from relic.core.functor import (
            FunctorMetadata,
            ProcessCategory,
            ExecutionPhase,
            SpatialScale,
            VariableKind,
        
    TemporalScale,
)
        
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            description="Farquhar-von Caemmerer-Berry C3 Photosynthesis Model",
            mathematical_expression="A = min(Ac, Aj) - Rd; Ac = Vcmax * (Ci - Gamma_star) / (Ci + Kc * (1 + O/Ko)); Aj = J * (Ci - Gamma_star) / (4 * (Ci + 2*Gamma_star))",
            references=[
                "Farquhar, G.D., von Caemmerer, S., Berry, J.A. (1980). A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. Planta 149, 78-90",
                "Medlyn et al. (2002). Temperature response of parameters of a biochemically based model of photosynthesis. Plant, Cell & Environment 25, 1167-1179",
                "Bernacchi et al. (2001). Improved temperature response functions for models of Rubisco-limited photosynthesis. Plant, Cell & Environment 24, 253-259"
            ],
            parameters={
                "biosphere.photosynthesis.pathway": "Photosynthetic pathway type (C3 or C4) [dimensionless]",
                "biosphere.photosynthesis.vcmax25": "Maximum carboxylation velocity at 25C [umol CO2 m-2 s-1]",
                "biosphere.photosynthesis.jmax25": "Maximum electron transport rate at 25C [umol e- m-2 s-1]",
                "biosphere.photosynthesis.rd25": "Dark respiration rate at 25C [umol CO2 m-2 s-1]",
                "biosphere.photosynthesis.kc25": "Michaelis constant for CO2 at 25C [umol mol-1]",
                "biosphere.photosynthesis.ko25": "Michaelis constant for O2 at 25C [mmol mol-1]",
                "biosphere.photosynthesis.gamma_star25": "CO2 compensation point without Rd at 25C [umol mol-1]",
                "biosphere.photosynthesis.ha_vcmax": "Activation energy for Vcmax [J mol-1]",
                "biosphere.photosynthesis.hd_vcmax": "Deactivation energy for Vcmax [J mol-1]",
                "biosphere.photosynthesis.s_vcmax": "Entropy factor for Vcmax [J mol-1 K-1]",
                "biosphere.photosynthesis.ha_jmax": "Activation energy for Jmax [J mol-1]",
                "biosphere.photosynthesis.hd_jmax": "Deactivation energy for Jmax [J mol-1]",
                "biosphere.photosynthesis.s_jmax": "Entropy factor for Jmax [J mol-1 K-1]",
                "biosphere.respiration.rd_q10": "Q10 coefficient for dark respiration [dimensionless]",
                "biosphere.photosynthesis.ha_kc": "Activation energy for Kc [J mol-1]",
                "biosphere.photosynthesis.ha_ko": "Activation energy for Ko [J mol-1]",
                "biosphere.photosynthesis.ha_gamma_star": "Activation energy for Gamma star [J mol-1]",
                "biosphere.photosynthesis.alpha": "Quantum yield of electron transport [dimensionless]",
                "biosphere.photosynthesis.theta": "Curvature parameter for light response curve [dimensionless]",
                "biosphere.photosynthesis.o2_concentration": "Atmospheric O2 concentration [mmol mol-1]",
                "biosphere.photosynthesis.vpmax25": "Maximum C4 PEP carboxylation rate at 25C [umol CO2 m-2 s-1]",
                "biosphere.photosynthesis.kp25": "C4 PEP carboxylase Michaelis constant at 25C [umol mol-1]",
                "biosphere.photosynthesis.phi": "C4 bundle-sheath leakiness fraction [dimensionless]",
                "biosphere.photosynthesis.ve_max25": "Maximum C4 bundle-sheath Rubisco capacity at 25C [umol CO2 m-2 s-1]",
                "biosphere.photosynthesis.ha_vpmax": "Activation energy for Vpmax [J mol-1]",
                "biosphere.photosynthesis.hd_vpmax": "Deactivation energy for Vpmax [J mol-1]",
                "biosphere.photosynthesis.s_vpmax": "Entropy term for Vpmax [J mol-1 K-1]",
                "biosphere.photosynthesis.ha_kp": "Activation energy for Kp [J mol-1]",
                "biosphere.photosynthesis.ha_ve_max": "Activation energy for Ve_max [J mol-1]",
                "biosphere.photosynthesis.hd_ve_max": "Deactivation energy for Ve_max [J mol-1]",
                "biosphere.photosynthesis.s_ve_max": "Entropy term for Ve_max [J mol-1 K-1]",
                "biosphere.photosynthesis.vcmax_min_scaling": "Minimum Vcmax scaling factor for numerical stability and biological realism in photosyn... [dimensionless]",
                "biosphere.photosynthesis.vcmax_max_scaling": "Maximum Vcmax scaling factor for numerical stability and biological realism in photosyn... [dimensionless]",
                "biosphere.photosynthesis.canopy_light_extinction": "Beer-Lambert light extinction coefficient for LAI scaling [dimensionless]",
                "biosphere.photosynthesis.rubisco_kinetic_factor1": "Rubisco kinetic factor for CO2 specificity [dimensionless]",
                "biosphere.photosynthesis.rubisco_kinetic_factor2": "Secondary Rubisco kinetic factor [dimensionless]",
                "biosphere.photosynthesis.c4_pathway_type": "Indicator variable for C4 pathway activation (1.0 for C4) [dimensionless]",
                "validation.numerical.epsilon_divide": "Small epsilon value to prevent division by zero [dimensionless]",
                "validation.numerical.clamp_min_zero": "Minimum clamp value for zero-bounded quantities [dimensionless]",
                "validation.numerical.clamp_min_small": "Small positive minimum clamp value [dimensionless]",
                "validation.numerical.clamp_max_one": "Maximum clamp value for unit-bounded quantities [dimensionless]",
                "validation.numerical.clamp_min_negative": "Minimum clamp threshold for negative pressure potentials to avoid numerical instability [MPa]",
                "validation.numerical.clamp_max_zero": "Maximum clamp threshold for pressure potentials (e.g., water potential) capped at 0 MPa [MPa]",
                "validation.numerical.clamp_min_temperature": "Minimum valid temperature for biological processes (-40 degC) [K]",
                "validation.numerical.clamp_max_temperature": "Maximum valid temperature for biological processes (50 degC) [K]",
                "validation.numerical.clamp_min_co2": "Minimum valid CO2 concentration [umol mol-1]",
                "validation.numerical.clamp_max_co2": "Maximum valid CO2 concentration [umol mol-1]",
                "validation.numerical.vpd_response_d0": "VPD response scale [Pa]",
                "validation.numerical.c4_electron_factor": "Electron transport factor for C4 photosynthesis [dimensionless]",
                "validation.numerical.max_ppfd_validation": "Maximum valid photosynthetic photon flux density [umol photons m-2 s-1]",
            },
            input_variables=[
                "air_temperature",
                "atmospheric_co2_ppm",
                "ppfd",
                "vapor_pressure_deficit",
                "vcmax_scaling",
                "vpmax_scaling",
                "jmax_scaling",
                "water_stress_factor",
                "temperature_stress_factor",
                "ozone_stress_factor",
                "leaf_area_index",
            ],
            output_variables=[
                "gross_photosynthesis",
                "net_photosynthesis",
                "gross_primary_productivity",
                "electron_transport_rate",
                "diagnostics.photosynthesis.dark_respiration",
                "diagnostics.photosynthesis.transport_limited",
                "diagnostics.photosynthesis.jmax_actual",
                "diagnostics.photosynthesis.limitation_factor",
                "diagnostics.photosynthesis.pathway_type"
            ],
            # This leaf-scale closure exposes open-system assimilation and
            # respiration fluxes; reservoir conservation is owned by the
            # coupled atmosphere-biosphere ledger. This process-specific law
            # checks the molar identity and canonical carbon-mass mapping.
            conservation_laws=["carbon_boundary"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.FLUX,
            coupling_interfaces=[
                "biosphere_atmosphere_photosynthesis",
                "stomatal_conductance",
                "canopy_energy_balance"
            ]
        )
        temporal_support = conf.get("temporal_support", "instantaneous")
        if temporal_support not in self.TEMPORAL_SUPPORTS:
            raise ValueError(
                "FarquharPhotosynthesis config temporal_support must be one of "
                f"{list(self.TEMPORAL_SUPPORTS)}, got {temporal_support!r}"
            )
        daylight_rule: Optional[Tuple[int, str]] = None
        if temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            daylight_rule = daylight_quadrature.daylight_config(conf, "FarquharPhotosynthesis")
            metadata.input_variables.remove("vapor_pressure_deficit")
            metadata.input_variables.extend(self.DAYLIGHT_INPUTS)
            metadata.output_variables.extend(self.DAYLIGHT_SAMPLE_OUTPUTS)
        else:
            daylight_quadrature.reject_daylight_keys(conf, "FarquharPhotosynthesis")
        
        # Initialize FunctorBase
        super().__init__(name=name, config=conf, metadata=metadata,
                         parameter_manager=parameter_manager)
        self.temporal_support = temporal_support
        self._daylight_rule = daylight_rule

        # Device and computation settings (set before _init_parameters to avoid device reference error)
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.enable_gpu = enable_gpu and self.device.type == "cuda"
        self.validate_inputs = validate_inputs
        
        self.pft_name = pft_name
        # Canonical VPD and PPFD providers. They are instance-level declarations
        # merged with config["dependencies"] (never overwriting it); an explicit
        # composition (set_dependencies, GraphExecutor links) rebinds them.
        # environment_source = external_forcing declares neither provider.
        self.environment_source = environment_source
        if environment_source == "canonical_providers":
            for provider in ("environmental_forcing", "atmospheric_radiation"):
                self.add_dependency(provider)
        # The scaling and stress factors are optional multiplicative modifiers
        # (absent = identity 1); the required-input rule
        # (FunctorBase.get_required_inputs) reads this declaration, leaving
        # air_temperature, atmospheric_co2_ppm, ppfd, vapor_pressure_deficit and
        # the canopy state leaf_area_index required.
        self.optional_input_variables = [
            "vcmax_scaling",
            "vpmax_scaling",
            "jmax_scaling",
            "water_stress_factor",
            "temperature_stress_factor",
            "ozone_stress_factor",
        ]

        # Initialize parameters from configuration (including overrides from constructor)
        self._init_parameters(conf, pft_name=pft_name)
        
        # Cache frequently used numerical parameters for hot path optimization
        self._cache_numerical_parameters()
        
        # Load PFT-specific parameters if provided
        if pft_name:
            self._load_pft_parameters(pft_name)
        
    def _init_parameters(self, config: Dict[str, Any], pft_name: Optional[str] = None):
        """
        Initialize biochemical parameters with literature values.

        Manager bindings keep their one owner. Private parameters remain private;
        other typed tensor values are persistent module buffers.
        """
        # Initialize parameter cache for hot path optimization
        self._param_cache = {}

        # Photosynthetic pathway (C3 default if not specified)
        self.pathway = self._resolve_photosynthesis_pathway(pft_name=pft_name)
        if self.pathway not in {"C3", "C4"}:
            raise ValueError(f"Unsupported pathway: {self.pathway}. Must be 'C3' or 'C4'")

        # References: Farquhar et al. (1980); architecture SSOT sections 4.2 and 6.1.
        # Mathematical expression: theta_functor = theta_source for
        #   theta = (Vcmax25, Jmax25, Rd25); A = min(Ac, Aj) - Rd is unchanged.
        # Parameter meanings: Vcmax25 is reference carboxylation capacity,
        #   Jmax25 reference electron transport capacity, Rd25 reference respiration;
        #   theta_source is the validated config-first or manager-owned value;
        #   A is net uptake, Ac carboxylation-limited uptake, Aj electron-limited
        #   uptake, and Rd temperature-adjusted respiration.
        # Units: Vcmax25,Rd25 [umol CO2 m-2 s-1]; Jmax25 [umol e- m-2 s-1].
        #   A,Ac,Aj,Rd [umol CO2 m-2 s-1].
        # Domain/constraints: the existing getter enforces declared units/bounds;
        #   preserve source dtype, gradient flag and graph; never create a new leaf.
        # Sign convention: capacities are nonnegative; Rd is subtracted once from
        #   gross uptake, and initialization performs no carbon transfer.
        for parameter_name in ("vcmax25", "jmax25", "rd25"):
            value = self.get_parameter_value(
                parameter_name,
                namespace="biosphere.photosynthesis",
                source_order=("config", "manager"),
                required=True,
            )
            if isinstance(value, nn.Parameter):
                setattr(self, parameter_name, value)
            else:
                self.register_buffer(parameter_name, value)

        # Michaelis-Menten constants
        self.kc25 = self.get_parameter_value("kc25", namespace="biosphere.photosynthesis", required=True)
        self.ko25 = self.get_parameter_value("ko25", namespace="biosphere.photosynthesis", required=True)
        self.gamma_star25 = self.get_parameter_value("gamma_star25", namespace="biosphere.photosynthesis", required=True)

        # Temperature response parameters (activation / deactivation energies)
        self.vcmax_ha = self.get_parameter_value("ha_vcmax", namespace="biosphere.photosynthesis", required=True)
        self.vcmax_hd = self.get_parameter_value("hd_vcmax", namespace="biosphere.photosynthesis", required=True)
        self.vcmax_s = self.get_parameter_value("s_vcmax", namespace="biosphere.photosynthesis", required=True)

        self.jmax_ha = self.get_parameter_value("ha_jmax", namespace="biosphere.photosynthesis", required=True)
        self.jmax_hd = self.get_parameter_value("hd_jmax", namespace="biosphere.photosynthesis", required=True)
        self.jmax_s = self.get_parameter_value("s_jmax", namespace="biosphere.photosynthesis", required=True)

        self.rd_q10 = self.get_parameter_value("rd_q10", namespace="biosphere.respiration", required=True)

        self.kc_ha = self.get_parameter_value("ha_kc", namespace="biosphere.photosynthesis", required=True)
        self.ko_ha = self.get_parameter_value("ha_ko", namespace="biosphere.photosynthesis", required=True)
        self.gamma_star_ha = self.get_parameter_value("ha_gamma_star", namespace="biosphere.photosynthesis", required=True)

        # Light response parameters
        self.alpha = self.get_parameter_value("alpha", namespace="biosphere.photosynthesis", required=True)
        self.theta = self.get_parameter_value("theta", namespace="biosphere.photosynthesis", required=True)

        # Physical constants are resolved only through the physical-constant SSOT API.
        self.gas_constant = get_physical_constant("gas_constant")
        self.reference_temp = get_physical_constant("t_ref")
        self.carbon_molar_mass_g_mol = get_physical_constant(
            "relic.physical.constants.molecular_weights.carbon"
        )
        self.o2_conc = self.get_parameter_value("o2_concentration", namespace="biosphere.photosynthesis", required=True)

        if self.pathway == 'C4':
            self._initialize_c4_parameters()

    def _initialize_c4_parameters(self) -> None:
        """Load C4-specific biochemical parameters from the parameter manager."""
        self.vpmax25 = self._ensure_tensor(
            self.get_parameter_value("vpmax25", namespace="biosphere.photosynthesis", required=True)
        )
        self.kp25 = self.get_parameter_value("kp25", namespace="biosphere.photosynthesis", required=True)
        self.phi = self.get_parameter_value("phi", namespace="biosphere.photosynthesis", required=True)
        self.ve_max25 = self._ensure_tensor(
            self.get_parameter_value("ve_max25", namespace="biosphere.photosynthesis", required=True)
        )

        self.vpmax_ha = self.get_parameter_value("ha_vpmax", namespace="biosphere.photosynthesis", required=True)
        self.vpmax_hd = self.get_parameter_value("hd_vpmax", namespace="biosphere.photosynthesis", required=True)
        self.vpmax_s = self.get_parameter_value("s_vpmax", namespace="biosphere.photosynthesis", required=True)

        self.kp_ha = self.get_parameter_value("ha_kp", namespace="biosphere.photosynthesis", required=True)
        self.ve_max_ha = self.get_parameter_value("ha_ve_max", namespace="biosphere.photosynthesis", required=True)
        self.ve_max_hd = self.get_parameter_value("hd_ve_max", namespace="biosphere.photosynthesis", required=True)
        self.ve_max_s = self.get_parameter_value("s_ve_max", namespace="biosphere.photosynthesis", required=True)

    def _resolve_photosynthesis_pathway(self, pft_name: Optional[str] = None) -> str:
        """
        Resolve the photosynthetic pathway from ParameterManager using PFT-specific overrides.
        """
        namespace_candidates: List[str] = []
        if pft_name:
            namespace_candidates.append(f"biosphere.pft.{pft_name}.photosynthesis")
        namespace_candidates.append("biosphere.photosynthesis")

        for namespace in namespace_candidates:
            try:
                pathway_value = self.get_parameter_value(
                    "pathway",
                    namespace=namespace,
                    source_order=("config", "manager"),
                )
                if pathway_value is not None:
                    return str(pathway_value).upper()
            except (KeyError, ValueError):
                continue

        raise ValueError(
            "Parameter 'pathway' was not found in biosphere photosynthesis namespaces. "
            "Please register biosphere.photosynthesis.pathway or the relevant PFT-specific override."
        )

    def _ensure_tensor(self, value, name: str = None):
        """Ensure value is a tensor with correct device and dtype."""
        dtype = torch.get_default_dtype()
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=self.device, dtype=dtype)
        else:
            if value.device != self.device or value.dtype != dtype:
                value = value.to(device=self.device, dtype=dtype)
        return value

    def _as_tensor_like(self, value: Union[float, Tensor], reference: Tensor) -> Tensor:
        """Convert scalar parameter values to the reference tensor device and dtype."""
        if isinstance(value, Tensor):
            return value.to(device=reference.device, dtype=reference.dtype)
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    
    def _params_like(self, reference: Tensor, *values: Union[float, Tensor]) -> Tuple[Tensor, ...]:
        """Align several parameter values to the reference tensor device and dtype."""
        return tuple(self._as_tensor_like(value, reference) for value in values)

    def _as_float(self, value: Union[float, Tensor]) -> float:
        """Convert scalar tensors to Python floats for bounds/defaults."""
        if isinstance(value, Tensor):
            return float(value.detach().reshape(-1)[0].tolist())
        return float(value)

    def _cache_numerical_parameters(self):
        """
        Cache frequently used numerical parameters to avoid repeated lookups in hot path.
        
        This method is called once during initialization to cache all validation
        and numerical stability parameters that are accessed frequently during
        forward passes.
        """
        # Cache validation parameters for numerical stability
        cache_target = self._param_cache
        cache_target['epsilon_divide'] = self.get_parameter_value(
            'epsilon_divide',
            namespace="validation.numerical", required=True)
        cache_target['clamp_min_zero'] = self.get_parameter_value(
            'clamp_min_zero',
            namespace="validation.numerical", required=True)
        cache_target['clamp_min_small'] = self.get_parameter_value(
            'clamp_min_small',
            namespace="validation.numerical", required=True)
        cache_target['clamp_max_one'] = self.get_parameter_value(
            'clamp_max_one',
            namespace="validation.numerical", required=True)
        cache_target['clamp_min_negative'] = self.get_parameter_value(
            'clamp_min_negative',
            namespace="validation.numerical", required=True)
        cache_target['clamp_max_zero'] = self.get_parameter_value(
            'clamp_max_zero',
            namespace="validation.numerical", required=True)
        
        # Cache temperature range parameters
        cache_target['clamp_min_temperature'] = self.get_parameter_value(
            'clamp_min_temperature',
            namespace="validation.numerical", required=True)
        cache_target['clamp_max_temperature'] = self.get_parameter_value(
            'clamp_max_temperature',
            namespace="validation.numerical", required=True)
        
        # Cache VPD and other model parameters
        cache_target['vpd_response_d0'] = self.get_parameter_value(
            'vpd_response_d0',
            namespace="validation.numerical", required=True)
        cache_target['c4_electron_factor'] = self.get_parameter_value(
            'c4_electron_factor',
            namespace="validation.numerical", required=True)
        
        # Cache vcmax scaling bounds
        cache_target['vcmax_max_scaling'] = self.get_parameter_value(
            'vcmax_max_scaling',
            namespace="biosphere.photosynthesis", required=True)
        cache_target['vcmax_min_scaling'] = self.get_parameter_value(
            'vcmax_min_scaling',
            namespace="biosphere.photosynthesis", required=True)
        
        # Cache the canopy light extinction coefficient
        cache_target['canopy_light_extinction'] = self.get_parameter_value(
            'canopy_light_extinction',
            namespace="biosphere.photosynthesis", required=True)
        cache_target['rubisco_kinetic_factor1'] = self.get_parameter_value(
            'rubisco_kinetic_factor1',
            namespace="biosphere.photosynthesis", required=True)
        cache_target['rubisco_kinetic_factor2'] = self.get_parameter_value(
            'rubisco_kinetic_factor2',
            namespace="biosphere.photosynthesis", required=True)
        
        # Cache C4-specific parameters if C4 pathway
        if self.pathway == 'C4':
            cache_target['c4_pathway_type'] = self.get_parameter_value(
                'c4_pathway_type',
                namespace="biosphere.photosynthesis", required=True)

    def _canopy_lai_scale(self, lai: torch.Tensor) -> torch.Tensor:
        # References: Monsi and Saeki (1953); de Pury and Farquhar (1997).
        # Mathematical expression: L_eff = integral_0^LAI exp(-k L) dL
        # = (1 - exp(-k * LAI)) / k.
        # Parameter meanings: k is the canopy light extinction coefficient;
        # LAI is leaf area index; L_eff is illuminated leaf area per ground area.
        extinction = torch.as_tensor(
            self._param_cache['canopy_light_extinction'],
            dtype=lai.dtype,
            device=lai.device,
        )
        return beer_lambert_effective_leaf_area(lai, extinction)
    
    def _check_numerical_consistency(self, output1: Dict[str, Tensor], output2: Dict[str, Tensor], 
                                    rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """
        Check numerical consistency between different computation paths.
        
        Parameters
        ----------
        output1, output2 : Dict[str, Tensor]
            Outputs to compare
        rtol : float
            Relative tolerance
        atol : float
            Absolute tolerance
            
        Returns
        -------
        bool
            True if outputs are numerically consistent
        """
        for key in output1.keys():
            if key in output2:
                if not torch.allclose(output1[key], output2[key], rtol=rtol, atol=atol):
                    import logging
                    logging.warning(f"Numerical inconsistency in {key}: "
                                  f"max diff = {torch.max(torch.abs(output1[key] - output2[key]))}")
                    return False
        return True
    
    def get_provides(self) -> list:
        """
        Get output variables provided by photosynthesis calculation.
        
        Returns
        -------  
        list
            Output variable names
        """
        return list(self.metadata.output_variables)
    
    def forward(
        # References: Farquhar et al. (1980); von Caemmerer (2000); Medlyn et al. (2002);
        #   Bernacchi et al. (2001).
        # Mathematical expression: Vcmax(T), Jmax(T) by the peaked Arrhenius response normalised
        #   at T_ref; J from the non-rectangular hyperbola
        #   theta J^2 - (alpha I + Jmax) J + alpha I Jmax = 0;
        #   Ac = Vcmax (Ci - Gamma*)/(Ci + Kc (1 + O/Ko)); Aj = J (Ci - Gamma*)/(4 Ci + 8 Gamma*);
        #   A = min(Ac, Aj) - Rd, scaled by the declared stress factors.
        # Parameter meanings: Vcmax, Jmax maximum carboxylation and electron transport rates;
        #   Ci intercellular CO2; Gamma* the compensation point; Kc, Ko Michaelis constants;
        #   O oxygen; Rd day respiration; alpha, theta the light-response parameters.
        # Units: A, Ac, Aj, Rd, Vcmax [umol CO2 m-2 s-1]; Jmax, J [umol e- m-2 s-1];
        #   I [umol photons m-2 s-1]; T [K];
        #   Ci, Gamma*, Kc, Ko, O in the concentration units declared by the module inputs
        #   (atmospheric_co2_ppm in umol mol-1).
        # Domain/constraints: the runtime dtype follows the floating inputs; temperature responses
        #   use the supplied leaf or air temperature; stress factors lie in [0, 1]; the limitation
        #   indicator is emitted in the runtime dtype.
        # Sign convention: positive A is net CO2 uptake by the leaf.
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs,
    ) -> FluxDict:
        """
        Forward pass through Farquhar photosynthesis model.
        
        This is the main computation method that performs the complete
        photosynthesis calculation with all outputs.
        
        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state containing:
            - air_temperature: Air temperature [K]
            - atmospheric_co2_ppm: Atmospheric CO2 mole fraction [umol mol-1]
            - ppfd: Photosynthetic photon flux density [umol photons m-2 s-1]
            - vapor_pressure_deficit: Vapor pressure deficit [Pa]
            Optional:
            - vcmax_scaling: Vcmax scaling factor [dimensionless]
            - jmax_scaling: Jmax scaling factor [dimensionless] (defaults to vcmax_scaling)
            - water_stress_factor: Water stress control [0-1] (absent: identity 1)
            - temperature_stress_factor: Temperature stress control [0-1] (absent: identity 1)
            - ozone_stress_factor: Ozone stress control [0-1] (absent: identity 1)
            Required canopy state:
            - leaf_area_index: LAI for canopy scaling [m^2 m^-2] (no default)
        time : float
            Current simulation time [s] (not used in current implementation)
        dt : float  
            Time step [s] (not used in current implementation)
        **kwargs
            Additional keyword arguments
            
        Returns
        -------
        Dict[str, UpdateItem]
            Canonical instantaneous outputs with explicit REPLACE semantics.
        """
        runtime_kwargs = dict(kwargs)
        runtime_kwargs.setdefault('time', float(time))
        runtime_kwargs.setdefault('dt', float(dt))

        if self.temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            return self._forward_daylight_gauss(state, **runtime_kwargs)
        return self._forward_core(state, **runtime_kwargs)
    
    def _forward_daylight_gauss(self, state: Dict[str, Union[Tensor, float]], **kwargs) -> Dict[str, UpdateItem]:
        """Daily means of the instantaneous law over the daylight Gauss samples."""
        # References: Goudriaan (1986), Agric. For. Meteorol. 38, 249-254; Goudriaan and
        #   van Laar (1994); relic.scientific.atmosphere.daylight_quadrature (FAO-56 Eqs.
        #   24, 25, 28, 34; SWAT+ Eq. 1:1.3.1 temperature course).
        # Mathematical expression: X = sum_k W_k X(T_k, I_k, D_k) for every output X of
        #   the law (A, A_gross, R_d, GPP, J, A_j, J_max, the limitation indicator);
        #   W_i = (N/24) w_i at the daylight Gauss instants, W_night = 1 - N/24 with
        #   I_night = 0, so the night term is -R_d(T_night) for A.
        # Parameter meanings: T_k the course of the daily mean, maximum and minimum air
        #   temperature; I_k the 24-hour mean ppfd redistributed by the solar elevation
        #   (sum_k W_k I_k = I_mean exactly); D_k = max(e0(T_k) - e_a, 0); CO2, LAI and the
        #   scaling and stress factors are held constant over the day.
        # Units: as the instantaneous law; samples carry a trailing axis of length n + 1.
        # Domain/constraints: dt = seconds_per_day; each sample passes the input checks
        #   of the instantaneous law; conservation (A = A_gross - R_d, GPP mapping) is
        #   enforced at the samples and holds for the weighted means (linear).
        # Sign convention: positive A is net uptake.
        daylight_quadrature.require_daily_step(kwargs["dt"], "FarquharPhotosynthesis")
        node_count, shortwave_shape = self._daylight_rule
        names = [key for key in self.metadata.input_variables if key in state]
        missing = [key for key in self.metadata.input_variables
                   if key not in state and key not in self.optional_input_variables]
        if missing:
            raise ValueError(f"Required input(s) {missing} not found in state")
        runtime_dtype = self._runtime_dtype(state)
        values = {}
        for key in names:
            value = state[key]
            value = value if isinstance(value, Tensor) else torch.as_tensor(value)
            values[key] = value.to(device=self.device, dtype=runtime_dtype)
        samples = daylight_quadrature.daylight_sample_state(
            values, {}, node_count=node_count, shortwave_shape=shortwave_shape,
            flux_keys=("ppfd",), temperature=True, vapor_pressure_deficit=True, dtype=runtime_dtype)
        sample_outputs = self._forward_core(samples.state, **kwargs)
        outputs = daylight_quadrature.daily_outputs(sample_outputs, samples.support)
        self._enforce_forward_conservation(state, outputs)
        net = sample_outputs["net_photosynthesis"]
        outputs["net_photosynthesis_daylight_samples"] = UpdateItem(
            value=net.value, mode=UpdateMode.REPLACE, unit=net.unit, per_timestep=False,
            source_functor=self.name)
        return outputs

    def _forward_core(self, state: Dict[str, Union[Tensor, float]], **kwargs) -> Dict[str, UpdateItem]:
        """
        Core photosynthesis computation implementation.

        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state dictionary
        **kwargs
            Runtime keywords (time, dt); the inputs come from the state only

        Returns
        -------
        Dict[str, UpdateItem]
            Canonical computation results.
        """
        # Route to appropriate pathway
        if self.pathway == 'C4':
            return self._forward_c4(dict(state))
        else:  # C3 (default)
            return self._forward_c3(dict(state))

    def _apply_iphs_photosynthesis_network(self, output: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Lower photosynthesis flux outputs through the explicit N_h r_h operator."""

        required = ("gross_photosynthesis", "net_photosynthesis", "dark_respiration")
        if not all(key in output for key in required):
            return output
        gross_photosynthesis = output["gross_photosynthesis"]
        net_photosynthesis = output["net_photosynthesis"]
        dark_respiration = output["dark_respiration"]
        species_rows = {
            "gross_photosynthesis": [1.0, 0.0, 0.0],
            "net_photosynthesis": [0.0, 1.0, 0.0],
            "dark_respiration": [0.0, 0.0, 1.0],
        }
        species = tuple(key for key in species_rows if key in output)
        # References: Farquhar et al. (1980); Collatz et al. (1992); Grmela and Ottinger (1997).
        # Mathematical expression: x_dot = N_h r_h with r_h = [A_gross, A_net, R_d].
        # Parameter meanings: N_h maps gross assimilation, net assimilation, and dark
        # respiration reaction channels onto the typed photosynthesis output basis.
        photosynthesis_network = advance_reaction_network(
            name=f"{self.name}.photosynthesis_reaction_network",
            species=species,
            reactions=("gross_assimilation", "net_assimilation", "dark_respiration"),
            stoichiometric_N=torch.as_tensor(
                [species_rows[key] for key in species],
                device=gross_photosynthesis.device,
                dtype=gross_photosynthesis.dtype,
            ),
            rates=(gross_photosynthesis, net_photosynthesis, dark_respiration),
            open_system_route="port",
        )
        for key in species:
            output[key] = photosynthesis_network.species_rate(key)
        output["gross_primary_productivity"] = self._gross_to_gpp_mass_flux(
            output["gross_photosynthesis"]
        )
        return output

    def _gross_to_gpp_mass_flux(self, gross_photosynthesis: Tensor) -> Tensor:
        """Convert gross CO2 assimilation to the canonical carbon mass flux.

        References:
            Farquhar, von Caemmerer, and Berry (1980); NIST carbon atomic mass.
        Mathematical expression:
            GPP_kgC = A_gross * M_C * 1e-9.
        Parameter meanings:
            A_gross is gross assimilation [umol CO2 m-2 s-1], M_C is the
            physical-constant SSOT carbon molar mass [g mol-1], and 1e-9
            converts micromoles to moles and grams to kilograms.
        """

        molar_mass = self._as_tensor_like(
            self.carbon_molar_mass_g_mol,
            gross_photosynthesis,
        )
        conversion = gross_photosynthesis.new_tensor(1.0e-9)
        return gross_photosynthesis * molar_mass * conversion

    @staticmethod
    def _require_finite(name: str, value: Tensor) -> None:
        """Reject non-finite scientific inputs instead of substituting defaults."""
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")

    def _validate_stress_factor(self, name: str, value: Tensor) -> None:
        """Reject dimensionless stress controls outside their declared bounds."""
        self._require_finite(name, value)
        lower = self._as_tensor_like(self._param_cache['clamp_min_zero'], value)
        upper = self._as_tensor_like(self._param_cache['clamp_max_one'], value)
        if torch.any(value < lower) or torch.any(value > upper):
            raise ValueError(
                f"{name} outside supported range "
                f"[{self._as_float(lower)}, {self._as_float(upper)}]"
            )

    def _validate_required_finite_inputs(
        self,
        air_temperature: Tensor,
        atmospheric_co2_ppm: Tensor,
        ppfd: Tensor,
        vapor_pressure_deficit: Tensor,
    ) -> None:
        """Apply unconditional finite validation to the four canonical forcings."""
        for name, value in (
            ("air_temperature", air_temperature),
            ("atmospheric_co2_ppm", atmospheric_co2_ppm),
            ("ppfd", ppfd),
            ("vapor_pressure_deficit", vapor_pressure_deficit),
        ):
            self._require_finite(name, value)

    def _finalize_output(
        self,
        output: Dict[str, Tensor],
        *,
        electron_transport_rate: Tensor,
        transport_limited: Tensor,
        jmax_actual: Tensor,
        limitation_factor: Tensor,
        pathway_type: Tensor,
    ) -> Dict[str, UpdateItem]:
        """Expose the exact canonical ports with explicit replacement semantics."""
        values = {
            "gross_photosynthesis": output["gross_photosynthesis"],
            "net_photosynthesis": output["net_photosynthesis"],
            "gross_primary_productivity": output["gross_primary_productivity"],
            "electron_transport_rate": electron_transport_rate,
            "diagnostics.photosynthesis.dark_respiration": output["dark_respiration"],
            "diagnostics.photosynthesis.transport_limited": transport_limited,
            "diagnostics.photosynthesis.jmax_actual": jmax_actual,
            "diagnostics.photosynthesis.limitation_factor": limitation_factor,
            "diagnostics.photosynthesis.pathway_type": pathway_type,
        }
        units = {
            "gross_photosynthesis": "umol CO2 m-2 s-1",
            "net_photosynthesis": "umol CO2 m-2 s-1",
            "gross_primary_productivity": "kg C m-2 s-1",
            "electron_transport_rate": "umol e- m-2 s-1",
            "diagnostics.photosynthesis.dark_respiration": "umol CO2 m-2 s-1",
            "diagnostics.photosynthesis.transport_limited": "umol CO2 m-2 s-1",
            "diagnostics.photosynthesis.jmax_actual": "umol e- m-2 s-1",
            "diagnostics.photosynthesis.limitation_factor": "dimensionless",
            "diagnostics.photosynthesis.pathway_type": "dimensionless",
        }
        unexpected = set(values) ^ (
            set(self.metadata.output_variables) - set(self.DAYLIGHT_SAMPLE_OUTPUTS)
        )
        if unexpected:
            raise RuntimeError(
                "Farquhar runtime output declaration mismatch: "
                f"{sorted(unexpected)}"
            )
        for name, value in values.items():
            self._require_finite(name, value)
        return {
            name: UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=units[name],
                per_timestep=False,
                source_functor=self.name,
            )
            for name, value in values.items()
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Return the open leaf carbon-boundary residual.

        References:
            Farquhar, von Caemmerer, and Berry (1980), Planta 149, 78-90.
        Mathematical expression:
            A_net + R_d - A_gross = 0 and
            GPP_kgC - A_gross M_C 1e-9 = 0.
        Parameter meanings:
            A_net and A_gross are net and gross CO2 assimilation fluxes, and
            R_d is dark respiration [umol CO2 m-2 s-1]. GPP_kgC is gross
            primary productivity [kg C m-2 s-1]. Atmospheric exchange is an
            open port; no local carbon-stock conservation is claimed.
        """

        del input_state
        port_names = (
            "gross_photosynthesis",
            "net_photosynthesis",
            "gross_primary_productivity",
            "diagnostics.photosynthesis.dark_respiration",
        )
        ports: Dict[str, Tensor] = {}
        for name in port_names:
            if name not in output_updates:
                raise KeyError(f"Missing conservation port '{name}'")
            value = output_updates[name]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, Tensor):
                raise TypeError(f"Conservation port '{name}' must be a Tensor")
            if value.numel() == 0:
                raise ValueError(f"Conservation port '{name}' must not be empty")
            if not value.is_floating_point():
                raise TypeError(
                    f"Conservation port '{name}' must use a floating dtype"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"Conservation port '{name}' must be finite")
            ports[name] = value

        shapes = {tuple(value.shape) for value in ports.values()}
        if len(shapes) != 1:
            raise ValueError("Photosynthesis conservation port shape mismatch")
        dtypes = {value.dtype for value in ports.values()}
        if len(dtypes) != 1:
            raise ValueError("Photosynthesis conservation port dtype mismatch")
        devices = {value.device for value in ports.values()}
        if len(devices) != 1:
            raise ValueError("Photosynthesis conservation port device mismatch")

        for name in (
            "gross_photosynthesis",
            "gross_primary_productivity",
            "diagnostics.photosynthesis.dark_respiration",
        ):
            if bool(torch.any(ports[name] < 0.0).item()):
                raise ValueError(
                    f"Conservation port '{name}' must be non-negative"
                )

        gross = ports["gross_photosynthesis"]
        net = ports["net_photosynthesis"]
        respiration = ports["diagnostics.photosynthesis.dark_respiration"]
        gpp_mass = ports["gross_primary_productivity"]
        expected_net = gross - respiration
        expected_gpp_mass = self._gross_to_gpp_mass_flux(gross)

        molar_scale = torch.maximum(torch.abs(net), torch.abs(expected_net))
        molar_scale = torch.where(
            molar_scale > 0.0,
            molar_scale,
            torch.ones_like(molar_scale),
        )
        mass_scale = torch.maximum(
            torch.abs(gpp_mass),
            torch.abs(expected_gpp_mass),
        )
        mass_scale = torch.where(
            mass_scale > 0.0,
            mass_scale,
            torch.ones_like(mass_scale),
        )
        molar_residual = torch.max(torch.abs(net - expected_net) / molar_scale)
        mass_residual = torch.max(
            torch.abs(gpp_mass - expected_gpp_mass) / mass_scale
        )
        magnitude = torch.maximum(molar_residual, mass_residual)
        return {"carbon_boundary": float(magnitude.detach().item())}
    
    def _forward_c3(self, state: Dict[str, Union[Tensor, float]]) -> Dict[str, UpdateItem]:
        """
        C3 photosynthesis computation (Farquhar model).
        
        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state dictionary with canonical forcing names.
            
        Returns
        -------
        Dict[str, UpdateItem]
            Canonical instantaneous outputs.
        """
        # References: Farquhar et al. 1980; von Caemmerer and Berry 1982; Bernacchi et al. 2001
        # Formula: A = min(Ac, Aj) - Rd
        #          Ac = Vcmax * (Ci - Gamma*) / (Ci + Kc * (1 + O/Ko))
        #          Aj = J * (Ci - Gamma*) / (4 * (Ci + 2*Gamma*))
        # Parameters: vcmax25, jmax25, rd25, rd_q10, kc25, ko25, gamma_star25,
        #             alpha, theta, o2_conc, gas_constant, reference_temp,
        #             water_stress_factor, temperature_stress_factor,
        #             ozone_stress_factor, vpd_response_d0
        mapped_state = state

        # Canonical forcing ports are required and are never reconstructed from aliases.
        temperature = self._extract_tensor(mapped_state, 'air_temperature', required=True)

        co2_conc = self._extract_tensor(
            mapped_state, 'atmospheric_co2_ppm', required=True
        )
        ppfd = self._extract_tensor(mapped_state, 'ppfd', required=True)
        vpd = self._extract_tensor(
            mapped_state, 'vapor_pressure_deficit', required=True
        )
        vcmax_scaling = self._extract_tensor(
            mapped_state,
            'vcmax_scaling',
            default=_ABSENT_MODIFIER_IDENTITY,
        )

        # jmax_scaling scales with vcmax_scaling unless supplied
        jmax_scaling = self._extract_tensor(mapped_state, 'jmax_scaling', default=vcmax_scaling)
        
        # Stress ownership remains external to the Farquhar biochemical closure.
        # References: Farquhar et al. (1980); Leuning (1995).
        # Mathematical expression: A_stressed = A_potential * f_w * f_T * f_O3.
        # Parameter meanings: f_w, f_T, and f_O3 are dimensionless controls in [0, 1].
        # Units: all stress factors are dimensionless.
        # Open-system route: the controls modulate the atmospheric CO2 uptake port.
        water_stress_factor = self._extract_tensor(
            mapped_state,
            'water_stress_factor',
            default=_ABSENT_MODIFIER_IDENTITY,
        )
        temp_stress = self._extract_tensor(
            mapped_state,
            'temperature_stress_factor',
            default=_ABSENT_MODIFIER_IDENTITY,
        )
        ozone_stress = self._extract_tensor(
            mapped_state,
            'ozone_stress_factor',
            default=_ABSENT_MODIFIER_IDENTITY,
        )
        lai = self._extract_tensor(mapped_state, 'leaf_area_index', required=True)

        for name, value in (
            ('jmax_scaling', jmax_scaling),
            ('water_stress_factor', water_stress_factor),
            ('temperature_stress_factor', temp_stress),
            ('ozone_stress_factor', ozone_stress),
            ('leaf_area_index', lai),
        ):
            self._require_finite(name, value)
        for name, value in (
            ('water_stress_factor', water_stress_factor),
            ('temperature_stress_factor', temp_stress),
            ('ozone_stress_factor', ozone_stress),
        ):
            self._validate_stress_factor(name, value)

        # Required forcings always receive finite validation. Optional physical
        # range validation remains controlled by validate_inputs.
        self._validate_required_finite_inputs(temperature, co2_conc, ppfd, vpd)
        if self.validate_inputs:
            temperature, co2_conc, ppfd, vpd, vcmax_scaling = self._validate_inputs(
                temperature, co2_conc, ppfd, vpd, vcmax_scaling
            )
        
        # 1. Calculate temperature-dependent parameters
        vcmax_temp = self._temperature_response_vcmax(temperature)
        jmax_temp = self._temperature_response_jmax(temperature) 
        rd_temp = self._temperature_response_rd(temperature)
        
        # Michaelis-Menten constants
        kc_temp = self._temperature_response_kc(temperature)
        ko_temp = self._temperature_response_ko(temperature)
        gamma_star_temp = self._temperature_response_gamma_star(temperature)
        
        # 2. Apply scaling factors (e.g., nitrogen limitation)
        vcmax_actual = vcmax_temp * vcmax_scaling
        jmax_actual = jmax_temp * jmax_scaling
        
        # 3. Calculate electron transport rate
        j_actual = self._electron_transport_rate(ppfd, jmax_actual)
        
        # 4. Calculate RuBisCO-limited photosynthesis
        ac = self._carboxylation_limited_rate(
            vcmax_actual, co2_conc, kc_temp, ko_temp, gamma_star_temp
        )
        
        # 5. Calculate light-limited photosynthesis  
        aj = self._light_limited_rate(j_actual, co2_conc, gamma_star_temp)
        
        # 6. Determine limiting process
        gross_photosynthesis = torch.minimum(ac, aj)
        limitation_factor = (aj < ac).to(dtype=aj.dtype)  # 0 for Ac-limited, 1 for Aj-limited
        
        # 7. Apply environmental stress factors
        # Water stress directly reduces both gross and respiration rates
        gross_photosynthesis = (
            gross_photosynthesis
            * water_stress_factor
            * temp_stress
            * ozone_stress
        )
        rd_temp = rd_temp * water_stress_factor
        
        # References: Leuning (1995), Plant Cell and Environment 18, 339-355.
        # Mathematical expression: f_vpd = 1 / (1 + D / D0).
        # Parameter meanings: D is vapor pressure deficit and D0 is the fitted
        # sensitivity scale. Both quantities are expressed in Pa.
        # Units: f_vpd [1], D [Pa], D0 [Pa].
        # Open-system route: this factor modulates the atmospheric CO2 uptake port.
        vpd_factor = 1.0 / (1.0 + vpd / self._as_tensor_like(self._param_cache['vpd_response_d0'], vpd))
        gross_photosynthesis = gross_photosynthesis * vpd_factor

        # 8. Calculate net photosynthesis
        net_photosynthesis = gross_photosynthesis - rd_temp
        
        # References: de Pury and Farquhar (1997); Monsi and Saeki (1953).
        # Mathematical expressions: X_canopy = X_leaf * L_eff for
        # X in {A_gross, A_net, R_d, J, A_j}; L_eff = (1-exp(-k*LAI))/k.
        # Parameter meanings: k is light extinction, LAI is leaf area per
        # ground area, and L_eff maps leaf-area rates to ground-area rates.
        lai_scale = self._canopy_lai_scale(lai)
        gross_photosynthesis = gross_photosynthesis * lai_scale
        net_photosynthesis = net_photosynthesis * lai_scale
        rd_temp = rd_temp * lai_scale
        j_actual = j_actual * lai_scale
        aj = aj * lai_scale
        
        # 10. Lower the open assimilation and respiration ports through N_h r_h.
        output = {
            'gross_photosynthesis': gross_photosynthesis,
            'net_photosynthesis': net_photosynthesis,
            'dark_respiration': rd_temp,
        }
        output = self._apply_iphs_photosynthesis_network(output)
        final_output = self._finalize_output(
            output,
            electron_transport_rate=j_actual,
            transport_limited=aj,
            jmax_actual=jmax_actual,
            limitation_factor=limitation_factor,
            pathway_type=torch.zeros_like(gross_photosynthesis),
        )
        self._enforce_forward_conservation(state, final_output)
        return final_output
    
    def _forward_c4(self, state: Dict[str, Union[Tensor, float]]) -> Dict[str, UpdateItem]:
        """
        C4 photosynthesis computation (Collatz 1992 three-limitation model).
        
        Implements the complete C4 photosynthesis model with PEP carboxylase,
        spatial separation of carbon fixation, and three-limitation approach.
        
        Mathematical Expression
        -----------------------
        A = min(Ap, Aj, Ae) - Rd
        where:
        - Ap = PEP carboxylation-limited rate
        - Aj = Electron transport-limited rate
        - Ae = Export/utilization-limited rate
        
        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state dictionary with the same canonical keys as C3.
            
        Returns
        -------
        Dict[str, UpdateItem]
            Canonical instantaneous outputs for C4 photosynthesis.
            
        References
        ----------
        - Collatz, G.J. et al. (1992). Coupled photosynthesis-stomatal conductance model for C4 plants
        - von Caemmerer, S. (2000). Biochemical Models of Leaf Photosynthesis
        - Yin & Struik (2021). C4 photosynthesis and concentration mechanisms
        """
        mapped_state = state

        # Canonical forcing ports are shared with the C3 closure.
        temperature = self._extract_tensor(mapped_state, 'air_temperature', required=True)

        co2_conc = self._extract_tensor(
            mapped_state, 'atmospheric_co2_ppm', required=True
        )
        ppfd = self._extract_tensor(mapped_state, 'ppfd', required=True)
        vpd = self._extract_tensor(
            mapped_state, 'vapor_pressure_deficit', required=True
        )
        vcmax_scaling = self._extract_tensor(
            mapped_state,
            'vcmax_scaling',
            default=_ABSENT_MODIFIER_IDENTITY,
        )

        # vpmax_scaling and jmax_scaling scale with vcmax_scaling unless supplied
        vpmax_scaling = self._extract_tensor(mapped_state, 'vpmax_scaling', default=vcmax_scaling)
        jmax_scaling = self._extract_tensor(mapped_state, 'jmax_scaling', default=vcmax_scaling)
        water_stress_factor = self._extract_tensor(
            mapped_state,
            'water_stress_factor',
            default=_ABSENT_MODIFIER_IDENTITY,
        )
        temp_stress = self._extract_tensor(
            mapped_state,
            'temperature_stress_factor',
            default=_ABSENT_MODIFIER_IDENTITY,
        )
        lai = self._extract_tensor(mapped_state, 'leaf_area_index', required=True)

        for name, value in (
            ('vpmax_scaling', vpmax_scaling),
            ('jmax_scaling', jmax_scaling),
            ('water_stress_factor', water_stress_factor),
            ('temperature_stress_factor', temp_stress),
            ('leaf_area_index', lai),
        ):
            self._require_finite(name, value)
        for name, value in (
            ('water_stress_factor', water_stress_factor),
            ('temperature_stress_factor', temp_stress),
        ):
            self._validate_stress_factor(name, value)

        self._validate_required_finite_inputs(temperature, co2_conc, ppfd, vpd)
        if self.validate_inputs:
            temperature, co2_conc, ppfd, vpd, vcmax_scaling = self._validate_inputs(
                temperature, co2_conc, ppfd, vpd, vcmax_scaling
            )
        
        # 1. Calculate temperature-dependent parameters
        vpmax_temp = self._temperature_response_vpmax(temperature)
        jmax_temp = self._temperature_response_jmax(temperature)
        rd_temp = self._temperature_response_rd(temperature)
        kp_temp = self._temperature_response_kp(temperature)
        ve_max_temp = self._temperature_response_ve_max(temperature)
        
        # 2. Apply scaling factors
        vpmax_actual = vpmax_temp * vpmax_scaling
        jmax_actual = jmax_temp * jmax_scaling
        ve_max_actual = ve_max_temp * vcmax_scaling  # Use vcmax_scaling for ve_max
        
        # 3. Calculate electron transport rate
        j_actual = self._electron_transport_rate(ppfd, jmax_actual)
        
        # 4. Calculate PEP carboxylation-limited rate
        ap = self._pep_carboxylation_limited_rate(vpmax_actual, co2_conc, kp_temp)
        
        # 5. Calculate electron transport-limited rate for C4
        aj = self._light_limited_rate_c4(j_actual, rd_temp)
        
        # 6. Calculate the bundle-sheath Rubisco-capacity limited rate
        ae = self._rubisco_limited_rate_c4(ve_max_actual)
        
        # 7. Determine limiting process (three-way minimum)
        gross_photosynthesis = torch.minimum(torch.minimum(ap, aj), ae)
        
        # Determine which process is limiting
        # 0 = Ap-limited, 1 = Aj-limited, 2 = Ae-limited
        limitation_factor = torch.zeros_like(gross_photosynthesis)
        limitation_factor = torch.where(aj < ap, torch.ones_like(limitation_factor), limitation_factor)
        limitation_factor = torch.where(
            (ae < ap) & (ae < aj), 
            2.0 * torch.ones_like(limitation_factor), 
            limitation_factor
        )
        
        # 8. Apply environmental stress factors
        gross_photosynthesis = (
            gross_photosynthesis * water_stress_factor * temp_stress
        )

        # References: Leuning (1995), Plant Cell and Environment 18, 339-355.
        # Mathematical expression: f_vpd = 1 / (1 + D / D0).
        # Parameter meanings: D is vapor pressure deficit and D0 is the fitted
        # sensitivity scale. Both quantities are expressed in Pa.
        # Units: f_vpd [1], D [Pa], D0 [Pa].
        # Open-system route: this factor modulates the atmospheric CO2 uptake port.
        vpd_factor = 1.0 / (1.0 + vpd / self._as_tensor_like(self._param_cache['vpd_response_d0'], vpd))
        gross_photosynthesis = gross_photosynthesis * vpd_factor

        # 9. Calculate net photosynthesis
        net_photosynthesis = gross_photosynthesis - rd_temp

        # References: de Pury and Farquhar (1997); Monsi and Saeki (1953).
        # Mathematical expressions: X_canopy = X_leaf * L_eff for
        # X in {A_gross, A_net, R_d, J, A_j}; L_eff = (1-exp(-k*LAI))/k.
        # Parameter meanings: k is light extinction, LAI is leaf area per
        # ground area, and L_eff maps leaf-area rates to ground-area rates.
        lai_scale = self._canopy_lai_scale(lai)
        gross_photosynthesis = gross_photosynthesis * lai_scale
        net_photosynthesis = net_photosynthesis * lai_scale
        rd_temp = rd_temp * lai_scale
        j_actual = j_actual * lai_scale
        aj = aj * lai_scale
        
        # 11. Lower the open assimilation and respiration ports through N_h r_h.
        output = {
            'gross_photosynthesis': gross_photosynthesis,
            'net_photosynthesis': net_photosynthesis,
            'dark_respiration': rd_temp,
        }
        output = self._apply_iphs_photosynthesis_network(output)
        final_output = self._finalize_output(
            output,
            electron_transport_rate=j_actual,
            transport_limited=aj,
            jmax_actual=jmax_actual,
            limitation_factor=limitation_factor,
            pathway_type=torch.ones_like(gross_photosynthesis),
        )
        self._enforce_forward_conservation(state, final_output)
        return final_output

    def _temperature_response_vcmax(self, temperature: Tensor) -> Tensor:
        """
        Temperature response of maximum carboxylation velocity.
        
        Uses peaked response function with high-temperature deactivation.
        """
        # References: Bernacchi et al. 2001; Medlyn et al. 2002
        # Mathematical expression: f(T) = f25 * exp(Ha/R*(1/Tref - 1/T))
        #   * (1 + exp((S*Tref - Hd)/(R*Tref))) / (1 + exp((S*T - Hd)/(R*T))),
        #   so that f(Tref) = f25 exactly (normalized peaked Arrhenius).
        # Parameter meanings: vcmax25 [umol CO2 m-2 s-1], vcmax_ha [J mol-1], vcmax_hd [J mol-1],
        #             vcmax_s [J mol-1 K-1], gas_constant [J mol-1 K-1], reference_temp [K]
        # Units: f25 and f(T) share units; Ha,Hd [J mol-1]; S [J mol-1 K-1];
        #   Q10 [1]; R [J mol-1 K-1]; T,Tref [K].
        # Domain/constraints: T > 0 K; evaluated in the input temperature dtype
        #   and device; parameter tensors keep their autograd graph through the
        #   dtype alignment instead of narrowing the temperature.
        # Sign convention: none; a positive reference rate stays positive.
        vcmax_ref = self._as_tensor_like(self.vcmax25, temperature)
        return peaked_temperature_response(
            vcmax_ref, temperature, *self._params_like(temperature, self.vcmax_ha, self.vcmax_hd, self.vcmax_s),
            self.gas_constant, self.reference_temp
        )
    
    def _temperature_response_jmax(self, temperature: Tensor) -> Tensor:
        """
        Temperature response of maximum electron transport rate.
        
        Uses peaked response function with high-temperature deactivation.
        """
        # References: Bernacchi et al. 2001; Medlyn et al. 2002
        # Mathematical expression: f(T) = f25 * exp(Ha/R*(1/Tref - 1/T))
        #   * (1 + exp((S*Tref - Hd)/(R*Tref))) / (1 + exp((S*T - Hd)/(R*T))),
        #   so that f(Tref) = f25 exactly (normalized peaked Arrhenius).
        # Parameter meanings: jmax25 [umol e- m-2 s-1], jmax_ha [J mol-1], jmax_hd [J mol-1],
        #             jmax_s [J mol-1 K-1], gas_constant [J mol-1 K-1], reference_temp [K]
        # Units: f25 and f(T) share units; Ha,Hd [J mol-1]; S [J mol-1 K-1];
        #   Q10 [1]; R [J mol-1 K-1]; T,Tref [K].
        # Domain/constraints: T > 0 K; evaluated in the input temperature dtype
        #   and device; parameter tensors keep their autograd graph through the
        #   dtype alignment instead of narrowing the temperature.
        # Sign convention: none; a positive reference rate stays positive.
        jmax_ref = self._as_tensor_like(self.jmax25, temperature)
        return peaked_temperature_response(
            jmax_ref, temperature, *self._params_like(temperature, self.jmax_ha, self.jmax_hd, self.jmax_s),
            self.gas_constant, self.reference_temp
        )
    
    def _temperature_response_rd(self, temperature: Tensor) -> Tensor:
        """
        Temperature response of dark respiration.
        
        Uses Q10 temperature relationship.
        """
        # References: Lloyd and Taylor 1994; Atkin et al. 2005
        # Mathematical expression: Rd(T) = Rd25 * Q10^((T - Tref)/10).
        # Parameter meanings: rd25 [umol CO2 m-2 s-1], rd_q10 [dimensionless], reference_temp [K]
        # Units: f25 and f(T) share units; Ha,Hd [J mol-1]; S [J mol-1 K-1];
        #   Q10 [1]; R [J mol-1 K-1]; T,Tref [K].
        # Domain/constraints: T > 0 K; evaluated in the input temperature dtype
        #   and device; parameter tensors keep their autograd graph through the
        #   dtype alignment instead of narrowing the temperature.
        # Sign convention: none; a positive reference rate stays positive.
        rd_ref = self._as_tensor_like(self.rd25, temperature)
        return q10_temperature_response(
            rd_ref, temperature, self._as_tensor_like(self.rd_q10, temperature), self.reference_temp
        )
    
    def _temperature_response_kc(self, temperature: Tensor) -> Tensor:
        """Temperature response of Michaelis constant for CO2."""
        # References: Bernacchi et al. 2001; Farquhar et al. 1980
        # Mathematical expression: Kc(T) = Kc25 * exp(Ha/R*(1/Tref - 1/T)).
        # Parameter meanings: kc25 [umol mol-1], kc_ha [J mol-1], gas_constant [J mol-1 K-1], reference_temp [K]
        # Units: f25 and f(T) share units; Ha,Hd [J mol-1]; S [J mol-1 K-1];
        #   Q10 [1]; R [J mol-1 K-1]; T,Tref [K].
        # Domain/constraints: T > 0 K; evaluated in the input temperature dtype
        #   and device; parameter tensors keep their autograd graph through the
        #   dtype alignment instead of narrowing the temperature.
        # Sign convention: none; a positive reference rate stays positive.
        kc_ref = self._as_tensor_like(self.kc25, temperature)
        return arrhenius_temperature_response(
            kc_ref, temperature, self._as_tensor_like(self.kc_ha, temperature), self.gas_constant, self.reference_temp
        )
    
    def _temperature_response_ko(self, temperature: Tensor) -> Tensor:
        """Temperature response of Michaelis constant for O2."""
        # References: Bernacchi et al. 2001; Farquhar et al. 1980
        # Mathematical expression: Ko(T) = Ko25 * exp(Ha/R*(1/Tref - 1/T)).
        # Parameter meanings: ko25 [mmol mol-1], ko_ha [J mol-1], gas_constant [J mol-1 K-1], reference_temp [K]
        # Units: f25 and f(T) share units; Ha,Hd [J mol-1]; S [J mol-1 K-1];
        #   Q10 [1]; R [J mol-1 K-1]; T,Tref [K].
        # Domain/constraints: T > 0 K; evaluated in the input temperature dtype
        #   and device; parameter tensors keep their autograd graph through the
        #   dtype alignment instead of narrowing the temperature.
        # Sign convention: none; a positive reference rate stays positive.
        ko_ref = self._as_tensor_like(self.ko25, temperature)
        return arrhenius_temperature_response(
            ko_ref, temperature, self._as_tensor_like(self.ko_ha, temperature), self.gas_constant, self.reference_temp
        )
    
    def _temperature_response_gamma_star(self, temperature: Tensor) -> Tensor:
        """Temperature response of CO2 compensation point."""
        # References: Bernacchi et al. 2001; Farquhar et al. 1980
        # Mathematical expression: Gamma*(T) = Gamma*25 * exp(Ha/R*(1/Tref - 1/T)).
        # Parameter meanings: gamma_star25 [umol mol-1], gamma_star_ha [J mol-1], gas_constant [J mol-1 K-1], reference_temp [K]
        # Units: f25 and f(T) share units; Ha,Hd [J mol-1]; S [J mol-1 K-1];
        #   Q10 [1]; R [J mol-1 K-1]; T,Tref [K].
        # Domain/constraints: T > 0 K; evaluated in the input temperature dtype
        #   and device; parameter tensors keep their autograd graph through the
        #   dtype alignment instead of narrowing the temperature.
        # Sign convention: none; a positive reference rate stays positive.
        gamma_ref = self._as_tensor_like(self.gamma_star25, temperature)
        return arrhenius_temperature_response(
            gamma_ref, temperature, self._as_tensor_like(self.gamma_star_ha, temperature),
            self.gas_constant, self.reference_temp
        )
    
    def _electron_transport_rate(self, ppfd: Tensor, jmax: Tensor) -> Tensor:
        """
        Calculate electron transport rate from light and Jmax.
        
        Uses the non-rectangular hyperbola light response (curvature theta),
        taking the lower root of the quadratic.
        """
        # References: Farquhar and Wong (1984); de Pury and Farquhar (1997).
        # Mathematical expression: theta*J^2 - (alpha*I + Jmax)*J + alpha*I*Jmax = 0,
        #   J = (alpha*I + Jmax - sqrt((alpha*I + Jmax)^2 - 4*theta*alpha*I*Jmax)) / (2*theta).
        # Parameter meanings: alpha quantum yield of electron transport, theta
        #   curvature, Jmax maximum electron transport rate, I absorbed PPFD;
        #   alpha and theta are aligned to the PPFD dtype before evaluation.
        # Units: alpha [mol e- mol-1 photon]; theta [1]; J,Jmax [umol e- m-2 s-1];
        #   I [umol photons m-2 s-1].
        # Domain/constraints: 0 < theta <= 1 gives a real lower root; the shared
        #   helper clamps the discriminant and the result at zero.
        # Sign convention: J is a nonnegative rate.
        alpha, theta = self._params_like(ppfd, self.alpha, self.theta)
        return light_response_rectangular_hyperbola(ppfd, alpha, jmax, theta)
    
    def _carboxylation_limited_rate(
        self, 
        vcmax: Tensor, 
        ci: Tensor, 
        kc: Tensor, 
        ko: Tensor, 
        gamma_star: Tensor
    ) -> Tensor:
        """
        Calculate RuBisCO-limited carboxylation rate.
        
        Mathematical Expression: Ac = Vcmax * (Ci - Gamma_star) / (Ci + Kc * (1 + O/Ko))
        """
        # References: Farquhar et al. (1980), Planta 149, 78-90; von Caemmerer and
        #   Berry (1982).
        # Mathematical expression: Ac = Vcmax (Ci - Gamma*) / (Ci + Kc (1 + O/Ko) + eps),
        #   clamped at zero.
        # Parameter meanings: Vcmax maximum carboxylation rate; Ci the CO2 mole fraction
        #   used as intercellular CO2; Gamma* the CO2 compensation point without day
        #   respiration; Kc, Ko Michaelis constants for CO2 and O2; O the O2 mole
        #   fraction (o2_conc); eps the epsilon_divide parameter.
        # Units: Vcmax and Ac [umol CO2 m-2 s-1]; Ci, Gamma*, Kc, eps [umol mol-1];
        #   Ko, O [mmol mol-1].
        # Domain/constraints: the denominator is positive for Ci >= 0; rates below the
        #   compensation point are clamped at zero (clamp_min_zero).
        # Sign convention: Ac >= 0 is CO2 uptake into the leaf.
        # Effective Michaelis constant for CO2 (competitive inhibition by O2)
        km_eff = kc * (1.0 + self._as_tensor_like(self.o2_conc, ko) / ko)
        
        # RuBisCO-limited rate
        numerator = vcmax * (ci - gamma_star)
        denominator = ci + km_eff
        
        ac = numerator / (denominator + self._as_tensor_like(self._param_cache['epsilon_divide'], denominator))  # Epsilon to avoid division by zero
        return torch.clamp(
            ac,
            min=self._as_tensor_like(self._param_cache['clamp_min_zero'], ac),
        )
    
    def _light_limited_rate(self, j: Tensor, ci: Tensor, gamma_star: Tensor) -> Tensor:
        """
        Calculate light-limited carboxylation rate.
        
        Mathematical Expression: Aj = J * (Ci - Gamma_star) / (4 * (Ci + 2*Gamma_star))
        """
        # References: Farquhar et al. (1980), Planta 149, 78-90; von Caemmerer and
        #   Berry (1982).
        # Mathematical expression: Aj = J (Ci - Gamma*) / (a1 (Ci + a2 Gamma*) + eps),
        #   clamped at zero; a1 = rubisco_kinetic_factor1 and a2 =
        #   rubisco_kinetic_factor2 (4 and 2 in Farquhar et al. 1980).
        # Parameter meanings: J electron transport rate; Ci the CO2 mole fraction used as
        #   intercellular CO2; Gamma* the CO2 compensation point; a1, a2 the electron
        #   requirement factors of carboxylation and photorespiration; eps the
        #   epsilon_divide parameter.
        # Units: J [umol e- m-2 s-1]; Aj [umol CO2 m-2 s-1]; Ci, Gamma*, eps
        #   [umol mol-1]; a1, a2 [dimensionless].
        # Domain/constraints: the denominator is positive for Ci >= 0; rates below the
        #   compensation point are clamped at zero (clamp_min_zero).
        # Sign convention: Aj >= 0 is CO2 uptake into the leaf.
        numerator = j * (ci - gamma_star)
        # Rubisco kinetic constants
        rubisco_factor1, rubisco_factor2 = self._params_like(
            j, self._param_cache['rubisco_kinetic_factor1'], self._param_cache['rubisco_kinetic_factor2'],
        )
        denominator = rubisco_factor1 * (ci + rubisco_factor2 * gamma_star)
        
        aj = numerator / (denominator + self._as_tensor_like(self._param_cache['epsilon_divide'], denominator))  # Epsilon to avoid division by zero
        return torch.clamp(
            aj,
            min=self._as_tensor_like(self._param_cache['clamp_min_zero'], aj),
        )
    
    _RUNTIME_DTYPE_KEYS = (
        'air_temperature',
        'atmospheric_co2_ppm',
        'ppfd',
        'vapor_pressure_deficit',
        'vcmax_scaling',
        'jmax_scaling',
        'vpmax_scaling',
        'water_stress_factor',
        'temperature_stress_factor',
        'ozone_stress_factor',
        'leaf_area_index',
    )

    def _runtime_dtype(self, state: Dict[str, Union[Tensor, float]]) -> torch.dtype:
        """Return the widest floating dtype among this Functor's tensor inputs.

        References:
          - PyTorch type promotion rules.
          - RELIC runtime-precision contract: runtime state owns dtype.
        Mathematical expression:
          dtype_run = promote_types over floating tensors in the declared
          input keys, or torch.get_default_dtype() when none is present.
        Parameter meanings:
          state is the caller's forcing and control mapping.
        Units:
          none; dtype resolution carries no physical unit.
        Domain/constraints:
          only the keys this Functor consumes participate, so unrelated
          state tensors cannot change the photosynthesis precision;
          parameters are aligned to dtype_run at their use sites.
        Sign convention:
          none.
        """
        resolved: Optional[torch.dtype] = None
        for key in self._RUNTIME_DTYPE_KEYS:
            value = state.get(key)
            if isinstance(value, Tensor) and value.is_floating_point():
                resolved = (
                    value.dtype if resolved is None
                    else torch.promote_types(resolved, value.dtype)
                )
        return resolved if resolved is not None else torch.get_default_dtype()

    def _extract_tensor(
        self, 
        state: Dict[str, Union[Tensor, float]], 
        key: str, 
        required: bool = False,
        default: Union[float, Tensor, None] = None
    ) -> Tensor:
        """
        Extract and convert input to tensor with proper device/dtype.
        
        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state dictionary
        key : str
            Key to extract
        required : bool
            Whether the key is required
        default : Union[float, Tensor, None]
            Default value if key not found
            
        Returns
        -------
        Tensor
            Extracted tensor on correct device
        """
        if key in state:
            value = state[key]
        elif required:
            raise ValueError(f"Required input '{key}' not found in state")
        elif default is not None:
            value = default
        else:
            raise ValueError(f"Input '{key}' not found and no default provided")
        
        # References: PyTorch type promotion; RELIC runtime-precision contract.
        # Mathematical expression: value_run = cast(value, dtype_run) with
        #   dtype_run from _runtime_dtype(state).
        # Parameter meanings: value is one forcing or control input; the
        #   caller's widest floating precision is preserved for every input.
        # Units: unchanged by the cast.
        # Domain/constraints: no fixed float32 narrowing; integer or boolean
        #   inputs are promoted to the runtime floating dtype; the autograd
        #   history of tensor inputs is kept by the cast.
        # Sign convention: none.
        runtime_dtype = self._runtime_dtype(state)
        if not isinstance(value, Tensor):
            value = torch.tensor(value, dtype=runtime_dtype, device=self.device)
        else:
            value = value.to(device=self.device, dtype=runtime_dtype)
        
        # Broadcast scalar / batch-level inputs to match tensor-shaped inputs.
        # This keeps defaults (e.g., stress factors) consistent with batched spatial inputs.
        if value.dim() <= 1:
            ref_shape = None
            for _, ref in state.items():
                if isinstance(ref, Tensor) and ref.dim() > 0:
                    if ref_shape is None or ref.dim() > len(ref_shape):
                        ref_shape = tuple(ref.shape)

            if ref_shape is not None:
                if value.dim() == 0:
                    value = value.expand(ref_shape)
                elif value.dim() == 1 and len(ref_shape) > 1 and value.shape[0] == ref_shape[0]:
                    view_shape = (ref_shape[0],) + (1,) * (len(ref_shape) - 1)
                    value = value.view(view_shape).expand(ref_shape)

        return value
    
    def _pep_carboxylation_limited_rate(self, vpmax: Tensor, ci: Tensor, kp: Tensor) -> Tensor:
        """
        Calculate PEP carboxylation-limited rate for C4 photosynthesis.
        
        Mathematical Expression: Ap = Vpmax * Ci / (Ci + Kp) * (1 - phi)
        
        Parameters
        ----------
        vpmax : Tensor
            Maximum PEP carboxylation velocity [umol CO2 m-2 s-1]
        ci : Tensor
            Intercellular CO2 concentration [umol mol^-1]
        kp : Tensor
            Michaelis constant for PEP carboxylase [umol mol^-1]
            
        Returns
        -------
        Tensor
            PEP carboxylation-limited rate [umol CO2 m-2 s-1]
        """
        # References: Collatz et al. (1992), Aust. J. Plant Physiol. 19, 519-538;
        #   von Caemmerer (2000), Biochemical Models of Leaf Photosynthesis, ch. 4.
        # Mathematical expression: Vp = Vpmax * Ci / (Ci + Kp + eps);
        #   Ap = max(Vp * (1 - phi), 0).
        # Parameter meanings: Vpmax PEP carboxylation capacity (one CO2, as
        #   HCO3-, fixed per PEP carboxylated); Ci the CO2 mole fraction used as
        #   intercellular CO2; Kp Michaelis constant of PEP carboxylase; phi
        #   bundle-sheath leakiness fraction; eps the YAML divide guard.
        # Units: Vpmax, Vp, Ap [umol CO2 m-2 s-1]; Ci, Kp, eps [umol mol-1];
        #   phi [dimensionless].
        # Domain/constraints: Ci >= 0 and Kp > 0 (validated forcing and YAML
        #   bounds); phi within its YAML bounds below 1; Ap is clamped at zero.
        # Sign convention: Ap >= 0 is CO2 fixation (uptake into the leaf).
        vp = vpmax * ci / (ci + kp + self._as_tensor_like(self._param_cache['epsilon_divide'], ci))
        
        # Account for bundle sheath leakiness
        # phi represents the fraction of fixed CO2 that leaks back
        ap = vp * (1.0 - self._as_tensor_like(self.phi, vp))
        
        return torch.clamp(
            ap,
            min=self._as_tensor_like(self._param_cache['clamp_min_zero'], ap),
        )
    
    def _light_limited_rate_c4(self, j: Tensor, rd: Tensor) -> Tensor:
        """
        Calculate electron transport-limited rate for C4 photosynthesis.
        
        Mathematical Expression: Aj = J/3 (gross electron transport-limited rate)
        Factor of 3 accounts for ATP requirements in C4 cycle.
        
        Parameters
        ----------
        j : Tensor
            Actual electron transport rate [umol e^-1 m^-2 s^-1]
        rd : Tensor
            Dark respiration rate [umol CO2 m-2 s-1]
            
        Returns
        -------
        Tensor
            Electron transport-limited rate [umol CO2 m-2 s-1]
        """
        # C4 requires more ATP per CO2 fixed than C3
        # Factor of 3 instead of 4 as in C3
        # References: Collatz et al. (1992), Aust. J. Plant Physiol. 19, 519-538;
        #   von Caemmerer (2000).
        # Mathematical expression: Aj = J / c4_electron_factor (gross), clamped at
        #   zero; Rd is subtracted later in the net assimilation.
        # Parameter meanings: J electron transport rate; c4_electron_factor the
        #   electrons required per CO2 fixed on the C4 path.
        # Units: J [umol e- m-2 s-1]; Aj [umol CO2 m-2 s-1];
        #   c4_electron_factor [dimensionless].
        # Domain/constraints: c4_electron_factor > 0 (parameter bounds); Aj >= 0.
        # Sign convention: Aj >= 0 is CO2 uptake into the leaf.
        aj = j / self._as_tensor_like(self._param_cache['c4_electron_factor'], j)
        
        return torch.clamp(
            aj,
            min=self._as_tensor_like(self._param_cache['clamp_min_zero'], aj),
        )
    
    def _rubisco_limited_rate_c4(self, ve_max: Tensor) -> Tensor:
        """
        Calculate the Rubisco-capacity limited rate for C4 photosynthesis.

        Mathematical Expression: Ae = Ve_max

        Ve_max is the maximum Rubisco carboxylation capacity of the bundle
        sheath: the Rubisco-limited rate V_max of Collatz et al. (1992) and
        Vcmax of the von Caemmerer (2000) C4 model.

        Parameters
        ----------
        ve_max : Tensor
            Bundle-sheath Rubisco capacity [umol CO2 m-2 s-1]

        Returns
        -------
        Tensor
            Rubisco-limited rate [umol CO2 m-2 s-1]
        """
        # References: Collatz et al. (1992), Aust. J. Plant Physiol. 19, 519-538;
        #   von Caemmerer (2000), Biochemical Models of Leaf Photosynthesis, ch. 4.
        # Mathematical expression: Ae = max(Ve_max, 0), the third limitation of
        #   A = min(Ap, Aj, Ae).
        # Parameter meanings: Ve_max the temperature- and control-adjusted
        #   bundle-sheath Rubisco capacity (Collatz V_max, von Caemmerer Vcmax)
        #   that caps gross assimilation; clamp_min_zero the YAML zero floor.
        # Units: Ve_max, Ae [umol CO2 m-2 s-1], the unit of Ap and Aj it is
        #   compared with; clamp_min_zero [dimensionless].
        # Domain/constraints: ve_max25 > 0 by its YAML bounds; Ve_max carries the
        #   sign of vcmax_scaling, and Ae is clamped at zero.
        # Sign convention: Ae >= 0 caps CO2 uptake into the leaf.
        return torch.clamp(
            ve_max,
            min=self._as_tensor_like(self._param_cache['clamp_min_zero'], ve_max),
        )
    
    def _temperature_response_vpmax(self, temperature: Tensor) -> Tensor:
        """
        Temperature response of maximum PEP carboxylation velocity.
        
        Uses peaked response function similar to Vcmax.
        """
        # References: Collatz et al. 1992; Bernacchi et al. 2001
        # Mathematical expression: f(T) = f25 * exp(Ha/R*(1/Tref - 1/T))
        #   * (1 + exp((S*Tref - Hd)/(R*Tref))) / (1 + exp((S*T - Hd)/(R*T))),
        #   so that f(Tref) = f25 exactly (normalized peaked Arrhenius).
        # Parameter meanings: vpmax25 [umol CO2 m-2 s-1], vpmax_ha [J mol-1], vpmax_hd [J mol-1],
        #             vpmax_s [J mol-1 K-1], gas_constant [J mol-1 K-1], reference_temp [K]
        # Units: f25 and f(T) share units; Ha,Hd [J mol-1]; S [J mol-1 K-1];
        #   Q10 [1]; R [J mol-1 K-1]; T,Tref [K].
        # Domain/constraints: T > 0 K; evaluated in the input temperature dtype
        #   and device; parameter tensors keep their autograd graph through the
        #   dtype alignment instead of narrowing the temperature.
        # Sign convention: none; a positive reference rate stays positive.
        return peaked_temperature_response(
            self._as_tensor_like(self.vpmax25, temperature), temperature,
            *self._params_like(temperature, self.vpmax_ha, self.vpmax_hd, self.vpmax_s),
            self.gas_constant, self.reference_temp
        )
    
    def _temperature_response_kp(self, temperature: Tensor) -> Tensor:
        """
        Temperature response of Michaelis constant for PEP carboxylase.
        
        Uses Arrhenius function (no high-temperature deactivation).
        """
        # References: Collatz et al. 1992; von Caemmerer 2000
        # Mathematical expression: Kp(T) = Kp25 * exp(Ha/R*(1/Tref - 1/T)).
        # Parameter meanings: kp25 [umol mol-1], kp_ha [J mol-1], gas_constant [J mol-1 K-1], reference_temp [K]
        # Units: f25 and f(T) share units; Ha,Hd [J mol-1]; S [J mol-1 K-1];
        #   Q10 [1]; R [J mol-1 K-1]; T,Tref [K].
        # Domain/constraints: T > 0 K; evaluated in the input temperature dtype
        #   and device; parameter tensors keep their autograd graph through the
        #   dtype alignment instead of narrowing the temperature.
        # Sign convention: none; a positive reference rate stays positive.
        return arrhenius_temperature_response(
            self._as_tensor_like(self.kp25, temperature), temperature,
            self._as_tensor_like(self.kp_ha, temperature), self.gas_constant, self.reference_temp
        )
    
    def _temperature_response_ve_max(self, temperature: Tensor) -> Tensor:
        """
        Temperature response of the bundle-sheath Rubisco capacity.
        
        Uses peaked response function.
        """
        # References: Collatz et al. 1992; von Caemmerer 2000
        # Mathematical expression: f(T) = f25 * exp(Ha/R*(1/Tref - 1/T))
        #   * (1 + exp((S*Tref - Hd)/(R*Tref))) / (1 + exp((S*T - Hd)/(R*T))),
        #   so that f(Tref) = f25 exactly (normalized peaked Arrhenius).
        # Parameter meanings: ve_max25 [umol CO2 m-2 s-1], ve_max_ha [J mol-1], ve_max_hd [J mol-1],
        #             ve_max_s [J mol-1 K-1], gas_constant [J mol-1 K-1], reference_temp [K]
        # Units: f25 and f(T) share units; Ha,Hd [J mol-1]; S [J mol-1 K-1];
        #   Q10 [1]; R [J mol-1 K-1]; T,Tref [K].
        # Domain/constraints: T > 0 K; evaluated in the input temperature dtype
        #   and device; parameter tensors keep their autograd graph through the
        #   dtype alignment instead of narrowing the temperature.
        # Sign convention: none; a positive reference rate stays positive.
        return peaked_temperature_response(
            self._as_tensor_like(self.ve_max25, temperature), temperature,
            *self._params_like(temperature, self.ve_max_ha, self.ve_max_hd, self.ve_max_s),
            self.gas_constant, self.reference_temp
        )
    
    def _validate_inputs(
        self,
        air_temperature: Tensor,
        atmospheric_co2_ppm: Tensor,
        ppfd: Tensor,
        vapor_pressure_deficit: Tensor,
        vcmax_scaling: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Validate canonical forcing inputs without clipping or replacement.

        Parameters
        ----------
        air_temperature : Tensor
            Air temperature [K].
        atmospheric_co2_ppm : Tensor
            Atmospheric CO2 mole fraction [umol mol-1].
        ppfd : Tensor
            Photosynthetic photon flux density [umol photons m-2 s-1].
        vapor_pressure_deficit : Tensor
            Vapor pressure deficit [Pa].
        vcmax_scaling : Tensor
            Vcmax scaling factor [dimensionless].
        """
        self._validate_required_finite_inputs(
            air_temperature,
            atmospheric_co2_ppm,
            ppfd,
            vapor_pressure_deficit,
        )
        self._require_finite("vcmax_scaling", vcmax_scaling)

        min_temp = self._as_tensor_like(
            self._param_cache['clamp_min_temperature'], air_temperature
        )
        max_temp = self._as_tensor_like(
            self._param_cache['clamp_max_temperature'], air_temperature
        )
        if torch.any(air_temperature < min_temp) or torch.any(air_temperature > max_temp):
            raise ValueError(
                "air_temperature outside supported range "
                f"[{self._as_float(min_temp)}, {self._as_float(max_temp)}] K"
            )

        min_co2 = self.get_parameter_value(
            'clamp_min_co2',
            namespace="validation.numerical", required=True)
        max_co2 = self.get_parameter_value(
            'clamp_max_co2',
            namespace="validation.numerical", required=True)
        min_co2 = self._as_tensor_like(min_co2, atmospheric_co2_ppm)
        max_co2 = self._as_tensor_like(max_co2, atmospheric_co2_ppm)
        if torch.any(atmospheric_co2_ppm < min_co2) or torch.any(atmospheric_co2_ppm > max_co2):
            raise ValueError(
                "atmospheric_co2_ppm outside supported range "
                f"[{self._as_float(min_co2)}, {self._as_float(max_co2)}] umol mol-1"
            )

        validate_positive_values(ppfd, "ppfd")
        max_ppfd = self.get_parameter_value(
            'max_ppfd_validation',
            namespace="validation.numerical", required=True)
        if torch.any(ppfd > max_ppfd):
            raise ValueError(
                f"ppfd exceeds supported maximum ({max_ppfd} umol photons m-2 s-1)"
            )

        validate_positive_values(vapor_pressure_deficit, "vapor_pressure_deficit")

        vcmax_min_scaling = self._as_tensor_like(
            self._param_cache['vcmax_min_scaling'],
            vcmax_scaling,
        )
        vcmax_max_scaling = self._as_tensor_like(
            self._param_cache['vcmax_max_scaling'],
            vcmax_scaling,
        )
        if torch.any(vcmax_scaling < vcmax_min_scaling) or torch.any(vcmax_scaling > vcmax_max_scaling):
            raise ValueError(
                "vcmax_scaling outside supported range "
                f"[{self._as_float(vcmax_min_scaling)}, "
                f"{self._as_float(vcmax_max_scaling)}]"
            )

        return (
            air_temperature,
            atmospheric_co2_ppm,
            ppfd,
            vapor_pressure_deficit,
            vcmax_scaling,
        )


    def _load_pft_parameters(self, pft_name: str) -> None:
        """Load PFT-specific photosynthesis parameters via ParameterManager."""
        namespace = f"biosphere.pft.{pft_name}.photosynthesis"

        def _get_value(key: str, *, expected_type: Optional[type] = None):
            full_key = f"{namespace}.{key}"
            return self.get_parameter_value(
                full_key,
                source_order=("manager",),
                expected_type=expected_type,
                validate=True,
            )

        def _assign(attr_name: str, value: Optional[Union[Tensor, float, int]]) -> None:
            if value is None:
                return
            if attr_name in ("vcmax25", "jmax25", "rd25"):
                # References: Farquhar et al. (1980); architecture SSOT section 4.2.
                # Mathematical expression: theta_functor = theta_PFT, while
                #   theta_base_after = theta_base_before for each reference capacity.
                # Parameter meanings: theta_PFT is the typed PFT source; theta_base
                #   is the separate base-database value; theta_functor is held here.
                # Units: Vcmax25,Rd25 [umol CO2 m-2 s-1]; Jmax25 [umol e- m-2 s-1].
                # Domain/constraints: use the existing validated getter; change only
                #   the module holding category, preserving source identity and dtype.
                # Sign convention: capacities stay nonnegative; rebinding is not a
                #   flux or state advance and never changes the base source.
                delattr(self, attr_name)
                if isinstance(value, nn.Parameter):
                    setattr(self, attr_name, value)
                else:
                    self.register_buffer(attr_name, value)
                return
            current = getattr(self, attr_name, None)
            if isinstance(current, nn.Parameter):
                tensor_value = self._ensure_tensor(value).to(
                    device=current.device,
                    dtype=current.dtype,
                )
                if tensor_value.shape != current.shape:
                    if tensor_value.numel() == 1:
                        tensor_value = tensor_value.expand_as(current)
                    elif tensor_value.numel() == current.numel():
                        tensor_value = tensor_value.reshape_as(current)
                    else:
                        raise ValueError(
                            f"PFT parameter '{attr_name}' shape {tuple(tensor_value.shape)} "
                            f"cannot update existing shape {tuple(current.shape)}."
                        )
                with torch.no_grad():
                    current.copy_(tensor_value)
            elif isinstance(current, torch.Tensor):
                setattr(self, attr_name, self._ensure_tensor(value))
            else:
                setattr(self, attr_name, value)

        # Required parameters
        required_mapping = {
            "vcmax25": "vcmax25",
            "jmax25": "jmax25",
            "rd25": "rd25",
        }

        namespace_available = any(
            _get_value(param_key) is not None
            for param_key in required_mapping.values()
        )

        missing: List[str] = []
        for attr_name, param_key in required_mapping.items():
            value = _get_value(param_key)
            if value is None:
                missing.append(param_key)
            else:
                _assign(attr_name, value)

        if missing:
            if namespace_available:
                raise ValueError(
                    f"Missing required PFT parameters {missing} in namespace '{namespace}'. "
                    "Please add them to the parameter database (YAML) before running the model."
                )
            logger.debug("No PFT override found for namespace '%s'; using base parameters", namespace)
            return

        # Optional pathway override (string)
        pathway_override = _get_value("pathway", expected_type=str)
        if pathway_override:
            normalized = pathway_override.upper()
            if normalized not in {"C3", "C4"}:
                raise ValueError(
                    f"Invalid pathway '{pathway_override}' for PFT '{pft_name}'. Expected 'C3' or 'C4'."
                )
            if normalized != self.pathway:
                self.pathway = normalized
                if self.pathway == "C4":
                    self._initialize_c4_parameters()

        # Optional scalar overrides shared by C3 and C4
        optional_mapping = {
            "alpha": "alpha",
            "theta": "theta",
            "vcmax_ha": "vcmax_ea",
            "vcmax_hd": "vcmax_hd",
            "vcmax_s": "vcmax_se",
            "jmax_ha": "jmax_ea",
            "jmax_hd": "jmax_hd",
            "jmax_s": "jmax_se",
            "rd_q10": "rd_q10",
        }

        for attr_name, param_key in optional_mapping.items():
            value = _get_value(param_key)
            _assign(attr_name, value)

        if self.pathway == "C4":
            c4_mapping = {
                "vpmax25": "vpmax25",
                "kp25": "kp25",
                "phi": "phi",
                "ve_max25": "ve_max25",
                "vpmax_ha": "ha_vpmax",
                "vpmax_hd": "hd_vpmax",
                "vpmax_s": "s_vpmax",
                "kp_ha": "ha_kp",
                "ve_max_ha": "ha_ve_max",
                "ve_max_hd": "hd_ve_max",
                "ve_max_s": "s_ve_max",
            }
            for attr_name, param_key in c4_mapping.items():
                value = _get_value(param_key)
                _assign(attr_name, value)

        # Refresh cached numerical parameters after overrides
        self._cache_numerical_parameters()

        logger.info("Loaded PFT-specific photosynthesis parameters for %s from %s", pft_name, namespace)


# Export main class
__all__ = [
    'FarquharPhotosynthesis'
]

