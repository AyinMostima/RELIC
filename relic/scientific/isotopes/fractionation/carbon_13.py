"""
Carbon-13 Isotope Fractionation Calculator

This module implements 13C isotope fractionation calculations for carbon
cycle processes including photosynthesis, respiration, and decomposition.
The calculations follow established models from the literature.

Key Features:
- Farquhar et al. (1989) model for C3 photosynthesis
- C4 pathway fractionation (Farquhar, 1983)
- Temperature-dependent respiration fractionation
- Substrate-specific decomposition fractionation
- Integrated with PyTorch for automatic differentiation

Mathematical Framework:
    delta^1^3C = (Rsample/Rstandard - 1) * 1000?
    where Rstandard = ParameterManager('isotopes.standards.carbon_13.pdb_ratio') (PDB standard, validated)

Fractionation Factor Calculation:
    alpha = klight/kheavy = 1/(1 + epsilon/1000)
    where epsilon is the fractionation in per mille

References:
    - Farquhar et al. (1989). Carbon isotope discrimination and photosynthesis
    - Farquhar (1983). On the nature of carbon isotope discrimination in C4 species
    - O'Leary (1988). Carbon isotopes in photosynthesis
    - Tcherkez et al. (2003). Natural 13C abundance in C3 leaves
    - Bowling et al. (2008). Carbon isotopes in respired CO2
"""

import torch
from typing import Dict, Optional, Union
import warnings

from relic.parameters import UnifiedParameterManager

from ..config.multi_isotope_models import (
    CarbonFractionationParameters, PhotosynthesisType, CarbonProcessType
)
from .base import IsotopeFractionationCalculator
class Carbon13FractionationCalculator(IsotopeFractionationCalculator):
    """
    Calculator for 13C isotope fractionation in carbon cycle processes.
    
    Implements process-specific fractionation models for carbon isotope
    discrimination in photosynthesis, respiration, and decomposition.
    
    Attributes:
        pdb_standard: PDB standard ratio (0.0112372)
        supported_processes: Set of supported carbon processes
        
    Methods:
        calculate_photosynthesis_fractionation: C3/C4 photosynthesis discrimination
        calculate_respiration_fractionation: Temperature-dependent respiration
        calculate_decomposition_fractionation: Substrate-specific decomposition
        farquhar_c3_model: Farquhar et al. (1989) C3 photosynthesis model
        c4_fractionation_model: C4 pathway fractionation model
    """
    
    def __init__(
        self,
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        """Initialize Carbon-13 fractionation calculator."""
        super().__init__("13C", parameter_manager=parameter_manager)

        # PDB standard for 13C/12C ratio
        self.pdb_standard = float(
            self.get_parameter_value(
                "pdb_ratio",
                namespace="isotopes.standards.carbon_13",
                required=True)
        )
        
        # Supported carbon processes
        self.supported_processes = {
            CarbonProcessType.PHOTOSYNTHESIS,
            CarbonProcessType.LEAF_RESPIRATION,
            CarbonProcessType.ROOT_RESPIRATION,
            CarbonProcessType.AUTOTROPHIC_RESPIRATION,
            CarbonProcessType.HETEROTROPHIC_RESPIRATION,
            CarbonProcessType.DECOMPOSITION,
            CarbonProcessType.BVOC_EMISSION
        }

        # Carbon fractionation defaults and bounds
        self.temperature_warning_min = float(
            self.get_parameter_value(
                "warning_min",
                namespace="isotopes.temperature",
                required=True)
        )
        self.temperature_warning_max = float(
            self.get_parameter_value(
                "warning_max",
                namespace="isotopes.temperature",
                required=True)
        )
        self.respiration_substrate_sensitivity = float(
            self.get_parameter_value(
                "respiration.substrate_sensitivity",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.respiration_parameters = {
            CarbonProcessType.LEAF_RESPIRATION: {
                "alpha_0": float(
                    self.get_parameter_value(
                        "respiration.leaf.alpha_0",
                        namespace="isotopes.fractionation.carbon",
                        required=True)
                ),
                "q10": float(
                    self.get_parameter_value(
                        "respiration.leaf.q10",
                        namespace="isotopes.fractionation.carbon",
                        required=True)
                ),
            },
            CarbonProcessType.ROOT_RESPIRATION: {
                "alpha_0": float(
                    self.get_parameter_value(
                        "respiration.root.alpha_0",
                        namespace="isotopes.fractionation.carbon",
                        required=True)
                ),
                "q10": float(
                    self.get_parameter_value(
                        "respiration.root.q10",
                        namespace="isotopes.fractionation.carbon",
                        required=True)
                ),
            },
            CarbonProcessType.AUTOTROPHIC_RESPIRATION: {
                "alpha_0": float(
                    self.get_parameter_value(
                        "respiration.autotrophic.alpha_0",
                        namespace="isotopes.fractionation.carbon",
                        required=True)
                ),
                "q10": float(
                    self.get_parameter_value(
                        "respiration.autotrophic.q10",
                        namespace="isotopes.fractionation.carbon",
                        required=True)
                ),
            },
            CarbonProcessType.HETEROTROPHIC_RESPIRATION: {
                "alpha_0": float(
                    self.get_parameter_value(
                        "respiration.heterotrophic.alpha_0",
                        namespace="isotopes.fractionation.carbon",
                        required=True)
                ),
                "q10": float(
                    self.get_parameter_value(
                        "respiration.heterotrophic.q10",
                        namespace="isotopes.fractionation.carbon",
                        required=True)
                ),
            },
        }
        self.decomposition_alpha_0 = float(
            self.get_parameter_value(
                "decomposition.alpha_0",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_q10 = float(
            self.get_parameter_value(
                "decomposition.q10",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_substrate_correction_labile = float(
            self.get_parameter_value(
                "decomposition.substrate_correction.labile",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_substrate_correction_recalcitrant = float(
            self.get_parameter_value(
                "decomposition.substrate_correction.recalcitrant",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_substrate_correction_mixed = float(
            self.get_parameter_value(
                "decomposition.substrate_correction.mixed",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_moisture_optimal = float(
            self.get_parameter_value(
                "decomposition.moisture_optimal",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_moisture_scale = float(
            self.get_parameter_value(
                "decomposition.moisture_scale",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_oxygen_scale = float(
            self.get_parameter_value(
                "decomposition.oxygen_scale",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_oxygen_reference = float(
            self.get_parameter_value(
                "decomposition.oxygen_reference",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_ph_optimal = float(
            self.get_parameter_value(
                "decomposition.ph_optimal",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_ph_scale = float(
            self.get_parameter_value(
                "decomposition.ph_scale",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.decomposition_ph_range = float(
            self.get_parameter_value(
                "decomposition.ph_range",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.bvoc_base_fractionations = {
            "isoprene": float(
                self.get_parameter_value(
                    "bvoc.base_fractionation.isoprene",
                    namespace="isotopes.fractionation.carbon",
                    required=True)
            ),
            "monoterpenes": float(
                self.get_parameter_value(
                    "bvoc.base_fractionation.monoterpenes",
                    namespace="isotopes.fractionation.carbon",
                    required=True)
            ),
            "sesquiterpenes": float(
                self.get_parameter_value(
                    "bvoc.base_fractionation.sesquiterpenes",
                    namespace="isotopes.fractionation.carbon",
                    required=True)
            ),
            "methanol": float(
                self.get_parameter_value(
                    "bvoc.base_fractionation.methanol",
                    namespace="isotopes.fractionation.carbon",
                    required=True)
            ),
            "acetaldehyde": float(
                self.get_parameter_value(
                    "bvoc.base_fractionation.acetaldehyde",
                    namespace="isotopes.fractionation.carbon",
                    required=True)
            ),
            "ethanol": float(
                self.get_parameter_value(
                    "bvoc.base_fractionation.ethanol",
                    namespace="isotopes.fractionation.carbon",
                    required=True)
            ),
        }
        self.bvoc_q10 = float(
            self.get_parameter_value(
                "bvoc.q10",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.bvoc_reference_temperature = float(
            self.get_parameter_value(
                "bvoc.reference_temperature",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.bvoc_par_optimal = float(
            self.get_parameter_value(
                "bvoc.par_optimal",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.bvoc_light_scale = float(
            self.get_parameter_value(
                "bvoc.light_scale",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.bvoc_substrate_correction_recent = float(
            self.get_parameter_value(
                "bvoc.substrate_correction.recent",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.bvoc_substrate_correction_stored = float(
            self.get_parameter_value(
                "bvoc.substrate_correction.stored",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        self.bvoc_substrate_correction_mixed = float(
            self.get_parameter_value(
                "bvoc.substrate_correction.mixed",
                namespace="isotopes.fractionation.carbon",
                required=True)
        )
        
        # Physical constants - use from base class
        # Gas constant is now loaded from ParameterManager  # J/(mol?K) from base class
    
    def _reference_temperature_tensor(self, temperature: torch.Tensor) -> torch.Tensor:
        """Return the sole ParameterManager-backed isotope reference temperature."""
        return torch.as_tensor(
            self.reference_temperature,
            device=temperature.device,
            dtype=temperature.dtype,
        )

    # References: Farquhar et al. (1989); O'Leary (1988).
    # Formula: alpha = f(process_type, T, environment, substrate).
    # Parameters: process_type, temperature (K), carbon_params (per mil coefficients).
    def calculate_fractionation_factor(
        self,
        process_type: Union[str, CarbonProcessType],
        temperature: torch.Tensor,
        carbon_params: Optional[CarbonFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 13C fractionation factor for carbon cycle processes.
        
        Process-specific calculations:
        - Photosynthesis: Farquhar model with ci/ca ratio dependence
        - Respiration: Temperature-dependent with Q10 or Arrhenius
        - Decomposition: Substrate and temperature dependent
        
        Parameters:
            process_type: Carbon cycle process type
            temperature: Temperature [K], shape [..., 1]
            carbon_params: Carbon-specific fractionation parameters
            environmental_state: Environmental variables (CO2, VPD, etc.)
            **kwargs: Additional process-specific parameters
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
            
        Mathematical Expression:
            alpha = f(T, environment, substrate) depending on process
            
        Raises:
            ValueError: For unsupported process types
            RuntimeError: For invalid parameter combinations
        """
        # Convert string to enum if needed
        if isinstance(process_type, str):
            try:
                process_type = CarbonProcessType(process_type.lower())
            except ValueError:
                raise ValueError(f"Unsupported carbon process: {process_type}")
        
        # Validate process type
        if process_type not in self.supported_processes:
            raise ValueError(f"Process {process_type} not supported for 13C fractionation")
        
        # Ensure temperature is valid
        temperature = self._validate_temperature(temperature)
        
        # Process-specific fractionation calculations
        if process_type == CarbonProcessType.PHOTOSYNTHESIS:
            return self.calculate_photosynthesis_fractionation(
                temperature, carbon_params, environmental_state, **kwargs
            )
            
        elif process_type in {
            CarbonProcessType.LEAF_RESPIRATION,
            CarbonProcessType.ROOT_RESPIRATION,
            CarbonProcessType.AUTOTROPHIC_RESPIRATION,
            CarbonProcessType.HETEROTROPHIC_RESPIRATION
        }:
            return self.calculate_respiration_fractionation(
                process_type, temperature, carbon_params, **kwargs
            )
            
        elif process_type == CarbonProcessType.DECOMPOSITION:
            return self.calculate_decomposition_fractionation(
                temperature, carbon_params, environmental_state, **kwargs
            )
            
        elif process_type == CarbonProcessType.BVOC_EMISSION:
            return self.calculate_bvoc_fractionation(
                temperature, carbon_params, environmental_state, **kwargs
            )
            
        else:
            raise ValueError(f"No fractionation model implemented for {process_type}")
    
    # References: Farquhar et al. (1989); Wei et al. (2014).
    # Formula: epsilon = a + (b - a) * (ci/ca) for C3; CAM uses weighted C3/C4.
    # Parameters: diffusion_fractionation (a), carboxylation_fractionation (b),
    #             ci/ca is an input of the call, validated on [0, 1].
    def calculate_photosynthesis_fractionation(
        self,
        temperature: torch.Tensor,
        carbon_params: Optional[CarbonFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 13C fractionation for photosynthesis.
        
        Implements the Farquhar et al. (1989) model for C3 photosynthesis
        and C4-specific modifications:
        
        C3: delta13C_leaf = delta13C_air - a - (b-a)(ci/ca)
        C4: Farquhar (1983) law with PEP carboxylation and bundle-sheath leakiness
        
        Parameters:
            temperature: Temperature [K], shape [..., 1]
            carbon_params: Carbon fractionation parameters
            environmental_state: Environmental conditions (ci, ca, par)
            ci_ca_ratio (keyword): ci/ca [1] when environmental_state has no ci and ca
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
            
        Mathematical Expression:
            C3: epsilon = a + (b - a)(ci/ca)
            C4: epsilon = a + (b4 + b3 phi - a)(ci/ca)  (Farquhar et al. 1989, Eq. 10)
            where a = diffusion, b = net C3 carboxylation, b4 = net PEP carboxylation
            of gaseous CO2, b3 = Rubisco, phi = bundle-sheath leakiness
        """
        # Parameter set of the calculator's ParameterManager when the call supplies none
        if carbon_params is None:
            carbon_params = CarbonFractionationParameters.from_manager(self.parameter_manager)

        # References: Farquhar, O'Leary and Berry (1982); Farquhar et al. (1989).
        # Mathematical expression: r = c_i / c_a.
        # Parameter meanings: c_i intercellular CO2, c_a ambient CO2, r their ratio.
        # Units: c_i and c_a in one CO2 unit (for example umol mol-1); r [1].
        # Domain/constraints: r is a physiological state of the leaf, not a parameter.
        #   Exactly one source is given: the ci_ca_ratio argument or the ci and ca
        #   entries of environmental_state; none or both raise ValueError, and no
        #   fixed ratio is substituted. RELIC has no stomatal law supplying r here.
        # Sign convention: none (ratio of two non-negative amounts).
        has_gases = (
            environmental_state is not None
            and 'ci' in environmental_state
            and 'ca' in environmental_state
        )
        has_ratio = 'ci_ca_ratio' in kwargs
        if has_gases == has_ratio:
            raise ValueError(
                "C13 photosynthesis requires exactly one ci/ca source: the ci_ca_ratio "
                "argument or the ci and ca entries of environmental_state"
            )
        if has_gases:
            ci_ca_ratio = environmental_state['ci'] / environmental_state['ca']
        else:
            ci_ca_ratio = kwargs['ci_ca_ratio']
        
        if not isinstance(ci_ca_ratio, torch.Tensor):
            ci_ca_ratio = torch.tensor(ci_ca_ratio, dtype=temperature.dtype, device=temperature.device)

        # References: Farquhar, O'Leary and Berry (1982); Farquhar et al. (1989).
        # Mathematical expression: 0 <= r = c_i / c_a <= 1.
        # Parameter meanings: r is the intercellular-to-ambient CO2 ratio of a leaf with
        #   net CO2 uptake (c_i <= c_a).
        # Units: r [1].
        # Domain/constraints: a non-finite r or r outside [0, 1] raises ValueError; r is
        #   not clipped, and no numerical-stability range narrows the physical domain.
        # Sign convention: none (ratio of two non-negative amounts).
        if not bool(torch.all(torch.isfinite(ci_ca_ratio))) or bool(
            torch.any((ci_ca_ratio < 0.0) | (ci_ca_ratio > 1.0))
        ):
            raise ValueError(
                "ci/ca must be a finite ratio in [0, 1] (c_i <= c_a during net CO2 uptake); "
                f"got {ci_ca_ratio}"
            )
        
        # Calculate fractionation based on photosynthesis type
        if carbon_params.photosynthesis_type == PhotosynthesisType.C3:
            epsilon = self.farquhar_c3_model(carbon_params, ci_ca_ratio)
            
        elif carbon_params.photosynthesis_type == PhotosynthesisType.C4:
            epsilon = self.c4_fractionation_model(carbon_params, ci_ca_ratio)
            
        elif carbon_params.photosynthesis_type == PhotosynthesisType.CAM:
            raise ValueError(
                "CAM fractionation is not implemented as a validated canonical process law"
            )
            
        else:
            raise ValueError(f"Unknown photosynthesis type: {carbon_params.photosynthesis_type}")
        
        # Convert discrimination to fractionation factor
        # alpha = 1 / (1 + epsilon/1000)
        alpha = 1.0 / (1.0 + epsilon / 1000.0)
        
        return alpha
    
    def farquhar_c3_model(
        self,
        carbon_params: CarbonFractionationParameters,
        ci_ca_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Farquhar et al. (1982, 1989) discrimination of C3 photosynthesis.

        Temperature, light and the vapour pressure deficit act on the discrimination
        only through ci/ca, which the caller supplies; no multiplier of epsilon applies.

        Parameters:
            carbon_params: Carbon fractionation parameters (a, b)
            ci_ca_ratio: Intercellular to ambient CO2 ratio in [0, 1]

        Returns:
            Discrimination epsilon [per mil]
        """
        # References: Farquhar, O'Leary and Berry (1982), Aust. J. Plant Physiol. 9, 121-137;
        #   Farquhar, Ehleringer and Hubick (1989), Annu. Rev. Plant Physiol. Plant Mol.
        #   Biol. 40, 503-537; CLM5 Technical Note (alpha_psn = 1 + (4.4 + 22.6 ci/ca)/1000).
        # Mathematical expression: epsilon = a + (b - a) r, r = c_i / c_a.
        # Parameter meanings: a diffusion discrimination in air, b Rubisco carboxylation
        #   discrimination, r the intercellular-to-ambient CO2 ratio.
        # Units: a, b, epsilon [per mil]; r [1].
        # Domain/constraints: r in [0, 1] (validated by the caller); no Q10, light or VPD
        #   multiplier exists in the law, so none is applied.
        # Sign convention: epsilon > 0 is discrimination against 13C.
        a = carbon_params.diffusion_fractionation
        b = carbon_params.carboxylation_fractionation
        return a + (b - a) * ci_ca_ratio
    
    def c4_fractionation_model(
        self,
        carbon_params: CarbonFractionationParameters,
        ci_ca_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """
        C4 photosynthesis discrimination of Farquhar et al. (1989), Eq. (10).

        Temperature acts only through ci/ca; no temperature multiplier of epsilon applies.

        Parameters:
            carbon_params: Carbon fractionation parameters (a, b4, b3, phi)
            ci_ca_ratio: Intercellular to ambient CO2 ratio in [0, 1]

        Returns:
            Discrimination epsilon [per mil]
        """
        # References: Farquhar, Ehleringer and Hubick (1989), Annu. Rev. Plant Physiol.
        #   Plant Mol. Biol. 40, 503-537, Eq. (10) and Table 1; Farquhar (1983), Aust. J.
        #   Plant Physiol. 10, 205-226; measured leakiness of Henderson et al. (1992).
        # Mathematical expression: epsilon = a (1 - r) + (b4 + b3 phi) r
        #   = a + (b4 + b3 phi - a) r, r = c_i / c_a.
        # Parameter meanings: a diffusion discrimination in air; b4 net discrimination
        #   of the fixation of gaseous CO2 (in equilibrium with HCO3-) by PEP
        #   carboxylase; b3 Rubisco discrimination in the bundle sheath; phi the
        #   bundle-sheath leakiness, the proportion of the PEP-fixed carbon that leaks
        #   out of the bundle sheath; r the intercellular-to-ambient CO2 ratio.
        # Units: a, b4, b3, epsilon [per mil]; phi, r [1].
        # Domain/constraints: r in [0, 1] (validated by the caller); phi within its
        #   measured range [0.08, 0.5] (Kromdijk et al. 2014, Table 1); b4 is its
        #   25 degC value (the temperature dependence of the CO2-HCO3- equilibrium is
        #   not carried); Eq. (10) has no leakage fractionation s, and none is
        #   applied (-s phi r is about -0.14 per mil at r = 0.4); no clamp.
        # Sign convention: epsilon > 0 is discrimination against 13C; b4 < 0 because
        #   13C concentrates in HCO3- at equilibrium.
        a = carbon_params.diffusion_fractionation
        b4 = carbon_params.c4_pep_net_fractionation
        b3 = carbon_params.c4_rubisco_fractionation
        phi = carbon_params.c4_bundle_sheath_leakiness
        return a + (b4 + b3 * phi - a) * ci_ca_ratio
    
    # References: Bowling et al. (2008); Tcherkez et al. (2003).
    # Formula: alpha(T) = alpha_0 * Q10^((T - T_ref)/10).
    # Parameters: respiration.* (alpha_0, q10), temperature.reference.
    def calculate_respiration_fractionation(
        self,
        respiration_type: CarbonProcessType,
        temperature: torch.Tensor,
        carbon_params: Optional[CarbonFractionationParameters] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 13C fractionation for respiration processes.
        
        Respiration fractionation depends on:
        - Substrate composition and 13C content
        - Temperature (canonical Q10 response)
        - Metabolic pathway (glycolysis vs. TCA cycle)
        - Tissue type (leaf vs. root vs. soil microbes)
        
        Parameters:
            respiration_type: Type of respiration process
            temperature: Temperature [K], shape [..., 1]
            carbon_params: Carbon fractionation parameters
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
            
        Mathematical Expression:
            alpha(T) = alpha_0 * Q_1_0^((T-T_ref)/10)
        """
        # Parameters of the respiration type (ParameterManager); a call may pass
        # alpha_0 or q10 explicitly. carbon_params holds the photosynthetic
        # discrimination parameters and does not enter the respiration law.
        if respiration_type not in self.respiration_parameters:
            raise ValueError(f"No parameters for respiration type: {respiration_type}")
        params = self.respiration_parameters[respiration_type]
        alpha_0 = kwargs.get('alpha_0', params['alpha_0'])
        q10 = kwargs.get('q10', params['q10'])
        
        if "model_type" in kwargs or "activation_energy" in kwargs:
            raise ValueError(
                "C13 respiration uses the canonical Q10 law; alternate runtime models are not supported"
            )
        if "T_ref" in kwargs:
            raise ValueError(
                "C13 respiration reference temperature is ParameterManager-owned"
            )
        T_ref = self._reference_temperature_tensor(temperature)
        alpha_0_tensor = torch.as_tensor(alpha_0, device=temperature.device, dtype=temperature.dtype)
        q10_tensor = torch.as_tensor(q10, device=temperature.device, dtype=temperature.dtype)
        alpha = alpha_0_tensor * torch.pow(q10_tensor, (temperature - T_ref) / 10.0)

        # Substrate-specific correction
        substrate_c13 = kwargs.get('substrate_delta_13c', None)
        if substrate_c13 is not None:
            # More enriched substrates show enhanced fractionation
            substrate_correction = 1.0 + self.respiration_substrate_sensitivity * substrate_c13
            alpha = alpha * substrate_correction
        return torch.clamp(
            alpha,
            min=self.min_fractionation_factor,
            max=self.max_fractionation_factor,
        )
    
    # References: Tcherkez et al. (2003); Manzoni and Porporato (2009).
    # Formula: alpha = alpha_0 * Q10^((T - T_ref)/10) * f_substrate * f_environment.
    # Parameters: decomposition.* (alpha_0, q10, substrate_corrections, moisture/oxygen/pH modifiers).
    def calculate_decomposition_fractionation(
        self,
        temperature: torch.Tensor,
        carbon_params: Optional[CarbonFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 13C fractionation for decomposition processes.
        
        Decomposition fractionation depends on:
        - Substrate quality (lignin vs. cellulose vs. labile compounds)
        - Microbial community composition
        - Temperature and moisture conditions
        - Oxygen availability (aerobic vs. anaerobic)
        
        Parameters:
            temperature: Temperature [K], shape [..., 1]
            carbon_params: Carbon fractionation parameters  
            environmental_state: Environmental conditions
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
            
        Mathematical Expression:
            alpha = alpha_base * f_substrate * f_environment * f_temperature
        """
        # Base decomposition fractionation
        alpha_base = kwargs.get('alpha_0', self.decomposition_alpha_0)
        
        # Temperature dependence
        q10 = kwargs.get('q10', self.decomposition_q10)
        if "T_ref" in kwargs:
            raise ValueError(
                "C13 decomposition reference temperature is ParameterManager-owned"
            )
        T_ref = self._reference_temperature_tensor(temperature)
        alpha_base_tensor = torch.as_tensor(alpha_base, device=temperature.device, dtype=temperature.dtype)
        q10_tensor = torch.as_tensor(q10, device=temperature.device, dtype=temperature.dtype)
        temp_factor = torch.pow(q10_tensor, (temperature - T_ref) / 10.0)
        
        # Substrate quality: a required input of the call (labile, recalcitrant or
        # mixed) selecting the substrate correction; an absent or unknown quality
        # raises, none is substituted.
        substrate_corrections = {
            'labile': self.decomposition_substrate_correction_labile,
            'recalcitrant': self.decomposition_substrate_correction_recalcitrant,
            'mixed': self.decomposition_substrate_correction_mixed,
        }
        substrate_quality = kwargs.get('substrate_quality')
        if substrate_quality not in substrate_corrections:
            raise ValueError(
                "C13 decomposition requires substrate_quality in "
                f"{sorted(substrate_corrections)}, received {substrate_quality!r}"
            )
        substrate_correction = substrate_corrections[substrate_quality]
        
        # Environmental modifiers
        environmental_factor = 1.0
        
        if environmental_state:
            # Moisture effects
            if 'soil_moisture' in environmental_state:
                moisture = environmental_state['soil_moisture']
                # Optimal moisture around 0.6 (field capacity)
                moisture_optimal = self.decomposition_moisture_optimal
                moisture_stress = torch.abs(moisture - moisture_optimal) / moisture_optimal
                moisture_correction = 1.0 + self.decomposition_moisture_scale * moisture_stress
                environmental_factor *= moisture_correction
            
            # Oxygen effects (aerobic vs. anaerobic)
            if 'oxygen_concentration' in environmental_state:
                oxygen = environmental_state['oxygen_concentration']
                # Lower oxygen enhances fractionation (anaerobic pathways)
                oxygen_correction = 1.0 + self.decomposition_oxygen_scale * torch.exp(
                    -oxygen / self.decomposition_oxygen_reference
                )
                environmental_factor *= oxygen_correction
            
            # pH effects
            if 'soil_ph' in environmental_state:
                ph = environmental_state['soil_ph']
                # Optimal pH around 6.5-7.0
                ph_optimal = self.decomposition_ph_optimal
                ph_stress = torch.abs(ph - ph_optimal) / self.decomposition_ph_range
                ph_correction = 1.0 + self.decomposition_ph_scale * ph_stress
                environmental_factor *= ph_correction
        
        # Calculate final fractionation factor
        alpha = alpha_base_tensor * temp_factor * substrate_correction * environmental_factor
        alpha = torch.clamp(
            alpha,
            min=self.min_fractionation_factor,
            max=self.max_fractionation_factor,
        )

        return alpha
    
    # References: Affek and Yakir (2003); Jardine et al. (2009).
    # Formula: alpha = alpha_base * Q10^((T - T_ref)/10) * light_factor * substrate_correction.
    # Parameters: bvoc.* (base_fractionations, q10, reference_temperature, par_optimal, light_scale).
    def calculate_bvoc_fractionation(
        self,
        temperature: torch.Tensor,
        carbon_params: Optional[CarbonFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 13C fractionation for BVOC (Biogenic Volatile Organic Compound) emissions.
        
        BVOC fractionation is generally small but depends on:
        - Compound type (isoprene vs. monoterpenes vs. sesquiterpenes)
        - Biosynthetic pathway (MEP vs. MVA pathways)
        - Temperature and light conditions
        - Substrate (recent photosynthate vs. stored reserves)
        
        Parameters:
            temperature: Temperature [K], shape [..., 1]
            carbon_params: Carbon fractionation parameters
            environmental_state: Environmental conditions
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
            
        References:
            - Affek & Yakir (2003). Natural abundance carbon isotope composition of isoprene
            - Jardine et al. (2009). Emissions and sources of BVOC isotopes
        """
        # BVOC compound: a required input of the call, one of the compounds with a
        # base fractionation in ParameterManager; no compound is substituted.
        bvoc_type = kwargs.get('bvoc_type')
        if bvoc_type not in self.bvoc_base_fractionations:
            raise ValueError(
                "C13 BVOC fractionation requires bvoc_type in "
                f"{sorted(self.bvoc_base_fractionations)}, received {bvoc_type!r}"
            )
        alpha_base = self.bvoc_base_fractionations[bvoc_type]
        
        # Temperature dependence (Q10 model)
        q10 = kwargs.get('q10', self.bvoc_q10)
        T_ref = kwargs.get('T_ref', self.bvoc_reference_temperature)
        temp_factor = q10 ** ((temperature - T_ref) / 10.0)
        
        # Light dependence for light-dependent emissions (e.g., isoprene)
        light_factor = 1.0
        if (environmental_state and 'par' in environmental_state and 
            bvoc_type in ['isoprene']):  # Light-dependent compounds
            par = environmental_state['par']
            par_optimal = self.bvoc_par_optimal
            
            # Light saturation curve
            light_saturation = par / (par + par_optimal)
            # Higher light slightly reduces fractionation (more efficient synthesis)
            light_factor = 1.0 - self.bvoc_light_scale * light_saturation
        
        # Substrate source: a required input of the call (recent photosynthate,
        # stored reserves or mixed); an absent or unknown source raises.
        substrate_corrections = {
            'recent': self.bvoc_substrate_correction_recent,
            'stored': self.bvoc_substrate_correction_stored,
            'mixed': self.bvoc_substrate_correction_mixed,
        }
        substrate_source = kwargs.get('substrate_source')
        if substrate_source not in substrate_corrections:
            raise ValueError(
                "C13 BVOC fractionation requires substrate_source in "
                f"{sorted(substrate_corrections)}, received {substrate_source!r}"
            )
        substrate_correction = substrate_corrections[substrate_source]
        
        # Calculate final fractionation factor
        alpha = alpha_base * temp_factor * light_factor * substrate_correction
        
        return alpha
    
    # References: Coplen et al. (2006).
    # Formula: delta = (R_sample / R_standard - 1) * 1000.
    # Parameters: pdb_standard (ratio), isotope_ratio (ratio).
    def delta_notation(self, isotope_ratio: torch.Tensor) -> torch.Tensor:
        """
        Convert isotope ratio to delta notation.
        
        delta^1^3C = (Rsample/RPDB - 1) * 1000?
        
        Parameters:
            isotope_ratio: 13C/12C ratio [dimensionless]
            
        Returns:
            Delta value delta^1^3C [?]
        """
        return (isotope_ratio / self.pdb_standard - 1.0) * 1000.0
    
    # References: Coplen et al. (2006).
    # Formula: R_sample = R_standard * (delta / 1000 + 1).
    # Parameters: pdb_standard (ratio), delta_value (per mil).
    def ratio_from_delta(self, delta_value: torch.Tensor) -> torch.Tensor:
        """
        Convert delta notation to isotope ratio.
        
        Rsample = RPDB * (delta^1^3C/1000 + 1)
        
        Parameters:
            delta_value: Delta value delta^1^3C [?]
            
        Returns:
            Isotope ratio 13C/12C [dimensionless]
        """
        return self.pdb_standard * (delta_value / 1000.0 + 1.0)
    
    def _validate_temperature(self, temperature: torch.Tensor) -> torch.Tensor:
        """Validate temperature tensor and apply reasonable bounds."""
        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor(temperature, dtype=torch.float32)
        
        # Check for reasonable temperature range (warnings only)
        if (
            torch.any(temperature < self.temperature_warning_min)
            or torch.any(temperature > self.temperature_warning_max)
        ):
            warnings.warn(
                "Temperature outside typical range. "
                "Fractionation calculations may be unreliable."
            )
        
        # Clamp to prevent numerical issues
        min_temp, max_temp = self.temperature_bounds
        temperature = torch.clamp(temperature, min=min_temp, max=max_temp)
        
        return temperature

# Export the calculator
__all__ = ['Carbon13FractionationCalculator']

