"""
Sesquiterpene Synthesis Pathway Model for RELIC

This module implements RELIC's reduced, mass-normalized sesquiterpene closure.
It uses MEGAN-family instantaneous activity factors but is not the canonical
area-normalized MEGAN2.1 canopy formulation.

Sesquiterpenes (C15H24) are complex terpenes often associated with:
- Stress responses (biotic and abiotic)
- Defense mechanisms
- Higher temperature sensitivity than monoterpenes
- Wood formation and resin production

References
----------
- Dudareva, N., et al. (2006). Plant volatiles: recent advances and future
  perspectives. Critical Reviews in Plant Sciences 25, 417-440
- Copolovici, L., et al. (2012). Emissions of green leaf volatiles and terpenoids
  from Solanum lycopersicum are quantitatively related to the severity of cold
  and heat shock treatments. Journal of Plant Physiology 169, 664-672
- Niinemets, U., et al. (2014). Bidirectional exchange of biogenic volatiles
  with vegetation: emission sources, reactions, breakdown and deposition.
  Plant, Cell & Environment 37, 1790-1809.

Extension Notes
---------------
This module can be extended to include:
- Farnesyl diphosphate synthesis pathway
- Stress-induced sesquiterpene production
- Temperature-dependent emission kinetics
- Defense response integration
"""

from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

# Import from core framework
from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
)
from relic.core.functor import BiologicalProcessFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter
from relic.scientific.biosphere.bvoc.responses import BVOCResponseIntegrator
from relic.computation.operators.iphs import advance_reaction_network
from relic.utils.physical_constants import get_physical_constant


class SesquiterpeneSynthesis(BiologicalProcessFunctor):
    """
    Sesquiterpene synthesis model with MEGAN-style activity factors.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    
    def __init__(
        self,
        name: str = "sesquiterpene_synthesis",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize sesquiterpene synthesis model."""
        config = config or {}

        metadata = FunctorMetadata(
            name=name,
            version="2.0.0",
            description="Sesquiterpene synthesis with MEGAN-style temperature/light/stress responses.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="""
            gamma_T = (1 - LDF_SQT) * exp(beta_SQT * (T - T_s))
                      + LDF_SQT * E_opt * C_T2 * exp(C_T1 * x)
                      / (C_T2 - C_T1 * (1 - exp(C_T2 * x))),
                      x = (1/T_opt - 1/T) / R   (Guenther et al. 2012, Eq. 7)
            gamma_L = (1 - LDF_SQT) + LDF_SQT * alpha * C_L * PPFD
                      / sqrt(1 + alpha^2 * PPFD^2)
            gamma_stress = min(supplied stress factors in [0, 1])
            gamma_total = gamma_T * gamma_L * gamma_stress
            E_leaf = EF * gamma_total
            rho_eff = LAI * k_f
            E_area = E_leaf * LAI * k_f / seconds_per_hour
            E_umol = E_area / MW_SQT
            E_mol = UnitConverter(E_umol, umol m-2 s-1 -> mol m-2 s-1)
            F_C_alloc = UnitConverter(E_mol * MW_SQT * f_C,
                                      g C m-2 s-1 -> kg C m-2 s-1)

            Where:
            - EF: Base emission factor [ug g-1 DW h-1]
            - T: Canopy temperature [K]
            - beta_SQT: Sesquiterpene temperature coefficient [K-1]
            - T_s: MEGAN standard temperature [K]
            - E_opt, C_T1: Table 4 'Other Sesquiterpenes' C_eo [1] and C_T1 [J mol-1]
            - T_opt [K], C_T2 [J mol-1]: class-independent Eq. (8) coefficients;
              R: molar gas constant [J mol-1 K-1]
            - LDF_SQT: Light-dependent fraction [1] (Guenther et al. 2012,
              Table 4)
            - alpha: PPFD response slope [m2 s umol-1]; C_L: light scale [1]
            - PPFD: Photosynthetic photon flux density [umol photons m-2 s-1]
            - gamma_T, gamma_L, gamma_stress: Dimensionless activity factors
            - LAI: Canonical RELIC leaf area index [m2 leaf m-2 ground]
            - k_f: Effective dry-leaf mass per canonical RELIC LAI area
              [g DW m-2 leaf]
            - rho_eff: Reduced dry-leaf mass per ground area [g DW m-2 ground]
            - MW_SQT: Sesquiterpene molecular weight [g mol-1]
            - E_area: Intermediate atmospheric mass flux [ug m-2 s-1]
            - E_umol: Diagnostic molar flux [umol m-2 s-1]
            - E_mol: Canonical atmospheric molar flux [mol m-2 s-1]
            - f_C: Sesquiterpene carbon mass fraction [1]
            - F_C_alloc: Reduced plant-carbon allocation proxy [kg C m-2 s-1]
            """,
            references=[
                "Guenther et al. (2012). MEGAN2.1. Geosci. Model Dev. 5, 1471-1492.",
                "Duhl et al. (2008). Sesquiterpene emissions from vegetation: a review."
            ],
            parameters={
                'biosphere.bvoc.default.sesquiterpenes': {
                    'description': (
                        'Base sesquiterpene emission factor at the MEGAN standard temperature '
                        'T_s'
                    ),
                    'units': 'ug g-1 DW h-1',
                },
                'biosphere.bvoc.megan.beta_sesquiterpenes': {
                    'description': 'Temperature coefficient beta of the delegated Eq. (11) term',
                    'units': 'K-1',
                },
                'biosphere.bvoc.molecular_weights.sesquiterpenes': {
                    'description': 'Molecular weight of sesquiterpenes (C15H24)',
                    'units': 'g mol-1',
                },
                'biosphere.bvoc.carbon_fraction.sesquiterpenes': {
                    'description': 'Carbon mass fraction of representative C15H24 sesquiterpenes',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.foliar_density_factor': {
                    'description': (
                        'Effective dry-leaf mass per unit canonical RELIC LAI leaf area'
                    ),
                    'units': 'g DW m-2',
                },
                'biosphere.bvoc.t_standard': {
                    'description': (
                        'MEGAN standard temperature T_s of the delegated Eq. (11) term'
                    ),
                    'units': 'K',
                },
                'biosphere.bvoc.megan.alpha_light': {
                    'description': (
                        'PPFD response slope alpha of the delegated Eq. (4) light activity'
                    ),
                    'units': 'm2 s umol-1',
                },
                'biosphere.bvoc.megan.cl_light': {
                    'description': (
                        'Light scaling coefficient C_L of the delegated Eq. (4) light activity'
                    ),
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.megan.ldf_sesquiterpenes': {
                    'description': (
                        'Sesquiterpene light-dependent fraction LDF of the delegated Eq. (3) '
                        'and Eq. (7) blends'
                    ),
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.megan.eopt_sesquiterpenes': {
                    'description': (
                        'Optimum activity E_opt (Table 4 C_eo) of the delegated Eq. (8) '
                        'temperature activity'
                    ),
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.megan.ct1_sesquiterpenes': {
                    'description': (
                        'Activation energy C_T1 of the delegated Eq. (8) temperature activity'
                    ),
                    'units': 'J mol-1',
                },
                'biosphere.bvoc.megan.topt': {
                    'description': (
                        'Class-independent optimum temperature T_opt of the delegated '
                        'Eq. (8) temperature activity'
                    ),
                    'units': 'K',
                },
                'biosphere.bvoc.megan.ct2': {
                    'description': (
                        'Class-independent deactivation energy C_T2 of the delegated '
                        'Eq. (8) temperature activity'
                    ),
                    'units': 'J mol-1',
                },
                'relic.physical.constants.universal.gas_constant': {
                    'description': (
                        'Molar gas constant R of the delegated Eq. (8) temperature activity'
                    ),
                    'units': 'J mol-1 K-1',
                },
            },
            # This law audits the open species-to-carbon boundary mapping. It
            # does not claim a closed local carbon storage ledger.
            conservation_laws=["carbon_boundary"],
            input_variables=[
                "temperature",
                "leaf_area_index",
                "ppfd",
                "soil_moisture",
                "water_stress_factor",
                "temperature_stress_factor",
                "nutrient_stress_factor",
            ],
            output_variables=[
                "sesquiterpenes_emission",
                "bvoc_carbon_allocation_flux",
                "diagnostics.bvoc.sesquiterpene.emission_rate_molar",
                "diagnostics.bvoc.sesquiterpene.gamma_temperature",
                "diagnostics.bvoc.sesquiterpene.gamma_light",
                "diagnostics.bvoc.sesquiterpene.gamma_stress",
                "diagnostics.bvoc.sesquiterpene.gamma_total",
            ],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )

        resolved_device = device if device is not None else torch.device("cpu")
        declared_parameters = dict(metadata.parameters)
        declared_references = list(metadata.references)

        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=resolved_device,
        )
        # The bridge adds its generic response parameters and references (Q10,
        # Arrhenius, optimum, moisture and pH shapes); this process executes only the
        # MEGAN2.1 Eq. (7) activity of temperature_function and declares only its own laws.
        metadata.parameters = declared_parameters
        metadata.references = declared_references

        self.optional_input_variables = {
            "ppfd",
            "soil_moisture",
            "water_stress_factor",
            "temperature_stress_factor",
            "nutrient_stress_factor",
        }

        self.device = resolved_device
        def _require(name: str, namespace: str) -> Tensor:
            value = self.get_parameter_value(name, namespace=namespace)
            if value is None:
                full_key = f"{namespace}.{name}"
                raise KeyError(f"Missing required parameter '{full_key}'")
            tensor_value = self._ensure_tensor(value)
            if isinstance(tensor_value, torch.nn.Parameter):
                return tensor_value.detach().clone()
            return tensor_value

        self.process_name = name

        # Get the model-specific emission factor from the parameter database.
        base_factor_mass = _require("default.sesquiterpenes", "biosphere.bvoc")
        self.register_buffer('base_emission_factor', base_factor_mass)

        sesquiterpene_mw = _require("molecular_weights.sesquiterpenes", "biosphere.bvoc")
        self.register_buffer('sesquiterpene_mw', sesquiterpene_mw)
        carbon_fraction_sesquiterpenes = _require(
            "carbon_fraction.sesquiterpenes",
            "biosphere.bvoc",
        )
        self.register_buffer(
            "carbon_fraction_sesquiterpenes",
            carbon_fraction_sesquiterpenes,
        )
        foliar_density_factor = _require("foliar_density_factor", "biosphere.bvoc")
        self.register_buffer('foliar_density_factor', foliar_density_factor)
        seconds_per_hour = get_physical_constant(
            "seconds_per_hour",
            as_tensor=True,
            device=resolved_device,
        )
        self.register_buffer('seconds_per_hour', seconds_per_hour)
        self.response_integrator = BVOCResponseIntegrator(
            device=self.device,
            parameter_manager=self.get_parameter_manager(),
        )
        # Align model device with actual buffer placement (captures explicit CUDA index)
        self.device = self.base_emission_factor.device
        self._validated_emission_factor()
        
    def get_dependencies(self) -> list:
        """Declare the stress Functors as optional same-step providers of the contracted
        factor inputs water_stress_factor (water_stress), temperature_stress_factor
        (thermal_stress) and nutrient_stress_factor (nutrient_stress); without them the
        committed or caller-supplied values are read, or the factor is absent."""
        return self._merge_declared_dependencies(
            {"optional": ["water_stress", "thermal_stress", "nutrient_stress"]}
        )

    def get_provides(self) -> list:
        """Get the canonical molar flux and namespaced diagnostics.
        
        Returns
        -------
        list
            Output variable names declared by FunctorMetadata.
        """
        return list(self.metadata.output_variables)

    def temperature_function(self, temperature: Tensor) -> Tensor:
        """Executed sesquiterpene temperature activity, Guenther et al. (2012) Eq. (7).

        The generic bridge responses (Q10, Arrhenius, optimum) are not used.
        """
        # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eqs. (7), (8),
        #   (11) and Table 4 (Other Sesquiterpenes: beta 0.17 K-1, LDF 0.5, C_T1 130 kJ
        #   mol-1, C_eo 2.37).
        # Mathematical expression: gamma_T = (1 - LDF) exp(beta_SQT (T - T_s)) + LDF E_opt
        #   C_T2 exp(C_T1 x) / (C_T2 - C_T1 (1 - exp(C_T2 x))), x = (1/T_opt - 1/T)/R,
        #   evaluated by the shared megan_temperature_activity law through the helper.
        # Parameter meanings: LDF, beta_SQT, C_T1 and E_opt come from the Table 4 row;
        #   T_opt and C_T2 are class independent; T_s is the standard temperature paired
        #   with the emission factor; R the molar gas constant.
        # Units: T, T_s, T_opt [K]; beta_SQT [K-1]; C_T1, C_T2 [J mol-1];
        #   R [J mol-1 K-1]; LDF, E_opt, gamma_T [1].
        # Domain/constraints: finite T > 0 (validated by the helper); no clamp is applied.
        # Sign convention: gamma_T is a positive multiplier of the emission magnitude.
        return self.response_integrator.temp_response.get_compound_temperature_response(
            "sesquiterpenes",
            temperature,
            param_manager=self.response_integrator.parameter_manager,
        )

    def _validated_emission_factor(self) -> Tensor:
        """Return the leaf-mass emission factor after rejecting illegal values."""
        emission_factor = self.base_emission_factor
        if not bool(torch.isfinite(emission_factor).all()) or bool((emission_factor < 0).any()):
            raise ValueError(
                "Sesquiterpene emission factor 'biosphere.bvoc.default.sesquiterpenes' must be "
                "finite and non-negative"
            )
        return emission_factor
        
    def _compute_rate_core(self, state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compute sesquiterpene emission rate using MEGAN-style activity factors.

        Parameters
        ----------
        state : Dict[str, Tensor]
            System state holding the declared inputs (temperature and the optional
            ppfd, soil_moisture and stress factors); nothing is read from elsewhere.
            
        Returns
        -------
        Dict[str, Tensor]
            Sesquiterpene emission diagnostics in mass basis [ug g-1 h-1].
        """
        # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eqs. (3), (4),
        #   (7), (8), (11) and Table 4; Duhl et al. (2008) Biogeosciences 5, 761-777.
        # Mathematical expression: gamma_T=(1-LDF_SQT)exp(beta_SQT(T-T_s))+LDF_SQT*gamma_T_LDF
        #   with the Eq. (8) gamma_T_LDF;
        #   gamma_L=(1-LDF_SQT)+LDF_SQT*alpha*C_L*PPFD/sqrt(1+alpha^2 PPFD^2);
        #   gamma_stress=min(supplied stress factors); E_mass=EF*gamma_T*gamma_L*gamma_stress.
        # Parameter meanings: beta_SQT, T_s, alpha, C_L and LDF_SQT are ParameterManager
        #   coefficients resolved by the shared response helper, which evaluates the
        #   megan_model laws with the T_s that MEGANModel also reads; an absent PPFD or
        #   stress driver gives the neutral factor one.
        # Units: E_mass,EF [ug g-1 DW h-1]; T,T_s [K]; beta_SQT [K-1];
        #   alpha [m2 s umol-1]; PPFD [umol photons m-2 s-1]; C_L, LDF_SQT, gamma_* [1].
        # Domain/constraints: Reduced instantaneous mass-normalized emission; finite
        #   temperature, PPFD >= 0 and stress factors in [0, 1] are enforced; a negative or
        #   non-finite EF raises; no activity factor or emission rate is clamped.
        # Sign convention: Positive E_mass is emission magnitude; no sesquiterpene storage is
        #   advanced.
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary of the declared inputs")
        inputs: Dict[str, Union[Tensor, float]] = {
            key: state[key]
            for key in (
                "temperature",
                "ppfd",
                "soil_moisture",
                "water_stress_factor",
                "temperature_stress_factor",
                "nutrient_stress_factor",
            )
            if key in state
        }

        temperature_value = inputs.get('temperature')
        if temperature_value is None:
            raise ValueError("Required input 'temperature' not found in state")
        temperature = self._coerce_finite_tensor(temperature_value, "temperature")
        inputs["temperature"] = temperature

        ppfd_value = inputs.get("ppfd")
        if ppfd_value is None:
            ppfd_tensor = None
        else:
            ppfd_tensor = self._coerce_finite_tensor(ppfd_value, "ppfd")
            if torch.any(ppfd_tensor < 0):
                raise ValueError("Input 'ppfd' must be non-negative")
            inputs["ppfd"] = ppfd_tensor

        for stress_key in (
            "soil_moisture",
            "water_stress_factor",
            "temperature_stress_factor",
            "nutrient_stress_factor",
        ):
            if stress_key in inputs:
                inputs[stress_key] = self._coerce_finite_tensor(
                    inputs[stress_key],
                    stress_key,
                )

        response_manager = self.response_integrator.parameter_manager
        gamma_temperature = self.temperature_function(temperature).to(
            device=self.device, dtype=torch.float32
        )
        if ppfd_tensor is None:
            gamma_light = torch.ones_like(gamma_temperature)
        else:
            gamma_light = self.response_integrator.light_response.get_compound_light_response(
                "sesquiterpenes",
                ppfd_tensor,
                param_manager=response_manager,
            ).to(device=self.device, dtype=torch.float32)
        stress_inputs_present = any(
            key in inputs
            for key in (
                "soil_moisture",
                "water_stress_factor",
                "temperature_stress_factor",
                "nutrient_stress_factor",
            )
        )
        if stress_inputs_present:
            gamma_stress = self.response_integrator.env_response.comprehensive_stress_response(
                inputs,
            ).to(device=self.device, dtype=torch.float32)
        else:
            gamma_stress = torch.ones_like(gamma_temperature)
        gamma_total = gamma_temperature * gamma_light * gamma_stress
        emission_rate = self._validated_emission_factor() * gamma_total

        return {
            'rate': emission_rate,
            'gamma_temperature': gamma_temperature,
            'gamma_light': gamma_light,
            'gamma_stress': gamma_stress,
            'gamma_total': gamma_total,
        }
        
    def forward(
        # References: Guenther et al. (2012); Duhl et al. (2008); explicit RELIC LAI-to-dry-mass
        #   bridge.
        # Mathematical expression: E_area=E_mass*LAI*k_f/seconds_per_hour;
        #   E_mol=convert(E_area/MW,umol to mol); F_C=convert(E_mol*MW*fC,g C to kg C).
        # Parameter meanings: k_f maps canonical LAI to foliage dry mass; MW and fC are the
        #   compound molecular weight and carbon fraction.
        # Units: E_area [ug m-2 s-1]; E_mol [mol m-2 s-1]; F_C [kg C m-2 s-1].
        # Domain/constraints: Finite required temperature and nonnegative LAI; direct-emission
        #   proxy without a stored sesquiterpene pool. The typed identity map does not certify
        #   strict IPHS.
        # Sign convention: Species emission and plant-carbon allocation proxy are positive outward
        #   rates.
        self,
        state: Dict[str, Union[Tensor, float]],
        time: float,
        dt: float,
    ) -> Dict[str, UpdateItem]:
        """
        Forward pass through sesquiterpene synthesis.

        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state
        time : float
            Current time
        dt : float
            Time step

        Returns
        -------
        Dict[str, UpdateItem]
            Sesquiterpene synthesis results
        """
        del time, dt
        temperature = self._extract_tensor(state, 'temperature')
        leaf_area_index = self._extract_tensor(state, 'leaf_area_index')
        if torch.any(leaf_area_index < 0.0):
            raise ValueError("Input 'leaf_area_index' must be non-negative")
        del temperature
        emission_outputs = self._compute_rate_core(state)

        # References: Guenther et al. (2012) for MEGAN-family response heritage;
        # the LAI-to-dry-mass conversion is an explicit RELIC reduced closure.
        # Mathematical expression: E_area = E_mass * LAI * k_f / t_h.
        # Parameter meanings: E_mass is the mass-normalized emission rate
        # [ug g-1 DW h-1], LAI is canonical RELIC leaf area index [1], k_f is
        # effective dry-leaf mass per canonical LAI area [g DW m-2], and t_h
        # is seconds per hour [s h-1].
        emission_rate, leaf_area_index = torch.broadcast_tensors(
            emission_outputs['rate'], leaf_area_index
        )
        emission_mass_area = (
            emission_rate
            * leaf_area_index
            * self.foliar_density_factor
            / self.seconds_per_hour
        )  # ug m-2 s-1
        
        # MW(sesquiterpene C15H24) is drawn from the parameter system.
        sesquiterpene_mw = self.sesquiterpene_mw

        # Unit identity: (ug m-2 s-1) / (g mol-1) = umol m-2 s-1.
        emission_molar_umol = emission_mass_area / sesquiterpene_mw

        # References: SI Brochure, 9th Edition; Guenther et al. (2012).
        # Mathematical expression: E_mol = convert(E_umol, umol m-2 s-1, mol m-2 s-1).
        # Parameter meanings: E_umol is the species molar flux in umol m-2 s-1;
        # E_mol is the canonical atmospheric port in mol m-2 s-1.
        # Units and route: the existing UnitConverter owns the exact SI-prefix conversion.
        emission_molar = unit_converter.convert(
            emission_molar_umol,
            variable_name="sesquiterpenes_emission",
            source_units="umol m-2 s-1",
            target_units="mol m-2 s-1",
        )

        # References: Guenther et al. (2012); Niinemets et al. (2014).
        # Mathematical expression:
        # F_C,alloc = convert(E_mol * MW_SQT * f_C, g C m-2 s-1,
        #                     kg C m-2 s-1).
        # Parameter meanings: E_mol is the reduced direct-emission proxy
        # [mol m-2 s-1], MW_SQT is molecular weight [g mol-1], and f_C is
        # the ParameterManager carbon mass fraction [1].
        # Units and route: F_C,alloc is the non-negative plant-carbon
        # allocation proxy [kg C m-2 s-1]. This no-storage closure does not
        # claim that stored sesquiterpene synthesis and release are coupled.
        carbon_allocation_flux_g = (
            emission_molar
            * self.sesquiterpene_mw
            * self.carbon_fraction_sesquiterpenes
        )
        carbon_allocation_flux = unit_converter.convert(
            carbon_allocation_flux_g,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        )

        # References: Guenther et al. (2012); Duhl et al. (2008);
        # Grmela and Ottinger (1997).
        # Mathematical expression: E_area=EF*gamma_T*gamma_L*gamma_stress*rho_f/seconds_per_hour;
        # x_dot = N_h r_h for the canonical molar port, plant-carbon
        # allocation port, and umol diagnostic.
        # Parameter meanings: EF [ug g-1 DW h-1], gamma_* [-], rho_f [g DW m-2],
        # MW_SQT [g mol-1], and seconds_per_hour [s h-1].
        # Units and route: E_mol [mol m-2 s-1] exits through the atmosphere
        # port and F_C,alloc [kg C m-2 s-1] routes to the plant-carbon ledger.
        sesquiterpene_network = advance_reaction_network(
            name=f"{self.name}.sesquiterpene_synthesis_reaction_network",
            species=(
                "sesquiterpenes_emission",
                "bvoc_carbon_allocation_flux",
                "diagnostics.bvoc.sesquiterpene.emission_rate_molar",
            ),
            reactions=(
                "sesquiterpene_molar_emission",
                "sesquiterpene_carbon_allocation",
                "sesquiterpene_umol_emission",
            ),
            stoichiometric_N=torch.eye(
                3,
                device=emission_molar.device,
                dtype=emission_molar.dtype,
            ),
            rates=(
                emission_molar,
                carbon_allocation_flux,
                emission_molar_umol,
            ),
            open_system_route="port",
        )

        return {
            "sesquiterpenes_emission": UpdateItem(
                value=sesquiterpene_network.species_rate(
                    "sesquiterpenes_emission"
                ),
                mode=UpdateMode.REPLACE,
                unit="mol m-2 s-1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "bvoc_carbon_allocation_flux": UpdateItem(
                value=sesquiterpene_network.species_rate(
                    "bvoc_carbon_allocation_flux"
                ),
                mode=UpdateMode.REPLACE,
                unit="kg C m-2 s-1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.bvoc.sesquiterpene.emission_rate_molar": UpdateItem(
                value=sesquiterpene_network.species_rate(
                    "diagnostics.bvoc.sesquiterpene.emission_rate_molar"
                ),
                mode=UpdateMode.REPLACE,
                unit="umol m-2 s-1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.bvoc.sesquiterpene.gamma_temperature": UpdateItem(
                value=emission_outputs["gamma_temperature"],
                mode=UpdateMode.REPLACE,
                unit="1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.bvoc.sesquiterpene.gamma_light": UpdateItem(
                value=emission_outputs["gamma_light"],
                mode=UpdateMode.REPLACE,
                unit="1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.bvoc.sesquiterpene.gamma_stress": UpdateItem(
                value=emission_outputs["gamma_stress"],
                mode=UpdateMode.REPLACE,
                unit="1",
                per_timestep=False,
                source_functor=self.name,
            ),
            "diagnostics.bvoc.sesquiterpene.gamma_total": UpdateItem(
                value=emission_outputs["gamma_total"],
                mode=UpdateMode.REPLACE,
                unit="1",
                per_timestep=False,
                source_functor=self.name,
            ),
        }

    def check_conservation_laws(
        self,
        input_state: Dict[str, Tensor],
        output_updates: Dict[str, Union[Tensor, UpdateItem]],
    ) -> Dict[str, float]:
        """Audit the open sesquiterpene molar-to-carbon boundary mapping.

        References:
            - Guenther et al. (2012), MEGAN2.1 emission framework.
            - Niinemets et al. (2014), BVOC formation, storage, and release.
        Mathematical expression:
            R_C = F_C,alloc
                  - convert(F_SQT MW_SQT f_C, g C to kg C).
        Parameter meanings:
            F_SQT is the atmospheric molar flux [mol m-2 s-1], MW_SQT is
            the ParameterManager-backed molecular weight [g mol-1], and f_C
            is the ParameterManager-backed carbon mass fraction [1].
        Open-system route:
            F_C,alloc leaves the vegetation budget and F_SQT enters the
            existing AtmosphereCoupling port. No closed plant stock balance
            or synthesis-storage equivalence is claimed.
        """
        # References: Guenther et al. (2012); declared sesquiterpene carbon-equivalent port
        #   mapping.
        # Mathematical expression: r=F_C_alloc-convert(E_mol*MW*fC,g C to kg C);
        #   report=max(abs(r)).
        # Parameter meanings: E_mol is canonical sesquiterpene emission; MW and fC share its
        #   promoted ledger dtype.
        # Units: Residual [kg C m-2 s-1]; E_mol [mol m-2 s-1].
        # Domain/constraints: Finite required shape-compatible outputs; this is an open no-storage
        #   port check, not an intracellular pathway balance.
        # Sign convention: Positive F_C_alloc is carbon export; residual magnitude is nonnegative.

        del input_state

        def require_tensor(key: str) -> Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing conservation output '{key}'")
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, Tensor):
                raise TypeError(
                    "Sesquiterpene conservation output "
                    f"'{key}' must be a torch.Tensor."
                )
            if value.numel() == 0:
                raise ValueError(
                    "Sesquiterpene conservation output "
                    f"'{key}' must be non-empty."
                )
            if not torch.is_floating_point(value):
                raise TypeError(
                    "Sesquiterpene conservation output "
                    f"'{key}' must be floating."
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    "Sesquiterpene conservation output "
                    f"'{key}' must be finite."
                )
            return value.to(device=self.device)

        emission_mol = require_tensor("sesquiterpenes_emission")
        allocation_carbon = require_tensor("bvoc_carbon_allocation_flux")
        if emission_mol.shape != allocation_carbon.shape:
            raise ValueError(
                "Sesquiterpene conservation outputs must have identical shapes."
            )
        ledger_dtype = torch.promote_types(
            emission_mol.dtype,
            allocation_carbon.dtype,
        )
        emission_mol = emission_mol.to(dtype=ledger_dtype)
        allocation_carbon = allocation_carbon.to(dtype=ledger_dtype)

        expected_carbon = unit_converter.convert(
            emission_mol
            * self.sesquiterpene_mw.to(
                device=self.device,
                dtype=ledger_dtype,
            )
            * self.carbon_fraction_sesquiterpenes.to(
                device=self.device,
                dtype=ledger_dtype,
            ),
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        ).to(dtype=ledger_dtype)
        residual = allocation_carbon - expected_carbon
        return {
            "carbon_boundary": float(
                torch.max(torch.abs(residual)).detach().item()
            )
        }

    def _extract_tensor(
        self,
        state: Dict[str, Union[Tensor, float]],
        key: str,
    ) -> Tensor:
        """Extract and convert input to tensor."""
        if key in state:
            value = state[key]
        else:
            raise ValueError(f"Required input '{key}' not found in state")
        
        if not isinstance(value, Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=self.device)
        else:
            value = value.to(device=self.device, dtype=torch.float32)
            
        if not torch.isfinite(value).all():
            raise ValueError(f"Input '{key}' must be finite")
        return value

    def _coerce_finite_tensor(
        self,
        value: Union[Tensor, float],
        key: str,
    ) -> Tensor:
        """Convert one declared input to a finite tensor on the Functor device."""
        return self._extract_tensor({key: value}, key)


class SesquiterpeneStressModel(BiologicalProcessFunctor):
    """
    Sesquiterpene stress response observation with threshold activation.

    One aggregated stress driver S (state "stress_level") gives the multiplier
    f = min(max(1 + max(S - S_th, 0) * A, 1), 1 + A). This threshold-linear, capped form is the RELIC
    reduced calibration documented in docs/source/theory/bvoc_emissions.md;
    no per-stress weighting and no recovery state are executed.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    
    def __init__(
        self,
        name: str = "sesquiterpene_stress_model",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize sesquiterpene stress response model."""
        config = config or {}
        
        metadata = FunctorMetadata(
            name=name,
            version="2.0.0",
            description=(
                "Sesquiterpene stress response observation: threshold-linear, capped "
                "amplification of one aggregated stress driver (no per-stress weighting, "
                "no recovery state)."
            ),
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "stress_response_factor = min(max(1 + max(S - S_th, 0) * A, 1), 1 + A); "
                "S = stress_level [1], one aggregated stress driver; "
                "S_th = biosphere.bvoc.sesquiterpene.stress_threshold [1]; "
                "A = biosphere.bvoc.sesquiterpene.stress_amplification [1]; "
                "algebraic in S, independent of dt, no recovery state; "
                "RELIC reduced threshold-linear calibration of the stress "
                "dose-response (docs/source/theory/bvoc_emissions.md)"
            ),
            references=[
                "Copolovici et al. (2012). Journal of Plant Physiology 169:664-672",
                "Loreto & Schnitzler (2010). Trends in Plant Science 15:154-166",
                "Niinemets et al. (2013). Frontiers in Plant Science 4:262"
            ],
            parameters={
                'biosphere.bvoc.sesquiterpene.stress_threshold': {
                    'description': 'Stress level S_th above which the sesquiterpene response is activated',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.sesquiterpene.stress_amplification': {
                    'description': 'Amplification A per unit stress above S_th; the factor is capped at 1 + A',
                    'units': 'dimensionless',
                },
            },
            # The stress factor is a dimensionless algebraic multiplier of the
            # emission potential: it advances no carbon, water or energy
            # reservoir, so it declares no conservation law (the emission
            # carbon boundary belongs to the synthesis Functors).
            conservation_laws=[],
            input_variables=['stress_level'],
            output_variables=['stress_response_factor'],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device
        )

        # The algebraic factor consumes none of the generic response parameters
        # (q10, activation energy, moisture and pH shapes) that the biological
        # bridge adds to the metadata; only the two threshold-law parameters apply.
        for injected_name in [
            name for name in self.metadata.parameters
            if name not in (
                'biosphere.bvoc.sesquiterpene.stress_threshold',
                'biosphere.bvoc.sesquiterpene.stress_amplification',
            )
        ]:
            del self.metadata.parameters[injected_name]

        def _fetch_required_parameter(param_name: str, namespace: str) -> torch.Tensor:
            value = self.get_parameter_value(
                param_name,
                namespace=namespace,
            )
            if value is None:
                full_key = f"{namespace}.{param_name}"
                raise KeyError(f"Missing required parameter '{full_key}'")
            return self._ensure_tensor(value, dtype=torch.float32)

        # Initialize stress response parameters
        self.register_buffer(
            'stress_threshold',
            self._ensure_tensor(
                _fetch_required_parameter(
                    "sesquiterpene.stress_threshold",
                    "biosphere.bvoc",
                ),
                dtype=torch.float32,
            ).to(device=self.device),
        )
        self.register_buffer(
            'stress_amplification',
            self._ensure_tensor(
                _fetch_required_parameter(
                    "sesquiterpene.stress_amplification",
                    "biosphere.bvoc",
                ),
                dtype=torch.float32,
            ).to(device=self.device),
        )

    def get_dependencies(self) -> list:
        """Declare optional upstream stress providers for execution ordering."""
        return self._merge_declared_dependencies([])

        
    def forward(
        # References: RELIC reduced calibration, docs/source/theory/bvoc_emissions.md
        #   (owner of the executed law); stress-enhanced sesquiterpene emission context:
        #   Loreto and Schnitzler (2010), Trends in Plant Science 15, 154-166; Duhl et al.
        #   (2008), Biogeosciences 5, 761-777.
        # Mathematical expression:
        #   f = min(max(1 + max(S - S_th, 0) * A, 1), 1 + A).
        # Parameter meanings: S is stress_level, one aggregated stress driver; S_th is
        #   biosphere.bvoc.sesquiterpene.stress_threshold; A is
        #   biosphere.bvoc.sesquiterpene.stress_amplification; f is stress_response_factor.
        # Units: S, S_th, A and f are dimensionless [1].
        # Domain/constraints: algebraic in S and independent of dt; 1 <= f <= 1 + A; no
        #   per-stress weighting and no recovery state; the threshold-linear shape is a
        #   reduced calibrated choice, not a universal stress law.
        # Sign convention: stress above S_th amplifies the emission potential (f >= 1);
        #   the factor carries no carbon, water or energy flux of its own.
        self,
        state: Dict[str, Union[Tensor, float]],
        time: float,
        dt: float,
    ) -> Dict[str, UpdateItem]:
        """
        Compute stress-induced sesquiterpene response.

        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Standard functor state dictionary containing `stress_level`.
        time : float
            Current simulation time [s].
        dt : float
            Time step size [s]. Unused in this algebraic response.

        Returns
        -------
        Dict[str, UpdateItem]
            `stress_response_factor` as one REPLACE UpdateItem in the SCM unit
            "1" (per_timestep=False): the algebraic factor replaces the previous
            value and is never accumulated over steps.
        """
        del time, dt

        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary containing 'stress_level'.")
        if "stress_level" not in state:
            raise KeyError("Required input 'stress_level' not found in state.")
        stress_source = state["stress_level"]

        if isinstance(stress_source, Tensor):
            stress_level = stress_source.to(device=self.device)
        else:
            stress_level = torch.as_tensor(
                float(stress_source),
                dtype=torch.float32,
                device=self.device,
            )

        # References: RELIC reduced threshold-linear calibration,
        #   docs/source/theory/bvoc_emissions.md; stress-enhanced emission context: Duhl
        #   et al. (2008), Biogeosciences 5, 761-777; Loreto and Schnitzler (2010), Trends
        #   in Plant Science 15, 154-166.
        # Mathematical expression:
        #   f = min(max(1 + max(S - S_th, 0) * A, 1), 1 + A).
        # Parameter meanings: S is stress_level; S_th is
        #   biosphere.bvoc.sesquiterpene.stress_threshold; A is
        #   biosphere.bvoc.sesquiterpene.stress_amplification; f is
        #   stress_response_factor; 1 is the neutral (unstressed) multiplier.
        # Units: S, S_th, A and f are dimensionless [1].
        # Domain/constraints: algebraic in S and independent of dt; 1 <= f <= 1 + A;
        #   the upper bound stays a tensor so the parameter keeps its autograd path;
        #   the value is committed with UpdateMode.REPLACE, never accumulated.
        # Sign convention: f >= 1; stress above the threshold amplifies the emission
        #   potential; the factor carries no mass or energy flux of its own.
        stress_factor = torch.where(
            stress_level > self.stress_threshold,
            1.0 + (stress_level - self.stress_threshold) * self.stress_amplification,
            torch.ones_like(stress_level)
        )
        upper_bound = 1.0 + self.stress_amplification.reshape(-1)[0]
        output = torch.minimum(torch.clamp(stress_factor, min=1.0), upper_bound)

        return {
            "stress_response_factor": UpdateItem(
                value=output,
                mode=UpdateMode.REPLACE,
                unit="1",
                per_timestep=False,
                source_functor=self.name,
            )
        }


# Export classes
__all__ = [
    'SesquiterpeneSynthesis',
    'SesquiterpeneStressModel'
]


