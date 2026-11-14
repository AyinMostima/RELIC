"""
Isoprene Synthesis Pathway Model for RELIC

This module provides mechanistic models of isoprene biosynthesis through the
MEP (methylerythritol phosphate) pathway, offering biochemical understanding
of the processes underlying empirical MEGAN relationships.

References
----------
- Lichtenthaler, H.K. (1999). The 1-deoxy-D-xylulose-5-phosphate pathway of
  isoprenoid biosynthesis in plants.
  Annual Review of Plant Physiology and Plant Molecular Biology 50, 47-65
- Vickers, C.E., et al. (2009). A unified mechanism of action for volatile 
  isoprenoids in plant abiotic stress. Nature Chemical Biology 5, 283-291
- Rasulov, B., et al. (2009). Evidence that light, carbon dioxide, and oxygen
  dependencies of leaf isoprene emission are driven by energy status in hybrid
  aspen. Plant Physiology 151, 448-460
- Morfopoulos, C., et al. (2014). A model of plant isoprene emission based on 
  available reducing power captures responses to atmospheric CO2. New Phytologist 203, 125-139

Mathematical Expressions
-------------------------
1. MEP pathway flux: J_MEP = V_max_MEP * [G3P] * [Pyruvate] / ((K_G3P + [G3P]) * (K_Pyr + [Pyruvate]))
2. Isoprene synthase: J_IS = V_max_IS * [DMAPP] / (K_DMAPP + [DMAPP])
3. ATP/NADPH dependency: J_MEP = J_MEP * min(ATP_avail/ATP_req, NADPH_avail/NADPH_req)
4. CO2 inhibition proxy: J_IS = J_IS * K_CO2 / (K_CO2 + C_a)
5. Temperature response: V_max(T) = V_max_25 * Q10^((T-T_ref)/10)
   where T is the canonical grid-cell air temperature used as an explicit
   proxy when no leaf energy-balance temperature is available, and T_ref is
   the physical constant ``reference_temperature`` (298.15 K).
6. Canopy support bridge:
   L_eff = (1 - exp(-k * LAI)) / k,
   F_leaf = F_ground / L_eff for L_eff > 0, and
   F_ground = F_leaf * L_eff.

Units
-----
- Internal biochemical fluxes (J): umol m-2 leaf s-1
- Public fluxes (J): umol m-2 ground s-1
- Concentrations: umol m-3 (cytoplasmic)
- Enzyme activities (V_max): umol m-2 leaf s-1
- Michaelis constants (K): umol m-3
- Temperatures: K
"""

from typing import Dict, Optional, Union, Any, Callable, Tuple
import torch
import logging
from torch import Tensor
import torch.nn as nn

# Import from core framework
from relic.core.functor import BiologicalProcessFunctor, TemporalScale
from relic.utils.unified_response_functions import (
    beer_lambert_effective_leaf_area,
    q10_temperature_response,
    michaelis_menten_response,
)
from relic.parameters import get_global_manager
from relic.computation.operators.iphs import advance_reaction_network
from relic.core.state import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter
from relic.utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)


def _scalar_from_value(value: Union[Tensor, float, int]) -> float:
    """Convert numeric values to Python float without tensor.item/cpu APIs."""
    if isinstance(value, Tensor):
        detached = value.detach().reshape(-1)
        if detached.numel() == 0:
            raise ValueError("Cannot convert empty tensor to scalar.")
        return detached[:1].to(dtype=torch.float64).tolist()[0]
    return float(value)


def _get_validated_parameter_from_database(name: str, namespace: str, units: str) -> float:
    """Fetch a parameter from the global manager in the requested units (the manager
    no longer takes a validate flag)."""
    lookup_kwargs = {"units": units}
    value = get_global_manager().get(
        name,
        namespace=namespace or None,
        **lookup_kwargs,
    )
    if value is None:
        raise KeyError(f"Parameter '{namespace}.{name}' not registered in database")
    if isinstance(value, Tensor):
        return _scalar_from_value(value)
    return _scalar_from_value(value)


class MEPPathwayModel(nn.Module):
    """
    Methylerythritol Phosphate (MEP) Pathway Model.
    
    Models the biochemical pathway from glyceraldehyde-3-phosphate (G3P) and 
    pyruvate to dimethylallyl diphosphate (DMAPP), the immediate precursor 
    of isoprene.
    
    Pathway Steps (simplified):
    1. DXS: G3P + Pyruvate -> DXP (rate-limiting)
    2. DXR: DXP + NADPH -> MEP 
    3. Multiple steps: MEP -> DMAPP (ATP-dependent)
    
    References
    ----------
    - Lichtenthaler (1999): Complete MEP pathway description
    - Vickers et al. (2009): Environmental controls
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parameter_getter: Optional[Callable[[str, str, str], Union[Tensor, float]]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if config:
            raise ValueError(
                "MEPPathwayModel kinetic parameters must be configured through ParameterManager."
            )

        self.device = device or (
            torch.device("cpu")
        )
        self._param_cache: Dict[str, Tensor] = {}
        self._param_specs: Dict[str, Tuple[str, str, str]] = {
            "v_max_dxs": ("v_max_dxs", "biosphere.bvoc.mep", "umol m-2 s-1"),
            "k_g3p": ("k_g3p", "biosphere.bvoc.mep", "umol m-3"),
            "k_pyruvate": ("k_pyruvate", "biosphere.bvoc.mep", "umol m-3"),
            "q10_dxs": ("q10_dxs", "biosphere.bvoc.mep", "dimensionless"),
            "atp_per_dmapp": ("atp_per_dmapp", "biosphere.bvoc.mep", "dimensionless"),
            "nadph_per_dmapp": ("nadph_per_dmapp", "biosphere.bvoc.mep", "dimensionless"),
            "energy_limitation_cap": ("coupling_max", "validation.numerical", "dimensionless"),
            "epsilon_dxs": ("epsilon", "validation.numerical", "dimensionless"),
            "epsilon_energy": ("epsilon_energy", "validation.numerical", "dimensionless"),
        }
        # Physical constants are resolved only through the physical-constant SSOT; the
        # Q10 reference temperature is the 25 degC standard (IUPAC, 298.15 K).
        self._physical_constant_names: Dict[str, str] = {
            "temp_ref": "reference_temperature",
        }
        if parameter_getter is None:
            def _parameter_getter(param_name: str, units: str, namespace: str) -> float:
                return _get_validated_parameter_from_database(param_name, namespace, units)
            self._parameter_getter = _parameter_getter
        else:
            self._parameter_getter = parameter_getter

    def _get_cached_parameter(self, attr: str) -> Tensor:
        if attr not in self._param_cache:
            if attr in self._physical_constant_names:
                value = get_physical_constant(self._physical_constant_names[attr])
            else:
                param_name, namespace, units = self._param_specs[attr]
                value = self._parameter_getter(param_name, units, namespace)
            # A learnable coefficient supplied by the owning Functor is the
            # ParameterManager-owned binding (architecture SSOT section 4.2). The
            # object is cached unchanged, so the gradient of every call reaches it
            # and its live value is read at every use. Every other value is cached
            # as a float32 tensor on the device of this helper.
            if not isinstance(value, nn.Parameter):
                value = self._ensure_tensor(value)
            self._param_cache[attr] = value
        # Tensor.to returns its operand when dtype and device already match,
        # otherwise a differentiable converted copy of a cached binding.
        return self._ensure_tensor(self._param_cache[attr])

    def to(self, device: torch.device) -> "MEPPathwayModel":
        """Manual device transfer helper for continuity with nn.Module APIs."""
        target = torch.device(device)
        if target == self.device:
            return self
        for name, value in self._param_cache.items():
            # A cached binding is relocated only by its owner, the manager.
            if not isinstance(value, nn.Parameter):
                self._param_cache[name] = value.to(target)
        self.device = target
        return self

    def refresh_from_manager(self) -> None:
        """Resolve every ParameterManager coefficient again through the getter."""
        self.clear_parameter_cache()
        for attr in self._param_specs:
            self._get_cached_parameter(attr)

    def clear_parameter_cache(self) -> None:
        """Clear cached parameter tensors to force reload on next access."""
        self._param_cache.clear()

    def __getattr__(self, name: str) -> Tensor:
        param_specs = object.__getattribute__(self, "_param_specs")
        physical_constant_names = object.__getattribute__(self, "_physical_constant_names")
        if name in param_specs or name in physical_constant_names:
            return self._get_cached_parameter(name)
        return super().__getattr__(name)

    def _ensure_tensor(self, value: Union[Tensor, float]) -> Tensor:
        if isinstance(value, Tensor):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.tensor(float(value), dtype=torch.float32, device=self.device)

    def __call__(
        self, 
        g3p_concentration: Tensor,
        pyruvate_concentration: Tensor,
        atp_availability: Tensor,
        nadph_availability: Tensor,
        temperature: Tensor
    ) -> Dict[str, Tensor]:
        """
        Calculate MEP pathway flux to DMAPP.
        
        Parameters
        ----------
        g3p_concentration : Tensor
            Glyceraldehyde-3-phosphate concentration [umol m-3]
        pyruvate_concentration : Tensor
            Pyruvate concentration [umol m-3]
        atp_availability : Tensor
            Available ATP flux [umol m-2 s-1]
        nadph_availability : Tensor
            Available NADPH flux [umol m-2 s-1]
        temperature : Tensor
            Temperature [K]
            
        Returns
        -------
        Dict[str, Tensor]
            MEP pathway results
        """
        return self.forward(
            g3p_concentration,
            pyruvate_concentration,
            atp_availability,
            nadph_availability,
            temperature,
        )

    def forward(
        # References: Lichtenthaler (1999); Vickers et al. (2009); existing reduced MEP allocation
        #   model.
        # Mathematical expression: J_DXS=V_DXS(T)*G3P*Pyr/((K_G3P+G3P)*(K_Pyr+Pyr)+eps_DXS);
        #   f_E=min(ATP/(J_DXS*a+eps_E),NADPH/(J_DXS*n+eps_E),cap); J_DMAPP=J_DXS*f_E.
        # Parameter meanings: a,n are ATP/NADPH cost per DMAPP; V_DXS(T) uses the shared Q10
        #   response; eps values are existing numerical regularizers.
        # Units: Substrates and Km [umol m-3]; J [umol m-2 leaf s-1]; energy supplies/demands [umol
        #   m-2 leaf s-1]; f_E [1].
        # Domain/constraints: Reduced algebraic substrate/energy closure with finite nonnegative
        #   physical inputs; this helper does not own an intracellular prognostic pool.
        # Sign convention: Positive J_DMAPP is synthesis supply; ATP and NADPH demands are positive
        #   consumption magnitudes.
        self,
        g3p_concentration: Union[Tensor, float],
        pyruvate_concentration: Union[Tensor, float],
        atp_availability: Union[Tensor, float],
        nadph_availability: Union[Tensor, float],
        temperature: Union[Tensor, float],
    ) -> Dict[str, Tensor]:
        """Calculate MEP pathway flux to DMAPP."""
        # References: Lichtenthaler 1999; Vickers et al. 2009; Zimmer et al. 2000
        # Equation: J_MEP = V_max_dxs * [G3P] * [Pyr] / ((K_G3P + [G3P]) * (K_Pyr + [Pyr]))
        # Parameters: v_max_dxs [umol m-2 s-1], k_g3p [umol m-3], k_pyruvate [umol m-3], atp_per_dmapp [mol mol-1], nadph_per_dmapp [mol mol-1]
        g3p_concentration = self._ensure_tensor(g3p_concentration)
        pyruvate_concentration = self._ensure_tensor(pyruvate_concentration)
        atp_availability = self._ensure_tensor(atp_availability)
        nadph_availability = self._ensure_tensor(nadph_availability)
        temperature = self._ensure_tensor(temperature)

        v_max_dxs_temp = q10_temperature_response(
            self.v_max_dxs, temperature, self.q10_dxs, self.temp_ref)
        
        # DXS reaction rate (double substrate Michaelis-Menten)
        # Rate = V_max * [G3P] * [Pyr] / ((K_G3P + [G3P]) * (K_Pyr + [Pyr]))
        numerator = v_max_dxs_temp * g3p_concentration * pyruvate_concentration
        denominator = (self.k_g3p + g3p_concentration) * (self.k_pyruvate + pyruvate_concentration)
        
        dxs_flux = numerator / (denominator + self.epsilon_dxs)
        
        # Potential energy demand determines the limiting energy supply.
        potential_atp_demand = dxs_flux * self.atp_per_dmapp
        potential_nadph_demand = dxs_flux * self.nadph_per_dmapp
        
        atp_limitation = atp_availability / (
            potential_atp_demand + self.epsilon_energy
        )
        nadph_limitation = nadph_availability / (
            potential_nadph_demand + self.epsilon_energy
        )
        
        # Most limiting energy source
        energy_limitation = torch.min(atp_limitation, nadph_limitation)
        # Numerical guard: cap by validation.numerical.coupling_max from ParameterManager
        energy_limitation = torch.clamp(energy_limitation, max=self.energy_limitation_cap)
        
        # Final DMAPP production flux
        dmapp_flux = dxs_flux * energy_limitation

        # References: Zhao et al. (2013); Rasulov et al. (2009).
        # Mathematical expressions: A_ATP = J_DMAPP * nu_ATP and
        # A_NADPH = J_DMAPP * nu_NADPH.
        # Parameter meanings: nu_ATP and nu_NADPH are the ParameterManager
        # energy requirements per unit of realized DMAPP production [mol mol-1].
        atp_demand = dmapp_flux * self.atp_per_dmapp
        nadph_demand = dmapp_flux * self.nadph_per_dmapp
        
        return {
            'dmapp_flux': dmapp_flux,
            'dxs_flux': dxs_flux,
            'energy_limitation': energy_limitation,
            'atp_demand': atp_demand,
            'nadph_demand': nadph_demand
        }


class IsopreneSynthesis(BiologicalProcessFunctor):
    """
    Mechanistic Isoprene Synthesis Model.

    Integrates MEP pathway with isoprene synthase to provide biochemically-based
    isoprene emission predictions. Links directly to photosynthetic metabolism
    and energy availability.
    
    Features
    --------
    - Complete MEP pathway representation
    - Photosynthetic substrate and energy coupling
    - CO2 inhibition of isoprene synthase
    - Temperature-dependent enzyme kinetics
    - Stress response integration
    
    Architecture
    ------------
    - Uses MEPPathwayModel for precursor synthesis
    - Implements isoprene synthase kinetics
    - Couples to photosynthetic carbon and energy fluxes
    - Provides mechanistic alternative to empirical MEGAN
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    
    def __init__(
        self,
        name: str = "mechanistic_isoprene_synthesis",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize mechanistic isoprene synthesis model.
        
        Parameters
        ----------
        name : str, optional
            Model name
        config : Dict[str, Any], optional
            Configuration parameters
        device : torch.device, optional
            Computation device
        """
        # References: Lichtenthaler (1999); Rasulov et al. (2009); Vranova et al. (2013).
        # Mathematical expressions: J_DXS = Vmax_DXS * [G3P][Pyr] / ((K_G3P + [G3P]) * (K_Pyr + [Pyr]));
        #   J_IS = Vmax_IS(T) * [DMAPP] / (K_DMAPP + [DMAPP]) * f_CO2 where f_CO2 = 1 - Ci/(Ci + K_CO2).
        # Parameters: v_max_dxs (maximum DXS velocity, umol m-2 s-1); k_g3p / k_pyruvate (substrate Michaelis constants, umol m-3);
        #   q10_* (dimensionless Q10 factors governing temperature response); f_g3p/f_pyruvate (allocation fractions);
        #   atp_per_dmapp/nadph_per_dmapp (energy demand per DMAPP, mol mol-1); stability guards (validation.numerical.*).
        from relic.core.functor import FunctorMetadata, ProcessCategory, ExecutionPhase, SpatialScale, VariableKind
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Mechanistic isoprene synthesis via MEP pathway with photosynthetic coupling",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            parameters={
                'biosphere.bvoc.mep.v_max_dxs': {
                    'description': 'Maximum velocity of DXS enzyme',
                    'units': 'umol m-2 s-1',
                },
                'biosphere.bvoc.mep.k_g3p': {
                    'description': 'Michaelis constant for G3P substrate',
                    'units': 'umol m-3',
                },
                'biosphere.bvoc.mep.k_pyruvate': {
                    'description': 'Michaelis constant for pyruvate substrate',
                    'units': 'umol m-3',
                },
                'biosphere.bvoc.mep.q10_dxs': {
                    'description': 'Temperature sensitivity factor for DXS enzyme',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.mep.atp_per_dmapp': {
                    'description': 'ATP molecules required per DMAPP produced',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.mep.nadph_per_dmapp': {
                    'description': 'NADPH molecules required per DMAPP produced',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.isoprene_synthase.v_max': {
                    'description': 'Maximum velocity of isoprene synthase',
                    'units': 'umol m-2 s-1',
                },
                'biosphere.bvoc.isoprene_synthase.k_dmapp': {
                    'description': 'Michaelis constant for DMAPP substrate',
                    'units': 'umol m-3',
                },
                'biosphere.bvoc.isoprene_synthase.k_co2_inhibition': {
                    'description': 'Ambient CO2 inhibition proxy constant',
                    'units': 'umol mol-1',
                },
                'biosphere.bvoc.isoprene_synthase.q10': {
                    'description': 'Temperature sensitivity factor for isoprene synthase',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.isoprene_synthase.enable_co2_inhibition': {
                    'description': 'Flag enabling CO2 inhibition feedback',
                    'units': 'dimensionless',
                },
                'relic.physical.constants.molecular_weights.isoprene': {
                    'description': 'Molecular weight of isoprene',
                    'units': 'g mol-1',
                },
                'biosphere.bvoc.carbon_fraction.isoprene': {
                    'description': (
                        'Carbon mass fraction for isoprene (C5H8) used in conservation '
                        'diagnostics'
                    ),
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.isoprene_synthase.f_g3p': {
                    'description': (
                        'Fraction of photosynthetic G3P allocated to isoprene synthesis'
                    ),
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.isoprene_synthase.f_pyruvate': {
                    'description': (
                        'Fraction of photosynthetic pyruvate allocated to isoprene synthesis'
                    ),
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.isoprene_synthase.g3p_flux_to_concentration': {
                    'description': 'Scaling factor converting G3P flux to concentration',
                    'units': 's m-1',
                },
                'biosphere.bvoc.isoprene_synthase.pyruvate_flux_to_concentration': {
                    'description': 'Scaling factor converting pyruvate flux to concentration',
                    'units': 's m-1',
                },
                'biosphere.bvoc.isoprene_synthase.dmapp_flux_to_concentration': {
                    'description': 'Scaling factor converting DMAPP flux to concentration',
                    'units': 's m-1',
                },
                'biosphere.bvoc.isoprene_synthase.atp_per_electron_transport': {
                    'description': 'ATP produced per unit electron transport flux',
                    'units': 'dimensionless',
                },
                'biosphere.bvoc.isoprene_synthase.nadph_per_electron_transport': {
                    'description': 'NADPH produced per unit electron transport flux',
                    'units': 'dimensionless',
                },
                'relic.physical.constants.temperature.reference_temperature': {
                    'description': (
                        'Q10 reference temperature (25 degC) from the physical-constant SSOT'
                    ),
                    'units': 'K',
                },
                'validation.numerical.coupling_max': {
                    'description': 'Upper bound of the MEP energy-limitation factor',
                    'units': 'dimensionless',
                },
                'biosphere.photosynthesis.canopy_light_extinction': {
                    'description': (
                        'Beer-Lambert extinction coefficient shared with canopy photosynthesis'
                    ),
                    'units': 'dimensionless',
                },
                'validation.numerical.epsilon': {
                    'description': 'Denominator floor of the DXS kinetics',
                    'units': 'dimensionless',
                },
                'validation.numerical.epsilon_energy': {
                    'description': 'Denominator floor of the ATP and NADPH demand ratios',
                    'units': 'dimensionless',
                },
            },
            input_variables=[
                "air_temperature",
                "gross_photosynthesis",
                "electron_transport_rate",
                "atmospheric_co2_ppm",
                "leaf_area_index",
            ],
            output_variables=[
                "isoprene_emission",
                "bvoc_carbon_allocation_flux",
                "diagnostics.bvoc.isoprene.emission_rate_molar",
                "diagnostics.bvoc.isoprene.emission_rate_mass",
                "diagnostics.bvoc.isoprene.dmapp_concentration",
                "diagnostics.bvoc.isoprene.mep_pathway_flux",
                "diagnostics.bvoc.isoprene.synthase_activity",
                "diagnostics.bvoc.isoprene.co2_inhibition_factor",
                "diagnostics.bvoc.isoprene.energy_limitation_factor",
                "diagnostics.bvoc.isoprene.g3p_availability",
                "diagnostics.bvoc.isoprene.pyruvate_availability",
                "diagnostics.bvoc.isoprene.atp_demand",
                "diagnostics.bvoc.isoprene.nadph_demand",
            ],
            conservation_laws=["carbon_boundary"],
            coupling_interfaces=[
                "photosynthesis_carbon_allocation",
                "electron_transport_energy_balance",
                "bvoc_emission_flux",
            ],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            mathematical_expression="""MEP Pathway: G3P + Pyruvate --DXS--> DXP --DXR--> MEP ---> DMAPP
            Isoprene Synthesis: DMAPP --IS--> Isoprene
            J_DXS = V_max_DXS * [G3P]*[Pyr] / ((K_G3P + [G3P]) * (K_Pyr + [Pyr]))
            J_IS(T) = V_max_IS(T) * [DMAPP] / (K_DMAPP + [DMAPP]) * K_CO2/(K_CO2 + C_a)
            L_eff = (1 - exp(-k * LAI)) / k
            F_leaf = F_ground / L_eff; F_ground = F_leaf * L_eff""",
            references=[
                "Guenther et al. (2012). The Model of Emissions of Gases and Aerosols from Nature version 2.1 (MEGAN2.1). Geoscientific Model Development.",
                "Lichtenthaler (1999). The 1-deoxy-D-xylulose-5-phosphate pathway of isoprenoid biosynthesis in plants. Annual Review of Plant Physiology and Plant Molecular Biology.",
                "Rasulov et al. (2009). Evidence that light, carbon dioxide, and oxygen dependencies of leaf isoprene emission are driven by energy status in hybrid aspen. Plant Physiology.",
                "Vranova et al. (2013). Network analysis of the MVA and MEP pathways for isoprenoid synthesis. Annual Review of Plant Biology.",
                "de Pury and Farquhar (1997). Simple scaling of photosynthesis from leaves to canopies without the errors of big-leaf models. Plant, Cell and Environment.",
                "Koppmann (2007). Volatile Organic Compounds in the Atmosphere. Wiley."
            ]
        )
        # Initialize BiologicalProcessFunctor properly
        super().__init__(
            name=name,
            metadata=metadata,
            config=config or {},
            device=device,
        )

        # Set biological process parameters
        self.process_name = name
        
        if device is None:
            self.device = (
                torch.device("cpu")
            )
        else:
            self.device = device
        # References: IUPAC/CIAAW standard atomic weights for C5H8.
        # Expression: F_mass = F_mol * M_isoprene.
        # Parameters: M_isoprene is the fixed isoprene molar mass [g mol-1].
        isoprene_mw = get_physical_constant(
            "relic.physical.constants.molecular_weights.isoprene",
            as_tensor=True,
            device=self.device,
        )
        self.register_buffer(
            "isoprene_mw",
            torch.as_tensor(
                isoprene_mw,
                dtype=torch.float32,
                device=self.device,
            ).detach().clone(),
            persistent=False,
        )
        self._param_cache: Dict[str, Tensor] = {}
        self._param_specs: Dict[str, Tuple[str, str]] = {
            # V_max of isoprene synthase is readable under both attribute names.
            "v_max_is": ("biosphere.bvoc.isoprene_synthase.v_max", "umol m-2 s-1"),
            "v_max_isoprene_synthase": ("biosphere.bvoc.isoprene_synthase.v_max", "umol m-2 s-1"),
            "k_dmapp": ("biosphere.bvoc.isoprene_synthase.k_dmapp", "umol m-3"),
            "k_co2_inhibition": ("biosphere.bvoc.isoprene_synthase.k_co2_inhibition", "umol mol-1"),
            "q10_is": ("biosphere.bvoc.isoprene_synthase.q10", "dimensionless"),
            "carbon_fraction_isoprene": ("biosphere.bvoc.carbon_fraction.isoprene", "dimensionless"),
            "f_g3p_from_photosynthesis": ("biosphere.bvoc.isoprene_synthase.f_g3p", "dimensionless"),
            "f_pyruvate_from_photosynthesis": ("biosphere.bvoc.isoprene_synthase.f_pyruvate", "dimensionless"),
            "g3p_flux_to_concentration": ("biosphere.bvoc.isoprene_synthase.g3p_flux_to_concentration", "s m-1"),
            "pyruvate_flux_to_concentration": ("biosphere.bvoc.isoprene_synthase.pyruvate_flux_to_concentration", "s m-1"),
            "dmapp_flux_to_concentration": ("biosphere.bvoc.isoprene_synthase.dmapp_flux_to_concentration", "s m-1"),
            "atp_per_electron_transport": ("biosphere.bvoc.isoprene_synthase.atp_per_electron_transport", "dimensionless"),
            "nadph_per_electron_transport": ("biosphere.bvoc.isoprene_synthase.nadph_per_electron_transport", "dimensionless"),
        }
            
        # Coefficients of the MEP helper, resolved and held by this Functor under
        # their ParameterManager keys (see _get_mep_parameter).
        self._mep_param_cache: Dict[str, Tensor] = {}
        self.mep_pathway = MEPPathwayModel(
            parameter_getter=self._get_mep_parameter,
            device=self.device,
        ).to(self.device)
        self._parameter_change_callback: Optional[Callable[[str, Any, Any], None]] = None
        self._load_isoprene_synthase_parameters(init=True)
        self._register_parameter_change_callback()
        
    def _load_isoprene_synthase_parameters(self, init: bool = True) -> None:
        """Initialize or refresh isoprene synthase parameters."""

        extinction_value = self.get_parameter_value(
            "canopy_light_extinction",
            namespace="biosphere.photosynthesis",
            required=True,
        )
        extinction_tensor = torch.as_tensor(
            extinction_value,
            dtype=torch.float32,
            device=self.device,
        )
        if (
            not torch.isfinite(extinction_tensor).all()
            or torch.any(extinction_tensor <= 0)
        ):
            raise ValueError(
                "Parameter 'biosphere.photosynthesis.canopy_light_extinction' "
                "must contain finite positive values."
            )
        if init:
            self.register_buffer(
                "canopy_light_extinction",
                extinction_tensor.detach().clone(),
            )
        else:
            with torch.no_grad():
                self.canopy_light_extinction.copy_(extinction_tensor)

        # Resolve every coefficient of _param_specs and of the MEP helper at construction
        # and at each refresh (architecture SSOT sections 4.2 and 4.4.3). The learnable
        # ones among them, V_max of isoprene synthase and the MEP kinetics included, are
        # ParameterManager bindings held by reference, which registration and the durable
        # checkpoint compare and the calibration path optimizes.
        self._param_cache.clear()
        for cached_name in self._param_specs:
            self._get_cached_parameter(cached_name)
        self._mep_param_cache.clear()
        self.mep_pathway.refresh_from_manager()

        enable_flag_value = self.param(
            "biosphere.bvoc.isoprene_synthase.enable_co2_inhibition",
            units="dimensionless",
        )
        self.enable_co2_inhibition = bool(_scalar_from_value(enable_flag_value))

    def _held_coefficient(self, param_path: str, units: str) -> Tensor:
        """Resolve one ParameterManager coefficient in the form this Functor holds."""
        value = self.param(param_path, units=units)
        # A learnable coefficient is the ParameterManager-owned binding
        # (architecture SSOT sections 4.2 and 4.4.2): it is held by reference
        # with the dtype and device of its owner, and registration relocates it
        # to the device of this Functor. A converted copy would hide it from the
        # registration and checkpoint owners. Every other value is cached as a
        # float32 tensor on the device of this Functor.
        if self.manager_binding_key(value) is None:
            value = self.to_tensor(value, dtype=torch.float32, device=self.device)
        return value

    def _get_cached_parameter(self, name: str) -> Tensor:
        if name not in self._param_cache:
            self._param_cache[name] = self._held_coefficient(*self._param_specs[name])
        # FunctorBase._ensure_tensor returns an owned binding itself when its dtype
        # and device match, otherwise a differentiable converted view of it.
        return self._ensure_tensor(self._param_cache[name], dtype=torch.float32)

    def _get_mep_parameter(self, param_name: str, units: str, namespace: str) -> Tensor:
        """Parameter getter of the MEP helper: a coefficient held by this Functor.

        The helper is no Functor, so a binding kept only in its cache would be
        invisible to held_manager_parameters(). The value is resolved once per
        refresh, kept in _mep_param_cache under its ParameterManager key and
        handed to the helper unchanged, which aligns it at each use.
        """
        key = f"{namespace}.{param_name}"
        if key not in self._mep_param_cache:
            self._mep_param_cache[key] = self._held_coefficient(key, units)
        return self._mep_param_cache[key]

    def __getattr__(self, name: str) -> Tensor:
        param_specs = object.__getattribute__(self, "_param_specs")
        if name in param_specs:
            return self._get_cached_parameter(name)
        return super().__getattr__(name)
        
    def get_dependencies(self) -> list:
        """Return the producing Functor required by this mechanism."""
        return self._merge_declared_dependencies(["farquhar_photosynthesis"])

    def get_provides(self) -> list:
        """Return the exact canonical runtime outputs."""
        return list(self.metadata.output_variables)
        
    def _register_parameter_change_callback(self) -> None:
        manager = self.get_parameter_manager()
        if manager is None:
            return

        def _callback(key: str, old_value: Any, new_value: Any) -> None:
            if self._is_relevant_parameter(key):
                self._refresh_cached_parameters()

        manager.add_change_callback(_callback)
        self._parameter_change_callback = _callback

    def _is_relevant_parameter(self, key: str) -> bool:
        prefixes = (
            "biosphere.bvoc.",
            "biosphere.photosynthesis.",
        )
        return any(key.startswith(prefix) for prefix in prefixes)

    def _refresh_cached_parameters(self) -> None:
        logger.debug("Refreshing cached parameters for %s after ParameterManager update", self.name)
        self._load_isoprene_synthase_parameters(init=False)

    def __del__(self) -> None:
        # Deregister the parameter-change callback. The manager removes it only when it
        # is registered (relic.parameters.events.remove_change_callback), so a repeated
        # call is a no-op; an error of the manager is not passed over (in a finaliser
        # Python reports it through sys.unraisablehook).
        manager = self.get_parameter_manager()
        if manager and getattr(self, "_parameter_change_callback", None):
            manager.remove_change_callback(self._parameter_change_callback)
        
    def forward(
        # References: Lichtenthaler (1999); Rasulov et al. (2009); Morfopoulos et al. (2014);
        #   existing RELIC canopy-support bridge.
        # Mathematical expression: L_eff=(1-exp(-k*LAI))/k; leaf drivers=ground drivers/L_eff where
        #   L_eff>0; E_leaf=V_IS(T)*DMAPP/(K_DMAPP+DMAPP)*f_CO2; E_ground=E_leaf*L_eff.
        # Parameter meanings: Gross photosynthesis and electron transport supply substrate/energy
        #   proxies; DMAPP is a flux-derived concentration proxy; CO2 inhibition is the unclamped
        #   K_CO2/(K_CO2+Ca) proxy.
        # Units: Internal rates [umol m-2 leaf s-1]; public emission [mol m-2 ground s-1]; carbon
        #   allocation [kg C m-2 ground s-1].
        # Domain/constraints: Zero effective leaf area gives zero leaf drivers; air_temperature is
        #   the declared foliage-temperature proxy. No prognostic DMAPP pool or complete
        #   intracellular carbon closure is claimed.
        # Sign convention: Emission and equivalent plant-carbon debit are positive outward rates;
        #   instantaneous outputs use REPLACE semantics.
        self,
        state: Dict[str, Union[Tensor, float]],
        time: float,
        dt: float,
        **kwargs
    ) -> Dict[str, UpdateItem]:
        """
        Forward pass through mechanistic isoprene synthesis.
        
        Parameters
        ----------
        state : Dict[str, Union[Tensor, float]]
            Input state containing:
            - air_temperature: Grid-cell air-temperature proxy [K]
            - gross_photosynthesis: GPP [umol CO2 m-2 s-1]
            - electron_transport_rate: J [umol e- m-2 s-1]
            - atmospheric_co2_ppm: Ambient CO2 proxy [umol mol-1]
            - leaf_area_index: Leaf area per ground area [m2 leaf m-2 ground]
        time : float
            Current time
        dt : float
            Time step
            
        Returns
        -------
        Dict[str, Tensor]
            Isoprene synthesis results
        """
        # Extract inputs
        air_temperature = self._extract_tensor(state, 'air_temperature')
        gpp = self._extract_tensor(state, 'gross_photosynthesis')
        j_rate = self._extract_tensor(state, 'electron_transport_rate')
        atmospheric_co2_ppm = self._extract_tensor(
            state,
            'atmospheric_co2_ppm',
        )
        leaf_area_index = self._extract_tensor(state, "leaf_area_index")

        # References: Monsi and Saeki (1953); de Pury and Farquhar (1997).
        # Mathematical expressions:
        # - L_eff = (1 - exp(-k * LAI)) / k.
        # - A_leaf = A_ground / L_eff and J_leaf = J_ground / L_eff for
        #   L_eff > 0; both leaf fluxes are zero when L_eff = 0.
        # Parameter meanings:
        # - k is the shared canopy light extinction coefficient [1].
        # - LAI is leaf area per ground area [m2 leaf m-2 ground].
        # - L_eff is effective illuminated leaf area per ground area
        #   [m2 leaf m-2 ground].
        effective_leaf_area = beer_lambert_effective_leaf_area(
            leaf_area_index,
            self.canopy_light_extinction,
        )
        positive_leaf_area = effective_leaf_area > 0
        safe_leaf_area = torch.where(
            positive_leaf_area,
            effective_leaf_area,
            torch.ones_like(effective_leaf_area),
        )
        gpp_leaf = torch.where(
            positive_leaf_area,
            gpp / safe_leaf_area,
            torch.zeros_like(gpp),
        )
        j_rate_leaf = torch.where(
            positive_leaf_area,
            j_rate / safe_leaf_area,
            torch.zeros_like(j_rate),
        )

        # References: Lichtenthaler (1999); Vickers et al. (2009).
        # Mathematical expressions:
        # - J_G3P,leaf = f_G3P * A_leaf.
        # - J_Pyr,leaf = f_Pyr * A_leaf.
        # Parameter meanings:
        # - f_G3P and f_Pyr are photosynthate allocation fractions [1].
        # - A_leaf is gross photosynthesis [umol CO2 m-2 leaf s-1].
        # Calculate photosynthetic substrates
        # G3P from photosynthesis (simplified allocation)
        g3p_flux_leaf = gpp_leaf * self.f_g3p_from_photosynthesis
        g3p_concentration = (
            g3p_flux_leaf * self.g3p_flux_to_concentration
        )
        
        # Pyruvate from photosynthesis
        pyruvate_flux_leaf = (
            gpp_leaf * self.f_pyruvate_from_photosynthesis
        )
        pyruvate_concentration = (
            pyruvate_flux_leaf * self.pyruvate_flux_to_concentration
        )
        
        # References: Vickers et al. (2009); Rasulov et al. (2009).
        # Mathematical expressions:
        # - A_ATP,leaf = nu_ATP,J * J_leaf.
        # - A_NADPH,leaf = nu_NADPH,J * J_leaf.
        # Parameter meanings:
        # - nu_ATP,J and nu_NADPH,J are energy yields per electron flux [1].
        # - J_leaf is electron transport [umol e- m-2 leaf s-1].
        # Energy availability from electron transport
        # Approximate ATP and NADPH from electron transport rate
        atp_availability = (
            j_rate_leaf * self.atp_per_electron_transport
        )
        nadph_availability = (
            j_rate_leaf * self.nadph_per_electron_transport
        )
        
        # MEP pathway flux to DMAPP
        mep_results = self.mep_pathway(
            g3p_concentration, pyruvate_concentration,
            atp_availability, nadph_availability, air_temperature
        )
        
        dmapp_flux_leaf = mep_results['dmapp_flux']
        
        # Isoprene synthase activity
        v_max_is_temp = q10_temperature_response(
            self.v_max_is,
            air_temperature,
            self.q10_is,
            self.mep_pathway.temp_ref,
        )
        
        # DMAPP concentration (simplified steady-state assumption)
        dmapp_concentration = (
            dmapp_flux_leaf * self.dmapp_flux_to_concentration
        )
        
        # Isoprene synthase rate
        isoprene_synthase_rate_leaf = michaelis_menten_response(
            dmapp_concentration, v_max_is_temp, self.k_dmapp
        )
        
        # References: RELIC reduced hyperbolic proxy of the CO2 inhibition of isoprene
        #   emission reported by Monson and Fall (1989) and Wilkinson et al. (2009); the
        #   form and K_CO2 are model defaults not verified in the cited literature.
        # Mathematical expression: f_CO2 = K_CO2 / (K_CO2 + C_a) when CO2 inhibition is
        #   enabled, otherwise f_CO2 = 1.
        # Parameter meanings: K_CO2 is the half-inhibition constant and C_a the ambient CO2
        #   mixing ratio proxy.
        # Units: K_CO2, C_a [umol mol-1]; f_CO2 [1].
        # Domain/constraints: C_a is finite and non-negative (validated by _extract_tensor)
        #   and K_CO2 > 0, so f_CO2 lies in (0, 1]; no floor or ceiling is applied.
        # Sign convention: f_CO2 is a non-negative multiplier of the synthase rate.
        if self.enable_co2_inhibition:
            co2_inhibition_factor = self.k_co2_inhibition / (
                self.k_co2_inhibition + atmospheric_co2_ppm
            )
        else:
            co2_inhibition_factor = torch.ones_like(atmospheric_co2_ppm)
        
        # References: Guenther et al. (2012); de Pury and Farquhar (1997).
        # Mathematical expressions:
        # - F_iso,leaf = J_IS,leaf * f_CO2.
        # - F_ground = F_leaf * L_eff.
        # Parameter meanings:
        # - f_CO2 is the dimensionless ambient-CO2 inhibition factor.
        # - L_eff is effective leaf area per ground area.
        # - Public rates are returned per ground area.
        isoprene_emission_flux_leaf = (
            isoprene_synthase_rate_leaf * co2_inhibition_factor
        )
        isoprene_emission_flux = (
            isoprene_emission_flux_leaf * effective_leaf_area
        )
        g3p_flux = g3p_flux_leaf * effective_leaf_area
        pyruvate_flux = pyruvate_flux_leaf * effective_leaf_area
        dmapp_flux = dmapp_flux_leaf * effective_leaf_area
        isoprene_synthase_rate = (
            isoprene_synthase_rate_leaf * effective_leaf_area
        )
        atp_demand = (
            mep_results["atp_demand"] * effective_leaf_area
        )
        nadph_demand = (
            mep_results["nadph_demand"] * effective_leaf_area
        )
        
        # References:
        # - Guenther et al. (2012), MEGAN2.1.
        # - Rasulov et al. (2009), energy control of leaf isoprene emission.
        # Mathematical expressions:
        # - F_iso,mol = convert(F_iso,umol) [mol m-2 s-1].
        # - F_iso,mass = F_iso,umol * MW_iso [ug C5H8 m-2 s-1].
        # - F_C = convert(F_iso,mol * MW_iso * f_C, g C -> kg C).
        # Parameter meanings:
        # - MW_iso is the isoprene molecular weight [g mol-1].
        # - f_C is the carbon mass fraction of C5H8 [1].
        # Units and route:
        # - isoprene_emission is the sole atmosphere-facing molar port.
        # - mass emission is diagnostic; carbon allocation is a non-negative
        #   open biosphere output magnitude consumed by the plant carbon ledger.
        isoprene_emission_mol = unit_converter.convert(
            isoprene_emission_flux,
            variable_name="molar_flux",
            source_units="umol m-2 s-1",
            target_units="mol m-2 s-1",
        )
        isoprene_emission_mass = isoprene_emission_flux * self.isoprene_mw
        carbon_mass_flux = unit_converter.convert(
            isoprene_emission_mol
            * self.isoprene_mw
            * self.carbon_fraction_isoprene,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        )

        # References: Lichtenthaler (1999); Rasulov et al. (2009); Grmela and Ottinger (1997).
        # Mathematical expression: x_dot = N_h r_h for MEP pathway flux, isoprene
        # synthase activity, molar emission, mass emission, and carbon-allocation
        # cochains.
        # Parameter meanings: N_h is the typed isoprene-output map; r_h contains
        # ParameterManager-backed MEP, synthase, emission, and carbon-allocation
        # rates.
        isoprene_network = advance_reaction_network(
            name=f"{self.name}.isoprene_synthesis_reaction_network",
            species=(
                "isoprene_emission",
                "bvoc_carbon_allocation_flux",
                "diagnostics.bvoc.isoprene.emission_rate_molar",
                "diagnostics.bvoc.isoprene.emission_rate_mass",
                "diagnostics.bvoc.isoprene.dmapp_concentration",
                "diagnostics.bvoc.isoprene.mep_pathway_flux",
                "diagnostics.bvoc.isoprene.synthase_activity",
            ),
            reactions=(
                "isoprene_molar_emission",
                "isoprene_carbon_allocation",
                "isoprene_umol_emission",
                "isoprene_mass_emission",
                "dmapp_pool_proxy",
                "mep_pathway_flux",
                "isoprene_synthase_activity",
            ),
            stoichiometric_N=torch.eye(7, device=isoprene_emission_flux.device, dtype=isoprene_emission_flux.dtype),
            rates=(
                isoprene_emission_mol,
                carbon_mass_flux,
                isoprene_emission_flux,
                isoprene_emission_mass,
                dmapp_concentration,
                dmapp_flux,
                isoprene_synthase_rate,
            ),
            open_system_route="port",
        )

        values_and_units = {
            "isoprene_emission": (
                isoprene_network.species_rate("isoprene_emission"),
                "mol m-2 s-1",
            ),
            "bvoc_carbon_allocation_flux": (
                isoprene_network.species_rate("bvoc_carbon_allocation_flux"),
                "kg C m-2 s-1",
            ),
            "diagnostics.bvoc.isoprene.emission_rate_molar": (
                isoprene_network.species_rate(
                    "diagnostics.bvoc.isoprene.emission_rate_molar"
                ),
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.isoprene.emission_rate_mass": (
                isoprene_network.species_rate(
                    "diagnostics.bvoc.isoprene.emission_rate_mass"
                ),
                "ug m-2 s-1",
            ),
            "diagnostics.bvoc.isoprene.dmapp_concentration": (
                isoprene_network.species_rate(
                    "diagnostics.bvoc.isoprene.dmapp_concentration"
                ),
                "umol m-3",
            ),
            "diagnostics.bvoc.isoprene.mep_pathway_flux": (
                isoprene_network.species_rate(
                    "diagnostics.bvoc.isoprene.mep_pathway_flux"
                ),
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.isoprene.synthase_activity": (
                isoprene_network.species_rate(
                    "diagnostics.bvoc.isoprene.synthase_activity"
                ),
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.isoprene.co2_inhibition_factor": (
                co2_inhibition_factor,
                "1",
            ),
            "diagnostics.bvoc.isoprene.energy_limitation_factor": (
                mep_results["energy_limitation"],
                "1",
            ),
            "diagnostics.bvoc.isoprene.g3p_availability": (
                g3p_flux,
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.isoprene.pyruvate_availability": (
                pyruvate_flux,
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.isoprene.atp_demand": (
                atp_demand,
                "umol m-2 s-1",
            ),
            "diagnostics.bvoc.isoprene.nadph_demand": (
                nadph_demand,
                "umol m-2 s-1",
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
        """
        Provide the non-negative carbon allocation for the conservation ledger.

        References:
            - Guenther et al. (2012) MEGAN 2.1
            - Koppmann (2007) Volatile Organic Compounds in the Atmosphere
        Mathematical expressions:
            m_C = F_iso * MW_iso * (n_C * M_C / MW_iso) * 1e-3
            where F_iso is isoprene molar flux [mol m-2 s-1],
                  MW_iso is molecular weight [g mol-1],
                  n_C = 5 is carbon atoms in C5H8,
                  M_C = 12 g mol-1 is molar mass of carbon.
        Parameter meanings:
            isoprene_mw: molecular weight of isoprene from parameter DB [g mol-1]
        """
        del state
        emission_update = updates.get("isoprene_emission")
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
        carbon_allocation = carbon_allocation.to(
            device=self.device,
            dtype=torch.float32,
        )

        return {
            "bvoc_carbon_allocation_flux": carbon_allocation,
            "bvoc_isoprene_emission_mol": emission_mol,
        }

    def check_conservation_laws(
        self,
        input_state: Dict[str, Tensor],
        output_updates: Dict[str, Union[Tensor, UpdateItem]],
    ) -> Dict[str, float]:
        """
        Audit the open isoprene carbon-boundary mapping.

        References:
            - Guenther et al. (2012), MEGAN2.1 emission-flux convention.
            - Lichtenthaler (1999), carbon stoichiometry of the MEP pathway.
        Mathematical expression:
            r_C = F_C,alloc - convert(F_iso * MW_iso * f_C, g C, kg C)
            carbon_boundary = ||r_C||_inf.
        Parameter meanings:
            - F_iso: canonical isoprene molar flux [mol m-2 s-1].
            - MW_iso: isoprene molecular weight [g mol-1].
            - f_C: carbon mass fraction of C5H8 [1].
            - F_C,alloc: positive plant-carbon allocation [kg C m-2 s-1].

        This residual verifies two representations of one open boundary port. It
        does not claim a closed local or global plant-carbon stock balance.
        """
        # References: Rasulov et al. (2009); declared isoprene molecular composition and SI
        #   conversion.
        # Mathematical expression: r=F_C_alloc-convert(E_mol*MW_isoprene*fC_isoprene,g C to kg C);
        #   report=max(abs(r)).
        # Parameter meanings: E_mol is the canonical atmosphere port; MW and fC come from the
        #   existing parameter owner.
        # Units: E_mol [mol m-2 s-1]; residual [kg C m-2 s-1]; MW [g mol-1].
        # Domain/constraints: Both ports must be present, nonempty, finite and shape-compatible;
        #   this checks the emission-carbon map, not hidden substrate stocks.
        # Sign convention: Positive allocation is plant-carbon export and the reported residual
        #   magnitude is nonnegative.
        del input_state

        def _require_output(key: str) -> Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing canonical output '{key}'.")
            update = output_updates[key]
            value = update.value if isinstance(update, UpdateItem) else update
            if not isinstance(value, Tensor):
                raise TypeError(f"Canonical output '{key}' must be a torch.Tensor")
            if value.numel() == 0 or not torch.isfinite(value).all():
                raise ValueError(
                    f"Canonical output '{key}' must be non-empty and finite"
                )
            return value

        emission_mol = _require_output("isoprene_emission")
        carbon_allocation = _require_output("bvoc_carbon_allocation_flux")
        if emission_mol.shape != carbon_allocation.shape:
            raise ValueError(
                "Canonical outputs 'isoprene_emission' and "
                "'bvoc_carbon_allocation_flux' must have identical shapes"
            )

        molecular_weight = self.isoprene_mw.to(
            device=emission_mol.device,
            dtype=emission_mol.dtype,
        )
        carbon_fraction = self.carbon_fraction_isoprene.to(
            device=emission_mol.device,
            dtype=emission_mol.dtype,
        )
        expected_carbon_allocation = unit_converter.convert(
            emission_mol * molecular_weight * carbon_fraction,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        )
        residual = carbon_allocation.to(
            device=emission_mol.device,
            dtype=emission_mol.dtype,
        ) - expected_carbon_allocation
        return {
            "carbon_boundary": float(
                torch.max(torch.abs(residual)).detach().cpu().item()
            )
        }
        
    def _extract_tensor(
        self,
        state: Dict[str, Union[Tensor, float]],
        key: str,
    ) -> Tensor:
        """Extract and validate one canonical state port."""
        if key not in state:
            raise ValueError(f"Required input '{key}' not found in state")
        value = state[key]

        if not isinstance(value, Tensor):
            raise TypeError(f"Required input '{key}' must be a torch.Tensor")

        value = value.to(device=self.device)
        if not torch.isfinite(value).all():
            raise ValueError(f"Required input '{key}' must contain only finite values")
        if key == "air_temperature":
            if torch.any(value <= 0):
                raise ValueError(
                    "Required input 'air_temperature' must be positive in K"
                )
        elif torch.any(value < 0):
            raise ValueError(f"Required input '{key}' must be non-negative")

        return value


# Export classes
__all__ = [
    'MEPPathwayModel',
    'IsopreneSynthesis'
]

