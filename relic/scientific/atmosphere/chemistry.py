"""
Atmospheric Chemistry Process Module for RELIC

This module implements reduced-form atmospheric chemistry calculations using
a box model approach for reactive species including NOx-O3 cycle and
BVOC interactions. Provides chemical species concentrations and reaction
rates for coupling with ecosystem BVOC emissions and air quality analysis.

Physical Framework:
    Box model assumption: well-mixed atmospheric layer
    Temperature-dependent kinetics: k(T) = k_ref * exp(Ea/R * (1/T_ref - 1/T))
    Photolysis rates: J = J_max * f_light * f_zenith_angle
    Chemical steady-state approximation for short-lived species

Mathematical Expressions:
    1. NOx-O3 cycle:
       NO2 + hnu -> NO + O   (J_NO2)
       O + O2 + M -> O3 + M
       NO + O3 -> NO2 + O2  (k_NO_O3)
    2. BVOC oxidation (surrogate):
       BVOC + OH -> products (k_BVOC_OH)
       BVOC + O3 -> products (k_BVOC_O3)
    3. Temperature dependence:
       k(T) = A * exp(-Ea/(R*T))
    4. Photolysis parameterization:
       J_NO2 = (1 + alpha) * (B1 * G + B2 * G^2) * R_NO2(T), G = SW_down (Trebs et al. 2009)
       J_O1D = (1 + alpha) * (B1 * G + B2 * G^2) * rho(theta) * R_O1D(T)
       R_X = 1 (mode none) or the spectral factor (mode spectral_quantum_yield)
    5. Primary OH production (steady-state OH diagnostic):
       O3 + hnu -> O(1D) + O2; O(1D) + H2O -> 2 OH; O(1D) + N2/O2 -> O(3P)
       P_OH = y f J_O1D [O3], f = k_H2O[H2O] / (k_H2O[H2O] + k_N2[N2] + k_O2[O2])

References:
    - Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics
    - Finlayson-Pitts, B.J. & Pitts, J.N. (2000). Chemistry of the Upper and Lower Atmosphere
    - Atkinson, R. & Arey, J. (2003). Chemical reviews of atmospheric oxidation mechanisms
    - Jenkin, M.E. & Clemitshaw, K.C. (2000). Ozone and other secondary photochemical pollutants
    - Warneck, P. & Williams, J. (2012). The Atmospheric Chemist's Companion (Tables 6.2-6.13, 7.1)
    - Matsumi, Y. et al. (2002). J. Geophys. Res. 107(D3), 4024 (O(1D) quantum yields)
    - Saunders, S.M. et al. (2003). Atmos. Chem. Phys. 3, 161-180 (MCM photolysis parameters)
    - Bodhaine, B.A. et al. (1999). J. Atmos. Oceanic Technol. 16, 1854-1861 (Rayleigh optical depth)

Scientific Validation:
    - Mass conservation: d[X]/dt = P - L (production - loss)
    - Photostationary state: [O3] ? J_NO2 * [NO2] / (k_NO_O3 * [NO])
    - Temperature dependence: higher T increases reaction rates
    - Light dependence: photolysis scales with radiation

Units (UDUNITS-compatible metadata):
    - Concentrations: mixing ratio [mol mol-1] (CF-1.8 standard)
    - Reaction rates: [molecules cm-3 s-1] or [mol mol-1 s-1]
    - Rate constants: [cm^3 molecule-1 s-1]
    - Photolysis rates: [s-1]
    - Temperature: K
"""

import torch
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple
import logging

from ...core.functor import (
    PhysicalFieldFunctor,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    DependencyList,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from ...core.functor.functor_bridge_utils import NamespaceDefaultMixin
from ...computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from ...computation.operators.iphs import (
    IPHSStructureSpec,
    PortSpecIPHS,
    advance_reaction_network,
)
from ...computation.solvers import SolverFactory, SolverStatus
from ...core.abstractions import StateDict, FluxDict
from ...core.state import UpdateItem, UpdateMode
from ...utils.unified_response_functions import (
    arrhenius_temperature_response,
    validate_positive_values,
    validate_temperature_range,
)
from ...utils.physical_constants import get_physical_constant


logger = logging.getLogger(__name__)


class ChemistryFunctor(NamespaceDefaultMixin, PhysicalFieldFunctor):
    """
    Reduced-form atmospheric chemistry functor using box model approach.
    
    Implements basic photochemical processes including NOx-O3 cycle and
    BVOC oxidation reactions with temperature-dependent kinetics and
    photolysis rate parameterizations. Designed for ecosystem-atmosphere
    coupling with focus on BVOC interactions.
    
    Key Features:
        - Temperature-dependent reaction rates (Arrhenius)
        - Light-dependent photolysis rates
        - NOx-O3 photostationary cycle
        - BVOC oxidation surrogate mechanisms
        - Chemical steady-state approximations

    Process Category: PHYSICAL_FIELD (atmospheric chemical field)
    Execution Phase: MAIN_PROCESSES
    
    Input Variables:
        - air_temperature [K]
        - atmospheric_pressure [Pa] (required; contract alias air_pressure is not read)
        - shortwave_down [W m-2]
        - cos_zenith [dimensionless] (required: J_NO2 geometry and J(O1D)/J(NO2) ratio)
        - no_mixing_ratio [mol mol-1] (CF-1.8 standard)
        - no2_mixing_ratio [mol mol-1] (CF-1.8 standard)
        - o3_mixing_ratio [mol mol-1] (CF-1.8 standard)
        - isoprene_mixing_ratio [mol mol-1] (optional, from BVOC)
        - co_mixing_ratio [mol mol-1] (OH sink)
        - ch4_mixing_ratio [mol mol-1] (OH sink)
        - specific_humidity [kg kg-1] (O(1D) + H2O fraction of the OH source)
    
    Output Variables:
        - no_mixing_ratio [mol mol-1 per step]
        - no2_mixing_ratio [mol mol-1 per step]
        - o3_mixing_ratio [mol mol-1 per step]
        - isoprene_mixing_ratio [mol mol-1 per step] when present
        - diagnostics.chemistry.* [declared diagnostic units]
    
    Coupling Interfaces:
        - BVOC emission modules (consume isoprene_mixing_ratio)
        - Radiation module (consume shortwave_down)
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    parameter_namespace = "atmosphere.chemistry"
    
    PHYSICAL_BOUNDARIES = ("open_mechanism", "closed_constant_volume")
    # Declared species layout of the open mechanism: the state cochain of the field
    # audit (declared forms, Hodge contract and dissipative structure).
    OPEN_MECHANISM_SPECIES = (
        "no_mixing_ratio",
        "no2_mixing_ratio",
        "o3_mixing_ratio",
        "isoprene_mixing_ratio",
    )
    # Explicit config overrides precede the ParameterManager (NamespaceDefaultMixin).
    parameter_lookup_source_order = ("config", "manager")
    CLOSED_BOX_SPECIES = (
        "box_no_amount",
        "box_no2_amount",
        "box_o3_amount",
        "box_o2_amount",
    )
    CLOSED_BOX_STATE = CLOSED_BOX_SPECIES + ("box_internal_energy",)
    CLOSED_BOX_INPUTS = CLOSED_BOX_STATE + ("box_volume",)
    CLOSED_BOX_DIAGNOSTICS = (
        "diagnostics.chemistry.box_temperature",
        "diagnostics.chemistry.box_pressure",
        "diagnostics.chemistry.box_entropy",
        "diagnostics.chemistry.box_entropy_production_rate",
        "diagnostics.chemistry.box_reaction_affinity",
        "diagnostics.chemistry.box_reaction_rate",
        "diagnostics.chemistry.box_reaction_power",
        "diagnostics.chemistry.box_rate_constant_no_o3",
    )
    # Override whitelist and runtime short names of each physical boundary: a
    # config "parameters" override names one of these keys of the namespace
    # atmosphere.chemistry, and the process reads it by that short name.
    # metadata.parameters declares the full ParameterManager keys.
    OPEN_MECHANISM_PARAMETERS = {
        'j_no2_irradiance_linear_coefficient': (
            'Trebs et al. (2009) B1 of J_NO2 = (1 + alpha)(B1 G + B2 G^2) [W-1 m2 s-1]'
        ),
        'j_no2_irradiance_quadratic_coefficient': (
            'Trebs et al. (2009) B2 of J_NO2 = (1 + alpha)(B1 G + B2 G^2) [W-2 m4 s-1]'
        ),
        'uv_a_surface_albedo': 'UV-A surface albedo alpha of the J_NO2 relation [dimensionless]',
        'photolysis_temperature_mode': (
            'Air-temperature dependence of J_NO2 and J(O1D), none or spectral_quantum_yield [dimensionless]'
        ),
        'mcm_o1d_photolysis_l': 'MCM J1 (O3 -> O(1D)) parameter l [s-1]',
        'mcm_o1d_photolysis_m': 'MCM J1 (O3 -> O(1D)) parameter m [dimensionless]',
        'mcm_o1d_photolysis_n': 'MCM J1 (O3 -> O(1D)) parameter n [dimensionless]',
        'mcm_no2_photolysis_l': 'MCM J4 (NO2 -> NO + O(3P)) parameter l [s-1]',
        'mcm_no2_photolysis_m': 'MCM J4 (NO2 -> NO + O(3P)) parameter m [dimensionless]',
        'mcm_no2_photolysis_n': 'MCM J4 (NO2 -> NO + O(3P)) parameter n [dimensionless]',
        'chemistry_temperature_q10': (
            'Positive mobility factor used only by the level-3 Hodge audit [dimensionless]'
        ),
        'temperature_min': 'Minimum temperature for chemistry validity [K]',
        'temperature_max': 'Maximum temperature for chemistry validity [K]',
        'k_no_o3_298': 'NO + O3 reaction rate coefficient [cm3 molecule-1 s-1]',
        'k_no_o3_activation_energy': 'Activation energy for NO + O3 [J mol-1]',
        'k_isoprene_oh_298': 'Isoprene + OH reaction rate [cm3 molecule-1 s-1]',
        'k_isoprene_oh_activation_energy': 'Activation energy for isoprene + OH [J mol-1]',
        'k_isoprene_o3_298': 'Isoprene + O3 reaction rate [cm3 molecule-1 s-1]',
        'k_isoprene_o3_activation_energy': 'Activation energy for isoprene + O3 [J mol-1]',
        'k_isoprene_no3_298': 'Isoprene + NO3 reaction rate [cm3 molecule-1 s-1]',
        'k_isoprene_no3_activation_energy': 'Activation energy for isoprene + NO3 [J mol-1]',
        'k_oh_co_298':'OH + CO reaction rate [cm3 molecule-1 s-1]',
        'k_oh_co_activation_energy': 'Activation energy for OH + CO [J mol-1]',
        'k_oh_ch4_298': 'OH + CH4 reaction rate [cm3 molecule-1 s-1]',
        'k_oh_ch4_activation_energy': 'Activation energy for OH + CH4 [J mol-1]',
        'k_oh_no2_298': 'OH + NO2 reaction rate [cm3 molecule-1 s-1]',
        'k_oh_no2_activation_energy': 'Activation energy for OH + NO2 [J mol-1]',
        'minimum_concentration': (
            'Positive metric scale used only by the level-3 Hodge audit [mol mol-1]'
        ),
        'k_o1d_h2o_298': 'O(1D) + H2O -> 2 OH reaction rate [cm3 molecule-1 s-1]',
        'k_o1d_h2o_activation_energy': 'Activation energy for O(1D) + H2O [J mol-1]',
        'k_o1d_n2_298': 'O(1D) + N2 quenching rate [cm3 molecule-1 s-1]',
        'k_o1d_n2_activation_energy': 'Activation energy for O(1D) + N2 [J mol-1]',
        'k_o1d_o2_298': 'O(1D) + O2 quenching rate [cm3 molecule-1 s-1]',
        'k_o1d_o2_activation_energy': 'Activation energy for O(1D) + O2 [J mol-1]',
        'o1d_water_oh_yield': 'OH molecules formed per O(1D) + H2O reaction [dimensionless]',
        'k_ro2_no_298': 'RO2 + NO -> RO + NO2 reaction rate coefficient [cm3 molecule-1 s-1]',
        'k_ro2_no_activation_energy': 'Activation energy for RO2 + NO [J mol-1]',
        'k_ro2_termination_loss': 'Effective first-order non-NO loss rate of RO2 radicals [s-1]',
        'isoprene_ro2_no2_stoichiometric_yield': (
            'Stoichiometric NO-to-NO2 conversions per OH + isoprene RO2 + NO branch [dimensionless]'
        ),
        'isoprene_carbon_atoms': 'Carbon atoms per molecule of isoprene (C5H8) [dimensionless]',
    }
    # Text-valued (non-numeric) open-mechanism parameters among the short names above.
    OPEN_MECHANISM_TEXT_PARAMETERS = frozenset({'photolysis_temperature_mode'})
    # Implemented photolysis temperature modes; the YAML allowed_values of
    # atmosphere.chemistry.photolysis_temperature_mode declare the same tuple.
    PHOTOLYSIS_TEMPERATURE_MODES = ("none", "spectral_quantum_yield")
    # Full keys of the spectral reference data of the photolysis temperature factor
    # (relic/parameters/database/relic/atmosphere/photolysis_spectra.yaml). They are
    # read from the ParameterManager only and bound once at construction.
    PHOTOLYSIS_NAMESPACE = "atmosphere.chemistry.photolysis"
    OPEN_MECHANISM_MANAGER_PARAMETERS = {
        f"atmosphere.chemistry.photolysis.{short_name}": description
        for short_name, description in {
            'solar_flux_wavelength': 'Centres of the 1-nm solar flux intervals [um]',
            'solar_photon_flux': 'Extraterrestrial solar photon flux [photons cm-2 s-1 um-1]',
            'no2_cross_section_wavelength': 'Centres of the 5-nm NO2 cross-section intervals [um]',
            'no2_cross_section': 'NO2 absorption cross section at 298 K [cm2]',
            'no2_quantum_yield_wavelength': 'NO2 quantum-yield wavelengths and integration grid [um]',
            'no2_quantum_yield_high_temperature': 'NO2 -> NO + O(3P) quantum yield at 298 K [dimensionless]',
            'no2_quantum_yield_low_temperature': 'NO2 -> NO + O(3P) quantum yield at 248 K [dimensionless]',
            'no2_quantum_yield_temperature_high': 'Temperature of the upper NO2 yield table [K]',
            'no2_quantum_yield_temperature_low': 'Temperature of the lower NO2 yield table [K]',
            'o3_cross_section_wavelength': 'Wavelengths of the O3 cross sections [um]',
            'o3_cross_section_high_temperature': 'O3 cross section at the upper tabulated temperature [cm2]',
            'o3_cross_section_low_temperature': 'O3 cross section at the lower tabulated temperature [cm2]',
            'o3_cross_section_temperature_high': 'Upper tabulation temperature of each O3 point [K]',
            'o3_cross_section_temperature_low': 'Lower tabulation temperature of each O3 point [K]',
            'o1d_integration_wavelength': 'Integration grid of the O(1D) factor [um]',
            'o1d_quantum_yield_short_wavelength': 'O(1D) yield below 306 nm [dimensionless]',
            'o1d_quantum_yield_long_wavelength': 'O(1D) yield above 328 nm [dimensionless]',
            'matsumi_wavelength_min': 'Lower wavelength limit of the Matsumi expression [um]',
            'matsumi_wavelength_max': 'Upper wavelength limit of the Matsumi expression [um]',
            'matsumi_amplitudes': 'Matsumi amplitudes A1, A2, A3 [dimensionless]',
            'matsumi_band_centers': 'Matsumi band centres x1, x2, x3 [um]',
            'matsumi_band_widths': 'Matsumi band widths w1, w2, w3 [um]',
            'matsumi_offset': 'Matsumi constant C [dimensionless]',
            'matsumi_energy_gap': 'Matsumi energy nu2 of q = exp(-nu2/(R T)) [cm-1]',
            'matsumi_reference_temperature': 'Matsumi normalization temperature [K]',
            'rayleigh_scale': 'Bodhaine Eq. 30 leading factor [dimensionless]',
            'rayleigh_numerator_constant': 'Bodhaine Eq. 30 numerator constant [dimensionless]',
            'rayleigh_numerator_inverse_square': 'Bodhaine Eq. 30 lambda^-2 numerator coefficient [dimensionless]',
            'rayleigh_numerator_square': 'Bodhaine Eq. 30 lambda^2 numerator coefficient [dimensionless]',
            'rayleigh_denominator_inverse_square': 'Bodhaine Eq. 30 lambda^-2 denominator coefficient [dimensionless]',
            'rayleigh_denominator_square': 'Bodhaine Eq. 30 lambda^2 denominator coefficient [dimensionless]',
            'reference_ozone_column': 'Reference vertical ozone column [molecule cm-2]',
            'reference_temperature': 'Reference temperature T_ref of R(T) [K]',
            'matsumi_temperature_min': 'Lower evaluated temperature of the Matsumi expression [K]',
            'matsumi_temperature_max': 'Upper evaluated temperature of the Matsumi expression [K]',
        }.items()
    }
    CLOSED_BOX_PARAMETERS = {
        'k_no_o3_298': 'NO + O3 reaction rate coefficient [cm3 molecule-1 s-1]',
        'k_no_o3_activation_energy': 'Activation energy for NO + O3 [J mol-1]',
        'temperature_min': 'Lower kinetic temperature domain and inversion bracket [K]',
        'temperature_max': 'Upper kinetic temperature domain and inversion bracket [K]',
    }
    # Full keys of other owners that the closed box reads from the ParameterManager
    # only (never overridden): the NASA7 thermodynamic tables, the settings of the
    # temperature inversion solver and the tolerance of the Level 5 runtime-point audit.
    CLOSED_BOX_MANAGER_PARAMETERS = {
        'atmosphere.chemistry.nasa7.model': 'NASA7 model identifier [dimensionless]',
        'atmosphere.chemistry.nasa7.species_order': (
            'Declared NASA7 species order NO, NO2, O3, O2 [dimensionless]'
        ),
        'atmosphere.chemistry.nasa7.temperature_ranges': 'NASA7 fit temperature intervals [K]',
        'atmosphere.chemistry.nasa7.a1': 'NASA7 coefficient a1 by species and interval [dimensionless]',
        'atmosphere.chemistry.nasa7.a2': 'NASA7 coefficient a2 by species and interval [K-1]',
        'atmosphere.chemistry.nasa7.a3': 'NASA7 coefficient a3 by species and interval [K-2]',
        'atmosphere.chemistry.nasa7.a4': 'NASA7 coefficient a4 by species and interval [K-3]',
        'atmosphere.chemistry.nasa7.a5': 'NASA7 coefficient a5 by species and interval [K-4]',
        'atmosphere.chemistry.nasa7.a6': 'NASA7 coefficient a6 by species and interval [K]',
        'atmosphere.chemistry.nasa7.a7': 'NASA7 coefficient a7 by species and interval [dimensionless]',
        'computation.solvers.bracketed_bisection.max_iterations': (
            'Iteration limit of the temperature inversion T(U, n) [dimensionless]'
        ),
        'computation.solvers.bracketed_bisection.rtol': (
            'Relative tolerance of the temperature inversion T(U, n) [dimensionless]'
        ),
        'computation.solvers.bracketed_bisection.atol': (
            'Absolute tolerance of the temperature inversion T(U, n) [dimensionless]'
        ),
        'validation.iphs.runtime_point_tolerance': (
            'Tolerance of the Level 5 runtime-point audit of each emitted step [dimensionless]'
        ),
    }
    # NO + O3 -> NO2 + O2 stoichiometry on (n_NO, n_NO2, n_O3, n_O2, U).
    CLOSED_BOX_STOICHIOMETRY = (-1.0, 1.0, -1.0, 1.0, 0.0)
    # Element ledgers C_N and C_O on the same state order; C nu = 0 for both.
    CLOSED_BOX_ELEMENT_LEDGERS = (
        ("nitrogen_atoms", (1.0, 1.0, 0.0, 0.0, 0.0)),
        ("oxygen_atoms", (1.0, 2.0, 3.0, 2.0, 0.0)),
    )
    CANONICAL_DRIFT_SOURCE = (
        "relic.computation.operators.iphs.IPHSStructureSpec.canonical_drift"
    )

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(
        self,
        name: str = "atmospheric_chemistry",
        *,
        physical_boundary: Literal["open_mechanism", "closed_constant_volume"],
        config: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        """Initialize chemistry functor with reaction parameters.

        ``physical_boundary`` selects the governed process law: the open
        reduced NOx-O3-isoprene mechanism, or the closed adiabatic
        constant-volume NO + O3 box advanced by the GENERIC absorbed-reaction
        drift.  It is a GraphIR selection field, not a parameter.
        """

        if physical_boundary not in self.PHYSICAL_BOUNDARIES:
            raise ValueError(
                "physical_boundary must be one of "
                f"{self.PHYSICAL_BOUNDARIES}, got {physical_boundary!r}."
            )
        # Extract device parameter if provided
        device = kwargs.pop("device", None)
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Chemistry config must be a mapping.")
        config_dict: Dict[str, Any] = dict(config) if config is not None else {}
        parameter_overrides = self._normalize_override_block(config_dict.pop("parameters", None))
        override_names = tuple(
            self.CLOSED_BOX_PARAMETERS
            if physical_boundary == "closed_constant_volume"
            else self.OPEN_MECHANISM_PARAMETERS
        )
        if physical_boundary == "closed_constant_volume":
            dependencies = config_dict.get("dependencies", [])
            if dependencies:
                raise ValueError(
                    "The closed constant-volume box declares no Functor dependencies; "
                    f"got {list(dependencies)!r}."
                )
            metadata = self._build_closed_constant_volume_metadata(name)
        else:
            dependencies = config_dict.get("dependencies")
            metadata = FunctorMetadata(
                name=name,
                version="1.0.0",
                description="Reduced-form atmospheric chemistry using box model approach",
                category=ProcessCategory.PHYSICAL_FIELD,
                execution_phase=ExecutionPhase.MAIN_PROCESSES,
                mathematical_expression=(
                    "d[NO]/dt = J_NO2*[NO2] - k_NO_O3*[NO]*[O3] - r_RO2_NO; "
                    "d[NO2]/dt = k_NO_O3*[NO]*[O3] - J_NO2*[NO2] + r_RO2_NO; "
                    "d[O3]/dt = J_NO2*[NO2] - k_NO_O3*[NO]*[O3] - r_iso_O3; "
                    "d[Ox]/dt = d[O3]/dt + d[NO2]/dt = r_RO2_NO - r_iso_O3; "
                    "r_RO2_NO = gamma_NO2 * beta_NO * r_iso_OH, "
                    "beta_NO = k_RO2_NO(T)[NO] / (k_RO2_NO(T)[NO] + k_loss_RO2); "
                    "Delta[X] = dt * d[X]/dt; "
                    "k(T) = k_298 * exp(Ea/R * (1/T_ref - 1/T)), T_ref = 298.15 K; "
                    "J_NO2 = (1 + alpha) * (B1 * G + B2 * G^2) * R_NO2(T), G = SW (held at G_v = B1/(2|B2|) above); "
                    "J_O1D = (1 + alpha) * (B1 * G + B2 * G^2) * rho(theta) * R_O1D(T), "
                    "rho = (l1/l4) cos(theta)^(m1-m4) exp(-(n1-n4)/cos(theta)); "
                    "R_X = 1 (none) or sum sigma_X phi_X F / sum sigma_X phi_X F at T_ref (spectral_quantum_yield); "
                    "[OH] = y f J_O1D [O3] / (k_OH_CO[CO] + k_OH_CH4[CH4] + k_OH_NO2[NO2] + k_OH_C5H8[C5H8]), "
                    "f = k_H2O[H2O] / (k_H2O[H2O] + k_N2[N2] + k_O2[O2]); "
                    "BVOC + OH -> products (k_BVOC_OH)"
                ),
                references=[
                    "Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics",
                    "Finlayson-Pitts, B.J. & Pitts, J.N. (2000). Chemistry of Upper and Lower Atmosphere",
                    "Atkinson, R. & Arey, J. (2003). Atmospheric degradation of volatile organic compounds",
                    "Jenkin, M.E. & Clemitshaw, K.C. (2000). Ozone and secondary photochemical pollutants",
                    "Warneck, P. & Williams, J. (2012). The Atmospheric Chemist's Companion, Springer",
                    "Matsumi, Y. et al. (2002). J. Geophys. Res. 107(D3), 4024",
                    "Saunders, S.M. et al. (2003). Atmos. Chem. Phys. 3, 161-180",
                    "Bodhaine, B.A. et al. (1999). J. Atmos. Oceanic Technol. 16, 1854-1861",
                    "Jenkin, M.E. et al. (2015). Atmos. Chem. Phys. 15, 11433-11459",
                    "Peeters, J. et al. (2014). J. Phys. Chem. A 118, 8625-8643",
                ],
                parameters={
                    **{
                        f"{self.parameter_namespace}.{short_name}": description
                        for short_name, description in self.OPEN_MECHANISM_PARAMETERS.items()
                    },
                    **self.OPEN_MECHANISM_MANAGER_PARAMETERS,
                },
                input_variables=[
                    'air_temperature',
                    'atmospheric_pressure',
                    'shortwave_down',
                    'cos_zenith',
                    'no_mixing_ratio',
                    'no2_mixing_ratio',
                    'o3_mixing_ratio',
                    'isoprene_mixing_ratio',
                    'co_mixing_ratio',
                    'ch4_mixing_ratio',
                    'specific_humidity',
                ],
                output_variables=[
                    'no_mixing_ratio',
                    'no2_mixing_ratio',
                    'o3_mixing_ratio',
                    'isoprene_mixing_ratio',
                    'diagnostics.chemistry.photolysis_rate_no2',
                    'diagnostics.chemistry.isoprene_oxidation_rate',
                    'diagnostics.chemistry.reaction_no_o3_rate',
                    'diagnostics.chemistry.photolysis_no2_rate',
                    'diagnostics.chemistry.rate_constant_no_o3',
                    'diagnostics.chemistry.shortwave_radiation_used',
                    'diagnostics.chemistry.oh_concentration',
                    'diagnostics.chemistry.air_number_density',
                    'diagnostics.chemistry.ro2_no_conversion_rate',
                    'diagnostics.chemistry.ro2_no_branching_fraction',
                    'diagnostics.chemistry.ox_net_chemical_tendency',
                    'diagnostics.chemistry.carbon_oxidized_rate',
                ],
                optional_output_variables=['isoprene_mixing_ratio'],
                conservation_laws=['nox_family'],
                spatial_dimensions=0,
                temporal_scale=TemporalScale.MINUTE,
                coupling_interfaces=['bvoc_emission', 'radiation'],
                spatial_scale=SpatialScale.GRID_CELL,
                variable_kind=VariableKind.STATE,
            )

        # Initialize FunctorBase
        unknown_names = sorted(set(parameter_overrides).difference(override_names))
        if unknown_names:
            raise ValueError("Unknown chemistry parameter override(s): " + ", ".join(unknown_names))
        clean_config: Dict[str, Any] = {"parameters": parameter_overrides}
        if dependencies is not None:
            clean_config['dependencies'] = dependencies

        super().__init__(name=name, config=clean_config, metadata=metadata, **kwargs)
        self.physical_boundary = physical_boundary
        self._override_parameter_names = override_names
        # Per-call Level 5 runtime-point audit of the closed box; evidence only,
        # never a second prognostic state. Cleared before every evaluation.
        self._last_strict_iphs_audit: Optional[Dict[str, Any]] = None

        if physical_boundary == "closed_constant_volume":
            self.optional_input_variables = []
        else:
            # CO and CH4 are required OH-sink inputs (no background closure); the
            # solar geometry and the humidity are required by the O(1D) OH source.
            self.optional_input_variables = [
                'isoprene_mixing_ratio',
            ]

        self._constants = self._load_constants()
        for parameter_name in self._override_parameter_names:
            self.get_parameter_value(parameter_name, required=True)
        if physical_boundary == "open_mechanism":
            self._photolysis_temperature_mode()
            self._load_photolysis_spectra()

        # Explicit constructor device uses the FunctorBase migration owner so the
        # registered torch modules and the Functor device attribute (consumed by
        # get_parameter_value) move together. Module.to alone leaves the CPU
        # placeholder in place. Registration migration remains the runtime owner.
        if device is not None:
            self.to_device(device)


    @classmethod
    def _build_closed_constant_volume_metadata(cls, name: str) -> FunctorMetadata:
        """Metadata of the closed adiabatic constant-volume NO + O3 box."""

        return FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Closed adiabatic constant-volume NO + O3 -> NO2 + O2 box advanced by the "
                "GENERIC absorbed-reaction drift with ideal-gas NASA7 thermodynamics"
            ),
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "x = (n_NO, n_NO2, n_O3, n_O2, U), fixed V; "
                "T = T(U, n) from sum_i n_i u_i(T) = U; "
                "mu_i = h_i(T) - T s_i(T) + R T ln(p_i / p0), p_i = n_i R T / V; "
                "A = -nu^T mu, nu = (-1, +1, -1, +1, 0); "
                "r = k(T) N_A n_NO n_O3 / (c_v V), k(T) = k_298 exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K; "
                "E = U, S = sum_i n_i (s_i(T) - R ln(p_i / p0)); "
                "L = 0, M = (r T / A) nu nu^T; x_dot = L grad(E) + M grad(S) = nu r; "
                "M grad(E) = 0, dE/dt = 0, dS/dt = grad(S)^T M grad(S) = A r / T >= 0; "
                "Delta x = dt x_dot"
            ),
            references=[
                "Grmela, M. & Ottinger, H.C. (1997). Phys. Rev. E 56, 6620-6632",
                "Ottinger, H.C. (2005). Beyond Equilibrium Thermodynamics, Wiley",
                "Mielke, A. (2011). Nonlinearity 24, 1329-1346",
                "Kondepudi, D. & Prigogine, I. (1998). Modern Thermodynamics, Wiley",
                "McBride, B.J., Gordon, S. & Reno, M.A. (1993). NASA TM-4513",
                "Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics",
                "Atkinson, R. et al. (2004). Atmos. Chem. Phys. 4, 1461-1738",
            ],
            parameters={
                **{
                    f"{cls.parameter_namespace}.{short_name}": description
                    for short_name, description in cls.CLOSED_BOX_PARAMETERS.items()
                },
                **cls.CLOSED_BOX_MANAGER_PARAMETERS,
            },
            input_variables=[str(name) for name in cls.CLOSED_BOX_INPUTS],
            output_variables=list(cls.CLOSED_BOX_STATE + cls.CLOSED_BOX_DIAGNOSTICS),
            conservation_laws=['nitrogen_atoms', 'oxygen_atoms', 'internal_energy'],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.MINUTE,
            coupling_interfaces=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )

    @staticmethod
    def _normalize_override_block(raw: Optional[Any]) -> Dict[str, Any]:
        """Normalize override dictionaries coming from user configuration."""
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError("Chemistry parameter overrides must be a mapping.")
        normalized: Dict[str, Any] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise ValueError("Chemistry override names must be strings.")
            if isinstance(value, dict):
                if "value" not in value:
                    raise ValueError(f"Chemistry override {key} requires an explicit value.")
                if "units" in value and (not isinstance(value["units"], str) or not value["units"].strip()):
                    raise ValueError(f"Chemistry override {key} requires a nonempty unit string.")
                normalized[key] = {"value": value["value"]}
                if "units" in value:
                    normalized[key]["units"] = value["units"]
            else:
                normalized[key] = value
        return normalized

    def _load_constants(self) -> Dict[str, float]:
        """Load atmospheric physical constants from the parameter database."""
        specs: List[Tuple[str, str]] = [
            ("avogadro_number", "relic.physical.constants.universal.avogadro_constant"),
            ("boltzmann_constant", "relic.physical.constants.universal.boltzmann_constant"),
            (
                "cubic_centimeters_per_cubic_meter",
                "relic.physical.constants.conversions.cubic_centimeters_per_cubic_meter",
            ),
            ("planck_constant", "relic.physical.constants.universal.planck_constant"),
            ("speed_of_light", "relic.physical.constants.universal.speed_of_light"),
            ("water_molar_mass", "relic.physical.constants.water.molecular_weight"),
            ("dry_air_molar_mass", "relic.physical.constants.air.molecular_weight_dry"),
            ("dry_air_n2_fraction", "relic.physical.constants.atmosphere.n2_fraction"),
            ("dry_air_o2_fraction", "relic.physical.constants.atmosphere.o2_fraction"),
        ]
        constants: Dict[str, float] = {}
        for attr, key in specs:
            constants[attr] = float(get_physical_constant(key))
        return constants

    def get_dependencies(self) -> DependencyList:
        """
        Chemistry may depend on radiation for photolysis rates.

        Returns:
            DependencyList: Dependencies for photolysis calculations

        Notes:
            Dependencies are architectural decisions, not configurable parameters.
            Photolysis (J_NO2) needs shortwave_down and cos_zenith; the open mechanism
            declares atmospheric_radiation as an optional provider of them.
        """
        # Architectural dependency - not a configurable parameter
        if self.physical_boundary == "closed_constant_volume":
            return self._merge_declared_dependencies([])
        # References: DeclaredDependencies contract (relic/core/dependency_graph.py);
        #   precedent AtmosphericAdvectionDiffusionFunctor (optional atmospheric_turbulence).
        # Mathematical expression: J_NO2 = J_max (SW_down / SW_ref) max(0, cos theta_z) reads
        #   shortwave_down [W m-2] and cos_zenith [1], both required inputs validated in forward.
        # Domain/constraints: atmospheric_radiation is an optional same-step provider; when it is
        #   registered it runs first, otherwise shortwave_down and cos_zenith must be forcing.
        return self._merge_declared_dependencies({"optional": ["atmospheric_radiation"]})

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare atmospheric chemistry field cochains for Phase 8 audit.

        References:
          - Seinfeld and Pandis (2016), box-model atmospheric chemistry.
          - Finlayson-Pitts and Pitts (2000), photochemical field tendencies.
        Mathematical expression:
          x_dot = M_h e_h + G_h u_h with field tendency r_h(C,T,J).
        Parameter meanings:
          C_h is the species mixing-ratio field, M_h stores positive rate
          mobility, and G_h u_h is the radiation-driven photolysis input.
        """

        if self.physical_boundary == "closed_constant_volume":
            return self._closed_box_declared_forms()
        n_species = len(self.OPEN_MECHANISM_SPECIES)
        return (
            CochainSpaceSpec(
                complex_id="atmosphere_chemistry_field",
                degree=0,
                primality="primal",
                n_dofs=n_species,
                units="mol mol-1",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="atmospheric_species_mixing_ratio",
                orientation_convention="positive_concentration",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="atmosphere_chemistry_field",
                degree=0,
                primality="dual",
                n_dofs=n_species,
                units="mol mol-1 s-1",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="atmospheric_chemistry_tendency",
                orientation_convention="production_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return tuple(sorted(self.metadata.parameters))

    def build_hodge_contract(
        self,
        n_dofs: int,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the positive chemistry field mobility Hodge contract.

        References:
          - Seinfeld and Pandis (2016), photolysis and kinetic rate constants.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          tendency_h = star_rxn e_h, star_rxn = diag(J_max Q10) >= 0, with
          J_max = (1 + alpha) B1^2 / (4 |B2|) the vertex value of the Trebs et al.
          (2009) relation J_NO2 = (1 + alpha)(B1 G + B2 G^2).
        Parameter meanings:
          alpha, B1, B2 (uv_a_surface_albedo, j_no2_irradiance_*_coefficient),
          chemistry_temperature_q10 and minimum_concentration scale the positive
          field-tendency mobility used for assembly audit only.
        """

        if self.physical_boundary == "closed_constant_volume":
            raise ValueError(
                "The closed constant-volume box declares no field Hodge; its mobility "
                "M = (r T / A) nu nu^T is state dependent and evaluated in forward."
            )
        n_local = int(n_dofs)
        if n_local < 1:
            raise ValueError("Chemistry Hodge degrees of freedom must be positive.")
        albedo = float(self.get_parameter_value("uv_a_surface_albedo", required=True))
        linear = float(self.get_parameter_value("j_no2_irradiance_linear_coefficient", required=True))
        quadratic = float(self.get_parameter_value("j_no2_irradiance_quadratic_coefficient", required=True))
        maximum_j_no2 = (1.0 + albedo) * linear * linear / (4.0 * abs(quadratic))
        mobility = maximum_j_no2 * float(self.get_parameter_value("chemistry_temperature_q10", required=True))
        values = torch.full(
            (n_local,),
            mobility,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        species_space = CochainSpaceSpec(
            complex_id="atmosphere_chemistry_field",
            degree=0,
            primality="primal",
            n_dofs=n_local,
            units="mol mol-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="atmospheric_species_mixing_ratio",
            orientation_convention="positive_concentration",
        )
        tendency_space = CochainSpaceSpec(
            complex_id="atmosphere_chemistry_field",
            degree=0,
            primality="dual",
            n_dofs=n_local,
            units="mol mol-1 s-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="atmospheric_chemistry_tendency",
            orientation_convention="production_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_chemistry_field_hodge",
            domain=species_space,
            codomain=tendency_space,
            matrix=torch.diag(values),
            parameter_keys=(
                "atmosphere.chemistry.uv_a_surface_albedo",
                "atmosphere.chemistry.j_no2_irradiance_linear_coefficient",
                "atmosphere.chemistry.j_no2_irradiance_quadratic_coefficient",
                "atmosphere.chemistry.chemistry_temperature_q10",
            ),
            metric_keys=("atmosphere.chemistry.minimum_concentration",),
            material_law="atmospheric_chemistry_field_mobility",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build chemistry field Hodge terms for Phase 8 assembly."""

        if self.physical_boundary == "closed_constant_volume":
            return ()
        # The Hodge acts on the declared species layout unless the complex bundle
        # declares its own degrees of freedom.
        bundle_dofs = getattr(complex_bundle, "n_dofs", None) if complex_bundle is not None else None
        n_dofs = len(self.OPEN_MECHANISM_SPECIES) if bundle_dofs is None else bundle_dofs
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a structural species-concentration audit view."""

        if self.physical_boundary == "closed_constant_volume":
            box = self._evaluate_closed_constant_volume_box(state)
            return {
                "storage_functional": "E = U",
                "internal_energy": box["energy"],
                "energy_effort": box["energy_effort"],
            }
        vectors: List[torch.Tensor] = []
        for state_var in self.OPEN_MECHANISM_SPECIES:
            if state_var in state:
                value = state[state_var]
                tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=self.device)
                vectors.append(tensor.to(device=self.device, dtype=torch.float32).reshape(-1))
        if not vectors:
            return {
                "quadratic_storage_audit": "H_audit = 0.5 * C_h.T C_h",
                "effort_audit": "e_h = C_h",
            }
        # References: Phase 8 structural audit convention.
        # Mathematical expression: H_audit = 0.5 * C_h.T C_h.
        # Parameter meanings: C_h is the atmospheric species mixing-ratio field.
        concentration = torch.cat(vectors)
        return {
            "quadratic_storage_audit": 0.5 * torch.sum(concentration * concentration),
            "effort_audit": concentration,
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return non-negative structural chemistry dissipation terms."""

        if self.physical_boundary == "closed_constant_volume":
            box = self._evaluate_closed_constant_volume_box(state)
            return {
                "entropy_functional": "S = sum_i n_i (s_i(T) - R ln(p_i / p0))",
                "entropy": box["entropy"],
                "dissipation_rate": box["entropy_production_rate"],
                "thermodynamic_force": box["entropy_effort"],
            }
        vectors: List[torch.Tensor] = []
        for state_var in self.OPEN_MECHANISM_SPECIES:
            if state_var in state:
                value = state[state_var]
                tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=self.device)
                vectors.append(torch.clamp(tensor.to(device=self.device, dtype=torch.float32).reshape(-1), min=0.0))
        if not vectors:
            return {
                "dissipation_rate": "sigma_h = C_h.T star_rxn C_h",
                "thermodynamic_force": "C_h",
            }
        concentration = torch.cat(vectors)
        hodge = self.build_hodge_contract(n_dofs=concentration.numel(), strict=True).matrix
        # References: atmospheric box chemistry and IPHS dissipative field form.
        # Mathematical expression: sigma_h = C_h.T star_rxn C_h >= 0.
        # Parameter meanings: star_rxn is the positive chemistry field mobility.
        return {
            "dissipation_rate": concentration @ hodge @ concentration,
            "thermodynamic_force": concentration,
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for this reduced chemistry field."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive chemistry field mobility."""

        if self.physical_boundary == "closed_constant_volume":
            box = self._evaluate_closed_constant_volume_box(state)
            return {
                "state_vars": self.CLOSED_BOX_STATE,
                "J": torch.zeros_like(box["mobility"]),
                "M": box["mobility"],
                "stoichiometric_N": box["stoichiometry"].reshape(-1, 1),
                "formulation": "generic",
                "poisson_operator": "zero_field",
                "reaction_representation": "absorbed",
                "ports": (),
                "open_system_route": None,
            }
        hodge = self.build_hodge_contract(
            n_dofs=len(self.OPEN_MECHANISM_SPECIES), strict=True
        ).matrix
        if bool(torch.any(hodge < 0.0)):
            raise ValueError("Chemistry field mobility must be non-negative in M_h.")
        return {
            "state_vars": self.OPEN_MECHANISM_SPECIES,
            "J": torch.zeros_like(hodge),
            "M": hodge,
            "ports": self.ports(),
            "open_system_route": "diagnostic_residual",
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the radiation-driven photolysis input port.

        References:
          - Seinfeld and Pandis (2016), photolysis rate parameterization.
          - Jenkin and Clemitshaw (2000), photochemical pollutants.
        Mathematical expression:
          P_port = e_h J_NO2, J_NO2 = (1 + alpha)(B1 G + B2 G^2), G = SW.
        Parameter meanings:
          e_h is NO2 field effort and J_NO2 is the photolysis input flow.
        """

        if self.physical_boundary == "closed_constant_volume":
            return ()
        effort_space = CochainSpaceSpec(
            complex_id="atmosphere_chemistry_field",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="mol mol-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="no2_photolysis_effort",
            orientation_convention="production_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="atmosphere_chemistry_field",
            degree=0,
            primality="dual",
            n_dofs=1,
            units="s-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="photolysis_rate_no2",
            orientation_convention="production_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="atmospheric_radiation_photolysis_port",
                effort_var="no2_mixing_ratio",
                flow_var="diagnostics.chemistry.photolysis_rate_no2",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="mol mol-1 s-1",
                positive_power_convention="into_system",
                port_kind="radiation",
            ),
        )

    def compute(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
    ) -> Mapping[str, Any]:
        """Run the canonical calculation through the existing Functor lifecycle."""

        return self.forward(dict(state), time, dt)

    def get_parameter_value(self, name: Any, **kwargs: Any) -> Any:
        """Keep config overrides scalar-shaped (text for the declared text parameters);
        the config-first source order is the NamespaceDefaultMixin class declaration
        (parameter_lookup_source_order)."""
        value = super().get_parameter_value(name, **kwargs)
        _, parameter_name, namespace = self._resolve_parameter_lookup(name, kwargs.get("namespace"))
        if namespace == self.parameter_namespace and parameter_name in self._override_parameter_names:
            if parameter_name in self.OPEN_MECHANISM_TEXT_PARAMETERS:
                if not isinstance(value, str):
                    raise ValueError(f"Chemistry parameter {name} must be text.")
            elif isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"Chemistry parameter {name} must be scalar.")
                if value.ndim != 0:
                    return value.reshape(())
            elif not isinstance(value, (int, float)):
                raise ValueError(f"Chemistry parameter {name} must be numeric and scalar.")
        return value

    def _resolve_nasa7_data(self) -> Dict[str, Any]:
        """Read one complete thermodynamic parameter projection from its owner."""
        names = ("model", "species_order", "temperature_ranges", *(f"a{i}" for i in range(1, 8)))
        return {
            name: self.get_parameter_value(
                "atmosphere.chemistry.nasa7." + name, required=True, source_order=("manager",),
            ) for name in names
        }

    def standard_molar_properties(
        self, temperature: torch.Tensor, *, _resolved_data: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the declared NASA7 standard gas properties on the input device.

        References: McBride, Gordon and Reno (1993), NASA TM-4513; the frozen
          source dataset and individual fit notes are recorded in chemistry.yaml.
        Mathematical expressions:
          cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4;
          h/(R*T) = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T;
          s/R = a1*ln(T/[1 K]) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7;
          u = h - R*T and cv = cp - R.
        Parameter meanings: a1..a7 use the YAML species and temperature interval
          axes; R and the standard pressure use the physical-constant owner.
        Units: T in K, cp/cv/s in J mol-1 K-1, h/u in J mol-1, pressure in Pa.
        Domain/constraints: finite float32/float64 temperatures within the declared
          fit; the lower interval owns a shared endpoint. No temperature guessing.
        Sign convention: h and u include formation terms and may be negative;
          standard entropy is referenced to the dataset's 1-bar convention.
        """
        if not isinstance(temperature, torch.Tensor) or temperature.dtype not in (torch.float32, torch.float64):
            raise ValueError("NASA7 temperature must be a float32 or float64 tensor.")
        if temperature.numel() == 0 or not bool(torch.isfinite(temperature).all()):
            raise ValueError("NASA7 temperature must be nonempty and finite.")
        data = self._resolve_nasa7_data() if _resolved_data is None else _resolved_data
        model, species, ranges = data["model"], data["species_order"], data["temperature_ranges"]
        if model != "NASA7":
            raise ValueError("NASA7 properties cannot use a different polynomial family.")
        if not isinstance(species, (list, tuple)) or len(species) != 4 or set(species) != {"NO", "NO2", "O3", "O2"}:
            raise ValueError("NASA7 species_order must contain NO, NO2, O3 and O2 exactly once.")
        if not isinstance(ranges, torch.Tensor) or ranges.ndim != 1 or ranges.numel() < 2:
            raise ValueError("NASA7 temperature_ranges must be an ordered one-dimensional tensor.")
        ranges = ranges.to(device=temperature.device, dtype=temperature.dtype)
        if not bool(torch.isfinite(ranges).all()) or not bool(torch.all(ranges[1:] > ranges[:-1])) or ranges[0] <= 0:
            raise ValueError("NASA7 temperature_ranges must be finite, positive and strictly increasing.")
        if bool(torch.any((temperature < ranges[0]) | (temperature > ranges[-1]))):
            raise ValueError("NASA7 temperature is outside the declared fit intervals.")
        interval = torch.bucketize(temperature.contiguous(), ranges[1:-1].contiguous(), right=False)
        coefficients = []
        for index in range(1, 8):
            coefficient = data[f"a{index}"]
            if not isinstance(coefficient, torch.Tensor) or tuple(coefficient.shape) != (len(species), ranges.numel() - 1):
                raise ValueError(f"NASA7 a{index} shape must match the species and temperature interval axes.")
            if not coefficient.is_floating_point() or not bool(torch.isfinite(coefficient).all()):
                raise ValueError(f"NASA7 a{index} must contain finite real coefficients.")
            coefficient = coefficient.to(device=temperature.device, dtype=temperature.dtype)
            coefficients.append(coefficient[:, interval].movedim(0, -1))
        a1, a2, a3, a4, a5, a6, a7 = coefficients
        t = temperature.unsqueeze(-1)
        gas_constant = float(get_physical_constant("R_GAS"))
        cp = gas_constant * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
        h = gas_constant * t * (a1 + t * (a2 / 2 + t * (a3 / 3 + t * (a4 / 4 + t * a5 / 5))) + a6 / t)
        s = gas_constant * (a1 * torch.log(t) + t * (a2 + t * (a3 / 2 + t * (a4 / 3 + t * a5 / 4))) + a7)
        properties: Dict[str, Any] = {
            "species": tuple(species), "cp": cp, "h": h, "s": s,
            "u": h - gas_constant * t, "cv": cp - gas_constant,
            "reference_pressure": temperature.new_tensor(float(get_physical_constant("reference_pressure"))),
        }
        if any(not bool(torch.isfinite(properties[key]).all()) for key in ("cp", "h", "s", "u", "cv")):
            raise ValueError("NASA7 evaluation produced nonfinite molar properties.")
        return properties

    def ideal_gas_mixture_properties(
        self, temperature: torch.Tensor, amounts: torch.Tensor, volume: torch.Tensor,
    ) -> Dict[str, Any]:
        """Evaluate ideal-mixture potentials without advancing any state.

        References: Seinfeld and Pandis (2016), ideal-gas thermodynamics;
          McBride, Gordon and Reno (1993), NASA TM-4513, standard properties.
        Mathematical expressions: p_i=n_i*R*T/V; mu_i=h_i-T*s_i+R*T*ln(p_i/p0);
          U=sum(n_i*u_i); S=sum(n_i*(s_i-R*ln(p_i/p0))); F=U-T*S;
          C_V=sum(n_i*cv_i). Thus F_n=mu, F_V=-p, F_T=-S, U_T=C_V.
        Parameter meanings: n follows the declared NASA7 species order;
          standard properties, R and p0 come from their existing owners.
        Units: n [mol], T [K], V [m3], p_i/p0 dimensionless with p_i and p0 [Pa],
          mu [J mol-1], U/F [J],
          S/C_V [J K-1].
        Domain/constraints: positive finite inventories and volume, valid NASA7
          temperatures, positive C_V; common dtype/device and explicit batch axes.
          Volume may be scalar and shared; no inventory regularization is used.
        Sign convention: U includes formation energy; pressure work is -p*dV.
          F is Helmholtz energy, not reaction affinity or Gibbs energy.
        """
        standard = self.standard_molar_properties(temperature)
        for name, value in (("amounts", amounts), ("volume", volume)):
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Mixture {name} must be a tensor.")
            if value.dtype != temperature.dtype or value.device != temperature.device:
                raise ValueError(f"Mixture {name} must share temperature dtype and device.")
            if value.numel() == 0 or not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
                raise ValueError(f"Mixture {name} must be nonempty, finite and positive.")
        if amounts.shape != standard["u"].shape:
            raise ValueError("Mixture amounts must append the species axis to the temperature shape.")
        if volume.ndim != 0 and volume.shape != temperature.shape:
            raise ValueError("Mixture volume must be scalar or match the temperature shape.")
        gas_constant = float(get_physical_constant("R_GAS"))
        t = temperature.unsqueeze(-1)
        partial_pressures = amounts * gas_constant * t / volume.unsqueeze(-1)
        if not bool(torch.isfinite(partial_pressures).all()) or not bool((partial_pressures > 0).all()):
            raise ValueError("Mixture partial pressures must remain finite and positive.")
        log_pressure_ratio = torch.log(partial_pressures) - torch.log(standard["reference_pressure"])
        partial_entropy = standard["s"] - gas_constant * log_pressure_ratio
        internal_energy = (amounts * standard["u"]).sum(dim=-1)
        entropy = (amounts * partial_entropy).sum(dim=-1)
        properties: Dict[str, Any] = {
            "species": standard["species"],
            "partial_pressures": partial_pressures,
            "pressure": partial_pressures.sum(dim=-1),
            "chemical_potential": standard["h"] - t * partial_entropy,
            "internal_energy": internal_energy,
            "entropy": entropy,
            "heat_capacity_cv": (amounts * standard["cv"]).sum(dim=-1),
            "helmholtz_energy": internal_energy - temperature * entropy,
        }
        if any(not bool(torch.isfinite(value).all()) for name, value in properties.items() if name != "species"):
            raise ValueError("Mixture evaluation produced nonfinite properties.")
        if not bool((properties["heat_capacity_cv"] > 0).all()):
            raise ValueError("Mixture heat capacity must be positive.")
        return properties

    def temperature_from_internal_energy(
        self, internal_energy: torch.Tensor, amounts: torch.Tensor, *,
        temperature_lower: torch.Tensor, temperature_upper: torch.Tensor,
    ) -> torch.Tensor:
        """Invert one continuous ideal-gas energy branch with explicit gradients.

        References: NASA TM-4513; Burden and Faires (2010), bisection;
          Blondel et al. (2022), implicit differentiation; Moore et al. (2009),
          Introduction to Interval Analysis, interval polynomial evaluation.
        Mathematical expressions: f(T)=sum(n_i*u_i(T))-U=0; f_T=C_V>0;
          T_U=1/C_V, T_ni=-u_i/C_V. Outward interval Horner bounds certify
          cv_i/R=(a1-1)+a2*T+a3*T^2+a4*T^3+a5*T^4>0 on the bracket.
        Parameter meanings: n and NASA coefficients are explicit root parameters;
          the existing solver owner supplies convergence settings.
        Units: U [J], n [mol], bracket/result T [K], C_V [J K-1].
        Domain/constraints: finite positive inventories, one NASA7 interval,
          positive heat capacities, matching scalar/batch dtype/device. No V term.
        Sign convention: formation internal energy is retained, including U<0;
          this inverse calculation performs no state update.
        """
        if not isinstance(internal_energy, torch.Tensor) or internal_energy.dtype not in (torch.float32, torch.float64):
            raise ValueError("Internal energy must be a float32 or float64 tensor.")
        if internal_energy.numel() == 0 or not bool(torch.isfinite(internal_energy).all()):
            raise ValueError("Internal energy must be nonempty and finite.")
        for name, value in (("amounts", amounts), ("temperature_lower", temperature_lower), ("temperature_upper", temperature_upper)):
            if not isinstance(value, torch.Tensor) or value.dtype != internal_energy.dtype or value.device != internal_energy.device:
                raise ValueError(f"Temperature inversion {name} must match energy dtype/device.")
            expected = internal_energy.shape + (4,) if name == "amounts" else internal_energy.shape
            if value.shape != expected or not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
                raise ValueError(f"Temperature inversion {name} requires positive finite values with the declared shape.")
        if not bool((temperature_upper > temperature_lower).all()):
            raise ValueError("Temperature inversion requires an increasing bracket.")
        data = self._resolve_nasa7_data()
        self.standard_molar_properties(temperature_lower, _resolved_data=data)
        self.standard_molar_properties(temperature_upper, _resolved_data=data)
        ranges = data["temperature_ranges"]
        if ranges.requires_grad:
            raise ValueError("Temperature fit boundaries are discrete metadata, not differentiable root parameters.")
        ranges = ranges.to(internal_energy).detach().clone()
        interval = torch.bucketize(temperature_lower.contiguous(), ranges[1:-1].contiguous(), right=False)
        upper_interval = torch.bucketize(temperature_upper.contiguous(), ranges[1:-1].contiguous(), right=False)
        if not bool((interval == upper_interval).all()):
            raise ValueError("Temperature inversion bracket must stay within one continuous NASA7 interval.")
        coefficient_shape = data["a1"].shape
        coefficients = tuple(data[f"a{i}"].to(internal_energy).clone() for i in range(1, 8))
        with torch.no_grad():
            selected = tuple(coefficient[:, interval].movedim(0, -1) for coefficient in coefficients[:5])
            low = high = selected[-1]
            t_low, t_high = temperature_lower.unsqueeze(-1), temperature_upper.unsqueeze(-1)
            negative_inf, positive_inf = torch.full_like(low, -torch.inf), torch.full_like(low, torch.inf)
            for index in range(3, -1, -1):
                term_low = term_high = selected[index]
                if index == 0:
                    term_low = torch.nextafter(term_low - 1, negative_inf)
                    term_high = torch.nextafter(term_high - 1, positive_inf)
                low = torch.nextafter(torch.minimum(low * t_low, low * t_high), negative_inf)
                high = torch.nextafter(torch.maximum(high * t_low, high * t_high), positive_inf)
                low = torch.nextafter(low + term_low, negative_inf)
                high = torch.nextafter(high + term_high, positive_inf)
            if not bool(torch.isfinite(low).all() & torch.isfinite(high).all()) or not bool((low > 0).all()):
                raise ValueError("Temperature bracket cannot certify positive species heat capacity; narrow the bracket.")
        parameter_count = coefficients[0].numel()
        frozen_metadata = {"model": data["model"], "species_order": tuple(data["species_order"]), "temperature_ranges": ranges}

        def residual(temperature: torch.Tensor, energy: torch.Tensor, *parameters: torch.Tensor) -> torch.Tensor:
            explicit_data = dict(frozen_metadata)
            for index in range(7):
                start = 4 + index * parameter_count
                explicit_data[f"a{index + 1}"] = torch.stack(parameters[start:start + parameter_count]).reshape(coefficient_shape)
            molar = self.standard_molar_properties(temperature, _resolved_data=explicit_data)
            inventory = torch.stack(parameters[:4], dim=-1)
            return (inventory * molar["u"]).sum(dim=-1) - energy

        settings = {
            name: self.get_parameter_value("computation.solvers.bracketed_bisection." + name, required=True, source_order=("manager",))
            for name in ("max_iterations", "rtol", "atol")
        }
        settings["rtol"] = max(float(settings["rtol"]), torch.finfo(internal_energy.dtype).eps)
        solver = SolverFactory.create(
            "bracketed_bisection", name="chemistry_temperature", config=settings,
            device=internal_energy.device, dtype=internal_energy.dtype,
        )
        parameters = (internal_energy, *amounts.unbind(-1), *(scalar for coefficient in coefficients for scalar in coefficient.reshape(-1).unbind()))
        result = solver(residual, temperature_lower, temperature_upper, parameters=parameters)
        if result.status is not SolverStatus.SUCCESS:
            raise ValueError("Temperature inversion did not converge within the declared solver settings.")
        temperature = result.solution
        resolved = dict(frozen_metadata)
        resolved.update({f"a{i}": coefficient for i, coefficient in enumerate(coefficients, 1)})
        final = self.standard_molar_properties(temperature, _resolved_data=resolved)
        energy_terms = amounts * final["u"]
        energy_residual = (energy_terms.sum(dim=-1) - internal_energy).abs()
        capacity_upper = (amounts * high).sum(dim=-1) * float(get_physical_constant("R_GAS"))
        temperature_tolerance = float(settings["atol"]) + settings["rtol"] * temperature.abs()
        roundoff = torch.finfo(internal_energy.dtype).eps * (amounts.shape[-1] + 1) * (energy_terms.abs().sum(dim=-1) + internal_energy.abs())
        if not bool(torch.isfinite(energy_residual).all()) or not bool((energy_residual <= capacity_upper * temperature_tolerance + roundoff).all()):
            raise ValueError("Temperature inversion energy residual exceeds the solver and roundoff bound.")
        return temperature

    def _photolysis_temperature_mode(self) -> str:
        """Return the declared photolysis temperature mode; an unknown mode raises."""
        mode = self.get_parameter_value("photolysis_temperature_mode", required=True)
        if mode not in self.PHOTOLYSIS_TEMPERATURE_MODES:
            raise ValueError(
                "atmosphere.chemistry.photolysis_temperature_mode must be one of "
                f"{self.PHOTOLYSIS_TEMPERATURE_MODES}, got {mode!r}"
            )
        return mode

    def _load_photolysis_spectra(self) -> None:
        """Bind the spectral reference data of the photolysis temperature factor once.

        The tables are static evaluated data (requires_grad false); they are read from
        the ParameterManager at construction and kept as non-persistent buffers that
        follow the Functor device. Later manager updates of these tables are not
        observed by an existing Functor. The slant-path attenuation depends on the
        solar geometry and is applied per call (_spectral_attenuation).
        """
        # References: Warneck and Williams (2012) Tables 6.2 and 6.4a (extraterrestrial
        #   photon flux), 6.11a, 6.11b and 6.12 (O3 cross sections), 6.13a and 6.13b (NO2
        #   cross sections and quantum yields); Bodhaine et al. (1999) Eq. (30).
        # Mathematical expression: tau(lambda) = tau_R(lambda) + sigma_O3,col(lambda) N_O3,
        #   tau_R = s (a - b lambda^-2 - c lambda^2) / (1 + d lambda^-2 - e lambda^2);
        #   F0 at an integer-nm grid wavelength is the linear interpolation (the mean) of
        #   the two adjacent 1-nm interval averages; sigma_NO2 at a grid wavelength is the
        #   5-nm interval average that contains it; the O3 columns are linearly
        #   interpolated in wavelength; tau_shift = tau - min_lambda(tau) per grid.
        # Parameter meanings: F0 extraterrestrial photon flux; tau vertical optical depth
        #   of the direct beam; tau_R sea-level Rayleigh optical depth; sigma_O3,col the
        #   lower-temperature column (218-226 K, the stratospheric range that holds most of
        #   the ozone column) of the O3 table; N_O3 reference vertical ozone column; s, a,
        #   b, c, d, e Bodhaine coefficients with lambda the wavelength in micrometres.
        # Units: lambda [um]; F0 [photons cm-2 s-1 um-1]; sigma [cm2]; N_O3
        #   [molecule cm-2]; tau, a-e, s [dimensionless].
        # Domain/constraints: direct-beam (Beer-Lambert) attenuation; diffuse sky
        #   radiation, surface albedo and aerosol are not represented, which is admissible
        #   only because the spectrum enters as a weight of the ratio R(T). Wavelength axes
        #   strictly increase, the integration grids lie inside every table, cross sections
        #   and fluxes are positive, yields lie in [0, 1], T_high > T_low, T_ref > 0 and
        #   0 < T_min(Matsumi) < T_max(Matsumi); a violation raises and nothing is clipped.
        # Sign convention: tau >= 0 removes photons from the beam; no flux or state is
        #   advanced here.
        namespace = self.PHOTOLYSIS_NAMESPACE

        def read(name: str) -> torch.Tensor:
            value = self.get_parameter_value(
                f"{namespace}.{name}", required=True, source_order=("manager",)
            )
            if isinstance(value, torch.Tensor):
                tensor = value.detach().to(dtype=torch.float64, device=torch.device("cpu"))
            else:
                tensor = torch.as_tensor(value, dtype=torch.float64)
            if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{namespace}.{name} must be nonempty and finite.")
            return tensor

        flux_axis, flux = read("solar_flux_wavelength"), read("solar_photon_flux")
        no2_sigma_axis, no2_sigma = read("no2_cross_section_wavelength"), read("no2_cross_section")
        no2_grid = read("no2_quantum_yield_wavelength")
        no2_yield_high = read("no2_quantum_yield_high_temperature")
        no2_yield_low = read("no2_quantum_yield_low_temperature")
        no2_t_high = read("no2_quantum_yield_temperature_high")
        no2_t_low = read("no2_quantum_yield_temperature_low")
        o3_axis = read("o3_cross_section_wavelength")
        o3_high, o3_low = read("o3_cross_section_high_temperature"), read("o3_cross_section_low_temperature")
        o3_t_high = read("o3_cross_section_temperature_high")
        o3_t_low = read("o3_cross_section_temperature_low")
        o1d_grid = read("o1d_integration_wavelength")
        (scale, numerator_constant, numerator_inverse_square, numerator_square,
         denominator_inverse_square, denominator_square) = (
            read(name) for name in (
                "rayleigh_scale", "rayleigh_numerator_constant", "rayleigh_numerator_inverse_square",
                "rayleigh_numerator_square", "rayleigh_denominator_inverse_square",
                "rayleigh_denominator_square",
            )
        )
        ozone_column = read("reference_ozone_column")
        t_ref = read("reference_temperature")
        matsumi_t_min, matsumi_t_max = read("matsumi_temperature_min"), read("matsumi_temperature_max")

        for label, axis, values in (
            ("solar flux", flux_axis, (flux,)),
            ("NO2 cross section", no2_sigma_axis, (no2_sigma,)),
            ("NO2 quantum yield", no2_grid, (no2_yield_high, no2_yield_low)),
            ("O3 cross section", o3_axis, (o3_high, o3_low, o3_t_high, o3_t_low)),
            ("O(1D) integration grid", o1d_grid, ()),
        ):
            if axis.ndim != 1 or axis.numel() < 2 or not bool(torch.all(axis[1:] > axis[:-1])) or bool(axis[0] <= 0):
                raise ValueError(f"The {label} wavelengths must be positive and strictly increasing.")
            if any(value.shape != axis.shape for value in values):
                raise ValueError(f"The {label} values must match their wavelength axis.")
        for label, grid in (("NO2 quantum yield", no2_grid), ("O(1D) integration", o1d_grid)):
            for table, axis in (("solar flux", flux_axis), ("O3 cross section", o3_axis)):
                if bool(grid[0] < axis[0]) or bool(grid[-1] > axis[-1]):
                    raise ValueError(f"The {label} grid extends beyond the {table} table.")
        no2_half_width = 0.5 * (no2_sigma_axis[1:] - no2_sigma_axis[:-1])
        if bool(no2_grid[0] < no2_sigma_axis[0] - no2_half_width[0]) or bool(
            no2_grid[-1] > no2_sigma_axis[-1] + no2_half_width[-1]
        ):
            raise ValueError("The NO2 quantum yield grid extends beyond the NO2 cross-section intervals.")
        if not (bool(torch.all(flux > 0)) and bool(torch.all(no2_sigma > 0))
                and bool(torch.all(o3_high > 0)) and bool(torch.all(o3_low > 0))):
            raise ValueError("Photolysis cross sections and photon fluxes must be positive.")
        for yields in (no2_yield_high, no2_yield_low):
            if bool(torch.any((yields < 0) | (yields > 1))):
                raise ValueError("NO2 quantum yields must lie in [0, 1].")
        if not bool(no2_t_high > no2_t_low) or not bool(torch.all(o3_t_high > o3_t_low)):
            raise ValueError("Each table's upper temperature must exceed its lower temperature.")
        if not (bool(t_ref > 0) and bool(matsumi_t_min > 0) and bool(matsumi_t_min < matsumi_t_max)):
            raise ValueError(
                "The photolysis reference temperature and the Matsumi temperature range must be positive "
                "and increasing."
            )

        def on_grid(values: torch.Tensor, axis: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
            return self._linear_interpolation_1d(values, axis.unsqueeze(-1), grid.unsqueeze(-1))

        def shifted_optical_depth(grid: torch.Tensor) -> torch.Tensor:
            tau_rayleigh = scale * (
                numerator_constant - numerator_inverse_square / grid ** 2 - numerator_square * grid ** 2
            ) / (1.0 + denominator_inverse_square / grid ** 2 - denominator_square * grid ** 2)
            if not bool(torch.all(tau_rayleigh > 0)) or not bool(torch.all(torch.isfinite(tau_rayleigh))):
                raise ValueError("The Rayleigh optical depth fit is outside its domain on the grid.")
            tau = tau_rayleigh + on_grid(o3_low, o3_axis, grid) * ozone_column
            # The common factor exp(-min(tau)/mu) cancels in R = S(T)/S(T_ref); removing it
            # keeps the largest weight at exp(0) for every solar geometry (no underflow).
            return tau - tau.min()

        # References: CODATA 2018 (SSOT constants k_B, h, c); Matsumi et al. (2002).
        # Mathematical expression: R_nu = k_B / (h c), converted from m-1 K-1 to cm-1 K-1.
        # Parameter meanings: k_B Boltzmann constant, h Planck constant, c speed of light.
        # Units: k_B [J K-1], h [J s], c [m s-1]; R_nu [cm-1 K-1] (0.695 cm-1 K-1).
        # Domain/constraints: exact SI constants; the conversion uses the unit owner.
        # Sign convention: not applicable (a constant).
        from ...data.standards.unit_converter import ConversionContext, unit_converter

        wavenumber_per_kelvin = unit_converter.convert(
            torch.tensor(
                self._constants["boltzmann_constant"]
                / (self._constants["planck_constant"] * self._constants["speed_of_light"]),
                dtype=torch.float64,
            ),
            "generic",
            "m-1 K-1",
            "cm-1 K-1",
            ConversionContext(allow_identity=True),
        )

        boundaries = 0.5 * (no2_sigma_axis[1:] + no2_sigma_axis[:-1])
        no2_sigma_on_grid = no2_sigma[torch.bucketize(no2_grid, boundaries)]
        buffers = {
            "_photolysis_no2_weight": no2_sigma_on_grid * on_grid(flux, flux_axis, no2_grid),
            "_photolysis_no2_optical_depth": shifted_optical_depth(no2_grid),
            "_photolysis_no2_yield_high": no2_yield_high,
            "_photolysis_no2_yield_low": no2_yield_low,
            "_photolysis_no2_temperature_high": no2_t_high.reshape(()),
            "_photolysis_no2_temperature_low": no2_t_low.reshape(()),
            "_photolysis_o1d_wavelength": o1d_grid,
            "_photolysis_o1d_flux": on_grid(flux, flux_axis, o1d_grid),
            "_photolysis_o1d_optical_depth": shifted_optical_depth(o1d_grid),
            "_photolysis_o3_sigma_high": on_grid(o3_high, o3_axis, o1d_grid),
            "_photolysis_o3_sigma_low": on_grid(o3_low, o3_axis, o1d_grid),
            "_photolysis_o3_temperature_high": on_grid(o3_t_high, o3_axis, o1d_grid),
            "_photolysis_o3_temperature_low": on_grid(o3_t_low, o3_axis, o1d_grid),
            "_photolysis_matsumi_amplitudes": read("matsumi_amplitudes"),
            "_photolysis_matsumi_centers": read("matsumi_band_centers"),
            "_photolysis_matsumi_widths": read("matsumi_band_widths"),
            "_photolysis_matsumi_offset": read("matsumi_offset").reshape(()),
            "_photolysis_matsumi_energy_gap": read("matsumi_energy_gap").reshape(()),
            "_photolysis_matsumi_temperature": read("matsumi_reference_temperature").reshape(()),
            "_photolysis_matsumi_temperature_min": matsumi_t_min.reshape(()),
            "_photolysis_matsumi_temperature_max": matsumi_t_max.reshape(()),
            "_photolysis_matsumi_wavelength_min": read("matsumi_wavelength_min").reshape(()),
            "_photolysis_matsumi_wavelength_max": read("matsumi_wavelength_max").reshape(()),
            "_photolysis_o1d_yield_short": read("o1d_quantum_yield_short_wavelength").reshape(()),
            "_photolysis_o1d_yield_long": read("o1d_quantum_yield_long_wavelength").reshape(()),
            "_photolysis_wavenumber_per_kelvin": wavenumber_per_kelvin.reshape(()),
            "_photolysis_reference_temperature": t_ref.reshape(()),
        }
        for label in ("_photolysis_matsumi_amplitudes", "_photolysis_matsumi_centers", "_photolysis_matsumi_widths"):
            if tuple(buffers[label].shape) != (3,):
                raise ValueError(f"{label} must hold the three terms of the Matsumi expression.")
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor.contiguous(), persistent=False)

    def _photolysis_temperature_factors(
        self, temperature: torch.Tensor, zenith_cosine: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Air-temperature factors (R_NO2, R_O1D) of the declared photolysis mode."""
        # References: Warneck and Williams (2012) Tables 6.11a-c and 6.13a-b; Matsumi et
        #   al. (2002); TUV practice of holding the nearest tabulated temperature (WRF
        #   module_xsections o3xs: no extrapolation in temperature).
        # Mathematical expression: mode none: R_NO2 = R_O1D = 1; mode
        #   spectral_quantum_yield: R_X(T, mu) = S_X(T, mu) / S_X(T_ref, mu),
        #   S_X(T, mu) = sum_lambda sigma_X(lambda, T) phi_X(lambda, T) F0(lambda)
        #   exp(-tau_shift(lambda) / mu) for mu > 0, and R_X = 1 for mu <= 0.
        # Parameter meanings: T air temperature; T_ref reference temperature; mu cosine of
        #   the actual solar zenith angle (direct-beam slant path); sigma_X, phi_X cross
        #   section and quantum yield of NO2 or O3 -> O(1D); F0 extraterrestrial photon
        #   flux; tau_shift the vertical optical depth less its grid minimum (a common
        #   factor of numerator and denominator).
        # Units: T, T_ref [K]; mu, tau, R_X [dimensionless].
        # Domain/constraints: finite positive T (else it raises); beyond the evaluated or
        #   tabulated temperatures the nearest one is held (tables: 248-298 K NO2, 218/226-
        #   295/298 K O3; Matsumi 200-300 K). mu <= 0 has no direct beam: R_X = 1 there,
        #   where J(O1D) = 0 (rho = 0) and J_NO2 is the small twilight value of the
        #   irradiance relation; the mu > 0 branch is evaluated on a safe cosine so the
        #   gradients stay finite.
        # Sign convention: R_X > 1 raises, R_X < 1 lowers the photolysis frequency.
        if not temperature.is_floating_point():
            temperature = temperature.to(dtype=torch.get_default_dtype())
        temperature, zenith_cosine = torch.broadcast_tensors(temperature, zenith_cosine.to(temperature))
        if self._photolysis_temperature_mode() == "none":
            ones = torch.ones_like(temperature)
            return ones, ones
        if not bool(torch.isfinite(temperature).all()) or bool(torch.any(temperature <= 0)):
            raise ValueError(
                "The spectral_quantum_yield photolysis temperature factor needs finite positive air "
                "temperatures [K]."
            )
        values = temperature.to(dtype=torch.float64)
        cosine = zenith_cosine.to(dtype=torch.float64)
        daytime = cosine > 0
        safe_cosine = torch.where(daytime, cosine, torch.ones_like(cosine))
        reference = torch.full_like(values, float(self._photolysis_reference_temperature))
        no2_attenuation = self._spectral_attenuation(self._photolysis_no2_optical_depth, safe_cosine)
        o1d_attenuation = self._spectral_attenuation(self._photolysis_o1d_optical_depth, safe_cosine)
        r_no2 = self._no2_spectral_sum(values, no2_attenuation) / self._no2_spectral_sum(reference, no2_attenuation)
        r_o1d = self._o1d_spectral_sum(values, o1d_attenuation) / self._o1d_spectral_sum(reference, o1d_attenuation)
        ones = torch.ones_like(r_no2)
        return (
            torch.where(daytime, r_no2, ones).to(dtype=temperature.dtype),
            torch.where(daytime, r_o1d, ones).to(dtype=temperature.dtype),
        )

    @staticmethod
    def _spectral_attenuation(optical_depth: torch.Tensor, zenith_cosine: torch.Tensor) -> torch.Tensor:
        """Direct-beam transmission exp(-tau_shift / mu), shape zenith_cosine.shape + (G,)."""
        # References: Beer-Lambert law of the direct solar beam (Seinfeld and Pandis 2016,
        #   ch. 4); Bodhaine et al. (1999) for tau_R.
        # Mathematical expression: T(lambda; mu) = exp(-tau_shift(lambda) / mu).
        # Parameter meanings: tau_shift vertical optical depth less its grid minimum; mu
        #   cosine of the solar zenith angle (plane-parallel slant path 1/mu).
        # Units: tau_shift, mu, T [dimensionless].
        # Domain/constraints: mu > 0 (the caller passes a safe cosine at night); the
        #   plane-parallel path is used without a sphericity correction.
        # Sign convention: T <= 1 (tau_shift >= 0).
        depth = optical_depth.to(zenith_cosine)
        return torch.exp(-depth / zenith_cosine.unsqueeze(-1))

    def _no2_spectral_sum(self, temperature: torch.Tensor, attenuation: torch.Tensor) -> torch.Tensor:
        """S_NO2(T) = sum sigma_NO2 phi_NO2(T) F0 T(mu) over the NO2 grid."""
        # References: Warneck and Williams (2012) Table 6.13b (Troe 2000); TUV practice.
        # Mathematical expression: phi(lambda, T) = phi_low + (phi_high - phi_low)
        #   (T_h - T_low) / (T_high - T_low), T_h = min(max(T, T_low), T_high);
        #   S = sum_lambda w(lambda) Tr(lambda; mu) phi(lambda, T), w = sigma_NO2 F0.
        # Parameter meanings: phi_high, phi_low quantum yields at T_high = 298 K and
        #   T_low = 248 K; T_h the held temperature; w the extraterrestrial spectral weight;
        #   Tr the direct-beam transmission.
        # Units: T [K]; phi, Tr [dimensionless]; w and S [photons s-1 um-1].
        # Domain/constraints: linear in T between the tables; the nearest table is held
        #   outside 248-298 K (zero temperature derivative there).
        # Sign convention: not applicable (a positive weighted sum).
        high = self._photolysis_no2_temperature_high.to(temperature)
        low = self._photolysis_no2_temperature_low.to(temperature)
        held = torch.minimum(torch.maximum(temperature, low), high)
        fraction = ((held - low) / (high - low)).unsqueeze(-1)
        yield_low = self._photolysis_no2_yield_low.to(temperature)
        yields = yield_low + (self._photolysis_no2_yield_high.to(temperature) - yield_low) * fraction
        weight = self._photolysis_no2_weight.to(temperature) * attenuation
        return (weight * yields).sum(dim=-1)

    def _o1d_spectral_sum(self, temperature: torch.Tensor, attenuation: torch.Tensor) -> torch.Tensor:
        """S_O1D(T) = sum sigma_O3(T) phi_O1D(T) F0 T(mu) over the O(1D) grid."""
        # References: Warneck and Williams (2012) Tables 6.11a and 6.11b (Molina and
        #   Molina 1986; Malicet et al. 1995; Cacciani et al. 1989); TUV practice.
        # Mathematical expression: sigma(lambda, T) = sigma_low + (sigma_high - sigma_low)
        #   (T_h - T_low) / (T_high - T_low), T_h = min(max(T, T_low), T_high);
        #   S = sum_lambda sigma(lambda, T) phi_O1D(lambda, T) F0(lambda) Tr(lambda; mu).
        # Parameter meanings: sigma_high, sigma_low O3 cross sections at the per-point
        #   tabulation temperatures T_high (298/295 K) and T_low (226/218/220 K); phi_O1D
        #   the O(1D) quantum yield; F0 the extraterrestrial photon flux; Tr the direct-beam
        #   transmission.
        # Units: T [K]; sigma [cm2]; phi, Tr [dimensionless]; F0 [photons cm-2 s-1 um-1].
        # Domain/constraints: linear in T between the two tables of each point; the
        #   nearest table is held outside them (zero derivative of sigma there).
        # Sign convention: not applicable (a positive weighted sum).
        values = temperature.unsqueeze(-1)
        high = self._photolysis_o3_temperature_high.to(temperature)
        low = self._photolysis_o3_temperature_low.to(temperature)
        held = torch.minimum(torch.maximum(values, low), high)
        sigma_low = self._photolysis_o3_sigma_low.to(temperature)
        sigma = sigma_low + (self._photolysis_o3_sigma_high.to(temperature) - sigma_low) * (held - low) / (high - low)
        yields = self._o1d_quantum_yield(self._photolysis_o1d_wavelength, temperature)
        flux = self._photolysis_o1d_flux.to(temperature) * attenuation
        return (sigma * yields * flux).sum(dim=-1)

    def _o1d_quantum_yield(self, wavelength: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """O3 + hv -> O(1D) + O2 quantum yield phi(lambda, T), shape temperature.shape + (G,)."""
        # References: Matsumi et al. (2002) J. Geophys. Res. 107(D3), 4024 (expression
        #   for 306-328 nm evaluated for 200-300 K, recommended 0.90 below 306 nm and 0.08
        #   above 328 nm); Warneck and Williams (2012) Table 6.11c and text.
        # Mathematical expression: phi = A1/(1+q) exp(-((x1-lambda)/w1)^4)
        #   + A2 q/(1+q) (T_M/T0)^2 exp(-((x2-lambda)/w2)^2)
        #   + A3 (T_M/T0)^1.5 exp(-((x3-lambda)/w3)^2) + C, q = exp(-nu2/(R_nu T_M)),
        #   T_M = min(max(T, T_min), T_max), for lambda_min <= lambda <= lambda_max;
        #   phi = phi_short below lambda_min and phi = phi_long above lambda_max.
        # Parameter meanings: A1-A3 amplitudes; x1-x3 band centres; w1-w3 widths; C offset;
        #   nu2 energy gap; R_nu = k_B/(h c); T0 normalization temperature; T_min, T_max
        #   the evaluated temperature range (200, 300 K); phi_short, phi_long
        #   temperature-independent plateau yields.
        # Units: lambda, x, w [um]; nu2 [cm-1]; R_nu [cm-1 K-1]; T, T_M, T0 [K]; phi, A,
        #   C, q [dimensionless].
        # Domain/constraints: outside 200-300 K the nearest evaluated temperature is held
        #   (zero temperature derivative there); T > 0 is checked by the caller.
        # Sign convention: not applicable (a probability of the O(1D) channel).
        lam = wavelength.to(dtype=torch.float64, device=temperature.device)
        values = temperature.to(dtype=torch.float64).unsqueeze(-1)
        held = torch.minimum(
            torch.maximum(values, self._photolysis_matsumi_temperature_min.to(lam)),
            self._photolysis_matsumi_temperature_max.to(lam),
        )
        amplitudes = self._photolysis_matsumi_amplitudes.to(lam)
        centers = self._photolysis_matsumi_centers.to(lam)
        widths = self._photolysis_matsumi_widths.to(lam)
        q = torch.exp(
            -self._photolysis_matsumi_energy_gap.to(lam)
            / (self._photolysis_wavenumber_per_kelvin.to(lam) * held)
        )
        relative = held / self._photolysis_matsumi_temperature.to(lam)
        formula = (
            amplitudes[0] / (1.0 + q) * torch.exp(-(((centers[0] - lam) / widths[0]) ** 4))
            + amplitudes[1] * q / (1.0 + q) * relative ** 2 * torch.exp(-(((centers[1] - lam) / widths[1]) ** 2))
            + amplitudes[2] * relative ** 1.5 * torch.exp(-(((centers[2] - lam) / widths[2]) ** 2))
            + self._photolysis_matsumi_offset.to(lam)
        )
        return torch.where(
            lam < self._photolysis_matsumi_wavelength_min.to(lam),
            self._photolysis_o1d_yield_short.to(lam),
            torch.where(
                lam > self._photolysis_matsumi_wavelength_max.to(lam),
                self._photolysis_o1d_yield_long.to(lam),
                formula,
            ),
        )

    def _o1d_to_no2_photolysis_ratio(self, zenith_cosine: torch.Tensor) -> torch.Tensor:
        """rho(chi) = J(O1D)/J(NO2) of the MCM clear-sky parameterisation."""
        # References: Saunders et al. (2003) ACP 3, 161-180; MCM v3.3.1 photolysis
        #   parameters J1 (O3 -> O(1D) + O2) and J4 (NO2 -> NO + O(3P)).
        # Mathematical expression: J_i = l_i cos(chi)^m_i exp(-n_i sec(chi)), so
        #   rho(chi) = J1/J4 = (l1/l4) cos(chi)^(m1-m4) exp(-(n1-n4)/cos(chi)) for
        #   cos(chi) > 0 and rho = 0 for cos(chi) <= 0 (sun at or below the horizon).
        # Parameter meanings: l, m, n MCM fit parameters (mcm_o1d_photolysis_*,
        #   mcm_no2_photolysis_*); chi solar zenith angle.
        # Units: l [s-1]; m, n, cos(chi), rho [dimensionless].
        # Domain/constraints: the MCM fits cover 0 <= chi < 90 deg; the night branch is
        #   the limit rho -> 0 (no direct beam); the positive-cosine branch is evaluated
        #   on a safe cosine so the gradient stays finite at cos(chi) = 0.
        # Sign convention: rho >= 0 converts a non-negative J_NO2 into J(O1D).
        def value(name: str) -> torch.Tensor:
            return self.get_parameter_value(name, required=True).to(zenith_cosine)

        l_o1d, m_o1d, n_o1d = (value(f"mcm_o1d_photolysis_{p}") for p in ("l", "m", "n"))
        l_no2, m_no2, n_no2 = (value(f"mcm_no2_photolysis_{p}") for p in ("l", "m", "n"))
        daytime = zenith_cosine > 0
        safe_cosine = torch.where(daytime, zenith_cosine, torch.ones_like(zenith_cosine))
        ratio = (l_o1d / l_no2) * safe_cosine ** (m_o1d - m_no2) * torch.exp(-(n_o1d - n_no2) / safe_cosine)
        return torch.where(daytime, ratio, torch.zeros_like(ratio))

    def _calculate_photolysis_rates(
        self,
        shortwave_down: torch.Tensor,
        cos_zenith: torch.Tensor,
        temperature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (J_NO2, J_O1D) from the global irradiance, solar geometry and temperature.

        J_NO2 is the Trebs et al. (2009) relation of the solar global irradiance;
        J(O1D) scales it by the MCM ratio rho(chi); both carry the photolysis
        temperature factor of the declared mode.

        Parameters:
            shortwave_down : torch.Tensor
                Solar global irradiance G (downwelling shortwave) [W m-2]
            cos_zenith : torch.Tensor
                Cosine of the solar zenith angle [dimensionless]
            temperature : torch.Tensor
                Air temperature [K]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: J_NO2 and J(O1D) [s-1]
        """
        # References: Trebs et al. (2009) Atmos. Meas. Tech. 2, 725-739, Eq. (1)-(2);
        #   Saunders et al. (2003) (MCM J1/J4); Warneck and Williams (2012) Table 6.7.
        # Mathematical expression: J_NO2,ref = (1 + alpha)(B1 G_h + B2 G_h^2),
        #   G_h = min(G, G_v), G_v = B1 / (2 |B2|); J_O1D,ref = J_NO2,ref rho(chi);
        #   J_NO2 = J_NO2,ref R_NO2(T, chi); J_O1D = J_O1D,ref R_O1D(T, chi).
        # Parameter meanings: G solar global irradiance (shortwave_down, it carries the
        #   solar geometry and cloud attenuation, so no cos(chi) factor is applied to
        #   J_NO2); alpha UV-A surface albedo; B1, B2 polynomial coefficients; G_v the
        #   vertex of the concave polynomial; rho MCM J1/J4 ratio; R_X temperature factors.
        # Units: G, G_v [W m-2]; B1 [W-1 m2 s-1]; B2 [W-2 m4 s-1]; J [s-1];
        #   alpha, cos(chi), rho, R [dimensionless]; T [K].
        # Domain/constraints: G >= 0 (a negative irradiance raises); fitted to half-hourly
        #   data below 800 m a.s.l., SZA 0-90 deg, G up to about 1000 W m-2, residual
        #   +-6e-4 s-1 (2 sigma); between the data and G_v = 1519 W m-2 the concave
        #   polynomial is extrapolated, at and above G_v the vertex value is held (monotone,
        #   zero slope); alpha <= 0.2 (not above snow, ice, sand).
        # Sign convention: positive J consumes the photolysed species.
        if not shortwave_down.is_floating_point():
            shortwave_down = shortwave_down.to(dtype=torch.get_default_dtype())
        if not bool(torch.isfinite(shortwave_down).all()) or bool(torch.any(shortwave_down < 0)):
            raise ValueError("shortwave_down (solar global irradiance) must be finite and non-negative")
        albedo = self.get_parameter_value("uv_a_surface_albedo", required=True).to(shortwave_down)
        linear = self.get_parameter_value("j_no2_irradiance_linear_coefficient", required=True).to(shortwave_down)
        quadratic = self.get_parameter_value(
            "j_no2_irradiance_quadratic_coefficient", required=True
        ).to(shortwave_down)
        vertex = linear / (2.0 * torch.abs(quadratic))
        held_irradiance = torch.minimum(shortwave_down, vertex)
        j_no2_reference = (1.0 + albedo) * (linear * held_irradiance + quadratic * held_irradiance ** 2)
        cosine = cos_zenith.to(shortwave_down)
        j_o1d_reference = j_no2_reference * self._o1d_to_no2_photolysis_ratio(cosine)
        r_no2, r_o1d = self._photolysis_temperature_factors(temperature, cosine)
        return j_no2_reference * r_no2.to(j_no2_reference), j_o1d_reference * r_o1d.to(j_o1d_reference)
    
    def _calculate_temperature_dependent_rates(
        self,
        temperature: torch.Tensor,
        *,
        reactions: Tuple[str, ...] = ("k_no_o3", "k_isoprene_oh", "k_isoprene_o3", "k_ro2_no"),
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate temperature-dependent reaction rate constants.
        
        Uses Arrhenius equation from unified response functions to ensure
        consistency with other modules in the ecosystem model.
        
        Mathematical Expression:
            k(T) = k_298 * exp(Ea/R * (1/T_ref - 1/T)), T_ref = 298.15 K,
            with k_298 the IUPAC preferred k298 and Ea = (E/R) R from A exp(-(E/R)/T)

        References:
            - Atkinson et al. (2004). ACP 4, 1461 (IUPAC Vol. I, datasheet I.A3.54)
            - Atkinson et al. (2006). ACP 6, 3625 (IUPAC Vol. II, II.A2.10, II.A1.157)
            - MCM v3.3.1 (Jenkin et al. 2015, ACP 15, 11433)

        Parameters:
            temperature : torch.Tensor
                Air temperature [K]
                
        Returns:
            Dict[str, torch.Tensor]: Temperature-corrected rate constants
        """
        # References: Atkinson et al. (2004) ACP 4:1461, IUPAC Vol. I datasheet I.A3.54;
        #   Atkinson et al. (2006) ACP 6:3625, IUPAC Vol. II datasheets II.A2.10, II.A1.157,
        #   II.A3.69 (NO3 + isoprene); MCM v3.3.1 (Jenkin et al. 2015, ACP 15:11433);
        #   O(1D) reactions: Warneck and Williams (2012) Table 7.1 (data of Sander et al.
        #   2006, JPL 06-2).
        # Mathematical expression: k(T) = k_298 exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K;
        #   with Ea = (E/R) R this is the evaluated expression A exp(-(E/R)/T) anchored at the
        #   evaluated k298 (k_no_o3_298, k_isoprene_oh_298, k_isoprene_o3_298, k_ro2_no_298,
        #   k_o1d_*_298); k_isoprene_no3_298 = 3.15e-12 exp(-450/T_ref) reproduces the MCM
        #   v3.3.1 expression 3.15e-12 exp(-450/T) at every T.
        # Parameter meanings: k_298 rate coefficient at T_ref; Ea activation energy
        #   (k_no_o3_activation_energy, k_isoprene_oh_activation_energy,
        #   k_isoprene_o3_activation_energy, k_isoprene_no3_activation_energy,
        #   k_ro2_no_activation_energy, k_o1d_h2o/n2/o2_activation_energy); R gas constant;
        #   T air temperature.
        # Units: T, T_ref [K]; Ea [J mol-1]; R [J mol-1 K-1]; k [cm3 molecule-1 s-1].
        # Domain/constraints: positive temperature; IUPAC evaluated ranges 195-308 K
        #   (NO + O3), 240-430 K (OH + isoprene), 240-360 K (O3 + isoprene) and 250-390 K
        #   (NO3 + isoprene); outside them the same Arrhenius form is extrapolated.
        # Sign convention: Ea > 0 raises k with temperature (NO + O3, O3 + isoprene,
        #   NO3 + isoprene), Ea < 0 lowers it (OH + isoprene, RO2 + NO, O(1D) +
        #   H2O/N2/O2); positive rate constants consume the reactants.
        if not temperature.is_floating_point():
            temperature = temperature.to(dtype=torch.get_default_dtype())
        parameter_names = {
            # NO + O3 -> NO2 + O2
            "k_no_o3": ("k_no_o3_298", "k_no_o3_activation_energy"),
            # BVOC + OH reactions
            "k_isoprene_oh": ("k_isoprene_oh_298", "k_isoprene_oh_activation_energy"),
            # BVOC + O3 reactions
            "k_isoprene_o3": ("k_isoprene_o3_298", "k_isoprene_o3_activation_energy"),
            # BVOC + NO3 reactions (isoprene chemical lifetime diagnostic only; the
            # open mechanism carries no NO3 species, so the default selection omits it)
            "k_isoprene_no3": ("k_isoprene_no3_298", "k_isoprene_no3_activation_energy"),
            # RO2 + NO -> RO + NO2 (high-NOx propagation branch)
            "k_ro2_no": ("k_ro2_no_298", "k_ro2_no_activation_energy"),
            # O(1D) + H2O -> 2 OH and the O(1D) quenching by N2 and O2
            "k_o1d_h2o": ("k_o1d_h2o_298", "k_o1d_h2o_activation_energy"),
            "k_o1d_n2": ("k_o1d_n2_298", "k_o1d_n2_activation_energy"),
            "k_o1d_o2": ("k_o1d_o2_298", "k_o1d_o2_activation_energy"),
        }
        unknown = [name for name in reactions if name not in parameter_names]
        if unknown or not reactions:
            raise ValueError(f"Unknown or empty rate-constant selection: {unknown!r}")
        rate_constants: Dict[str, torch.Tensor] = {}
        for name in reactions:
            reference_name, activation_name = parameter_names[name]
            k_reference = self.get_parameter_value(reference_name, required=True).to(temperature)
            # Preserve activation energy sign; negative Ea implies faster rates at lower temperature.
            activation_energy = self.get_parameter_value(activation_name, required=True).to(temperature)
            rate_constants[name] = arrhenius_temperature_response(
                k_reference, temperature, activation_energy
            ).to(dtype=temperature.dtype)
        return rate_constants
    
    def _o1d_water_reaction_fraction(
        self,
        specific_humidity: torch.Tensor,
        temperature: torch.Tensor,
        air_number_density: torch.Tensor,
    ) -> torch.Tensor:
        """Fraction f of O(1D) atoms that react with water vapour (and form OH)."""
        # References: Warneck and Williams (2012) Table 7.1 (data of Sander et al. 2006,
        #   JPL 06-2); Seinfeld and Pandis (2016) ch. 6 (OH source O(1D) + H2O).
        # Mathematical expression: f = k_H2O [H2O] / (k_H2O [H2O] + k_N2 [N2] + k_O2 [O2]),
        #   [H2O] = x_w n_air, [N2] = x_N2 (1 - x_w) n_air, [O2] = x_O2 (1 - x_w) n_air,
        #   x_w = (q/M_w) / (q/M_w + (1 - q)/M_d).
        # Parameter meanings: k_H2O, k_N2, k_O2 Arrhenius rate coefficients of O(1D) with
        #   H2O (reaction) and N2, O2 (quenching); q specific humidity; M_w, M_d molar
        #   masses of water and dry air; x_N2, x_O2 dry-air mole fractions (SSOT
        #   constants); x_w water-vapour mole fraction of moist air; n_air air number
        #   density.
        # Units: k [cm3 molecule-1 s-1]; n_air, [X] [molecule cm-3]; q [kg kg-1]; M
        #   [g mol-1]; x, f [dimensionless].
        # Domain/constraints: 0 <= q < 1 (checked by the caller); the other O(1D) sinks
        #   (O3, CH4, CO2, ...) are neglected against N2 and O2, and O(1D) is in steady state.
        # Sign convention: not applicable (a branching fraction in [0, 1)).
        dtype_reference = air_number_density
        humidity = specific_humidity.to(dtype_reference)
        water_molar_mass = dtype_reference.new_tensor(self._constants["water_molar_mass"])
        dry_molar_mass = dtype_reference.new_tensor(self._constants["dry_air_molar_mass"])
        water_amount = humidity / water_molar_mass
        water_fraction = water_amount / (water_amount + (1.0 - humidity) / dry_molar_mass)
        dry_density = (1.0 - water_fraction) * air_number_density
        water_density = water_fraction * air_number_density
        nitrogen_density = dtype_reference.new_tensor(self._constants["dry_air_n2_fraction"]) * dry_density
        oxygen_density = dtype_reference.new_tensor(self._constants["dry_air_o2_fraction"]) * dry_density
        rates = self._calculate_temperature_dependent_rates(
            temperature, reactions=("k_o1d_h2o", "k_o1d_n2", "k_o1d_o2")
        )
        water_rate = rates["k_o1d_h2o"].to(dtype_reference) * water_density
        quenching_rate = (
            rates["k_o1d_n2"].to(dtype_reference) * nitrogen_density
            + rates["k_o1d_o2"].to(dtype_reference) * oxygen_density
        )
        return water_rate / (water_rate + quenching_rate)

    def _primary_oh_production(
        self,
        state: StateDict,
        photolysis_rate_o1d: torch.Tensor,
        temperature: torch.Tensor,
        air_number_density: torch.Tensor,
    ) -> torch.Tensor:
        """Primary OH production P_OH from O3 photolysis to O(1D) and O(1D) + H2O."""
        # References: Seinfeld and Pandis (2016) ch. 6; Warneck and Williams (2012) Tables
        #   6.7 and 7.1; Levy (1971). NO2 + hv -> NO + O(3P) forms no OH and is no source.
        # Mathematical expression: P_OH = y f J_O1D [O3], [O3] = chi_O3 n_air.
        # Parameter meanings: y OH yield of O(1D) + H2O -> 2 OH (o1d_water_oh_yield = 2);
        #   f fraction of O(1D) that reacts with H2O; J_O1D photolysis frequency of
        #   O3 -> O(1D) + O2; chi_O3 ozone mixing ratio; n_air air number density.
        # Units: J_O1D [s-1]; chi_O3 [mol mol-1]; n_air, [O3] [molecule cm-3]; P_OH
        #   [molecule cm-3 s-1]; y, f [dimensionless].
        # Domain/constraints: specific_humidity is a required finite mass fraction
        #   0 <= q < 1; no humidity is substituted when it is missing.
        # Sign convention: P_OH >= 0 is a source of the OH diagnostic.
        if "specific_humidity" not in state:
            raise ValueError(
                "ChemistryFunctor requires the input specific_humidity for the O(1D) + H2O OH "
                "source; no humidity is substituted"
            )
        humidity = state["specific_humidity"]
        if not isinstance(humidity, torch.Tensor):
            raise ValueError("specific_humidity must be a torch.Tensor")
        if (
            not bool(torch.isfinite(humidity).all())
            or bool(torch.any(humidity < 0.0))
            or bool(torch.any(humidity >= 1.0))
        ):
            raise ValueError("specific_humidity must be a finite mass fraction with 0 <= q < 1 [kg kg-1]")
        ozone_molecules = self._mixing_ratio_to_number_density(state["o3_mixing_ratio"], air_number_density)
        oh_yield = self.get_parameter_value("o1d_water_oh_yield", required=True).to(air_number_density)
        water_fraction = self._o1d_water_reaction_fraction(humidity, temperature, air_number_density)
        return oh_yield * water_fraction * photolysis_rate_o1d * ozone_molecules

    def compute_oh_concentration(self,
                                 state: StateDict,
                                 photolysis_rate_o1d: torch.Tensor,
                                 temperature: torch.Tensor) -> torch.Tensor:
        """
        Calculate the steady-state OH number density from its primary source and sinks.

        Mathematical Expression:
            P_OH = y f J_O1D [O3],  f = k_H2O[H2O] / (k_H2O[H2O] + k_N2[N2] + k_O2[O2])
            L_OH = k_CO [CO] + k_CH4 [CH4] + k_NO2 [NO2] + k_C5H8 [C5H8]
            [OH] = P_OH / L_OH

        References:
            - Levy, H. (1971). Normal atmosphere: Large radical and formaldehyde concentrations predicted
            - Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics, ch. 6
            - Warneck, P. & Williams, J. (2012). The Atmospheric Chemist's Companion, Table 7.1

        Parameters:
            state : StateDict
                Required: atmospheric_pressure, specific_humidity, no2_mixing_ratio,
                o3_mixing_ratio, co_mixing_ratio, ch4_mixing_ratio
                Optional: isoprene_mixing_ratio
            photolysis_rate_o1d : torch.Tensor
                Photolysis frequency of O3 -> O(1D) + O2 [s-1]
            temperature : torch.Tensor
                Air temperature for rate constant calculations [K]

        Returns:
            torch.Tensor: OH concentration [molecules cm-3]

        Notes:
            - Primary production only: HO2 -> OH recycling (HO2 + NO) is not represented
            - CO and CH4 are required inputs; no background value is substituted
            - No lower or upper bound is applied; a non-positive loss frequency raises
        """
        # References: Seinfeld and Pandis (2016), ch. 2 (ideal-gas number density).
        # Mathematical expression: n = p / (k_B T) with p the atmospheric_pressure input.
        # Parameter meanings: p air pressure, T air temperature, k_B Boltzmann constant.
        # Units: p [Pa]; T [K]; n [molecules cm-3] after the cm3-per-m3 conversion.
        # Domain/constraints: p is the required canonical input atmospheric_pressure
        #   (its contract alias air_pressure is not read); no standard atmosphere is
        #   substituted for a missing pressure.
        # Sign convention: not applicable (a state conversion).
        if 'atmospheric_pressure' not in state:
            raise ValueError(
                'ChemistryFunctor requires the input atmospheric_pressure; no standard '
                'atmosphere is substituted'
            )
        air_pressure = state['atmospheric_pressure']
        air_number_density = self._calculate_number_density(temperature, air_pressure)

        no2_molecules = self._mixing_ratio_to_number_density(state['no2_mixing_ratio'], air_number_density)
        p_oh_total = self._primary_oh_production(state, photolysis_rate_o1d, temperature, air_number_density)

        # OH Loss terms (reaction with various species)
        # References: Atkinson et al. (2006) ACP 6:3625, IUPAC Vol. II datasheets II.A2.12
        #   (OH + CO) and II.A2.3 (OH + CH4); Atkinson et al. (2004) ACP 4:1461, IUPAC Vol. I
        #   datasheet I.A3.43 (OH + NO2 + M); MCM v3.3.1 (Jenkin et al. 2015).
        # Mathematical expression: k(T) = k_298 exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K;
        #   OH + CH4: Ea = (E/R) R with E/R = 1690 K; OH + CO: Ea = 0 (IUPAC II.A2.12,
        #   no temperature dependence over 200-300 K, k_298 taken at 1 atm); OH + NO2:
        #   Ea = 0 by convention (the I.A3.43 falloff is not implemented).
        # Parameter meanings: k_oh_*_298 rate coefficients at T_ref; k_oh_*_activation_energy
        #   activation energies; R gas constant; T air temperature.
        # Units: T, T_ref [K]; Ea [J mol-1]; R [J mol-1 K-1]; k [cm3 molecule-1 s-1].
        # Domain/constraints: positive temperature; OH + CH4 evaluated over 200-300 K; the
        #   OH + CO and OH + NO2 coefficients hold their 1 atm values.
        # Sign convention: Ea > 0 raises k with temperature; positive rate constants
        #   consume OH and the named reactant.

        # OH + CO -> HO2 + CO2
        k_oh_co_298 = self.get_parameter_value("k_oh_co_298", required=True).to(temperature)
        ea_oh_co = self.get_parameter_value("k_oh_co_activation_energy", required=True).to(temperature)
        k_oh_co = arrhenius_temperature_response(k_oh_co_298, temperature, ea_oh_co)

        # OH + CH4 -> CH3 + H2O
        k_oh_ch4_298 = self.get_parameter_value("k_oh_ch4_298", required=True).to(temperature)
        ea_oh_ch4 = self.get_parameter_value("k_oh_ch4_activation_energy", required=True).to(temperature)
        k_oh_ch4 = arrhenius_temperature_response(k_oh_ch4_298, temperature, ea_oh_ch4)

        # OH + NO2 -> HNO3
        k_oh_no2_298 = self.get_parameter_value("k_oh_no2_298", required=True).to(temperature)
        ea_oh_no2 = self.get_parameter_value("k_oh_no2_activation_energy", required=True).to(temperature)
        k_oh_no2 = arrhenius_temperature_response(k_oh_no2_298, temperature, ea_oh_no2)

        # References: Seinfeld and Pandis (2016), ch. 6 (tropospheric OH budget);
        #   Levy (1971); Logan (1985).
        # Mathematical expression: L_OH = k_OH+CO(T) [CO] + k_OH+CH4(T) [CH4]
        #   + k_OH+NO2(T) [NO2] + k_OH+isoprene(T) [C5H8]; [OH] = P_OH / L_OH.
        # Parameter meanings: [CO] and [CH4] are the supplied co_mixing_ratio and
        #   ch4_mixing_ratio converted to number densities with the air number density.
        # Units: mixing ratios mol mol-1; number densities molecules cm-3; rate
        #   coefficients cm3 molecule-1 s-1; L_OH s-1; [OH] molecule cm-3.
        # Domain/constraints: CO and CH4 are always-present OH sinks, so both are
        #   required inputs; no background mixing ratio is substituted for a missing one.
        #   The steady state needs L_OH > 0: a non-positive loss frequency raises; no
        #   loss floor and no bound of [OH] is applied (the OH number density drives the
        #   isoprene + OH sink of the prognostic isoprene state).
        # Sign convention: every sink term is a non-negative first-order OH loss rate.
        missing_sinks = [
            name for name in ('co_mixing_ratio', 'ch4_mixing_ratio') if name not in state
        ]
        if missing_sinks:
            raise ValueError(
                "ChemistryFunctor requires the OH-sink inputs " + ", ".join(missing_sinks)
                + "; no background mixing ratio is substituted"
            )
        co_conc = state['co_mixing_ratio']
        ch4_conc = state['ch4_mixing_ratio']
        self._validate_mixing_ratio(co_conc, 'co_mixing_ratio')
        self._validate_mixing_ratio(ch4_conc, 'ch4_mixing_ratio')

        # Convert to molecules cm-3
        co_molecules = self._mixing_ratio_to_number_density(co_conc, air_number_density)
        ch4_molecules = self._mixing_ratio_to_number_density(ch4_conc, air_number_density)

        # Calculate loss rates
        l_oh_co = k_oh_co * co_molecules
        l_oh_ch4 = k_oh_ch4 * ch4_molecules
        l_oh_no2 = k_oh_no2 * no2_molecules

        # BVOC loss (if present)
        l_oh_bvoc = torch.zeros_like(l_oh_co)
        if 'isoprene_mixing_ratio' in state:
            isoprene_conc = state['isoprene_mixing_ratio']
            self._validate_mixing_ratio(isoprene_conc, 'isoprene_mixing_ratio')
            isoprene_molecules = self._mixing_ratio_to_number_density(isoprene_conc, air_number_density)
            rate_constants = self._calculate_temperature_dependent_rates(
                temperature, reactions=("k_isoprene_oh",)
            )
            l_oh_bvoc = rate_constants['k_isoprene_oh'] * isoprene_molecules

        # Total OH loss rate
        l_oh_total = l_oh_co + l_oh_ch4 + l_oh_no2 + l_oh_bvoc
        if not bool(torch.all(l_oh_total > 0.0)):
            raise ValueError(
                "The steady-state OH diagnostic requires a positive OH loss frequency "
                "L_OH = k_CO[CO] + k_CH4[CH4] + k_NO2[NO2] + k_C5H8[C5H8]; the supplied "
                "sink species give a non-positive value and no loss floor is substituted"
            )
        return p_oh_total / l_oh_total
    
    # Accepted concentration bases of compute_isoprene_chemical_lifetime.
    ISOPRENE_LIFETIME_CONCENTRATION_BASES = ("number_density", "mixing_ratio")

    def compute_isoprene_chemical_lifetime(
        self,
        temperature: torch.Tensor,
        *,
        oh: torch.Tensor,
        o3: torch.Tensor,
        no3: torch.Tensor,
        concentration_basis: Literal["number_density", "mixing_ratio"],
        atmospheric_pressure: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Chemical lifetime of isoprene against OH, O3 and NO3 (pseudo-first-order).

        Mathematical Expression:
            tau_C5H8 = 1 / (k_OH(T)[OH] + k_O3(T)[O3] + k_NO3(T)[NO3])
            k_X(T) = k_X,298 exp(Ea_X/R (1/T_ref - 1/T)), T_ref = 298.15 K

        References:
            - Atkinson et al. (2006). ACP 6, 3625 (IUPAC Vol. II, II.A2.10, II.A1.157, II.A3.69)
            - MCM v3.3.1 (Jenkin et al. 2015, ACP 15, 11433)
            - Atkinson, R. & Arey, J. (2003). Chem. Rev. 103, 4605 (lifetime definition)

        Parameters:
            temperature : torch.Tensor
                Air temperature [K]
            oh, o3, no3 : torch.Tensor
                Oxidant amounts: number densities [molecule cm-3] for
                concentration_basis "number_density", or mixing ratios [mol mol-1]
                for "mixing_ratio"
            concentration_basis : str
                "number_density" or "mixing_ratio" (required; no unit is inferred)
            atmospheric_pressure : torch.Tensor, optional
                Air pressure [Pa]; required by, and only accepted with, "mixing_ratio"

        Returns:
            torch.Tensor: Isoprene chemical lifetime tau_C5H8 [s]

        Notes:
            - A diagnostic of the existing rate owner; it does not change the forward
              mechanism, which carries no NO3 species.
            - A non-positive total loss frequency raises; no lifetime cap or loss floor
              is substituted.
        """
        # References: Atkinson and Arey (2003) Chem. Rev. 103:4605 (atmospheric lifetime
        #   tau = 1/(k[X]) for each oxidant X); Atkinson et al. (2006) ACP 6:3625, IUPAC
        #   Vol. II datasheets II.A2.10 (OH), II.A1.157 (O3), II.A3.69 (NO3); MCM v3.3.1
        #   (Jenkin et al. 2015, ACP 15:11433), NO3 + C5H8 = 3.15e-12 exp(-450/T).
        # Mathematical expression: tau = 1 / (k_OH(T)[OH] + k_O3(T)[O3] + k_NO3(T)[NO3]),
        #   k_X(T) = k_X,298 exp(Ea_X/R (1/T_ref - 1/T)) from the owner rate table; for the
        #   mixing-ratio basis [X] = chi_X n_air, n_air = p / (k_B T).
        # Parameter meanings: k_OH, k_O3, k_NO3 second-order rate coefficients of
        #   isoprene with OH, O3 and NO3 (k_isoprene_oh/o3/no3_298 and their activation
        #   energies); [X] oxidant number density; chi_X oxidant mixing ratio; n_air air
        #   number density; p air pressure; k_B Boltzmann constant; T air temperature.
        # Units: T [K]; p [Pa]; k [cm3 molecule-1 s-1]; [X], n_air [molecule cm-3];
        #   chi_X [mol mol-1]; loss frequency [s-1]; tau [s].
        # Domain/constraints: temperature finite inside [temperature_min, temperature_max]
        #   of the chemistry owner; the rate expressions are evaluated over 240-430 K (OH),
        #   240-360 K (O3) and 250-390 K (NO3) and extrapolated outside; oxidant amounts
        #   finite and non-negative; the total loss frequency must be positive (raises
        #   otherwise, no floor). Pseudo-first-order in isoprene: oxidant amounts are held
        #   fixed over the lifetime, and no other isoprene sink (photolysis, deposition,
        #   Cl) is included.
        # Sign convention: every term k_X[X] >= 0 is an isoprene loss frequency; tau > 0.
        if concentration_basis not in self.ISOPRENE_LIFETIME_CONCENTRATION_BASES:
            raise ValueError(
                "concentration_basis must be one of "
                f"{self.ISOPRENE_LIFETIME_CONCENTRATION_BASES}, got {concentration_basis!r}"
            )
        if not isinstance(temperature, torch.Tensor):
            raise ValueError("temperature must be a torch.Tensor [K]")
        if not temperature.is_floating_point():
            temperature = temperature.to(dtype=torch.get_default_dtype())
        if not torch.isfinite(temperature).all():
            raise ValueError("air_temperature must contain only finite values")
        temp_min = self.get_parameter_value("temperature_min", required=True)
        temp_max = self.get_parameter_value("temperature_max", required=True)
        validate_temperature_range(temperature, min_temp=temp_min, max_temp=temp_max)
        oxidants = {"oh": oh, "o3": o3, "no3": no3}
        for name, value in oxidants.items():
            self._validate_mixing_ratio(value, name)
        if concentration_basis == "mixing_ratio":
            if atmospheric_pressure is None:
                raise ValueError(
                    "The mixing_ratio basis requires atmospheric_pressure [Pa]; no standard "
                    "atmosphere is substituted"
                )
            air_number_density = self._calculate_number_density(
                temperature, atmospheric_pressure.to(temperature)
            )
            densities = {
                name: self._mixing_ratio_to_number_density(value.to(air_number_density), air_number_density)
                for name, value in oxidants.items()
            }
        else:
            if atmospheric_pressure is not None:
                raise ValueError(
                    "The number_density basis takes no atmospheric_pressure; pass "
                    "concentration_basis='mixing_ratio' for mixing ratios"
                )
            densities = {name: value.to(temperature) for name, value in oxidants.items()}
        rates = self._calculate_temperature_dependent_rates(
            temperature, reactions=("k_isoprene_oh", "k_isoprene_o3", "k_isoprene_no3")
        )
        loss_frequency = (
            rates["k_isoprene_oh"] * densities["oh"]
            + rates["k_isoprene_o3"] * densities["o3"]
            + rates["k_isoprene_no3"] * densities["no3"]
        )
        if not bool(torch.all(loss_frequency > 0.0)):
            raise ValueError(
                "The isoprene chemical lifetime requires a positive loss frequency "
                "k_OH[OH] + k_O3[O3] + k_NO3[NO3]; the supplied oxidants give a "
                "non-positive value and no loss floor is substituted"
            )
        return 1.0 / loss_frequency

    def _calculate_number_density(self, 
                                  temperature: torch.Tensor, 
                                  pressure: torch.Tensor) -> torch.Tensor:
        """
        Calculate air number density using ideal gas law.
        
        Mathematical Expression:
            n = P / (k_B * T) [molecules m-3]
            Convert to molecules cm-3: n_cm3 = n * 1e-6
            
        Parameters:
            temperature : torch.Tensor
                Air temperature [K]
            pressure : torch.Tensor
                Air pressure [Pa]
                
        Returns:
            torch.Tensor: Number density [molecules cm-3]
        """
        # References: Seinfeld and Pandis 2016
        # Formula: n = P / (k_B * T); n_cm3 = n * 1e-6
        # Parameters: boltzmann_constant
        if not torch.isfinite(temperature).all():
            raise ValueError("air_temperature must contain only finite values")
        if not torch.isfinite(pressure).all() or torch.any(pressure <= 0):
            raise ValueError("atmospheric_pressure must contain only finite positive values")
        # Number density in molecules m-3
        boltzmann = temperature.new_tensor(self._constants['boltzmann_constant'])
        n_m3 = pressure / (boltzmann * temperature)
        # Convert to molecules cm-3 with the SSOT cm3-per-m3 factor
        n_cm3 = n_m3 / temperature.new_tensor(self._constants['cubic_centimeters_per_cubic_meter'])
        return n_cm3
    
    def _mixing_ratio_to_number_density(self,
                                       mixing_ratio: torch.Tensor,
                                       air_density: torch.Tensor) -> torch.Tensor:
        """
        Convert mixing ratio to number density.
        
        Mathematical Expression:
            n_species = mixing_ratio * n_air [molecules cm-3]
            
        Parameters:
            mixing_ratio : torch.Tensor
                Volume mixing ratio [mol mol-1]
            air_density : torch.Tensor
                Air number density [molecules cm-3]
                
        Returns:
            torch.Tensor: Species number density [molecules cm-3]
        """
        # References: Seinfeld and Pandis 2016
        # Formula: n_species = mixing_ratio * n_air
        # Parameters: air_number_density
        return mixing_ratio * air_density

    @staticmethod
    def _validate_mixing_ratio(value: torch.Tensor, name: str) -> None:
        """Require a finite, non-negative mixing-ratio tensor."""

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        validate_positive_values(value, name)


    def _closed_box_declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare the closed-box inventory, energy, and tendency cochains."""

        def space(
            primality: Literal["primal", "dual"],
            n_dofs: int,
            units: str,
            semantic_name: str,
            orientation: str,
        ) -> CochainSpaceSpec:
            return CochainSpaceSpec(
                complex_id="atmosphere_chemistry_closed_box",
                degree=0,
                primality=primality,
                n_dofs=n_dofs,
                units=units,
                extensivity="extensive",
                physical_domain="atmosphere",
                semantic_name=semantic_name,
                orientation_convention=orientation,
                boundary_role="interior",
            )

        return (
            space("primal", 4, "mol", "closed_box_species_amount", "positive_inventory"),
            space("primal", 1, "J", "closed_box_internal_energy", "energy_stored_positive"),
            space("dual", 4, "mol s-1", "closed_box_species_tendency", "production_positive"),
            space("dual", 1, "W", "closed_box_internal_energy_tendency", "energy_gain_positive"),
        )

    def last_strict_iphs_audit(self) -> Optional[Dict[str, Any]]:
        """Return the Level 5 runtime-point audit of the last emitted closed-box step."""

        audit = self._last_strict_iphs_audit
        return None if audit is None else dict(audit)

    def _closed_box_scalar(self, state: Mapping[str, Any], key: str) -> torch.Tensor:
        """Return one closed-box input as a finite scalar tensor."""

        if key not in state:
            raise ValueError(f"{key} is required for the closed constant-volume box")
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{key} must be a torch.Tensor for the closed constant-volume box")
        if not value.is_floating_point():
            raise ValueError(f"{key} must be a floating-point tensor")
        if value.numel() != 1:
            raise ValueError(
                f"{key} must be a single scalar; the closed box uses scalar support and "
                f"does not treat shape {tuple(value.shape)} as a batch of boxes"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} must be finite")
        return value.reshape(())

    def _closed_box_thermodynamic_point(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        """Evaluate the existing closed-box thermodynamic domain without drift or audit writes.

        References:
          - NASA TM-4513; Seinfeld and Pandis (2016); Kondepudi and Prigogine (1998).
        Mathematical expression:
          sum_i n_i u_i(T) = U; p_i = n_i R T / V;
          mu_i = h_i(T) - T s_i(T) + R T ln(p_i / p0); A = -nu^T mu.
        Parameter meanings:
          n_i species inventory; U internal energy; V fixed volume; nu existing
          stoichiometry; T inverted temperature; A forward reaction affinity.
        Units:
          n mol; U J; V m3; T K; p Pa; mu and A J mol-1.
        Domain/constraints:
          finite scalar inputs with one dtype/device, n_i > 0, V > 0,
          T in one NASA7 interval within the declared kinetic bounds, A > 0.
          The same owner checks initial and actual Euler-candidate states;
          no inventory floor, reverse reaction, state advance, or point audit.
        Sign convention:
          A > 0 supports the existing forward NO + O3 -> NO2 + O2 law.
        """

        names = self.CLOSED_BOX_STATE
        components = [self._closed_box_scalar(state, key) for key in names]
        volume = self._closed_box_scalar(state, "box_volume")
        reference = components[0]
        for key, value in zip(self.CLOSED_BOX_INPUTS, components + [volume]):
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(
                    f"{key} must share dtype {reference.dtype} and device "
                    f"{reference.device} with the other closed-box inputs"
                )
        for key, value in zip(self.CLOSED_BOX_SPECIES, components[:4]):
            if not bool(value > 0.0):
                raise ValueError(
                    f"{key} must be strictly positive; zero inventory is outside "
                    "the ideal-gas chemical-potential domain"
                )
        if not bool(volume > 0.0):
            raise ValueError("box_volume must be strictly positive")

        # Same-step state graph: upstream gradients are kept; inference inputs
        # obtain a local leaf so the potentials can be differentiated.
        x = torch.stack(components)
        if not x.requires_grad:
            x.requires_grad_(True)
        amounts = x[:4]
        energy = x[4]

        temperature_lower = self.get_parameter_value("temperature_min", required=True).to(x)
        temperature_upper = self.get_parameter_value("temperature_max", required=True).to(x)
        temperature = self.temperature_from_internal_energy(
            energy,
            amounts,
            temperature_lower=temperature_lower,
            temperature_upper=temperature_upper,
        )
        mixture = self.ideal_gas_mixture_properties(temperature, amounts, volume)
        stoichiometry = x.new_tensor(self.CLOSED_BOX_STOICHIOMETRY)
        affinity = -(stoichiometry[:4] @ mixture["chemical_potential"])
        if not bool(torch.isfinite(affinity)) or not bool(affinity > 0.0):
            raise ValueError(
                "closed box reaction affinity A = -nu^T mu must be positive for the "
                f"forward mass-action law; got {float(affinity.detach()):.6g} J mol-1"
            )
        return {
            "state": x,
            "amounts": amounts,
            "energy": energy,
            "volume": volume,
            "temperature": temperature,
            "mixture": mixture,
            "stoichiometry": stoichiometry,
            "affinity": affinity,
        }

    def _evaluate_closed_constant_volume_box(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        """Evaluate the closed-box thermodynamic state, structure, and drift once.

        References:
          - Grmela and Ottinger (1997), Phys. Rev. E 56, 6620; Ottinger (2005) ch. 1-2.
          - Mielke (2011), Nonlinearity 24, 1329: Onsager operator of one reaction.
          - Kondepudi and Prigogine (1998) ch. 9: affinity and entropy production.
          - McBride, Gordon and Reno (1993), NASA TM-4513; Seinfeld and Pandis (2016).
          - Atkinson et al. (2004), ACP 4, 1461: NO + O3 rate coefficient.
        Mathematical expression:
          x = (n_NO, n_NO2, n_O3, n_O2, U), fixed V;
          T = T(U, n) from sum_i n_i u_i(T) = U;
          p_i = n_i R T / V, mu_i = h_i(T) - T s_i(T) + R T ln(p_i / p0);
          E = U, S = sum_i n_i (s_i(T) - R ln(p_i / p0));
          A = -nu^T mu, nu = (-1, +1, -1, +1, 0);
          r = k(T) N_A n_NO n_O3 / (c_v V);
          L = 0 (zero_field), M = (r T / A) nu nu^T (absorbed reaction);
          x_dot = L grad(E) + M grad(S) = nu r;  M grad(E) = 0;
          sigma = grad(S)^T M grad(S) = A r / T;  P = A r.
        Parameter meanings:
          n_i molar inventories; U total internal energy including NASA7
          formation terms; V fixed box volume; T temperature; mu_i chemical
          potentials; A physical affinity; k Arrhenius NO + O3 coefficient;
          N_A Avogadro constant; c_v cm3-per-m3 factor; nu stoichiometry;
          M GENERIC friction operator; sigma entropy production rate; P power.
        Units:
          n mol; U J; V m3; T K; p Pa; mu, A J mol-1; k cm3 molecule-1 s-1;
          N_A mol-1; c_v cm3 m-3; r mol s-1; S J K-1; M mol2 J-1 K s-1;
          x_dot mol s-1 and W; sigma J K-1 s-1; P W.
        Domain/constraints:
          all inputs finite scalar tensors of one dtype/device; n_i > 0 and
          V > 0 (ideal-gas chemical potential domain, no epsilon); T solved
          inside [temperature_min, temperature_max], one NASA7 interval;
          A > 0 so that M is symmetric positive semidefinite; forward mass
          action only, no reverse reaction is added.
        Sign convention:
          r > 0 consumes NO and O3 and produces NO2 and O2; A > 0 drives the
          forward reaction; sigma >= 0; U is unchanged (closed, adiabatic,
          rigid box, no work and no heat exchange).
        """

        self._last_strict_iphs_audit = None
        names = self.CLOSED_BOX_STATE
        point = self._closed_box_thermodynamic_point(state)
        x = point["state"]
        amounts = point["amounts"]
        energy = point["energy"]
        volume = point["volume"]
        temperature = point["temperature"]
        mixture = point["mixture"]
        stoichiometry = point["stoichiometry"]
        affinity = point["affinity"]
        rate_constant = self._calculate_temperature_dependent_rates(
            temperature, reactions=("k_no_o3",)
        )["k_no_o3"]
        avogadro = x.new_tensor(self._constants["avogadro_number"])
        cm3_per_m3 = x.new_tensor(self._constants["cubic_centimeters_per_cubic_meter"])
        rate = rate_constant * avogadro / cm3_per_m3 * amounts[0] * amounts[2] / volume
        entropy = mixture["entropy"]
        mobility = (rate * temperature / affinity) * torch.outer(stoichiometry, stoichiometry)

        # One SSOT tolerance governs the skew, PSD, degeneracy, entropy, and
        # reaction-closure audits of the runtime point (validation.iphs).
        runtime_point_tolerance = float(
            self.get_parameter_value(
                "runtime_point_tolerance",
                namespace="validation.iphs",
                source_order=("manager",),
                required=True,
            )
        )
        spec = IPHSStructureSpec(
            name=f"{self.name}.closed_constant_volume_box",
            state_vars=names,
            energy_functional="E = U",
            entropy_functional="S = sum_i n_i (s_i(T) - R ln(p_i / p0)), T = T(U, n)",
            J=torch.zeros_like(mobility),
            M=mobility,
            G=None,
            stoichiometric_N=stoichiometry.reshape(-1, 1),
            ports=(),
            formulation="generic",
            poisson_operator="zero_field",
            reaction_representation="absorbed",
            skew_tol=runtime_point_tolerance,
            psd_tol=runtime_point_tolerance,
            degeneracy_tol=runtime_point_tolerance,
            contribution_count=1,
            contribution_sources=(self.CANONICAL_DRIFT_SOURCE,),
        )
        drift = spec.canonical_drift(
            state=x,
            energy_value=energy,
            entropy_value=entropy,
            reaction_rates=rate.reshape(1),
        )
        entropy_effort = torch.autograd.grad(
            entropy, x, create_graph=torch.is_grad_enabled(), retain_graph=True,
        )[0]
        energy_effort = torch.autograd.grad(
            energy, x, create_graph=torch.is_grad_enabled(), retain_graph=True,
        )[0]

        # Ledger and affinity-power audit through the canonical reaction owner;
        # no second state advance is taken from it (dt=None).
        network = advance_reaction_network(
            name=f"{self.name}.closed_constant_volume_box_network",
            species=names,
            reactions=("no_o3_titration",),
            stoichiometric_N=stoichiometry.reshape(-1, 1),
            rates=rate.reshape(1),
            dt=None,
            conservation_matrix=x.new_tensor(
                [ledger for _, ledger in self.CLOSED_BOX_ELEMENT_LEDGERS]
            ),
            strict_conservation=True,
            require_conservation_matrix=True,
            affinities=affinity.reshape(1),
            affinity_sign_convention="affinity_times_rate_nonnegative",
            require_thermodynamic_point_audit=True,
        )
        # Level 5 runtime-point verification on the same bound potentials:
        # L e_S = 0, M e_E = 0, sigma >= 0, e_E^T N r = 0, e_S^T N r >= 0,
        # absorbed M e_S = N r, canonical zero-field Jacobi basis.
        strict_audit = spec.validate(
            required_level=5,
            state=x,
            energy_value=energy,
            entropy_value=entropy,
            reaction_rates=rate.reshape(1),
        )
        power = affinity * rate
        entropy_production_rate = power / temperature
        return {
            "strict_audit": strict_audit,
            "state": x,
            "amounts": amounts,
            "energy": energy,
            "volume": volume,
            "temperature": temperature,
            "pressure": mixture["pressure"],
            "entropy": entropy,
            "affinity": affinity,
            "rate_constant": rate_constant,
            "rate": rate,
            "stoichiometry": stoichiometry,
            "mobility": mobility,
            "energy_effort": energy_effort,
            "entropy_effort": entropy_effort,
            "drift": drift,
            "power": power,
            "entropy_production_rate": entropy_production_rate,
            "network": network,
            "spec": spec,
        }

    def _forward_closed_constant_volume(
        self,
        state: StateDict,
        time: float,
        dt: float,
    ) -> FluxDict:
        """Advance the closed constant-volume box by one explicit Euler step.

        References: see _evaluate_closed_constant_volume_box; Sandu et al.
          (1997) for the positivity domain of the explicit step.
        Mathematical expression: Delta x = dt * x_dot with x_dot from the
          GENERIC absorbed-reaction drift; Delta U = dt * (M grad E)_U = 0.
        Parameter meanings: dt integration step; x_dot canonical drift.
        Units: Delta n mol; Delta U J; dt s.
        Domain/constraints: dt finite and non-negative; the actual candidate
          must retain the thermodynamic domain (n_i > 0, valid T(U, n), A > 0).
          An invalid step is rejected before emission without changing dt or inventories.
        Sign convention: positive species increments produce material; Delta U is zero.
        """

        del time
        box = self._evaluate_closed_constant_volume_box(state)
        drift = box["drift"]
        dt_value = float(dt)
        if not torch.isfinite(drift.new_tensor(dt_value)) or dt_value < 0.0:
            raise ValueError("dt must be finite and non-negative")
        dt_tensor = drift.new_tensor(dt_value)
        self._validate_explicit_euler_positivity(
            {name: box["amounts"][i] for i, name in enumerate(self.CLOSED_BOX_SPECIES)},
            {name: drift[i] for i, name in enumerate(self.CLOSED_BOX_SPECIES)},
            dt_tensor,
        )
        increments = drift * dt_tensor
        # References: the closed-box thermodynamic domain above; IEEE-754 arithmetic.
        # Mathematical expression: x_candidate = x + dt*x_dot; require n_i > 0 and A > 0.
        # Parameter meanings: x is the current inventory/energy; dt*x_dot is the emitted increment.
        # Units: candidate inventories mol, energy J; fixed volume m3.
        # Domain/constraints: inspect the actual tensor sum, including round-to-zero endpoints.
        # Sign convention: no projection, clipping, extra drift, or second state advance.
        candidate = box["state"] + increments
        candidate_state = {name: candidate[i] for i, name in enumerate(self.CLOSED_BOX_STATE)}
        candidate_state["box_volume"] = box["volume"]
        self._closed_box_thermodynamic_point(candidate_state)
        self._last_strict_iphs_audit = dict(box["strict_audit"])

        output: FluxDict = {}
        for index, name in enumerate(self.CLOSED_BOX_STATE):
            output[name] = UpdateItem(
                value=increments[index],
                mode=UpdateMode.ACCUMULATE,
                unit='J' if name == 'box_internal_energy' else 'mol',
                per_timestep=True,
                source_functor=self.name,
            )
        diagnostics = {
            'diagnostics.chemistry.box_temperature': (box["temperature"], 'K'),
            'diagnostics.chemistry.box_pressure': (box["pressure"], 'Pa'),
            'diagnostics.chemistry.box_entropy': (box["entropy"], 'J K-1'),
            'diagnostics.chemistry.box_entropy_production_rate': (
                box["entropy_production_rate"], 'J K-1 s-1'
            ),
            'diagnostics.chemistry.box_reaction_affinity': (box["affinity"], 'J mol-1'),
            'diagnostics.chemistry.box_reaction_rate': (box["rate"], 'mol s-1'),
            'diagnostics.chemistry.box_reaction_power': (box["power"], 'W'),
            'diagnostics.chemistry.box_rate_constant_no_o3': (
                box["rate_constant"], 'cm3 molecule-1 s-1'
            ),
        }
        for name, (value, unit) in diagnostics.items():
            output[name] = UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=True,
                source_functor=self.name,
            )
        return output

    def _check_closed_box_ledgers(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Audit element and energy ledgers of the closed box.

        References: Feinberg (2019), reaction network stoichiometry;
          Kondepudi and Prigogine (1998), closed-system energy balance.
        Mathematical expression:
          R_C = |C Delta n| / sum_i |C_i Delta n_i| for C in {C_N, C_O};
          R_U = |Delta U| / max(|U|, |Delta U|).
        Parameter meanings: Delta n integrated inventory increments, C element
          ledger rows, Delta U emitted internal-energy increment.
        Units: all residuals dimensionless; Delta n mol; Delta U, U J.
        Domain/constraints: a zero scale (no reaction) yields a zero residual.
        Sign convention: residuals are magnitudes; exact ledgers give 0.
        """

        def increment(key: str) -> torch.Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing conservation output '{key}'")
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise TypeError(f"Closed-box conservation output '{key}' must be a scalar tensor.")
            if not bool(torch.isfinite(value)):
                raise ValueError(f"Closed-box conservation output '{key}' must be finite.")
            return value.reshape(())

        deltas = torch.stack([increment(name) for name in self.CLOSED_BOX_STATE]).detach()
        residuals: Dict[str, float] = {}
        for law, ledger in self.CLOSED_BOX_ELEMENT_LEDGERS:
            row = deltas.new_tensor(ledger)
            scale = torch.sum(torch.abs(row * deltas))
            residual = torch.abs(row @ deltas)
            residuals[law] = 0.0 if float(scale) == 0.0 else float(residual / scale)
        delta_energy = deltas[4]
        energy = self._closed_box_scalar(input_state, "box_internal_energy").detach()
        scale_energy = torch.maximum(torch.abs(energy), torch.abs(delta_energy))
        residuals["internal_energy"] = (
            0.0 if float(delta_energy) == 0.0 else float(torch.abs(delta_energy) / scale_energy)
        )
        return residuals

    @staticmethod
    def _validate_explicit_euler_positivity(
        concentrations: Mapping[str, torch.Tensor],
        tendencies: Mapping[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> None:
        """Reject an Euler step that would make a retained species negative.

        References:
          - Sandu et al. (1997), Atmos. Environ. 31, 3151-3166.
          - Hairer and Wanner (1996), Solving ODEs II, stiff systems.
        Mathematical expression:
          x_next = x + dt * f(x); for f_i < 0,
          dt <= min_i(x_i / -f_i) is required for x_next >= 0.
        Parameter meanings:
          x is a retained species mixing ratio [mol mol-1], f is its net
          tendency [mol mol-1 s-1], and dt is the integration step [s].
        Open/closed route:
          An inadmissible step is rejected before any UpdateItem is emitted;
          no negative residual is clipped, discarded, or routed as a source.
        """

        violating_species: List[str] = []
        admissible_limits: List[torch.Tensor] = []
        for name, concentration in concentrations.items():
            tendency = tendencies[name]
            negative_tendency = tendency < 0.0
            if bool(torch.any(negative_tendency).item()):
                limits = torch.where(
                    negative_tendency,
                    concentration / (-tendency),
                    torch.full_like(tendency, torch.inf),
                )
                admissible_limits.append(torch.amin(limits))
            candidate = concentration + dt * tendency
            if bool(torch.any(candidate < 0.0).item()):
                violating_species.append(name)

        if not violating_species:
            return

        max_admissible_dt = min(
            float(limit.detach().cpu())
            for limit in admissible_limits
        )
        dt_value = float(dt.detach().cpu())
        raise ValueError(
            "Chemistry explicit Euler positivity domain exceeded: "
            f"dt={dt_value:.15g} s, "
            f"max_admissible_dt={max_admissible_dt:.15g} s, "
            f"violating_species={','.join(violating_species)}. "
            "Reduce dt or use simulator-managed FunctorMetadata.substep_count; "
            "no state update was emitted."
        )
    
    def forward(self,
        # References: Seinfeld and Pandis (2016); Finlayson-Pitts and Pitts (2000);
        #   Atkinson and Arey (2003); Jenkin and Clemitshaw (2000).
        # Mathematical expression: d[NO]/dt = J_NO2 [NO2] - k_NO_O3 [NO][O3];
        #   d[NO2]/dt = k_NO_O3 [NO][O3] - J_NO2 [NO2]; d[O3]/dt = J_NO2 [NO2] - k_NO_O3 [NO][O3];
        #   Delta[X] = dt d[X]/dt; k(T) = k_298 exp(Ea/R (1/T_ref - 1/T)), T_ref = 298.15 K;
        #   J_NO2 = (1 + alpha)(B1 G + B2 G^2) R_NO2(T), G = SW; BVOC + OH -> products
        #   (k_BVOC_OH) with [OH] = y f J_O1D [O3] / L_OH.
        # Parameter meanings: [X] mixing ratios of the box species; k_298 and Ea Arrhenius
        #   parameters; alpha, B1, B2 Trebs et al. (2009) coefficients; SW downwelling shortwave; theta zenith
        #   angle; R_NO2 photolysis temperature factor; y, f, J_O1D, L_OH as in
        #   compute_oh_concentration.
        # Units: mixing ratios [mol mol-1]; tendencies [mol mol-1 s-1]; number densities
        #   [molecule cm-3]; k [cm3 molecule-1 s-1]; J [s-1]; SW [W m-2]; T [K].
        # Domain/constraints: reduced-form box model for ecosystem-atmosphere coupling; the
        #   required temperature, NO, NO2, O3, shortwave, cos_zenith and humidity inputs
        #   must be finite; increments are explicit Euler over dt.
        # Sign convention: production terms are positive for the species formed; a negative
        #   Delta[X] is net loss.
                state: StateDict,
                time: float,
                dt: float,
                **kwargs) -> FluxDict:
        """
        Compute atmospheric chemistry reaction rates and species changes.
        
        This method calculates the time evolution of chemical species using
        a reduced-form box model with temperature-dependent kinetics and
        light-dependent photolysis. Designed for ecosystem-atmosphere coupling.
        
        Parameters:
            state : StateDict
                Current system state containing chemical and meteorological variables
                Required keys: air_temperature, no_mixing_ratio, no2_mixing_ratio, 
                              o3_mixing_ratio
                Required radiation keys: shortwave_down, cos_zenith
                Required pressure key: atmospheric_pressure
                Required OH keys: co_mixing_ratio, ch4_mixing_ratio, specific_humidity
                Optional keys: isoprene_mixing_ratio
            time : float
                Current simulation time [s]
            dt : float
                Time step [s]
                
        Returns:
            FluxDict: Canonical state increments and namespaced diagnostics.
                Species increments are UpdateItem(ACCUMULATE) values in
                [mol mol-1] already integrated over the current time step.
        
        Raises:
            ValueError: If a required input is missing or an input is invalid
        """
        if self.physical_boundary == "closed_constant_volume":
            return self._forward_closed_constant_volume(state, time, dt)

        # References: Seinfeld and Pandis 2016; Jenkin and Clemitshaw 2000
        # Mathematical expression: d[X]/dt = P - L and Delta[X] = dt*d[X]/dt.
        # Parameters: temperature_min/max.
        # Units: tendencies are mol mol-1 s-1; state increments are mol mol-1.
        # Open/closed route: NO and NO2 form a closed reaction ledger. O3 and
        # isoprene losses remain explicit canonical state increments consumed
        # by the shared atmospheric species ledger.
        # Extract required variables
        if 'air_temperature' not in state:
            raise ValueError("air_temperature is required for chemistry calculations")
        if 'no_mixing_ratio' not in state:
            raise ValueError("no_mixing_ratio is required")
        if 'no2_mixing_ratio' not in state:
            raise ValueError("no2_mixing_ratio is required")
        if 'o3_mixing_ratio' not in state:
            raise ValueError("o3_mixing_ratio is required")
        if 'shortwave_down' not in state:
            raise ValueError("shortwave_down is required for chemistry calculations")
            
        temperature = state['air_temperature']
        no_conc = state['no_mixing_ratio']
        no2_conc = state['no2_mixing_ratio']
        o3_conc = state['o3_mixing_ratio']

        # Validate inputs - use wider temperature range for atmospheric chemistry
        # Atmospheric chemistry can occur over wider temperature range than biological processes
        if not torch.isfinite(temperature).all():
            raise ValueError("air_temperature must contain only finite values")
        temp_min = self.get_parameter_value("temperature_min", required=True)
        temp_max = self.get_parameter_value("temperature_max", required=True)
        validate_temperature_range(temperature, min_temp=temp_min, max_temp=temp_max)

        self._validate_mixing_ratio(no_conc, 'no_mixing_ratio')
        self._validate_mixing_ratio(no2_conc, 'no2_mixing_ratio')
        self._validate_mixing_ratio(o3_conc, 'o3_mixing_ratio')
        
        # References: Trebs et al. (2009) Atmos. Meas. Tech. 2, 725; Saunders et al. (2003).
        # Mathematical expression: J_NO2 = (1 + alpha)(B1 G + B2 G^2) R_NO2(T, theta), G = SW_down;
        #   J_O1D = (1 + alpha)(B1 G + B2 G^2) rho(theta) R_O1D(T, theta).
        # Parameter meanings: SW_down the solar global irradiance G (it carries the solar
        #   geometry and cloud attenuation); theta the solar zenith angle of the required
        #   cos_zenith input; rho the MCM J1/J4 ratio; R_X the photolysis temperature factors.
        # Units: SW_down [W m-2]; J [s-1]; cos(theta) [dimensionless].
        # Domain/constraints: SW_down >= 0 and -1 <= cos(theta) <= 1, finite; cos_zenith is
        #   required: rho depends on the ozone slant path, which broadband shortwave cannot
        #   separate from cloud attenuation, so no geometry is derived from SW_down.
        # Sign convention: radiation enters through the atmospheric_radiation port;
        #   positive J consumes the photolysed species.
        shortwave_down = state['shortwave_down']
        if not torch.isfinite(shortwave_down).all() or torch.any(shortwave_down < 0):
            raise ValueError("shortwave_down must contain only finite non-negative values")
        if 'cos_zenith' not in state:
            raise ValueError(
                "cos_zenith is required: J_NO2 uses max(0, cos(theta)) and J(O1D) the MCM "
                "ratio rho(theta); no solar geometry is derived from shortwave_down"
            )
        cos_zenith = state['cos_zenith']
        if not torch.isfinite(cos_zenith).all():
            raise ValueError("cos_zenith must contain only finite values")
        if bool(torch.any(cos_zenith.abs() > 1.0)):
            raise ValueError("cos_zenith must lie in [-1, 1] (the cosine of the solar zenith angle)")

        # Calculate photolysis rates
        j_no2, j_o1d = self._calculate_photolysis_rates(shortwave_down, cos_zenith, temperature)
        
        # Calculate temperature-dependent rate constants
        rate_constants = self._calculate_temperature_dependent_rates(temperature)
        k_no_o3 = rate_constants['k_no_o3']
        
        # Calculate air number density (needed for unit conversions)
        # References: Seinfeld and Pandis (2016), ch. 2 (ideal-gas number density).
        # Mathematical expression: n = p / (k_B T) with p the atmospheric_pressure input.
        # Parameter meanings: p air pressure, T air temperature, k_B Boltzmann constant.
        # Units: p [Pa]; T [K]; n [molecules cm-3] after the cm3-per-m3 conversion.
        # Domain/constraints: p is the required canonical input atmospheric_pressure
        #   (its contract alias air_pressure is not read); no standard atmosphere is
        #   substituted for a missing pressure.
        # Sign convention: not applicable (a state conversion).
        if 'atmospheric_pressure' not in state:
            raise ValueError(
                'ChemistryFunctor requires the input atmospheric_pressure; no standard '
                'atmosphere is substituted'
            )
        air_pressure = state['atmospheric_pressure']
        air_number_density = self._calculate_number_density(temperature, air_pressure)
        
        # Chemical reaction rates (box model)
        # Convert mixing ratios to number densities for proper unit handling
        no_molecules = self._mixing_ratio_to_number_density(no_conc, air_number_density)
        no2_molecules = self._mixing_ratio_to_number_density(no2_conc, air_number_density)
        o3_molecules = self._mixing_ratio_to_number_density(o3_conc, air_number_density)
        
        # NO + O3 -> NO2 + O2 [molecules cm^-3 s^-1]
        reaction_no_o3_molecules = k_no_o3 * no_molecules * o3_molecules
        
        # NO2 photolysis: NO2 + hnu -> NO + O (followed by O + O2 -> O3) [molecules cm^-3 s^-1]
        photolysis_no2_molecules = j_no2 * no2_molecules
        
        # Convert back to mixing ratio change rates [mol mol^-1 s^-1]
        reaction_no_o3 = reaction_no_o3_molecules / air_number_density
        photolysis_no2 = photolysis_no2_molecules / air_number_density
        
        # Steady-state OH from the O(1D) + H2O source and the CO/CH4/NO2/isoprene sinks
        oh_conc = self.compute_oh_concentration(state, j_o1d, temperature)

        # References:
        #   1. Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics,
        #      3rd ed., Ch. 6, Eqs. (6.45)-(6.49) (peroxy radical NO-to-NO2 oxidation
        #      and net photochemical odd-oxygen production).
        #   2. Saunders, S.M. et al. (2003). Atmos. Chem. Phys., 3, 161-180 (MCM v3.3.1
        #      peroxy radical kinetics k_RO2_NO = 2.7e-12 * exp(360/T)).
        #   3. Peeters, J. et al. (2014). J. Phys. Chem. A, 118, 8625-8643 (LIM1
        #      low-NOx peroxy radical termination and isomerization competing sink).
        # Mathematical expression:
        #   beta_NO = (k_RO2_NO(T) * [NO]) / (k_RO2_NO(T) * [NO] + k_loss,RO2)
        #   R_RO2_NO = gamma_NO2 * beta_NO * R_C5H8_OH
        #   d[Ox]/dt = d([O3] + [NO2])/dt = R_RO2_NO - R_C5H8_O3
        #   R_C,ox = N_C,isop * (R_C5H8_OH + R_C5H8_O3)
        # Parameter meanings:
        #   - k_RO2_NO(T): temperature-dependent RO2 + NO rate coefficient [cm3 molecule-1 s-1]
        #   - k_loss,RO2: first-order RO2 termination/isomerization loss rate [s-1]
        #   - beta_NO: dimensionless NO branching fraction of isoprene peroxy radicals [0, 1]
        #   - gamma_NO2: stoichiometric NO -> NO2 conversions per OH-oxidized isoprene [dimensionless]
        #   - N_C,isop: carbon atoms per isoprene molecule [dimensionless, 5.0]
        k_ro2_no = rate_constants['k_ro2_no']
        k_ro2_loss = self.get_parameter_value(
            "k_ro2_termination_loss", required=True
        ).to(photolysis_no2)
        if bool(torch.any(~torch.isfinite(k_ro2_loss) | (k_ro2_loss <= 0.0))):
            raise ValueError(
                "atmosphere.chemistry.k_ro2_termination_loss must be finite and positive."
            )
        ro2_no_pseudo_first_order = k_ro2_no * no_molecules
        ro2_no_branching_fraction = ro2_no_pseudo_first_order / (
            ro2_no_pseudo_first_order + k_ro2_loss
        )

        isoprene_oh_rate = torch.zeros_like(photolysis_no2)
        isoprene_o3_rate = torch.zeros_like(photolysis_no2)
        isoprene_ro2_no_rate = torch.zeros_like(photolysis_no2)
        carbon_oxidized_rate = torch.zeros_like(photolysis_no2)
        has_isoprene = 'isoprene_mixing_ratio' in state
        concentrations = {
            'no_mixing_ratio': no_conc,
            'no2_mixing_ratio': no2_conc,
            'o3_mixing_ratio': o3_conc,
        }

        # BVOC chemistry (if isoprene is present)
        if has_isoprene:
            isoprene_conc = state['isoprene_mixing_ratio']
            self._validate_mixing_ratio(isoprene_conc, 'isoprene_mixing_ratio')
            concentrations['isoprene_mixing_ratio'] = isoprene_conc
            k_isoprene_oh = rate_constants['k_isoprene_oh']
            k_isoprene_o3 = rate_constants['k_isoprene_o3']
            gamma_no2 = self.get_parameter_value(
                "isoprene_ro2_no2_stoichiometric_yield", required=True
            ).to(photolysis_no2)
            n_carbon_isoprene = self.get_parameter_value(
                "isoprene_carbon_atoms", required=True
            ).to(photolysis_no2)
            
            # Convert mixing ratios to molecules cm-3 using proper air density
            isoprene_molecules = self._mixing_ratio_to_number_density(isoprene_conc, air_number_density)
            o3_molecules = self._mixing_ratio_to_number_density(o3_conc, air_number_density)
            
            # BVOC reaction rates [molecules cm-3 s-1]
            isoprene_oh_rxn = k_isoprene_oh * isoprene_molecules * oh_conc
            isoprene_o3_rxn = k_isoprene_o3 * isoprene_molecules * o3_molecules
            
            # Convert reaction rates back to mixing ratio units [mol mol-1 s-1]
            # Rate [molecules cm-3 s-1] / air_density [molecules cm-3] = mixing ratio rate [s-1]
            isoprene_oh_rate = isoprene_oh_rxn / air_number_density
            isoprene_o3_rate = isoprene_o3_rxn / air_number_density
            isoprene_ro2_no_rate = (
                gamma_no2 * ro2_no_branching_fraction * isoprene_oh_rate
            )
            carbon_oxidized_rate = n_carbon_isoprene * (
                isoprene_oh_rate + isoprene_o3_rate
            )

        isoprene_oxidation_rate = isoprene_oh_rate + isoprene_o3_rate
        ox_net_chemical_tendency = isoprene_ro2_no_rate - isoprene_o3_rate

        dt_value = float(dt)
        if (
            not torch.isfinite(photolysis_no2.new_tensor(dt_value))
            or dt_value < 0.0
        ):
            raise ValueError("dt must be finite and non-negative")
        dt_tensor = photolysis_no2.new_tensor(dt_value)

        # References: Seinfeld and Pandis (2016); Feinberg (2019).
        # Mathematical expression: x_dot = N_h r_h for the NOx-O3-isoprene
        # reaction family, with C_NOx N_h = 0 and C_NOx = [1, 1, 0, 0].
        # Parameter meanings: r_h contains the ParameterManager-backed NO2
        # photolysis, NO-O3, isoprene-OH, isoprene-O3, and RO2-NO reaction rates.
        # Units and route: x_dot uses mol mol-1 s-1; O/O2/OH and oxidation
        # products leave the reduced box through the boundary-export route.
        chemistry_network = advance_reaction_network(
            name=f"{self.name}.gas_phase_reaction_network",
            species=(
                'no_mixing_ratio',
                'no2_mixing_ratio',
                'o3_mixing_ratio',
                'isoprene_mixing_ratio',
            ),
            reactions=(
                'no2_photolysis',
                'no_o3_titration',
                'isoprene_oh_oxidation',
                'isoprene_o3_oxidation',
                'isoprene_ro2_no_oxidation',
            ),
            stoichiometric_N=photolysis_no2.new_tensor(
                [
                    [1.0, -1.0, 0.0, 0.0, -1.0],
                    [-1.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, -1.0, 0.0],
                ]
            ),
            rates=(
                photolysis_no2,
                reaction_no_o3,
                isoprene_oh_rate,
                isoprene_o3_rate,
                isoprene_ro2_no_rate,
            ),
            dt=dt_tensor,
            conservation_matrix=photolysis_no2.new_tensor(
                [[1.0, 1.0, 0.0, 0.0]]
            ),
            strict_conservation=True,
            require_conservation_matrix=True,
            open_system_route='boundary_export',
        )
        if chemistry_network.state_delta is None:
            raise RuntimeError(
                "Chemistry reaction network did not return integrated state deltas."
            )
        delta_no, delta_no2, delta_o3, delta_isoprene = (
            chemistry_network.state_derivative.unbind(0)
        )
        step_no, step_no2, step_o3, step_isoprene = (
            chemistry_network.state_delta.unbind(0)
        )

        tendencies = {
            'no_mixing_ratio': delta_no,
            'no2_mixing_ratio': delta_no2,
            'o3_mixing_ratio': delta_o3,
        }
        if has_isoprene:
            tendencies['isoprene_mixing_ratio'] = delta_isoprene
        self._validate_explicit_euler_positivity(
            concentrations,
            tendencies,
            dt_tensor,
        )

        output: FluxDict = {
            'no_mixing_ratio': UpdateItem(
                value=step_no,
                mode=UpdateMode.ACCUMULATE,
                unit='mol mol-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'no2_mixing_ratio': UpdateItem(
                value=step_no2,
                mode=UpdateMode.ACCUMULATE,
                unit='mol mol-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'o3_mixing_ratio': UpdateItem(
                value=step_o3,
                mode=UpdateMode.ACCUMULATE,
                unit='mol mol-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.photolysis_rate_no2': UpdateItem(
                value=j_no2,
                mode=UpdateMode.REPLACE,
                unit='s-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.isoprene_oxidation_rate': UpdateItem(
                value=isoprene_oxidation_rate,
                mode=UpdateMode.REPLACE,
                unit='mol mol-1 s-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.reaction_no_o3_rate': UpdateItem(
                value=reaction_no_o3,
                mode=UpdateMode.REPLACE,
                unit='mol mol-1 s-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.photolysis_no2_rate': UpdateItem(
                value=photolysis_no2,
                mode=UpdateMode.REPLACE,
                unit='mol mol-1 s-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.rate_constant_no_o3': UpdateItem(
                value=k_no_o3,
                mode=UpdateMode.REPLACE,
                unit='cm3 molecule-1 s-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.shortwave_radiation_used': UpdateItem(
                value=shortwave_down,
                mode=UpdateMode.REPLACE,
                unit='W m-2',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.oh_concentration': UpdateItem(
                value=oh_conc,
                mode=UpdateMode.REPLACE,
                unit='molecule cm-3',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.air_number_density': UpdateItem(
                value=air_number_density,
                mode=UpdateMode.REPLACE,
                unit='molecule cm-3',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.ro2_no_conversion_rate': UpdateItem(
                value=isoprene_ro2_no_rate,
                mode=UpdateMode.REPLACE,
                unit='mol mol-1 s-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.ro2_no_branching_fraction': UpdateItem(
                value=ro2_no_branching_fraction,
                mode=UpdateMode.REPLACE,
                unit='dimensionless',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.ox_net_chemical_tendency': UpdateItem(
                value=ox_net_chemical_tendency,
                mode=UpdateMode.REPLACE,
                unit='mol mol-1 s-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.chemistry.carbon_oxidized_rate': UpdateItem(
                value=carbon_oxidized_rate,
                mode=UpdateMode.REPLACE,
                unit='mol mol-1 s-1',
                per_timestep=True,
                source_functor=self.name,
            ),
        }
        if has_isoprene:
            output['isoprene_mixing_ratio'] = UpdateItem(
                value=step_isoprene,
                mode=UpdateMode.ACCUMULATE,
                unit='mol mol-1',
                per_timestep=True,
                source_functor=self.name,
            )
        return output

    def compute_chemical_ledger(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
        dt: float,
    ) -> Dict[str, Any]:
        """Compute explicit NOx, odd-oxygen (Ox = O3 + NO2), and reactive-carbon ledgers.

        References:
            - Seinfeld, J.H. & Pandis, S.N. (2016). Atmospheric Chemistry and Physics,
              3rd ed., Ch. 6 (NOx family conservation and Ox = O3 + NO2 budget).
            - Atkinson, R. & Arey, J. (2003). Chem. Rev., 103, 4605-4638 (C5H8 stoichiometry).
        Mathematical expression:
            Delta_NOx = Delta[NO] + Delta[NO2] = 0
            Delta_Ox = Delta[O3] + Delta[NO2] = dt * (R_RO2_NO - R_C5H8_O3)
            Delta_C_reactive = 5 * Delta[C5H8] + dt * R_C,ox = 0
        Parameter meanings:
            - Delta[X]: per-step mixing ratio increment [mol mol-1]
            - R_RO2_NO: peroxy radical NO-to-NO2 conversion rate [mol mol-1 s-1]
            - R_C5H8_O3: isoprene ozonolysis rate [mol mol-1 s-1]
            - R_C,ox: reactive carbon oxidation export rate [mol C mol-1 s-1]
            - dt: integration timestep [s]
        """
        del input_state
        dt_val = float(dt)
        if dt_val < 0.0:
            raise ValueError("dt must be non-negative for chemical ledger audit.")

        def _extract(key: str) -> torch.Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing chemical ledger key '{key}'")
            item = output_updates[key]
            return item.value if isinstance(item, UpdateItem) else item

        delta_no = _extract('no_mixing_ratio')
        delta_no2 = _extract('no2_mixing_ratio')
        delta_o3 = _extract('o3_mixing_ratio')
        r_ro2_no = _extract('diagnostics.chemistry.ro2_no_conversion_rate')
        r_ox_net = _extract('diagnostics.chemistry.ox_net_chemical_tendency')
        r_c_ox = _extract('diagnostics.chemistry.carbon_oxidized_rate')

        nox_net = delta_no + delta_no2
        ox_step = delta_o3 + delta_no2
        ox_expected = r_ox_net * dt_val
        ox_residual = ox_step - ox_expected

        n_carbon = self.get_parameter_value(
            "isoprene_carbon_atoms", required=True
        ).to(delta_o3)
        if 'isoprene_mixing_ratio' in output_updates:
            delta_isop = _extract('isoprene_mixing_ratio')
        else:
            delta_isop = torch.zeros_like(delta_o3)
        carbon_step = n_carbon * delta_isop
        carbon_exported = r_c_ox * dt_val
        carbon_residual = carbon_step + carbon_exported

        return {
            'nox_molar_balance': {
                'delta_no': delta_no,
                'delta_no2': delta_no2,
                'net_residual': nox_net,
                'max_abs_residual': float(torch.max(torch.abs(nox_net)).detach().item()),
            },
            'odd_oxygen_balance': {
                'delta_o3': delta_o3,
                'delta_no2': delta_no2,
                'delta_ox': ox_step,
                'ro2_no_production': r_ro2_no * dt_val,
                'net_chemical_production': ox_expected,
                'net_residual': ox_residual,
                'max_abs_residual': float(torch.max(torch.abs(ox_residual)).detach().item()),
            },
            'reactive_carbon_balance': {
                'delta_isoprene_carbon': carbon_step,
                'oxidized_carbon_export': carbon_exported,
                'net_residual': carbon_residual,
                'max_abs_residual': float(torch.max(torch.abs(carbon_residual)).detach().item()),
            },
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Audit the closed NOx family within the reduced chemistry box.

        References:
            - Seinfeld and Pandis (2016), atmospheric NOx-O3 chemistry.
            - Feinberg (2019), chemical reaction network stoichiometry.
        Mathematical expression:
            R_NOx = |Delta NO + Delta NO2|
                    / max(|Delta NO|, |Delta NO2|, 1).
        Parameter meanings:
            Delta NO and Delta NO2 are integrated mixing-ratio increments
            [mol mol-1]. The unit floor is used only when both increments are
            zero, so the returned audit is dimensionless.
        Open/closed route:
            NO plus NO2 is closed for the declared local reaction family.
            O3, OH, O/O2, isoprene, and products use the declared open boundary.
        """

        if self.physical_boundary == "closed_constant_volume":
            return self._check_closed_box_ledgers(input_state, output_updates)
        del input_state

        def require_increment(key: str) -> torch.Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing conservation output '{key}'")
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Chemistry conservation output '{key}' must be a torch.Tensor."
                )
            if value.numel() == 0:
                raise ValueError(
                    f"Chemistry conservation output '{key}' must be non-empty."
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    f"Chemistry conservation output '{key}' must be finite."
                )
            return value

        delta_no = require_increment('no_mixing_ratio')
        delta_no2 = require_increment('no2_mixing_ratio')
        if delta_no.shape != delta_no2.shape:
            raise ValueError(
                "NO and NO2 conservation increments must have identical shapes."
            )
        scale = torch.maximum(torch.abs(delta_no), torch.abs(delta_no2))
        scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
        residual = torch.abs(delta_no + delta_no2) / scale
        return {
            'nox_family': float(torch.max(residual).detach().item())
        }


# Export the main functor class
__all__ = ['ChemistryFunctor']
