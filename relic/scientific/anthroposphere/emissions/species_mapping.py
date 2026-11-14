"""
Species Mapping and Unit Conversion for Emission-Atmosphere Coupling

This module provides bidirectional mapping between emission inventory species
and atmospheric chemistry mechanism species, along with unit conversions.

References:
    - Yarwood, G. et al. (2005). Updates to the CB05 chemical mechanism
    - Kuenen, J. et al. (2014). TNO-MACC_II emission inventory
    - IPCC (2019). Refinement to the 2006 IPCC Guidelines

Mathematical Expressions:
    Mass flux to molar flux conversion:
    F_mol = F_mass / MW

    Mixing ratio from molar flux:
    chi = (F_mol * dt * R * T) / (P * V)

    where:
    - F_mol: molar flux [mol m-2 s-1]
    - F_mass: mass flux [kg m^-2 s^-1]
    - MW: molecular weight [kg mol^-1]
    - chi: mixing ratio [mol mol^-1]
    - R: gas constant [J mol-1 K^-1]
    - T: temperature [K]
    - P: pressure [Pa]
    - V: volume [m^3]
"""
import logging
import torch
from typing import Dict, Optional, List, Union, Any, Mapping
from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.utils.physical_constants import get_physical_constant
from relic.core.functor import ScaleCouplingFunctor
from relic.core.functor import (
    DependencyDeclaration,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.state import UpdateItem, UpdateMode
from .atmosphere_interface_contract import ChemicalMechanism
logger = logging.getLogger(__name__)

class SpeciesMapper(ScaleCouplingFunctor):
    """
    Maps emission inventory species to chemical mechanism species
    and provides unit conversion utilities.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    CANONICAL_SECTORS = (
        "power",
        "transport",
        "industry",
        "buildings",
        "agriculture",
    )
    CANONICAL_INVENTORY_SPECIES = (
        "NOx",
        "CO",
        "SO2",
        "VOC",
        "NH3",
        "PM25",
    )
    MECHANISM_GAS_SPECIES = (
        "NO",
        "NO2",
        "CO",
        "SO2",
        "NH3",
        "PAR",
        "OLE",
        "TOL",
        "XYL",
        "FORM",
        "ALD2",
    )
    MECHANISM_AEROSOL_SPECIES = ("BC", "OC", "NO3", "SO4", "NH4", "OTHER")
    MOLECULAR_WEIGHT_SPECIES = (
        "NO",
        "NO2",
        "CO",
        "SO2",
        "NH3",
        "PAR",
        "OLE",
        "TOL",
        "XYL",
        "FORM",
        "ALD2",
        "NOx",
    )
    # CO and FORM molar masses are owned by the physical-constant SSOT
    # (relic/parameters/database/relic/physical/constants.yaml); the remaining
    # declared species retain their existing domain parameter entries.
    CO_MOLAR_MASS_KEY = "relic.physical.constants.molecular_weights.co"
    VOC_PROFILE_SPECIES = ("PAR", "OLE", "TOL", "XYL", "FORM", "ALD2")
    VOC_PARAMETER_SECTORS = {
        "power": "power_plant",
        "transport": "transport",
        "industry": "industrial",
        "buildings": "residential",
        "agriculture": "agriculture",
    }
    PM_PARAMETER_SECTORS = {
        "power": "power",
        "transport": "transport",
        "industry": "industrial",
        "buildings": "residential",
        "agriculture": "agriculture",
    }
    PM_COMPONENT_PARAMETERS = (
        "bc_fraction",
        "oc_fraction",
        "nitrate_fraction",
        "sulfate_fraction",
        "ammonium_fraction",
    )

    def __init__(
        self,
        name: str = 'species_mapping',
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize SpeciesMapper functor.

        Args:
            name: Functor name
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
        """
        if not isinstance(name, str):
            raise TypeError("SpeciesMapper requires a string name.")
        # Scientific values come only from the ParameterManager YAML; the one
        # accepted config key is the FunctorBase dependency declaration.
        runtime_config = dict(config or {})
        unsupported = sorted(set(runtime_config) - {"dependencies"})
        if unsupported:
            raise ValueError(
                "SpeciesMapper runtime config overrides are not supported "
                f"({', '.join(unsupported)}); register values in the "
                "ParameterManager YAML database."
            )
        metadata = FunctorMetadata(
            name=name,
            version='1.1.0',
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description='Maps emission inventory species to chemistry mechanism species across scales',
            mathematical_expression=(
                'IPHS chain map: y_mech = S_mech(theta) y_inventory, '
                'sum_i S_mech,ij = 1 for mass-conserving VOC and PM splits; '
                'NOx partition conserves nitrogen amount on an NO2-equivalent basis; '
                'F_mol = star_MW_inv F_mass; chi_dot = F_mol / (c_air H)'
            ),
            references=[
                'Yarwood, G. et al. (2005). Updates to the CB05 chemical mechanism',
                'Kuenen, J. et al. (2014). TNO-MACC_II emission inventory',
                'IPCC (2019). Refinement to the 2006 IPCC Guidelines',
            ],
            parameters={
                'anthroposphere.emissions.species_mapping.default_mechanism': (
                    'Identifier of the default chemical mechanism [dimensionless]'
                ),
                'anthroposphere.emissions.species_mapping.mass_balance_tolerance': (
                    'Relative tolerance of the speciation mass balance [dimensionless]'
                ),
                'anthroposphere.emissions.partition_factors.no_fraction': (
                    'Fraction of NOx (NO2-equivalent mass) emitted as NO [dimensionless]'
                ),
                'anthroposphere.emissions.molecular_weights.NO': (
                    'Molecular weight of emitted NO [g mol-1]'
                ),
                'anthroposphere.emissions.molecular_weights.NO2': (
                    'Molecular weight of emitted NO2 [g mol-1]'
                ),
                'anthroposphere.emissions.molecular_weights.NOx': (
                    'Molecular weight of emitted NOx [g mol-1]'
                ),
                'relic.physical.constants.molecular_weights.co': (
                    'Molar mass of emitted CO from the physical-constant SSOT [g mol-1]'
                ),
                'anthroposphere.emissions.molecular_weights.SO2': (
                    'Molecular weight of emitted SO2 [g mol-1]'
                ),
                'anthroposphere.emissions.molecular_weights.NH3': (
                    'Molecular weight of emitted NH3 [g mol-1]'
                ),
                'anthroposphere.emissions.molecular_weights.PAR': (
                    'Molecular weight of emitted PAR [g mol-1]'
                ),
                'anthroposphere.emissions.molecular_weights.OLE': (
                    'Molecular weight of emitted OLE [g mol-1]'
                ),
                'anthroposphere.emissions.molecular_weights.TOL': (
                    'Molecular weight of emitted TOL [g mol-1]'
                ),
                'anthroposphere.emissions.molecular_weights.XYL': (
                    'Molecular weight of emitted XYL [g mol-1]'
                ),
                'relic.physical.constants.molecular_weights.hcho': (
                    'Molar mass of emitted FORM (formaldehyde) from the physical-constant SSOT [g mol-1]'
                ),
                'anthroposphere.emissions.molecular_weights.ALD2': (
                    'Molecular weight of emitted ALD2 [g mol-1]'
                ),
                **{
                    'anthroposphere.emissions.speciation.voc_split_fractions.'
                    f'{parameter_sector}.{species}': (
                        f'VOC split fraction of {species} in {parameter_sector} emissions '
                        '[dimensionless]'
                    )
                    for parameter_sector in self.VOC_PARAMETER_SECTORS.values()
                    for species in self.VOC_PROFILE_SPECIES
                },
                **{
                    f'anthroposphere.emissions.pm_composition.{parameter_sector}.{component}': (
                        f'PM2.5 {component} of {parameter_sector} emissions [dimensionless]'
                    )
                    for parameter_sector in self.PM_PARAMETER_SECTORS.values()
                    for component in self.PM_COMPONENT_PARAMETERS
                },
            },
            input_variables=['sector_emission_flux'],
            output_variables=[
                'mechanism_gas_molar_flux',
                'mechanism_aerosol_mass_flux',
                'diagnostics.species_mapping.total_input_mass_flux',
                'diagnostics.species_mapping.total_output_mass_flux',
                'diagnostics.species_mapping.inventory_equivalent_output_mass_flux',
                'diagnostics.species_mapping.inventory_basis_adjustment_flux',
                'diagnostics.species_mapping.relative_error',
                'diagnostics.species_mapping.canonical_iphs_runtime',
            ],
            conservation_laws=['mass'],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=['emissions', 'chemical_mechanism'],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        super().__init__(
            name=name,
            config=runtime_config,
            metadata=metadata,
            source_scale='inventory',
            target_scale='mechanism',
            coupling_type='hierarchical',
            **kwargs,
        )
        self._emissions_namespace = 'anthroposphere.emissions'
        self.molecular_weights: Dict[str, float] = {}
        self.voc_speciation_profiles: Dict[str, Dict[str, float]] = {}
        self._initialize_parameters()
        mechanism_value = self.get_parameter_value(
            'default_mechanism',
            namespace='anthroposphere.emissions.species_mapping',
        )
        if mechanism_value is None:
            raise KeyError(
                "Missing required parameter "
                "'anthroposphere.emissions.species_mapping.default_mechanism'"
            )
        if isinstance(mechanism_value, torch.Tensor):
            mechanism_value = mechanism_value.detach().reshape(-1)[0].tolist()
        if not isinstance(mechanism_value, (str, ChemicalMechanism)):
            raise TypeError(
                "SpeciesMapper default_mechanism must resolve to a string or ChemicalMechanism."
            )
        self.default_mechanism = self._normalize_mechanism(mechanism_value)
        if self.default_mechanism is not ChemicalMechanism.CB05:
            raise NotImplementedError(
                "SpeciesMapper canonical runtime currently supports only CB05; "
                "other mechanisms require complete typed species and molecular-weight contracts."
            )
        self.mechanism_gas_species_order = self.MECHANISM_GAS_SPECIES
        self.mechanism_aerosol_species_order = self.MECHANISM_AEROSOL_SPECIES
        self.runtime_path = "canonical_iphs_species_mapping_chain_map"

    def get_dependencies(self) -> DependencyDeclaration:
        """Declare sector_allocation as the optional same-step producer of the input.

        SectorAllocation returns sector_emission_flux with write mode DERIVED: it
        is never committed and is visible only to the Functors that run after it
        in the same group call. The declaration orders it before this Functor
        whenever it is registered, whatever the registration order. It is
        optional because the input can also be supplied as input state when no
        allocator runs; a missing sector_emission_flux still raises.
        """
        return self._merge_declared_dependencies({"optional": ["sector_allocation"]})

    def _structural_n_dofs(self, n_dofs: Optional[int] = None) -> int:
        """Resolve structural audit size without changing runtime tensor shape."""

        value = len(self.MECHANISM_GAS_SPECIES) if n_dofs is None else int(n_dofs)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare cochain spaces for species mapping.

        References:
          - Yarwood et al. (2005), CB05 chemical mechanism.
          - CMAQ User's Guide v5.3, emissions speciation profiles.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          y_mech = S_mech(theta) y_inventory and F_mol = star_MW_inv F_mass.
        Parameter meanings:
          y_inventory and y_mech are mass-flux cochains [kg m-2 s-1],
          S_mech is a mass-conserving speciation matrix [1], and
          star_MW_inv is the molecular-weight inverse Hodge [mol kg-1].
        """

        return (
            CochainSpaceSpec(
                complex_id="anthroposphere_species_mapping",
                degree=2,
                primality="dual",
                n_dofs=len(self.CANONICAL_SECTORS)
                * len(self.CANONICAL_INVENTORY_SPECIES),
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="anthroposphere",
                semantic_name="sector_inventory_species_mass_flux",
                orientation_convention="emission_positive",
                boundary_role="boundary",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_species_mapping",
                degree=2,
                primality="dual",
                n_dofs=len(self.MECHANISM_GAS_SPECIES),
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="anthroposphere",
                semantic_name="mechanism_gas_mass_flux",
                orientation_convention="emission_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_species_mapping",
                degree=2,
                primality="dual",
                n_dofs=len(self.MECHANISM_GAS_SPECIES),
                units="mol m-2 s-1",
                extensivity="integral",
                physical_domain="atmosphere",
                semantic_name="mechanism_gas_molar_flux",
                orientation_convention="emission_positive",
                boundary_role="port",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_species_mapping",
                degree=2,
                primality="dual",
                n_dofs=len(self.MECHANISM_AEROSOL_SPECIES),
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="atmosphere",
                semantic_name="mechanism_aerosol_mass_flux",
                orientation_convention="emission_positive",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        molecular_keys = tuple(
            self._molecular_weight_key(species)
            for species in self.MOLECULAR_WEIGHT_SPECIES
        )
        voc_keys = tuple(
            "anthroposphere.emissions.speciation.voc_split_fractions."
            f"{parameter_sector}.{species}"
            for parameter_sector in self.VOC_PARAMETER_SECTORS.values()
            for species in self.VOC_PROFILE_SPECIES
        )
        pm_keys = tuple(
            "anthroposphere.emissions.pm_composition."
            f"{parameter_sector}.{component}"
            for parameter_sector in self.PM_PARAMETER_SECTORS.values()
            for component in self.PM_COMPONENT_PARAMETERS
        )
        return (
            "anthroposphere.emissions.species_mapping.default_mechanism",
            "anthroposphere.emissions.species_mapping.mass_balance_tolerance",
            "anthroposphere.emissions.partition_factors.no_fraction",
            *molecular_keys,
            *voc_keys,
            *pm_keys,
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the positive molecular-weight inverse Hodge contract.

        References:
          - Seinfeld and Pandis (2016), atmospheric chemistry unit conversion.
          - CMAQ User's Guide v5.3, emissions preprocessing.
          - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        Mathematical expression:
          star_MW_inv = diag(1 / MW_i), F_mol = star_MW_inv F_mass.
        Parameter meanings:
          MW_i is species molecular weight [kg mol-1], F_mass is mass flux
          [kg m-2 s-1], and F_mol is molar flux [mol m-2 s-1]. Each diagonal
          entry is bound to its molar-mass key: CO to
          relic.physical.constants.molecular_weights.co and FORM to
          relic.physical.constants.molecular_weights.hcho; remaining species
          use anthroposphere.emissions.molecular_weights.<species>.
        Units:
          MW_i in kg mol-1; star_MW_inv entries in mol kg-1; F_mass in
          kg m-2 s-1; F_mol in mol m-2 s-1.
        Domain/constraints:
          MW_i > 0 for every declared gas species (ValueError otherwise), so
          star_MW_inv is diagonal and positive definite.
        Sign convention:
          Emission-positive fluxes; the positive diagonal map preserves sign.
        """

        n_dofs = self._structural_n_dofs(n_dofs)
        values = []
        parameter_keys = []
        for idx in range(n_dofs):
            species = self.MECHANISM_GAS_SPECIES[
                idx % len(self.MECHANISM_GAS_SPECIES)
            ]
            mw = float(self.get_molecular_weight(species))
            if mw <= 0.0:
                raise ValueError(f"Molecular weight for {species} must be positive.")
            values.append(1.0 / mw)
            parameter_keys.append(self._molecular_weight_key(species))
        matrix = torch.diag(
            torch.tensor(
                values,
                dtype=torch.float32,
                device=getattr(self, "device", torch.device("cpu")),
            )
        )
        domain = CochainSpaceSpec(
            complex_id="anthroposphere_species_mapping",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="anthroposphere",
            semantic_name="mechanism_gas_mass_flux",
            orientation_convention="emission_positive",
            boundary_role="interior",
        )
        codomain = CochainSpaceSpec(
            complex_id="anthroposphere_species_mapping",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="mol m-2 s-1",
            extensivity="integral",
            physical_domain="atmosphere",
            semantic_name="mechanism_gas_molar_flux",
            orientation_convention="emission_positive",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_molecular_weight_inverse_hodge",
            domain=domain,
            codomain=codomain,
            matrix=matrix,
            parameter_keys=tuple(parameter_keys),
            metric_keys=tuple(parameter_keys),
            material_law="positive_molecular_weight_inverse",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build Hodge terms for IPHS assembly."""

        _ = complex_bundle, resolved_parameters
        return (self.build_hodge_contract(strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no stored energy term for algebraic species mapping."""

        _ = state
        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no standalone entropy term for algebraic species mapping."""

        _ = state
        return {}

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the mass-conserving chain-map structure."""

        _ = state, complex_bundle
        return {
            "state_vars": tuple(self.CANONICAL_INVENTORY_SPECIES),
            "chain_map": "S_cb05(theta)",
            "inventory_equivalent_conserving": True,
            "ports": self.ports(),
        }

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return no dissipative term for an algebraic unit conversion."""

        _ = state, complex_bundle
        return {}

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare separate gas-molar and aerosol-mass matter ports."""

        gas_effort_space = CochainSpaceSpec(
            complex_id="anthroposphere_species_mapping",
            degree=0,
            primality="primal",
            n_dofs=len(self.MECHANISM_GAS_SPECIES),
            units="J mol-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="chemical_species_potential",
            orientation_convention="emission_positive",
            boundary_role="port",
        )
        gas_flow_space = CochainSpaceSpec(
            complex_id="anthroposphere_species_mapping",
            degree=2,
            primality="dual",
            n_dofs=len(self.MECHANISM_GAS_SPECIES),
            units="mol m-2 s-1",
            extensivity="integral",
            physical_domain="atmosphere",
            semantic_name="mechanism_gas_molar_flux",
            orientation_convention="emission_positive",
            boundary_role="port",
        )
        aerosol_effort_space = CochainSpaceSpec(
            complex_id="anthroposphere_species_mapping",
            degree=0,
            primality="primal",
            n_dofs=len(self.MECHANISM_AEROSOL_SPECIES),
            units="J kg-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="aerosol_mass_specific_potential",
            orientation_convention="emission_positive",
            boundary_role="port",
        )
        aerosol_flow_space = CochainSpaceSpec(
            complex_id="anthroposphere_species_mapping",
            degree=2,
            primality="dual",
            n_dofs=len(self.MECHANISM_AEROSOL_SPECIES),
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="atmosphere",
            semantic_name="mechanism_aerosol_mass_flux",
            orientation_convention="emission_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="species_mapping_gas_molar_flux",
                effort_var="chemical_species_potential",
                flow_var="mechanism_gas_molar_flux",
                effort_space=gas_effort_space,
                flow_space=gas_flow_space,
                power_units="W m-2",
                positive_power_convention="into_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="species_mapping_aerosol_mass_flux",
                effort_var="aerosol_mass_specific_potential",
                flow_var="mechanism_aerosol_mass_flux",
                effort_space=aerosol_effort_space,
                flow_space=aerosol_flow_space,
                power_units="W m-2",
                positive_power_convention="into_system",
                port_kind="matter",
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

    def validate_state(
        self,
        state: Dict[str, torch.Tensor],
    ) -> List[str]:
        """
        Validate the fixed sector/species mass-flux tensor.
        """
        if not isinstance(state, dict):
            return ["SpeciesMapper state must be a dictionary."]
        if 'sector_emission_flux' not in state:
            return ["Missing required canonical port: sector_emission_flux"]
        flux = state['sector_emission_flux']
        if not isinstance(flux, torch.Tensor):
            return ["sector_emission_flux must be a torch.Tensor"]
        expected_prefix = (
            len(self.CANONICAL_SECTORS),
            len(self.CANONICAL_INVENTORY_SPECIES),
        )
        if flux.ndim < 2 or tuple(flux.shape[:2]) != expected_prefix:
            return [
                "sector_emission_flux must use [sector, species, ...] with prefix "
                f"{expected_prefix}; got {tuple(flux.shape)}"
            ]
        if not flux.is_floating_point():
            return ["sector_emission_flux must use a floating dtype"]
        if not torch.isfinite(flux).all():
            return ["sector_emission_flux must contain only finite values"]
        if torch.any(flux < 0.0):
            return ["sector_emission_flux cannot contain negative values"]
        return []

    def _require_dimensionless_parameter(self, full_key: str) -> float:
        """Fetch one required dimensionless parameter through FunctorBase."""
        if '.' not in full_key or full_key.endswith('.'):
            raise ValueError(f'full_key must include namespace and parameter name: {full_key}')
        namespace, param_name = full_key.rsplit('.', 1)
        value = self.get_parameter_value(
            param_name,
            namespace=namespace,
            source_order=('manager',),
            required=True,
        )
        if value is None:
            raise KeyError(f"Missing required parameter '{full_key}'")
        return float(value)


    def forward(
        self,
        state: Dict,
        time: float,
        dt: float,
    ) -> Dict[str, Any]:
        """Map sector-resolved inventory fluxes to fixed CB05 ports.

        References:
            Yarwood et al. (2005); CMAQ/SMOKE emissions preprocessing.
        Mathematical expression:
            F_gas,mol = star_MW_inv S_gas F_sector and
            F_aer,mass = S_aer F_sector.
        Parameter meanings:
            S_gas and S_aer use ParameterManager-backed sector profiles [1].
            MW is molecular weight [kg mol-1]. Input and aerosol outputs use
            kg m-2 s-1; gas output uses mol m-2 s-1.
        Open/closed route:
            VOC and PM splits are closed. NOx is closed on its declared
            NO2-equivalent inventory basis; the true oxygen-mass adjustment is
            returned explicitly as a diagnostic flux.
        """
        # References: Yarwood et al. (2005); CMAQ/SMOKE preprocessing.
        # Mathematical expression: F_gas = star_MW_inv S_gas F_sector.
        # Parameter meanings: S is unitless and MW uses kg mol-1.
        # Every input comes from the state (sector_emission_flux); forward takes no
        # other input mapping, and an unexpected keyword raises TypeError.
        _ = time, dt
        validation_errors = self.validate_state(state)
        if validation_errors:
            raise ValueError(f'SpeciesMapper state validation failed: {validation_errors}')
        sector_flux = state['sector_emission_flux'].to(self.device)
        template = sector_flux[0, 0]
        gas_mass = {
            species: torch.zeros_like(template)
            for species in self.MECHANISM_GAS_SPECIES
        }
        aerosol_mass = {
            species: torch.zeros_like(template)
            for species in self.MECHANISM_AEROSOL_SPECIES
        }

        for sector_index, sector in enumerate(self.CANONICAL_SECTORS):
            nox_split = self._speciate_nox(sector_flux[sector_index, 0])
            voc_split = self._speciate_voc(sector_flux[sector_index, 3], sector)
            pm_split = self._speciate_pm25(sector_flux[sector_index, 5], sector)
            direct_gases = {
                'CO': sector_flux[sector_index, 1],
                'SO2': sector_flux[sector_index, 2],
                'NH3': sector_flux[sector_index, 4],
            }
            for species, value in {**nox_split, **voc_split, **direct_gases}.items():
                if species not in gas_mass:
                    raise KeyError(f"CB05 gas output species '{species}' is undeclared.")
                gas_mass[species] = gas_mass[species] + value
            for species, value in pm_split.items():
                if species not in aerosol_mass:
                    raise KeyError(f"Aerosol output species '{species}' is undeclared.")
                aerosol_mass[species] = aerosol_mass[species] + value

        gas_molar_tensor = torch.stack(
            [
                self.mass_to_molar_flux(gas_mass[species], species)
                for species in self.MECHANISM_GAS_SPECIES
            ],
            dim=0,
        )
        aerosol_mass_tensor = torch.stack(
            [aerosol_mass[species] for species in self.MECHANISM_AEROSOL_SPECIES],
            dim=0,
        )
        gas_mass_tensor = torch.stack(
            [gas_mass[species] for species in self.MECHANISM_GAS_SPECIES],
            dim=0,
        )
        total_in = sector_flux.sum(dim=(0, 1))
        total_out = gas_mass_tensor.sum(dim=0) + aerosol_mass_tensor.sum(dim=0)

        no_index = self.MECHANISM_GAS_SPECIES.index('NO')
        no2_index = self.MECHANISM_GAS_SPECIES.index('NO2')
        nox_mw = torch.as_tensor(
            self.get_molecular_weight('NOx'),
            device=total_in.device,
            dtype=total_in.dtype,
        )
        nox_true_mass = gas_mass_tensor[no_index] + gas_mass_tensor[no2_index]
        nox_equivalent_mass = (
            gas_molar_tensor[no_index] + gas_molar_tensor[no2_index]
        ) * nox_mw
        equivalent_out = total_out - nox_true_mass + nox_equivalent_mass
        residual = equivalent_out - total_in
        rel_error = torch.where(
            torch.abs(total_in) > 0.0,
            torch.abs(residual) / torch.abs(total_in),
            torch.abs(residual),
        )
        tolerance = self._require_dimensionless_parameter(
            'anthroposphere.emissions.species_mapping.mass_balance_tolerance'
        )
        max_error = float(rel_error.max().detach().tolist())
        if max_error > tolerance:
            raise RuntimeError(
                f"Species mapping residual {max_error:.6e} exceeds {tolerance:.6e}."
            )

        return {
            'mechanism_gas_molar_flux': self.wrap_update(
                gas_molar_tensor,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
            ),
            'mechanism_aerosol_mass_flux': self.wrap_update(
                aerosol_mass_tensor,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
            ),
            'diagnostics.species_mapping.total_input_mass_flux': total_in,
            'diagnostics.species_mapping.total_output_mass_flux': total_out,
            'diagnostics.species_mapping.inventory_equivalent_output_mass_flux': (
                equivalent_out
            ),
            'diagnostics.species_mapping.inventory_basis_adjustment_flux': (
                total_out - equivalent_out
            ),
            'diagnostics.species_mapping.relative_error': rel_error,
            'diagnostics.species_mapping.canonical_iphs_runtime': total_in.new_ones(()),
        }

    def check_conservation_laws(
        self,
        input_state: Dict[str, torch.Tensor],
        output_updates: Dict[str, Any],
    ) -> Dict[str, float]:
        """Check the mapping ledger on the declared inventory mass basis.

        References:
            Yarwood et al. (2005); CMAQ/SMOKE emissions preprocessing.
        Mathematical expression:
            F_eq = sum_i(F_gas,mol,i MW_i) + sum_j(F_aer,mass,j)
                   - F_NO,true - F_NO2,true
                   + (F_NO,mol + F_NO2,mol) MW_NOx,
            epsilon = abs(F_eq - F_in) / abs(F_in).
        Parameter meanings:
            MW_i is the ParameterManager-backed molecular weight [kg mol-1].
            F_in is the sector/species inventory flux [kg m-2 s-1].
        Open/closed route:
            VOC and PM splits are closed. NOx is checked on the declared
            NO2-equivalent inventory basis; oxygen-mass adjustment remains an
            explicit diagnostic and is not hidden in this ledger.
        """
        sector_flux = input_state.get('sector_emission_flux')
        gas_item = output_updates.get('mechanism_gas_molar_flux')
        aerosol_item = output_updates.get('mechanism_aerosol_mass_flux')
        if not isinstance(sector_flux, torch.Tensor):
            raise KeyError('sector_emission_flux tensor is required for conservation.')
        if not isinstance(gas_item, UpdateItem) or not isinstance(
            gas_item.value, torch.Tensor
        ):
            raise KeyError(
                'mechanism_gas_molar_flux UpdateItem is required for conservation.'
            )
        if not isinstance(aerosol_item, UpdateItem) or not isinstance(
            aerosol_item.value, torch.Tensor
        ):
            raise KeyError(
                'mechanism_aerosol_mass_flux UpdateItem is required for conservation.'
            )

        gas_molar = gas_item.value
        aerosol_mass = aerosol_item.value
        if gas_molar.shape[0] != len(self.MECHANISM_GAS_SPECIES):
            raise ValueError('Gas mechanism species axis does not match the declaration.')
        if aerosol_mass.shape[0] != len(self.MECHANISM_AEROSOL_SPECIES):
            raise ValueError('Aerosol species axis does not match the declaration.')

        gas_mass = torch.stack(
            [
                gas_molar[index]
                * gas_molar.new_tensor(self.get_molecular_weight(species))
                for index, species in enumerate(self.MECHANISM_GAS_SPECIES)
            ],
            dim=0,
        )
        no_index = self.MECHANISM_GAS_SPECIES.index('NO')
        no2_index = self.MECHANISM_GAS_SPECIES.index('NO2')
        nox_equivalent_mass = (
            gas_molar[no_index] + gas_molar[no2_index]
        ) * gas_molar.new_tensor(self.get_molecular_weight('NOx'))
        equivalent_output = (
            gas_mass.sum(dim=0)
            + aerosol_mass.sum(dim=0)
            - gas_mass[no_index]
            - gas_mass[no2_index]
            + nox_equivalent_mass
        )
        input_total = sector_flux.sum(dim=(0, 1))
        residual = equivalent_output - input_total
        relative = torch.where(
            torch.abs(input_total) > 0.0,
            torch.abs(residual) / torch.abs(input_total),
            torch.abs(residual),
        )
        return {'mass': float(relative.max().detach().tolist())}

    def _initialize_parameters(self) -> None:
        """Load the exact runtime parameter surface through FunctorBase."""
        mw_namespace = 'anthroposphere.emissions.molecular_weights'
        # References:
        # - CIAAW (2024) abridged standard atomic weights, C = 12.011 and
        #   O = 15.999, https://ciaaw.org/abridged-atomic-weights.htm (M_CO).
        # - NIST Chemistry WebBook, SRD 69, Formaldehyde, CAS 50-00-0,
        #   https://webbook.nist.gov/cgi/cbook.cgi?ID=50000 (M_FORM).
        # - Yarwood et al. (2005) CB05 mechanism documentation (other species).
        # Mathematical expression:
        # MW_i = M_i * 1.0e-3
        # Parameter meanings:
        # M_i: stored bulk molar mass of species i; CO and FORM use their
        #   canonical physical-constant keys through get_physical_constant.
        #   Remaining species use the existing emissions parameter namespace.
        # MW_i: cached molar mass used by the mass/molar flux conversions.
        # Units:
        # M_i in g mol-1; 1.0e-3 in kg g-1; MW_i in kg mol-1.
        # Domain/constraints:
        # M_i > 0 is required (ValueError otherwise); canonical CO and HCHO
        #   entries have fixed bounds and are not isotopic uncertainty models.
        # Sign convention:
        # A positive scalar constant; it sets no flux direction.
        for species in self.MOLECULAR_WEIGHT_SPECIES:
            if species in ('CO', 'FORM'):
                raw = get_physical_constant(self._molecular_weight_key(species))
            else:
                raw = self.get_parameter_value(
                    species,
                    namespace=mw_namespace,
                    source_order=('manager',),
                    required=True,
                )
            value_g_per_mol = float(raw)
            if value_g_per_mol <= 0.0:
                raise ValueError(f"Molecular weight for {species} must be positive.")
            self.molecular_weights[species] = value_g_per_mol * 1.0e-3

        voc_root = 'anthroposphere.emissions.speciation.voc_split_fractions'
        tolerance = self._require_dimensionless_parameter(
            'anthroposphere.emissions.species_mapping.mass_balance_tolerance'
        )
        for sector, parameter_sector in self.VOC_PARAMETER_SECTORS.items():
            profile: Dict[str, float] = {}
            namespace = f'{voc_root}.{parameter_sector}'
            for species in self.VOC_PROFILE_SPECIES:
                raw = self.get_parameter_value(
                    species,
                    namespace=namespace,
                    source_order=('manager',),
                    required=True,
                )
                value = float(raw)
                if value < 0.0:
                    raise ValueError(
                        f"VOC profile {parameter_sector}.{species} must be non-negative."
                    )
                profile[species] = value
            total = sum(profile.values())
            if abs(total - 1.0) > tolerance:
                raise ValueError(
                    f"VOC profile '{parameter_sector}' must sum to 1; got {total:.12g}."
                )
            self.voc_speciation_profiles[sector] = profile

    def _resolve_sector_key(self, sector: str) -> str:
        """Require one exact SectorAllocation sector identifier."""
        if not isinstance(sector, str) or sector not in self.CANONICAL_SECTORS:
            raise KeyError(f"Unknown canonical emissions sector '{sector}'.")
        return sector
    def _ensure_tensor(self, value: Union[float, torch.Tensor]) -> torch.Tensor:
        """Return a tensor on the functor device."""
        device = getattr(
            self,
            'device',
            torch.device("cpu"),
        )
        if isinstance(value, torch.Tensor):
            return value.to(device)
        return torch.as_tensor(float(value), device=device, dtype=torch.float32)

    def _normalize_mechanism(self, mechanism: Optional[Union[str, ChemicalMechanism]]) -> ChemicalMechanism:
        """Coerce mechanism argument into a ChemicalMechanism enum."""
        if mechanism is None:
            raise ValueError("Chemical mechanism must be declared explicitly.")
        if isinstance(mechanism, ChemicalMechanism):
            return mechanism
        normalized = str(mechanism).strip().lower()
        for candidate in ChemicalMechanism:
            if candidate.value == normalized:
                return candidate
        raise ValueError(f"Unsupported chemical mechanism '{mechanism}'.")

    def _molecular_weight_key(self, species: str) -> str:
        """Return the full parameter key of one species molar mass [g mol-1]."""
        if species == 'CO':
            return self.CO_MOLAR_MASS_KEY
        if species == 'FORM':
            return "relic.physical.constants.molecular_weights.hcho"
        return f"anthroposphere.emissions.molecular_weights.{species}"

    def get_molecular_weight(self, species: str) -> float:
        """
        Get molecular weight for a species.

        Args:
            species: Species name

        Returns:
            Molecular weight [kg/mol]
        """
        base_value = self.molecular_weights.get(species)
        if base_value is None:
            raise KeyError(f"Unknown species '{species}' for molecular weight lookup")
        return base_value

    # References:
    # - Yarwood et al. (2005) CB05 mechanism documentation.
    #
    # Mathematical expression:
    # F_mol = F_mass / MW
    #
    # Parameter meanings:
    # F_mass: mass flux [kg m-2 s-1]
    # MW: molecular weight [kg mol-1]
    # F_mol: molar flux [mol m-2 s-1]
    def mass_to_molar_flux(self, mass_flux: torch.Tensor, species: str) -> torch.Tensor:
        """
        Convert mass flux to molar flux.

        Args:
            mass_flux: Mass flux [kg m^-2 s^-1]
            species: Species name

        Returns:
            Molar flux [mol m-2 s-1]
        """
        mw = self.get_molecular_weight(species)
        mass_device = mass_flux.device if torch.is_tensor(mass_flux) else None
        mass_dtype = mass_flux.dtype if torch.is_tensor(mass_flux) else torch.float32
        mass_tensor = torch.as_tensor(mass_flux, device=mass_device, dtype=mass_dtype)
        if torch.any(mass_tensor < 0):
            raise ValueError("Mass flux cannot be negative for species mapping.")
        return mass_tensor / mw

    # References:
    # - Yarwood et al. (2005) CB05 mechanism documentation.
    #
    # Mathematical expression:
    # F_mass = F_mol * MW
    #
    # Parameter meanings:
    # F_mol: molar flux [mol m-2 s-1]
    # MW: molecular weight [kg mol-1]
    # F_mass: mass flux [kg m-2 s-1]
    def molar_to_mass_flux(self, molar_flux: torch.Tensor, species: str) -> torch.Tensor:
        """
        Convert molar flux to mass flux.

        Args:
            molar_flux: Molar flux [mol m-2 s-1]
            species: Species name

        Returns:
            Mass flux [kg m^-2 s^-1]
        """
        mw = self.get_molecular_weight(species)
        molar_tensor = self._ensure_tensor(molar_flux)
        if torch.any(molar_tensor < 0.0):
            raise ValueError("Molar flux cannot be negative for species mapping.")
        return molar_tensor * mw

    # References:
    # - IPCC (2019) guidance on concentration units.
    #
    # Mathematical expression:
    # chi = ppm * 1e-6
    #
    # Parameter meanings:
    # ppm: concentration in parts per million [1e-6]
    # chi: mixing ratio [mol mol-1]
    @staticmethod
    def ppm_to_mixing_ratio(ppm: torch.Tensor) -> torch.Tensor:
        """
        Convert concentration from ppm to dimensionless mixing ratio.

        Mathematical expression: chi = ppm * 1e-6, where chi is mixing ratio.
        """
        ppm_device = ppm.device if torch.is_tensor(ppm) else None
        ppm_dtype = ppm.dtype if torch.is_tensor(ppm) else torch.float32
        ppm_tensor = torch.as_tensor(ppm, device=ppm_device, dtype=ppm_dtype)
        return ppm_tensor * 1e-06

    # References:
    # - IPCC (2019) guidance on concentration units.
    #
    # Mathematical expression:
    # ppm = chi * 1e6
    #
    # Parameter meanings:
    # chi: mixing ratio [mol mol-1]
    # ppm: concentration in parts per million [1e-6]
    @staticmethod
    def mixing_ratio_to_ppm(mixing_ratio: torch.Tensor) -> torch.Tensor:
        """
        Convert dimensionless mixing ratio to ppm concentration.

        Mathematical expression: ppm = chi * 1e6, where chi is mixing ratio.
        """
        ratio_device = mixing_ratio.device if torch.is_tensor(mixing_ratio) else None
        ratio_dtype = mixing_ratio.dtype if torch.is_tensor(mixing_ratio) else torch.float32
        ratio_tensor = torch.as_tensor(mixing_ratio, device=ratio_device, dtype=ratio_dtype)
        return ratio_tensor * 1000000.0

    # References:
    # - CMAQ User Guide (2019) emissions conversion.
    #
    # Mathematical expression:
    # delta_chi = F_mol * dt * R * T / (P * H)
    #
    # Parameter meanings:
    # F_mol: molar flux [mol m-2 s-1]
    # dt: time step [s]
    # R: gas constant [J mol-1 K-1]
    # P: pressure [Pa]
    # H: layer height [m]
    # T: temperature [K]
    @staticmethod
    def flux_to_mixing_ratio(
        molar_flux: torch.Tensor,
        pressure: torch.Tensor,
        temperature: torch.Tensor,
        layer_height: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Convert molar flux to mixing ratio change.

        Args:
            molar_flux: Molar flux [mol m-2 s-1]
            pressure: Air pressure [Pa]
            temperature: Air temperature [K]
            layer_height: Height of atmospheric layer [m]
            dt: Time step [s]

        Returns:
            Mixing ratio change [mol mol^-1]
        """
        R = get_physical_constant('gas_constant')
        target_device = None
        target_dtype = torch.float32
        for tensor_candidate in (molar_flux, pressure, temperature, layer_height):
            if torch.is_tensor(tensor_candidate):
                target_device = tensor_candidate.device
                target_dtype = tensor_candidate.dtype
                break
        R_tensor = torch.as_tensor(R, device=target_device, dtype=target_dtype)
        molar_flux_tensor = torch.as_tensor(molar_flux, device=target_device, dtype=target_dtype)
        pressure_tensor = torch.as_tensor(pressure, device=target_device, dtype=target_dtype)
        temperature_tensor = torch.as_tensor(temperature, device=target_device, dtype=target_dtype)
        layer_tensor = torch.as_tensor(layer_height, device=target_device, dtype=target_dtype)
        dt_tensor = torch.as_tensor(dt, device=target_device, dtype=target_dtype)
        if torch.any(molar_flux_tensor < 0.0):
            raise ValueError("Molar flux cannot be negative for mixing-ratio conversion.")
        if torch.any(temperature_tensor <= 0.0):
            raise ValueError("Air temperature must be positive for mixing-ratio conversion.")
        if torch.any(pressure_tensor <= 0.0):
            raise ValueError("Air pressure must be positive for mixing-ratio conversion.")
        if torch.any(layer_tensor <= 0.0):
            raise ValueError("Layer height must be positive for mixing-ratio conversion.")
        if torch.any(dt_tensor <= 0.0):
            raise ValueError("Time step must be positive for mixing-ratio conversion.")
        mixing_ratio_change = (
            molar_flux_tensor
            * dt_tensor
            * R_tensor
            * temperature_tensor
            / (pressure_tensor * layer_tensor)
        )
        return mixing_ratio_change

    def _speciate_voc(
        self,
        total_voc: torch.Tensor,
        sector: str,
    ) -> Dict[str, torch.Tensor]:
        """Speciate one sector VOC flux using its declared CB05 profile.

        References:
            CMAQ/SMOKE emissions preprocessing; Yarwood et al. (2005).
        Mathematical expression:
            F_i = f_i(sector) F_VOC and sum_i f_i = 1.
        Parameter meanings:
            f_i is the ParameterManager VOC profile fraction [1]. F_VOC and
            F_i use kg m-2 s-1.
        Open/closed route:
            This is a closed mass split with no discarded residual.

        Args:
            total_voc: Total VOC emissions
            sector: Emission sector

        Returns:
            Dictionary of CB05 VOC species and amounts
        """
        total_voc_tensor = torch.as_tensor(total_voc, dtype=torch.get_default_dtype())
        if torch.is_tensor(total_voc):
            total_voc_tensor = torch.as_tensor(
                total_voc,
                device=total_voc.device,
                dtype=total_voc.dtype,
            )
        if torch.any(total_voc_tensor < 0.0):
            raise ValueError("VOC flux cannot be negative for species mapping.")
        sector_key = self._resolve_sector_key(sector)
        profile = self.voc_speciation_profiles[sector_key]
        cb05_spec = {}
        for species, fraction in profile.items():
            if fraction > 0:
                cb05_spec[species] = total_voc_tensor * fraction
        return cb05_spec

    def get_supported_mechanisms(self) -> List[ChemicalMechanism]:
        """
        Get list of supported chemical mechanisms.

        Returns:
            List of supported mechanisms
        """
        return [ChemicalMechanism.CB05]

    def get_mechanism_species_list(self, mechanism: ChemicalMechanism) -> List[str]:
        """
        Get list of species for a given chemical mechanism.

        Args:
            mechanism: Chemical mechanism

        Returns:
            List of species names
        """
        if mechanism is not ChemicalMechanism.CB05:
            raise NotImplementedError(
                f"Mechanism '{mechanism.value}' lacks a complete typed runtime contract."
            )
        return [*self.MECHANISM_GAS_SPECIES, *self.MECHANISM_AEROSOL_SPECIES]

    def _get_partition_fraction(self, name: str) -> float:
        """Fetch partition fraction from partition_factors namespace."""
        full_key = f'anthroposphere.emissions.partition_factors.{name}'
        return self._require_dimensionless_parameter(full_key)

    def _get_pm_fraction(self, sector: str, component: str) -> float:
        """Fetch PM composition fraction for a given sector/component."""
        sector_key = self._resolve_sector_key(sector)
        if component not in self.PM_COMPONENT_PARAMETERS:
            raise KeyError(f"Unknown PM composition component '{component}'.")
        parameter_sector = self.PM_PARAMETER_SECTORS[sector_key]
        full_key = f'anthroposphere.emissions.pm_composition.{parameter_sector}.{component}'
        return self._require_dimensionless_parameter(full_key)

    def _speciate_nox(self, nox_flux: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Partition NOx emissions into NO and NO2.

        Args:
            nox_flux: Total NOx emissions [kg m^-2 s^-1]

        Returns:
            Dictionary with NO and NO2 fluxes [kg m^-2 s^-1]
        """
        # References:
        # - US EPA NEI convention: inventory NOx mass is reported as NO2 equivalent.
        # - US EPA emissions-modeling guidance: typical combustion split is 90% NO.
        # Mathematical expression:
        # - F_NO = f_NO * F_NOx,eq * MW_NO / MW_NOx.
        # - F_NO2 = (1 - f_NO) * F_NOx,eq * MW_NO2 / MW_NOx.
        # Parameter meanings and units:
        # - f_NO is anthroposphere.emissions.partition_factors.no_fraction [1].
        # - F values are mass fluxes [kg m-2 s-1]; MW values are [kg mol-1].
        # Open-system route:
        # - Nitrogen amount is closed; oxygen mass differs only by the NO2-equivalent basis.
        nox_tensor = self._ensure_tensor(nox_flux)
        if torch.any(nox_tensor < 0.0):
            raise ValueError("NOx flux cannot be negative for species mapping.")
        dtype = nox_tensor.dtype
        no_fraction_value = self._get_partition_fraction('no_fraction')
        if not 0.0 <= no_fraction_value <= 1.0:
            raise ValueError("NO partition fraction must be within [0, 1].")
        no_fraction = torch.as_tensor(
            no_fraction_value,
            device=nox_tensor.device,
            dtype=dtype,
        )
        mw_nox = torch.as_tensor(
            self.get_molecular_weight('NOx'),
            device=nox_tensor.device,
            dtype=dtype,
        )
        mw_no = torch.as_tensor(
            self.get_molecular_weight('NO'),
            device=nox_tensor.device,
            dtype=dtype,
        )
        mw_no2 = torch.as_tensor(
            self.get_molecular_weight('NO2'),
            device=nox_tensor.device,
            dtype=dtype,
        )
        no_equivalent_flux = nox_tensor * no_fraction
        no2_equivalent_flux = nox_tensor - no_equivalent_flux
        no_flux = no_equivalent_flux * mw_no / mw_nox
        no2_flux = no2_equivalent_flux * mw_no2 / mw_nox
        return {'NO2': no2_flux, 'NO': no_flux}

    def _speciate_pm25(
        self,
        pm25_flux: torch.Tensor,
        sector: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Partition PM2.5 emissions into primary components.

        Args:
            pm25_flux: Total PM2.5 emissions [kg m^-2 s^-1]
            sector: Emission sector

        Returns:
            Dictionary with PM components [kg m^-2 s^-1]
        """
        # References:
        # - US EPA SPECIATE profiles for primary particulate matter components.
        # - CMAQ emissions preprocessing guidance for aerosol mass speciation.
        # Mathematical expression:
        # - F_i = f_i * F_PM25 and F_OTHER = (1 - sum_i f_i) * F_PM25.
        # Parameter meanings and units:
        # - f_i are sector PM composition fractions [1].
        # - F_PM25 and F_i are mass fluxes [kg m-2 s-1].
        # Open-system route:
        # - This is a closed mass split; the six component fluxes sum to F_PM25.
        pm25_tensor = self._ensure_tensor(pm25_flux)
        if torch.any(pm25_tensor < 0.0):
            raise ValueError("PM2.5 flux cannot be negative for species mapping.")
        sector_key = self._resolve_sector_key(sector)
        fractions = torch.as_tensor(
            [
                self._get_pm_fraction(sector_key, 'bc_fraction'),
                self._get_pm_fraction(sector_key, 'oc_fraction'),
                self._get_pm_fraction(sector_key, 'nitrate_fraction'),
                self._get_pm_fraction(sector_key, 'sulfate_fraction'),
                self._get_pm_fraction(sector_key, 'ammonium_fraction'),
            ],
            device=pm25_tensor.device,
            dtype=pm25_tensor.dtype,
        )
        if torch.any((fractions < 0.0) | (fractions > 1.0)):
            raise ValueError("PM2.5 composition fractions must each be within [0, 1].")
        total_fraction = fractions.sum()
        if total_fraction > 1.0:
            raise ValueError("PM2.5 composition fractions must sum to at most 1.")
        other_fraction = (
            torch.as_tensor(1.0, device=pm25_tensor.device, dtype=pm25_tensor.dtype)
            - total_fraction
        )
        bc_fraction, oc_fraction, nitrate_fraction, sulfate_fraction, ammonium_fraction = fractions
        pm_bc_flux = pm25_tensor * bc_fraction
        pm_oc_flux = pm25_tensor * oc_fraction
        pm_no3_flux = pm25_tensor * nitrate_fraction
        pm_so4_flux = pm25_tensor * sulfate_fraction
        pm_nh4_flux = pm25_tensor * ammonium_fraction
        pm_other_flux = pm25_tensor * other_fraction
        return {
            'BC': pm_bc_flux,
            'OC': pm_oc_flux,
            'NO3': pm_no3_flux,
            'SO4': pm_so4_flux,
            'NH4': pm_nh4_flux,
            'OTHER': pm_other_flux,
        }

    def upscale(
        self,
        fine_data: Union[Dict[str, torch.Tensor], torch.Tensor],
        scaling_factor: int = 10,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Reject undeclared spatial aggregation for chemical-axis mapping."""
        _ = fine_data, scaling_factor
        raise RuntimeError(
            "SpeciesMapper does not own a spatial upscale operator; use the declared grid mapper."
        )

    def downscale(
        self,
        coarse_data: Union[Dict[str, torch.Tensor], torch.Tensor],
        spatial_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reject undeclared spatial disaggregation for chemical-axis mapping."""
        _ = coarse_data, spatial_weights
        raise RuntimeError(
            "SpeciesMapper does not own a spatial downscale operator; use SectorAllocation."
        )

    def mixing_ratio_to_flux(
        self,
        mixing_ratio: torch.Tensor,
        pressure: torch.Tensor,
        temperature: torch.Tensor,
        layer_height: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Convert mixing ratio change to molar flux.

        Args:
            mixing_ratio: Mixing ratio change [mol mol^-1]
            pressure: Air pressure [Pa]
            temperature: Air temperature [K]
            layer_height: Height of atmospheric layer [m]
            dt: Time step [s]

        Returns:
            Molar flux [mol m-2 s-1]
        """
        ratio = self._ensure_tensor(mixing_ratio)
        pressure_tensor = self._ensure_tensor(pressure)
        temperature_tensor = self._ensure_tensor(temperature)
        layer_tensor = self._ensure_tensor(layer_height)
        dt_tensor = self._ensure_tensor(dt)
        if torch.any(temperature_tensor <= 0.0):
            raise ValueError("Air temperature must be positive for flux conversion.")
        if torch.any(pressure_tensor <= 0.0):
            raise ValueError("Air pressure must be positive for flux conversion.")
        if torch.any(layer_tensor <= 0.0):
            raise ValueError("Layer height must be positive for flux conversion.")
        if torch.any(dt_tensor <= 0.0):
            raise ValueError("Time step must be positive for flux conversion.")
        gas_constant = torch.as_tensor(
            get_physical_constant('gas_constant'),
            device=ratio.device,
            dtype=ratio.dtype,
        )
        return (
            ratio
            * pressure_tensor
            * layer_tensor
            / (gas_constant * temperature_tensor * dt_tensor)
        )


__all__ = ['SpeciesMapper']

