"""
BVOC Emission Factors Database for RELIC

This module provides comprehensive emission factor data for different plant 
functional types (PFTs) and species. It centralizes all emission factor 
information used across the BVOC modeling framework.

References
----------
- Guenther, A., et al. (2012). The Model of Emissions of Gases and Aerosols from 
  Nature version 2.1 (MEGAN2.1): an extended and updated framework for modeling 
  biogenic emissions. Geoscientific Model Development 5, 1471-1492
- Guenther, A.B., et al. (2006). Estimates of global terrestrial isoprene emissions 
  using MEGAN. Atmospheric Chemistry and Physics 6, 3181-3210
- Monson, R.K., et al. (2007). Isoprene emission from terrestrial ecosystems in
  response to global change: minding the gap between models and observations.
  Philosophical Transactions of the Royal Society A 365, 1677-1695
- Arneth, A., et al. (2007). Process-based estimates of terrestrial ecosystem
  isoprene emissions: incorporating the effects of a direct CO2-isoprene
  interaction. Atmospheric Chemistry and Physics 7, 31-53

Data Sources
------------
- MEGAN v2.1 global emission factor database
- Regional and species-specific measurements from literature
- PFT-averaged values from global synthesis studies
- Uncertainty estimates from measurement variability

Units
-----
- Emission factors: mug g-1 DW h-1 (micrograms per gram dry weight per hour)
- Standard conditions: 30degC, 1000 mumol m-2 s-1 PPFD
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from torch import Tensor
from dataclasses import dataclass
from enum import Enum
from threading import RLock
import json
import warnings

from relic.parameters.ops.key_normalization import normalize_parameter_key
from relic.parameters import get_global_manager

BVOC_EMISSION_FACTOR_UNITS = "ug g-1 DW h-1"


class PlantFunctionalType(Enum):
    """Plant Functional Type categories for emission factors."""
    BROADLEAF_EVERGREEN_TROPICAL = "broadleaf_evergreen_tropical"
    BROADLEAF_EVERGREEN_TEMPERATE = "broadleaf_evergreen_temperate"
    BROADLEAF_DECIDUOUS_TROPICAL = "broadleaf_deciduous_tropical"
    BROADLEAF_DECIDUOUS_TEMPERATE = "broadleaf_deciduous_temperate"
    NEEDLELEAF_EVERGREEN_TEMPERATE = "needleleaf_evergreen_temperate"
    NEEDLELEAF_EVERGREEN_BOREAL = "needleleaf_evergreen_boreal"
    NEEDLELEAF_DECIDUOUS_BOREAL = "needleleaf_deciduous_boreal"
    C3_GRASS = "c3_grass"
    C4_GRASS = "c4_grass"
    SHRUB = "shrub"
    CROP = "crop"
    WETLAND = "wetland"
    DEFAULT = "default"


class CompoundCategory(Enum):
    """BVOC compound categories."""
    ISOPRENE = "isoprene"
    MONOTERPENES = "monoterpenes"
    SESQUITERPENES = "sesquiterpenes"
    OVOCS = "ovocs"
    NO = "no"
    AEROSOLS = "aerosols"
    CO = "co"


DEFAULT_COMPOUND_CATEGORIES = (
    CompoundCategory.ISOPRENE,
    CompoundCategory.MONOTERPENES,
    CompoundCategory.SESQUITERPENES,
    CompoundCategory.OVOCS,
)


@dataclass
class EmissionFactorEntry:
    """Single emission factor entry with metadata."""
    value: float                    # Emission factor [mug g-1 DW h-1]
    uncertainty: float              # Uncertainty factor (multiplicative)
    sample_size: Optional[int] = None    # Number of measurements
    reference: Optional[str] = None      # Literature reference
    notes: Optional[str] = None          # Additional notes
    units: str = "ug g-1 DW h-1"
    
    def get_value_with_uncertainty(self, uncertainty_method: str = "log_normal") -> Tuple[float, float]:
        """
        Get emission factor value with uncertainty bounds.
        
        Parameters
        ----------
        uncertainty_method : str
            Method for uncertainty calculation ('log_normal', 'linear')
            
        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound)
        """
        if uncertainty_method == "log_normal":
            # Log-normal uncertainty (typical for emission factors)
            lower = self.value / self.uncertainty
            upper = self.value * self.uncertainty
        else:  # linear
            # Linear uncertainty
            lower = self.value - self.uncertainty
            upper = self.value + self.uncertainty
            
        return (max(0.0, lower), upper)


_EmissionFactorTable = Dict[
    PlantFunctionalType,
    Dict[CompoundCategory, EmissionFactorEntry],
]
_EmissionFactorSnapshot = Tuple[Any, _EmissionFactorTable]


class EmissionFactorDatabase:
    """
    Comprehensive emission factor database.
    
    Stores emission factors organized by PFT and compound category,
    with uncertainty information and literature references.
    """
    
    def __init__(self, custom_factors: Optional[Dict] = None, parameter_manager: Optional[Any] = None):
        """
        Initialize emission factor database.
        
        Parameters
        ----------
        custom_factors : Dict, optional
            Custom emission factors to override defaults.
        """
        self._explicit_parameter_manager: Optional[Any] = parameter_manager
        self._snapshot_lock = RLock()
        self._custom_factors = custom_factors
        manager = (
            parameter_manager
            if parameter_manager is not None
            else get_global_manager()
        )
        self._snapshot = self._build_snapshot(manager)

    def _build_snapshot(self, manager: Any) -> _EmissionFactorSnapshot:
        """Build a complete immutable-by-owner database snapshot."""
        database = self._build_database_from_parameters(manager)
        if self._custom_factors:
            self._update_database(database, self._custom_factors)
        return manager, database

    def _current_snapshot(self) -> _EmissionFactorSnapshot:
        """Return one coherent snapshot and reconcile a replaced global owner."""
        snapshot = self._snapshot
        if self._explicit_parameter_manager is not None:
            return snapshot
        current_manager = get_global_manager()
        if current_manager is snapshot[0]:
            return snapshot
        with self._snapshot_lock:
            snapshot = self._snapshot
            current_manager = get_global_manager()
            if current_manager is not snapshot[0]:
                snapshot = self._build_snapshot(current_manager)
                self._snapshot = snapshot
            return snapshot

    @property
    def param_manager(self) -> Any:
        """Return the ParameterManager that owns the current database snapshot."""
        return self._current_snapshot()[0]

    @property
    def database(self) -> _EmissionFactorTable:
        """Return the complete table from the current coherent snapshot."""
        return self._current_snapshot()[1]

    def refresh_from_manager(self) -> None:
        """Atomically rebuild from the explicit or current global manager."""
        manager = (
            self._explicit_parameter_manager
            if self._explicit_parameter_manager is not None
            else get_global_manager()
        )
        snapshot = self._build_snapshot(manager)
        with self._snapshot_lock:
            self._snapshot = snapshot

    def _build_database_from_parameters(self, manager: Any) -> _EmissionFactorTable:
        """Build emission factors from one captured ParameterManager."""
        database: _EmissionFactorTable = {}
        missing: List[str] = []

        for pft in PlantFunctionalType:
            database[pft] = {}
            for compound in DEFAULT_COMPOUND_CATEGORIES:
                param_key = normalize_parameter_key(f"biosphere.bvoc.{pft.value}.{compound.value}")
                try:
                    entry = self._build_entry_from_parameters(param_key, manager)
                except KeyError as exc:
                    missing.append(str(exc))
                    continue
                database[pft][compound] = entry

        if missing:
            missing_list = "\n".join(sorted(set(missing)))
            raise KeyError(
                "Missing BVOC emission factor parameters detected. "
                "Please add the following keys to "
                "relic\\parameters\\database\\relic\\biosphere\\bvoc_parameters.yaml:\n"
                f"{missing_list}"
            )
        return database

    def _build_entry_from_parameters(
        self,
        param_key: str,
        manager: Any,
    ) -> EmissionFactorEntry:
        """
        Build an EmissionFactorEntry from the parameter database.

        Parameters
        ----------
        param_key : str
            Fully qualified parameter key in the parameter database.

        Returns
        -------
        EmissionFactorEntry
            Entry populated from ParameterManager.

        Raises
        ------
        KeyError
            If the parameter is missing or unresolved.
        """
        spec = manager.get_parameter(param_key)
        if spec is None:
            raise KeyError(f"Parameter '{param_key}' not found in ParameterManager.")

        raw_value = manager.get(
            param_key,
            units=BVOC_EMISSION_FACTOR_UNITS,
            required=True,
            expected_type=None,
        )
        if isinstance(raw_value, Tensor):
            scalar_value = raw_value.detach().reshape(-1)[0].tolist()
            value = float(scalar_value)
        else:
            value = float(raw_value)

        metadata = self._extract_metadata_from_spec(spec)
        uncertainty = metadata.get("uncertainty")
        if uncertainty is None:
            uncertainty = 1.0
        return EmissionFactorEntry(
            value=value,
            uncertainty=float(uncertainty),
            sample_size=metadata.get("sample_size"),
            reference=metadata.get("reference"),
            notes=metadata.get("notes"),
            units=BVOC_EMISSION_FACTOR_UNITS,
        )

    @staticmethod
    def _extract_metadata_from_spec(spec: Any) -> Dict[str, Any]:
        """Extract uncertainty/sample size/reference metadata from a ParameterSpec."""
        provenance = getattr(spec, "provenance", {}) or {}
        references = getattr(spec, "references", None) or []

        reference = None
        if references:
            first_ref = references[0]
            if isinstance(first_ref, dict):
                reference = next(iter(first_ref.values())) if first_ref else None
            else:
                reference = str(first_ref)

        return {
            "uncertainty": provenance.get("uncertainty"),
            "sample_size": provenance.get("sample_size"),
            "notes": provenance.get("notes"),
            "reference": reference,
        }

    def _update_database(
        self,
        database: _EmissionFactorTable,
        custom_factors: Dict,
    ) -> None:
        """Update database with custom emission factors."""
        for pft_key, compounds in custom_factors.items():
            if isinstance(pft_key, str):
                try:
                    pft = PlantFunctionalType(pft_key)
                except ValueError:
                    warnings.warn(f"Unknown PFT: {pft_key}")
                    continue
            else:
                pft = pft_key
            
            if pft not in database:
                database[pft] = {}
            
            for compound_key, value in compounds.items():
                if isinstance(compound_key, str):
                    try:
                        compound = CompoundCategory(compound_key)
                    except ValueError:
                        warnings.warn(f"Unknown compound: {compound_key}")
                        continue
                else:
                    compound = compound_key
                
                if isinstance(value, (int, float)):
                    # Simple value, create EmissionFactorEntry
                    database[pft][compound] = EmissionFactorEntry(
                        value=float(value), uncertainty=2.0, reference="Custom"
                    )
                else:
                    # Assume it's already an EmissionFactorEntry or dict
                    if isinstance(value, dict):
                        database[pft][compound] = EmissionFactorEntry(**value)
                    else:
                        database[pft][compound] = value
    
    def get_emission_factor(
        self, 
        pft: Union[PlantFunctionalType, str], 
        compound: Union[CompoundCategory, str]
    ) -> EmissionFactorEntry:
        """
        Get emission factor for specific PFT and compound.
        
        Parameters
        ----------
        pft : PlantFunctionalType or str
            Plant functional type
        compound : CompoundCategory or str
            BVOC compound category
            
        Returns
        -------
        EmissionFactorEntry
            Emission factor with metadata
        """
        # Convert strings to enums if needed
        if isinstance(pft, str):
            try:
                pft = PlantFunctionalType(pft)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown PFT '{pft}'. Ensure the PFT is registered in "
                    "PlantFunctionalType and has parameters in the BVOC database."
                ) from exc
        
        if isinstance(compound, str):
            try:
                compound = CompoundCategory(compound)
            except ValueError as exc:
                raise ValueError(f"Unknown compound category: {compound}") from exc

        database = self._current_snapshot()[1]
        if pft not in database:
            raise KeyError(
                f"Emission factors for PFT '{pft.value}' are not loaded. "
                "Add the corresponding parameters to "
                "relic\\parameters\\database\\relic\\biosphere\\bvoc_parameters.yaml."
            )

        if compound not in database[pft]:
            raise KeyError(
                f"Emission factor '{pft.value}.{compound.value}' is missing. "
                "Declare it in relic\\parameters\\database\\relic\\biosphere\\bvoc_parameters.yaml."
            )

        return database[pft][compound]
    
    def get_all_factors_for_pft(self, pft: Union[PlantFunctionalType, str]) -> Dict[CompoundCategory, EmissionFactorEntry]:
        """
        Get all emission factors for a specific PFT.
        
        Parameters
        ----------
        pft : PlantFunctionalType or str
            Plant functional type
            
        Returns
        -------
        Dict[CompoundCategory, EmissionFactorEntry]
            All emission factors for the PFT
        """
        if isinstance(pft, str):
            try:
                pft = PlantFunctionalType(pft)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown PFT '{pft}'. Provide a valid PlantFunctionalType."
                ) from exc
        
        database = self._current_snapshot()[1]
        if pft not in database:
            raise KeyError(
                f"Emission factors for PFT '{pft.value}' are not available. "
                "Ensure parameters are declared in the BVOC parameter YAML."
            )

        return database[pft].copy()
    
    def get_factor_tensor(
        self, 
        pft: Union[PlantFunctionalType, str], 
        compound: Union[CompoundCategory, str],
        device: Optional[torch.device] = None
    ) -> Tensor:
        """
        Get emission factor as PyTorch tensor.
        
        Parameters
        ----------
        pft : PlantFunctionalType or str
            Plant functional type
        compound : CompoundCategory or str
            BVOC compound category
        device : torch.device, optional
            Target device for tensor
            
        Returns
        -------
        Tensor
            Emission factor as tensor [mug g-1 DW h-1]
        """
        factor_entry = self.get_emission_factor(pft, compound)
        return torch.tensor(factor_entry.value, dtype=torch.float32, device=device)
    
    def get_uncertainty_bounds(
        self, 
        pft: Union[PlantFunctionalType, str], 
        compound: Union[CompoundCategory, str],
        device: Optional[torch.device] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Get uncertainty bounds as tensors.
        
        Parameters
        ----------
        pft : PlantFunctionalType or str
            Plant functional type
        compound : CompoundCategory or str  
            BVOC compound category
        device : torch.device, optional
            Target device for tensors
            
        Returns
        -------
        Tuple[Tensor, Tensor]
            (lower_bound, upper_bound) tensors
        """
        factor_entry = self.get_emission_factor(pft, compound)
        lower, upper = factor_entry.get_value_with_uncertainty()
        
        lower_tensor = torch.tensor(lower, dtype=torch.float32, device=device)
        upper_tensor = torch.tensor(upper, dtype=torch.float32, device=device)
        
        return lower_tensor, upper_tensor
    
    def list_available_pfts(self) -> List[PlantFunctionalType]:
        """List all available plant functional types."""
        database = self._current_snapshot()[1]
        return list(database.keys())
    
    def list_available_compounds(self) -> List[CompoundCategory]:
        """List all available compound categories."""
        database = self._current_snapshot()[1]
        compounds = set()
        for pft_data in database.values():
            compounds.update(pft_data.keys())
        return list(compounds)
    
    def export_database(self) -> Dict:
        """Export database in JSON-serializable format."""
        database = self._current_snapshot()[1]
        export_data = {}

        for pft, compounds in database.items():
            pft_key = pft.value if isinstance(pft, PlantFunctionalType) else str(pft)
            export_data[pft_key] = {}
            
            for compound, entry in compounds.items():
                compound_key = compound.value if isinstance(compound, CompoundCategory) else str(compound)
                export_data[pft_key][compound_key] = {
                    'value': entry.value,
                    'uncertainty': entry.uncertainty,
                    'sample_size': entry.sample_size,
                    'reference': entry.reference,
                    'notes': entry.notes
                }
        
        return export_data
    
    def save_to_file(self, filepath: str):
        """Save database to JSON file."""
        export_data = self.export_database()
        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'EmissionFactorDatabase':
        """Load database from JSON file."""
        with open(filepath, 'r', encoding="utf-8-sig") as f:
            data = json.load(f)
        return cls(custom_factors=data)


# Global default instance
DEFAULT_EMISSION_FACTORS = EmissionFactorDatabase()


# Export classes and instances
__all__ = [
    'EmissionFactorDatabase',
    'EmissionFactorEntry',
    'PlantFunctionalType',
    'CompoundCategory',
    'DEFAULT_EMISSION_FACTORS'
]

