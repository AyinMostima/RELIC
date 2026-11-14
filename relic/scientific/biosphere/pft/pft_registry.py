"""
Plant Functional Type (PFT) Registry and Parameter Management System

This module provides the central registry for managing PFT definitions,
parameter loading, caching, and validation. It serves as the single source
of truth for PFT parameters across all biosphere processes.

Features
--------
- Thread-safe PFT registration and lookup
- YAML-based parameter file loading  
- Parameter validation and consistency checking
- Caching for performance optimization
- Version control and parameter provenance tracking
- Integration with parameter database system

Architecture
------------
- PFTRegistry: Main registry class with singleton pattern
- Parameter loading from YAML configuration files
- Lazy loading and caching of PFT definitions
- Validation pipeline with configurable strictness levels
- Integration with relic parameter management system

References
----------
- Design patterns from CLM parameter management
- YAML configuration standards from CESM/CIME
- Parameter validation approaches from ED2/FATES models
"""

from typing import Dict, List, Optional, Union, Any, Set
from pathlib import Path
import yaml
import logging
import threading

# Import PFT type definitions
from .pft_types import (PFTDefinition, 
    PFTCategory, 
    PhotosynthesisPathway,
    PFTCollection
)

# Parameter management imports
from relic.parameters import get_database_root, get_global_manager

logger = logging.getLogger(__name__)

_PFT_CATEGORY_BY_NAME = {category.value: category for category in PFTCategory}
_NON_PFT_PARAMETER_NAMES = {"default", "metadata"}


def _pft_identity_declared_in_file(filepath: Path) -> tuple[Set[str], bool]:
    """Return declared PFT names and whether the file is shared metadata."""
    payload = yaml.safe_load(filepath.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(payload, dict):
        return set(), False

    names: Set[str] = set()
    namespace = payload.get("namespace")
    if namespace == "biosphere.pft":
        return set(), True
    if isinstance(namespace, str) and namespace.startswith("biosphere.pft."):
        parts = namespace.split(".")
        if len(parts) >= 3:
            names.add(parts[2])

    parameters = payload.get("parameters") or {}
    if isinstance(parameters, dict):
        for parameter_key in parameters:
            if not isinstance(parameter_key, str):
                continue
            if not parameter_key.startswith("biosphere.pft."):
                continue
            parts = parameter_key.split(".")
            if len(parts) >= 4:
                names.add(parts[2])
    return names, False


def _get_parameter_manager():
    """Resolve ParameterManager through the canonical global manager entry."""
    return get_global_manager()


class PFTRegistryError(Exception):
    """Custom exception for PFT registry errors."""


class PFTValidationError(PFTRegistryError):
    """Exception for PFT parameter validation errors."""


class PFTRegistry:
    """
    Central registry for Plant Functional Type definitions and parameters.
    
    This class implements a singleton pattern to ensure consistent PFT
    parameter access across all biosphere modules. It provides methods
    for loading, validating, caching, and retrieving PFT definitions.
    
    Features
    --------
    - Thread-safe singleton pattern
    - Lazy loading of PFT definitions
    - Parameter validation with configurable strictness
    - YAML configuration file support
    - Parameter versioning and provenance tracking
    - Integration with parameter database system
    
    Examples
    --------
    >>> registry = PFTRegistry.get_instance()
    >>> pft = registry.get_pft('broadleaf_deciduous_temperate')
    >>> vcmax = pft.photosynthesis.vcmax25
    >>> print(f"Vcmax25: {vcmax} umol CO2 m-2 s-1")
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PFTRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize PFT registry."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._pft_definitions: PFTCollection = {}
        self._lightweight_pfts: Set[str] = set()
        self._parameter_files: List[Path] = []
        self._validation_strict = False  # Set to False to allow loading with minor issues
        self._cache_enabled = True
        self._version = "1.0.0"
        
        # Threading locks for thread safety
        self._registry_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        
        # Cache for performance
        self._parameter_cache: Dict[str, Any] = {}
        self._validation_cache: Dict[str, bool] = {}
        
        logger.info("PFT Registry initialized (lazy default loading)")
    
    @classmethod
    def get_instance(cls) -> 'PFTRegistry':
        """
        Get the singleton instance of PFT registry.
        
        Returns
        -------
        PFTRegistry
            The singleton registry instance
        """
        return cls()
    
    def load_default_pfts(self, parameter_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Load the PFT definitions of the packaged parameter database.

        Parameters
        ----------
        parameter_dir : Optional[Union[str, Path]]
            Directory whose ``pft`` folder holds one YAML file per PFT. None selects
            the relic tier of the packaged parameter database,
            ``relic.parameters.get_database_root() / "relic" / "biosphere"``: the fixed
            location of the PFT files indexed by ``pft/__init__.yaml``.

        Raises
        ------
        PFTRegistryError
            If the directory holds no PFT folder or no PFT file; no default PFT is
            fabricated. A file that cannot be loaded raises from load_pft_file.
        """
        with self._registry_lock:
            if parameter_dir is None:
                parameter_dir = get_database_root() / "relic" / "biosphere"
            else:
                parameter_dir = Path(parameter_dir)

            pft_subdir = parameter_dir / "pft"
            if not pft_subdir.is_dir():
                raise PFTRegistryError(f"PFT parameter directory not found: {pft_subdir}")

            # One YAML file per PFT plus the shared metadata file; the index and report
            # files are not parameter files.
            skipped = {'__init__', 'split_validation_report', 'phase_6_4_cleanup_report'}
            yaml_files = sorted(
                path
                for pattern in ("*.yaml", "*.yml")
                for path in pft_subdir.glob(pattern)
                if path.stem not in skipped
            )
            if not yaml_files:
                raise PFTRegistryError(f"No PFT parameter files found in {pft_subdir}")

            for yaml_file in yaml_files:
                self.load_pft_file(yaml_file, lightweight=True)
                logger.info(f"Loaded PFT parameters from {yaml_file}")
    
    def load_pft_file(self, filepath: Union[str, Path], lightweight: bool = False) -> None:
        """
        Load PFT parameters from ParameterSpec format YAML file.
        
        This method loads parameters in ParameterSpec format and registers them
        with the ParameterManager, then creates PFT definitions from the registered
        parameters to avoid duplicate parameter sources.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to YAML file containing ParameterSpec format parameters
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"PFT parameter file not found: {filepath}")
        
        # Load ParameterSpec format YAML file via ParameterManager
        pm = _get_parameter_manager()
        loaded_count = pm.load_from_database(str(filepath))

        if loaded_count == 0:
            logger.warning(f"No parameters loaded from {filepath}")
            return

        logger.info(f"Loaded {loaded_count} parameters from {filepath}")

        # Materialize only the PFT declared by this file. Parameter values remain owned
        # by ParameterManager; YAML is read only for identity. A file without a PFT
        # namespace, an unregistered PFT type or a PFT whose definition cannot be built
        # raises: the registry never drops a PFT file silently.
        pft_names, shared_metadata = _pft_identity_declared_in_file(filepath)
        if shared_metadata:
            if filepath not in self._parameter_files:
                self._parameter_files.append(filepath)
            return
        if not pft_names:
            raise ValueError(f"No PFT namespace declared in {filepath}")

        for pft_name in sorted(pft_names):
            if pft_name in _NON_PFT_PARAMETER_NAMES:
                logger.debug("Skipping non-PFT metadata entry %s", pft_name)
                continue
            category = _PFT_CATEGORY_BY_NAME.get(pft_name)
            if category is None:
                raise ValueError(
                    f"Unknown PFT type: {pft_name}. Available types: "
                    f"{sorted(_PFT_CATEGORY_BY_NAME)}"
                )

            existing_pft = self._pft_definitions.get(pft_name)
            if existing_pft is not None and existing_pft.source == str(filepath):
                logger.debug(
                    "PFT '%s' already loaded from %s",
                    pft_name,
                    filepath,
                )
                continue

            pft_def = PFTDefinition(
                name=pft_name,
                category=category,
                description=f"PFT definition loaded from {filepath.name}",
                source=str(filepath),
                _use_param_manager=(not lightweight),
                parameter_manager=pm,
            )

            # Validate the PFT
            if self._validation_strict:
                self._validate_pft(pft_def)

            self.register_pft(pft_def)
            if lightweight:
                self._lightweight_pfts.add(pft_name)
            logger.info(f"Created PFT definition: {pft_name}")

        if filepath not in self._parameter_files:
            self._parameter_files.append(filepath)
    
    def register_pft(self, pft_definition: PFTDefinition) -> None:
        """
        Register a PFT definition in the registry.
        
        Parameters
        ----------
        pft_definition : PFTDefinition
            PFT definition to register
        """
        with self._registry_lock:
            pft_name = pft_definition.name
            
            # Check for existing PFT
            if pft_name in self._pft_definitions:
                logger.warning(f"Overwriting existing PFT definition: {pft_name}")
            
            # Validate if strict mode enabled
            if self._validation_strict:
                self._validate_pft(pft_definition)
            
            # Register the PFT
            self._pft_definitions[pft_name] = pft_definition
            if pft_definition.photosynthesis is not None:
                self._lightweight_pfts.discard(pft_name)
            
            # Clear relevant caches
            with self._cache_lock:
                self._parameter_cache.pop(pft_name, None)
                self._validation_cache.pop(pft_name, None)
            
            logger.info(f"Registered PFT: {pft_name} ({pft_definition.category.value})")
    
    def get_pft(self, pft_name: str) -> PFTDefinition:
        """
        Get PFT definition by name.
        
        Parameters
        ----------
        pft_name : str
            Name of the PFT to retrieve
            
        Returns
        -------
        PFTDefinition
            PFT definition
            
        Raises
        ------
        PFTRegistryError
            If PFT is not found
        """
        with self._registry_lock:
            if pft_name not in self._pft_definitions:
                # Try to load defaults if not loaded
                if not self._pft_definitions:
                    logger.info("No PFTs loaded, attempting to load defaults")
                    self.load_default_pfts()
                
                # Check again
                if pft_name not in self._pft_definitions:
                    available = list(self._pft_definitions.keys())
                    raise PFTRegistryError(
                        f"PFT '{pft_name}' not found. Available PFTs: {available}"
                    )
            
            pft_definition = self._pft_definitions[pft_name]
            if pft_name in self._lightweight_pfts:
                pft_definition = PFTDefinition(
                    name=pft_definition.name,
                    category=pft_definition.category,
                    description=pft_definition.description,
                    version=pft_definition.version,
                    source=pft_definition.source,
                    validation_status=pft_definition.validation_status,
                    _use_param_manager=True,
                    parameter_manager=_get_parameter_manager(),
                )
                if self._validation_strict:
                    self._validate_pft(pft_definition)
                self._pft_definitions[pft_name] = pft_definition
                self._lightweight_pfts.discard(pft_name)

            return pft_definition
    
    def get_pft_parameter(self, pft_name: str, parameter_path: str) -> Any:
        """
        Get specific parameter from PFT.
        
        Parameters
        ----------
        pft_name : str
            Name of the PFT
        parameter_path : str
            Dot notation path to parameter (e.g., 'photosynthesis.vcmax25')
            
        Returns
        -------
        Any
            Parameter value
        """
        # Check cache first
        cache_key = f"{pft_name}.{parameter_path}"
        if self._cache_enabled:
            with self._cache_lock:
                if cache_key in self._parameter_cache:
                    return self._parameter_cache[cache_key]
        
        # Get parameter from PFT
        pft = self.get_pft(pft_name)
        parameter_value = pft.get_parameter(parameter_path)
        
        # Cache the result
        if self._cache_enabled:
            with self._cache_lock:
                self._parameter_cache[cache_key] = parameter_value
        
        return parameter_value
    
    def list_pfts(self) -> List[str]:
        """
        Get list of all registered PFT names.
        
        Returns
        -------
        List[str]
            List of PFT names
        """
        with self._registry_lock:
            if not self._pft_definitions:
                logger.info("No PFTs loaded, attempting to load defaults")
                self.load_default_pfts()
            return list(self._pft_definitions.keys())
    
    def list_pft_categories(self) -> List[str]:
        """
        Get list of all PFT categories in registry.
        
        Returns
        -------
        List[str]
            List of PFT category names
        """
        with self._registry_lock:
            categories = set()
            for pft in self._pft_definitions.values():
                categories.add(pft.category.value)
            return list(categories)
    
    def get_pfts_by_category(self, category: Union[str, PFTCategory]) -> List[str]:
        """
        Get PFT names by category.
        
        Parameters
        ----------
        category : Union[str, PFTCategory]
            PFT category to filter by
            
        Returns
        -------
        List[str]
            List of PFT names in the category
        """
        if isinstance(category, str):
            category = PFTCategory(category)
        
        with self._registry_lock:
            pfts = []
            for pft_name, pft_def in self._pft_definitions.items():
                if pft_def.category == category:
                    pfts.append(pft_name)
            return pfts
    
    def validate_all_pfts(self) -> Dict[str, List[str]]:
        """
        Validate all registered PFTs and return validation results.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping PFT names to list of validation errors
            Empty list means no errors
        """
        results = {}
        
        with self._registry_lock:
            for pft_name, pft_def in self._pft_definitions.items():
                try:
                    self._validate_pft(pft_def)
                    results[pft_name] = []
                except PFTValidationError as e:
                    results[pft_name] = [str(e)]
        
        return results
    
    def _validate_pft(self, pft_definition: PFTDefinition) -> None:
        """
        Validate a PFT definition.
        
        Parameters
        ----------
        pft_definition : PFTDefinition
            PFT to validate
            
        Raises
        ------
        PFTValidationError
            If validation fails
        """
        # Check cache first
        cache_key = f"validation_{pft_definition.name}"
        if self._cache_enabled:
            with self._cache_lock:
                if cache_key in self._validation_cache:
                    if not self._validation_cache[cache_key]:
                        raise PFTValidationError(f"Cached validation failure for {pft_definition.name}")
                    return
        
        errors = []

        try:
            # Additional cross-parameter validation
            errors.extend(self._validate_photosynthesis_consistency(pft_definition))
            errors.extend(self._validate_allocation_consistency(pft_definition))
            errors.extend(self._validate_phenology_consistency(pft_definition))
            errors.extend(self._validate_stress_consistency(pft_definition))
            
            # Check for unrealistic parameter combinations
            errors.extend(self._validate_parameter_ranges(pft_definition))
            
        except TypeError as e:
            # A parameter value that is not a real number (for example a string or a
            # non-scalar) cannot enter the consistency checks: the PFT fails validation.
            errors.append(f"Parameter value of the wrong type: {e}")
        
        # Cache validation result
        if self._cache_enabled:
            with self._cache_lock:
                self._validation_cache[cache_key] = len(errors) == 0
        
        if errors:
            error_msg = f"PFT validation failed for '{pft_definition.name}': " + "; ".join(errors)
            raise PFTValidationError(error_msg)
    
    def _validate_photosynthesis_consistency(self, pft: PFTDefinition) -> List[str]:
        """Validate photosynthesis parameter consistency."""
        errors = []
        
        # Skip validation if parameters are not set
        if (pft.photosynthesis is None or 
            pft.photosynthesis.vcmax25 is None or 
            pft.photosynthesis.jmax25 is None):
            return errors
        
        # Check Vcmax/Jmax relationship
        if pft.photosynthesis.vcmax25 > 0:
            ratio = pft.photosynthesis.jmax25 / pft.photosynthesis.vcmax25
            if pft.photosynthesis.pathway == PhotosynthesisPathway.C3:
                if ratio < 1.0 or ratio > 3.0:
                    errors.append(f"C3 Jmax/Vcmax ratio {ratio:.2f} outside acceptable range [1.0, 3.0]")
            elif pft.photosynthesis.pathway == PhotosynthesisPathway.C4:
                if ratio < 0.5 or ratio > 2.0:
                    errors.append(f"C4 Jmax/Vcmax ratio {ratio:.2f} outside acceptable range [0.5, 2.0]")
        
        # Check dark respiration
        if pft.photosynthesis.rd25 is not None and pft.photosynthesis.vcmax25 > 0:
            rd_vcmax_ratio = pft.photosynthesis.rd25 / pft.photosynthesis.vcmax25
            if rd_vcmax_ratio > 0.1:
                errors.append(f"Rd25/Vcmax25 ratio {rd_vcmax_ratio:.3f} unusually high (> 0.1)")
        
        return errors
    
    def _validate_allocation_consistency(self, pft: PFTDefinition) -> List[str]:
        """Validate allocation parameter consistency."""
        errors = []
        
        # Skip validation if parameters are not set
        if (pft.allocation is None or 
            pft.allocation.f_leaf is None or 
            pft.allocation.f_stem is None or 
            pft.allocation.f_root is None):
            return errors
        
        # Check structural allocation fractions sum to 1 (storage handled separately)
        total = pft.allocation.f_leaf + pft.allocation.f_stem + pft.allocation.f_root
            
        if abs(total - 1.0) > 0.05:  # Allow 5% tolerance
            errors.append(f"Allocation fractions sum to {total:.3f}, should be 1.0")
        
        # Check reasonable allocation ranges
        if pft.allocation.f_leaf < 0.05 or pft.allocation.f_leaf > 0.8:
            errors.append(f"Leaf allocation fraction {pft.allocation.f_leaf:.2f} outside range [0.05, 0.8]")
        
        return errors
    
    def _validate_phenology_consistency(self, pft: PFTDefinition) -> List[str]:
        """Validate phenology parameter consistency."""
        errors = []
        
        # Skip validation if parameters are not set
        if (pft.phenology is None or 
            pft.phenology.temp_growth_opt is None or 
            pft.phenology.temp_growth_min is None or
            pft.phenology.temp_growth_max is None):
            return errors
        
        # Check temperature ranges
        if pft.phenology.temp_growth_opt <= pft.phenology.temp_growth_min:
            errors.append("Optimal growth temperature must be above minimum growth temperature")
        
        if pft.phenology.temp_growth_max <= pft.phenology.temp_growth_opt:
            errors.append("Maximum growth temperature must be above optimal growth temperature")
        
        return errors
    
    def _validate_stress_consistency(self, pft: PFTDefinition) -> List[str]:
        """Validate stress parameter consistency.""" 
        errors = []
        
        # Skip validation if parameters are not set
        if (pft.stress is None or pft.phenology is None or
            pft.stress.temp_stress_min is None or 
            pft.phenology.temp_growth_min is None):
            return errors
        
        # Check temperature stress vs phenology
        # Allow tiny float tolerance to avoid false positives from float32 rounding.
        if pft.stress.temp_stress_min > pft.phenology.temp_growth_min + 1.0e-4:
            errors.append("Temperature stress minimum should not be above growth minimum")
        
        return errors
    
    def _validate_parameter_ranges(self, pft: PFTDefinition) -> List[str]:
        """Validate parameter ranges against literature values."""
        errors = []
        
        # Photosynthesis parameter ranges (from literature)
        if pft.photosynthesis is not None and pft.photosynthesis.vcmax25 is not None:
            if not 10 <= pft.photosynthesis.vcmax25 <= 200:
                errors.append(f"Vcmax25 {pft.photosynthesis.vcmax25} outside typical range [10, 200]")
        
        if pft.photosynthesis is not None and pft.photosynthesis.jmax25 is not None:
            if not 20 <= pft.photosynthesis.jmax25 <= 400:
                errors.append(f"Jmax25 {pft.photosynthesis.jmax25} outside typical range [20, 400]")
        
        # Specific leaf area ranges
        if pft.phenology is not None and pft.phenology.specific_leaf_area is not None:
            if not 5 <= pft.phenology.specific_leaf_area <= 50:
                errors.append(f"SLA {pft.phenology.specific_leaf_area} outside typical range [5, 50]")
        
        return errors
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._cache_lock:
            self._parameter_cache.clear()
            self._validation_cache.clear()
        logger.info("PFT Registry caches cleared")
    
    def set_validation_strict(self, strict: bool) -> None:
        """
        Set validation strictness level.
        
        Parameters
        ----------
        strict : bool
            If True, raise errors on validation failures
            If False, log warnings but continue
        """
        self._validation_strict = strict
        logger.info(f"PFT validation strictness set to: {strict}")
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get registry information and statistics.
        
        Returns
        -------
        Dict[str, Any]
            Registry information dictionary
        """
        with self._registry_lock:
            return {
                'version': self._version,
                'num_pfts': len(self._pft_definitions),
                'pft_names': list(self._pft_definitions.keys()),
                'categories': self.list_pft_categories(),
                'parameter_files': [str(p) for p in self._parameter_files],
                'cache_enabled': self._cache_enabled,
                'validation_strict': self._validation_strict,
                'cache_size': {
                    'parameters': len(self._parameter_cache),
                    'validation': len(self._validation_cache)
                }
            }


# Convenience functions for easy access
def get_pft_registry() -> PFTRegistry:
    """Get the singleton PFT registry instance."""
    return PFTRegistry.get_instance()


def get_pft(pft_name: str) -> PFTDefinition:
    """Get PFT definition by name."""
    return get_pft_registry().get_pft(pft_name)


def get_pft_parameter(pft_name: str, parameter_path: str) -> Any:
    """Get specific parameter from PFT."""
    return get_pft_registry().get_pft_parameter(pft_name, parameter_path)


def list_available_pfts() -> List[str]:
    """List all available PFT names."""
    return get_pft_registry().list_pfts()


# Export main classes and functions
__all__ = [
    'PFTRegistry',
    'PFTRegistryError', 
    'PFTValidationError',
    'get_pft_registry',
    'get_pft',
    'get_pft_parameter', 
    'list_available_pfts'
]

