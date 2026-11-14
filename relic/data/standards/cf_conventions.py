"""
CF standard-name vocabulary and metadata utilities.

The bundled CF Standard Name Table v94 is the sole source for official entry,
alias, and canonical-unit lookups. Single-variable checks in this module do not
establish dataset-level CF conformance.

References:
    - CF Standard Name Table v94, 2026-06-09:
      https://cfconventions.org/Data/cf-standard-names/94/
    - CF Metadata Conventions 1.13, Section 3.3 and Appendix C:
      https://cfconventions.org/Data/cf-conventions/cf-conventions-1.13/
    - CF FAQ unit exceptions: https://cfconventions.org/faq.html
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
import logging
import re
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import xml.etree.ElementTree as ET

from cf_units import Unit
from torch import Tensor

from .dimensionality import dimensionality_from_unit, element_annotations_from_unit

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CFVariableMetadataValidationResult:
    """Scope-explicit result for one variable's standard-name/unit subset."""

    is_valid: bool
    issues: Tuple[str, ...]
    warnings: Tuple[str, ...] = ()
    membership_checked: bool = False
    unit_contract_defined: Optional[bool] = None
    unit_equivalence_checked: bool = False
    validation_scope: str = "single_variable_standard_name_and_units"
    dataset_conformance_checked: bool = False
    coordinates_checked: bool = False
    cell_methods_checked: bool = False

    def __iter__(self):
        """Preserve two-value unpacking while exposing the bounded scope."""
        yield self.is_valid
        yield list(self.issues)


# CF-1.8 section 4.4.1: the calendar attribute vocabulary.
CF_CALENDARS = frozenset({
    "standard", "gregorian", "proleptic_gregorian", "noleap", "365_day",
    "all_leap", "366_day", "360_day", "julian", "none",
})
_CF_AXES = frozenset({"X", "Y", "Z", "T"})
# Scope label recorded by CFConventions.validate_dataset_conformance: the bounded
# rule set listed in that method's docstring and nothing outside it.
CF_DATASET_CONFORMANCE_SCOPE = "declared_dataset_scope"
_TIME_UNITS_PATTERN = re.compile(
    r"^\s*(seconds?|minutes?|hours?|days?|s|min|h|d)\s+since\s+\S+", re.IGNORECASE
)
_CELL_METHODS_PATTERN = re.compile(
    r"^(\s*[A-Za-z_][A-Za-z0-9_]*(\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*\s*:\s*[A-Za-z_]+(\s*\([^()]*\))?\s*)+$"
)
# CF-1.13 Appendix E, Table E.1: official cell-method names.
_CF_CELL_METHOD_NAMES = frozenset({
    "point", "sum", "anomaly_wrt", "maximum", "maximum_absolute_value",
    "median", "mid_range", "minimum", "minimum_absolute_value", "mean",
    "mean_absolute_value", "mean_of_upper_decile", "mode", "range",
    "root_mean_square", "standard_deviation", "sum_of_squares", "variance",
})
# CF-1.8 sections 5 and 5.6: the grid fields a regridding target must declare --
# the grid_mapping variable name and the latitude/longitude coordinate variable
# names written to the regridded variable's ``coordinates``. Each must be an
# explicit non-empty string; none is inferred from tensor shape or scale labels.
CF_REGRID_TARGET_GRID_FIELDS: Tuple[str, ...] = ("grid_mapping", "lat_name", "lon_name")
# CF-1.8 sections 3.3 and 7.3: several variables may carry one standard_name when a
# declared qualifier tells them apart (air_temperature with cell_methods
# "time: maximum" or "time: minimum"). This module is the single owner of that
# identity rule for SSOTCatalog and VariableNameMapper; it imports no relic module at
# load time, so the mapper can use it while the ParameterManager is constructed.
CF_STANDARD_NAME_QUALIFIERS: Tuple[str, ...] = ("cell_methods",)


@dataclass(frozen=True)
class CFVariableIdentity:
    """Identity of one declared variable: standard_name plus normalized qualifiers.

    ``standard_name_declared`` is False when the standard_name defaulted to the
    variable name; such a default yields to a declared standard_name.
    """

    standard_name: str
    qualifiers: Tuple[Tuple[str, str], ...]
    standard_name_declared: bool

    @property
    def key(self) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        return (self.standard_name, self.qualifiers)


def cf_variable_identity(name: str, declaration: Mapping[str, Any]) -> CFVariableIdentity:
    """Return the CF identity of variable ``name`` from its declaration mapping.

    The standard_name defaults to ``name``; qualifiers (cell_methods) are
    whitespace-normalized. An empty standard_name or a non-string or empty
    qualifier raises ValueError naming the variable.
    """
    declared = "standard_name" in declaration
    standard_name = str(declaration.get("standard_name", name)).strip()
    if not standard_name:
        raise ValueError(f"Variable {name!r} declares an empty standard_name")
    qualifiers = []
    for key in CF_STANDARD_NAME_QUALIFIERS:
        value = declaration.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Variable {name!r} declares {key} {value!r}; a non-empty string is required"
            )
        qualifiers.append((key, " ".join(value.split())))
    return CFVariableIdentity(standard_name, tuple(qualifiers), declared)


def merge_cf_variable_identity(
    name: str,
    first: CFVariableIdentity,
    second: CFVariableIdentity,
) -> CFVariableIdentity:
    """Combine two declarations of one variable, or raise when they contradict.

    A declared standard_name wins over a defaulted one; two declared standard_names
    must be equal. Qualifiers are united; one qualifier declared with two values
    raises.
    """
    if (
        first.standard_name_declared
        and second.standard_name_declared
        and first.standard_name != second.standard_name
    ):
        raise ValueError(
            f"Variable {name!r} is declared with standard_name {first.standard_name!r} "
            f"and {second.standard_name!r}"
        )
    winner = second if second.standard_name_declared and not first.standard_name_declared else first
    qualifiers = dict(first.qualifiers)
    for key, value in second.qualifiers:
        if key in qualifiers and qualifiers[key] != value:
            raise ValueError(
                f"Variable {name!r} is declared with {key} {qualifiers[key]!r} and {value!r}"
            )
        qualifiers[key] = value
    return CFVariableIdentity(
        winner.standard_name,
        tuple((key, qualifiers[key]) for key in CF_STANDARD_NAME_QUALIFIERS if key in qualifiers),
        first.standard_name_declared or second.standard_name_declared,
    )


def register_cf_identity_holder(
    holders: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], str],
    name: str,
    identity: CFVariableIdentity,
) -> None:
    """Record ``name`` as the one variable holding ``identity``; a second name raises."""
    holder = holders.get(identity.key)
    if holder is not None and holder != name:
        qualifiers = dict(identity.qualifiers) or "no qualifier"
        raise ValueError(
            f"Variables {holder!r} and {name!r} both declare standard_name "
            f"{identity.standard_name!r} with {qualifiers}; one CF identity has one variable"
        )
    holders[identity.key] = name


# CF-1.8 section 7.3 cell methods a disaggregated field can carry, per axis.
# area: "sum" for a per-cell total (cell_total basis), "mean" for a per-cell mean
# (intensive or area_density basis). time: "sum" for a per-interval accumulation,
# "mean" for an interval mean or rate, "point" for instantaneous states
# (state_at_boundary). Any other axis has no declared disaggregation contract.
_DISAGGREGATION_CELL_METHODS: Dict[str, frozenset] = {
    "area": frozenset({"sum", "mean"}),
    "time": frozenset({"sum", "mean", "point"}),
}


@dataclass(frozen=True)
class CFDatasetConformanceResult:
    """Outcome of the bounded dataset-level CF conformance check.

    ``scope`` names exactly what was evaluated; ``is_valid`` is True only when
    every rule inside that scope passed. Rules outside the scope (grid
    mappings, discrete sampling geometries, compression by gathering, external
    variables, group hierarchies) are not evaluated and never reported as
    conforming.
    """

    is_valid: bool
    variables_checked: int
    coordinates_checked: int
    scope: str
    issues: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def __iter__(self):
        yield self.is_valid
        yield list(self.issues)


class CFConventions:
    """
    Sole runtime owner for the bundled CF v94 standard-name vocabulary.

    This class verifies official entry/alias membership, standard-name
    modifiers, canonical-unit availability, and physical unit equivalence.
    Dataset-level CF conformance remains outside this scope.
    """

    _TABLE_RESOURCE_NAME = "cf_standard_name_table.xml"
    _EXPECTED_TABLE_VERSION = "94"
    _STANDARD_NAME_MODIFIERS = frozenset(
        {
            "detection_minimum",
            "number_of_observations",
            "standard_error",
            "status_flag",
        }
    )
    _STANDARD_NAME_PATTERN = re.compile(
        r"^(?P<base>[a-z][a-z0-9_]*)"
        r"(?: +(?P<modifier>detection_minimum|number_of_observations|"
        r"standard_error|status_flag))?$"
    )
    _DB_UNIT_SPELLINGS = frozenset({"dB", "decibel", "db", "dbel"})
    # Element labels of RELIC units and the element words of CF names.
    _ELEMENT_WORDS = {
        "C": "carbon",
        "N": "nitrogen",
        "P": "phosphorus",
        "S": "sulfur",
        "H": "hydrogen",
        "K": "potassium",
    }
    # Words that may follow the substance of a CF "mass_<quantity>_of_<X>"
    # phrase; any other word (e.g. "dioxide") makes X part of a compound name.
    _SUBSTANCE_PHRASE_ENDS = frozenset({"in", "due", "at", "expressed", "out", "into"})
    _catalog_lock = RLock()
    _catalog_loaded = False
    _standard_name_aliases: Dict[str, Tuple[str, ...]] = {}
    _standard_units: Dict[str, Optional[str]] = {}
    _standard_descriptions: Dict[str, Optional[str]] = {}
    _table_metadata: Dict[str, Any] = {}

    @classmethod
    def _load_catalog(cls, force: bool = False) -> None:
        with cls._catalog_lock:
            if cls._catalog_loaded and not force:
                return

            try:
                payload = resources.files(__package__).joinpath(
                    cls._TABLE_RESOURCE_NAME
                ).read_bytes()
                root = ET.fromstring(payload)
            except (FileNotFoundError, OSError, ET.ParseError) as exc:
                raise RuntimeError(
                    "Bundled CF standard-name table could not be loaded"
                ) from exc

            if root.tag != "standard_name_table":
                raise RuntimeError("Invalid CF standard-name table root element")
            version = (root.findtext("version_number") or "").strip()
            conventions = (root.findtext("conventions") or "").strip()
            last_modified = (root.findtext("last_modified") or "").strip()
            if version != cls._EXPECTED_TABLE_VERSION:
                raise RuntimeError(
                    "Bundled CF standard-name table version must be 94"
                )
            if conventions != "CF-StandardNameTable-94" or not last_modified:
                raise RuntimeError("Bundled CF standard-name table metadata is invalid")

            units: Dict[str, Optional[str]] = {}
            descriptions: Dict[str, Optional[str]] = {}
            for entry in root.findall("entry"):
                identifier = entry.get("id")
                if (
                    not isinstance(identifier, str)
                    or not identifier
                    or identifier != identifier.strip()
                    or identifier in units
                ):
                    raise RuntimeError("Invalid or duplicate CF entry identifier")
                units_element = entry.find("canonical_units")
                if units_element is None:
                    raise RuntimeError(
                        f"CF entry '{identifier}' lacks canonical_units element"
                    )
                units[identifier] = (units_element.text or "").strip() or None
                description_element = entry.find("description")
                descriptions[identifier] = (
                    (description_element.text or "").strip() or None
                    if description_element is not None
                    else None
                )

            aliases: Dict[str, Tuple[str, ...]] = {}
            for alias in root.findall("alias"):
                identifier = alias.get("id")
                targets = tuple(
                    (target.text or "").strip()
                    for target in alias.findall("entry_id")
                )
                if (
                    not isinstance(identifier, str)
                    or not identifier
                    or identifier != identifier.strip()
                    or identifier in aliases
                ):
                    raise RuntimeError("Invalid or duplicate CF alias identifier")
                if len(targets) not in {1, 2} or any(not target for target in targets):
                    raise RuntimeError(
                        f"CF alias '{identifier}' has invalid target cardinality"
                    )
                if len(set(targets)) != len(targets):
                    raise RuntimeError(f"CF alias '{identifier}' repeats a target")
                missing_targets = [target for target in targets if target not in units]
                if missing_targets:
                    raise RuntimeError(
                        f"CF alias '{identifier}' targets missing entries: "
                        f"{missing_targets}"
                    )
                aliases[identifier] = targets

            cls._standard_units = units
            cls._standard_descriptions = descriptions
            cls._standard_name_aliases = aliases
            cls._table_metadata = {
                "version_number": version,
                "conventions": conventions,
                "last_modified": last_modified,
                "entry_count": len(units),
                "alias_count": len(aliases),
            }
            cls._catalog_loaded = True

    @classmethod
    def get_table_metadata(cls) -> Dict[str, Any]:
        """Return bounded metadata reported by the bundled official table."""
        cls._load_catalog()
        return dict(cls._table_metadata)

    @classmethod
    def _parse_standard_name(
        cls,
        value: Any,
    ) -> Tuple[str, Optional[str], str, Tuple[str, ...]]:
        cls._load_catalog()
        if not isinstance(value, str):
            raise TypeError("CF standard_name must be a string")
        if not value or value != value.strip():
            raise ValueError("CF standard_name must be an exact non-empty string")
        match = cls._STANDARD_NAME_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError("CF standard_name has invalid lexical or modifier syntax")
        base = match.group("base")
        modifier = match.group("modifier")
        if base in cls._standard_units:
            return base, modifier, "entry", (base,)
        if base in cls._standard_name_aliases:
            return base, modifier, "alias", cls._standard_name_aliases[base]
        raise ValueError(f"Unknown CF v94 standard name: {base}")

    @classmethod
    def get_standard_name(cls, variable_name: str) -> str:
        """Return one exact official v94 entry/alias identity and modifier."""
        base, modifier, _, _ = cls._parse_standard_name(variable_name)
        return f"{base} {modifier}" if modifier else base

    @classmethod
    def is_valid_standard_name(cls, variable_name: Any) -> bool:
        """Return whether a value is an official v94 entry/alias plus modifier."""
        try:
            cls._parse_standard_name(variable_name)
        except (TypeError, ValueError):
            return False
        return True

    @classmethod
    def get_standard_units(cls, standard_name: str) -> Optional[str]:
        """Return the official canonical unit when the artifact defines one."""
        base, modifier, kind, targets = cls._parse_standard_name(standard_name)
        if modifier == "number_of_observations":
            return "1"
        if modifier == "status_flag":
            return None
        if kind == "entry":
            return cls._standard_units[base]
        target_units = {cls._standard_units[target] for target in targets}
        if len(target_units) != 1:
            raise ValueError(
                f"CF alias '{base}' target canonical units are inconsistent"
            )
        return next(iter(target_units))

    @classmethod
    def get_standard_name_info(cls, variable_name: str) -> Dict[str, Any]:
        """Return scope-explicit official membership and unit information."""
        base, modifier, kind, targets = cls._parse_standard_name(variable_name)
        standard_name = f"{base} {modifier}" if modifier else base
        units = cls.get_standard_units(standard_name)
        description = None
        if kind == "entry":
            description = cls._standard_descriptions.get(base)
        elif len(targets) == 1:
            description = cls._standard_descriptions.get(targets[0])
        return {
            "standard_name": standard_name,
            "base_name": base,
            "modifier": modifier,
            "membership_kind": kind,
            "canonical_candidates": targets,
            "semantic_resolution": (
                "ambiguous" if kind == "alias" and len(targets) > 1 else "resolved"
            ),
            "canonical_units": units,
            "unit_contract_defined": units is not None,
            "description": description,
            "colliding_alias_targets": (
                cls._standard_name_aliases.get(base, ()) if kind == "entry" else ()
            ),
            "table_version": cls._table_metadata["version_number"],
            "dataset_conformance_checked": False,
        }

    @classmethod
    def are_standard_name_units_equivalent(
        cls,
        standard_name: str,
        actual_units: Optional[str],
    ) -> bool:
        """Check the official name-unit relation without converting data values."""
        _, modifier, _, _ = cls._parse_standard_name(standard_name)
        canonical_units = cls.get_standard_units(standard_name)
        if modifier in {"number_of_observations", "status_flag"}:
            return actual_units in {None, "1"}
        if canonical_units is None:
            raise ValueError(
                f"CF standard name '{standard_name}' unit contract is not defined"
            )
        if not isinstance(actual_units, str) or not actual_units:
            return False
        if actual_units != actual_units.strip():
            return False
        if canonical_units == "dB":
            return actual_units in cls._DB_UNIT_SPELLINGS
        elements = element_annotations_from_unit(actual_units)
        if elements:
            return cls._element_labelled_units_equivalent(
                standard_name, canonical_units, actual_units, elements
            )
        try:
            actual = Unit(actual_units)
            canonical = Unit(canonical_units)
        except (TypeError, ValueError):
            return False
        if actual.is_unknown() or canonical.is_unknown():
            return False
        return bool(actual.is_convertible(canonical))

    @classmethod
    def _element_basis_of_name(cls, name: str) -> Optional[str]:
        """Element word a CF name states as its mass or amount basis, or None.

        CF names carry the element in the name: "A_expressed_as_B" (B),
        "<element>_mass_..." (e.g. carbon_mass_transport_in_river_channel) or
        "..._mass_<quantity>_of_<element>" ending the substance phrase (e.g.
        vegetation_mass_content_of_carbon, not ..._of_carbon_dioxide_...).
        """
        tokens = name.split("_")
        for index in range(len(tokens) - 2):
            if tokens[index] == "expressed" and tokens[index + 1] == "as":
                return tokens[index + 2]
        element_words = set(cls._ELEMENT_WORDS.values())
        if len(tokens) > 1 and tokens[0] in element_words and tokens[1] == "mass":
            return tokens[0]
        for index in range(len(tokens) - 3):
            if tokens[index] == "mass" and tokens[index + 2] == "of":
                substance = tokens[index + 3]
                following = tokens[index + 4] if index + 4 < len(tokens) else None
                if following is None or following in cls._SUBSTANCE_PHRASE_ENDS:
                    return substance
        return None

    @classmethod
    def element_basis_of_standard_name(cls, standard_name: str) -> Optional[str]:
        """Element symbol (for example 'C') a CF name states as its mass or amount basis.

        The element word comes from _element_basis_of_name for every target of
        the name (an entry, or each target of an alias); it is mapped to the
        RELIC element label of _ELEMENT_WORDS. None when the name states no
        element basis (or a basis without a RELIC element label); targets that
        state different bases raise ValueError.
        """
        _, _, _, targets = cls._parse_standard_name(standard_name)
        words = {cls._element_basis_of_name(target) for target in targets}
        if len(words) != 1:
            raise ValueError(
                f"CF standard name '{standard_name}' has targets with different element bases: {sorted(map(str, words))}"
            )
        symbols = {word: symbol for symbol, word in cls._ELEMENT_WORDS.items()}
        return symbols.get(words.pop())

    @classmethod
    def _element_labelled_units_equivalent(
        cls,
        standard_name: str,
        canonical_units: str,
        actual_units: str,
        elements: Tuple[str, ...],
    ) -> bool:
        """Compatibility of an element-labelled RELIC unit with a CF name.

        References:
            CF Standard Name guidelines ("expressed_as" construction; the
            canonical units exclude the element); BIPM SI Brochure (2019)
            for the dimension vector; RELIC dimensionality derivation
            (relic.data.standards.dimensionality).
        Mathematical expression:
            compatible = dim(u) == dim(U(s)) and elem(u) == basis(s) for every
            target entry of s, where dim is the element-aware SI vector.
        Parameter meanings:
            u the declared unit (e.g. kg C m-2 s-1), U(s) the canonical CF
            units of the name s (kg m-2 s-1), elem(u) the element label of u,
            basis(s) the element the name states (expressed_as_carbon).
        Units:
            dimension exponents of M, L, T, Theta, Q, N, J (dimensionless).
        Domain/constraints:
            exactly one element label; a unit or canonical unit without an SI
            derivation, several labels, or a name without the same element
            basis is incompatible (no guess).
        Sign convention:
            not applicable.
        """
        if len(set(elements)) != 1:
            return False
        element_word = cls._ELEMENT_WORDS.get(elements[0])
        actual_dimension = dimensionality_from_unit(actual_units)
        canonical_dimension = dimensionality_from_unit(canonical_units)
        if element_word is None or actual_dimension is None or canonical_dimension is None:
            return False
        if not actual_dimension.is_compatible(canonical_dimension):
            return False
        _, _, _, targets = cls._parse_standard_name(standard_name)
        return all(cls._element_basis_of_name(target) == element_word for target in targets)

    @staticmethod
    def update_cell_methods(
        current_methods: str,
        dimension: str,
        operation: str,
        comment: Optional[str] = None
    ) -> str:
        """
        Update CF-1.8 cell_methods string.
        
        Parameters:
            current_methods : str
                Current cell_methods string
            dimension : str
                Dimension name (e.g., 'time', 'lat', 'lon')
            operation : str
                Statistical operation (e.g., 'mean', 'sum', 'maximum')
            comment : Optional[str]
                Optional comment (e.g., 'where land')
                
        Returns:
            str: Updated cell_methods string
            
        Example:
            >>> update_cell_methods("", "time", "mean")
            "time: mean"
            >>> update_cell_methods("time: mean", "area", "mean", "where land")
            "time: mean area: mean (where land)"
        """
        new_method = f"{dimension}: {operation}"
        if comment:
            new_method += f" ({comment})"
        
        if current_methods:
            # Check if dimension already exists and replace it
            if f"{dimension}:" in current_methods:
                # Parse existing methods and replace the dimension
                parts = current_methods.split()
                updated_parts = []
                i = 0
                while i < len(parts):
                    if parts[i] == f"{dimension}:":
                        # Replace this dimension's method
                        updated_parts.extend(new_method.split())
                        # Skip old method and any parenthetical content
                        i += 1
                        while i < len(parts) and (
                            not parts[i].endswith(':') or parts[i].startswith('(')
                        ):
                            i += 1
                        continue
                    else:
                        updated_parts.append(parts[i])
                    i += 1
                return " ".join(updated_parts)
            else:
                return f"{current_methods} {new_method}"
        else:
            return new_method
    
    @staticmethod
    def validate_units(value: Union[Tensor, float], units: str) -> bool:
        """
        Check whether a unit string is recognized by CF/UDUNITS syntax.
        
        Parameters:
            value : Union[Tensor, float]
                Data value(s) to check
            units : str
                Units string to validate
                
        Returns:
            bool: True only for a recognized unit string
        """
        if not isinstance(units, str) or not units or units != units.strip():
            return False
        if units in CFConventions._DB_UNIT_SPELLINGS:
            return True
        try:
            parsed = Unit(units)
        except (TypeError, ValueError):
            return False
        return not parsed.is_unknown()
    
    @staticmethod
    def create_bounds_variable_name(coord_name: str) -> str:
        """
        Create CF-1.8 compliant bounds variable name.
        
        Parameters:
            coord_name : str
                Coordinate variable name
                
        Returns:
            str: Bounds variable name
        """
        return f"{coord_name}_bounds"
    
    @staticmethod
    def create_global_attributes(
        title: Optional[str] = None,
        institution: Optional[str] = None,
        source: Optional[str] = None,
        history: Optional[str] = None,
        references: Optional[str] = None,
        comment: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create neutral descriptive global attributes.
        
        Parameters:
            title : Optional[str]
                Dataset title
            institution : Optional[str]
                Institution name
            source : Optional[str]
                Method of production
            history : Optional[str]
                Processing history
            references : Optional[str]
                References
            comment : Optional[str]
                Additional comments
                
        Returns:
            Dict[str, str]: Attributes that do not claim dataset CF conformance
        """
        attrs = {
            'creation_date': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }
        
        if title:
            attrs['title'] = title
        if institution:
            attrs['institution'] = institution
        if source:
            attrs['source'] = source
        if history:
            attrs['history'] = history
        if references:
            attrs['references'] = references
        if comment:
            attrs['comment'] = comment
            
        return attrs
    
    @staticmethod
    def sync_variable_attributes(
        data: Tensor,
        metadata: Dict[str, Any],
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Propagate validated variable attributes after an operation.
        
        Parameters:
            data : Tensor
                Processed data tensor
            metadata : Dict[str, Any]
                Current metadata dictionary
            operation : Optional[str]
                Operation performed (e.g., 'temporal_aggregation', 'spatial_regrid')
                
        Returns:
            Dict[str, Any]: Updated metadata without identity or unit inference
        """
        validation = CFConventions.validate_cf_compliance(metadata)
        undefined_unit_only = (
            validation.membership_checked
            and validation.unit_contract_defined is False
            and not validation.unit_equivalence_checked
            and validation.issues
            and all(
                "unit contract is not defined" in issue
                for issue in validation.issues
            )
        )
        if not validation.is_valid and not undefined_unit_only:
            raise ValueError(
                "CF standard-name metadata invalid: "
                + "; ".join(validation.issues)
            )

        updated_metadata = metadata.copy()
        
        # Update cell_methods based on operation
        if operation:
            current_methods = metadata.get('cell_methods', '')
            
            if operation == 'temporal_aggregation':
                method = metadata.get('aggregation_method', 'mean')
                updated_metadata['cell_methods'] = CFConventions.update_cell_methods(
                    current_methods, 'time', method
                )
        
        # Add coordinate references if present
        if 'coordinates' in metadata:
            updated_metadata['coordinates'] = metadata['coordinates']
        
        # Add bounds references if present
        for coord in ['time', 'lat', 'lon', 'x', 'y']:
            if f'{coord}_bounds' in metadata:
                bounds_name = CFConventions.create_bounds_variable_name(coord)
                if 'bounds' not in updated_metadata:
                    updated_metadata['bounds'] = {}
                updated_metadata['bounds'][coord] = bounds_name
        
        # Add shape information
        updated_metadata['shape'] = list(data.shape)
        
        # Add data range for quality control
        if data.numel() > 0:
            updated_metadata['actual_range'] = [
                float(data.min().item()),
                float(data.max().item())
            ]
        
        return updated_metadata
    
    @classmethod
    def validate_cf_compliance(
        cls,
        metadata: Dict[str, Any],
    ) -> CFVariableMetadataValidationResult:
        """
        Validate one variable's CF standard-name and unit metadata subset.

        References:
            CF 1.13 Section 3.3 and Appendix C; CF Standard Name Table v94.

        Mathematical expression:
            member(s) and compatible(u, U(s)), where compatibility is tested
            with UDUNITS convertibility when the artifact defines U(s).

        Parameter meanings:
            metadata.standard_name: Optional official entry/alias plus modifier.
            metadata.units: Unit string for the variable values.
            metadata.units_metadata: Optional temperature scale/difference label.

        Units:
            Canonical units come only from the bundled official artifact.

        Domain and constraints:
            This method does not validate dataset dimensions, coordinates,
            bounds, grid mappings, cell-method semantics, or global attributes.

        Sign convention:
            Not applicable.
        """
        if not isinstance(metadata, dict):
            return CFVariableMetadataValidationResult(
                is_valid=False,
                issues=("metadata must be a mapping",),
            )

        issues: List[str] = []
        warnings: List[str] = []
        membership_checked = False
        unit_contract_defined: Optional[bool] = None
        unit_equivalence_checked = False

        if "standard_name" not in metadata:
            if "units" in metadata and not cls.validate_units(None, metadata["units"]):
                issues.append(f"Unrecognized UDUNITS string: {metadata['units']}")
            return CFVariableMetadataValidationResult(
                is_valid=not issues,
                issues=tuple(issues),
                membership_checked=False,
                unit_contract_defined=None,
                unit_equivalence_checked=False,
            )

        standard_name = metadata.get("standard_name")
        try:
            base, modifier, _, _ = cls._parse_standard_name(standard_name)
        except (TypeError, ValueError) as exc:
            issues.append(str(exc))
            if "units" in metadata and not cls.validate_units(None, metadata["units"]):
                issues.append(f"Unrecognized UDUNITS string: {metadata['units']}")
            return CFVariableMetadataValidationResult(
                is_valid=False,
                issues=tuple(issues),
                membership_checked=False,
                unit_contract_defined=None,
                unit_equivalence_checked=False,
            )

        membership_checked = True
        canonical_units = cls.get_standard_units(standard_name)
        unit_contract_defined = canonical_units is not None
        actual_units = metadata.get("units")

        if modifier in {"number_of_observations", "status_flag"}:
            unit_equivalence_checked = True
            if not cls.are_standard_name_units_equivalent(
                standard_name,
                actual_units,
            ):
                issues.append(
                    f"Units {actual_units!r} violate modifier '{modifier}' contract"
                )
        elif canonical_units is None:
            issues.append(
                f"CF standard name '{standard_name}' unit contract is not defined"
            )
        elif "units" not in metadata:
            issues.append(
                f"Missing units for CF standard name '{standard_name}'"
            )
        else:
            unit_equivalence_checked = True
            if not cls.are_standard_name_units_equivalent(
                standard_name,
                actual_units,
            ):
                issues.append(
                    f"Units {actual_units!r} are not physically compatible with "
                    f"canonical units {canonical_units!r} for '{standard_name}'"
                )

        if modifier == "standard_error" and canonical_units is not None:
            is_temperature = False
            if canonical_units != "dB":
                try:
                    is_temperature = Unit(canonical_units).is_convertible(Unit("K"))
                except (TypeError, ValueError):
                    is_temperature = False
            if is_temperature:
                units_metadata = metadata.get("units_metadata")
                if units_metadata is None:
                    warnings.append(
                        "temperature standard_error lacks explicit "
                        "units_metadata='temperature: difference'"
                    )
                elif units_metadata != "temperature: difference":
                    issues.append(
                        "temperature standard_error units_metadata must be "
                        "'temperature: difference'"
                    )

        return CFVariableMetadataValidationResult(
            is_valid=not issues,
            issues=tuple(issues),
            warnings=tuple(warnings),
            membership_checked=membership_checked,
            unit_contract_defined=unit_contract_defined,
            unit_equivalence_checked=unit_equivalence_checked,
        )


    @classmethod
    def validate_dataset_conformance(cls, dataset: Any) -> CFDatasetConformanceResult:
        """Validate one whole dataset against the declared CF scope.

        References:
            CF-1.8 sections 2.5.1 (missing data), 2.6.1 (Conventions), 3.1
            (units), 3.3 (standard names), 4 (coordinate types and axis),
            4.4 (time coordinate and calendar), 5 (coordinate systems), 7.1
            (cell boundaries), 7.3 (cell methods); CF Standard Name Table v94.
        Mathematical expression:
            valid = AND over rules r in the declared scope of r(dataset).
        Parameter meanings:
            dataset: an xarray Dataset (attrs, variables, coords, encoding).
        Units:
            Variable units must be recognized UDUNITS strings; standard-name
            units must be convertible to the canonical units of the artifact.
        Domain and constraints:
            Only the rules listed above are evaluated; decoded time coordinates
            expose their units and calendar through ``encoding``, which is
            consulted after ``attrs``.
        Sign convention:
            Not applicable.
        """
        issues: List[str] = []
        warnings: List[str] = []
        attrs = dict(getattr(dataset, "attrs", {}) or {})
        conventions = str(attrs.get("Conventions", ""))
        if "Conventions" not in attrs:
            issues.append("global attribute 'Conventions' is missing")
        elif not re.search(r"CF-\d+\.\d+", conventions):
            issues.append(f"global attribute 'Conventions' does not declare a CF version: {conventions!r}")

        variables = getattr(dataset, "variables", {})
        coord_names = set(getattr(dataset, "coords", {}).keys())
        dims = dict(getattr(dataset, "sizes", None) or getattr(dataset, "dims", {}) or {})
        # xarray keeps decoded-away attributes (bounds, units of time) in
        # ``encoding``; both places are consulted.
        bounds_variables = set()
        for name in variables:
            candidate = variables[name].attrs.get("bounds", getattr(variables[name], "encoding", {}).get("bounds"))
            if candidate is not None:
                bounds_variables.add(str(candidate))
        variables_checked = 0
        coordinates_checked = 0

        def _attr(name: str, key: str) -> Any:
            variable = variables[name]
            if key in variable.attrs:
                return variable.attrs[key]
            return getattr(variable, "encoding", {}).get(key)

        for name in variables:
            variable = variables[name]
            is_coordinate = name in coord_names
            if is_coordinate:
                coordinates_checked += 1
            else:
                variables_checked += 1
            kind = getattr(getattr(variable, "dtype", None), "kind", "")
            is_flag = "flag_values" in variable.attrs or "flag_masks" in variable.attrs
            units = _attr(name, "units")
            if kind in {"U", "S", "O", "b"}:
                pass
            elif units is None:
                # CF-1.8 section 7.1: a boundary variable inherits units from
                # its parent coordinate and need not repeat them.
                if not is_flag and name not in bounds_variables:
                    issues.append(f"variable '{name}' has no units attribute")
            elif not cls.validate_units(0.0, str(units)) and not _TIME_UNITS_PATTERN.match(str(units)):
                issues.append(f"variable '{name}' units {units!r} are not a recognized UDUNITS string")
            if "standard_name" in variable.attrs:
                metadata = {"standard_name": variable.attrs["standard_name"]}
                if units is not None:
                    metadata["units"] = str(units)
                if "units_metadata" in variable.attrs:
                    metadata["units_metadata"] = variable.attrs["units_metadata"]
                if _TIME_UNITS_PATTERN.match(str(units or "")):
                    if not cls.is_valid_standard_name(str(variable.attrs["standard_name"]).split(" ")[0]):
                        issues.append(f"variable '{name}' standard_name is not an official entry")
                else:
                    result = cls.validate_cf_compliance(metadata)
                    issues.extend(f"variable '{name}': {issue}" for issue in result.issues)
                    warnings.extend(f"variable '{name}': {warning}" for warning in result.warnings)
            axis = variable.attrs.get("axis")
            if axis is not None and str(axis) not in _CF_AXES:
                issues.append(f"variable '{name}' axis {axis!r} is not one of X, Y, Z, T")
            calendar = _attr(name, "calendar")
            is_time = _TIME_UNITS_PATTERN.match(str(units or "")) is not None or str(axis) == "T"
            if is_time and units is not None and not _TIME_UNITS_PATTERN.match(str(units)):
                issues.append(f"time coordinate '{name}' units {units!r} are not '<unit> since <reference>'")
            if calendar is not None and str(calendar).lower() not in CF_CALENDARS:
                issues.append(f"variable '{name}' calendar {calendar!r} is not a CF calendar")
            if is_time and calendar is None:
                warnings.append(f"time coordinate '{name}' has no calendar attribute (CF default 'standard' assumed)")
            bounds = _attr(name, "bounds")
            if bounds is not None:
                if bounds not in variables:
                    issues.append(f"variable '{name}' bounds attribute references missing variable {bounds!r}")
                else:
                    expected = tuple(variable.shape) + (2,)
                    if tuple(variables[bounds].shape) != expected:
                        issues.append(
                            f"bounds variable {bounds!r} has shape {tuple(variables[bounds].shape)}, expected {expected}"
                        )
            coordinates = _attr(name, "coordinates")
            if coordinates is not None:
                for token in str(coordinates).split():
                    if token not in variables:
                        issues.append(f"variable '{name}' coordinates attribute references missing variable {token!r}")
            fill = _attr(name, "_FillValue")
            missing = variable.attrs.get("missing_value")
            if fill is not None and missing is not None and fill != missing:
                issues.append(f"variable '{name}' _FillValue and missing_value differ")
            if is_coordinate and variable.attrs.get("_FillValue") is not None:
                issues.append(f"coordinate variable '{name}' must not carry _FillValue")
            cell_methods = variable.attrs.get("cell_methods")
            if cell_methods is not None:
                # References: CF-1.13 sections 7.3, 7.3.2 and Appendix E.
                # Mathematical expression: valid = syntax AND known_axes AND known_methods.
                # Parameter meanings: axes and methods are top-level declaration tokens;
                # parenthesized interval/comment text describes their sampling support.
                # Units: metadata strings only; no values or units are converted here.
                # Domain/constraints: validate the whole supported grammar before
                # removing balanced, non-nested annotations from token inspection.
                # Sign convention: no flux, source, sink or sign transformation.
                text = str(cell_methods)
                if not _CELL_METHODS_PATTERN.fullmatch(text):
                    issues.append(f"variable '{name}' cell_methods {text!r} is not 'name: method' syntax")
                else:
                    declarations = re.sub(r"\([^()]*\)", " ", text)
                    for axes, method in re.findall(
                        r"([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)"
                        r"\s*:\s*([A-Za-z_]+)", declarations
                    ):
                        for token in (axis.strip() for axis in axes.split(",")):
                            if token not in dims and token not in variables and token not in {"area", "time"}:
                                issues.append(f"variable '{name}' cell_methods names unknown dimension {token!r}")
                        if method not in _CF_CELL_METHOD_NAMES:
                            issues.append(f"variable '{name}' cell_methods names unknown method {method!r}")
        return CFDatasetConformanceResult(
            is_valid=not issues,
            issues=tuple(issues),
            warnings=tuple(warnings),
            variables_checked=variables_checked,
            coordinates_checked=coordinates_checked,
            scope=CF_DATASET_CONFORMANCE_SCOPE,
        )


class CFAttributeSynchronizer:
    """
    Mechanically propagate explicit CF metadata through processing steps.

    It does not infer quantity identity, units, variable role, grid mapping,
    coordinates, or dataset-level conformance.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize CF attribute synchronizer.
        
        Parameters:
            strict_mode : bool
                If True, raise errors for the bounded metadata subset
        """
        self.strict_mode = strict_mode
        self.conventions = CFConventions()
    
    def sync_after_aggregation(
        self,
        data: Tensor,
        metadata: Dict[str, Any],
        aggregation_method: str,
        dimension: str,
        *,
        source_label: str,
        target_label: str,
    ) -> Dict[str, Any]:
        """
        Synchronize attributes after aggregation operation.

        Parameters:
            data : Tensor
                Aggregated data
            metadata : Dict[str, Any]
                Original metadata
            aggregation_method : str
                CF cell method written for ``dimension`` (e.g. mean, sum)
            dimension : str
                Dimension aggregated over; required, it names the
                ``cell_methods`` entry that is written
            source_label, target_label : str
                Explicit non-empty labels of the fine source scale and the
                coarse target scale, recorded in ``history``

        Returns:
            Dict[str, Any]: Updated metadata
        """
        return self._sync_after_rescaling(
            data,
            metadata,
            direction="aggregation",
            method=aggregation_method,
            dimension=dimension,
            source_label=source_label,
            target_label=target_label,
        )

    def sync_after_disaggregation(
        self,
        data: Tensor,
        metadata: Dict[str, Any],
        disaggregation_method: str,
        dimension: str,
        *,
        source_label: str,
        target_label: str,
    ) -> Dict[str, Any]:
        """
        Synchronize attributes after disaggregation to a finer scale.

        Parameters:
            data : Tensor
                Disaggregated data
            metadata : Dict[str, Any]
                Original metadata
            disaggregation_method : str
                ``"sum"`` when each fine cell carries a total over its own
                cell (cell_total basis, area-proportional split) or ``"mean"``
                when each fine cell carries a cell mean (intensive or
                area_density basis); for ``dimension="time"`` also ``"point"``
                for instantaneous states. Other methods and other dimensions
                have no declared contract and raise ``ValueError``
            dimension : str
                Dimension disaggregated over (``"area"`` or ``"time"``);
                required, it names the ``cell_methods`` entry that is written
            source_label, target_label : str
                Explicit non-empty labels of the coarse source scale and the
                fine target scale, recorded in ``history``

        Returns:
            Dict[str, Any]: Updated metadata
        """
        allowed = _DISAGGREGATION_CELL_METHODS.get(dimension)
        if allowed is None:
            raise ValueError(
                f"no disaggregation cell-method contract for dimension {dimension!r}; "
                f"declared: {sorted(_DISAGGREGATION_CELL_METHODS)}"
            )
        if disaggregation_method not in allowed:
            raise ValueError(
                f"disaggregation_method for {dimension!r} must be one of "
                f"{sorted(allowed)}, got {disaggregation_method!r}"
            )
        return self._sync_after_rescaling(
            data,
            metadata,
            direction="disaggregation",
            method=disaggregation_method,
            dimension=dimension,
            source_label=source_label,
            target_label=target_label,
        )

    def _sync_after_rescaling(
        self,
        data: Tensor,
        metadata: Dict[str, Any],
        *,
        direction: str,
        method: str,
        dimension: str,
        source_label: str,
        target_label: str,
    ) -> Dict[str, Any]:
        """Shared aggregation/disaggregation path: attributes, cell_methods, history, strict check.

        Metadata contract (CF-1.8 Section 7.3, cell_methods):
            cell_methods gains or replaces the entry ``"<dimension>: <method>"``
            describing the value of each output cell, and history gains
            ``"<Aggregated|Disaggregated> <dimension> from <source_label> to
            <target_label> (<dimension>: <method>)"``. No value is computed or
            converted here; units and standard_name pass through unchanged.
        """
        for name, label in (("source_label", source_label), ("target_label", target_label)):
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{name} must be a non-empty scale label, got {label!r}")

        updated = self.conventions.sync_variable_attributes(
            data, metadata, f'{dimension}_{direction}'
        )
        updated['cell_methods'] = self.conventions.update_cell_methods(
            metadata.get('cell_methods', ''), dimension, method
        )
        verbs = {"aggregation": "Aggregated", "disaggregation": "Disaggregated"}
        if direction not in verbs:
            raise ValueError(f"unknown rescaling direction {direction!r}")
        verb = verbs[direction]
        updated['history'] = self._append_history(
            metadata,
            f"{verb} {dimension} from {source_label} to {target_label} ({dimension}: {method})",
        )
        self._validate_strict(updated)
        return updated

    @staticmethod
    def _append_history(metadata: Dict[str, Any], entry: str) -> str:
        """Return ``metadata['history']`` with ``entry`` appended as ``"; "``-separated text."""
        history = metadata.get('history', '')
        return f"{history}; {entry}" if history else entry

    def _validate_strict(self, updated: Dict[str, Any]) -> None:
        """Raise on any bounded CF standard-name/unit issue when strict mode is enabled."""
        if self.strict_mode:
            is_valid, issues = self.conventions.validate_cf_compliance(updated)
            if not is_valid:
                raise ValueError(f"CF standard-name metadata issues: {issues}")

    def sync_after_regridding(
        self,
        data: Tensor,
        metadata: Dict[str, Any],
        regrid_method: str,
        source_grid: Dict[str, Any],
        target_grid: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synchronize attributes after spatial regridding.

        Parameters:
            data : Tensor
                Regridded data
            metadata : Dict[str, Any]
                Original metadata
            regrid_method : str
                Method used (conservative, bilinear, nearest)
            source_grid : Dict[str, Any]
                Source grid information
            target_grid : Dict[str, Any]
                Target grid information

        Returns:
            Dict[str, Any]: Updated metadata
        """
        missing_target = [
            name
            for name in CF_REGRID_TARGET_GRID_FIELDS
            if not isinstance(target_grid.get(name), str) or not target_grid[name].strip()
        ]
        if "shape" not in target_grid:
            missing_target.append("shape")
        if missing_target:
            raise ValueError(
                "explicit target_grid metadata is required: "
                + ", ".join(missing_target)
            )
        if "shape" not in source_grid:
            raise ValueError("explicit source_grid shape is required")

        updated = self.conventions.sync_variable_attributes(
            data, metadata, 'spatial_regrid'
        )

        # Update grid information
        updated['grid_mapping'] = target_grid['grid_mapping']
        updated['coordinates'] = f"{target_grid['lat_name']} {target_grid['lon_name']}"

        # Add regridding information to history
        updated['history'] = self._append_history(
            metadata,
            f"Regridded from {source_grid['shape']} to "
            f"{target_grid['shape']} using {regrid_method} method",
        )
        self._validate_strict(updated)
        return updated

    def infer_variable_type(self, metadata: Dict[str, Any]) -> str:
        """
        Return one explicitly declared spatial quantity role.

        Parameters:
            metadata : Dict[str, Any]
                Variable metadata containing ``variable_type``

        Returns:
            str: Variable type ('extensive', 'intensive', or 'categorical')
        """
        variable_type = metadata.get("variable_type") if isinstance(metadata, dict) else None
        allowed = {"extensive", "intensive", "categorical"}
        if not isinstance(variable_type, str) or variable_type not in allowed:
            raise ValueError(
                "An explicit variable_type in "
                "{'extensive', 'intensive', 'categorical'} is required"
            )
        return variable_type
    
    def sync_after_temporal_aggregation(
        self,
        data: Tensor,
        metadata: Dict[str, Any],
        time_coord,  # TimeCoordinate object
        operation: str,
        method: str,
        coverage_stats: Dict[str, Any],
        *,
        source_label: str,
        target_label: str,
    ) -> Dict[str, Any]:
        """
        Synchronize a variable's attributes after temporal aggregation or disaggregation.

        The variable keeps its own standard_name, units and long_name (a rate
        stays a rate, an interval accumulation keeps its accumulated unit).
        cell_methods gains ``"time: <method>"`` and history one entry through
        the same path as :meth:`sync_after_aggregation` /
        :meth:`sync_after_disaggregation`; ``time_coordinate`` carries the CF
        attributes of the result time coordinate (units, calendar, axis and
        ``bounds`` when the coordinate has bounds); coverage statistics and the
        quality classification come from explicit ``coverage_stats``.

        Parameters:
            data : Tensor
                Rescaled data
            metadata : Dict[str, Any]
                Original variable metadata
            time_coord : TimeCoordinate
                Result time coordinate
            operation : str
                ``"aggregation"`` or ``"disaggregation"``
            method : str
                CF cell method written for the time axis (e.g. mean, sum)
            coverage_stats : Dict[str, Any]
                Required coverage statistics: coverage_fraction, or valid_points
                and total_points. Coverage is never assumed.
            source_label, target_label : str
                Explicit non-empty labels of the source and target time scales

        Returns:
            Dict[str, Any]: Synchronized variable metadata
        """
        from ...coupling.scales.temporal.cf_functions import (
            create_temporal_rescaling_metadata,
            validate_cf_temporal_compliance,
        )

        if operation == "aggregation":
            rescale = self.sync_after_aggregation
        elif operation == "disaggregation":
            rescale = self.sync_after_disaggregation
        else:
            raise ValueError(
                "operation must be 'aggregation' or 'disaggregation', "
                f"got {operation!r}"
            )
        updated = rescale(
            data,
            metadata,
            method,
            'time',
            source_label=source_label,
            target_label=target_label,
        )
        temporal = create_temporal_rescaling_metadata(
            time_coord, coverage_stats=coverage_stats
        )
        if self.strict_mode:
            issues = validate_cf_temporal_compliance(temporal["time_coordinate"])
            if issues:
                raise ValueError(f"CF-1.8 temporal compliance issues: {issues}")
        updated.update(temporal)
        return updated


# Module-level wrappers for unified CF entry points.
# Lazy resolution avoids import-time circular dependencies while keeping strict
# new-architecture behavior (no legacy fallback paths).
def create_cf_time_attributes(*args, **kwargs):
    from ...coupling.scales.temporal.cf_functions import create_cf_time_attributes as _impl
    return _impl(*args, **kwargs)


def update_cell_methods_temporal(*args, **kwargs):
    from ...coupling.scales.temporal.cf_functions import update_cell_methods as _impl
    return _impl(*args, **kwargs)


def compute_time_coverage(*args, **kwargs):
    from ...coupling.scales.temporal.cf_functions import compute_time_coverage as _impl
    return _impl(*args, **kwargs)


def create_quality_metadata(*args, **kwargs):
    from ...coupling.scales.temporal.cf_functions import create_quality_metadata as _impl
    return _impl(*args, **kwargs)


def validate_cf_temporal_compliance(*args, **kwargs):
    from ...coupling.scales.temporal.cf_functions import validate_cf_temporal_compliance as _impl
    return _impl(*args, **kwargs)


__all__ = [
    'CF_DATASET_CONFORMANCE_SCOPE',
    'CF_REGRID_TARGET_GRID_FIELDS',
    'CF_STANDARD_NAME_QUALIFIERS',
    'CFVariableIdentity',
    'cf_variable_identity',
    'merge_cf_variable_identity',
    'register_cf_identity_holder',
    'CFConventions',
    'CFAttributeSynchronizer',
    'create_cf_time_attributes',
    'update_cell_methods_temporal',
    'compute_time_coverage',
    'create_quality_metadata',
    'validate_cf_temporal_compliance',
]


def conserve_check(dataset: Any,
                   tolerance: Optional[float] = None,
                   check_variables: Optional[List[str]] = None,
                   verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Check conservation of mass/energy in xarray Dataset.
    
    This function checks only variables with an explicit extensive role.
    CF standard names and units do not determine that representation contract.
    
    Mathematical Expression
    -----------------------
    For extensive variables with area/volume integration:
        sum(data * weights) = constant (within tolerance)
    
    For flux variables with time integration:
        integral(flux * dt) = delta(stock)
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to check for conservation
    tolerance : float, optional
        Relative tolerance for the conservation check. When omitted, the
        canonical mode-less scale tolerance
        ``coupling.scales.defaults.conservation_tolerance`` is read from the
        parameter database; there is no literal default.
    check_variables : List[str], optional
        Specific variables to check. If None, checks all extensive variables
    verbose : bool, optional
        Print detailed conservation diagnostics
        
    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (is_conserved, diagnostics_dict) where diagnostics contains:
        - 'passed': bool, overall conservation status
        - 'variable_results': Dict[str, Dict], per-variable results
        - 'violations': List[str], list of conservation violations
        - 'summary': Dict, summary statistics
        
    References
    ----------
    - CF Conventions 1.8, Section 7.3: Cell Methods
    - Conservation laws in biogeochemical modeling
    - Numerical precision in scientific computing
        
    Examples
    --------
    >>> is_conserved, diagnostics = conserve_check(dataset, tolerance=1e-5)
    >>> if not is_conserved:
    ...     print("Conservation violations:", diagnostics['violations'])
    """
    import numpy as np

    if tolerance is None:
        # Scale precision policy: mode-less scale conservation checks read exactly
        # one key, coupling.scales.defaults.conservation_tolerance (no fallback).
        from relic.parameters import get_global_manager

        tolerance = float(
            get_global_manager().get("coupling.scales.defaults.conservation_tolerance", required=True)
        )

    # Initialize results
    diagnostics = {
        'passed': True,
        'variable_results': {},
        'violations': [],
        'summary': {
            'total_variables_checked': 0,
            'variables_passed': 0,
            'variables_failed': 0,
            'max_violation': 0.0
        }
    }
    
    # Determine which variables to check
    if check_variables is None:
        # Check only explicitly declared extensive variables.
        check_variables = []
        for var_name in dataset.data_vars:
            var = dataset[var_name]
            variable_type = var.attrs.get("variable_type")
            if variable_type not in {"extensive", "intensive", "categorical"}:
                diagnostics["passed"] = False
                diagnostics["violations"].append(
                    f"{var_name}: explicit variable_type is required before "
                    "conservation selection"
                )
                diagnostics["variable_results"][var_name] = {
                    "checked": False,
                    "conserved": False,
                    "error": "explicit variable_type is required",
                }
                continue
            if variable_type == 'extensive':
                check_variables.append(var_name)
    
    if not check_variables:
        # No extensive variables to check
        diagnostics['summary']['message'] = 'No extensive variables found to check'
        return diagnostics["passed"], diagnostics
    
    # Check each variable
    for var_name in check_variables:
        if var_name not in dataset.data_vars:
            # A requested variable the dataset does not hold fails the check.
            diagnostics['passed'] = False
            diagnostics['violations'].append(f"Variable '{var_name}' not found in dataset")
            continue
            
        var = dataset[var_name]
        if var.attrs.get("variable_type") != "extensive":
            diagnostics["passed"] = False
            diagnostics["violations"].append(
                f"{var_name}: explicit variable_type='extensive' is required "
                "for conservation checking"
            )
            diagnostics["variable_results"][var_name] = {
                "checked": False,
                "conserved": False,
                "error": "explicit variable_type='extensive' is required",
            }
            continue
        var_result = {
            'checked': True,
            'conserved': True,
            'violation_amount': 0.0,
            'details': {}
        }
        
        try:
            # Get variable metadata
            cell_methods = var.attrs.get('cell_methods', '')
            # CF-1.8 section 3.1: a dimensional variable carries a units attribute.
            # A missing attribute is a finding about the dataset, never an empty unit.
            units = var.attrs.get('units')
            if units is None:
                var_result['conserved'] = False
                var_result['details']['units'] = {'present': False}
                violation_msg = f"{var_name}: units attribute is missing"
                diagnostics['violations'].append(violation_msg)
                if verbose:
                    logger.error("[FAIL] %s", violation_msg)

            # Check for spatial conservation
            if any(dim in var.dims for dim in ['lat', 'latitude', 'lon', 'longitude', 'x', 'y']):
                spatial_conserved, spatial_violation = _check_spatial_conservation(
                    var, dataset, tolerance, cell_methods
                )
                var_result['details']['spatial'] = {
                    'conserved': spatial_conserved,
                    'violation': spatial_violation
                }
                
                if not spatial_conserved:
                    var_result['conserved'] = False
                    var_result['violation_amount'] = max(
                        var_result['violation_amount'], spatial_violation
                    )
                    violation_msg = (
                        f"{var_name}: Spatial conservation violated "
                        f"(relative error: {spatial_violation:.2e}, tolerance: {tolerance:.2e})"
                    )
                    diagnostics['violations'].append(violation_msg)
                    if verbose:
                        logger.error("[FAIL] %s", violation_msg)
            
            # Check for temporal conservation (flux-stock consistency)
            if 'time' in var.dims and '_flux' in var_name.lower():
                temporal_conserved, temporal_violation = _check_temporal_conservation(
                    var, dataset, tolerance
                )
                var_result['details']['temporal'] = {
                    'conserved': temporal_conserved,
                    'violation': temporal_violation
                }
                
                if not temporal_conserved:
                    var_result['conserved'] = False
                    var_result['violation_amount'] = max(
                        var_result['violation_amount'], temporal_violation
                    )
                    violation_msg = (
                        f"{var_name}: Temporal conservation violated "
                        f"(relative error: {temporal_violation:.2e}, tolerance: {tolerance:.2e})"
                    )
                    diagnostics['violations'].append(violation_msg)
                    if verbose:
                        logger.error("[FAIL] %s", violation_msg)
            
            # Check for non-negative values (physical constraint)
            if units is not None and any(
                indicator in str(units).lower() for indicator in ['flux', 'concentration', 'content']
            ):
                if np.any(var.values < -tolerance):
                    var_result['conserved'] = False
                    min_val = float(np.min(var.values))
                    violation_msg = f"{var_name}: Negative values detected (min: {min_val:.2e})"
                    diagnostics['violations'].append(violation_msg)
                    var_result['details']['non_negative'] = {
                        'conserved': False,
                        'min_value': min_val
                    }
                    if verbose:
                        logger.error("[FAIL] %s", violation_msg)
            
            # Update summary
            diagnostics['summary']['total_variables_checked'] += 1
            if var_result['conserved']:
                diagnostics['summary']['variables_passed'] += 1
                if verbose:
                    logger.info("[OK] %s: Conservation check passed", var_name)
            else:
                diagnostics['summary']['variables_failed'] += 1
                diagnostics['passed'] = False
                diagnostics['summary']['max_violation'] = max(
                    diagnostics['summary']['max_violation'],
                    var_result['violation_amount']
                )
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            # Audit boundary: the variable's check error is recorded and fails the report.
            var_result['checked'] = False
            var_result['conserved'] = False
            var_result['error'] = str(e)
            diagnostics['passed'] = False
            diagnostics['violations'].append(f"{var_name}: Check failed with error: {e}")
            if verbose:
                logger.error("[ERROR] %s: %s", var_name, e)
        
        diagnostics['variable_results'][var_name] = var_result
    
    return diagnostics['passed'], diagnostics


def _check_spatial_conservation(var: Any, 
                                dataset: Any,
                                tolerance: float,
                                cell_methods: str) -> Tuple[bool, float]:
    """
    Check spatial conservation for a variable.
    
    For extensive variables, the sum over space (weighted by cell area)
    should remain constant over time (within tolerance).
    """
    import numpy as np
    
    # Get spatial dimensions
    spatial_dims = [d for d in var.dims if d in ['lat', 'latitude', 'lon', 'longitude', 'x', 'y']]

    if not spatial_dims:
        return True, 0.0

    # Try to get cell area weights
    area_weights = None
    if 'cell_area' in dataset.data_vars:
        area_weights = dataset['cell_area'].values
    elif 'area' in dataset.data_vars:
        area_weights = dataset['area'].values
    elif 'lat' in dataset.coords or 'latitude' in dataset.coords:
        # Compute approximate area weights from latitude
        lat_name = 'lat' if 'lat' in dataset.coords else 'latitude'
        lat = dataset[lat_name].values
        # Simple cosine weighting
        area_weights = np.cos(np.deg2rad(lat))

    # Compute spatial sum
    if area_weights is not None and area_weights.shape == var.values.shape[-len(spatial_dims):]:
        # Weighted sum
        spatial_sum = np.sum(var.values * area_weights, axis=tuple(range(-len(spatial_dims), 0)))
    else:
        # Unweighted sum
        spatial_sum = np.sum(var.values, axis=tuple(range(-len(spatial_dims), 0)))

    # Check if sum is conserved over non-spatial dimensions
    if 'time' in var.dims and len(spatial_sum.shape) > 0:
        # Check temporal variation of spatial sum
        mean_sum = np.mean(spatial_sum)
        if mean_sum != 0:
            max_deviation = np.max(np.abs(spatial_sum - mean_sum))
            relative_violation = max_deviation / np.abs(mean_sum)
            return relative_violation <= tolerance, relative_violation
        else:
            # Mean is zero, check absolute deviation
            max_deviation = np.max(np.abs(spatial_sum))
            return max_deviation <= tolerance, max_deviation
    else:
        # No time dimension or scalar result - conservation check not applicable
        return True, 0.0
        


def _check_temporal_conservation(var: Any,
                                dataset: Any, 
                                tolerance: float) -> Tuple[bool, float]:
    """
    Check temporal conservation (flux-stock consistency).
    
    For flux variables, the time-integrated flux should equal the change
    in the corresponding stock variable (if available).
    """
    import numpy as np
    
    # Try to find corresponding stock variable
    flux_name = var.name
    stock_name = None

    # Common flux-stock naming patterns
    if '_flux' in flux_name:
        stock_name = flux_name.replace('_flux', '')
        if stock_name not in dataset.data_vars:
            stock_name = flux_name.replace('_flux', '_stock')
    elif 'flux' in flux_name.lower():
        stock_name = flux_name.lower().replace('flux', 'stock')

    if stock_name and stock_name in dataset.data_vars:
        stock_var = dataset[stock_name]
        
        # Check if stock has time dimension
        if 'time' in stock_var.dims and len(stock_var.time) > 1:
            # Compute stock change
            stock_change = np.diff(stock_var.values, axis=stock_var.dims.index('time'))
            
            # Compute integrated flux
            if 'time' in var.dims and len(var.time) > 1:
                # Get time step (assume uniform for simplicity)
                time_vals = var.time.values
                if hasattr(time_vals[0], 'astype'):
                    # Numpy datetime
                    dt_seconds = float((time_vals[1] - time_vals[0]) / np.timedelta64(1, 's'))
                else:
                    from relic.utils.physical_constants import (
                        get_physical_constant,
                    )

                    # References: CF 1.13 Section 4.4; UDUNITS time coordinates.
                    # Mathematical expression: Delta t_s = (t_1 - t_0) * S_d.
                    # Parameter meanings: t_i are day coordinates; S_d is seconds/day.
                    # Units: coordinate difference [day] becomes Delta t_s [s].
                    # Domain/constraints: numeric coordinates use uniform day spacing.
                    # Sign convention: increasing time gives positive Delta t_s.
                    dt_seconds = (
                        float(time_vals[1] - time_vals[0])
                        * float(get_physical_constant("seconds_per_day"))
                    )

                # Integrate flux (trapezoidal rule for middle points)
                flux_values = var.values
                if len(flux_values.shape) == len(stock_change.shape):
                    integrated_flux = flux_values[:-1] * dt_seconds
                else:
                    integrated_flux = flux_values * dt_seconds

                # Compare integrated flux with stock change
                max_stock = np.max(np.abs(stock_change))
                if max_stock > 0:
                    diff = np.abs(integrated_flux - stock_change)
                    max_diff = np.max(diff)
                    relative_violation = max_diff / max_stock
                    return relative_violation <= tolerance, relative_violation
                else:
                    # Stock is constant, flux should be near zero
                    max_flux = np.max(np.abs(integrated_flux))
                    return max_flux <= tolerance, max_flux

    # If no stock variable found or check not applicable
    return True, 0.0
