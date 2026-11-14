"""SCM contract loader and registry.

Loads variable-level YAML contracts and builds runtime lookup structures:
- owner_map: variable -> owner functor
- alias_map: alias -> canonical variable
- flux_whitelist: variable -> {flux_in, flux_out}
- writers_whitelist: variable -> allowed functors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging

import yaml
import jsonschema

YAML_AVAILABLE = True
JSONSCHEMA_AVAILABLE = True

logger = logging.getLogger(__name__)

# The variable contracts shipped with RELIC.
PACKAGED_CONTRACTS_DIR = Path(__file__).resolve().parent / "variables"


@dataclass
class VariableContract:
    """Variable-level SCM contract definition."""

    variable: str
    kind: str
    udunits: str
    cf_standard_name: Optional[str] = None
    # CF-1.8 section 7.3 cell_methods that qualify cf_standard_name (for example
    # "time: maximum"); None when the CF identity carries no qualifier.
    cell_methods: Optional[str] = None
    owner: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    incrementable: bool = True
    accumulates: Optional[str] = None
    flux_in: List[str] = field(default_factory=list)
    flux_out: List[str] = field(default_factory=list)
    writers: List[str] = field(default_factory=list)
    increment_writers: List[str] = field(default_factory=list)
    tendency_writers: List[str] = field(default_factory=list)
    atol: float = 1e-12
    rtol: float = 1e-9
    # State variable [m] holding the thickness of the layer this variable
    # describes (for example soil_moisture -> soil_depth); None when the
    # variable declares no layer. Consumers read it pointwise from the state.
    layer_thickness_variable: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class ContractRegistry:
    """Central registry for all variable contracts."""

    def __init__(self, schema_path: Optional[Path] = None, strict: bool = False) -> None:
        self._contracts: Dict[str, VariableContract] = {}
        self._contract_sources: Dict[str, str] = {}
        self._owner_map: Dict[str, str] = {}
        self._alias_map: Dict[str, str] = {}
        self._flux_whitelist: Dict[str, Dict[str, List[str]]] = {}
        self._writers_whitelist: Dict[str, Set[str]] = {}
        self._increment_writers: Dict[str, Set[str]] = {}
        self._tendency_writers: Dict[str, Set[str]] = {}
        self._conservation_domains: Dict[str, List[str]] = {}
        self._conflicts: List[str] = []
        self._schema_path = schema_path or (
            Path(__file__).parent / "schemas" / "scm_variable_schema.yaml"
        )
        self._schema_cache: Optional[Dict[str, Any]] = None
        self._strict = strict
        self.load_domain_ledger()

    @property
    def conflicts(self) -> List[str]:
        return list(self._conflicts)

    def load_contracts_dir(self, contracts_dir: Path) -> None:
        """Load all YAML contracts from a directory; a missing directory raises."""
        if not contracts_dir.is_dir():
            raise FileNotFoundError(f"SCM contract directory not found: {contracts_dir}")
        for yaml_file in sorted(contracts_dir.rglob("*.yml")) + sorted(contracts_dir.rglob("*.yaml")):
            if "schemas" in yaml_file.parts or "domains" in yaml_file.parts:
                continue
            self.load_contract_file(yaml_file)

    def load_domain_ledger(self, force_reload: bool = False) -> None:
        """Load conservation domain members from parameter ledger."""
        from relic.core.conservation_registry import load_parameter_ledger
        ledger = load_parameter_ledger(force_reload=force_reload)
        if not ledger:
            return
        for element, members in ledger.items():
            domain = self._normalize_domain_name(element)
            self._merge_domain_members(domain, members)

    @staticmethod
    def _normalize_domain_name(name: str) -> str:
        raw = str(name)
        if "." in raw:
            return raw
        return f"{raw}.total"

    def _merge_domain_members(self, domain: str, members: List[str]) -> None:
        if not members:
            return
        target = self._conservation_domains.setdefault(domain, [])
        for member in members:
            member_name = str(member)
            if member_name not in target:
                target.append(member_name)

    def load_contract_file(self, path: Path) -> None:
        """Load and validate a single contract file.

        In strict mode a parse, schema or field failure is raised as a
        ValueError that names the contract file.
        """
        raw_text = path.read_text(encoding="utf-8-sig")
        data: Optional[Dict[str, Any]] = None
        sanitized = raw_text.replace("\\", "/")
        try:
            data = yaml.safe_load(sanitized) or {}
        except (yaml.YAMLError, ValueError) as exc:
            if self._strict:
                raise ValueError(f"SCM contract parse failed for {path}: {exc}") from exc
            logger.warning("SCM contract parse failed for %s: %s", path, exc)
        if data is None:
            if self._strict:
                raise ValueError(f"SCM contract parse failed for {path}")
            logger.warning("SCM contract parse failed for %s", path)
            return
        try:
            self._validate_schema(data, path)
        except (jsonschema.ValidationError, jsonschema.SchemaError) as exc:
            raise ValueError(
                f"SCM contract schema validation failed for {path}: {exc.message}"
            ) from exc
        try:
            contract = self._parse_contract(data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SCM contract {path}: {exc}") from exc
        self._register_contract(contract, source_path=path)

    def _load_schema(self) -> Optional[Dict[str, Any]]:
        if self._schema_cache is not None:
            return self._schema_cache
        if not self._schema_path.exists():
            if self._strict:
                raise FileNotFoundError(f"SCM schema not found: {self._schema_path}")
            logger.warning("SCM schema not found at %s", self._schema_path)
            return None
        try:
            raw_text = self._schema_path.read_text(encoding="utf-8-sig")
            sanitized = raw_text.replace("\\", "/")
            self._schema_cache = yaml.safe_load(sanitized) or {}
        except (OSError, yaml.YAMLError) as exc:
            if self._strict:
                raise
            logger.warning("SCM schema load failed for %s: %s", self._schema_path, exc)
            self._schema_cache = {}
        return self._schema_cache

    def _validate_schema(self, data: Dict[str, Any], path: Path) -> None:
        schema = self._load_schema()
        if schema is None:
            return
        try:
            jsonschema.validate(instance=data, schema=schema)
        except (jsonschema.ValidationError, jsonschema.SchemaError) as exc:
            if self._strict:
                raise
            logger.warning("SCM schema validation failed for %s: %s", path, exc)

    def _parse_contract(self, data: Dict[str, Any]) -> VariableContract:
        """Parse YAML data into VariableContract."""
        dimension = data.get("dimension", {}) if isinstance(data.get("dimension"), dict) else {}
        writers_raw = data.get("writers")
        writers_data = writers_raw if isinstance(writers_raw, dict) else {}
        flux_contract = data.get("flux_contract", {}) if isinstance(data.get("flux_contract"), dict) else {}
        numerics = data.get("numerics", {}) if isinstance(data.get("numerics"), dict) else {}
        aliases_raw = data.get("aliases")

        writer_names: List[str] = []
        increment_writers: List[str] = []
        tendency_writers: List[str] = []
        if isinstance(writers_data, dict):
            for inc in writers_data.get("increments", []) or []:
                if isinstance(inc, dict) and inc.get("name"):
                    inc_name = str(inc.get("name"))
                    writer_names.append(inc_name)
                    increment_writers.append(inc_name)
            for tend in writers_data.get("tendencies", []) or []:
                if isinstance(tend, dict) and tend.get("name"):
                    tend_name = str(tend.get("name"))
                    writer_names.append(tend_name)
                    tendency_writers.append(tend_name)
        if isinstance(writers_raw, list):
            for name in writers_raw:
                if isinstance(name, str):
                    writer_names.append(name)

        aliases: List[str] = []
        if isinstance(aliases_raw, list):
            aliases = [str(alias) for alias in aliases_raw]

        cell_methods = dimension.get("cell_methods")
        if cell_methods is not None and (
            not isinstance(cell_methods, str) or not cell_methods.strip()
        ):
            raise ValueError(
                f"Contract {data.get('variable')!r}: dimension.cell_methods must be a "
                f"non-empty CF cell_methods string, got {cell_methods!r}"
            )

        layer_thickness_variable = data.get("layer_thickness_variable")
        if layer_thickness_variable is not None and (
            not isinstance(layer_thickness_variable, str)
            or not layer_thickness_variable.strip()
        ):
            raise ValueError(
                f"Contract {data.get('variable')!r}: layer_thickness_variable must be "
                f"a non-empty state variable name, got {layer_thickness_variable!r}"
            )

        return VariableContract(
            variable=str(data["variable"]),
            kind=str(data.get("kind", "state")),
            udunits=str(dimension.get("udunits", "dimensionless")),
            cf_standard_name=dimension.get("cf_standard_name"),
            cell_methods=cell_methods,
            owner=data.get("owner"),
            aliases=aliases,
            incrementable=bool(data.get("incrementable", True)),
            accumulates=data.get("accumulates"),
            flux_in=list(flux_contract.get("allowed_in", []) or []),
            flux_out=list(flux_contract.get("allowed_out", []) or []),
            writers=writer_names,
            increment_writers=increment_writers,
            tendency_writers=tendency_writers,
            atol=float(numerics.get("atol", 1e-12)),
            rtol=float(numerics.get("rtol", 1e-9)),
            layer_thickness_variable=layer_thickness_variable,
            raw=dict(data),
        )

    def _register_contract(
        self,
        contract: VariableContract,
        source_path: Optional[Path] = None,
    ) -> None:
        """Register contract in all lookup structures."""
        var = contract.variable
        if var in self._contracts:
            previous_source = self._contract_sources.get(var, "<in-memory>")
            current_source = str(source_path) if source_path is not None else "<in-memory>"
            conflict = (
                f"duplicate_variable:{var}:{previous_source}:{current_source}"
            )
            if self._strict:
                raise ValueError(
                    "Duplicate SCM variable contract for "
                    f"'{var}': {previous_source} and {current_source}"
                )
            self._conflicts.append(conflict)
            logger.warning(
                "Duplicate SCM variable contract for %s; keeping %s and ignoring %s",
                var,
                previous_source,
                current_source,
            )
            return

        self._contracts[var] = contract
        self._contract_sources[var] = (
            str(source_path) if source_path is not None else "<in-memory>"
        )

        if contract.owner:
            self._owner_map[var] = str(contract.owner)

        for alias in contract.aliases:
            alias_str = str(alias)
            existing = self._alias_map.get(alias_str)
            if existing and existing != var:
                self._conflicts.append(f"alias_conflict:{alias_str}:{existing}:{var}")
            else:
                self._alias_map[alias_str] = var

        self._flux_whitelist[var] = {
            "flux_in": list(contract.flux_in),
            "flux_out": list(contract.flux_out),
        }

        allowed: Set[str] = set(contract.writers)
        if contract.owner:
            allowed.add(str(contract.owner))
        self._writers_whitelist[var] = allowed
        self._increment_writers[var] = set(contract.increment_writers)
        self._tendency_writers[var] = set(contract.tendency_writers)

        if contract.accumulates:
            domain = str(contract.accumulates)
            self._merge_domain_members(domain, [var])

    def get_owner(self, variable: str) -> Optional[str]:
        return self._owner_map.get(variable)

    def get_contract(self, variable: str) -> Optional[VariableContract]:
        return self._contracts.get(variable)

    def is_alias(self, name: str) -> bool:
        return name in self._alias_map

    def resolve_alias(self, name: str) -> str:
        return self._alias_map.get(name, name)

    def can_write(self, variable: str, functor_name: str) -> bool:
        """Check whether a scientific Functor may write a variable.

        Variables without an SCM remain open for compatibility with transient
        diagnostics. A declared SCM is closed by default: an empty owner/writer
        set denotes an externally supplied forcing or observation, not an
        unrestricted scientific output.
        """
        if variable not in self._contracts:
            return True
        allowed = self._writers_whitelist.get(variable, set())
        return functor_name in allowed

    def is_incrementable(self, variable: str) -> bool:
        """Check whether a variable permits accumulated multi-writer updates."""
        contract = self._contracts.get(variable)
        if contract is None:
            return True
        return bool(contract.incrementable)

    def can_increment(self, variable: str, functor_name: str) -> bool:
        """
        Check whether a functor can contribute ACCUMULATE increments.

        If increment writers are explicitly declared, only those writers (plus owner)
        are accepted. Otherwise falls back to general write permissions.
        """
        contract = self._contracts.get(variable)
        if contract is None:
            return True
        if not contract.incrementable:
            return False
        allowed_increment = set(self._increment_writers.get(variable, set()))
        allowed_increment.update(self._tendency_writers.get(variable, set()))
        if contract.owner:
            allowed_increment.add(str(contract.owner))
        if allowed_increment:
            return functor_name in allowed_increment
        return self.can_write(variable, functor_name)

    def get_flux_whitelist(self, variable: str) -> Dict[str, List[str]]:
        return self._flux_whitelist.get(variable, {"flux_in": [], "flux_out": []})

    def get_domain_members(self, domain: str) -> List[str]:
        return list(self._conservation_domains.get(domain, []))

    def list_variables(self) -> List[str]:
        """Return all registered variable names."""
        return list(self._contracts.keys())

    def list_domains(self) -> List[str]:
        """Return all conservation domain names."""
        return list(self._conservation_domains.keys())

    def to_state_variable_spec(self, var_name: str) -> Optional[Any]:
        """Convert contract to StateVariableSpec for schema building.

        Parameters
        ----------
        var_name : str
            Variable name to convert.

        Returns
        -------
        StateVariableSpec or None
            The specification if contract exists.
        """
        contract = self._contracts.get(var_name)
        if not contract:
            return None

        from relic.core.state.types import StateVariableSpec, StateVariableType

        # The contract kind (state/diagnostic/flux/forcing/parameter) is carried
        # as the spec kind. The storage role is not derived from it: every
        # projected variable is a one-dimensional grid field (spatial_dims=1
        # below) except a parameter. The quantity basis (extensive/intensive)
        # belongs to SpatialQuantityBasis, not to the storage role.
        storage_role = (
            StateVariableType.PARAMETER
            if contract.kind == "parameter"
            else StateVariableType.GRID
        )

        # Bounds are enforced exactly as declared in numerics: both sides ->
        # [lb, ub]; lower only -> [lb, +inf); upper only -> (-inf, ub]; none ->
        # not bounded. A signed quantity declares no bound on its signed side.
        bounds = None
        numerics = contract.raw.get("numerics", {})
        if numerics:
            lo = numerics.get("lower_bound")
            hi = numerics.get("upper_bound")
            if lo is not None or hi is not None:
                bounds = (
                    float(lo) if lo is not None else float("-inf"),
                    float(hi) if hi is not None else float("inf"),
                )
                if not bounds[0] <= bounds[1]:
                    raise ValueError(
                        f"Contract {contract.variable!r} declares lower_bound "
                        f"{bounds[0]} above upper_bound {bounds[1]}"
                    )

        return StateVariableSpec(
            name=contract.variable,
            units=contract.udunits,
            bounds=bounds,
            conserved=contract.accumulates is not None,
            spatial_dims=1,
            required=False,
            variable_type=storage_role,
            kind=contract.kind,
            owner=contract.owner,
            writers=contract.writers if contract.writers else [],
            aliases=contract.aliases,
            incrementable=contract.incrementable,
            accumulates=contract.accumulates,
        )


def load_contract_registry(contracts_dir: Optional[Path] = None) -> ContractRegistry:
    """Return the strict ContractRegistry of one contract directory.

    This is the single owner of reading contract directories. The directory
    defaults to the packaged contracts (``PACKAGED_CONTRACTS_DIR``, read at
    call time). Each resolved directory is loaded once per process and the
    same registry object is returned afterwards, so callers treat it as
    read-only. Loading is strict: a missing directory raises
    ``FileNotFoundError`` and an unreadable, schema-invalid or duplicate
    contract raises ``ValueError`` naming the file. A failed load is not
    cached.
    """
    directory = PACKAGED_CONTRACTS_DIR if contracts_dir is None else Path(contracts_dir)
    return _load_contract_registry(directory.resolve())


@lru_cache(maxsize=None)
def _load_contract_registry(resolved_dir: Path) -> ContractRegistry:
    """Load one resolved contract directory strictly (cached by path)."""
    registry = ContractRegistry(strict=True)
    registry.load_contracts_dir(resolved_dir)
    return registry

