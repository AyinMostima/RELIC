"""
SSOT catalog loader for typed port metadata.

This module pulls variable metadata from the centralized parameter system.
It must not hardcode units, bounds, or defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from relic.data.standards.cf_conventions import (
    CFVariableIdentity,
    cf_variable_identity,
    merge_cf_variable_identity,
    register_cf_identity_holder,
)
from relic.data.standards.dimensionality import Dimensionality, dimensionality_from_unit
from relic.parameters import get_global_manager
from relic.parameters.ops.unit_validation import _is_dimensionless_equivalent

if TYPE_CHECKING:
    from relic.coupling.scales.types import SpatialQuantityBasis

# The CF identity of an entry (standard_name plus qualifiers such as cell_methods) and
# its conflict rules are owned by relic.data.standards.cf_conventions; the catalog key
# (variable name) is the unique identity of an entry.
# Fields that two declarations of one variable name must not contradict.
_ENTRY_IDENTITY_FIELDS: Tuple[str, ...] = (
    "units",
    "semantics",
    "cf_standard_name",
    "update_mode",
    "write_mode",
    "spatial_basis",
    "quantity",
    "element",
    "datatype",
)
# A variable contract's kind is a different vocabulary from parameter semantics, so a
# contract merge compares only the fields both sources state in one vocabulary. A
# contract always declares cf_standard_name, so its null is a declared value too.
_CONTRACT_IDENTITY_FIELDS: Tuple[str, ...] = ("units", "cf_standard_name")
_CONTRACT_DECLARED_NULL_FIELDS: Tuple[str, ...] = ("cf_standard_name",)
_FILLED_FIELDS: Tuple[str, ...] = _ENTRY_IDENTITY_FIELDS + ("shape",)


@dataclass
class SSOTEntry:
    """Single SSOT variable entry."""

    name: str
    standard_name: str
    units: Optional[str]
    # Spatial representation declared by the entry (spatial_basis, or variable_type
    # intensive/categorical) and resolved by the single owner resolve_spatial_basis;
    # None when the entry declares neither.
    spatial_basis: Optional[SpatialQuantityBasis] = None
    quantity: Optional[str] = None
    cf_standard_name: Optional[str] = None
    element: Optional[str] = None
    semantics: Optional[str] = None
    update_mode: Optional[str] = None
    write_mode: Optional[str] = None
    context_requirements: Dict[str, str] = field(default_factory=dict)
    datatype: Optional[str] = None
    shape: Optional[Dict[str, Any]] = None
    dimensionality: Optional[Dimensionality] = None
    references: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


class SSOTCatalog:
    """Container for SSOT variable entries."""

    def __init__(self) -> None:
        self._entries: Dict[str, SSOTEntry] = {}
        # Explicit aliases (parameter alias table, contract aliases) -> variable name.
        self._aliases: Dict[str, str] = {}
        # CF identity key (standard_name, qualifiers) -> the one variable holding it.
        self._identity_holders: Dict[Any, str] = {}

    @staticmethod
    def _normalize_optional_cf_standard_name(
        value: Any,
        *,
        source: str,
    ) -> Optional[str]:
        """Preserve an absent CF name without conflating it with local identity."""

        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"{source} cf_standard_name must be a string or null"
            )
        if not value or value != value.strip():
            raise ValueError(
                f"{source} cf_standard_name must be an exact non-empty "
                "string or null"
            )
        return value

    @staticmethod
    def _identity(entry: SSOTEntry) -> CFVariableIdentity:
        """CF identity of an entry; a standard_name equal to the key counts as defaulted."""
        declaration: Dict[str, Any] = {"cell_methods": entry.raw.get("cell_methods")}
        if "standard_name" in entry.raw or entry.standard_name != entry.name:
            declaration["standard_name"] = entry.standard_name
        return cf_variable_identity(entry.name, declaration)

    @staticmethod
    def _field_values_agree(field_name: str, first: Any, second: Any) -> bool:
        if field_name == "units":
            left, right = str(first).strip(), str(second).strip()
            return left == right or (
                _is_dimensionless_equivalent(left) and _is_dimensionless_equivalent(right)
            )
        return first == second

    def _merge_declarations(
        self,
        existing: SSOTEntry,
        incoming: SSOTEntry,
        compared_fields: Tuple[str, ...],
        declared_null_fields: Tuple[str, ...] = (),
    ) -> None:
        """Merge a second declaration of one variable name, or raise on a contradiction.

        ``declared_null_fields`` are fields whose None in ``incoming`` is a declared
        value (it contradicts a non-None value of ``existing``).
        """
        previous_identity = self._identity(existing)
        merged_identity = merge_cf_variable_identity(
            existing.name, previous_identity, self._identity(incoming)
        )
        contradictions = []
        for field_name in compared_fields:
            first, second = getattr(existing, field_name), getattr(incoming, field_name)
            if first is None or (second is None and field_name not in declared_null_fields):
                continue
            if second is None or not self._field_values_agree(field_name, first, second):
                contradictions.append(f"{field_name} {first!r} vs {second!r}")
        for key, value in incoming.context_requirements.items():
            if key in existing.context_requirements and existing.context_requirements[key] != value:
                contradictions.append(
                    f"context_requirements[{key!r}] "
                    f"{existing.context_requirements[key]!r} vs {value!r}"
                )
        if contradictions:
            raise ValueError(
                f"SSOT entry {existing.name!r} is declared twice with contradicting "
                + "; ".join(contradictions)
            )

        for field_name in _FILLED_FIELDS:
            if getattr(existing, field_name) is None and getattr(incoming, field_name) is not None:
                setattr(existing, field_name, getattr(incoming, field_name))
        if existing.units and existing.dimensionality is None:
            existing.dimensionality = dimensionality_from_unit(existing.units)
        existing.references.extend(
            reference for reference in incoming.references if reference not in existing.references
        )
        for key, value in incoming.context_requirements.items():
            existing.context_requirements.setdefault(key, value)
        merged_raw = dict(incoming.raw)
        for key, value in existing.raw.items():
            incoming_value = merged_raw.get(key)
            if isinstance(value, dict) and isinstance(incoming_value, dict):
                merged_raw[key] = {**incoming_value, **value}
            else:
                merged_raw[key] = value
        existing.raw = merged_raw
        existing.standard_name = merged_identity.standard_name
        if merged_identity.key != previous_identity.key:
            del self._identity_holders[previous_identity.key]
            register_cf_identity_holder(self._identity_holders, existing.name, merged_identity)

    def add_entry(self, entry: SSOTEntry) -> None:
        """Register an SSOT entry.

        A second declaration of an existing variable name is merged when it agrees
        and raises when it contradicts. One CF identity (standard_name plus
        cell_methods) belongs to one variable: a second variable declaring it raises
        (cf_conventions.register_cf_identity_holder); nothing is dropped silently.
        """
        existing = self._entries.get(entry.name)
        if existing is not None:
            self._merge_declarations(existing, entry, _ENTRY_IDENTITY_FIELDS)
            return
        if entry.name in self._aliases:
            raise ValueError(
                f"SSOT entry {entry.name!r} collides with the alias of {self._aliases[entry.name]!r}"
            )
        register_cf_identity_holder(self._identity_holders, entry.name, self._identity(entry))
        self._entries[entry.name] = entry

    def _add_alias(self, alias: str, target: str) -> None:
        """Register one alias of one variable; an alias never shadows or is shared."""
        if alias in self._entries and alias != target:
            raise ValueError(
                f"SSOT alias {alias!r} of {target!r} names the existing variable {alias!r}"
            )
        existing = self._aliases.get(alias)
        if existing is not None and existing != target:
            raise ValueError(
                f"SSOT alias {alias!r} is claimed by both {existing!r} and {target!r}"
            )
        self._aliases[alias] = target

    def get(self, name: str) -> Optional[SSOTEntry]:
        """Return the SSOT entry of a variable name or explicit alias.

        A name that is neither falls back to :meth:`get_by_standard_name` without
        qualifiers.
        """
        if name in self._entries:
            return self._entries[name]
        alias = self._aliases.get(name)
        if alias is not None:
            return self._entries[alias]
        return self.get_by_standard_name(name)

    def get_by_standard_name(
        self,
        standard_name: Optional[str],
        *,
        cell_methods: Optional[str] = None,
    ) -> Optional[SSOTEntry]:
        """Return the entry identified by ``standard_name`` and its qualifiers.

        CF-1.8 7.3: the qualifiers are part of the identity, so a request without
        ``cell_methods`` selects the entry that declares none (air_temperature, not
        its daily maximum). One identity has at most one entry; None when no entry
        holds it.
        """
        if standard_name is None:
            return None
        identity = cf_variable_identity(
            standard_name,
            {"standard_name": standard_name, "cell_methods": cell_methods},
        )
        holder = self._identity_holders.get(identity.key)
        return None if holder is None else self._entries[holder]

    @classmethod
    def from_parameter_manager(cls, manager: Any = None) -> "SSOTCatalog":
        """Load SSOT catalogs from the ParameterManager."""
        pm = manager or get_global_manager()
        if pm is None:
            raise RuntimeError("ParameterManager is required for SSOT catalog loading.")

        catalog = cls()
        external_key = "coupling.external.variables.variable_definitions"
        scale_key = "coupling.scales.defaults.variables"
        alias_key = "coupling.external.variables.aliases"

        external_vars = pm.get(
            external_key,
            expected_type=dict,
            required=True,
        )
        scale_vars = pm.get(
            scale_key,
            expected_type=dict,
            required=True,
        )

        if not isinstance(external_vars, dict):
            raise TypeError("SSOT external variables must be a dict.")
        if not isinstance(scale_vars, dict):
            raise TypeError("SSOT scale variables must be a dict.")

        catalog._load_entries(external_vars)
        catalog._load_entries(scale_vars)
        catalog._load_contract_entries()
        # The alias table is optional (absent -> None); a present but invalid table raises.
        alias_payload = pm.get(
            alias_key,
            expected_type=dict,
            required=False,
        )
        catalog._load_aliases(alias_payload)
        return catalog

    @staticmethod
    def _entry_from_variable_contract(contract: Any) -> SSOTEntry:
        raw = dict(getattr(contract, "raw", {}) or {})
        support = (
            raw.get("support")
            if isinstance(raw.get("support"), dict)
            else {}
        )
        shape = raw.get("shape")
        if shape is None:
            shape = support.get("shape")
        kind = str(getattr(contract, "kind", "") or "").strip().lower()
        write_mode = None
        if kind == "diagnostic":
            write_mode = "derived"
        elif kind in {"parameter", "forcing"}:
            write_mode = "immutable"
        units = str(getattr(contract, "udunits", "") or "").strip() or None
        variable = str(getattr(contract, "variable", "")).strip()
        cf_standard_name = SSOTCatalog._normalize_optional_cf_standard_name(
            getattr(contract, "cf_standard_name", None),
            source=f"VariableContract '{variable}'",
        )
        return SSOTEntry(
            name=variable,
            standard_name=variable,
            units=units,
            cf_standard_name=cf_standard_name,
            semantics=kind or None,
            write_mode=write_mode,
            shape=shape if isinstance(shape, dict) else None,
            dimensionality=dimensionality_from_unit(units) if units else None,
            references=SSOTCatalog._normalize_references(
                raw.get("references")
            ),
            raw=raw,
        )

    def _load_contract_entries(self, contracts_dir: Optional[Path] = None) -> None:
        """Merge the canonical ContractRegistry into this catalog.

        The contracts are read through the single contract loader
        (``load_contract_registry``; the packaged contracts when
        ``contracts_dir`` is None), which raises on a missing directory.
        A contract that declares ``cf_standard_name`` also holds that CF
        identity, qualified by its ``dimension.cell_methods``: a second
        variable declaring the same identity, as a contract CF name or as a
        catalog standard_name, raises through the same
        ``cf_conventions.register_cf_identity_holder``. The entry's local
        ``standard_name`` (its connection identity) is not changed.
        """
        from relic.contracts import load_contract_registry

        registry = load_contract_registry(contracts_dir)
        if registry.conflicts:
            raise ValueError(
                "Variable contract conflicts prevent SSOT loading: "
                + "; ".join(registry.conflicts)
            )
        contract_variables = set(registry.list_variables())
        for variable in sorted(contract_variables):
            contract = registry.get_contract(variable)
            if contract is None:
                continue
            contract_entry = self._entry_from_variable_contract(contract)
            existing = self._entries.get(variable)
            if existing is None:
                self.add_entry(contract_entry)
            else:
                self._merge_declarations(
                    existing,
                    contract_entry,
                    _CONTRACT_IDENTITY_FIELDS,
                    _CONTRACT_DECLARED_NULL_FIELDS,
                )
            for alias in getattr(contract, "aliases", []) or []:
                alias_name = str(alias).strip()
                if not alias_name:
                    continue
                alias_entry = self._entries.get(alias_name)
                if alias_entry is not None and alias_name != variable:
                    if alias_name in contract_variables:
                        raise ValueError(
                            f"Variable contract {variable!r} declares alias {alias_name!r}, "
                            f"which is itself the variable contract {alias_name!r}"
                        )
                    # The contract declares that the parameter catalog lists this variable
                    # under its alias: one variable keeps one entry, and the alias entry's
                    # declarations are merged into it (contradictions raise).
                    del self._entries[alias_name]
                    del self._identity_holders[self._identity(alias_entry).key]
                    self._merge_declarations(
                        self._entries[variable],
                        alias_entry,
                        _CONTRACT_IDENTITY_FIELDS,
                    )
                self._add_alias(alias_name, variable)
            # The CF identity (cf_standard_name plus cell_methods, CF-1.8 section
            # 7.3) claimed by the contract belongs to this one variable; its
            # aliases were merged into it above.
            if contract_entry.cf_standard_name is not None:
                register_cf_identity_holder(
                    self._identity_holders,
                    variable,
                    cf_variable_identity(
                        variable,
                        {
                            "standard_name": contract_entry.cf_standard_name,
                            "cell_methods": contract.cell_methods,
                        },
                    ),
                )

    @staticmethod
    def _normalize_mode_fields(
        update_mode_raw: Any,
        write_mode_raw: Any,
    ) -> tuple[Optional[str], Optional[str]]:
        update_mode: Optional[str] = None
        write_mode: Optional[str] = None

        if isinstance(write_mode_raw, str):
            write_mode = write_mode_raw.strip().lower()

        if update_mode_raw is None:
            return update_mode, write_mode

        mode_str = str(update_mode_raw).strip().lower()
        if mode_str in {"derived", "immutable", "mutable"}:
            return None, write_mode or mode_str
        return mode_str, write_mode

    @staticmethod
    def _normalize_context_requirements(raw_context: Any) -> Dict[str, str]:
        if not isinstance(raw_context, dict):
            return {}
        return {str(k): str(v) for k, v in raw_context.items() if isinstance(k, str)}

    @staticmethod
    def _normalize_references(raw_references: Any) -> List[str]:
        if not isinstance(raw_references, list):
            return []
        return [str(ref) for ref in raw_references if isinstance(ref, str)]

    @staticmethod
    def _normalize_quantity(payload: Dict[str, Any]) -> Optional[str]:
        quantity_raw = payload.get("quantity")
        if quantity_raw is None:
            quantity_raw = payload.get("category")
        return str(quantity_raw).strip().lower() if quantity_raw is not None else None

    @staticmethod
    def _coerce_shape(raw_shape: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw_shape, dict):
            return raw_shape
        return None

    def _build_entry(self, key: Any, value: Any) -> Optional[SSOTEntry]:
        if not isinstance(key, str) or key.startswith("__") or not isinstance(value, dict):
            return None

        standard_name = str(value.get("standard_name", key)).strip()
        units = value.get("units")
        units_str = str(units) if units is not None else None
        spatial_basis = None
        if "spatial_basis" in value or "variable_type" in value:
            # Imported at use: relic.coupling imports the functor layer, whose
            # initialization reads this catalog.
            from relic.coupling.scales.types import resolve_spatial_basis

            spatial_basis = resolve_spatial_basis(key, value)
        references = self._normalize_references(value.get("references"))
        quantity = self._normalize_quantity(value)

        cf_standard_name = self._normalize_optional_cf_standard_name(
            value.get("cf_standard_name"),
            source=f"SSOT entry '{key}'",
        )
        element_raw = value.get("element")
        element = str(element_raw).strip().lower() if element_raw is not None else None
        semantics = value.get("semantics")
        datatype = value.get("datatype")
        shape = self._coerce_shape(value.get("shape"))

        update_mode, write_mode = self._normalize_mode_fields(
            value.get("update_mode"),
            value.get("write_mode") or value.get("semantic_mode"),
        )

        return SSOTEntry(
            name=key,
            standard_name=standard_name,
            units=units_str,
            spatial_basis=spatial_basis,
            quantity=quantity,
            cf_standard_name=cf_standard_name,
            element=element,
            semantics=str(semantics).lower() if semantics is not None else None,
            update_mode=update_mode,
            write_mode=write_mode,
            context_requirements=self._normalize_context_requirements(value.get("context_requirements")),
            datatype=str(datatype) if datatype is not None else None,
            shape=shape,
            dimensionality=dimensionality_from_unit(units_str) if units_str else None,
            references=references,
            raw=dict(value),
        )

    def _load_entries(self, payload: Dict[str, Any]) -> None:
        for key, value in payload.items():
            entry = self._build_entry(key, value)
            if entry is None:
                continue
            self.add_entry(entry)

    def _load_aliases(self, payload: Any) -> None:
        """Register the parameter alias table (alias -> variable name); conflicts raise."""
        if payload is None:
            return
        if not isinstance(payload, dict):
            raise TypeError("SSOT alias table must be a mapping of alias -> variable name")
        for alias, target in payload.items():
            if (
                not isinstance(alias, str)
                or not alias.strip()
                or not isinstance(target, str)
                or not target.strip()
            ):
                raise ValueError(f"SSOT alias entry {alias!r} -> {target!r} is not a name pair")
            target_entry = self.get(target.strip())
            if target_entry is None:
                raise ValueError(
                    f"SSOT alias {alias!r} targets unknown variable {target.strip()!r}"
                )
            self._add_alias(alias.strip(), target_entry.name)
