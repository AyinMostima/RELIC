"""
Variable Name Mapper for RELIC

Provides variable name mapping functionality between different naming conventions
and module-specific names. This module is intentionally separated from state_manager
to avoid circular import issues with the data I/O layer.

The mapper supports:
- Standard name to module-specific name conversion
- Module-specific name to standard name conversion
- Identity mapping when no aliases are configured
- Mapped view creation for state dictionaries

References:
    - CF-1.8 Conventions for standard naming
    - RELIC Architecture Documentation
"""

from typing import Dict, Any, List, Optional
import importlib


class VariableNameMapper:
    """
    Variable name mapper for RELIC.

    Provides mapping between standard variable names (following CF-1.8 conventions)
    and module-specific variable names. Defaults to identity mapping unless
    alias rules are provided.

    Methods
    -------
    to_standard(name: str) -> str
        Map module-specific name to standard name.
    to_module(name: str) -> str
        Map standard name to module-specific name.
    map_name(name: str, target: str = 'standard') -> str
        Map a variable name to the target namespace.
    create_mapped_view(state: Dict[str, Any], target: str = 'standard') -> Dict[str, Any]
        Return a shallow copy with mapped variable names.

    Examples
    --------
    >>> mapper = VariableNameMapper()
    >>> standard_name = mapper.to_standard("temp")
    >>> module_name = mapper.to_module("air_temperature")
    >>> mapped_state = mapper.create_mapped_view(state, target='standard')
    """

    def __init__(self) -> None:
        """Initialize the variable name mapper."""
        self._module_to_standard: Dict[str, str] = {}
        self._standard_to_module: Dict[str, str] = {}
        self._conflicts: List[str] = []

    def to_standard(self, name: str) -> str:
        """
        Map module-specific name to standard CF-1.8 name.

        Currently implements identity mapping. Can be extended with
        actual mapping rules from configuration.

        Parameters
        ----------
        name : str
            Module-specific variable name

        Returns
        -------
        str
            Standard CF-1.8 variable name
        """
        return self._module_to_standard.get(name, name)

    def to_module(self, name: str) -> str:
        """
        Map standard CF-1.8 name to module-specific name.

        Currently implements identity mapping. Can be extended with
        actual mapping rules from configuration.

        Parameters
        ----------
        name : str
            Standard CF-1.8 variable name

        Returns
        -------
        str
            Module-specific variable name
        """
        return self._standard_to_module.get(name, name)

    @property
    def conflicts(self) -> List[str]:
        """Return any alias conflicts detected during refresh."""
        return list(self._conflicts)

    def refresh_from_parameter_manager(self, manager: Optional[Any] = None) -> None:
        """Load alias mappings from the ParameterManager SSOT catalogs."""
        pm = manager
        if pm is None:
            if importlib.util.find_spec("relic.parameters.manager") is None:
                raise RuntimeError("relic.parameters.manager is not available")
            manager_module = importlib.import_module("relic.parameters.manager")
            pm = manager_module.get_global_manager()

        external_key = "coupling.external.variables.variable_definitions"
        scale_key = "coupling.scales.defaults.variables"
        alias_key = "coupling.external.variables.aliases"
        # Load errors propagate: the previous mapping is kept only because nothing is reset
        # before every catalog has been read and validated.
        external_vars = pm.get(external_key, expected_type=dict, required=True)
        scale_vars = pm.get(scale_key, expected_type=dict, required=True)
        explicit_aliases = pm.get(alias_key, expected_type=dict, required=False)
        if explicit_aliases is not None and not isinstance(explicit_aliases, dict):
            raise TypeError(f"{alias_key} must be a mapping, got {type(explicit_aliases).__name__}.")
        for key, value in (explicit_aliases or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(f"{alias_key} entries must map strings to strings, got {key!r}: {value!r}.")
            if not key.strip() or not value.strip():
                raise ValueError(f"{alias_key} entry {key!r}: {value!r} has an empty alias or standard name.")

        # CF-1.8 section 7.3: cell_methods qualify the identity of a standard_name. The SSOT
        # catalog resolves (standard_name, cell_methods); an entry with qualifiers keeps its
        # variable name as its identity (daily extrema never map to the mean), several names
        # for one identity or contradicting declarations raise.
        catalog_aliases = self._catalog_aliases(external_vars, scale_vars)
        aliases: Dict[str, str] = {}
        self._module_to_standard = {}
        self._standard_to_module = {}
        self._conflicts = []
        for alias, standard_name in catalog_aliases.items():
            self._set_alias(aliases, alias, standard_name)
        for key, value in (explicit_aliases or {}).items():
            self._set_alias(aliases, key, value)
        if aliases:
            self._apply_aliases(aliases)

    @staticmethod
    def _catalog_aliases(*payloads: Any) -> Dict[str, str]:
        """Variable name -> standard_name for the unqualified entries of the catalogs.

        The CF identity rule (standard_name plus whitespace-normalized cell_methods,
        CF-1.8 section 7.3), the merge of two declarations of one name, and the rule that
        one identity has one variable are owned by relic.data.standards.cf_conventions,
        shared with SSOTCatalog; cf_conventions imports no relic module at load time, so
        it is usable while the ParameterManager is constructed. An entry with
        cell_methods keeps its variable name as its identity.
        """
        from ..data.standards.cf_conventions import (
            cf_variable_identity,
            merge_cf_variable_identity,
            register_cf_identity_holder,
        )

        identities: Dict[str, Any] = {}
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                if not isinstance(key, str) or not key.strip() or key.startswith("__"):
                    continue
                if not isinstance(value, dict):
                    continue
                name = key.strip()
                identity = cf_variable_identity(name, value)
                previous = identities.get(name)
                identities[name] = (
                    identity
                    if previous is None
                    else merge_cf_variable_identity(name, previous, identity)
                )
        holders: Dict[Any, str] = {}
        for name, identity in identities.items():
            register_cf_identity_holder(holders, name, identity)
        return {
            name: identity.standard_name
            for name, identity in identities.items()
            if not identity.qualifiers and name != identity.standard_name
        }

    def _set_alias(
        self,
        aliases: Dict[str, str],
        alias: str,
        standard_name: str,
    ) -> None:
        alias = alias.strip()
        standard_name = standard_name.strip()
        if not alias or not standard_name:
            return
        existing = aliases.get(alias)
        if existing and existing != standard_name:
            self._conflicts.append(
                f"alias_conflict:{alias}:{existing}:{standard_name}"
            )
        aliases[alias] = standard_name

    def _apply_aliases(self, aliases: Dict[str, str]) -> None:
        for alias, standard_name in aliases.items():
            self._module_to_standard[alias] = standard_name
            if standard_name not in self._standard_to_module:
                self._standard_to_module[standard_name] = alias

    def map_name(self, name: str, target: str = 'standard') -> str:
        """
        Map a variable name to the target namespace.

        Parameters
        ----------
        name : str
            Variable name to map
        target : str, optional
            Target namespace ('standard' or 'module'), default is 'standard'

        Returns
        -------
        str
            Mapped variable name
        """
        if target == 'standard':
            return self.to_standard(name)
        elif target == 'module':
            return self.to_module(name)
        else:
            return name

    def create_mapped_view(self, state: Dict[str, Any], target: str = 'standard') -> Dict[str, Any]:
        """
        Create a mapped view of a state dictionary.

        Returns a shallow copy of the state dictionary with variable names
        mapped to the target namespace. Defaults to identity mapping when no
        aliases are configured.

        Parameters
        ----------
        state : Dict[str, Any]
            Original state dictionary
        target : str, optional
            Target namespace ('standard' or 'module'), default is 'standard'

        Returns
        -------
        Dict[str, Any]
            Mapped state dictionary
        """
        mapped_state = dict(state)
        if target not in {'standard', 'module'}:
            return mapped_state

        if target == 'standard':
            for key, value in state.items():
                mapped_name = self.to_standard(key)
                if mapped_name not in mapped_state:
                    mapped_state[mapped_name] = value
        else:  # target == 'module'
            for key, value in state.items():
                mapped_name = self.to_module(key)
                if mapped_name not in mapped_state:
                    mapped_state[mapped_name] = value
        return mapped_state


# Create module-level shared instance
variable_name_mapper = VariableNameMapper()


__all__ = [
    'VariableNameMapper',
    'variable_name_mapper',
]
