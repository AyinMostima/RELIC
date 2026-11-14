"""
NUOPC-style state container for import and export fields.
"""

from typing import Any, Dict, List, Optional

from ...orchestration.port_spec import PortSpec
from ...orchestration.ssot_catalog import SSOTCatalog

from .field import Field


class State:
    """
    NUOPC-style state container.

    Parameters
    ----------
    name : str
        State name.
    ssot_catalog : SSOTCatalog, optional
        Catalog used for validating standard names and units.
    strict : bool, optional
        If True, invalid fields raise exceptions.
    """

    def __init__(
        self,
        name: str,
        ssot_catalog: Optional[SSOTCatalog] = None,
        strict: bool = False,
    ) -> None:
        self.name = name
        self._catalog = ssot_catalog
        self._strict = bool(strict)
        self._fields: Dict[str, Field] = {}
        self._directions: Dict[str, str] = {}

    def add_field(self, name: str, field: Field, direction: str) -> None:
        """
        Add a field to the state.

        Parameters
        ----------
        name : str
            Field name.
        field : Field
            Field instance.
        direction : str
            One of "import", "export", or "both".
        """
        direction_value = direction.strip().lower()
        if direction_value not in {"import", "export", "both"}:
            raise ValueError(f"Unsupported direction: {direction}")
        if name in self._fields:
            raise ValueError(f"Field already registered: {name}")
        self._validate_field(field)
        self._fields[name] = field
        self._directions[name] = direction_value

    def add_port_spec(self, port: PortSpec, direction: str, data: Any = None) -> None:
        """
        Add a Field based on a PortSpec.

        Parameters
        ----------
        port : PortSpec
            Port specification.
        direction : str
            Direction label.
        data : Any, optional
            Field payload.
        """
        field = Field.from_port_spec(port, data=data)
        self.add_field(port.name, field, direction)

    def export_state(self) -> Dict[str, Any]:
        """
        Export state data for fields marked as export or both.

        Returns
        -------
        dict
            Mapping of field name to payload.
        """
        export_data: Dict[str, Any] = {}
        for name, field in self._fields.items():
            if self._directions[name] in {"export", "both"}:
                export_data[name] = field.data
        return export_data

    def import_state(self, data: Dict[str, Any]) -> None:
        """
        Import state payloads into registered fields.

        Parameters
        ----------
        data : dict
            Mapping of field name to payload.
        """
        for name, value in data.items():
            field = self._fields.get(name)
            if field is None:
                if self._strict:
                    raise KeyError(f"Unknown field: {name}")
                continue
            if self._directions.get(name) not in {"import", "both"}:
                if self._strict:
                    raise ValueError(f"Field not marked for import: {name}")
                continue
            field.data = value

    def manifest(self) -> List[Dict[str, Any]]:
        """
        Return field metadata for manifest export.

        Returns
        -------
        list of dict
            Field manifest entries.
        """
        return [
            {
                "name": name,
                "direction": self._directions[name],
                "per_timestep": bool(field.metadata.get("per_timestep", False)),
                "conservative": bool(field.metadata.get("conservative", False)),
                **field.to_dict(),
            }
            for name, field in self._fields.items()
        ]

    def _validate_field(self, field: Field) -> None:
        if self._catalog is None or field.standard_name is None:
            return
        entry = self._catalog.get(field.name) or self._catalog.get_by_standard_name(
            field.standard_name
        )
        if entry is None:
            if self._strict:
                raise ValueError(f"Unknown standard_name: {field.standard_name}")
            return
        if field.units and entry.units and field.units != entry.units:
            if self._strict:
                raise ValueError(
                    f"Unit mismatch for {field.standard_name}: {field.units} vs {entry.units}"
                )
