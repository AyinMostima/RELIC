"""Manifest validation helpers for coupling plugins."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

from ...data.standards.dimension_check import check_port_dimension_consistency
from ...orchestration.port_spec import PortSpec
from ...orchestration.ssot_catalog import SSOTCatalog


class ManifestValidator:
    """Validate plugin manifests against JSON Schema and SSOT metadata."""

    def __init__(
        self,
        schema_path: Optional[Path] = None,
        catalog: Optional[SSOTCatalog] = None,
    ) -> None:
        self.schema_path = schema_path or Path(__file__).with_name("manifest.schema.json")
        self._schema = json.loads(self.schema_path.read_text(encoding="utf-8-sig"))
        self._validator = jsonschema.Draft7Validator(self._schema)
        self._catalog = catalog

    def validate(self, manifest_dict: Dict[str, Any]) -> List[str]:
        """
        Validate manifest payload.

        Returns a list of validation errors. Empty list means valid.
        """
        errors: List[str] = []
        for error in sorted(self._validator.iter_errors(manifest_dict), key=lambda e: list(e.path)):
            path = ".".join(str(item) for item in error.path)
            if path:
                errors.append(f"{path}: {error.message}")
            else:
                errors.append(error.message)

        errors.extend(self._validate_ssot_consistency(manifest_dict))
        return errors

    def _resolve_catalog(self) -> Optional[SSOTCatalog]:
        if self._catalog is not None:
            return self._catalog
        # A catalog that fails to load raises: the SSOT consistency checks are never skipped.
        self._catalog = SSOTCatalog.from_parameter_manager()
        return self._catalog

    def _validate_ssot_consistency(self, manifest_dict: Dict[str, Any]) -> List[str]:
        """
        Check lightweight SSOT consistency at port level.

        The check is non-blocking when SSOT is unavailable.
        """
        catalog = self._resolve_catalog()
        if catalog is None:
            return []

        ports = manifest_dict.get("ports")
        if not isinstance(ports, list):
            return []

        errors: List[str] = []
        ports_for_dimension_check: List[PortSpec] = []
        for index, raw_port in enumerate(ports):
            if not isinstance(raw_port, dict):
                continue

            port_name = str(raw_port.get("name", "")).strip()
            if not port_name:
                continue

            standard_name = str(raw_port.get("standard_name", port_name)).strip()
            ports_for_dimension_check.append(
                PortSpec(
                    name=port_name,
                    standard_name=standard_name,
                    units=(
                        str(raw_port.get("units")).strip()
                        if raw_port.get("units") is not None
                        else None
                    ),
                    direction=str(raw_port.get("direction", "inout")).strip().lower() or "inout",
                    update_mode=(
                        str(raw_port.get("update_mode")).strip().lower()
                        if raw_port.get("update_mode") is not None
                        else None
                    ),
                    per_timestep=bool(raw_port.get("per_timestep", False)),
                    conservative=bool(raw_port.get("conservative", False)),
                )
            )
            entry = catalog.get(port_name) or catalog.get_by_standard_name(standard_name)
            if entry is None:
                continue

            port_units = raw_port.get("units")
            if port_units is None or entry.units is None:
                continue
            if str(port_units).strip() != str(entry.units).strip():
                errors.append(
                    f"ports.{index}.units: '{port_units}' conflicts with SSOT '{entry.units}' for '{standard_name}'"
                )

        errors.extend(check_port_dimension_consistency(ports_for_dimension_check, catalog))
        return errors


__all__ = ["ManifestValidator"]
