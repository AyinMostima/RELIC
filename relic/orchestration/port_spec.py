"""
Typed port specifications for graph orchestration.

This module defines the minimal PortSpec and NodeDefinition structures used
by the orchestration layer. Units and standard names must come from the
parameter system (SSOT) and must not be hardcoded here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


PORT_SUPPORT_FIELDS = (
    "shape",
    "spatial_kind",
    "area_basis",
    "temporal_basis",
    "vertical_basis",
)


_SUPPORT_VOCABULARY = {
    "spatial_kind": {
        "leaf",
        "canopy",
        "grid_cell",
        "footprint",
        "tower_height",
        "vertical_column",
    },
    "area_basis": {"leaf_area", "ground_area", "grid_area"},
    "temporal_basis": {
        "instantaneous",
        "interval_mean",
        "interval_integral",
    },
    "vertical_basis": {"none", "height_level", "layer_integral"},
}


@dataclass
class PortSpec:
    """
    Typed port contract used by graph validation.

    Fields are intentionally minimal and SSOT-driven.
    """

    name: str
    standard_name: str
    units: Optional[str] = None
    quantity: Optional[str] = None
    cf_standard_name: Optional[str] = None
    element: Optional[str] = None
    datatype: str = "tensor"
    semantics: Optional[str] = None
    update_mode: Optional[str] = None
    write_mode: Optional[str] = None
    required: bool = True
    context_requirements: Dict[str, str] = field(default_factory=dict)
    shape: Optional[Dict[str, Any]] = None
    direction: str = "inout"
    per_timestep: bool = False
    conservative: bool = False
    form_degree: Optional[int] = None
    primality: Optional[str] = None
    extensivity: Optional[str] = None
    complex_id: Optional[str] = None
    physical_domain: Optional[str] = None
    orientation_convention: Optional[str] = None
    boundary_role: Optional[str] = None
    port_kind: Optional[str] = None
    power_conjugate: Optional[str] = None
    spatial_kind: Optional[str] = None
    area_basis: Optional[str] = None
    temporal_basis: Optional[str] = None
    vertical_basis: Optional[str] = None
    declared_support_fields: frozenset[str] = field(
        default_factory=frozenset,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Reject support labels outside the publication contract vocabulary."""
        declared_fields = set(self.declared_support_fields)
        unknown_fields = declared_fields.difference(PORT_SUPPORT_FIELDS)
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unsupported support declaration fields: {unknown}.")
        declared_fields.update(
            field_name
            for field_name in PORT_SUPPORT_FIELDS
            if getattr(self, field_name) is not None
        )
        self.declared_support_fields = frozenset(declared_fields)
        for field_name, allowed_values in _SUPPORT_VOCABULARY.items():
            value = getattr(self, field_name)
            if value is not None and value not in allowed_values:
                allowed = ", ".join(sorted(allowed_values))
                raise ValueError(
                    f"Unsupported {field_name} '{value}'. Allowed values: {allowed}."
                )

    def is_support_declared(self, field_name: str) -> bool:
        """Return whether a support field was explicitly declared."""
        if field_name not in PORT_SUPPORT_FIELDS:
            raise ValueError(f"Unknown support field: {field_name}.")
        return field_name in self.declared_support_fields

    def missing_support_declarations(self) -> tuple[str, ...]:
        """Return support fields that have no explicit declaration."""
        return tuple(
            field_name
            for field_name in PORT_SUPPORT_FIELDS
            if field_name not in self.declared_support_fields
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PortSpec to a plain dictionary."""
        return {
            "name": self.name,
            "standard_name": self.standard_name,
            "units": self.units,
            "quantity": self.quantity,
            "cf_standard_name": self.cf_standard_name,
            "element": self.element,
            "datatype": self.datatype,
            "semantics": self.semantics,
            "update_mode": self.update_mode,
            "write_mode": self.write_mode,
            "required": self.required,
            "context_requirements": dict(self.context_requirements),
            "shape": deepcopy(self.shape),
            "direction": self.direction,
            "per_timestep": self.per_timestep,
            "conservative": self.conservative,
            "form_degree": self.form_degree,
            "primality": self.primality,
            "extensivity": self.extensivity,
            "complex_id": self.complex_id,
            "physical_domain": self.physical_domain,
            "orientation_convention": self.orientation_convention,
            "boundary_role": self.boundary_role,
            "port_kind": self.port_kind,
            "power_conjugate": self.power_conjugate,
            "spatial_kind": self.spatial_kind,
            "area_basis": self.area_basis,
            "temporal_basis": self.temporal_basis,
            "vertical_basis": self.vertical_basis,
            "declared_support_fields": sorted(self.declared_support_fields),
        }


@dataclass
class NodeDefinition:
    """Typed port definition for a node type."""

    node_type_id: str
    inputs: Dict[str, PortSpec] = field(default_factory=dict)
    outputs: Dict[str, PortSpec] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize NodeDefinition to a plain dictionary."""
        return {
            "node_type_id": self.node_type_id,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "metadata": deepcopy(self.metadata),
        }
