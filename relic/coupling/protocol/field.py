"""
NUOPC-style field container.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from relic.orchestration.port_spec import PortSpec


@dataclass
class Field:
    """
    Field container with minimal NUOPC-style metadata.

    Parameters
    ----------
    name : str
        Local field name.
    data : Any, optional
        Field payload.
    standard_name : str, optional
        CF standard name.
    units : str, optional
        Units following CF conventions.
    semantics : str, optional
        Semantic label (state, flux, diagnostic).
    metadata : dict, optional
        Additional metadata.
    """

    name: str
    data: Any = None
    standard_name: Optional[str] = None
    units: Optional[str] = None
    semantics: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_port_spec(cls, port: PortSpec, data: Any = None) -> "Field":
        """
        Build a Field from an orchestration PortSpec.

        Parameters
        ----------
        port : PortSpec
            Port specification with SSOT metadata.
        data : Any, optional
            Field payload.

        Returns
        -------
        Field
            Field instance.
        """
        return cls(
            name=port.name,
            data=data,
            standard_name=port.standard_name,
            units=port.units,
            semantics=port.semantics,
            metadata={
                "datatype": port.datatype,
                "update_mode": port.update_mode,
                "write_mode": port.write_mode,
                "required": port.required,
                "direction": port.direction,
                "per_timestep": bool(port.per_timestep),
                "conservative": bool(port.conservative),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize Field metadata to a dictionary.

        Returns
        -------
        dict
            Serialized field metadata.
        """
        return {
            "name": self.name,
            "standard_name": self.standard_name,
            "units": self.units,
            "semantics": self.semantics,
            "metadata": dict(self.metadata),
        }
