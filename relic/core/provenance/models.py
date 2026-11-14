"""
Provenance data models for W3C PROV JSON-LD payloads.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProvEntity:
    """Represents a PROV entity."""

    identifier: str
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvActivity:
    """Represents a PROV activity."""

    identifier: str
    label: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvAgent:
    """Represents a PROV agent."""

    identifier: str
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)
