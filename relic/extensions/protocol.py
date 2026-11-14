"""
Declarative manifest protocol for RELIC extensions.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from relic.core.simulator.dependency_ops import _require_execution_group

MANIFEST_FILENAME = "relic_extension.yaml"

_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
_VALID_BRIDGES = {
    "BiologicalProcessFunctor",
    "PhysicalFieldFunctor",
    "TransportMechanismFunctor",
    "ScaleCouplingFunctor",
}
_VALID_CATEGORIES = {
    "biological_process",
    "physical_field",
    "transport_mechanism",
    "scale_coupling",
}


class ManifestError(ValueError):
    """Raised when an extension manifest cannot be parsed or validated."""


class FunctorDeclaration(BaseModel):
    """Single functor declaration inside a manifest."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    class_path: str = Field(alias="class")
    bridge: str
    category: str
    # Declared per functor (fast / medium / slow); a manifest without it is rejected.
    execution_group: str
    config: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_path")
    @classmethod
    def _validate_class_path(cls, value: str) -> str:
        text = (value or "").strip()
        if "." not in text:
            raise ValueError(
                "Functor class path must be a dotted import path, e.g. module.ClassName"
            )
        return text

    @field_validator("bridge")
    @classmethod
    def _validate_bridge(cls, value: str) -> str:
        text = (value or "").strip()
        if text not in _VALID_BRIDGES:
            raise ValueError(
                f"Invalid bridge '{text}'. Allowed: {sorted(_VALID_BRIDGES)}"
            )
        return text

    @field_validator("category")
    @classmethod
    def _validate_category(cls, value: str) -> str:
        text = (value or "").strip()
        if text not in _VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{text}'. Allowed: {sorted(_VALID_CATEGORIES)}"
            )
        return text

    @field_validator("execution_group")
    @classmethod
    def _validate_execution_group(cls, value: str) -> str:
        # The simulator's one execution-group predicate and message
        # (dependency_ops._require_execution_group) judge the declaration.
        return _require_execution_group(
            (value or "").strip(),
            source="the functor declaration of an extension manifest",
        )


class ParameterFileDeclaration(BaseModel):
    """Parameter file declaration used by the extension."""

    model_config = ConfigDict(extra="ignore")
    path: str
    namespace_prefix: Optional[str] = None

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("Parameter file path cannot be empty")
        return text

    @field_validator("namespace_prefix")
    @classmethod
    def _validate_namespace_prefix(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        return text


class DependencyDeclaration(BaseModel):
    """Dependency declaration for functor-level runtime ordering hints."""

    model_config = ConfigDict(extra="ignore")
    required: List[str] = Field(default_factory=list)
    optional: List[str] = Field(default_factory=list)


class ExtensionManifest(BaseModel):
    """Top-level extension manifest."""

    model_config = ConfigDict(extra="ignore")

    name: str
    version: str
    author: str = ""
    description: str = ""
    min_relic_version: Optional[str] = None
    calibrated_output: Optional[str] = None
    functors: List[FunctorDeclaration]
    parameters: List[ParameterFileDeclaration] = Field(default_factory=list)
    dependencies: DependencyDeclaration = Field(default_factory=DependencyDeclaration)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("Extension name cannot be empty")
        return text

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        text = (value or "").strip()
        if not _SEMVER_PATTERN.match(text):
            raise ValueError("Version must follow semantic version format: X.Y.Z")
        return text

    @field_validator("min_relic_version")
    @classmethod
    def _validate_min_relic_version(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        if not _SEMVER_PATTERN.match(text):
            raise ValueError("min_relic_version must follow semantic version format: X.Y.Z")
        return text

    @field_validator("calibrated_output")
    @classmethod
    def _validate_calibrated_output(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        return text

    @field_validator("functors")
    @classmethod
    def _validate_functors(cls, value: List[FunctorDeclaration]) -> List[FunctorDeclaration]:
        if not value:
            raise ValueError("Manifest must declare at least one functor")
        return value


def parse_manifest(path: Path) -> ExtensionManifest:
    """
    Parse and validate an extension manifest.

    Parameters
    ----------
    path
        Either the extension root directory or the manifest file path.
    """

    path = Path(path)
    manifest_path = path / MANIFEST_FILENAME if path.is_dir() else path

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8-sig")) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"Invalid YAML in manifest {manifest_path}: {exc}") from exc
    except OSError as exc:
        raise ManifestError(f"Unable to read manifest {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ManifestError(
            f"Manifest payload must be a mapping in {manifest_path}, got {type(payload).__name__}"
        )

    try:
        return ExtensionManifest.model_validate(payload)
    except ValidationError as exc:
        raise ManifestError(f"Manifest schema validation failed for {manifest_path}: {exc}") from exc

