"""Pydantic schemas for validating parameter YAML files."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..spec import parameter_bounds_error


class _SchemaBase(BaseModel):  # type: ignore[misc]
    """Base schema with extra fields allowed for forward consistency."""

    model_config = ConfigDict(extra="allow")


class ParameterSpec(_SchemaBase):
    """Schema for an individual parameter entry."""

    value: Any
    units: str
    dtype: str
    bounds: Optional[List[Any]]
    category: str
    description: str = Field(min_length=1)
    references: List[Any] = Field(min_length=1)
    tags: List[Any]
    dimension: str
    requires_grad: bool
    provenance: Dict[str, Any]
    aliases: List[str] = Field(default_factory=list)
    canonical_name: Optional[str] = None
    deprecation: Optional[Dict[str, Any]] = None

    @field_validator("description")
    @classmethod
    def _description_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be blank")
        return value

    @model_validator(mode="after")
    def _bounds_follow_the_dtype(self) -> "ParameterSpec":
        # None only for a non-numeric dtype; a numeric entry writes -inf/inf
        # for an open side (relic.parameters.spec.parameter_bounds_error).
        error = parameter_bounds_error(self.bounds, self.dtype)
        if error is not None:
            raise ValueError(error)
        return self


class ParameterDatabase(_SchemaBase):
    """Schema for a parameter database YAML file."""

    schema_version: str
    tier: str
    namespace: str
    description: str
    parameters: Dict[str, ParameterSpec]

