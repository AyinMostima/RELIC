"""
Contract validator for RELIC extensions.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import inspect
import logging
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorBase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    TransportMechanismFunctor,
    VariableKind,
)

from .discovery import ExtensionInfo

logger = logging.getLogger(__name__)

_BRIDGE_CLASS_MAP: Dict[str, Type[FunctorBase]] = {
    "BiologicalProcessFunctor": BiologicalProcessFunctor,
    "PhysicalFieldFunctor": PhysicalFieldFunctor,
    "TransportMechanismFunctor": TransportMechanismFunctor,
    "ScaleCouplingFunctor": ScaleCouplingFunctor,
}

_CATEGORY_MAP: Dict[str, ProcessCategory] = {
    "biological_process": ProcessCategory.BIOLOGICAL_PROCESS,
    "physical_field": ProcessCategory.PHYSICAL_FIELD,
    "transport_mechanism": ProcessCategory.TRANSPORT_MECHANISM,
    "scale_coupling": ProcessCategory.SCALE_COUPLING,
}


@dataclass
class ExtensionValidationResult:
    """Result of extension validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


ValidationResult = ExtensionValidationResult


@contextmanager
def _temporary_sys_path(path: Path):
    target = str(Path(path).resolve())
    inserted = False
    if target not in sys.path:
        sys.path.insert(0, target)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(target)
            except ValueError:
                pass


def _module_belongs_to_root(module: Any, extension_root: Path) -> bool:
    root = Path(extension_root).resolve()

    module_file = getattr(module, "__file__", None)
    if module_file:
        try:
            file_path = Path(module_file).resolve()
            return root == file_path.parent or root in file_path.parents
        except OSError:
            return False

    module_paths = getattr(module, "__path__", None)
    if module_paths:
        for raw_path in module_paths:
            try:
                path_obj = Path(raw_path).resolve()
            except OSError:
                continue
            if root == path_obj or root in path_obj.parents:
                return True
    return False


def _import_class(class_path: str, extension_root: Path) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    existing = sys.modules.get(module_path)
    if existing is not None and not _module_belongs_to_root(existing, extension_root):
        sys.modules.pop(module_path, None)
        importlib.invalidate_caches()
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _instantiate_functor(
    cls: type,
    class_path: str,
    declared_config: Optional[Dict[str, Any]] = None,
) -> FunctorBase:
    """
    Instantiate a functor class the way the loader will: with the config the
    manifest declares for it, so validation exercises the real constructor.
    """

    init_signature = inspect.signature(cls.__init__)
    params = init_signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in params.values()
    )

    kwargs: Dict[str, Any] = {}
    if "config" in params or accepts_kwargs:
        kwargs["config"] = dict(declared_config or {})
    if "name" in params:
        name_param = params["name"]
        if name_param.default is inspect.Parameter.empty:
            kwargs["name"] = class_path.rsplit(".", 1)[-1].lower()

    # Built only from the manifest declaration, exactly as the loader builds
    # it; a constructor TypeError is reported, never retried without config.
    instance = cls(**kwargs)
    if not isinstance(instance, FunctorBase):
        raise TypeError(
            f"Instantiated object for '{class_path}' is not a FunctorBase subclass."
        )
    return instance


def _check_forward_signature(cls: type) -> Optional[str]:
    forward = getattr(cls, "forward", None)
    if not callable(forward):
        return "Missing callable forward() method."

    signature = inspect.signature(forward)
    parameters = list(signature.parameters.values())
    if len(parameters) < 4:
        return "forward() must accept (self, state, time, dt, **kwargs)."

    expected_names = ("self", "state", "time", "dt")
    for expected, parameter in zip(expected_names, parameters[:4]):
        if parameter.name != expected:
            return (
                "forward() signature mismatch, expected first parameters "
                "(self, state, time, dt)."
            )

    has_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if not has_kwargs:
        return "forward() must include **kwargs for runtime context compatibility."
    return None


def _check_metadata(
    declaration_category: str,
    metadata: FunctorMetadata,
) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(metadata, FunctorMetadata):
        errors.append("metadata must be an instance of FunctorMetadata.")
        return errors, warnings

    category = getattr(metadata.category, "value", metadata.category)
    expected_category = _CATEGORY_MAP.get(declaration_category)
    if expected_category is not None and category != expected_category.value:
        errors.append(
            "Manifest category '%s' does not match metadata.category '%s'."
            % (declaration_category, category)
        )

    has_inputs = bool(getattr(metadata, "input_variables", None))
    has_outputs = bool(getattr(metadata, "output_variables", None))
    if not (has_inputs or has_outputs):
        errors.append("metadata must declare input_variables and/or output_variables.")

    if metadata.spatial_scale in (None, SpatialScale.UNKNOWN):
        warnings.append("metadata.spatial_scale is missing or UNKNOWN.")
    if metadata.variable_kind in (None, VariableKind.UNKNOWN):
        warnings.append("metadata.variable_kind is missing or UNKNOWN.")

    return errors, warnings


def validate_extension(info: ExtensionInfo) -> ExtensionValidationResult:
    """
    Validate an extension against RELIC contracts.
    """

    errors: List[str] = []
    warnings: List[str] = []

    # V8: Parameter file existence
    for parameter_decl in info.manifest.parameters:
        parameter_path = (info.root_path / parameter_decl.path).resolve()
        if not parameter_path.exists():
            warnings.append(f"Parameter file not found: {parameter_path}")

    with _temporary_sys_path(info.root_path):
        for declaration in info.manifest.functors:
            prefix = f"[{info.name}:{declaration.class_path}]"

            # V2: Class importability
            try:
                cls = _import_class(declaration.class_path, info.root_path)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                errors.append(f"{prefix} Import failed: {exc}")
                continue

            # V3: Bridge inheritance
            bridge_class = _BRIDGE_CLASS_MAP.get(declaration.bridge)
            if bridge_class is None:
                errors.append(f"{prefix} Unknown bridge class '{declaration.bridge}'.")
                continue
            if not issubclass(cls, bridge_class):
                errors.append(
                    f"{prefix} {cls.__name__} does not inherit from {declaration.bridge}."
                )
                continue

            # V5: forward() signature
            forward_error = _check_forward_signature(cls)
            if forward_error:
                errors.append(f"{prefix} {forward_error}")

            # V6 and V4/V7 require instance-level checks
            instance: Optional[FunctorBase] = None
            metadata_obj: Optional[FunctorMetadata] = None
            try:
                instance = _instantiate_functor(cls, declaration.class_path, declaration.config)
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                errors.append(f"{prefix} Instantiation failed: {exc}")

            if instance is not None:
                metadata_obj = getattr(instance, "metadata", None)
            if metadata_obj is None:
                class_meta = getattr(cls, "metadata", None)
                if isinstance(class_meta, FunctorMetadata):
                    metadata_obj = class_meta

            # V4 and V7: metadata checks
            if metadata_obj is None:
                errors.append(f"{prefix} Missing FunctorMetadata on class or instance.")
            else:
                meta_errors, meta_warnings = _check_metadata(
                    declaration.category,
                    metadata_obj,
                )
                errors.extend(f"{prefix} {message}" for message in meta_errors)
                warnings.extend(f"{prefix} {message}" for message in meta_warnings)

            # V6: get_dependencies contract
            dependencies_method = getattr(cls, "get_dependencies", None)
            if not callable(dependencies_method):
                errors.append(f"{prefix} Missing callable get_dependencies() method.")
            elif instance is not None:
                try:
                    deps = instance.get_dependencies()
                    if not isinstance(deps, (list, tuple, set, dict)):
                        errors.append(
                            f"{prefix} get_dependencies() must return list/tuple/set/dict."
                        )
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                    errors.append(f"{prefix} get_dependencies() execution failed: {exc}")

    return ExtensionValidationResult(valid=not errors, errors=errors, warnings=warnings)

