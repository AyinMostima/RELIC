"""
Safe dynamic loader for RELIC extensions.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import inspect
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, cast

from relic.core.functor import FunctorBase

from .discovery import ExtensionInfo, build_extension_info, discover_extensions
from .validator import validate_extension

logger = logging.getLogger(__name__)

_EXTENSION_REGISTRY: Dict[str, str] = {}
_EXTENSION_NAME_TO_FUNCTORS: Dict[str, List[str]] = {}
# Instances the loader created, kept so a retried unload can still release a
# native host whose provider was already retired from the simulator.
_EXTENSION_INSTANCES: Dict[str, FunctorBase] = {}


class ExtensionValidationError(RuntimeError):
    """Raised when strict validation fails for one or more extensions."""


class ExtensionLoadError(RuntimeError):
    """Raised when loading an extension functor fails."""


class LoadedFunctors(dict[str, FunctorBase]):
    """Functor instances loaded by one call, keyed by instance name.

    ``skipped_extensions`` lists every item a non-strict load skipped, as
    ``{"extension": name, "reason": message}`` (with ``"functor"`` for a
    skipped duplicate instance), so a partial result is never silent. A
    strict load raises instead and leaves this list empty.
    """

    def __init__(self) -> None:
        super().__init__()
        self.skipped_extensions: list[dict[str, str]] = []


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
    declared_config: Optional[Dict[str, Any]],
) -> FunctorBase:
    init_signature = inspect.signature(cls.__init__)
    parameters = init_signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    kwargs: Dict[str, Any] = {}
    if "config" in parameters or accepts_kwargs:
        kwargs["config"] = dict(declared_config or {})
    if "name" in parameters:
        name_param = parameters["name"]
        if name_param.default is inspect.Parameter.empty:
            kwargs["name"] = class_path.rsplit(".", 1)[-1].lower()

    # The functor is built only from its manifest declaration; a constructor
    # error (including TypeError) propagates and is never retried without
    # the declared config.
    instance = cls(**kwargs)

    if not isinstance(instance, FunctorBase):
        raise TypeError(f"Class '{class_path}' is not a FunctorBase subtype.")
    return instance


def _register_core_functor_class(functor_cls: type) -> None:
    from relic.scientific import register_functor

    register_functor(functor_cls)


def _record_loaded_functor(info: ExtensionInfo, class_path: str, functor: FunctorBase) -> None:
    """Record one constructed functor under its extension.

    Recording happens per functor, before any later declaration is loaded, so
    a load that fails part-way leaves every constructed instance reachable by
    ``unload_extension`` (for example to release a native host).
    """
    _EXTENSION_REGISTRY[functor.name] = class_path
    _EXTENSION_INSTANCES[functor.name] = functor
    existing = _EXTENSION_NAME_TO_FUNCTORS.get(info.name, [])
    _EXTENSION_NAME_TO_FUNCTORS[info.name] = sorted(set(existing).union({functor.name}))


def _load_extension_info(
    info: ExtensionInfo,
    strict: bool,
    populate_core_registry: bool,
    reserved_names: set[str],
    skipped: list[dict[str, str]],
) -> Dict[str, FunctorBase]:
    """Import, construct, and record the functors declared by one extension.

    An import, construction, or core-registry failure raises
    ``ExtensionLoadError`` with the original cause in every mode. A duplicate
    instance name (within this extension or against ``reserved_names``, the
    names already loaded by the same call) is checked before it is recorded:
    strict mode raises, non-strict mode appends it to ``skipped``.
    """
    loaded: Dict[str, FunctorBase] = {}

    with _temporary_sys_path(info.root_path):
        for declaration in info.manifest.functors:
            try:
                cls = _import_class(declaration.class_path, info.root_path)
                functor = _instantiate_functor(
                    cls=cls,
                    class_path=declaration.class_path,
                    declared_config=declaration.config,
                )
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                raise ExtensionLoadError(
                    "Failed to load functor '%s' from extension '%s': %s"
                    % (declaration.class_path, info.name, exc)
                ) from exc

            setattr(functor, "_extension_name", info.name)
            setattr(functor, "_extension_source", info.source)
            setattr(functor, "_extension_execution_group", declaration.execution_group)

            if functor.name in loaded or functor.name in reserved_names:
                message = (
                    f"Duplicate functor instance name '{functor.name}' in extension "
                    f"'{info.name}'; the first loaded instance is kept."
                )
                if strict:
                    raise ExtensionLoadError(message)
                logger.error(message)
                skipped.append(
                    {"extension": info.name, "functor": functor.name, "reason": message}
                )
                continue

            loaded[functor.name] = functor
            _record_loaded_functor(info, declaration.class_path, functor)

            if populate_core_registry:
                try:
                    _register_core_functor_class(cls)
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                    raise ExtensionLoadError(
                        "Failed to populate core registry for '%s': %s"
                        % (declaration.class_path, exc)
                    ) from exc

    return loaded


def _validation_failure_message(info: ExtensionInfo) -> str | None:
    """Validate one extension; return the failure message, or None when valid."""
    result = validate_extension(info)
    for warning in result.warnings:
        logger.warning("[Extension:%s] %s", info.name, warning)
    if result.valid:
        return None
    return f"Validation failed for extension '{info.name}': {'; '.join(result.errors)}"


def _partition_valid_extensions(
    discovered: list[ExtensionInfo],
    *,
    validate: bool,
    strict: bool,
) -> tuple[list[ExtensionInfo], list[dict[str, str]]]:
    """Split discovered extensions into loadable ones and skipped ones.

    Strict mode raises ExtensionValidationError listing every failure before
    anything is loaded. Non-strict mode returns the failures as
    ``{"extension": name, "reason": message}`` records.
    """
    loadable: list[ExtensionInfo] = []
    skipped: list[dict[str, str]] = []
    for info in discovered:
        message = _validation_failure_message(info) if validate else None
        if message is None:
            loadable.append(info)
            continue
        logger.error(message)
        skipped.append({"extension": info.name, "reason": message})
    if strict and skipped:
        raise ExtensionValidationError("\n".join(item["reason"] for item in skipped))
    return loadable, skipped


def load_extensions(
    extra_dirs: Optional[List[Path]] = None,
    validate: bool = True,
    strict: bool = True,
    populate_core_registry: bool = False,
    discovered_infos: Optional[List[ExtensionInfo]] = None,
) -> LoadedFunctors:
    """
    Discover, validate, and instantiate all extension functors.

    Every discovered extension is validated before any is loaded. In strict
    mode (the default) a validation failure raises ExtensionValidationError
    before anything is loaded, and a duplicate instance name raises
    ExtensionLoadError. ``strict=False`` loads the valid extensions and lists
    every skipped extension or duplicate with its reason in
    ``skipped_extensions`` of the returned mapping. Import, construction, and
    core-registry failures raise ExtensionLoadError in both modes.
    """

    discovered = (
        list(discovered_infos)
        if discovered_infos is not None
        else discover_extensions(extra_dirs=extra_dirs)
    )
    loaded_all = LoadedFunctors()
    loadable, skipped = _partition_valid_extensions(
        discovered,
        validate=validate,
        strict=strict,
    )
    loaded_all.skipped_extensions.extend(skipped)

    for info in loadable:
        loaded_all.update(
            _load_extension_info(
                info=info,
                strict=strict,
                populate_core_registry=populate_core_registry,
                reserved_names=set(loaded_all),
                skipped=loaded_all.skipped_extensions,
            )
        )

    return loaded_all


def load_single_extension(
    extension_path: Path,
    validate: bool = True,
    strict: bool = True,
    populate_core_registry: bool = False,
) -> LoadedFunctors:
    """
    Load functors from a single extension directory.

    Strict mode (the default) raises ExtensionValidationError for an invalid
    extension. ``strict=False`` returns an empty mapping whose
    ``skipped_extensions`` names the extension and the validation failure.
    """

    info = build_extension_info(Path(extension_path), source="directory")
    loaded = LoadedFunctors()
    if validate:
        message = _validation_failure_message(info)
        if message is not None:
            if strict:
                raise ExtensionValidationError(message)
            logger.error(message)
            loaded.skipped_extensions.append({"extension": info.name, "reason": message})
            return loaded

    loaded.update(
        _load_extension_info(
            info=info,
            strict=strict,
            populate_core_registry=populate_core_registry,
            reserved_names=set(),
            skipped=loaded.skipped_extensions,
        )
    )
    return loaded


def get_extension_registry() -> Dict[str, str]:
    """Return a snapshot of loaded extension class registry."""

    return dict(_EXTENSION_REGISTRY)


def get_loaded_extensions() -> Dict[str, List[str]]:
    """Return loaded extension names mapped to registered functor instance names."""
    return {name: list(functors) for name, functors in _EXTENSION_NAME_TO_FUNCTORS.items()}


def _consume_lifecycle_receipt(simulator: Any, functor_name: str) -> Optional[Dict[str, Any]]:
    """Return the retirement receipt the simulator lifecycle owner produced."""
    getter: Any = getattr(simulator, "get_lifecycle_receipts", None)
    if not callable(getter):
        return None
    history: Any = getter()
    for receipt in reversed(list(history)):
        if getattr(receipt, "operation", None) == "unregister" and getattr(receipt, "provider_name", None) == functor_name:
            return {
                "receipt_id": receipt.receipt_id,
                "provider_uid": receipt.provider_uid,
                "status": receipt.status,
                "epoch_before": receipt.epoch_before,
                "epoch_after": receipt.epoch_after,
                "view_published": receipt.view_published,
            }
    return None


def _release_native_host(functor: Any, *, strict: bool, drain_timeout: Optional[float]) -> Optional[Dict[str, Any]]:
    """Retire, drain, and release a native plugin host owned by the functor."""
    host: Any = getattr(functor, "host", None)
    unload: Any = getattr(host, "unload", None)
    if host is None or not callable(unload):
        return None
    if not getattr(host, "loaded", False):
        return {"path": str(getattr(host, "path", "")), "released": False, "state": getattr(host, "state", "unloaded"), "skipped": True}
    # Native hosts are owned by relic.coupling.plugin (PluginHost); import its
    # lifecycle error only when a host is present.
    from relic.coupling.plugin.host import PluginLifecycleError

    try:
        receipt: Any = unload(drain_timeout=drain_timeout)
    except (PluginLifecycleError, OSError) as exc:
        # Documented native release failures (refused lifecycle transition,
        # FreeLibrary/dlclose failure) become an incomplete receipt that keeps
        # the loader record for a retry; any other error propagates.
        if strict:
            raise
        return {"path": str(getattr(host, "path", "")), "released": False, "state": getattr(host, "state", None), "error": f"{type(exc).__name__}: {exc}"}
    record: Dict[str, Any] = dict(receipt)
    record["abi"] = getattr(host, "abi", {})
    return record


def unload_extension(
    extension_name: str,
    *,
    simulator: Optional[Any] = None,
    registry: Optional[Any] = None,
    strict: bool = False,
    native_drain_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Unload a previously loaded extension and unregister related runtime entries.

    Order per functor: the simulator retires the provider (its lifecycle
    receipt is consumed into the summary), then a native plugin host owned by
    the functor is retired, drained, and released, and only then is the
    loader record cleared. A functor whose retirement or native release fails
    keeps its loader record (``complete`` is False) so the unload can be
    retried; a half-unloaded extension is never reported as removed. An
    extension that is not loaded raises KeyError in every mode.
    """
    functor_names = list(_EXTENSION_NAME_TO_FUNCTORS.get(extension_name, []))
    if not functor_names:
        raise KeyError(f"Extension '{extension_name}' is not loaded")

    removed_functors: List[str] = []
    missing_functors: List[str] = []
    failed_functors: Dict[str, str] = {}
    receipts: Dict[str, Any] = {}
    native: Dict[str, Any] = {}
    removed_node_types: List[str] = []
    cleared: List[str] = []

    runtime = cast(Any, simulator) if hasattr(simulator, "unregister_functor") else None
    registered: Optional[Dict[str, Any]] = None
    if runtime is not None:
        registered = getattr(runtime, "functors", {})

    for functor_name in functor_names:
        functor = _EXTENSION_INSTANCES.get(functor_name)
        removed = False
        if runtime is not None:
            if registered and functor_name in registered:
                functor = registered[functor_name]
            try:
                removed = bool(runtime.unregister_functor(functor_name, strict=strict))
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                if strict:
                    raise
                failed_functors[functor_name] = f"{type(exc).__name__}: {exc}"
                continue
            if removed:
                receipt = _consume_lifecycle_receipt(runtime, functor_name)
                if receipt is not None:
                    receipts[functor_name] = receipt
            else:
                # Already retired by an earlier (possibly half-completed)
                # unload; the native host may still need releasing.
                missing_functors.append(functor_name)
        # Without a simulator nothing is retired from a runtime, so nothing
        # is reported as removed; the loader still releases the native host
        # of the instance it created before clearing its record.
        native_receipt = _release_native_host(functor, strict=strict, drain_timeout=native_drain_timeout)
        if native_receipt is not None:
            native[functor_name] = native_receipt
            if not native_receipt.get("released", False) and not native_receipt.get("skipped", False):
                failed_functors[functor_name] = "native release incomplete: " + str(
                    native_receipt.get("error", native_receipt.get("state"))
                )
                if removed:
                    removed_functors.append(functor_name)
                continue
        if removed:
            removed_functors.append(functor_name)
        cleared.append(functor_name)

    registry_error: Optional[str] = None
    if registry is not None and hasattr(registry, "remove_definitions_by_functor_names"):
        try:
            removed_node_types = list(registry.remove_definitions_by_functor_names(cleared))
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            if strict:
                raise
            registry_error = f"{type(exc).__name__}: {exc}"
            removed_node_types = []

    for functor_name in cleared:
        _EXTENSION_REGISTRY.pop(functor_name, None)
        _EXTENSION_INSTANCES.pop(functor_name, None)
    remaining = [name for name in functor_names if name not in cleared]
    if remaining:
        _EXTENSION_NAME_TO_FUNCTORS[extension_name] = remaining
    else:
        _EXTENSION_NAME_TO_FUNCTORS.pop(extension_name, None)

    return {
        "extension": extension_name,
        "removed_functors": removed_functors,
        "removed_node_types": removed_node_types,
        "missing_functors": missing_functors,
        "failed_functors": failed_functors,
        "receipts": receipts,
        "native": native,
        "loader_records_cleared": cleared,
        "registry_error": registry_error,
        "complete": not remaining and registry_error is None,
    }
