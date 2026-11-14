"""
Functor registration helpers for RELICSimulator.
"""

from __future__ import annotations

import copy
import importlib
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from ...parameters import get_global_manager
from types import MappingProxyType
from typing import Any, Optional, Set
from uuid import uuid4

import torch

from ..functor import (
    ConflictResolutionPolicy,
    FunctorBase,
    ProcessCategory,
    VariableKind,
)
from ..device_policy import DeviceFence, record_device_fence
from .checkpoint import _held_parameter_bindings, _payload_sha256, _scientific_signature_value
from ...parameters.ops.nn_params import relocate_binding
from ..dependency_graph import normalize_dependency_declaration
from .output_overlap_governance import (
    OVERLAP_POLICY_AGGREGATED,
    OVERLAP_POLICY_MUTUALLY_EXCLUSIVE,
    get_output_overlap_policy,
)

logger = logging.getLogger(__name__)

_CRITICAL_SINGLE_WRITER_OUTPUTS = {
    "evapotranspiration_flux",
    "soil_moisture",
}


@dataclass(frozen=True)
class _RegistrationSnapshot:
    """Pre-call values for the existing registration owners, never a registry."""

    functors: tuple[tuple[str, Any], ...]
    graph: Any
    graph_state: Any
    attributes: dict[str, Any]
    state_manager: Any
    ledger: Any
    lifecycle_maps: Optional[dict[str, Any]]

    @classmethod
    def capture(cls, simulator: Any) -> _RegistrationSnapshot:
        graph = simulator.dependency_graph
        state_manager = simulator.state_manager
        lifecycle = getattr(simulator, "_provider_lifecycle", None)
        lifecycle_maps = None
        if lifecycle is not None:
            lifecycle_maps = {
                "provider_uids": dict(lifecycle.provider_uids),
                "unpublished_providers": dict(lifecycle.unpublished_providers),
                "retired_uids": set(lifecycle.retired_uids),
            }
        attributes = {
            "_groups": {
                name: list(members) for name, members in simulator._groups.items()
            },
            "_bootstrap_groups": set(simulator._bootstrap_groups),
            "_groups_dirty": simulator._groups_dirty,
            "execution_order": (
                list(simulator.execution_order)
                if simulator.execution_order is not None else None
            ),
            **{
                name: list(getattr(simulator, name))
                for name in ("_order_fast", "_order_medium", "_order_slow")
            },
        }
        return cls(
            functors=tuple(simulator.functors.items()),
            graph=graph,
            graph_state=graph.capture_mutation_state() if graph is not None else None,
            attributes=attributes,
            state_manager=state_manager,
            ledger=(
                copy.deepcopy(state_manager.get_conservation_ledger())
                if state_manager is not None else None
            ),
            lifecycle_maps=lifecycle_maps,
        )

    def restore(self, simulator: Any, cause: BaseException) -> None:
        """Attempt every owner even when one compensation fails or is cancelled."""
        errors: list[BaseException] = []
        try:
            simulator.functors.clear()
            simulator.functors.update(self.functors)
        except BaseException as exc:
            errors.append(exc)
        for name, value in self.attributes.items():
            try:
                setattr(simulator, name, value)
            except BaseException as exc:
                errors.append(exc)
        try:
            simulator.dependency_graph = self.graph
            if self.graph is not None:
                self.graph.restore_mutation_state(self.graph_state)
        except BaseException as exc:
            errors.append(exc)
        try:
            if self.state_manager is not None and self.ledger is not None:
                self.state_manager.set_conservation_ledger(self.ledger)
        except BaseException as exc:
            errors.append(exc)
        try:
            lifecycle = getattr(simulator, "_provider_lifecycle", None)
            if lifecycle is not None and self.lifecycle_maps is not None:
                lifecycle.provider_uids = dict(self.lifecycle_maps["provider_uids"])
                lifecycle.unpublished_providers = dict(
                    self.lifecycle_maps["unpublished_providers"]
                )
                lifecycle.retired_uids = set(self.lifecycle_maps["retired_uids"])
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise BaseExceptionGroup(
                "Registration failed and compensation was incomplete",
                [cause, *errors],
            ) from cause


# ---------------------------------------------------------------------------
# Provider identity, committed view, lifecycle receipts, and reader tokens.
#
# The registration owner mints provider UIDs, computes the three provider
# digests, publishes the immutable committed view, and retires old views.
# The simulator stores the lifecycle state; the dependency graph and the
# execution loop only read it. No second registry exists: ``self.functors``
# remains the provider registry and ``self.dependency_graph`` the resolver.
# ---------------------------------------------------------------------------

PROVIDER_VIEW_SCHEMA_VERSION = 1
_PROVIDER_ORDER_ATTRIBUTES = ("_order_fast", "_order_medium", "_order_slow")
_REGISTRATION_RESOURCES = (
    "module_membership",
    "group_membership",
    "bootstrap_group",
    "dependency_graph",
    "execution_order",
    "conservation_ledger",
    "candidate_device",
    "provider_uid",
)
_RETIREMENT_RESOURCES = (
    "dependency_graph",
    "module_membership",
    "group_membership",
    "execution_order",
    "conservation_ledger",
    "provider_uid",
)
_TOPOLOGY_RESOURCES = (
    "module_membership",
    "group_membership",
    "dependency_graph",
    "execution_order",
    "conservation_ledger",
    "provider_uid",
    "provider_view",
    "state_remap",
)


def _provider_digest(payload: Mapping[str, Any], *, path: str) -> str:
    """Canonical SHA-256 of a scientific-contract payload (checkpoint grammar)."""
    return _payload_sha256(_scientific_signature_value(payload, path=path))


def _bridge_class_name(functor: FunctorBase) -> str:
    for bridge in _load_bridge_types():
        if isinstance(functor, bridge):
            return f"{bridge.__module__}.{bridge.__qualname__}"
    raise TypeError(
        f"Functor '{functor.name}' is not bound through a canonical bridge class."
    )


def _implementation_identity(
    functor: FunctorBase,
) -> tuple[Optional[str], bool, Optional[str]]:
    """Return (implementation_digest, strict_identity, rejection_reason).

    The only identity algorithm is ``FunctorBase.get_kernel_identity``. An
    untraceable kernel is recorded as a narrowed, non-strict identity with the
    rejection reason; it is never replaced by a ``repr`` or path fallback.
    """
    try:
        identity = functor.get_kernel_identity()
    except ValueError as exc:
        return None, False, str(exc)
    digest = _provider_digest(
        {"kernel_identity": identity},
        path=f"providers.{functor.name}.implementation",
    )
    return digest, True, None


def _contract_payload(functor: FunctorBase) -> dict[str, Any]:
    """Declared scientific contract: class, bridge, metadata, dependencies, config."""
    metadata = getattr(functor, "metadata", None)
    metadata_value = (
        metadata.to_dict()
        if metadata is not None and callable(getattr(metadata, "to_dict", None))
        else metadata
    )
    return {
        "schema_version": PROVIDER_VIEW_SCHEMA_VERSION,
        "class": f"{type(functor).__module__}.{type(functor).__qualname__}",
        "bridge": _bridge_class_name(functor),
        "metadata": metadata_value,
        "dependencies": normalize_dependency_declaration(
            functor.get_dependencies(), owner=functor.name
        ).canonical(),
        "config": getattr(functor, "_original_config", {}),
        "parameter_namespace": functor._default_parameter_namespace(),
    }


def _binding_payload(
    functor: FunctorBase,
    execution_group: str,
    substep_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Value binding of one provider inside this simulator (not its spec).

    A per-run sub-step override of the provider (the simulator setting
    substep_overrides, computation.scheduling.substep_overrides, already
    validated) belongs to its binding: it changes how this simulator executes
    the provider, not what the provider declares, so it enters the binding
    digest of the committed view and the exact-restart topology. It is recorded
    only when present, so a provider without an override keeps the binding
    digest it had before overrides existed.
    """
    payload: dict[str, Any] = {
        "schema_version": PROVIDER_VIEW_SCHEMA_VERSION,
        "name": functor.name,
        "execution_group": execution_group,
        "parameter_values": getattr(functor, "parameter_values", {}),
        "force_initial_execution": bool(
            getattr(functor, "force_initial_execution", False)
        ),
    }
    override = (substep_overrides or {}).get(functor.name)
    if override is not None:
        payload["substep_override"] = dict(override)
    return payload


@dataclass(frozen=True)
class ProviderIdentity:
    """Identity of one registered provider inside a committed view.

    ``provider_uid`` is a lifecycle identifier minted once per registration
    and never reused after retirement. The three digests are content-derived
    and therefore equal for re-instantiated providers with the same content.
    The Functor object is carried for execution and excluded from equality.
    """

    provider_uid: str
    name: str
    execution_group: str
    implementation_digest: Optional[str]
    strict_identity: bool
    identity_rejection: Optional[str]
    contract_digest: str
    binding_digest: str
    functor: Any = field(compare=False, repr=False, hash=False)

    def digest_record(self) -> dict[str, Any]:
        return {
            "implementation_digest": self.implementation_digest,
            "strict_identity": self.strict_identity,
            "identity_rejection": self.identity_rejection,
            "contract_digest": self.contract_digest,
            "binding_digest": self.binding_digest,
        }


@dataclass(frozen=True)
class CommittedProviderView:
    """Immutable published mapping key -> provider identity plus its plan."""

    schema_version: int
    epoch: int
    providers: Mapping[str, ProviderIdentity]
    groups: Mapping[str, tuple[str, ...]]
    execution_order: tuple[str, ...]
    order_fast: tuple[str, ...]
    order_medium: tuple[str, ...]
    order_slow: tuple[str, ...]
    registry_signature: tuple[tuple[str, str], ...]
    unresolved_required: tuple[str, ...] = ()

    def provider(self, name: str) -> ProviderIdentity:
        try:
            return self.providers[name]
        except KeyError:
            raise KeyError(
                f"Provider '{name}' is not part of committed provider view "
                f"epoch {self.epoch}"
            ) from None

    def group_order(self, group_name: str) -> tuple[str, ...]:
        if group_name == "fast":
            return self.order_fast
        if group_name == "medium":
            return self.order_medium
        if group_name == "slow":
            return self.order_slow
        raise KeyError(f"Unknown execution group '{group_name}'")


@dataclass
class LifecycleReceipt:
    """Narrow record of one registration or retirement transaction."""

    receipt_id: str
    operation: str
    provider_uid: Optional[str]
    provider_name: str
    implementation_digest: Optional[str]
    contract_digest: Optional[str]
    binding_digest: Optional[str]
    epoch_before: int
    epoch_after: Optional[int]
    view_published: bool
    managed_delta: dict[str, Any]
    completed_actions: tuple[str, ...]
    compensated_actions: tuple[str, ...]
    recoverable_resources: tuple[str, ...]
    status: str
    error: Optional[str]


class ProviderViewToken:
    """Reader admission token for one committed view epoch.

    Release is idempotent. The retired view a token pins is disposed only
    after the last token on it is released, whichever thread releases it.
    """

    __slots__ = ("_lifecycle", "view", "_released")

    def __init__(self, lifecycle: "_ProviderLifecycleState", view: CommittedProviderView):
        self._lifecycle = lifecycle
        self.view = view
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        # Python control leaving the step is not completion of the device
        # work it enqueued: the fence recorded on the view's device streams
        # is what allows the retired view's resources to be released (E01).
        fence = record_device_fence(_view_devices(self.view))
        self._lifecycle.release_reader(self.view.epoch, fence=fence)

    def __enter__(self) -> "ProviderViewToken":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


def _view_devices(view: CommittedProviderView) -> tuple[torch.device, ...]:
    """Devices of the providers a committed view resolves to."""
    devices: list[torch.device] = []
    for identity in view.providers.values():
        device = getattr(identity.functor, "device", None)
        if device is not None:
            devices.append(torch.device(device))
    return tuple(devices)


class _ProviderLifecycleState:
    """Registration-owned epoch, view, reader, and receipt bookkeeping."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.quiescent = threading.Condition(self.lock)
        self.epoch = 0
        self.uid_prefix = uuid4().hex[:12]
        self.uid_sequence = 0
        self.committed_view: Optional[CommittedProviderView] = None
        self.pending_publication_reason: Optional[str] = None
        self.reader_counts: dict[int, int] = {}
        self.retired_views: dict[int, CommittedProviderView] = {}
        self.retired_status: dict[int, str] = {}
        self.dispose_errors: dict[int, str] = {}
        # Device fences still pending for a retired epoch (E01): the view is
        # disposed only after its last reader left and every fence completed.
        self.device_fences: dict[int, list[DeviceFence]] = {}
        self.retired_uids: set[str] = set()
        self.provider_uids: dict[str, str] = {}
        self.unpublished_providers: dict[str, Any] = {}
        self.receipts: list[LifecycleReceipt] = []
        self.publication_history: list[dict[str, Any]] = []
        # Declared topology transitions and their commit history (G09-TOPO).
        self.topology_transitions: dict[str, TopologyTransition] = {}
        self.topology_history: list[dict[str, Any]] = []
        self.deferred_publications = 0
        # Declared topology transitions and their commit history (G09-TOPO).
        self.topology_transitions: dict[str, TopologyTransition] = {}
        self.topology_history: list[dict[str, Any]] = []
        self.deferred_publications = 0

    # -- identity -----------------------------------------------------------
    def mint_provider_uid(self, active_uids: Set[str]) -> str:
        with self.lock:
            while True:
                self.uid_sequence += 1
                candidate = f"{self.uid_prefix}-{self.uid_sequence:08d}"
                if candidate not in self.retired_uids and candidate not in active_uids:
                    return candidate

    # -- readers ------------------------------------------------------------
    def acquire_reader(self, view: CommittedProviderView) -> ProviderViewToken:
        with self.lock:
            self.reader_counts[view.epoch] = self.reader_counts.get(view.epoch, 0) + 1
            return ProviderViewToken(self, view)

    def release_reader(self, epoch: int, *, fence: Optional[DeviceFence] = None) -> None:
        with self.lock:
            remaining = self.reader_counts.get(epoch, 0) - 1
            if remaining < 0:
                raise RuntimeError(
                    f"Provider view epoch {epoch} released more readers than acquired"
                )
            if fence is not None and not fence.completed:
                self.device_fences.setdefault(epoch, []).append(fence)
            if remaining == 0:
                del self.reader_counts[epoch]
                if self.retired_status.get(epoch) in ("pending_dispose", "pending_device_fence"):
                    self._dispose_if_ready(epoch)
            else:
                self.reader_counts[epoch] = remaining

    def active_reader_count(self, epoch: int) -> int:
        with self.lock:
            return self.reader_counts.get(epoch, 0)

    # -- retirement and disposal -------------------------------------------
    def retire_view(self, view: CommittedProviderView) -> None:
        with self.lock:
            if view.epoch in self.retired_status:
                return
            self.retired_views[view.epoch] = view
            self.retired_status[view.epoch] = "pending_dispose"
            # Work enqueued by the last step on the view's devices may still
            # be running although no reader token is held any more.
            fence = record_device_fence(_view_devices(view))
            if not fence.completed:
                self.device_fences.setdefault(view.epoch, []).append(fence)
            if self.reader_counts.get(view.epoch, 0) == 0:
                self._dispose_if_ready(view.epoch)

    def _pending_fences(self, epoch: int) -> list[DeviceFence]:
        pending = [fence for fence in self.device_fences.get(epoch, []) if not fence.completed]
        if pending:
            self.device_fences[epoch] = pending
        else:
            self.device_fences.pop(epoch, None)
        return pending

    def _dispose_if_ready(self, epoch: int) -> str:
        """Dispose a reader-free retired view once its device fences completed."""
        if self._pending_fences(epoch):
            self.retired_status[epoch] = "pending_device_fence"
            self.quiescent.notify_all()
            return self.retired_status[epoch]
        self._dispose(epoch)
        return self.retired_status[epoch]

    def device_fence_status(self, epoch: int) -> dict[str, Any]:
        with self.lock:
            pending = self._pending_fences(epoch)
            return {
                "epoch": epoch,
                "status": self.retired_status.get(epoch),
                "pending_fences": len(pending),
                "devices": sorted({str(device) for fence in pending for device in fence.devices}),
            }

    def dispose_view(self, view: CommittedProviderView) -> None:
        """Release the last simulator references held by a retired view.

        The disposal action is the reference drop itself; provider objects
        are owned by the registry while registered and by nothing afterwards.
        """
        del self.retired_views[view.epoch]

    def _dispose(self, epoch: int) -> None:
        view = self.retired_views[epoch]
        try:
            self.dispose_view(view)
        except Exception as exc:  # noqa: BLE001 - observable cleanup state
            self.retired_status[epoch] = "dispose_failed"
            self.dispose_errors[epoch] = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Disposal of retired provider view epoch %s failed: %s", epoch, exc
            )
        else:
            self.retired_status[epoch] = "disposed"
            self.dispose_errors.pop(epoch, None)
        self.quiescent.notify_all()

    def retry_dispose(self, epoch: int) -> str:
        with self.lock:
            status = self.retired_status.get(epoch)
            if status is None:
                raise KeyError(f"Epoch {epoch} was never retired")
            if status == "dispose_failed":
                self._dispose(epoch)
            elif status == "pending_device_fence" and self.reader_counts.get(epoch, 0) == 0:
                self._dispose_if_ready(epoch)
            return self.retired_status[epoch]

    def wait_quiescent(self, epoch: int, timeout: Optional[float] = None) -> bool:
        """Wait until the retired epoch is disposed: readers gone, fences done.

        Device completion is awaited outside the lifecycle lock so a blocked
        device wait never stalls readers of the live view.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))

        def _remaining() -> Optional[float]:
            return None if deadline is None else max(0.0, deadline - time.monotonic())

        settled = ("disposed", "pending_device_fence", "dispose_failed")
        while True:
            with self.quiescent:
                reached = self.quiescent.wait_for(
                    lambda: self.retired_status.get(epoch) in settled,
                    timeout=_remaining(),
                )
                if not reached:
                    return False
                status = self.retired_status.get(epoch)
                if status == "disposed":
                    return True
                if status != "pending_device_fence":
                    return False
                fences = list(self._pending_fences(epoch))
            for fence in fences:
                if not fence.wait(_remaining()):
                    return False
            with self.lock:
                if self.reader_counts.get(epoch, 0) == 0:
                    self._dispose_if_ready(epoch)
                if self.retired_status.get(epoch) == "disposed":
                    return True
                if deadline is not None and time.monotonic() >= deadline:
                    return False


def _lifecycle(simulator: Any) -> _ProviderLifecycleState:
    lifecycle = getattr(simulator, "_provider_lifecycle", None)
    if lifecycle is None:
        raise RuntimeError(
            "Simulator has no provider lifecycle state; RELICSimulator owns it."
        )
    return lifecycle


def _registry_signature(simulator: Any, identities: Mapping[str, ProviderIdentity]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, identities[name].provider_uid) for name in simulator.functors))


def _current_view_matches_registry(simulator: Any, view: CommittedProviderView) -> bool:
    functors = simulator.functors
    if set(functors) != set(view.providers):
        return False
    return all(view.providers[name].functor is functors[name] for name in functors)


def _build_provider_identities(
    simulator: Any,
    previous: Optional[CommittedProviderView],
    *,
    new_uids: Mapping[str, str],
) -> dict[str, ProviderIdentity]:
    """Recompute the three digests for every registered provider.

    UIDs are stable for an unchanged registered object and come from
    ``new_uids`` for providers minted by the current transaction.
    """
    lifecycle = _lifecycle(simulator)
    group_of: dict[str, str] = {}
    for group_name, members in simulator._groups.items():
        for member in members:
            group_of[member] = group_name
    identities: dict[str, ProviderIdentity] = {}
    active_uids: Set[str] = set(new_uids.values())
    for name, functor in simulator.functors.items():
        same_published_object = (
            previous is not None
            and name in previous.providers
            and previous.providers[name].functor is functor
        )
        same_unpublished_object = lifecycle.unpublished_providers.get(name) is functor
        if name in new_uids:
            uid = new_uids[name]
        elif name in lifecycle.provider_uids and (
            same_published_object or same_unpublished_object
        ):
            uid = lifecycle.provider_uids[name]
        else:
            # The registry was replaced outside the registration owner; the
            # object is a new provider and receives a fresh UID.
            uid = lifecycle.mint_provider_uid(active_uids | set(lifecycle.provider_uids.values()))
            lifecycle.provider_uids[name] = uid
        active_uids.add(uid)
        group_name = group_of.get(name)
        if group_name is None:
            raise RuntimeError(
                f"Registered provider '{name}' has no execution group membership"
            )
        implementation_digest, strict, rejection = _implementation_identity(functor)
        identities[name] = ProviderIdentity(
            provider_uid=uid,
            name=name,
            execution_group=group_name,
            implementation_digest=implementation_digest,
            strict_identity=strict,
            identity_rejection=rejection,
            contract_digest=_provider_digest(
                _contract_payload(functor), path=f"providers.{name}.contract"
            ),
            binding_digest=_provider_digest(
                _binding_payload(
                    functor, group_name, getattr(simulator, "_substep_overrides", None)
                ),
                path=f"providers.{name}.binding",
            ),
            functor=functor,
        )
    return identities


def _publish_provider_view(
    simulator: Any,
    *,
    expected_epoch: int,
    operation: str,
    provider_uid: str,
    new_uids: Mapping[str, str],
) -> Optional[CommittedProviderView]:
    """Publish the committed view at the single publication point.

    Strict dependency mode refuses a registration with unmet required
    dependencies before this point (the transaction is rolled back). In the
    ordinary mode the view is published with the unresolved requirements
    recorded, matching the existing "execution order remains unresolved"
    contract, so partial graphs keep running with an explicit record.
    The previous view is retired and disposed after its last reader leaves.
    """
    lifecycle = _lifecycle(simulator)
    with lifecycle.lock:
        if lifecycle.deferred_publications > 0:
            # A topology transition commits several registry changes and
            # publishes exactly one epoch at its end.
            lifecycle.pending_publication_reason = (
                "publication deferred by an in-progress topology transition"
            )
            return None
        missing = simulator._find_missing_required_dependencies()
        if missing and getattr(simulator, "strict_dependencies", False):
            summary = ", ".join(f"{source} -> {target}" for source, target in missing)
            raise RuntimeError(f"Missing required dependencies: {summary}")
        unresolved_required = tuple(
            sorted(f"{source} -> {target}" for source, target in missing)
        )
        if unresolved_required:
            logger.warning(
                "Committed provider view published with unresolved required "
                "dependencies: %s",
                ", ".join(unresolved_required),
            )
        simulator._compute_group_execution_orders()
        if simulator._groups_dirty:
            raise RuntimeError(
                "Group execution orders remain dirty after recomputation"
            )
        # Cross-group lag governance (warnings, strict refusal) and any
        # configured slow -> medium promotion belong to the published plan,
        # not to a per-step mutation. A promotion moves group membership, so
        # the identities built below carry the published group.
        from .execution_loop import _govern_splitting_admissibility, _warn_cross_group_dependencies

        promoted_to_medium = _warn_cross_group_dependencies(simulator)
        # Splitting admissibility of the fast composition (S1 flux transfers
        # under strang, S3 require_declared, S5 fast_order reverse) is resolved
        # before the view is built: a refusal raises and the transaction rolls
        # back; an admissible plan installs its published fast order.
        _govern_splitting_admissibility(simulator)
        previous = lifecycle.committed_view
        identities = _build_provider_identities(simulator, previous, new_uids=new_uids)
        # Compare-and-swap at the swap point: an epoch moved by anything other
        # than this transaction means a second plan exists and must not be
        # exposed.
        if lifecycle.epoch != expected_epoch:
            raise RuntimeError(
                "Provider view publication failed compare-and-swap: expected "
                f"epoch {expected_epoch}, found {lifecycle.epoch}"
            )
        view = CommittedProviderView(
            schema_version=PROVIDER_VIEW_SCHEMA_VERSION,
            epoch=lifecycle.epoch + 1,
            providers=MappingProxyType(dict(identities)),
            groups=MappingProxyType(
                {name: tuple(members) for name, members in simulator._groups.items()}
            ),
            execution_order=tuple(simulator.execution_order or ()),
            order_fast=tuple(simulator._order_fast),
            order_medium=tuple(simulator._order_medium),
            order_slow=tuple(simulator._order_slow),
            registry_signature=_registry_signature(simulator, identities),
            unresolved_required=unresolved_required,
        )
        lifecycle.committed_view = view
        lifecycle.epoch = view.epoch
        lifecycle.unpublished_providers.clear()
        for stale in [name for name in lifecycle.provider_uids if name not in identities]:
            del lifecycle.provider_uids[stale]
        lifecycle.publication_history.append(
            {
                "epoch": view.epoch,
                "operation": operation,
                "provider_uid": provider_uid,
                "registry_signature": view.registry_signature,
                "promoted_to_medium": tuple(promoted_to_medium),
            }
        )
        if previous is not None:
            lifecycle.retire_view(previous)
        return view


# ---------------------------------------------------------------------------
# Topology transitions (G09-TOPO): declared connectivity changes committed at
# an accepted-step boundary with one epoch publication and a ledger audit.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RemapRow:
    """One row of the affine state remap x^+ = P x^- + R eta.

    ``target`` receives ``offset + sum(coefficient * state[source])``; an
    empty ``sources`` mapping with a zero offset resets the target.
    """

    target: str
    sources: Mapping[str, float] = field(default_factory=dict)
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("RemapRow target must be a non-empty variable name.")
        if not isinstance(self.sources, Mapping) or any(
            not isinstance(name, str) or not name or isinstance(coefficient, bool)
            or not isinstance(coefficient, (int, float))
            for name, coefficient in self.sources.items()
        ):
            raise TypeError("RemapRow sources must map variable names to numeric coefficients.")
        if isinstance(self.offset, bool) or not isinstance(self.offset, (int, float)):
            raise TypeError("RemapRow offset must be numeric.")
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))


@dataclass(frozen=True)
class TopologyTransition:
    """Declared hybrid topology transition of the process graph.

    References:
      - Goebel, Sanfelice, and Teel (2012), hybrid dynamical systems.
      - Revised RELIC foundations, section E: x_h^+ = P_h x_h^- + R_h eta_h
        with C_h^+ x_h^+ = C_h^- x_h^- + Pi_port.
    Mathematical expression:
      providers: retire the named ones, connect the given instances;
      state: every RemapRow target_j = offset_j + sum_i P_ji x_i;
      ledger: C x^+ - C x^- - Pi = 0 with C the indicator of
      ``ledger_variables`` and Pi = -x^-[ledger_change_source] (or 0).
    Parameter meanings:
      guard: (variable, expected value) that must hold before the commit,
      which rejects a repeated reset; parameter_keys: ParameterManager keys
      controlling the event thresholds; audit_route/ledger_change_route:
      diagnostic names receiving the residual and the declared change.
    Units: those of the remapped variables (one common unit per ledger).
    Domain/constraints: at least one provider change; unique remap targets;
      the residual must vanish within ``tolerance`` relative to the ledger
      scale before anything is committed.
    Sign convention: Pi < 0 is ledger content leaving through the declared
      route; the residual is reported with sign, audited by magnitude.
    """

    name: str
    event_name: str
    retire: tuple[str, ...] = ()
    connect: tuple[tuple[Any, str], ...] = ()
    state_remap: tuple[RemapRow, ...] = ()
    ledger_variables: tuple[str, ...] = ()
    ledger_change_source: Optional[str] = None
    guard: Optional[tuple[str, float]] = None
    parameter_keys: tuple[str, ...] = ()
    audit_route: str = field(kw_only=True)
    ledger_change_route: str = field(kw_only=True)
    # None reads computation.scheduling.topology_transition_tolerance.
    tolerance: Optional[float] = None

    def __post_init__(self) -> None:
        for label in ("name", "event_name", "audit_route", "ledger_change_route"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise ValueError(f"TopologyTransition {label} must be a non-empty string.")
        if self.tolerance is None:
            object.__setattr__(
                self,
                "tolerance",
                float(get_global_manager().get("computation.scheduling.topology_transition_tolerance", required=True)),
            )
        self._validate_membership()
        self._validate_ledger_and_guard()

    def _validate_membership(self) -> None:
        """Validate the retire, connect, and state remap declarations."""
        if not isinstance(self.retire, tuple) or any(not isinstance(n, str) or not n for n in self.retire):
            raise TypeError("retire must be a tuple of Functor names.")
        if not isinstance(self.connect, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2
            or not isinstance(item[0], FunctorBase) or not isinstance(item[1], str) or not item[1]
            for item in self.connect
        ):
            raise TypeError("connect must be a tuple of (FunctorBase instance, execution group) pairs.")
        if not self.retire and not self.connect:
            raise ValueError("A topology transition must retire or connect at least one provider.")
        if not isinstance(self.state_remap, tuple) or any(not isinstance(r, RemapRow) for r in self.state_remap):
            raise TypeError("state_remap must be a tuple of RemapRow.")
        targets = [row.target for row in self.state_remap]
        if len(set(targets)) != len(targets):
            raise ValueError("state_remap targets must be unique.")

    def _validate_ledger_and_guard(self) -> None:
        """Validate the ledger rows, guard, parameter keys, and tolerance."""
        if not isinstance(self.ledger_variables, tuple) or not self.ledger_variables or any(
            not isinstance(n, str) or not n for n in self.ledger_variables
        ):
            raise ValueError("ledger_variables must be a non-empty tuple of variable names.")
        if self.ledger_change_source is not None and (
            not isinstance(self.ledger_change_source, str) or not self.ledger_change_source
        ):
            raise ValueError("ledger_change_source must be a variable name or None.")
        if self.guard is not None and (
            not isinstance(self.guard, tuple) or len(self.guard) != 2
            or not isinstance(self.guard[0], str) or isinstance(self.guard[1], bool)
            or not isinstance(self.guard[1], (int, float))
        ):
            raise TypeError("guard must be a (variable, expected value) pair.")
        if not isinstance(self.parameter_keys, tuple) or any(
            not isinstance(k, str) or not k for k in self.parameter_keys
        ):
            raise TypeError("parameter_keys must be a tuple of ParameterManager keys.")
        if isinstance(self.tolerance, bool) or not isinstance(self.tolerance, (int, float)) or self.tolerance < 0.0:
            raise ValueError("tolerance must be a nonnegative number.")

    def remap_matrix(self, order: tuple[str, ...]) -> list[list[float]]:
        """P_h over the ordered state vector (identity rows for untouched entries)."""
        rows: list[list[float]] = []
        by_target = {row.target: row for row in self.state_remap}
        for target in order:
            row = by_target.get(target)
            if row is None:
                rows.append([1.0 if name == target else 0.0 for name in order])
                continue
            unknown = sorted(set(row.sources).difference(order))
            if unknown:
                raise KeyError(f"remap sources outside the ordered state: {unknown}")
            rows.append([float(row.sources.get(name, 0.0)) for name in order])
        return rows

    def reset_vector(self, order: tuple[str, ...]) -> list[float]:
        """R_h eta over the ordered state vector (the affine offsets)."""
        by_target = {row.target: row for row in self.state_remap}
        return [float(by_target[name].offset) if name in by_target else 0.0 for name in order]


def declare_topology_transition(simulator: Any, transition: TopologyTransition) -> None:
    """Register one declared transition with the lifecycle owner (idempotent)."""
    if not isinstance(transition, TopologyTransition):
        raise TypeError("transition must be a TopologyTransition.")
    lifecycle = _lifecycle(simulator)
    with lifecycle.lock:
        existing = lifecycle.topology_transitions.get(transition.name)
        if existing is not None and existing is not transition:
            raise ValueError(
                f"Topology transition '{transition.name}' is already declared with another definition."
            )
        lifecycle.topology_transitions[transition.name] = transition


def _ledger_sum(state: Mapping[str, Any], variables: tuple[str, ...]) -> Any:
    total = None
    for name in variables:
        if name not in state:
            raise ValueError(f"Ledger variable '{name}' is not part of the committed state.")
        value = state[name]
        total = value.clone() if total is None else total + value
    return total


def apply_topology_transition(
    simulator: Any,
    name: str,
    *,
    accepted_step: Optional[int] = None,
    accepted_time: Optional[float] = None,
) -> LifecycleReceipt:
    """Commit one declared transition at a quiescent boundary.

    Order: guard and membership checks, planned remap and ledger audit (pure),
    registry retire/connect with publication deferred, one transactional
    state remap, one epoch publication, receipt. Any failure compensates the
    registry through the registration snapshot and restores the remapped
    variables; a rejected transition never exposes a mixed provider view.
    """
    from ..state import UpdateItem, UpdateMode

    lifecycle = _lifecycle(simulator)
    with lifecycle.lock:
        if getattr(simulator, "_active_step_view", None) is not None:
            raise RuntimeError(
                f"Topology transition '{name}' cannot be committed inside a step transaction."
            )
        transition = lifecycle.topology_transitions.get(name)
        if transition is None:
            raise ValueError(f"Topology transition '{name}' is not declared.")
        state_manager = getattr(simulator, "state_manager", None)
        if state_manager is None:
            raise RuntimeError("Topology transitions require an initialized StateManager.")
        state = state_manager.get_state()
        if transition.guard is not None:
            guard_name, expected = transition.guard
            if guard_name not in state:
                raise ValueError(f"Topology guard variable '{guard_name}' is not part of the state.")
            if not bool((state[guard_name] == float(expected)).all()):
                raise RuntimeError(
                    f"Topology transition '{name}' guard failed: {guard_name} != {expected} "
                    "(repeated reset or wrong phase)."
                )
        for retired in transition.retire:
            if retired not in simulator.functors:
                raise RuntimeError(f"Topology transition '{name}' retires unregistered provider '{retired}'.")
        for functor, _group in transition.connect:
            if functor.name in simulator.functors:
                raise RuntimeError(
                    f"Topology transition '{name}' connects already registered provider '{functor.name}'."
                )

        # Planned remap (pure) and ledger audit before any mutation.
        planned: dict[str, Any] = {}
        for row in transition.state_remap:
            if row.target not in state:
                raise ValueError(f"Remap target '{row.target}' is not part of the committed state.")
            value = torch.full_like(state[row.target], float(row.offset))
            for source, coefficient in row.sources.items():
                if source not in state:
                    raise ValueError(f"Remap source '{source}' is not part of the committed state.")
                value = value + float(coefficient) * state[source].to(value)
            planned[row.target] = value
        ledger_before = _ledger_sum(state, transition.ledger_variables)
        after_state = dict(state)
        after_state.update(planned)
        ledger_after = _ledger_sum(after_state, transition.ledger_variables)
        if transition.ledger_change_source is not None:
            if transition.ledger_change_source not in state:
                raise ValueError(
                    f"Ledger change source '{transition.ledger_change_source}' is not part of the state."
                )
            declared_change = -state[transition.ledger_change_source].to(ledger_before)
        else:
            declared_change = torch.zeros_like(ledger_before)
        residual = ledger_after - ledger_before - declared_change
        scale = torch.maximum(torch.abs(ledger_before).max(), torch.ones((), dtype=ledger_before.dtype, device=ledger_before.device))
        residual_max = float(torch.abs(residual).max().detach())
        if residual_max > float(transition.tolerance) * float(scale.detach()):
            raise ValueError(
                f"Topology transition '{name}' leaks the declared ledger: residual {residual_max:.3e} "
                f"exceeds tolerance {transition.tolerance!r} x scale {float(scale):.3e}."
            )

        epoch_before = lifecycle.epoch
        snapshot = _RegistrationSnapshot.capture(simulator)
        previous_values = {target: state[target].clone() for target in planned}
        completed: list[str] = []
        remap_applied = False
        lifecycle.deferred_publications += 1
        try:
            try:
                for retired in transition.retire:
                    simulator.unregister_functor(retired)
                    completed.append(f"retire:{retired}")
                for functor, group in transition.connect:
                    simulator.register_functor(functor, execution_group=group)
                    completed.append(f"connect:{functor.name}:{group}")
                if planned:
                    source_label = f"topology:{name}"
                    state_manager.apply_updates(
                        {
                            target: UpdateItem(
                                value=value,
                                mode=UpdateMode.REPLACE,
                                source_functor=source_label,
                            )
                            for target, value in planned.items()
                        },
                        None,
                        check_conservation=False,
                        rollback_on_failure=True,
                        update_sources={target: source_label for target in planned},
                    )
                    remap_applied = True
                    completed.append("state_remap")
            finally:
                lifecycle.deferred_publications -= 1
            view = _publish_provider_view(
                simulator,
                expected_epoch=epoch_before,
                operation="topology_transition",
                provider_uid=name,
                new_uids={},
            )
            if view is None:
                raise RuntimeError(
                    f"Topology transition '{name}' could not publish a committed view: "
                    f"{lifecycle.pending_publication_reason}"
                )
            completed.append(f"provider_view:publish:{view.epoch}")
        except BaseException as exc:
            receipt = LifecycleReceipt(
                receipt_id=uuid4().hex,
                operation="topology_transition",
                provider_uid=name,
                provider_name=transition.event_name,
                implementation_digest=None,
                contract_digest=None,
                binding_digest=None,
                epoch_before=epoch_before,
                epoch_after=None,
                view_published=False,
                managed_delta={"accepted_step": accepted_step, "accepted_time": accepted_time},
                completed_actions=tuple(completed),
                compensated_actions=tuple(reversed(completed)),
                recoverable_resources=_TOPOLOGY_RESOURCES,
                status="rolled_back",
                error=f"{type(exc).__name__}: {exc}",
            )
            errors: list[BaseException] = []
            try:
                snapshot.restore(simulator, exc)
            except BaseException as compensation_exc:
                errors.append(compensation_exc)
            if remap_applied:
                try:
                    state_manager.set_state(previous_values, validate=False)
                except BaseException as compensation_exc:
                    errors.append(compensation_exc)
            if errors:
                receipt.status = "compensation_incomplete"
                receipt.error = "; ".join(f"{type(e).__name__}: {e}" for e in errors)
                _append_receipt(simulator, receipt)
                raise BaseExceptionGroup(
                    f"Topology transition '{name}' failed and compensation was incomplete",
                    [exc, *errors],
                ) from exc
            _append_receipt(simulator, receipt)
            raise

        state_manager.ingest_module_diagnostics(
            "topology_transition",
            {
                "ledger_residual": residual_max,
                "ledger_change": float(declared_change.sum().detach()),
                "event": float(accepted_step if accepted_step is not None else -1),
            },
        )
        receipt = LifecycleReceipt(
            receipt_id=uuid4().hex,
            operation="topology_transition",
            provider_uid=name,
            provider_name=transition.event_name,
            implementation_digest=None,
            contract_digest=None,
            binding_digest=None,
            epoch_before=epoch_before,
            epoch_after=view.epoch,
            view_published=True,
            managed_delta={
                "accepted_step": accepted_step,
                "accepted_time": accepted_time,
                "retired": list(transition.retire),
                "connected": [(functor.name, group) for functor, group in transition.connect],
                "state_remap": {
                    row.target: {"sources": dict(row.sources), "offset": float(row.offset)}
                    for row in transition.state_remap
                },
                "ledger": {
                    "variables": list(transition.ledger_variables),
                    "before_max": float(torch.abs(ledger_before).max().detach()),
                    "after_max": float(torch.abs(ledger_after).max().detach()),
                    "declared_change_sum": float(declared_change.sum().detach()),
                    "residual_max": residual_max,
                    "audit_route": transition.audit_route,
                    "ledger_change_route": transition.ledger_change_route,
                },
            },
            completed_actions=tuple(completed),
            compensated_actions=(),
            recoverable_resources=_TOPOLOGY_RESOURCES,
            status="committed",
            error=None,
        )
        _append_receipt(simulator, receipt)
        lifecycle.topology_history.append(
            {
                "transition": name,
                "epoch_before": epoch_before,
                "epoch_after": view.epoch,
                "accepted_step": accepted_step,
                "accepted_time": accepted_time,
                "ledger_residual_max": residual_max,
            }
        )
        return receipt


def acquire_provider_view(simulator: Any) -> ProviderViewToken:
    """Admit one reader to the current committed view (fail-closed).

    A registry that differs from the committed view means a registration
    committed without publication, so the step is refused with the reason.
    """
    lifecycle = _lifecycle(simulator)
    with lifecycle.lock:
        view = lifecycle.committed_view
        if view is None or not _current_view_matches_registry(simulator, view):
            raise RuntimeError(
                "Cannot execute without a committed provider view: the registry "
                "was changed outside the registration owner"
            )
        if simulator._groups_dirty:
            raise RuntimeError(
                "Execution group membership changed outside the registration owner; "
                "no committed provider view covers the current groups"
            )
        return lifecycle.acquire_reader(view)


def _current_provider_view(simulator: Any) -> CommittedProviderView:
    """Resolve providers for execution: the frozen step view or the committed view."""
    active = getattr(simulator, "_active_step_view", None)
    if active is not None:
        return active
    lifecycle = _lifecycle(simulator)
    with lifecycle.lock:
        view = lifecycle.committed_view
        if view is None or not _current_view_matches_registry(simulator, view):
            raise RuntimeError(
                "Cannot resolve providers without a committed view: the registry "
                "was changed outside the registration owner"
            )
        return view


def _append_receipt(simulator: Any, receipt: LifecycleReceipt) -> None:
    lifecycle = _lifecycle(simulator)
    with lifecycle.lock:
        lifecycle.receipts.append(receipt)


def _describe_exception(exc: BaseException) -> str:
    """Flatten an exception (group) into one receipt error line."""
    if isinstance(exc, BaseExceptionGroup):
        nested = "; ".join(_describe_exception(item) for item in exc.exceptions)
        return f"{type(exc).__name__}: {exc.message} [{nested}]"
    return f"{type(exc).__name__}: {exc}"


def _device_label(functor: Any) -> Optional[str]:
    device = getattr(functor, "device", None)
    return None if device is None else str(device)


def _migrate_tensor_payload(payload: Any, device: torch.device, visited: Set[int]) -> Any:
    """Recursively move plain tensor payloads to target device."""
    payload_id = id(payload)
    if payload_id in visited:
        return payload
    visited.add(payload_id)

    if isinstance(payload, torch.nn.Parameter):
        return payload
    if isinstance(payload, torch.Tensor):
        return payload.to(device=device)
    if isinstance(payload, dict):
        for key, value in list(payload.items()):
            payload[key] = _migrate_tensor_payload(value, device, visited)
        return payload
    if isinstance(payload, list):
        for index, value in enumerate(payload):
            payload[index] = _migrate_tensor_payload(value, device, visited)
        return payload
    if isinstance(payload, tuple):
        return tuple(_migrate_tensor_payload(value, device, visited) for value in payload)
    return payload


def _migrate_non_registered_tensor_attributes(functor: Any, device: torch.device) -> None:
    """
    Move non-parameter/non-buffer tensor attributes to target device.

    Notes:
    - ``nn.Module.to`` only guarantees parameters/buffers migration.
    - Many scientific functors cache plain tensor attributes; those must be migrated
      to avoid mixed CPU/CUDA arithmetic during forward().
    - An ``nn.Parameter`` attribute is skipped: a ParameterManager binding held by
      reference is relocated once, in place, through the manager
      (``_plan_binding_relocations``); a copy here would detach the holder from
      the one binding.
    """
    visited: Set[int] = set()
    for attr_name, attr_value in list(vars(functor).items()):
        if attr_name in {"_parameters", "_buffers", "_modules"}:
            continue
        if isinstance(attr_value, torch.nn.Parameter):
            continue
        if isinstance(attr_value, torch.Tensor):
            setattr(functor, attr_name, attr_value.to(device=device))
            continue
        if isinstance(attr_value, (dict, list, tuple)):
            migrated = _migrate_tensor_payload(attr_value, device, visited)
            if isinstance(attr_value, tuple):
                setattr(functor, attr_name, migrated)


def _capture_nested_device_owners(
    functor: FunctorBase,
) -> list[tuple[Any, torch.device]]:
    """Capture directly owned non-Module device participants for rollback."""
    snapshots: list[tuple[Any, torch.device]] = []
    visited = {id(functor)}
    for value in vars(functor).values():
        value_id = id(value)
        if value_id in visited or isinstance(value, (torch.Tensor, torch.nn.Module)):
            continue
        visited.add(value_id)
        to_device = getattr(value, "to_device", None)
        raw_device = getattr(value, "device", None)
        if not callable(to_device) or raw_device is None:
            continue
        try:
            snapshots.append((value, torch.device(raw_device)))
        except (TypeError, ValueError, RuntimeError):
            continue
    return snapshots


def _restore_nested_device_owners(
    snapshots: list[tuple[Any, torch.device]],
) -> None:
    """Restore all captured nested device owners and report aggregate failure."""
    rollback_errors: list[BaseException] = []
    for owner, original_device in snapshots:
        try:
            owner.to_device(original_device)
        except BaseException as exc:
            rollback_errors.append(exc)
    if rollback_errors:
        raise BaseExceptionGroup("Nested device compensation failed", rollback_errors)


def _require_functor_base(functor: Any) -> FunctorBase:
    """Enforce the canonical FunctorBase simulator registration boundary."""
    if not isinstance(functor, FunctorBase):
        raise TypeError(
            "Registered scientific processes must inherit from FunctorBase "
            "through an approved bridge class."
        )
    return functor


def _simulator_has_functor_class(simulator: Any, class_name: str) -> bool:
    return any(
        existing.__class__.__name__ == class_name
        for existing in simulator.functors.values()
    )


def _enforce_mutually_exclusive_functors(simulator: Any, new_class_name: str) -> None:
    """Prevent simultaneous registration of mutually-exclusive process models."""
    simulator_config = getattr(simulator, "config", {}) or {}
    allow_bucket_surface_coexistence = bool(
        simulator_config.get("allow_bucket_surface_coexistence", False)
    )

    transport_single_writer_set = {
        "LinearAdvection",
        "FickianDiffusion",
        "AdvectionDiffusion",
    }
    mutually_exclusive_sets = (
        {"VerticalPercolationTransport", "RichardsSoilMoisture"},
        {"BucketSoilMoisture", "RichardsSoilMoisture"},
        {"BucketSoilMoisture", "SurfaceRunoff"},
        {"PenmanMonteithET", "PriestleyTaylorET"},
        transport_single_writer_set,
    )

    for class_set in mutually_exclusive_sets:
        if (
            allow_bucket_surface_coexistence
            and class_set == {"BucketSoilMoisture", "SurfaceRunoff"}
        ):
            continue
        if class_set is transport_single_writer_set and not getattr(
            simulator,
            "strict_dependencies",
            False,
        ):
            continue
        if new_class_name not in class_set:
            continue
        for existing_class in class_set:
            if existing_class == new_class_name:
                continue
            if _simulator_has_functor_class(simulator, existing_class):
                raise ValueError(
                    f"{new_class_name} cannot be registered with {existing_class}; "
                    "select one mutually exclusive process model."
                )


def _load_bridge_types() -> tuple[type[Any], ...]:
    """Load authoritative bridge classes used by executable functors."""
    try:
        bridge_module = importlib.import_module("relic.core.functor.functor_bridges")
        BiologicalProcessFunctor = getattr(bridge_module, "BiologicalProcessFunctor")
        PhysicalFieldFunctor = getattr(bridge_module, "PhysicalFieldFunctor")
        ScaleCouplingFunctor = getattr(bridge_module, "ScaleCouplingFunctor")
        TransportMechanismFunctor = getattr(bridge_module, "TransportMechanismFunctor")
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
        raise RuntimeError(
            "Failed to import bridge functor classes required for registration. "
            "This indicates an internal packaging/refactor inconsistency."
        ) from exc
    return (
        BiologicalProcessFunctor,
        TransportMechanismFunctor,
        PhysicalFieldFunctor,
        ScaleCouplingFunctor,
    )


def _require_category_bridge(functor: FunctorBase) -> FunctorBase:
    """Require one canonical bridge matching the declared scientific category."""
    (
        biological_bridge,
        transport_bridge,
        physical_bridge,
        scale_bridge,
    ) = _load_bridge_types()
    bridge_by_category = {
        ProcessCategory.BIOLOGICAL_PROCESS: biological_bridge,
        ProcessCategory.TRANSPORT_MECHANISM: transport_bridge,
        ProcessCategory.PHYSICAL_FIELD: physical_bridge,
        ProcessCategory.SCALE_COUPLING: scale_bridge,
    }
    category = getattr(getattr(functor, "metadata", None), "category", None)
    expected_bridge = (
        bridge_by_category.get(category) if isinstance(category, ProcessCategory) else None
    )
    if expected_bridge is None:
        raise TypeError(
            "Registered scientific processes must declare one canonical "
            "ProcessCategory before registration."
        )

    matched_bridges = tuple(
        bridge_type
        for bridge_type in bridge_by_category.values()
        if isinstance(functor, bridge_type)
    )
    if matched_bridges != (expected_bridge,):
        category_value = getattr(category, "value", str(category))
        raise TypeError(
            f"Scientific category '{category_value}' requires "
            f"{expected_bridge.__name__}; received "
            f"{functor.__class__.__name__}."
        )
    return functor


def _resolve_execution_group(
    simulator: Any,
    functor: FunctorBase,
    execution_group: Optional[str],
) -> str:
    """Resolve the execution group of a registration without mutating anything.

    The priority chain and the validation of every declaration belong to
    ``simulator._determine_execution_group`` (dependency_ops); the register
    argument is its highest-priority rule. A resolved group that is not one of
    the simulator's execution groups is an invariant violation, not a reason to
    fall back to another group.
    """
    group = simulator._determine_execution_group(functor, execution_group)
    if group not in simulator._groups:
        raise RuntimeError(
            f"Execution-group invariant violated: Functor '{functor.name}' resolved to "
            f"{group!r}, which is not one of the simulator groups {sorted(simulator._groups)}."
        )
    return group


def _ensure_dependency_graph(simulator: Any) -> None:
    if simulator.dependency_graph is None:
        from ..dependency_graph import DependencyGraphResolver

        simulator.dependency_graph = DependencyGraphResolver()


def _attach_functor_dependencies(simulator: Any, functor: FunctorBase) -> None:
    """Attach required and optional dependency edges for a functor."""
    declared = normalize_dependency_declaration(functor.get_dependencies(), owner=functor.name)
    required_deps = list(declared.required)
    optional_deps = list(declared.optional)
    missing_required = [dep for dep in required_deps if dep not in simulator.functors]
    if missing_required:
        msg = "Missing required dependencies for %s: %s" % (
            functor.name,
            ", ".join(missing_required),
        )
        if simulator.strict_dependencies:
            logger.error(msg)
        else:
            logger.warning(msg)

    for dep in required_deps:
        if dep in simulator.functors:
            simulator.dependency_graph.add_edge(functor.name, dep, required=True)
    for dep in optional_deps:
        if dep in simulator.functors:
            simulator.dependency_graph.add_edge(functor.name, dep, required=False)
        else:
            logger.debug(
                "Optional dependency not available: %s depends on %s",
                functor.name,
                dep,
            )


def _refresh_execution_order(simulator: Any, functor: FunctorBase) -> None:
    """Recompute topological order while honoring strict dependency mode."""
    missing_all = simulator._find_missing_required_dependencies()
    if missing_all:
        missing_summary = ", ".join(
            f"{source} -> {target}" for source, target in missing_all
        )
        if simulator.strict_dependencies:
            raise RuntimeError(
                f"Missing required dependencies: {missing_summary}"
            )
        logger.warning(
            "Execution order remains unresolved until required dependencies "
            "are registered: %s",
            missing_summary,
        )
        simulator.execution_order = None
        return
    simulator.execution_order = simulator.dependency_graph.topological_sort()


def _plan_binding_relocations(
    simulator: Any, functor: FunctorBase, target_device: torch.device,
) -> list[tuple[Any, str]]:
    """Held ParameterManager bindings to relocate to the Functor's effective device.

    A learnable binding has one owner (the ParameterManager) and one device.
    Each binding the Functor holds that is not on ``target_device`` is
    relocated once, in place, through the manager, unless another registered
    Functor holds it on another device: that raises ValueError naming the
    parameter, both Functors and their devices (assign both the same device
    through ``device`` / ``device_overrides``).
    """
    target = torch.empty(0, device=target_device).device
    plan: list[tuple[Any, str]] = []
    for key, (manager, binding) in _held_parameter_bindings(functor).items():
        if binding.device == target:
            continue
        for other in getattr(simulator, "functors", {}).values():
            if other is functor:
                continue
            held = _held_parameter_bindings(other).get(key)
            if held is not None and held[1] is binding:
                raise ValueError(
                    f"ParameterManager binding '{key}' is held by Functor '{other.name}' "
                    f"on {binding.device} and by Functor '{functor.name}' assigned to "
                    f"{target}; a shared learnable parameter has one device, so assign "
                    "both Functors the same device (device / device_overrides)."
                )
        plan.append((manager, key))
    return plan


def _maybe_migrate_functor_device(simulator: Any, functor: FunctorBase) -> Optional[dict[str, Any]]:
    """Move functor to its effective device (per-functor override or simulator global).

    Resolution order:
        1. Per-functor override from SimulatorConfig["device_overrides"]
        2. Simulator global device from SimulatorConfig["device"]

    Returns the description of the source-device fence awaited before the
    move (None when no move was needed). Work already enqueued on the source
    device completes before its tensors are copied; the substream generator
    is re-created on the target by ``to_device`` (no old handle is reused).
    """
    if not hasattr(simulator, "device") or simulator.device is None:
        return None

    # Per-functor override takes priority, otherwise use simulator global device.
    # Prefer simulator public API when available.
    if hasattr(simulator, "get_effective_device") and callable(getattr(simulator, "get_effective_device")):
        target_device = torch.device(simulator.get_effective_device(functor.name))
    else:
        overrides = getattr(simulator, "_device_overrides", {})
        target_device = torch.device(overrides.get(functor.name, simulator.device))

    nested_device_snapshots = _capture_nested_device_owners(functor)
    relocations = _plan_binding_relocations(simulator, functor, target_device)

    # Skip only when the Functor, all directly owned device participants and
    # every held ParameterManager binding agree.
    if (
        hasattr(functor, "device")
        and functor.device == target_device
        and all(device == target_device for _, device in nested_device_snapshots)
        and not relocations
    ):
        return None

    original_device = torch.device(getattr(functor, "device", "cpu"))
    binding_devices = [manager.nn_parameters[key].device for manager, key in relocations]
    source_fence = record_device_fence(
        (
            original_device,
            *[device for _, device in nested_device_snapshots],
            *binding_devices,
        )
    )
    source_fence.wait()
    fence_record = source_fence.describe()
    relocated: list[tuple[Any, str, torch.device]] = []
    try:
        # Held bindings move once, in place, through the manager that owns them.
        for (manager, key), source_device in zip(relocations, binding_devices):
            if relocate_binding(manager, key, target_device):
                relocated.append((manager, key, source_device))
        if hasattr(functor, "to_device") and callable(getattr(functor, "to_device")):
            functor.to_device(target_device)
        else:
            functor.to(target_device)
        for module in functor.modules():
            if hasattr(module, "device"):
                object.__setattr__(module, "device", target_device)
            _migrate_non_registered_tensor_attributes(module, target_device)
        for owner, _ in nested_device_snapshots:
            if torch.device(owner.device) != target_device:
                owner.to_device(target_device)
        logger.debug("Migrated functor %s to device %s", functor.name, target_device)
        return fence_record
    except BaseException as exc:
        rollback_errors: list[BaseException] = []
        try:
            torch.nn.Module.to(functor, original_device)
            for module in functor.modules():
                _migrate_non_registered_tensor_attributes(module, original_device)
                if hasattr(module, "device"):
                    object.__setattr__(module, "device", original_device)
        except BaseException as rollback_exc:
            rollback_errors.append(rollback_exc)
        try:
            _restore_nested_device_owners(nested_device_snapshots)
        except BaseException as rollback_exc:
            rollback_errors.append(rollback_exc)
        for manager, key, source_device in reversed(relocated):
            try:
                relocate_binding(manager, key, source_device)
            except BaseException as rollback_exc:
                rollback_errors.append(rollback_exc)
        if rollback_errors:
            raise BaseExceptionGroup(
                f"Failed to migrate functor '{functor.name}' to {target_device}: "
                f"device rollback to {original_device} also failed",
                [exc, *rollback_errors],
            ) from exc
        if not isinstance(exc, Exception):
            raise
        raise RuntimeError(
            f"Failed to migrate functor '{functor.name}' to {target_device}: {exc}"
        ) from exc


def register_functor(
    self: Any,
    functor: FunctorBase,
    execution_group: Optional[str] = None,
    *,
    _prepare_device: bool = True,
) -> None:
    """
    Register a scientific process functor for execution with multi-rate group assignment.

    The transaction prepares identity and validation before any mutation,
    commits the existing owners in order, migrates the device last, and
    publishes the committed provider view as the final visible action.
    The compiler's private witness mode defers only device preparation.
    """
    functor = _require_functor_base(functor)
    functor = _require_category_bridge(functor)

    base_name = functor.name
    if base_name in self.functors:
        raise ValueError(f"Functor name '{base_name}' already registered")

    _enforce_mutually_exclusive_functors(self, functor.__class__.__name__)

    # Validate output ownership and resolve the execution group before
    # mutating any simulator registry. A rejected alternative closure or an
    # invalid group declaration must not leave a functor, group member,
    # dependency-graph node, or minted provider UID behind.
    _detect_output_overlap(self, functor)
    execution_group = _resolve_execution_group(self, functor, execution_group)

    lifecycle = _lifecycle(self)
    with lifecycle.lock:
        # Prepare: identity digests and the provider UID are fixed before any
        # owner mutates, so a rejected contract never becomes a half product.
        implementation_digest, strict_identity, identity_rejection = (
            _implementation_identity(functor)
        )
        contract_digest = _provider_digest(
            _contract_payload(functor), path=f"providers.{base_name}.contract"
        )
        provider_uid = lifecycle.mint_provider_uid(set(lifecycle.provider_uids.values()))
        epoch_before = lifecycle.epoch
        device_before = _device_label(functor)
        completed: list[str] = []
        snapshot = _RegistrationSnapshot.capture(self)
        binding_digest: Optional[str] = None
        view: Optional[CommittedProviderView] = None
        edges_before = 0

        try:
            self.functors[functor.name] = functor
            completed.append(f"module_membership:add:{provider_uid}")
            self._groups[execution_group].append(functor.name)
            completed.append(f"group_membership:add:{execution_group}:{provider_uid}")

            if getattr(functor, "force_initial_execution", False):
                self._bootstrap_groups.add(execution_group)
                completed.append(f"bootstrap_group:add:{execution_group}:{provider_uid}")

            self._groups_dirty = True

            _ensure_dependency_graph(self)
            edges_before = len(self.dependency_graph.edges)
            self.dependency_graph.add_functor(functor, functor.name)
            _attach_functor_dependencies(self, functor)
            completed.append(f"dependency_graph:add:{provider_uid}")
            _refresh_execution_order(self, functor)
            completed.append("execution_order:refresh")
            self._refresh_conservation_ledger()
            completed.append("conservation_ledger:refresh")

            meta = functor.metadata
            input_variables = getattr(meta, "input_variables", [])
            output_variables = getattr(meta, "output_variables", [])
            # The contract payload above already rejected non-sequence declarations.
            n_in = len(input_variables)
            n_out = len(output_variables)
            spatial_scale = getattr(meta, "spatial_scale", None)
            scale = getattr(spatial_scale, "value", "unknown")
            category = getattr(getattr(meta, "category", None), "value", "unknown")
            logger.info(
                "Registered functor: %s in group: %s",
                functor.name,
                execution_group,
            )
            logger.info(
                "  Metadata: category=%s, inputs=%d, outputs=%d, spatial_scale=%s",
                category,
                n_in,
                n_out,
                scale,
            )
            # Device migration is the final owner commit action. All fail-prone
            # graph, dependency, ledger, and metadata work has completed, so a
            # rejected registration cannot mutate the caller-owned candidate
            # device.
            migration_fence = None
            if _prepare_device:
                migration_fence = _maybe_migrate_functor_device(self, functor)
                completed.append(f"device_migration:{provider_uid}:{_device_label(functor)}")
            else:
                completed.append(f"device_preparation_deferred:{provider_uid}")

            binding_digest = _provider_digest(
                _binding_payload(
                    functor, execution_group, getattr(self, "_substep_overrides", None)
                ),
                path=f"providers.{base_name}.binding",
            )
            lifecycle.provider_uids[functor.name] = provider_uid
            lifecycle.unpublished_providers[functor.name] = functor
            # Publication is the single point at which the new provider becomes
            # visible to readers; it is skipped when required dependencies are
            # still unmet and the registration remains committed but unpublished.
            view = _publish_provider_view(
                self,
                expected_epoch=epoch_before,
                operation="register",
                provider_uid=provider_uid,
                new_uids={functor.name: provider_uid},
            )
            if view is not None:
                completed.append(f"provider_view:publish:{view.epoch}")
                # Cross-group promotion at publication may move the new provider
                # to another group; the receipt records the published binding.
                published_identity = view.providers[functor.name]
                execution_group = published_identity.execution_group
                binding_digest = published_identity.binding_digest
        except BaseException as exc:
            receipt = LifecycleReceipt(
                receipt_id=uuid4().hex,
                operation="register",
                provider_uid=provider_uid,
                provider_name=base_name,
                implementation_digest=implementation_digest,
                contract_digest=contract_digest,
                binding_digest=binding_digest,
                epoch_before=epoch_before,
                epoch_after=None,
                view_published=False,
                managed_delta={"device": {"before": device_before, "after": device_before}},
                completed_actions=tuple(completed),
                compensated_actions=tuple(reversed(completed)),
                recoverable_resources=_REGISTRATION_RESOURCES,
                status="rolled_back",
                error=f"{type(exc).__name__}: {exc}",
            )
            try:
                snapshot.restore(self, exc)
            except BaseException as compensation_exc:
                receipt.status = "compensation_incomplete"
                receipt.error = _describe_exception(compensation_exc)
                _append_receipt(self, receipt)
                raise
            _append_receipt(self, receipt)
            raise

        _append_receipt(
            self,
            LifecycleReceipt(
                receipt_id=uuid4().hex,
                operation="register",
                provider_uid=provider_uid,
                provider_name=base_name,
                implementation_digest=implementation_digest,
                contract_digest=contract_digest,
                binding_digest=binding_digest,
                epoch_before=epoch_before,
                epoch_after=None if view is None else view.epoch,
                view_published=view is not None,
                managed_delta={
                    "module": {"added": base_name},
                    "group": {"name": execution_group, "added": base_name},
                    "bootstrap_group": bool(getattr(functor, "force_initial_execution", False)),
                    "dependency_graph": {
                        "node_added": base_name,
                        "edges_added": len(self.dependency_graph.edges) - edges_before,
                    },
                    "execution_order": {
                        "before": snapshot.attributes["execution_order"],
                        "after": (
                            None if self.execution_order is None
                            else list(self.execution_order)
                        ),
                    },
                    "conservation_ledger": "refreshed",
                    "device": {
                        "before": device_before,
                        "after": _device_label(functor),
                        "source_fence": migration_fence,
                    },
                    "provider_view": {
                        "published": view is not None,
                        "epoch": None if view is None else view.epoch,
                        "unresolved_required": (
                            [] if view is None else list(view.unresolved_required)
                        ),
                        "retired_view": (
                            None if view is None or epoch_before == 0
                            else lifecycle.device_fence_status(epoch_before)
                        ),
                    },
                },
                completed_actions=tuple(completed),
                compensated_actions=(),
                recoverable_resources=_REGISTRATION_RESOURCES,
                status="committed",
                error=None,
            ),
        )


def unregister_functor(
    self: Any,
    functor_name: str,
    *,
    strict: bool = True,
) -> bool:
    """Retire one Functor with strong exception safety.

    Retirement removes the provider from every owner, records its UID as
    retired, and publishes the successor view. The retired view is disposed
    only after its last reader token is released.
    """
    if functor_name not in self.functors:
        if strict:
            raise KeyError(f"Functor '{functor_name}' is not registered")
        return False

    lifecycle = _lifecycle(self)
    with lifecycle.lock:
        functor = self.functors[functor_name]
        provider_uid = lifecycle.provider_uids.get(functor_name)
        previous_view = lifecycle.committed_view
        identity = (
            previous_view.providers.get(functor_name)
            if previous_view is not None else None
        )
        if identity is not None and identity.functor is not functor:
            identity = None
        epoch_before = lifecycle.epoch
        completed: list[str] = []
        snapshot = _RegistrationSnapshot.capture(self)
        view: Optional[CommittedProviderView] = None
        uid_label = provider_uid or "unpublished"

        try:
            if self.dependency_graph is not None:
                if functor_name not in self.dependency_graph.nodes:
                    raise RuntimeError(
                        "Cannot unregister Functor because the dependency graph is "
                        f"inconsistent: '{functor_name}' is not present"
                    )
                self.dependency_graph.remove_functor(
                    functor_name,
                    rebuild_graph=False,
                )
                completed.append(f"dependency_graph:remove:{uid_label}")

            del self.functors[functor_name]
            completed.append(f"module_membership:remove:{uid_label}")
            for group_name, members in self._groups.items():
                if functor_name in members:
                    self._groups[group_name] = [
                        name for name in members if name != functor_name
                    ]
                    completed.append(f"group_membership:remove:{group_name}:{uid_label}")

            self.execution_order = [
                name for name in (self.execution_order or []) if name != functor_name
            ]
            for attribute in _PROVIDER_ORDER_ATTRIBUTES:
                setattr(
                    self,
                    attribute,
                    [name for name in getattr(self, attribute) if name != functor_name],
                )
            self._groups_dirty = True
            completed.append("execution_order:prune")

            if self.dependency_graph is not None:
                try:
                    self.execution_order = self.dependency_graph.topological_sort()
                except Exception as exc:  # noqa: BLE001 - rollback every late failure
                    raise RuntimeError(
                        "Failed to rebuild dependency graph after removing "
                        f"'{functor_name}': {exc}"
                    ) from exc
                completed.append("execution_order:refresh")

            self._refresh_conservation_ledger()
            completed.append("conservation_ledger:refresh")

            lifecycle.provider_uids.pop(functor_name, None)
            lifecycle.unpublished_providers.pop(functor_name, None)
            if provider_uid is not None:
                lifecycle.retired_uids.add(provider_uid)
                completed.append(f"provider_uid:retire:{provider_uid}")
            view = _publish_provider_view(
                self,
                expected_epoch=epoch_before,
                operation="unregister",
                provider_uid=uid_label,
                new_uids={},
            )
            if view is not None:
                completed.append(f"provider_view:publish:{view.epoch}")
        except BaseException as exc:
            receipt = LifecycleReceipt(
                receipt_id=uuid4().hex,
                operation="unregister",
                provider_uid=provider_uid,
                provider_name=functor_name,
                implementation_digest=(
                    None if identity is None else identity.implementation_digest
                ),
                contract_digest=None if identity is None else identity.contract_digest,
                binding_digest=None if identity is None else identity.binding_digest,
                epoch_before=epoch_before,
                epoch_after=None,
                view_published=False,
                managed_delta={},
                completed_actions=tuple(completed),
                compensated_actions=tuple(reversed(completed)),
                recoverable_resources=_RETIREMENT_RESOURCES,
                status="rolled_back",
                error=f"{type(exc).__name__}: {exc}",
            )
            try:
                snapshot.restore(self, exc)
            except BaseException as compensation_exc:
                receipt.status = "compensation_incomplete"
                receipt.error = _describe_exception(compensation_exc)
                _append_receipt(self, receipt)
                raise
            _append_receipt(self, receipt)
            raise

    _append_receipt(
        self,
        LifecycleReceipt(
            receipt_id=uuid4().hex,
            operation="unregister",
            provider_uid=provider_uid,
            provider_name=functor_name,
            implementation_digest=(
                None if identity is None else identity.implementation_digest
            ),
            contract_digest=None if identity is None else identity.contract_digest,
            binding_digest=None if identity is None else identity.binding_digest,
            epoch_before=epoch_before,
            epoch_after=None if view is None else view.epoch,
            view_published=view is not None,
            managed_delta={
                "module": {"removed": functor_name},
                "group": {
                    "removed": functor_name,
                    "names": [
                        name for name, members in snapshot.attributes["_groups"].items()
                        if functor_name in members
                    ],
                },
                "dependency_graph": {"node_removed": functor_name},
                "execution_order": {
                    "before": snapshot.attributes["execution_order"],
                    "after": list(self.execution_order or []),
                },
                "conservation_ledger": "refreshed",
                "provider_view": {
                    "published": view is not None,
                    "epoch": None if view is None else view.epoch,
                    "retired_epoch": None if previous_view is None else previous_view.epoch,
                    "unresolved_required": (
                        [] if view is None else list(view.unresolved_required)
                    ),
                    "retired_view": (
                        None if previous_view is None
                        else lifecycle.device_fence_status(previous_view.epoch)
                    ),
                },
            },
            completed_actions=tuple(completed),
            compensated_actions=(),
            recoverable_resources=_RETIREMENT_RESOURCES,
            status="committed",
            error=None,
        ),
    )
    logger.info("Unregistered functor: %s", functor_name)
    return True


def _detect_output_overlap(
    simulator: Any,
    new_functor: FunctorBase,
) -> None:
    """
    Detect conflicts when multiple functors declare the same output variable.

    Diagnostic variables (prefixed with 'diagnostics.') are excluded because
    they are allowed to be written by multiple functors without conflict.

    The behaviour is controlled by ``simulator.strict_dependencies``:
    * True  -> raise ``ValueError`` on the first conflict
    * False -> emit ``logger.warning`` for each conflict
    """
    new_outputs = set(getattr(new_functor.metadata, "output_variables", []) or [])
    if not new_outputs:
        return

    new_policy = _resolve_conflict_policy(
        getattr(
            new_functor.metadata,
            "conflict_resolution_policy",
            ConflictResolutionPolicy.INHERIT,
        )
    )

    for existing_name, existing_functor in simulator.functors.items():
        if existing_name == new_functor.name:
            continue
        overlap = _compute_unmanaged_overlap(
            new_functor=new_functor,
            existing_name=existing_name,
            existing_functor=existing_functor,
            new_outputs=new_outputs,
        )
        if overlap:
            _enforce_overlap_policy(
                simulator=simulator,
                new_functor=new_functor,
                existing_functor=existing_functor,
                existing_name=existing_name,
                overlap=overlap,
                new_policy=new_policy,
            )


def _resolve_conflict_policy(policy: Any) -> ConflictResolutionPolicy:
    if isinstance(policy, ConflictResolutionPolicy):
        return policy
    return ConflictResolutionPolicy.INHERIT


def _compute_unmanaged_overlap(
    *,
    new_functor: FunctorBase,
    existing_name: str,
    existing_functor: FunctorBase,
    new_outputs: set[str],
) -> set[str]:
    existing_outputs = set(
        getattr(existing_functor.metadata, "output_variables", []) or []
    )
    overlap = new_outputs & existing_outputs
    overlap = {v for v in overlap if not v.startswith("diagnostics.")}
    unmanaged_overlap: set[str] = set()
    for variable in overlap:
        overlap_policy = get_output_overlap_policy(variable)
        if overlap_policy == OVERLAP_POLICY_AGGREGATED:
            logger.info(
                "Governed aggregated overlap accepted between '%s' and '%s' for '%s'",
                new_functor.name,
                existing_name,
                variable,
            )
            continue
        if overlap_policy == OVERLAP_POLICY_MUTUALLY_EXCLUSIVE:
            raise ValueError(
                "Mutually exclusive overlap variable '%s' detected between "
                "functor '%s' and '%s'. Register only one writer."
                % (variable, new_functor.name, existing_name)
            )
        unmanaged_overlap.add(variable)
    return unmanaged_overlap


def _enforce_overlap_policy(
    *,
    simulator: Any,
    new_functor: FunctorBase,
    existing_functor: FunctorBase,
    existing_name: str,
    overlap: set[str],
    new_policy: ConflictResolutionPolicy,
) -> None:
    existing_policy = _resolve_conflict_policy(
        getattr(
            existing_functor.metadata,
            "conflict_resolution_policy",
            ConflictResolutionPolicy.INHERIT,
        )
    )
    critical_overlap = overlap & _CRITICAL_SINGLE_WRITER_OUTPUTS
    msg = (
        f"Output variable overlap detected: functor '{new_functor.name}' "
        f"and '{existing_name}' both declare outputs: {sorted(overlap)}"
    )
    new_kind = getattr(new_functor.metadata, "variable_kind", None)
    existing_kind = getattr(existing_functor.metadata, "variable_kind", None)
    if new_kind == VariableKind.STATE and existing_kind == VariableKind.STATE:
        raise ValueError(
            f"{msg}. STATE outputs require a single writer or an explicit "
            "overlap governance policy."
        )
    if critical_overlap:
        raise ValueError(
            f"{msg}. Critical single-writer outputs "
            f"must be unique: {sorted(critical_overlap)}"
        )

    force_error = (
        new_policy in {
            ConflictResolutionPolicy.ERROR,
            ConflictResolutionPolicy.EXCLUSIVE,
        }
        or existing_policy
        in {
            ConflictResolutionPolicy.ERROR,
            ConflictResolutionPolicy.EXCLUSIVE,
        }
    )
    force_warn = (
        new_policy == ConflictResolutionPolicy.WARN
        or existing_policy == ConflictResolutionPolicy.WARN
    )

    if force_error or (
        not force_warn and getattr(simulator, "strict_dependencies", False)
    ):
        raise ValueError(msg)
    logger.warning(msg)
