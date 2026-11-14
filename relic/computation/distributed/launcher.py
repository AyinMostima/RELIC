"""
Distributed Computing Module for RELIC

This module provides distributed computing capabilities with support for 
multi-node and multi-GPU training/simulation. Focuses on Windows-friendly
Gloo backend with comprehensive error handling and explicit policy.

Key Features:
    - Windows distributed computing (Gloo backend)
    - Multi-GPU support with NCCL on Linux
    - Process group management and cleanup
    - Distributed data parallel (DDP) integration
    - Automatic rank and world size detection
    - Graceful error handling and process cleanup

Mathematical Considerations:
    - Gradient synchronization across distributed processes
    - All-reduce operations for parameter averaging
    - Distributed ensemble simulations for uncertainty quantification
    - Load balancing across heterogeneous compute nodes
    - Consistent random number generation across processes

References:
    - PyTorch Distributed documentation
    - Gloo collective communication library
    - Distributed training best practices
    - Windows-specific distributed computing considerations
"""

import os
import logging
import socket
import time
import uuid
from datetime import timedelta
from typing import Dict, Any, Optional, Callable, List, Tuple
from contextlib import contextmanager
from functools import wraps
import threading
from enum import Enum
from dataclasses import asdict, dataclass, field, replace

# Required PyTorch distributed runtime
import torch
from relic.core import device_policy
import torch.distributed as dist
TORCH_AVAILABLE = True
DISTRIBUTED_AVAILABLE = dist.is_available()

# Configure logging
logger = logging.getLogger(__name__)

# Global distributed state
_DISTRIBUTED_STATE = {
    'initialized': False,
    'rank': -1,
    'world_size': -1,
    'backend': None,
    'master_addr': None,
    'master_port': None,
    'local_rank': -1
}


class DistributedBackend(Enum):
    """Available distributed backends."""
    GLOO = "gloo"
    NCCL = "nccl"  
    MPI = "mpi"


@dataclass
class DistributedConfig:
    """Configuration for distributed computing setup.

    The single owner of the launch defaults (software and deployment settings,
    not scientific parameters): ProcessGroupManager, init_distributed and
    setup_distributed_simulation take a DistributedConfig or overrides of it.
    """
    backend: str = "gloo"  # Windows-friendly default
    master_addr: str = "localhost"
    master_port: int = 29500
    world_size: int = 1
    rank: Optional[int] = None
    timeout_minutes: int = 30
    init_method: Optional[str] = None
    
    # GPU settings
    local_rank: Optional[int] = None
    # Distributed default: prefer GPU; fallback to CPU when CUDA is unavailable.
    use_cuda: bool = True


class DistributedError(Exception):
    """Exception raised for distributed computing errors."""


class ProcessGroupManager:
    """
    Manager for PyTorch distributed process groups (v2).
    
    Handles initialization, cleanup, and monitoring of distributed
    process groups with Windows configurations.
    Mathematical Expression:
        Distributed Communication: AllReduce(x) = Sigma(x_i) / N across N processes
        Broadcast: x_all = x_src for synchronization
        
    References:
        - PyTorch Distributed documentation
        - Gloo collective communication library
        - Distributed training best practices
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        # The construction configuration (DistributedConfig owns the defaults);
        # initialize() starts every cycle from a copy of it.
        self._init_config = config if config is not None else DistributedConfig()
        
        # Core attributes (re-created in _do_initialize)
        self._lock: threading.Lock = threading.Lock()
        self._cleanup_callbacks: List[Callable] = []
        self._initialized = False
        # Configuration of the initialized group (None while not initialized).
        self._active_config: Optional[DistributedConfig] = None
        self._config: DistributedConfig = replace(self._init_config)
        self.initialize()

    def initialize(self) -> None:
        """Initialize ProcessGroupManager components."""
        self._do_initialize()

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure ProcessGroupManager with provided settings."""
        self._do_configure(config)

    def validate(self) -> List[str]:
        """Validate ProcessGroupManager configuration."""
        return self._do_validate()

    def cleanup(self) -> None:
        """Clean up ProcessGroupManager resources."""
        self._do_cleanup()

    def _do_initialize(self) -> None:
        """Perform ProcessGroupManager-specific initialization."""
        self._lock = threading.Lock()
        self._cleanup_callbacks = []
        self._initialized = False
        self._active_config = None
        
        # Start from a copy of the construction configuration
        self._config = replace(self._init_config)
    
    def _do_configure(self, config: Dict[str, Any]) -> None:
        """Configure ProcessGroupManager with provided settings."""
        settings = config or {}
        if "backend" in settings:
            self._config.backend = settings["backend"]
        
        if "master_addr" in settings:
            self._config.master_addr = settings["master_addr"]
        
        if "master_port" in settings:
            self._config.master_port = settings["master_port"]
        
        if "world_size" in settings:
            self._config.world_size = settings["world_size"]
        
        if "timeout_minutes" in settings:
            self._config.timeout_minutes = settings["timeout_minutes"]
    
    def _do_validate(self) -> List[str]:
        """Validate ProcessGroupManager configuration."""
        errors = []
        
        if not TORCH_AVAILABLE:
            errors.append("PyTorch not available - distributed computing disabled")
        
        if not DISTRIBUTED_AVAILABLE:
            errors.append("PyTorch distributed not available")
        
        if DISTRIBUTED_AVAILABLE:
            # Check available backends
            available_backends = self._get_available_backends()
            if self._config.backend not in available_backends:
                errors.append(f"Backend {self._config.backend} not available. Available: {available_backends}")
        
        if self._config.world_size < 1:
            errors.append("World size must be at least 1")
        
        if self._config.master_port < 1024 or self._config.master_port > 65535:
            errors.append("Master port must be between 1024 and 65535")
        
        return errors
    
    def _do_cleanup(self) -> None:
        """Clean up ProcessGroupManager resources."""
        # Destroy process group if initialized
        if self._initialized:
            self.destroy_process_group()
        
        # Clear cleanup callbacks
        if hasattr(self, '_cleanup_callbacks'):
            self._cleanup_callbacks.clear()
    
    def _get_available_backends(self) -> List[str]:
        """Get list of available distributed backends."""
        if not DISTRIBUTED_AVAILABLE:
            return []
        
        backends = []
        if dist.is_gloo_available():
            backends.append("gloo")
        if dist.is_nccl_available():
            backends.append("nccl")
        if dist.is_mpi_available():
            backends.append("mpi")
        
        return backends
    
    def _find_free_port(self) -> int:
        """Find a free port for master process communication."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def _auto_detect_rank_and_world_size(self) -> Tuple[int, int]:
        """Auto-detect rank and world size from environment variables."""
        # Check for torchrun/torchx environment variables
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            return rank, world_size
        
        # Check for SLURM environment
        if 'SLURM_PROCID' in os.environ and 'SLURM_NPROCS' in os.environ:
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NPROCS'])
            return rank, world_size
        
        # Default to single process
        return 0, 1
    
    def init_process_group(self, config: DistributedConfig) -> bool:
        """
        Initialize distributed process group.
        
        Parameters:
            config : DistributedConfig
                Distributed computing configuration
        
        Returns:
            bool: True once the process group is initialized.

        Raises:
            DistributedError: If torch.distributed is unavailable, the requested
                backend is not available (it is never replaced by another), or
                a GPU local rank has no visible CUDA device. Initialization
                errors propagate after the environment is cleaned up.
        """
        if not DISTRIBUTED_AVAILABLE:
            raise DistributedError("PyTorch distributed (torch.distributed) is not available")

        with self._lock:
            # Auto-detect rank and world size if not specified
            if config.rank is None:
                rank, world_size = self._auto_detect_rank_and_world_size()
                config.rank = rank
                if config.world_size == 1:  # Only override if default
                    config.world_size = world_size

            if self._initialized:
                # Idempotent for the configuration the group was initialized
                # with; any other configuration is a conflicting request.
                active = asdict(self._active_config) if self._active_config is not None else {}
                requested = asdict(config)
                differing = sorted(key for key in requested if requested[key] != active.get(key))
                if differing:
                    raise DistributedError(
                        "Process group already initialized with a different configuration; "
                        f"differing fields: {differing}"
                    )
                return True

            # The requested backend is used or the request fails (the default
            # "gloo" is DistributedConfig's declared backend, not a substitute).
            available_backends = self._get_available_backends()
            if config.backend not in available_backends:
                raise DistributedError(
                    f"Requested distributed backend '{config.backend}' is not available; "
                    f"available backends: {available_backends}"
                )

            # Device policy (relic.core.device_policy): GPU first, CPU when CUDA
            # is unavailable. With CUDA available, the rank's device index must
            # exist; it is checked before any process-group or environment change.
            cuda_device_id: Optional[int] = None
            if config.use_cuda and device_policy.is_cuda_available():
                cuda_device_id = (
                    config.local_rank
                    if config.local_rank is not None
                    else config.rank
                )
                device_count = torch.cuda.device_count()
                if not 0 <= cuda_device_id < device_count:
                    raise DistributedError(
                        f"local_rank {cuda_device_id} has no visible CUDA device "
                        f"({device_count} visible); set use_cuda=False to run this rank on the CPU"
                    )

            try:
                # Set up environment variables
                os.environ['MASTER_ADDR'] = config.master_addr
                os.environ['MASTER_PORT'] = str(config.master_port)
                os.environ['RANK'] = str(config.rank)
                os.environ['WORLD_SIZE'] = str(config.world_size)
                
                if config.local_rank is not None:
                    os.environ['LOCAL_RANK'] = str(config.local_rank)
                
                # Initialize process group
                init_method = config.init_method or f"tcp://{config.master_addr}:{config.master_port}"
                
                timeout = timedelta(minutes=config.timeout_minutes)
                
                dist.init_process_group(
                    backend=config.backend,
                    init_method=init_method,
                    world_size=config.world_size,
                    rank=config.rank,
                    timeout=timeout
                )
                
                # Update global state
                global _DISTRIBUTED_STATE
                _DISTRIBUTED_STATE.update({
                    'initialized': True,
                    'rank': config.rank,
                    'world_size': config.world_size,
                    'backend': config.backend,
                    'master_addr': config.master_addr,
                    'master_port': config.master_port,
                    'local_rank': (
                        config.local_rank
                        if config.local_rank is not None
                        else config.rank
                    )
                })
                
                self._initialized = True
                self._active_config = replace(config)

                logger.info(
                    f"Distributed process group initialized: "
                    f"backend={config.backend}, rank={config.rank}, "
                    f"world_size={config.world_size}"
                )
                
                if cuda_device_id is not None:
                    torch.cuda.set_device(cuda_device_id)
                    logger.info(f"Set CUDA device to {cuda_device_id}")
                elif config.use_cuda:
                    # Documented device policy: CUDA unavailable -> the rank runs on the CPU.
                    logger.warning(
                        "CUDA requested for distributed runtime but unavailable; this rank runs on the CPU."
                    )

                return True

            except BaseException:
                # Undo the environment this call set, then re-raise the cause.
                self._cleanup_failed_init()
                raise
    
    def _cleanup_failed_init(self):
        """Clean up after failed initialization."""
        env_vars_to_remove = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK']
        for var in env_vars_to_remove:
            if var in os.environ:
                del os.environ[var]
    
    def destroy_process_group(self):
        """Destroy the distributed process group."""
        with self._lock:
            if not self._initialized:
                return

            # Every cleanup callback runs even when an earlier one fails; the
            # group is then destroyed. Failures are raised together at the end
            # (ExceptionGroup); the owner state is reset only when the torch
            # group was actually destroyed.
            errors: List[Exception] = []
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as exc:  # noqa: BLE001 - collected and re-raised below
                    errors.append(exc)

            destroyed = True
            if DISTRIBUTED_AVAILABLE and dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception as exc:  # noqa: BLE001 - collected and re-raised below
                    errors.append(exc)
                    destroyed = False

            if destroyed:
                global _DISTRIBUTED_STATE
                _DISTRIBUTED_STATE.update({
                    'initialized': False,
                    'rank': -1,
                    'world_size': -1,
                    'backend': None,
                    'master_addr': None,
                    'master_port': None,
                    'local_rank': -1
                })
                self._initialized = False
                self._active_config = None
                logger.info("Distributed process group destroyed")
            if errors:
                raise ExceptionGroup("Distributed process group cleanup failed", errors)
    
    def add_cleanup_callback(self, callback: Callable):
        """Add cleanup callback to be called on process group destruction."""
        self._cleanup_callbacks.append(callback)
    
    def is_initialized(self) -> bool:
        """Check if process group is initialized."""
        return self._initialized and DISTRIBUTED_AVAILABLE and dist.is_initialized()
    
    def get_rank(self) -> int:
        """Get current process rank."""
        if self.is_initialized():
            return dist.get_rank()
        return 0
    
    def get_world_size(self) -> int:
        """Get total number of processes."""
        if self.is_initialized():
            return dist.get_world_size()
        return 1
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized():
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=None):
        """All-reduce operation across all processes."""
        if not self.is_initialized():
            return tensor
        
        if op is None:
            op = dist.ReduceOp.SUM if hasattr(dist, 'ReduceOp') else dist.reduce_op.SUM
        
        dist.all_reduce(tensor, op=op)
        return tensor
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """Broadcast tensor from source rank to all ranks."""
        if self.is_initialized():
            dist.broadcast(tensor, src=src)
        return tensor


# ---------------------------------------------------------------------------
# Coordinated transactions (E02): prepare / decision / apply across ranks
# ---------------------------------------------------------------------------

COORDINATION_SCHEMA_VERSION = 1


@dataclass
class CoordinationOutcome:
    """Ledger of one coordinated transaction as observed by this rank.

    Failure model (explicit, not a promise of atomicity or liveness):

    - participants are the ranks of the initialized process group; a rank may
      crash-stop at any point (no Byzantine behaviour is tolerated);
    - every phase boundary is one ``all_gather_object`` collective; a peer
      that stops responding surfaces as a collective error or the process
      group timeout, and the local outcome is ``partition`` for the phase in
      which it happened (bounded wait, no unbounded liveness guarantee);
    - the decision is a pure function of the gathered votes, so ranks that
      completed the vote gather compute identical decisions; the decision
      gather verifies that and refuses to apply on divergence;
    - after a ``commit`` decision an apply failure on any rank leaves the
      transaction ``commit_partial``; ranks whose apply succeeded run the
      compensation and the transaction ends ``compensated`` or
      ``compensation_incomplete``. A rank that loses contact after applying
      reports ``partition`` with ``applied=True`` so the operator can see the
      unresolved state; nothing is silently rolled forward or back.

    Collective order per transaction: vote gather, decision gather, apply
    gather, and (only after an apply failure) compensation gather.
    """

    transaction_id: str
    operation: str
    rank: int
    world_size: int
    phase: str = "prepare"
    decision: str = "undecided"
    status: str = "undecided"
    applied: bool = False
    compensated: Optional[bool] = None
    votes: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    apply_results: List[Dict[str, Any]] = field(default_factory=list)
    compensation_results: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": COORDINATION_SCHEMA_VERSION,
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "rank": self.rank,
            "world_size": self.world_size,
            "phase": self.phase,
            "decision": self.decision,
            "status": self.status,
            "applied": self.applied,
            "compensated": self.compensated,
            "votes": list(self.votes),
            "decisions": list(self.decisions),
            "apply_results": list(self.apply_results),
            "compensation_results": list(self.compensation_results),
            "reasons": list(self.reasons),
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }


_VOTE_KEYS = frozenset({"rank", "ok", "identity", "error"})


def decide_coordination(votes: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Pure decision over the gathered votes: commit only when every rank
    prepared successfully and every identity field agrees across ranks.

    References: Gray (1978) two-phase commit; Lampson and Sturgis (1979)
    crash recovery in distributed storage; Skeen (1981) nonblocking commit
    (why a coordinator-free decision needs identical inputs).
    Mathematical expression: decision = commit iff AND_r ok_r AND
    AND_k |{identity_r[k] : r}| = 1, else abort with one reason per failing
    conjunct.
    Parameter meanings: ok_r prepare outcome of rank r; identity_r[k] the
    value of identity field k (epoch, contract, binding, device, rng) on
    rank r.
    Units: none.
    Domain/constraints: votes must be the complete world (one per rank);
    an incomplete vote list aborts. Identity keys prefixed ``local_`` are
    rank-local information (for example a provider UID minted per process)
    and are recorded but never compared.
    Sign convention: not applicable.
    """
    reasons: List[str] = []
    for vote in votes:
        missing = sorted(_VOTE_KEYS - set(vote))
        if missing:
            raise KeyError(f"coordination vote lacks {missing}: {vote!r}")
    ranks = sorted(int(vote["rank"]) for vote in votes)
    if not votes or ranks != list(range(len(votes))):
        reasons.append(f"incomplete votes: ranks {ranks}")
    ordered = sorted(votes, key=lambda item: int(item["rank"]))
    for vote in ordered:
        if not vote["ok"]:
            reasons.append(f"rank {vote['rank']} prepare failed: {vote['error']}")
    # A failed prepare votes no identity (None); an identity key absent on one
    # rank is itself a mismatch and is shown as <absent>.
    identities = [vote["identity"] or {} for vote in ordered]
    keys = sorted({key for identity in identities for key in identity if not str(key).startswith("local_")})
    for key in keys:
        values = [repr(identity[key]) if key in identity else "<absent>" for identity in identities]
        if len(set(values)) != 1:
            per_rank = ", ".join(
                f"rank {vote['rank']}={value}" for vote, value in zip(ordered, values)
            )
            reasons.append(f"{key} mismatch: {per_rank}")
    return ("commit" if not reasons else "abort"), reasons


def _describe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def process_group_world() -> Tuple[bool, int, int]:
    """(initialized, rank, world_size) of the real torch process group.

    Coordination is keyed to the actual ``torch.distributed`` group, never to
    a manager object: a mocked or half-initialized manager is not a world.
    """
    if not DISTRIBUTED_AVAILABLE or not dist.is_initialized():
        return False, 0, 1
    return True, int(dist.get_rank()), int(dist.get_world_size())


def coordinate_transaction(
    operation: str,
    *,
    prepare: Callable[[], Dict[str, Any]],
    apply: Callable[[], Any],
    compensate: Optional[Callable[[], Any]] = None,
    rollback_prepare: Optional[Callable[[], Any]] = None,
    transaction_id: Optional[str] = None,
) -> CoordinationOutcome:
    """Run one prepare/decision/apply transaction over the process group.

    ``prepare`` returns the identity payload this rank votes with (or raises
    to vote abort); ``apply`` performs the local change after a commit
    decision; ``compensate`` undoes a local apply when another rank's apply
    failed; ``rollback_prepare`` releases prepare-time resources on abort or
    partition. A single-process group runs the same phases locally. The
    outcome is always returned; the caller decides how to surface it.
    """
    initialized, rank, world_size = process_group_world()
    outcome = CoordinationOutcome(
        transaction_id=transaction_id or uuid.uuid4().hex,
        operation=operation,
        rank=rank,
        world_size=world_size,
    )
    started = time.perf_counter()

    def _gather(payload: Any, phase: str) -> Optional[List[Any]]:
        outcome.phase = phase
        if world_size == 1 or not initialized:
            return [payload]
        gathered: List[Any] = [None] * world_size
        try:
            dist.all_gather_object(gathered, payload)
        except BaseException as exc:  # noqa: BLE001 - partition is an observable outcome
            outcome.status = "partition"
            outcome.error = _describe_error(exc)
            logger.error(
                "Coordinated %s %s: contact lost during %s on rank %s: %s",
                operation, outcome.transaction_id, phase, rank, outcome.error,
            )
            if not isinstance(exc, Exception):
                raise
            return None
        return gathered

    def _finish() -> CoordinationOutcome:
        outcome.elapsed_seconds = time.perf_counter() - started
        return outcome

    def _rollback_prepare() -> None:
        if rollback_prepare is None:
            return
        try:
            rollback_prepare()
        except Exception as exc:  # noqa: BLE001 - recorded, never masks the phase
            outcome.reasons.append(f"rollback_prepare failed: {_describe_error(exc)}")

    # Phase 1: local prepare and vote gather.
    vote: Dict[str, Any] = {"rank": rank, "ok": False, "identity": None, "error": None}
    try:
        vote["identity"] = dict(prepare())
        vote["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the vote carries the failure
        vote["error"] = _describe_error(exc)
        logger.error("Coordinated %s: prepare failed on rank %s: %s", operation, rank, vote["error"])
    votes = _gather(vote, "vote_gather")
    if votes is None:
        _rollback_prepare()
        return _finish()
    outcome.votes = sorted(votes, key=lambda item: int(item["rank"]))

    # Phase 2: pure decision, then verify every rank decided identically.
    decision, reasons = decide_coordination(outcome.votes)
    outcome.decision = decision
    outcome.reasons.extend(reasons)
    decisions = _gather(decision, "decision_gather")
    if decisions is None:
        _rollback_prepare()
        return _finish()
    outcome.decisions = list(decisions)
    if any(item != decision for item in decisions):
        outcome.decision = "abort"
        outcome.status = "decision_divergence"
        outcome.reasons.append(f"decisions diverged across ranks: {decisions}")
        _rollback_prepare()
        return _finish()
    if decision != "commit":
        outcome.status = "aborted"
        _rollback_prepare()
        return _finish()

    # Phase 3: local apply and apply gather.
    apply_result: Dict[str, Any] = {"rank": rank, "ok": False, "error": None}
    try:
        apply()
        apply_result["ok"] = True
        outcome.applied = True
    except Exception as exc:  # noqa: BLE001 - reported to every rank
        apply_result["error"] = _describe_error(exc)
        logger.error("Coordinated %s: apply failed on rank %s: %s", operation, rank, apply_result["error"])
    outcome.phase = "apply"
    results = _gather(apply_result, "apply_gather")
    if results is None:
        return _finish()
    outcome.apply_results = sorted(results, key=lambda item: int(item["rank"]))
    if all(item["ok"] for item in outcome.apply_results):
        outcome.status = "committed"
        return _finish()

    # Phase 4: compensation on the ranks whose apply succeeded.
    outcome.status = "commit_partial"
    failed = [item["rank"] for item in outcome.apply_results if not item["ok"]]
    outcome.reasons.append(f"apply failed on ranks {failed}")
    compensation: Dict[str, Any] = {"rank": rank, "applied": outcome.applied, "ok": None, "error": None}
    if outcome.applied:
        if compensate is None:
            compensation["ok"] = False
            compensation["error"] = "no compensation available"
        else:
            try:
                compensate()
                compensation["ok"] = True
            except Exception as exc:  # noqa: BLE001 - incomplete compensation is reported
                compensation["ok"] = False
                compensation["error"] = _describe_error(exc)
        outcome.compensated = bool(compensation["ok"])
    compensations = _gather(compensation, "compensation_gather")
    if compensations is None:
        return _finish()
    outcome.compensation_results = sorted(compensations, key=lambda item: int(item["rank"]))
    incomplete = [
        item["rank"]
        for item in outcome.compensation_results
        if item["applied"] and not item["ok"]
    ]
    if not any(item["applied"] for item in outcome.compensation_results):
        outcome.status = "apply_failed"
    elif incomplete:
        outcome.status = "compensation_incomplete"
        outcome.reasons.append(f"compensation incomplete on ranks {incomplete}")
    else:
        outcome.status = "compensated"
    return _finish()


# Module-level process group manager
_process_group_manager = ProcessGroupManager()

def get_process_group_manager() -> ProcessGroupManager:
    """Return module-level ProcessGroupManager instance."""
    return _process_group_manager


def is_distributed() -> bool:
    """Check if distributed computing is available and initialized."""
    return _process_group_manager.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    return _process_group_manager.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    return _process_group_manager.get_world_size()


def is_master() -> bool:
    """Check if current process is master (rank 0)."""
    return get_rank() == 0


def init_distributed(**overrides: Any) -> bool:
    """
    Initialize distributed computing.

    Parameters:
        **overrides : DistributedConfig fields (backend, master_addr,
            master_port, world_size, rank, timeout_minutes, init_method,
            local_rank, use_cuda); the others keep the DistributedConfig
            defaults, and an unknown name raises TypeError.

    Returns:
        bool: True if successful
    """
    config = DistributedConfig(**overrides)
    
    return _process_group_manager.init_process_group(config)


def cleanup_distributed():
    """Clean up distributed computing resources."""
    _process_group_manager.destroy_process_group()


@contextmanager
def distributed_context(**kwargs):
    """
    Context manager for distributed computing setup.
    
    Parameters:
        **kwargs : Distributed configuration options
    
    Yields:
        bool: True if distributed is initialized
    """
    success = init_distributed(**kwargs)
    try:
        yield success
    finally:
        if success:
            cleanup_distributed()


def barrier():
    """Synchronize all distributed processes."""
    _process_group_manager.barrier()


def all_reduce(tensor: torch.Tensor, op=None):
    """All-reduce operation across all processes."""
    return _process_group_manager.all_reduce(tensor, op)


def broadcast(tensor: torch.Tensor, src: int = 0):
    """Broadcast tensor from source to all processes."""
    return _process_group_manager.broadcast(tensor, src)


def distributed_wrapper(func: Callable) -> Callable:
    """
    Decorator to make function distributed-aware.
    
    The wrapped function will only execute on rank 0 by default,
    unless specified otherwise.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if function should run on all ranks
        run_on_all_ranks = kwargs.pop('_distributed_run_on_all_ranks', False)
        
        if not is_distributed() or run_on_all_ranks or is_master():
            result = func(*args, **kwargs)
            
            # Broadcast result if distributed and this is master
            if is_distributed() and is_master() and not run_on_all_ranks:
                # Create a simple broadcast for basic return types
                if isinstance(result, (int, float)):
                    result_tensor = torch.tensor(result, dtype=torch.float32)
                    broadcast(result_tensor)
                    return result_tensor.item()
            
            return result
        else:
            # Non-master ranks wait for broadcast
            if isinstance(func, type):  # If it's expected to return a number
                result_tensor = torch.tensor(0.0, dtype=torch.float32)
                broadcast(result_tensor)
                return result_tensor.item()
            
            return None
    
    return wrapper


class DistributedSimulationLauncher:
    """
    Launcher for distributed scientific simulations.
    
    Provides utilities for launching multi-process scientific simulations
    with proper process management and result aggregation.
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self._results = {}
    
    def launch_ensemble_simulation(
        self,
        simulation_func: Callable,
        parameter_sets: List[Dict],
        allow_single_process: bool = False,
        **kwargs,
    ) -> List[Any]:
        """
        Launch ensemble simulation across distributed processes.
        
        Parameters:
            simulation_func : Callable
                Simulation function to execute
            parameter_sets : List[Dict]
                List of parameter dictionaries
            **kwargs : Additional launch options
        
        Returns:
            List[Any]: Simulation results from all processes
        """
        if not is_distributed():
            if not allow_single_process:
                raise RuntimeError("Distributed computing not initialized")
            logger.info("Running ensemble simulation in single-process mode")
            return [simulation_func(params) for params in parameter_sets]
        
        # Distribute parameter sets across ranks
        world_size = get_world_size()
        rank = get_rank()
        
        # Simple round-robin distribution
        my_parameter_sets = [
            parameter_sets[i] for i in range(rank, len(parameter_sets), world_size)
        ]
        
        logger.info(f"Rank {rank}: processing {len(my_parameter_sets)} parameter sets")
        
        # Run simulations on this rank
        my_results = []
        my_errors: List[Dict[str, Any]] = []
        my_indices = [
            rank + index * world_size
            for index in range(len(my_parameter_sets))
        ]
        for member_id, params in zip(my_indices, my_parameter_sets):
            try:
                result = simulation_func(params)
                my_results.append(result)
            except Exception as e:  # noqa: BLE001 - gather the complete failure ledger
                logger.error(
                    "Simulation member %s failed on rank %s: %s",
                    member_id,
                    rank,
                    e,
                )
                my_results.append(None)
                my_errors.append(
                    {
                        "member_id": member_id,
                        "rank": rank,
                        "type": type(e).__name__,
                        "message": str(e),
                    }
                )

        local_payload = {
            "rank": rank,
            "indices": my_indices,
            "results": my_results,
            "errors": my_errors,
        }

        def _merge_payloads(payloads: List[Optional[Dict[str, Any]]]) -> List[Any]:
            missing_ranks = [
                rank_id
                for rank_id, payload in enumerate(payloads)
                if payload is None
            ]
            if missing_ranks:
                raise RuntimeError(
                    f"Distributed ensemble missing rank payloads: {missing_ranks}"
                )

            gathered_errors = sorted(
                (
                    error
                    for payload in payloads
                    if payload is not None
                    for error in payload["errors"]
                ),
                key=lambda error: int(error["member_id"]),
            )
            if gathered_errors:
                details = "; ".join(
                    "member "
                    f"{error['member_id']} rank {error['rank']} "
                    f"{error['type']}: {error['message']}"
                    for error in gathered_errors
                )
                raise RuntimeError(f"Distributed ensemble failed: {details}")

            merged = [None] * len(parameter_sets)
            for payload in payloads:
                assert payload is not None
                indices = payload["indices"]
                values = payload["results"]
                for idx, value in zip(indices, values):
                    if 0 <= idx < len(merged):
                        merged[idx] = value
            missing_members = [
                member_id
                for member_id, value in enumerate(merged)
                if value is None
            ]
            if missing_members:
                raise RuntimeError(
                    f"Distributed ensemble missing member results: {missing_members}"
                )
            return merged

        # Use gather_object when available to collect payload only on rank-0.
        if hasattr(dist, "gather_object"):
            if is_master():
                gathered_payloads: List[Optional[Dict[str, Any]]] = [None] * world_size
                dist.gather_object(local_payload, gathered_payloads, dst=0)
                return _merge_payloads(gathered_payloads)
            dist.gather_object(local_payload, None, dst=0)
            if my_errors:
                error = my_errors[0]
                raise RuntimeError(
                    "Distributed ensemble local failure: member "
                    f"{error['member_id']} {error['type']}: {error['message']}"
                )
            return my_results

        # Fallback path for runtimes exposing all_gather_object only.
        if hasattr(dist, "all_gather_object"):
            gathered_payloads = [None] * world_size
            dist.all_gather_object(gathered_payloads, local_payload)
            if is_master():
                return _merge_payloads(gathered_payloads)
            if my_errors:
                error = my_errors[0]
                raise RuntimeError(
                    "Distributed ensemble local failure: member "
                    f"{error['member_id']} {error['type']}: {error['message']}"
                )
            return my_results

        raise RuntimeError(
            "Distributed ensemble requires gather_object or all_gather_object; "
            "partial local results are forbidden."
        )
    
    def run_single_node_multi_gpu(
        self,
        simulation_func: Callable,
        gpu_count: Optional[int] = None,
        allow_single_gpu: bool = False,
        allow_cpu: bool = False,
        **kwargs,
    ):
        """
        Run simulation on single node with multiple GPUs.
        
        Parameters:
            simulation_func : Callable
                Simulation function
            gpu_count : Optional[int]
                Number of GPUs to use
            **kwargs : Additional options
        """
        if not device_policy.is_cuda_available():
            if not allow_cpu:
                raise RuntimeError(
                    "CUDA is unavailable for single-node GPU execution; "
                    "pass allow_cpu=True to authorize CPU fallback."
                )
            logger.warning("CUDA not available; running on CPU fallback mode")
            return simulation_func(**kwargs)
        
        available_gpus = torch.cuda.device_count()
        gpu_count = gpu_count or available_gpus
        
        if gpu_count > available_gpus:
            logger.warning(f"Requested {gpu_count} GPUs, only {available_gpus} available")
            gpu_count = available_gpus
        
        # This would typically use torch.multiprocessing.spawn
        if not allow_single_gpu:
            raise RuntimeError("Multi-GPU execution not implemented")
        logger.info("Running on single GPU (multi-GPU not implemented)")
        
        if available_gpus > 0:
            with torch.cuda.device(0):
                return simulation_func(**kwargs)
        else:
            return simulation_func(**kwargs)


def distributed_healthcheck() -> Dict[str, Any]:
    """
    Perform distributed computing health check.
    
    Returns:
        Dict[str, Any]: Health check results
    """
    status = {
        'torch_available': TORCH_AVAILABLE,
        'distributed_available': DISTRIBUTED_AVAILABLE,
        'initialized': is_distributed(),
        'available_backends': [],
        'current_state': _DISTRIBUTED_STATE.copy(),
        'warnings': [],
        'recommendations': []
    }
    
    if not TORCH_AVAILABLE:
        status['warnings'].append("PyTorch not available")
        status['recommendations'].append("Install PyTorch for distributed computing")
        return status
    
    if not DISTRIBUTED_AVAILABLE:
        status['warnings'].append("PyTorch distributed not available")
        status['recommendations'].append("Install PyTorch with distributed support")
        return status
    
    # Check available backends
    if _process_group_manager:
        status['available_backends'] = _process_group_manager._get_available_backends()
    
    if not status['available_backends']:
        status['warnings'].append("No distributed backends available")
    
    # Platform-specific checks
    import platform
    if platform.system() == "Windows":
        if 'nccl' in status['available_backends']:
            status['warnings'].append("NCCL may have limited Windows support")
            status['recommendations'].append("Use Gloo backend on Windows")
        
        if 'gloo' not in status['available_backends']:
            status['warnings'].append("Gloo backend not available - required for Windows")
    
    # Check current initialization status
    if status['initialized']:
        rank = get_rank()
        world_size = get_world_size()
        status['current_rank'] = rank
        status['current_world_size'] = world_size
        
        if world_size == 1:
            status['warnings'].append("Single process - no distributed benefit")
    
    return status


# Convenience functions
def setup_distributed_simulation(**overrides: Any) -> bool:
    """Set up distributed computing for simulation (DistributedConfig overrides)."""
    return init_distributed(**overrides)


def finalize_distributed_simulation():
    """Clean up distributed resources after simulation."""
    cleanup_distributed()


# Export main classes and functions
__all__ = [
    # Classes
    'ProcessGroupManager',
    'DistributedConfig',
    'DistributedSimulationLauncher',
    'DistributedBackend',
    'DistributedError',
    
    # Functions
    'get_process_group_manager',
    'is_distributed',
    'get_rank',
    'get_world_size',
    'is_master',
    'init_distributed',
    'cleanup_distributed',
    'distributed_context',
    'barrier',
    'all_reduce',
    'broadcast',
    'distributed_wrapper',
    'distributed_healthcheck',
    'setup_distributed_simulation',
    'finalize_distributed_simulation',
    'CoordinationOutcome',
    'coordinate_transaction',
    'decide_coordination',
    'process_group_world',
]
