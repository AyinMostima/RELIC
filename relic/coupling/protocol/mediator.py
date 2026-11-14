"""
NUOPC-style mediator interface.
"""

from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Set

from .state import State


class InitPhase(Enum):
    """
    NUOPC-style initialization phases.

    References
    ----------
    - ESMF 8.9.1 NUOPC_refdoc, Section 2.3 Initialization Phases

    Mathematical Expression
    -----------------------
    Phase transition: P_i -> P_{i+1} only if validate(P_i) is True.

    Parameters
    ----------
    UNCONFIGURED
        Initial state before any configuration.
    ADVERTISED
        Fields have been declared (Advertise phase complete).
    REALIZED
        Fields bound to data containers (Realize phase complete).
    DATA_INITIALIZED
        Initial data values set (DataInitialize phase complete).
    READY
        All phases complete, ready for execution.
    """

    UNCONFIGURED = auto()
    ADVERTISED = auto()
    REALIZED = auto()
    DATA_INITIALIZED = auto()
    READY = auto()


class InitPhaseError(RuntimeError):
    """Raised when an initialization phase transition is invalid."""


class Mediator:
    """
    Minimal mediator for exchanging import and export states.

    Extended with NUOPC-style initialization phases.

    Parameters
    ----------
    import_state : State
        Incoming state container.
    export_state : State
        Outgoing state container.
    """

    def __init__(self, import_state: State, export_state: State) -> None:
        self.import_state = import_state
        self.export_state = export_state
        self._initialized = False
        self._phase = InitPhase.UNCONFIGURED
        self._advertised_fields: Set[str] = set()
        self._realized_fields: Set[str] = set()
        self._phase_callbacks: Dict[InitPhase, Callable[[], None]] = {}

    @property
    def phase(self) -> InitPhase:
        """Return the current initialization phase."""
        return self._phase

    def register_phase_callback(self, phase: InitPhase, callback: Callable[[], None]) -> None:
        """
        Register a callback invoked when entering a phase.

        References
        ----------
        NUOPC SetServices phase registration pattern.
        """
        self._phase_callbacks[phase] = callback

    def _enter_phase(self, phase: InitPhase) -> None:
        """Set phase and run registered callback."""
        self._phase = phase
        callback = self._phase_callbacks.get(phase)
        if callback is not None:
            callback()

    def advertise(self, field_names: Optional[Set[str]] = None) -> None:
        """
        NUOPC Advertise phase: declare fields that will be exchanged.

        References
        ----------
        - ESMF 8.9.1 NUOPC_refdoc Section 2.3.1

        Mathematical Expression
        -----------------------
        F_advertised = F_import union F_export

        Parameters
        ----------
        field_names : set of str, optional
            Explicit field set; if None, uses all registered fields.

        Raises
        ------
        InitPhaseError
            If not in UNCONFIGURED phase.
        """
        if self._phase != InitPhase.UNCONFIGURED:
            raise InitPhaseError(
                f"advertise() requires UNCONFIGURED phase, current: {self._phase.name}"
            )

        if field_names is None:
            field_names = set(self.import_state._fields.keys()) | set(  # noqa: SLF001
                self.export_state._fields.keys()  # noqa: SLF001
            )

        self._advertised_fields = set(field_names)
        self._enter_phase(InitPhase.ADVERTISED)

    def realize(self, validate_all: bool = True) -> None:
        """
        NUOPC Realize phase: bind advertised fields to data containers.

        References
        ----------
        - ESMF 8.9.1 NUOPC_refdoc Section 2.3.2

        Mathematical Expression
        -----------------------
        For f in F_advertised: bind(f, data_container(f))

        Parameters
        ----------
        validate_all : bool, optional
            If True, require all advertised fields to be realized.

        Raises
        ------
        InitPhaseError
            If not in ADVERTISED phase or validation fails.
        """
        if self._phase != InitPhase.ADVERTISED:
            raise InitPhaseError(
                f"realize() requires ADVERTISED phase, current: {self._phase.name}"
            )

        import_fields = set(self.import_state._fields.keys())  # noqa: SLF001
        export_fields = set(self.export_state._fields.keys())  # noqa: SLF001
        realized = import_fields | export_fields

        if validate_all:
            missing = self._advertised_fields - realized
            if missing:
                raise InitPhaseError(f"Unrealized fields after realize(): {sorted(missing)}")

        self._realized_fields = realized
        self._enter_phase(InitPhase.REALIZED)

    def data_initialize(self, initial_data: Optional[Dict[str, Any]] = None) -> None:
        """
        NUOPC DataInitialize phase: set initial field values.

        References
        ----------
        - ESMF 8.9.1 NUOPC_refdoc Section 2.3.3

        Mathematical Expression
        -----------------------
        For f in F_realized: f.data = initial_value(f)

        Parameters
        ----------
        initial_data : dict, optional
            Mapping of field_name to initial_value.

        Raises
        ------
        InitPhaseError
            If not in REALIZED phase.
        """
        if self._phase != InitPhase.REALIZED:
            raise InitPhaseError(
                f"data_initialize() requires REALIZED phase, current: {self._phase.name}"
            )

        if initial_data:
            self.import_state.import_state(initial_data)

        self._enter_phase(InitPhase.DATA_INITIALIZED)

    def complete_initialization(self) -> None:
        """
        Complete initialization sequence and transition to READY phase.

        This method can be called after data_initialize() or directly
        if manual phase transitions have been completed.
        """
        if self._phase == InitPhase.DATA_INITIALIZED:
            self._initialized = True
            self._enter_phase(InitPhase.READY)
            return
        if self._phase == InitPhase.READY:
            return
        raise InitPhaseError(
            f"complete_initialization() requires DATA_INITIALIZED phase, current: {self._phase.name}"
        )

    def initialize(self) -> None:
        """
        Compatibility initialization method that orchestrates all phases.

        Sequence:
            UNCONFIGURED -> ADVERTISED -> REALIZED -> DATA_INITIALIZED -> READY
        """
        if self._phase == InitPhase.UNCONFIGURED:
            self.advertise()
        if self._phase == InitPhase.ADVERTISED:
            self.realize(validate_all=False)
        if self._phase == InitPhase.REALIZED:
            self.data_initialize()
        if self._phase == InitPhase.DATA_INITIALIZED:
            self.complete_initialization()
        self._initialized = True

    def advance(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Advance the mediator one step.

        Parameters
        ----------
        dt : float, optional
            Time step in seconds.

        Returns
        -------
        dict
            Export payloads keyed by field name.
        """
        if self._phase != InitPhase.READY and not self._initialized:
            raise RuntimeError(
                f"Mediator not ready (phase={self._phase.name}, initialized={self._initialized})"
            )
        _ = dt
        return self.export_state.export_state()

    def finalize(self) -> None:
        """Finalize the mediator lifecycle."""
        self._initialized = False
        self._phase = InitPhase.UNCONFIGURED
        self._advertised_fields.clear()
        self._realized_fields.clear()
