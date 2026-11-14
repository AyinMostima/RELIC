"""Equilibrium (Freundlich) sorption with dissolved-phase degradation as an index-1 DAE.

Theory document: docs/source/theory/pedosphere_equilibrium_sorption.md

The total solute mass per bulk soil volume y = theta C + rho_b S is the
prognostic (differential) state; the dissolved concentration C and the
sorbed concentration S are algebraic states tied to y by the mass partition
and the Freundlich isotherm S = K_f C^n (Jury and Horton 2004).  The
dissolved phase degrades with first-order rate lambda.  The system is a
semi-explicit index-1 DAE: g_z = theta + rho_b K_f n C^{n-1} > 0, so the
algebraic variables can be eliminated by the implicit function theorem
(Hairer and Wanner 1996).  Every step solves the scalar implicit equation
for C(t+dt) with the existing bracketed bisection owner (with implicit
parameter gradients), verifies the consistent initial triple, and reports the
constraint residual instead of silently projecting.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor

from ....computation.solvers.base import SolverStatus
from ....computation.solvers.nonlinear import BracketedBisectionSolver
from ....core.abstractions import FluxDict, StateDict
from ....core.functor.functor_bridge_utils import NamespaceDefaultMixin
from ....core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from ....core.state import UpdateItem, UpdateMode


class EquilibriumSorptionFunctor(NamespaceDefaultMixin, PhysicalFieldFunctor):
    """Index-1 equilibrium sorption DAE with implicit dissolved decay (PhysicalField contract).

    References:
      - Jury, W.A. & Horton, R. (2004). Soil Physics, 6th ed., ch. 7
        (Freundlich isotherm, retardation).
      - Mayes, M.A. et al. (2012). Soil Sci. Soc. Am. J. 76, 199-206
        (nonlinear isotherms of dissolved organic carbon in soils).
      - Hairer, E. & Wanner, G. (1996). Solving Ordinary Differential
        Equations II, ch. VI (index-1 DAEs, consistent initialization).
      - Burden & Faires (2010): bracketed bisection (existing solver owner).
    Mathematical expression:
      dy/dt = -lambda theta C,   0 = g(y, C) = theta C + rho_b K_f C^n - y,
      S = K_f C^n; implicit step: theta C' + rho_b K_f C'^n + dt lambda theta C' = y,
      y' = y - dt lambda theta C', S' = K_f C'^n.
    Parameter meanings:
      y total mass per bulk volume, C dissolved concentration, S sorbed
      concentration, theta volumetric water content, rho_b bulk density,
      K_f Freundlich coefficient, n Freundlich exponent, lambda dissolved
      decay rate, dt the simulator step.
    Units:
      y and C kg m-3; S kg kg-1; theta m3 m-3 (a volume fraction); rho_b kg m-3;
      K_f m3 kg-1 (with C
      in kg m-3); n 1; lambda s-1; dt s; degradation_flux kg m-3 s-1;
      constraint_residual kg m-3.
    Domain/constraints:
      theta > 0; y, C, S >= 0 and finite; the input triple must satisfy both
      constraints within the consistency tolerance (no silent projection);
      the scalar equation is monotone so the bracket [0, y/theta] is exact;
      solver non-convergence fails the step.
    Sign convention:
      degradation removes mass from y and from the dissolved phase; S follows
      C on the isotherm.
    """

    parameter_namespace = "pedosphere.transport"
    TOTAL_VARIABLE = "equilibrium_total_solute_mass"
    DISSOLVED_VARIABLE = "equilibrium_dissolved_concentration"
    SORBED_VARIABLE = "equilibrium_sorbed_concentration"
    # theta is the volumetric liquid soil water content, the prognostic soil_moisture state
    # [m3 m-3] (supplied as initial state or advanced by the selected soil water model).
    WATER_CONTENT_VARIABLE = "soil_moisture"
    # The forcing input of the bulk_density contract (soil_sorption_geometry_input).
    BULK_DENSITY_VARIABLE = "bulk_density"
    STATE_VARIABLES = (TOTAL_VARIABLE, DISSOLVED_VARIABLE, SORBED_VARIABLE)
    ALGEBRAIC_VARIABLES = (DISSOLVED_VARIABLE, SORBED_VARIABLE)
    DIAGNOSTICS = (
        "diagnostics.equilibrium_sorption.constraint_residual",
        "diagnostics.equilibrium_sorption.degradation_flux",
        "diagnostics.equilibrium_sorption.solver_iterations",
    )
    PARAMETERS = {
        "equilibrium_sorption_freundlich_coefficient": "Freundlich coefficient K_f [m3 kg-1]",
        "equilibrium_sorption_freundlich_exponent": "Freundlich exponent n [dimensionless]",
        "equilibrium_sorption_dissolved_decay_rate": "Dissolved-phase decay rate lambda [s-1]",
        "equilibrium_sorption_consistency_tolerance": (
            "Relative consistency tolerance [dimensionless]"
        ),
    }
    constraint_category = "descriptor_dae"
    constraint_index = 1
    elimination_method = "algebraic_elimination"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    # Explicit config overrides precede the ParameterManager (NamespaceDefaultMixin).
    parameter_lookup_source_order = ("config", "manager")

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(
        self,
        name: str = "equilibrium_sorption",
        *,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Equilibrium sorption config must be a mapping.")
        config_dict: Dict[str, Any] = dict(config) if config is not None else {}
        parameter_overrides = config_dict.pop("parameters", None)
        if parameter_overrides is None:
            parameter_overrides = {}
        if not isinstance(parameter_overrides, Mapping):
            raise ValueError("Equilibrium sorption parameter overrides must be a mapping.")
        device = kwargs.pop("device", None)

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Index-1 equilibrium Freundlich sorption DAE with implicit dissolved-phase "
                "decay solved by algebraic elimination"
            ),
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "dy/dt = -lambda theta C; 0 = theta C + rho_b K_f C^n - y; S = K_f C^n; "
                "implicit step theta C' + rho_b K_f C'^n + dt lambda theta C' = y"
            ),
            references=[
                "Jury, W.A. & Horton, R. (2004). Soil Physics, 6th ed., Wiley, ch. 7",
                "Mayes, M.A. et al. (2012). Soil Sci. Soc. Am. J. 76, 199-206",
                "Hairer, E. & Wanner, G. (1996). Solving Ordinary Differential Equations II, ch. VI",
            ],
            parameters={
                **{
                    f"{self.parameter_namespace}.{name}": text
                    for name, text in self.PARAMETERS.items()
                },
                "computation.solvers.bracketed_bisection.atol": (
                    "Absolute interval tolerance of the bisection root solve [dimensionless]"
                ),
                "computation.solvers.bracketed_bisection.rtol": (
                    "Relative interval tolerance of the bisection root solve [dimensionless]"
                ),
                "computation.solvers.bracketed_bisection.max_iterations": (
                    "Maximum bracket contractions of the bisection root solve [dimensionless]"
                ),
            },
            input_variables=[
                *self.STATE_VARIABLES, self.WATER_CONTENT_VARIABLE, self.BULK_DENSITY_VARIABLE,
            ],
            output_variables=[*self.STATE_VARIABLES, *self.DIAGNOSTICS],
            conservation_laws=["mass"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            coupling_interfaces=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        unknown = sorted(set(parameter_overrides).difference(self.PARAMETERS))
        if unknown:
            raise ValueError("Unknown equilibrium sorption parameter override(s): " + ", ".join(unknown))
        normalized_overrides: Dict[str, Any] = {
            str(key): (dict(value) if isinstance(value, Mapping) else {"value": value})
            for key, value in parameter_overrides.items()
        }
        # config["dependencies"] seeds the instance dependencies through FunctorBase
        # (the process itself declares no static Functor dependency).
        clean_config: Dict[str, Any] = {
            "parameters": normalized_overrides,
            "dependencies": config_dict.get("dependencies", []),
        }
        super().__init__(name=name, config=clean_config, metadata=metadata, **kwargs)
        self.optional_input_variables = []
        self._solvers: Dict[tuple[torch.dtype, str], BracketedBisectionSolver] = {}
        for parameter_name in self.PARAMETERS:
            self.get_parameter_value(parameter_name, required=True)
        if device is not None:
            self.to_device(device)

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------

    def _root_solver(self, prototype: Tensor) -> BracketedBisectionSolver:
        """Return the existing bracketed bisection owner for this dtype/device."""
        key = (prototype.dtype, str(prototype.device))
        solver = self._solvers.get(key)
        if solver is None:
            solver = BracketedBisectionSolver(
                name="equilibrium_sorption_bisection",
                device=prototype.device,
                dtype=prototype.dtype,
            )
            self._solvers[key] = solver
        return solver

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Report the total-mass ledger residual |Delta y + dt lambda theta C'|.

        References: Hairer and Wanner (1996); RELIC conservation architecture.
        Mathematical expression: violation = max |Delta y + dt * degradation_flux|.
        Parameter meanings: Delta y the emitted increment, degradation_flux the
          emitted implicit sink.
        Units: kg m-3.
        Domain/constraints: the step duration is carried by the emitted
          diagnostic pair; a missing pair with a state update is an error.
        Sign convention: magnitude only.
        """
        if self.TOTAL_VARIABLE not in output_updates:
            return {"mass": 0.0}
        residual = getattr(self, "_last_mass_ledger_residual", None)
        if residual is None:
            raise ValueError("Equilibrium sorption updates were emitted without a mass ledger residual.")
        return {"mass": float(torch.max(torch.abs(residual)).detach())}

    # ------------------------------------------------------------------
    # Scientific process
    # ------------------------------------------------------------------
    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """Advance the DAE by one implicit step with algebraic elimination.

        References:
          - Jury and Horton (2004); Hairer and Wanner (1996); Burden and Faires (2010).
        Mathematical expression:
          F(C') = theta C' + rho_b K_f C'^n + dt lambda theta C' - y = 0 solved on
          [0, y/theta] (F(0) = -y < 0, F(y/theta) > 0, F increasing);
          y' = y - dt lambda theta C', S' = K_f C'^n.
        Parameter meanings:
          as in the class docstring.
        Units:
          as in the class docstring.
        Domain/constraints:
          consistent input triple within the relative tolerance; theta > 0;
          y == 0 cells are exact zeros without a solve; non-convergence is a
          RuntimeError.
        Sign convention:
          as in the class docstring.
        """
        # References: Jury and Horton (2004) ch. 7; Mayes et al. (2012); Hairer and
        #   Wanner (1996) ch. VI (index-1 DAE); Burden and Faires (2010) bisection.
        # Mathematical expression: theta C' + rho_b K_f C'^n + dt lambda theta C' = y,
        #   y' = y - dt lambda theta C', S' = K_f C'^n (backward Euler on y, algebraic
        #   states re-solved by bracketed bisection on [0, y/theta]).
        # Parameter meanings: y total mass, C dissolved, S sorbed, theta water
        #   content, rho_b bulk density, K_f and n Freundlich, lambda decay rate.
        # Units: y, C kg m-3; S kg kg-1; theta m3 m-3; rho_b kg m-3; K_f m3 kg-1; n 1;
        #   lambda s-1; dt s.
        # Domain/constraints: 0 < theta <= 1; y, C, S >= 0 finite; consistent input
        #   triple within tolerance; non-convergence raises.
        # Sign convention: degradation removes mass from y and the dissolved phase.

        if kwargs:
            raise TypeError(f"Unsupported equilibrium sorption forward arguments: {sorted(kwargs)}")
        for name in (*self.STATE_VARIABLES, self.WATER_CONTENT_VARIABLE, self.BULK_DENSITY_VARIABLE):
            if name not in state:
                raise ValueError(f"{name} is required for the equilibrium sorption DAE")
            value = state[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"State '{name}' must be a torch.Tensor")
            if not value.is_floating_point():
                raise ValueError(f"State '{name}' must be a floating-point tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"State '{name}' must contain only finite values")
        total = state[self.TOTAL_VARIABLE]
        dissolved = state[self.DISSOLVED_VARIABLE]
        sorbed = state[self.SORBED_VARIABLE]
        theta = state[self.WATER_CONTENT_VARIABLE]
        if not (total.shape == dissolved.shape == sorbed.shape == theta.shape):
            raise ValueError("Equilibrium sorption states and water content must share one shape")
        for name, value in ((self.TOTAL_VARIABLE, total), (self.DISSOLVED_VARIABLE, dissolved), (self.SORBED_VARIABLE, sorbed)):
            if bool((value < 0).any()):
                raise ValueError(f"State '{name}' must be nonnegative")
        if bool((theta <= 0).any()) or bool((theta > 1).any()):
            raise ValueError(f"State '{self.WATER_CONTENT_VARIABLE}' must lie in (0, 1]")
        dt_value = float(dt)
        if not torch.isfinite(total.new_tensor(dt_value)) or dt_value <= 0.0:
            raise ValueError("dt must be finite and positive")

        coefficient = self.get_parameter_value("equilibrium_sorption_freundlich_coefficient", required=True).to(total).reshape(())
        exponent = self.get_parameter_value("equilibrium_sorption_freundlich_exponent", required=True).to(total).reshape(())
        decay = self.get_parameter_value("equilibrium_sorption_dissolved_decay_rate", required=True).to(total).reshape(())
        tolerance = float(self.get_parameter_value("equilibrium_sorption_consistency_tolerance", required=True).detach())
        for label, value in (("K_f", coefficient), ("n", exponent)):
            if not float(value.detach()) > 0.0:
                raise ValueError(f"Equilibrium sorption parameter {label} must be positive")
        if float(decay.detach()) < 0.0:
            raise ValueError("Equilibrium sorption decay rate must be nonnegative")
        bulk_density = state[self.BULK_DENSITY_VARIABLE].to(total)
        if bool((bulk_density <= 0).any()):
            raise ValueError(f"State '{self.BULK_DENSITY_VARIABLE}' must be positive")
        tiny = torch.finfo(total.dtype).tiny

        # References: Hairer and Wanner (1996), consistent initialization of index-1 DAEs.
        # Mathematical expression: |theta C + rho_b S - y| <= tol max(|y|, tiny) and
        #   |S - K_f C^n| <= tol max(|S|, tiny) on every cell.
        # Parameter meanings: tol the relative consistency tolerance.
        # Units: partition residual kg m-3; isotherm residual kg kg-1.
        # Domain/constraints: an inconsistent triple is rejected, never projected.
        # Sign convention: magnitudes only.
        partition_gap = theta * dissolved + bulk_density * sorbed - total
        isotherm_gap = sorbed - coefficient * dissolved**exponent
        if bool((partition_gap.abs() > tolerance * torch.clamp(total.abs(), min=tiny)).any()) or bool(
            (isotherm_gap.abs() > tolerance * torch.clamp(sorbed.abs(), min=tiny)).any()
        ):
            raise ValueError(
                "Equilibrium sorption requires a consistent initial triple (y, C, S): "
                f"max partition gap {float(partition_gap.abs().max()):.3e} kg m-3, "
                f"max isotherm gap {float(isotherm_gap.abs().max()):.3e} kg kg-1."
            )

        decay_dt = decay * dt_value

        def residual_fn(point: Tensor, y: Tensor, water: Tensor, rho: Tensor, kf: Tensor, n: Tensor, lam_dt: Tensor) -> Tensor:
            return water * point + rho * kf * point**n + lam_dt * water * point - y

        active = total > 0
        iterations = torch.zeros_like(total, dtype=torch.int64)
        new_dissolved = torch.zeros_like(total)
        if bool(active.any()):
            y_active = total[active]
            theta_active = theta[active]
            lower = torch.zeros_like(y_active)
            upper = y_active / theta_active
            solver = self._root_solver(total)
            result = solver.solve(
                residual_fn,
                lower,
                upper,
                parameters=(
                    y_active,
                    theta_active,
                    bulk_density,
                    coefficient,
                    exponent,
                    decay_dt,
                ),
            )
            if result.status is not SolverStatus.SUCCESS:
                raise RuntimeError(
                    "Equilibrium sorption constraint solve did not converge: "
                    f"status={result.status}, iterations={result.iterations}, "
                    f"residual_norm={result.residual_norm}."
                )
            solution = result.solution
            if not isinstance(solution, Tensor):
                raise RuntimeError("Equilibrium sorption expects a tensor root from the bracketed solver.")
            new_dissolved = torch.zeros_like(total).masked_scatter(active, solution)
            iterations = torch.where(active, torch.full_like(iterations, int(result.iterations)), iterations)
        new_sorbed = coefficient * new_dissolved**exponent
        degradation_flux = decay * theta * new_dissolved
        new_total = total - dt_value * degradation_flux
        constraint_residual = (theta * new_dissolved + bulk_density * new_sorbed - new_total).detach()
        self._last_mass_ledger_residual = ((new_total - total) + dt_value * degradation_flux).detach()

        output: FluxDict = {
            self.TOTAL_VARIABLE: UpdateItem(
                value=new_total - total,
                mode=UpdateMode.ACCUMULATE,
                unit="kg m-3",
                per_timestep=True,
                source_functor=self.name,
            ),
            self.DISSOLVED_VARIABLE: UpdateItem(
                value=new_dissolved,
                mode=UpdateMode.REPLACE,
                unit="kg m-3",
                per_timestep=True,
                source_functor=self.name,
            ),
            self.SORBED_VARIABLE: UpdateItem(
                value=new_sorbed,
                mode=UpdateMode.REPLACE,
                unit="kg kg-1",
                per_timestep=True,
                source_functor=self.name,
            ),
        }
        for key, value, unit in (
            (self.DIAGNOSTICS[0], constraint_residual.abs(), "kg m-3"),
            (self.DIAGNOSTICS[1], degradation_flux.detach(), "kg m-3 s-1"),
            (self.DIAGNOSTICS[2], iterations.to(total.dtype), "1"),
        ):
            output[key] = UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=True,
                source_functor=self.name,
            )
        return output


__all__ = ["EquilibriumSorptionFunctor"]
