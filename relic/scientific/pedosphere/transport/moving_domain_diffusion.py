"""Moving-domain (ALE) column diffusion with a prescribed boundary velocity.

Theory document: docs/source/theory/pedosphere_moving_domain_diffusion.md

A one-dimensional column [0, L(t)] (for example the unsaturated zone above
a moving water table or a growing root zone) is discretized on the fixed
reference coordinate xi in [0, 1] through the uniformly stretched mapping
x = xi L(t).  The mesh velocity is w(x, t) = xi dL/dt and the lower boundary
moves with the prescribed velocity w_b = dL/dt.  Cell masses are advanced in
the arbitrary Lagrangian-Eulerian (ALE) conservation form with the relative
flux F - c w (Donea et al. 2004), the diffusive flux F = -D dc/dx from the
canonical finite-volume owner, and cell volumes that satisfy the discrete
geometric conservation law exactly (Thomas and Lombard 1979), so a constant
field is preserved under pure mesh motion.  Fluid swept into the column by a
growing boundary carries the prescribed boundary concentration; fluid
released by a shrinking boundary carries the bottom-cell concentration.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor

from ....computation.operators.functional import _compute_diffusion_rate_and_boundary_flux_1d
from ....core.abstractions import FluxDict, StateDict
from ....core.functor.functor_bridge_utils import NamespaceDefaultMixin
from ....core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    TransportMechanismFunctor,
    VariableKind,
)
from ....core.state import UpdateItem, UpdateMode


class MovingDomainDiffusionFunctor(NamespaceDefaultMixin, TransportMechanismFunctor):
    """ALE finite-volume diffusion on a uniformly stretched column (TransportMechanism contract).

    References:
      - Thomas, P.D. & Lombard, C.K. (1979). Geometric conservation law and
        its application to flow computations on moving grids. AIAA J. 17,
        1030-1037.
      - Donea, J., Huerta, A., Ponthot, J.-Ph. & Rodriguez-Ferran, A. (2004).
        Arbitrary Lagrangian-Eulerian methods. Encyclopedia of Computational
        Mechanics, ch. 14.
      - Crank, J. (1975). The Mathematics of Diffusion; Patankar (1980).
    Mathematical expression:
      d/dt int_{x_k}^{x_{k+1}} c dx = -(F_{k+1} - F_k) + c_{k+1} w_{k+1} - c_k w_k,
      F = -D dc/dx, w_k = (k/N) w_b, dL/dt = w_b;
      GCL: V_i(t+dt) = V_i(t) + dt (w_{i+1} - w_i) = L(t+dt)/N;
      m_i(t+dt) = m_i(t) + dt [rate_i dx + s_{i+1} - s_i], c_i(t+dt) = m_i(t+dt)/V_i(t+dt),
      with s_k = c_face,k w_k the swept (relative) mass flux; c_face upwind
      of the face motion; the outside face uses c_b when the column grows.
      Geometry power: P_geo = -0.5 c^T dot(star) c with dot(star) = diag(w_b/N).
    Parameter meanings:
      c cell concentrations, L column length, w_b boundary velocity, c_b
      boundary (entrained) concentration, D diffusivity, N the cell count
      defined by the concentration state shape.
    Units:
      c and c_b kg m-3; L m; w_b m s-1; D m2 s-1; dt s; boundary_mass_flux
      kg m-2 s-1; mass_balance_residual kg m-2; gcl_residual m;
      geometry_power kg2 m-5 s-1.
    Domain/constraints:
      N >= 2; L > 0 and L + w_b dt > 0; nonnegative finite concentrations;
      explicit stability D dt/dx^2 <= transport.numerical.cfl_diffusion_max
      and mesh Courant |w_b| dt/dx <= pedosphere.transport.moving_domain_mesh_courant_max;
      x increases downward from the fixed top face (w_0 = 0).
    Sign convention:
      w_b > 0 grows the column and entrains fluid with concentration c_b
      (boundary_mass_flux positive inward); w_b < 0 releases bottom-cell
      fluid (boundary_mass_flux negative).
    """

    TRANSPORT_TYPE = "diffusion"

    parameter_namespace = "pedosphere.transport"
    CONCENTRATION_VARIABLE = "moving_domain_solute_concentration"
    LENGTH_VARIABLE = "moving_domain_length"
    VELOCITY_VARIABLE = "moving_domain_boundary_velocity"
    BOUNDARY_CONCENTRATION_VARIABLE = "moving_boundary_solute_concentration"
    STATE_VARIABLES = (CONCENTRATION_VARIABLE, LENGTH_VARIABLE)
    FORCING_VARIABLES = (VELOCITY_VARIABLE, BOUNDARY_CONCENTRATION_VARIABLE)
    DIAGNOSTICS = (
        "diagnostics.moving_domain.boundary_mass_flux",
        "diagnostics.moving_domain.mass_balance_residual",
        "diagnostics.moving_domain.gcl_residual",
        "diagnostics.moving_domain.geometry_power",
    )
    PARAMETERS = {
        "moving_domain_diffusivity": "Effective solute diffusivity D [m2 s-1]",
        "moving_domain_mesh_courant_max": (
            "Maximum mesh Courant number |w_b| dt / dx [dimensionless]"
        ),
    }
    geometry_category = "ale"
    geometry_mapping = "uniform_stretching"

    # Explicit config overrides precede the ParameterManager (NamespaceDefaultMixin).
    parameter_lookup_source_order = ("config", "manager")

    def __init__(
        self,
        name: str = "moving_domain_diffusion",
        *,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("Moving-domain config must be a mapping.")
        config_dict: Dict[str, Any] = dict(config) if config is not None else {}
        parameter_overrides = config_dict.pop("parameters", None)
        if parameter_overrides is None:
            parameter_overrides = {}
        if not isinstance(parameter_overrides, Mapping):
            raise ValueError("Moving-domain parameter overrides must be a mapping.")
        device = kwargs.pop("device", None)

        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "ALE finite-volume diffusion on a uniformly stretched column with "
                "prescribed boundary velocity, relative fluxes, and exact discrete GCL"
            ),
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "d/dt int_cell c dx = -(F_{k+1} - F_k) + c_{k+1} w_{k+1} - c_k w_k; "
                "F = -D dc/dx; w_k = (k/N) w_b; dL/dt = w_b; V_i' = V_i + dt (w_{i+1} - w_i)"
            ),
            references=[
                "Thomas, P.D. & Lombard, C.K. (1979). AIAA J. 17, 1030-1037",
                "Donea, J. et al. (2004). ALE methods, Encyclopedia of Computational Mechanics",
                "Crank, J. (1975). The Mathematics of Diffusion",
            ],
            parameters={
                **{
                    f"{self.parameter_namespace}.{name}": text
                    for name, text in self.PARAMETERS.items()
                },
                "transport.numerical.cfl_diffusion_max": (
                    "Largest accepted diffusion number (von Neumann) [dimensionless]"
                ),
            },
            input_variables=[*self.STATE_VARIABLES, *self.FORCING_VARIABLES],
            output_variables=[*self.STATE_VARIABLES, *self.DIAGNOSTICS],
            conservation_laws=["mass"],
            spatial_dimensions=1,
            temporal_scale=TemporalScale.HOUR,
            coupling_interfaces=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        unknown = sorted(set(parameter_overrides).difference(self.PARAMETERS))
        if unknown:
            raise ValueError("Unknown moving-domain parameter override(s): " + ", ".join(unknown))
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
        transport_kwargs = {
            "device": device if device is not None else torch.device("cpu"),
        }
        super().__init__(name=name, config=clean_config, metadata=metadata, **transport_kwargs, **kwargs)
        self.optional_input_variables = []
        for parameter_name in self.PARAMETERS:
            self.get_parameter_value(parameter_name, required=True)
        # The TransportMechanism diffusivity is the live ParameterManager object that
        # forward and dissipative_structure read (same object: a calibration update moves
        # both; no float copy detached from autograd).
        self.set_scalar_diffusivity(
            self.get_parameter_value("moving_domain_diffusivity", required=True)
        )
        if device is not None:
            self.to_device(device)

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Declared IPHS structure (level 3 audit hooks; no runtime replacement)
    # ------------------------------------------------------------------
    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the moving-metric storage functional and geometry correction."""

        return {
            "storage": "H = 0.5 c^T star c with star = diag(L/N) [kg2 m-5]",
            "geometry_power": "P_geo = -0.5 c^T dot(star) c, dot(star) = diag(w_b/N)",
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the nonnegative diffusive norm dissipation of the column."""

        return {
            "diffusive_dissipation": "sigma = sum_faces D (dc/dx)^2 dx >= 0 [kg2 m-5 s-1]",
            "boundary_exchange": "swept mass c_face w_b through the moving face",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for the diffusive column."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the finite-volume dissipation R = d^T diag(D/dx) d on the current metric.

        References:
          - Patankar (1980); Thomas and Lombard (1979).
        Mathematical expression:
          R_h = d_h^T diag(K_f) d_h / dx^2 with K_f = D on interior faces;
          dot(star) = diag(w_b / N) is the metric time derivative.
        Parameter meanings:
          d_h the cell-to-face difference, dx = L/N, D diffusivity, N the
          concentration state length.
        Units: R_h s-1; dot(star) m s-1.
        Domain/constraints: L > 0; N >= 2.
        Sign convention: -R_h c is the diffusive rate.
        """

        for required in (self.CONCENTRATION_VARIABLE, self.LENGTH_VARIABLE, self.VELOCITY_VARIABLE):
            if required not in state:
                raise ValueError(f"Structural hook requires state variable '{required}'")
        concentration = state[self.CONCENTRATION_VARIABLE]
        if not isinstance(concentration, Tensor) or concentration.ndim != 1 or concentration.numel() < 2:
            raise ValueError(f"{self.CONCENTRATION_VARIABLE} must be a one-dimensional tensor with at least two cells")
        dtype = concentration.dtype
        length = torch.as_tensor(state[self.LENGTH_VARIABLE], dtype=dtype).reshape(())
        velocity = torch.as_tensor(state[self.VELOCITY_VARIABLE], dtype=dtype).reshape(())
        diffusivity = torch.as_tensor(
            self.get_parameter_value("moving_domain_diffusivity", required=True), dtype=dtype
        ).reshape(())
        n = int(concentration.numel())
        difference = torch.zeros((n - 1, n), dtype=dtype)
        for face in range(n - 1):
            difference[face, face] = -1.0
            difference[face, face + 1] = 1.0
        dx = length / n
        dissipation = difference.T @ difference * (diffusivity / (dx * dx))
        return {
            "state_vars": tuple(f"{self.CONCENTRATION_VARIABLE}[{i}]" for i in range(n)),
            "J": torch.zeros_like(dissipation),
            "R": dissipation,
            "hodge_time_derivative": torch.eye(n, dtype=dtype) * (velocity / n),
            "ports": (),
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Report the column mass-balance residual of one step.

        References: Donea et al. (2004); RELIC conservation architecture.
        Mathematical expression:
          violation = |sum_i m_i(t+dt) - sum_i m_i(t) - dt s_N|.
        Parameter meanings: the emitted mass_balance_residual diagnostic.
        Units: kg m-2.
        Domain/constraints: a missing residual with emitted state updates is an error.
        Sign convention: magnitude only.
        """
        residual = output_updates.get(self.DIAGNOSTICS[1])
        if residual is None:
            if any(name in output_updates for name in self.STATE_VARIABLES):
                raise ValueError("Moving-domain updates were emitted without the mass-balance residual.")
            return {"mass": 0.0}
        value = residual.value if isinstance(residual, UpdateItem) else residual
        return {"mass": float(torch.max(torch.abs(torch.as_tensor(value))).detach())}

    # ------------------------------------------------------------------
    # Scientific process
    # ------------------------------------------------------------------
    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """Advance the column by one explicit ALE finite-volume step.

        References:
          - Thomas and Lombard (1979); Donea et al. (2004); Patankar (1980).
        Mathematical expression:
          see the class docstring; the diffusive rate comes from the canonical
          _compute_diffusion_rate_and_boundary_flux_1d with closed (zero-flux)
          top and bottom faces, the mesh-motion exchange s_k = c_face,k w_k is
          added at every face, and c_i(t+dt) = m_i(t+dt) / V_i(t+dt).
        Parameter meanings:
          as in the class docstring.
        Units:
          as in the class docstring.
        Domain/constraints:
          stability and positivity limits are enforced before any update.
        Sign convention:
          as in the class docstring.
        """
        # References: Donea et al. (2004) ALE; Thomas and Lombard (1979) GCL;
        #   Crank (1975) diffusion.
        # Mathematical expression: on x = xi L(t), d(c L)/dt = -(1/L) d/dxi(F_rel)
        #   with mesh velocity w_k = (k/N) w_b; discrete GCL d(L)/dt = w_b exactly.
        # Parameter meanings: c cell concentration, L domain length, w_b boundary
        #   velocity, D diffusivity, N cell count, F_rel flux relative to the mesh.
        # Units: c kg m-3; L m; w_b m s-1; D m2 s-1; dt s.
        # Domain/constraints: L > 0; N >= 2; diffusion and mesh Courant limits
        #   enforced; finite positive dt.
        # Sign convention: positive w_b stretches the domain; outward flux at the
        #   moving boundary is positive.

        if kwargs:
            raise TypeError(f"Unsupported moving-domain forward arguments: {sorted(kwargs)}")
        for name in (*self.STATE_VARIABLES, *self.FORCING_VARIABLES):
            if name not in state:
                raise ValueError(f"{name} is required for the moving-domain diffusion")
            value = state[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"State '{name}' must be a torch.Tensor")
            if not value.is_floating_point():
                raise ValueError(f"State '{name}' must be a floating-point tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"State '{name}' must contain only finite values")
        concentration = state[self.CONCENTRATION_VARIABLE]
        if concentration.ndim != 1 or concentration.numel() < 2:
            raise ValueError(
                f"{self.CONCENTRATION_VARIABLE} must be a one-dimensional column with at least two cells, "
                f"got shape {tuple(concentration.shape)}"
            )
        n = int(concentration.numel())
        if bool((concentration < 0).any()):
            raise ValueError(f"State '{self.CONCENTRATION_VARIABLE}' must be nonnegative")
        length = state[self.LENGTH_VARIABLE].reshape(())
        velocity = state[self.VELOCITY_VARIABLE].reshape(())
        boundary_concentration = state[self.BOUNDARY_CONCENTRATION_VARIABLE].reshape(())
        if not float(length.detach()) > 0.0:
            raise ValueError(f"State '{self.LENGTH_VARIABLE}' must be positive")
        if float(boundary_concentration.detach()) < 0.0:
            raise ValueError(f"State '{self.BOUNDARY_CONCENTRATION_VARIABLE}' must be nonnegative")
        dt_value = float(dt)
        if not torch.isfinite(concentration.new_tensor(dt_value)) or dt_value <= 0.0:
            raise ValueError("dt must be finite and positive")

        diffusivity = self.get_parameter_value("moving_domain_diffusivity", required=True).to(concentration)
        courant_max = float(
            self.get_parameter_value("moving_domain_mesh_courant_max", required=True).detach()
        )
        cfl_max = float(
            self.get_parameter_value(
                "cfl_diffusion_max",
                namespace="transport.numerical",
                source_order=("manager",),
                required=True,
            )
        )
        if diffusivity.numel() != 1 or not float(diffusivity.detach()) > 0.0:
            raise ValueError("moving_domain_diffusivity must be a positive scalar")
        diffusivity = diffusivity.reshape(())

        # References: Von Neumann stability; Thomas and Lombard (1979).
        # Mathematical expression: D dt/dx^2 <= cfl_max; |w_b| dt/dx <= courant_max; L + w_b dt > 0.
        # Parameter meanings: dx = L/N the current cell width.
        # Units: dx m; ratios dimensionless.
        # Domain/constraints: both limits are fail-closed, never clipped.
        # Sign convention: not applicable.
        dx = float(length.detach()) / n
        velocity_value = float(velocity.detach())
        diffusion_number = float(diffusivity.detach()) * dt_value / (dx * dx)
        if diffusion_number > cfl_max:
            raise ValueError(
                f"Moving-domain diffusion number {diffusion_number:.6g} exceeds cfl_diffusion_max {cfl_max:.6g}"
            )
        mesh_courant = abs(velocity_value) * dt_value / dx
        if mesh_courant > courant_max:
            raise ValueError(
                f"Moving-domain mesh Courant number {mesh_courant:.6g} exceeds moving_domain_mesh_courant_max {courant_max:.6g}"
            )
        new_length = length + velocity * dt_value
        if not float(new_length.detach()) > 0.0:
            raise ValueError("The moving boundary would collapse the column within one step")

        # References: Patankar (1980) closed finite-volume diffusion on the current metric.
        # Mathematical expression: rate_i = (G_{i+1/2} - G_{i-1/2}) / dx, G = D dc/dx, zero-flux faces.
        # Parameter meanings: D uniform across cells; dx the current cell width.
        # Units: rate kg m-3 s-1.
        # Domain/constraints: N >= 2 (owner requirement).
        # Sign convention: diffusion moves mass down the gradient.
        diffusion_rate, _ = _compute_diffusion_rate_and_boundary_flux_1d(
            concentration,
            diffusivity.expand_as(concentration),
            dx,
            "zero",
        )
        cell_width = length / n
        old_mass = concentration * cell_width

        # References: Donea et al. (2004) ALE relative flux; Thomas and Lombard (1979) GCL.
        # Mathematical expression: s_k = c_face,k w_k, w_k = (k/N) w_b; the face
        #   concentration is that of the fluid crossing the face (upwind of the
        #   face motion; c_b for the outside when the column grows).
        # Parameter meanings: k = 0..N faces, k = 0 fixed top, k = N moving bottom.
        # Units: s kg m-2 s-1.
        # Domain/constraints: mesh Courant limit enforced above.
        # Sign convention: s_k > 0 moves mass from below the face into the cell above.
        face_index = torch.arange(n + 1, dtype=concentration.dtype, device=concentration.device)
        face_velocity = face_index / n * velocity
        lower = torch.cat([concentration, boundary_concentration.reshape(1)])
        upper = torch.cat([concentration[:1], concentration])
        face_concentration = torch.where(face_velocity > 0, lower, upper)
        swept = face_concentration * face_velocity
        swept = torch.cat([torch.zeros(1, dtype=swept.dtype, device=swept.device), swept[1:]])
        new_mass = old_mass + dt_value * (diffusion_rate * cell_width + swept[1:] - swept[:-1])
        new_width = new_length / n
        new_concentration = new_mass / new_width
        concentration_increment = new_concentration - concentration
        length_increment = new_length - length

        gcl_residual = torch.abs(
            new_width - cell_width - dt_value * (face_velocity[1:] - face_velocity[:-1])
        ).max().detach()
        boundary_mass_flux = swept[-1].detach()
        mass_balance_residual = (new_mass.sum() - old_mass.sum() - dt_value * swept[-1]).detach()
        geometry_power = (-0.5 * (concentration * concentration).sum() * velocity / n).detach()

        output: FluxDict = {
            self.CONCENTRATION_VARIABLE: UpdateItem(
                value=concentration_increment,
                mode=UpdateMode.ACCUMULATE,
                unit="kg m-3",
                per_timestep=True,
                source_functor=self.name,
            ),
            self.LENGTH_VARIABLE: UpdateItem(
                value=length_increment,
                mode=UpdateMode.ACCUMULATE,
                unit="m",
                per_timestep=True,
                source_functor=self.name,
            ),
        }
        for key, value, unit in (
            (self.DIAGNOSTICS[0], boundary_mass_flux, "kg m-2 s-1"),
            (self.DIAGNOSTICS[1], mass_balance_residual, "kg m-2"),
            (self.DIAGNOSTICS[2], gcl_residual, "m"),
            (self.DIAGNOSTICS[3], geometry_power, "kg2 m-5 s-1"),
        ):
            output[key] = UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=True,
                source_functor=self.name,
            )
        return output


__all__ = ["MovingDomainDiffusionFunctor"]
