"""Vertical averaging-kernel operator for satellite column comparisons.

Theory owner: docs/source/theory/observation_operators.md, section
"Satellite column kernel operator".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch import Tensor

from ...computation.operators.dec.forms import (
    DiagnosticSpec,
    ObservationOperatorSpec,
)
from ...core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    VariableKind,
)
from ...core.state import StateWriteMode, UpdateItem, UpdateMode
from ...utils.physical_constants import get_physical_constant


class PartialColumnRemap(NamedTuple):
    """Remapped partial columns and the source amount outside the target range."""

    partial_columns: Tensor
    unmapped_column: Tensor


class SatelliteColumnKernelOperator(ScaleCouplingFunctor):
    """Map a model profile to a satellite-equivalent column, N_v = sum_T A_l x_pc,l."""

    LAYER_EDGE_METHODS = ("arithmetic_mean", "log_pressure")
    TROPOPAUSE_TRUNCATIONS = ("inclusive", "exclusive", "full_column")
    CONSERVATION_TOLERANCE_KEY = "coupling.scales.defaults.conservation_tolerance"

    HYBRID_A_INPUT = "observation.satellite_column.hybrid_coefficient_a"
    HYBRID_B_INPUT = "observation.satellite_column.hybrid_coefficient_b"
    SURFACE_PRESSURE_INPUT = "observation.satellite_column.surface_pressure"
    MIXING_RATIO_INPUT = "observation.satellite_column.model_mixing_ratio"
    AVERAGING_KERNEL_INPUT = "observation.satellite_column.averaging_kernel"
    TROPOPAUSE_INDEX_INPUT = "observation.satellite_column.tropopause_layer_index"
    MODEL_COLUMN_OUTPUT = "diagnostics.observation.satellite_column.model_column"
    AIR_PARTIAL_COLUMN_OUTPUT = (
        "diagnostics.observation.satellite_column.air_partial_column"
    )

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        layer_edge_method: str = "arithmetic_mean",
        tropopause_truncation: str = "inclusive",
    ) -> None:
        edge_method = str(layer_edge_method).strip().lower()
        truncation = str(tropopause_truncation).strip().lower()
        if edge_method not in self.LAYER_EDGE_METHODS:
            raise ValueError(
                "layer_edge_method must be 'arithmetic_mean' or 'log_pressure'."
            )
        if truncation not in self.TROPOPAUSE_TRUNCATIONS:
            raise ValueError(
                "tropopause_truncation must be 'inclusive', 'exclusive' or "
                "'full_column'."
            )
        functor_name = name or "satellite_column_kernel_operator"
        metadata = FunctorMetadata(
            name=functor_name,
            version="1.0.0",
            description=(
                "Read-only averaging-kernel column of a model mixing-ratio "
                "profile on hybrid-pressure product layers."
            ),
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.POSTPROCESSING,
            mathematical_expression=(
                "p_k = a_k + b_k p_s; n_k = (p_e,k - p_e,k+1) / (g M_air); "
                "x_pc,k = chi_k n_k; N_v = sum_{l in T} A_l x_pc,l"
            ),
            references=[
                (
                    "S5P/TROPOMI HCHO ATBD S5P-BIRA-L2-400F issue 2.4.1, "
                    "section 5.4.4 and Eqs. 5-1, 5-14, 7-3, 9-5, appendix C."
                ),
                (
                    "Eskes and Boersma (2003), Atmos. Chem. Phys. 3, "
                    "1285-1291, doi:10.5194/acp-3-1285-2003."
                ),
                (
                    "De Smedt et al. (2018), Atmos. Meas. Tech. 11, "
                    "2395-2426, doi:10.5194/amt-11-2395-2018, appendix C."
                ),
                "Jones (1999), Mon. Weather Rev. 127, 2204-2210.",
            ],
            parameters={
                self.CONSERVATION_TOLERANCE_KEY: (
                    "Relative closure tolerance of the vertical remap [1]"
                ),
                "relic.physical.constants.earth.gravity_acceleration": (
                    "Standard gravity g [m s-2]"
                ),
                "relic.physical.constants.air.molecular_weight_dry": (
                    "Dry-air molar mass M_air [g mol-1]"
                ),
            },
            input_variables=[
                self.HYBRID_A_INPUT,
                self.HYBRID_B_INPUT,
                self.SURFACE_PRESSURE_INPUT,
                self.MIXING_RATIO_INPUT,
                self.AVERAGING_KERNEL_INPUT,
                self.TROPOPAUSE_INDEX_INPUT,
            ],
            output_variables=[
                self.MODEL_COLUMN_OUTPUT,
                self.AIR_PARTIAL_COLUMN_OUTPUT,
            ],
            conservation_laws=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
            coupling_interfaces=["satellite_column_observation"],
        )
        super().__init__(
            name=functor_name,
            config=config or {},
            metadata=metadata,
            source_scale="vertical_column",
            target_scale="satellite_pixel_column",
            coupling_type="hierarchical",
            conservation_properties=[],
        )
        self.layer_edge_method = edge_method
        self.tropopause_truncation = truncation
        if truncation == "full_column":
            self.optional_input_variables = {self.TROPOPAUSE_INDEX_INPUT}

        gravity = float(get_physical_constant("gravity_acceleration"))
        molar_mass_air_g = float(get_physical_constant("air_molecular_weight"))
        if not (gravity > 0.0 and molar_mass_air_g > 0.0):
            raise ValueError("g and M_air must be positive.")
        self.register_buffer(
            "gravity",
            torch.tensor(gravity, dtype=torch.float64),
            persistent=False,
        )
        # M_air is registered in g mol-1; 1e-3 converts it to kg mol-1.
        self.register_buffer(
            "molar_mass_air_kg_per_mol",
            torch.tensor(molar_mass_air_g * 1.0e-3, dtype=torch.float64),
            persistent=False,
        )
        self._conservation_tolerance()

    def _conservation_tolerance(self) -> float:
        value = self.get_parameter_value(
            self.CONSERVATION_TOLERANCE_KEY,
            required=True,
        )
        tolerance = float(torch.as_tensor(value, dtype=torch.float64).item())
        if not 0.0 < tolerance < 1.0:
            raise ValueError("The remap conservation tolerance must be in (0, 1).")
        return tolerance

    @staticmethod
    def _require_finite(value: Any, label: str) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{label} must be a torch.Tensor.")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{label} must be finite.")
        return value

    @staticmethod
    def _require_strictly_decreasing(values: Tensor, label: str) -> None:
        if values.ndim < 1 or values.shape[-1] < 2:
            raise ValueError(f"{label} needs at least two values along the layer axis.")
        if not bool(torch.all(values[..., 1:] < values[..., :-1])):
            raise ValueError(
                f"{label} must be strictly decreasing from the surface upward."
            )

    def hybrid_pressure(
        self,
        coefficient_a: Tensor,
        coefficient_b: Tensor,
        surface_pressure: Tensor,
    ) -> Tensor:
        """Return p_k = a_k + b_k p_s [Pa] with layers on the last axis."""

        a = self._require_finite(coefficient_a, "coefficient_a")
        b = self._require_finite(coefficient_b, "coefficient_b")
        p_s = self._require_finite(surface_pressure, "surface_pressure")
        # References: S5P HCHO ATBD S5P-BIRA-L2-400F issue 2.4.1, section 5.4.4, p. 33;
        #   PUM section 8.5, p. 17 (also the IFS/CAMS half-level form p = a + b p_s).
        # Mathematical expression: p_k = a_k + b_k p_s.
        # Parameter meanings: a_k pressure coefficient, b_k surface-pressure fraction,
        #   p_s elevation-corrected surface pressure of the pixel.
        # Units: a_k, p_s, p_k [Pa]; b_k [1].
        # Domain/constraints: finite inputs; a, b broadcast on the last (layer) axis
        #   against p_s[..., None]; ordering is checked where edges are built.
        # Sign convention: not applicable; pressures are positive magnitudes.
        return a + b * p_s.unsqueeze(-1)

    def layer_edges(self, centre_pressure: Tensor, surface_pressure: Tensor) -> Tensor:
        """Return L + 1 edges [Pa] from L surface-first layer-centre pressures."""

        centres = self._require_finite(centre_pressure, "centre_pressure")
        p_s = self._require_finite(surface_pressure, "surface_pressure")
        if centres.ndim < 1:
            raise ValueError("centre_pressure needs a layer axis.")
        batch = torch.broadcast_shapes(centres.shape[:-1], p_s.shape)
        centres = centres.expand(*batch, centres.shape[-1])
        p_s = p_s.expand(batch)
        if bool(torch.any(centres <= 0)):
            raise ValueError("centre_pressure must be positive.")
        if centres.shape[-1] > 1:
            self._require_strictly_decreasing(centres, "centre_pressure")
        if not bool(torch.all(p_s > centres[..., 0])):
            raise ValueError(
                "surface pressure must exceed the lowest layer-centre pressure."
            )

        # References: paper-3 registered design g02_3_hcho_operator_design_20261018,
        #   layer_edges_for_partial_columns (primary arithmetic mean, sensitivity
        #   log-pressure midpoints); layer index 0 at the surface (registered inference).
        # Mathematical expression: p_e,0 = p_s; p_e,k = (p_k-1 + p_k) / 2 (primary) or
        #   sqrt(p_k-1 p_k) (log-pressure) for k = 1..L-1; p_e,L = 0.
        # Parameter meanings: p_k layer-centre pressure, p_s surface pressure, p_e,k
        #   lower edge of layer k (upper edge of layer k-1).
        # Units: all pressures [Pa].
        # Domain/constraints: finite, positive, strictly decreasing centres with
        #   p_s > p_0, so every centre lies inside its own layer and the edges are
        #   strictly decreasing; 0 Pa is the registered top edge.
        # Sign convention: not applicable; pressures are positive magnitudes.
        lower, upper = centres[..., :-1], centres[..., 1:]
        if self.layer_edge_method == "arithmetic_mean":
            interior = 0.5 * (lower + upper)
        else:
            interior = torch.sqrt(lower * upper)
        top = torch.zeros_like(p_s).unsqueeze(-1)
        return torch.cat([p_s.unsqueeze(-1), interior, top], dim=-1)

    def air_partial_columns(self, edges: Tensor) -> Tensor:
        """Return the air amount n_k of every layer [mol m-2]."""

        edges = self._require_finite(edges, "edges")
        self._require_strictly_decreasing(edges, "edges")
        if bool(torch.any(edges[..., -1] < 0)):
            raise ValueError("The top edge pressure must be non-negative.")
        gravity = self.gravity.to(dtype=edges.dtype, device=edges.device)
        molar_mass = self.molar_mass_air_kg_per_mol.to(
            dtype=edges.dtype,
            device=edges.device,
        )
        # References: hydrostatic balance dp = -rho g dz (e.g. Seinfeld and Pandis,
        #   2016, ch. 1); paper-3 registered partial_column_units.
        # Mathematical expression: n_k = (p_e,k - p_e,k+1) / (g M_air).
        # Parameter meanings: p_e,k and p_e,k+1 lower and upper layer edges, g standard
        #   gravity, M_air dry-air molar mass.
        # Units: p [Pa]; g [m s-2]; M_air [kg mol-1]; n_k [mol m-2].
        # Domain/constraints: finite, strictly decreasing edges with a non-negative top;
        #   constant g and dry-air M_air are the registered convention (no moist-air
        #   or latitude correction).
        # Sign convention: n_k > 0 is the air amount above unit area within layer k.
        return (edges[..., :-1] - edges[..., 1:]) / (gravity * molar_mass)

    def tracer_partial_columns(self, mixing_ratio: Tensor, edges: Tensor) -> Tensor:
        """Return x_pc,k = chi_k n_k [mol m-2] for a mixing ratio chi [mol mol-1]."""

        chi = self._require_finite(mixing_ratio, "mixing_ratio")
        edges = self._require_finite(edges, "edges")
        if chi.ndim < 1 or edges.ndim < 1 or chi.shape[-1] != edges.shape[-1] - 1:
            raise ValueError(
                "mixing_ratio must have one value per layer (edges minus one)."
            )
        if bool(torch.any(chi < 0)):
            raise ValueError("mixing_ratio must be non-negative.")
        # References: paper-3 registered partial_column_units (x_pc = vmr times air
        #   partial column).
        # Mathematical expression: x_pc,k = chi_k n_k.
        # Parameter meanings: chi_k layer-mean tracer mole fraction, n_k air amount.
        # Units: chi [mol mol-1]; n_k, x_pc,k [mol m-2].
        # Domain/constraints: finite, non-negative chi with one value per layer.
        # Sign convention: x_pc,k >= 0 is tracer amount above unit area in layer k.
        return chi * self.air_partial_columns(edges)

    def remap_partial_columns(
        self,
        source_partial_columns: Tensor,
        source_edges: Tensor,
        target_edges: Tensor,
    ) -> PartialColumnRemap:
        """Move partial columns to target layers by pressure overlap (amount conserving)."""

        source = self._require_finite(source_partial_columns, "source_partial_columns")
        source_edges = self._require_finite(source_edges, "source_edges")
        target_edges = self._require_finite(target_edges, "target_edges")
        self._require_strictly_decreasing(source_edges, "source_edges")
        self._require_strictly_decreasing(target_edges, "target_edges")
        if source.ndim < 1 or source.shape[-1] != source_edges.shape[-1] - 1:
            raise ValueError(
                "source_partial_columns must have one value per source layer."
            )

        # References: Jones (1999) Mon. Weather Rev. 127, 2204-2210 (first-order
        #   conservative remapping, here in one pressure dimension); paper-3 registered
        #   model_profile_shape (mass-conserving pressure mapping to the TM5 layers).
        # Mathematical expression: x_j = sum_i x_i O_ij / dp_i with
        #   O_ij = max(0, min(p_e,i, P_e,j) - max(p_e,i+1, P_e,j+1)),
        #   dp_i = p_e,i - p_e,i+1; x_unmapped = sum_i x_i (dp_i - sum_j O_ij) / dp_i.
        # Parameter meanings: x_i source partial column, p_e source edges, P_e target
        #   edges, O_ij pressure overlap of source layer i and target layer j.
        # Units: x [mol m-2]; p, O, dp [Pa]; weights O_ij / dp_i [1].
        # Domain/constraints: finite, strictly decreasing surface-first edges; uniform
        #   mixing ratio within each source layer in pressure; source amount outside
        #   the target range is returned, never dropped; closure is checked with
        #   coupling.scales.defaults.conservation_tolerance (relative to sum |x_i|).
        # Sign convention: amounts keep their sign; weights are non-negative.
        source_upper = source_edges[..., :-1].unsqueeze(-1)
        source_lower = source_edges[..., 1:].unsqueeze(-1)
        target_upper = target_edges[..., :-1].unsqueeze(-2)
        target_lower = target_edges[..., 1:].unsqueeze(-2)
        overlap = torch.clamp(
            torch.minimum(source_upper, target_upper)
            - torch.maximum(source_lower, target_lower),
            min=0.0,
        )
        thickness = source_edges[..., :-1] - source_edges[..., 1:]
        density = source / thickness
        remapped = (density.unsqueeze(-1) * overlap).sum(dim=-2)
        unmapped = (density * (thickness - overlap.sum(dim=-1))).sum(dim=-1)

        residual = remapped.sum(dim=-1) + unmapped - source.sum(dim=-1)
        scale = source.abs().sum(dim=-1)
        tolerance = self._conservation_tolerance()
        if not bool(torch.all(residual.abs() <= tolerance * scale)):
            raise ValueError(
                "Vertical remap does not close the source column within "
                f"{self.CONSERVATION_TOLERANCE_KEY}."
            )
        return PartialColumnRemap(partial_columns=remapped, unmapped_column=unmapped)

    def scale_shape_to_column(
        self,
        shape_partial_columns: Tensor,
        column: Tensor,
        *,
        layer_mask: Tensor,
    ) -> Tensor:
        """Scale a profile shape so that its sum over layer_mask equals column."""

        shape = self._require_finite(shape_partial_columns, "shape_partial_columns")
        target = self._require_finite(column, "column")
        if not isinstance(layer_mask, Tensor) or layer_mask.dtype != torch.bool:
            raise TypeError("layer_mask must be a boolean torch.Tensor.")
        # References: paper-3 registered model_profile_shape (CAMS-prior shape scaled
        #   to the RELIC model column); the normalization layer set K is not registered
        #   and is therefore a required argument without default.
        # Mathematical expression: x_pc,k = x_shape,k Omega / sum_{k in K} x_shape,k.
        # Parameter meanings: x_shape,k shape partial column, Omega target column, K the
        #   caller-selected layer set (layer_mask).
        # Units: x_shape, x_pc, Omega [mol m-2].
        # Domain/constraints: finite inputs; the masked shape sum must be positive.
        # Sign convention: the scaled profile keeps the sign of Omega.
        denominator = (shape * layer_mask.to(dtype=shape.dtype)).sum(dim=-1)
        if not bool(torch.all(denominator > 0)):
            raise ValueError("The masked shape sum must be positive.")
        return shape * (target / denominator).unsqueeze(-1)

    def tropospheric_averaging_kernel(
        self,
        box_air_mass_factors: Tensor,
        tropospheric_air_mass_factor: Tensor,
    ) -> Tensor:
        """Return A_l = m_l / M [1] (ATBD Eq. 7-3)."""

        box = self._require_finite(box_air_mass_factors, "box_air_mass_factors")
        total = self._require_finite(
            tropospheric_air_mass_factor,
            "tropospheric_air_mass_factor",
        )
        if not bool(torch.all(total > 0)):
            raise ValueError("tropospheric_air_mass_factor must be positive.")
        # References: S5P HCHO ATBD Eq. 7-3 and appendix C; Eskes and Boersma (2003).
        # Mathematical expression: A_l = m_l / M.
        # Parameter meanings: m_l box air mass factor of layer l, M tropospheric air
        #   mass factor of the product (the PUM 'total air mass factor').
        # Units: m_l, M, A_l [1].
        # Domain/constraints: finite m_l (any sign), M > 0; no rescaling to another
        #   air mass factor because the product has no separate total M.
        # Sign convention: not applicable; ratios of sensitivities.
        return box / total.unsqueeze(-1)

    def _tropopause_mask(
        self,
        layer_count: int,
        tropopause_layer_index: Tensor | None,
        like: Tensor,
    ) -> Tensor:
        if tropopause_layer_index is None:
            if self.tropopause_truncation != "full_column":
                raise ValueError(
                    "tropopause_layer_index is required for "
                    f"'{self.tropopause_truncation}' truncation."
                )
            return torch.ones(layer_count, dtype=like.dtype, device=like.device)
        index = torch.as_tensor(tropopause_layer_index, device=like.device)
        if index.is_floating_point():
            if not bool(torch.all(torch.isfinite(index))) or not bool(
                torch.all(index == torch.round(index))
            ):
                raise ValueError("tropopause_layer_index must be integral.")
        index = index.to(torch.int64)
        if bool(torch.any(index < 0)) or bool(torch.any(index >= layer_count)):
            raise ValueError(
                f"tropopause_layer_index must lie in [0, {layer_count - 1}]."
            )
        if self.tropopause_truncation == "full_column":
            return torch.ones(layer_count, dtype=like.dtype, device=like.device)
        layers = torch.arange(layer_count, device=like.device)
        if self.tropopause_truncation == "inclusive":
            mask = layers <= index.unsqueeze(-1)
        else:
            mask = layers < index.unsqueeze(-1)
        return mask.to(dtype=like.dtype)

    def apply_averaging_kernel(
        self,
        averaging_kernel: Tensor,
        partial_columns: Tensor,
        tropopause_layer_index: Tensor | None = None,
    ) -> Tensor:
        """Return N_v = sum_{l in T} A_l x_pc,l [mol m-2] (ATBD Eq. 9-5, reduced form)."""

        kernel = self._require_finite(averaging_kernel, "averaging_kernel")
        partial = self._require_finite(partial_columns, "partial_columns")
        if kernel.shape != partial.shape or kernel.ndim < 1:
            raise ValueError(
                "averaging_kernel and partial_columns must have the same shape "
                "with layers on the last axis."
            )
        mask = self._tropopause_mask(kernel.shape[-1], tropopause_layer_index, partial)
        # References: S5P HCHO ATBD Eq. 9-5 and appendix C; De Smedt et al. (2018)
        #   appendix C (N_v - N_v,a = A (x_pc - n_a,pc) reduces to N_v = A x_pc for
        #   A = m / M); paper-3 registered tropopause_truncation.
        # Mathematical expression: N_v = sum_l w_l A_l x_pc,l with w_l = 1 for l <= l_t
        #   (inclusive, primary), l < l_t (exclusive) or all l (full_column), else 0.
        # Parameter meanings: A_l product averaging kernel, x_pc,l model partial column,
        #   l_t 0-based tm5_tropopause_layer_index of the pixel, w_l layer selection.
        # Units: A_l [1]; x_pc,l and N_v [mol m-2] (the product column unit).
        # Domain/constraints: equal shapes, finite values, integral l_t in [0, L-1]; no
        #   a priori term and no background term are added (the retrieved column
        #   already contains the CTM background, ATBD Eqs. 5-1 and 5-14).
        # Sign convention: kernels and columns keep their sign; no positivity rule.
        return (kernel * partial * mask).sum(dim=-1)

    def observation_operator(self) -> ObservationOperatorSpec:
        """Return the typed profile-to-satellite-column observation map."""

        return ObservationOperatorSpec(
            name="satellite_column_kernel_observation",
            input_state_vars=tuple(self.metadata.input_variables),
            output_name=self.MODEL_COLUMN_OUTPUT,
            output_units="mol m-2",
            support="satellite_pixel_hybrid_pressure_layers",
            averaging_window="instantaneous_at_pixel_time",
            footprint_description=(
                "Product layers p_k = a_k + b_k p_s of one pixel; horizontal "
                "and temporal support are owned by the existing adapters"
            ),
            error_model="product_kernel_and_truncation_variants",
            conservative_projection_required=False,
            metadata={
                "layer_edge_method": self.layer_edge_method,
                "tropopause_truncation": self.tropopause_truncation,
                "a_priori_term": False,
                "background_term": False,
            },
        )

    def diagnostic_operator(self) -> DiagnosticSpec:
        """Return the read-only air partial-column diagnostic."""

        return DiagnosticSpec(
            name="satellite_column_air_partial_column_diagnostic",
            input_state_vars=(
                self.HYBRID_A_INPUT,
                self.HYBRID_B_INPUT,
                self.SURFACE_PRESSURE_INPUT,
            ),
            output_units="mol m-2",
            diagnostic_kind="air_partial_column",
            writes_state=False,
            metadata={"outputs": tuple(self.metadata.output_variables)},
        )

    @staticmethod
    def _derived(value: Tensor, source_functor: str) -> UpdateItem:
        return UpdateItem(
            value=value,
            mode=UpdateMode.REPLACE,
            write_mode=StateWriteMode.DERIVED,
            unit="mol m-2",
            per_timestep=False,
            source_functor=source_functor,
        )

    def forward(
        self,
        state: Mapping[str, Tensor],
        time: float,
        dt: Any,
        subprocess_outputs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, UpdateItem]:
        """
        Evaluate the satellite-equivalent model column of the declared inputs.

        References:
            S5P HCHO ATBD Eqs. 7-3 and 9-5; De Smedt et al. (2018), appendix C.
        Mathematical Expression:
            p_k = a_k + b_k p_s; edges from centres; n_k = dp_k / (g M_air);
            N_v = sum_{l in T} A_l chi_l n_l.
        Parameters:
            a [Pa], b [1], p_s [Pa], chi [mol mol-1] on product layers, A [1],
            l_t 0-based tropopause layer index; outputs in mol m-2.
        """
        # References, expression, parameter meanings, units, domain and sign
        #   convention: as in hybrid_pressure, layer_edges, air_partial_columns,
        #   tracer_partial_columns and apply_averaging_kernel; both outputs are
        #   REPLACE UpdateItems with StateWriteMode.DERIVED in mol m-2.
        del time, dt, subprocess_outputs, kwargs
        optional = getattr(self, "optional_input_variables", set())
        for key in self.metadata.input_variables:
            if key not in state and key not in optional:
                raise KeyError(f"Required satellite column input '{key}' is missing.")
        centres = self.hybrid_pressure(
            state[self.HYBRID_A_INPUT],
            state[self.HYBRID_B_INPUT],
            state[self.SURFACE_PRESSURE_INPUT],
        )
        edges = self.layer_edges(centres, state[self.SURFACE_PRESSURE_INPUT])
        air = self.air_partial_columns(edges)
        mixing_ratio = state[self.MIXING_RATIO_INPUT]
        partial = self.tracer_partial_columns(mixing_ratio, edges)
        column = self.apply_averaging_kernel(
            state[self.AVERAGING_KERNEL_INPUT],
            partial,
            tropopause_layer_index=(
                state[self.TROPOPAUSE_INDEX_INPUT]
                if self.TROPOPAUSE_INDEX_INPUT in state
                else None
            ),
        )
        return {
            self.MODEL_COLUMN_OUTPUT: self._derived(column, self.name),
            self.AIR_PARTIAL_COLUMN_OUTPUT: self._derived(air, self.name),
        }


__all__ = ["PartialColumnRemap", "SatelliteColumnKernelOperator"]
