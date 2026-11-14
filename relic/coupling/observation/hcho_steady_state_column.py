"""HCHO steady-state column relation for satellite formaldehyde comparisons.

Theory owner: docs/source/theory/observation_operators.md, section
"HCHO steady-state column relation".
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


class HchoColumnSensitivity(NamedTuple):
    """Column-emission sensitivity S with its regime and range flags."""

    seconds: Tensor
    linear_regime: Tensor
    below_calibrated_range: Tensor


class HchoSteadyStateColumnOperator(ScaleCouplingFunctor):
    """Map an isoprene emission to a steady-state HCHO column, Omega = S n_C E + B."""

    PARAMETER_PREFIX = "coupling.observation.hcho_steady_state_column"
    COEFFICIENT_SETS = ("marais2012_standard", "marais2012_paulot")
    COEFFICIENT_NAMES = (
        "low_nox_slope",
        "low_nox_intercept",
        "high_nox_constant",
    )
    NO2_THRESHOLD_KEY = f"{PARAMETER_PREFIX}.no2_regime_threshold"
    NO2_LOWER_END_KEY = f"{PARAMETER_PREFIX}.calibrated_no2_lower_end"
    CARBON_ATOMS_KEY = "atmosphere.chemistry.isoprene_carbon_atoms"
    NO2_UNITS = ("molecule cm-2", "mol m-2")

    ISOPRENE_EMISSION_INPUT = "observation.hcho_column.isoprene_emission"
    NO2_COLUMN_INPUT = "observation.hcho_column.no2_column"
    BACKGROUND_COLUMN_INPUT = "observation.hcho_column.background_column"
    COLUMN_OUTPUT = "diagnostics.observation.hcho_column.steady_state_column"
    SENSITIVITY_OUTPUT = (
        "diagnostics.observation.hcho_column.emission_column_sensitivity"
    )
    LINEAR_REGIME_OUTPUT = "diagnostics.observation.hcho_column.no2_linear_regime"
    BELOW_CALIBRATED_OUTPUT = (
        "diagnostics.observation.hcho_column.no2_below_calibrated_range"
    )

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        *,
        coefficient_set: str = "marais2012_standard",
    ) -> None:
        selected_set = str(coefficient_set).strip().lower()
        if selected_set not in self.COEFFICIENT_SETS:
            raise ValueError(
                "coefficient_set must be one of "
                f"{', '.join(self.COEFFICIENT_SETS)}."
            )
        functor_name = name or "hcho_steady_state_column_operator"
        coefficient_keys = {
            coefficient: f"{self.PARAMETER_PREFIX}.{selected_set}.{coefficient}"
            for coefficient in self.COEFFICIENT_NAMES
        }
        parameter_metadata = {
            coefficient_keys["low_nox_slope"]: (
                "Low-NOx slope m of S(Omega_NO2) [s cm2 molecule-1]"
            ),
            coefficient_keys["low_nox_intercept"]: (
                "Low-NOx intercept c of S(Omega_NO2) [s]"
            ),
            coefficient_keys["high_nox_constant"]: (
                "High-NOx constant S_high [s]"
            ),
            self.NO2_THRESHOLD_KEY: (
                "NO2 column switch Omega_t between linear and constant S "
                "[molecule cm-2]"
            ),
            self.NO2_LOWER_END_KEY: (
                "Lower end of the calibrated NO2 column range [molecule cm-2]"
            ),
            self.CARBON_ATOMS_KEY: (
                "Carbon atoms per isoprene molecule n_C [dimensionless]"
            ),
        }
        for coefficient, key in coefficient_keys.items():
            parameter_metadata[f"{key}_standard_deviation"] = (
                f"One-sigma standard deviation of {coefficient}"
            )
        metadata = FunctorMetadata(
            name=functor_name,
            version="1.0.0",
            description=(
                "Read-only steady-state HCHO column from isoprene emission "
                "with the NOx-dependent column-emission sensitivity."
            ),
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.POSTPROCESSING,
            mathematical_expression=(
                "S = m Omega_NO2 + c if Omega_NO2 <= Omega_t else S_high; "
                "Omega_HCHO = S n_C E + B"
            ),
            references=[
                (
                    "Marais et al. (2012), Atmos. Chem. Phys. 12, 6219-6235, "
                    "doi:10.5194/acp-12-6219-2012, Eq. (2) p. 6225, Eq. (4) "
                    "p. 6226, Sect. 4.2-4.3 and Fig. 9 p. 6229."
                ),
                (
                    "Palmer et al. (2003), J. Geophys. Res. 108(D6), 4180, "
                    "doi:10.1029/2002JD002153, Eq. (1) p. ACH 2-4."
                ),
            ],
            parameters=parameter_metadata,
            input_variables=[
                self.ISOPRENE_EMISSION_INPUT,
                self.NO2_COLUMN_INPUT,
                self.BACKGROUND_COLUMN_INPUT,
            ],
            output_variables=[
                self.COLUMN_OUTPUT,
                self.SENSITIVITY_OUTPUT,
                self.LINEAR_REGIME_OUTPUT,
                self.BELOW_CALIBRATED_OUTPUT,
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
            source_scale="surface_emission",
            target_scale="vertical_column",
            coupling_type="hierarchical",
            conservation_properties=[],
        )
        self.coefficient_set = selected_set
        self._coefficient_keys = coefficient_keys
        # Fail closed at construction if any registered entry is absent.
        self.coefficients()
        self.coefficient_standard_deviations()
        self._regime_limits()
        _ = self.isoprene_carbon_atoms

    def _scalar_parameter(self, key: str) -> float:
        value = self.get_parameter_value(key, required=True)
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if tensor.ndim != 0 or not bool(torch.isfinite(tensor)):
            raise ValueError(f"Parameter '{key}' must be a finite scalar.")
        return float(tensor.item())

    def required_parameters(self) -> tuple[str, ...]:
        """Return the ParameterManager keys read by this operator instance."""

        keys = list(self._coefficient_keys.values())
        keys += [f"{key}_standard_deviation" for key in self._coefficient_keys.values()]
        keys += [
            self.NO2_THRESHOLD_KEY,
            self.NO2_LOWER_END_KEY,
            self.CARBON_ATOMS_KEY,
        ]
        return tuple(keys)

    def coefficients(self) -> dict[str, float]:
        """Return m [s cm2 molecule-1], c [s] and S_high [s] of the selected set."""

        return {
            coefficient: self._scalar_parameter(key)
            for coefficient, key in self._coefficient_keys.items()
        }

    def coefficient_standard_deviations(self) -> dict[str, float]:
        """Return the registered one-sigma deviations; they are not applied."""

        return {
            coefficient: self._scalar_parameter(f"{key}_standard_deviation")
            for coefficient, key in self._coefficient_keys.items()
        }

    def _regime_limits(self) -> tuple[float, float]:
        threshold = self._scalar_parameter(self.NO2_THRESHOLD_KEY)
        lower_end = self._scalar_parameter(self.NO2_LOWER_END_KEY)
        if not 0.0 < lower_end < threshold:
            raise ValueError(
                "The calibrated NO2 lower end must be positive and below the "
                "regime threshold."
            )
        return threshold, lower_end

    @property
    def isoprene_carbon_atoms(self) -> float:
        """Carbon atoms per isoprene molecule from the chemistry owner [1]."""

        n_carbon = self._scalar_parameter(self.CARBON_ATOMS_KEY)
        if n_carbon <= 0.0:
            raise ValueError("isoprene_carbon_atoms must be positive.")
        return n_carbon

    @staticmethod
    def _require_finite(value: Any, label: str) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{label} must be a torch.Tensor.")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{label} must be finite.")
        return value

    def _no2_in_molecule_cm2(self, no2_column: Tensor, units: str) -> Tensor:
        # References: CODATA 2018 Avogadro constant (RELIC physical-constant owner).
        # Mathematical expression: Omega[molecule cm-2] = Omega[mol m-2] N_A 1e-4.
        # Parameter meanings: N_A Avogadro constant; 1e-4 converts m-2 to cm-2.
        # Units: Omega_NO2 in mol m-2 or molecule cm-2; N_A [mol-1]; result
        #   [molecule cm-2].
        # Domain/constraints: units must be 'molecule cm-2' or 'mol m-2'; finite values.
        # Sign convention: not applicable; a unit conversion keeps the sign.
        if units == "molecule cm-2":
            return no2_column
        avogadro = float(get_physical_constant("avogadro_constant"))
        return no2_column * (avogadro * 1.0e-4)

    def emission_column_sensitivity(
        self,
        no2_column: Tensor,
        *,
        no2_column_units: str,
    ) -> HchoColumnSensitivity:
        """Return S(Omega_NO2) [s] with the linear-regime and calibrated-range flags."""

        units = str(no2_column_units).strip()
        if units not in self.NO2_UNITS:
            raise ValueError(
                "no2_column_units must be 'molecule cm-2' or 'mol m-2'."
            )
        no2_column = self._require_finite(no2_column, "no2_column")
        no2 = self._no2_in_molecule_cm2(no2_column, units)
        coefficients = self.coefficients()
        threshold, lower_end = self._regime_limits()

        # References: Marais et al. (2012) ACP 12, 6219-6235, Sect. 4.2 and Fig. 9,
        #   p. 6229 (linear for Omega_NO2 <= 1e15, S levels off above); Sect. 4.3
        #   (0.2-1e15 low-NOx range of the error analysis); paper-3 registered
        #   handling s_threshold_20261018.
        # Mathematical expression: S = m Omega_NO2 + c if Omega_NO2 <= Omega_t,
        #   S = S_high if Omega_NO2 > Omega_t; flag_below = Omega_NO2 < Omega_low.
        # Parameter meanings: m low-NOx slope, c low-NOx intercept, S_high high-NOx
        #   mean, Omega_t regime switch, Omega_low lower end of the calibrated range,
        #   Omega_NO2 tropospheric NO2 column.
        # Units: Omega_NO2, Omega_t, Omega_low [molecule cm-2]; m [s cm2 molecule-1];
        #   c, S_high, S [s]; flags boolean.
        # Domain/constraints: finite Omega_NO2; the boundary point is linear; values
        #   below Omega_low (including negative columns) keep the linear S and are
        #   only flagged, never clipped; no upper limit is stated by the source.
        # Sign convention: S > 0 is the HCHO column gained per unit carbon emission
        #   rate; Omega_NO2 enters with its retrieved or modelled sign.
        slope = no2.new_tensor(coefficients["low_nox_slope"])
        intercept = no2.new_tensor(coefficients["low_nox_intercept"])
        high_constant = no2.new_tensor(coefficients["high_nox_constant"])
        linear_regime = no2 <= threshold
        seconds = torch.where(
            linear_regime,
            slope * no2 + intercept,
            high_constant.expand_as(no2),
        )
        below_calibrated_range = no2 < lower_end
        return HchoColumnSensitivity(
            seconds=seconds,
            linear_regime=linear_regime,
            below_calibrated_range=below_calibrated_range,
        )

    def steady_state_column(
        self,
        isoprene_emission: Tensor,
        column_sensitivity: Tensor,
        background_column: Tensor,
    ) -> Tensor:
        """Return Omega_HCHO = S n_C E + B [mol m-2]."""

        emission = self._require_finite(isoprene_emission, "isoprene_emission")
        sensitivity = self._require_finite(column_sensitivity, "column_sensitivity")
        background = self._require_finite(background_column, "background_column")
        if bool(torch.any(emission < 0)):
            raise ValueError("isoprene_emission must be non-negative.")

        # References: Marais et al. (2012) Eq. (4), p. 6226 (Omega = S E_ISOP + B) and
        #   Eq. (2) text, p. 6225 (E_ISOP in atoms C cm-2 s-1); paper-3 registered
        #   yield_basis_rule (n_C = 5 from the isoprene_carbon_atoms owner).
        # Mathematical expression: Omega_HCHO = S n_C E + B.
        # Parameter meanings: S per-carbon column-emission sensitivity, n_C carbon atoms
        #   per isoprene molecule, E isoprene emission per molecule, B named non-isoprene
        #   background column supplied by the caller.
        # Units: S [s]; n_C [1]; E [mol C5H8 m-2 s-1]; n_C E [mol C m-2 s-1];
        #   B, Omega_HCHO [mol m-2] (the molecule cm-2 to mol m-2 factor is common to
        #   both sides of Eq. (4), so S keeps its value).
        # Domain/constraints: finite tensors broadcastable to one shape; E >= 0; B is
        #   never the product background correction, which the retrieved column contains.
        # Sign convention: E > 0 is emission into the atmosphere; Omega_HCHO and B are
        #   column amounts with their own sign (a residual B may be negative).
        n_carbon = self.isoprene_carbon_atoms
        return sensitivity * (n_carbon * emission) + background

    def observation_operator(self) -> ObservationOperatorSpec:
        """Return the typed emission-to-column observation map."""

        return ObservationOperatorSpec(
            name="hcho_steady_state_column_observation",
            input_state_vars=tuple(self.metadata.input_variables),
            output_name=self.COLUMN_OUTPUT,
            output_units="mol m-2",
            support="vertical_column_at_model_cell",
            averaging_window="instantaneous_at_supplied_time",
            footprint_description=(
                "Local steady-state column; Marais et al. (2012) smearing "
                "limits set the minimum comparison support"
            ),
            error_model="registered_coefficient_standard_deviations_not_propagated",
            conservative_projection_required=False,
            metadata={
                "coefficient_set": self.coefficient_set,
                "parameter_keys": self.required_parameters(),
                "emission_basis": "mol C5H8 m-2 s-1 converted with n_C",
                "below_calibrated_range": "flagged_not_clipped",
            },
        )

    def diagnostic_operator(self) -> DiagnosticSpec:
        """Return the read-only regime and calibrated-range diagnostic."""

        return DiagnosticSpec(
            name="hcho_steady_state_column_regime_diagnostic",
            input_state_vars=(self.NO2_COLUMN_INPUT,),
            output_units="1",
            diagnostic_kind="no2_regime_and_calibrated_range",
            writes_state=False,
            metadata={"outputs": tuple(self.metadata.output_variables)},
        )

    @staticmethod
    def _derived(value: Tensor, unit: str, source_functor: str) -> UpdateItem:
        return UpdateItem(
            value=value,
            mode=UpdateMode.REPLACE,
            write_mode=StateWriteMode.DERIVED,
            unit=unit,
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
        Evaluate the steady-state HCHO column from the declared inputs.

        References:
            Marais et al. (2012), Eqs. (2) and (4), Fig. 9.
        Mathematical Expression:
            S = m Omega_NO2 + c (Omega_NO2 <= Omega_t), S_high otherwise;
            Omega_HCHO = S n_C E + B.
        Parameters:
            E isoprene emission [mol m-2 s-1]; Omega_NO2 tropospheric NO2
            column [mol m-2]; B background column [mol m-2]; S [s].
        """
        # References, expression, parameter meanings, units, domain and sign
        #   convention: as in emission_column_sensitivity and steady_state_column;
        #   the state carries Omega_NO2 in mol m-2 and every output is a REPLACE
        #   UpdateItem with StateWriteMode.DERIVED; flags are 1.0 (true) or 0.0.
        del time, dt, subprocess_outputs, kwargs
        for key in self.metadata.input_variables:
            if key not in state:
                raise KeyError(f"Required HCHO column input '{key}' is missing.")
        sensitivity = self.emission_column_sensitivity(
            state[self.NO2_COLUMN_INPUT],
            no2_column_units="mol m-2",
        )
        column = self.steady_state_column(
            state[self.ISOPRENE_EMISSION_INPUT],
            sensitivity.seconds,
            state[self.BACKGROUND_COLUMN_INPUT],
        )
        dtype = sensitivity.seconds.dtype
        return {
            self.COLUMN_OUTPUT: self._derived(column, "mol m-2", self.name),
            self.SENSITIVITY_OUTPUT: self._derived(
                sensitivity.seconds,
                "s",
                self.name,
            ),
            self.LINEAR_REGIME_OUTPUT: self._derived(
                sensitivity.linear_regime.to(dtype),
                "1",
                self.name,
            ),
            self.BELOW_CALIBRATED_OUTPUT: self._derived(
                sensitivity.below_calibrated_range.to(dtype),
                "1",
                self.name,
            ),
        }


__all__ = ["HchoColumnSensitivity", "HchoSteadyStateColumnOperator"]
