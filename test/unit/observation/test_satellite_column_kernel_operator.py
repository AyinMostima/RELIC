"""L1 tests of the satellite column kernel operator (TROPOMI HCHO comparison).

Hand cases follow docs/source/theory/observation_operators.md, section
"Satellite column kernel operator": p_k = a_k + b_k p_s; arithmetic-mean (primary)
or log-pressure (sensitivity) interior edges with p_s at the surface and 0 Pa at
the top; n_k = delta_p_k / (g M_air); x_pc = chi n; N_v = sum_{l in T} A_l x_pc,l.
"""

import math

import pytest
import torch

from relic.core.state import StateWriteMode, UpdateMode
from relic.coupling.observation.satellite_column_kernel import (
    SatelliteColumnKernelOperator,
)
from relic.utils.physical_constants import get_physical_constant

F64 = torch.float64


def _t(values):
    return torch.tensor(values, dtype=F64)


def _g_m_air() -> float:
    gravity = float(get_physical_constant("gravity_acceleration"))
    molar_mass_air = float(get_physical_constant("air_molecular_weight")) * 1.0e-3
    return gravity * molar_mass_air


def test_hybrid_pressure_hand_case_and_broadcast():
    operator = SatelliteColumnKernelOperator()
    a = _t([1000.0, 500.0])
    b = _t([0.9, 0.4])
    surface = _t([100000.0, 90000.0])
    pressure = operator.hybrid_pressure(a, b, surface)
    torch.testing.assert_close(
        pressure,
        _t([[91000.0, 40500.0], [82000.0, 36500.0]]),
    )


def test_primary_edges_are_arithmetic_means_with_surface_and_zero_top():
    operator = SatelliteColumnKernelOperator()
    edges = operator.layer_edges(_t([90000.0, 50000.0]), _t(100000.0))
    torch.testing.assert_close(edges, _t([100000.0, 70000.0, 0.0]))


def test_sensitivity_edges_are_log_pressure_midpoints():
    operator = SatelliteColumnKernelOperator(layer_edge_method="log_pressure")
    edges = operator.layer_edges(_t([90000.0, 50000.0]), _t(100000.0))
    torch.testing.assert_close(
        edges,
        _t([100000.0, math.sqrt(90000.0 * 50000.0), 0.0]),
    )


def test_two_layer_air_and_tracer_partial_columns_hand_case():
    operator = SatelliteColumnKernelOperator()
    edges = _t([100000.0, 70000.0, 0.0])
    air = operator.air_partial_columns(edges)
    g_m = _g_m_air()
    torch.testing.assert_close(air, _t([30000.0 / g_m, 70000.0 / g_m]))
    # Sanity of magnitude: the whole 1e5 Pa column holds about 3.52e5 mol m-2 of air.
    assert float(air.sum()) == pytest.approx(100000.0 / g_m)
    tracer = operator.tracer_partial_columns(_t([2.0e-9, 1.0e-9]), edges)
    torch.testing.assert_close(
        tracer,
        _t([2.0e-9 * 30000.0 / g_m, 1.0e-9 * 70000.0 / g_m]),
    )


def test_kernel_of_ones_reproduces_total_column():
    operator = SatelliteColumnKernelOperator(tropopause_truncation="full_column")
    partial = _t([[1.0e-6, 2.0e-6, 3.0e-6], [4.0e-6, -1.0e-6, 0.5e-6]])
    column = operator.apply_averaging_kernel(torch.ones_like(partial), partial)
    torch.testing.assert_close(column, partial.sum(dim=-1))


def test_tropopause_truncation_variants_hand_case():
    kernel = _t([1.2, 0.8, 0.5, 2.0])
    partial = _t([1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6])
    index = torch.tensor(1, dtype=torch.int64)
    inclusive = SatelliteColumnKernelOperator().apply_averaging_kernel(
        kernel, partial, tropopause_layer_index=index
    )
    exclusive = SatelliteColumnKernelOperator(
        tropopause_truncation="exclusive"
    ).apply_averaging_kernel(kernel, partial, tropopause_layer_index=index)
    full = SatelliteColumnKernelOperator(
        tropopause_truncation="full_column"
    ).apply_averaging_kernel(kernel, partial, tropopause_layer_index=index)
    torch.testing.assert_close(inclusive, _t(2.8e-6))
    torch.testing.assert_close(exclusive, _t(1.2e-6))
    torch.testing.assert_close(full, _t(1.23e-5))


def test_per_pixel_tropopause_index_and_negative_kernel_values():
    operator = SatelliteColumnKernelOperator()
    kernel = _t([[1.0, -0.5, 2.0], [1.0, -0.5, 2.0]])
    partial = _t([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    column = operator.apply_averaging_kernel(
        kernel,
        partial,
        tropopause_layer_index=torch.tensor([0, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(column, _t([1.0, 2.5]))


def test_tropospheric_kernel_from_box_air_mass_factors():
    operator = SatelliteColumnKernelOperator()
    kernel = operator.tropospheric_averaging_kernel(_t([0.5, 1.0, 1.5]), _t(1.25))
    torch.testing.assert_close(kernel, _t([0.4, 0.8, 1.2]))


def test_remap_conserves_the_column_hand_case():
    operator = SatelliteColumnKernelOperator()
    result = operator.remap_partial_columns(
        _t([3.0, 2.0, 1.0]),
        _t([100000.0, 60000.0, 20000.0, 0.0]),
        _t([100000.0, 80000.0, 30000.0, 0.0]),
    )
    torch.testing.assert_close(result.partial_columns, _t([1.5, 3.0, 1.5]))
    torch.testing.assert_close(result.unmapped_column, _t(0.0))
    torch.testing.assert_close(result.partial_columns.sum(), _t(6.0))


def test_remap_reports_unmapped_amount_below_target_surface():
    operator = SatelliteColumnKernelOperator()
    result = operator.remap_partial_columns(
        _t([4.2, 6.0]),
        _t([102000.0, 60000.0, 0.0]),
        _t([100000.0, 50000.0, 0.0]),
    )
    torch.testing.assert_close(result.partial_columns, _t([5.0, 5.0]))
    torch.testing.assert_close(result.unmapped_column, _t(0.2))
    torch.testing.assert_close(
        result.partial_columns.sum() + result.unmapped_column,
        _t(10.2),
    )


def test_remap_of_a_sixty_level_hybrid_profile_conserves_the_column():
    operator = SatelliteColumnKernelOperator()
    surface = _t(98000.0)
    # Surface-first interfaces of a 60-layer hybrid grid, p_half = a + b p_s.
    fraction = torch.linspace(0.0, 1.0, 61, dtype=F64)
    b_half = (1.0 - fraction) ** 2
    a_half = 2000.0 * torch.sin(math.pi * fraction) * (1.0 - fraction)
    source_edges = a_half + b_half * surface
    source_edges[-1] = 0.0
    vmr = torch.linspace(3.0e-9, 1.0e-11, 60, dtype=F64)
    source = operator.tracer_partial_columns(vmr, source_edges)
    target_edges = operator.layer_edges(
        operator.hybrid_pressure(
            torch.linspace(0.0, 20.0, 34, dtype=F64),
            torch.linspace(0.995, 0.0, 34, dtype=F64),
            surface,
        ),
        surface,
    )
    result = operator.remap_partial_columns(source, source_edges, target_edges)
    assert result.partial_columns.shape == (34,)
    torch.testing.assert_close(result.partial_columns.sum(), source.sum())
    torch.testing.assert_close(result.unmapped_column, _t(0.0))


def test_scale_shape_requires_an_explicit_layer_set():
    operator = SatelliteColumnKernelOperator()
    shape = _t([1.0, 2.0, 3.0, 4.0])
    scaled = operator.scale_shape_to_column(
        shape,
        _t(6.0),
        layer_mask=torch.tensor([True, True, False, False]),
    )
    torch.testing.assert_close(scaled, _t([2.0, 4.0, 6.0, 8.0]))
    with pytest.raises(TypeError):
        operator.scale_shape_to_column(shape, _t(6.0))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="positive"):
        operator.scale_shape_to_column(
            shape,
            _t(6.0),
            layer_mask=torch.tensor([False, False, False, False]),
        )


def test_model_column_gradient_matches_hand_jacobian():
    operator = SatelliteColumnKernelOperator()
    edges = _t([100000.0, 70000.0, 30000.0, 0.0])
    vmr = _t([2.0e-9, 1.0e-9, 5.0e-10]).requires_grad_(True)
    kernel = _t([0.6, 1.1, 1.4])
    partial = operator.tracer_partial_columns(vmr, edges)
    column = operator.apply_averaging_kernel(
        kernel,
        partial,
        tropopause_layer_index=torch.tensor(1),
    )
    column.backward()
    g_m = _g_m_air()
    expected = _t([0.6 * 30000.0 / g_m, 1.1 * 40000.0 / g_m, 0.0])
    torch.testing.assert_close(vmr.grad, expected)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError, match="layer_edge_method"):
        SatelliteColumnKernelOperator(layer_edge_method="linear")
    with pytest.raises(ValueError, match="tropopause_truncation"):
        SatelliteColumnKernelOperator(tropopause_truncation="none")
    operator = SatelliteColumnKernelOperator()
    with pytest.raises(ValueError, match="strictly decreasing"):
        operator.layer_edges(_t([50000.0, 90000.0]), _t(100000.0))
    with pytest.raises(ValueError, match="surface pressure"):
        operator.layer_edges(_t([90000.0, 50000.0]), _t(80000.0))
    with pytest.raises(ValueError, match="finite"):
        operator.layer_edges(_t([90000.0, float("nan")]), _t(100000.0))
    with pytest.raises(ValueError, match="strictly decreasing"):
        operator.air_partial_columns(_t([100000.0, 100000.0, 0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        operator.tracer_partial_columns(_t([-1.0e-9, 1.0e-9]), _t([1.0e5, 5.0e4, 0.0]))
    with pytest.raises(ValueError, match="layer"):
        operator.tracer_partial_columns(_t([1.0e-9]), _t([1.0e5, 5.0e4, 0.0]))
    kernel = _t([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="tropopause_layer_index"):
        operator.apply_averaging_kernel(kernel, kernel)
    with pytest.raises(ValueError, match="tropopause_layer_index"):
        operator.apply_averaging_kernel(
            kernel, kernel, tropopause_layer_index=torch.tensor(3)
        )
    with pytest.raises(ValueError, match="integral"):
        operator.apply_averaging_kernel(
            kernel, kernel, tropopause_layer_index=_t(1.5)
        )
    with pytest.raises(ValueError, match="shape"):
        operator.apply_averaging_kernel(
            _t([1.0, 1.0]), kernel, tropopause_layer_index=torch.tensor(1)
        )
    with pytest.raises(ValueError, match="positive"):
        operator.tropospheric_averaging_kernel(_t([1.0]), _t(0.0))


def test_forward_returns_derived_replace_column_and_air_partial_columns():
    operator = SatelliteColumnKernelOperator()
    state = {
        operator.HYBRID_A_INPUT: _t([1000.0, 500.0, 0.0]),
        operator.HYBRID_B_INPUT: _t([0.9, 0.4, 0.1]),
        operator.SURFACE_PRESSURE_INPUT: _t([100000.0]),
        operator.MIXING_RATIO_INPUT: _t([[2.0e-9, 1.0e-9, 5.0e-10]]),
        operator.AVERAGING_KERNEL_INPUT: _t([[0.6, 1.1, 1.4]]),
        operator.TROPOPAUSE_INDEX_INPUT: torch.tensor([1], dtype=torch.int64),
    }
    outputs = operator.forward(state=state, time=0.0, dt=3600.0)
    assert set(outputs) == set(operator.metadata.output_variables)
    for item in outputs.values():
        assert item.mode == UpdateMode.REPLACE
        assert item.write_mode == StateWriteMode.DERIVED
        assert item.per_timestep is False
        assert item.unit == "mol m-2"
    # Centres 91000, 40500, 10000 Pa -> edges 1e5, 65750, 25250, 0 Pa.
    g_m = _g_m_air()
    air = _t([[34250.0 / g_m, 40500.0 / g_m, 25250.0 / g_m]])
    torch.testing.assert_close(outputs[operator.AIR_PARTIAL_COLUMN_OUTPUT].value, air)
    expected = _t([0.6 * 2.0e-9 * 34250.0 / g_m + 1.1 * 1.0e-9 * 40500.0 / g_m])
    torch.testing.assert_close(outputs[operator.MODEL_COLUMN_OUTPUT].value, expected)
