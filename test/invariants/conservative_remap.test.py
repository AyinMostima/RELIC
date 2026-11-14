"""MR-B conservative remap invariant tests with PBT/MT.

References:
- Jones, P. W. (1999). First- and second-order conservative remapping schemes.
- Dukowicz and Baumgardner (2000). Incremental remapping for transport/advection.

Mathematical expressions:
- MR-B1 integral conservation (extensive quantity):
  sum_i(src_i * A_src_i) = sum_j(dst_j * A_dst_j)
- MR-B2 linearity:
  R(alpha * x + beta * 1) = alpha * R(x) + beta * R(1)

Parameter meanings:
- W: non-negative conservative transfer matrix.
- src: source field value for each source cell.
- area_src, area_dst: source and destination cell areas.
- alpha, beta: linear transform coefficients.
"""

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st


def _make_conservative_weights(n_src: int, n_dst: int) -> torch.Tensor:
    """Create a deterministic column-stochastic transfer matrix."""
    w = torch.arange(1, n_dst * n_src + 1, dtype=torch.float64).reshape(n_dst, n_src)
    w = w / w.sum(dim=0, keepdim=True)
    return w


def _area_conservative_remap(
    src: torch.Tensor,
    weights: torch.Tensor,
    area_src: torch.Tensor,
    area_dst: torch.Tensor,
) -> torch.Tensor:
    """Apply first-order area-based conservative remapping for intensive source."""
    return (weights @ (src * area_src)) / area_dst


@pytest.mark.property
class TestConservativeRemapMRB:
    """MR-B tests for conservation and linearity."""

    @given(
        n_src=st.integers(min_value=2, max_value=20),
        n_dst=st.integers(min_value=2, max_value=20),
        data=st.data(),
    )
    @settings(max_examples=30, deadline=None)
    def test_integral_conservation(self, n_src: int, n_dst: int, data) -> None:
        """MR-B1: remap preserves total extensive mass."""
        src = torch.tensor(
            data.draw(
                st.lists(
                    st.floats(min_value=1e-6, max_value=1e4, allow_nan=False, allow_infinity=False),
                    min_size=n_src,
                    max_size=n_src,
                )
            ),
            dtype=torch.float64,
        )
        area_src = torch.tensor(
            data.draw(
                st.lists(
                    st.floats(min_value=1e-4, max_value=1e3, allow_nan=False, allow_infinity=False),
                    min_size=n_src,
                    max_size=n_src,
                )
            ),
            dtype=torch.float64,
        )
        area_dst = torch.tensor(
            data.draw(
                st.lists(
                    st.floats(min_value=1e-4, max_value=1e3, allow_nan=False, allow_infinity=False),
                    min_size=n_dst,
                    max_size=n_dst,
                )
            ),
            dtype=torch.float64,
        )
        assume(float(area_src.sum().item()) > 0.0)
        assume(float(area_dst.sum().item()) > 0.0)

        # Normalize destination total area to source total area.
        area_dst = area_dst * (area_src.sum() / area_dst.sum())
        weights = _make_conservative_weights(n_src, n_dst)
        dst = _area_conservative_remap(src, weights, area_src, area_dst)

        total_src = float((src * area_src).sum().item())
        total_dst = float((dst * area_dst).sum().item())
        tol = max(1e-8, 1e-8 * abs(total_src))
        assert abs(total_dst - total_src) <= tol

    @given(
        n=st.integers(min_value=2, max_value=14),
        alpha=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        beta=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        data=st.data(),
    )
    @settings(max_examples=30, deadline=None)
    def test_linearity(self, n: int, alpha: float, beta: float, data) -> None:
        """MR-B2: conservative remap operator is linear."""
        x = torch.tensor(
            data.draw(
                st.lists(
                    st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                )
            ),
            dtype=torch.float64,
        )
        ones = torch.ones(n, dtype=torch.float64)
        area_src = torch.tensor(
            data.draw(
                st.lists(
                    st.floats(min_value=1e-4, max_value=1e3, allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                )
            ),
            dtype=torch.float64,
        )
        area_dst = torch.tensor(
            data.draw(
                st.lists(
                    st.floats(min_value=1e-4, max_value=1e3, allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                )
            ),
            dtype=torch.float64,
        )
        assume(float(area_src.sum().item()) > 0.0)
        assume(float(area_dst.sum().item()) > 0.0)
        area_dst = area_dst * (area_src.sum() / area_dst.sum())

        weights = _make_conservative_weights(n, n)
        lhs = _area_conservative_remap(alpha * x + beta * ones, weights, area_src, area_dst)
        rhs = (
            alpha * _area_conservative_remap(x, weights, area_src, area_dst)
            + beta * _area_conservative_remap(ones, weights, area_src, area_dst)
        )
        diff = float(torch.max(torch.abs(lhs - rhs)).item())
        scale = max(
            1.0,
            float(torch.max(torch.abs(lhs)).item()),
            float(torch.max(torch.abs(rhs)).item()),
        )
        tol = max(1e-5, 1e-12 * scale)
        assert diff <= tol
