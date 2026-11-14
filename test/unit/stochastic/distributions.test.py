"""L1 tests for parameter distributions and spatial correlation.

Module under test: relic/stochastic/distributions.py (ParameterDistribution,
DistributionSpec, SpatialCorrelation).

References:
    Rasmussen and Williams (2006), Gaussian Processes for Machine Learning,
    ch. 4 (Matern kernels) and appendix A.2 (Cholesky sampling);
    Cressie (1993), Statistics for Spatial Data (nugget effect);
    Williams and Seeger (2001), NIPS 13 (Nystroem approximation).
Mathematical expression:
    theta = mu + sigma L z with L L^T = K + tau I, z ~ N(0, I);
    log-normal field: theta = exp(mu + sigma L z);
    Nystroem: lambda_n = (n / m) lambda_m, u_n ~ K_nm u_m / lambda_m.
Parameter meanings:
    K is the kernel correlation matrix, tau the declared nugget, n the number
    of points, m the subset size.

A correlation matrix that is not positive definite is an error with its
diagnostics unless a nugget is declared; no distribution is replaced by
another (triangular, truncated normal), and no statistic is estimated in
place of an unavailable analytical one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import special

from relic.parameters.spec import ParameterSpec
from relic.stochastic.distributions import (
    CorrelationKernel,
    DistributionSpec,
    DistributionType,
    ParameterDistribution,
    SpatialCorrelation,
)


def _normal_spec(**overrides) -> DistributionSpec:
    values = {
        "dist_type": DistributionType.NORMAL,
        "params": {"loc": 10.0, "scale": 2.0},
        "correlation_length": 5.0,
        "kernel": CorrelationKernel.EXPONENTIAL,
        "correlation_nugget": 0.0,
    }
    values.update(overrides)
    return DistributionSpec(**values)


def _uncorrelated_normal_spec() -> DistributionSpec:
    return _normal_spec(correlation_length=None, kernel=None, correlation_nugget=None)


def _duplicate_coordinates() -> torch.Tensor:
    # Two identical points make the kernel matrix exactly singular.
    return torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Correlated sampling: Cholesky failure and the explicit nugget.
# ---------------------------------------------------------------------------


def test_singular_correlation_matrix_is_an_error_with_diagnostics() -> None:
    distribution = ParameterDistribution({"p": _normal_spec()}, _duplicate_coordinates())

    with pytest.raises(ValueError, match="minimum eigenvalue") as excinfo:
        distribution.sample(torch.empty(0), num_samples=4, seed=1)
    assert isinstance(excinfo.value.__cause__, torch.linalg.LinAlgError)


def test_cholesky_failure_is_not_replaced_by_an_eigen_projection(monkeypatch) -> None:
    coords = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    distribution = ParameterDistribution({"p": _normal_spec()}, coords)

    def failing_cholesky(_matrix):
        raise torch.linalg.LinAlgError("probe failure")

    monkeypatch.setattr(torch.linalg, "cholesky", failing_cholesky)

    with pytest.raises(ValueError, match="not positive definite"):
        distribution.sample(torch.empty(0), num_samples=2, seed=1)


def test_declared_nugget_regularizes_the_correlation_matrix() -> None:
    spec = _normal_spec(correlation_nugget=1.0e-6)
    distribution = ParameterDistribution({"p": spec}, _duplicate_coordinates())

    samples = distribution.sample(torch.empty(0), num_samples=2000, seed=3)["p"]

    assert samples.shape == (2000, 3)
    duplicate_correlation = torch.corrcoef(samples[:, :2].T)[0, 1]
    assert float(duplicate_correlation) > 0.999


@pytest.mark.parametrize("nugget", [-1.0e-6, float("nan"), float("inf")])
def test_invalid_nugget_is_rejected(nugget: float) -> None:
    with pytest.raises(ValueError, match="correlation_nugget"):
        ParameterDistribution({"p": _normal_spec(correlation_nugget=nugget)}, _duplicate_coordinates())


def test_correlated_lognormal_field_is_the_exponential_of_the_gaussian_field() -> None:
    spec = DistributionSpec(
        dist_type=DistributionType.LOGNORMAL,
        params={"loc": 0.5, "scale": 0.25},
        correlation_length=5.0,
        kernel=CorrelationKernel.EXPONENTIAL,
        correlation_nugget=0.0,
    )
    coords = torch.tensor([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0], [50.0, 50.0]])
    samples = ParameterDistribution({"p": spec}, coords).sample(
        torch.empty(0), num_samples=4000, seed=5
    )["p"]

    assert bool((samples > 0.0).all())
    log_samples = torch.log(samples)
    assert float(log_samples.mean()) == pytest.approx(0.5, abs=0.02)
    assert float(log_samples.std()) == pytest.approx(0.25, abs=0.02)


# ---------------------------------------------------------------------------
# Distribution families: no substitute family.
# ---------------------------------------------------------------------------


def test_triangular_is_not_replaced_by_a_uniform_distribution() -> None:
    spec = DistributionSpec(
        dist_type=DistributionType.TRIANGULAR,
        params={"low": 0.0, "high": 1.0, "mode": 0.2},
    )

    with pytest.raises(NotImplementedError, match="triangular"):
        ParameterDistribution({"p": spec})


def test_truncated_normal_is_not_replaced_by_a_logit_transform() -> None:
    spec = DistributionSpec(
        dist_type=DistributionType.TRUNCATED_NORMAL,
        params={"loc": 0.5, "scale": 0.1},
        bounds=(0.0, 1.0),
    )

    with pytest.raises(NotImplementedError, match="truncated_normal"):
        ParameterDistribution({"p": spec})


def test_unbounded_parameter_spec_has_no_invented_spread() -> None:
    spec = ParameterSpec(key="probe.value", value=4.0, units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")

    with pytest.raises(ValueError, match="DistributionSpec"):
        ParameterDistribution({"p": spec})


def test_half_bounded_parameter_spec_is_rejected() -> None:
    spec = ParameterSpec(key="probe.value", value=4.0, bounds=(0.0, float("inf")), units="dimensionless", dtype="float32", category="model_parameter")

    with pytest.raises(ValueError, match="DistributionSpec"):
        ParameterDistribution({"p": spec})


def test_bounded_parameter_spec_is_uniform_over_its_bounds() -> None:
    spec = ParameterSpec(key="probe.value", value=4.0, bounds=(2.0, 6.0), units="dimensionless", dtype="float32", category="model_parameter")
    samples = ParameterDistribution({"p": spec}).sample(
        torch.empty(0), num_samples=5000, seed=2
    )["p"]

    assert float(samples.min()) >= 2.0 and float(samples.max()) <= 6.0
    assert float(samples.mean()) == pytest.approx(4.0, abs=0.1)


def test_log_scale_integer_samples_are_transformed_bounded_then_rounded() -> None:
    spec = DistributionSpec(
        dist_type=DistributionType.NORMAL,
        params={"loc": math.log(20.0), "scale": 0.5},
        bounds=(5.0, 60.0),
        log_scale=True,
        integer_valued=True,
    )
    samples = ParameterDistribution({"p": spec}).sample(
        torch.empty(0), num_samples=500, seed=4
    )["p"]

    assert torch.equal(samples, torch.round(samples))
    assert float(samples.min()) >= 5.0 and float(samples.max()) <= 60.0


def test_sampling_with_a_seed_does_not_touch_the_global_generator() -> None:
    distribution = ParameterDistribution({"p": _uncorrelated_normal_spec()})
    before = torch.random.get_rng_state()

    first = distribution.sample(torch.empty(0), num_samples=3, seed=9)["p"]
    after = torch.random.get_rng_state()
    second = distribution.sample(torch.empty(0), num_samples=3, seed=9)["p"]

    assert torch.equal(before, after)
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# Densities and moments: nothing estimated in place of the declared quantity.
# ---------------------------------------------------------------------------


def test_correlated_log_probability_is_not_an_independent_approximation() -> None:
    coords = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    distribution = ParameterDistribution({"p": _normal_spec()}, coords)

    with pytest.raises(NotImplementedError, match="correlated"):
        distribution.log_probability(torch.empty(0), torch.ones(3, 1))


def test_unavailable_analytical_moments_are_not_replaced_by_sample_estimates() -> None:
    spec = DistributionSpec(
        dist_type=DistributionType.BETA,
        params={"alpha": 2.0, "beta": 3.0, "low": 1.0, "high": 3.0},
    )
    distribution = ParameterDistribution({"p": spec})

    with pytest.raises(NotImplementedError, match="moments"):
        distribution.moments(torch.empty(0))


# ---------------------------------------------------------------------------
# Kernels and the Nystroem approximation.
# ---------------------------------------------------------------------------


def _matern_32(distances: torch.Tensor, length: float) -> torch.Tensor:
    scaled = distances * math.sqrt(3.0) / length
    return (1.0 + scaled) * torch.exp(-scaled)


def _pairwise(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.norm(left.unsqueeze(1) - right.unsqueeze(0), dim=2)


def test_nystroem_extension_uses_the_requested_kernel_and_scales_eigenvalues() -> None:
    torch.manual_seed(0)
    coords = torch.rand(12, 2, dtype=torch.float64) * 10.0
    n_points, n_eig, length = 12, 4, 3.0
    subset_size = min(n_eig * 4, n_points // 2)

    torch.manual_seed(123)
    indices = torch.randperm(n_points)[:subset_size]
    subset = coords[indices]
    k_subset = _matern_32(_pairwise(subset, subset), length)
    k_cross = _matern_32(_pairwise(coords, subset), length)
    eigenvalues, eigenvectors = torch.linalg.eigh(k_subset)
    order = torch.argsort(eigenvalues, descending=True)[:n_eig]
    expected_values = eigenvalues[order] * (n_points / subset_size)
    expected_vectors = k_cross @ (eigenvectors[:, order] / eigenvalues[order])
    expected_vectors = expected_vectors / torch.norm(expected_vectors, dim=0, keepdim=True)

    torch.manual_seed(123)
    values, vectors = SpatialCorrelation.approximate_correlation_matrix(
        coords, length, CorrelationKernel.MATERN_32, n_eigenfunctions=n_eig
    )

    torch.testing.assert_close(values, expected_values)
    for column in range(n_eig):
        sign = torch.sign(torch.dot(vectors[:, column], expected_vectors[:, column]))
        torch.testing.assert_close(sign * vectors[:, column], expected_vectors[:, column])


def test_generic_matern_rejects_invalid_distances() -> None:
    with pytest.raises(ValueError, match="distances"):
        SpatialCorrelation.matern_correlation(torch.tensor([-1.0, 1.0]), 1.0, nu=0.7)
    with pytest.raises(ValueError, match="distances"):
        SpatialCorrelation.matern_correlation(torch.tensor([float("nan")]), 1.0, nu=0.7)


def test_generic_matern_matches_the_closed_form_and_its_limits() -> None:
    nu, length = 0.7, 2.0
    distances = torch.tensor([0.0, 0.5, 2.0, 10.0, 1.0e4], dtype=torch.float64)

    values = SpatialCorrelation.matern_correlation(distances, length, nu=nu)

    s = distances.numpy()[1:4] * math.sqrt(2.0 * nu) / length
    expected = (2.0 ** (1.0 - nu) / special.gamma(nu)) * s**nu * special.kv(nu, s)
    assert float(values[0]) == 1.0
    np.testing.assert_allclose(values.numpy()[1:4], expected, rtol=1e-12)
    assert float(values[4]) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA generator to observe")
def test_seeded_cpu_sampling_does_not_reseed_the_cuda_generator() -> None:
    """torch.manual_seed would also reseed every CUDA generator, outside the fork."""
    distribution = ParameterDistribution({"p": _uncorrelated_normal_spec()})
    cuda_before = torch.cuda.get_rng_state().clone()

    distribution.sample(torch.empty(0), num_samples=3, seed=9)

    assert torch.equal(cuda_before, torch.cuda.get_rng_state())


# ---------------------------------------------------------------------------
# Round 5: a correlated random field declares its kernel and its nugget; an
# uncorrelated specification declares neither (no kernel is chosen for it).
# ---------------------------------------------------------------------------


def test_correlated_field_needs_a_declared_kernel() -> None:
    with pytest.raises(ValueError, match="kernel"):
        _normal_spec(kernel=None)


def test_correlated_field_needs_a_declared_nugget() -> None:
    with pytest.raises(ValueError, match="correlation_nugget"):
        _normal_spec(correlation_nugget=None)


def test_kernel_must_be_a_correlation_kernel() -> None:
    with pytest.raises(TypeError, match="CorrelationKernel"):
        _normal_spec(kernel="exponential")


@pytest.mark.parametrize(
    "declared", [{"kernel": CorrelationKernel.EXPONENTIAL}, {"correlation_nugget": 0.0}]
)
def test_kernel_or_nugget_without_a_correlation_length_is_rejected(declared) -> None:
    values = {"correlation_length": None, "kernel": None, "correlation_nugget": None}
    values.update(declared)
    with pytest.raises(ValueError, match="correlation_length"):
        _normal_spec(**values)


def test_uncorrelated_spec_declares_no_kernel_and_no_nugget() -> None:
    spec = DistributionSpec(dist_type=DistributionType.NORMAL, params={"loc": 1.0, "scale": 0.5})
    assert spec.kernel is None
    assert spec.correlation_nugget is None


@pytest.mark.parametrize("nugget", [-1.0e-6, float("nan"), float("inf")])
def test_invalid_nugget_is_rejected_when_the_spec_is_declared(nugget: float) -> None:
    with pytest.raises(ValueError, match="correlation_nugget"):
        _normal_spec(correlation_nugget=nugget)
