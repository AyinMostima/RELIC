"""Benchmark kernels for computation CLI."""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
from relic.utils.physical_constants import get_physical_constant


SYNTHETIC_BENCHMARK_SCOPE = "synthetic_single_environment_microbenchmark"


def _mark_synthetic_benchmark(
    benchmark: Callable[[torch.Tensor], torch.Tensor],
    target: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Attach non-scientific scope metadata to one synthetic benchmark kernel."""

    setattr(benchmark, "relic_benchmark_scope", SYNTHETIC_BENCHMARK_SCOPE)
    setattr(benchmark, "scientific_equivalence", False)
    setattr(benchmark, "scientific_runtime_owner", None)
    setattr(benchmark, "synthetic_target_label", target)
    return benchmark


def create_benchmark_function(target: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a synthetic local microbenchmark, never a scientific Functor."""
    target = target.lower()

    if target == "photosynthesis":
        gas_constant = float(get_physical_constant("gas_constant"))
        reference_temperature = float(get_physical_constant("reference_temperature"))
        celsius_to_kelvin = float(get_physical_constant("celsius_to_kelvin"))

        # References: Farquhar et al. 1980 (C3 photosynthesis), DOI: 10.1007/BF00386231
        # Expression:
        #   K = K25 * exp(Ea * (T - T_ref) / (T_ref * R * T))
        #   Wc = Vcmax * (Ci - Gamma) / (Ci + Kc * (1 + O2 / Ko))
        #   Wj = Jmax * (Ci - Gamma) / (4 * (Ci + 2 * Gamma))
        #   A = min(Wc, Wj)
        # Parameters:
        #   Vcmax: maximum carboxylation rate at 25C
        #   Jmax: maximum electron transport rate at 25C
        #   Ci: intercellular CO2 concentration
        #   T: leaf temperature in Kelvin
        #   Kc, Ko, Gamma: temperature-dependent Michaelis constants and CO2 compensation point
        #   O2: oxygen partial pressure
        def benchmark_func(inputs: torch.Tensor) -> torch.Tensor:
            vcmax = inputs[..., 0]
            jmax = inputs[..., 1]
            ci = inputs[..., 2]
            temp = inputs[..., 3]

            temp_k = temp + celsius_to_kelvin
            kc = 404.9 * torch.exp(
                79430.0 * (temp_k - reference_temperature) / (reference_temperature * gas_constant * temp_k)
            )
            ko = 278400.0 * torch.exp(
                36380.0 * (temp_k - reference_temperature) / (reference_temperature * gas_constant * temp_k)
            )
            gamma_star = 42.75 * torch.exp(
                37830.0 * (temp_k - reference_temperature) / (reference_temperature * gas_constant * temp_k)
            )

            co2 = ci * 0.7
            o2 = 210000.0

            wc = vcmax * (co2 - gamma_star) / (co2 + kc * (1.0 + o2 / ko))
            wj = jmax * (co2 - gamma_star) / (4.0 * (co2 + 2.0 * gamma_star))

            return torch.min(wc, wj)

        return _mark_synthetic_benchmark(benchmark_func, target)

    if target == "transport":
        # References: advection-diffusion equation (e.g., Bear 1972, Dynamics of Fluids in Porous Media)
        # Expression: dC/dt = D * d2C/dx2 - u * dC/dx + S
        # Parameters:
        #   C: concentration
        #   u: velocity
        #   D: diffusivity
        #   S: source term
        #   dx: spatial step
        def benchmark_func(inputs: torch.Tensor) -> torch.Tensor:
            concentration = inputs[..., 0]
            velocity = inputs[..., 1]
            diffusivity = inputs[..., 2]
            source = inputs[..., 3]

            dx = 0.1

            conc_padded = torch.nn.functional.pad(concentration, (1, 1), mode="replicate")
            d2c_dx2 = (conc_padded[..., 2:] - 2.0 * conc_padded[..., 1:-1] + conc_padded[..., :-2]) / (dx ** 2)
            dc_dx = (conc_padded[..., 2:] - conc_padded[..., :-2]) / (2.0 * dx)

            dcdt = diffusivity * d2c_dx2 - velocity * dc_dx + source

            return dcdt.mean(dim=-1)

        return _mark_synthetic_benchmark(benchmark_func, target)

    if target == "decomposition":
        # References: first-order decomposition with Q10 temperature response (e.g., Parton et al. 1987)
        # Expression:
        #   f_T = Q10 ^ ((T - Tref) / 10)
        #   f_W = piecewise moisture scalar
        #   dC/dt = k * f_T * f_W * C
        # Parameters:
        #   C: carbon pool
        #   T: temperature in degC
        #   W: moisture (0 to 1)
        #   k: base decomposition rate
        def benchmark_func(inputs: torch.Tensor) -> torch.Tensor:
            carbon_pool = inputs[..., 0]
            temperature = inputs[..., 1]
            moisture = inputs[..., 2]

            temp_factor = torch.pow(2.0, (temperature - 20.0) / 10.0)

            moisture_factor = torch.where(
                moisture < 0.3,
                moisture / 0.3,
                torch.where(moisture > 0.7, (1.0 - moisture) / 0.3, torch.ones_like(moisture)),
            )

            k_decomp = 0.1 * temp_factor * moisture_factor

            return carbon_pool * k_decomp

        return _mark_synthetic_benchmark(benchmark_func, target)

    raise ValueError(f"Unknown benchmark target: {target}")


def create_test_inputs(target: str, batch_size: int, n_points: int, device: torch.device) -> torch.Tensor:
    """Create synthetic inputs for one local microbenchmark target label."""
    target = target.lower()

    if target == "photosynthesis":
        inputs = torch.randn(batch_size, n_points, 4, device=device, dtype=torch.float32)
        inputs[..., 0] = inputs[..., 0] * 20.0 + 50.0
        inputs[..., 1] = inputs[..., 1] * 50.0 + 125.0
        inputs[..., 2] = torch.abs(inputs[..., 2]) * 200.0 + 200.0
        inputs[..., 3] = inputs[..., 3] * 15.0 + 25.0
        return inputs

    if target == "transport":
        concentration = torch.randn(batch_size, n_points, device=device, dtype=torch.float32)
        velocity = (
            torch.randn(batch_size, 1, device=device, dtype=torch.float32) * 0.5
        ).expand(-1, n_points)
        diffusivity = (
            torch.abs(
                torch.randn(batch_size, 1, device=device, dtype=torch.float32)
            )
            * 0.1
        ).expand(-1, n_points)
        source = (
            torch.randn(batch_size, 1, device=device, dtype=torch.float32) * 0.01
        ).expand(-1, n_points)

        return torch.stack([concentration, velocity, diffusivity, source], dim=-1)

    if target == "decomposition":
        inputs = torch.randn(batch_size, n_points, 3, device=device, dtype=torch.float32)
        inputs[..., 0] = torch.abs(inputs[..., 0]) * 100.0 + 50.0
        inputs[..., 1] = inputs[..., 1] * 20.0 + 15.0
        inputs[..., 2] = torch.abs(inputs[..., 2]) * 0.5 + 0.2
        return inputs

    raise ValueError(f"Unknown benchmark target: {target}")


def compute_spectral_signature(time_series: torch.Tensor, dt: float) -> Dict[str, torch.Tensor]:
    """Compute power spectral signature for a state variable time series.

    Mathematical Expression:
        S(f) = |FFT(x(t))|^2 / N
        spectral_slope = d(log S) / d(log f) from least squares fit

    References:
        - Lovejoy and Schertzer (2013), Weather and Climate.
        - Fraedrich (2003), Physica D.

    Parameters:
        time_series: tensor with shape [n_timesteps]
        dt: sampling interval [s]
    """
    x = time_series.reshape(-1).to(dtype=torch.float64)
    if x.numel() < 2:
        raise ValueError("time_series must contain at least 2 samples.")

    x_centered = x - torch.mean(x)
    n_samples = x_centered.numel()
    fft_values = torch.fft.rfft(x_centered)
    power_spectrum = (fft_values.abs() ** 2) / float(n_samples)
    frequencies = torch.fft.rfftfreq(n_samples, d=float(dt), device=x.device)

    valid = (frequencies > 0.0) & (power_spectrum > 0.0)
    spectral_slope = torch.tensor(0.0, dtype=torch.float64, device=x.device)
    if int(valid.sum().item()) >= 2:
        log_f = torch.log(frequencies[valid])
        log_s = torch.log(power_spectrum[valid])
        x_mean = torch.mean(log_f)
        y_mean = torch.mean(log_s)
        denom = torch.sum((log_f - x_mean) ** 2)
        if float(denom.item()) > 0.0:
            spectral_slope = torch.sum((log_f - x_mean) * (log_s - y_mean)) / denom

    if n_samples > 2:
        dominant_idx = torch.argmax(power_spectrum[1:]) + 1
    else:
        dominant_idx = torch.tensor(0, device=x.device)

    return {
        "frequencies": frequencies.to(dtype=time_series.dtype),
        "power_spectrum": power_spectrum.to(dtype=time_series.dtype),
        "spectral_slope": spectral_slope.to(dtype=time_series.dtype),
        "dominant_frequency": frequencies[dominant_idx].to(dtype=time_series.dtype),
    }


def compute_structure_function(
    field: torch.Tensor,
    max_lag: int,
    order: int,
) -> Dict[str, torch.Tensor]:
    """Compute structure function for a one-dimensional field.

    Mathematical Expression:
        S_q(r) = (1 / N) * sum_i |f(x_i + r) - f(x_i)|^q
        scaling_exponent = d(log S_q) / d(log r)

    References:
        - Pope (2000), Turbulent Flows.
        - Kolmogorov (1941), Dokl. Akad. Nauk SSSR.

    Parameters:
        field: tensor with shape [n_points]
        max_lag: maximum lag in grid cells
        order: structure function order q
    """
    values = field.reshape(-1).to(dtype=torch.float64)
    n_points = values.numel()
    if n_points < 2:
        raise ValueError("field must contain at least 2 points.")

    max_effective_lag = min(max(int(max_lag), 1), max(int(n_points) - 1, 1))
    lags = torch.arange(1, max_effective_lag + 1, dtype=torch.float64, device=values.device)
    structure_vals = []
    q = int(order)
    if q < 1:
        raise ValueError("order must be >= 1.")

    for lag in range(1, max_effective_lag + 1):
        diffs = torch.abs(values[lag:] - values[:-lag]) ** q
        structure_vals.append(torch.mean(diffs))
    structure_function = torch.stack(structure_vals)

    valid = structure_function > 0.0
    scaling_exponent = torch.tensor(0.0, dtype=torch.float64, device=values.device)
    if int(valid.sum().item()) >= 2:
        log_r = torch.log(lags[valid])
        log_s = torch.log(structure_function[valid])
        x_mean = torch.mean(log_r)
        y_mean = torch.mean(log_s)
        denom = torch.sum((log_r - x_mean) ** 2)
        if float(denom.item()) > 0.0:
            scaling_exponent = torch.sum((log_r - x_mean) * (log_s - y_mean)) / denom

    return {
        "lags": lags.to(dtype=field.dtype),
        "structure_function": structure_function.to(dtype=field.dtype),
        "scaling_exponent": scaling_exponent.to(dtype=field.dtype),
    }


def compute_dynamical_embedding(
    state_histories: Dict[str, torch.Tensor],
    n_components: int = 3,
) -> Dict[str, Any]:
    """Compute low-dimensional embedding using SVD on state trajectories.

    Mathematical Expression:
        X = [x1(t), x2(t), ..., xn(t)]^T
        U, S, V = SVD(X)
        embedding = X @ V[:, :k]
        explained_variance = S_k^2 / sum(S^2)

    References:
        - Broomhead and King (1986), Physica D.
        - Ghil et al. (2002), Rev. Mod. Phys.

    Parameters:
        state_histories: mapping variable name -> Tensor[n_timesteps]
        n_components: embedding dimension
    """
    if not state_histories:
        raise ValueError("state_histories cannot be empty.")

    lengths = [tensor.reshape(-1).numel() for tensor in state_histories.values()]
    min_length = min(lengths)
    if min_length < 2:
        raise ValueError("state histories must contain at least 2 samples.")

    ordered_names = sorted(state_histories.keys())
    trajectory_columns = []
    for name in ordered_names:
        vector = state_histories[name].reshape(-1)[:min_length].to(dtype=torch.float64)
        trajectory_columns.append(vector)
    trajectory_matrix = torch.stack(trajectory_columns, dim=1)  # [time, vars]

    centered = trajectory_matrix - torch.mean(trajectory_matrix, dim=0, keepdim=True)
    u_mat, singular_values, vh_mat = torch.linalg.svd(centered, full_matrices=False)

    k = max(1, min(int(n_components), int(vh_mat.shape[0])))
    components = vh_mat[:k].transpose(0, 1)  # [vars, k]
    embedding = centered @ components
    variance_all = singular_values ** 2
    variance_top = variance_all[:k]
    explained_variance_ratio = variance_top / torch.sum(variance_all)

    return {
        "variables": ordered_names,
        "embedding": embedding.to(dtype=torch.float32),
        "components": components.to(dtype=torch.float32),
        "explained_variance_ratio": explained_variance_ratio.to(dtype=torch.float32),
        "singular_values": singular_values.to(dtype=torch.float32),
    }


def compute_mode_similarity_score(
    embedding_a: Dict[str, Any],
    embedding_b: Dict[str, Any],
) -> float:
    """Compute similarity score between two dynamical embeddings.

    Mathematical Expression:
        cos_similarity_i = (v_a_i dot v_b_i) / (|v_a_i| |v_b_i|)
        score = mean_i(|cos_similarity_i|)

    References:
        - Ghil et al. (2002), Rev. Mod. Phys.

    Parameters:
        embedding_a: output dict from compute_dynamical_embedding
        embedding_b: output dict from compute_dynamical_embedding
    """
    comp_a = embedding_a.get("components")
    comp_b = embedding_b.get("components")
    if not isinstance(comp_a, torch.Tensor) or not isinstance(comp_b, torch.Tensor):
        raise ValueError("Both embeddings must contain tensor field 'components'.")
    if comp_a.ndim != 2 or comp_b.ndim != 2:
        raise ValueError("Embedding components must be rank-2 tensors.")

    common_modes = min(comp_a.shape[1], comp_b.shape[1])
    if common_modes <= 0:
        return 0.0

    cos_scores = []
    eps = torch.as_tensor(1e-12, dtype=torch.float64)
    for mode_idx in range(common_modes):
        vec_a = comp_a[:, mode_idx].to(dtype=torch.float64)
        vec_b = comp_b[:, mode_idx].to(dtype=torch.float64)
        dot = torch.dot(vec_a, vec_b)
        denom = torch.maximum(torch.linalg.norm(vec_a) * torch.linalg.norm(vec_b), eps)
        cos = torch.abs(dot / denom)
        cos_scores.append(torch.clamp(cos, 0.0, 1.0))

    return float(torch.mean(torch.stack(cos_scores)).item())


__all__ = [
    "SYNTHETIC_BENCHMARK_SCOPE",
    "create_benchmark_function",
    "create_test_inputs",
    "compute_spectral_signature",
    "compute_structure_function",
    "compute_dynamical_embedding",
    "compute_mode_similarity_score",
]
