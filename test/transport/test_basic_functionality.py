"""
Basic Functionality Tests for Transport Operators

Simple tests to verify the core functionality works correctly.
"""

import torch

# Import the operators to test
from relic.computation.operators.functional import (
    apply_diffusion, apply_advection, advance_advection_diffusion,
    compute_grid_metrics, compute_cfl_number, check_conservation
)

def test_basic_diffusion():
    """Test basic diffusion functionality."""
    print("Testing basic diffusion...")

    # Create simple test case
    conc = torch.ones((1, 11))  # Batch dim + 11 points
    dx = 0.1
    dt = 0.01
    diffusivity = 1e-5

    result = apply_diffusion(
        concentration=conc,
        diffusivity=diffusivity,
        grid_spacing=dx,
        spatial_dims=1,
        time_step=dt,
        check_stability=True,
        enforce_non_negative=True, boundary_method="zero"
    )

    # Verify result structure
    assert 'diffusion_rate' in result
    assert 'concentration' in result
    assert 'stability_number' in result
    assert result['diffusion_rate'].shape == conc.shape
    assert result['concentration'].shape == conc.shape
    assert torch.all(torch.isfinite(result['diffusion_rate']))
    assert torch.all(torch.isfinite(result['concentration']))

    # For constant field, diffusion rate should be approximately zero
    assert torch.allclose(result['diffusion_rate'], torch.zeros_like(result['diffusion_rate']), atol=1e-10)

    # Concentration should remain approximately constant for uniform field
    assert torch.allclose(result['concentration'], conc, atol=1e-10)

    print("[OK] Basic diffusion test passed")

def test_basic_advection():
    """Test basic advection functionality."""
    print("Testing basic advection...")

    # Create simple test case
    conc = torch.ones((1, 11))
    vel = torch.ones((1, 11)) * 0.5  # Constant velocity
    dx = 0.1
    dt = 0.01

    result = apply_advection(
        concentration=conc,
        velocity=vel,
        grid_spacing=dx,
        spatial_dims=1,
        time_step=dt,
        check_cfl=True, scheme="upwind", boundary_method="extrapolate"
    )

    # Verify result structure
    assert 'advection_rate' in result
    assert 'concentration' in result
    assert 'cfl_number' in result
    assert result['advection_rate'].shape == conc.shape
    assert result['concentration'].shape == conc.shape
    assert torch.all(torch.isfinite(result['advection_rate']))
    assert torch.all(torch.isfinite(result['concentration']))

    # For constant field and positive velocity, advection rate should be negative at inflow boundary
    # (concentration decreases due to outflow at boundaries)
    assert torch.all(result['advection_rate'] <= 0)  # Should be non-positive

    print("[OK] Basic advection test passed")

def test_combined_operator():
    """Test combined advection-diffusion operator."""
    print("Testing combined operator...")

    # Create simple test case
    conc = torch.ones((1, 11))
    vel = torch.ones((1, 11)) * 0.5
    dx = 0.1
    dt = 0.01
    diffusivity = 1e-5

    result = advance_advection_diffusion(
        concentration=conc,
        velocity=vel,
        diffusivity=diffusivity,
        grid_spacing=dx,
        spatial_dims=1,
        time_step=dt,
        splitting_method="sequential",
        check_stability=True,
        enforce_non_negative=True, advection_scheme="upwind", boundary_method="zero"
    )

    # Verify result structure
    assert 'concentration' in result
    assert 'advection_rate' in result
    assert 'diffusion_rate' in result
    assert 'conservation_error' in result
    assert result['concentration'].shape == conc.shape
    assert result['advection_rate'].shape == conc.shape
    assert result['diffusion_rate'].shape == conc.shape
    assert torch.all(torch.isfinite(result['concentration']))
    assert torch.all(torch.isfinite(result['advection_rate']))
    assert torch.all(torch.isfinite(result['diffusion_rate']))

    print("[OK] Combined operator test passed")

def test_utility_functions():
    """Test utility functions."""
    print("Testing utility functions...")
    
    # Test grid metrics
    metrics = compute_grid_metrics((11,), 0.1, torch.device("cpu"))
    assert 'volumes' in metrics
    assert 'face_areas' in metrics
    assert 'dx' in metrics
    
    # Test CFL number
    velocity = torch.tensor([1.0, 2.0, 0.5])
    cfl = compute_cfl_number(velocity, 0.1, 0.05)
    expected = torch.abs(velocity) * 0.05 / 0.1
    assert torch.allclose(cfl, expected)
    
    # Test conservation check
    conc_before = torch.ones((1, 10))
    conc_after = torch.ones((1, 10))
    source = torch.zeros((1, 10))
    dt = 0.1
    volumes = torch.ones(10)
    
    error = check_conservation(conc_before, conc_after, source, dt, volumes)
    assert error < 1e-10
    
    print("[OK] Utility functions test passed")

def test_gaussian_pulse_diffusion():
    """Test diffusion of Gaussian pulse."""
    print("Testing Gaussian pulse diffusion...")

    # Create Gaussian pulse
    x = torch.linspace(0, 1, 101)
    conc = torch.exp(-((x - 0.5) / 0.1) ** 2).unsqueeze(0)

    result = apply_diffusion(
        concentration=conc,
        diffusivity=1e-4,  # Higher diffusivity for visible effect
        grid_spacing=0.01,
        spatial_dims=1,
        time_step=0.01,
        check_stability=True,
        enforce_non_negative=True, boundary_method="zero"
    )

    # Peak should decrease
    diffusion_rate = result['diffusion_rate']
    center_idx = 50
    assert diffusion_rate[0, center_idx] < 0  # Diffusion should reduce peak

    # Verify physical realism
    new_conc = result['concentration']
    assert torch.all(new_conc >= 0)
    assert torch.all(torch.isfinite(new_conc))

    # Peak should be lower than original
    assert new_conc[0, center_idx] < conc[0, center_idx]

    print("[OK] Gaussian pulse diffusion test passed")

def test_pulse_translation():
    """Test advection of pulse."""
    print("Testing pulse translation...")

    # Create pulse
    x = torch.linspace(0, 1, 101)
    conc = torch.exp(-((x - 0.3) / 0.05) ** 2).unsqueeze(0)
    vel = torch.ones((1, 101)) * 0.5

    result = apply_advection(
        concentration=conc,
        velocity=vel,
        grid_spacing=0.01,
        spatial_dims=1,
        time_step=0.01,
        check_cfl=True, scheme="upwind", boundary_method="extrapolate"
    )

    # Peak should move in direction of velocity
    new_conc = result['concentration']

    peak_idx_before = torch.argmax(conc[0]).item()
    peak_idx_after = torch.argmax(new_conc[0]).item()

    # With positive velocity, peak should move right
    assert peak_idx_after >= peak_idx_before

    # Reject physical undershoot while allowing boundary roundoff below 1e-12.
    torch.testing.assert_close(
        new_conc,
        new_conc.clamp_min(0.0),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        result["conservation_error"],
        torch.zeros_like(result["conservation_error"]),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert torch.all(torch.isfinite(new_conc))

    print("[OK] Pulse translation test passed")

def test_check_conservation_default_tolerance_resolves_from_the_parameter_database():
    """check_conservation defaults to computation.operators.diagnostics.tolerance.

    Mathematical expression: error = |dM - dt integral(S dV)| / (|M_old| + 1e-12);
    M_old = 4 and dM = 2e-6 give a relative error of 5e-7.
    """
    import warnings
    from pathlib import Path

    import pytest
    import yaml

    from relic.parameters import get_global_manager

    key = "computation.operators.diagnostics.tolerance"
    path = (
        Path(__file__).resolve().parents[2]
        / "relic" / "parameters" / "database" / "relic" / "computation" / "operators.yaml"
    )
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"][key]
    for field_name in ("value", "units", "dtype", "bounds", "category", "description",
                       "references", "tags", "requires_grad", "provenance"):
        assert field_name in entry, field_name
    manager = get_global_manager()
    original = manager.get(key, required=True)
    assert original == pytest.approx(float(entry["value"]))
    old = torch.ones(4, dtype=torch.float64)
    new = old.clone()
    new[0] = new[0] + 2.0e-6

    def _violation_warnings(**kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            error = check_conservation(old, new, **kwargs)
        assert float(error) == pytest.approx(5.0e-7, rel=1.0e-6)
        return [item for item in caught if "exceeds tolerance" in str(item.message)]

    assert _violation_warnings() == []
    assert len(_violation_warnings(tolerance=1.0e-7)) == 1
    try:
        manager.set_value(key, 1.0e-7)
        assert len(_violation_warnings()) == 1
    finally:
        manager.set_value(key, original)
    assert manager.get(key, required=True) == original


def test_check_conservation_relative_guard_scales_with_dtype_precision():
    """The denominator guard is a roundoff floor, not an absolute 1e-12 mass.

    Mathematical expression: error = |dM - E| / max(|M_old| + n eps S, tiny) with
    S = sum|phi_old dV| + sum|phi_new dV| + |E|. Tiny masses keep their relative
    error, all-zero fields give 0 (never nan), and a cancelling signed field with
    a real change is still reported as a large violation.
    """
    import pytest

    for dtype in (torch.float32, torch.float64):
        old = torch.full((4,), 1.0e-20, dtype=dtype)
        new = old.clone()
        new[0] = new[0] * (1.0 + 4.0e-3)
        error = check_conservation(old, new, tolerance=1.0, warn_on_violation=False)
        assert float(error) == pytest.approx(1.0e-3, rel=1.0e-3), dtype
        zero = torch.zeros(4, dtype=dtype)
        zero_error = check_conservation(zero, zero.clone(), tolerance=1.0, warn_on_violation=False)
        assert float(zero_error) == 0.0, dtype
        signed_old = torch.tensor([1.0, -1.0], dtype=dtype)
        signed_new = torch.tensor([1.0, -1.0 + 1.0e-2], dtype=dtype)
        signed_error = check_conservation(signed_old, signed_new, tolerance=1.0, warn_on_violation=False)
        assert float(signed_error) > 1.0e3, dtype


if __name__ == "__main__":
    print("Running basic functionality tests...")
    print("=" * 50)
    
    test_basic_diffusion()
    test_basic_advection()
    test_combined_operator()
    test_utility_functions()
    test_gaussian_pulse_diffusion()
    test_pulse_translation()
    
    print("=" * 50)
    print("All basic functionality tests passed! [OK]")
    print("The unified transport operators are working correctly.")

