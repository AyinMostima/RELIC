import pytest
import torch

from relic.computation.operators.functional import apply_advection


def _gaussian_2d(nx: int, ny: int, dtype: torch.dtype) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx, dtype=dtype)
    y = torch.linspace(0.0, 1.0, ny, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    sigma2 = 0.01
    center_x, center_y = 0.5, 0.5
    return torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma2))


def _gaussian_axis(length: int, center: float, sigma2: float, dtype: torch.dtype) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, length, dtype=dtype)
    return torch.exp(-((grid - center) ** 2) / (2.0 * sigma2))


def test_2d_advection_preserves_mass_with_periodic_boundaries():
    dtype = torch.float64
    nx = ny = 32
    dx = dy = 1.0 / nx
    dt = 0.2 * min(dx / 1.0, dy / 0.5)
    n_steps = 100

    concentration = _gaussian_2d(nx, ny, dtype=dtype).unsqueeze(0)
    initial_mass = concentration.sum().item()

    u = torch.full_like(concentration, 1.0)
    v = torch.full_like(concentration, 0.5)

    for _ in range(n_steps):
        result = apply_advection(
            concentration=concentration,
            velocity=(u, v),
            grid_spacing=(dy, dx),
            spatial_dims=2,
            scheme="upwind",
            boundary_method="periodic",
            time_step=dt,
            check_cfl=True,
            time_integration_method="euler",
        )
        concentration = result["concentration"]

    final_mass = concentration.sum().item()
    relative_error = abs(final_mass - initial_mass) / max(abs(initial_mass), 1.0e-30)

    assert relative_error < 1.0e-12


def test_4d_advection_preserves_mass_with_periodic_boundaries():
    dtype = torch.float64
    nx = ny = nz = nw = 8
    dx = dy = dz = dw = 1.0 / nx
    dt = 0.15 * min(dx / 1.0, dy / 0.5, dz / 0.25, dw / 0.1)
    n_steps = 20

    gx = _gaussian_axis(nx, 0.30, 0.02, dtype)
    gy = _gaussian_axis(ny, 0.45, 0.02, dtype)
    gz = _gaussian_axis(nz, 0.60, 0.02, dtype)
    gw = _gaussian_axis(nw, 0.25, 0.02, dtype)

    concentration = (
        gw[:, None, None, None]
        * gz[None, :, None, None]
        * gy[None, None, :, None]
        * gx[None, None, None, :]
    ).unsqueeze(0)
    initial_mass = concentration.sum().item()

    u = torch.full_like(concentration, 1.0)
    v = torch.full_like(concentration, 0.5)
    w = torch.full_like(concentration, 0.25)
    q = torch.full_like(concentration, 0.1)

    for _ in range(n_steps):
        result = apply_advection(
            concentration=concentration,
            velocity=(u, v, w, q),
            grid_spacing=(dw, dz, dy, dx),
            spatial_dims=4,
            scheme="upwind",
            boundary_method="periodic",
            time_step=dt,
            check_cfl=True,
            time_integration_method="euler",
        )
        concentration = result["concentration"]

    final_mass = concentration.sum().item()
    relative_error = abs(final_mass - initial_mass) / max(abs(initial_mass), 1.0e-30)

    assert relative_error < 1.0e-11


def test_periodic_zero_field_has_no_hidden_mass_source():
    concentration = torch.zeros((1, 8), dtype=torch.float64)
    velocity = torch.ones_like(concentration)

    result = apply_advection(
        concentration=concentration,
        velocity=velocity,
        grid_spacing=1.0,
        spatial_dims=1,
        scheme="upwind",
        boundary_method="periodic",
        time_step=0.25,
        check_cfl=True,
        time_integration_method="euler",
    )

    assert torch.equal(result["advection_rate"], torch.zeros_like(concentration))
    assert torch.equal(result["concentration"], concentration)
    assert result["boundary_fluxes"] == {}


def test_open_1d_advection_exposes_outward_flux_and_closes_mass_ledger():
    concentration = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    velocity = torch.ones_like(concentration)
    boundary_state = torch.zeros_like(concentration)
    dx = 1.0
    dt = 0.1

    result = apply_advection(
        concentration=concentration,
        velocity=velocity,
        grid_spacing=dx,
        spatial_dims=1,
        scheme="upwind",
        boundary_method="open",
        boundary_state=boundary_state,
        time_step=dt,
        check_cfl=True,
        time_integration_method="euler",
    )

    boundary_fluxes = result["boundary_fluxes"]
    assert set(boundary_fluxes) == {"x_min", "x_max"}
    torch.testing.assert_close(boundary_fluxes["x_min"], torch.tensor([-0.0]))
    torch.testing.assert_close(boundary_fluxes["x_max"], torch.tensor([4.0]))

    storage_rate = torch.sum(result["advection_rate"]) * dx
    outward_boundary_rate = sum(torch.sum(value) for value in boundary_fluxes.values())
    torch.testing.assert_close(storage_rate + outward_boundary_rate, torch.zeros(()))
    torch.testing.assert_close(
        torch.sum(result["concentration"] - concentration),
        -outward_boundary_rate * dt / dx,
    )
    torch.testing.assert_close(
        result["conservation_error"],
        torch.zeros((), dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )


def test_open_2d_advection_exposes_face_fluxes_and_closes_mass_ledger():
    concentration = torch.arange(1.0, 7.0, dtype=torch.float64).reshape(1, 2, 3)
    velocity_x = torch.ones_like(concentration)
    velocity_y = torch.full_like(concentration, 0.5)
    boundary_state = torch.zeros_like(concentration)
    dy, dx = 2.0, 1.0

    result = apply_advection(
        concentration=concentration,
        velocity=(velocity_x, velocity_y),
        grid_spacing=(dy, dx),
        spatial_dims=2,
        scheme="upwind",
        boundary_method="open",
        boundary_state=boundary_state,
        time_step=0.1,
        check_cfl=True,
        time_integration_method="euler",
    )

    boundary_fluxes = result["boundary_fluxes"]
    assert set(boundary_fluxes) == {"x_min", "x_max", "y_min", "y_max"}
    storage_rate = torch.sum(result["advection_rate"]) * dx * dy
    outward_boundary_rate = (
        torch.sum(boundary_fluxes["x_min"]) * dy
        + torch.sum(boundary_fluxes["x_max"]) * dy
        + torch.sum(boundary_fluxes["y_min"]) * dx
        + torch.sum(boundary_fluxes["y_max"]) * dx
    )
    torch.testing.assert_close(storage_rate + outward_boundary_rate, torch.zeros(()))
    assert float(result["conservation_error"]) < 1.0e-14


def test_unbatched_open_3d_advection_closes_face_flux_mass_ledger():
    concentration = torch.arange(1.0, 25.0, dtype=torch.float64).reshape(2, 3, 4)
    velocity_x = torch.ones_like(concentration)
    velocity_y = torch.full_like(concentration, 0.5)
    velocity_z = torch.full_like(concentration, 0.25)
    boundary_state = torch.zeros_like(concentration)
    dz, dy, dx = 3.0, 2.0, 1.0

    result = apply_advection(
        concentration=concentration,
        velocity=(velocity_x, velocity_y, velocity_z),
        grid_spacing=(dz, dy, dx),
        spatial_dims=3,
        scheme="upwind",
        boundary_method="open",
        boundary_state=boundary_state,
        time_step=0.1,
        check_cfl=True,
        time_integration_method="euler",
    )

    boundary_fluxes = result["boundary_fluxes"]
    assert set(boundary_fluxes) == {
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
    }
    assert boundary_fluxes["x_min"].shape == (2, 3)
    assert boundary_fluxes["y_min"].shape == (2, 4)
    assert boundary_fluxes["z_min"].shape == (3, 4)

    storage_rate = torch.sum(result["advection_rate"]) * dx * dy * dz
    outward_boundary_rate = (
        torch.sum(boundary_fluxes["x_min"]) * dy * dz
        + torch.sum(boundary_fluxes["x_max"]) * dy * dz
        + torch.sum(boundary_fluxes["y_min"]) * dx * dz
        + torch.sum(boundary_fluxes["y_max"]) * dx * dz
        + torch.sum(boundary_fluxes["z_min"]) * dx * dy
        + torch.sum(boundary_fluxes["z_max"]) * dx * dy
    )
    torch.testing.assert_close(storage_rate + outward_boundary_rate, torch.zeros(()))


def test_advection_operator_rejects_unstable_requested_timestep():
    concentration = torch.ones((1, 8), dtype=torch.float64)
    velocity = torch.full_like(concentration, 2.0)

    with pytest.raises(ValueError, match="CFL"):
        apply_advection(
            concentration=concentration,
            velocity=velocity,
            grid_spacing=1.0,
            spatial_dims=1,
            scheme="upwind",
            boundary_method="periodic",
            time_step=1.0,
            check_cfl=True,
            time_integration_method="euler",
        )


def test_2d_divergent_velocity_uses_outward_face_courant_limit():
    concentration = torch.zeros((1, 3, 3), dtype=torch.float64)
    concentration[0, 1, 1] = 1.0
    u = torch.zeros_like(concentration)
    v = torch.zeros_like(concentration)
    u[0, 1, 0] = -1.0
    u[0, 1, 2] = 1.0
    v[0, 0, 1] = -1.0
    v[0, 2, 1] = 1.0

    with pytest.raises(ValueError, match="CFL"):
        apply_advection(
            concentration,
            (u, v),
            (1.0, 1.0),
            spatial_dims=2,
            scheme="upwind",
            boundary_method="periodic",
            time_step=1.0,
            check_cfl=True,
            time_integration_method="euler",
        )


def test_4d_advection_broadcasts_scalar_velocity_components():
    concentration = torch.ones((2, 2, 2, 2), dtype=torch.float64)
    velocity = tuple(torch.tensor(value, dtype=torch.float64) for value in (0.1, 0.2, 0.3, 0.4))
    result = apply_advection(
        concentration,
        velocity,
        (1.0, 1.0, 1.0, 1.0),
        spatial_dims=4,
        scheme="upwind",
        boundary_method="periodic",
        time_step=0.1,
        check_cfl=True,
        time_integration_method="euler",
    )
    torch.testing.assert_close(result["concentration"], concentration)


@pytest.mark.parametrize("scheme", ["superbee", "van_leer"])
def test_open_tvd_advection_is_symmetric_under_axis_and_velocity_reversal(scheme):
    concentration = torch.tensor(
        [[0.0, 0.0, 0.2, 1.0, 0.8, 0.1, 0.0, 0.0]],
        dtype=torch.float64,
    )
    positive_velocity = torch.ones_like(concentration)
    boundary_state = torch.zeros_like(concentration)

    positive_result = apply_advection(
        concentration=concentration,
        velocity=positive_velocity,
        grid_spacing=1.0,
        spatial_dims=1,
        scheme=scheme,
        boundary_method="open",
        boundary_state=boundary_state,
        time_step=0.2,
        check_cfl=True,
        time_integration_method="euler",
    )["concentration"]

    reversed_concentration = torch.flip(concentration, dims=(-1,))
    negative_result = apply_advection(
        concentration=reversed_concentration,
        velocity=-positive_velocity,
        grid_spacing=1.0,
        spatial_dims=1,
        scheme=scheme,
        boundary_method="open",
        boundary_state=boundary_state,
        time_step=0.2,
        check_cfl=True,
        time_integration_method="euler",
    )["concentration"]

    torch.testing.assert_close(
        positive_result,
        torch.flip(negative_result, dims=(-1,)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("scheme", ["superbee", "van_leer"])
def test_tvd_advection_is_invariant_to_concentration_unit_scale(scheme):
    """Dimensionless limiter ratios must not depend on concentration scale."""
    concentration = torch.tensor(
        [[0.0, 0.0, 0.2, 1.0, 0.8, 0.1, 0.0, 0.0]],
        dtype=torch.float64,
    )
    velocity = torch.ones_like(concentration)
    scale = 1.0e-15

    baseline = apply_advection(
        concentration=concentration,
        velocity=velocity,
        grid_spacing=1.0,
        spatial_dims=1,
        scheme=scheme,
        boundary_method="periodic",
        time_step=0.2, check_cfl=True,
    )["concentration"]
    scaled = apply_advection(
        concentration=concentration * scale,
        velocity=velocity,
        grid_spacing=1.0,
        spatial_dims=1,
        scheme=scheme,
        boundary_method="periodic",
        time_step=0.2, check_cfl=True,
    )["concentration"]

    torch.testing.assert_close(
        scaled / scale,
        baseline,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("method", ["implicit_midpoint", "crank_nicolson"])
def test_implicit_open_advection_uses_time_averaged_boundary_ledger(method):
    concentration = torch.linspace(1.0, 0.2, 10, dtype=torch.float64).unsqueeze(0)
    velocity = torch.full_like(concentration, 0.2)
    boundary_state = torch.zeros_like(concentration)
    dx = 1.0
    dt = 0.1

    result = apply_advection(
        concentration=concentration,
        velocity=velocity,
        grid_spacing=dx,
        spatial_dims=1,
        scheme="upwind",
        boundary_method="open",
        boundary_state=boundary_state,
        time_step=dt,
        check_cfl=True,
        time_integration_method=method,
        implicit_midpoint_max_iter=100,
        implicit_midpoint_tolerance=1.0e-13,
    )

    storage_change = torch.sum(result["concentration"] - concentration, dim=-1) * dx
    outward_export = dt * sum(result["boundary_fluxes"].values())
    torch.testing.assert_close(
        storage_change + outward_export,
        torch.zeros_like(storage_change),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("boundary_method", ["dirichlet", "open"])
def test_advection_inflow_boundaries_require_explicit_state(boundary_method):
    concentration = torch.ones((1, 4), dtype=torch.float64)

    with pytest.raises(ValueError, match="requires explicit boundary_state"):
        apply_advection(
            concentration=concentration,
            velocity=torch.ones_like(concentration),
            grid_spacing=1.0,
            spatial_dims=1,
            scheme="upwind",
            boundary_method=boundary_method,
            time_step=0.1, check_cfl=True,
        )


def test_periodic_advection_rejects_unused_boundary_state():
    concentration = torch.ones((1, 4), dtype=torch.float64)

    with pytest.raises(ValueError, match="only supported"):
        apply_advection(
            concentration=concentration,
            velocity=torch.ones_like(concentration),
            grid_spacing=1.0,
            spatial_dims=1,
            scheme="upwind",
            boundary_method="periodic",
            boundary_state=torch.zeros_like(concentration),
            time_step=0.1, check_cfl=True,
        )


def test_advection_rejects_complex_boundary_state_before_cast():
    """Complex boundary states must fail without discarding the imaginary part."""
    concentration = torch.ones((1, 4), dtype=torch.float64)
    boundary_state = torch.full((1, 4), 1.0 + 0.5j, dtype=torch.complex128)

    with pytest.raises(TypeError, match="real floating-point"):
        apply_advection(
            concentration=concentration,
            velocity=torch.ones_like(concentration),
            grid_spacing=1.0,
            spatial_dims=1,
            scheme="upwind",
            boundary_method="open",
            boundary_state=boundary_state,
            time_step=0.1, check_cfl=True,
        )


def test_open_and_extrapolate_advection_have_distinct_inflow_laws():
    concentration = torch.tensor([[2.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    velocity = torch.ones_like(concentration)

    open_result = apply_advection(
        concentration=concentration,
        velocity=velocity,
        grid_spacing=1.0,
        spatial_dims=1,
        scheme="upwind",
        boundary_method="open",
        boundary_state=torch.zeros_like(concentration), time_step=None, check_cfl=True,
    )
    extrapolated_result = apply_advection(
        concentration=concentration,
        velocity=velocity,
        grid_spacing=1.0,
        spatial_dims=1,
        scheme="upwind",
        boundary_method="extrapolate", time_step=None, check_cfl=True,
    )

    assert not torch.equal(
        open_result["advection_rate"],
        extrapolated_result["advection_rate"],
    )


def test_3d_advection_supports_multiple_leading_axes():
    concentration = torch.arange(
        1.0,
        2.0 * 3.0 * 2.0 * 3.0 * 4.0 + 1.0,
        dtype=torch.float64,
    ).reshape(2, 3, 2, 3, 4)
    velocity = tuple(
        torch.full_like(concentration, value)
        for value in (0.2, 0.1, 0.05)
    )

    result = apply_advection(
        concentration=concentration,
        velocity=velocity,
        grid_spacing=(3.0, 2.0, 1.0),
        spatial_dims=3,
        scheme="upwind",
        boundary_method="periodic",
        time_step=0.1,
        time_integration_method="euler", check_cfl=True,
    )

    assert result["concentration"].shape == concentration.shape
    torch.testing.assert_close(
        torch.sum(result["concentration"], dim=(-3, -2, -1)),
        torch.sum(concentration, dim=(-3, -2, -1)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
