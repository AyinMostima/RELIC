import pytest
import torch

from relic.computation.operators.functional import apply_diffusion, compute_laplacian_1d


def _gaussian_1d(nx: int, dtype: torch.dtype) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx, dtype=dtype)
    sigma2 = 0.01
    center = 0.5
    return torch.exp(-((x - center) ** 2) / (2.0 * sigma2))


def test_variable_diffusivity_uses_conservative_form_and_preserves_mass():
    dtype = torch.float64
    nx = 128
    dx = 1.0 / nx
    dt = 0.1 * dx * dx
    n_steps = 200

    concentration_new = _gaussian_1d(nx, dtype).unsqueeze(0)
    concentration_old = concentration_new.clone()

    diffusivity = torch.ones_like(concentration_new)
    diffusivity[..., nx // 2 :] = 0.1

    initial_mass = concentration_new.sum().item()

    for _ in range(n_steps):
        # New conservative path in apply_diffusion (tensor diffusivity).
        result = apply_diffusion(
            concentration=concentration_new,
            diffusivity=diffusivity,
            diffusivity_layout="scalar_field",
            grid_spacing=dx,
            spatial_dims=1,
            boundary_method="periodic",
            time_step=dt,
            check_stability=True,
            time_integration_method="euler", enforce_non_negative=True,
        )
        concentration_new = result["concentration"]

        # Legacy non-conservative reference: K * laplacian(c).
        lap = compute_laplacian_1d(concentration_old, dx, boundary_method="periodic")
        concentration_old = concentration_old + dt * (diffusivity * lap)

    final_mass_new = concentration_new.sum().item()
    final_mass_old = concentration_old.sum().item()

    rel_err_new = abs(final_mass_new - initial_mass) / max(abs(initial_mass), 1.0e-30)
    rel_err_old = abs(final_mass_old - initial_mass) / max(abs(initial_mass), 1.0e-30)

    assert rel_err_new < 1.0e-12
    assert rel_err_old > rel_err_new


def test_scalar_and_tensor_diffusivity_share_closed_zero_flux_semantics():
    concentration = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=torch.float64)
    scalar_diffusivity = 0.1
    tensor_diffusivity = torch.full_like(concentration, scalar_diffusivity)

    scalar_result = apply_diffusion(
        concentration=concentration,
        diffusivity=scalar_diffusivity,
        grid_spacing=1.0,
        spatial_dims=1,
        boundary_method="zero",
        time_step=1.0,
        check_stability=True,
        time_integration_method="euler", enforce_non_negative=True,
    )
    tensor_result = apply_diffusion(
        concentration=concentration,
        diffusivity=tensor_diffusivity,
        diffusivity_layout="scalar_field",
        grid_spacing=1.0,
        spatial_dims=1,
        boundary_method="zero",
        time_step=1.0,
        check_stability=True,
        time_integration_method="euler", enforce_non_negative=True,
    )

    expected_rate = torch.tensor([[0.1, 0.1, 0.2, -0.4]], dtype=torch.float64)
    torch.testing.assert_close(scalar_result["diffusion_rate"], expected_rate)
    torch.testing.assert_close(
        scalar_result["diffusion_rate"], tensor_result["diffusion_rate"]
    )
    torch.testing.assert_close(
        torch.sum(scalar_result["concentration"]), torch.sum(concentration)
    )
    torch.testing.assert_close(
        torch.sum(tensor_result["concentration"]), torch.sum(concentration)
    )
    for result in (scalar_result, tensor_result):
        assert set(result["boundary_fluxes"]) == {"x_min", "x_max"}
        torch.testing.assert_close(
            result["boundary_fluxes"]["x_min"],
            torch.zeros(1, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result["boundary_fluxes"]["x_max"],
            torch.zeros(1, dtype=torch.float64),
        )
    assert scalar_result["conservation_error"] == 0.0
    assert tensor_result["conservation_error"] == 0.0


def test_directional_diffusivity_uses_storage_axis_order_and_conserves_mass():
    """A diagonal tensor must use (Ky, Kx) without a second stencil path."""
    concentration = torch.zeros((1, 3, 3), dtype=torch.float64)
    concentration[0, 1, 1] = 1.0
    directional_diffusivity = torch.tensor([0.1, 0.4], dtype=torch.float64)

    result = apply_diffusion(
        concentration=concentration,
        diffusivity=directional_diffusivity,
        diffusivity_layout="directional",
        grid_spacing=(1.0, 1.0),
        spatial_dims=2,
        boundary_method="zero",
        time_step=0.1,
        check_stability=True,
        time_integration_method="euler", enforce_non_negative=True,
    )

    expected_rate = torch.zeros_like(concentration)
    expected_rate[0, 1, 0] = 0.4
    expected_rate[0, 1, 2] = 0.4
    expected_rate[0, 0, 1] = 0.1
    expected_rate[0, 2, 1] = 0.1
    expected_rate[0, 1, 1] = -1.0
    torch.testing.assert_close(result["diffusion_rate"], expected_rate)
    torch.testing.assert_close(
        torch.sum(result["concentration"]),
        torch.sum(concentration),
        rtol=0.0,
        atol=1.0e-14,
    )
    torch.testing.assert_close(
        result["stability_number"],
        torch.full_like(concentration, 0.05),
    )
    assert set(result["boundary_fluxes"]) == {
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    }
    for boundary_flux in result["boundary_fluxes"].values():
        torch.testing.assert_close(
            boundary_flux,
            torch.zeros_like(boundary_flux),
        )
    torch.testing.assert_close(result["conservation_error"], torch.zeros(()))


def test_diffusion_source_is_integrated_once_and_closes_open_mass_ledger():
    concentration = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    source = torch.tensor([[0.1, 0.0, -0.05, 0.2]], dtype=torch.float64)
    dt = 0.1

    diffusion_only = apply_diffusion(
        concentration=concentration,
        diffusivity=0.1,
        grid_spacing=1.0,
        spatial_dims=1,
        boundary_method="zero",
        time_step=dt,
        check_stability=True,
        time_integration_method="euler", enforce_non_negative=True,
    )
    with_source = apply_diffusion(
        concentration=concentration,
        diffusivity=0.1,
        grid_spacing=1.0,
        spatial_dims=1,
        boundary_method="zero",
        source_term=source,
        time_step=dt,
        check_stability=True,
        time_integration_method="euler", enforce_non_negative=True,
    )

    torch.testing.assert_close(
        with_source["concentration"],
        diffusion_only["concentration"] + source * dt,
    )
    storage_change = torch.sum(with_source["concentration"] - concentration)
    source_input = torch.sum(source) * dt
    torch.testing.assert_close(storage_change, source_input)
    torch.testing.assert_close(with_source["conservation_error"], torch.zeros(()))


def test_conservation_audit_preserves_autograd_without_scalar_conversion_warning():
    concentration = torch.tensor(
        [[1.0, 2.0, 4.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = apply_diffusion(
        concentration=concentration,
        diffusivity=0.1,
        grid_spacing=1.0,
        spatial_dims=1,
        boundary_method="periodic",
        time_step=0.1,
        check_stability=True,
        time_integration_method="euler", enforce_non_negative=True,
    )

    loss = torch.sum(result["concentration"] ** 2)
    loss.backward()

    assert concentration.grad is not None
    assert torch.isfinite(concentration.grad).all()
    torch.testing.assert_close(result["conservation_error"], torch.zeros(()))


def test_dirichlet_diffusion_requires_explicit_boundary_state():
    concentration = torch.ones((1, 4), dtype=torch.float64)

    with pytest.raises(ValueError, match="boundary_state"):
        apply_diffusion(
            concentration=concentration,
            diffusivity=0.1,
            grid_spacing=1.0,
            spatial_dims=1,
            boundary_method="dirichlet",
            time_step=0.1, check_stability=True, enforce_non_negative=True,
        )


def test_dirichlet_diffusion_exposes_outward_flux_and_closes_mass_ledger():
    concentration = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    boundary_state = torch.zeros_like(concentration)
    diffusivity = 0.5
    dx = 1.0
    dt = 0.1

    result = apply_diffusion(
        concentration=concentration,
        diffusivity=diffusivity,
        grid_spacing=dx,
        spatial_dims=1,
        boundary_method="dirichlet",
        boundary_state=boundary_state,
        time_step=dt,
        check_stability=True,
        time_integration_method="euler", enforce_non_negative=True,
    )

    boundary_fluxes = result["boundary_fluxes"]
    assert set(boundary_fluxes) == {"x_min", "x_max"}
    torch.testing.assert_close(
        boundary_fluxes["x_min"], torch.tensor([1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        boundary_fluxes["x_max"], torch.tensor([4.0], dtype=torch.float64)
    )
    storage_change = torch.sum(result["concentration"] - concentration) * dx
    outward_export = sum(torch.sum(flux) for flux in boundary_fluxes.values()) * dt
    torch.testing.assert_close(storage_change + outward_export, torch.zeros(()))
    torch.testing.assert_close(result["conservation_error"], torch.zeros(()))


def test_dirichlet_face_geometry_has_stricter_explicit_stability_bound():
    concentration = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="Diffusion stability violation"):
        apply_diffusion(
            concentration=concentration,
            diffusivity=1.0,
            grid_spacing=1.0,
            spatial_dims=1,
            boundary_method="dirichlet",
            boundary_state=torch.zeros_like(concentration),
            time_step=0.4,
            check_stability=True,
            time_integration_method="euler", enforce_non_negative=True,
        )


def test_3d_dirichlet_diffusion_reports_canonical_face_shapes_per_batch():
    concentration = torch.arange(1.0, 49.0, dtype=torch.float64).reshape(2, 2, 3, 4)
    boundary_state = torch.zeros_like(concentration)
    diffusivity = 0.01
    dz, dy, dx = 3.0, 2.0, 1.0
    dt = 0.1

    result = apply_diffusion(
        concentration=concentration,
        diffusivity=diffusivity,
        grid_spacing=(dz, dy, dx),
        spatial_dims=3,
        boundary_method="dirichlet",
        boundary_state=boundary_state,
        time_step=dt,
        check_stability=True,
        time_integration_method="euler", enforce_non_negative=True,
    )

    boundary_fluxes = result["boundary_fluxes"]
    assert boundary_fluxes["x_min"].shape == (2, 2, 3)
    assert boundary_fluxes["y_min"].shape == (2, 2, 4)
    assert boundary_fluxes["z_min"].shape == (2, 3, 4)

    cell_volume = dx * dy * dz
    storage_change = torch.sum(
        result["concentration"] - concentration,
        dim=(1, 2, 3),
    ) * cell_volume
    outward_export = dt * (
        torch.sum(boundary_fluxes["x_min"], dim=(1, 2)) * dy * dz
        + torch.sum(boundary_fluxes["x_max"], dim=(1, 2)) * dy * dz
        + torch.sum(boundary_fluxes["y_min"], dim=(1, 2)) * dx * dz
        + torch.sum(boundary_fluxes["y_max"], dim=(1, 2)) * dx * dz
        + torch.sum(boundary_fluxes["z_min"], dim=(1, 2)) * dx * dy
        + torch.sum(boundary_fluxes["z_max"], dim=(1, 2)) * dx * dy
    )
    torch.testing.assert_close(
        storage_change + outward_export,
        torch.zeros_like(storage_change),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("boundary_state", "message"),
    [
        (torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]), "finite"),
        (torch.tensor([[-1.0, 0.0, 0.0, 0.0]]), "non-negative"),
    ],
)
def test_dirichlet_boundary_state_validation(boundary_state, message):
    with pytest.raises(ValueError, match=message):
        apply_diffusion(
            concentration=torch.ones((1, 4)),
            diffusivity=0.1,
            grid_spacing=1.0,
            spatial_dims=1,
            boundary_method="dirichlet",
            boundary_state=boundary_state,
            time_step=0.1,
            enforce_non_negative=True, check_stability=True,
        )


def test_closed_diffusion_rejects_unused_boundary_state():
    with pytest.raises(ValueError, match="boundary_state.*Dirichlet"):
        apply_diffusion(
            concentration=torch.ones((1, 4)),
            diffusivity=0.1,
            grid_spacing=1.0,
            spatial_dims=1,
            boundary_method="zero",
            boundary_state=torch.zeros((1, 4)),
            time_step=0.1, check_stability=True, enforce_non_negative=True,
        )


def test_diffusion_requires_floating_concentration():
    with pytest.raises(TypeError, match="floating"):
        apply_diffusion(
            concentration=torch.ones((1, 4), dtype=torch.int64),
            diffusivity=0.1,
            grid_spacing=1.0,
            spatial_dims=1,
            time_step=0.1, boundary_method="zero", check_stability=True, enforce_non_negative=True,
        )


def test_diffusion_rejects_complex_container_diffusivity_before_cast():
    """Complex diffusivity must fail without discarding its imaginary part."""
    concentration = torch.ones((1, 4), dtype=torch.float64)
    diffusivity = [[0.1 + 0.2j, 0.1 + 0.2j, 0.1 + 0.2j, 0.1 + 0.2j]]

    with pytest.raises(TypeError, match="real floating-point"):
        apply_diffusion(
            concentration=concentration,
            diffusivity=diffusivity,
            grid_spacing=1.0,
            spatial_dims=1,
            time_step=0.1, boundary_method="zero", check_stability=True, enforce_non_negative=True,
        )


def test_diffusion_rejects_complex_dirichlet_boundary_before_cast():
    """Complex boundary states must fail without an implicit real cast."""
    concentration = torch.ones((1, 4), dtype=torch.float64)
    boundary_state = torch.full((1, 4), 1.0 + 0.5j, dtype=torch.complex128)

    with pytest.raises(TypeError, match="real floating-point"):
        apply_diffusion(
            concentration=concentration,
            diffusivity=0.1,
            grid_spacing=1.0,
            spatial_dims=1,
            boundary_method="dirichlet",
            boundary_state=boundary_state,
            time_step=0.1, check_stability=True, enforce_non_negative=True,
        )


@pytest.mark.parametrize("method", ["implicit_midpoint", "crank_nicolson"])
def test_implicit_diffusion_fails_when_canonical_solver_does_not_converge(method):
    concentration = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    with pytest.raises(RuntimeError, match=rf"{method}.*iterations.*residual.*dt"):
        apply_diffusion(
            concentration=concentration,
            diffusivity=1.0,
            grid_spacing=1.0,
            spatial_dims=1,
            boundary_method="periodic",
            time_step=1.0,
            check_stability=True,
            enforce_non_negative=False,
            time_integration_method=method,
            implicit_midpoint_max_iter=2,
            implicit_midpoint_tolerance=1.0e-12,
        )


def test_unbatched_one_dimensional_diffusion_preserves_public_shape():
    concentration = torch.tensor([1.0, 0.5, 0.0, 0.0], dtype=torch.float64)

    result = apply_diffusion(
        concentration=concentration,
        diffusivity=0.1,
        grid_spacing=1.0,
        spatial_dims=1,
        boundary_method="periodic",
        time_step=0.1, check_stability=True, enforce_non_negative=True,
    )

    assert result["concentration"].shape == concentration.shape
    assert result["diffusion_rate"].shape == concentration.shape
    assert result["boundary_fluxes"] == {}
    torch.testing.assert_close(
        torch.sum(result["concentration"]),
        torch.sum(concentration),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def _ambiguous_case():
    """A 2 x 2 grid with a length-2 coefficient: an x-profile or (Ky, Kx)."""
    concentration = torch.zeros((2, 2), dtype=torch.float64)
    concentration[0, 0] = 1.0
    return concentration, torch.tensor([0.1, 0.4], dtype=torch.float64)


def _diffuse(concentration, diffusivity, **layout):
    return apply_diffusion(
        concentration=concentration,
        diffusivity=diffusivity,
        grid_spacing=(1.0, 1.0),
        spatial_dims=2,
        boundary_method="zero",
        time_step=0.1,
        check_stability=True,
        time_integration_method="euler",
        enforce_non_negative=False,
        **layout,
    )


def test_ambiguous_diffusivity_layout_must_be_declared():
    """Both readings are valid and differ, so the layout is not guessed."""
    concentration, diffusivity = _ambiguous_case()
    with pytest.raises(ValueError, match="diffusivity_layout"):
        _diffuse(concentration, diffusivity)


def test_declared_diffusivity_layout_selects_the_reading():
    concentration, diffusivity = _ambiguous_case()
    directional = _diffuse(concentration, diffusivity, diffusivity_layout="directional")
    scalar_field = _diffuse(concentration, diffusivity, diffusivity_layout="scalar_field")

    # (Ky, Kx) = (0.1, 0.4): the x-neighbour of cell (0, 0) gains 0.4 * 1.
    assert float(directional["diffusion_rate"][0, 1]) == pytest.approx(0.4)
    # x-profile K(x) = (0.1, 0.4): the face between x = 0 and x = 1 averages them.
    assert float(scalar_field["diffusion_rate"][0, 1]) != pytest.approx(0.4)


@pytest.mark.parametrize(
    "diffusivity",
    [
        torch.full((2, 2), 0.1, dtype=torch.float64),  # fits only the scalar-field reading
        torch.tensor([0.1], dtype=torch.float64),  # one element but ndim 1
        torch.tensor([0.1, 0.4, 0.2], dtype=torch.float64),  # fits no directional reading
    ],
)
def test_tensor_diffusivity_requires_a_declared_layout_even_with_one_reading(diffusivity):
    """A tensor coefficient (ndim >= 1) always declares its layout; the shape is never read."""
    concentration, _ = _ambiguous_case()
    with pytest.raises(ValueError, match="diffusivity_layout"):
        _diffuse(concentration, diffusivity)


@pytest.mark.parametrize("diffusivity", [0.1, torch.tensor(0.1, dtype=torch.float64)])
def test_scalar_diffusivity_needs_no_layout(diffusivity):
    concentration, _ = _ambiguous_case()
    undeclared = _diffuse(concentration, diffusivity)
    declared = _diffuse(concentration, diffusivity, diffusivity_layout="scalar_field")
    torch.testing.assert_close(undeclared["concentration"], declared["concentration"])


def test_advection_diffusion_tensor_diffusivity_requires_a_declared_layout():
    from relic.computation.operators.functional import advance_advection_diffusion

    concentration = torch.ones((1, 4, 3), dtype=torch.float64)
    velocity = (torch.full_like(concentration, 0.2), torch.full_like(concentration, 0.1))
    options = dict(
        grid_spacing=(1.0, 0.5), time_step=0.01, spatial_dims=2, splitting_method="sequential",
        advection_scheme="upwind", boundary_method="periodic", check_stability=True,
        enforce_non_negative=False, time_integration_method="euler",
    )
    with pytest.raises(ValueError, match="diffusivity_layout"):
        advance_advection_diffusion(concentration, velocity, torch.full_like(concentration, 0.4), **options)
    result = advance_advection_diffusion(
        concentration, velocity, torch.full_like(concentration, 0.4),
        diffusivity_layout="scalar_field", **options,
    )
    scalar = advance_advection_diffusion(concentration, velocity, 0.4, **options)
    torch.testing.assert_close(result["concentration"], scalar["concentration"])


def test_declared_layout_that_does_not_fit_the_shape_is_rejected():
    concentration = torch.zeros((1, 3, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="scalar_field"):
        _diffuse(concentration, torch.tensor([0.1, 0.4], dtype=torch.float64), diffusivity_layout="scalar_field")


def test_peclet_diagnostic_uses_the_declared_directional_diffusivity():
    # References: Patankar (1980); Hundsdorfer and Verwer (2003), cell Peclet number.
    # Mathematical expression: Pe = max_k max_i |u_k,i| dx_k / D_k,i.
    # Parameter meanings: (Ky, Kx) = (0.05, 0.4) m2 s-1, u_x = 0.2 and
    #   u_y = 0.1 m s-1, dx = 0.5 m and dy = 1.0 m.
    # Units: Pe dimensionless.
    # Domain/constraints: Pe_x = 0.2 * 0.5 / 0.4 = 0.25, Pe_y = 0.1 * 1.0 / 0.05
    #   = 2.0; a (2,) vector does not broadcast to the (1, 4, 3) grid, so the
    #   diagnostic must use the per-axis reading, not a broadcast.
    # Sign convention: magnitudes only.
    from relic.computation.operators.functional import advance_advection_diffusion

    concentration = torch.ones((1, 4, 3), dtype=torch.float64)
    concentration[0, 1, 1] = 2.0
    velocity = (torch.full_like(concentration, 0.2), torch.full_like(concentration, 0.1))
    options = dict(
        grid_spacing=(1.0, 0.5), time_step=0.01, spatial_dims=2, splitting_method="sequential",
        advection_scheme="upwind", boundary_method="periodic", check_stability=True,
        enforce_non_negative=False, time_integration_method="euler",
    )
    directional = advance_advection_diffusion(
        concentration, velocity, torch.tensor([0.05, 0.4], dtype=torch.float64),
        diffusivity_layout="directional", **options,
    )
    assert float(directional["peclet_number"]) == pytest.approx(2.0)
    field = torch.full_like(concentration, 0.4)
    field[0, 2, 0] = 0.05
    scalar_field = advance_advection_diffusion(
        concentration, velocity, field, diffusivity_layout="scalar_field", **options,
    )
    # max(0.2 * 0.5 / 0.05, 0.1 * 1.0 / 0.05) = 2.0 at the low-diffusivity cell.
    assert float(scalar_field["peclet_number"]) == pytest.approx(2.0)
