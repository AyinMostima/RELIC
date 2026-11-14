import pytest
import torch

from relic.core.state import StateManager
from relic.coupling.core.policy import (
    ScalingMode,
    get_precision_policy,
    reset_precision_policy_cache,
)
from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.spatial.grid_utils import GridDescriptor
from relic.parameters import get_global_manager


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


class DummyPrecisionPolicy:
    def __init__(self, tolerance: float, enforce_conservation: bool = True):
        self.tolerance = tolerance
        self.enforce_conservation = enforce_conservation
        self.dtype = torch.float64
        self.max_iterations = 10
        self.relative_tolerance = tolerance
        self.absolute_tolerance = tolerance * 0.1


def test_scale_manager_uses_parameter_manager(monkeypatch):
    """ScaleManager should pass through the provided ParameterManager to precision policy lookups."""

    captured = {}

    def fake_get_precision_policy(mode, parameter_manager=None):
        captured["pm"] = parameter_manager
        return DummyPrecisionPolicy(tolerance=1e-9, enforce_conservation=True)

    monkeypatch.setattr(
        "relic.coupling.core.policy.get_precision_policy",
        fake_get_precision_policy,
    )

    pm = get_global_manager()

    mgr = ScaleManager(parameter_manager=pm, state_manager=StateManager())
    mgr.initialize()

    assert captured.get("pm") is pm
    assert mgr.conservation_tolerance == pytest.approx(1e-9)


def test_regrid_variable_uses_precision_mode(monkeypatch):
    """regrid_variable should operate without missing precision policy attributes."""
    class DummyManager:
        def __init__(self):
            self.calls = []

        def clear_cache(self):
            self.calls.clear()

        def create_regridder(self, *args, **kwargs):
            class _Stub:
                def upscale(self, value):
                    return value

            return _Stub()

        def regrid_variable(
            self,
            var_name,
            var_value,
            source_grid,
            target_grid,
            strategy="conservative",
            metadata=None,
            mode=None,
        ):
            self.calls.append(
                {
                    "var_name": var_name,
                    "strategy": strategy,
                    "mode": mode,
                }
            )
            return var_value, {}

    grid = GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(1, 1),
        x_coords=torch.tensor([0.0]),
        y_coords=torch.tensor([0.0]),
        x_bounds=torch.tensor([[0.0, 1.0]]),
        y_bounds=torch.tensor([[0.0, 1.0]]),
    )
    scale_manager = ScaleManager(
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    scale_manager.initialize()
    scale_manager.regridder_factory = DummyManager()
    # Isolate precision dispatch; CF regridding metadata is covered separately.
    scale_manager.cf_synchronizer = None
    input_metadata = {
        "standard_name": "air_temperature",
        "units": "K",
        "variable_type": "intensive",
        "source_grid": {"shape": (1, 1)},
        "target_grid": {"shape": (1, 1)},
    }

    result, metadata = scale_manager.regrid_variable(
        var_name="air_temperature",
        var_value=torch.ones((1, 1)),
        source_scale="source_grid",
        target_scale="target_grid",
        strategy="conservative",
        source_grid=grid,
        target_grid=grid,
        metadata=input_metadata,
    )

    if isinstance(result, dict):
        if "regridded" in result:
            result = result["regridded"]
        elif "regridded_data" in result:
            result = result["regridded_data"]
        else:
            for val in result.values():
                if isinstance(val, torch.Tensor):
                    result = val
                    break
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.ones((1, 1)))
    assert scale_manager.regridder_factory.calls == []
    assert metadata["operation"] == "identity"
    assert metadata["standard_name"] == "air_temperature"
    assert metadata["units"] == "K"
    assert metadata["variable_type"] == "intensive"
    assert "source_grid" not in metadata
    assert "target_grid" not in metadata
    assert input_metadata["source_grid"]["shape"] == (1, 1)
    assert input_metadata["target_grid"]["shape"] == (1, 1)


def _strip_grid(x_edges, cell_areas) -> GridDescriptor:
    """One-row cartesian grid with explicit x bounds on y in [0, 1] (areas in m2)."""
    n_cells = len(x_edges) - 1
    x_bounds = torch.tensor([[x_edges[i], x_edges[i + 1]] for i in range(n_cells)], dtype=torch.float64)
    return GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(1, n_cells),
        x_coords=x_bounds.mean(dim=1),
        y_coords=torch.tensor([0.5], dtype=torch.float64),
        x_bounds=x_bounds,
        y_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        cell_areas=torch.tensor(cell_areas, dtype=torch.float64).reshape(1, n_cells),
    )


def _regrid_metadata(basis: str, target_cells: int) -> dict:
    """Declared basis and the CF target-grid declaration required by the regrid CF synchronizer."""
    return {
        "spatial_basis": basis,
        "target_grid": {"shape": (1, target_cells), "grid_mapping": "crs", "lat_name": "y", "lon_name": "x"},
    }


def _enforcing_manager() -> ScaleManager:
    scale_manager = ScaleManager(
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    scale_manager.initialize()
    scale_manager.enforce_conservation = True
    return scale_manager


@pytest.mark.parametrize("failure_mode", ["residual", "no_report"])
def test_conservative_regrid_fails_closed_without_conservation_proof(monkeypatch, failure_mode):
    """Requested conservative remapping rejects a covered-support residual above tolerance and a
    regridder without its covered-support report."""
    import relic.coupling.scales.spatial.regrid_conservative as conservative_module

    scale_manager = _enforcing_manager()
    if failure_mode == "residual":
        # A wrong (doubled) map: the float64 ledger of the returned field shows the residual.
        original_map = conservative_module.ConservativeRegridder._map_field
        monkeypatch.setattr(
            conservative_module.ConservativeRegridder,
            "_map_field",
            lambda self, *args, **kwargs: 2.0 * original_map(self, *args, **kwargs),
        )
        expected = "covered-support residual"
    else:
        class _Unreported:
            def upscale(self, data):
                return data.sum(dim=-1, keepdim=True)

        monkeypatch.setattr(
            scale_manager.regridder_factory, "create_regridder", lambda *args, **kwargs: _Unreported()
        )
        expected = "covered-support report"

    with pytest.raises(RuntimeError, match=expected):
        scale_manager.regrid_variable(
            var_name="carbon_mass",
            var_value=torch.tensor([[4.0, 8.0]], dtype=torch.float64),
            source_scale="source_grid",
            target_scale="target_grid",
            strategy="conservative",
            source_grid=_strip_grid([0.0, 1.0, 2.0], [1.0, 1.0]),
            target_grid=_strip_grid([0.0, 2.0], [2.0]),
            metadata=_regrid_metadata("cell_total", 1),
        )


@pytest.mark.parametrize(
    ("basis", "field", "expected", "covered"),
    [
        # cell totals 4 and 8 kg: the target [0, 1.5] holds 4 + 0.5 * 8 = 8 kg.
        ("cell_total", [[4.0, 8.0]], [[8.0]], 8.0),
        # densities 4 and 8 kg m-2 on 1 m2 cells: 4 + 0.5 * 8 = 8 kg over 1.5 m2.
        ("area_density", [[4.0, 8.0]], [[8.0 / 1.5]], 8.0),
    ],
)
def test_regrid_audit_is_the_covered_support_ledger_of_the_regridder(basis, field, expected, covered):
    """The one audit is the regridder's covered-support ledger on the declared basis: partial
    coverage keeps the uncovered source amount out of the residual."""
    scale_manager = _enforcing_manager()
    result, _metadata = scale_manager.regrid_variable(
        var_name="carbon_mass",
        var_value=torch.tensor(field, dtype=torch.float64),
        source_scale="source_grid",
        target_scale="target_grid",
        strategy="conservative",
        source_grid=_strip_grid([0.0, 1.0, 2.0], [1.0, 1.0]),
        target_grid=_strip_grid([0.0, 1.5], [1.5]),
        metadata=_regrid_metadata(basis, 1),
    )
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
    record = scale_manager.audit_collector.records[-1]
    diagnostic = record.conservation["carbon_mass"]
    assert diagnostic.is_conserved is True
    assert diagnostic.source_total == pytest.approx(covered)
    assert diagnostic.target_total == pytest.approx(covered)
    assert diagnostic.relative_error < 1e-12
    assert record.metadata_updates["conservation_passed"] is True


def test_regrid_ledger_tolerance_is_the_precision_tolerance_of_the_calls_mode(monkeypatch):
    """The covered-support audit of one call compares with coupling.scales.defaults.precision.<mode>.tolerance
    of that call's mode (the default mode when the call passes none)."""
    import relic.coupling.scales.spatial.regrid_conservative as conservative_module

    scale_manager = _enforcing_manager()
    # A map off by a relative 1e-7: the float64 ledger of the returned field shows the residual.
    original_map = conservative_module.ConservativeRegridder._map_field
    monkeypatch.setattr(
        conservative_module.ConservativeRegridder,
        "_map_field",
        lambda self, *args, **kwargs: (1.0 + 1.0e-7) * original_map(self, *args, **kwargs),
    )
    request = dict(
        var_name="carbon_mass",
        var_value=torch.tensor([[4.0, 8.0]], dtype=torch.float64),
        source_scale="source_grid",
        target_scale="target_grid",
        strategy="conservative",
        source_grid=_strip_grid([0.0, 1.0, 2.0], [1.0, 1.0]),
        target_grid=_strip_grid([0.0, 2.0], [2.0]),
    )
    # conservative (the default mode, 1e-9) rejects a relative residual of 1e-7 ...
    for mode in (None, ScalingMode.CONSERVATIVE):
        with pytest.raises(RuntimeError, match="covered-support residual"):
            scale_manager.regrid_variable(mode=mode, metadata=_regrid_metadata("cell_total", 1), **request)
    # ... balanced (1e-6) accepts it.
    scale_manager.regrid_variable(
        mode=ScalingMode.BALANCED, metadata=_regrid_metadata("cell_total", 1), **request
    )
    record = scale_manager.audit_collector.records[-1]
    balanced = get_precision_policy(ScalingMode.BALANCED).tolerance
    assert record.metadata_updates["conservation_passed"] is True
    assert record.metadata_updates["conservation_tolerance"] == balanced
    assert scale_manager.ledger_tolerance(ScalingMode.BALANCED) == balanced


def test_same_shape_targets_of_different_geometry_do_not_share_weights():
    """The facade reuses a regridder only for the same geometry: totals 4 and 8 kg on [0, 1] and
    [1, 2] give 12 kg on the target [0, 2] and then 4 kg on the target [0, 1] of the same shape."""
    scale_manager = _enforcing_manager()
    request = dict(
        var_name="carbon_mass",
        var_value=torch.tensor([[4.0, 8.0]], dtype=torch.float64),
        source_scale="source_grid",
        target_scale="target_grid",
        strategy="conservative",
        source_grid=_strip_grid([0.0, 1.0, 2.0], [1.0, 1.0]),
    )
    wide, _ = scale_manager.regrid_variable(
        target_grid=_strip_grid([0.0, 2.0], [2.0]), metadata=_regrid_metadata("cell_total", 1), **request
    )
    narrow, _ = scale_manager.regrid_variable(
        target_grid=_strip_grid([0.0, 1.0], [1.0]), metadata=_regrid_metadata("cell_total", 1), **request
    )
    assert wide.reshape(-1).tolist() == [12.0]
    assert narrow.reshape(-1).tolist() == [4.0]


def _float32_strip_request(n_cells: int) -> dict:
    """Float32 cell totals near 1 kg on n unit cells regridded to one cell of n m2."""
    edges = [float(index) for index in range(n_cells + 1)]
    generator = torch.Generator().manual_seed(7)
    values = (torch.rand((1, n_cells), generator=generator, dtype=torch.float64) * 1.0e-3 + 1.0).to(torch.float32)
    return dict(
        var_name="carbon_mass",
        var_value=values,
        source_scale="source_grid",
        target_scale="target_grid",
        strategy="conservative",
        source_grid=_strip_grid(edges, [1.0] * n_cells),
        target_grid=_strip_grid([0.0, float(n_cells)], [float(n_cells)]),
    )


def test_float32_rounding_is_accepted_by_the_shared_ledger_rule(monkeypatch):
    """The default enforcement accepts the float32 rounding of the returned field (it exceeds the
    float64-scale tolerance) and still rejects a wrong result."""
    import relic.coupling.scales.spatial.regrid_conservative as conservative_module

    scale_manager = _enforcing_manager()
    request = _float32_strip_request(64)
    scale_manager.regrid_variable(metadata=_regrid_metadata("cell_total", 1), **request)
    record = scale_manager.audit_collector.records[-1]
    assert record.metadata_updates["conservation_passed"] is True
    assert record.conservation["carbon_mass"].relative_error > scale_manager.ledger_tolerance()
    original_map = conservative_module.ConservativeRegridder._map_field
    monkeypatch.setattr(
        conservative_module.ConservativeRegridder,
        "_map_field",
        lambda self, *args, **kwargs: 2.0 * original_map(self, *args, **kwargs),
    )
    with pytest.raises(RuntimeError, match="covered-support residual"):
        scale_manager.regrid_variable(metadata=_regrid_metadata("cell_total", 1), **request)


def test_facade_regridders_follow_the_mode_of_the_call(monkeypatch):
    """The conservative regridder of one call is built with the ledger tolerance of that call's
    mode; the facade factory keeps no tolerance of its own."""
    from relic.coupling.scales.spatial.factory import SpatialRegridderFactory

    built = []
    original_create = SpatialRegridderFactory.create_regridder

    def spy(self, *args, **kwargs):
        regridder = original_create(self, *args, **kwargs)
        built.append(regridder)
        return regridder

    monkeypatch.setattr(SpatialRegridderFactory, "create_regridder", spy)
    scale_manager = _enforcing_manager()
    assert scale_manager.regridder_factory.conservation_tolerance is None
    request = dict(
        var_name="carbon_mass",
        var_value=torch.tensor([[4.0, 8.0]], dtype=torch.float64),
        source_scale="source_grid",
        target_scale="target_grid",
        strategy="conservative",
        source_grid=_strip_grid([0.0, 1.0, 2.0], [1.0, 1.0]),
        target_grid=_strip_grid([0.0, 2.0], [2.0]),
    )
    scale_manager.regrid_variable(metadata=_regrid_metadata("cell_total", 1), **request)
    scale_manager.regrid_variable(mode=ScalingMode.BALANCED, metadata=_regrid_metadata("cell_total", 1), **request)
    assert [regridder.conservation_tolerance for regridder in built] == [
        scale_manager.ledger_tolerance(),
        scale_manager.ledger_tolerance(ScalingMode.BALANCED),
    ]
    assert built[0] is not built[1]


def test_scale_manager_has_one_regrid_audit():
    import relic.coupling.scales.scale_strategies.manager_spatial_ops as spatial_ops

    assert not hasattr(spatial_ops, "check_mass_conservation")


def test_default_enforcement_regrids_a_categorical_variable_by_nearest_neighbour():
    """The default auto-regrid enforces conservation; a categorical field is exempt (nearest)."""
    scale_manager = ScaleManager(
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    scale_manager.initialize()
    assert scale_manager.enforce_conservation is True
    result, _metadata = scale_manager.regrid_variable(
        var_name="land_cover_class",
        var_value=torch.tensor([[3.0, 7.0]], dtype=torch.float64),
        source_scale="source_grid",
        target_scale="target_grid",
        strategy="nearest",
        source_grid=_strip_grid([0.0, 1.0, 2.0], [1.0, 1.0]),
        target_grid=_strip_grid([0.0, 0.5, 1.0, 1.5, 2.0], [0.5, 0.5, 0.5, 0.5]),
        metadata=_regrid_metadata("categorical", 4),
    )
    assert result.reshape(-1).tolist() == [3.0, 3.0, 7.0, 7.0]


def test_conservative_identity_audit_preserves_diagnostic_values():
    """The audit record must carry the conservation checker values verbatim."""
    grid = GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(1, 1),
        x_coords=torch.tensor([0.0]),
        y_coords=torch.tensor([0.0]),
        x_bounds=torch.tensor([[0.0, 1.0]]),
        y_bounds=torch.tensor([[0.0, 1.0]]),
        cell_areas=torch.tensor([[1.0]]),
    )
    scale_manager = ScaleManager(
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    scale_manager.initialize()
    scale_manager.enforce_conservation = True

    scale_manager.regrid_variable(
        var_name="carbon_mass",
        var_value=torch.tensor([[3.0]]),
        source_scale="source_grid",
        target_scale="target_grid",
        strategy="conservative",
        source_grid=grid,
        target_grid=grid,
        metadata={"spatial_basis": "cell_total"},
    )

    record = scale_manager.audit_collector.records[-1]
    diagnostic = record.conservation["carbon_mass"]
    assert diagnostic.is_conserved is True
    assert diagnostic.absolute_error == pytest.approx(0.0)
    assert diagnostic.relative_error == pytest.approx(0.0)
    assert diagnostic.source_total == pytest.approx(3.0)
    assert diagnostic.target_total == pytest.approx(3.0)
    assert record.custom_data["conservation_enforced"] is True
    assert record.metadata_updates["conservation_requested"] is True
    assert record.metadata_updates["conservation_evaluated"] is True
    assert record.metadata_updates["conservation_passed"] is True


def test_regrid_uses_cf_synchronizer_alias(monkeypatch):
    """ScaleManager should call sync_after_regridding when available."""

    class DummyFactory:
        def regrid_variable(
            self,
            var_name,
            var_value,
            source_grid,
            target_grid,
            strategy="conservative",
            metadata=None,
            mode=None,
        ):
            return torch.ones(
                target_grid.dims,
                dtype=var_value.dtype,
                device=var_value.device,
            ), {}

        def create_regridder(self, *args, **kwargs):
            class _Stub:
                def upscale(self, value):
                    return value

            return _Stub()

        def clear_cache(self):
            return None

    class DummyCFSynchronizer:
        def __init__(self):
            self.calls = []

        def sync_after_regridding(self, data, metadata, regrid_method, source_grid, target_grid):
            self.calls.append(
                {
                    "method": regrid_method,
                    "source_grid": source_grid,
                    "target_grid": target_grid,
                }
            )
            tagged = metadata.copy() if metadata else {}
            tagged["called"] = True
            return tagged

    grid = GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(1, 1),
        x_coords=torch.tensor([0.0]),
        y_coords=torch.tensor([0.0]),
        x_bounds=torch.tensor([[0.0, 1.0]]),
        y_bounds=torch.tensor([[0.0, 1.0]]),
    )
    target_grid = GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(2, 2),
        x_coords=torch.tensor([0.0, 1.0]),
        y_coords=torch.tensor([0.0, 1.0]),
        x_bounds=torch.tensor([[0.0, 0.5], [0.5, 1.0]]),
        y_bounds=torch.tensor([[0.0, 0.5], [0.5, 1.0]]),
    )

    scale_manager = ScaleManager(
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    scale_manager.initialize()
    scale_manager.regridder_factory = DummyFactory()
    # The dummy factory has no covered-support ledger; conservation is not under test here.
    scale_manager.enforce_conservation = False
    dummy_cf = DummyCFSynchronizer()
    scale_manager.cf_synchronizer = dummy_cf

    _, metadata = scale_manager.regrid_variable(
        var_name="air_temperature",
        var_value=torch.ones((1, 1)),
        source_scale="source_grid",
        target_scale="target_grid",
        strategy="conservative",
        source_grid=grid,
        target_grid=target_grid,
    )

    assert metadata.get("called") is True


def test_precision_policy_missing_parameters_raises(monkeypatch):
    """Precision policy should raise when ParameterManager lacks required entries."""
    reset_precision_policy_cache()
    pm = get_global_manager()

    def _raise_missing(*_args, **_kwargs):
        raise KeyError("missing parameter")

    monkeypatch.setattr(pm, "get", _raise_missing, raising=True)

    with pytest.raises(RuntimeError):
        get_precision_policy(ScalingMode.CONSERVATIVE, parameter_manager=pm)


def _two_to_one_grids():
    source = GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(1, 2),
        x_coords=torch.tensor([0.5, 1.5]),
        y_coords=torch.tensor([0.5]),
        x_bounds=torch.tensor([[0.0, 1.0], [1.0, 2.0]]),
        y_bounds=torch.tensor([[0.0, 1.0]]),
    )
    target = GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(1, 1),
        x_coords=torch.tensor([1.0]),
        y_coords=torch.tensor([0.5]),
        x_bounds=torch.tensor([[0.0, 2.0]]),
        y_bounds=torch.tensor([[0.0, 1.0]]),
    )
    return source, target


def test_governed_regrid_passes_declared_cf_grid_fields():
    """The real CF synchronizer receives the declared target-grid CF fields (never invented)."""
    from relic.data.standards.cf_conventions import CF_REGRID_TARGET_GRID_FIELDS

    source, target = _two_to_one_grids()
    scale_manager = ScaleManager(parameter_manager=get_global_manager(), state_manager=StateManager())
    scale_manager.initialize()
    declared = {"shape": (1, 1), "grid_mapping": "crs_local", "lat_name": "y", "lon_name": "x"}
    assert set(CF_REGRID_TARGET_GRID_FIELDS) <= set(declared)
    base = {"standard_name": "air_temperature", "units": "K", "variable_type": "intensive"}
    _, metadata = scale_manager.regrid_variable(
        var_name="air_temperature",
        var_value=torch.tensor([[280.0, 282.0]]),
        source_scale="source_grid",
        target_scale="target_grid",
        strategy="conservative",
        source_grid=source,
        target_grid=target,
        metadata=dict(base, source_grid={"shape": (1, 2)}, target_grid=dict(declared)),
    )
    assert metadata["grid_mapping"] == "crs_local"
    assert metadata["coordinates"] == "y x"
    with pytest.raises(ValueError, match="grid_mapping, lat_name, lon_name"):
        scale_manager.regrid_variable(
            var_name="air_temperature",
            var_value=torch.tensor([[280.0, 282.0]]),
            source_scale="source_grid",
            target_scale="target_grid",
            strategy="conservative",
            source_grid=source,
            target_grid=target,
            metadata=dict(base, source_grid={"shape": (1, 2)}, target_grid={"shape": (1, 1)}),
        )


def test_precision_policy_cache_key_resolves_the_database_path_without_a_fallback(tmp_path):
    """The key uses the resolved database path; a path that cannot be resolved raises."""
    import pytest

    from relic.coupling.core import policy as policy_module

    class _Manager:
        database_path = str(tmp_path / "db")

    key = policy_module._make_cache_key(ScalingMode.CONSERVATIVE, _Manager())
    assert key == (ScalingMode.CONSERVATIVE, str((tmp_path / "db").resolve()))

    class _Unresolvable:
        class database_path:
            @staticmethod
            def resolve():
                raise OSError("path loop")

    with pytest.raises((OSError, TypeError)):
        policy_module._make_cache_key(ScalingMode.CONSERVATIVE, _Unresolvable())
