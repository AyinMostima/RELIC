import torch

from relic.coupling.core.policy import ScalingMode
from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.spatial.grid_utils import GridDescriptor
from relic.coupling.scales.types import ScalingStrategy
from relic.core.state import StateManager


def test_regrid_audit_tracks_method_and_strategy():
    """Regrid audit must reflect chosen method/strategy and conservation flags."""
    manager = ScaleManager(
        device=torch.device("cpu"),
        state_manager=StateManager(),
    )
    manager.initialize()

    class DummyRegridder:
        """Sums the cell totals into the target cell and reports its covered-support ledger."""

        def __init__(self, target_shape):
            self.target_shape = target_shape
            self.last_regrid_report = None

        def upscale(self, data):
            result = data.sum().expand(self.target_shape).clone()
            amount = float(data.sum())
            residual = abs(float(result.sum()) - amount)
            # The report schema of ConservativeRegridder.last_regrid_report: one ledger row.
            self.last_regrid_report = {
                "quantity_basis": "cell_total",
                "covered_source_amount": amount,
                "mapped_target_amount": float(result.sum()),
                "uncovered_source_amount": 0.0,
                "absolute_residual": residual,
                "relative_residual": residual / amount,
                "activity": abs(amount),
                "rows": [{"covered_source_amount": amount, "absolute_residual": residual, "activity": abs(amount)}],
                "rounding_dtype": str(data.dtype).replace("torch.", ""),
                "term_count": 2 * data.numel() + 8,
            }
            return result

    manager.regridder_factory.create_regridder = (
        lambda src_grid, tgt_grid, method=None, variable_metadata=None, conservation_tolerance=None: DummyRegridder(
            tgt_grid.dims
        )
    )

    src = GridDescriptor(
        shape=(2, 2),
        spacing=(1.0, 1.0),
        origin=(0.0, 0.0),
        grid_type="regular",
        coord_system="cartesian",
        device=manager.device,
    )
    tgt = GridDescriptor(
        shape=(1, 1),
        spacing=(2.0, 2.0),
        origin=(0.0, 0.0),
        grid_type="regular",
        coord_system="cartesian",
        device=manager.device,
    )
    value = torch.ones((2, 2), dtype=torch.float32, device=manager.device)

    scenarios = [
        ("conservative", ScalingStrategy.CONSERVATIVE, False, True),
        ("bilinear", ScalingStrategy.BILINEAR, False, False),
        ("nearest", ScalingStrategy.NEAREST, False, False),
    ]

    for method, expected_strategy, expected_cf, expected_enforced in scenarios:
        manager.audit_collector.records.clear()
        manager.regrid_variable(
            "var",
            value,
            src,
            tgt,
            strategy=method,
            metadata={
                "spatial_basis": "cell_total",
                "source_grid": {
                    "shape": (2, 2),
                    "grid_mapping": "cartesian",
                    "lat_name": "y",
                    "lon_name": "x",
                },
                "target_grid": {
                    "shape": (1, 1),
                    "grid_mapping": "cartesian",
                    "lat_name": "y",
                    "lon_name": "x",
                },
            },
            mode=ScalingMode.CONSERVATIVE,
        )
        record = manager.audit_collector.records[-1]
        assert record.strategy == str(expected_strategy)
        assert record.cf_compliant == expected_cf
        assert record.custom_data.get("conservation_enforced") == expected_enforced
        assert record.metadata_updates.get("method") == method
        assert record.metadata_updates["cf_validation_scope"] == "not_evaluated"
        assert record.metadata_updates["dataset_conformance_checked"] is False


def test_regrid_variable_maps_the_source_grid_to_the_target_grid_for_equal_cell_counts():
    """Nearest neighbour between two equal-size, shifted grids: no direction label is guessed.

    regrid_variable maps the field from the source grid to the target grid (the regridder's
    upscale for every class), so equal cell counts need no operation and the nearest-neighbour
    result is the nearest source value of each target, with the target coverage reported.
    """
    manager = ScaleManager(device=torch.device("cpu"), state_manager=StateManager())
    manager.initialize()
    manager.enforce_conservation = False
    manager.cf_synchronizer = None

    def grid(x):
        return GridDescriptor(
            grid_type="regular",
            coord_system="cartesian",
            dims=(1, len(x)),
            x_coords=torch.tensor(x),
            y_coords=torch.tensor([0.0]),
            x_bounds=torch.tensor([[value - 0.5, value + 0.5] for value in x]),
            y_bounds=torch.tensor([[-0.5, 0.5]]),
        )

    source, target = grid([0.0, 1.0, 2.0]), grid([0.9, 1.9, 2.9])
    result, metadata = manager.regrid_variable(
        "land_cover",
        torch.tensor([[1.0, 2.0, 3.0]]),
        source,
        target,
        strategy="nearest",
        metadata={"spatial_basis": "categorical"},
        mode=ScalingMode.CONSERVATIVE,
    )
    assert result.tolist() == [[2.0, 3.0, 3.0]]
    assert metadata["coverage"] == {"target_cells": 3, "covered_target_cells": 3, "uncovered_target_cells": 0}


def _metadata_2x2_to_1x1():
    grid = {"grid_mapping": "cartesian", "lat_name": "y", "lon_name": "x"}
    return {
        "spatial_basis": "cell_total",
        "source_grid": {"shape": (2, 2), **grid},
        "target_grid": {"shape": (1, 1), **grid},
    }


def test_regrid_variable_needs_a_known_method_and_assumes_none():
    """The spatial method is the caller's explicit choice among the ScaleManager methods: no
    parameter default exists, and a missing or unknown method raises instead of becoming the
    conservative method (no audit record is written for a refused call)."""
    import inspect

    import pytest

    manager = ScaleManager(device=torch.device("cpu"), state_manager=StateManager())
    manager.initialize()
    parameter = inspect.signature(ScaleManager.regrid_variable).parameters["strategy"]
    assert parameter.default is inspect.Parameter.empty

    src = GridDescriptor(shape=(2, 2), spacing=(1.0, 1.0), origin=(0.0, 0.0), grid_type="regular",
                         coord_system="cartesian", device=manager.device)
    tgt = GridDescriptor(shape=(1, 1), spacing=(2.0, 2.0), origin=(0.0, 0.0), grid_type="regular",
                         coord_system="cartesian", device=manager.device)
    value = torch.ones((2, 2), dtype=torch.float32)
    manager.audit_collector.records.clear()
    for method in (None, "irregular_nearest", "patch", "Conservative"):
        with pytest.raises(ValueError, match="method"):
            manager.regrid_variable(
                "var", value, src, tgt, strategy=method, metadata=_metadata_2x2_to_1x1(),
                mode=ScalingMode.CONSERVATIVE,
            )
    assert manager.audit_collector.records == []


def test_default_scaling_mode_is_a_required_parameter():
    """The default scaling mode is the declared coupling.scales.defaults.auto_scale_mode; an
    undeclared mode raises instead of silently becoming the conservative mode."""
    import pytest

    from relic.parameters import get_global_manager

    class _WithoutAutoScaleMode:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def get(self, key, *args, **kwargs):
            if str(key).endswith("auto_scale_mode"):
                if kwargs.get("required", True):
                    raise KeyError(f"undeclared parameter {key}")
                return None
            return self._inner.get(key, *args, **kwargs)

    declared = ScaleManager(device=torch.device("cpu"), state_manager=StateManager())
    declared.initialize()
    assert isinstance(declared.default_mode, ScalingMode)

    manager = ScaleManager(
        device=torch.device("cpu"),
        state_manager=StateManager(),
        parameter_manager=_WithoutAutoScaleMode(get_global_manager()),
    )
    with pytest.raises(Exception, match="auto_scale_mode"):
        manager.initialize()
    assert manager.default_mode is None
