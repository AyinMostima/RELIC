"""
Unit tests for simulator scale detection and error handling.
"""

from types import SimpleNamespace

import pytest
import torch

from relic.core.functor import BiologicalProcessFunctor, FunctorMetadata, SpatialScale
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.core.simulator.execution_group import _prepare_group_state
from relic.core.simulator.scale_ops import _build_metadata_scale_map, _needs_regrid
from relic.scientific.transport.advection import LinearAdvection


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'basin_flux': 'dimensionless',
    'grid_output': 'dimensionless',
    'patch_soil_moisture': 'dimensionless',
    'point_carbon': 'kg C m-2',
    'point_output': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class _ScaleMetadataModule(torch.nn.Module):
    """Minimal module exposing one authoritative output scale for tests."""

    def __init__(self, output_name: str, spatial_scale: SpatialScale) -> None:
        super().__init__()
        self.metadata = SimpleNamespace(
            spatial_scale=spatial_scale,
            output_variables=[output_name],
        )


class TestSimulatorScaleInference:
    """Validate scale detection behavior when auto-scale is disabled or enabled."""

    def test_mixed_scales_raise_when_auto_scale_disabled(self):
        """Mixed scales with strict detection should raise a clear error when auto-scale is off."""
        simulator = _declared_simulator(
            {"enable_auto_scale": False, "strict_scale_detection": True, "time_step": 60.0}
        )
        simulator.initialize_state(
            {
                "patch_soil_moisture": torch.ones(3),
                "basin_flux": torch.ones(2),
            }
        )

        with pytest.raises(ValueError):
            simulator._execute_group(
                ["noop_functor"], dt=simulator.time_step, start_time=simulator.current_time
            )

    def test_scale_check_returns_target_when_auto_scale_enabled(self):
        """Scale check should infer a target scale when auto-scale is enabled."""
        simulator = _declared_simulator(
            {"enable_auto_scale": True, "strict_scale_detection": True, "time_step": 60.0}
        )
        state = {
            "patch_soil_moisture": torch.ones(3),
            "basin_flux": torch.ones(2),
        }

        scale_check = simulator._check_scale_consistency(state)
        assert scale_check["needs_regrid"] is True
        assert scale_check["target_scale"] is not None

    def test_detection_can_be_suppressed_for_consistency(self):
        """historical behavior: no error when strict_scale_detection is disabled."""
        simulator = _declared_simulator(
            {"enable_auto_scale": False, "strict_scale_detection": False, "time_step": 60.0}
        )
        simulator.initialize_state(
            {
                "patch_soil_moisture": torch.ones(3),
                "basin_flux": torch.ones(2),
            }
        )

        result = simulator._execute_group(
            ["noop_functor"], dt=simulator.time_step, start_time=simulator.current_time
        )
        assert result == {}

    def test_declared_output_scale_excludes_controls_and_external_forcing(self):
        """Strict detection should use authoritative process output scales only."""
        simulator = _declared_simulator(
            {
                "conservation_checks": False,
                "enable_auto_scale": False,
                "strict_scale_detection": True,
                "time_step": 0.1,
            }
        )
        tracer = torch.ones(1, 16, dtype=torch.float32)
        simulator.initialize_state(
            {
                "tracer_concentration": tracer,
                "velocity_field": torch.full_like(tracer, 0.1),
                "source_term": torch.zeros_like(tracer),
            }
        )
        advection = LinearAdvection(
            name="linear_advection",
            spatial_dims=1,
            grid_spacing=[1.0],
        )
        simulator.register_functor(advection)

        current_state = simulator._get_current_state()
        scale_check = simulator._check_scale_consistency(current_state, force=True)

        assert scale_check["detected_scales"] == {
            "tracer_concentration": SpatialScale.GRID_CELL.value
        }
        assert scale_check["needs_regrid"] is False
        updates = simulator._execute_group(
            [advection.name], dt=0.1, start_time=simulator.current_time
        )
        assert "tracer_concentration" in updates

    def test_conflicting_declared_output_scales_fail_fast(self):
        """One runtime port cannot have two authoritative output scales."""
        point_functor = SimpleNamespace(
            metadata=SimpleNamespace(
                spatial_scale=SpatialScale.POINT,
                output_variables=["shared_output"],
            )
        )
        grid_functor = SimpleNamespace(
            metadata=SimpleNamespace(
                spatial_scale=SpatialScale.GRID_CELL,
                output_variables=["shared_output"],
            )
        )
        simulator = SimpleNamespace(
            functors={"point_writer": point_functor, "grid_writer": grid_functor}
        )

        with pytest.raises(ValueError, match="Conflicting spatial scales.*shared_output"):
            _build_metadata_scale_map(simulator)

    def test_unknown_scale_cannot_hide_two_known_scales(self):
        """Unknown evidence must not suppress a real known-scale conflict."""
        assert _needs_regrid(
            {
                SpatialScale.UNKNOWN.value,
                SpatialScale.POINT.value,
                SpatialScale.GRID_CELL.value,
            }
        ) is True

    def test_group_scale_check_ignores_other_execution_group_outputs(self):
        """Each execution group should validate only its active retained outputs."""
        simulator = _declared_simulator(
            {
                "enable_auto_scale": False,
                "strict_scale_detection": True,
            }
        )
        simulator.initialize_state(
            {
                "point_output": torch.ones(2),
                "grid_output": torch.ones(2),
            }
        )
        simulator.functors["point_writer"] = _ScaleMetadataModule(
            "point_output", SpatialScale.POINT
        )
        simulator.functors["grid_writer"] = _ScaleMetadataModule(
            "grid_output", SpatialScale.GRID_CELL
        )

        state = _prepare_group_state(simulator, ["point_writer"])

        assert "point_output" in state
        assert simulator._scale_check_history[-1]["detected_scales"] == {
            "point_output": SpatialScale.POINT.value
        }

    def test_group_scale_check_rejects_mixed_outputs_within_one_group(self):
        """Two active output scales still require an explicit coupling operator."""
        simulator = _declared_simulator(
            {
                "enable_auto_scale": False,
                "strict_scale_detection": True,
            }
        )
        simulator.initialize_state(
            {
                "point_output": torch.ones(2),
                "grid_output": torch.ones(2),
            }
        )
        simulator.functors["point_writer"] = _ScaleMetadataModule(
            "point_output", SpatialScale.POINT
        )
        simulator.functors["grid_writer"] = _ScaleMetadataModule(
            "grid_output", SpatialScale.GRID_CELL
        )

        with pytest.raises(ValueError, match="multiple scales detected"):
            _prepare_group_state(simulator, ["point_writer", "grid_writer"])


# --------------------------------------------------------------------------- governed automatic regridding


_CF_GRID_FIELDS = {"grid_mapping": "crs", "lat_name": "lat", "lon_name": "lon"}


def _grid_descriptor(n: int, spacing: float, *, cf: bool = True) -> dict:
    descriptor = {
        "grid_type": "regular",
        "coord_system": "cartesian",
        "shape": (n, n),
        "spacing": (spacing, spacing),
        "origin": (0.0, 0.0),
    }
    if cf:
        descriptor.update(_CF_GRID_FIELDS)
    return descriptor


def _regridding_simulator(target_grid: dict) -> RELICSimulator:
    return _declared_simulator(
        {
            "device": "cpu",
            "time_step": 60.0,
            "enable_auto_scale": True,
            "strict_scale_detection": False,
            "conservation_checks": False,
            "auto_scale": {
                "source_grid": _grid_descriptor(4, 1.0),
                "target_grid": target_grid,
            },
        }
    )


class _PointCarbonReader(BiologicalProcessFunctor):
    """Reads the working-copy soil carbon (regridded to the point grid)."""

    def __init__(self) -> None:
        super().__init__(
            name="point_carbon_reader",
            config={},
            metadata=FunctorMetadata(
                name="point_carbon_reader",
                description="Copies the point-grid soil carbon density",
                references=["Jones (1999) Mon. Weather Rev. 127:2204-2210"],
                input_variables=["soil_carbon"],
                output_variables=["point_carbon"],
                spatial_scale=SpatialScale.POINT,
            ),
        )
        self.seen_shapes = []

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        carbon = state["soil_carbon"]
        self.seen_shapes.append(tuple(carbon.shape))
        return {
            "point_carbon": UpdateItem(
                value=carbon.clone(), mode=UpdateMode.REPLACE, unit="kg C m-2",
                source_functor=self.name,
            )
        }


class _GridCarbonLoader(BiologicalProcessFunctor):
    """Replaces soil carbon with its grid-cell field (a data-loader Functor)."""

    def __init__(self, field: torch.Tensor) -> None:
        super().__init__(
            name="grid_carbon_loader",
            config={},
            metadata=FunctorMetadata(
                name="grid_carbon_loader",
                description="Loads the grid-cell soil carbon density field",
                references=["RELIC regridding integration fixture"],
                output_variables=["soil_carbon"],
                spatial_scale=SpatialScale.GRID_CELL,
            ),
        )
        self.field = field

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        return {
            "soil_carbon": UpdateItem(
                value=self.field.clone(), mode=UpdateMode.REPLACE, unit="kg C m-2",
                source_functor=self.name,
            )
        }


def test_real_step_regrids_through_the_cf_synchronizer_and_audits_conservation():
    """A step with two output labels regrids the working copy through the governed path."""
    simulator = _regridding_simulator(_grid_descriptor(2, 2.0))
    field = torch.arange(1.0, 17.0, dtype=torch.float64).reshape(4, 4)
    simulator.initialize_state(
        {"soil_carbon": field.clone(), "point_carbon": torch.zeros(2, 2, dtype=torch.float64)}
    )
    reader = _PointCarbonReader()
    simulator.register_functor(reader)
    simulator.register_functor(_GridCarbonLoader(field))
    assert simulator.scale_manager.cf_synchronizer is not None

    simulator.step()

    # Conservative area-density remap on the explicit grids: every 2 m x 2 m target
    # cell is the area-weighted mean of its four 1 m x 1 m source cells.
    expected = field.reshape(2, 2, 2, 2).mean(dim=(1, 3))
    assert reader.seen_shapes == [(2, 2)]
    torch.testing.assert_close(simulator.get_state()["point_carbon"], expected)
    torch.testing.assert_close(simulator.get_state()["soil_carbon"], field)
    record = simulator.scale_manager.audit_collector.records[-1]
    assert record.operation == "regrid"
    assert record.variables == ["soil_carbon"]
    assert record.source_scale == SpatialScale.GRID_CELL.value
    assert record.target_scale == SpatialScale.POINT.value
    diagnostics = record.conservation["soil_carbon"]
    assert diagnostics.is_conserved is True
    assert diagnostics.relative_error <= 1e-12
    assert float(diagnostics.source_total) == pytest.approx(float(field.sum()))
    assert float(diagnostics.target_total) == pytest.approx(float(expected.sum() * 4.0))
    assert record.metadata_updates["grid_mapping"] == "crs"
    assert record.metadata_updates["coordinates"] == "lat lon"


@pytest.mark.parametrize("missing", ["grid_mapping", "lat_name", "lon_name"])
def test_target_descriptor_without_cf_fields_is_rejected_at_validation(missing):
    target = _grid_descriptor(2, 2.0)
    del target[missing]
    simulator = _regridding_simulator(target)

    with pytest.raises(ValueError, match=rf"auto_scale\.target_grid.*{missing}"):
        simulator.initialize_state({"soil_carbon": torch.ones(4, 4, dtype=torch.float64)})


def test_cf_fields_are_never_inferred_from_shape_or_labels():
    simulator = _regridding_simulator(_grid_descriptor(2, 2.0, cf=False))

    with pytest.raises(ValueError, match="grid_mapping, lat_name, lon_name"):
        simulator.initialize_state({"soil_carbon": torch.ones(4, 4, dtype=torch.float64)})


def test_identical_descriptors_need_no_cf_fields_or_variable_metadata(monkeypatch):
    """An identity mapping transforms nothing, so it needs neither CF fields nor a basis.

    Only literally identical geometry qualifies; any other pair must declare CF
    fields, and a real regrid still needs each variable's catalog metadata.
    """
    identical = _grid_descriptor(4, 1.0, cf=False)
    simulator = _declared_simulator(
        {
            "device": "cpu",
            "time_step": 60.0,
            "enable_auto_scale": True,
            "strict_scale_detection": False,
            "conservation_checks": False,
            "auto_scale": {"source_grid": dict(identical), "target_grid": dict(identical)},
        }
    )
    field = torch.arange(1.0, 17.0, dtype=torch.float64).reshape(4, 4)
    simulator.initialize_state(
        {"soil_carbon": field.clone(), "point_carbon": torch.zeros(4, 4, dtype=torch.float64)}
    )
    reader = _PointCarbonReader()
    simulator.register_functor(reader)
    simulator.register_functor(_GridCarbonLoader(field))

    def _no_metadata_lookup(*args, **kwargs):
        raise AssertionError("an identity mapping looked up variable metadata")

    monkeypatch.setattr(simulator.scale_manager, "get_variable_metadata", _no_metadata_lookup)

    simulator.step()

    assert reader.seen_shapes == [(4, 4)]
    torch.testing.assert_close(simulator.get_state()["point_carbon"], field)
    assert not any(
        "soil_carbon" in record.variables
        for record in simulator.scale_manager.audit_collector.records
    )


class _GridFieldLoader(BiologicalProcessFunctor):
    """Stands in for the contract writer of one grid-cell field (a data loader)."""

    def __init__(self, output: str, field: torch.Tensor, writer: str = "atmospheric_radiation") -> None:
        super().__init__(
            name=writer,
            config={},
            metadata=FunctorMetadata(
                name=writer,
                description="Loads one grid-cell field",
                references=["RELIC regridding integration fixture"],
                output_variables=[output],
                spatial_scale=SpatialScale.GRID_CELL,
            ),
        )
        self.output = output
        self.field = field

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        return {
            self.output: UpdateItem(
                value=self.field.clone(), mode=UpdateMode.REPLACE, unit="W m-2",
                source_functor=self.name,
            )
        }


class _PointFieldReader(BiologicalProcessFunctor):
    """Copies one working-copy field into a point-scale output."""

    def __init__(self, input_name: str) -> None:
        super().__init__(
            name="point_field_reader",
            config={},
            metadata=FunctorMetadata(
                name="point_field_reader",
                description="Copies one field to a point-scale output",
                references=["RELIC regridding integration fixture"],
                input_variables=[input_name],
                output_variables=["point_output"],
                spatial_scale=SpatialScale.POINT,
            ),
        )
        self.input_name = input_name

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        return {
            "point_output": UpdateItem(
                value=state[self.input_name].clone(), mode=UpdateMode.REPLACE,
                unit="dimensionless", source_functor=self.name,
            )
        }


def test_identity_mapping_of_a_variable_without_catalog_metadata_passes_through():
    """RUN-004 PROFILE-001/002: identical descriptors and an uncatalogued radiation field."""
    identical = _grid_descriptor(4, 1.0, cf=False)
    simulator = _declared_simulator(
        {
            "device": "cpu",
            "time_step": 60.0,
            "enable_auto_scale": True,
            "strict_scale_detection": False,
            "conservation_checks": False,
            "auto_scale": {"source_grid": dict(identical), "target_grid": dict(identical)},
        }
    )
    field = torch.full((4, 4), 500.0, dtype=torch.float64)
    simulator.initialize_state(
        {"shortwave_down": field.clone(), "point_output": torch.zeros(4, 4, dtype=torch.float64)}
    )
    simulator.register_functor(_PointFieldReader("shortwave_down"))
    simulator.register_functor(_GridFieldLoader("shortwave_down", field))

    simulator.step()

    torch.testing.assert_close(simulator.get_state()["point_output"], field)


def test_real_regrid_of_a_variable_without_catalog_metadata_fails_closed():
    simulator = _regridding_simulator(_grid_descriptor(2, 2.0))
    field = torch.full((4, 4), 500.0, dtype=torch.float64)
    simulator.initialize_state(
        {"shortwave_down": field.clone(), "point_output": torch.zeros(2, 2, dtype=torch.float64)}
    )
    simulator.register_functor(_PointFieldReader("shortwave_down"))
    simulator.register_functor(_GridFieldLoader("shortwave_down", field))

    with pytest.raises(Exception, match="Variable metadata for 'shortwave_down' not found"):
        simulator.step()


def test_geometrically_equal_but_differently_written_descriptors_still_need_cf_fields():
    source = _grid_descriptor(4, 1.0, cf=False)
    target = dict(source, origin=(0.0, 0.0), spacing=[1.0, 1.0])
    simulator = _declared_simulator(
        {
            "device": "cpu",
            "time_step": 60.0,
            "enable_auto_scale": True,
            "auto_scale": {"source_grid": source, "target_grid": target},
        }
    )

    with pytest.raises(ValueError, match="grid_mapping, lat_name, lon_name"):
        simulator.initialize_state({"soil_carbon": torch.ones(4, 4, dtype=torch.float64)})


def test_regridding_uses_the_published_owners_of_cf_fields_and_variable_metadata():
    """No local copies: CF fields from cf_conventions, metadata from the public ScaleManager API."""
    import inspect

    import relic.core.simulator.regridding_ops as regridding_ops
    from relic.data.standards.cf_conventions import CF_REGRID_TARGET_GRID_FIELDS

    assert not hasattr(regridding_ops, "_CF_GRID_FIELDS")
    assert regridding_ops.CF_REGRID_TARGET_GRID_FIELDS is CF_REGRID_TARGET_GRID_FIELDS
    source = inspect.getsource(regridding_ops)
    assert "_get_variable_metadata" not in source
    assert "get_variable_metadata(" in source


@pytest.mark.parametrize("declared", ["grid", "gridcell ", "pach", "", "   "])
def test_spatial_scale_from_value_rejects_a_string_that_names_no_scale(declared):
    """A misspelled or blank declaration is an error, not a silent UNKNOWN."""
    with pytest.raises(ValueError, match="spatial scale") as error:
        SpatialScale.from_value(declared)

    message = str(error.value)
    assert repr(declared) in message
    for member in SpatialScale:
        assert repr(member.value) in message


@pytest.mark.parametrize("declared", [3, 2.0, object()])
def test_spatial_scale_from_value_rejects_a_value_that_is_not_a_string(declared):
    with pytest.raises(TypeError, match="SpatialScale"):
        SpatialScale.from_value(declared)


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (None, SpatialScale.UNKNOWN),
        (SpatialScale.UNKNOWN, SpatialScale.UNKNOWN),
        ("unknown", SpatialScale.UNKNOWN),
        (" Unknown ", SpatialScale.UNKNOWN),
        ("grid_cell", SpatialScale.GRID_CELL),
        ("GRID_CELL", SpatialScale.GRID_CELL),
        (" grid_2d ", SpatialScale.GRID_2D),
        (SpatialScale.POINT, SpatialScale.POINT),
    ],
)
def test_spatial_scale_from_value_accepts_members_values_and_the_undeclared_forms(
    declared, expected
):
    assert SpatialScale.from_value(declared) is expected


@pytest.mark.parametrize("declared", ["grid2d", "grid3d", "GRID2D"])
def test_spatial_scale_from_value_rejects_the_removed_aliases(declared):
    with pytest.raises(ValueError, match="spatial scale") as error:
        SpatialScale.from_value(declared)

    message = str(error.value)
    assert "alias" not in message
    assert [member.value for member in SpatialScale].__repr__() in message


def test_functor_metadata_rejects_a_misspelled_spatial_scale():
    with pytest.raises(ValueError, match="'grid'"):
        FunctorMetadata(
            name="misspelled_spatial_scale",
            description="Spatial scale declaration test",
            references=["RELIC SpatialScale contract"],
            spatial_scale="grid",
        )


def test_metadata_scale_detection_rejects_a_misspelled_label_instead_of_falling_back():
    """Tier 1 parses a declaration with SpatialScale.from_value; it never hands an
    unknown label on to the Tier 2 heuristics or the regrid decision."""
    simulator = SimpleNamespace(
        functors={"typo": _ScaleMetadataModule("point_output", "pointt")}
    )

    with pytest.raises(ValueError, match="pointt"):
        _build_metadata_scale_map(simulator)


def test_metadata_scale_detection_normalizes_the_case_of_a_member_value():
    simulator = SimpleNamespace(
        functors={"upper": _ScaleMetadataModule("grid_output", " GRID_2D ")}
    )

    assert _build_metadata_scale_map(simulator) == {"grid_output": "grid_2d"}

def test_scale_check_failure_is_not_read_as_no_regrid(monkeypatch):
    """A crashing detection never reports needs_regrid=False, whatever the strictness."""
    import relic.core.simulator.scale_ops as scale_ops

    simulator = _declared_simulator(
        {"enable_auto_scale": True, "strict_scale_detection": False, "time_step": 60.0}
    )

    def _broken(sim):
        raise RuntimeError("metadata scale map inconsistent")

    monkeypatch.setattr(scale_ops, "_build_metadata_scale_map", _broken)
    with pytest.raises(RuntimeError, match="inconsistent"):
        simulator._check_scale_consistency({"patch_soil_moisture": torch.ones(3)})
