import pytest

from relic.coupling.scales.spatial import factory as factory_module
from relic.coupling.scales.spatial.grid_utils import GridDescriptor
from relic.parameters import get_global_manager


def test_regridders_have_one_default_tolerance_source():
    """Without an explicit tolerance a regridder reads coupling.scales.defaults.conservation_tolerance,
    whether it is built directly or by a factory; the factory resolves no tolerance of its own."""
    from relic.coupling.scales.spatial.regrid_conservative import ConservativeRegridder

    pm = get_global_manager()
    default = float(pm.get("conservation_tolerance", namespace="coupling.scales.defaults", required=True))
    assert not hasattr(factory_module, "get_conservation_tolerance")
    factory = factory_module.SpatialRegridderFactory(parameter_manager=pm)
    assert factory.conservation_tolerance is None
    source, target = _factory_grids()
    request = dict(method="conservative", variable_metadata={"spatial_basis": "cell_total"})
    built = factory.create_regridder(source, target, **request)
    direct = ConservativeRegridder(source, target, quantity_basis="cell_total", parameter_manager=pm)
    assert built.conservation_tolerance == direct.conservation_tolerance == default
    # An explicit per-call tolerance is used (and is part of the cache key).
    explicit = factory.create_regridder(source, target, conservation_tolerance=1.0e-6, **request)
    assert explicit.conservation_tolerance == 1.0e-6 and explicit is not built
    with pytest.raises(ValueError, match="conservative method only"):
        factory.create_regridder(source, target, method="bilinear", conservation_tolerance=1.0e-6)
    for invalid in (float("nan"), 0.0, 1.0):
        with pytest.raises(ValueError, match="conservation_tolerance"):
            factory_module.SpatialRegridderFactory(conservation_tolerance=invalid, parameter_manager=pm)


def test_classmethod_resolves_parameter_manager_and_builds_regridder(monkeypatch):
    pm = get_global_manager()

    class DummyRegridder:
        def __init__(self, source_grid, target_grid, device=None, parameter_manager=None, **kwargs):
            self.source_grid = source_grid
            self.target_grid = target_grid
            self.parameter_manager = parameter_manager

        def upscale(self, data):
            return data

    monkeypatch.setitem(
        factory_module.SpatialRegridderFactory.REGRIDDER_CLASSES,
        "bilinear",
        DummyRegridder,
    )

    src = GridDescriptor.from_regular_grid(
        shape=(2, 2),
        spacing=(1.0, 1.0),
        origin=(0.0, 0.0),
        coord_system="cartesian",
    )
    tgt = GridDescriptor.from_regular_grid(
        shape=(1, 1),
        spacing=(2.0, 2.0),
        origin=(0.0, 0.0),
        coord_system="cartesian",
    )

    regridder = factory_module.SpatialRegridderFactory.create_regridder(
        "bilinear",
        src,
        tgt,
        parameter_manager=pm,
    )

    assert isinstance(regridder, DummyRegridder)
    assert regridder.parameter_manager is pm


def _factory_grids():
    source = GridDescriptor.from_regular_grid(
        shape=(2, 2),
        spacing=(1.0, 1.0),
        origin=(0.0, 0.0),
        coord_system="cartesian",
    )
    target = GridDescriptor.from_regular_grid(
        shape=(1, 1),
        spacing=(2.0, 2.0),
        origin=(0.0, 0.0),
        coord_system="cartesian",
    )
    return source, target


def test_factory_requires_explicit_regridding_method_contract():
    factory = factory_module.SpatialRegridderFactory(
        conservation_tolerance=1.0e-6,
        parameter_manager=get_global_manager(),
    )
    source, target = _factory_grids()

    with pytest.raises(ValueError, match="explicit regridding_method"):
        factory.create_regridder(source, target)
    with pytest.raises(ValueError, match="explicit regridding_method"):
        factory.create_regridder(
            source,
            target,
            variable_metadata={"variable_type": "intensive"},
        )


def test_factory_accepts_explicit_metadata_method(monkeypatch):
    class DummyRegridder:
        def __init__(self, source_grid, target_grid, **kwargs):
            self.source_grid = source_grid
            self.target_grid = target_grid

    monkeypatch.setitem(
        factory_module.SpatialRegridderFactory.REGRIDDER_CLASSES,
        "bilinear",
        DummyRegridder,
    )
    factory = factory_module.SpatialRegridderFactory(
        conservation_tolerance=1.0e-6,
        parameter_manager=get_global_manager(),
    )
    source, target = _factory_grids()

    regridder = factory.create_regridder(
        source,
        target,
        variable_metadata={"regridding_method": "bilinear"},
    )

    assert regridder.selected_method == "bilinear"


def test_factory_rejects_conservation_policy_method_conflict():
    factory = factory_module.SpatialRegridderFactory(
        enforce_conservation=True,
        conservation_tolerance=1.0e-6,
        parameter_manager=get_global_manager(),
    )
    source, target = _factory_grids()

    with pytest.raises(ValueError, match="conflicts with enforce_conservation"):
        factory.create_regridder(source, target, method="bilinear")


def test_factory_has_no_second_cf_synchronized_regrid_path():
    """CF regrid metadata is synchronized only by ScaleManager.regrid_variable, which reads the
    declared target-grid CF fields; the factory builds regridders and carries no synchronizer."""
    import inspect

    factory_cls = factory_module.SpatialRegridderFactory
    assert not hasattr(factory_cls, "regrid_with_metadata")
    assert "cf_synchronizer" not in inspect.signature(factory_cls.__init__).parameters
    factory = factory_cls(parameter_manager=get_global_manager())
    assert not hasattr(factory, "cf_synchronizer")


def test_conservation_enforcement_is_basis_aware():
    """enforce_conservation binds amounts and intensive fields to the conservative method; a
    categorical field is exempt and is regridded by nearest neighbour only."""
    from relic.coupling.scales.spatial.regrid_nearest import NearestNeighborRegridder

    enforced = factory_module.SpatialRegridderFactory(
        enforce_conservation=True,
        conservation_tolerance=1.0e-6,
        parameter_manager=get_global_manager(),
    )
    source, target = _factory_grids()
    regridder = enforced.create_regridder(
        source, target, method="nearest", variable_metadata={"spatial_basis": "categorical"}
    )
    assert isinstance(regridder, NearestNeighborRegridder)
    for basis in ("intensive", "cell_total", "area_density"):
        with pytest.raises(ValueError, match="enforce_conservation"):
            enforced.create_regridder(source, target, method="nearest", variable_metadata={"spatial_basis": basis})
    with pytest.raises(ValueError, match="spatial basis"):
        enforced.create_regridder(source, target, method="nearest")
    unenforced = factory_module.SpatialRegridderFactory(
        enforce_conservation=False,
        conservation_tolerance=1.0e-6,
        parameter_manager=get_global_manager(),
    )
    for method in ("bilinear", "conservative"):
        with pytest.raises(ValueError, match="categorical"):
            unenforced.create_regridder(
                source, target, method=method, variable_metadata={"spatial_basis": "categorical"}
            )
