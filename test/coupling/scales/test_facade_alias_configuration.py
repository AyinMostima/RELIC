import logging

import torch

from relic.coupling.scales.scale_manager import ScaleManager
from relic.core.state import StateManager
from relic.parameters import get_global_manager


def test_facade_aliases_loaded_from_parameters():
    """Ensure alias mappings come from ParameterManager YAML definitions, not hardcoded defaults."""
    manager = ScaleManager(
        device=torch.device("cpu"),
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    manager.initialize()

    expected_transitions = {
        ("daily", "monthly", "up"),
        ("monthly", "daily", "down"),
        ("fine_grid", "coarse_grid", "up"),
        ("coarse_grid", "fine_grid", "down"),
        ("source_grid", "target_grid", "up"),
        ("target_grid", "source_grid", "down"),
    }
    expected_temporal_up = {("daily", "monthly")}
    expected_temporal_down = {("monthly", "daily")}

    assert expected_transitions.issubset(manager._alias_transitions)  # pylint: disable=protected-access
    assert expected_temporal_up.issubset(manager._temporal_alias_up)  # pylint: disable=protected-access
    assert expected_temporal_down.issubset(manager._temporal_alias_down)  # pylint: disable=protected-access
    # Spatial and conservative alias pairs are only valid transitions; they have no second path.
    for name in ("_spatial_alias_up", "_spatial_alias_down", "_conservative_alias_up", "_conservative_alias_down"):
        assert not hasattr(manager, name), name


class _AliasStubParameterManager:
    def get(self, key, namespace=None, expected_type=None, required=False, **kwargs):
        if namespace != "coupling.scales.aliases":
            if required:
                raise KeyError(f"{namespace}.{key}" if namespace else key)
            return None
        if key == "priority_order":
            return ["temporal", "spatial"]
        if key == "temporal_transitions":
            return [["daily", "monthly", "up"]]
        if key == "spatial_transitions":
            # Conflict: same transition labeled as spatial instead of temporal
            return [["daily", "monthly", "up"]]
        if key == "conservative_transitions":
            return []
        if required:
            raise KeyError(key)
        return None


def test_alias_conflict_emits_warning(caplog):
    """Alias conflicts should be logged for observability."""
    manager = ScaleManager(device=torch.device("cpu"), parameter_manager=_AliasStubParameterManager())
    manager.parameter_manager = manager._init_parameter_manager  # type: ignore[attr-defined]

    with caplog.at_level(logging.WARNING):
        manager._load_alias_configuration()

    assert any("Alias conflict" in record.message for record in caplog.records)
    assert any("Alias conflict" in warning for warning in manager._warnings)


# ---------------------------------------------------------------------------------------------
# Alias scale pairs run through the same basis-aware hierarchical kernels and audit as every
# other scale pair (HierarchicalAggregator; ConservationEnforcer ledger of the returned result).
# Topology: two children of 1 and 3 m2 under one parent of 4 m2.
# ---------------------------------------------------------------------------------------------
import pytest  # noqa: E402

from relic.coupling.core.policy import ScalingMode  # noqa: E402
from relic.coupling.scales.types import ScalingStrategy  # noqa: E402

_TOPOLOGY = {
    "parent_map": {0: 0, 1: 0},
    "source_areas": [1.0, 3.0],
    "target_areas": [4.0],
    "n_source_units": 2,
    "n_target_units": 1,
}
_ALIAS_UP = [("fine_grid", "coarse_grid"), ("source_grid", "target_grid")]
_ALIAS_DOWN = [("coarse_grid", "fine_grid"), ("target_grid", "source_grid")]


def _alias_manager() -> ScaleManager:
    manager = ScaleManager(
        device=torch.device("cpu"),
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    manager.initialize()
    return manager


@pytest.mark.parametrize(("source", "target"), _ALIAS_UP)
@pytest.mark.parametrize(
    ("basis", "strategy", "values", "expected", "ledger"),
    [
        # rho = (4 * 1 + 8 * 3) / 4 = 7 kg m-2 (not the area-free sum 12); ledger 28 kg.
        ("area_density", ScalingStrategy.CONSERVATIVE, [4.0, 8.0], [7.0], 28.0),
        ("area_density", None, [4.0, 8.0], [7.0], 28.0),
        # cell totals are summed: 4 + 8 = 12 kg.
        ("cell_total", None, [4.0, 8.0], [12.0], 12.0),
        # intensive: area-weighted mean (280 * 1 + 290 * 3) / 4 = 287.5 K; no amount ledger.
        ("intensive", None, [280.0, 290.0], [287.5], None),
        # categorical: area-weighted mode, code 7 covers 3 of 4 m2.
        ("categorical", None, [3.0, 7.0], [7.0], None),
    ],
)
def test_alias_pairs_aggregate_on_the_declared_basis_and_topology(
    source, target, basis, strategy, values, expected, ledger
):
    manager = _alias_manager()
    request = dict(
        variables=["q"],
        strategy=strategy,
        metadata={"q": {"spatial_basis": basis}},
    )
    result, audit = manager.aggregate(
        torch.tensor(values, dtype=torch.float64), source, target, topology=dict(_TOPOLOGY), **request
    )
    # A tensor in gives a tensor out; a mapping in gives a mapping out, with the same values.
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
    mapped, _ = manager.aggregate(
        {"q": torch.tensor(values, dtype=torch.float64)}, source, target, topology=dict(_TOPOLOGY), **request
    )
    assert isinstance(mapped, dict)
    torch.testing.assert_close(mapped["q"], result)
    assert "alias" not in audit.metadata
    if ledger is not None:
        diagnostics = audit.metadata["conservation_audit"]["q"]
        assert diagnostics.source_total == pytest.approx(ledger)
        assert diagnostics.target_total == pytest.approx(ledger)
        assert diagnostics.relative_error < 1e-12
        assert audit.relative_error < 1e-12


@pytest.mark.parametrize(("source", "target"), _ALIAS_DOWN)
@pytest.mark.parametrize(
    ("basis", "values", "expected"),
    [
        # a parent amount of 12 kg is split by child area: 12 * 1/4 and 12 * 3/4.
        ("cell_total", [12.0], [3.0, 9.0]),
        # a parent density is the density of every child.
        ("area_density", [7.0], [7.0, 7.0]),
        ("intensive", [287.5], [287.5, 287.5]),
    ],
)
def test_alias_pairs_disaggregate_on_the_declared_basis_and_topology(source, target, basis, values, expected):
    manager = _alias_manager()
    request = dict(variables=["q"], metadata={"q": {"spatial_basis": basis}})
    result, audit = manager.disaggregate(
        torch.tensor(values, dtype=torch.float64), source, target, topology=dict(_TOPOLOGY), **request
    )
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
    mapped, _ = manager.disaggregate(
        {"q": torch.tensor(values, dtype=torch.float64)}, source, target, topology=dict(_TOPOLOGY), **request
    )
    assert isinstance(mapped, dict)
    torch.testing.assert_close(mapped["q"], result)
    assert not audit.warnings or all("pass-through" not in warning for warning in audit.warnings)
    with pytest.raises(RuntimeError, match="[Cc]ategorical"):
        manager.disaggregate(
            torch.tensor([7.0], dtype=torch.float64),
            source,
            target,
            topology=dict(_TOPOLOGY),
            variables=["q"],
            metadata={"q": {"spatial_basis": "categorical"}},
        )


@pytest.mark.parametrize(
    ("operation", "source", "target"),
    [("aggregate", *pair) for pair in _ALIAS_UP] + [("disaggregate", *pair) for pair in _ALIAS_DOWN],
)
def test_alias_pairs_without_topology_raise_instead_of_summing(operation, source, target):
    manager = _alias_manager()
    with pytest.raises(RuntimeError, match="topology"):
        getattr(manager, operation)(
            torch.tensor([4.0, 8.0], dtype=torch.float64),
            source,
            target,
            variables=["q"],
            metadata={"q": {"spatial_basis": "cell_total"}},
        )


def test_the_audit_is_the_ledger_of_the_returned_result(monkeypatch):
    """A wrong result can never report relative_error 0: the audit integrates the returned field."""
    from relic.coupling.scales.hierarchical.aggregator import HierarchicalAggregator

    original = HierarchicalAggregator._apply_aggregation_method

    def doubled(self, *args, **kwargs):
        return 2.0 * original(self, *args, **kwargs)

    monkeypatch.setattr(HierarchicalAggregator, "_apply_aggregation_method", doubled)
    request = dict(
        variables=["q"],
        strategy=ScalingStrategy.CONSERVATIVE,
        metadata={"q": {"spatial_basis": "area_density"}},
        topology=dict(_TOPOLOGY),
    )
    enforced = _alias_manager()
    enforced.enforce_conservation = True
    with pytest.raises(RuntimeError, match="violates conservation"):
        enforced.aggregate(torch.tensor([4.0, 8.0], dtype=torch.float64), "fine_grid", "coarse_grid", **request)
    relaxed = _alias_manager()
    relaxed.enforce_conservation = False
    result, audit = relaxed.aggregate(
        torch.tensor([4.0, 8.0], dtype=torch.float64), "fine_grid", "coarse_grid", mode=ScalingMode.BALANCED, **request
    )
    torch.testing.assert_close(result, torch.tensor([14.0], dtype=torch.float64))
    assert audit.relative_error == pytest.approx(1.0)
    assert audit.metadata["conservation_audit"]["q"].is_conserved is False


@pytest.mark.parametrize(
    ("operation", "source", "target", "values"),
    [("aggregate", "daily", "monthly", [4.0, 8.0]), ("disaggregate", "monthly", "daily", [12.0])],
)
def test_the_temporal_alias_guard_cannot_be_skipped(operation, source, target, values):
    """Without the temporal strategy a temporal alias pair must raise, not run the spatial path."""
    manager = _alias_manager()
    manager.temporal_strategy = None
    with pytest.raises(RuntimeError, match="temporal alias guard"):
        getattr(manager, operation)(
            torch.tensor(values, dtype=torch.float64),
            source,
            target,
            topology=dict(_TOPOLOGY),
            variables=["q"],
            metadata={"q": {"spatial_basis": "cell_total"}},
        )
