"""I7: Bounds clipping produces closed-loop clip_loss flux.

References:
- Beven, K. (2012). Rainfall-Runoff Modelling: The Primer
- Dingman, S. L. (2015). Physical Hydrology

Mathematical expression:
If S + delta < L: S_new = L, clip_loss = (S + delta) - L
If S + delta > U: S_new = U, clip_loss = (S + delta) - U

Parameters:
S state value (kg), delta increment (kg per step), L lower bound, U upper bound
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
)
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager, UpdateItem, StateVariableSpec
from relic.core.state import UpdateMode


class _FailingConstraintOwner:
    def apply_constraints(self, *_args, **_kwargs):
        raise RuntimeError("canonical constraint owner failed")


class _BoundsOvershootFunctor(TransportMechanismFunctor):
    """Test fixture that requests one bounded accumulated update."""

    TRANSPORT_TYPE = "diffusion"

    def __init__(self) -> None:
        metadata = FunctorMetadata(
            name="bounds_overshoot",
            version="0.0.0",
            description="Test-only bounded-state overshoot.",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="x_{n+1} = x_n + 20 kg",
            references=["Test fixture for accepted-step bound projection."],
            input_variables=["bounded_state"],
            output_variables=["bounded_state"],
        )
        super().__init__(name=metadata.name, config={}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        # Reference: accepted-step projection contract in StateManager.
        # Expression: x_requested = x_n + 20 kg.
        # Parameters: x_n is bounded_state; 20 kg is the requested increment.
        return {
            "bounded_state": UpdateItem(
                value=torch.tensor(20.0, device=self.device),
                mode=UpdateMode.ACCUMULATE,
                unit="kg",
                per_timestep=True,
                source_functor=self.name,
            )
        }


class TestBoundsClipping:
    """Test bounds enforcement with clip_loss tracking."""

    def test_non_negativity_generates_clip_loss(self) -> None:
        """Clipping to lower bound records negative clip_loss."""
        sm = self._build_bounded_manager(lower=0.0, upper=100.0, initial=10.0)
        inc = UpdateItem(value=-25.0, unit="kg", basis="per_step")
        sm.apply_increment("bounded_state", inc, dt=1.0)

        current = sm.get_state()["bounded_state"].item()
        assert current >= 0.0

        diagnostics = sm.get_diagnostics()
        clip_loss = diagnostics.get("clip_loss", {}).get("bounded_state", 0.0)
        assert pytest.approx(clip_loss, abs=1e-9) == -15.0

    def test_upper_bound_clipping(self) -> None:
        """Clipping to upper bound records positive clip_loss."""
        sm = self._build_bounded_manager(lower=0.0, upper=100.0, initial=90.0)
        inc = UpdateItem(value=20.0, unit="kg", basis="per_step")
        sm.apply_increment("bounded_state", inc, dt=1.0)

        current = sm.get_state()["bounded_state"].item()
        assert current <= 100.0

        diagnostics = sm.get_diagnostics()
        clip_loss = diagnostics.get("clip_loss", {}).get("bounded_state", 0.0)
        assert pytest.approx(clip_loss, abs=1e-9) == 10.0

    @pytest.mark.parametrize(
        "name,bounds,initial,increment,expected_value,expected_clip",
        [
            # A flux-named variable that declares non-negative bounds is clipped.
            ("tracer_flux_store", (0.0, float("inf")), 5.0, -8.0, 0.0, -3.0),
            ("respiration_index_store", (0.0, float("inf")), 1.0, -2.5, 0.0, -1.5),
            ("net_exchange_store", (0.0, float("inf")), 0.0, -1.0, 0.0, -1.0),
            # A potential that declares a non-positive range is clipped above zero.
            ("soil_matric_potential_store", (-float("inf"), 0.0), -1.0, 3.0, 0.0, 2.0),
        ],
    )
    def test_bounds_follow_declarations_not_variable_names(
        self, name, bounds, initial, increment, expected_value, expected_clip
    ) -> None:
        """Bounds enforcement is decided by the declared bounds, never by name tokens.

        Expression: x_req = x_n + delta; x_new = clamp(x_req, L, U);
        clip_loss = x_req - x_new (negative when the lower bound adds storage).
        """
        spec = StateVariableSpec(name=name, units="kg", bounds=bounds, conserved=False)
        sm = StateManager(schema={name: spec}, autoload_contract_registry=False)
        sm.set_state({name: torch.tensor(initial)})
        sm.apply_increment(name, UpdateItem(value=increment, unit="kg", basis="per_step"), dt=1.0)

        assert sm.get_state()[name].item() == pytest.approx(expected_value)
        clip_loss = sm.get_diagnostics()["clip_loss"][name]
        assert clip_loss == pytest.approx(expected_clip)

    @pytest.mark.parametrize(
        "numerics,initial,increment,expected_value,expected_clip",
        [
            ("  lower_bound: 0.0\n", 1.0, -3.0, 0.0, -2.0),
            ("  upper_bound: 0.0\n", -1.0, 3.0, 0.0, 2.0),
        ],
    )
    def test_one_sided_contract_bound_is_enforced_with_clip_loss(
        self, tmp_path, monkeypatch, numerics, initial, increment, expected_value, expected_clip
    ) -> None:
        """A contract declaring one bound is enforced on that side only:
        lower only -> [lb, +inf), upper only -> (-inf, ub]; the clipped amount
        is reported as clip_loss = x_req - x_new."""
        import relic.contracts.contract_loader as contract_loader

        contracts = tmp_path / "variables"
        contracts.mkdir()
        (contracts / "one_sided_store.yaml").write_text(
            "variable: one_sided_store\nkind: state\ndimension:\n  udunits: kg\n"
            f"numerics:\n{numerics}",
            encoding="utf-8",
        )
        monkeypatch.delenv("RELIC_AUTOLOAD_CONTRACT_REGISTRY", raising=False)
        monkeypatch.setattr(contract_loader, "PACKAGED_CONTRACTS_DIR", contracts)
        sm = StateManager()
        sm.set_state({"one_sided_store": torch.tensor(initial)})
        sm.apply_increment(
            "one_sided_store",
            UpdateItem(value=increment, unit="kg", basis="per_step"),
            dt=1.0,
        )

        assert sm.get_state()["one_sided_store"].item() == pytest.approx(expected_value)
        assert sm.get_diagnostics()["clip_loss"]["one_sided_store"] == pytest.approx(
            expected_clip
        )

    def test_upper_only_contract_bound_reaches_the_simulator_schema_and_clips(
        self, tmp_path, monkeypatch
    ) -> None:
        """End to end: a YAML contract declaring only upper_bound -> strict
        ContractRegistry -> to_state_variable_spec -> simulator initial-state
        schema -> StateManager clipping; the undeclared lower side stays open."""
        import relic.contracts.contract_loader as contract_loader

        contracts = tmp_path / "variables"
        contracts.mkdir()
        (contracts / "suction_store.yaml").write_text(
            "variable: suction_store\nkind: state\ndimension:\n  udunits: Pa\n"
            "numerics:\n  lower_bound: null\n  upper_bound: 0.0\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("RELIC_AUTOLOAD_CONTRACT_REGISTRY", raising=False)
        monkeypatch.setattr(contract_loader, "PACKAGED_CONTRACTS_DIR", contracts)

        registry = contract_loader.load_contract_registry()
        assert registry.to_state_variable_spec("suction_store").bounds == (
            float("-inf"),
            0.0,
        )

        simulator = RELICSimulator(
            {"time_step": 60.0, "device": "cpu", "enable_bootstrap_modules": False}
        )
        simulator.initialize_state({"suction_store": torch.tensor([-5.0e4, -1.0e3])})
        manager = simulator.state_manager
        assert manager._schema["suction_store"].bounds == (float("-inf"), 0.0)

        manager.apply_increment(
            "suction_store",
            UpdateItem(value=torch.tensor([-1.0e9, 3.0e3]), unit="Pa", basis="per_step"),
            dt=1.0,
        )
        state = manager.get_state()["suction_store"]
        assert state[0].item() == pytest.approx(-5.0e4 - 1.0e9)
        assert state[1].item() == pytest.approx(0.0)
        assert manager.get_diagnostics()["clip_loss"]["suction_store"] == pytest.approx(2.0e3)

        rejecting = RELICSimulator(
            {"time_step": 60.0, "device": "cpu", "enable_bootstrap_modules": False}
        )
        with pytest.raises(ValueError, match="suction_store"):
            rejecting.initialize_state({"suction_store": torch.tensor([1.0])})

    def test_signed_variable_without_bounds_is_not_clipped(self) -> None:
        """A signed quantity (for example a net exchange) declares no bounds."""
        spec = StateVariableSpec(name="net_ecosystem_exchange_store", units="kg", bounds=None)
        sm = StateManager(
            schema={"net_ecosystem_exchange_store": spec}, autoload_contract_registry=False
        )
        sm.set_state({"net_ecosystem_exchange_store": torch.tensor(1.0)})
        sm.apply_increment(
            "net_ecosystem_exchange_store",
            UpdateItem(value=-4.0, unit="kg", basis="per_step"),
            dt=1.0,
        )
        assert sm.get_state()["net_ecosystem_exchange_store"].item() == pytest.approx(-3.0)
        assert "net_ecosystem_exchange_store" not in sm.get_diagnostics()["clip_loss"]

    def test_no_clipping_no_loss(self) -> None:
        """No clipping results in zero clip_loss."""
        sm = self._build_bounded_manager(lower=0.0, upper=100.0, initial=50.0)
        inc = UpdateItem(value=10.0, unit="kg", basis="per_step")
        sm.apply_increment("bounded_state", inc, dt=1.0)

        diagnostics = sm.get_diagnostics()
        clip_loss = diagnostics.get("clip_loss", {}).get("bounded_state", 0.0)
        assert clip_loss == 0.0

    def test_constraint_owner_failure_does_not_enable_local_clamp(self) -> None:
        """A constraint-owner failure must abort instead of using a second formula."""
        spec = StateVariableSpec(
            name="bounded_state",
            units="kg",
            bounds=(0.0, 100.0),
            conserved=False,
        )
        sm = StateManager(
            schema={"bounded_state": spec},
            physical_constraint_manager=_FailingConstraintOwner(),
        )
        sm.set_state({"bounded_state": torch.tensor(90.0)})

        with pytest.raises(RuntimeError, match="canonical constraint owner failed"):
            sm.apply_increment(
                "bounded_state",
                UpdateItem(value=20.0, unit="kg", basis="per_step"),
                dt=1.0,
            )

        assert float(sm.get_state()["bounded_state"].item()) == 90.0

    def test_simulator_routes_bounds_residual_through_state_manager(self) -> None:
        """The simulator must not rewrite an overshoot before StateManager sees it."""
        spec = StateVariableSpec(
            name="bounded_state",
            units="kg",
            bounds=(0.0, 100.0),
            conserved=False,
        )
        sm = StateManager(
            schema={"bounded_state": spec},
            autoload_contract_registry=False,
        )
        sm.initialize({"bounded_state": torch.tensor(90.0)})
        simulator = RELICSimulator(
            {
                "max_time": 1.0,
                "time_step": 1.0,
                "conservation_checks": False,
                "enable_auto_scale": False,
                "device": "cpu",
            }
        )
        simulator.initialize(sm, DependencyGraphResolver())
        simulator.register_functor(_BoundsOvershootFunctor(), execution_group="fast")

        simulator.step(dt=1.0)

        assert float(sm.get_state()["bounded_state"].item()) == pytest.approx(100.0)
        clip_loss = sm.get_diagnostics()["clip_loss"]["bounded_state"]
        assert float(clip_loss) == pytest.approx(10.0)

    def test_clip_loss_in_conservation_balance(self) -> None:
        """clip_loss should not break conservation reporting."""
        carbon_unit = "g C m-2"
        sm = self._build_bounded_manager(
            lower=0.0,
            upper=100.0,
            initial=10.0,
            units=carbon_unit,
            accumulates="carbon.total",
            autoload_contract_registry=False,
        )
        sm.register_variable(
            StateVariableSpec(
                name="carbon_loss",
                units=carbon_unit,
                conserved=True,
                default_value=0.0,
                accumulates="carbon.total",
            )
        )
        sm.set_conservation_ledger(
            {"carbon": ["bounded_state", "carbon_loss"]}
        )

        # Reference: E3SM/ELM elemental budget diagnostics.
        # Expression: requested delta = -25, actual delta = -10, clip_loss = -15,
        # and residual_C = actual delta + routed loss = -10 + 10 = 0.
        # Parameters: all terms are integrated carbon amounts in g C m-2.
        # Route: actual removed carbon enters carbon_loss; rejected removal is diagnostic.
        bounded_update = UpdateItem(
            value=-25.0,
            mode=UpdateMode.ACCUMULATE,
            unit=carbon_unit,
            per_timestep=True,
            element="C",
            source_functor="bounds_clipping_test",
        )
        loss_update = UpdateItem(
            value=10.0,
            mode=UpdateMode.ACCUMULATE,
            unit=carbon_unit,
            per_timestep=True,
            element="C",
            source_functor="bounds_clipping_test",
        )
        sm.apply_updates(
            {
                "bounded_state": bounded_update,
                "carbon_loss": loss_update,
            },
            dt=1.0,
        )

        diagnostics = sm.get_diagnostics()
        clip_loss = diagnostics.get("clip_loss", {}).get("bounded_state", 0.0)
        assert pytest.approx(clip_loss, abs=1e-9) == -15.0
        assert pytest.approx(
            float(sm.get_state()["carbon_loss"].item()),
            abs=1e-9,
        ) == 10.0

        report = sm.generate_conservation_report(
            updates={},
            current_time=1.0,
            store_history=False,
        )
        assert report.total_violations == 0

    @staticmethod
    def _build_bounded_manager(
        lower: float,
        upper: float,
        initial: float,
        *,
        units: str = "kg",
        accumulates: str | None = None,
        autoload_contract_registry: bool = True,
    ) -> StateManager:
        spec = StateVariableSpec(
            name="bounded_state",
            units=units,
            bounds=(lower, upper),
            conserved=True,
            accumulates=accumulates,
        )
        sm = StateManager(
            schema={"bounded_state": spec},
            autoload_contract_registry=autoload_contract_registry,
        )
        sm.set_state({"bounded_state": torch.tensor(initial)})
        return sm


@pytest.mark.property
class TestBoundsClippingPBT:
    """I7 property tests for clipping and clip_loss diagnostics."""

    @given(
        initial=st.floats(min_value=10.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        increment=st.floats(min_value=-300.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_clipped_values_within_bounds(self, initial: float, increment: float) -> None:
        """Final value must be clamped into [0, 100] after update."""
        lower = 0.0
        upper = 100.0
        sm = StateManager()
        sm.register_variable(
            StateVariableSpec(
                name="bounded_x",
                units="kg",
                bounds=(lower, upper),
                conserved=False,
                default_value=initial,
            )
        )

        sm.apply_increment(
            "bounded_x",
            UpdateItem(value=increment, mode=UpdateMode.ACCUMULATE, unit="kg", basis="per_step"),
            dt=1.0,
        )
        final = float(sm.get_state()["bounded_x"].item())
        assert lower - 1e-10 <= final <= upper + 1e-10

    @given(
        initial=st.floats(min_value=10.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        increment=st.floats(min_value=-300.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_clip_loss_matches_clamp_difference(self, initial: float, increment: float) -> None:
        """clip_loss should equal unclamped-minus-clamped scalar difference."""
        lower = 0.0
        upper = 100.0
        sm = StateManager()
        sm.register_variable(
            StateVariableSpec(
                name="bounded_x",
                units="kg",
                bounds=(lower, upper),
                conserved=False,
                default_value=initial,
            )
        )

        unclamped = initial + increment
        clamped = min(max(unclamped, lower), upper)

        sm.apply_increment(
            "bounded_x",
            UpdateItem(value=increment, mode=UpdateMode.ACCUMULATE, unit="kg", basis="per_step"),
            dt=1.0,
        )

        diagnostics = sm.get_diagnostics()
        clip_loss = diagnostics.get("clip_loss", {}).get("bounded_x", 0.0)
        expected = unclamped - clamped
        tol = max(1e-8, 1e-8 * abs(expected))
        assert abs(float(clip_loss) - expected) <= tol


class TestClipResidualSignConvention:
    """Every clip residual follows the StateManager clip_loss sign: requested value minus clipped
    value, so amount before the clip = amount after the clip + residual; a lower bound that adds
    material gives a negative residual, an upper bound that removes material a positive one."""

    def test_lower_bound_clips_report_negative_residuals_in_every_owner(self) -> None:
        import numpy as np
        import xarray as xr

        from relic.coupling.external.common.regrid import RegridConfig, RegridMethod, RegridOperator
        from relic.coupling.scales.adapters import ConservativeRemap
        from relic.coupling.scales.spatial.grid_utils import GridDescriptor
        from relic.coupling.scales.spatial.regrid_conservative import ConservativeRegridder
        from relic.coupling.scales.types import SpatialQuantityBasis

        # StateManager: 10 kg - 25 kg requested, clipped to 0 kg: clip_loss = -15 kg.
        sm = TestBoundsClipping._build_bounded_manager(lower=0.0, upper=100.0, initial=10.0)
        sm.apply_increment("bounded_state", UpdateItem(value=-25.0, unit="kg", basis="per_step"), dt=1.0)
        assert sm.get_diagnostics()["clip_loss"]["bounded_state"] == pytest.approx(-15.0)

        # ConservativeRemap positivity clip of cell totals: Y_0 = -8 + 2 = -6 kg raised to 0.
        remap = ConservativeRemap(
            preserve_mass=True, preserve_energy=False, preserve_positivity=True, device=torch.device("cpu")
        )
        out = remap.forward(
            {"litter_mass": torch.tensor([-8.0, 4.0], dtype=torch.float64)},
            0.0,
            1.0,
            variables=["litter_mass"],
            overlap_areas=torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float64),
            source_areas=torch.tensor([1.0, 1.0], dtype=torch.float64),
            target_areas=torch.tensor([1.5, 0.5], dtype=torch.float64),
            variable_metadata={"litter_mass": {"spatial_basis": "cell_total"}},
        )
        assert float(out["diagnostics.conservative_remap.positivity_clip_residual"]) == pytest.approx(-6.0)

        # ConservativeRegridder range clamp: a half-covered 4 m2 target gets y = 3 kg m-2 below the
        # source minimum 4 kg m-2; the clamp adds (4 - 3) * 4 m2 = 4 kg: residual -4 kg.
        def strip(edges, areas):
            n = len(edges) - 1
            bounds = torch.tensor([[edges[i], edges[i + 1]] for i in range(n)], dtype=torch.float64)
            return GridDescriptor(
                grid_type="regular", coord_system="cartesian", dims=(1, n),
                x_coords=bounds.mean(dim=1), y_coords=torch.tensor([0.5], dtype=torch.float64),
                x_bounds=bounds, y_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
                cell_areas=torch.tensor(areas, dtype=torch.float64).reshape(1, n),
            )

        regridder = ConservativeRegridder(
            source_grid=strip([0.0, 1.0, 2.0], [1.0, 1.0]),
            target_grid=strip([0.0, 4.0], [4.0]),
            use_sparse_weights=False,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )
        regridder.downscale(torch.tensor([[4.0, 8.0]], dtype=torch.float64), range_clamp=True)
        assert regridder.last_regrid_report["range_clip_residual"] == pytest.approx(-4.0)

        # RegridOperator positive_definite: the remapped density -2.75 is raised to 0.
        lat = np.array([-0.5, 0.5])
        lon = np.array([0.5, 1.5, 2.5, 3.5])
        source = xr.DataArray(
            np.array([[1.0, 2.0, -30.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, attrs={"units": "kg m-2 s-1"},
        )
        source_grid = {
            "lat": lat, "lon": lon,
            "lat_bounds": np.array([[-1.0, 0.0], [0.0, 1.0]]),
            "lon_bounds": np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        }
        target = {
            "lat": np.array([0.0]), "lon": np.array([1.0, 3.0]),
            "lat_bounds": np.array([[-1.0, 1.0]]), "lon_bounds": np.array([[0.0, 2.0], [2.0, 4.0]]),
        }
        config = RegridConfig(
            spatial_basis=SpatialQuantityBasis.AREA_DENSITY, method=RegridMethod.CONSERVATIVE,
            check_conservation=False, positive_definite=True,
        )
        ledger = RegridOperator().regrid(source, source_grid, target, config).positive_definite_ledger
        assert ledger["clip_amount"] < 0.0
        assert ledger["amount_before"] == pytest.approx(ledger["amount_after"] + ledger["clip_amount"], rel=1e-12)
