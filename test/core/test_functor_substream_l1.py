"""L1 tests for the Functor-owned random substream mixin.

References:
    - PyTorch Module extra-state contract.
    - Revised RELIC foundations: accepted-step rollback and restore-and-replay
      of Functor-owned random streams.
"""

import pytest
import torch

from relic.core.functor import (
    FunctorMetadata,
    FunctorSubstreamMixin,
    PhysicalFieldFunctor,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from relic.core.state import UpdateItem, UpdateMode


class _CounterFunctor(FunctorSubstreamMixin, PhysicalFieldFunctor):
    """Minimal stochastic Functor: one uniform draw per emitted step."""

    EXTRA_STATE_FORMAT = "relic.test.counter_substream"
    SUBSTREAM_LABEL = "Counter"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(self, seed, **kwargs):
        metadata = FunctorMetadata(
            name="counter",
            version="1.0.0",
            description="test substream owner",
            category=ProcessCategory.PHYSICAL_FIELD,
            input_variables=["counter_state"],
            output_variables=["counter_state"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        super().__init__(name="counter", config={"parameters": {}, "dependencies": []}, metadata=metadata, **kwargs)
        self._initialize_substream(seed)

    def get_dependencies(self):
        return []

    def draw(self):
        value = torch.rand((), dtype=torch.float64, generator=self.generator)
        self.draw_count += 1
        return value

    def forward(self, state, time, dt, **kwargs):
        value = state["counter_state"]
        self._require_substream_device(value)
        return {
            "counter_state": UpdateItem(
                value=self.draw().to(value) * 0.0,
                mode=UpdateMode.ACCUMULATE,
                unit="1",
                per_timestep=True,
                source_functor=self.name,
            )
        }


class TestSeedValidation:
    @pytest.mark.parametrize("seed", [None, -1, 1.5, True, 2**63])
    def test_invalid_seeds_fail_closed(self, seed):
        with pytest.raises(ValueError, match="seed"):
            _CounterFunctor(seed)

    def test_valid_seed_initializes_a_fresh_stream(self):
        functor = _CounterFunctor(11)
        assert functor.seed == 11
        assert functor.draw_count == 0
        assert isinstance(functor.generator, torch.Generator)
        assert torch.device(functor.generator.device).type == "cpu"


class TestExtraState:
    def test_extra_state_roundtrip_replays_the_same_stream(self):
        source = _CounterFunctor(5)
        first = source.draw()
        snapshot = source.get_extra_state()
        assert set(snapshot) == {
            "format", "version", "seed", "draw_count", "generator_state", "wiener_pending",
        }
        assert snapshot["wiener_pending"] == []
        assert snapshot["format"] == "relic.test.counter_substream"
        assert snapshot["draw_count"] == 1
        second_source = source.draw()

        restored = _CounterFunctor(99)
        restored.set_extra_state(snapshot)
        assert restored.seed == 5
        assert restored.draw_count == 1
        second_restored = restored.draw()
        assert float(second_restored) == float(second_source)
        assert float(first) != float(second_source)

    def test_state_dict_carries_the_extra_state(self):
        source = _CounterFunctor(3)
        source.draw()
        state_dict = source.state_dict()
        restored = _CounterFunctor(4)
        restored.load_state_dict(state_dict)
        assert restored.draw_count == 1
        assert float(restored.draw()) == float(source.draw())

    @pytest.mark.parametrize(
        "mutation, error",
        [
            (lambda s: "not a mapping", TypeError),
            (lambda s: {**s, "extra": 1}, ValueError),
            (lambda s: {**s, "format": "other"}, ValueError),
            (lambda s: {**s, "version": 3}, ValueError),
            (lambda s: {**s, "seed": -1}, ValueError),
            (lambda s: {**s, "draw_count": True}, ValueError),
            (lambda s: {**s, "generator_state": torch.zeros(3)}, TypeError),
        ],
    )
    def test_invalid_extra_state_is_rejected(self, mutation, error):
        functor = _CounterFunctor(1)
        snapshot = functor.get_extra_state()
        with pytest.raises(error):
            functor.set_extra_state(mutation(snapshot))


class TestDeviceMigration:
    def test_same_device_type_migration_keeps_the_stream(self):
        functor = _CounterFunctor(7)
        functor.draw()
        before = functor.generator.get_state().clone()
        functor.to_device("cpu")
        assert torch.equal(functor.generator.get_state(), before)
        assert functor.draw_count == 1

    def test_state_on_another_device_type_is_rejected(self, monkeypatch):
        functor = _CounterFunctor(7)

        class _FakeDevice:
            type = "cuda"

        class _FakeTensor:
            device = _FakeDevice()

        with pytest.raises(ValueError, match="substream lives on the Functor device"):
            functor._require_substream_device(_FakeTensor())

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA migration test requires a CUDA device")
    def test_cross_device_migration_only_before_draws(self):
        functor = _CounterFunctor(7)
        functor.to_device("cuda")
        assert torch.device(functor.generator.device).type == "cuda"
        functor.draw_count = 1
        with pytest.raises(ValueError, match="migration across device types after draws"):
            functor.to_device("cpu")



class TestBrownianTree:
    """Path-consistent Wiener increments under step-size changes (F05).

    References: Gaines and Lyons (1997) Brownian tree; Karlin and Taylor
    (1981) ch. 9 Brownian bridge: Delta W(h) | Delta W(H) ~ N((h/H) Delta W(H),
    h (H - h) / H).
    """

    def _like(self, n=4):
        return torch.zeros(n, dtype=torch.float64)

    def test_fresh_draw_has_wiener_scaling_and_advances_only_the_generator(self):
        functor = _CounterFunctor(11)
        like = self._like()
        before = functor.generator.get_state().clone()
        first = functor.draw_wiener_increment(0.0, 4.0, like=like)
        functor.generator.set_state(before)
        second = functor.draw_wiener_increment(0.0, 1.0, like=like)
        # Same underlying normal, scaled by sqrt(width): fixed-step replay.
        torch.testing.assert_close(first / 2.0, second, rtol=0.0, atol=0.0)
        assert functor.wiener_pending == []
        assert functor.draw_count == 0

    def test_split_sum_identity_and_conditional_law(self):
        functor = _CounterFunctor(5)
        like = self._like(2000)
        whole = functor.draw_wiener_increment(0.0, 2.0, like=like)
        functor.wiener_pending.append({"time": 0.0, "width": 2.0, "increment": whole.clone()})
        first = functor.draw_wiener_increment(0.0, 0.5, like=like)
        assert len(functor.wiener_pending) == 1
        remainder = functor.wiener_pending[0]
        assert remainder["time"] == 0.5 and remainder["width"] == 1.5
        # Delta W(H) = Delta W(h) + Delta W(H - h) holds to rounding (one ulp).
        torch.testing.assert_close(first + remainder["increment"], whole, rtol=1e-12, atol=1e-12)
        second = functor.draw_wiener_increment(0.5, 1.5, like=like)
        torch.testing.assert_close(first + second, whole, rtol=1e-12, atol=1e-12)
        assert functor.wiener_pending == []
        # Conditional law over the 2000 independent components: mean h/H W,
        # variance h (H - h) / H = 0.5 * 1.5 / 2 = 0.375.
        residual = first - 0.25 * whole
        assert abs(float(residual.mean())) < 0.05
        assert abs(float(residual.var()) - 0.375) < 0.04

    def test_merge_consumes_pending_and_extends_with_a_fresh_piece(self):
        functor = _CounterFunctor(9)
        like = self._like()
        pending = torch.ones(4, dtype=torch.float64)
        functor.wiener_pending.append({"time": 3.0, "width": 1.0, "increment": pending.clone()})
        before = functor.generator.get_state().clone()
        merged = functor.draw_wiener_increment(3.0, 3.0, like=like)
        functor.generator.set_state(before)
        fresh_tail = functor.draw_wiener_increment(4.0, 2.0, like=like)
        torch.testing.assert_close(merged, pending + fresh_tail, rtol=0.0, atol=0.0)
        assert functor.wiener_pending == []

    def test_overlapping_pending_without_shared_start_fails_closed(self):
        functor = _CounterFunctor(9)
        functor.wiener_pending.append({"time": 1.0, "width": 2.0, "increment": torch.zeros(4, dtype=torch.float64)})
        with pytest.raises(RuntimeError, match="inconsistent"):
            functor.draw_wiener_increment(0.0, 2.0, like=self._like())
        with pytest.raises(RuntimeError, match="does not match the state shape"):
            functor.draw_wiener_increment(1.0, 2.0, like=self._like(3))
        with pytest.raises(ValueError, match="finite and positive"):
            functor.draw_wiener_increment(0.0, 0.0, like=self._like())

    def test_rejected_trial_is_pinned_and_refined_not_redrawn(self):
        functor = _CounterFunctor(21)
        like = self._like()
        snapshot = functor.get_extra_state()
        trial = functor.draw_wiener_increment(10.0, 8.0, like=like)
        # Simulator rejects the trial: extra state rewinds to the snapshot.
        functor.set_extra_state(snapshot)
        functor._trial_draw_spec = (tuple(like.shape), like.dtype, like.device, 10.0, 8.0)
        functor.record_rejected_trial_path()
        assert len(functor.wiener_pending) == 1
        torch.testing.assert_close(functor.wiener_pending[0]["increment"], trial, rtol=0.0, atol=0.0)
        retry = functor.draw_wiener_increment(10.0, 2.0, like=like)
        remainder = functor.wiener_pending[0]["increment"]
        torch.testing.assert_close(retry + remainder, trial, rtol=1e-12, atol=1e-12)
        # Redrawing the short step from the rewound generator would have given
        # sqrt(2/8) * trial, which is the wrong (path-breaking) increment.
        assert not torch.allclose(retry, trial * 0.5)
        assert functor._trial_draw_spec is not None
        functor._trial_draw_spec = None
        functor.record_rejected_trial_path()  # nothing pending to pin
        assert len(functor.wiener_pending) == 1

    def test_pending_ledger_round_trips_through_extra_state(self):
        functor = _CounterFunctor(3)
        functor.wiener_pending.append({"time": 2.5, "width": 0.5, "increment": torch.tensor([1.0, -1.0])})
        state = functor.get_extra_state()
        assert state["version"] == 2 and len(state["wiener_pending"]) == 1
        other = _CounterFunctor(4)
        other.set_extra_state(state)
        assert other.wiener_pending[0]["time"] == 2.5 and other.wiener_pending[0]["width"] == 0.5
        torch.testing.assert_close(other.wiener_pending[0]["increment"], torch.tensor([1.0, -1.0]))
        for bad in (
            [{"time": 0.0, "width": 0.0, "increment": torch.zeros(1)}],
            [{"time": float("nan"), "width": 1.0, "increment": torch.zeros(1)}],
            [{"time": 0.0, "width": 1.0}],
            [{"time": 0.0, "width": 1.0, "increment": [1.0]}],
            "not-a-list",
        ):
            with pytest.raises((TypeError, ValueError)):
                other.set_extra_state(dict(state, wiener_pending=bad))
