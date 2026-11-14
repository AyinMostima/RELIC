"""
L1 Unit and L2 Integration Tests for RELICSimulator Ensemble Functionality

Tests ensemble simulation capabilities including:
- Parameter sampling from various distributions
- Ensemble execution with different modes
- Statistical aggregation and uncertainty quantification
- Integration with EnsembleRunner
- Convergence monitoring and quality control

Test Level: L1 (Unit) + L2 (Integration)
Dependencies: RELICSimulator, EnsembleRunner, parameter management
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import relic.stochastic.ensemble_runner as ensemble_runner_module

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.core.simulator import RELICSimulator
from relic.core.simulator.ensemble_ops import run_ensemble_simulation
from relic.core.state import StateManager
from relic.core.functor import (
    BiologicalProcessFunctor,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
)
from relic.stochastic.ensemble_runner import (
    EnsembleResults,
    EnsembleRunner,
    ExecutionConfig,
    ExecutionMode,
    StatisticType,
)
from test.fixtures.parallel_workers import ensemble_linear_model


class TestParameterSampling:
    """Test parameter sampling from distributions (P2.3-1)"""

    def test_sample_normal_distribution(self):
        """Test sampling from normal distribution"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
        }
        simulator = RELICSimulator(config)

        distributions = {
            'param1': {
                'type': 'normal',
                'mean': 10.0,
                'std': 2.0
            }
        }

        samples = simulator._sample_parameters(distributions, num_samples=100, seed=42)

        assert 'param1' in samples
        assert samples['param1'].shape == (100,)

        # Check statistical properties
        sample_mean = samples['param1'].mean().item()
        sample_std = samples['param1'].std().item()

        # Should be close to specified mean and std (within tolerance)
        assert abs(sample_mean - 10.0) < 1.0  # Within 1 unit
        assert abs(sample_std - 2.0) < 0.5    # Within 0.5 units

    def test_sample_lognormal_distribution(self):
        """Test sampling from lognormal distribution"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
        }
        simulator = RELICSimulator(config)

        distributions = {
            'param2': {
                'type': 'lognormal',
                'mean': 5.0,
                'std': 1.0
            }
        }

        samples = simulator._sample_parameters(distributions, num_samples=100, seed=42)

        assert 'param2' in samples
        assert samples['param2'].shape == (100,)

        # All samples should be positive
        assert torch.all(samples['param2'] > 0)

        # Check that distribution is skewed (lognormal property)
        median = samples['param2'].median().item()
        mean = samples['param2'].mean().item()
        # For lognormal, mean > median
        assert mean >= median

    def test_sample_uniform_distribution(self):
        """Test sampling from uniform distribution"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
        }
        simulator = RELICSimulator(config)

        distributions = {
            'param3': {
                'type': 'uniform',
                'low': 0.0,
                'high': 10.0
            }
        }

        samples = simulator._sample_parameters(distributions, num_samples=100, seed=42)

        assert 'param3' in samples
        assert samples['param3'].shape == (100,)

        # All samples should be within bounds
        assert torch.all(samples['param3'] >= 0.0)
        assert torch.all(samples['param3'] <= 10.0)

    def test_sample_constant_distribution(self):
        """Test constant parameter (no variation)"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
        }
        simulator = RELICSimulator(config)

        distributions = {
            'param4': {
                'type': 'constant',
                'value': 7.5
            }
        }

        samples = simulator._sample_parameters(distributions, num_samples=100, seed=42)

        assert 'param4' in samples
        assert samples['param4'].shape == (100,)

        # All samples should be identical
        assert torch.allclose(samples['param4'], torch.tensor(7.5))

    def test_sample_multiple_parameters(self):
        """Test sampling multiple parameters simultaneously"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
        }
        simulator = RELICSimulator(config)

        distributions = {
            'param_a': {'type': 'normal', 'mean': 5.0, 'std': 1.0},
            'param_b': {'type': 'uniform', 'low': 0.0, 'high': 1.0},
            'param_c': {'type': 'constant', 'value': 42.0}
        }

        samples = simulator._sample_parameters(distributions, num_samples=50, seed=42)

        assert len(samples) == 3
        assert all(param in samples for param in ['param_a', 'param_b', 'param_c'])
        assert all(samples[param].shape == (50,) for param in samples)

    def test_sample_reproducibility_with_seed(self):
        """Test that sampling is reproducible with same seed"""
        config = {
            'time_step': 3600.0,
            'max_time': 86400.0,
        }
        simulator = RELICSimulator(config)

        distributions = {
            'param1': {'type': 'normal', 'mean': 10.0, 'std': 2.0}
        }

        samples1 = simulator._sample_parameters(distributions, num_samples=100, seed=123)
        samples2 = simulator._sample_parameters(distributions, num_samples=100, seed=123)

        assert torch.allclose(samples1['param1'], samples2['param1'])

    def test_sampling_uses_local_rng_substream(self):
        """Sampling must not mutate process-global RNG streams."""
        simulator = RELICSimulator({'time_step': 3600.0, 'max_time': 86400.0})
        distributions = {
            'param1': {'type': 'normal', 'mean': 10.0, 'std': 2.0},
            'param2': {'type': 'lognormal', 'mean': 5.0, 'std': 1.0},
            'param3': {'type': 'uniform', 'low': 0.0, 'high': 1.0},
        }

        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        python_state_before = random.getstate()
        numpy_state_before = np.random.get_state()
        torch_state_before = torch.get_rng_state().clone()

        simulator._sample_parameters(distributions, num_samples=16, seed=404)

        assert random.getstate() == python_state_before
        numpy_state_after = np.random.get_state()
        assert numpy_state_after[0] == numpy_state_before[0]
        assert np.array_equal(numpy_state_after[1], numpy_state_before[1])
        assert numpy_state_after[2:] == numpy_state_before[2:]
        assert torch.equal(torch.get_rng_state(), torch_state_before)

    def test_different_sampling_seeds_produce_different_substreams(self):
        simulator = RELICSimulator({'time_step': 3600.0, 'max_time': 86400.0})
        distributions = {
            'param1': {'type': 'normal', 'mean': 10.0, 'std': 2.0}
        }

        first = simulator._sample_parameters(distributions, num_samples=32, seed=1)
        second = simulator._sample_parameters(distributions, num_samples=32, seed=2)

        assert not torch.equal(first['param1'], second['param1'])

    @pytest.mark.parametrize('num_samples', [0, -1, 1.5, True])
    def test_sampling_rejects_invalid_sample_count(self, num_samples):
        simulator = RELICSimulator({'time_step': 3600.0, 'max_time': 86400.0})
        distributions = {
            'param1': {'type': 'normal', 'mean': 10.0, 'std': 2.0}
        }

        with pytest.raises(ValueError, match='num_samples'):
            simulator._sample_parameters(distributions, num_samples=num_samples, seed=42)

    @pytest.mark.parametrize(
        'distribution, expected_message',
        [
            ({'type': 'unknown', 'value': 1.0}, 'Unsupported distribution'),
            ({'type': 'normal', 'mean': 0.0, 'std': -1.0}, 'std'),
            ({'type': 'lognormal', 'mean': 0.0, 'std': 1.0}, 'mean'),
            ({'type': 'lognormal', 'mean': 1.0, 'std': -1.0}, 'std'),
            ({'type': 'uniform', 'low': 1.0, 'high': 1.0}, 'low < high'),
            ({'type': 'constant', 'value': float('nan')}, 'finite'),
            ({'type': 'normal', 'mean': float('inf'), 'std': 1.0}, 'finite'),
        ],
    )
    def test_sampling_rejects_invalid_distribution_domain(
        self,
        distribution,
        expected_message,
    ):
        simulator = RELICSimulator({'time_step': 3600.0, 'max_time': 86400.0})

        with pytest.raises(ValueError, match=expected_message):
            simulator._sample_parameters(
                {'param1': distribution},
                num_samples=8,
                seed=42,
            )

    @pytest.mark.parametrize(
        'distribution',
        [
            {},
            {'type': 'normal', 'std': 1.0},
            {'type': 'normal', 'mean': 0.0},
            {'type': 'lognormal', 'std': 1.0},
            {'type': 'uniform', 'low': 0.0},
            {'type': 'constant'},
        ],
    )
    def test_sampling_requires_explicit_distribution_fields(self, distribution):
        simulator = RELICSimulator({'time_step': 3600.0, 'max_time': 86400.0})

        with pytest.raises(ValueError, match='requires'):
            simulator._sample_parameters(
                {'param1': distribution},
                num_samples=8,
                seed=42,
            )


class TestEnsembleRunnerContracts:
    """Executable contracts for the canonical stochastic ensemble runner."""

    @staticmethod
    def _sequential_config() -> ExecutionConfig:
        return ExecutionConfig(
            mode=ExecutionMode.SEQUENTIAL,
            outlier_detection=False,
            convergence_check=False,
        )

    @pytest.mark.parametrize(
        "overrides, expected_message",
        [
            ({"n_workers": 0}, "n_workers"),
            ({"chunk_size": 0}, "chunk_size"),
            ({"outlier_threshold": 0.0}, "outlier_threshold"),
            ({"convergence_window": 1}, "convergence_window"),
            ({"convergence_tolerance": 0.0}, "convergence_tolerance"),
        ],
    )
    def test_execution_config_rejects_invalid_numeric_contracts(
        self,
        overrides,
        expected_message,
    ):
        with pytest.raises(ValueError, match=expected_message):
            ExecutionConfig(**overrides)

    def test_runner_does_not_mutate_process_global_rng_streams(self):
        runner = EnsembleRunner(config=self._sequential_config())
        samples = {"theta": torch.tensor([1.0, 2.0, 3.0])}

        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        python_state_before = random.getstate()
        numpy_state_before = np.random.get_state()
        torch_state_before = torch.get_rng_state().clone()

        runner.run_ensemble(
            model_callable=lambda inputs: {"y": inputs["theta"]},
            parameter_samples=samples,
            statistics=[StatisticType.MEAN],
            seed=404,
        )

        assert random.getstate() == python_state_before
        numpy_state_after = np.random.get_state()
        assert numpy_state_after[0] == numpy_state_before[0]
        assert np.array_equal(numpy_state_after[1], numpy_state_before[1])
        assert numpy_state_after[2:] == numpy_state_before[2:]
        assert torch.equal(torch.get_rng_state(), torch_state_before)

    def test_stochastic_sample_interface_is_fail_closed(self):
        runner = EnsembleRunner(config=self._sequential_config())

        with pytest.raises(NotImplementedError, match="run_ensemble"):
            runner.sample(torch.tensor([1.0]), num_samples=1)

    def test_process_mode_requires_canonical_purity_contract_before_execution(self):
        runner = EnsembleRunner(
            config=ExecutionConfig(
                mode=ExecutionMode.PARALLEL_PROCESS,
                outlier_detection=False,
                convergence_check=False,
            )
        )

        with pytest.raises(NotImplementedError, match=r"\[N\].*deterministic_pure"):
            runner.run_ensemble(
                model_callable=lambda inputs: {"y": inputs["theta"]},
                parameter_samples={"theta": torch.tensor([1.0, 2.0])},
                statistics=[StatisticType.MEAN],
            )

    @pytest.mark.serial
    def test_process_mode_runs_real_isolated_workers_in_member_order(self):
        runner = EnsembleRunner(
            config=ExecutionConfig(
                mode=ExecutionMode.PARALLEL_PROCESS,
                callable_contract=(
                    ensemble_runner_module.CallableContract.DETERMINISTIC_PURE
                ),
                n_workers=2,
                outlier_detection=False,
                convergence_check=False,
            )
        )

        results = runner.run_ensemble(
            model_callable=ensemble_linear_model,
            parameter_samples={"theta": torch.tensor([3.0, 1.0, 2.0, 4.0])},
            fixed_inputs={"offset": torch.tensor(1.0)},
            statistics=[StatisticType.MEAN],
            store_samples=True,
            seed=123,
        )

        assert results.samples is not None
        assert torch.equal(
            results.samples["y"],
            torch.tensor([7.0, 3.0, 5.0, 9.0]),
        )
        worker_pids = {
            int(pid)
            for pid in results.samples["worker_pid"].tolist()
        }
        assert worker_pids
        assert all(pid != os.getpid() for pid in worker_pids)
        assert results.execution_info["requested_backend"] == "processes"
        assert results.execution_info["effective_backend"] == "processes"
        assert results.execution_info["expected_count"] == 4
        assert results.execution_info["actual_count"] == 4
        assert set(results.execution_info["worker_pids"]) == worker_pids

    def test_thread_mode_requires_canonical_purity_contract_before_execution(self):
        runner = EnsembleRunner(
            config=ExecutionConfig(
                mode=ExecutionMode.PARALLEL_THREAD,
                n_workers=2,
                outlier_detection=False,
                convergence_check=False,
            )
        )
        model_calls = 0

        def model(inputs):
            nonlocal model_calls
            model_calls += 1
            return {"y": inputs["theta"]}

        with pytest.raises(NotImplementedError, match=r"\[N\].*deterministic_pure"):
            runner.run_ensemble(
                model_callable=model,
                parameter_samples={"theta": torch.tensor([1.0, 2.0])},
                statistics=[StatisticType.MEAN],
            )

        assert model_calls == 0

    def test_execution_config_rejects_free_string_purity_contract(self):
        with pytest.raises(TypeError, match="callable_contract"):
            ExecutionConfig(callable_contract="deterministic_pure")

    def test_thread_mode_preserves_member_order_for_pure_callable(self):
        runner = EnsembleRunner(
            config=ExecutionConfig(
                mode=ExecutionMode.PARALLEL_THREAD,
                callable_contract=(
                    ensemble_runner_module.CallableContract.DETERMINISTIC_PURE
                ),
                n_workers=2,
                outlier_detection=False,
                convergence_check=False,
            )
        )

        results = runner.run_ensemble(
            model_callable=lambda inputs: {"y": inputs["theta"] * 2.0},
            parameter_samples={"theta": torch.tensor([3.0, 1.0, 2.0])},
            statistics=[StatisticType.MEAN],
            store_samples=True,
        )

        assert results.samples is not None
        assert torch.equal(results.samples["y"], torch.tensor([6.0, 2.0, 4.0]))

    def test_sequential_runner_preserves_state_shape(self):
        runner = EnsembleRunner(config=self._sequential_config())
        samples = {"theta": torch.tensor([1.0, 2.0, 3.0, 4.0])}

        def model(inputs):
            value = inputs["theta"].reshape(())
            return {"field": torch.ones((2, 3)) * value}

        result = runner.run_ensemble(
            model_callable=model,
            parameter_samples=samples,
            statistics=[StatisticType.MEAN, StatisticType.STD],
            store_samples=True,
        )

        assert result.samples is not None
        assert result.samples["field"].shape == (4, 2, 3)
        assert result.statistics["field_mean"].shape == (2, 3)
        assert result.statistics["field_std"].shape == (2, 3)
        assert torch.equal(
            result.statistics["field_mean"],
            torch.full((2, 3), 2.5),
        )

    def test_vectorized_runner_preserves_state_shape(self):
        config = ExecutionConfig(
            mode=ExecutionMode.VECTORIZED,
            chunk_size=2,
            outlier_detection=False,
            convergence_check=False,
        )
        runner = EnsembleRunner(config=config)
        samples = {"theta": torch.tensor([1.0, 2.0, 3.0, 4.0])}

        def model(inputs):
            return {"field": inputs["theta"][:, None, None].expand(-1, 2, 3)}

        result = runner.run_ensemble(
            model_callable=model,
            parameter_samples=samples,
            statistics=[StatisticType.MEAN],
            store_samples=True,
        )

        assert result.samples is not None
        assert result.samples["field"].shape == (4, 2, 3)
        assert result.statistics["field_mean"].shape == (2, 3)

    def test_member_failure_is_fail_closed_without_nan_placeholder(self):
        runner = EnsembleRunner(config=self._sequential_config())
        samples = {"theta": torch.tensor([0.0, 1.0, 2.0])}

        def model(inputs):
            if bool((inputs["theta"] == 1.0).item()):
                raise ValueError("intentional member failure")
            return {"y": inputs["theta"]}

        with pytest.raises(RuntimeError, match="member 1"):
            runner.run_ensemble(
                model_callable=model,
                parameter_samples=samples,
                statistics=[StatisticType.MEAN],
                store_samples=True,
            )

    @pytest.mark.parametrize(
        "parameter_samples, expected_message",
        [
            ({"theta": torch.tensor(1.0)}, "at least one dimension"),
            ({"theta": torch.empty(0)}, "at least one member"),
            ({"theta": torch.tensor([1.0, torch.nan])}, "finite"),
            (
                {"theta": torch.tensor([1.0, 2.0]), "phi": torch.tensor([1.0])},
                "same ensemble size",
            ),
        ],
    )
    def test_runner_rejects_invalid_parameter_sample_ledgers(
        self,
        parameter_samples,
        expected_message,
    ):
        runner = EnsembleRunner(config=self._sequential_config())

        with pytest.raises(ValueError, match=expected_message):
            runner.run_ensemble(
                model_callable=lambda inputs: {"y": inputs["theta"]},
                parameter_samples=parameter_samples,
                statistics=[StatisticType.MEAN],
            )

    def test_fixed_inputs_cannot_shadow_sampled_parameters(self):
        runner = EnsembleRunner(config=self._sequential_config())

        with pytest.raises(ValueError, match="overlap"):
            runner.run_ensemble(
                model_callable=lambda inputs: {"y": inputs["theta"]},
                parameter_samples={"theta": torch.tensor([1.0, 2.0])},
                fixed_inputs={"theta": torch.tensor(99.0)},
                statistics=[StatisticType.MEAN],
            )

    def test_output_keys_must_be_identical_for_every_member(self):
        runner = EnsembleRunner(config=self._sequential_config())

        def model(inputs):
            if bool((inputs["theta"] == 0.0).item()):
                return {"first": inputs["theta"]}
            return {"second": inputs["theta"]}

        with pytest.raises(ValueError, match="output keys"):
            runner.run_ensemble(
                model_callable=model,
                parameter_samples={"theta": torch.tensor([0.0, 1.0])},
                statistics=[StatisticType.MEAN],
            )

    def test_runner_rejects_empty_or_nonfinite_outputs(self):
        runner = EnsembleRunner(config=self._sequential_config())
        samples = {"theta": torch.tensor([1.0, 2.0])}

        with pytest.raises(ValueError, match="non-empty"):
            runner.run_ensemble(
                model_callable=lambda _inputs: {},
                parameter_samples=samples,
                statistics=[StatisticType.MEAN],
            )

        with pytest.raises(ValueError, match="finite"):
            runner.run_ensemble(
                model_callable=lambda _inputs: {"y": torch.tensor(torch.inf)},
                parameter_samples=samples,
                statistics=[StatisticType.MEAN],
            )

    def test_execution_statistics_are_scoped_to_one_run(self):
        runner = EnsembleRunner(config=self._sequential_config())
        samples = {"theta": torch.tensor([1.0, 2.0])}
        kwargs = {
            "model_callable": lambda inputs: {"y": inputs["theta"]},
            "parameter_samples": samples,
            "statistics": [StatisticType.MEAN],
        }

        first = runner.run_ensemble(**kwargs)
        second = runner.run_ensemble(**kwargs)

        assert first.execution_info["successful_runs"] == 2
        assert second.execution_info["successful_runs"] == 2
        assert second.execution_info["total_runs"] == 2

    def test_netcdf_export_requires_explicit_dimension_ledger(self):
        results = EnsembleResults(
            statistics={"field_mean": torch.ones((2, 3))},
        )

        with pytest.raises(ValueError, match="variable_dimensions"):
            results.to_netcdf_dict()

    def test_netcdf_export_uses_only_explicit_dimensions_and_attributes(self):
        results = EnsembleResults(
            statistics={
                "field_mean": torch.ones((2, 3)),
                "field_std": torch.zeros((2, 3)),
            },
            metadata={
                "variable_dimensions": {
                    "field_mean": ["latitude", "longitude"],
                    "field_std": ["latitude", "longitude"],
                },
                "variable_attributes": {
                    "field_mean": {
                        "standard_name": "air_temperature",
                        "units": "K",
                    },
                    "field_std": {
                        "long_name": "ensemble standard deviation of field",
                        "units": "K",
                    },
                },
            },
        )

        payload = results.to_netcdf_dict()

        assert payload["field_mean"]["dims"] == ["latitude", "longitude"]
        assert payload["field_std"]["dims"] == ["latitude", "longitude"]
        assert payload["field_mean"]["attrs"]["standard_name"] == "air_temperature"
        assert "standard_name" not in payload["field_std"]["attrs"]


class _EnsembleParameterSpec:
    def __init__(self, *, units="1", description="Test ensemble parameter"):
        self.units = units
        self.description = description
        self.bounds = None
        self.requires_grad = False
        self.dtype = "float32"


class _EnsembleParameterManager:
    def __init__(self, values):
        self.values = dict(values)
        self.parameters = {
            key: _EnsembleParameterSpec()
            for key in self.values
        }
        self.set_batch_calls = []

    def get_parameter(self, key):
        return self.parameters.get(key)

    def get(self, key, **_kwargs):
        return self.values.get(key)

    def get_batch(self, keys):
        return [self.values[key] for key in keys]

    def set_batch(self, updates, *, validate=True, notify=True):
        assert validate is True
        assert notify is True
        self.set_batch_calls.append(dict(updates))
        self.values.update(updates)


def _bind_ensemble_parameter(simulator, owner_name, key="test.param", value=1.0):
    """Bind a Functor-owned ParameterManager holding ``key`` and return a constant
    distribution over it (the smallest valid ensemble parameter declaration)."""
    if owner_name in simulator.functors:
        owner = simulator.functors[owner_name]
    else:
        class _ParameterOwner(torch.nn.Module):
            pass

        owner = _ParameterOwner()
        simulator.functors[owner_name] = owner
    owner._parameter_manager = _EnsembleParameterManager({key: value})
    return {key: {"type": "constant", "value": value}}


class TestSimulatorEnsembleConfiguration:
    """The simulator ensemble declares no literal execution settings (item 71).

    The sample count and the outlier-removal switch (which drops members) are
    the caller's; the convergence check is enabled from the member floor
    optimization.uncertainty.ensemble.convergence_min_members.
    """

    def test_sample_count_and_outlier_removal_are_required(self):
        import inspect

        for function in (RELICSimulator.run_ensemble, run_ensemble_simulation):
            parameters = inspect.signature(function).parameters
            for name in ("num_samples", "outlier_detection"):
                assert parameters[name].default is inspect.Parameter.empty, (function.__name__, name)
            assert "n_workers" not in parameters, function.__name__
        assert "chunk_size" not in inspect.getsource(run_ensemble_simulation)

    @pytest.mark.parametrize("outlier_detection", [True, False])
    def test_convergence_check_follows_the_minimum_member_count(
        self, monkeypatch, restore_global_parameter_manager, outlier_detection,
    ):
        key = "optimization.uncertainty.ensemble.convergence_min_members"
        restore_global_parameter_manager.set_batch({key: 3}, validate=True, notify=False)
        captured = []
        original_init = ensemble_runner_module.EnsembleRunner.__init__

        def capturing_init(runner, config=None, **kwargs):
            captured.append(config)
            original_init(runner, config=config, **kwargs)

        monkeypatch.setattr(ensemble_runner_module.EnsembleRunner, "__init__", capturing_init)
        simulator = RELICSimulator({"time_step": 3600.0, "max_time": 3600.0, "device": "cpu"})
        state_manager = StateManager()
        state_manager.initialize()
        simulator.set_state_manager(state_manager)
        monkeypatch.setattr(simulator, "reset", lambda: None)
        monkeypatch.setattr(simulator, "run", lambda **_kwargs: None)
        monkeypatch.setattr(simulator, "_get_current_state", lambda: {"member_output": torch.tensor(1.0)})
        for num_samples in (2, 3):
            simulator.run_ensemble(
                end_time=3600.0,
                dt=3600.0,
                num_samples=num_samples,
                parameter_distributions=_bind_ensemble_parameter(simulator, "parameter_owner"),
                outlier_detection=outlier_detection,
                statistics=["mean"],
                seed=42,
            )
        assert [config.convergence_check for config in captured] == [False, True]
        assert [config.outlier_detection for config in captured] == [outlier_detection] * 2

    def test_runner_settings_come_from_the_stochastic_ensemble_database(self):
        """The simulator passes no runner numerics; ExecutionConfig reads
        stochastic.ensemble.* (the simulator declares no second value)."""
        import inspect

        from relic.parameters import get_global_manager

        source = inspect.getsource(run_ensemble_simulation)
        for name in ("outlier_threshold", "convergence_window", "convergence_tolerance"):
            assert f"{name}=" not in source, name
        manager = get_global_manager()
        config = ExecutionConfig(mode=ExecutionMode.SEQUENTIAL, outlier_detection=True)
        assert config.outlier_threshold == manager.get("stochastic.ensemble.outlier_threshold", required=True) == 3.0
        assert config.convergence_window == manager.get("stochastic.ensemble.convergence_window", required=True) == 50
        assert config.convergence_tolerance == manager.get("stochastic.ensemble.convergence_tolerance", required=True) == 0.01


class TestEnsembleExecution:
    """Test ensemble execution with simple functors (P2.3-2)"""

    @pytest.fixture
    def simple_simulator(self):
        """Create simulator with simple test functor"""
        config = {
            'time_step': 3600.0,
            'max_time': 7200.0,  # Short simulation for testing
            'device': 'cpu',
        }

        simulator = RELICSimulator(config)

        # Initialize state manager
        state_manager = StateManager()
        state_manager.initialize()
        simulator.set_state_manager(state_manager)

        # Register simple functor
        class SimpleTestFunctor(BiologicalProcessFunctor):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="simple_test",
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                    description="Simple test functor for ensemble",
                    references=["RELIC ensemble simulator architecture test fixture"],
                    output_variables=["test_output"],
                )
                super().__init__(name="simple_test", metadata=metadata, config={})
                # Register a parameter
                self.param_value = 10.0

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                # Simple computation: output = param_value * time factor
                time_factor = 1.0 + (time / 10000.0)
                output = self.param_value * time_factor
                return {"test_output": torch.tensor(output)}

        functor = SimpleTestFunctor()
        simulator.register_functor(functor)

        return simulator

    @pytest.mark.parametrize("distributions", [None, {}])
    def test_ensemble_without_parameter_distributions_is_rejected(self, simple_simulator, distributions):
        """Members restore one runtime transaction and replay it: without a parameter
        distribution every member is the same run, so the request is rejected."""
        with pytest.raises(ValueError, match="parameter_distributions"):
            simple_simulator.run_ensemble(
                end_time=3600.0,
                dt=3600.0,
                num_samples=3,
                outlier_detection=True,
                parameter_distributions=distributions,
                statistics=["mean"],
                seed=42,
            )

    def test_ensemble_basic_execution(self, simple_simulator):
        """Basic ensemble execution with a constant parameter distribution"""
        distributions = _bind_ensemble_parameter(simple_simulator, "simple_test")
        results = simple_simulator.run_ensemble(
            start_time=0.0,
            end_time=7200.0,
            dt=3600.0,
            num_samples=3,
            outlier_detection=True,
            parameter_distributions=distributions,
            execution_mode='sequential',
            statistics=['mean', 'std'],
            store_samples=True,
            seed=42
        )

        # Verify results structure
        assert isinstance(results, EnsembleResults)
        assert results.execution_info.get('successful_runs', 0) == 3
        assert "test_output_mean" in results.statistics
        assert "test_output_std" in results.statistics

        # With no parameter variation, std should be zero or very small
        std_value = results.statistics["test_output_std"]
        mean_value = results.statistics["test_output_mean"]
        rounding_bound = 4.0 * torch.finfo(std_value.dtype).eps * mean_value.abs()
        assert std_value <= rounding_bound

    def test_simulator_default_ensemble_mode_is_operational_sequential(
        self,
        simple_simulator,
    ):
        results = simple_simulator.run_ensemble(
            end_time=3600.0,
            dt=3600.0,
            num_samples=2,
            outlier_detection=True,
            parameter_distributions=_bind_ensemble_parameter(simple_simulator, "simple_test"),
            statistics=["mean"],
            seed=42,
        )

        assert results.execution_info.get("successful_runs", 0) == 2

    def test_internal_ensemble_entrypoint_default_is_operational_sequential(
        self,
        simple_simulator,
    ):
        results = run_ensemble_simulation(
            simple_simulator,
            end_time=3600.0,
            dt=3600.0,
            num_samples=2,
            outlier_detection=True,
            parameter_distributions=_bind_ensemble_parameter(simple_simulator, "simple_test"),
            statistics=["mean"],
            seed=42,
        )

        assert results.execution_info.get("successful_runs", 0) == 2

    def test_ensemble_uses_functor_bound_parameter_manager_for_execution_and_evidence(
        self,
        simple_simulator,
        monkeypatch,
    ):
        _ = simple_simulator.state_manager.physical_constraint_manager
        parameter_manager = _EnsembleParameterManager({"test.param": 7.5})
        functor = simple_simulator.functors["simple_test"]
        functor._parameter_manager = parameter_manager

        def parameter_bound_forward(state, time, dt, **kwargs):
            value = functor.get_parameter_value("test.param", required=True)
            return {"test_output": value.reshape(())}

        monkeypatch.setattr(functor, "forward", parameter_bound_forward)

        def forbidden_global_manager():
            raise AssertionError("ensemble bypassed the Functor-bound ParameterManager")

        monkeypatch.setattr(
            "relic.parameters.get_global_manager",
            forbidden_global_manager,
        )

        distributions = {
            'test.param': {
                'type': 'constant',
                'value': 12.0,
            }
        }

        # Run ensemble
        results = simple_simulator.run_ensemble(
            start_time=0.0,
            end_time=3600.0,
            dt=3600.0,
            num_samples=2,
            outlier_detection=True,
            parameter_distributions=distributions,
            execution_mode='sequential',
            statistics=['mean', 'std', 'var'],
            store_samples=False,
            seed=42
        )

        assert isinstance(results, EnsembleResults)
        assert results.execution_info.get('successful_runs', 0) == 2
        assert results.metadata["ensemble_size"] == 2
        assert "test.param" in results.metadata["parameter_statistics"]
        parameter_evidence = results.metadata["parameter_statistics"]["test.param"]
        assert parameter_evidence["units"] == "1"
        assert parameter_evidence["description"] == "Test ensemble parameter"
        assert "test_output_mean" in results.statistics
        assert torch.equal(
            results.statistics["test_output_mean"],
            torch.tensor(12.0),
        )
        assert parameter_manager.values["test.param"] == 7.5
        assert parameter_manager.set_batch_calls

    def test_ensemble_rejects_duplicate_parameter_owners_before_mutation(
        self,
        simple_simulator,
        monkeypatch,
    ):
        first_manager = _EnsembleParameterManager({"test.param": 7.5})
        second_manager = _EnsembleParameterManager({"test.param": 8.5})
        simple_simulator.functors["simple_test"]._parameter_manager = first_manager

        class SecondOwner(torch.nn.Module):
            def __init__(self, parameter_manager):
                super().__init__()
                self._parameter_manager = parameter_manager

        second_owner = SecondOwner(second_manager)
        simple_simulator.functors["second_owner"] = second_owner
        reset_calls = 0

        def forbidden_reset():
            nonlocal reset_calls
            reset_calls += 1

        monkeypatch.setattr(simple_simulator, "reset", forbidden_reset)

        with pytest.raises(ValueError, match="multiple ParameterManager owners"):
            simple_simulator.run_ensemble(
                end_time=3600.0,
                dt=3600.0,
                num_samples=2,
                outlier_detection=True,
                parameter_distributions={
                    "test.param": {"type": "constant", "value": 12.0}
                },
                statistics=["mean"],
                seed=42,
            )

        assert reset_calls == 0
        assert first_manager.set_batch_calls == []
        assert second_manager.set_batch_calls == []

    def test_ensemble_rejects_parameter_ledger_spanning_multiple_managers(
        self,
        simple_simulator,
        monkeypatch,
    ):
        first_manager = _EnsembleParameterManager({"test.alpha": 1.0})
        second_manager = _EnsembleParameterManager({"test.beta": 2.0})
        simple_simulator.functors["simple_test"]._parameter_manager = first_manager

        class SecondOwner(torch.nn.Module):
            def __init__(self, parameter_manager):
                super().__init__()
                self._parameter_manager = parameter_manager

        second_owner = SecondOwner(second_manager)
        simple_simulator.functors["second_owner"] = second_owner
        reset_calls = 0

        def forbidden_reset():
            nonlocal reset_calls
            reset_calls += 1

        monkeypatch.setattr(simple_simulator, "reset", forbidden_reset)

        with pytest.raises(ValueError, match="multiple ParameterManager instances"):
            simple_simulator.run_ensemble(
                end_time=3600.0,
                dt=3600.0,
                num_samples=2,
                outlier_detection=True,
                parameter_distributions={
                    "test.alpha": {"type": "constant", "value": 3.0},
                    "test.beta": {"type": "constant", "value": 4.0},
                },
                statistics=["mean"],
                seed=42,
            )

        assert reset_calls == 0
        assert first_manager.set_batch_calls == []
        assert second_manager.set_batch_calls == []

    def test_ensemble_restores_runtime_functor_state_and_rng_on_success(
        self,
        simple_simulator,
        monkeypatch,
    ):
        functor = simple_simulator.functors["simple_test"]
        functor.register_buffer("hidden_counter", torch.tensor(3.0))
        original_forward = functor.forward

        def mutating_forward(state, time, dt, **kwargs):
            with torch.no_grad():
                functor.hidden_counter.add_(1.0)
            random.random()
            np.random.random()
            torch.rand(())
            return original_forward(state, time, dt, **kwargs)

        monkeypatch.setattr(functor, "forward", mutating_forward)
        simple_simulator.state_manager.set_state(
            {"caller_marker": torch.tensor(9.0)},
            validate=False,
        )
        simple_simulator.current_time = 123.0
        simple_simulator.step_count = 7

        random.seed(101)
        python_state = random.getstate()
        expected_python = random.random()
        random.setstate(python_state)
        np.random.seed(202)
        numpy_state = np.random.get_state()
        expected_numpy = np.random.random()
        np.random.set_state(numpy_state)
        torch.manual_seed(303)
        torch_state = torch.get_rng_state()
        expected_torch = torch.rand(())
        torch.set_rng_state(torch_state)

        results = simple_simulator.run_ensemble(
            start_time=0.0,
            end_time=3600.0,
            dt=3600.0,
            num_samples=2,
            outlier_detection=True,
            parameter_distributions=_bind_ensemble_parameter(simple_simulator, "simple_test"),
            statistics=["mean"],
            seed=42,
        )

        assert results.execution_info["successful_runs"] == 2
        assert simple_simulator.current_time == 123.0
        assert simple_simulator.step_count == 7
        assert set(simple_simulator.state_manager.get_state()) == {"caller_marker"}
        assert torch.equal(
            simple_simulator.state_manager.get_state()["caller_marker"],
            torch.tensor(9.0),
        )
        assert torch.equal(functor.hidden_counter, torch.tensor(3.0))
        assert random.random() == expected_python
        assert np.random.random() == expected_numpy
        assert torch.equal(torch.rand(()), expected_torch)

    def test_ensemble_restores_runtime_parameters_and_rng_on_member_failure(
        self,
        simple_simulator,
        monkeypatch,
    ):
        parameter_manager = _EnsembleParameterManager({"test.param": 6.0})
        functor = simple_simulator.functors["simple_test"]
        functor._parameter_manager = parameter_manager

        def forbidden_global_manager():
            raise AssertionError("ensemble bypassed the Functor-bound ParameterManager")

        monkeypatch.setattr(
            "relic.parameters.get_global_manager",
            forbidden_global_manager,
        )
        functor.register_buffer("hidden_counter", torch.tensor(4.0))
        simple_simulator.state_manager.set_state(
            {"caller_marker": torch.tensor(11.0)},
            validate=False,
        )
        simple_simulator.current_time = 321.0
        simple_simulator.step_count = 8

        random.seed(404)
        python_state = random.getstate()
        expected_python = random.random()
        random.setstate(python_state)
        np.random.seed(505)
        numpy_state = np.random.get_state()
        expected_numpy = np.random.random()
        np.random.set_state(numpy_state)
        torch.manual_seed(606)
        torch_state = torch.get_rng_state()
        expected_torch = torch.rand(())
        torch.set_rng_state(torch_state)

        def failing_run(*, start_time, end_time, dt):
            simple_simulator.current_time = 999.0
            simple_simulator.step_count = 999
            simple_simulator.state_manager.set_state(
                {"leaked_state": torch.tensor(1.0)},
                validate=False,
            )
            with torch.no_grad():
                functor.hidden_counter.add_(5.0)
            random.random()
            np.random.random()
            torch.rand(())
            raise ValueError("member failure")

        monkeypatch.setattr(simple_simulator, "run", failing_run)

        with pytest.raises(RuntimeError, match="member failure"):
            simple_simulator.run_ensemble(
                start_time=0.0,
                end_time=3600.0,
                dt=3600.0,
                num_samples=2,
                outlier_detection=True,
                parameter_distributions={
                    "test.param": {"type": "constant", "value": 12.0}
                },
                statistics=["mean"],
                seed=42,
            )

        assert parameter_manager.values["test.param"] == 6.0
        assert simple_simulator.current_time == 321.0
        assert simple_simulator.step_count == 8
        assert set(simple_simulator.state_manager.get_state()) == {"caller_marker"}
        assert torch.equal(
            simple_simulator.state_manager.get_state()["caller_marker"],
            torch.tensor(11.0),
        )
        assert torch.equal(functor.hidden_counter, torch.tensor(4.0))
        assert random.random() == expected_python
        assert np.random.random() == expected_numpy
        assert torch.equal(torch.rand(()), expected_torch)

    def test_ensemble_rejects_unregistered_parameters_before_runtime_mutation(
        self,
        simple_simulator,
        monkeypatch,
    ):
        parameter_manager = _EnsembleParameterManager({})
        simple_simulator.functors["simple_test"]._parameter_manager = parameter_manager
        reset_calls = 0

        def forbidden_reset():
            nonlocal reset_calls
            reset_calls += 1

        monkeypatch.setattr(
            "relic.parameters.get_global_manager",
            lambda: (_ for _ in ()).throw(
                AssertionError("ensemble used the global ParameterManager")
            ),
        )
        monkeypatch.setattr(simple_simulator, "reset", forbidden_reset)

        with pytest.raises(KeyError, match="not registered"):
            simple_simulator.run_ensemble(
                end_time=3600.0,
                dt=3600.0,
                num_samples=2,
                outlier_detection=True,
                parameter_distributions={
                    "missing.param": {"type": "constant", "value": 1.0}
                },
                statistics=["mean"],
                seed=42,
            )

        assert reset_calls == 0

    def test_simulator_vectorized_mode_is_fail_closed(
        self,
        simple_simulator,
        monkeypatch,
    ):
        """Shared mutable Simulator members cannot claim vectorized execution."""
        distributions = {
            'test.param': {
                'type': 'constant',
                'value': 15.0
            }
        }

        # Run with sequential mode
        results_seq = simple_simulator.run_ensemble(
            start_time=0.0,
            end_time=3600.0,
            dt=3600.0,
            num_samples=3,
            outlier_detection=True,
            parameter_distributions=_bind_ensemble_parameter(simple_simulator, "simple_test"),
            execution_mode='sequential',
            statistics=['mean'],
            seed=123
        )

        assert results_seq.execution_info.get('successful_runs', 0) == 3
        reset_calls = 0

        def forbidden_reset():
            nonlocal reset_calls
            reset_calls += 1

        monkeypatch.setattr(simple_simulator, "reset", forbidden_reset)
        with pytest.raises(NotImplementedError, match=r"\[N\].*[Vv]ectorized"):
            simple_simulator.run_ensemble(
                start_time=0.0,
                end_time=3600.0,
                dt=3600.0,
                num_samples=3,
                outlier_detection=True,
                parameter_distributions=distributions,
                execution_mode='vectorized',
                statistics=['mean'],
                seed=123
            )
        assert reset_calls == 0

    @pytest.mark.parametrize("execution_mode", ["parallel_thread", "parallel_process"])
    def test_simulator_parallel_modes_reject_before_runtime_mutation(
        self,
        simple_simulator,
        monkeypatch,
        execution_mode,
    ):
        reset_calls = 0

        def forbidden_reset():
            nonlocal reset_calls
            reset_calls += 1

        monkeypatch.setattr(simple_simulator, "reset", forbidden_reset)
        with pytest.raises(NotImplementedError, match=r"\[N\]"):
            simple_simulator.run_ensemble(
                end_time=3600.0,
                dt=3600.0,
                num_samples=2,
                outlier_detection=True,
                execution_mode=execution_mode,
                statistics=["mean"],
                seed=123,
            )
        assert reset_calls == 0


class TestEnsembleStatistics:
    """Test statistical aggregation in ensemble results (P2.3-3)"""

    @pytest.fixture
    def mock_ensemble_results(self):
        """Create mock ensemble results for testing statistics"""
        # Create sample data
        num_samples = 100
        sample_data = torch.randn(num_samples) * 10 + 50  # mean=50, std=10

        statistics = {
            'output_var': {
                'mean': sample_data.mean().item(),
                'std': sample_data.std().item(),
                'var': sample_data.var().item(),
                'min': sample_data.min().item(),
                'max': sample_data.max().item(),
                'median': sample_data.median().item(),
            }
        }

        results = EnsembleResults(
            statistics=statistics,
            samples={'output_var': sample_data},
            metadata={'test': True, 'ensemble_size': num_samples},
            execution_info={'successful_runs': num_samples, 'execution_time': 10.0}
        )

        return results, sample_data

    def test_ensemble_mean_computation(self, mock_ensemble_results):
        """Test mean statistic is correctly computed"""
        results, sample_data = mock_ensemble_results

        computed_mean = results.statistics['output_var']['mean']
        expected_mean = sample_data.mean().item()

        assert abs(computed_mean - expected_mean) < 1e-5

    def test_ensemble_std_computation(self, mock_ensemble_results):
        """Test standard deviation statistic"""
        results, sample_data = mock_ensemble_results

        computed_std = results.statistics['output_var']['std']
        expected_std = sample_data.std().item()

        assert abs(computed_std - expected_std) < 1e-5

    def test_ensemble_variance_computation(self, mock_ensemble_results):
        """Test variance statistic"""
        results, sample_data = mock_ensemble_results

        computed_var = results.statistics['output_var']['var']
        expected_var = sample_data.var().item()

        assert abs(computed_var - expected_var) < 1e-4

    def test_ensemble_quantiles(self):
        """Test quantile statistics"""
        # Create results with quantiles
        sample_data = torch.linspace(0, 100, 101)  # 0, 1, 2, ..., 100

        statistics = {
            'output_var': {
                'mean': sample_data.mean().item(),
                'quantile_0.25': torch.quantile(sample_data, 0.25).item(),
                'quantile_0.50': torch.quantile(sample_data, 0.50).item(),
                'quantile_0.75': torch.quantile(sample_data, 0.75).item(),
            }
        }

        results = EnsembleResults(
            statistics=statistics,
            samples={'output_var': sample_data},
            metadata={'ensemble_size': 101},
            execution_info={'successful_runs': 101, 'execution_time': 5.0}
        )

        # Check quantiles are in expected range
        assert 20 <= results.statistics['output_var']['quantile_0.25'] <= 30
        assert 45 <= results.statistics['output_var']['quantile_0.50'] <= 55
        assert 70 <= results.statistics['output_var']['quantile_0.75'] <= 80


class TestEnsembleIntegration:
    """Integration tests with EnsembleRunner (P2.3-4)"""

    def test_ensemble_runner_integration(self):
        """Test that simulator correctly integrates with EnsembleRunner"""
        config = {
            'time_step': 3600.0,
            'max_time': 3600.0,  # Single step
            'device': 'cpu',
        }

        simulator = RELICSimulator(config)

        # Initialize state
        state_manager = StateManager()
        state_manager.initialize()
        simulator.set_state_manager(state_manager)

        # Register minimal functor
        class MinimalFunctor(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def __init__(self):
                metadata = FunctorMetadata(
                    name="minimal",
                    category=ProcessCategory.TRANSPORT_MECHANISM,
                    description="Minimal ensemble integration functor",
                    references=["RELIC ensemble simulator architecture test fixture"],
                    output_variables=["minimal_output"],
                )
                super().__init__(name="minimal", metadata=metadata, config={})

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {"minimal_output": torch.tensor(1.0)}

        simulator.register_functor(MinimalFunctor())

        # Run ensemble
        results = simulator.run_ensemble(
            start_time=0.0,
            end_time=3600.0,
            dt=3600.0,
            num_samples=3,
            outlier_detection=True,
            parameter_distributions=_bind_ensemble_parameter(simulator, "minimal"),
            execution_mode='sequential',
            statistics=['mean'],
            seed=42
        )

        assert isinstance(results, EnsembleResults)
        assert results.execution_info.get('successful_runs', 0) == 3

    def test_ensemble_convergence_monitoring(self, monkeypatch):
        """Test that convergence monitoring works when enabled"""
        config = {
            'time_step': 3600.0,
            'max_time': 3600.0,
            'device': 'cpu',
        }

        simulator = RELICSimulator(config)
        state_manager = StateManager()
        state_manager.initialize()
        simulator.set_state_manager(state_manager)
        monkeypatch.setattr(simulator, "reset", lambda: None)
        monkeypatch.setattr(simulator, "run", lambda **_kwargs: None)
        monkeypatch.setattr(
            simulator,
            "_get_current_state",
            lambda: {"conv_output": torch.tensor(10.0)},
        )

        # Run with enough samples for convergence check
        results = simulator.run_ensemble(
            start_time=0.0,
            end_time=3600.0,
            dt=3600.0,
            num_samples=100,
            outlier_detection=True,
            parameter_distributions=_bind_ensemble_parameter(simulator, "parameter_owner"),
            execution_mode='sequential',
            statistics=['mean', 'std'],
            seed=42
        )

        assert results.execution_info.get('successful_runs', 0) == 100
        assert results.convergence_info is not None
        assert results.convergence_info["conv_output"]["converged"] is True
        assert results.convergence_info["conv_output"]["max_rel_change"] == pytest.approx(0.0)
        assert results.convergence_info["overall_converged"] is True


class TestEnsemblePerformance:
    """Performance tests for ensemble execution (P2.3-5)"""

    def test_simulator_vectorized_mode_has_no_performance_claim(self):
        """Unisolated Simulator members have no vectorized performance claim."""
        config = {
            'time_step': 3600.0,
            'max_time': 3600.0,
            'device': 'cpu',
        }

        simulator = RELICSimulator(config)
        state_manager = StateManager()
        state_manager.initialize()
        simulator.set_state_manager(state_manager)

        # Register simple functor
        class FastFunctor(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def __init__(self):
                metadata = FunctorMetadata(
                    name="fast_test",
                    category=ProcessCategory.TRANSPORT_MECHANISM,
                    description="Ensemble execution-mode performance test functor",
                    references=["RELIC ensemble simulator architecture test fixture"],
                    output_variables=["fast_output"],
                )
                super().__init__(name="fast_test", metadata=metadata, config={})

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {"fast_output": torch.tensor(5.0)}

        simulator.register_functor(FastFunctor())
        distributions = _bind_ensemble_parameter(simulator, "fast_test")

        results_seq = simulator.run_ensemble(
            num_samples=3,
            outlier_detection=True,
            parameter_distributions=distributions,
            execution_mode='sequential',
            statistics=['mean'],
            seed=42
        )

        assert results_seq.execution_info.get("successful_runs", 0) == 3
        with pytest.raises(NotImplementedError, match=r"\[N\].*[Vv]ectorized"):
            simulator.run_ensemble(
                num_samples=3,
                outlier_detection=True,
                parameter_distributions=distributions,
                execution_mode='vectorized',
                statistics=['mean'],
                seed=42
            )


class TestEnsembleEdgeCases:
    """Test edge cases and error handling (P2.3-6)"""

    def test_ensemble_with_zero_samples(self):
        """Test that zero samples raises appropriate error"""
        config = {
            'time_step': 3600.0,
            'max_time': 3600.0,
        }
        simulator = RELICSimulator(config)

        with pytest.raises((ValueError, RuntimeError)):
            simulator.run_ensemble(num_samples=0, outlier_detection=True)

    def test_ensemble_with_invalid_execution_mode(self):
        """Test that invalid execution mode raises error"""
        config = {
            'time_step': 3600.0,
            'max_time': 3600.0,
        }
        simulator = RELICSimulator(config)

        with pytest.raises(ValueError, match="Unsupported ensemble execution mode"):
            simulator.run_ensemble(
                num_samples=5,
                outlier_detection=True,
                execution_mode='invalid_mode',
                statistics=['mean']
            )

    def test_ensemble_with_empty_parameter_distributions(self):
        """Test ensemble with explicitly empty distributions"""
        config = {
            'time_step': 3600.0,
            'max_time': 3600.0,
        }
        simulator = RELICSimulator(config)

        state_manager = StateManager()
        state_manager.initialize()
        simulator.set_state_manager(state_manager)

        # Should run successfully with no parameter variation
        samples = simulator._sample_parameters({}, num_samples=10, seed=42)

        assert isinstance(samples, dict)
        assert len(samples) == 0


class TestEnsembleRunnerStatisticsContracts:
    """Statistics are computed as defined or omitted; none is fabricated (MA-21 C7).

    References: Hyndman and Fan (1996) sample quantiles; Robert and Casella
    (2004), ch. 4 (monitoring Monte Carlo averages).
    Mathematical expression: s^2 = sum (x_i - xbar)^2 / (N - 1) needs N >= 2;
    running-mean change r_k = |m_k - m_(k-1)| / |m_(k-1)|.
    """

    @staticmethod
    def _runner(**overrides):
        values = {
            "mode": ExecutionMode.SEQUENTIAL,
            "outlier_detection": False,
            "convergence_check": False,
        }
        values.update(overrides)
        return EnsembleRunner(config=ExecutionConfig(**values))

    def test_spread_of_a_single_member_is_omitted_not_zero(self):
        runner = self._runner()
        stats = runner._compute_statistics(
            {"y": torch.tensor([[1.0, 2.0]])},
            [StatisticType.MEAN, StatisticType.STD, StatisticType.VAR],
            [],
        )
        assert "y_mean" in stats
        assert "y_std" not in stats
        assert "y_var" not in stats

    def test_quantile_names_follow_the_declared_levels(self):
        runner = self._runner()
        stats = runner._compute_statistics(
            {"y": torch.linspace(0.0, 1.0, 11)},
            [StatisticType.QUANTILE],
            [0.05, 0.29, 0.97, 0.975],
        )
        assert set(stats) == {"y_q05", "y_q29", "y_q97", "y_q97.5"}

    def test_quantile_labels_are_exact_for_fractional_and_single_precision_levels(self):
        from relic.stochastic.ensemble_runner import quantile_label

        assert quantile_label(0.025) == "q02.5"
        assert quantile_label(torch.tensor(0.1, dtype=torch.float32)) == "q10"
        assert quantile_label(1.0) == "q100"

    def test_duplicate_quantile_levels_are_rejected(self):
        with pytest.raises(ValueError, match="quantile"):
            EnsembleRunner._validate_statistics_contract(
                [StatisticType.QUANTILE], [0.5, 0.5]
            )

    def test_small_magnitude_output_is_not_declared_converged_by_a_floor(self):
        """A 1e-6 denominator floor made any |mean| << 1e-6 look converged."""
        runner = self._runner(convergence_check=True, convergence_window=4)
        values = torch.tensor([1e-9, 3e-9, 1e-9, 3e-9, 9e-9, 9e-9, 9e-9, 9e-9]).reshape(8, 1)

        info = runner._check_convergence({"flux": values})

        assert info["flux"]["converged"] is False


class TestExecutionConfigParameterDatabase:
    """Round 5: unstated screening and convergence settings come from YAML.

    stochastic.ensemble.outlier_threshold (shared with EnsembleConfig),
    stochastic.ensemble.convergence_window and
    stochastic.ensemble.convergence_tolerance are the single source; a value
    stated in the configuration is used as stated.
    """

    def test_unstated_settings_are_the_parameter_database_values(self):
        config = ExecutionConfig()
        assert config.outlier_threshold == pytest.approx(3.0)
        assert config.convergence_window == 50
        assert config.convergence_tolerance == pytest.approx(0.01)

    @pytest.mark.parametrize(
        "field_name, key, value",
        [
            ("outlier_threshold", "stochastic.ensemble.outlier_threshold", 2.5),
            ("convergence_window", "stochastic.ensemble.convergence_window", 20),
            ("convergence_tolerance", "stochastic.ensemble.convergence_tolerance", 0.02),
        ],
    )
    def test_parameter_database_is_the_single_source(self, field_name, key, value):
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        original = manager.get(key, required=True)
        try:
            manager.set_value(key, value)
            assert getattr(ExecutionConfig(), field_name) == pytest.approx(value)
        finally:
            manager.set_value(key, original)

    def test_stated_settings_are_used_as_stated(self):
        config = ExecutionConfig(
            outlier_threshold=4.0, convergence_window=10, convergence_tolerance=0.05
        )
        assert config.outlier_threshold == 4.0
        assert config.convergence_window == 10
        assert config.convergence_tolerance == 0.05


class TestConvergenceStrideParameterDatabase:
    """Round 6: the running-mean stride divisor is stochastic.ensemble.convergence_stride_divisor."""

    def test_running_mean_stride_divisor_is_the_parameter_database_value(self):
        from relic.parameters import get_global_manager

        runner = EnsembleRunner(
            config=ExecutionConfig(convergence_window=20, convergence_tolerance=0.5)
        )
        values = {"y": torch.linspace(1.0, 2.0, 100, dtype=torch.float64).reshape(100, 1)}
        # stride max(1, 20 // 4) = 5: running means at 20, 25, ..., 100 -> 17
        assert runner._check_convergence(values)["y"]["n_windows"] == 17
        manager = get_global_manager()
        key = "stochastic.ensemble.convergence_stride_divisor"
        original = manager.get(key, required=True)
        try:
            manager.set_value(key, 2)
            # stride max(1, 20 // 2) = 10: running means at 20, 30, ..., 100 -> 9
            assert runner._check_convergence(values)["y"]["n_windows"] == 9
        finally:
            manager.set_value(key, original)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

