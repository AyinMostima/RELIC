"""
L1 Unit Tests for Gradient-Based Parameter Calibration

Tests for P7 implementation:
- UnifiedParameterManager.get_trainable_parameters()
- GradientCalibrator basic functionality
- Gradient flow verification
"""

import sys
import inspect
import pytest
import torch
import torch.nn as nn
from pathlib import Path

import relic.parameters.calibration as calibration_package
import relic.parameters.calibration.hooks as calibration_hooks

# Add relic to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.parameters import (
    get_global_manager, 
    reset_global_manager,
    ParameterSpec,
    ParameterCategory
)
from relic.parameters.calibration import (
    GradientCalibrator,
    GradientCalibrationConfig,
    CalibrationMethod
)
from relic.parameters.ops import manager_registry


@pytest.fixture(autouse=True)
def _isolate_parameter_manager_singleton(monkeypatch):
    """Each test registers its parameters on a fresh process-wide manager; the
    manager that was global before the test is put back afterwards, so the
    registrations of one test never reach a later test."""
    monkeypatch.setattr(manager_registry, "_global_manager", None)
    yield


def test_calibration_package_does_not_export_checkpoint_callback():
    assert "CheckpointCallback" not in calibration_package.__all__
    assert not hasattr(calibration_package, "CheckpointCallback")


def test_calibration_hooks_have_no_checkpoint_schema_or_callback():
    signature = inspect.signature(calibration_hooks.create_default_hooks)

    assert "CUSTOM_CHECKPOINT" not in calibration_hooks.CalibrationEvent.__members__
    assert not hasattr(calibration_hooks, "CheckpointCallback")
    assert not hasattr(calibration_hooks.CalibrationHooks, "add_checkpointing")
    assert "checkpoint_dir" not in signature.parameters
    assert "enable_checkpointing" not in signature.parameters


def test_legacy_checkpoint_argument_rejected_before_filesystem_effect(tmp_path):
    target = tmp_path / "must_not_exist"

    with pytest.raises(TypeError, match="unexpected keyword argument 'checkpoint_dir'"):
        calibration_hooks.create_default_hooks(checkpoint_dir=target)

    assert not target.exists()


def test_default_hooks_keep_non_durable_callbacks():
    hooks = calibration_hooks.create_default_hooks()
    callback_types = {type(callback) for callback in hooks.global_callbacks}

    assert callback_types == {
        calibration_hooks.LoggingCallback,
        calibration_hooks.ProgressCallback,
        calibration_hooks.ConvergenceCallback,
        calibration_hooks.AdaptiveCallback,
    }
    hooks.cleanup()


class TestGetTrainableParameters:
    """Test UnifiedParameterManager.get_trainable_parameters() method"""
    
    def test_get_trainable_parameters_empty(self):
        """Test get_trainable_parameters with no trainable parameters"""
        reset_global_manager()
        manager = get_global_manager()
        
        # Register non-trainable parameter
        spec = ParameterSpec(
            key='test.param1',
            value=1.0,
            units='dimensionless',
            category=ParameterCategory.PHYSICAL_CONSTANT,
            requires_grad=False, dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(spec)
        
        trainable = manager.get_trainable_parameters()
        assert len(trainable) == 0, "Should return empty dict when no trainable parameters"
    
    def test_get_trainable_parameters_single(self):
        """Test get_trainable_parameters with one trainable parameter"""
        reset_global_manager()
        manager = get_global_manager()
        
        # Register trainable parameter
        spec = ParameterSpec(
            key='test.param_trainable',
            value=2.0,
            units='m',
            category=ParameterCategory.CALIBRATION,
            requires_grad=True, dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(spec)
        
        trainable = manager.get_trainable_parameters()
        
        assert len(trainable) == 1, "Should return 1 trainable parameter"
        assert 'test.param_trainable' in trainable, "Should contain registered key"
        assert isinstance(trainable['test.param_trainable'], nn.Parameter), "Should be nn.Parameter"
        assert trainable['test.param_trainable'].requires_grad, "Should have requires_grad=True"
    
    def test_get_trainable_parameters_mixed(self):
        """Test get_trainable_parameters filters correctly"""
        reset_global_manager()
        manager = get_global_manager()
        
        # Register mix of trainable and non-trainable
        specs = [
            ParameterSpec(key='test.train1', value=1.0, units='m', requires_grad=True,
                         category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf"))),
            ParameterSpec(key='test.const1', value=2.0, units='m', requires_grad=False,
                         category=ParameterCategory.PHYSICAL_CONSTANT, dtype="float32", bounds=(-float("inf"), float("inf"))),
            ParameterSpec(key='test.train2', value=3.0, units='m', requires_grad=True,
                         category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf"))),
            ParameterSpec(key='test.const2', value=4.0, units='m', requires_grad=False,
                         category=ParameterCategory.PHYSICAL_CONSTANT, dtype="float32", bounds=(-float("inf"), float("inf"))),
        ]
        
        for spec in specs:
            manager.register_parameter(spec)
        
        trainable = manager.get_trainable_parameters()
        
        assert len(trainable) == 2, "Should return only 2 trainable parameters"
        assert 'test.train1' in trainable, "Should include test.train1"
        assert 'test.train2' in trainable, "Should include test.train2"
        assert 'test.const1' not in trainable, "Should exclude test.const1"
        assert 'test.const2' not in trainable, "Should exclude test.const2"
        
        # Verify all returned are nn.Parameter with requires_grad
        for key, param in trainable.items():
            assert isinstance(param, nn.Parameter), f"{key} should be nn.Parameter"
            assert param.requires_grad, f"{key} should have requires_grad=True"
    
    def test_get_trainable_parameters_after_freeze(self):
        """Test get_trainable_parameters respects freeze()"""
        reset_global_manager()
        manager = get_global_manager()
        
        # Register trainable parameter
        spec = ParameterSpec(
            key='test.freezable',
            value=5.0,
            units='kg',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(spec)
        
        # Initially trainable
        trainable_before = manager.get_trainable_parameters()
        assert len(trainable_before) == 1, "Should have 1 trainable before freeze"
        
        # Freeze the parameter
        manager.freeze(['test.freezable'])
        
        # Should no longer appear in trainable
        trainable_after = manager.get_trainable_parameters()
        assert len(trainable_after) == 0, "Should have 0 trainable after freeze"
        
        # Unfreeze
        manager.unfreeze(['test.freezable'])
        
        # Should reappear
        trainable_unfrozen = manager.get_trainable_parameters()
        assert len(trainable_unfrozen) == 1, "Should have 1 trainable after unfreeze"


class TestGradientCalibratorBasics:
    """Test GradientCalibrator basic functionality"""

    def test_gradient_calibrator_requires_explicit_parameter_manager(self):
        """Calibration must not bind to process-global parameter state implicitly."""
        with pytest.raises(ValueError, match="explicit parameter_manager"):
            GradientCalibrator(None)
    
    def test_gradient_calibrator_initialization(self):
        """Test GradientCalibrator initializes correctly"""
        reset_global_manager()
        manager = get_global_manager()
        
        config = GradientCalibrationConfig(
            method=CalibrationMethod.ADAM,
            learning_rate=0.01,
            max_iterations=10
        )
        
        calibrator = GradientCalibrator(manager, config)
        
        assert calibrator.parameter_manager is manager, "Should store parameter manager"
        assert calibrator.config is config, "Should store config"
        assert calibrator.optimizer is None, "Optimizer not created until calibrate()"
        assert len(calibrator.convergence_history) == 0, "Convergence history empty initially"
    
    def test_gradient_calibrator_config_defaults(self):
        """The method is a required choice; numeric settings come from the parameter database."""
        with pytest.raises(TypeError, match="method"):
            GradientCalibrationConfig()
        config = GradientCalibrationConfig(method=CalibrationMethod.ADAM)
        manager = get_global_manager()
        expected = {
            "learning_rate": "optimization.calibration.gradient.learning_rate",
            "beta1": "optimization.calibration.gradient.adam_beta1",
            "beta2": "optimization.calibration.gradient.adam_beta2",
            "epsilon": "optimization.calibration.gradient.adam_epsilon",
            "weight_decay": "optimization.calibration.gradient.weight_decay",
            "history_size": "optimization.calibration.gradient.lbfgs_history_size",
            "max_iterations": "optimization.calibration.max_iterations",
            "tolerance": "optimization.calibration.objective_tolerance",
            "patience": "optimization.calibration.early_stopping_patience",
            "min_delta": "optimization.calibration.min_objective_improvement",
        }
        for field_name, key in expected.items():
            assert getattr(config, field_name) == manager.get(key, required=True), field_name
        # Kingma & Ba (2015), Algorithm 1 default settings.
        assert (config.learning_rate, config.beta1, config.beta2, config.epsilon) == (0.001, 0.9, 0.999, 1e-8)

    def test_gradient_calibrator_requires_an_explicit_config(self):
        """No optimizer method is chosen on the caller's behalf."""
        reset_global_manager()
        with pytest.raises(ValueError, match="GradientCalibrationConfig"):
            GradientCalibrator(get_global_manager())
    
    def test_gradient_calibrator_simple_quadratic(self):
        """Test gradient calibration on simple quadratic objective"""
        reset_global_manager()
        manager = get_global_manager()
        
        # Register trainable parameter
        spec = ParameterSpec(
            key='test.x',
            value=5.0,  # Start far from optimum
            units='dimensionless',
            bounds=[0.0, 10.0],
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32"
        )
        manager.register_parameter(spec)
        
        # Simple quadratic objective: min (x - 2)^2
        # Optimum at x = 2
        def objective(params, obs):
            # params is now Dict[str, torch.Tensor], not Dict[str, float]
            x = params['test.x']
            loss = (x - 2.0) ** 2
            return loss
        
        config = GradientCalibrationConfig(
            method=CalibrationMethod.ADAM,
            learning_rate=0.1,
            max_iterations=50,
            tolerance=1e-4,
            verbose=False
        )
        
        calibrator = GradientCalibrator(manager, config)
        
        # Calibrate (observations not used for this simple case)
        result = calibrator.calibrate(objective, None, target_parameters=['test.x'])
        
        # Check convergence
        assert len(result.convergence_history) > 0, "Should have convergence history"
        assert result.objective_value < 0.1, f"Should converge close to optimum, got {result.objective_value}"
        
        # Check optimized parameter
        optimized_x = result.optimized_parameters['test.x']
        assert abs(optimized_x - 2.0) < 0.5, f"Should converge near x=2, got {optimized_x}"
        
        # Verify parameter was updated in manager
        manager_value = manager.get('test.x')
        assert abs(float(manager_value) - optimized_x) < 1e-6, "Manager should be updated"
    
    def test_gradient_calibrator_bounds_enforcement(self):
        """Test that parameter bounds are enforced during optimization"""
        reset_global_manager()
        manager = get_global_manager()
        
        # Register parameter with tight bounds
        spec = ParameterSpec(
            key='test.bounded',
            value=0.5,
            units='dimensionless',
            bounds=[0.0, 1.0],  # Constrained to [0, 1]
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32"
        )
        manager.register_parameter(spec)
        
        # Objective that would push parameter outside bounds
        # min (x - 5)^2, optimum at x=5 but constrained to [0,1]
        def objective(params, obs):
            x = params['test.bounded']
            loss = (x - 5.0) ** 2
            return loss
        
        config = GradientCalibrationConfig(
            method=CalibrationMethod.ADAM,
            learning_rate=0.1,
            max_iterations=20,
            verbose=False
        )
        
        calibrator = GradientCalibrator(manager, config)
        result = calibrator.calibrate(objective, None, target_parameters=['test.bounded'])
        
        # Should be pushed to upper bound
        optimized_x = result.optimized_parameters['test.bounded']
        assert 0.0 <= optimized_x <= 1.0, f"Parameter should stay within bounds, got {optimized_x}"
        assert optimized_x > 0.9, f"Should be near upper bound, got {optimized_x}"
    
    def test_gradient_calibrator_requires_differentiable_objective(self):
        """Test that non-differentiable objective raises error"""
        reset_global_manager()
        manager = get_global_manager()
        
        spec = ParameterSpec(
            key='test.param',
            value=1.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(spec)
        
        # Non-differentiable objective (returns float instead of tensor)
        def bad_objective(params, obs):
            return 1.0  # NOT a tensor
        
        config = GradientCalibrationConfig(
            method=CalibrationMethod.ADAM,
            max_iterations=5,
            verbose=False
        )
        
        calibrator = GradientCalibrator(manager, config)
        
        with pytest.raises(RuntimeError, match="must return torch.Tensor"):
            calibrator.calibrate(bad_objective, None, target_parameters=['test.param'])

    def test_gradient_calibrator_rejects_any_missing_target_parameter(self):
        """A partial target match must fail instead of calibrating a different set."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.available',
            value=1.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
        ))
        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(method=CalibrationMethod.ADAM, max_iterations=1, verbose=False),
        )

        with pytest.raises(ValueError, match="not trainable.*test.missing"):
            calibrator.calibrate(
                lambda params, obs: params['test.available'] ** 2,
                None,
                target_parameters=['test.available', 'test.missing'],
            )

    def test_gradient_calibrator_rejects_non_ssot_bounds(self):
        """Call-site bounds may confirm but never override ParameterSpec bounds."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.bounded_ssot',
            value=0.5,
            units='dimensionless',
            bounds=[0.0, 1.0],
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32",
        ))
        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(method=CalibrationMethod.ADAM, max_iterations=1, verbose=False),
        )

        with pytest.raises(ValueError, match="ParameterManager bounds"):
            calibrator.calibrate(
                lambda params, obs: params['test.bounded_ssot'] ** 2,
                None,
                target_parameters=['test.bounded_ssot'],
                parameter_bounds={'test.bounded_ssot': (0.0, 2.0)},
            )

    def test_gradient_calibrator_rolls_back_manager_owned_tensor_on_failure(self):
        """A failed optimization restores values without replacing nn.Parameter."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.rollback',
            value=1.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
        ))
        original_parameter = manager.get_trainable_parameters()['test.rollback']
        calls = 0

        def failing_objective(params, obs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected calibration failure")
            return (params['test.rollback'] - 5.0) ** 2

        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(
                method=CalibrationMethod.GRADIENT_DESCENT,
                learning_rate=1.0,
                max_iterations=3,
                verbose=False,
            ),
        )

        with pytest.raises(RuntimeError, match="injected calibration failure"):
            calibrator.calibrate(
                failing_objective,
                None,
                target_parameters=['test.rollback'],
            )

        restored_parameter = manager.get_trainable_parameters()['test.rollback']
        assert restored_parameter is original_parameter
        assert restored_parameter.item() == pytest.approx(1.0)
        assert float(manager.get('test.rollback')) == pytest.approx(1.0)

    def test_gradient_calibrator_commits_one_atomic_manager_batch(self, monkeypatch):
        """The calibrator must use the canonical batch mutation owner once."""
        reset_global_manager()
        manager = get_global_manager()
        for key, value in [('test.batch_a', 2.0), ('test.batch_b', 3.0)]:
            manager.register_parameter(ParameterSpec(
                key=key,
                value=value,
                units='dimensionless',
                requires_grad=True,
                category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
            ))
        calibrator = GradientCalibrator(
            manager, GradientCalibrationConfig(method=CalibrationMethod.ADAM, verbose=False)
        )
        calibrator.trainable_params = manager.get_trainable_parameters()
        original_set_batch = manager.set_batch
        calls = []

        def recording_set_batch(updates, **kwargs):
            calls.append((dict(updates), dict(kwargs)))
            return original_set_batch(updates, **kwargs)

        monkeypatch.setattr(manager, 'set_batch', recording_set_batch)
        calibrator._update_parameter_manager({
            'test.batch_a': 4.0,
            'test.batch_b': 5.0,
        })

        assert len(calls) == 1
        assert set(calls[0][0]) == {'test.batch_a', 'test.batch_b'}
        assert float(manager.get('test.batch_a')) == pytest.approx(4.0)
        assert float(manager.get('test.batch_b')) == pytest.approx(5.0)

    def test_gradient_calibrator_binds_objective_to_accepted_parameter_point(self):
        """Reported loss, history, and parameters must describe the same point."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.accepted_point',
            value=1.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
        ))
        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(
                method=CalibrationMethod.GRADIENT_DESCENT,
                learning_rate=0.1,
                max_iterations=1,
                verbose=False,
            ),
        )

        result = calibrator.calibrate(
            lambda params, obs: (params['test.accepted_point'] - 3.0) ** 2,
            None,
            target_parameters=['test.accepted_point'],
        )

        accepted = result.optimized_parameters['test.accepted_point']
        accepted_loss = (accepted - 3.0) ** 2
        assert result.objective_value == pytest.approx(accepted_loss)
        assert result.convergence_history[-1] == pytest.approx(accepted_loss)
        assert calibrator.parameter_history[-1]['test.accepted_point'] == pytest.approx(accepted)

    def test_gradient_calibrator_history_pairs_losses_with_parameter_snapshots(self):
        """Every history index must describe one accepted optimization point."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.history_pair',
            value=1.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
        ))
        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(
                method=CalibrationMethod.GRADIENT_DESCENT,
                learning_rate=0.1,
                max_iterations=3,
                tolerance=0.0,
                min_delta=0.0,
                patience=10,
                verbose=False,
            ),
        )

        result = calibrator.calibrate(
            lambda params, obs: (params['test.history_pair'] - 3.0) ** 2,
            None,
            target_parameters=['test.history_pair'],
        )

        assert len(calibrator.parameter_history) == result.iteration_count
        assert len(result.convergence_history) == result.iteration_count
        for snapshot, loss in zip(
            calibrator.parameter_history,
            result.convergence_history,
        ):
            assert loss == pytest.approx((snapshot['test.history_pair'] - 3.0) ** 2)

    def test_gradient_calibrator_keeps_manager_unchanged_during_search(self):
        """Only the final accepted batch may mutate canonical manager state."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.search_candidate',
            value=1.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
        ))
        canonical_values = []

        def objective(params, obs):
            canonical = manager.get_trainable_parameters()['test.search_candidate']
            canonical_values.append(float(canonical.detach().item()))
            return (params['test.search_candidate'] - 3.0) ** 2

        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(
                method=CalibrationMethod.GRADIENT_DESCENT,
                learning_rate=0.1,
                max_iterations=2,
                tolerance=0.0,
                min_delta=0.0,
                patience=10,
                verbose=False,
            ),
        )

        result = calibrator.calibrate(
            objective,
            None,
            target_parameters=['test.search_candidate'],
        )

        assert canonical_values
        assert canonical_values == pytest.approx([1.0] * len(canonical_values))
        assert float(manager.get('test.search_candidate')) == pytest.approx(
            result.optimized_parameters['test.search_candidate']
        )

    @pytest.mark.parametrize('invalid_kind', ['nan', 'inf', 'vector'])
    def test_gradient_calibrator_rejects_invalid_objective_and_rolls_back(self, invalid_kind):
        """The objective contract is differentiable, scalar, and finite."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.invalid_objective',
            value=1.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
        ))
        original_parameter = manager.get_trainable_parameters()['test.invalid_objective']

        def invalid_objective(params, obs):
            parameter = params['test.invalid_objective']
            if invalid_kind == 'vector':
                return torch.stack((parameter, parameter))
            invalid_value = float(invalid_kind)
            return parameter * torch.as_tensor(
                invalid_value,
                dtype=parameter.dtype,
                device=parameter.device,
            )

        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(
                method=CalibrationMethod.GRADIENT_DESCENT,
                max_iterations=1,
                verbose=False,
            ),
        )

        with pytest.raises(RuntimeError, match='finite scalar'):
            calibrator.calibrate(
                invalid_objective,
                None,
                target_parameters=['test.invalid_objective'],
            )

        restored_parameter = manager.get_trainable_parameters()['test.invalid_objective']
        assert restored_parameter is original_parameter
        assert restored_parameter.item() == pytest.approx(1.0)

    def test_gradient_calibrator_reports_min_delta_termination(self):
        """A min-delta stop must not be mislabeled as max-iteration exhaustion."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.plateau',
            value=2.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
        ))
        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(
                method=CalibrationMethod.GRADIENT_DESCENT,
                learning_rate=0.0,
                max_iterations=5,
                tolerance=0.0,
                min_delta=1.0e-12,
                patience=5,
                verbose=False,
            ),
        )

        result = calibrator.calibrate(
            lambda params, obs: (params['test.plateau'] - 4.0) ** 2,
            None,
            target_parameters=['test.plateau'],
        )

        assert result.iteration_count == 2
        assert result.termination_reason == 'Objective change below min_delta'
        assert result.success is False

    def test_gradient_calibrator_rejects_disconnected_target_gradient(self):
        """Every requested target must participate in the objective graph."""
        reset_global_manager()
        manager = get_global_manager()
        for key in ('test.connected', 'test.disconnected'):
            manager.register_parameter(ParameterSpec(
                key=key,
                value=1.0,
                units='dimensionless',
                requires_grad=True,
                category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
            ))
        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(method=CalibrationMethod.ADAM, max_iterations=1, verbose=False),
        )

        with pytest.raises(RuntimeError, match='not connected.*test.disconnected'):
            calibrator.calibrate(
                lambda params, obs: params['test.connected'] ** 2,
                None,
                target_parameters=['test.connected', 'test.disconnected'],
            )

    def test_gradient_calibrator_rejects_non_finite_target_gradient(self):
        """A finite loss with an infinite gradient is a numerical failure."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.non_finite_gradient',
            value=0.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
        ))
        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(
                method=CalibrationMethod.GRADIENT_DESCENT,
                max_iterations=1,
                verbose=False,
            ),
        )

        with pytest.raises(RuntimeError, match='non-finite gradient'):
            calibrator.calibrate(
                lambda params, obs: torch.sqrt(params['test.non_finite_gradient']),
                None,
                target_parameters=['test.non_finite_gradient'],
            )
    
    def test_gradient_calibrator_convergence_history(self):
        """Test that convergence history is properly recorded"""
        reset_global_manager()
        manager = get_global_manager()
        
        spec = ParameterSpec(
            key='test.x',
            value=3.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(spec)
        
        def objective(params, obs):
            x = params['test.x']
            return (x - 1.0) ** 2
        
        config = GradientCalibrationConfig(
            method=CalibrationMethod.ADAM,
            learning_rate=0.1,
            max_iterations=20,
            verbose=False
        )
        
        calibrator = GradientCalibrator(manager, config)
        result = calibrator.calibrate(objective, None, target_parameters=['test.x'])
        
        # Check history
        assert len(result.convergence_history) > 0, "Should have convergence history"
        assert len(result.convergence_history) <= config.max_iterations, "Should not exceed max iterations"
        
        # Loss should generally decrease
        first_loss = result.convergence_history[0]
        last_loss = result.convergence_history[-1]
        assert last_loss < first_loss, "Loss should decrease during optimization"


class TestGradientCalibratorOptimizers:
    """Test different optimizers in GradientCalibrator"""
    
    def test_adam_optimizer(self):
        """Test Adam optimizer works"""
        reset_global_manager()
        manager = get_global_manager()
        
        spec = ParameterSpec(
            key='test.adam_param',
            value=4.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(spec)
        
        def objective(params, obs):
            x = params['test.adam_param']
            return (x - 2.0) ** 2
        
        config = GradientCalibrationConfig(
            method=CalibrationMethod.ADAM,
            learning_rate=0.1,
            max_iterations=30,
            verbose=False
        )
        
        calibrator = GradientCalibrator(manager, config)
        result = calibrator.calibrate(objective, None, target_parameters=['test.adam_param'])
        
        assert result.objective_value < 0.5, "Adam should converge"
        assert abs(result.optimized_parameters['test.adam_param'] - 2.0) < 0.5, "Should be near optimum"
    
    def test_sgd_optimizer(self):
        """Test SGD optimizer works"""
        reset_global_manager()
        manager = get_global_manager()
        
        spec = ParameterSpec(
            key='test.sgd_param',
            value=5.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(spec)
        
        def objective(params, obs):
            x = params['test.sgd_param']
            return (x - 3.0) ** 2
        
        config = GradientCalibrationConfig(
            method=CalibrationMethod.GRADIENT_DESCENT,
            learning_rate=0.1,
            max_iterations=50,
            verbose=False
        )
        
        calibrator = GradientCalibrator(manager, config)
        result = calibrator.calibrate(objective, None, target_parameters=['test.sgd_param'])
        
        assert result.objective_value < 1.0, "SGD should make progress"
        # SGD might not converge as well as Adam, but should improve
        assert result.convergence_history[-1] < result.convergence_history[0], "Should reduce loss"

    def test_lbfgs_optimizer_returns_coherent_accepted_point(self):
        """L-BFGS closure and result evidence must bind to the accepted point."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.lbfgs_param',
            value=5.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf")),
        ))
        evaluations = 0

        def objective(params, obs):
            nonlocal evaluations
            evaluations += 1
            return (params['test.lbfgs_param'] - 2.0) ** 2

        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(
                method=CalibrationMethod.LBFGS,
                learning_rate=1.0,
                max_iterations=1,
                max_eval=8,
                history_size=10,
                line_search_fn='strong_wolfe',
                verbose=False,
            ),
        )

        result = calibrator.calibrate(
            objective,
            None,
            target_parameters=['test.lbfgs_param'],
        )

        accepted = result.optimized_parameters['test.lbfgs_param']
        accepted_loss = (accepted - 2.0) ** 2
        assert evaluations >= 2
        assert result.objective_value == pytest.approx(accepted_loss)
        assert result.convergence_history[-1] == pytest.approx(accepted_loss)

    def test_lbfgs_rejects_projected_bounds(self):
        """PyTorch L-BFGS is unbounded; projection must not masquerade as L-BFGS."""
        reset_global_manager()
        manager = get_global_manager()
        manager.register_parameter(ParameterSpec(
            key='test.lbfgs_bounded',
            value=0.5,
            units='dimensionless',
            bounds=[0.0, 1.0],
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32",
        ))
        calibrator = GradientCalibrator(
            manager,
            GradientCalibrationConfig(
                method=CalibrationMethod.LBFGS,
                max_iterations=1,
                verbose=False,
            ),
        )

        with pytest.raises(ValueError, match='does not support bounded parameters'):
            calibrator.calibrate(
                lambda params, obs: (params['test.lbfgs_bounded'] - 0.75) ** 2,
                None,
                target_parameters=['test.lbfgs_bounded'],
            )


class TestGradientFlow:
    """Test that gradients flow correctly through the system"""
    
    def test_parameter_gradients_computed(self):
        """Test that gradients are actually computed"""
        reset_global_manager()
        manager = get_global_manager()
        
        spec = ParameterSpec(
            key='test.grad_check',
            value=2.0,
            units='dimensionless',
            requires_grad=True,
            category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        manager.register_parameter(spec)
        
        # Get the nn.Parameter
        trainable = manager.get_trainable_parameters()
        param = trainable['test.grad_check']
        
        # Forward pass
        loss = param ** 2
        
        # Backward pass
        loss.backward()
        
        # Check gradient
        assert param.grad is not None, "Gradient should be computed"
        assert param.grad.item() != 0, "Gradient should be non-zero"
        
        # For f(x) = x^2, df/dx = 2x
        # At x=2, gradient should be 4
        expected_grad = 2.0 * 2.0
        assert abs(param.grad.item() - expected_grad) < 1e-5, f"Gradient should be {expected_grad}"
    
    def test_multi_parameter_gradients(self):
        """Test gradients for multiple parameters"""
        reset_global_manager()
        manager = get_global_manager()
        
        # Register two trainable parameters
        for i in range(2):
            spec = ParameterSpec(
                key=f'test.param{i}',
                value=float(i + 1),
                units='dimensionless',
                requires_grad=True,
                category=ParameterCategory.CALIBRATION, dtype="float32", bounds=(-float("inf"), float("inf"))
            )
            manager.register_parameter(spec)
        
        trainable = manager.get_trainable_parameters()
        assert len(trainable) == 2, "Should have 2 trainable parameters"
        
        # Forward pass: loss = param0^2 + param1^2
        loss = trainable['test.param0'] ** 2 + trainable['test.param1'] ** 2
        
        # Backward pass
        loss.backward()
        
        # Check both gradients
        for key, param in trainable.items():
            assert param.grad is not None, f"{key} should have gradient"
            assert param.grad.item() != 0, f"{key} gradient should be non-zero"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

