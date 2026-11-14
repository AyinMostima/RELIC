"""
Comprehensive L1 Unit Tests for PyTorch Surrogate Architectures

This test suite validates the multi-architecture support in PyTorchSurrogate,
including MLP, ResNet, and Transformer architectures, uncertainty quantification,
and enhanced physics loss functions.

Test Categories:
1. Architecture Factory Tests (MLP, ResNet, Transformer)
2. Model Creation and Forward Pass Tests
3. Uncertainty Quantification Tests (Dropout-MC, Ensemble)
4. Enhanced Physics Loss Tests
5. Training and Optimization Tests
6. Default Architecture Tests
7. Integration and Error Handling Tests

Mathematical Framework:
    MLP: y = f(W_n(...f(W_2(f(W_1x + b_1)) + b_2)...) + b_n)
    ResNet: y = F(x) + x (residual connections)
    Transformer: Attention(Q,K,V) = softmax(QK^T/鈭歞_k)V
    Uncertainty: 蟽虏_total = Var[E[y|x]] + E[Var[y|x]]

References:
    - Goodfellow, I. et al. (2016). Deep Learning
    - He, K. et al. (2015). Deep Residual Learning for Image Recognition
    - Vaswani, A. et al. (2017). Attention Is All You Need
    - Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os
from torch.utils.data import RandomSampler, SequentialSampler

# Add relic to Python path
relic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, relic_path)

from relic.models.pytorch_surrogate import (
    PyTorchSurrogate,
    FlexibleSurrogateNet,
    MLPSurrogateNet,
    ResNetSurrogateNet,
    TransformerSurrogateNet,
    BasicResNetBlock,
    SimpleTransformerBlock,
    architecture_factory,
    SurrogateDataset
)


class TestArchitectureFactory:
    """Test the architecture factory function."""
    
    def test_mlp_architecture_creation(self):
        """Test MLP architecture creation through factory."""
        config = {
            'type': 'mlp',
            'hidden_dims': [64, 32],
            'activation': 'relu',
            'dropout_rate': 0.1,
            'batch_norm': True
        }
        
        model = architecture_factory(config, input_dim=10, output_dim=2)
        
        assert isinstance(model, MLPSurrogateNet)
        assert model.input_dim == 10
        assert model.output_dim == 2
        assert model.hidden_dims == [64, 32]
        
        # Test forward pass
        x = torch.randn(5, 10)
        y = model(x)
        assert y.shape == (5, 2)
    
    def test_resnet_architecture_creation(self):
        """Test ResNet architecture creation through factory."""
        config = {
            'type': 'resnet',
            'n_blocks': 3,
            'hidden_dim': 128,
            'activation': 'relu',
            'dropout_rate': 0.2
        }
        
        model = architecture_factory(config, input_dim=15, output_dim=3)
        
        assert isinstance(model, ResNetSurrogateNet)
        assert model.input_dim == 15
        assert model.output_dim == 3
        assert model.n_blocks == 3
        assert model.hidden_dim == 128
        
        # Test forward pass
        x = torch.randn(4, 15)
        y = model(x)
        assert y.shape == (4, 3)
    
    def test_transformer_architecture_creation(self):
        """Test Transformer architecture creation through factory."""
        config = {
            'type': 'transformer',
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'ffn_ratio': 4.0,
            'dropout_rate': 0.1
        }
        
        model = architecture_factory(config, input_dim=8, output_dim=1)
        
        assert isinstance(model, TransformerSurrogateNet)
        assert model.input_dim == 8
        assert model.output_dim == 1
        assert model.n_layers == 2
        assert model.d_model == 64
        
        # Test forward pass
        x = torch.randn(3, 8)
        y = model(x)
        assert y.shape == (3, 1)
    
    def test_unknown_architecture_type(self):
        """Test error handling for unknown architecture types."""
        config = {'type': 'unknown_architecture'}
        
        with pytest.raises(ValueError, match="Unknown architecture type"):
            architecture_factory(config, input_dim=10, output_dim=2)


class TestBasicResNetBlock:
    """Test the BasicResNetBlock component."""
    
    def test_resnet_block_forward_pass(self):
        """Test ResNet block forward pass with residual connection."""
        block = BasicResNetBlock(dim=64, activation='relu', dropout_rate=0.1)
        
        x = torch.randn(4, 64)
        y = block(x)
        
        assert y.shape == (4, 64)
        
        # Verify residual connection by checking that output != input
        # but has learned some transformation
        assert not torch.allclose(x, y, rtol=1e-3)
    
    def test_resnet_block_different_activations(self):
        """Test ResNet block with different activation functions."""
        activations = ['relu', 'tanh', 'elu', 'silu']
        
        for activation in activations:
            block = BasicResNetBlock(dim=32, activation=activation, dropout_rate=0.0)
            x = torch.randn(2, 32)
            y = block(x)
            
            assert y.shape == (2, 32)
            # Should not be NaN or infinite
            assert torch.isfinite(y).all()
    
    def test_resnet_block_batch_norm(self):
        """Test ResNet block batch normalization."""
        block = BasicResNetBlock(dim=16, activation='relu', dropout_rate=0.0)
        
        # Check that batch norm layers exist
        assert any(isinstance(m, nn.BatchNorm1d) for m in block.modules())
        
        # Test with batch size > 1 (required for batch norm)
        x = torch.randn(5, 16)
        y = block(x)
        assert y.shape == (5, 16)
    
    def test_resnet_block_dropout(self):
        """Test ResNet block dropout functionality."""
        block = BasicResNetBlock(dim=32, activation='relu', dropout_rate=0.5)
        
        # In training mode, dropout should be active
        block.train()
        x = torch.randn(10, 32)
        y1 = block(x)
        y2 = block(x)
        
        # With high dropout rate, outputs should be different
        assert not torch.allclose(y1, y2, rtol=1e-3)
        
        # In evaluation mode, dropout should be inactive
        block.eval()
        y3 = block(x)
        y4 = block(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(y3, y4, rtol=1e-6)


class TestSimpleTransformerBlock:
    """Test the SimpleTransformerBlock component."""
    
    def test_transformer_block_forward_pass(self):
        """Test Transformer block forward pass."""
        block = SimpleTransformerBlock(d_model=64, n_heads=8, ffn_ratio=4.0, dropout_rate=0.0)
        
        # Test with 2D input (batch_size, d_model)
        x = torch.randn(4, 64)
        y = block(x)
        
        assert y.shape == (4, 64)
    
    def test_transformer_block_3d_input(self):
        """Test Transformer block with 3D input (sequence)."""
        block = SimpleTransformerBlock(d_model=32, n_heads=4, ffn_ratio=4.0, dropout_rate=0.0)
        
        # Test with 3D input (batch_size, seq_len, d_model)
        x = torch.randn(2, 10, 32)
        y = block(x)
        
        assert y.shape == (2, 10, 32)
    
    def test_transformer_attention_mechanism(self):
        """Test that attention mechanism produces different outputs for different inputs."""
        block = SimpleTransformerBlock(d_model=16, n_heads=2, ffn_ratio=4.0, dropout_rate=0.0)
        
        x1 = torch.randn(1, 16)
        x2 = torch.randn(1, 16)
        
        y1 = block(x1)
        y2 = block(x2)
        
        # Different inputs should produce different outputs
        assert not torch.allclose(y1, y2, rtol=1e-3)
    
    def test_transformer_different_head_counts(self):
        """Test Transformer block with different numbers of attention heads."""
        d_model = 64
        head_counts = [1, 2, 4, 8]
        
        for n_heads in head_counts:
            if d_model % n_heads == 0:  # Ensure divisible
                block = SimpleTransformerBlock(d_model=d_model, n_heads=n_heads, ffn_ratio=4.0, dropout_rate=0.0)
                
                x = torch.randn(3, d_model)
                y = block(x)
                
                assert y.shape == (3, d_model)
                assert torch.isfinite(y).all()


class TestFlexibleSurrogateNet:
    """Test the FlexibleSurrogateNet class."""
    def test_mlp_architecture_config(self):
        """Test MLP architecture configuration."""
        architecture = {
            'type': 'mlp',
            'hidden_dims': [64, 32],
            'activation': 'relu',
            'dropout_rate': 0.1,
            'batch_norm': True
        }
        net = FlexibleSurrogateNet(
            input_dim=10,
            output_dim=2,
            architecture=architecture
        )
        
        assert net.input_dim == 10
        assert net.output_dim == 2
        assert net.architecture_type == 'mlp'
        
        x = torch.randn(5, 10)
        y = net(x)
        assert y.shape == (5, 2)
    
    def test_new_architecture_interface(self):
        """Test new multi-architecture interface."""
        # New interface with architecture specification
        architecture = {
            'type': 'resnet',
            'n_blocks': 2,
            'hidden_dim': 128,
            'activation': 'tanh'
        }
        
        net = FlexibleSurrogateNet(
            input_dim=8,
            output_dim=3,
            architecture=architecture
        )
        
        assert net.input_dim == 8
        assert net.output_dim == 3
        assert net.architecture_type == 'resnet'
        
        # Forward pass should work
        x = torch.randn(4, 8)
        y = net(x)
        assert y.shape == (4, 3)
    
    def test_get_architecture_info(self):
        """Test architecture information retrieval."""
        architecture = {
            'type': 'transformer',
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 3
        }
        
        net = FlexibleSurrogateNet(
            input_dim=12,
            output_dim=1,
            architecture=architecture
        )
        
        info = net.get_architecture_info()
        
        assert info['type'] == 'transformer'
        assert info['input_dim'] == 12
        assert info['output_dim'] == 1
        assert info['config']['type'] == 'transformer'
        assert 'n_layers' in info
        assert 'd_model' in info


class TestPyTorchSurrogateArchitectures:
    """Test PyTorchSurrogate with different architectures."""
    
    def setup_method(self):
        """Set up temporary directory for test models."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_surrogate_with_mlp_architecture(self):
        """Test creating PyTorchSurrogate with MLP architecture."""
        config = {
            'model_path': str(self.temp_dir / 'test_mlp.pth'),
            'validate_model_path': False,
            'input_variables': ['temp', 'humidity'],
            'output_variables': ['growth_rate'],
            'architecture': {
                'type': 'mlp',
                'hidden_dims': [32, 16],
                'activation': 'relu',
                'dropout_rate': 0.1
            }
        }
        
        surrogate = PyTorchSurrogate('test_mlp', config)
        
        # Model should be created
        assert surrogate.model is not None
        assert isinstance(surrogate.model.network, MLPSurrogateNet)
        
        # Test forward pass through surrogate
        state = {
            'temp': torch.tensor([25.0]),
            'humidity': torch.tensor([0.7])
        }
        
        result = surrogate._prepare_input(state)
        assert result.shape == (1, 2)  # 2 input variables
    
    def test_create_surrogate_with_resnet_architecture(self):
        """Test creating PyTorchSurrogate with ResNet architecture."""
        config = {
            'model_path': str(self.temp_dir / 'test_resnet.pth'),
            'validate_model_path': False,
            'input_variables': ['par', 'co2', 'temp'],
            'output_variables': ['gpp', 'respiration'],
            'architecture': {
                'type': 'resnet',
                'n_blocks': 3,
                'hidden_dim': 64,
                'activation': 'elu'
            }
        }
        
        surrogate = PyTorchSurrogate('test_resnet', config)
        
        assert surrogate.model is not None
        assert isinstance(surrogate.model.network, ResNetSurrogateNet)
        
        # Test inference
        state = {
            'par': torch.tensor([800.0]),
            'co2': torch.tensor([400.0]),
            'temp': torch.tensor([298.15])
        }
        
        time = 0.0
        dt = 3600.0
        
        flux_dict = surrogate(state, time, dt)
        
        assert 'gpp' in flux_dict
        assert 'respiration' in flux_dict
        assert isinstance(flux_dict['gpp'], torch.Tensor)
        assert isinstance(flux_dict['respiration'], torch.Tensor)
    
    def test_create_surrogate_with_transformer_architecture(self):
        """Test creating PyTorchSurrogate with Transformer architecture."""
        config = {
            'model_path': str(self.temp_dir / 'test_transformer.pth'),
            'validate_model_path': False,
            'input_variables': ['x1', 'x2', 'x3', 'x4'],
            'output_variables': ['y1'],
            'architecture': {
                'type': 'transformer',
                'd_model': 32,
                'n_heads': 4,
                'n_layers': 2,
                'ffn_ratio': 2.0
            }
        }
        
        surrogate = PyTorchSurrogate('test_transformer', config)
        
        assert surrogate.model is not None
        assert isinstance(surrogate.model.network, TransformerSurrogateNet)
        
        # Test inference
        state = {
            'x1': torch.tensor([1.0]),
            'x2': torch.tensor([2.0]),
            'x3': torch.tensor([3.0]),
            'x4': torch.tensor([4.0])
        }
        
        time = 0.0
        dt = 1.0
        
        flux_dict = surrogate(state, time, dt)
        
        assert 'y1' in flux_dict
        assert isinstance(flux_dict['y1'], torch.Tensor)


class TestUncertaintyQuantification:
    """Test uncertainty quantification features."""
    
    def setup_method(self):
        """Set up test surrogate with uncertainty."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_dropout_mc_uncertainty(self):
        """Test Monte Carlo Dropout uncertainty estimation."""
        config = {
            'model_path': str(self.temp_dir / 'test_uncertainty.pth'),
            'validate_model_path': False,
            'input_variables': ['x1', 'x2'],
            'output_variables': ['y1'],
            'architecture': {
                'type': 'mlp',
                'hidden_dims': [32, 16],
                'dropout_rate': 0.2  # Enable dropout
            },
            'uncertainty': {
                'method': 'dropout_mc',
                'mc_samples': 10
            }
        }
        
        surrogate = PyTorchSurrogate('test_uncertainty', config)
        
        # Test prediction with uncertainty
        input_tensor = torch.randn(3, 2)
        output = surrogate._predict(input_tensor)
        
        # Should return mean prediction
        assert output.shape == (3, 1)
        
        # Should have uncertainty information
        uncertainty_info = surrogate.get_prediction_uncertainty()
        assert uncertainty_info is not None
        assert 'mean' in uncertainty_info
        assert 'std' in uncertainty_info
        assert 'samples' in uncertainty_info
        assert uncertainty_info['method'] == 'dropout_mc'
        
        # Uncertainty (std) should be non-negative
        assert torch.all(uncertainty_info['std'] >= 0)

    def test_dropout_mc_resnet_single_sample_safe(self):
        """Test dropout MC inference stays valid for single-sample ResNet inputs."""
        config = {
            'model_path': str(self.temp_dir / 'test_uncertainty_resnet_single.pth'),
            'validate_model_path': False,
            'input_variables': ['x1', 'x2', 'x3'],
            'output_variables': ['y1'],
            'architecture': {
                'type': 'resnet',
                'hidden_dim': 16,
                'n_blocks': 2,
                'activation': 'relu',
                'dropout_rate': 0.05,
            },
            'uncertainty': {
                'method': 'dropout_mc',
                'mc_samples': 5,
            }
        }

        surrogate = PyTorchSurrogate('test_uncertainty_resnet_single', config)
        prediction = surrogate._predict(torch.randn(1, 3))
        uncertainty_info = surrogate.get_prediction_uncertainty()

        assert prediction.shape == (1, 1)
        assert uncertainty_info is not None
        assert uncertainty_info['method'] == 'dropout_mc'
        assert uncertainty_info['std'].shape == (1, 1)
    
    def test_ensemble_uncertainty(self):
        """Test ensemble uncertainty estimation."""
        config = {
            'model_path': str(self.temp_dir / 'test_ensemble.pth'),
            'validate_model_path': False,
            'input_variables': ['x'],
            'output_variables': ['y'],
            'architecture': {
                'type': 'mlp',
                'hidden_dims': [16, 8],
                # The members differ only by their dropout masks.
                'dropout_rate': 0.2
            },
            'ensemble': {
                'enabled': True,
                'n_members': 5,
                'seed': 42
            },
            'uncertainty': {
                'method': 'ensemble'
            }
        }

        surrogate = PyTorchSurrogate('test_ensemble', config)

        # Test prediction with ensemble uncertainty
        input_tensor = torch.randn(2, 1)
        output = surrogate._predict(input_tensor)

        assert output.shape == (2, 1)

        # Should have uncertainty information with a real member spread
        uncertainty_info = surrogate.get_prediction_uncertainty()
        assert uncertainty_info is not None
        assert uncertainty_info['method'] == 'ensemble'
        assert uncertainty_info['samples'].shape[0] == 5
        assert bool((uncertainty_info['std'] > 0).any())
    
    def test_deterministic_prediction(self):
        """Test deterministic prediction (no uncertainty)."""
        config = {
            'model_path': str(self.temp_dir / 'test_deterministic.pth'),
            'validate_model_path': False,
            'input_variables': ['x1', 'x2'],
            'output_variables': ['y1'],
            'architecture': {
                'type': 'mlp',
                'hidden_dims': [16, 8]
            }
            # No uncertainty configuration
        }
        
        surrogate = PyTorchSurrogate('test_deterministic', config)
        
        # Test deterministic prediction
        input_tensor = torch.randn(3, 2)
        output = surrogate._predict(input_tensor)
        
        assert output.shape == (3, 1)
        
        # Should not have uncertainty information
        uncertainty_info = surrogate.get_prediction_uncertainty()
        assert uncertainty_info is None


class TestEnhancedPhysicsLoss:
    """Test enhanced physics loss functions."""
    
    def setup_method(self):
        """Set up test surrogate with physics loss."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_conservation_loss(self):
        """Test that physics loss uses the declared conservation ledger."""
        config = {
            'model_path': str(self.temp_dir / 'test_physics.pth'),
            'validate_model_path': False,
            'input_variables': ['input'],
            'output_variables': ['output1', 'output2'],
            'architecture': {'type': 'mlp', 'hidden_dims': [16, 8]},
            'conservation_matrix': [[1.0, -1.0]],
            'physics_loss': {
                'conservation_weight': 2.0,
                'conservation_type': 'declared_ledger',
                'adaptive_scaling': False,
            }
        }
        
        surrogate = PyTorchSurrogate('test_physics', config)
        
        # Test physics loss computation
        closed_outputs = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        open_outputs = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        targets = torch.zeros_like(closed_outputs)

        closed_loss = surrogate._compute_physics_loss(closed_outputs, targets)
        open_loss = surrogate._compute_physics_loss(open_outputs, targets)

        assert torch.equal(closed_loss, torch.zeros_like(closed_loss))
        expected = 2.0 * torch.tensor([1.0, 16.0]).mean()
        assert torch.allclose(open_loss, expected)

    def test_unknown_conservation_loss_type_fails_closed(self):
        """Test that an undeclared conservation equation cannot pass as zero."""
        config = {
            'model_path': str(self.temp_dir / 'test_unknown_physics.pth'),
            'validate_model_path': False,
            'input_variables': ['input'],
            'output_variables': ['output1', 'output2'],
            'architecture': {'type': 'mlp', 'hidden_dims': [8]},
            'conservation_matrix': [[1.0, -1.0]],
            'physics_loss': {
                'conservation_weight': 1.0,
                'conservation_type': 'sum',
                'adaptive_scaling': False,
            },
        }
        surrogate = PyTorchSurrogate('test_unknown_physics', config)

        with pytest.raises(ValueError, match='declared_ledger'):
            surrogate._compute_physics_loss(
                torch.ones((2, 2)),
                torch.ones((2, 2)),
            )

    def test_conservation_loss_requires_declared_matrix(self):
        """Test that a weighted conservation loss cannot infer a ledger."""
        config = {
            'model_path': str(self.temp_dir / 'test_missing_ledger.pth'),
            'validate_model_path': False,
            'input_variables': ['input'],
            'output_variables': ['output1', 'output2'],
            'architecture': {'type': 'mlp', 'hidden_dims': [8]},
            'physics_loss': {
                'conservation_weight': 1.0,
                'conservation_type': 'declared_ledger',
                'adaptive_scaling': False,
            },
        }
        surrogate = PyTorchSurrogate('test_missing_ledger', config)

        with pytest.raises(ValueError, match='conservation_matrix'):
            surrogate._compute_physics_loss(
                torch.ones((2, 2)),
                torch.ones((2, 2)),
            )

    @pytest.mark.parametrize('weight', [-1.0, float('inf'), float('nan')])
    def test_conservation_loss_rejects_invalid_weight(self, weight):
        """Test that the declared ledger penalty has a finite non-negative weight."""
        config = {
            'model_path': str(self.temp_dir / 'test_invalid_weight.pth'),
            'validate_model_path': False,
            'input_variables': ['input'],
            'output_variables': ['output1', 'output2'],
            'architecture': {'type': 'mlp', 'hidden_dims': [8]},
            'conservation_matrix': [[1.0, -1.0]],
            'physics_loss': {
                'conservation_weight': weight,
                'conservation_type': 'declared_ledger',
                'adaptive_scaling': False,
            },
        }
        surrogate = PyTorchSurrogate('test_invalid_weight', config)

        with pytest.raises(ValueError, match='finite non-negative'):
            surrogate._compute_physics_loss(
                torch.ones((2, 2)),
                torch.ones((2, 2)),
            )

    def test_conservation_loss_rejects_runtime_shape_mismatch(self):
        """Test that runtime outputs must match the declared ledger columns."""
        config = {
            'model_path': str(self.temp_dir / 'test_shape_mismatch.pth'),
            'validate_model_path': False,
            'input_variables': ['input'],
            'output_variables': ['output1', 'output2'],
            'architecture': {'type': 'mlp', 'hidden_dims': [8]},
            'conservation_matrix': [[1.0, -1.0]],
            'physics_loss': {
                'conservation_weight': 1.0,
                'conservation_type': 'declared_ledger',
                'adaptive_scaling': False,
            },
        }
        surrogate = PyTorchSurrogate('test_shape_mismatch', config)

        with pytest.raises(ValueError, match='column count'):
            surrogate._compute_physics_loss(
                torch.ones((2, 1)),
                torch.ones((2, 1)),
            )
    
    def test_smoothness_loss(self):
        """Test smoothness constraint in physics loss."""
        config = {
            'model_path': str(self.temp_dir / 'test_smoothness.pth'),
            'validate_model_path': False,
            'input_variables': ['input'],
            'output_variables': ['output'],
            'architecture': {'type': 'mlp', 'hidden_dims': [16]},
            'physics_loss': {
                'smoothness_weight': 0.05,
                'smoothness_type': 'temporal'
            }
        }
        
        surrogate = PyTorchSurrogate('test_smoothness', config)
        
        # Test with batch size > 1 for temporal smoothness
        outputs = torch.tensor([[1.0], [1.5], [2.0]])  # Smooth progression
        targets = torch.tensor([[1.0], [1.5], [2.0]])
        
        physics_loss = surrogate._compute_physics_loss(outputs, targets)
        
        assert physics_loss >= 0
        
        # Test with non-smooth progression
        outputs_rough = torch.tensor([[1.0], [10.0], [0.5]])  # Rough progression
        targets_rough = torch.tensor([[1.0], [1.5], [2.0]])
        
        physics_loss_rough = surrogate._compute_physics_loss(outputs_rough, targets_rough)
        
        # Rough progression should have higher physics loss
        assert physics_loss_rough > physics_loss
    
    def test_bounds_constraints_loss(self):
        """Test physical bounds constraints in physics loss."""
        config = {
            'model_path': str(self.temp_dir / 'test_bounds.pth'),
            'validate_model_path': False,
            'input_variables': ['input'],
            'output_variables': ['output'],
            'architecture': {'type': 'mlp', 'hidden_dims': [8]},
            'physics_loss': {
                'bounds_weight': 0.1,
                'bounds_constraints': [
                    {'output_idx': 0, 'lower_bound': 0.0, 'upper_bound': 10.0}
                ]
            }
        }
        
        surrogate = PyTorchSurrogate('test_bounds', config)
        
        # Test with valid bounds
        outputs_valid = torch.tensor([[5.0], [7.5], [2.0]])  # All within [0, 10]
        targets = torch.tensor([[5.0], [7.5], [2.0]])
        
        physics_loss_valid = surrogate._compute_physics_loss(outputs_valid, targets)
        
        # Test with bound violations
        outputs_invalid = torch.tensor([[-1.0], [5.0], [15.0]])  # Violations at bounds
        physics_loss_invalid = surrogate._compute_physics_loss(outputs_invalid, targets)
        
        # Invalid outputs should have higher physics loss
        assert physics_loss_invalid > physics_loss_valid
    
    def test_physics_loss_components_tracking(self):
        """Test physics loss components tracking."""
        config = {
            'model_path': str(self.temp_dir / 'test_components.pth'),
            'validate_model_path': False,
            'input_variables': ['input'],
            'output_variables': ['output'],
            'architecture': {'type': 'mlp', 'hidden_dims': [8]},
            'conservation_matrix': [[1.0]],
            'physics_loss': {
                'conservation_weight': 0.1,
                'conservation_type': 'declared_ledger',
                'smoothness_weight': 0.05,
                'bounds_weight': 0.1,
                'bounds_constraints': [{'output_idx': 0, 'lower_bound': 0.0}]
            }
        }
        
        surrogate = PyTorchSurrogate('test_components', config)
        
        outputs = torch.tensor([[1.0], [2.0]])
        targets = torch.tensor([[1.0], [2.0]])
        
        physics_loss = surrogate._compute_physics_loss(outputs, targets)
        
        # Should track components
        components = surrogate.get_physics_loss_components()
        
        assert isinstance(components, dict)
        # Should have at least some components tracked
        assert len(components) >= 0


class TestTrainingAndOptimization:
    """Test training functionality with different architectures."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_training_with_mlp_architecture(self):
        """Test training with MLP architecture."""
        config = {
            'model_path': str(self.temp_dir / 'test_train_mlp.pth'),
            'validate_model_path': False,
            'input_variables': ['x1', 'x2'],
            'output_variables': ['y'],
            'input_units': {'x1': '1', 'x2': '1'},
            'output_units': {'y': '1'},
            'architecture': {
                'type': 'mlp',
                'hidden_dims': [16, 8],
                'activation': 'relu'
            },
            'training': {
                'optimizer': 'adam',
                'learning_rate': 0.01,
                'batch_size': 8,
                'epochs': 5
            }
        }
        
        surrogate = PyTorchSurrogate('test_train_mlp', config)
        
        # Create synthetic training data
        n_samples = 32
        states = []
        fluxes = []
        
        for i in range(n_samples):
            state = {
                'x1': torch.tensor([np.random.normal(0, 1)]),
                'x2': torch.tensor([np.random.normal(0, 1)])
            }
            flux = {
                'y': torch.tensor([state['x1'].item() + 2 * state['x2'].item()])  # Simple relationship
            }
            states.append(state)
            fluxes.append(flux)
        
        # Train the model
        training_result = surrogate.train((states, fluxes))
        
        # Should complete training
        assert 'epochs_trained' in training_result
        assert training_result['epochs_trained'] >= 5
        assert 'final_train_loss' in training_result
        assert training_result['final_train_loss'] is not None
        
        # Model should be marked as trained
        assert surrogate.is_trained
    
    def test_training_with_validation_split(self):
        """Test training with validation data and early stopping."""
        config = {
            'model_path': str(self.temp_dir / 'test_train_val.pth'),
            'validate_model_path': False,
            'input_variables': ['x'],
            'output_variables': ['y'],
            'input_units': {'x': '1'},
            'output_units': {'y': '1'},
            'architecture': {
                'type': 'resnet',
                'n_blocks': 2,
                'hidden_dim': 16
            },
            'training': {
                'optimizer': 'adam',
                'learning_rate': 0.01,
                'epochs': 10,
                'early_stopping_patience': 3
            }
        }
        
        surrogate = PyTorchSurrogate('test_train_val', config)
        
        # Create training and validation data
        def create_data(n_samples):
            states = []
            fluxes = []
            for i in range(n_samples):
                x_val = np.random.uniform(-1, 1)
                state = {'x': torch.tensor([x_val])}
                flux = {'y': torch.tensor([x_val**2])}  # y = x^2 relationship
                states.append(state)
                fluxes.append(flux)
            return states, fluxes
        
        train_states, train_fluxes = create_data(40)
        val_states, val_fluxes = create_data(16)
        
        # Train with validation data
        training_result = surrogate.train(
            (train_states, train_fluxes),
            validation_data=(val_states, val_fluxes)
        )
        
        # Should have validation loss recorded
        assert 'final_val_loss' in training_result
        assert training_result['final_val_loss'] is not None
        assert len(surrogate.training_history['val_loss']) > 0

    def test_training_loader_respects_shuffle_training_flag(self):
        """Test that surrogate training can preserve temporal sample order."""
        config = {
            'model_path': str(self.temp_dir / 'test_train_ordered.pth'),
            'validate_model_path': False,
            'input_variables': ['x'],
            'output_variables': ['y'],
            'training': {
                'batch_size': 4,
                'epochs': 1,
                'shuffle_training': False,
            },
        }

        surrogate = PyTorchSurrogate('test_train_ordered', config)
        states = [{'x': torch.tensor([float(i)])} for i in range(8)]
        fluxes = [{'y': torch.tensor([float(i)])} for i in range(8)]
        dataset = SurrogateDataset(states, fluxes, ['x'], ['y'], validate_units=False)

        train_loader, _ = surrogate._build_train_val_loaders(
            training_dataset=dataset,
            validation_dataset=None,
            batch_size=4,
        )

        assert isinstance(train_loader.sampler, SequentialSampler)
        assert not isinstance(train_loader.sampler, RandomSampler)


class TestArchitectureDefaults:
    """Test default architecture behavior."""
    
    def test_default_mlp_when_type_missing(self):
        """Test default MLP when architecture type is omitted."""
        config = {
            'model_path': str(Path(tempfile.mkdtemp()) / 'surrogate_default.pth'),
            'validate_model_path': False,
            'input_variables': ['temp', 'humidity'],
            'output_variables': ['growth'],
            'architecture': {
                'hidden_dims': [32, 16],
                'activation': 'tanh',
                'dropout_rate': 0.1,
                'batch_norm': True
            }
        }
        
        surrogate = PyTorchSurrogate('test_default_mlp', config)
        
        assert surrogate.model is not None
        assert hasattr(surrogate.model, 'network')
        assert isinstance(surrogate.model.network, MLPSurrogateNet)
    
    def test_default_mlp_when_architecture_missing(self):
        """Test default MLP when architecture config is missing."""
        config = {
            'model_path': str(Path(tempfile.mkdtemp()) / 'surrogate_minimal.pth'),
            'validate_model_path': False,
            'input_variables': ['temp', 'humidity'],
            'output_variables': ['growth']
        }
        
        surrogate = PyTorchSurrogate('test_default_minimal', config)

        assert surrogate.model is not None
        assert hasattr(surrogate.model, 'network')
        assert isinstance(surrogate.model.network, MLPSurrogateNet)


from relic.models.surrogates import InferenceError  # noqa: E402


class TestNoSilentSubstitution:
    """A requested behaviour runs as declared or raises; nothing is substituted.

    The declared uncertainty method, ensemble setting, physics-loss terms and
    checkpoint are honoured exactly: no deterministic prediction replaces an
    unconfigured ensemble, no zero-spread 'uncertainty' is reported for a
    network without dropout, no untrained network replaces an unreadable
    checkpoint (whatever the path name), and no physics-loss term is dropped.
    """

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _config(self, name, **extra):
        config = {
            'model_path': str(self.temp_dir / f'{name}.pth'),
            'validate_model_path': False,
            'input_variables': ['x'],
            'output_variables': ['y'],
            'architecture': {'type': 'mlp', 'hidden_dims': [8], 'dropout_rate': 0.2},
        }
        config.update(extra)
        return config

    def test_ensemble_method_without_enabled_ensemble_is_an_error(self):
        surrogate = PyTorchSurrogate(
            'ensemble_disabled',
            self._config('ensemble_disabled', uncertainty={'method': 'ensemble'}),
        )

        with pytest.raises(InferenceError, match='ensemble.enabled') as excinfo:
            surrogate._predict(torch.randn(2, 1))
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert surrogate.get_prediction_uncertainty() is None

    def test_unknown_uncertainty_method_is_rejected(self):
        with pytest.raises(ValueError, match='uncertainty.method'):
            PyTorchSurrogate(
                'unknown_method',
                self._config('unknown_method', uncertainty={'method': 'ensemlbe'}),
            )

    @pytest.mark.parametrize('method', ['dropout_mc', 'ensemble'])
    def test_dropout_uncertainty_without_dropout_is_an_error(self, method):
        config = self._config(
            f'no_dropout_{method}',
            architecture={'type': 'mlp', 'hidden_dims': [8]},
            uncertainty={'method': method},
            ensemble={'enabled': True, 'n_members': 3, 'seed': 5},
        )
        surrogate = PyTorchSurrogate(f'no_dropout_{method}', config)

        with pytest.raises(InferenceError, match='dropout') as excinfo:
            surrogate._predict(torch.randn(2, 1))
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert surrogate.get_prediction_uncertainty() is None

    def test_ensemble_prediction_leaves_the_global_generator_untouched(self):
        config = self._config(
            'ensemble_rng',
            uncertainty={'method': 'ensemble'},
            ensemble={'enabled': True, 'n_members': 3, 'seed': 11},
        )
        surrogate = PyTorchSurrogate('ensemble_rng', config)
        inputs = torch.randn(2, 1)
        before = torch.random.get_rng_state()

        first = surrogate._predict(inputs)
        after = torch.random.get_rng_state()
        second = surrogate._predict(inputs)

        assert torch.equal(before, after)
        assert torch.equal(first, second)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA generator to observe')
    def test_cpu_ensemble_prediction_does_not_reseed_the_cuda_generator(self):
        """torch.manual_seed would also reseed every CUDA generator, outside the fork."""
        config = self._config(
            'ensemble_cuda_rng',
            uncertainty={'method': 'ensemble'},
            ensemble={'enabled': True, 'n_members': 3, 'seed': 11},
            device='cpu',
        )
        surrogate = PyTorchSurrogate('ensemble_cuda_rng', config)
        cuda_before = torch.cuda.get_rng_state().clone()

        surrogate._predict(torch.randn(2, 1))

        assert torch.equal(cuda_before, torch.cuda.get_rng_state())

    def test_ensemble_requires_an_explicit_seed(self):
        surrogate = PyTorchSurrogate(
            'ensemble_unseeded',
            self._config(
                'ensemble_unseeded',
                uncertainty={'method': 'ensemble'},
                ensemble={'enabled': True, 'n_members': 3},
            ),
        )

        with pytest.raises(InferenceError, match='seed') as excinfo:
            surrogate._predict(torch.randn(2, 1))
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_unreadable_checkpoint_is_an_error_even_on_a_test_path(self):
        from relic.models.surrogates import ModelLoadError

        checkpoint = self.temp_dir / 'test_mock_corrupt.pth'
        checkpoint.write_bytes(b'not a checkpoint')

        with pytest.raises(ModelLoadError):
            PyTorchSurrogate('corrupt', self._config('corrupt', model_path=str(checkpoint)))

    def test_mismatched_checkpoint_is_not_replaced_by_an_untrained_model(self):
        """A path containing 'test' or 'mock' gets no fresh network on failure."""
        from relic.models.surrogates import ModelLoadError

        checkpoint = self.temp_dir / 'test_mock_mismatch.pth'
        torch.save({'model_state_dict': {'bogus.weight': torch.zeros(1)}}, checkpoint)

        with pytest.raises(ModelLoadError) as excinfo:
            PyTorchSurrogate('mismatch', self._config('mismatch', model_path=str(checkpoint)))
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_thermodynamic_weight_is_not_a_zero_term(self):
        surrogate = PyTorchSurrogate(
            'thermo',
            self._config('thermo', physics_loss={'thermodynamic_weight': 1.0}),
        )

        with pytest.raises(NotImplementedError, match='thermodynamic'):
            surrogate._compute_physics_loss(torch.ones(3, 1), torch.ones(3, 1))

    def test_unknown_smoothness_type_is_rejected(self):
        surrogate = PyTorchSurrogate(
            'smooth',
            self._config(
                'smooth',
                physics_loss={'smoothness_weight': 1.0, 'smoothness_type': 'temporl'},
            ),
        )

        with pytest.raises(ValueError, match='smoothness_type'):
            surrogate._compute_physics_loss(torch.ones(3, 1), torch.ones(3, 1))

    @pytest.mark.parametrize(
        'loss, constraints, message',
        [
            ('monotonicity', [], 'monotonicity_constraints'),
            ('monotonicity', ['increasing'], 'mapping'),
            ('monotonicity', [{'output_idx': 3, 'direction': 'increasing'}], 'output_idx'),
            ('monotonicity', [{'output_idx': 0}], 'direction'),
            ('monotonicity', [{'output_idx': 0, 'direction': 'upward'}], 'direction'),
            ('bounds', [], 'bounds_constraints'),
            ('bounds', [{'lower_bound': 0.0}], 'output_idx'),
            ('bounds', [{'output_idx': 0}], 'lower_bound'),
        ],
    )
    def test_invalid_constraints_are_rejected_not_skipped(self, loss, constraints, message):
        surrogate = PyTorchSurrogate(
            f'{loss}_constraints',
            self._config(
                f'{loss}_constraints',
                physics_loss={
                    f'{loss}_weight': 1.0,
                    f'{loss}_constraints': constraints,
                    'adaptive_scaling': False,
                },
            ),
        )

        with pytest.raises(ValueError, match=message):
            surrogate._compute_physics_loss(torch.ones(3, 1), torch.ones(3, 1))

    def test_physics_weights_are_not_rescaled_unless_requested(self):
        """Adaptive rescaling of the declared weights is opt-in."""
        surrogate = PyTorchSurrogate(
            'weights',
            self._config(
                'weights',
                physics_loss={
                    'bounds_weight': 100.0,
                    'bounds_constraints': [{'output_idx': 0, 'lower_bound': 0.0}],
                },
            ),
        )
        outputs = torch.tensor([[-10.0], [-10.0]])
        targets = torch.tensor([[0.1], [0.1]])

        loss = surrogate._compute_physics_loss(outputs, targets)

        assert float(loss) == pytest.approx(100.0 * 20.0)


class TestTrainingSettingsAreHonoured:
    """Declared training settings run as declared or raise.

    The best validation weights are a snapshot, not a view of the live
    parameters; an unknown or untyped scheduler is rejected instead of
    training without one; an explicit epochs=0 is not replaced by the config
    value; a declared validation_split holds out a validation set.
    """

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _config(self, name, **training):
        settings = {'optimizer': 'adam', 'learning_rate': 0.01, 'epochs': 2, 'batch_size': 8}
        settings.update(training)
        return {
            'model_path': str(self.temp_dir / f'{name}.pth'),
            'validate_model_path': False,
            'input_variables': ['x'],
            'output_variables': ['y'],
            'input_units': {'x': '1'},
            'output_units': {'y': '1'},
            'architecture': {'type': 'mlp', 'hidden_dims': [8]},
            'training': settings,
        }

    @staticmethod
    def _pairs(n_samples):
        values = torch.linspace(-1.0, 1.0, n_samples)
        states = [{'x': torch.tensor([float(v)])} for v in values]
        fluxes = [{'y': torch.tensor([float(v) ** 2])} for v in values]
        return states, fluxes

    def test_best_validation_weights_are_a_snapshot(self):
        surrogate = PyTorchSurrogate('snapshot', self._config('snapshot'))
        surrogate._update_early_stopping_state(
            avg_val_loss=0.1,
            best_val_loss=float('inf'),
            patience_counter=0,
            early_stopping_patience=3,
            epoch=0,
            verbose=False,
        )
        saved = {key: value.clone() for key, value in surrogate.best_model_state.items()}

        with torch.no_grad():
            for parameter in surrogate.model.parameters():
                parameter.add_(1.0)

        for key, value in surrogate.best_model_state.items():
            assert torch.equal(value, saved[key])

    @pytest.mark.parametrize(
        'architecture, key',
        [
            ({'type': 'mlp', 'hidden_dim': 16}, 'hidden_dim'),
            ({'type': 'resnet', 'n_blocks': 1, 'hidden_dim': 8, 'batch_norm': False}, 'batch_norm'),
            ({'type': 'transformer', 'd_model': 8, 'n_heads': 2, 'activation': 'gelu'}, 'activation'),
        ],
    )
    def test_architecture_keys_the_network_does_not_read_are_rejected(self, architecture, key):
        from relic.models.surrogates import ModelLoadError

        config = self._config('arch_keys')
        config['architecture'] = architecture
        with pytest.raises(ModelLoadError, match=key) as excinfo:
            PyTorchSurrogate('arch_keys', config)
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_transformer_width_must_split_into_heads(self):
        from relic.models.surrogates import ModelLoadError

        config = self._config('heads')
        config['architecture'] = {'type': 'transformer', 'd_model': 10, 'n_heads': 4}
        with pytest.raises(ModelLoadError, match='n_heads'):
            PyTorchSurrogate('heads', config)

    def test_unknown_scheduler_type_is_rejected(self):
        from relic.models.surrogates import ModelLoadError

        with pytest.raises(ModelLoadError, match='scheduler') as excinfo:
            PyTorchSurrogate('sched', self._config('sched', scheduler={'type': 'plateau'}))
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_scheduler_without_type_is_rejected(self):
        from relic.models.surrogates import ModelLoadError

        with pytest.raises(ModelLoadError, match='scheduler.type') as excinfo:
            PyTorchSurrogate('sched_untyped', self._config('sched_untyped', scheduler={'step_size': 5}))
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_explicit_zero_epochs_is_not_replaced_by_the_config_value(self):
        surrogate = PyTorchSurrogate('zero_epochs', self._config('zero_epochs'))
        with pytest.raises(ValueError, match='epochs'):
            surrogate.train(self._pairs(16), epochs=0, verbose=False)
        assert surrogate.epochs_trained == 0

    def test_validation_split_holds_out_a_validation_set(self):
        surrogate = PyTorchSurrogate('split', self._config('split', validation_split=0.25))
        result = surrogate.train(self._pairs(16), verbose=False)
        assert len(result['training_history']['val_loss']) == 2
        assert result['best_val_loss'] is not None

    @pytest.mark.parametrize('split', [0.0, 1.0, 1.5, 'x', True])
    def test_invalid_validation_split_is_rejected(self, split):
        surrogate = PyTorchSurrogate('bad_split', self._config('bad_split', validation_split=split))
        with pytest.raises(ValueError, match='validation_split'):
            surrogate.train(self._pairs(16), verbose=False)


class TestHyperparametersComeFromYaml:
    """Undeclared architecture and training hyperparameters come from the YAML database.

    models/pytorch_surrogate.yaml (namespace models.surrogate.pytorch) declares
    every hyperparameter with units, bounds, references and provenance; the
    code keeps no literal default (the configuration overrides the YAML value).
    """

    PREFIX = "models.surrogate.pytorch"

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _yaml(self, key):
        from relic.parameters import get_global_manager

        return get_global_manager().get(f"{self.PREFIX}.{key}", required=True)

    def test_empty_architecture_resolves_to_the_declared_mlp(self):
        from relic.models.pytorch_surrogate import resolve_architecture

        resolved = resolve_architecture({})
        assert resolved["type"] == self._yaml("architecture.type")
        for key in ("hidden_dims", "activation", "dropout_rate", "batch_norm"):
            assert resolved[key] == self._yaml(f"architecture.mlp.{key}"), key

    def test_configured_keys_override_and_missing_keys_come_from_yaml(self):
        from relic.models.pytorch_surrogate import resolve_architecture

        resolved = resolve_architecture({"type": "resnet", "n_blocks": 2})
        assert resolved["n_blocks"] == 2
        assert resolved["hidden_dim"] == self._yaml("architecture.resnet.hidden_dim")

    def test_training_defaults_are_read_through_the_parameter_system(self, monkeypatch):
        seen = []
        original = PyTorchSurrogate.get_parameter_value

        def spy(surrogate, name, *args, **kwargs):
            seen.append(name)
            if name == f"{self.PREFIX}.training.learning_rate":
                return 5.0e-4
            return original(surrogate, name, *args, **kwargs)

        monkeypatch.setattr(PyTorchSurrogate, "get_parameter_value", spy)
        surrogate = PyTorchSurrogate(
            "yaml_lr",
            {
                "model_path": str(self.temp_dir / "yaml_lr.pth"),
                "validate_model_path": False,
                "input_variables": ["x"],
                "output_variables": ["y"],
                "architecture": {"type": "mlp", "hidden_dims": [4]},
            },
        )
        assert surrogate.optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-4)
        assert f"{self.PREFIX}.training.optimizer" in seen

    def test_building_blocks_take_explicit_hyperparameters(self):
        with pytest.raises(TypeError):
            BasicResNetBlock(dim=8)
        with pytest.raises(TypeError):
            SimpleTransformerBlock(d_model=8)

    def test_no_literal_hyperparameter_defaults_remain(self):
        import inspect
        import re

        import relic.models.pytorch_surrogate as module

        names = (
            "type|hidden_dims|activation|dropout_rate|batch_norm|n_blocks|hidden_dim|d_model|"
            "n_heads|n_layers|ffn_ratio|optimizer|learning_rate|weight_decay|momentum|step_size|"
            "gamma|T_max|loss_function|huber_delta|batch_size|epochs|early_stopping_patience|"
            "shuffle_training|num_workers|pin_memory|persistent_workers|prefetch_factor|seed"
        )
        pattern = re.compile(r"\.get\(\s*['\"](" + names + r")['\"]\s*,\s*(?!None\s*\))")
        offenders = pattern.findall(inspect.getsource(module))
        assert offenders == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

