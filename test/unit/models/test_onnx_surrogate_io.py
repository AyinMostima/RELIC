"""
Comprehensive L1 Unit Tests for ONNXSurrogate I/O Operations

This test suite provides exhaustive testing of the ONNXSurrogate class,
focusing specifically on ONNX Runtime integration, model loading, inference 
operations, device management, and performance characteristics.

Test Categories:
1. ONNX Model Loading Tests
2. Inference Engine and Runtime Tests  
3. Device consistency Tests (CPU/GPU)
4. Input/Output Shape Validation Tests
5. Performance and Benchmarking Tests
6. Memory Management Tests
7. Error Handling and Recovery Tests
8. ONNX Provider Configuration Tests

Mathematical Framework:
    ONNX Inference: f虃 = M_ONNX(x | 胃, providers)
    where M_ONNX is the ONNX Runtime model, x is input tensor,
    胃 are model parameters, providers are execution providers

Performance Requirements:
    - Model loading: < 5 seconds for typical models
    - Single inference: < 100ms for batch_size=1  
    - Batch inference: Throughput > 100 samples/second
    - Memory usage: < 2x model file size

References:
    - ONNX Runtime Documentation: https://onnxruntime.ai/
    - ONNX Model Zoo: https://github.com/onnx/models
    - Performance Optimization Guide: https://onnxruntime.ai/docs/performance/
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import time
import sys

# Add relic to Python path for testing
relic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, relic_path)

from relic.models.surrogates import ONNXSurrogate, ONNX_AVAILABLE, ModelLoadError, InferenceError
from relic.core.functor import FunctorMetadata

# Optional ONNX Runtime imports for test data generation
try:
    from onnx import helper, TensorProto
    ONNX_TEST_AVAILABLE = True
except ImportError:
    ONNX_TEST_AVAILABLE = False


class TestONNXModelCreationUtils:
    """Utilities for creating test ONNX models."""
    
    @staticmethod
    def create_simple_linear_model(input_size: int, output_size: int) -> bytes:
        """
        Create a simple linear ONNX model for testing.
        
        Model: output = weight @ input + bias
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            
        Returns:
            ONNX model as bytes
        """
        if not ONNX_TEST_AVAILABLE:
            pytest.skip("ONNX not available for test model creation")
        
        # Create input
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, input_size])
        
        # Create output  
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, output_size])
        
        # Create weights and bias as initializers
        weight_data = np.random.randn(input_size, output_size).astype(np.float32)
        bias_data = np.random.randn(output_size).astype(np.float32)
        
        weight_tensor = helper.make_tensor('weight', TensorProto.FLOAT, [input_size, output_size], weight_data.flatten())
        bias_tensor = helper.make_tensor('bias', TensorProto.FLOAT, [output_size], bias_data.flatten())
        
        # Create MatMul node
        matmul_node = helper.make_node('MatMul', ['input', 'weight'], ['matmul_out'])
        
        # Create Add node
        add_node = helper.make_node('Add', ['matmul_out', 'bias'], ['output'])
        
        # Create graph
        graph = helper.make_graph(
            [matmul_node, add_node],
            'simple_linear',
            [input_tensor],
            [output_tensor],
            [weight_tensor, bias_tensor]
        )
        
        # Create model with consistent opset version
        opset_imports = [helper.make_opsetid("", 15)]
        model = helper.make_model(graph, producer_name='test', opset_imports=opset_imports)
        # Set IR version to 10 for consistency with ONNX Runtime 1.22.1
        model.ir_version = 10
        
        return model.SerializeToString()
    
    @staticmethod
    def create_identity_model(size: int) -> bytes:
        """Create an identity ONNX model (output = input)."""
        if not ONNX_TEST_AVAILABLE:
            pytest.skip("ONNX not available for test model creation")
        
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, size])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, size])
        
        identity_node = helper.make_node('Identity', ['input'], ['output'])
        
        graph = helper.make_graph([identity_node], 'identity', [input_tensor], [output_tensor])
        opset_imports = [helper.make_opsetid("", 15)]
        model = helper.make_model(graph, producer_name='test', opset_imports=opset_imports)
        # Set IR version to 10 for consistency
        model.ir_version = 10
        
        return model.SerializeToString()
    
    @staticmethod
    def create_nonlinear_model(input_size: int) -> bytes:
        """Create a nonlinear ONNX model with activation functions."""
        if not ONNX_TEST_AVAILABLE:
            pytest.skip("ONNX not available for test model creation")
        
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, input_size])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 1])
        
        # Create weight tensors
        w1_data = np.random.randn(input_size, 10).astype(np.float32) * 0.1
        w2_data = np.random.randn(10, 1).astype(np.float32) * 0.1
        b1_data = np.zeros(10, dtype=np.float32)
        b2_data = np.zeros(1, dtype=np.float32)
        
        w1_tensor = helper.make_tensor('w1', TensorProto.FLOAT, [input_size, 10], w1_data.flatten())
        w2_tensor = helper.make_tensor('w2', TensorProto.FLOAT, [10, 1], w2_data.flatten())
        b1_tensor = helper.make_tensor('b1', TensorProto.FLOAT, [10], b1_data)
        b2_tensor = helper.make_tensor('b2', TensorProto.FLOAT, [1], b2_data)
        
        # Create computation nodes
        matmul1_node = helper.make_node('MatMul', ['input', 'w1'], ['hidden_raw'])
        add1_node = helper.make_node('Add', ['hidden_raw', 'b1'], ['hidden_biased'])
        relu_node = helper.make_node('Relu', ['hidden_biased'], ['hidden_activated'])
        matmul2_node = helper.make_node('MatMul', ['hidden_activated', 'w2'], ['output_raw'])
        add2_node = helper.make_node('Add', ['output_raw', 'b2'], ['output'])
        
        graph = helper.make_graph(
            [matmul1_node, add1_node, relu_node, matmul2_node, add2_node],
            'nonlinear_model',
            [input_tensor],
            [output_tensor],
            [w1_tensor, w2_tensor, b1_tensor, b2_tensor]
        )
        
        opset_imports = [helper.make_opsetid("", 15)]
        model = helper.make_model(graph, producer_name='test', opset_imports=opset_imports)
        # Set IR version to 10 for consistency with ONNX Runtime 1.22.1
        model.ir_version = 10
        return model.SerializeToString()


@pytest.fixture
def temp_model_file():
    """Create a temporary ONNX model file for testing."""
    if not ONNX_TEST_AVAILABLE:
        pytest.skip("ONNX not available for test model creation")
    
    model_bytes = TestONNXModelCreationUtils.create_simple_linear_model(3, 2)
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        f.write(model_bytes)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture  
def identity_model_file():
    """Create a temporary identity ONNX model file."""
    if not ONNX_TEST_AVAILABLE:
        pytest.skip("ONNX not available for test model creation")
    
    model_bytes = TestONNXModelCreationUtils.create_identity_model(2)
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        f.write(model_bytes)
        temp_path = f.name
    
    yield temp_path
    
    try:
        os.unlink(temp_path)
    except OSError:
        pass


class TestONNXModelLoading:
    """Test ONNX model loading functionality."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_model_loading_success(self, temp_model_file):
        """Test successful ONNX model loading."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['input1', 'input2', 'input3'],
            'output_variables': ['output1', 'output2']
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Check that model loaded successfully
        assert hasattr(surrogate, 'ort_session')
        assert surrogate.ort_session is not None
        assert surrogate.model_loaded is True
        
        # Check input/output specifications
        assert surrogate.input_name is not None
        assert surrogate.output_name is not None
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_model_loading_nonexistent_file(self):
        """Test error handling for nonexistent model files."""
        config = {
            'model_path': '/path/to/nonexistent/model.onnx',
            'input_variables': ['input'],
            'output_variables': ['output'],
            'validate_model_path': False
        }
        
        with pytest.raises(ModelLoadError, match="Failed to load ONNX model"):
            ONNXSurrogate("test_onnx", config)
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available") 
    def test_model_loading_invalid_file(self):
        """Test error handling for invalid ONNX files."""
        # Create a file with invalid content
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"This is not a valid ONNX model")
            invalid_path = f.name
        
        config = {
            'model_path': invalid_path,
            'input_variables': ['input'],
            'output_variables': ['output']
        }
        
        try:
            with pytest.raises(ModelLoadError):
                ONNXSurrogate("test_onnx", config)
        finally:
            os.unlink(invalid_path)
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_provider_configuration(self, temp_model_file):
        """Test ONNX execution provider configuration."""
        # Test CPU provider
        config = {
            'model_path': temp_model_file,
            'input_variables': ['input1', 'input2', 'input3'],
            'output_variables': ['output1', 'output2'],
            'device': 'cpu'
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Check that CPU provider is used
        providers = surrogate.ort_session.get_providers()
        assert 'CPUExecutionProvider' in providers
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_auto_device_selection(self, temp_model_file):
        """Test automatic device selection."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['input1', 'input2', 'input3'],
            'output_variables': ['output1', 'output2'],
            'device': 'auto'
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Should select best available provider
        providers = surrogate.ort_session.get_providers()
        assert len(providers) > 0
        assert isinstance(providers[0], str)


class TestONNXInferenceOperations:
    """Test ONNX model inference operations."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_basic_inference(self, identity_model_file):
        """Test basic model inference with identity model."""
        config = {
            'model_path': identity_model_file,
            'input_variables': ['var1', 'var2'],
            'output_variables': ['out1', 'out2']
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Test inference
        state = {
            'var1': torch.tensor([1.0, 2.0]),
            'var2': torch.tensor([3.0, 4.0])
        }
        
        result = surrogate.forward(state, time=0.0, dt=60.0)
        
        # For identity model, output should equal input
        assert 'out1' in result
        assert 'out2' in result
        assert torch.allclose(
            result['out1'].cpu(),
            state['var1'].to(result['out1'].dtype),
            atol=1e-5
        )
        assert torch.allclose(
            result['out2'].cpu(),
            state['var2'].to(result['out2'].dtype),
            atol=1e-5
        )
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_batch_inference(self, temp_model_file):
        """Test batch inference with multiple samples."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['temp', 'humidity', 'pressure'],
            'output_variables': ['rate1', 'rate2'],
            'conservation_check': False  # Disable conservation check for test
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Create batch of samples
        batch_size = 5
        state = {
            'temp': torch.randn(batch_size),
            'humidity': torch.randn(batch_size),
            'pressure': torch.randn(batch_size)
        }
        
        result = surrogate.forward(state, time=0.0, dt=60.0)
        
        # Check output shapes
        assert result['rate1'].shape == (batch_size,)
        assert result['rate2'].shape == (batch_size,)
        
        # Check outputs are finite
        assert torch.isfinite(result['rate1']).all()
        assert torch.isfinite(result['rate2']).all()
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_input_shape_validation(self, temp_model_file):
        """Test input shape validation and error handling."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['var1', 'var2', 'var3'],
            'output_variables': ['out1', 'out2']
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Test with wrong number of input variables
        invalid_state = {
            'var1': torch.tensor([1.0]),
            'var2': torch.tensor([2.0])
            # Missing var3
        }
        
        with pytest.raises(KeyError):
            surrogate.forward(invalid_state, time=0.0, dt=60.0)
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_dtype_consistency(self, temp_model_file):
        """Test data type consistency between input and output."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['a', 'b', 'c'],
            'output_variables': ['x', 'y']
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Test different input dtypes
        dtypes_to_test = [torch.float32, torch.float64]
        
        for dtype in dtypes_to_test:
            state = {
                'a': torch.tensor([1.0], dtype=dtype),
                'b': torch.tensor([2.0], dtype=dtype),
                'c': torch.tensor([3.0], dtype=dtype)
            }
            
            result = surrogate.forward(state, time=0.0, dt=60.0)
            
            # Output should be float32 (ONNX default)
            assert result['x'].dtype == torch.float32
            assert result['y'].dtype == torch.float32


class TestDeviceconsistency:
    """Test device consistency and memory management."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_cpu_device_inference(self, temp_model_file):
        """Test inference on CPU device."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['x', 'y', 'z'],
            'output_variables': ['a', 'b'],
            'device': 'cpu'
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        state = {
            'x': torch.tensor([1.0]),
            'y': torch.tensor([2.0]),
            'z': torch.tensor([3.0])
        }
        
        result = surrogate.forward(state, time=0.0, dt=60.0)
        
        # Verify results are on CPU
        assert result['a'].device.type == 'cpu'
        assert result['b'].device.type == 'cpu'
    
    @pytest.mark.skipif(not ONNX_AVAILABLE or not torch.cuda.is_available(), 
                       reason="ONNX Runtime or CUDA not available")
    def test_gpu_device_consistency(self, temp_model_file):
        """Test GPU device consistency if available."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['x', 'y', 'z'],
            'output_variables': ['a', 'b'],
            'device': 'cuda'
        }
        
        try:
            surrogate = ONNXSurrogate("test_onnx", config)
            
            state = {
                'x': torch.tensor([1.0]).cuda(),
                'y': torch.tensor([2.0]).cuda(), 
                'z': torch.tensor([3.0]).cuda()
            }
            
            result = surrogate.forward(state, time=0.0, dt=60.0)
            
            # Results should be moved back to same device as input
            assert result['a'].device.type == 'cuda'
            assert result['b'].device.type == 'cuda'
            
        except Exception as e:
            # GPU provider may not be available in test environment
            if "CUDAExecutionProvider" in str(e):
                pytest.skip("CUDA execution provider not available")
            else:
                raise
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_mixed_device_handling(self, temp_model_file):
        """Test handling of mixed device inputs."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['x', 'y', 'z'],
            'output_variables': ['a', 'b'],
            'device': 'cpu'
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Create inputs on different devices (if CUDA available)
        state = {'x': torch.tensor([1.0]), 'y': torch.tensor([2.0]), 'z': torch.tensor([3.0])}
        
        if torch.cuda.is_available():
            state['y'] = state['y'].cuda()
        
        # Should handle device mismatch gracefully
        try:
            result = surrogate.forward(state, time=0.0, dt=60.0)
            assert 'a' in result and 'b' in result
        except Exception as e:
            # May raise error for device mismatch, which is acceptable behavior
            assert "device" in str(e).lower() or "cuda" in str(e).lower()


class TestPerformanceAndBenchmarking:
    """Test performance characteristics and benchmarking."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_single_inference_latency(self, temp_model_file):
        """Test single inference latency requirements."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['input1', 'input2', 'input3'],
            'output_variables': ['output1', 'output2']
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        state = {
            'input1': torch.tensor([1.0]),
            'input2': torch.tensor([2.0]),
            'input3': torch.tensor([3.0])
        }
        
        # Warmup
        for _ in range(5):
            surrogate.forward(state, time=0.0, dt=60.0)
        
        # Measure latency
        start_time = time.perf_counter()
        result = surrogate.forward(state, time=0.0, dt=60.0)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Should complete within 100ms for simple model
        assert latency_ms < 100, f"Single inference took {latency_ms:.2f}ms (expected < 100ms)"
        assert 'output1' in result and 'output2' in result
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_batch_inference_throughput(self, temp_model_file):
        """Test batch inference throughput requirements."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['x1', 'x2', 'x3'],
            'output_variables': ['y1', 'y2']
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        batch_size = 100
        state = {
            'x1': torch.randn(batch_size),
            'x2': torch.randn(batch_size),
            'x3': torch.randn(batch_size)
        }
        
        # Warmup
        for _ in range(3):
            surrogate.forward(state, time=0.0, dt=60.0)
        
        # Measure throughput
        start_time = time.perf_counter()
        result = surrogate.forward(state, time=0.0, dt=60.0)
        end_time = time.perf_counter()
        
        duration_s = end_time - start_time
        throughput = batch_size / duration_s
        
        # Should achieve > 100 samples/second for simple model
        assert throughput > 100, f"Throughput: {throughput:.1f} samples/s (expected > 100)"
        assert result['y1'].shape[0] == batch_size
        assert result['y2'].shape[0] == batch_size
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_memory_usage_efficiency(self, temp_model_file):
        """Test memory usage efficiency."""
        import psutil
        import gc
        
        config = {
            'model_path': temp_model_file,
            'input_variables': ['a', 'b', 'c'],
            'output_variables': ['x', 'y']
        }
        
        # Measure memory before model loading
        gc.collect()
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Measure memory after model loading
        gc.collect()
        memory_after = process.memory_info().rss
        
        memory_increase_mb = (memory_after - memory_before) / 1024 / 1024
        
        # Model loading should not use excessive memory
        assert memory_increase_mb < 500, f"Memory increase: {memory_increase_mb:.1f}MB"
        
        # Test that multiple inferences don't leak memory
        state = {
            'a': torch.randn(10),
            'b': torch.randn(10),
            'c': torch.randn(10)
        }
        
        for i in range(10):
            result = surrogate.forward(state, time=0.0, dt=60.0)
            
            if i == 4:  # Check memory usage mid-way
                gc.collect()
                memory_mid = process.memory_info().rss
        
        gc.collect()
        memory_final = process.memory_info().rss
        
        # Memory should not increase significantly during inference
        inference_increase_mb = (memory_final - memory_after) / 1024 / 1024
        assert inference_increase_mb < 100, f"Inference memory leak: {inference_increase_mb:.1f}MB"


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_inference_error_handling(self, temp_model_file):
        """Test handling of inference errors."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['input1', 'input2', 'input3'],
            'output_variables': ['output1', 'output2']
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Test with NaN inputs
        state_with_nan = {
            'input1': torch.tensor([float('nan')]),
            'input2': torch.tensor([1.0]),
            'input3': torch.tensor([2.0])
        }
        
        try:
            result = surrogate.forward(state_with_nan, time=0.0, dt=60.0)
            # If it succeeds, outputs might be NaN, which is acceptable
            assert 'output1' in result and 'output2' in result
        except (InferenceError, RuntimeError):
            # Acceptable to fail on NaN inputs
            pass
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_extreme_value_handling(self, temp_model_file):
        """Test handling of extreme input values."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['x', 'y', 'z'],
            'output_variables': ['out1', 'out2']
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        extreme_values = [1e10, -1e10, 1e-10, -1e-10]
        
        for extreme_val in extreme_values:
            state = {
                'x': torch.tensor([extreme_val]),
                'y': torch.tensor([1.0]),
                'z': torch.tensor([1.0])
            }
            
            try:
                result = surrogate.forward(state, time=0.0, dt=60.0)
                # Check that outputs are reasonable
                assert torch.isfinite(result['out1']).all() or torch.abs(result['out1']).max() < 1e20
                assert torch.isfinite(result['out2']).all() or torch.abs(result['out2']).max() < 1e20
            except (InferenceError, RuntimeError, ValueError):
                # Acceptable to fail on extreme values
                pass
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_resource_cleanup(self, temp_model_file):
        """Test proper resource cleanup."""
        config = {
            'model_path': temp_model_file,
            'input_variables': ['a', 'b', 'c'],
            'output_variables': ['x', 'y']
        }
        
        surrogate = ONNXSurrogate("test_onnx", config)
        
        # Verify session is created
        assert hasattr(surrogate, 'ort_session')
        assert surrogate.ort_session is not None
        assert surrogate.model_loaded is True
        
        # Test cleanup (if implemented)
        if hasattr(surrogate, '__del__') or hasattr(surrogate, 'cleanup'):
            # Simulate object destruction
            try:
                if hasattr(surrogate, 'cleanup'):
                    surrogate.cleanup()
                del surrogate
            except Exception:
                # Cleanup may not be implemented yet
                pass


class TestIntegrationWithRELICFramework:
    """Test integration with RELIC framework components."""
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_scientific_workflow_integration(self, temp_model_file):
        """Test integration with typical scientific modeling workflow."""
        # Create metadata similar to real scientific process
        metadata = FunctorMetadata(
            name="photosynthesis_onnx",
            description="ONNX surrogate for Farquhar photosynthesis model",
            category="biological_process",
            mathematical_expression="A_net = f_ONNX(T_air, PPFD, CO2_atm)",
            references=["Farquhar et al. (1980)", "ML Surrogate (2024)"],
            input_variables=["air_temperature", "ppfd", "co2"],
            output_variables=["net_photosynthesis", "gross_photosynthesis"],
        )
        
        config = {
            'model_path': temp_model_file,
            'input_variables': ['air_temperature', 'ppfd', 'co2'],
            'output_variables': ['net_photosynthesis', 'gross_photosynthesis'],
            'units_conversion': {
                'air_temperature': ('celsius', 'kelvin'),
                'net_photosynthesis': ('umol_m2_s1', 'mol_m2_s1')
            }
        }
        
        surrogate = ONNXSurrogate("photosynthesis_onnx", config, metadata)
        
        # Simulate typical environmental conditions
        state = {
            'air_temperature': torch.tensor([20.0, 25.0, 30.0]),  # Celsius
            'ppfd': torch.tensor([800.0, 1200.0, 1600.0]),       # umol/m2/s
            'co2': torch.tensor([380.0, 400.0, 420.0])           # ppm
        }
        
        result = surrogate.forward(state, time=43200.0, dt=3600.0)  # Noon, hourly step
        
        # Check scientific validity
        assert 'net_photosynthesis' in result
        assert result['net_photosynthesis'].shape == (3,)
        
        # Photosynthesis rates should be reasonable (converted to mol/m2/s)
        gpp_values = result['net_photosynthesis']
        # Allow small negative values (net respiration is physiologically possible)
        assert gpp_values.min() >= -1e-2, "Net photosynthesis should be reasonable (small negative values allowed for respiration)"
        assert gpp_values.max() < 1e-2, "Converted rates should be reasonable (mol/m2/s)"


class TestNoSilentConfiguration:
    """A declared ONNX setting is honoured or rejected, never ignored."""

    @staticmethod
    def _config(model_file, **extra):
        config = {
            'model_path': model_file,
            'input_variables': ['input1', 'input2', 'input3'],
            'output_variables': ['output1', 'output2'],
        }
        config.update(extra)
        return config

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_unknown_session_option_is_rejected(self, temp_model_file):
        config = self._config(temp_model_file, session_options={'intra_op_threds': 2})

        with pytest.raises(ModelLoadError, match="intra_op_threds") as excinfo:
            ONNXSurrogate("test_onnx", config)
        assert isinstance(excinfo.value.__cause__, ValueError)

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_input_feature_mismatch_is_rejected(self, temp_model_file):
        config = self._config(temp_model_file, input_variables=['input1', 'input2'])

        with pytest.raises(ModelLoadError, match="input feature"):
            ONNXSurrogate("test_onnx", config)

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_output_dimension_mismatch_is_rejected(self, temp_model_file):
        config = self._config(
            temp_model_file, output_variables=['output1', 'output2', 'output3']
        )

        with pytest.raises(ModelLoadError, match="output dimension"):
            ONNXSurrogate("test_onnx", config)

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX Runtime not available")
    def test_requested_uncertainty_is_not_silently_omitted(self, temp_model_file):
        config = self._config(temp_model_file, uncertainty_quantification=True)

        with pytest.raises(NotImplementedError, match="uncertainty"):
            ONNXSurrogate("test_onnx", config)

    @pytest.mark.skipif(
        not ONNX_AVAILABLE or not torch.cuda.is_available(),
        reason="ONNX Runtime or CUDA not available",
    )
    def test_cuda_device_without_cuda_provider_is_an_error(self, temp_model_file, monkeypatch):
        import relic.models.surrogates as surrogates_module

        monkeypatch.setattr(
            surrogates_module.ort,
            "get_available_providers",
            lambda: ['CPUExecutionProvider'],
        )
        config = self._config(temp_model_file, device='cuda')

        with pytest.raises(ModelLoadError, match="CUDAExecutionProvider"):
            ONNXSurrogate("test_onnx", config)


if __name__ == "__main__":
    # Configure test execution
    test_args = [__file__, "-v", "--tb=short"]
    
    # Add performance tests if requested
    if '--benchmark' in sys.argv:
        test_args.extend(["-k", "test_performance"])
    
    # Add integration tests if requested  
    if '--integration' in sys.argv:
        test_args.extend(["-k", "test_integration"])
    
    pytest.main(test_args)


