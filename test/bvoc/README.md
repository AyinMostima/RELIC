# BVOC Module Test Suite

Comprehensive test suite for the Biogenic Volatile Organic Compounds (BVOC) emission module in RELIC, following the L1-L2-L3 testing framework for scientific software validation.

## Test Structure

### L1 Unit Tests (`test_bvoc_unit.py`)
- **Purpose**: Test individual functions and methods in isolation
- **Scope**: Parameter loading, MEGAN calculations, response functions, emission factors
- **Dependencies**: Mocked (StateManager, ParameterManager, etc.)
- **Target**: Fast execution, high coverage of edge cases
- **Scientific Focus**: Algorithm correctness, numerical stability

### L2 Integration Tests (`test_bvoc_integration.py`)
- **Purpose**: Test component interactions and data flow
- **Scope**: MEGAN model integration, photosynthesis coupling, conservation principles
- **Dependencies**: Real components with realistic data
- **Target**: System behavior validation, coupling mechanisms
- **Scientific Focus**: Process interactions, mass/energy conservation

### L3 System Tests (`test_bvoc_system.py`)
- **Purpose**: End-to-end validation against real-world data
- **Scope**: FluxNet validation, performance benchmarks, long-term stability
- **Dependencies**: Full system integration
- **Target**: Scientific accuracy, computational performance
- **Scientific Focus**: Model-measurement comparison, operational requirements

## Quick Start

### Prerequisites
```bash
# Required packages
pip install pytest torch numpy

# Ensure relic is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/RELIC-REBUILD"
```

### Running Tests

#### Run All Tests
```bash
cd test/bvoc
python run_bvoc_tests.py --level all --verbose --report
```

#### Run Specific Test Level
```bash
# Unit tests only (fast)
python run_bvoc_tests.py --level L1 --quick

# Integration tests with detailed output
python run_bvoc_tests.py --level L2 --verbose

# System tests (comprehensive but slow)
python run_bvoc_tests.py --level L3 --report
```

#### Using pytest Directly
```bash
# Run specific test file
pytest test_bvoc_unit.py -v

# Run with coverage
pytest test_bvoc_unit.py --cov=relic.scientific.biosphere.bvoc

# Run specific test class
pytest test_bvoc_unit.py::TestMEGANModelCalculations -v
```

## Test Configuration

### Environment Variables
```bash
# Enable CUDA testing (if available)
export CUDA_VISIBLE_DEVICES=0

# Adjust test verbosity
export PYTEST_VERBOSITY=2

# Skip slow tests
export PYTEST_MARKERS="not slow"
```

### Test Markers
- `@pytest.mark.slow`: Long-running tests (>10s)
- `@pytest.mark.gpu`: Requires CUDA-capable GPU
- `@pytest.mark.integration`: Requires real data/components
- `@pytest.mark.scientific`: Validates against literature values

## Scientific Validation

### Literature References
Tests validate against published literature:
- **Guenther et al. (2012)**: MEGAN 2.1 algorithm and parameters
- **Niinemets et al. (2010)**: Response function parameters
- **FluxNet Database**: Real-world emission measurements
- **MEGAN Global Inventory**: Regional and global emission totals

### Validation Targets
- **Isoprene Emissions**: 15-35 μg m⁻² s⁻¹ (temperate deciduous, summer midday)
- **Monoterpene Emissions**: 0.5-2.0 μg m⁻² s⁻¹ (temperate deciduous, summer midday)
- **Annual Totals**: 2-8 kg C ha⁻¹ year⁻¹ (temperate deciduous forest)
- **Diurnal Patterns**: 3x higher daytime vs nighttime emissions
- **Seasonal Patterns**: Summer peak, winter minimum (deciduous)

### Performance Requirements
- **Single Timestep**: < 1 ms per grid cell
- **Batch Processing**: > 1000 grid cells per second
- **Memory Usage**: < 100 MB for continental domain
- **GPU Acceleration**: > 3x speedup over CPU

## Common Test Patterns

### Mocking Parameter Manager
```python
from unittest.mock import Mock, patch

@patch('relic.parameters.manager.get_global_manager')
def test_with_mocked_params(mock_manager):
    mock_instance = Mock()
    mock_instance.get.side_effect = lambda key, default=None, **kwargs: default
    mock_manager.return_value = mock_instance
    
    # Test code here
```

### Testing Scientific Accuracy
```python
def test_emission_against_literature():
    """Test against Guenther et al. (2012) values."""
    # Standard MEGAN conditions
    state = {
        'temperature': torch.tensor(303.15),  # 30°C
        'par': torch.tensor(1000.0),          # 1000 μmol m⁻² s⁻¹
        'foliar_density': torch.tensor(200.0) # 200 g DW m⁻²
    }
    
    result = model.forward(state, time=0, dt=3600)
    emission = result['isoprene_emission'].item()
    
    # Literature range for temperate deciduous forest
    assert 15.0 <= emission <= 35.0, f"Emission {emission} outside literature range"
```

### Performance Testing
```python
import time

def test_performance_benchmark():
    """Test computational performance."""
    # Warm up
    for _ in range(5):
        model.forward(test_state, time=0, dt=3600)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        result = model.forward(test_state, time=0, dt=3600)
    end_time = time.time()
    
    time_per_call = (end_time - start_time) / 100
    assert time_per_call < 0.001, f"Too slow: {time_per_call*1000:.2f} ms"
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
ModuleNotFoundError: No module named 'relic'
```
**Solution**: Ensure relic is in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/RELIC-REBUILD"
```

#### Parameter Manager Errors
```bash
AttributeError: 'Mock' object has no attribute 'get'
```
**Solution**: Check mock configuration in test setup.

#### CUDA Errors
```bash
RuntimeError: CUDA out of memory
```
**Solution**: Skip GPU tests or reduce batch size:
```bash
pytest -m "not gpu" test_bvoc_unit.py
```

#### Scientific Validation Failures
```bash
AssertionError: Emission 45.2 outside literature range [15.0, 35.0]
```
**Solution**: Check parameter values and model configuration against literature.

### Debug Mode
```bash
# Run with debugging
pytest --pdb test_bvoc_unit.py::test_specific_function

# Verbose output with print statements
pytest -s -v test_bvoc_unit.py
```

## Test Data and Fixtures

### Realistic Test Data
- **Environmental Conditions**: Based on FluxNet site measurements
- **Parameter Values**: Literature-validated values from MEGAN database
- **Expected Results**: Ranges from field measurements and model intercomparisons

### Test Fixtures
- `@pytest.fixture realistic_forest_state`: Standard forest conditions
- `@pytest.fixture mock_parameter_manager`: Configured parameter manager mock
- `@pytest.fixture comprehensive_validation_suite`: Full validation configuration

## Contributing

### Adding New Tests
1. Follow the L1-L2-L3 hierarchy
2. Include scientific references in docstrings
3. Use literature-based validation targets
4. Add performance benchmarks for new features
5. Mock dependencies appropriately

### Test Naming Convention
```python
class TestComponentName:
    def test_specific_functionality(self):
        """Test description with scientific context.
        
        Mathematical Expression: E = epsilon * gamma_T * gamma_L
        References: Guenther et al. (2012)
        Parameters: epsilon [μg g⁻¹ DW h⁻¹], gamma_T [dimensionless]
        """
```

### Validation Requirements
- All tests must have scientific justification
- Expected values must cite literature sources
- Performance tests must meet operational requirements
- Conservation tests must verify physical principles

## Contact

For questions about the BVOC test suite:
- Review RELIC documentation
- Check scientific literature references
- Consult FluxNet database for validation targets
- Follow MEGAN model documentation for parameter values

---

*This test suite ensures the BVOC module meets both scientific accuracy and computational performance requirements for operational Earth system modeling.*