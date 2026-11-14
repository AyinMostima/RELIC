"""
Unit Tests for Pedosphere Modules

This package contains L1 unit tests for the pedosphere (soil biogeochemistry) modules
in the RELIC framework, covering:

- Decomposition processes (first-order, microbial-enzyme)
- Carbon pool dynamics (transfer matrix, mass balance)
- Nutrient cycling (mineralization, nitrification, denitrification)
- Transport mechanisms (soil-specific transport)

Test Organization:
- test_first_order_decomposition.py: FirstOrderDecomposition L1 tests
- test_pools_matrix.py: PoolsMatrix L1 tests
- test_mineralization_process.py: MineralizationProcess L1 tests

Each test module follows the L1 testing strategy with categories:
- L1.1: Basic instantiation and configuration
- L1.2: Scientific accuracy against literature values
- L1.3: Environmental response functions
- L1.4: Process-specific mechanisms (transfers, stoichiometry, etc.)
- L1.5: Boundary condition handling
- L1.6: Parameter sensitivity analysis
- L1.7: Performance and stability tests
"""

__all__ = [
    'TestFirstOrderDecompositionL1',
    'TestPoolsMatrixL1', 
    'TestMineralizationProcessL1'
]

