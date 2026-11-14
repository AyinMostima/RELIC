"""
Respiration Processes for RELIC

This module provides the canonical plant maintenance, growth, and aggregate
autotrophic-respiration carbon ports. Each process uses the unified parameter
and Functor lifecycle.

Main Components
------------------------------------
- MaintenanceRespiration: Temperature-dependent maintenance respiration with Q10 response
- GrowthRespiration: Carbon-limited construction respiration
- AutotrophicRespiration: State-driven aggregation of scheduled respiration ports
- Full integration with unified parameter management system

Mathematical Framework
-----------------------
1. Maintenance respiration: R_i = k_i(T) * B_i * f_sub * f_water;
   R_m = sum_i(R_i)
2. Available carbon: C_avail = max(GPP - R_m, 0)
3. Growth respiration: R_g = min((1 - Y_g) * C_avail * f_T * f_S, C_avail)
4. Carbon ledger: R_a = R_m + R_g; NPP = GPP - R_a

Units
-----
- GPP, respiration, and NPP rates: kg C m-2 s-1
- Biomass: g C m-2
- Temperature: K
- Response factors and carbon-use efficiency: dimensionless

References
----------------------------------
- Ryan, M.G. (1991). Effects of climate change on plant respiration. 
  Ecological Applications 1, 157-167
- Penning de Vries, F.W.T., et al. (1974). Products, requirements and 
  efficiency of biosynthesis. Journal of Theoretical Biology 45, 339-377
- Amthor, J.S. (2000). The McCree-de Wit-Penning de Vries-Thornley 
  respiration paradigms. New Phytologist 146, 201-205

Example Usage
----------------------
    from relic.scientific.biosphere.respiration import AutotrophicRespiration
    # Parameters loaded automatically via ParameterManager
    
    # Parameters are automatically loaded from YAML files via ParameterManager
    # No manual parameter passing required
    respiration = AutotrophicRespiration()
    
    # Optional: customize parameters through ParameterManager
    # parameter_manager = ...  # obtain from system bootstrap utilities
    # parameter_manager.set_value("biosphere.respiration.maintenance.q10_respiration", 2.2)
    
    # Provide required state variables
    # Note: Use test fixtures from test/fixtures/biosphere_test_data.yaml for actual testing
    # Example values shown here are for documentation purposes only
    # state = {...}  # Provide required state variables (temperature, biomass pools, nutrient status, etc.)
    # result = respiration.forward(state, time=0.0, dt=1.0)
    # total_resp = result['total_autotrophic_respiration']
"""

# Main respiration model imports
from .maintenance_respiration import MaintenanceRespiration
from .growth_respiration import GrowthRespiration
from .autotrophic_respiration import AutotrophicRespiration

__all__ = [
    "MaintenanceRespiration",
    "GrowthRespiration",
    "AutotrophicRespiration",
]
