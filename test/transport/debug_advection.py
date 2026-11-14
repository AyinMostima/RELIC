"""
Debug script to check operator return values
"""

import torch
from relic.computation.operators.functional import apply_advection

# Create simple test case
conc = torch.ones((1, 11))
vel = torch.ones((1, 11)) * 0.5
dx = 0.1
dt = 0.01

result = apply_advection(
    concentration=conc,
    velocity=vel,
    grid_spacing=dx,
    spatial_dims=1,
    time_step=dt,
    check_cfl=True, scheme="upwind", boundary_method="extrapolate"
)

print("Result keys:", list(result.keys()))
print("Advection rate shape:", result['advection_rate'].shape)
print("Advection rate values:", result['advection_rate'])
print("Concentration shape:", result['concentration'].shape)
print("Concentration values:", result['concentration'])
print("CFL number:", result['cfl_number'])

