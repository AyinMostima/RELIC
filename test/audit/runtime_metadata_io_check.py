from relic.scientific.anthroposphere.land_use.mapping import LandUseParameterMapping
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.coupling.scales.spatial.spatial_scaling import SpatialScaling
from relic.scientific.biosphere.phenology.models import StaticPhenology

constructors = [
    ('LandUseParameterMapping', lambda: LandUseParameterMapping()),
    ('FickianDiffusion', lambda: FickianDiffusion()),
    ('SpatialScaling', lambda: SpatialScaling(source_scale='grid', target_scale='region', source_resolution=(1.0, 1.0), target_resolution=(2.0, 2.0), method='conservative')),
    ('StaticPhenology', lambda: StaticPhenology(name='static_phenology_probe')),
]
for name, ctor in constructors:
    obj = ctor()
    metadata = obj.metadata
    inputs = metadata.input_variables or []
    outputs = metadata.output_variables or []
    if not inputs or not outputs:
        raise RuntimeError(f"{name} has an incomplete runtime metadata interface")
    print('[OK]', name, len(inputs), len(outputs))
