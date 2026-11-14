from relic.coupling.scales.spatial.spatial_scaling import SpatialScaling

obj = SpatialScaling(source_scale='grid', target_scale='region', source_resolution=(1.0, 1.0), target_resolution=(2.0, 2.0), method='conservative')
assert obj.metadata.input_variables, "metadata.input_variables must not be empty"
assert obj.metadata.output_variables, "metadata.output_variables must not be empty"
print(obj.metadata.name)
