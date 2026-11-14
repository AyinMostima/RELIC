from relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake import RootNutrientUptake
import torch

f = RootNutrientUptake()
print('INIT_OK')
try:
    f({}, 0.0, torch.tensor(1.0))
    print('CALL_OK')
except Exception as exc:
    print(type(exc).__name__ + ':' + str(exc))
