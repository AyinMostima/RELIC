from relic.core.simulator import RELICSimulator
from relic.scientific.pedosphere.methane.registry import register_methane_functors

sim = RELICSimulator(config={"strict_dependencies": True})
print("SIM_INIT_OK")
try:
    register_methane_functors(sim)
    print("REGISTER_OK")
except Exception as exc:
    print(type(exc).__name__ + ":" + str(exc))
