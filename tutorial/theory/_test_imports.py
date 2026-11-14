"""Quick import test for RELIC modules needed by tutorials."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

results = {}
modules_to_test = [
    "relic.core.dependency_graph",
    "relic.core.functor.functor_types",
    "relic.core.functor.functor_base",
    "relic.core.state.manager",
    "relic.core.state.conservation",
    "relic.core.simulator.main",
    "relic.core.simulator.execution_loop",
    "relic.core.simulator.scale_ops",
    "torch",
    "matplotlib",
    "numpy",
    "networkx",
]

for mod in modules_to_test:
    try:
        __import__(mod)
        results[mod] = "OK"
    except Exception as e:
        results[mod] = f"FAIL: {e}"

for mod, status in results.items():
    print(f"  {mod}: {status}")

print(f"\nTotal: {sum(1 for v in results.values() if v == 'OK')}/{len(results)} OK")
