import torch
from relic.scientific.biosphere.bvoc.synthesis.monoterpene import MonoterpeneSynthesis

f = MonoterpeneSynthesis()
state = {
    "air_temperature": torch.tensor(300.0),
    "gross_photosynthesis": torch.tensor(10.0),
    MonoterpeneSynthesis.POOL_STATE_KEY: f.reset_pools(),
}
keys_before = set(state.keys())
_ = f.forward(state, time=0.0, dt=1.0)
keys_after = set(state.keys())
added = sorted(list(keys_after - keys_before))
print("ADDED_KEYS=" + ",".join(added))
