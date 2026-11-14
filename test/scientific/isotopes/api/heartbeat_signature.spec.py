import inspect

from relic.scientific.isotopes.functor import IsotopeHeartbeatFunctor


def test_heartbeat_forward_signature_matches_functor_base_contract() -> None:
    signature = inspect.signature(IsotopeHeartbeatFunctor.forward)
    parameter_names = list(signature.parameters.keys())
    assert parameter_names[:4] == ["self", "state", "time", "dt"]

