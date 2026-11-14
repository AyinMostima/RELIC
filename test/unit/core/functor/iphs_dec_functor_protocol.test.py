from relic.core.functor import IPHSDECFunctor


class MinimalStructuralFunctor:
    name = "minimal_structural_functor"
    metadata = {}

    def declared_forms(self):
        return ()

    def required_parameters(self):
        return ()

    def build_hodge_terms(self, complex_bundle, resolved_parameters):
        return ()

    def energy_terms(self, state):
        return {}

    def entropy_terms(self, state):
        return {}

    def reversible_structure(self, state, complex_bundle):
        return None

    def dissipative_structure(self, state, complex_bundle):
        return None

    def ports(self):
        return ()

    def compute(self, state, time, dt):
        return {}


def test_iphs_dec_functor_protocol_is_runtime_checkable():
    functor = MinimalStructuralFunctor()

    assert isinstance(functor, IPHSDECFunctor)
    assert functor.compute({}, 0.0, 1.0) == {}
