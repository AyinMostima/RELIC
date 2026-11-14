"""Tests for irreversible port-Hamiltonian structure specs."""

import pytest
import torch


_EXECUTABLE_IPHS_SOURCE = (
    "relic.computation.operators.iphs.IPHSStructureSpec.canonical_drift"
)


def _zero_poisson_spec(**overrides):
    from relic.computation.operators.iphs import IPHSStructureSpec

    fields = dict(
        name="zero_poisson_field", state_vars=("energy", "irreversible_coordinate"),
        energy_functional="E=x0", entropy_functional="S=3*x1",
        J=torch.zeros((2, 2), dtype=torch.float64),
        M=torch.diag(torch.tensor([0.0, 2.0], dtype=torch.float64)),
        G=None, stoichiometric_N=None, ports=(), formulation="generic",
        energy_gradient=torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True),
        entropy_gradient=torch.tensor([0.0, 3.0], dtype=torch.float64),
        contribution_count=1, contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        poisson_operator="zero_field", reaction_representation="independent",
    )
    fields.update(overrides)
    return IPHSStructureSpec(**fields)


def test_zero_poisson_field_executes_irreversible_flow_with_a_zero_energy_derivative():
    # References: Grmela and Ottinger (1997), GENERIC formalism.
    # Mathematical expression: J(x)=0; M=diag(0,2); dx/dt=M*(0,3)=(0,6).
    # Parameter meanings: e_E=(1,0), e_S=(0,3), M is irreversible mobility.
    # Units: normalized test coordinates; this is not a physical-process certificate.
    # Domain/constraints: a finite two-coordinate structural pre-gate, without Nr.
    # Sign convention: positive e_S dot dx/dt is entropy production.
    spec = _zero_poisson_spec()
    audit = spec.validate()
    assert audit["jacobi_identity_verified"] is True
    assert audit["jacobi_identity_basis"] == "canonical_zero_field"
    assert audit["process_level_status"] == "[N] runtime_unbound"
    drift = spec.canonical_drift()
    torch.testing.assert_close(drift, torch.tensor([0.0, 6.0], dtype=torch.float64))
    assert torch.dot(spec.energy_gradient, drift).item() == 0.0
    assert torch.dot(spec.entropy_gradient, drift).item() == 18.0
    gradient = torch.autograd.grad(drift.sum(), spec.energy_gradient)[0]
    torch.testing.assert_close(gradient, torch.zeros(2, dtype=torch.float64))


def test_zero_poisson_field_is_not_a_process_runtime_binding():
    with pytest.raises(ValueError, match="process runtime.*linkage"):
        _zero_poisson_spec().validate(required_level=5)


@pytest.mark.parametrize("magnitude", [1.0, 1.0e-30])
def test_zero_poisson_field_rejects_any_nonzero_declared_matrix(magnitude):
    matrix = torch.tensor([[0.0, magnitude], [-magnitude, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="zero_field"):
        _zero_poisson_spec(J=matrix)


@pytest.mark.parametrize("operation", ["validate", "canonical_drift"])
def test_zero_poisson_field_rechecks_a_mutated_matrix_at_each_evaluation(operation):
    spec = _zero_poisson_spec()
    spec.J[0, 1] = 1.0e-30
    spec.J[1, 0] = -1.0e-30
    with pytest.raises(ValueError, match="zero_field"):
        if operation == "validate":
            spec.validate(required_level=5)
        else:
            spec.canonical_drift()


@pytest.mark.parametrize("magnitude", [1.0, 1.0e-30])
def test_zero_poisson_field_keeps_reaction_and_degeneracy_gates(magnitude):
    spec = _zero_poisson_spec(stoichiometric_N=magnitude * torch.tensor([[-1.0], [1.0]], dtype=torch.float64))
    with pytest.raises(ValueError, match="reaction closure"):
        spec.validate(required_level=5)
    spec = _zero_poisson_spec(M=torch.eye(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="M e_E"):
        spec.validate(required_level=5)


def test_zero_poisson_field_does_not_hide_an_inactive_resistance_below_tolerance():
    spec = _zero_poisson_spec(R=torch.diag(torch.tensor([1.0e-30, 0.0], dtype=torch.float64)))
    with pytest.raises(ValueError, match="R=0|resistance R"):
        spec.validate(required_level=5)
    with pytest.raises(ValueError, match="resistance R"):
        spec.canonical_drift()


@pytest.mark.parametrize("iterator_payload", [False, True])
def test_zero_field_assembly_preserves_execution_and_runtime_boundary(iterator_payload):
    from relic.computation.operators.iphs import assemble_iphs_structure

    local = _zero_poisson_spec()
    declared_j = [[0.0, 0.0], [0.0, 0.0]]
    if iterator_payload:
        declared_j = (iter(row) for row in declared_j)
    spec = assemble_iphs_structure(
        name="zero_assembly", state_vars=local.state_vars,
        energy_functional=local.energy_functional, entropy_functional=local.entropy_functional,
        formulation="generic", poisson_operator="zero_field", validate=True,
        contributions=[dict(J=declared_j, M=local.M,
                            energy_gradient=local.energy_gradient,
                            entropy_gradient=local.entropy_gradient,
                            runtime_owner=_EXECUTABLE_IPHS_SOURCE)], reaction_representation="independent",
    )
    assert spec.poisson_operator == "zero_field"
    assert spec.validate()["jacobi_identity_basis"] == "canonical_zero_field"
    torch.testing.assert_close(spec.canonical_drift(), local.canonical_drift())
    with pytest.raises(ValueError, match="process runtime.*linkage"):
        spec.validate(required_level=5)


@pytest.mark.parametrize("magnitude", [1.0, 1.0e-50])
@pytest.mark.parametrize("container", ["list", "tensor", "iterator"])
def test_zero_field_assembly_rejects_local_cancellation_before_precision_conversion(magnitude, container):
    from relic.computation.operators.iphs import assemble_iphs_structure

    matrix = [[0.0, magnitude], [-magnitude, 0.0]]
    opposite = [[0.0, -magnitude], [magnitude, 0.0]]
    if container == "tensor":
        matrix = torch.tensor(matrix, dtype=torch.float64)
    elif container == "iterator":
        matrix = (iter(row) for row in matrix)
    with pytest.raises(ValueError, match="zero_field.*J"):
        assemble_iphs_structure(
            name="cancelled_local", state_vars=("x", "y"),
            energy_functional="E", entropy_functional="S",
            poisson_operator="zero_field",
            contributions=[{"M": torch.eye(2, dtype=torch.float32)},
                           {"J": matrix}, {"J": opposite}], reaction_representation="independent",
        )


@pytest.mark.parametrize("magnitude", [1.0, 1.0e-50])
def test_zero_field_assembly_rejects_feedback_even_when_embedding_annihilates_it(magnitude):
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="zero_field.*port_interconnection_C"):
        assemble_iphs_structure(
            name="annihilated_feedback", state_vars=("x", "y"),
            energy_functional="E", entropy_functional="S",
            poisson_operator="zero_field",
            contributions=[
                {
                    "G": torch.zeros((2, 2)),
                    "ports": (
                        _scalar_iphs_port("feedback_a", role="internal"),
                        _scalar_iphs_port("feedback_b", role="internal"),
                    ),
                }
            ],
            port_interconnection_C=[[0.0, magnitude], [-magnitude, 0.0]], reaction_representation="independent",
        )


@pytest.mark.parametrize("formulation", ["generic", "dissipative_ph"])
@pytest.mark.parametrize("representation", ["independent", "absorbed"])
def test_reaction_representation_controls_single_drift_and_rate_gradient(formulation, representation):
    from relic.computation.operators.iphs import IPHSStructureSpec

    # References: Grmela and Ottinger (1997), GENERIC; van der Schaft (2022), dPH.
    # Mathematical expression: nu=(-1,1,0); M=r*nu*nu.T; M*(0,1,0)=nu*r.
    #   For dPH, R=r*nu*nu.T and -R*(1,0,0)=nu*r.
    # Parameter meanings: r is the reaction extent rate, nu the species incidence.
    # Units: consistent normalized coordinates and time; not a physical certificate.
    # Domain/constraints: the full irreversible operator represents this one reaction.
    # Sign convention: species 0 is removed and species 1 is produced.
    rate = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    nu = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float64)
    operator = rate[0] * torch.outer(nu, nu)
    zero = torch.zeros((3, 3), dtype=torch.float64)
    spec = IPHSStructureSpec(
        name="reaction_representation", state_vars=("a", "b", "energy"),
        energy_functional="E", entropy_functional="S", J=zero,
        M=operator if formulation == "generic" else zero,
        R=operator if formulation == "dissipative_ph" else zero,
        G=None, stoichiometric_N=nu[:, None], ports=(), formulation=formulation,
        energy_gradient=torch.tensor([0.0, 0.0, 1.0] if formulation == "generic" else [1.0, 0.0, 0.0], dtype=torch.float64),
        entropy_gradient=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        reaction_representation=representation, poisson_operator="matrix_snapshot",
    )
    multiplier = 1.0 if representation == "absorbed" else 2.0
    drift = spec.canonical_drift(reaction_rates=rate)
    torch.testing.assert_close(drift, multiplier * nu * rate)
    torch.testing.assert_close(torch.autograd.grad(drift[1], rate)[0], torch.tensor([multiplier], dtype=torch.float64))
    state = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    for _ in range(3):
        state = state + 0.1 * spec.canonical_drift(reaction_rates=rate)
    torch.testing.assert_close(state, torch.tensor([1.0-0.09*multiplier, 0.09*multiplier, 2.0], dtype=torch.float64))


@pytest.mark.parametrize("rate", [1.0, 1.0e-200])
def test_absorbed_reaction_rejects_mismatched_operator_without_absolute_tolerance_floor(rate):
    spec = _zero_poisson_spec(
        stoichiometric_N=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        M=torch.diag(torch.tensor([0.0, rate], dtype=torch.float64)),
        reaction_representation="absorbed",
    )
    with pytest.raises(ValueError, match="absorbed.*N r"):
        spec.canonical_drift(reaction_rates=torch.tensor([rate], dtype=torch.float64))


@pytest.mark.parametrize("representation", ["unknown", None, True])
def test_reaction_representation_rejects_invalid_selection(representation):
    with pytest.raises(ValueError, match="reaction_representation"):
        _zero_poisson_spec(reaction_representation=representation)


def test_absorbed_reaction_requires_a_real_stoichiometric_declaration():
    with pytest.raises(ValueError, match="absorbed.*stoichiometric_N"):
        _zero_poisson_spec(reaction_representation="absorbed")


def test_assembler_preserves_absorbed_representation_without_hiding_stoichiometry():
    from relic.computation.operators.iphs import assemble_iphs_structure

    spec = assemble_iphs_structure(
        name="absorbed_assembly", state_vars=("energy", "coordinate"),
        energy_functional="E", entropy_functional="S", formulation="generic",
        poisson_operator="zero_field", reaction_representation="absorbed",
        contributions=[{"M": [[0.0, 0.0], [0.0, 2.0]],
                        "stoichiometric_N": [[0.0], [1.0]],
                        "energy_gradient": [1.0, 0.0], "entropy_gradient": [0.0, 3.0]}],
        validate=True,
    )
    assert spec.validate()["reaction_representation"] == "absorbed"
    assert spec.stoichiometric_N is not None
    torch.testing.assert_close(
        spec.canonical_drift(reaction_rates=torch.tensor([6.0], dtype=torch.float32)),
        torch.tensor([0.0, 6.0], dtype=torch.float32),
    )


@pytest.mark.parametrize("outer,local", [("absorbed", "independent"), ("independent", "absorbed")])
def test_assembler_rejects_mixed_reaction_representations(outer, local):
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="reaction_representation"):
        assemble_iphs_structure(
            name="mixed_reactions", state_vars=("x",), energy_functional="E",
            entropy_functional="S", formulation="generic", reaction_representation=outer,
            contributions=[{"M": [[1.0]], "stoichiometric_N": [[1.0]],
                            "reaction_representation": local}], poisson_operator="matrix_snapshot",
        )


def test_absorbed_reaction_cannot_use_structural_tolerance_to_change_the_law():
    spec = _zero_poisson_spec(
        reaction_representation="absorbed", degeneracy_tol=0.5,
        M=torch.diag(torch.tensor([0.0, (1.0 + 1.0e-8) / 3.0], dtype=torch.float64)),
        stoichiometric_N=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="absorbed.*N r"):
        spec.canonical_drift(reaction_rates=torch.tensor([1.0], dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_absorbed_reaction_accepts_only_roundoff_difference(dtype):
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="rounded_representation", state_vars=("x",), energy_functional="E",
        entropy_functional="S", formulation="generic", reaction_representation="absorbed",
        J=torch.zeros((1, 1), dtype=dtype), M=torch.tensor([[0.3]], dtype=dtype),
        stoichiometric_N=torch.ones((1, 1), dtype=dtype), G=None, ports=(),
        energy_gradient=torch.zeros(1, dtype=dtype), entropy_gradient=torch.ones(1, dtype=dtype), poisson_operator="matrix_snapshot",
    )
    rates = torch.tensor([0.1], dtype=dtype) + torch.tensor([0.2], dtype=dtype)
    torch.testing.assert_close(spec.canonical_drift(reaction_rates=rates), torch.tensor([0.3], dtype=dtype))


def test_dph_drift_derives_energy_effort_with_mixed_parameter_derivatives():
    from relic.computation.operators.iphs import IPHSStructureSpec

    # References: van der Schaft (2022), dissipative port-Hamiltonian systems.
    # Mathematical expression: H=k*sum(x^2)/2; e_H=k*x; dx/dt=(J-R)*e_H.
    # Parameter meanings: k is storage curvature; R=diag(2,1); J=[[0,1],[-1,0]].
    # Units: consistent normalized coordinates, energy, and time.
    # Domain/constraints: smooth quadratic potential, no reaction or external port.
    # Sign convention: dH/dt=-e_H.T*R*e_H=-272 at x=(2,3), k=4.
    state = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    energy = 0.5 * coefficient * state.square().sum()
    spec = IPHSStructureSpec(
        name="bound_dph_potential", state_vars=("x", "y"),
        energy_functional="k*sum(x^2)/2", entropy_functional="unused",
        J=torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64),
        R=torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64)),
        M=torch.zeros((2, 2), dtype=torch.float64), G=None, stoichiometric_N=None,
        ports=(), formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )
    drift = spec.canonical_drift(state=state, energy_value=energy)
    torch.testing.assert_close(drift, torch.tensor([-4.0, -20.0], dtype=torch.float64))
    torch.testing.assert_close(torch.dot(coefficient * state, drift), torch.tensor(-272.0, dtype=torch.float64))
    state_gradient, parameter_gradient = torch.autograd.grad(drift.sum(), (state, coefficient), create_graph=True)
    torch.testing.assert_close(state_gradient, torch.tensor([-12.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(parameter_gradient, torch.tensor(-6.0, dtype=torch.float64))
    torch.testing.assert_close(torch.autograd.grad(state_gradient[0], coefficient)[0], torch.tensor(-3.0, dtype=torch.float64))


def test_generic_drift_uses_energy_and_entropy_from_the_same_state_graph():
    # References: Grmela and Ottinger (1997), GENERIC formalism.
    # Mathematical expression: E=x0; S=k*log(x1); M=diag(0,2); dx1/dt=2*k/x1.
    # Parameter meanings: x0 is energy, x1 the irreversible coordinate, k>0.
    # Units: normalized coordinates for a low-level potential-binding oracle.
    # Domain/constraints: x1>0, zero Poisson field, no reaction or port.
    # Sign convention: energy is constant and entropy production is positive.
    state = torch.tensor([5.0, 2.0], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    spec = _zero_poisson_spec(
        energy_gradient=None, entropy_gradient=None, entropy_functional="k*log(x1)",
    )
    drift = spec.canonical_drift(
        state=state, energy_value=state[0], entropy_value=coefficient * torch.log(state[1]),
    )
    torch.testing.assert_close(drift, torch.tensor([0.0, 3.0], dtype=torch.float64))
    state_gradient, parameter_gradient = torch.autograd.grad(drift[1], (state, coefficient), create_graph=True)
    torch.testing.assert_close(state_gradient, torch.tensor([0.0, -1.5], dtype=torch.float64))
    torch.testing.assert_close(parameter_gradient, torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(torch.autograd.grad(state_gradient[1], coefficient)[0], torch.tensor(-0.5, dtype=torch.float64))


def test_entropy_effort_through_implicit_root_retains_drift_state_sensitivity():
    from relic.computation.solvers import BracketedBisectionSolver, SolverStatus

    # References: Blondel et al. (2022); Grmela and Ottinger (1997).
    # Mathematical expression: T^2-q=0, S=log(T), dS/dq=1/(2q);
    #   M=diag(0,2), dq/dt=1/q, d(dq/dt)/dq=-1/q^2.
    # Parameter meanings: q is the positive state parameter of a scalar closure.
    # Units: normalized root/entropy coordinates, not a physical thermochemistry model.
    # Domain/constraints: smooth positive root; both AD stages use the same q.
    # Sign convention: q grows with positive entropy effort; energy stays constant.
    state = torch.tensor([5.0, 4.0], dtype=torch.float64, requires_grad=True)
    solver = BracketedBisectionSolver(
        config={"max_iterations": 80, "rtol": 1e-12, "atol": 1e-12}, dtype=torch.float64,
    )
    root = solver(
        lambda value, q: value.square() - q,
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(3.0, dtype=torch.float64),
        parameters=(state[1],),
    )
    assert root.status is SolverStatus.SUCCESS
    spec = _zero_poisson_spec(
        energy_gradient=None, entropy_gradient=None, entropy_functional="log(sqrt(q))",
    )
    drift = spec.canonical_drift(
        state=state, energy_value=state[0], entropy_value=torch.log(root.solution),
    )
    torch.testing.assert_close(drift, torch.tensor([0.0, 0.25], dtype=torch.float64))
    torch.testing.assert_close(
        torch.autograd.grad(drift[1], state)[0],
        torch.tensor([0.0, -0.0625], dtype=torch.float64),
    )


def test_absorbed_reaction_uses_derived_entropy_effort_once():
    # References: Grmela and Ottinger (1997), entropy-gradient representation.
    # Mathematical expression: S=3*log(q), r=k*q, M_qq=r*q/3; M*dS/dq=r.
    # Parameter meanings: q is the positive coordinate, k the rate coefficient.
    # Units: normalized coordinate and time for the representation oracle.
    # Domain/constraints: the entire irreversible term represents the same N r.
    # Sign convention: positive r increases q; the energy coordinate is unchanged.
    state = torch.tensor([5.0, 4.0], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    rate = coefficient * state[1]
    mobility = torch.diag(torch.stack((torch.zeros_like(rate), rate * state[1] / 3.0)))
    spec = _zero_poisson_spec(
        energy_gradient=None, entropy_gradient=None, entropy_functional="3*log(q)",
        M=mobility, stoichiometric_N=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        reaction_representation="absorbed",
    )
    drift = spec.canonical_drift(
        state=state, energy_value=state[0], entropy_value=3.0 * torch.log(state[1]),
        reaction_rates=rate.reshape(1),
    )
    torch.testing.assert_close(drift, torch.tensor([0.0, 0.8], dtype=torch.float64))
    state_gradient, parameter_gradient = torch.autograd.grad(drift[1], (state, coefficient))
    torch.testing.assert_close(state_gradient, torch.tensor([0.0, 0.2], dtype=torch.float64))
    torch.testing.assert_close(parameter_gradient, torch.tensor(4.0, dtype=torch.float64))


def test_dph_potential_binding_rejects_an_unused_entropy_value():
    state = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    spec = _zero_poisson_spec(
        formulation="dissipative_ph", M=torch.zeros((2, 2), dtype=torch.float64),
        R=torch.eye(2, dtype=torch.float64), energy_gradient=None, entropy_gradient=None,
    )
    with pytest.raises(ValueError, match="dissipative_ph.*entropy_value"):
        spec.canonical_drift(state=state, energy_value=state.square().sum(), entropy_value=state.sum())


@pytest.mark.parametrize("case,error", [
    ("missing_state", "state.*energy_value"),
    ("missing_energy", "state.*energy_value"),
    ("missing_entropy", "entropy_value"),
    ("detached", "autograd"),
    ("unrelated", "not have been used"),
    ("vector", "scalar"),
    ("dtype", "dtype.*device"),
    ("state_without_grad", "requires_grad"),
    ("nonfinite", "finite"),
    ("explicit_effort", "mutually exclusive"),
    ("stored_effort", "mutually exclusive"),
])
def test_potential_binding_rejects_ambiguous_or_unconnected_inputs(case, error):
    state = torch.tensor([5.0, 2.0], dtype=torch.float64, requires_grad=True)
    spec = _zero_poisson_spec(energy_gradient=None, entropy_gradient=None)
    arguments = dict(state=state, energy_value=state[0], entropy_value=state[1].square())
    if case == "missing_state":
        arguments.pop("state")
    elif case == "missing_energy":
        arguments.pop("energy_value")
    elif case == "missing_entropy":
        arguments.pop("entropy_value")
    elif case == "detached":
        arguments["energy_value"] = state[0].detach()
    elif case == "unrelated":
        arguments["energy_value"] = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    elif case == "vector":
        arguments["energy_value"] = state[:1]
    elif case == "dtype":
        arguments["energy_value"] = state[0].to(torch.float32)
    elif case == "state_without_grad":
        arguments["state"] = state.detach()
    elif case == "nonfinite":
        arguments["energy_value"] = state[0] * float("inf")
    elif case == "explicit_effort":
        arguments["energy_effort"] = torch.ones_like(state)
    elif case == "stored_effort":
        spec = _zero_poisson_spec()
    with pytest.raises((ValueError, RuntimeError), match=error):
        spec.canonical_drift(**arguments)


def _scalar_iphs_port(
    name,
    *,
    role=None,
    effort_n_dofs=1,
    flow_n_dofs=None,
):
    from relic.computation.operators.dec.forms import CochainSpaceSpec
    from relic.computation.operators.iphs import PortSpecIPHS

    effort_space = CochainSpaceSpec(
        complex_id="test_complex",
        degree=0,
        primality="dual",
        n_dofs=effort_n_dofs,
        units="J",
        extensivity="intensive",
        physical_domain="test_domain",
        semantic_name=f"{name}_effort",
        orientation_convention="positive_into_system",
        boundary_role="port",
    )
    flow_space = CochainSpaceSpec(
        complex_id="test_complex",
        degree=0,
        primality="primal",
        n_dofs=effort_n_dofs if flow_n_dofs is None else flow_n_dofs,
        units="s-1",
        extensivity="integral",
        physical_domain="test_domain",
        semantic_name=f"{name}_flow",
        orientation_convention="positive_into_system",
        boundary_role="port",
    )
    kwargs = {
        "name": name,
        "effort_var": f"{name}_effort",
        "flow_var": f"{name}_flow",
        "effort_space": effort_space,
        "flow_space": flow_space,
        "power_units": "W",
        "positive_power_convention": "into_system",
        "port_kind": "energy",
    }
    if role is not None:
        kwargs["role"] = role
    return PortSpecIPHS(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", ""),
        ("effort_var", ""),
        ("flow_var", ""),
        ("power_units", ""),
        ("positive_power_convention", "sideways"),
        ("port_kind", "unknown"),
    ],
)
def test_iphs_port_rejects_invalid_local_contract(field_name, value):
    from dataclasses import replace

    port = _scalar_iphs_port("invalid_local_contract", role="external")

    with pytest.raises(ValueError, match=field_name):
        replace(port, **{field_name: value})


def test_iphs_port_rejects_interior_only_support():
    from dataclasses import replace

    port = _scalar_iphs_port("interior_only", role="external")
    effort_space = replace(port.effort_space, boundary_role="interior")
    flow_space = replace(port.flow_space, boundary_role="interior")

    with pytest.raises(ValueError, match="boundary or port support"):
        replace(port, effort_space=effort_space, flow_space=flow_space)


def test_iphs_structure_accepts_skew_j_and_psd_m():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="damped_mode",
        state_vars=("x", "p"),
        energy_functional="0.5 * (x^2 + p^2)",
        entropy_functional="S",
        J=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        M=torch.diag(torch.tensor([0.0, 0.1])),
        G=None,
        stoichiometric_N=None,
        ports=(), poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    audit = spec.validate()

    assert audit["skew_symmetric"] is True
    assert audit["dissipation_psd"] is True
    assert audit["min_dissipation_eigenvalue"] == pytest.approx(0.0)


def test_iphs_structure_rejects_non_skew_j():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="bad_j",
        state_vars=("x", "p"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.ones(2, 2),
        M=torch.eye(2),
        G=None,
        stoichiometric_N=None,
        ports=(), poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="skew"):
        spec.validate()


def test_iphs_structure_rejects_negative_dissipation():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="bad_m",
        state_vars=("x", "p"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        M=torch.diag(torch.tensor([1.0, -0.1])),
        G=None,
        stoichiometric_N=None,
        ports=(), poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        spec.validate()


def test_assemble_iphs_structure_sums_declared_contributions():
    from relic.computation.operators.iphs import assemble_iphs_structure

    spec = assemble_iphs_structure(
        name="assembled_mode",
        state_vars=("x", "s"),
        energy_functional="quadratic_energy",
        entropy_functional="linear_entropy",
        contributions=[
            {"J": torch.tensor([[0.0, 1.0], [-1.0, 0.0]])},
            {"M": torch.diag(torch.tensor([0.0, 0.5]))},
        ],
        validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    assert torch.allclose(spec.J, torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))
    assert torch.allclose(spec.M, torch.diag(torch.tensor([0.0, 0.5])))


def test_assemble_iphs_structure_embeds_local_state_contributions():
    from relic.computation.operators.iphs import assemble_iphs_structure

    spec = assemble_iphs_structure(
        name="local_embed",
        state_vars=("x", "s", "w"),
        energy_functional="summed_energy",
        entropy_functional="summed_entropy",
        contributions=[
            {
                "state_vars": ("s", "x"),
                "J": torch.tensor([[0.0, 2.0], [-2.0, 0.0]]),
                "M": torch.diag(torch.tensor([0.3, 0.1])),
                "energy_gradient": torch.tensor([3.0, 1.0]),
            },
            {
                "state_vars": ("w",),
                "M": torch.tensor([[0.5]]),
                "energy_gradient": torch.tensor([5.0]),
            },
        ], poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    assert torch.allclose(
        spec.J,
        torch.tensor(
            [
                [0.0, -2.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.allclose(spec.M, torch.diag(torch.tensor([0.1, 0.3, 0.5])))
    assert torch.allclose(spec.energy_gradient, torch.tensor([1.0, 3.0, 5.0]))


def test_assemble_iphs_structure_rejects_unknown_local_state_var():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="subset"):
        assemble_iphs_structure(
            name="bad_state_order",
            state_vars=("x", "s"),
            energy_functional="quadratic_energy",
            entropy_functional="linear_entropy",
            contributions=[
                {
                    "state_vars": ("s", "missing"),
                    "J": torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
                }
            ], poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_assemble_iphs_structure_adds_power_and_dissipative_port_interconnections():
    from relic.computation.operators.iphs import assemble_iphs_structure

    spec = assemble_iphs_structure(
        name="port_connected",
        state_vars=("x", "z"),
        energy_functional="E",
        entropy_functional="S",
        contributions=[
            {
                "state_vars": ("x",),
                "G": torch.tensor([[1.0]]),
                "ports": (_scalar_iphs_port("x_port", role="internal"),),
            },
            {
                "state_vars": ("z",),
                "G": torch.tensor([[1.0]]),
                "ports": (_scalar_iphs_port("z_port", role="internal"),),
            },
        ],
        port_interconnection_C=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        port_dissipation_D=torch.diag(torch.tensor([0.2, 0.4])),
        validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    assert torch.allclose(spec.G, torch.eye(2))
    assert torch.allclose(spec.J, torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))
    assert torch.allclose(spec.R, torch.diag(torch.tensor([0.2, 0.4])))
    assert torch.count_nonzero(spec.M) == 0


def test_port_feedback_without_declared_port_roles_is_rejected():
    """C and D act on internal ports; ports without roles are not assumed internal."""
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="explicit internal"):
        assemble_iphs_structure(
            name="unroled_feedback",
            state_vars=("x", "z"),
            energy_functional="E",
            entropy_functional="S",
            contributions=[{"G": torch.eye(2)}],
            port_dissipation_D=torch.diag(torch.tensor([0.2, 0.4])),
            validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_assemble_iphs_structure_partitions_internal_feedback_and_external_input():
    from relic.computation.operators.iphs import assemble_iphs_structure

    energy_effort = torch.tensor([2.0, 3.0])
    spec = assemble_iphs_structure(
        name="partitioned_ports",
        state_vars=("x", "z"),
        energy_functional="E",
        entropy_functional="S",
        formulation="dissipative_ph",
        contributions=[
            {
                "G": torch.eye(2),
                "ports": (
                    _scalar_iphs_port("internal_loss", role="internal"),
                    _scalar_iphs_port("external_forcing", role="external"),
                ),
                "energy_gradient": energy_effort,
            }
        ],
        port_dissipation_D=torch.tensor([[0.25]]),
        validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    drift = spec.canonical_drift(port_input=torch.tensor([4.0]))

    assert spec.internal_port_indices == (0,)
    assert spec.external_port_indices == (1,)
    assert torch.allclose(spec.R, torch.diag(torch.tensor([0.25, 0.0])))
    assert torch.allclose(drift, torch.tensor([-0.5, 4.0]))


def test_assemble_iphs_structure_rejects_feedback_shape_for_all_ports():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="internal port count"):
        assemble_iphs_structure(
            name="mismatched_feedback_ports",
            state_vars=("x", "z"),
            energy_functional="E",
            entropy_functional="S",
            contributions=[
                {
                    "G": torch.eye(2),
                    "ports": (
                        _scalar_iphs_port("internal_loss", role="internal"),
                        _scalar_iphs_port("external_forcing", role="external"),
                    ),
                }
            ],
            port_dissipation_D=torch.eye(2), poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_strict_iphs_rejects_unspecified_port_role():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="explicit internal or external role"):
        assemble_iphs_structure(
            name="strict_unspecified_port_role",
            state_vars=("x",),
            energy_functional="E",
            entropy_functional="S",
            formulation="dissipative_ph",
            contributions=[
                {
                    "runtime_owner": "test.iphs.strict_port_role",
                    "G": torch.ones((1, 1)),
                    "ports": (_scalar_iphs_port("ambiguous_port"),),
                    "energy_gradient": torch.ones(1),
                }
            ],
            strict=True,
            required_level=4,
            validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_strict_iphs_zero_g_still_requires_explicit_port_role():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="explicit internal or external role"):
        assemble_iphs_structure(
            name="strict_zero_g_unspecified_port_role",
            state_vars=("x",),
            energy_functional="E",
            entropy_functional="S",
            formulation="dissipative_ph",
            contributions=[
                {
                    "runtime_owner": "test.iphs.strict_zero_port_role",
                    "G": torch.zeros((1, 1)),
                    "ports": (_scalar_iphs_port("ambiguous_zero_port"),),
                    "R": torch.ones((1, 1)),
                    "energy_gradient": torch.ones(1),
                }
            ],
            strict=True,
            required_level=4,
            validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_iphs_port_width_is_defined_by_typed_cochain_dofs():
    from relic.computation.operators.iphs import assemble_iphs_structure

    vector_port = _scalar_iphs_port(
        "vector_boundary",
        role="external",
        effort_n_dofs=2,
    )
    with pytest.raises(ValueError, match="typed port degrees of freedom"):
        assemble_iphs_structure(
            name="bad_vector_port_width",
            state_vars=("x",),
            energy_functional="E",
            entropy_functional="S",
            contributions=[{"G": torch.ones((1, 1)), "ports": (vector_port,)}], poisson_operator="matrix_snapshot", reaction_representation="independent",
        )

    spec = assemble_iphs_structure(
        name="vector_port_width",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        formulation="dissipative_ph",
        contributions=[
            {
                "G": torch.tensor([[1.0, 2.0]]),
                "ports": (vector_port,),
                "energy_gradient": torch.ones(1),
            }
        ], poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    assert spec.external_port_indices == (0, 1)
    assert torch.allclose(
        spec.canonical_drift(port_input=torch.tensor([3.0, 4.0])),
        torch.tensor([11.0]),
    )


def test_iphs_port_rejects_effort_flow_dof_mismatch():
    with pytest.raises(ValueError, match="effort and flow spaces must have equal n_dofs"):
        _scalar_iphs_port(
            "mismatched_port",
            role="external",
            effort_n_dofs=2,
            flow_n_dofs=1,
        )


def test_assemble_iphs_structure_uses_internal_and_external_dof_blocks():
    from relic.computation.operators.iphs import assemble_iphs_structure

    internal_port = _scalar_iphs_port(
        "internal_vector",
        role="internal",
        effort_n_dofs=2,
    )
    external_port = _scalar_iphs_port(
        "external_vector",
        role="external",
        effort_n_dofs=3,
    )
    spec = assemble_iphs_structure(
        name="vector_port_partition",
        state_vars=("x", "z"),
        energy_functional="E",
        entropy_functional="S",
        formulation="dissipative_ph",
        contributions=[
            {
                "G": torch.tensor(
                    [
                        [1.0, 0.0, 1.0, 2.0, 3.0],
                        [0.0, 1.0, 4.0, 5.0, 6.0],
                    ]
                ),
                "ports": (internal_port, external_port),
                "energy_gradient": torch.tensor([1.0, 1.0]),
            }
        ],
        port_interconnection_C=torch.tensor([[0.0, 0.5], [-0.5, 0.0]]),
        port_dissipation_D=torch.diag(torch.tensor([0.25, 0.5])), poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    assert spec.internal_port_indices == (0, 1)
    assert spec.external_port_indices == (2, 3, 4)
    expected = (spec.J - spec.R) @ torch.ones(2)
    expected = expected + spec.G[:, 2:] @ torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(
        spec.canonical_drift(port_input=torch.tensor([1.0, 2.0, 3.0])),
        expected,
    )

    with pytest.raises(ValueError, match="internal port count"):
        assemble_iphs_structure(
            name="bad_internal_dof_feedback",
            state_vars=("x", "z"),
            energy_functional="E",
            entropy_functional="S",
            contributions=[
                {
                    "G": torch.ones((2, 5)),
                    "ports": (internal_port, external_port),
                }
            ],
            port_dissipation_D=torch.eye(1), poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_assemble_iphs_structure_rejects_cross_contribution_port_pairing():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="same contribution"):
        assemble_iphs_structure(
            name="cross_contribution_port_pairing",
            state_vars=("x",),
            energy_functional="E",
            entropy_functional="S",
            contributions=[
                {"G": torch.ones((1, 1))},
                {"ports": (_scalar_iphs_port("late_port", role="external"),)},
            ], poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_iphs_canonical_drift_rejects_unresolved_or_internal_port_input():
    from relic.computation.operators.iphs import IPHSStructureSpec

    common = {
        "state_vars": ("x",),
        "energy_functional": "E",
        "entropy_functional": "S",
        "J": torch.zeros((1, 1)),
        "M": torch.zeros((1, 1)),
        "G": torch.ones((1, 1)),
        "stoichiometric_N": None,
        "energy_gradient": torch.ones(1),
        "formulation": "dissipative_ph", "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
    }
    unresolved = IPHSStructureSpec(
        name="unresolved_port",
        ports=(),
        **common,
    )
    internal = IPHSStructureSpec(
        name="internal_port",
        ports=(_scalar_iphs_port("internal_only", role="internal"),),
        **common,
    )
    unspecified = IPHSStructureSpec(
        name="unspecified_port",
        ports=(_scalar_iphs_port("unspecified_only"),),
        **common,
    )

    with pytest.raises(ValueError, match="explicit port role partition"):
        unresolved.canonical_drift(port_input=torch.ones(1))
    with pytest.raises(ValueError, match="explicit port role partition"):
        unspecified.canonical_drift(port_input=torch.ones(1))
    with pytest.raises(ValueError, match="no external G column"):
        internal.canonical_drift(port_input=torch.ones(1))


def test_iphs_canonical_drift_rejects_dtype_mismatches_before_matmul():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="dtype_checked_drift",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1), dtype=torch.float32),
        M=torch.zeros((1, 1), dtype=torch.float32),
        G=torch.ones((1, 1), dtype=torch.float32),
        stoichiometric_N=torch.ones((1, 1), dtype=torch.float32),
        ports=(_scalar_iphs_port("dtype_external", role="external"),),
        energy_gradient=torch.ones(1, dtype=torch.float32),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="energy_effort.*dtype and device"):
        spec.canonical_drift(
            energy_effort=torch.ones(1, dtype=torch.float64),
            port_input=torch.ones(1),
            reaction_rates=torch.ones(1),
        )
    with pytest.raises(ValueError, match="port_input.*dtype and device"):
        spec.canonical_drift(
            port_input=torch.ones(1, dtype=torch.float64),
            reaction_rates=torch.ones(1),
        )
    with pytest.raises(ValueError, match="reaction_rates.*dtype and device"):
        spec.canonical_drift(
            port_input=torch.ones(1, dtype=torch.float32),
            reaction_rates=torch.ones(1, dtype=torch.float64),
        )


def test_level5_assembly_rejects_declared_port_dissipation_even_when_zero():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="GENERIC.*port_dissipation_D"):
        assemble_iphs_structure(
            name="generic_with_declared_d",
            state_vars=("x", "s"),
            energy_functional="E",
            entropy_functional="S",
            formulation="generic",
            reversible_operator_state_dependence="constant",
            contributions=[
                {
                    "runtime_owner": _EXECUTABLE_IPHS_SOURCE,
                    "J": torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
                    "M": torch.diag(torch.tensor([0.0, 1.0])),
                    "G": torch.ones((2, 1)),
                    "ports": (_scalar_iphs_port("generic_internal", role="internal"),),
                    "energy_gradient": torch.tensor([1.0, 0.0]),
                    "entropy_gradient": torch.tensor([0.0, 0.0]),
                }
            ],
            port_dissipation_D=torch.zeros((1, 1)),
            strict=True,
            required_level=5,
            validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_assemble_iphs_structure_rejects_non_skew_port_interconnection():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="skew"):
        assemble_iphs_structure(
            name="bad_port_interconnection",
            state_vars=("x", "z"),
            energy_functional="E",
            entropy_functional="S",
            contributions=[
                {
                    "G": torch.eye(2),
                    "ports": (
                        _scalar_iphs_port("x_port", role="internal"),
                        _scalar_iphs_port("z_port", role="internal"),
                    ),
                }
            ],
            port_interconnection_C=torch.ones(2, 2), poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_advance_reaction_network_computes_explicit_nh_rh_state_derivative():
    from relic.computation.operators.iphs import advance_reaction_network

    rates = torch.tensor(
        [
            [2.0, 3.0],
            [0.5, 1.0],
        ]
    )
    stoichiometric_n = torch.tensor(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
            [0.0, 1.0],
        ]
    )

    advance = advance_reaction_network(
        name="chain_reaction",
        species=("A", "B", "C"),
        reactions=("A_to_B", "B_to_C"),
        stoichiometric_N=stoichiometric_n,
        rates=rates,
        dt=10.0,
    )

    expected_derivative = torch.tensor(
        [
            [-2.0, -3.0],
            [1.5, 2.0],
            [0.5, 1.0],
        ]
    )
    assert torch.allclose(advance.state_derivative, expected_derivative)
    assert torch.allclose(advance.state_delta, expected_derivative * 10.0)
    assert torch.allclose(advance.species_rate("B"), torch.tensor([1.5, 2.0]))


def test_advance_reaction_network_validates_conserved_element_matrix():
    from relic.computation.operators.iphs import advance_reaction_network

    advance = advance_reaction_network(
        name="closed_chain",
        species=("A", "B", "C"),
        reactions=("A_to_B", "B_to_C"),
        stoichiometric_N=torch.tensor(
            [
                [-1.0, 0.0],
                [1.0, -1.0],
                [0.0, 1.0],
            ]
        ),
        rates=torch.tensor([2.0, 0.5]),
    )

    audit = advance.validate_conservation(torch.ones(1, 3))

    assert audit["passed"] is True
    assert audit["max_abs_residual"] == pytest.approx(0.0)


def test_advance_reaction_network_auto_validates_declared_conservation_matrix():
    from relic.computation.operators.iphs import advance_reaction_network

    advance = advance_reaction_network(
        name="closed_runtime_chain",
        species=("A", "B"),
        reactions=("A_to_B",),
        stoichiometric_N=torch.tensor([[-1.0], [1.0]]),
        rates=torch.tensor([2.0]),
        conservation_matrix=torch.ones(1, 2),
    )

    assert advance.conservation_audit is not None
    assert advance.conservation_audit["passed"] is True


def test_advance_reaction_network_rejects_declared_nonconservative_matrix():
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="C_h N_h"):
        advance_reaction_network(
            name="bad_runtime_chain",
            species=("A", "B"),
            reactions=("loss",),
            stoichiometric_N=torch.tensor([[-1.0], [0.0]]),
            rates=torch.tensor([2.0]),
            conservation_matrix=torch.ones(1, 2),
        )


@pytest.mark.parametrize(
    "conservation_matrix",
    [
        torch.zeros((1, 1)),
        torch.empty((0, 1)),
    ],
)
def test_advance_reaction_network_rejects_vacuous_conservation_ledgers(
    conservation_matrix,
):
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="non-empty|nonzero"):
        advance_reaction_network(
            name="vacuous_closed_ledger",
            species=("A",),
            reactions=("source",),
            stoichiometric_N=torch.ones((1, 1)),
            rates=torch.ones(1),
            conservation_matrix=conservation_matrix,
            require_conservation_matrix=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stoichiometric_N", torch.tensor([[float("nan")]])),
        ("rates", torch.tensor([float("nan")])),
        ("dt", float("nan")),
        ("dt", 0.0),
        ("dt", -1.0),
        ("dt", True),
        ("conservation_tolerance", float("nan")),
        ("conservation_tolerance", -1.0),
        ("conservation_tolerance", True),
    ],
)
def test_advance_reaction_network_rejects_invalid_numeric_domains(field, value):
    from relic.computation.operators.iphs import advance_reaction_network

    kwargs = {
        "name": "invalid_numeric_reaction",
        "species": ("A",),
        "reactions": ("source",),
        "stoichiometric_N": torch.ones((1, 1)),
        "rates": torch.ones(1),
        "dt": 1.0,
        "conservation_matrix": torch.ones((1, 1)),
        "strict_conservation": False,
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match="finite|positive|real scalar"):
        advance_reaction_network(**kwargs)


@pytest.mark.parametrize(
    ("stoichiometric_n", "rates", "dt"),
    [
        (
            torch.tensor([[1.0e38]], dtype=torch.float32),
            torch.tensor([1.0e38], dtype=torch.float32),
            1.0,
        ),
        (
            torch.ones((1, 1), dtype=torch.float32),
            torch.tensor([1.0e38], dtype=torch.float32),
            1.0e38,
        ),
    ],
)
def test_advance_reaction_network_rejects_nonfinite_derived_state(
    stoichiometric_n,
    rates,
    dt,
):
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="non-finite"):
        advance_reaction_network(
            name="overflowing_reaction",
            species=("A",),
            reactions=("source",),
            stoichiometric_N=stoichiometric_n,
            rates=rates,
            dt=dt,
        )


def test_advance_reaction_network_records_missing_conservation_matrix():
    from relic.computation.operators.iphs import advance_reaction_network

    advance = advance_reaction_network(
        name="undeclared_runtime_chain",
        species=("A", "B"),
        reactions=("A_to_B",),
        stoichiometric_N=torch.tensor([[-1.0], [1.0]]),
        rates=torch.tensor([2.0]),
    )

    assert advance.conservation_audit is not None
    assert advance.conservation_audit["validated"] is False
    assert advance.conservation_audit["reason"] == "conservation_matrix_not_declared"


def test_advance_reaction_network_can_require_conservation_matrix():
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="requires declared conservation_matrix"):
        advance_reaction_network(
            name="strict_undeclared_runtime_chain",
            species=("A", "B"),
            reactions=("A_to_B",),
            stoichiometric_N=torch.tensor([[-1.0], [1.0]]),
            rates=torch.tensor([2.0]),
            require_conservation_matrix=True,
        )


def test_advance_reaction_network_strict_open_system_requires_route_declaration():
    from relic.computation.operators.iphs import advance_reaction_network

    advance = advance_reaction_network(
        name="managed_export_runtime_chain",
        species=("soil_pool", "managed_export"),
        reactions=("export",),
        stoichiometric_N=torch.tensor([[-1.0], [1.0]]),
        rates=torch.tensor([2.0]),
        require_conservation_matrix=True,
        open_system_route="managed_export",
    )

    assert advance.conservation_audit is not None
    assert advance.conservation_audit["validated"] is False
    assert advance.conservation_audit["reason"] == "open_system_route_declared"
    assert advance.conservation_audit["open_system_route"] == "managed_export"


def test_advance_reaction_network_rejects_unknown_open_system_route():
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="open_system_route"):
        advance_reaction_network(
            name="bad_open_route",
            species=("A", "B"),
            reactions=("A_to_B",),
            stoichiometric_N=torch.tensor([[-1.0], [1.0]]),
            rates=torch.tensor([2.0]),
            open_system_route="hidden_sink",
        )


def test_advance_reaction_network_rejects_reaction_count_mismatch():
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="reaction count"):
        advance_reaction_network(
            name="bad_reaction_count",
            species=("A", "B"),
            reactions=("A_to_B",),
            stoichiometric_N=torch.tensor([[-1.0, 0.0], [1.0, -1.0]]),
            rates=torch.tensor([1.0]),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stoichiometric_N", __import__("numpy").array([[1.0 + 2.0j]])),
        ("rates", __import__("numpy").array([1.0 + 2.0j])),
        ("conservation_matrix", __import__("numpy").array([[1.0 + 2.0j]])),
        ("dt", __import__("numpy").array(1.0 + 2.0j)),
    ],
)
def test_advance_reaction_network_rejects_non_tensor_complex_inputs(field, value):
    from relic.computation.operators.iphs import advance_reaction_network

    kwargs = {
        "name": "complex_reaction_input",
        "species": ("A",),
        "reactions": ("source",),
        "stoichiometric_N": torch.ones((1, 1)),
        "rates": torch.ones(1),
        "dt": 1.0,
        "conservation_matrix": torch.ones((1, 1)),
        "strict_conservation": False,
    }
    kwargs[field] = value

    with pytest.raises(TypeError, match="must be real"):
        advance_reaction_network(**kwargs)


def test_reaction_point_audit_uses_aggregate_affinity_rate_product():
    from relic.computation.operators.iphs import advance_reaction_network

    # References:
    #   - Rao and Esposito (2016), thermodynamics of chemical reaction networks.
    # Mathematical expression:
    #   P = A.T r = (-1)(2) + (3)(1) = 1 >= 0.
    # Parameter meanings:
    #   A is physical affinity; r gives the coupled A->B and B->C rates.
    # Units:
    #   A [J mol-1], r [mol s-1], P [J s-1].
    # Domain/constraints:
    #   This is an aggregate numerical audit, not a kinetic-law certificate.
    # Sign convention:
    #   Only total power is required nonnegative; one channel may be uphill.
    advance = advance_reaction_network(
        name="aggregate_affinity_audit",
        species=("A", "B", "C"),
        reactions=("A_to_B", "B_to_C"),
        stoichiometric_N=torch.tensor(
            [[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]],
            dtype=torch.float64,
        ),
        rates=torch.tensor([2.0, 1.0], dtype=torch.float64),
        affinities=torch.tensor([-1.0, 3.0], dtype=torch.float64),
        affinity_sign_convention="affinity_times_rate_nonnegative",
        conservation_matrix=torch.ones((1, 3), dtype=torch.float64),
        require_thermodynamic_point_audit=True,
    )

    assert advance.thermodynamic_audit is not None
    assert advance.thermodynamic_audit["passed"] is True
    assert advance.thermodynamic_audit["affinity_rate_power_min"] == pytest.approx(1.0)
    assert advance.thermodynamic_audit["componentwise_required"] is False


@pytest.mark.parametrize(
    ("affinities", "convention", "error"),
    [
        (torch.tensor([-2.0], dtype=torch.float64), "affinity_times_rate_nonnegative", "affinity-rate power"),
        (torch.tensor([float("nan")], dtype=torch.float64), "affinity_times_rate_nonnegative", "finite"),
        (torch.tensor([float("inf")], dtype=torch.float64), "affinity_times_rate_nonnegative", "finite"),
        (torch.tensor([1.0], dtype=torch.float64), None, "sign convention"),
        (torch.tensor([1.0], dtype=torch.float64), "rate_times_negative_affinity", "sign convention"),
    ],
)
def test_reaction_point_audit_rejects_negative_or_ambiguous_affinity_contract(
    affinities,
    convention,
    error,
):
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match=error):
        advance_reaction_network(
            name="invalid_affinity_contract",
            species=("A", "B"),
            reactions=("A_to_B",),
            stoichiometric_N=torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
            rates=torch.tensor([2.0], dtype=torch.float64),
            affinities=affinities,
            affinity_sign_convention=convention,
            conservation_matrix=torch.ones((1, 2), dtype=torch.float64),
            require_thermodynamic_point_audit=True,
        )


def test_reaction_point_audit_rejects_affinity_broadcasting():
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="affinities shape"):
        advance_reaction_network(
            name="broadcast_affinity",
            species=("A", "B"),
            reactions=("forward", "reverse"),
            stoichiometric_N=torch.tensor(
                [[-1.0, 1.0], [1.0, -1.0]],
                dtype=torch.float64,
            ),
            rates=torch.ones((2, 3), dtype=torch.float64),
            affinities=torch.ones((2, 1), dtype=torch.float64),
            affinity_sign_convention="affinity_times_rate_nonnegative",
            conservation_matrix=torch.ones((1, 2), dtype=torch.float64),
            require_thermodynamic_point_audit=True,
        )


def test_reaction_point_power_preserves_batches_gradients_and_amount_ledger():
    from relic.computation.operators.iphs import advance_reaction_network

    # References:
    #   - Rao and Esposito (2016), Phys. Rev. X 6, 041064, Eq. (55).
    # Mathematical expression:
    #   P = A.T r = [1, 2]; dx = N r dt; C dx = 0.
    # Parameter meanings:
    #   N maps A->B and B->C; C sums species; each column is one batch.
    # Units:
    #   A [J mol-1], r [mol s-1], P [J s-1], dt [s], dx [mol].
    # Domain/constraints:
    #   Two finite channels; no temperature or entropy binding is supplied.
    # Sign convention:
    #   Negative species increments leave donors; positive P is dissipative.
    rates = torch.tensor([[2.0, 4.0], [1.0, 2.0]], requires_grad=True)
    affinities = torch.tensor([[-1.0, -1.0], [3.0, 3.0]], requires_grad=True)
    advance = advance_reaction_network(
        name="batched_physical_power",
        species=("A", "B", "C"),
        reactions=("A_to_B", "B_to_C"),
        stoichiometric_N=torch.tensor([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]]),
        rates=rates,
        dt=0.5,
        affinities=affinities,
        affinity_sign_convention="affinity_times_rate_nonnegative",
        conservation_matrix=torch.ones((1, 3)),
        require_thermodynamic_point_audit=True,
    )

    audit = advance.thermodynamic_audit
    assert audit is not None
    assert "entropy_production" not in audit
    power = audit["affinity_rate_power"]
    torch.testing.assert_close(power, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(
        advance.state_delta, torch.tensor([[-1.0, -2.0], [0.5, 1.0], [0.5, 1.0]])
    )
    torch.testing.assert_close(advance.state_delta.sum(dim=0), torch.zeros(2))
    power.sum().backward()
    torch.testing.assert_close(rates.grad, affinities.detach())
    torch.testing.assert_close(affinities.grad, rates.detach())


@pytest.mark.parametrize("second_rate", [1.0e308, -1.0e308])
def test_reaction_power_reduction_rejects_overflow(second_rate):
    from relic.computation.operators.iphs import advance_reaction_network

    # References:
    #   - IEEE 754-2019; Rao and Esposito (2016), PRX 6, 041064.
    # Mathematical expression:
    #   P = sum(A*r); scale = sum(abs(A*r)); finite terms can overflow a sum.
    # Parameter meanings:
    #   Two channels have bounded N*r but overflow P or its tolerance scale.
    # Units:
    #   A [J mol-1], r [mol s-1], P and scale [J s-1].
    # Domain/constraints:
    #   float64 terms are finite; non-finite reductions must be rejected.
    # Sign convention:
    #   Signed rates permit cancellation, which does not make an infinite scale valid.
    with pytest.raises(ValueError, match="affinity-rate power reduction"):
        advance_reaction_network(
            name="overflowing_power_reduction",
            species=("A", "B"),
            reactions=("first", "second"),
            stoichiometric_N=torch.tensor([[-0.25, -0.25], [0.25, 0.25]]),
            rates=torch.tensor([1.0e308, second_rate]),
            affinities=torch.ones(2),
            affinity_sign_convention="affinity_times_rate_nonnegative",
            conservation_matrix=torch.ones((1, 2)),
            require_thermodynamic_point_audit=True,
        )


def test_reaction_point_audit_zero_scale_has_zero_tolerance():
    from relic.computation.operators.iphs import advance_reaction_network

    advance = advance_reaction_network(
        name="zero_scale_affinity",
        species=("A", "B"),
        reactions=("A_to_B",),
        stoichiometric_N=torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
        rates=torch.tensor([2.0], dtype=torch.float64),
        affinities=torch.zeros(1, dtype=torch.float64),
        affinity_sign_convention="affinity_times_rate_nonnegative",
        conservation_matrix=torch.ones((1, 2), dtype=torch.float64),
        require_thermodynamic_point_audit=True,
    )

    assert advance.thermodynamic_audit is not None
    assert advance.thermodynamic_audit["affinity_rate_power_scale_max"] == pytest.approx(0.0)
    assert advance.thermodynamic_audit["affinity_rate_power_tolerance_max"] == pytest.approx(0.0)
    assert advance.thermodynamic_audit["zero_scale"] is True


def test_reaction_point_audit_rejects_rtol_below_dtype_precision_floor():
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="precision floor"):
        advance_reaction_network(
            name="float32_false_precision",
            species=("A", "B"),
            reactions=("A_to_B",),
            stoichiometric_N=torch.tensor([[-1.0], [1.0]], dtype=torch.float32),
            rates=torch.ones(1, dtype=torch.float32),
            affinities=torch.ones(1, dtype=torch.float32),
            affinity_sign_convention="affinity_times_rate_nonnegative",
            conservation_matrix=torch.ones((1, 2), dtype=torch.float32),
            require_thermodynamic_point_audit=True,
        )


def test_reaction_point_audit_rejects_nonfinite_affinity_rate_product():
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="affinity-rate product.*finite"):
        advance_reaction_network(
            name="overflowing_affinity_product",
            species=("A", "B"),
            reactions=("A_to_B",),
            stoichiometric_N=torch.tensor([[-1.0], [1.0]], dtype=torch.float32),
            rates=torch.tensor([1.0e30], dtype=torch.float32),
            affinities=torch.tensor([1.0e30], dtype=torch.float32),
            affinity_sign_convention="affinity_times_rate_nonnegative",
            conservation_matrix=torch.ones((1, 2), dtype=torch.float32),
            thermodynamic_rtol=1.0e-5,
            require_thermodynamic_point_audit=True,
        )


def test_open_reaction_point_audit_validates_ledger_without_double_counting():
    from relic.computation.operators.iphs import advance_reaction_network

    # References:
    #   - Rao and Esposito (2016), open chemical reaction networks.
    # Mathematical expression:
    #   C N = 0 and C q_open = ledger_rate; x_dot remains N r here.
    # Parameter meanings and units:
    #   q_open is the separately routed state-space exchange rate and C maps it
    #   to the declared ledger-rate basis. This helper audits but does not add q_open.
    advance = advance_reaction_network(
        name="managed_open_reaction",
        species=("A", "B"),
        reactions=("A_to_B",),
        stoichiometric_N=torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
        rates=torch.tensor([2.0], dtype=torch.float64),
        affinities=torch.tensor([1.0], dtype=torch.float64),
        affinity_sign_convention="affinity_times_rate_nonnegative",
        conservation_matrix=torch.ones((1, 2), dtype=torch.float64),
        open_system_route="managed_export",
        open_exchange=torch.tensor([-1.0, 0.0], dtype=torch.float64),
        open_ledger_rate=torch.tensor([-1.0], dtype=torch.float64),
        require_thermodynamic_point_audit=True,
    )

    assert advance.open_system_audit is not None
    assert advance.open_system_audit["passed"] is True
    assert advance.open_system_audit["route"] == "managed_export"
    assert torch.allclose(
        advance.state_derivative,
        torch.tensor([-2.0, 2.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("open_exchange", None, "open_exchange"),
        ("open_ledger_rate", None, "open_ledger_rate"),
        ("open_system_route", None, "open_system_route"),
        ("conservation_matrix", None, "conservation_matrix"),
        ("open_system_route", "observation", "cannot satisfy"),
        ("open_system_route", "diagnostic_residual", "cannot satisfy"),
    ],
)
def test_open_reaction_point_audit_requires_complete_supported_contract(
    field,
    value,
    error,
):
    from relic.computation.operators.iphs import advance_reaction_network

    kwargs = {
        "name": "incomplete_open_reaction",
        "species": ("A", "B"),
        "reactions": ("A_to_B",),
        "stoichiometric_N": torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
        "rates": torch.tensor([2.0], dtype=torch.float64),
        "affinities": torch.tensor([1.0], dtype=torch.float64),
        "affinity_sign_convention": "affinity_times_rate_nonnegative",
        "conservation_matrix": torch.ones((1, 2), dtype=torch.float64),
        "open_system_route": "managed_export",
        "open_exchange": torch.tensor([-1.0, 0.0], dtype=torch.float64),
        "open_ledger_rate": torch.tensor([-1.0], dtype=torch.float64),
        "require_thermodynamic_point_audit": True,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=error):
        advance_reaction_network(**kwargs)


def test_open_reaction_point_audit_rejects_broadcast_or_wrong_ledger():
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="open_ledger_rate shape"):
        advance_reaction_network(
            name="broadcast_open_ledger",
            species=("A", "B"),
            reactions=("A_to_B",),
            stoichiometric_N=torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
            rates=torch.ones((1, 2), dtype=torch.float64),
            affinities=torch.ones((1, 2), dtype=torch.float64),
            affinity_sign_convention="affinity_times_rate_nonnegative",
            conservation_matrix=torch.ones((1, 2), dtype=torch.float64),
            open_system_route="managed_export",
            open_exchange=torch.tensor(
                [[-1.0, -2.0], [0.0, 0.0]],
                dtype=torch.float64,
            ),
            open_ledger_rate=torch.tensor([[-1.0]], dtype=torch.float64),
            require_thermodynamic_point_audit=True,
        )


def test_open_reaction_point_audit_does_not_hide_internal_nonconservation():
    from relic.computation.operators.iphs import advance_reaction_network

    with pytest.raises(ValueError, match="C_h N_h"):
        advance_reaction_network(
            name="open_route_cannot_hide_internal_loss",
            species=("A", "B"),
            reactions=("loss",),
            stoichiometric_N=torch.tensor([[-1.0], [0.0]], dtype=torch.float64),
            rates=torch.tensor([2.0], dtype=torch.float64),
            affinities=torch.tensor([1.0], dtype=torch.float64),
            affinity_sign_convention="affinity_times_rate_nonnegative",
            conservation_matrix=torch.ones((1, 2), dtype=torch.float64),
            strict_conservation=False,
            open_system_route="managed_export",
            open_exchange=torch.zeros(2, dtype=torch.float64),
            open_ledger_rate=torch.zeros(1, dtype=torch.float64),
            require_thermodynamic_point_audit=True,
        )


def test_strict_iphs_assembly_rejects_empty_contributions():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="runtime-linked contribution"):
        assemble_iphs_structure(
            name="empty_strict_assembly",
            state_vars=("x",),
            energy_functional="E",
            entropy_functional="S",
            contributions=[],
            strict=True,
            required_level=4,
            validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_strict_iphs_assembly_rejects_contribution_without_runtime_owner():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="runtime owner"):
        assemble_iphs_structure(
            name="unbound_strict_assembly",
            state_vars=("x",),
            energy_functional="E",
            entropy_functional="S",
            contributions=[{"M": torch.ones((1, 1))}],
            strict=True,
            required_level=4,
            validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_strict_iphs_assembly_rejects_required_level_downgrade():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="strict IPHS required_level cannot be lower than 4"):
        assemble_iphs_structure(
            name="downgraded_strict_assembly",
            state_vars=("x",),
            energy_functional="E",
            entropy_functional="S",
            formulation="dissipative_ph",
            contributions=[
                {
                    "runtime_owner": "test.iphs.downgraded_strict_assembly",
                    "R": torch.ones((1, 1)),
                    "energy_gradient": torch.ones(1),
                }
            ],
            strict=True,
            required_level=1,
            validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_strict_level4_iphs_requires_explicit_runtime_effort():
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match="explicit runtime energy_effort"):
        assemble_iphs_structure(
            name="missing_efforts",
            state_vars=("x", "s"),
            energy_functional="E",
            entropy_functional="S",
            formulation="dissipative_ph",
            contributions=[
                {
                    "runtime_owner": "test.iphs.missing_efforts",
                    "J": torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
                    "R": torch.eye(2),
                }
            ],
            strict=True,
            required_level=4,
            validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_level5_iphs_computes_and_rejects_nonzero_generic_residual():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="bad_generic",
        state_vars=("x", "s"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        M=torch.diag(torch.tensor([0.0, 1.0])),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.tensor([1.0, 0.0]),
        entropy_gradient=torch.tensor([1.0, 0.0]),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="generic",
        reversible_operator_state_dependence="constant", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="J e_S"):
        spec.validate(required_level=5)


def test_level5_rejects_state_dependent_poisson_sample_mislabeled_constant():
    from relic.computation.operators.iphs import IPHSStructureSpec

    # References:
    # - Edwards and Oettinger (1997), Phys. Rev. E 56, 4097.
    # - McLachlan et al. (1999), J. Comput. Appl. Math. 111, 181-191.
    # Mathematical expressions:
    #   L(x) = [[0, -1, x0], [1, 0, x1], [-x0, -x1, 0]],
    #   B = (-x1, x0, 1), L(x) B = 0, and B dot curl(B) = 2 != 0.
    # Parameter meanings:
    #   x is the state, L (stored as J) is the reversible operator, and B is
    #   the entropy gradient. The nonzero Jacobi residual makes L non-Poisson.
    state = torch.tensor([1.0, 2.0, 0.0])
    zero = torch.zeros((), dtype=state.dtype)
    one = torch.ones((), dtype=state.dtype)
    reversible_operator = torch.stack(
        (
            torch.stack((zero, -one, state[0])),
            torch.stack((one, zero, state[1])),
            torch.stack((-state[0], -state[1], zero)),
        )
    ).detach()
    entropy_gradient = torch.stack((-state[1], state[0], one)).detach()
    spec = IPHSStructureSpec(
        name="non_poisson_sample_mislabeled_constant",
        state_vars=("x0", "x1", "s"),
        energy_functional="E",
        entropy_functional="S",
        J=reversible_operator,
        M=torch.zeros((3, 3)),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.zeros(3),
        entropy_gradient=entropy_gradient,
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="generic",
        reversible_operator_state_dependence="constant", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    assert torch.allclose(reversible_operator @ entropy_gradient, torch.zeros(3))
    with pytest.raises(
        ValueError,
        match=r"Level 5 executable Jacobi identity verification remains \[N\]",
    ):
        spec.validate(required_level=5)


def test_iphs_canonical_drift_evaluates_all_declared_terms():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="canonical_drift",
        state_vars=("x", "s"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        M=torch.diag(torch.tensor([0.0, 3.0])),
        G=torch.tensor([[1.0], [2.0]]),
        stoichiometric_N=torch.tensor([[-1.0], [1.0]]),
        ports=(_scalar_iphs_port("canonical_external", role="external"),),
        energy_gradient=torch.tensor([1.0, 2.0]),
        entropy_gradient=torch.tensor([4.0, 5.0]),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="generic", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    drift = spec.canonical_drift(
        port_input=torch.tensor([2.0]),
        reaction_rates=torch.tensor([0.5]),
    )

    assert torch.allclose(drift, torch.tensor([3.5, 18.5]))


def test_iphs_canonical_drift_rejects_unspecified_formulation():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="unspecified_drift",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        M=torch.zeros((1, 1)),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.ones(1),
        entropy_gradient=torch.ones(1),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,), poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="explicit formulation"):
        spec.canonical_drift()


def test_strict_level4_rejects_unspecified_formulation():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="unspecified_strict",
        state_vars=("x", "s"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        M=torch.zeros((2, 2)),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.tensor([1.0, 1.0]),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,), poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="formulation"):
        spec.validate(required_level=4)


def test_dph_single_runtime_point_uses_resistance_and_dissipates_energy():
    from relic.computation.operators.iphs import IPHSStructureSpec

    energy_effort = torch.tensor([1.0, 1.0])
    spec = IPHSStructureSpec(
        name="strict_dissipative_ph",
        state_vars=("x", "p"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((2, 2)),
        M=torch.zeros((2, 2)),
        G=None,
        stoichiometric_N=None,
        ports=(),
        R=torch.eye(2),
        energy_gradient=energy_effort,
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    audit = spec.validate(energy_effort=energy_effort)
    drift = spec.canonical_drift()

    assert audit["formulation"] == "dissipative_ph"
    assert audit["evidence_scope"] == "single_runtime_point"
    assert audit["closed_port"] is True
    assert audit["dissipation_power"] == pytest.approx(2.0)
    assert audit["port_power_into_system"] == pytest.approx(0.0)
    assert audit["power_balance_residual"] == pytest.approx(0.0)
    assert "iphs_level" not in audit
    assert torch.allclose(drift, -energy_effort)
    assert torch.dot(energy_effort, drift).item() == pytest.approx(-2.0)

    with pytest.raises(ValueError, match=r"process runtime linkage remains \[N\]"):
        spec.validate(required_level=4, energy_effort=energy_effort)


def test_strict_level4_rejects_unbound_process_even_at_zero_effort():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="unbound_zero_effort",
        state_vars=("x", "p"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((2, 2)),
        M=torch.zeros((2, 2)),
        G=None,
        stoichiometric_N=None,
        ports=(),
        R=torch.eye(2),
        energy_gradient=torch.zeros(2),
        contribution_count=1,
        contribution_sources=("not.an.owner",),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match=r"process runtime linkage remains \[N\]"):
        spec.validate(required_level=4, energy_effort=torch.zeros(2))


def test_strict_level4_rejects_generic_mobility_energy_creation_counterexample():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="energy_creation_counterexample",
        state_vars=("x", "s"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((2, 2)),
        M=torch.eye(2),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.ones(2),
        entropy_gradient=torch.ones(2),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="mobility M"):
        spec.validate(required_level=4)


@pytest.mark.parametrize(
    ("required_level", "formulation"),
    [
        (4, "dissipative_ph"),
        (5, "generic"),
    ],
)
def test_strict_iphs_rejects_unbound_reaction_energy_entropy_terms(
    required_level,
    formulation,
):
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name=f"unbound_reaction_level_{required_level}",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        M=torch.zeros((1, 1)),
        G=None,
        stoichiometric_N=torch.ones((1, 1)),
        ports=(),
        R=torch.zeros((1, 1)),
        energy_gradient=torch.ones(1),
        entropy_gradient=torch.zeros(1),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation=formulation,
        reversible_operator_state_dependence="constant", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match=r"reaction closure linkage.*\[N\]"):
        spec.validate(required_level=required_level)


def test_level5_generic_rejects_dissipative_ph_formulation():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="dph_not_generic",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        M=torch.zeros((1, 1)),
        G=None,
        stoichiometric_N=None,
        ports=(),
        R=torch.ones((1, 1)),
        energy_gradient=torch.ones(1),
        entropy_gradient=torch.ones(1),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="generic"):
        spec.validate(required_level=5)


def test_level5_state_dependent_generic_requires_jacobi_evidence():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="generic_without_jacobi",
        state_vars=("x", "s"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((2, 2)),
        M=torch.diag(torch.tensor([0.0, 1.0])),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.tensor([1.0, 0.0]),
        entropy_gradient=torch.tensor([0.0, 1.0]),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="generic",
        reversible_operator_state_dependence="state_dependent", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="Jacobi"):
        spec.validate(required_level=5)


def test_level5_state_dependent_generic_rejects_unexecutable_jacobi_claim():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="generic_with_text_only_jacobi_claim",
        state_vars=("x", "s"),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((2, 2)),
        M=torch.diag(torch.tensor([0.0, 1.0])),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.tensor([1.0, 0.0]),
        entropy_gradient=torch.tensor([0.0, 1.0]),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="generic",
        reversible_operator_state_dependence="state_dependent",
        jacobi_identity_evidence="trust me", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(
        ValueError,
        match=r"Level 5 executable Jacobi identity verification remains \[N\]",
    ):
        spec.validate(required_level=5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("R", torch.ones((3, 3))),
        ("G", torch.ones((3, 1))),
        ("stoichiometric_N", torch.ones((3, 1))),
        ("energy_gradient", torch.ones(3)),
    ],
)
def test_iphs_structure_rejects_auxiliary_shape_mismatch(field, value):
    from relic.computation.operators.iphs import IPHSStructureSpec

    values = {
        "name": "bad_aux_shape",
        "state_vars": ("x", "s"),
        "energy_functional": "E",
        "entropy_functional": "S",
        "J": torch.zeros((2, 2)),
        "M": torch.zeros((2, 2)),
        "G": None,
        "stoichiometric_N": None,
        "ports": (), "poisson_operator": "matrix_snapshot", "reaction_representation": "independent",
    }
    values[field] = value

    with pytest.raises(ValueError, match="shape|row count|length"):
        IPHSStructureSpec(**values)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.int64])
def test_iphs_structure_rejects_non_real_floating_dtype(dtype):
    from relic.computation.operators.iphs import IPHSStructureSpec

    with pytest.raises(ValueError, match="real floating.*float32 or float64"):
        IPHSStructureSpec(
            name="non_real_floating_structure",
            state_vars=("x",),
            energy_functional="E",
            entropy_functional="S",
            J=torch.zeros((1, 1), dtype=dtype),
            R=torch.zeros((1, 1), dtype=dtype),
            M=torch.zeros((1, 1), dtype=dtype),
            G=None,
            stoichiometric_N=None,
            ports=(), poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


def test_canonical_drift_revalidates_mutated_tensor_contract():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="mutated_resistance",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        R=torch.ones((1, 1)),
        M=torch.zeros((1, 1)),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.ones(1),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )
    spec.R[0, 0] = -1.0

    with pytest.raises(ValueError, match="positive semidefinite"):
        spec.canonical_drift()


def test_canonical_drift_rejects_nonfinite_derived_result():
    from relic.computation.operators.iphs import IPHSStructureSpec

    # References:
    #   - van der Schaft (2022), dissipative port-Hamiltonian systems.
    # Mathematical expression:
    #   x_dot = (J - R)e_H; finite operands can overflow in R e_H.
    # Parameter meanings and units:
    #   R maps energy effort to state rate; this test uses scaled abstract units.
    huge = torch.tensor([1.0e38], dtype=torch.float32)
    spec = IPHSStructureSpec(
        name="overflowing_drift",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1), dtype=torch.float32),
        R=torch.tensor([[1.0e38]], dtype=torch.float32),
        M=torch.zeros((1, 1), dtype=torch.float32),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=huge,
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="dissipative_rate.*finite"):
        spec.canonical_drift()


def test_out_of_system_port_is_sign_normalized_exactly_once():
    from dataclasses import replace

    from relic.computation.operators.iphs import IPHSStructureSpec

    # References:
    #   - van der Schaft (2022), input-state-output port-Hamiltonian systems.
    # Mathematical expression:
    #   u_into = -u_out; x_dot = -R e_H + G u_into; P = y.T u_into.
    # Parameter meanings and units:
    #   e_H is effort, u_out is outward-positive flow, and P is power into system.
    effort = torch.tensor([2.0])
    outward_flow = torch.tensor([3.0])
    port = replace(
        _scalar_iphs_port("external_export", role="external"),
        positive_power_convention="out_of_system",
    )
    spec = IPHSStructureSpec(
        name="outward_power",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        R=torch.ones((1, 1)),
        M=torch.zeros((1, 1)),
        G=torch.ones((1, 1)),
        stoichiometric_N=None,
        ports=(port,),
        energy_gradient=effort,
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    drift = spec.canonical_drift(port_input=outward_flow)
    audit = spec.validate(energy_effort=effort, port_input=outward_flow)

    assert torch.allclose(drift, torch.tensor([-5.0]))
    assert audit["open_port_energy_rate"] == pytest.approx(-10.0)
    assert audit["dissipation_power"] == pytest.approx(4.0)
    assert audit["port_power_into_system"] == pytest.approx(-6.0)
    assert audit["power_balance_residual"] == pytest.approx(0.0)


def test_point_audit_requires_external_input_and_rejects_closed_input():
    from relic.computation.operators.iphs import IPHSStructureSpec

    external = IPHSStructureSpec(
        name="missing_external_input",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        R=torch.ones((1, 1)),
        M=torch.zeros((1, 1)),
        G=torch.ones((1, 1)),
        stoichiometric_N=None,
        ports=(_scalar_iphs_port("required_input", role="external"),),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )
    closed = IPHSStructureSpec(
        name="closed_with_input",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        R=torch.ones((1, 1)),
        M=torch.zeros((1, 1)),
        G=None,
        stoichiometric_N=None,
        ports=(),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="requires port_input"):
        external.validate(energy_effort=torch.ones(1))
    with pytest.raises(ValueError, match="no external G column"):
        closed.validate(
            energy_effort=torch.ones(1),
            port_input=torch.ones(1),
        )


def test_open_port_point_audit_computes_signed_energy_balance():
    from relic.computation.operators.iphs import IPHSStructureSpec

    # References:
    #   - van der Schaft (2022), passivity balance for dissipative pH systems.
    # Mathematical expression:
    #   dH/dt = -e_H.T R e_H + (G.T e_H).T u.
    # Parameter meanings and units:
    #   e_H=2, R=1, G=1, and u=3 use a consistent abstract power basis.
    effort = torch.tensor([2.0])
    port_input = torch.tensor([3.0])
    spec = IPHSStructureSpec(
        name="open_balance",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        R=torch.ones((1, 1)),
        M=torch.zeros((1, 1)),
        G=torch.ones((1, 1)),
        stoichiometric_N=None,
        ports=(_scalar_iphs_port("open_input", role="external"),),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    audit = spec.validate(energy_effort=effort, port_input=port_input)

    assert audit["evidence_scope"] == "single_runtime_point"
    assert audit["closed_port"] is False
    assert audit["open_port_energy_rate"] == pytest.approx(2.0)
    assert audit["dissipation_power"] == pytest.approx(4.0)
    assert audit["port_power_into_system"] == pytest.approx(6.0)
    assert audit["power_balance_residual"] == pytest.approx(0.0)
    assert audit["balance_scale"] > 0.0
    assert audit["balance_tolerance"] >= 0.0
    assert "closed_port_energy_rate" not in audit
    assert "iphs_level" not in audit


def test_open_port_point_audit_rejects_nonfinite_derived_port_rate():
    from relic.computation.operators.iphs import IPHSStructureSpec

    huge = torch.tensor([1.0e20], dtype=torch.float32)
    spec = IPHSStructureSpec(
        name="overflowing_open_port",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1), dtype=torch.float32),
        R=torch.zeros((1, 1), dtype=torch.float32),
        M=torch.zeros((1, 1), dtype=torch.float32),
        G=torch.tensor([[1.0e20]], dtype=torch.float32),
        stoichiometric_N=None,
        ports=(_scalar_iphs_port("overflow_input", role="external"),),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match="port_rate.*finite"):
        spec.validate(energy_effort=huge, port_input=huge)


def test_single_point_balance_and_static_gradient_do_not_satisfy_level4():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="static_effort_is_not_proven_gradient",
        state_vars=("x",),
        energy_functional="H_zero",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        R=torch.ones((1, 1)),
        M=torch.zeros((1, 1)),
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.tensor([2.0]),
        contribution_count=1,
        contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="dissipative_ph", poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    point_audit = spec.validate(energy_effort=torch.tensor([2.0]))
    assert point_audit["power_balance_residual"] == pytest.approx(0.0)
    assert point_audit["evidence_scope"] == "single_runtime_point"
    with pytest.raises(ValueError, match=r"process runtime linkage remains \[N\]"):
        spec.validate(required_level=4, energy_effort=torch.tensor([2.0]))
    with pytest.raises(ValueError, match="explicit runtime energy_effort"):
        spec.validate(required_level=4)


@pytest.mark.parametrize(
    "runtime_owner",
    [
        "definitely.not.a.real.owner",
        "relic.computation.operators.iphs.IPHSStructureSpec.validate",
    ],
)
def test_mapping_runtime_owner_cannot_self_bind_level4(runtime_owner):
    from relic.computation.operators.iphs import assemble_iphs_structure

    with pytest.raises(ValueError, match=r"process runtime linkage remains \[N\]"):
        assemble_iphs_structure(
            name="metadata_only_owner",
            state_vars=("x",),
            energy_functional="E",
            entropy_functional="S",
            formulation="dissipative_ph",
            contributions=[
                {
                    "runtime_owner": runtime_owner,
                    "R": torch.ones((1, 1)),
                    "energy_gradient": torch.ones(1),
                }
            ],
            strict=True,
            required_level=4,
            validate=True, poisson_operator="matrix_snapshot", reaction_representation="independent",
        )


@pytest.mark.parametrize(
    ("formulation", "resistance", "mobility", "error"),
    [
        ("dissipative_ph", torch.zeros((1, 1)), torch.ones((1, 1)), "mobility M"),
        ("generic", torch.ones((1, 1)), torch.zeros((1, 1)), "resistance R"),
    ],
)
def test_canonical_drift_rejects_inactive_formulation_operator(
    formulation,
    resistance,
    mobility,
    error,
):
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="inactive_formulation_operator",
        state_vars=("x",),
        energy_functional="E",
        entropy_functional="S",
        J=torch.zeros((1, 1)),
        R=resistance,
        M=mobility,
        G=None,
        stoichiometric_N=None,
        ports=(),
        energy_gradient=torch.ones(1),
        entropy_gradient=torch.ones(1),
        formulation=formulation, poisson_operator="matrix_snapshot", reaction_representation="independent",
    )

    with pytest.raises(ValueError, match=error):
        spec.canonical_drift()


def _level5_reaction_spec(**overrides):
    """GENERIC zero-field spec with one absorbed reaction, S = 3 log(q), r = k q."""
    from relic.computation.operators.iphs import IPHSStructureSpec

    state = torch.tensor([5.0, 4.0], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    rate = coefficient * state[1]
    mobility = torch.diag(torch.stack((torch.zeros_like(rate), rate * state[1] / 3.0)))
    fields = dict(
        name="level5_runtime_reaction", state_vars=("energy", "irreversible_coordinate"),
        energy_functional="E=x0", entropy_functional="S=3*log(x1)",
        J=torch.zeros((2, 2), dtype=torch.float64), M=mobility, G=None,
        stoichiometric_N=torch.tensor([[0.0], [1.0]], dtype=torch.float64), ports=(),
        formulation="generic", energy_gradient=None, entropy_gradient=None,
        contribution_count=1, contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        poisson_operator="zero_field", reaction_representation="absorbed",
    )
    fields.update(overrides)
    return IPHSStructureSpec(**fields), state, rate


def test_level5_runtime_point_verifies_generic_absorbed_reaction_closure():
    # References: Grmela and Ottinger (1997); Mielke (2011) reaction Onsager operator.
    # Mathematical expression: e_E=(1,0), e_S=(0,3/q); M e_S = N r; e_E^T N r = 0;
    #   e_S^T N r = 3 r / q >= 0; sigma = e_S^T M e_S = 3 r / q.
    # Parameter meanings: q positive coordinate, r = k q reaction rate.
    # Units: normalized coordinate and time for the representation oracle.
    # Domain/constraints: q > 0, k > 0; zero Poisson field.
    # Sign convention: positive r increases q; energy is untouched by the reaction.
    spec, state, rate = _level5_reaction_spec()
    audit = spec.validate(
        required_level=5,
        state=state,
        energy_value=state[0],
        entropy_value=3.0 * torch.log(state[1]),
        reaction_rates=rate.reshape(1),
    )
    assert audit["required_level"] == 5
    assert audit["runtime_verified_level"] == 5
    assert audit["process_level_status"] == "runtime_point_verified"
    assert audit["evidence_scope"] == "runtime_point"
    assert audit["jacobi_identity_verified"] is True
    assert audit["jacobi_identity_basis"] == "canonical_zero_field"
    assert audit["reaction_energy_rate"] == 0.0
    assert audit["reaction_entropy_rate"] == pytest.approx(3.0 * 0.8 / 4.0)
    assert audit["entropy_production"] == pytest.approx(3.0 * 0.8 / 4.0)
    assert audit["M_energy_gradient_residual"] == 0.0
    assert audit["J_entropy_gradient_residual"] == 0.0


def test_level5_runtime_point_still_requires_binding_for_stored_efforts():
    from relic.computation.operators.iphs import IPHSStructureSpec

    spec = IPHSStructureSpec(
        name="stored_effort_reaction", state_vars=("energy", "irreversible_coordinate"),
        energy_functional="E=x0", entropy_functional="S=3*x1",
        J=torch.zeros((2, 2), dtype=torch.float64),
        M=torch.diag(torch.tensor([0.0, 2.0], dtype=torch.float64)), G=None,
        stoichiometric_N=torch.tensor([[0.0], [1.0]], dtype=torch.float64), ports=(),
        formulation="generic",
        energy_gradient=torch.tensor([1.0, 0.0], dtype=torch.float64),
        entropy_gradient=torch.tensor([0.0, 3.0], dtype=torch.float64),
        contribution_count=1, contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        poisson_operator="zero_field", reaction_representation="independent",
    )
    with pytest.raises(ValueError, match=r"reaction closure linkage.*\[N\]"):
        spec.validate(required_level=5)


def test_level5_runtime_point_rejects_reaction_energy_leak():
    spec, state, rate = _level5_reaction_spec(
        stoichiometric_N=torch.tensor([[1.0], [1.0]], dtype=torch.float64),
        reaction_representation="independent",
    )
    with pytest.raises(ValueError, match="closed reaction energy"):
        spec.validate(
            required_level=5, state=state, energy_value=state[0],
            entropy_value=3.0 * torch.log(state[1]), reaction_rates=rate.reshape(1),
        )


def test_level5_runtime_point_rejects_negative_reaction_entropy_rate():
    spec, state, rate = _level5_reaction_spec(
        stoichiometric_N=torch.tensor([[0.0], [-1.0]], dtype=torch.float64),
        reaction_representation="independent",
        M=torch.diag(torch.tensor([0.0, 2.0], dtype=torch.float64)),
    )
    with pytest.raises(ValueError, match="reaction entropy rate"):
        spec.validate(
            required_level=5, state=state, energy_value=state[0],
            entropy_value=3.0 * torch.log(state[1]), reaction_rates=rate.reshape(1),
        )


def test_level5_runtime_point_rejects_missing_reaction_rates_for_declared_N():
    spec, state, _ = _level5_reaction_spec()
    with pytest.raises(ValueError, match="reaction_rates"):
        spec.validate(
            required_level=5, state=state, energy_value=state[0],
            entropy_value=3.0 * torch.log(state[1]),
        )


def test_level5_runtime_point_keeps_matrix_snapshot_jacobi_unbound():
    # A constant skew J with e_S in its null space passes the degeneracy checks,
    # but a matrix snapshot still cannot prove the global Jacobi identity.
    from relic.computation.operators.iphs import IPHSStructureSpec

    state = torch.tensor([5.0, 1.0, 4.0], dtype=torch.float64, requires_grad=True)
    spec = IPHSStructureSpec(
        name="level5_matrix_snapshot", state_vars=("energy", "rotation", "irreversible_coordinate"),
        energy_functional="E=x0", entropy_functional="S=3*log(x2)",
        J=torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
        M=torch.diag(torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)), G=None,
        stoichiometric_N=None, ports=(), formulation="generic",
        energy_gradient=None, entropy_gradient=None,
        contribution_count=1, contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        poisson_operator="matrix_snapshot", reversible_operator_state_dependence="constant", reaction_representation="independent",
    )
    with pytest.raises(ValueError, match=r"Jacobi identity verification remains \[N\]"):
        spec.validate(
            required_level=5, state=state, energy_value=state[0],
            entropy_value=3.0 * torch.log(state[2]),
        )


def test_level5_runtime_point_without_reaction_passes_on_degenerate_structure():
    spec, state, _ = _level5_reaction_spec(
        stoichiometric_N=None, reaction_representation="independent",
        M=torch.diag(torch.tensor([0.0, 2.0], dtype=torch.float64)),
    )
    audit = spec.validate(
        required_level=5, state=state, energy_value=state[0],
        entropy_value=3.0 * torch.log(state[1]),
    )
    assert audit["runtime_verified_level"] == 5
    assert audit["entropy_production"] == pytest.approx(2.0 * (3.0 / 4.0) ** 2)
    assert "reaction_energy_rate" not in audit


def _level4_storage_spec(**overrides):
    """dPH zero-field spec with quadratic storage H = 0.5 x.x and R = 2 I."""
    from relic.computation.operators.iphs import IPHSStructureSpec

    fields = dict(
        name="level4_runtime_storage", state_vars=("x", "p"),
        energy_functional="H=0.5*x.x", entropy_functional="S",
        J=torch.zeros((2, 2), dtype=torch.float64), M=torch.zeros((2, 2), dtype=torch.float64),
        G=None, stoichiometric_N=None, ports=(), R=2.0 * torch.eye(2, dtype=torch.float64),
        energy_gradient=None, entropy_gradient=None,
        contribution_count=1, contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        formulation="dissipative_ph", poisson_operator="zero_field", reaction_representation="independent",
    )
    fields.update(overrides)
    return IPHSStructureSpec(**fields)


def test_level4_runtime_point_verifies_bound_dph_storage():
    # References: van der Schaft and Jeltsema (2014), dissipative pH passivity.
    # Mathematical expression: e_H = dH/dx = x; x_dot = -R e_H; dH/dt = -e_H^T R e_H <= 0.
    # Parameter meanings: x state, R = 2 I resistance, H quadratic storage.
    # Units: normalized state and power for the representation oracle.
    # Domain/constraints: closed ports, zero Poisson field, no reaction.
    # Sign convention: dissipation power positive, closed-port energy rate negative.
    spec = _level4_storage_spec()
    state = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
    audit = spec.validate(required_level=4, state=state, energy_value=0.5 * (state * state).sum())
    assert audit["required_level"] == 4
    assert audit["runtime_verified_level"] == 4
    assert audit["process_level_status"] == "runtime_point_verified"
    assert audit["evidence_scope"] == "runtime_point"
    assert audit["closed_port"] is True
    assert audit["dissipation_power"] == pytest.approx(50.0)
    assert audit["closed_port_energy_rate"] == pytest.approx(-50.0)
    assert audit["power_balance_residual"] == pytest.approx(0.0)


def test_level4_runtime_point_rejects_explicit_effort_without_state_binding():
    spec = _level4_storage_spec()
    with pytest.raises(ValueError, match=r"process runtime linkage remains \[N\]"):
        spec.validate(required_level=4, energy_effort=torch.tensor([3.0, 4.0], dtype=torch.float64))


def test_level4_runtime_point_keeps_reaction_pairing_unbound():
    spec = _level4_storage_spec(stoichiometric_N=torch.tensor([[1.0], [0.0]], dtype=torch.float64))
    state = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match=r"reaction closure linkage.*\[N\]"):
        spec.validate(
            required_level=4, state=state, energy_value=0.5 * (state * state).sum(),
            reaction_rates=torch.tensor([0.5], dtype=torch.float64),
        )


def test_level4_runtime_point_rejects_entropy_value_for_dph():
    spec = _level4_storage_spec()
    state = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="entropy_value"):
        spec.validate(
            required_level=4, state=state, energy_value=0.5 * (state * state).sum(),
            entropy_value=state.sum(),
        )


def test_level4_runtime_point_rejects_energy_creating_resistance():
    spec = _level4_storage_spec(R=-2.0 * torch.eye(2, dtype=torch.float64))
    state = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="positive semidefinite"):
        spec.validate(required_level=4, state=state, energy_value=0.5 * (state * state).sum())


def _level4_reaction_spec(*, storage_kind, stoichiometry, representation="independent", resistance=None):
    """dPH runtime-point spec with one declared reaction on x = (energy, q)."""
    from relic.computation.operators.iphs import IPHSStructureSpec

    if resistance is None:
        resistance = torch.diag(torch.tensor([0.0, 2.0], dtype=torch.float64))
    return IPHSStructureSpec(
        name="level4_reaction_point", state_vars=("energy", "q"),
        energy_functional="H", entropy_functional="S",
        J=torch.zeros((2, 2), dtype=torch.float64), M=torch.zeros((2, 2), dtype=torch.float64),
        R=resistance, G=None, stoichiometric_N=torch.tensor(stoichiometry, dtype=torch.float64),
        ports=(), formulation="dissipative_ph", poisson_operator="zero_field",
        storage_kind=storage_kind, reaction_representation=representation,
        energy_gradient=None, entropy_gradient=None,
        contribution_count=1, contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
    )


def test_level4_runtime_point_verifies_total_energy_reaction_pairing():
    # References: van der Schaft (2022) dPH with reaction; MA-19.2 independent dPH pairing.
    # Mathematical expression: H = x0 (total energy), e_H = (1, 0), N = (0, 1)^T,
    #   e_H^T N r = 0; dH/dt = -e_H^T R e_H + e_H^T N r = 0 for closed reaction.
    # Parameter meanings: q irreversible coordinate, r = k q reaction rate.
    # Units: normalized state, energy, and time for the representation oracle.
    # Domain/constraints: zero Poisson field, closed ports, k > 0.
    # Sign convention: reaction energy rate must vanish for a total-energy storage.
    spec = _level4_reaction_spec(storage_kind="total_energy", stoichiometry=[[0.0], [1.0]])
    state = torch.tensor([5.0, 4.0], dtype=torch.float64, requires_grad=True)
    audit = spec.validate(
        required_level=4, state=state, energy_value=state[0],
        reaction_rates=(0.2 * state[1]).reshape(1),
    )
    assert audit["runtime_verified_level"] == 4
    assert audit["storage_kind"] == "total_energy"
    assert audit["reaction_energy_rate"] == 0.0
    assert audit["dissipation_power"] == 0.0
    assert audit["power_balance_residual"] == pytest.approx(0.0)


def test_level4_runtime_point_verifies_free_energy_reaction_pairing_sign():
    # H = 0.5 x.x (free energy), N = (0, -1)^T consumes q: e_H^T N r = -q r <= 0.
    spec = _level4_reaction_spec(storage_kind="free_energy", stoichiometry=[[0.0], [-1.0]])
    state = torch.tensor([5.0, 4.0], dtype=torch.float64, requires_grad=True)
    rate = (0.2 * state[1]).reshape(1)
    audit = spec.validate(
        required_level=4, state=state, energy_value=0.5 * (state * state).sum(), reaction_rates=rate,
    )
    assert audit["runtime_verified_level"] == 4
    assert audit["reaction_energy_rate"] == pytest.approx(-4.0 * 0.8)
    assert audit["closed_port_energy_rate"] == pytest.approx(-2.0 * 16.0 - 3.2)
    assert audit["power_balance_residual"] == pytest.approx(0.0)

    producing = _level4_reaction_spec(storage_kind="free_energy", stoichiometry=[[0.0], [1.0]])
    with pytest.raises(ValueError, match="free-energy reaction pairing"):
        producing.validate(
            required_level=4, state=state, energy_value=0.5 * (state * state).sum(), reaction_rates=rate,
        )


def test_level4_runtime_point_rejects_total_energy_reaction_leak():
    spec = _level4_reaction_spec(storage_kind="total_energy", stoichiometry=[[1.0], [0.0]])
    state = torch.tensor([5.0, 4.0], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="closed reaction energy"):
        spec.validate(
            required_level=4, state=state, energy_value=state[0],
            reaction_rates=(0.2 * state[1]).reshape(1),
        )


def test_level4_runtime_point_requires_storage_kind_for_reactions():
    spec = _level4_reaction_spec(storage_kind="unspecified", stoichiometry=[[0.0], [1.0]])
    state = torch.tensor([5.0, 4.0], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match=r"storage_kind.*reaction closure linkage.*\[N\]"):
        spec.validate(
            required_level=4, state=state, energy_value=state[0],
            reaction_rates=(0.2 * state[1]).reshape(1),
        )


def test_level4_runtime_point_verifies_absorbed_dph_reaction():
    # -R e_H = N r: R = diag(0, k), H = 0.5 x.x, N = (0, -1)^T, r = k q.
    state = torch.tensor([5.0, 4.0], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(0.2, dtype=torch.float64)
    spec = _level4_reaction_spec(
        storage_kind="free_energy", stoichiometry=[[0.0], [-1.0]], representation="absorbed",
        resistance=torch.diag(torch.stack((torch.zeros_like(coefficient), coefficient))),
    )
    audit = spec.validate(
        required_level=4, state=state, energy_value=0.5 * (state * state).sum(),
        reaction_rates=(coefficient * state[1]).reshape(1),
    )
    assert audit["runtime_verified_level"] == 4
    assert audit["reaction_representation"] == "absorbed"
    assert audit["reaction_energy_rate"] == pytest.approx(-0.2 * 16.0)
    assert audit["dissipation_power"] == pytest.approx(0.2 * 16.0)
    assert audit["closed_port_energy_rate"] == pytest.approx(-0.2 * 16.0)


def test_storage_kind_selection_is_validated():
    with pytest.raises(ValueError, match="storage_kind"):
        _level4_reaction_spec(storage_kind="enthalpy", stoichiometry=[[0.0], [1.0]])


# ---------------------------------------------------------------------------
# D72: executable nonzero Poisson-field bases (constant_field / state_field).
# References: Grmela and Ottinger (1997); Marsden and Ratiu (1999) Ch. 10;
#   Plank (1995) J. Math. Phys. 36, Hamiltonian structures for Lotka-Volterra systems.
# Mathematical expression: Jacobi residual R_ijk = sum_l (J_il d_l J_jk + cyclic) = 0.
# Parameter meanings: x state vector on a normalized three-coordinate basis; J(x) the
#   candidate Poisson tensor; e_E, e_S energy/entropy gradients on the same basis.
# Units: normalized test coordinates; these are structural pre-gates, not process claims.
# Domain/constraints: finite sample points; float64; degeneracy J e_S = 0 and M e_E = 0 hold
#   at the declared reference point by construction of the test gradients.
# Sign convention: not applicable to an algebraic identity residual.
# ---------------------------------------------------------------------------


def _three_state_spec(**overrides):
    from relic.computation.operators.iphs import IPHSStructureSpec

    fields = dict(
        name="poisson_field_probe", state_vars=("x0", "x1", "x2"),
        energy_functional="E=x0+x1", entropy_functional="S=3*x2",
        J=torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
        M=torch.diag(torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)),
        G=None, stoichiometric_N=None, ports=(), formulation="generic",
        energy_gradient=torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True),
        entropy_gradient=torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64),
        contribution_count=1, contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        poisson_operator="constant_field", reversible_operator_state_dependence="constant", reaction_representation="independent",
    )
    fields.update(overrides)
    return IPHSStructureSpec(**fields)


def _so3_field(x):
    # Lie-Poisson structure of so(3): J_ij = epsilon_ijk x_k, a Poisson tensor for every x.
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.stack(
        [
            torch.stack([zero, x[2], -x[1]]),
            torch.stack([-x[2], zero, x[0]]),
            torch.stack([x[1], -x[0], zero]),
        ]
    )


def _lotka_volterra_field(x):
    # J_ij = A_ij x_i x_j with constant antisymmetric A is a Poisson tensor (Plank 1995).
    a = torch.tensor([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]], dtype=x.dtype, device=x.device)
    return a * x.reshape(-1, 1) * x.reshape(1, -1)


def _non_poisson_field(x):
    # Skew but J12 = x0 and J23 = x1 give a nonzero cyclic residual (R_123 = x0).
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.stack(
        [
            torch.stack([zero, x[0], zero]),
            torch.stack([-x[0], zero, x[1]]),
            torch.stack([zero, -x[1], zero]),
        ]
    )


def test_jacobi_residual_vanishes_for_so3_and_lotka_volterra_and_detects_a_violation():
    from relic.computation.operators.iphs import poisson_jacobi_residual

    point = torch.tensor([0.7, -1.3, 2.1], dtype=torch.float64)
    assert float(torch.max(torch.abs(poisson_jacobi_residual(_so3_field, point)))) < 1e-12
    assert float(torch.max(torch.abs(poisson_jacobi_residual(_lotka_volterra_field, point)))) < 1e-12
    residual = poisson_jacobi_residual(_non_poisson_field, point)
    assert float(residual[0, 1, 2]) == pytest.approx(0.7, abs=1e-12)


def test_constant_field_executes_a_nonzero_skew_operator_with_an_exact_jacobi_basis():
    spec = _three_state_spec()
    audit = spec.validate()
    assert audit["jacobi_identity_verified"] is True
    assert audit["jacobi_identity_basis"] == "constant_field_exact"
    assert audit["process_level_status"] == "[N] runtime_unbound"
    drift = spec.canonical_drift()
    # J e_E = (1, -1, 0); M e_S = (0, 0, 6).
    torch.testing.assert_close(drift, torch.tensor([1.0, -1.0, 6.0], dtype=torch.float64))
    assert torch.dot(spec.energy_gradient, drift).item() == pytest.approx(0.0)


def test_constant_field_checks_a_declared_callable_on_the_domain_and_rejects_drift():
    constant = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    domain = torch.tensor([[0.0, 0.0, 1.0], [0.5, -0.5, 2.0]], dtype=torch.float64)
    spec = _three_state_spec(poisson_field=lambda x: constant.clone(), poisson_domain=domain)
    audit = spec.validate()
    assert audit["jacobi_sample_count"] == 2
    with pytest.raises(ValueError, match="constant_field requires poisson_field to equal"):
        _three_state_spec(poisson_field=lambda x: constant * (1.0 + x[2]), poisson_domain=domain)


def test_constant_field_rejects_a_state_dependent_declaration():
    with pytest.raises(ValueError, match="constant_field cannot declare"):
        _three_state_spec(reversible_operator_state_dependence="state_dependent")


def test_state_field_verifies_so3_on_the_declared_domain_and_evaluates_at_the_runtime_point():
    reference = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    domain = torch.stack([reference, torch.tensor([0.3, -0.4, 0.9], dtype=torch.float64), torch.tensor([-1.0, 2.0, 0.5], dtype=torch.float64)])
    spec = _three_state_spec(
        J=_so3_field(reference),
        M=torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)),
        energy_gradient=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True),
        entropy_gradient=torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64),
        poisson_operator="state_field", reversible_operator_state_dependence="state_dependent",
        poisson_field=_so3_field, poisson_domain=domain,
    )
    audit = spec.validate()
    assert audit["jacobi_identity_verified"] is True
    assert audit["jacobi_identity_basis"] == "sampled_domain_jacobi_residual"
    assert audit["jacobi_sample_count"] == 3
    assert audit["jacobi_max_residual"] < 1e-12
    runtime_point = torch.tensor([0.2, 0.1, 0.9], dtype=torch.float64)
    torch.testing.assert_close(spec._evaluate_poisson_matrix(state=runtime_point), _so3_field(runtime_point))


def test_state_field_accepts_the_lotka_volterra_poisson_tensor():
    reference = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    domain = torch.stack([reference, torch.tensor([0.5, 2.0, 1.5], dtype=torch.float64)])
    spec = _three_state_spec(
        J=_lotka_volterra_field(reference),
        M=torch.diag(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)),
        energy_gradient=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True),
        entropy_gradient=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        poisson_operator="state_field", reversible_operator_state_dependence="state_dependent",
        poisson_field=_lotka_volterra_field, poisson_domain=domain,
    )
    assert spec.validate()["jacobi_identity_basis"] == "sampled_domain_jacobi_residual"


def test_state_field_refuses_a_skew_field_that_violates_the_jacobi_identity():
    reference = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    domain = torch.stack([reference, torch.tensor([0.7, 0.2, 0.4], dtype=torch.float64)])
    with pytest.raises(ValueError, match="violates the Jacobi identity"):
        _three_state_spec(
            J=_non_poisson_field(reference),
            M=torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)),
            energy_gradient=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True),
            entropy_gradient=torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64),
            poisson_operator="state_field", reversible_operator_state_dependence="state_dependent",
            poisson_field=_non_poisson_field, poisson_domain=domain,
        )


def test_state_field_requires_callable_domain_and_state_dependent_declaration():
    with pytest.raises(ValueError, match="state_field requires poisson_field and poisson_domain"):
        _three_state_spec(poisson_operator="state_field", reversible_operator_state_dependence="state_dependent")
    domain = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="state_dependent"):
        _three_state_spec(
            J=_so3_field(domain[0]), poisson_operator="state_field",
            reversible_operator_state_dependence="constant",
            poisson_field=_so3_field, poisson_domain=domain,
        )


def test_matrix_snapshot_still_cannot_claim_a_level_five_jacobi_basis():
    spec = _three_state_spec(poisson_operator="matrix_snapshot", reversible_operator_state_dependence="unspecified")
    audit = spec.validate()
    assert "jacobi_identity_basis" not in audit


# ---------------------------------------------------------------------------
# MA-21: the production drift evaluates the declared state field on the autograd
# path, and a Level 5 runtime point audits the Jacobi residual at the bound state.
# References: Grmela and Ottinger (1997); Marsden and Ratiu (1999) Ch. 10 and 14
#   (Lie-Poisson bracket of so(3); Casimir functions).
# Mathematical expression: E(x) = a.x, S(x) = 0.5 x.x, J(x)_ij = eps_ijk x_k,
#   M = I - a a^T / (a.a); x_dot = J(x) e_E + M e_S = a cross x + M x;
#   d x_dot_i / d x_k = sum_j eps_ijk a_j + M_ik.
# Parameter meanings: a is a constant energy covector; S is a Casimir of so(3),
#   so J(x) e_S = J(x) x = 0 and M e_E = M a = 0 hold at every state x.
# Units: normalized test coordinates; structural evidence, not a process claim.
# Domain/constraints: float64; finite declared samples plus the bound runtime state.
# Sign convention: e_S^T M e_S = x^T M x >= 0 is the entropy production.
# ---------------------------------------------------------------------------

_SO3_ENERGY_COVECTOR = (1.0, 2.0, 3.0)
_SO3_CASIMIR_DOMAIN = ((0.0, 0.0, 1.0), (0.3, -0.4, 0.9), (-1.0, 2.0, 0.5))


def _levi_civita(dtype):
    tensor = torch.zeros((3, 3, 3), dtype=dtype)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        tensor[i, j, k] = 1.0
        tensor[i, k, j] = -1.0
    return tensor


def _casimir_generic_spec(field=None, domain=_SO3_CASIMIR_DOMAIN, **overrides):
    from relic.computation.operators.iphs import IPHSStructureSpec

    a = torch.tensor(_SO3_ENERGY_COVECTOR, dtype=torch.float64)
    mobility = torch.eye(3, dtype=torch.float64) - torch.outer(a, a) / torch.dot(a, a)
    field = _so3_field if field is None else field
    samples = torch.tensor(domain, dtype=torch.float64)
    fields = dict(
        name="casimir_generic_state_field", state_vars=("x0", "x1", "x2"),
        energy_functional="E=a.x", entropy_functional="S=0.5 x.x",
        J=field(samples[0]).detach(), M=mobility, G=None, stoichiometric_N=None, ports=(),
        formulation="generic", poisson_operator="state_field",
        reversible_operator_state_dependence="state_dependent",
        poisson_field=field, poisson_domain=samples,
        contribution_count=1, contribution_sources=(_EXECUTABLE_IPHS_SOURCE,), reaction_representation="independent",
    )
    fields.update(overrides)
    return IPHSStructureSpec(**fields)


def _casimir_potentials(x):
    a = torch.tensor(_SO3_ENERGY_COVECTOR, dtype=x.dtype)
    return torch.dot(a, x), 0.5 * torch.dot(x, x)


def _casimir_drift(spec, x):
    energy, entropy = _casimir_potentials(x)
    return spec.canonical_drift(state=x, energy_value=energy, entropy_value=entropy)


def test_state_field_drift_jacobian_includes_the_field_derivative():
    spec = _casimir_generic_spec()
    point = torch.tensor([5.0, 7.0, -3.0], dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(lambda z: _casimir_drift(spec, z), point.clone())
    a = torch.tensor(_SO3_ENERGY_COVECTOR, dtype=torch.float64)
    analytic = torch.einsum("ijk,j->ik", _levi_civita(torch.float64), a) + spec.M
    torch.testing.assert_close(jacobian, analytic, rtol=0.0, atol=1.0e-12)
    step = 1.0e-6
    columns = []
    for index in range(3):
        offset = torch.zeros(3, dtype=torch.float64)
        offset[index] = step
        forward = _casimir_drift(spec, (point + offset).requires_grad_(True)).detach()
        backward = _casimir_drift(spec, (point - offset).requires_grad_(True)).detach()
        columns.append((forward - backward) / (2.0 * step))
    torch.testing.assert_close(jacobian, torch.stack(columns, dim=1), rtol=0.0, atol=1.0e-7)


def test_state_field_drift_carries_the_parameter_sensitivity_of_the_field():
    theta = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

    def scaled_field(x):
        return theta * _so3_field(x)

    spec = _casimir_generic_spec(field=scaled_field)
    point = torch.tensor([0.2, -0.7, 1.1], dtype=torch.float64, requires_grad=True)
    drift = _casimir_drift(spec, point)
    # d x_dot / d theta = J(x) a / theta = a cross x.
    expected = torch.linalg.cross(torch.tensor(_SO3_ENERGY_COVECTOR, dtype=torch.float64), point.detach())
    for index in range(3):
        (gradient,) = torch.autograd.grad(drift[index], theta, retain_graph=True, allow_unused=True)
        assert gradient is not None
        assert float(gradient) == pytest.approx(float(expected[index]), abs=1.0e-12)


def test_level5_state_field_runtime_point_audits_the_jacobi_residual_at_the_bound_state():
    spec = _casimir_generic_spec()
    point = torch.tensor([5.0, 7.0, -3.0], dtype=torch.float64, requires_grad=True)
    assert not any(torch.equal(point.detach(), sample) for sample in spec.poisson_domain)
    energy, entropy = _casimir_potentials(point)
    audit = spec.validate(required_level=5, state=point, energy_value=energy, entropy_value=entropy)
    assert audit["runtime_verified_level"] == 5
    assert audit["jacobi_identity_basis"] == "sampled_domain_jacobi_residual"
    assert audit["jacobi_sample_count"] == 3
    assert audit["runtime_point_jacobi_residual"] == pytest.approx(0.0, abs=1.0e-12)
    assert audit["runtime_point_skew_residual"] == pytest.approx(0.0, abs=1.0e-12)
    assert audit["runtime_point_jacobi_point"] == "bound_state"


def _jacobi_violation_off_samples_field(x):
    # v(x) = x + (x0 - 5) e_z and J_ij = eps_ijk v_k. The Jacobi residual of such a
    # bivector is proportional to v . curl(v) = -x1, so it vanishes on samples with
    # x1 = 0 and not at a runtime state with x1 != 0; at x0 = 5 the field equals the
    # so(3) tensor, so the degeneracy J(x) x = 0 still holds there.
    shifted = torch.stack([x[0], x[1], x[2] + (x[0] - 5.0)])
    return _so3_field(shifted)


@pytest.mark.parametrize("tolerance", [float("inf"), float("nan"), 0.0, -1.0e-8, True, "1e-8"])
def test_state_field_rejects_a_vacuous_or_malformed_jacobi_tolerance(tolerance):
    with pytest.raises(ValueError, match="jacobi_tol must be a finite positive number"):
        _casimir_generic_spec(jacobi_tol=tolerance)


_JACOBI_TOLERANCE_KEY = "validation.iphs.jacobi_tolerance"


def _jacobi_tolerance_yaml_entry():
    from pathlib import Path

    import yaml

    path = (
        Path(__file__).resolve().parents[4]
        / "relic" / "parameters" / "database" / "relic" / "validation" / "iphs.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"][_JACOBI_TOLERANCE_KEY]


def test_default_jacobi_tolerance_resolves_from_the_parameter_database():
    from relic.computation.operators.iphs import assemble_iphs_structure
    from relic.parameters import get_global_manager

    entry = _jacobi_tolerance_yaml_entry()
    for field_name in ("value", "units", "dtype", "bounds", "category", "description",
                       "references", "tags", "requires_grad", "provenance"):
        assert field_name in entry, field_name
    value = float(entry["value"])
    assert get_global_manager().get(_JACOBI_TOLERANCE_KEY, required=True) == pytest.approx(value)
    spec = _casimir_generic_spec()
    assert spec.jacobi_tol == pytest.approx(value)
    assert spec.validate()["jacobi_tolerance"] == pytest.approx(value)
    assembled = assemble_iphs_structure(
        name="assembled_state_field", state_vars=("x0", "x1", "x2"),
        energy_functional="E", entropy_functional="S",
        contributions=[{"J": spec.J, "M": spec.M}], formulation="generic",
        poisson_operator="state_field", reversible_operator_state_dependence="state_dependent",
        poisson_field=_so3_field, poisson_domain=spec.poisson_domain, reaction_representation="independent",
    )
    assert assembled.jacobi_tol == pytest.approx(value)
    # An explicit caller tolerance stays explicit.
    assert _casimir_generic_spec(jacobi_tol=1.0e-6).jacobi_tol == pytest.approx(1.0e-6)


def test_changing_the_jacobi_tolerance_parameter_changes_the_state_field_check():
    from relic.parameters import get_global_manager

    # J(x) = s N(x) with the non-Poisson field N: the Jacobi residual is s^2 x0,
    # i.e. 9e-10 at the sample x0 = 1 for s = 3e-5.
    def weak_violation(x):
        return 3.0e-5 * _non_poisson_field(x)

    domain = ((1.0, 0.0, 0.5),)
    manager = get_global_manager()
    original = manager.get(_JACOBI_TOLERANCE_KEY, required=True)
    assert original > 9.0e-10
    audit = _casimir_generic_spec(field=weak_violation, domain=domain).validate()
    assert audit["jacobi_max_residual"] == pytest.approx(9.0e-10, rel=1.0e-9)
    try:
        manager.set_value(_JACOBI_TOLERANCE_KEY, 1.0e-10)
        with pytest.raises(ValueError, match="violates the Jacobi identity"):
            _casimir_generic_spec(field=weak_violation, domain=domain)
    finally:
        manager.set_value(_JACOBI_TOLERANCE_KEY, original)
    assert manager.get(_JACOBI_TOLERANCE_KEY, required=True) == original


_STRUCTURE_TOLERANCE_KEYS = {
    "skew_tol": "validation.iphs.skew_tolerance",
    "psd_tol": "validation.iphs.psd_tolerance",
    "degeneracy_tol": "validation.iphs.degeneracy_tolerance",
}


def _iphs_validation_yaml_entry(key):
    from pathlib import Path

    import yaml

    path = (
        Path(__file__).resolve().parents[4]
        / "relic" / "parameters" / "database" / "relic" / "validation" / "iphs.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"][key]


@pytest.mark.parametrize("field_name,key", sorted(_STRUCTURE_TOLERANCE_KEYS.items()))
def test_default_structure_tolerances_resolve_from_the_parameter_database(field_name, key):
    from relic.computation.operators.iphs import assemble_iphs_structure
    from relic.parameters import get_global_manager

    entry = _iphs_validation_yaml_entry(key)
    for metadata_field in ("value", "units", "dtype", "bounds", "category", "description",
                           "references", "tags", "requires_grad", "provenance"):
        assert metadata_field in entry, metadata_field
    value = float(entry["value"])
    assert get_global_manager().get(key, required=True) == pytest.approx(value)
    assert getattr(_three_state_spec(), field_name) == pytest.approx(value)
    assembled = assemble_iphs_structure(
        name="assembled_defaults", state_vars=("x", "y"), energy_functional="E",
        entropy_functional="S", contributions=[{"J": torch.tensor([[0.0, 1.0], [-1.0, 0.0]])}], poisson_operator="matrix_snapshot", reaction_representation="independent",
    )
    assert getattr(assembled, field_name) == pytest.approx(value)
    # An explicit caller tolerance stays explicit.
    assert getattr(_three_state_spec(**{field_name: 0.25}), field_name) == pytest.approx(0.25)


def _skew_check(tolerance_changed):
    # J + J^T has Frobenius norm sqrt(2) 1e-11.
    perturbed = torch.tensor([[0.0, 1.0, 0.0], [-1.0 + 1.0e-11, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    spec = _three_state_spec(
        J=perturbed, poisson_operator="matrix_snapshot", reversible_operator_state_dependence="unspecified",
    )
    if tolerance_changed:
        with pytest.raises(ValueError, match="non-skew J"):
            spec.validate()
    else:
        assert spec.validate()["skew_residual"] == pytest.approx(2.0 ** 0.5 * 1.0e-11, rel=1.0e-4)


def _psd_check(tolerance_changed):
    spec = _three_state_spec(M=torch.diag(torch.tensor([0.0, 0.0, -1.0e-11], dtype=torch.float64)))
    if tolerance_changed:
        with pytest.raises(ValueError, match="M is not positive semidefinite"):
            spec.validate()
    else:
        assert spec.validate()["min_dissipation_eigenvalue"] == pytest.approx(-1.0e-11)


def _degeneracy_check(tolerance_changed):
    # J e_S = (1e-11, 0, 0) for e_S = (0, 1e-11, 3).
    spec = _three_state_spec(entropy_gradient=torch.tensor([0.0, 1.0e-11, 3.0], dtype=torch.float64))
    message = "violates J e_S = 0" if tolerance_changed else "linkage remains"
    with pytest.raises(ValueError, match=message):
        spec.validate(required_level=5)


@pytest.mark.parametrize("field_name,check", [
    ("skew_tol", _skew_check), ("psd_tol", _psd_check), ("degeneracy_tol", _degeneracy_check),
])
def test_changing_a_structure_tolerance_parameter_changes_the_check(field_name, check):
    from relic.parameters import get_global_manager

    key = _STRUCTURE_TOLERANCE_KEYS[field_name]
    manager = get_global_manager()
    original = manager.get(key, required=True)
    assert original > 1.0e-10 * 0.5
    check(tolerance_changed=False)
    try:
        manager.set_value(key, 1.0e-12)
        check(tolerance_changed=True)
    finally:
        manager.set_value(key, original)
    assert manager.get(key, required=True) == original


_REACTION_TOLERANCE_KEYS = {
    "conservation_tolerance": "validation.iphs.reaction_conservation_tolerance",
    "thermodynamic_rtol": "validation.iphs.reaction_thermodynamic_rtol",
}


def _closed_exchange_network(**overrides):
    # X <-> Y with r = (1, 1) and A = (1, -(1 + 1e-10)): P = sum A r = -1e-10 and
    # sum |A r| = 2 + 1e-10, so the aggregate check passes for rtol >= 5e-11.
    from relic.computation.operators.iphs import advance_reaction_network

    payload = dict(
        name="closed_exchange", species=("X", "Y"), reactions=("X_to_Y", "Y_to_X"),
        stoichiometric_N=torch.tensor([[-1.0, 1.0], [1.0, -1.0]], dtype=torch.float64),
        rates=torch.tensor([1.0, 1.0], dtype=torch.float64),
        conservation_matrix=torch.ones((1, 2), dtype=torch.float64),
        affinities=torch.tensor([1.0, -(1.0 + 1.0e-10)], dtype=torch.float64),
        affinity_sign_convention="affinity_times_rate_nonnegative",
        require_thermodynamic_point_audit=True,
    )
    payload.update(overrides)
    return advance_reaction_network(**payload)


def _leaky_single_reaction_network():
    # C N = 1 * (-1) + 1 * (1 + 5e-11) = 5e-11 for the ledger C = (1, 1).
    from relic.computation.operators.iphs import advance_reaction_network

    return advance_reaction_network(
        name="leaky_single_reaction", species=("A", "B"), reactions=("A_to_B",),
        stoichiometric_N=torch.tensor([[-1.0], [1.0 + 5.0e-11]], dtype=torch.float64),
        rates=torch.tensor([1.0], dtype=torch.float64),
        conservation_matrix=torch.ones((1, 2), dtype=torch.float64),
    )


@pytest.mark.parametrize("argument,key", sorted(_REACTION_TOLERANCE_KEYS.items()))
def test_default_reaction_network_tolerances_resolve_from_the_parameter_database(argument, key):
    from relic.parameters import get_global_manager

    entry = _iphs_validation_yaml_entry(key)
    for metadata_field in ("value", "units", "dtype", "bounds", "category", "description",
                           "references", "tags", "requires_grad", "provenance"):
        assert metadata_field in entry, metadata_field
    value = float(entry["value"])
    assert get_global_manager().get(key, required=True) == pytest.approx(value)
    advance = _closed_exchange_network()
    ledger = torch.ones((1, 2), dtype=torch.float64)
    if argument == "conservation_tolerance":
        assert advance.conservation_audit["tolerance"] == pytest.approx(value)
        assert advance.validate_conservation(ledger)["tolerance"] == pytest.approx(value)
        assert advance.validate_conservation(ledger, tolerance=0.5)["tolerance"] == pytest.approx(0.5)
        explicit = _closed_exchange_network(conservation_tolerance=1.0e-8)
        assert explicit.conservation_audit["tolerance"] == pytest.approx(1.0e-8)
    else:
        assert advance.thermodynamic_audit["relative_tolerance"] == pytest.approx(value)
        explicit = _closed_exchange_network(thermodynamic_rtol=1.0e-6)
        assert explicit.thermodynamic_audit["relative_tolerance"] == pytest.approx(1.0e-6)


def test_changing_the_reaction_conservation_parameter_changes_the_check():
    from relic.parameters import get_global_manager

    key = _REACTION_TOLERANCE_KEYS["conservation_tolerance"]
    manager = get_global_manager()
    original = manager.get(key, required=True)
    ledger = torch.ones((1, 2), dtype=torch.float64)
    advance = _leaky_single_reaction_network()
    assert advance.conservation_audit["max_abs_residual"] == pytest.approx(5.0e-11, rel=1.0e-4)
    try:
        manager.set_value(key, 1.0e-12)
        assert advance.validate_conservation(ledger)["passed"] is False
        with pytest.raises(ValueError, match="violates declared C_h N_h = 0"):
            _leaky_single_reaction_network()
    finally:
        manager.set_value(key, original)
    assert manager.get(key, required=True) == original


def test_changing_the_reaction_thermodynamic_parameter_changes_the_check():
    from relic.parameters import get_global_manager

    key = _REACTION_TOLERANCE_KEYS["thermodynamic_rtol"]
    manager = get_global_manager()
    original = manager.get(key, required=True)
    assert _closed_exchange_network().thermodynamic_audit["affinity_rate_power_min"] == pytest.approx(
        -1.0e-10, rel=1.0e-4
    )
    try:
        manager.set_value(key, 1.0e-11)
        with pytest.raises(ValueError, match="negative beyond the declared relative tolerance"):
            _closed_exchange_network()
    finally:
        manager.set_value(key, original)
    assert manager.get(key, required=True) == original


def test_level5_state_field_runtime_point_rejects_a_jacobi_violation_off_the_samples():
    domain = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.5), (-2.0, 0.0, 3.0))
    spec = _casimir_generic_spec(field=_jacobi_violation_off_samples_field, domain=domain)
    assert spec.validate()["jacobi_max_residual"] == pytest.approx(0.0, abs=1.0e-12)
    point = torch.tensor([5.0, 7.0, -3.0], dtype=torch.float64, requires_grad=True)
    energy, entropy = _casimir_potentials(point)
    with pytest.raises(ValueError, match="Jacobi identity at the runtime point"):
        spec.validate(required_level=5, state=point, energy_value=energy, entropy_value=entropy)


def _small_generic_reaction_leak():
    """GENERIC point with r = 1e-3 and an energy leak e_E^T N r = 5e-11 (N = (5e-8, 1)^T)."""
    from relic.computation.operators.iphs import IPHSStructureSpec

    state = torch.tensor([5.0, 5.0e-3], dtype=torch.float64, requires_grad=True)
    rate = torch.tensor(0.2, dtype=torch.float64) * state[1]
    mobility = torch.diag(torch.stack((torch.zeros_like(rate), rate * state[1] / 3.0)))
    spec = IPHSStructureSpec(
        name="small_reaction_leak", state_vars=("energy", "irreversible_coordinate"),
        energy_functional="E=x0", entropy_functional="S=3*log(x1)",
        J=torch.zeros((2, 2), dtype=torch.float64), M=mobility, G=None,
        stoichiometric_N=torch.tensor([[5.0e-8], [1.0]], dtype=torch.float64), ports=(),
        formulation="generic", energy_gradient=None, entropy_gradient=None,
        contribution_count=1, contribution_sources=(_EXECUTABLE_IPHS_SOURCE,),
        poisson_operator="zero_field", reaction_representation="independent",
    )
    return spec, state, rate


def test_reaction_energy_closure_scale_floor_is_read_from_the_parameter_database():
    # References: Hairer, Norsett and Wanner (1993) Sect. II.4 (mixed absolute/relative
    #   tolerance); validation.iphs.reaction_power_scale_floor (convention value 1.0).
    # Mathematical expression: |e_E^T N r| <= tol * max(floor, ||e_E|| ||N r||).
    # Parameter meanings: tol = validation.iphs.degeneracy_tolerance (1e-10); here
    #   ||e_E|| ||N r|| = 1e-3 and |e_E^T N r| = 5e-11.
    # Units: normalized power of the declared basis.
    # Domain/constraints: floor 1.0 -> threshold 1e-10 (passes); floor 1e-6 ->
    #   threshold 1e-13 (the relative test applies and the leak is refused).
    # Sign convention: only magnitudes are compared.
    from relic.parameters import get_global_manager

    key = "validation.iphs.reaction_power_scale_floor"
    manager = get_global_manager()
    original = manager.get(key, required=True)
    assert original == 1.0
    spec, state, rate = _small_generic_reaction_leak()
    audit = spec.validate(
        required_level=5, state=state, energy_value=state[0],
        entropy_value=3.0 * torch.log(state[1]), reaction_rates=rate.reshape(1),
    )
    assert audit["reaction_energy_rate"] == pytest.approx(5.0e-11)
    try:
        manager.set_value(key, 1.0e-6)
        spec, state, rate = _small_generic_reaction_leak()
        with pytest.raises(ValueError, match="closed reaction energy"):
            spec.validate(
                required_level=5, state=state, energy_value=state[0],
                entropy_value=3.0 * torch.log(state[1]), reaction_rates=rate.reshape(1),
            )
    finally:
        manager.set_value(key, original)
    assert manager.get(key, required=True) == original


# ---------------------------------------------------------------------------
# Round 5: the reversible evaluation policy and the reaction representation
# are declared by every structure and every assembly (no implicit choice).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["poisson_operator", "reaction_representation"])
def test_structure_spec_declares_poisson_operator_and_reaction_representation(missing):
    from relic.computation.operators.iphs import IPHSStructureSpec

    fields = dict(
        name="declared_choices", state_vars=("energy", "irreversible_coordinate"),
        energy_functional="E=x0", entropy_functional="S=3*x1",
        J=torch.zeros((2, 2), dtype=torch.float64),
        M=torch.diag(torch.tensor([0.0, 2.0], dtype=torch.float64)),
        G=None, stoichiometric_N=None, ports=(), formulation="generic",
        poisson_operator="zero_field", reaction_representation="independent",
    )
    fields.pop(missing)
    with pytest.raises(TypeError, match=missing):
        IPHSStructureSpec(**fields)


@pytest.mark.parametrize("missing", ["poisson_operator", "reaction_representation"])
def test_assembly_declares_poisson_operator_and_reaction_representation(missing):
    from relic.computation.operators.iphs import assemble_iphs_structure

    arguments = dict(
        name="declared_assembly",
        state_vars=("x", "s"),
        energy_functional="quadratic_energy",
        entropy_functional="linear_entropy",
        contributions=[
            {"J": torch.tensor([[0.0, 1.0], [-1.0, 0.0]])},
            {"M": torch.diag(torch.tensor([0.0, 0.5]))},
        ],
        poisson_operator="matrix_snapshot",
        reaction_representation="independent",
        validate=True,
    )
    arguments.pop(missing)
    with pytest.raises(TypeError, match=missing):
        assemble_iphs_structure(**arguments)
