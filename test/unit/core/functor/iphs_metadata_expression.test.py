from relic.core.functor.functor_types import FunctorMetadata, ProcessCategory, VariableKind
from relic.core.functor.iphs_metadata import (
    IPHS_MARKER,
    ensure_iphs_equivalent_expression,
)


def test_transport_metadata_exposes_canonical_iphs_form_only():
    metadata = FunctorMetadata(
        name="diffusion_test",
        description="Diffusion metadata smoke test",
        category=ProcessCategory.TRANSPORT_MECHANISM,
        mathematical_expression="dc/dt = div(K * grad(c))",
        references=["Ramirez and Le Gorrec IPHS review", "RELIC-IPHS-DEC plan"],
        parameters={
            "transport.diffusion.diffusivity": "Diffusion mobility coefficient",
        },
        input_variables=["concentration"],
        output_variables=["delta_concentration"],
        conservation_laws=["mass"],
    )

    expression = metadata.mathematical_expression

    assert IPHS_MARKER in expression
    assert "Base law:" not in expression
    assert "IPHS equivalent:" not in expression
    assert "dc/dt = div(K * grad(c))" not in expression
    assert "x_dot = J_h e_E + M_h e_S + G_h u_h + N_h r_h" in expression
    assert "non-strict embedding template; not executable proof" in expression
    assert "e_E = delta E_h / delta x_h" not in expression
    assert "e_S = delta S_h / delta x_h" not in expression
    assert "M_h = M_h.T >= 0" not in expression
    assert "sigma_h = e_S.T M_h e_S >= 0" not in expression
    assert "J_h e_S = 0" not in expression
    assert "M_h e_E = 0" not in expression
    assert "energy and entropy gradients are not inferred by metadata" in expression
    assert "operator properties and strict GENERIC degeneracy are not inferred" in expression
    assert "entropy-production closure is not inferred by metadata" in expression
    assert "transport.diffusion.diffusivity" in expression
    assert "ParameterManager-backed metadata.parameters" in expression
    assert "passive transport uses X_h = -D_h phi_h" in expression


def test_biological_metadata_uses_reaction_network_iphs_mapping():
    metadata = FunctorMetadata(
        name="decomposition_test",
        description="Biological process metadata smoke test",
        category=ProcessCategory.BIOLOGICAL_PROCESS,
        mathematical_expression="dC/dt = -k(T,theta) * C",
        references=["Maas and Mielke gradient structures", "RELIC-IPHS-DEC plan"],
        parameters={"pedosphere.decomposition.k_ref": "Reference decay rate"},
        input_variables=["soil_carbon"],
        output_variables=["delta_soil_carbon"],
        conservation_laws=["carbon"],
    )

    expression = metadata.mathematical_expression

    assert "Base law:" not in expression
    assert "dC/dt = -k(T,theta) * C" not in expression
    assert "reaction or growth terms use x_dot = N_h r_h(x_h, theta)" in expression
    assert "requires declared stoichiometry and entropy evidence" in expression


def test_diagnostic_biological_metadata_uses_observation_iphs_mapping():
    metadata = FunctorMetadata(
        name="phenology_diagnostic_test",
        description="Phenology diagnostic metadata smoke test",
        category=ProcessCategory.BIOLOGICAL_PROCESS,
        mathematical_expression="phenology_factor = f(GDD, photoperiod)",
        references=["Richardson et al. 2013", "RELIC-IPHS-DEC plan"],
        parameters={"biosphere.phenology.gdd.threshold_start": "Budburst threshold"},
        input_variables=["accumulated_gdd", "photoperiod"],
        output_variables=["phenological_factor", "growth_active"],
        conservation_laws=["phenology timing"],
        variable_kind=VariableKind.DIAGNOSTIC,
    )

    expression = metadata.mathematical_expression

    assert "Base law:" not in expression
    assert "phenology_factor = f(GDD, photoperiod)" not in expression
    assert "observation: y_h = O_h x_h + B_h u_h" in expression
    assert "diagnostic-only; no state advance is performed" in expression
    assert "typed cochain construction requires certificate or runtime evidence" in expression
    assert "reaction or growth terms use x_dot = N_h r_h" not in expression


def test_the_functor_name_is_required():
    import inspect

    parameter = inspect.signature(ensure_iphs_equivalent_expression).parameters["functor_name"]
    assert parameter.default is inspect.Parameter.empty


def test_iphs_expression_wrapper_is_idempotent():
    first = ensure_iphs_equivalent_expression(
        "ET = alpha * energy",
        category=ProcessCategory.PHYSICAL_FIELD,
        parameters={"hydrosphere.et.alpha": "Priestley-Taylor coefficient"},
        input_variables=["net_radiation"],
        output_variables=["evapotranspiration"],
        conservation_laws=["energy"],
        functor_name="evapotranspiration_test",
    )
    second = ensure_iphs_equivalent_expression(
        first,
        category=ProcessCategory.PHYSICAL_FIELD,
        parameters={"hydrosphere.et.alpha": "Priestley-Taylor coefficient"},
        input_variables=["net_radiation"],
        output_variables=["evapotranspiration"],
        conservation_laws=["energy"],
        functor_name="evapotranspiration_test",
    )

    assert second == first
    assert second.count(IPHS_MARKER) == 1


def test_functor_metadata_exposes_empty_structural_contract_fields():
    metadata = FunctorMetadata(
        name="structural_fields_test",
        description="Structural metadata fields smoke test",
        category=ProcessCategory.TRANSPORT_MECHANISM,
        mathematical_expression="x_dot = -D F",
        references=["RELIC-IPHS-DEC plan"],
        input_variables=["x"],
        output_variables=["delta_x"],
    )

    payload = metadata.to_dict()

    assert metadata.form_contracts == ()
    assert metadata.hodge_contracts == ()
    assert metadata.iphs_terms == ()
    assert metadata.port_power_contracts == ()
    assert metadata.stoichiometric_contracts == ()
    assert metadata.structural_role == ()
    assert payload["form_contracts"] == ()
    assert payload["hodge_contracts"] == ()
    assert payload["iphs_terms"] == ()
    assert payload["port_power_contracts"] == ()
    assert payload["stoichiometric_contracts"] == ()
    assert payload["structural_role"] == ()
