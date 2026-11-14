import re
from pathlib import Path

import pytest
import yaml

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import TransportMechanismFunctor
from relic.parameters import get_global_manager
from relic.scientific.anthroposphere.emissions.sector_allocation import (
    SectorAllocation,
)
from relic.scientific.anthroposphere.emissions.species_mapping import SpeciesMapper
from relic.scientific.anthroposphere.emissions import (
    coupling_to_atmosphere as coupling_module,
)
from relic.scientific.anthroposphere.emissions.coupling_to_atmosphere import (
    AtmosphereCoupling,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_BLOCKER = (
    "[N] runtime_dependency_unbound: AtmosphereCoupling requires the canonical "
    "emission_inventory dependency and six anthropogenic surface-flux providers, "
    "but no executable Graph writer is currently bound"
)
DEAD_COUPLING_PARAMETER_KEYS = (
    "anthroposphere.emissions.coupling.injection_efficiency",
    "anthroposphere.emissions.coupling.conservation_tolerance",
    "anthroposphere.emissions.stack_heights.power_plant",
    "anthroposphere.emissions.stack_heights.industrial",
    "anthroposphere.emissions.vertical_distribution.n_vertical_levels",
    "anthroposphere.emissions.vertical_distribution.gaussian_distribution.power_plant_sigma",
    "anthroposphere.emissions.vertical_distribution.gaussian_distribution.industrial_sigma",
    "anthroposphere.emissions.vertical_distribution.surface_distribution.layer_0_fraction",
    "anthroposphere.emissions.vertical_distribution.surface_distribution.layer_1_fraction",
    "anthroposphere.emissions.vertical_distribution.surface_distribution.layer_2_fraction",
    "anthroposphere.emissions.emission.sector.default",
)
ADJACENT_UNREVIEWED_PARAMETER_KEYS = (
    "anthroposphere.emissions.stack_heights.residential",
    "anthroposphere.emissions.vertical_distribution.mixing_height",
    "anthroposphere.emissions.clamp.max_sector",
    "anthroposphere.emissions.clamp.min_sector",
    "anthroposphere.emissions.atmospheric_stability.default_class",
    "anthroposphere.emissions.atmospheric_stability.neutral_class",
)


class _ExplosiveState(dict):
    def __contains__(self, key):
        raise AssertionError(f"state membership was accessed: {key}")

    def __getitem__(self, key):
        raise AssertionError(f"state item was accessed: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"state get was accessed: {key}")

    def items(self):
        raise AssertionError("state items were accessed")


def test_atmosphere_coupling_keeps_only_the_canonical_class_identity():
    assert issubclass(AtmosphereCoupling, TransportMechanismFunctor)
    assert coupling_module.__all__ == ["AtmosphereCoupling"]
    assert not hasattr(coupling_module, "VerticalProfile")


def test_constructor_rejects_before_functor_or_parameter_effects(monkeypatch):
    base_calls = []

    def _reject_base_init(*args, **kwargs):
        base_calls.append((args, kwargs))
        raise AssertionError("FunctorBase initialization was reached")

    monkeypatch.setattr(TransportMechanismFunctor, "__init__", _reject_base_init)

    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        AtmosphereCoupling(
            name="forbidden",
            config={"bvoc_emission_owner": "megan_bvoc_model"},
        )

    assert base_calls == []


@pytest.mark.parametrize(
    "args,kwargs",
    [
        pytest.param((), {}, id="default"),
        pytest.param(({"injection_efficiency": 0.95},), {}, id="dict-name"),
        pytest.param(
            (),
            {"name": "coupling", "bvoc_emission_owner": "megan_bvoc_model"},
            id="selected-bvoc",
        ),
    ],
)
def test_constructor_has_one_deterministic_runtime_blocker(args, kwargs):
    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        AtmosphereCoupling(*args, **kwargs)


def test_unsafe_constructor_bypass_cannot_read_state_or_emit_updates():
    coupling = object.__new__(AtmosphereCoupling)
    state = _ExplosiveState()

    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        AtmosphereCoupling.forward(coupling, state, time=0.0, dt=60.0)

    assert dict(state) == {}
    assert not hasattr(coupling, "_parameters")
    assert not hasattr(coupling, "controls")
    assert not hasattr(coupling, "cf_mapping")


def test_coupling_exposes_no_formula_metadata_or_false_structure():
    forbidden_class_api = {
        "compute",
        "get_dependencies",
        "required_parameters",
        "validate_state",
        "check_conservation_laws",
        "declared_forms",
        "build_hodge_contract",
        "build_hodge_terms",
        "energy_terms",
        "entropy_terms",
        "reversible_structure",
        "dissipative_structure",
        "ports",
        "_inject_surface_fluxes",
        "_calculate_vertical_distribution",
        "_check_conservation",
        "_apply_cf_mapping",
        "ANTHROPOGENIC_FLUX_PORTS",
        "BVOC_SURFACE_PORTS",
        "PHASE8_RUNTIME_REPLACEMENT_STATUS",
    }

    assert forbidden_class_api.isdisjoint(AtmosphereCoupling.__dict__)


def test_coupling_source_contains_only_the_fail_closed_boundary():
    source = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "anthroposphere"
        / "emissions"
        / "coupling_to_atmosphere.py"
    ).read_text(encoding="utf-8")
    forbidden_tokens = (
        "import torch",
        "UpdateItem(",
        "ParameterManager",
        "AnthroposphereCFMapping",
        "canonical_iphs_runtime",
        "runtime_path",
        "emission_field_3d",
        "bvoc_emission_field_3d",
        "_inject_surface_fluxes",
        "_calculate_vertical_distribution",
        "torch.exp",
        "torch.stack",
    )

    assert "[N] runtime_dependency_unbound" in source
    for token in forbidden_tokens:
        assert token not in source


def test_anthropogenic_3d_field_contract_has_no_unbound_coupling_writer():
    path = (
        PROJECT_ROOT
        / "relic"
        / "contracts"
        / "variables"
        / "anthroposphere"
        / "emission_field_3d.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["dimension"]["cf_standard_name"] is None
    assert payload["dimension"]["udunits"] == "kg m-2 s-1"
    assert payload["title"] == "Vertically Distributed Anthropogenic Emission Flux"
    assert payload["owner"] is None
    assert payload["writers"] == {"increments": [], "tendencies": []}
    assert payload["flux_contract"]["allowed_in"] == []
    assert payload["provenance"]["runtime_writer"] is None
    assert payload["provenance"]["runtime_role"] == "runtime_dependency_unbound"


def test_bvoc_3d_field_contract_has_no_unbound_coupling_writer():
    path = (
        PROJECT_ROOT
        / "relic"
        / "contracts"
        / "variables"
        / "atmosphere"
        / "bvoc_emission_field_3d.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["dimension"]["cf_standard_name"] is None
    assert payload["dimension"]["udunits"] == "mol m-2 s-1"
    assert payload["title"] == "Vertically Distributed Biogenic VOC Molar Emission Flux"
    assert payload["owner"] is None
    assert payload["writers"] == {"increments": [], "tendencies": []}
    assert payload["flux_contract"]["allowed_in"] == []
    assert payload["provenance"]["runtime_writer"] is None
    assert payload["provenance"]["runtime_role"] == "runtime_dependency_unbound"


def _assert_bvoc_source_has_no_blocked_coupling_route(variable: str) -> None:
    path = (
        PROJECT_ROOT
        / "relic"
        / "contracts"
        / "variables"
        / "biosphere"
        / f"{variable}.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["writers"]["tendencies"]
    assert "atmosphere_coupling" not in payload["flux_contract"]["allowed_out"]


def test_isoprene_source_has_no_blocked_coupling_route():
    _assert_bvoc_source_has_no_blocked_coupling_route("isoprene_emission")


def test_monoterpenes_source_has_no_blocked_coupling_route():
    _assert_bvoc_source_has_no_blocked_coupling_route("monoterpenes_emission")


def test_sesquiterpenes_source_has_no_blocked_coupling_route():
    _assert_bvoc_source_has_no_blocked_coupling_route("sesquiterpenes_emission")


def test_ovocs_source_has_no_blocked_coupling_route():
    _assert_bvoc_source_has_no_blocked_coupling_route("ovocs_emission")


def test_coupling_certificate_is_level_zero_and_non_operational():
    path = (
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_emissions_coupling_to_atmosphere_atmosphere_coupling.yaml"
    )
    certificate = load_iphs_equivalence_certificate(path)
    audit = certificate.validate()
    source = path.read_text(encoding="utf-8")

    assert certificate.iphs_level == 0
    assert audit["iphs_level_name"] == "ordinary_empirical"
    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert audit["runtime_evidence_status"] == "descriptive_only"
    assert certificate.input_variables == ()
    assert certificate.output_variables == ()
    assert certificate.composition_ports == ()
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False
    assert "[N] runtime_dependency_unbound" in source
    for token in (
        "Hodge",
        "PortSpec",
        "canonical_iphs_runtime",
        "StateManager commit",
        "runtime_verified_level: 1",
    ):
        assert token not in source


def test_obsolete_coupling_formula_and_fake_runtime_suites_are_removed():
    obsolete = (
        PROJECT_ROOT / "test" / "anthroposphere" / "emissions" / "test_emissions_l1.py",
        PROJECT_ROOT / "test" / "anthroposphere" / "emissions" / "test_L1_unit.py",
        PROJECT_ROOT / "test" / "anthroposphere" / "emissions" / "test_L2_integration.py",
        PROJECT_ROOT / "test" / "anthroposphere" / "emissions" / "test_L3_system.py",
        PROJECT_ROOT
        / "test"
        / "integration"
        / "anthroposphere"
        / "emissions"
        / "test_emissions_integration.py",
    )

    assert not [path for path in obsolete if path.exists()]


def test_obsolete_coupling_system_and_unit_formula_suites_are_removed():
    obsolete = (
        PROJECT_ROOT
        / "test"
        / "system"
        / "anthroposphere"
        / "emissions"
        / "test_emissions_system.py",
        PROJECT_ROOT
        / "test"
        / "unit"
        / "anthroposphere"
        / "emissions"
        / "test_coupling.py",
    )

    assert not [path for path in obsolete if path.exists()]


def test_obsolete_direct_anthro_atmosphere_integration_is_removed():
    path = (
        PROJECT_ROOT
        / "test"
        / "integration"
        / "coupling"
        / "anthro_atmosphere_coupling_test.py"
    )
    assert not path.exists()


def test_deleted_coupling_parameters_are_unknown_without_scope_expansion():
    manager = get_global_manager()

    for key in DEAD_COUPLING_PARAMETER_KEYS:
        assert manager.get_parameter(key) is None
        with pytest.raises(KeyError, match=re.escape(key)):
            manager.get(key, required=True)

    assert manager.get_parameter(
        "anthroposphere.emissions.vertical_distribution.layer_thickness"
    ) is not None
    for key in ADJACENT_UNREVIEWED_PARAMETER_KEYS:
        assert manager.get_parameter(key) is not None

    for owner in (
        SectorAllocation(),
        SpeciesMapper(),
    ):
        for key in owner.required_parameters():
            assert manager.get_parameter(key) is not None, key
