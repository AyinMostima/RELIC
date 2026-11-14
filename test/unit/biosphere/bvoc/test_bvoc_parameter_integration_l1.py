"""
Active L1 tests for MEGAN ParameterManager and physical-constant binding.

This file owns parameter-schema and runtime-binding coverage for the reduced
instantaneous MEGAN closure. Generic environmental-response helpers are tested
separately in ``test_responses_parameter_integration_l1.py``.
"""

import pytest
import torch
from relic.core.state.types import UpdateItem
from relic.parameters import get_global_manager
from relic.scientific.biosphere.bvoc.megan_model import (
    CouplingStrategy,
    EmissionPathway,
    MEGANModel,
)
from relic.utils.physical_constants import get_physical_constant


COMPOUNDS = ("isoprene", "monoterpenes", "sesquiterpenes", "ovocs")
EXPECTED_PARAMETER_KEYS = {
    "biosphere.bvoc.megan.hybrid_weight",
    *(f"biosphere.bvoc.carbon_fraction.{name}" for name in COMPOUNDS),
    "biosphere.bvoc.validation.temperature.min_operational",
    "biosphere.bvoc.validation.temperature.max_operational",
    *(f"biosphere.bvoc.emission_factors.{name}" for name in COMPOUNDS),
    "biosphere.bvoc.megan.alpha_light",
    "biosphere.bvoc.megan.cl_light",
    "biosphere.bvoc.megan.ldf_monoterpenes",
    "biosphere.bvoc.megan.ldf_sesquiterpenes",
    "biosphere.bvoc.megan.ovoc.methanol.global_emission",
    "biosphere.bvoc.megan.ovoc.methanol.beta",
    "biosphere.bvoc.megan.ovoc.methanol.ldf",
    "biosphere.bvoc.megan.ovoc.methanol.ct1",
    "biosphere.bvoc.megan.ovoc.methanol.eopt",
    "biosphere.bvoc.megan.ovoc.acetone.global_emission",
    "biosphere.bvoc.megan.ovoc.acetone.beta",
    "biosphere.bvoc.megan.ovoc.acetone.ldf",
    "biosphere.bvoc.megan.ovoc.acetone.ct1",
    "biosphere.bvoc.megan.ovoc.acetone.eopt",
    "biosphere.bvoc.megan.ovoc.acetaldehyde.global_emission",
    "biosphere.bvoc.megan.ovoc.acetaldehyde.beta",
    "biosphere.bvoc.megan.ovoc.acetaldehyde.ldf",
    "biosphere.bvoc.megan.ovoc.acetaldehyde.ct1",
    "biosphere.bvoc.megan.ovoc.acetaldehyde.eopt",
    "biosphere.bvoc.megan.topt",
    "biosphere.bvoc.t_standard",
    "biosphere.bvoc.megan.eopt_isoprene",
    "biosphere.bvoc.megan.ct1_isoprene",
    "biosphere.bvoc.megan.ct2",
    "biosphere.bvoc.megan.ct1_monoterpenes",
    "biosphere.bvoc.megan.eopt_monoterpenes",
    "biosphere.bvoc.megan.ct1_sesquiterpenes",
    "biosphere.bvoc.megan.eopt_sesquiterpenes",
    *(
        f"biosphere.bvoc.photosynthetic_coupling.f_carbon_{name}"
        for name in COMPOUNDS
    ),
    "relic.physical.constants.molecular_weights.isoprene",
    *(
        f"biosphere.bvoc.molecular_weights.{name}"
        for name in COMPOUNDS
        if name != "isoprene"
    ),
    *(f"biosphere.bvoc.carbon_atoms.{name}" for name in COMPOUNDS),
    "biosphere.bvoc.foliar_density_factor",
    "biosphere.bvoc.megan.beta_monoterpenes",
    "biosphere.bvoc.megan.beta_sesquiterpenes",
    "biosphere.bvoc.co2_inhibition.ismax",
    "biosphere.bvoc.co2_inhibition.c_star",
    "biosphere.bvoc.co2_inhibition.hill_exponent",
    "biosphere.bvoc.co2_inhibition.intercellular_to_ambient_ratio",
    "validation.numerical.carbon_flux_min",
    "validation.numerical.epsilon",
    "relic.physical.constants.conversions.micrograms_to_grams",
    "relic.physical.constants.universal.gas_constant",
    "relic.physical.constants.time.seconds_per_hour",
}
# forward() ends with FunctorBase._enforce_forward_conservation, which reads these two
# framework keys; the metadata declares them next to the closure parameters.
DIRECT_FORWARD_CONSERVATION_KEYS = {
    "validation.conservation.conservation_tolerance",
    "validation.conservation.strict_conservation",
}


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def _standard_state():
    return {
        "air_temperature": torch.tensor(303.15),
        "ppfd": torch.tensor(1000.0),
        "leaf_area_index": torch.tensor(1.0),
        "atmospheric_co2_ppm": torch.tensor(400.0),
    }


class TestParameterNamespaceResolution:
    """Test that all parameters use correct namespaces."""

    def test_emission_factors_use_default_namespace(self):
        """Emission factors must use the single canonical parameter namespace."""
        model = MEGANModel()
        manager = get_global_manager()
        for compound in COMPOUNDS:
            key = f"biosphere.bvoc.emission_factors.{compound}"
            assert key in model.metadata.parameters
            assert _scalar(model.emission_factors[compound]) == pytest.approx(
                _scalar(manager.get(key)), rel=1.0e-6
            )
            assert f"biosphere.bvoc.default.{compound}" not in model.metadata.parameters

    def test_no_shadow_parameters_created(self):
        """Ensure no incorrect parameters are auto-registered."""
        param_manager = get_global_manager()

        # Check that incorrect namespace parameters don't exist
        # These would be created if namespace was wrong (e.g., "biosphere.bvoc.isoprene")
        missing_requests = param_manager.get_missing_requests()

        # Verify no auto-registered shadow parameters for base emission factors
        if 'biosphere.bvoc.isoprene' in missing_requests:
            pytest.fail("Shadow parameter 'biosphere.bvoc.isoprene' was created - namespace error!")
        if 'biosphere.bvoc.monoterpenes' in missing_requests:
            pytest.fail("Shadow parameter 'biosphere.bvoc.monoterpenes' was created - namespace error!")

    def test_megan_parameters_namespace(self):
        """Runtime metadata must expose only parameters used by this closure."""
        model = MEGANModel()
        assert set(model.metadata.parameters) == (
            EXPECTED_PARAMETER_KEYS | DIRECT_FORWARD_CONSERVATION_KEYS
        )
        assert len(model.metadata.parameters) == 65


class TestPhysicalConstants:
    """Test physical constants are correctly loaded."""

    def test_gas_constant_correct_value(self):
        """The runtime buffer must equal the physical-constant SSOT."""
        model = MEGANModel()
        expected = _scalar(get_physical_constant("gas_constant"))
        assert _scalar(model.gas_constant) == pytest.approx(expected, rel=1.0e-7)
        assert model.gas_constant.dtype == torch.float32

    def test_physical_constants_namespace(self):
        """Verify physical constants use correct namespace."""
        param_manager = get_global_manager()

        gas_const = param_manager.get(
            "relic.physical.constants.universal.gas_constant"
        )
        assert _scalar(gas_const) == pytest.approx(
            _scalar(get_physical_constant("gas_constant")), rel=1.0e-12
        )

    def test_mass_and_time_conversions_use_physical_constant_ssot(self):
        model = MEGANModel()
        assert _scalar(model.micrograms_to_grams) == pytest.approx(
            _scalar(
                get_physical_constant(
                    "relic.physical.constants.conversions.micrograms_to_grams"
                )
            ),
            rel=1.0e-7,
        )
        assert _scalar(model.seconds_per_hour) == pytest.approx(
            _scalar(get_physical_constant("seconds_per_hour")), rel=1.0e-7
        )


class TestEnvironmentalResponseParameters:
    """Test the environmental responses owned directly by MEGANModel."""

    def test_co2_inhibition_uses_parameter_manager(self):
        """Guenther Eq. 14 coefficients must come from canonical keys."""
        manager = get_global_manager()
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        ca = torch.tensor([400.0])
        ismax = _scalar(manager.get("biosphere.bvoc.co2_inhibition.ismax"))
        c_star = _scalar(manager.get("biosphere.bvoc.co2_inhibition.c_star"))
        hill = _scalar(
            manager.get("biosphere.bvoc.co2_inhibition.hill_exponent")
        )
        ratio = _scalar(
            manager.get(
                "biosphere.bvoc.co2_inhibition.intercellular_to_ambient_ratio"
            )
        )
        ci = ratio * ca
        expected = ismax - ismax * ci.pow(hill) / (c_star**hill + ci.pow(hill))
        assert torch.allclose(
            model._compute_co2_inhibition_factor(ca), expected, rtol=1.0e-6
        )

    def test_water_stress_uses_parameter_manager(self):
        """Stress-enabled MEGAN consumes declared factors without duplicating laws."""
        model = MEGANModel(
            config={"use_stress_functors": True},
            emission_pathway=EmissionPathway.STANDARD,
        )
        assert {
            "water_stress_factor",
            "temperature_stress_factor",
            "ozone_stress_factor",
        }.issubset(model.get_required_inputs())
        assert "biosphere.bvoc.water_stress.wilting_point" not in set(
            model.metadata.parameters
        )

    def test_no_hardcoded_defaults_in_responses(self):
        """Required environmental inputs cannot be synthesized from parameters."""
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        for missing in _standard_state():
            state = _standard_state()
            state.pop(missing)
            with pytest.raises(ValueError, match=missing):
                model.forward(state, time=0.0, dt=60.0)


class TestParameterUnitsAndBounds:
    """Test that parameters have proper units and bounds specified."""

    def test_foliar_mass_conversion_has_one_explicit_area_basis(self):
        """The reduced closure must expose one unambiguous LAI-area conversion."""
        manager = get_global_manager()
        conversion = manager.get_parameter(
            "biosphere.bvoc.foliar_density_factor"
        )

        assert conversion is not None
        assert conversion.units == "g DW m-2"
        assert conversion.dimension == "mass_per_area"
        assert {
            "dry_weight",
            "canonical_lai_area_basis",
            "lai_conversion",
        }.issubset(conversion.tags)
        assert "canonical RELIC LAI leaf area" in conversion.description
        assert manager.get_parameter("biosphere.defaults.foliar_density") is None

    def test_emission_factors_have_units(self):
        """Emission-factor units and bounds must be supplied by YAML specs."""
        manager = get_global_manager()
        for compound in COMPOUNDS:
            spec = manager.get_parameter(
                f"biosphere.bvoc.emission_factors.{compound}"
            )
            assert spec is not None
            assert spec.units == "ug g-1 DW h-1"
            assert spec.bounds[0] >= 0.0
            assert spec.bounds[0] <= _scalar(spec.value) <= spec.bounds[1]

    def test_molecular_weights_have_units(self):
        """Verify molecular weights are accessed with proper units."""
        model = MEGANModel()
        manager = get_global_manager()
        for compound in COMPOUNDS:
            key = (
                "relic.physical.constants.molecular_weights.isoprene"
                if compound == "isoprene"
                else f"biosphere.bvoc.molecular_weights.{compound}"
            )
            spec = manager.get_parameter(key)
            assert spec is not None
            assert spec.units == "g mol-1"
            assert model.molecular_weights[compound] == pytest.approx(
                _scalar(spec.value)
            )
        assert (
            "biosphere.bvoc.molecular_weights.isoprene"
            not in model.metadata.parameters
        )

    def test_temperature_parameters_have_units(self):
        """Test temperature-related parameters have K units."""
        model = MEGANModel()
        manager = get_global_manager()
        bindings = {
            "biosphere.bvoc.megan.topt": model.topt,
            "biosphere.bvoc.t_standard": model.temp_std,
            "biosphere.bvoc.megan.ct1_isoprene": model.ct1_iso,
            "biosphere.bvoc.megan.ct2": model.ct2,
        }
        for key, actual in bindings.items():
            spec = manager.get_parameter(key)
            assert spec is not None
            expected_unit = "K" if "topt" in key or "t_standard" in key else "J mol-1"
            assert spec.units == expected_unit
            assert _scalar(actual) == pytest.approx(_scalar(spec.value), rel=1.0e-6)


class TestParameterConsistency:
    """Test parameter values are consistent across modules."""

    def test_gas_constant_consistent(self):
        """The model and public SSOT accessor must return the same constant."""
        model = MEGANModel()
        assert _scalar(model.gas_constant) == pytest.approx(
            _scalar(get_physical_constant("gas_constant")), rel=1.0e-7
        )

    def test_reference_temperatures_consistent(self):
        """Check reference temperatures are consistent."""
        model = MEGANModel()
        param_manager = get_global_manager()
        reference = param_manager.get("biosphere.bvoc.t_standard")
        assert _scalar(model.temp_std) == pytest.approx(
            _scalar(reference), rel=1.0e-6
        )

    def test_co2_parameters_match_database(self):
        """All four Guenther Eq. 14 parameters must be complete YAML specs."""
        model = MEGANModel()
        manager = get_global_manager()
        keys = {
            "biosphere.bvoc.co2_inhibition.ismax",
            "biosphere.bvoc.co2_inhibition.c_star",
            "biosphere.bvoc.co2_inhibition.hill_exponent",
            "biosphere.bvoc.co2_inhibition.intercellular_to_ambient_ratio",
        }
        assert keys.issubset(model.metadata.parameters)
        for key in keys:
            spec = manager.get_parameter(key)
            assert spec is not None
            assert spec.units in {"dimensionless", "ppm"}
            assert spec.description
            assert spec.references
            assert spec.provenance


class TestParameterAudit:
    """Test complete ParameterSpec coverage for runtime parameters."""

    def test_no_missing_required_parameters(self):
        """Ensure all required BVOC parameters exist in database."""
        manager = get_global_manager()
        missing = [
            key for key in EXPECTED_PARAMETER_KEYS
            if manager.get_parameter(key) is None
        ]
        assert missing == []

    def test_parameter_units_valid(self):
        """Every MEGAN parameter must carry the complete YAML schema."""
        manager = get_global_manager()
        for key in EXPECTED_PARAMETER_KEYS:
            spec = manager.get_parameter(key)
            assert spec is not None
            assert spec.units
            assert spec.dtype
            assert spec.bounds is not None and len(spec.bounds) == 2
            assert spec.category is not None
            assert spec.description
            assert spec.references
            assert isinstance(spec.tags, set)
            assert isinstance(spec.requires_grad, bool)
            assert spec.provenance

    def test_representative_units_are_exact(self):
        manager = get_global_manager()
        expected_units = {
            "biosphere.bvoc.emission_factors.isoprene": "ug g-1 DW h-1",
            "biosphere.bvoc.megan.topt": "K",
            "biosphere.bvoc.megan.ct1_isoprene": "J mol-1",
            "relic.physical.constants.universal.gas_constant": "J mol-1 K-1",
            "relic.physical.constants.conversions.micrograms_to_grams": "g ug-1",
        }
        for key, unit in expected_units.items():
            spec = manager.get_parameter(key)
            assert spec is not None
            assert spec.units == unit


class TestModelInitialization:
    """Test MEGAN model initializes correctly with parameter system."""

    def test_model_initializes_without_errors(self):
        """Basic smoke test for model initialization."""
        model = MEGANModel()
        assert model.name == "megan_bvoc_model"
        assert model.device.type == "cpu"

    def test_model_has_all_required_attributes(self):
        """Constants and non-learnable coefficients are module buffers; the learnable
        database coefficients are the ParameterManager bindings held by reference
        (architecture SSOT Sections 4.2 and 4.4.2), never buffer copies."""
        model = MEGANModel()
        required_buffers = {
            "micrograms_to_grams",
            "gas_constant",
            "seconds_per_hour",
            "temp_std",
            *(f"{name}_carbon_fraction" for name in COMPOUNDS),
            *(f"f_carbon_{name}" for name in COMPOUNDS),
        }
        assert required_buffers.issubset(dict(model.named_buffers()))
        held_attributes = (
            "alpha_light", "cl_light", "topt", "eopt_isoprene", "ct1_iso", "ct2",
            *(f"{name}_ef" for name in COMPOUNDS),
        )
        buffers = dict(model.named_buffers())
        for attribute in held_attributes:
            assert model.manager_binding_key(getattr(model, attribute)) is not None, attribute
            assert attribute not in buffers, attribute
        assert set(model.molecular_weights) == set(COMPOUNDS)
        assert set(model.carbon_atoms) == set(COMPOUNDS)

    def test_coupling_strategies_work(self):
        """Test different coupling strategies initialize correctly."""
        strategies = [
            CouplingStrategy.INDEPENDENT,
            CouplingStrategy.SEQUENTIAL,
            CouplingStrategy.SIMULTANEOUS
        ]

        for strategy in strategies:
            model = MEGANModel(coupling_strategy=strategy)
            assert model.coupling_strategy == strategy


class TestForwardPass:
    """Test model forward pass with parameter system."""

    def test_forward_pass_with_minimal_inputs(self):
        """Test forward pass works with minimal required inputs."""
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        output = model.forward(_standard_state(), time=0.0, dt=1.0)
        assert set(output) == set(model.metadata.output_variables)
        assert isinstance(output["total_bvoc_emission"], UpdateItem)
        assert output["total_bvoc_emission"].value.item() > 0.0

    def test_parameter_sensitivity_analysis(self):
        """Sensitivity of a non-learnable member through the base path; a learnable
        binding is refused there and goes through the manager calibration path."""
        model = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
        sensitivity = model.parameter_sensitivity(
            "temp_std", _standard_state(), time=0.0, dt=1.0,
            target_output="total_bvoc_emission",
        )
        assert sensitivity.numel() == 1
        assert torch.isfinite(sensitivity).all()
        assert sensitivity.item() != 0.0
        with pytest.raises(TypeError, match="calibration path"):
            model.parameter_sensitivity(
                "alpha_light", _standard_state(), time=0.0, dt=1.0,
                target_output="total_bvoc_emission",
            )


class TestParameterLiteratureAudit:
    """Every BVOC parameter cites registered works and records whether its value
    was found in them (literature audit of bvoc_parameters.yaml)."""

    RETIRED_CITATIONS = (
        "acp-8-4129-2008",
        "Atmos Chem Phys 8:4129",
        "pp.109.144568",
        "j.1365-3040.2008.01824.x",
        "j.1469-8137.2007.02330.x",
        "j.1365-3040.2011.02472.x",
        "phytochem.2004.06.023",
        "pnas.0810258106",
        "j.1365-3040.2009.01946.x",
        "nph.12768",
        "tplants.2010.01.004",
        "j.1365-3040.2010.02164.x",
        "978-0-470-01542-1",
        "Plant Cell Environ 32:1671",
        "Plant Cell Environ 34:1671",
        "ESSD 14, 3833",
    )
    CHECK_PREFIXES = (
        "verified",
        "not verified in the cited literature (model default)",
    )

    @staticmethod
    def _database():
        from pathlib import Path

        import yaml

        path = (
            Path(__file__).resolve().parents[4]
            / "relic"
            / "parameters"
            / "database"
            / "relic"
            / "biosphere"
            / "bvoc_parameters.yaml"
        )
        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))

    def test_references_carry_no_unregistered_or_misattributed_citation(self):
        """Unregistered DOIs, DOIs of unrelated articles and wrong journal locators
        are replaced by registered records."""
        data = self._database()
        offenders = []
        for key, spec in data["parameters"].items():
            references = spec.get("references") or []
            assert references, key
            for reference in references:
                offenders.extend(
                    (key, retired)
                    for retired in self.RETIRED_CITATIONS
                    if retired in str(reference)
                )
        for reference in data["functor_metadata"]["references"]:
            offenders.extend(
                ("functor_metadata", retired)
                for retired in self.RETIRED_CITATIONS
                if retired in reference
            )
        assert offenders == []

    def test_every_value_records_its_literature_check(self):
        """Each value states whether it was found in the cited literature."""
        data = self._database()
        unlabelled = [
            key
            for key, spec in data["parameters"].items()
            if not str(
                (spec.get("provenance") or {}).get("literature_check", "")
            ).startswith(self.CHECK_PREFIXES)
        ]
        assert unlabelled == []

    def test_database_declares_no_unloaded_conversion_factor_section(self):
        """MA-21 16.3: the loader reads only the parameters section of a file that has
        one, so a top-level conversion_factors mapping is never loaded. Its SI factors
        (ug -> kg, g -> kg, h -> s and their 1e-9/3600 products) had no reader and
        duplicate relic.utils.time_constants and the UnitConverter."""
        data = self._database()
        assert "conversion_factors" not in data
        unloaded = [
            section
            for section, content in data.items()
            if section != "parameters"
            and isinstance(content, dict)
            and any(
                isinstance(entry, dict) and "value" in entry and "units" in entry
                for entry in content.values()
            )
        ]
        assert unloaded == []

    def test_verified_values_name_their_location(self):
        """Verified values point to the table or equation that contains them;
        RELIC calibrations are labelled as unverified model defaults."""
        data = self._database()
        checks = {
            key: (spec.get("provenance") or {}).get("literature_check", "")
            for key, spec in data["parameters"].items()
        }
        assert checks["biosphere.bvoc.megan.beta_sesquiterpenes"].startswith("verified")
        assert "Table 4" in checks["biosphere.bvoc.megan.beta_sesquiterpenes"]
        assert checks["biosphere.bvoc.megan.eopt_isoprene"].startswith("verified")
        assert "Table 4" in checks["biosphere.bvoc.megan.eopt_isoprene"]
        assert checks["biosphere.bvoc.sesquiterpene.stress_threshold"].startswith(
            "not verified in the cited literature (model default)"
        )

    def test_standard_temperature_is_paired_with_the_emission_factor_basis(self):
        """T_s = 303.15 K belongs to the 30 degC dry-mass emission factors RELIC
        uses (Duhl et al. 2008 Eq. 1; Guenther et al. 2006 standard air
        temperature); the MEGAN2.1 T_s = 297 K of Guenther et al. (2012) Eqs. (9)-(11)
        belongs to the MEGAN2.1 Table 2 emission factors, and the pairs are not mixed."""
        spec = self._database()["parameters"]["biosphere.bvoc.t_standard"]
        assert spec["value"] == 303.15
        description = spec["description"]
        for fragment in ("303.15 K", "30 degC", "297 K", "must not be mixed"):
            assert fragment in description, fragment
        references = " | ".join(str(reference) for reference in spec["references"])
        for fragment in (
            "Duhl et al. (2008)",
            "Sect. 2.1, Eq. (1)",
            "Guenther et al. (2006)",
            "Guenther et al. (2012)",
            "Eqs. (9)-(11)",
        ):
            assert fragment in references, fragment

    RETIRED_UNREAD_PARAMETERS = {
        ("biosphere", "bvoc_parameters.yaml"): (
            "biosphere.bvoc.megan.amax",
            "biosphere.bvoc.megan.theta_light",
            "biosphere.bvoc.megan.ct1_light",
            "biosphere.bvoc.megan.eopt_ovocs",
            "biosphere.bvoc.megan.ldf_ovocs",
            "biosphere.bvoc.megan.beta_ovocs",
            "biosphere.bvoc.megan.ppfd_standard",
            "biosphere.bvoc.megan.beta_temperature",
            "biosphere.bvoc.megan.monoterpene_mw",
            "biosphere.bvoc.mep.v_max_dxr",
            "biosphere.bvoc.mep.temp_ref",
            "biosphere.bvoc.q10_factor_min",
            "biosphere.bvoc.q10_factor_max",
            "biosphere.bvoc.co2_inhibition.reference_concentration",
            "biosphere.bvoc.co2_inhibition.inhibition_factor",
            "biosphere.bvoc.pft_aliases",
        ),
        ("validation_numerical.yaml",): (
            "validation.numerical.light_factor_min",
            "validation.numerical.light_factor_max",
            "validation.numerical.isoprene_temp_factor_min",
            "validation.numerical.isoprene_temp_factor_max",
            "validation.numerical.emission_min",
        ),
        ("biosphere", "constants.yaml"): (
            "biosphere.constants.q10.temperature_interval",
            "biosphere.constants.temperature.standard_25c",
        ),
    }

    def test_parameters_without_readers_are_removed(self):
        """Entries that no code reads, including the monoterpene molar-mass alias,
        the unread MEP reference temperature (303.15 K) and its domain copy of the
        25 degC physical constant, are removed from the database."""
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[4] / "relic" / "parameters" / "database" / "relic"
        present = []
        for parts, keys in self.RETIRED_UNREAD_PARAMETERS.items():
            parameters = yaml.safe_load(root.joinpath(*parts).read_text(encoding="utf-8-sig"))[
                "parameters"
            ]
            present.extend(key for key in keys if key in parameters)
        assert present == []

    def test_monoterpene_light_dependent_fraction_is_the_representative_table4_value(self):
        """RELIC's lumped monoterpene class (undifferentiated C10H16) uses the
        Guenther et al. (2012) Table 4 'Other Monoterpenes' LDF as a representative
        value, stated as such."""
        spec = self._database()["parameters"]["biosphere.bvoc.megan.ldf_monoterpenes"]
        assert spec["value"] == 0.4
        assert spec["units"] == "dimensionless"
        assert spec["bounds"] == [0.0, 1.0]
        check = spec["provenance"]["literature_check"]
        assert check.startswith("verified")
        assert "Other Monoterpenes" in check and "representative" in check
        assert any("Table 4" in str(reference) for reference in spec["references"])

    def test_temperature_activity_parameters_come_from_one_table4_row(self):
        """beta, LDF, C_T1 and C_eo of each class come from one Guenther et al. (2012)
        Table 4 row (Isoprene, Other Monoterpenes, Other Sesquiterpenes); the
        superseded mixed-source values (monoterpene beta 0.09 of Guenther et al. 1993,
        isoprene E_opt 2.034 of Guenther et al. 2006 Eq. 9) are recorded as such."""
        parameters = self._database()["parameters"]
        rows = {
            "Isoprene": {"ct1_isoprene": 95000.0, "eopt_isoprene": 2.0},
            "Other Monoterpenes": {
                "beta_monoterpenes": 0.1,
                "ldf_monoterpenes": 0.4,
                "ct1_monoterpenes": 80000.0,
                "eopt_monoterpenes": 1.83,
            },
            "Other Sesquiterpenes": {
                "beta_sesquiterpenes": 0.17,
                "ldf_sesquiterpenes": 0.5,
                "ct1_sesquiterpenes": 130000.0,
                "eopt_sesquiterpenes": 2.37,
            },
            "Methanol": {
                "ovoc.methanol.beta": 0.08,
                "ovoc.methanol.ldf": 0.8,
                "ovoc.methanol.ct1": 60000.0,
                "ovoc.methanol.eopt": 1.6,
            },
            "Acetone": {
                "ovoc.acetone.beta": 0.1,
                "ovoc.acetone.ldf": 0.2,
                "ovoc.acetone.ct1": 80000.0,
                "ovoc.acetone.eopt": 1.83,
            },
            "Bidirectional VOC": {
                "ovoc.acetaldehyde.beta": 0.13,
                "ovoc.acetaldehyde.ldf": 0.8,
                "ovoc.acetaldehyde.ct1": 95000.0,
                "ovoc.acetaldehyde.eopt": 2.0,
            },
        }
        for name, emission in (("methanol", 9.96e13), ("acetone", 4.37e13), ("acetaldehyde", 2.07e13)):
            spec = parameters[f"biosphere.bvoc.megan.ovoc.{name}.global_emission"]
            assert spec["value"] == pytest.approx(emission) and spec["units"] == "g yr-1"
            check = spec["provenance"]["literature_check"]
            assert check.startswith("verified") and "Table 6" in check, name
        for row, values in rows.items():
            for name, value in values.items():
                spec = parameters[f"biosphere.bvoc.megan.{name}"]
                assert spec["value"] == pytest.approx(value), name
                check = spec["provenance"]["literature_check"]
                assert check.startswith("verified"), name
                assert "Table 4" in check and row in check, name
        monoterpene_beta = parameters["biosphere.bvoc.megan.beta_monoterpenes"]
        isoprene_eopt = parameters["biosphere.bvoc.megan.eopt_isoprene"]
        for spec, former in ((monoterpene_beta, "0.09"), (isoprene_eopt, "2.034")):
            check = spec["provenance"]["literature_check"]
            assert former in check and "superseded" in check

    def test_ovoc_class_is_the_megan21_emission_weighted_equivalent(self):
        """The lumped OVOC class (methanol, acetone, acetaldehyde) takes the MEGAN2.1
        global-emission-weighted equivalent (Guenther et al. 2012 Table 6: 99.6, 43.7
        and 20.7 Tg yr-1): harmonic mass-weighted molar mass, weighted carbon fraction,
        n_C = MW f_C / A_r(C) and the weighted Table 4 LDF, with no conflicting
        'other VOCs' wording."""
        parameters = self._database()["parameters"]
        molar_mass = parameters["biosphere.bvoc.molecular_weights.ovocs"]
        carbon_fraction = parameters["biosphere.bvoc.carbon_fraction.ovocs"]
        carbon_atoms = parameters["biosphere.bvoc.carbon_atoms.ovocs"]
        assert molar_mass["value"] == 37.87
        assert carbon_fraction["value"] == 0.4618
        assert carbon_atoms["value"] == 1.456
        assert carbon_fraction["value"] * molar_mass["value"] == pytest.approx(
            carbon_atoms["value"] * 12.011, rel=1.0e-3
        )
        for spec in (molar_mass, carbon_fraction, carbon_atoms):
            text = " | ".join(
                [str(spec["description"]), *map(str, spec["references"]),
                 str(spec["provenance"]["literature_check"])]
            )
            for fragment in ("Table 6", "99.6", "43.7", "20.7", "emission-weighted"):
                assert fragment in text, fragment
            assert spec["provenance"]["literature_check"].startswith("verified")
        for key in (
            "biosphere.bvoc.molecular_weights.ovocs",
            "biosphere.bvoc.carbon_fraction.ovocs",
            "biosphere.bvoc.carbon_atoms.ovocs",
            "biosphere.bvoc.default.ovocs",
        ):
            assert "other voc" not in str(parameters[key]["description"]).lower(), key

    def test_co2_inhibition_bounds_are_retired_with_the_clamp(self):
        """The isoprene CO2-inhibition floor and ceiling of validation_numerical.yaml
        were read only by the retired IsopreneSynthesis clamp and are removed."""
        from pathlib import Path

        import yaml

        path = (
            Path(__file__).resolve().parents[4]
            / "relic"
            / "parameters"
            / "database"
            / "relic"
            / "validation_numerical.yaml"
        )
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        assert [
            key
            for key in data["parameters"]
            if key.startswith("validation.numerical.co2_inhibition_")
        ] == []


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])


