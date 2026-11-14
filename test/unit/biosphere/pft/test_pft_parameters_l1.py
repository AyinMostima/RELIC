"""
L1 Unit Tests for PFT Parameter System Integration

Tests individual PFT parameter loading, validation, and unit conversions.
Ensures parameters are correctly loaded from the unified parameter system
and that all units conform to CF-1.8 standards.
"""

import pytest
import inspect

from relic.parameters import get_global_manager
from relic.scientific.biosphere.pft import pft_types as pft_types_module
from relic.scientific.biosphere.pft import pft_parameter_base as pft_base_module
from relic.scientific.biosphere.pft.pft_parameter_base import PFTParameterBase
from relic.scientific.biosphere.pft.pft_types import (
    AllocationParameters,
    BVOCParameters,
    PFTCategory,
    PFTDefinition,
    PhenologyParameters,
    PhotosynthesisParameters,
    PhotosynthesisPathway,
    RespirationParameters,
    StressParameters,
)


class TestPFTParameterLoading:
    """Test loading of PFT parameters from the unified parameter system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        # Ensure parameter database is loaded
        self.param_manager.load_from_database()

    def test_parameter_groups_and_definition_require_explicit_manager(self):
        """Keep manager ownership mandatory across the PFT dataclass graph."""
        classes = (
            PFTParameterBase,
            PhotosynthesisParameters,
            PhenologyParameters,
            AllocationParameters,
            RespirationParameters,
            BVOCParameters,
            StressParameters,
            PFTDefinition,
        )

        for parameter_class in classes:
            parameter = inspect.signature(parameter_class).parameters[
                "parameter_manager"
            ]
            assert parameter.default is inspect.Parameter.empty

        with pytest.raises(TypeError):
            PhotosynthesisParameters()
        with pytest.raises(TypeError):
            PFTDefinition(name="missing_manager")

    def test_pft_parameter_sources_have_no_global_manager_bypass(self):
        """Forbid global lookup in the base and its six parameter groups."""
        assert "get_global_manager" not in inspect.getsource(pft_base_module)
        assert "get_global_manager" not in inspect.getsource(pft_types_module)

    def test_pft_base_delegates_to_injected_manager(self):
        """Forward full parameter keys to one explicitly injected manager."""

        class RecordingManager:
            def __init__(self):
                self.calls = []

            def get(self, key, **kwargs):
                self.calls.append((key, kwargs))
                return 17.0

        manager = RecordingManager()
        params = PFTParameterBase(
            parameter_manager=manager,
            _pft_namespace="biosphere.pft.test",
        )
        value = params.get_parameter_from_manager(
            "biosphere.pft.test.photosynthesis.vcmax25",
            required=True,
        )

        assert value == 17.0
        assert manager.calls == [
            (
                "biosphere.pft.test.photosynthesis.vcmax25",
                {
                    "required": True,
                    "expected_type": None,
                },
            )
        ]

    def test_pft_definition_propagates_one_manager_to_all_parameter_groups(self):
        """Construct every group with the definition's single manager owner."""
        pft = PFTDefinition(
            name="broadleaf_deciduous_temperate",
            parameter_manager=self.param_manager,
        )

        for group_name in (
            "photosynthesis",
            "phenology",
            "allocation",
            "respiration",
            "bvoc",
            "stress",
        ):
            assert getattr(pft, group_name)._parameter_manager is self.param_manager

    def test_pft_definition_serialization_excludes_and_restores_manager(self):
        """Keep manager ownership outside serialized PFT business data."""
        pft = PFTDefinition(
            name="broadleaf_deciduous_temperate",
            parameter_manager=self.param_manager,
        )

        payload = pft.to_dict()
        assert "parameter_manager" not in payload
        assert "_parameter_manager" not in payload

        restored = PFTDefinition.from_dict(
            payload,
            parameter_manager=self.param_manager,
        )
        assert restored._parameter_manager is self.param_manager
        for group_name in (
            "photosynthesis",
            "phenology",
            "allocation",
            "respiration",
            "bvoc",
            "stress",
        ):
            assert (
                getattr(restored, group_name)._parameter_manager
                is self.param_manager
            )

    def test_base_has_no_retired_file_loading_hook(self):
        """PFT parameters must use the live ParameterManager path only."""
        assert not hasattr(PFTParameterBase, "_load_pft_file")

    def test_retired_parameter_alias_helpers_are_absent(self):
        """PFT loading must not retain compatibility alias machinery."""
        retired_symbols = (
            "_get_alias_value",
            "_get_growth_default_temperature",
            "_PHENOLOGY_ALIAS_MAP",
            "_ALLOCATION_ALIAS_MAP",
            "_RESPIRATION_ALIAS_MAP",
            "_BVOC_ALIAS_MAP",
            "_REPORTED_ALIAS_USAGE",
        )

        for symbol in retired_symbols:
            assert not hasattr(pft_types_module, symbol)

    def test_photosynthesis_uses_only_declared_shared_process_binding(self, monkeypatch):
        """Only rd_q10 may bind from the shared photosynthesis namespace."""
        shared_values = {
            "biosphere.photosynthesis.rd_q10": 2.2,
            "biosphere.photosynthesis.g0": 0.99,
        }

        class SharedManager:
            def get(self, key, **kwargs):
                return shared_values.get(key)

        shared_manager = SharedManager()
        monkeypatch.setattr(
            PhotosynthesisParameters,
            "get_parameter_from_manager",
            lambda self, *args, **kwargs: None,
        )

        params = PhotosynthesisParameters(
            vcmax25=50.0,
            jmax25=100.0,
            rd25=1.0,
            pathway=PhotosynthesisPathway.C3,
            _pft_namespace="biosphere.pft.test",
            parameter_manager=shared_manager,
        )

        assert params.rd_q10 == 2.2
        assert params.g0 is None

    def test_respiration_uses_only_declared_shared_process_bindings(self, monkeypatch):
        """Only the three process-level respiration parameters may be shared."""
        shared_values = {
            "biosphere.respiration.maintenance.q10_respiration": 2.0,
            "biosphere.respiration.growth.np_response_coeff": 0.5,
            "biosphere.respiration.maintenance.min_respiration_rate": 0.01,
            "biosphere.respiration.maintenance.rm_stem_25": 9.99,
        }

        class SharedManager:
            def get(self, key, **kwargs):
                return shared_values.get(key)

        shared_manager = SharedManager()
        monkeypatch.setattr(
            RespirationParameters,
            "get_parameter_from_manager",
            lambda self, *args, **kwargs: None,
        )

        params = RespirationParameters(
            _pft_namespace="biosphere.pft.test",
            parameter_manager=shared_manager,
        )

        assert params.q10_maint_resp == 2.0
        assert params.n_effect_maint_resp == 0.5
        assert params.base_resp_rate == 0.01
        assert params.maint_resp_stem is None

    def test_stress_parameters_do_not_invent_temperature_bounds(self, monkeypatch):
        """Missing PFT stress thresholds must remain missing and fail downstream validation."""
        class SharedManager:
            def get(self, key, **kwargs):
                if key == "biosphere.growth.defaults.growth_default_temperature":
                    return 290.0
                return None

        shared_manager = SharedManager()
        monkeypatch.setattr(
            StressParameters,
            "get_parameter_from_manager",
            lambda self, *args, **kwargs: None,
        )

        params = StressParameters(
            _pft_namespace="biosphere.pft.test",
            parameter_manager=shared_manager,
        )

        assert params.temp_stress_min is None
        assert params.temp_stress_max is None

    def test_photosynthesis_parameters_loading(self):
        """Test loading photosynthesis parameters for a specific PFT."""
        # Test for broadleaf_deciduous_temperate
        namespace = "biosphere.pft.broadleaf_deciduous_temperate"
        photo_params = PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)

        # Check key parameters are loaded
        assert photo_params.vcmax25 is not None, "Vcmax25 should be loaded"
        assert photo_params.jmax25 is not None, "Jmax25 should be loaded"
        assert photo_params.rd25 is not None, "Rd25 should be loaded"

        # Check parameter ranges
        assert 20 <= photo_params.vcmax25 <= 150, f"Vcmax25 out of range: {photo_params.vcmax25}"
        assert 30 <= photo_params.jmax25 <= 300, f"Jmax25 out of range: {photo_params.jmax25}"
        assert 0 <= photo_params.rd25 <= 10, f"Rd25 out of range: {photo_params.rd25}"

        # Check Jmax/Vcmax ratio
        ratio = photo_params.jmax25 / photo_params.vcmax25
        assert 1.2 <= ratio <= 2.5, f"Jmax/Vcmax ratio out of typical range: {ratio}"

        # Check pathway is set correctly
        assert photo_params.pathway == PhotosynthesisPathway.C3, "Should be C3 pathway"

    def test_temperature_parameters_units(self):
        """Test that all temperature parameters are in Kelvin."""
        namespace = "biosphere.pft.broadleaf_deciduous_temperate"
        phenology_params = PhenologyParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)

        # Check temperature parameters are in reasonable Kelvin ranges
        assert 240 <= phenology_params.temp_cold_kill <= 280, f"temp_cold_kill not in Kelvin: {phenology_params.temp_cold_kill}"
        assert 260 <= phenology_params.temp_growth_min <= 290, f"temp_growth_min not in Kelvin: {phenology_params.temp_growth_min}"
        assert 285 <= phenology_params.temp_growth_opt <= 310, f"temp_growth_opt not in Kelvin: {phenology_params.temp_growth_opt}"
        assert 300 <= phenology_params.temp_growth_max <= 330, f"temp_growth_max not in Kelvin: {phenology_params.temp_growth_max}"

        # Check GDD base temperature
        assert 268 <= phenology_params.gdd_min <= 278, f"gdd_min not in Kelvin: {phenology_params.gdd_min}"

    def test_allocation_fractions_sum(self):
        """Test that allocation fractions sum to 1.0."""
        namespace = "biosphere.pft.broadleaf_deciduous_temperate"
        alloc_params = AllocationParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)

        # Check individual fractions
        assert 0 <= alloc_params.f_leaf <= 1, f"f_leaf out of range: {alloc_params.f_leaf}"
        assert 0 <= alloc_params.f_stem <= 1, f"f_stem out of range: {alloc_params.f_stem}"
        assert 0 <= alloc_params.f_root <= 1, f"f_root out of range: {alloc_params.f_root}"

        # Check sum equals 1.0 (within tolerance)
        total = alloc_params.f_leaf + alloc_params.f_stem + alloc_params.f_root
        assert abs(total - 1.0) < 0.01, f"Allocation fractions don't sum to 1.0: {total}"

    def test_bvoc_units_cf_compliance(self):
        """Test that BVOC emission factors use CF-1.8 compliant units."""
        namespace = "biosphere.pft.broadleaf_deciduous_temperate"
        bvoc_params = BVOCParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)

        # Check emission factors are loaded
        assert bvoc_params.ef_isoprene is not None, "Isoprene emission factor should be loaded"
        assert bvoc_params.ef_monoterpene is not None, "Monoterpene emission factor should be loaded"
        assert bvoc_params.ef_sesquiterpene is not None, "Sesquiterpene emission factor should be loaded"

        # Check values are in reasonable ranges (ug g-1 h-1)
        assert 0 <= bvoc_params.ef_isoprene <= 100, f"ef_isoprene out of range: {bvoc_params.ef_isoprene}"
        assert 0 <= bvoc_params.ef_monoterpene <= 50, f"ef_monoterpene out of range: {bvoc_params.ef_monoterpene}"
        assert 0 <= bvoc_params.ef_sesquiterpene <= 10, f"ef_sesquiterpene out of range: {bvoc_params.ef_sesquiterpene}"

    def test_stress_parameters_completeness(self):
        """Test that all stress parameters are present and loaded."""
        namespace = "biosphere.pft.broadleaf_deciduous_temperate"
        stress_params = StressParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)

        # Check all required stress parameters are loaded
        required_params = [
            'water_stress_threshold', 'water_stress_slope', 'osmotic_adjustment',
            'temp_stress_min', 'temp_stress_max', 'heat_shock_threshold',
            'frost_tolerance', 'nitrogen_stress_half_sat', 'phosphorus_stress_half_sat',
            'ozone_uptake_rate', 'ozone_detox_capacity', 'light_stress_threshold',
            'temp_stress_threshold'
        ]

        for param_name in required_params:
            value = getattr(stress_params, param_name)
            assert value is not None, f"Stress parameter {param_name} not loaded"

        # Check temperature stress parameters are in Kelvin
        assert 250 <= stress_params.temp_stress_min <= 290, f"temp_stress_min not in Kelvin: {stress_params.temp_stress_min}"
        assert 300 <= stress_params.temp_stress_max <= 340, f"temp_stress_max not in Kelvin: {stress_params.temp_stress_max}"
        assert 250 <= stress_params.frost_tolerance <= 280, f"frost_tolerance not in Kelvin: {stress_params.frost_tolerance}"
        assert 300 <= stress_params.heat_shock_threshold <= 330, f"heat_shock_threshold not in Kelvin: {stress_params.heat_shock_threshold}"

    def test_all_pft_types_available(self):
        """Test that all 15 PFT types are available in the parameter database."""
        expected_pfts = [
            'broadleaf_deciduous_temperate',
            'broadleaf_evergreen_tropical',
            'broadleaf_evergreen_temperate',
            'broadleaf_deciduous_tropical',
            'broadleaf_deciduous_boreal',
            'needleleaf_evergreen_temperate',
            'needleleaf_evergreen_boreal',
            'needleleaf_deciduous_boreal',
            'shrub_deciduous_temperate',
            'shrub_evergreen_temperate',
            'c3_grass_temperate',
            'c3_grass_arctic',
            'c4_grass_temperate',
            'crop_temperate',
            'crop_tropical'
        ]

        for pft_name in expected_pfts:
            namespace = f"biosphere.pft.{pft_name}"

            # Test that we can load photosynthesis parameters for each PFT
            try:
                photo_params = PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
                assert photo_params.vcmax25 is not None, f"Vcmax25 not loaded for {pft_name}"
            except Exception as e:
                pytest.fail(f"Failed to load parameters for PFT {pft_name}: {e}")

    def test_pft_definition_complete_loading(self):
        """Test loading a complete PFT definition with all parameter groups."""
        pft_name = "broadleaf_deciduous_temperate"
        pft_def = PFTDefinition(
            name=pft_name,
            category=PFTCategory.BROADLEAF_DECIDUOUS_TEMPERATE,
            description=f"Test PFT: {pft_name}",
            _use_param_manager=True,
            _pft_namespace=f"biosphere.pft.{pft_name}",
            parameter_manager=self.param_manager,
        )

        # Check all parameter groups are loaded
        assert pft_def.photosynthesis is not None, "Photosynthesis parameters not loaded"
        assert pft_def.phenology is not None, "Phenology parameters not loaded"
        assert pft_def.allocation is not None, "Allocation parameters not loaded"
        assert pft_def.respiration is not None, "Respiration parameters not loaded"
        assert pft_def.bvoc is not None, "BVOC parameters not loaded"
        assert pft_def.stress is not None, "Stress parameters not loaded"

        # Check a few key values
        assert pft_def.photosynthesis.vcmax25 > 0, "Vcmax25 should be positive"
        assert pft_def.allocation.f_leaf > 0, "Leaf allocation should be positive"
        assert pft_def.phenology.leaf_longevity > 0, "Leaf longevity should be positive"


class TestPFTParameterValidation:
    """Test parameter validation and constraint checking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Provide the explicit parameter owner for every validation case."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()

    def test_photosynthesis_temperature_response_validation(self):
        """Test validation of temperature response parameters."""
        namespace = "biosphere.pft.broadleaf_deciduous_temperate"
        photo_params = PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)

        # Activation energies should be positive
        assert photo_params.vcmax_ea > 0, "Vcmax activation energy should be positive"
        assert photo_params.jmax_ea > 0, "Jmax activation energy should be positive"
        assert photo_params.rd_ea > 0, "Rd activation energy should be positive"

        # Q10 should be in reasonable range
        assert 1.5 <= photo_params.rd_q10 <= 3.0, f"Q10 out of range: {photo_params.rd_q10}"

    def test_phenology_gdd_parameters(self):
        """Test GDD-related phenology parameters."""
        namespace = "biosphere.pft.broadleaf_deciduous_temperate"
        phenology_params = PhenologyParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)

        # GDD thresholds should be ordered correctly
        assert phenology_params.gdd_leaf_on < phenology_params.gdd_leaf_off, \
            "GDD leaf onset should be less than leaf offset"

        # GDD min (base temperature) should be reasonable
        assert 268 <= phenology_params.gdd_min <= 283, \
            f"GDD base temperature out of range: {phenology_params.gdd_min} K"

    def test_respiration_parameters_consistency(self):
        """Test consistency of respiration parameters."""
        namespace = "biosphere.pft.broadleaf_deciduous_temperate"
        resp_params = RespirationParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)

        # Maintenance respiration coefficients should be positive
        assert resp_params.maint_resp_leaf > 0, "Leaf maintenance respiration should be positive"
        assert resp_params.maint_resp_stem > 0, "Stem maintenance respiration should be positive"
        assert resp_params.maint_resp_root > 0, "Root maintenance respiration should be positive"

        # Q10 values should be in reasonable range
        assert 1.5 <= resp_params.q10_maint_resp <= 3.0, f"Q10 maint resp out of range: {resp_params.q10_maint_resp}"
        if resp_params.q10_growth_resp is not None:
            assert 1.5 <= resp_params.q10_growth_resp <= 3.0, f"Q10 growth resp out of range: {resp_params.q10_growth_resp}"


class TestPFTCrossComparison:
    """Test parameter relationships across different PFT types."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Provide the explicit parameter owner for every comparison case."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()

    def test_tropical_vs_boreal_temperature_ranges(self):
        """Test that tropical PFTs have higher temperature thresholds than boreal."""
        # Load tropical PFT
        tropical_namespace = "biosphere.pft.broadleaf_evergreen_tropical"
        tropical_stress = StressParameters(parameter_manager=self.param_manager, _pft_namespace=tropical_namespace)

        # Load boreal PFT
        boreal_namespace = "biosphere.pft.needleleaf_evergreen_boreal"
        boreal_stress = StressParameters(parameter_manager=self.param_manager, _pft_namespace=boreal_namespace)

        # Tropical should have higher temperature thresholds
        assert tropical_stress.temp_stress_min > boreal_stress.temp_stress_min, \
            "Tropical min temp should be higher than boreal"
        assert tropical_stress.heat_shock_threshold > boreal_stress.heat_shock_threshold, \
            "Tropical heat shock threshold should be higher"

        # Boreal should have better frost tolerance (lower temperature)
        assert boreal_stress.frost_tolerance < tropical_stress.frost_tolerance, \
            "Boreal should have better frost tolerance (lower temp threshold)"

    def test_evergreen_vs_deciduous_turnover(self):
        """Test that evergreen PFTs have lower leaf turnover than deciduous."""
        # Load evergreen PFT
        evergreen_namespace = "biosphere.pft.needleleaf_evergreen_temperate"
        evergreen_alloc = AllocationParameters(parameter_manager=self.param_manager, _pft_namespace=evergreen_namespace)

        # Load deciduous PFT
        deciduous_namespace = "biosphere.pft.broadleaf_deciduous_temperate"
        deciduous_alloc = AllocationParameters(parameter_manager=self.param_manager, _pft_namespace=deciduous_namespace)

        # Note: Turnover rates might be in phenology parameters
        evergreen_phenology = PhenologyParameters(parameter_manager=self.param_manager, _pft_namespace=evergreen_namespace)
        deciduous_phenology = PhenologyParameters(parameter_manager=self.param_manager, _pft_namespace=deciduous_namespace)

        # Check if leaf longevity is available
        if evergreen_phenology.leaf_longevity is not None and deciduous_phenology.leaf_longevity is not None:
            # Evergreen should have longer leaf longevity
            assert evergreen_phenology.leaf_longevity > deciduous_phenology.leaf_longevity, \
                "Evergreen should have longer leaf longevity"

    def test_c3_vs_c4_photosynthesis(self):
        """Test that C3 and C4 grasses have appropriate photosynthesis parameters."""
        # Load C3 grass
        c3_namespace = "biosphere.pft.c3_grass_temperate"
        c3_photo = PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace=c3_namespace)

        # Load C4 grass
        c4_namespace = "biosphere.pft.c4_grass_temperate"
        c4_photo = PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace=c4_namespace)

        # Check pathways
        assert c3_photo.pathway == PhotosynthesisPathway.C3, "C3 grass should have C3 pathway"
        assert c4_photo.pathway == PhotosynthesisPathway.C4, "C4 grass should have C4 pathway"

        # C4 plants typically have lower Vcmax but can still have high productivity
        # This is a simplified check - actual C4 biochemistry is more complex
        assert c4_photo.vcmax25 > 0, "C4 should have positive Vcmax25"
        assert c3_photo.vcmax25 > 0, "C3 should have positive Vcmax25"

    def test_bvoc_emissions_by_pft_type(self):
        """Test that BVOC emissions vary appropriately by PFT type."""
        # Broadleaf tropical should have high isoprene emissions
        tropical_namespace = "biosphere.pft.broadleaf_evergreen_tropical"
        tropical_bvoc = BVOCParameters(parameter_manager=self.param_manager, _pft_namespace=tropical_namespace)

        # Needleleaf should have high monoterpene emissions
        needleleaf_namespace = "biosphere.pft.needleleaf_evergreen_temperate"
        needleleaf_bvoc = BVOCParameters(parameter_manager=self.param_manager, _pft_namespace=needleleaf_namespace)

        # Check emission patterns (these are general tendencies, not strict rules)
        assert tropical_bvoc.ef_isoprene > 0, "Tropical should emit isoprene"
        assert needleleaf_bvoc.ef_monoterpene > 0, "Needleleaf should emit monoterpenes"


class TestPFTEmissionFactorDescriptions:
    """BVOC emission-factor descriptions of each PFT file name that PFT."""

    def test_emission_factor_descriptions_name_their_own_pft(self):
        """Each ef_* description carries the human-readable descriptor of its own PFT
        (biosphere/pft/metadata.yaml), names no other PFT, and drops the copied
        '(moderate for deciduous trees)' and 'other VOCs' wording."""
        from pathlib import Path

        import yaml

        root = (
            Path(__file__).resolve().parents[4]
            / "relic"
            / "parameters"
            / "database"
            / "relic"
            / "biosphere"
            / "pft"
        )
        metadata = yaml.safe_load((root / "metadata.yaml").read_text(encoding="utf-8-sig"))
        descriptors = {
            key.split(".")[2]: str(spec["value"])
            for key, spec in metadata["parameters"].items()
            if key.endswith(".description") and key.split(".")[2] != "default"
        }
        problems = []
        for pft, descriptor in descriptors.items():
            parameters = yaml.safe_load(
                (root / f"{pft}.yaml").read_text(encoding="utf-8-sig")
            )["parameters"]
            for name in ("ef_isoprene", "ef_monoterpene", "ef_sesquiterpene", "ef_other_voc"):
                description = str(parameters[f"biosphere.pft.{pft}.bvoc.{name}"]["description"])
                lowered = description.lower()
                if descriptor.lower() not in lowered:
                    problems.append((pft, name, "missing its own PFT"))
                for other, other_descriptor in descriptors.items():
                    if other != pft and other_descriptor.lower() in lowered:
                        problems.append((pft, name, f"names {other}"))
                if "moderate for deciduous trees" in lowered or "other voc" in lowered:
                    problems.append((pft, name, "copied wording"))
        assert problems == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

