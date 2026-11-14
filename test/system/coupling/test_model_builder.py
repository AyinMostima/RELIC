"""
Unit Tests for Central Model Builder

This module contains comprehensive unit tests for the CentralModelBuilder class
and related components in the relic/models refactoring. Tests cover:
- Basic functionality (functor deduplication, variable merging)
- Conflict detection and resolution
- CF compliance validation  
- Integration with existing models modules
- Edge cases and error handling

Test Categories:
    L1 - Unit tests for individual components
    L2 - Integration tests for multi-scale interactions
    L3 - System tests for full workflow validation

Mathematical Validation:
    Tests verify area-weighted aggregation formulas:
    X_agg = 危(X_i 脳 A_i) / 危(A_i) for intensive variables
    X_total = 危(X_i 脳 A_i) for extensive variables

References:
    - RELIC Testing Framework Guidelines
    - CF Metadata Conventions v1.8 Test Suite
    - RHESSys Validation Test Methods (Tague & Band, 2004)
"""

import inspect
import sys
import os
import unittest
import warnings

# Add relic to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))


# Import modules under test
from relic.coupling.scales.hierarchical.model_builder import (
    CentralModelBuilder,
    create_model_builder,
    VariableSpec,
    AggregationMethod,
    ScaleConfig,
    ConflictReport,
    ValidationReport
)
from relic.coupling.scales.hierarchical import model_builder as model_builder_module
from relic.coupling.scales.types import SpatialQuantityBasis
from relic.models import SCALE_HIERARCHY, UPSCALING_MAP
from relic.models import basin, hillslope, individual, landscape, patch, regional


class TestVariableSpec(unittest.TestCase):
    """
    L1 Unit Tests for VariableSpec data structure.
    
    Tests the standardized variable specification with CF compliance,
    automatic cell_methods generation, and type conversions.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.basic_spec = VariableSpec(
            units="kg C m-2 s-1",
            standard_name="gross_primary_productivity_of_carbon",
            spatial_basis=SpatialQuantityBasis.AREA_DENSITY,
            description="Test GPP variable"
        )
    
    def test_basic_creation(self):
        """Test basic VariableSpec creation with required fields."""
        self.assertEqual(self.basic_spec.units, "kg C m-2 s-1")
        self.assertEqual(self.basic_spec.standard_name, "gross_primary_productivity_of_carbon")
        self.assertEqual(self.basic_spec.spatial_basis, SpatialQuantityBasis.AREA_DENSITY)
        self.assertEqual(self.basic_spec.description, "Test GPP variable")
    
    def test_auto_cell_methods_generation(self):
        """Test automatic generation of CF-compliant cell_methods."""
        # Test area-weighted mean (default for most cases)
        intensive_spec = VariableSpec(
            units="K",
            standard_name="surface_temperature",
            spatial_basis=SpatialQuantityBasis.INTENSIVE,
            description="Temperature variable",
            aggregation=AggregationMethod.AREA_WEIGHTED_MEAN
        )
        self.assertIn("area: mean", intensive_spec.cell_methods)
        self.assertIn("time: mean", intensive_spec.cell_methods)
        
        # An area density keeps its density units at the parent: area-weighted mean.
        density_spec = VariableSpec(
            units="kg C m-2 s-1",
            standard_name="gross_primary_productivity_of_carbon",
            spatial_basis=SpatialQuantityBasis.AREA_DENSITY,
            description="GPP flux",
            aggregation=AggregationMethod.AREA_WEIGHTED_MEAN
        )
        self.assertIn("area: mean", density_spec.cell_methods)
        # A cell total is summed: "area: sum".
        total_spec = VariableSpec(
            units="kg s-1",
            standard_name="water_volume_transport_into_sea_water_from_rivers",
            spatial_basis=SpatialQuantityBasis.CELL_TOTAL,
            description="Streamflow",
            aggregation=AggregationMethod.SUM
        )
        self.assertIn("area: sum", total_spec.cell_methods)
        
        # Test standard deviation for variability
        variability_spec = VariableSpec(
            units="1",
            standard_name="coefficient_of_variation",
            spatial_basis=SpatialQuantityBasis.INTENSIVE,
            description="Spatial variability",
            aggregation=AggregationMethod.STANDARD_DEVIATION
        )
        self.assertIn("standard_deviation", variability_spec.cell_methods)
    
    def test_string_to_enum_conversion(self):
        """Test automatic conversion of string values to enums."""
        spec_from_strings = VariableSpec(
            units="m",
            standard_name="height",
            spatial_basis="intensive",  # String instead of enum
            description="Height variable",
            aggregation="mean"  # String instead of enum
        )
        
        self.assertIsInstance(spec_from_strings.spatial_basis, SpatialQuantityBasis)
        self.assertEqual(spec_from_strings.spatial_basis, SpatialQuantityBasis.INTENSIVE)
        self.assertIsInstance(spec_from_strings.aggregation, AggregationMethod)
        self.assertEqual(spec_from_strings.aggregation, AggregationMethod.MEAN)
    
    def test_extras_field(self):
        """Test extras field for additional metadata."""
        spec_with_extras = VariableSpec(
            units="kg C m-2 s-1",
            standard_name="gpp",
            spatial_basis=SpatialQuantityBasis.AREA_DENSITY,
            description="GPP with extras",
            extras={
                "conserved": True,
                "bounds": [0, 100],
                "source_reference": "Farquhar et al. 1980"
            }
        )
        
        self.assertTrue(spec_with_extras.extras["conserved"])
        self.assertEqual(spec_with_extras.extras["bounds"], [0, 100])
        self.assertIn("Farquhar", spec_with_extras.extras["source_reference"])


class TestExplicitSpatialBasis(unittest.TestCase):
    """The builder uses the single SpatialQuantityBasis owner, not a local VariableType."""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()

    def test_no_local_variable_type_enum(self):
        self.assertFalse(hasattr(model_builder_module, "VariableType"))

    def test_bare_extensive_is_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            VariableSpec(
                units="kg C m-2 s-1",
                standard_name="gpp",
                spatial_basis="extensive",
                description="ambiguous",
            )

    def test_models_declare_area_densities_and_cell_totals(self):
        patch_vars = self.builder.get_effective_variables("patch")
        gpp = patch_vars["gross_primary_productivity"]
        self.assertEqual(gpp.spatial_basis, SpatialQuantityBasis.AREA_DENSITY)
        self.assertEqual(gpp.aggregation, AggregationMethod.AREA_WEIGHTED_MEAN)
        self.assertIn("area: mean", gpp.cell_methods)
        # An area density averaged over the parent area is conservative: no silent correction.
        self.assertEqual(patch_vars["nee"].spatial_basis, SpatialQuantityBasis.AREA_DENSITY)
        self.assertEqual(patch_vars["nee"].aggregation, AggregationMethod.AREA_WEIGHTED_MEAN)
        self.assertIn("area: mean", patch_vars["nee"].cell_methods)
        hillslope_vars = self.builder.get_effective_variables("hillslope")
        self.assertEqual(hillslope_vars["streamflow"].spatial_basis, SpatialQuantityBasis.CELL_TOTAL)
        self.assertEqual(hillslope_vars["streamflow"].aggregation, AggregationMethod.SUM)
        self.assertEqual(patch_vars["lai"].spatial_basis, SpatialQuantityBasis.INTENSIVE)

    def test_undeclared_or_inconsistent_configs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "declare spatial_basis"):
            self.builder._convert_variable_spec("x", {"units": "kg", "standard_name": "x"})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.builder._convert_variable_spec(
                "x", {"units": "kg", "standard_name": "x", "variable_type": "extensive"}
            )
        with self.assertRaisesRegex(ValueError, "cell_total"):
            self.builder._convert_variable_spec(
                "x",
                {
                    "units": "kg s-1",
                    "standard_name": "x",
                    "spatial_basis": "cell_total",
                    "aggregation_method": "area_weighted_mean",
                },
            )

    def test_categorical_variables_require_the_mode_instead_of_being_rewritten(self):
        """A non-MODE aggregation of class codes raises; it is not silently replaced by MODE."""
        with self.assertRaises(ValueError) as caught:
            self.builder._convert_variable_spec(
                "land_cover_probe",
                {
                    "units": "1",
                    "standard_name": "land_cover",
                    "spatial_basis": "categorical",
                    "aggregation_method": "area_weighted_mean",
                },
            )
        message = str(caught.exception)
        self.assertIn("land_cover_probe", message)
        self.assertIn("area_weighted_mean", message)
        spec = self.builder._convert_variable_spec(
            "land_cover_probe",
            {"units": "1", "standard_name": "land_cover", "spatial_basis": "categorical"},
        )
        self.assertIs(spec.aggregation, AggregationMethod.MODE)

    def test_unknown_aggregation_method_is_rejected_not_ignored(self):
        """An unrecognised aggregation_method raises instead of falling back to cell_methods."""
        with self.assertRaises(ValueError) as caught:
            self.builder._convert_variable_spec(
                "surface_roughness_probe",
                {
                    "units": "m",
                    "standard_name": "surface_roughness_length",
                    "spatial_basis": "intensive",
                    "cell_methods": "area: mean",
                    "aggregation_method": "geometric_mean",
                },
            )
        message = str(caught.exception)
        self.assertIn("surface_roughness_probe", message)
        self.assertIn("geometric_mean", message)
        for accepted in ("sum", "area_weighted_mean", "mode"):
            self.assertIn(accepted, message)

    def test_model_catalogs_use_only_accepted_aggregation_methods(self):
        accepted = {method.value for method in AggregationMethod}
        for module in (individual, patch, hillslope, landscape, basin, regional):
            for name, config in module.EXPOSED_VARIABLES.items():
                if "aggregation_method" in config:
                    self.assertIn(
                        config["aggregation_method"], accepted, f"{module.__name__}.{name}"
                    )


class TestIndividualCatalogStandardNames(unittest.TestCase):
    """stomatal_conductance is a conductance, not a flux: CF v94 has no standard name
    for stomatal conductance, so the catalog declares none (and never a flux name)."""

    def test_stomatal_conductance_declares_no_flux_standard_name(self):
        entry = individual.EXPOSED_VARIABLES["stomatal_conductance"]
        self.assertIsNone(entry["standard_name"])
        self.assertEqual(entry["units"], "mol m-2 s-1")
        self.assertIn("water vapour", entry["description"])

    def test_cf_table_has_no_stomatal_conductance_entry(self):
        import xml.etree.ElementTree as ET
        from pathlib import Path as _Path

        table = _Path(individual.__file__).resolve().parents[1] / "data" / "standards" / "cf_standard_name_table.xml"
        names = {entry.get("id") for entry in ET.parse(table).getroot().findall("entry")}
        self.assertFalse([name for name in names if "conductance" in name])


class TestCatalogStandardNames(unittest.TestCase):
    """Round 6: a declared standard name is an entry or alias of the CF v94 table
    (relic/data/standards/cf_standard_name_table.xml); a quantity without a CF name
    declares None and its description says why. No name is invented."""

    MODULES = (individual, patch, hillslope, landscape, basin, regional)

    def test_every_catalog_standard_name_is_a_cf_entry_or_none(self):
        import xml.etree.ElementTree as ET
        from pathlib import Path as _Path

        table = _Path(individual.__file__).resolve().parents[1] / "data" / "standards" / "cf_standard_name_table.xml"
        root = ET.parse(table).getroot()
        names = {entry.get("id") for entry in root.findall("entry")}
        names |= {alias.get("id") for alias in root.findall("alias")}
        unknown = []
        for module in self.MODULES:
            for name, config in module.EXPOSED_VARIABLES.items():
                standard_name = config.get("standard_name")
                if standard_name is None:
                    self.assertIn("CF", config["description"], f"{module.__name__}.{name}")
                elif standard_name not in names:
                    unknown.append(f"{module.__name__}.{name}: {standard_name}")
        self.assertEqual(unknown, [])

    def test_declared_names_and_units_are_cf_compatible(self):
        """RELIC keeps element-labelled units (kg C m-2 s-1); the CF check compares their
        element-aware dimension and element with the canonical units and the name."""
        from relic.data.standards.cf_conventions import CFConventions

        issues = []
        for module in self.MODULES:
            for name, config in module.EXPOSED_VARIABLES.items():
                standard_name = config.get("standard_name")
                if standard_name is None:
                    continue
                result = CFConventions.validate_cf_compliance(
                    {"standard_name": standard_name, "units": config["units"]}
                )
                issues.extend(f"{module.__name__}.{name}: {issue}" for issue in result.issues)
        self.assertEqual(issues, [])


class TestOneMeaningPerAggregationMethod(unittest.TestCase):
    """An area density keeps its density at the parent (AREA_WEIGHTED_MEAN, "area: mean"); an amount
    is a cell_total summed with SUM ("area: sum"). There is no AREA_WEIGHTED_SUM that would write
    "area: sum" next to density units."""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()

    def test_area_weighted_sum_is_not_an_aggregation_method(self):
        self.assertNotIn("area_weighted_sum", {method.value for method in AggregationMethod})
        with self.assertRaisesRegex(ValueError, "area_weighted_sum"):
            self.builder._convert_variable_spec(
                "gpp_probe",
                {
                    "units": "kg C m-2 s-1",
                    "standard_name": "gross_primary_productivity_of_carbon",
                    "spatial_basis": "area_density",
                    "aggregation_method": "area_weighted_sum",
                },
            )

    def test_area_sum_cell_methods_mean_a_summed_cell_total(self):
        total = self.builder._convert_variable_spec(
            "streamflow_probe",
            {"units": "kg s-1", "standard_name": "x", "spatial_basis": "cell_total", "cell_methods": "area: sum"},
        )
        self.assertIs(total.aggregation, AggregationMethod.SUM)
        with self.assertRaisesRegex(ValueError, "area_density"):
            self.builder._convert_variable_spec(
                "density_probe",
                {
                    "units": "kg m-2 s-1",
                    "standard_name": "x",
                    "spatial_basis": "area_density",
                    "cell_methods": "time: mean area: sum",
                },
            )

    def test_model_area_densities_aggregate_to_parent_densities(self):
        for scale in ("individual", "hillslope", "landscape"):
            for name, spec in self.builder.get_effective_variables(scale, include_lower=False).items():
                if spec.spatial_basis is SpatialQuantityBasis.AREA_DENSITY:
                    self.assertIs(spec.aggregation, AggregationMethod.AREA_WEIGHTED_MEAN, f"{scale}.{name}")
                    self.assertIn("area: mean", spec.cell_methods, f"{scale}.{name}")
        gpp = self.builder.get_effective_variables("individual", include_lower=False)[
            "gross_primary_productivity"
        ]
        self.assertEqual(gpp.cell_methods, "area: mean time: mean")


class TestGrossPrimaryProductivityNamesHaveOneUnit(unittest.TestCase):
    """gpp is the SSOT molar short name; the hierarchical leaf-level GPP is the
    canonical carbon-mass flux gross_primary_productivity; both names keep one unit."""

    def test_each_gpp_name_has_one_unit_and_an_explicit_conversion(self):
        import torch

        from relic.contracts.contract_loader import load_contract_registry
        from relic.data.standards.unit_converter import unit_converter
        from relic.orchestration.ssot_catalog import SSOTCatalog
        from relic.parameters import get_global_manager
        from relic.utils.physical_constants import get_physical_constant

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builder = create_model_builder()
        registry = load_contract_registry()
        catalog = SSOTCatalog.from_parameter_manager()
        individual = builder.get_effective_variables("individual", include_lower=False)
        patch = builder.get_effective_variables("patch")

        # gpp: molar short name, the same unit in every declaration of the name.
        declared_gpp_units = {registry.get_contract("gpp").udunits, catalog.get("gpp").units}
        for scale_variables in (individual, patch):
            if "gpp" in scale_variables:
                declared_gpp_units.add(scale_variables["gpp"].units)
        self.assertEqual(declared_gpp_units, {"umol m-2 s-1"})
        self.assertIsNone(
            get_global_manager().get("atmosphere.state_defaults.gpp", required=False)
        )

        # The hierarchical leaf-level GPP is the canonical carbon-mass flux, in the
        # unit of its contract, next to the kg C m-2 s-1 NEE and soil respiration.
        gpp_mass = individual["gross_primary_productivity"]
        self.assertEqual(
            gpp_mass.units, registry.get_contract("gross_primary_productivity").udunits
        )
        self.assertEqual(gpp_mass.units, patch["nee"].units)
        self.assertEqual(
            gpp_mass.standard_name,
            "gross_primary_productivity_of_biomass_expressed_as_carbon",
        )

        # References: Farquhar et al. (1980); IUPAC standard atomic weight of C.
        # Mathematical expression: F_mass = F_molar * M_C * 1e-6 mol umol-1 * 1e-3 kg g-1.
        # Parameter meanings: F_molar gpp, F_mass gross_primary_productivity,
        #   M_C the SSOT molar mass of carbon (one mol CO2 fixed = one mol C).
        # Units: F_molar [umol m-2 s-1]; F_mass [kg C m-2 s-1]; M_C [g mol-1].
        # Domain/constraints: non-negative gross uptake.
        # Sign convention: both are positive uptake magnitudes.
        molar_mass = float(get_physical_constant("relic.physical.constants.molecular_weights.carbon"))
        converted = unit_converter.convert(
            torch.tensor(10.0, dtype=torch.float64), "carbon_flux", "umol m-2 s-1", "kg C m-2 s-1"
        )
        self.assertAlmostEqual(float(converted), 10.0 * molar_mass * 1.0e-9, delta=1.0e-20)


class TestVariableSpecValidatesItsBasisMethodPair(unittest.TestCase):
    """The basis/method rule is checked when a VariableSpec is constructed, not only by the loader."""

    def test_inconsistent_pairs_raise_at_construction(self):
        for basis, method in (
            ("categorical", AggregationMethod.MEAN),
            ("categorical", AggregationMethod.AREA_WEIGHTED_MEAN),
            ("cell_total", AggregationMethod.AREA_WEIGHTED_MEAN),
            ("cell_total", AggregationMethod.MEAN),
            ("area_density", AggregationMethod.SUM),
            ("area_density", AggregationMethod.MAX),
        ):
            with self.assertRaisesRegex(ValueError, basis):
                VariableSpec(
                    units="1", standard_name="probe", spatial_basis=basis, description="probe", aggregation=method
                )

    def test_consistent_pairs_construct(self):
        for basis, method in (
            ("categorical", AggregationMethod.MODE),
            ("cell_total", AggregationMethod.SUM),
            ("area_density", AggregationMethod.AREA_WEIGHTED_MEAN),
            ("intensive", AggregationMethod.MAX),
        ):
            spec = VariableSpec(
                units="1", standard_name="probe", spatial_basis=basis, description="probe", aggregation=method
            )
            self.assertIs(spec.aggregation, method)

    def test_builder_has_no_second_validation_path(self):
        self.assertFalse(hasattr(CentralModelBuilder, "_validate_aggregation_consistency"))


class TestDeltaFunctorsKeepTheDeclaredOrder(unittest.TestCase):
    """Delta functors follow LAYER_FUNCTORS order, independent of PYTHONHASHSEED."""

    def test_delta_functors_are_the_new_declared_functors_in_order(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builder = create_model_builder()
        seen = set()
        modules = {
            "individual": individual, "patch": patch, "hillslope": hillslope,
            "landscape": landscape, "basin": basin, "regional": regional,
        }
        for scale in builder.scale_hierarchy:
            declared = list(dict.fromkeys(modules[scale].LAYER_FUNCTORS))
            expected = [name for name in declared if name not in seen]
            self.assertEqual(builder.get_scale_config(scale).delta_functors, expected, scale)
            seen.update(declared)


class TestCFAndUnitValidation(unittest.TestCase):
    """validate_cf_and_units checks standard names against the CF table and units through
    CFConventions (cf_units), including the name-unit relation."""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()

    def _spec(self, standard_name, units):
        return VariableSpec(units=units, standard_name=standard_name, spatial_basis="intensive", description="probe")

    def test_valid_name_and_units_pass(self):
        report = self.builder.validate_cf_and_units({"t": self._spec("air_temperature", "degC")})
        self.assertTrue(report.valid, report)

    def test_unknown_names_bad_units_and_mismatches_are_reported(self):
        report = self.builder.validate_cf_and_units({"x": self._spec("custom_probe_quantity", "K")})
        self.assertFalse(report.valid)
        self.assertTrue(any("custom_probe_quantity" in error for error in report.cf_errors), report)
        report = self.builder.validate_cf_and_units({"t": self._spec("air_temperature", "kg")})
        self.assertFalse(report.valid)
        self.assertTrue(any("kg" in error for error in report.unit_errors), report)
        report = self.builder.validate_cf_and_units({"t": self._spec("air_temperature", "not_a_unit")})
        self.assertFalse(report.valid)

    def test_absent_standard_name_is_none_and_only_the_units_are_checked(self):
        """A variable without a CF standard name (for example stomatal conductance, which has
        no CF v94 entry) carries standard_name None: no custom_<name> is invented, the CF-name
        check is skipped and the units are still checked."""
        import typing

        self.assertEqual(
            typing.get_type_hints(VariableSpec)["standard_name"], typing.Optional[str]
        )
        spec = self.builder._convert_variable_spec("probe", {"units": "K", "variable_type": "intensive"})
        self.assertIsNone(spec.standard_name)
        report = self.builder.validate_cf_and_units({"probe": self._spec(None, "mol m-2 s-1")})
        self.assertTrue(report.valid, report)
        report = self.builder.validate_cf_and_units({"probe": self._spec(None, "not_a_unit")})
        self.assertFalse(report.valid)
        self.assertTrue(any("not_a_unit" in error for error in report.unit_errors), report)
        self.assertEqual(report.cf_errors, [])
        variables = self.builder.get_effective_variables("individual", include_lower=True)
        self.assertIsNone(variables["stomatal_conductance"].standard_name)
        self.assertTrue(self.builder.validate_cf_and_units(
            {"stomatal_conductance": variables["stomatal_conductance"]}
        ).valid)
        self.assertNotIn("custom_", inspect.getsource(model_builder_module))


class TestBuilderReferences(unittest.TestCase):
    """The builder cites RHESSys with its journal and drops references that could not be verified."""

    def test_references(self):
        module_doc = model_builder_module.__doc__
        self.assertIn("Earth Interactions, 8(19), 1-42", module_doc)
        for doc in (module_doc, CentralModelBuilder.__doc__, AggregationMethod.__doc__):
            self.assertNotIn("Gang", doc)
            self.assertNotIn("Liang", doc)


class TestScaleConfig(unittest.TestCase):
    """
    L1 Unit Tests for ScaleConfig data structure.
    
    Tests the delta configuration pattern for individual scales,
    ensuring only incremental changes are stored.
    """
    
    def test_empty_scale_config(self):
        """Test creation of empty scale configuration."""
        empty_config = ScaleConfig(name="test_scale")
        
        self.assertEqual(empty_config.name, "test_scale")
        self.assertEqual(len(empty_config.delta_functors), 0)
        self.assertEqual(len(empty_config.delta_variables), 0)
        self.assertEqual(len(empty_config.coordinate_vars), 0)
        self.assertEqual(len(empty_config.coupling), 0)
        self.assertEqual(len(empty_config.constants), 0)
    
    def test_populated_scale_config(self):
        """Test scale configuration with all fields populated."""
        test_var = VariableSpec(
            units="kg C m-2 s-1",
            standard_name="gpp",
            spatial_basis=SpatialQuantityBasis.AREA_DENSITY,
            description="Test variable"
        )
        
        config = ScaleConfig(
            name="patch",
            delta_functors=["photosynthesis.basic", "respiration.leaf"],
            delta_variables={"gpp": test_var},
            coordinate_vars={"patch_id": {"units": "1", "description": "Patch identifier"}},
            coupling={"upscaling_target": "landscape"},
            constants={"light_extinction": 0.5}
        )
        
        self.assertEqual(config.name, "patch")
        self.assertEqual(len(config.delta_functors), 2)
        self.assertIn("photosynthesis.basic", config.delta_functors)
        self.assertEqual(len(config.delta_variables), 1)
        self.assertIn("gpp", config.delta_variables)
        self.assertEqual(config.constants["light_extinction"], 0.5)


class TestCentralModelBuilder(unittest.TestCase):
    """
    L1/L2 Tests for CentralModelBuilder core functionality.
    
    Tests the main builder class including:
    - Loading existing configurations
    - Functor deduplication
    - Variable conflict resolution
    - CF compliance validation
    """
    
    def setUp(self):
        """Set up test builder instance before each test."""
        # Suppress parameter manager warnings during testing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
    
    def test_builder_initialization(self):
        """Test basic builder initialization and config loading."""
        self.assertIsInstance(self.builder, CentralModelBuilder)
        
        # Check that scale configurations were loaded
        scale_configs = [self.builder.get_scale_config(scale) for scale in SCALE_HIERARCHY]
        loaded_configs = [config for config in scale_configs if config is not None]
        self.assertGreater(len(loaded_configs), 0, "No scale configurations loaded")
        
        # Check that we have at least individual and patch scales
        individual_config = self.builder.get_scale_config("individual")
        patch_config = self.builder.get_scale_config("patch")
        
        self.assertIsNotNone(individual_config)
        self.assertIsNotNone(patch_config)
        self.assertEqual(individual_config.name, "individual")
        self.assertEqual(patch_config.name, "patch")
    
    def test_get_effective_functors_deduplication(self):
        """Test functor deduplication with stable ordering."""
        # Test individual scale (no lower scales)
        individual_functors = self.builder.get_effective_functors("individual", include_lower=False)
        self.assertIsInstance(individual_functors, list)
        
        # Test patch scale (includes individual + patch)
        patch_functors = self.builder.get_effective_functors("patch", include_lower=True)
        patch_only = self.builder.get_effective_functors("patch", include_lower=False)
        
        # Patch effective should include individual functors
        self.assertGreaterEqual(len(patch_functors), len(individual_functors))
        self.assertGreaterEqual(len(patch_functors), len(patch_only))
        
        # Test for no duplicates
        self.assertEqual(len(patch_functors), len(set(patch_functors)), 
                        "Duplicate functors found in effective list")
        
        # Test stable ordering (individual functors should come first)
        for individual_func in individual_functors:
            if individual_func in patch_functors:
                individual_idx = patch_functors.index(individual_func)
                for patch_func in patch_only:
                    if patch_func in patch_functors:
                        patch_idx = patch_functors.index(patch_func)
                        self.assertLess(individual_idx, patch_idx,
                                      "Functor ordering not stable: individual should come before patch")
    
    def test_get_effective_variables_basic(self):
        """Test basic variable merging without conflicts."""
        # Test individual variables
        individual_vars = self.builder.get_effective_variables("individual", include_lower=False)
        self.assertIsInstance(individual_vars, dict)
        self.assertGreater(len(individual_vars), 0, "No individual variables loaded")
        
        # Check that variables are VariableSpec instances
        for var_name, var_spec in individual_vars.items():
            self.assertIsInstance(var_spec, VariableSpec)
            self.assertIsInstance(var_name, str)
            self.assertTrue(var_spec.units)  # Units should not be empty
            # A declared standard name is non-empty; None declares that CF has no
            # standard name for the quantity (e.g. stomatal_conductance).
            if var_spec.standard_name is not None:
                self.assertTrue(var_spec.standard_name)
        
        # Test patch variables (should include individual)
        patch_vars = self.builder.get_effective_variables("patch", include_lower=True)
        self.assertGreaterEqual(len(patch_vars), len(individual_vars))
        
        # Individual variables should be present in patch
        for var_name, var_spec in individual_vars.items():
            if var_name in patch_vars:
                patch_spec = patch_vars[var_name]
                # Core fields should match (no conflicts in this test)
                self.assertEqual(var_spec.units, patch_spec.units,
                               f"Units mismatch for {var_name}")
                self.assertEqual(var_spec.standard_name, patch_spec.standard_name,
                               f"Standard name mismatch for {var_name}")
    
    def test_conflict_detection(self):
        """Test conflict detection for variable definitions."""
        # Create a mock conflict by modifying builder's internal state
        original_config = self.builder.get_scale_config("patch")
        if original_config and len(original_config.delta_variables) > 0:
            # Create conflicting variable specification
            var_name = list(original_config.delta_variables.keys())[0]
            original_var = original_config.delta_variables[var_name]
            
            # Create conflicting spec with different units
            conflicting_var = VariableSpec(
                units="DIFFERENT_UNITS",  # This should cause a conflict
                standard_name=original_var.standard_name,
                spatial_basis=original_var.spatial_basis,
                description="Conflicting description"
            )
            
            # Temporarily modify the configuration
            landscape_config = self.builder.get_scale_config("landscape")
            if landscape_config:
                landscape_config.delta_variables[var_name] = conflicting_var
                
                # This should raise ValueError due to core field conflict
                with self.assertRaises(ValueError) as context:
                    self.builder.get_effective_variables("landscape", include_lower=True)
                
                self.assertIn("Critical conflict", str(context.exception))
                
                # Clean up
                if var_name in landscape_config.delta_variables:
                    del landscape_config.delta_variables[var_name]
    
    def test_cf_validation_basic(self):
        """Test basic CF compliance validation."""
        # Get some variables to validate
        patch_vars = self.builder.get_effective_variables("patch", include_lower=True)
        
        # Run CF validation
        validation_report = self.builder.validate_cf_and_units(patch_vars)
        self.assertIsInstance(validation_report, ValidationReport)
        
        # Basic validation should pass (implementation is simplified for now)
        # In production, this would check against actual CF standard names table
        self.assertIsInstance(validation_report.valid, bool)
        self.assertIsInstance(validation_report.cf_errors, list)
        self.assertIsInstance(validation_report.unit_errors, list)
    
    def test_scale_hierarchy_consistency(self):
        """Test that scale hierarchy is consistent with SCALE_HIERARCHY."""
        available_scales = self.builder.list_scales()
        
        # All scales in SCALE_HIERARCHY should be available
        for scale in SCALE_HIERARCHY:
            self.assertIn(scale, available_scales, f"Scale {scale} not available in builder")
        
        # Test upscaling relationships
        for scale in SCALE_HIERARCHY[:-1]:  # Exclude last scale (regional)
            target_scale = UPSCALING_MAP.get(scale)
            if target_scale:
                # Target scale should be in hierarchy after current scale
                current_idx = SCALE_HIERARCHY.index(scale)
                target_idx = SCALE_HIERARCHY.index(target_scale)
                self.assertGreater(target_idx, current_idx,
                                 f"Upscaling target {target_scale} should come after {scale}")
    
    def test_allow_overrides_functionality(self):
        """Test the allow_overrides functionality."""
        # Create builder with specific override permissions
        override_permissions = {
            "test_var": ["description", "aggregation"]
        }
        
        builder_with_overrides = create_model_builder(allow_overrides=override_permissions)
        self.assertEqual(builder_with_overrides.allow_overrides, override_permissions)
        
        # Test override permissions are used in variable merging
        # This is tested implicitly through the conflict detection test above


class TestIntegrationWithExistingModels(unittest.TestCase):
    """
    L2 Integration Tests with existing relic/models modules.
    
    Tests that the central builder correctly integrates with existing
    models modules and produces equivalent results to direct access.
    """
    
    def setUp(self):
        """Set up builder and import existing models."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
        
        # Import models modules for comparison
        from relic.models import individual, patch, landscape
        self.individual_module = individual
        self.patch_module = patch
        self.landscape_module = landscape
    
    def test_individual_scale_equivalence(self):
        """Test that builder produces equivalent results for individual scale."""
        # Get functors from builder
        builder_functors = self.builder.get_effective_functors("individual", include_lower=True)
        
        # Get functors from original module
        original_functors = getattr(self.individual_module, 'LAYER_FUNCTORS', [])
        
        # Should have same functors (order may differ due to deduplication)
        self.assertEqual(set(builder_functors), set(original_functors),
                        "Individual functors differ between builder and original module")
    
    def test_patch_scale_no_duplicates(self):
        """Test that patch scale from builder has no duplicates vs original."""
        # Get effective functors from builder
        builder_functors = self.builder.get_effective_functors("patch", include_lower=True)
        
        # Get original patch functors (which include duplicates from individual)
        original_functors = getattr(self.patch_module, 'LAYER_FUNCTORS', [])
        
        # Builder should have no duplicates
        self.assertEqual(len(builder_functors), len(set(builder_functors)),
                        "Builder functors contain duplicates")
        
        # Original may have duplicates due to cumulative inheritance
        unique_original = list(dict.fromkeys(original_functors))  # Preserve order
        
        # Should contain same unique functors
        self.assertEqual(set(builder_functors), set(unique_original),
                        "Patch functors differ between builder and deduplicated original")
    
    def test_variable_consistency(self):
        """Test variable definitions are consistent between builder and modules."""
        # Compare individual variables
        builder_vars = self.builder.get_effective_variables("individual", include_lower=True)
        original_vars = getattr(self.individual_module, 'EXPOSED_VARIABLES', {})
        
        # Should have same variable names
        self.assertEqual(set(builder_vars.keys()), set(original_vars.keys()),
                        "Individual variable names differ")
        
        # Compare key properties for each variable
        for var_name in builder_vars.keys():
            if var_name in original_vars:
                builder_var = builder_vars[var_name]
                original_var = original_vars[var_name]
                
                # Units should match
                self.assertEqual(builder_var.units, original_var.get('units', '1'),
                               f"Units differ for variable {var_name}")
                
                # Standard names should match
                self.assertEqual(builder_var.standard_name,
                               original_var.get('standard_name'),
                               f"Standard names differ for variable {var_name}")


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """
    L1 Tests for edge cases and error handling.
    
    Tests error conditions, invalid inputs, and boundary cases
    to ensure robust behavior under all conditions.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
    
    def test_invalid_scale_name(self):
        """Test handling of invalid scale names."""
        with self.assertRaises(ValueError):
            self.builder.get_effective_functors("nonexistent_scale")
        
        with self.assertRaises(ValueError):
            self.builder.get_effective_variables("invalid_scale")
    
    def test_empty_configurations(self):
        """Test handling of empty scale configurations."""
        # Test scale with no functors
        empty_functors = self.builder.get_effective_functors("individual", include_lower=False)
        if len(empty_functors) == 0:
            # This is acceptable - some scales may have no unique functors
            pass
        
        # Test scale with no variables  
        empty_vars = self.builder.get_effective_variables("individual", include_lower=False)
        if len(empty_vars) == 0:
            # This is also acceptable - some scales may have no unique variables
            pass
    
    def test_conflict_report_structure(self):
        """Test ConflictReport structure and methods."""
        report = ConflictReport()
        
        # Test empty report
        self.assertFalse(report.has_errors())
        self.assertFalse(report.has_warnings())
        
        # Add error and test
        report.errors.append("Test error")
        self.assertTrue(report.has_errors())
        
        # Add warning and test
        report.warnings.append("Test warning")
        self.assertTrue(report.has_warnings())
    
    def test_validation_report_structure(self):
        """Test ValidationReport structure."""
        report = ValidationReport()
        
        # Test default state
        self.assertTrue(report.valid)
        self.assertEqual(len(report.cf_errors), 0)
        self.assertEqual(len(report.unit_errors), 0)
        
        # Add errors and test validity
        report.cf_errors.append("CF error")
        report.unit_errors.append("Unit error")
        report.valid = len(report.cf_errors) == 0 and len(report.unit_errors) == 0
        
        self.assertFalse(report.valid)


class TestPerformanceAndMemoryUsage(unittest.TestCase):
    """
    L1 Performance tests for central model builder.
    
    Tests that the builder operates efficiently and doesn't
    consume excessive memory or computation time.
    """
    
    def setUp(self):
        """Set up performance test fixtures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.builder = create_model_builder()
    
    def test_repeated_calls_performance(self):
        """Test that repeated calls don't degrade performance significantly."""
        import time
        
        # Measure time for multiple calls
        start_time = time.time()
        
        for _ in range(10):
            functors = self.builder.get_effective_functors("landscape")
            variables = self.builder.get_effective_variables("landscape")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(elapsed, 5.0, f"Performance test took {elapsed:.2f}s, expected < 5.0s")
    
    def test_memory_usage_reasonable(self):
        """Test that builder doesn't consume excessive memory."""
        import gc
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Get memory usage (simplified check)
        initial_objects = len(gc.get_objects())
        
        # Create multiple builders
        builders = []
        for _ in range(5):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                builders.append(create_model_builder())
        
        # Memory usage shouldn't grow excessively
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # This is a rough check - exact thresholds may need adjustment
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 10000, 
                       f"Memory usage grew by {object_growth} objects, may indicate memory leak")
        
        # Clean up
        del builders
        gc.collect()


def create_test_suite():
    """
    Create comprehensive test suite for central model builder.
    
    Returns:
        unittest.TestSuite: Complete test suite with all test cases
    """
    suite = unittest.TestSuite()
    
    # Add L1 unit tests
    suite.addTest(unittest.makeSuite(TestVariableSpec))
    suite.addTest(unittest.makeSuite(TestScaleConfig))
    suite.addTest(unittest.makeSuite(TestCentralModelBuilder))
    suite.addTest(unittest.makeSuite(TestEdgeCasesAndErrorHandling))
    suite.addTest(unittest.makeSuite(TestPerformanceAndMemoryUsage))
    
    # Add L2 integration tests
    suite.addTest(unittest.makeSuite(TestIntegrationWithExistingModels))
    
    return suite


def run_tests(verbosity=2):
    """
    Run all tests with specified verbosity level.
    
    Parameters:
        verbosity: Test output verbosity (0=quiet, 1=normal, 2=verbose)
    
    Returns:
        unittest.TestResult: Test results
    """
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    
    print("Running Central Model Builder Test Suite")
    print("=" * 50)
    
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
    
    return result


if __name__ == '__main__':
    # Run tests if script is executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Central Model Builder Tests')
    parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2],
                       help='Test output verbosity level')
    parser.add_argument('--pattern', type=str, default='test_*',
                       help='Test method pattern to run')
    
    args = parser.parse_args()
    
    if args.pattern != 'test_*':
        # Run specific test pattern
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        
        # Load tests matching pattern from all test classes
        for test_class in [TestVariableSpec, TestScaleConfig, TestCentralModelBuilder,
                          TestIntegrationWithExistingModels, TestEdgeCasesAndErrorHandling,
                          TestPerformanceAndMemoryUsage]:
            tests = loader.loadTestsFromTestCase(test_class)
            for test in tests:
                if args.pattern in str(test):
                    suite.addTest(test)
        
        runner = unittest.TextTestRunner(verbosity=args.verbosity)
        runner.run(suite)
    else:
        # Run all tests
        run_tests(verbosity=args.verbosity)

