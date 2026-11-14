"""
Integration tests for RELIC configuration system.
"""
import pytest
import tempfile
import yaml
from pathlib import Path

from relic.core.config.models import RELICConfig, SimulationConfig, PhysicsConfig, IntegrationMethod
from relic.core.config.validator import ConfigValidator
from relic.core.config.loader import ConfigLoader, ConfigurationError
from relic.core.irf_specs import load_irf_specs, validate_irf_specs
from relic.core.state import StateManager, StateVariableSpec
from relic.parameters import get_global_manager


@pytest.mark.filterwarnings("ignore:Consider using MIMICS or CENTURY for nitrogen cycle")
def test_config_models():
    """Test configuration models with Pydantic V2."""
    print("\n" + "="*60)
    print("Testing Configuration Models...")
    print("="*60)
    
    # Create config model
    config = RELICConfig()
    print("[OK] Created RELICConfig model")
    assert "cf_compliant" not in type(config.data).model_fields
    assert "standard_names" not in type(config.data).model_fields
    assert "add_bounds" not in type(config.data).model_fields
    assert "Conventions" not in config.data.global_attributes
    
    # Test simulation settings
    config.simulation.time_step = 60.0
    config.simulation.end_time = 86400.0
    print(f"  - Time step: {config.simulation.time_step}")
    print(f"  - End time: {config.simulation.end_time}")
    
    # Test physics settings
    config.physics.carbon_cycle = True
    config.physics.nitrogen_cycle = True
    print(f"  - Carbon cycle: {config.physics.carbon_cycle}")
    print(f"  - Nitrogen cycle: {config.physics.nitrogen_cycle}")
    
    # Test stochastic settings
    config.stochastic.enable_weather_generator = True
    config.stochastic.num_realizations = 100
    print(f"  - Weather generator: {config.stochastic.enable_weather_generator}")
    print(f"  - Num realizations: {config.stochastic.num_realizations}")
    
    # Test validation
    from pydantic import ValidationError
    try:
        config.simulation.time_step = -1.0  # Should fail
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("[OK] Pydantic validation working correctly")
    
    # Test completed successfully


def test_simulation_config_has_one_source_for_time_defaults():
    """The typed simulation section declares no adaptive-step fields (the runtime
    keys adaptive_dt_* and computation.scheduling.adaptive.* own them) and takes
    its time defaults from the time-constant SSOT."""
    import pytest
    from pydantic import ValidationError

    from relic.core.config.models import SimulationConfig
    from relic.utils.time_constants import SECONDS_PER_DAY, SECONDS_PER_HOUR

    for removed in ("adaptive_timestep", "min_timestep", "max_timestep"):
        assert removed not in SimulationConfig.model_fields, removed
    config = SimulationConfig()
    assert config.end_time == SECONDS_PER_DAY
    SimulationConfig(time_step=SECONDS_PER_HOUR)
    with pytest.raises(ValidationError, match="time_step"):
        SimulationConfig(time_step=SECONDS_PER_HOUR * 2.0, end_time=SECONDS_PER_DAY)


def test_config_validator():
    """Test configuration validator with rules."""
    print("\n" + "="*60)
    print("Testing Configuration Validator...")
    print("="*60)
    
    # Create validator
    validator = ConfigValidator()
    stats = validator.get_validation_statistics()
    print(f"[OK] Created ConfigValidator")
    print(f"  - Number of rules: {stats['num_rules']}")
    print(f"  - Rule IDs: {', '.join(stats['rule_ids'][:5])}...")
    assert "cf_unit_consistency" not in stats["rule_ids"]
    
    # Create test configuration - using model_construct to bypass Pydantic validation
    # This allows us to test our custom validator rules
    config_data = {
        'simulation': {
            'start_time': 0.0,
            'end_time': 86400.0,
            'time_step': -1.0,  # Invalid negative time step for testing
            'integration_method': IntegrationMethod.EULER,
            'tolerance_relative': 1e-6,
            'tolerance_absolute': 1e-9,
            'max_iterations': 100,
            'convergence_criterion': 1e-8
        },
        'physics': {
            'nitrogen_cycle': True,
            'carbon_cycle': False,  # Invalid: N cycle needs C cycle
            'phosphorus_cycle': False,
            'water_balance': True,
            'energy_balance': True,
            'canopy_interception': True,
            'soil_evaporation': True,
            'snow_dynamics': False,
            'phenology': True,
            'allocation': True,
            'mortality': True,
            'fire': False,
            'herbivory': False,
            'radiation_scheme': 'two_stream',
            'stomatal_model': 'ball_berry',
            'photosynthesis_model': 'farquhar',
            'soil_biogeochemistry': 'century',
            'hydraulics': False,
            'mycorrhizae': False,
            'root_dynamics': True,
            'litter_decomposition': True,
            'soil_thermal': True,
            'permafrost': False,
            'wetlands': False,
            'voc_emissions': False,
            'ozone_damage': False,
            'nutrient_limitation': 'liebig',
            'carbon_allocation': 'allometric'
        }
    }
    
    # Use model_construct to bypass Pydantic validation for testing
    config = RELICConfig.model_construct(
        simulation=SimulationConfig.model_construct(**config_data['simulation']),
        physics=PhysicsConfig.model_construct(**config_data['physics'])
    )
    
    # Validate configuration using our custom validator
    result = validator.validate(config)
    print(f"[OK] Validated configuration")
    print(f"  - Valid: {result.is_valid}")
    print(f"  - Errors: {result.num_errors}")
    print(f"  - Warnings: {result.num_warnings}")
    print(f"  - Validation time: {result.validation_time:.3f}s")
    
    # Show some error messages
    if result.num_errors > 0:
        print("  - Sample errors:")
        for msg in result.get_messages_by_severity('error')[:3]:
            print(f"    - {msg.message}")
    
    # Test should find validation errors
    assert not result.is_valid, "Should detect validation errors"
    assert result.num_errors > 0, "Should have errors for invalid configuration"
    
    # Test completed successfully


@pytest.mark.filterwarnings("ignore:Consider using MIMICS or CENTURY for nitrogen cycle")
def test_config_loader():
    """Test configuration loader with YAML files."""
    print("\n" + "="*60)
    print("Testing Configuration Loader...")
    print("="*60)
    
    # Create loader
    loader = ConfigLoader()
    print("[OK] Created ConfigLoader")
    
    # Create temporary YAML config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'simulation': {
                'time_step': 30.0,
                'end_time': 172800.0,
                'max_iterations': 200  # Use valid field
            },
            'physics': {
                'carbon_cycle': True,
                'nitrogen_cycle': True,
                'water_balance': True
            }
        }, f)
        temp_file = f.name
    
    try:
        # Load configuration
        config = loader.load_config(Path(temp_file), env_overrides=False)
        print(f"[OK] Loaded configuration from YAML")
        print(f"  - Time step: {config.simulation.time_step}")
        print(f"  - End time: {config.simulation.end_time}")
        print(f"  - Carbon cycle: {config.physics.carbon_cycle}")
        
        # Test validation during loading
        loader.validate_on_load = True
        config_validated = loader.load_config(Path(temp_file), env_overrides=False)
        print("[OK] Configuration validated during loading")
        
        # Test overrides
        config_override = loader.load_config(
            Path(temp_file),
            overrides={'simulation.time_step': 60.0},
            env_overrides=False
        )
        print(f"[OK] Applied overrides")
        print(f"  - New time step: {config_override.simulation.time_step}")
        
    finally:
        # Cleanup
        Path(temp_file).unlink()
    
    # Test completed successfully


def test_parameter_integration():
    """Register, validate, read, and remove a parameter on the global manager.

    The global manager is a process-wide singleton, so the registered key must
    be removed again: checkpoint scientific contracts record every parameter
    accessed in the process, and a leaked runtime-registered key breaks restart
    contract comparison in later tests.
    """
    from relic.parameters import ParameterSpec, get_global_manager

    manager = get_global_manager()
    key = "test_param"
    spec = ParameterSpec(
        key=key,
        value=0.5,
        units="dimensionless",
        dtype="float32",
        bounds=(0.0, 1.0),
        category="biophysical",
        description="Test parameter",
        references=[],
        tags=set(["test"]),
        dimension="dimensionless",
        provenance={"source": "test", "version": "1.0.0"},
        requires_grad=False,
    )
    manager.register_parameter(spec)
    try:
        assert manager.validate_parameters([key]) == []
        assert manager.get(key, required=True) == pytest.approx(0.5)
        registered = manager.get_parameter(key)
        assert registered is not None and registered.units == "dimensionless"
    finally:
        manager.delete(key)
    assert manager.get_parameter(key) is None
    assert manager.get(key, default=None) is None


def test_irf_loader_and_validation_strict_off(tmp_path: Path):
    irf_path = tmp_path / "irf_specs.yaml"

    pm = get_global_manager(preload_all=False, warmup_cache=False)
    spec = pm.get_parameter("relic.physical.constants.time.seconds_per_day")
    assert spec is not None

    param_entry = {
        "key": spec.key,
        "units": spec.units,
        "required": True,
    }
    if spec.bounds is not None:
        param_entry["bounds"] = list(spec.bounds)

    irf_payload = {
        "schema_version": "1.0.0",
        "strict": False,
        "init_specs": {
            "parameters": [param_entry],
            "state_variables": [{
                "name": "air_temperature",
                "units": "K",
                "bounds": [0.0, 400.0],
            }],
        },
        "run_specs": {
            "default_mode": "replace",
        },
    }
    irf_path.write_text(yaml.safe_dump(irf_payload), encoding="utf-8")

    irf_specs = load_irf_specs(irf_path)
    state_schema = {
        "air_temperature": StateVariableSpec(
            name="air_temperature",
            units="K",
            bounds=(0.0, 400.0),
            conserved=False,
            required=False,
        )
    }
    state_manager = StateManager(schema=state_schema)
    report = validate_irf_specs(
        irf_specs,
        parameter_manager=pm,
        state_manager=state_manager,
        strict=False,
    )
    assert report.errors == []


def test_irf_validation_strict_raises(tmp_path: Path):
    irf_path = tmp_path / "irf_specs_bad.yaml"

    irf_payload = {
        "schema_version": "1.0.0",
        "strict": True,
        "init_specs": {
            "parameters": [{
                "key": "missing.parameter.key",
                "units": "1",
                "required": True,
            }],
        },
    }
    irf_path.write_text(yaml.safe_dump(irf_payload), encoding="utf-8")

    irf_specs = load_irf_specs(irf_path)
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    with pytest.raises(ValueError):
        validate_irf_specs(irf_specs, parameter_manager=pm, strict=True)


def test_schema_export():
    """Test configuration schema export."""
    print("\n" + "="*60)
    print("Testing Schema Export...")
    print("="*60)
    
    # Create config
    config = RELICConfig()
    
    # Export schema
    schema = config.model_json_schema()
    print(f"[OK] Exported JSON schema")
    print(f"  - Schema title: {schema.get('title', 'N/A')}")
    print(f"  - Number of properties: {len(schema.get('properties', {}))}")
    
    # Verify key sections
    assert 'simulation' in schema['properties']
    assert 'physics' in schema['properties']
    assert 'stochastic' in schema['properties']
    print("[OK] Schema contains expected sections")
    
    # Test completed successfully


# Non-ASCII text whose UTF-8 bytes are not decodable as GBK (the platform
# default on a Chinese-locale Windows host): an EM DASH followed by a space.
BOM_DESCRIPTION = "Units umol m-2 s-1 \u2014 BOM round trip"


@pytest.mark.parametrize("entry_point", ["RELICConfig.from_yaml", "ConfigLoader.load_config"])
def test_configuration_loaders_read_utf8_with_bom(tmp_path: Path, entry_point: str):
    """Configuration files are UTF-8, a leading BOM is accepted, never the locale codec."""
    path = tmp_path / "bom_config.yaml"
    body = yaml.safe_dump(
        {"description": BOM_DESCRIPTION, "simulation": {"time_step": 30.0, "end_time": 172800.0}},
        allow_unicode=True,
    )
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))

    if entry_point == "RELICConfig.from_yaml":
        config = RELICConfig.from_yaml(path)
    else:
        config = ConfigLoader().load_config(path, env_overrides=False)

    assert config.description == BOM_DESCRIPTION
    assert config.simulation.time_step == 30.0


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
# RELICConfig fields read by a RELIC entry point (all others are read by no runtime code):
#   simulation.end_time / time_step / integration_method -> RELICSimulator max_time /
#     time_step / time_integration_method through the documented runtime mapping;
#   parameters.parameter_overrides -> ConfigLoader.load_config -> ParameterManager.set_value;
#   computation.device / enable_distributed / enable_profiling -> RELICSimulator settings
#     (computation section of a dumped RELICConfig);
#   computation.parallel_backend / chunk_size / num_workers / enable_profiling ->
#     ParallelConfig.from_computation_config (parallel_map(computation_config=...));
#   computation.compile_backend / compile_mode -> relic -c <file> compute bench --jit.
# simulation_name and description are identity labels, not settings.
CONSUMED_RELICCONFIG_FIELDS = {
    "simulation_name",
    "description",
    "simulation.end_time",
    "simulation.time_step",
    "simulation.integration_method",
    "parameters.parameter_overrides",
    "computation.device",
    "computation.enable_distributed",
    "computation.enable_profiling",
    "computation.parallel_backend",
    "computation.chunk_size",
    "computation.num_workers",
    "computation.compile_backend",
    "computation.compile_mode",
}


def _typed_example_paths():
    """Example files written for RELICConfig (named or made only of RELICConfig fields)."""
    paths = []
    for path in sorted(EXAMPLES_DIR.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        if "simulation_name" in payload or set(payload) <= set(RELICConfig.model_fields):
            paths.append(path)
    return paths


def _set_fields(payload):
    fields = set()
    for key, value in payload.items():
        model_field = RELICConfig.model_fields.get(key)
        is_section = model_field is not None and isinstance(value, dict) and key != "surrogates"
        if is_section:
            fields.update(f"{key}.{name}" for name in value)
        else:
            fields.add(key)
    return fields


@pytest.mark.parametrize("path", _typed_example_paths(), ids=lambda path: path.name)
def test_typed_example_loads_and_sets_only_consumed_fields(path: Path):
    """A typed example validates against the schema and suggests no setting that does nothing."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))

    loader = ConfigLoader(validate_on_load=True)
    overrides = _ReadOnlyParameterManager()
    loader.set_parameter_manager(overrides)
    validated = loader.load_config(path, env_overrides=False)

    assert isinstance(validated, RELICConfig)
    assert _set_fields(payload) - CONSUMED_RELICCONFIG_FIELDS == set()
    assert dict(overrides.writes) == (payload.get("parameters") or {}).get("parameter_overrides", {})
    for key in (payload.get("parameters") or {}).get("parameter_overrides") or {}:
        assert get_global_manager().get_parameter(key) is not None, key


class _ReadOnlyParameterManager:
    """Resolves specs from the global manager but records override writes without applying them."""

    def __init__(self):
        self.writes = []

    def get_parameter(self, key):
        return get_global_manager().get_parameter(key)

    def set_value(self, key, value, **kwargs):
        self.writes.append((key, value))


class _RecordingParameterManager:
    """Parameter manager stand-in that knows no parameter and records every write."""

    def __init__(self):
        self.writes = []

    def get_parameter(self, key):
        return None

    def set_value(self, key, value, **kwargs):
        self.writes.append((key, value))


def test_parameter_override_of_an_unregistered_key_is_rejected(tmp_path: Path):
    """An override that names no registered parameter would change nothing: it fails."""
    path = tmp_path / "override.yaml"
    path.write_text(
        yaml.safe_dump({"parameters": {"parameter_overrides": {"examples.unregistered_key": 2.0}}}),
        encoding="utf-8",
    )
    loader = ConfigLoader()
    manager = _RecordingParameterManager()
    loader.set_parameter_manager(manager)

    with pytest.raises(ConfigurationError, match="examples.unregistered_key"):
        loader.load_config(path, env_overrides=False)
    assert manager.writes == []


@pytest.mark.parametrize("overrides", [{"simulation.time_step": 1800.0}, ["simulation.time_step=1800.0"]])
def test_config_loader_applies_dotted_overrides(tmp_path: Path, overrides):
    """Dotted overrides given as a mapping or as key=value strings change the loaded value."""
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump({"simulation": {"time_step": 60.0, "end_time": 86400.0}}), encoding="utf-8")

    config = ConfigLoader().load_config(path, overrides=overrides, env_overrides=False)

    assert config.simulation.time_step == 1800.0


def test_config_loader_rejects_an_override_without_a_value(tmp_path: Path):
    """An override entry that cannot be applied fails instead of being skipped."""
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump({"simulation": {"time_step": 60.0, "end_time": 86400.0}}), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="simulation.time_step"):
        ConfigLoader().load_config(path, overrides=["simulation.time_step"], env_overrides=False)


def test_environment_overrides_map_schema_fields_only(tmp_path: Path, monkeypatch):
    """RELIC_<SECTION>_<FIELD> overrides that field; other RELIC_* process switches are not config."""
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump({"simulation": {"time_step": 60.0, "end_time": 86400.0}}), encoding="utf-8")
    monkeypatch.setenv("RELIC_SIMULATION_TIME_STEP", "120.0")
    monkeypatch.setenv("RELIC_PHYSICS_CARBON_CYCLE", "false")
    monkeypatch.setenv("RELIC_DATA_IO_COMPRESSION_LEVEL", "7")
    monkeypatch.setenv("RELIC_SKIP_CALIBRATED", "1")
    monkeypatch.setenv("RELIC_PROJECT_ROOT", str(tmp_path))

    config = ConfigLoader().load_config(path, env_overrides=True)

    assert config.simulation.time_step == 120.0
    assert config.physics.carbon_cycle is False
    assert config.data.io.compression_level == 7


def test_default_output_variables_are_canonical_written_variables():
    """Each default output name is a ContractRegistry variable (not an alias or a
    short name) with at least one declared writer: gpp is the molar SSOT name of GPP
    [umol m-2 s-1] that no Functor writes, while the photosynthesis Functors write
    gross_primary_productivity [kg C m-2 s-1]; npp and et have no contract."""
    from relic.contracts import load_contract_registry
    from relic.core.config.models import DataConfig

    registry = load_contract_registry()
    problems = []
    for name in DataConfig().output_variables:
        contract = registry.get_contract(name)
        if contract is None:
            problems.append(f"{name}: no variable contract")
        elif not (contract.owner or contract.writers):
            problems.append(f"{name}: no declared writer")
    assert problems == []


def test_integration_method_schema_equals_the_runtime_choice():
    """simulation.integration_method feeds RELICSimulator time_integration_method.

    The runtime owner of the choice is transport.numerical.time_integration_method
    (allowed_values), resolved by the transport operators; the schema enum must
    accept exactly those values.
    """
    from relic.computation.operators.functional import _resolve_transport_time_integration

    spec = get_global_manager().get_parameter("transport.numerical.time_integration_method")
    schema_values = {method.value for method in IntegrationMethod}

    assert schema_values == set(spec.allowed_values)
    for method in schema_values:
        assert _resolve_transport_time_integration(method, 4, 1.0e-10)[0] == method
    with pytest.raises(ValueError, match="Unsupported time integration method"):
        _resolve_transport_time_integration("rk4", 4, 1.0e-10)


def test_from_yaml_and_config_loader_share_one_load_path(tmp_path: Path, monkeypatch):
    """RELICConfig.from_yaml is ConfigLoader.load_config: same overrides, same environment."""
    path = tmp_path / "overrides.yaml"
    path.write_text(
        yaml.safe_dump({
            "simulation": {"time_step": 60.0, "end_time": 86400.0},
            "parameters": {"parameter_overrides": {"biosphere.photosynthesis.vcmax25": 61.0}},
        }),
        encoding="utf-8",
    )
    writes = []
    monkeypatch.setattr(
        get_global_manager(), "set_value", lambda key, value, **kwargs: writes.append((key, value))
    )
    monkeypatch.setenv("RELIC_SIMULATION_TIME_STEP", "90.0")

    from_yaml = RELICConfig.from_yaml(path)
    from_loader = ConfigLoader().load_config(path)

    assert writes == [("biosphere.photosynthesis.vcmax25", 61.0)] * 2
    assert from_yaml.simulation.time_step == from_loader.simulation.time_step == 90.0
    assert from_yaml.model_dump(exclude={"created_at"}) == from_loader.model_dump(exclude={"created_at"})


def test_config_loader_uses_the_one_configuration_source_rule(tmp_path: Path, monkeypatch):
    """Named file (argument, else RELIC_CONFIG_PATH) must exist; otherwise the
    first existing DEFAULT_CONFIG_PATHS entry; otherwise the global defaults.
    No directory search guesses a file."""
    import relic.core.config.loader as loader_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RELIC_CONFIG_PATH", raising=False)
    monkeypatch.delenv("RELIC_SIMULATION_TIME_STEP", raising=False)
    monkeypatch.setattr(loader_module, "DEFAULT_CONFIG_PATHS", [])
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "relic_other.yaml").write_text(
        yaml.safe_dump({"simulation": {"time_step": 45.0, "end_time": 86400.0}}), encoding="utf-8"
    )

    defaults = ConfigLoader().load_config()
    assert defaults.model_dump(mode="json", exclude={"created_at"}) == RELICConfig().model_dump(
        mode="json", exclude={"created_at"}
    )

    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        ConfigLoader().load_config(tmp_path / "absent.yaml")
    monkeypatch.setenv("RELIC_CONFIG_PATH", str(tmp_path / "absent.yaml"))
    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        ConfigLoader().load_config()

    monkeypatch.setenv("RELIC_CONFIG_PATH", str(tmp_path / "config" / "relic_other.yaml"))
    assert ConfigLoader().load_config().simulation.time_step == 45.0


def test_directory_configuration_is_composed_from_that_directory(tmp_path: Path):
    """load_config(<dir>) composes <dir>/config.yaml with Hydra, not ./config."""
    (tmp_path / "config.yaml").write_text(
        "simulation:\n  time_step: 123.0\n  end_time: 3600.0\n", encoding="utf-8"
    )
    loader = ConfigLoader(validate_on_load=False)

    config = loader.load_config(tmp_path, overrides=["simulation.end_time=7200.0"], env_overrides=False)

    assert config.simulation.time_step == 123.0
    assert config.simulation.end_time == 7200.0


def test_hydra_integration_failures_propagate(tmp_path: Path, monkeypatch):
    """Setup, composition and default merging raise; nothing returns None or the unmerged input."""
    from hydra.core.config_store import ConfigStore
    from hydra.errors import MissingConfigException

    import relic.core.config.loader as loader_module

    integration = loader_module.HydraIntegration(config_path=tmp_path)
    with pytest.raises(MissingConfigException):
        integration.compose_config("missing_name")
    with pytest.raises(MissingConfigException):
        integration.merge_with_defaults({"a": 1}, ["missing_default"])

    def _broken_store():
        raise RuntimeError("config store unavailable")

    monkeypatch.setattr(ConfigStore, "instance", staticmethod(_broken_store))
    with pytest.raises(RuntimeError, match="config store unavailable"):
        loader_module.HydraIntegration(config_path=tmp_path)


def test_omegaconf_bridge_failure_is_not_replaced_by_direct_conversion():
    from omegaconf import OmegaConf

    class _BrokenBridge:
        def get_parameter(self, key):
            return None

        def load_omegaconf_config(self, cfg, namespace):
            raise RuntimeError("bridge failed")

    loader = ConfigLoader(validate_on_load=False, parameter_manager=_BrokenBridge())

    with pytest.raises(RuntimeError, match="bridge failed"):
        loader._omegaconf_to_dict(OmegaConf.create({"a": 1}), namespace="test")
    with pytest.raises(RuntimeError, match="bridge failed"):
        loader._apply_hydra_overrides({"simulation": {"time_step": 1.0}}, ["simulation.time_step=2.0"])


def test_temperature_rule_does_not_skip_the_celsius_check_when_the_constant_fails(monkeypatch):
    """A failed constant lookup fails the rule; the validator records it as an error."""
    from relic.core.config.validator import ConfigValidator

    manager = get_global_manager()
    original_get = manager.get

    def _get(key, *args, **kwargs):
        if key == "relic.physical.constants.temperature.celsius_to_kelvin":
            raise RuntimeError("constant store unreadable")
        return original_get(key, *args, **kwargs)

    monkeypatch.setattr(manager, "get", _get)
    result = ConfigValidator().validate(RELICConfig())

    assert not result.is_valid
    assert any("constant store unreadable" in str(message) for message in result.messages)


def test_directory_configuration_without_hydra_is_rejected(tmp_path: Path):
    """A directory is composed by Hydra only; without Hydra it is not read as config.yaml."""
    (tmp_path / "config.yaml").write_text("simulation:\n  time_step: 123.0\n", encoding="utf-8")
    loader = ConfigLoader(validate_on_load=False, use_hydra=False)

    with pytest.raises(ConfigurationError, match="Hydra"):
        loader._load_hydra_config(tmp_path)


if __name__ == "__main__":
    # Run all tests
    print("\n" + "="*70)
    print(" RELIC Configuration System Integration Tests ")
    print("="*70)
    
    tests = [
        test_config_models,
        test_config_validator,
        test_config_loader,
        test_parameter_integration,
        test_schema_export
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"[ERROR] Test {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f" Results: {passed} passed, {failed} failed ")
    print("="*70)

