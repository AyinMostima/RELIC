"""Extension load and unload lifecycle (E03).

An extension whose functor wraps a native plugin (``PluginFunctorAdapter``)
is loaded, registered on a simulator, and unloaded. The loader must consume
the simulator lifecycle receipt, release the native library only after its
in-flight call drained, and keep the loader record of a half-unloaded
functor so the unload can be retried.

Load, parameter-manager, and registration failures of a Python extension
propagate with their cause; the functors constructed before the failure stay
recorded under their extension so ``unload_extension`` can release them.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from relic.core.simulator import RELICSimulator
from relic.coupling.plugin.host import PluginLifecycleError
from relic.extensions import loader as loader_module
from relic.extensions.hooks import load_and_register_extensions, unload_from_runtime
from relic.extensions.loader import (
    ExtensionLoadError,
    ExtensionValidationError,
    get_extension_registry,
    get_loaded_extensions,
    load_extensions,
    load_single_extension,
    unload_extension,
)
from relic.extensions.protocol import MANIFEST_FILENAME
from relic.parameters import get_global_manager
from relic.parameters.manager import UnifiedParameterManager

REPO_ROOT = Path(__file__).resolve().parents[3]
RELAY_SOURCE = REPO_ROOT / "examples" / "plugins" / "relay_plugin.cpp"
RELAY_PORT = "plugin_relay_value"
DELAY_PORT = "plugin_relay_delay_s"
EXTENSION_NAME = "relay_native_extension"
FUNCTOR_NAME = "relay_coupler"


@pytest.fixture(scope="session")
def relay_plugin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("g++") is None:
        pytest.skip("C++23 toolchain (g++) unavailable for the relay plugin")
    suffix = ".dll" if shutil.os.name == "nt" else ".so"
    output = tmp_path_factory.mktemp("relay_plugin_loader") / f"relay_plugin{suffix}"
    result = subprocess.run(
        ["g++", "-shared", "-O2", "-std=c++23", "-static-libstdc++", "-static-libgcc", "-o", str(output), str(RELAY_SOURCE)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"relay plugin did not compile: {result.stderr.strip()}")
    return output


def _create_native_extension(root: Path, plugin_path: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "functor.py").write_text(
        dedent(
            """
            from relic.coupling.plugin import PluginFunctorAdapter


            class RelayCoupler(PluginFunctorAdapter):
                def __init__(self, config=None, name="relay_coupler", **kwargs):
                    config = dict(config or {})
                    super().__init__(name=name, plugin_path=config["plugin_path"], config=config, **kwargs)
            """
        ),
        encoding="utf-8",
    )
    manifest = {
        "name": EXTENSION_NAME,
        "version": "0.1.0",
        "author": "unit-test",
        "description": "native relay plugin extension",
        "min_relic_version": "1.0.0",
        "functors": [
            {
                "class": "functor.RelayCoupler",
                "bridge": "ScaleCouplingFunctor",
                "category": "scale_coupling",
                "execution_group": "fast",
                "config": {
                    "plugin_path": str(plugin_path),
                    # The plugin adapter declares the scales it couples (required config keys).
                    "source_scale": "external_grid",
                    "target_scale": "grid_cell",
                    "coupling_type": "spatial",
                },
            }
        ],
        "parameters": [],
        "dependencies": {"required": [], "optional": []},
    }
    (root / MANIFEST_FILENAME).write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return root


PY_EXTENSION_NAME = "python_load_failure_extension"
PY_SECOND_EXTENSION_NAME = "python_second_extension"

_PY_FUNCTOR_CLASS = """
class {class_name}(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, config=None, name="{functor_name}", **kwargs):
        metadata = FunctorMetadata(
            name=name,
            category=ProcessCategory.TRANSPORT_MECHANISM,
            description="python extension functor",
            references=["Fick (1855)."],
            input_variables=["gradient"],
            output_variables=["{functor_name}_flux"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        super().__init__(name=name, config=config or {{}}, metadata=metadata, **kwargs)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {{"{functor_name}_flux": torch.tensor(0.0)}}
"""


def _create_python_extension(
    root: Path,
    module_name: str,
    class_names: list[str],
    functor_names: list[str],
    *,
    extension_name: str = PY_EXTENSION_NAME,
    parameter_namespace: str | None = None,
) -> Path:
    """Write a Python extension; a class name without a functor name is declared but not defined."""
    root.mkdir(parents=True, exist_ok=True)
    header = (
        "import torch\n"
        "from relic.core.functor import (\n"
        "    FunctorMetadata,\n"
        "    ProcessCategory,\n"
        "    SpatialScale,\n"
        "    TransportMechanismFunctor,\n"
        "    VariableKind,\n"
        ")\n"
    )
    bodies = [
        _PY_FUNCTOR_CLASS.format(class_name=class_name, functor_name=functor_name)
        for class_name, functor_name in zip(class_names, functor_names, strict=False)
    ]
    (root / f"{module_name}.py").write_text(header + "\n".join(bodies), encoding="utf-8")
    manifest = {
        "name": extension_name,
        "version": "0.1.0",
        "author": "unit-test",
        "description": "python extension for load failure propagation",
        "min_relic_version": "1.0.0",
        "functors": [
            {
                "class": f"{module_name}.{class_name}",
                "bridge": "TransportMechanismFunctor",
                "category": "transport_mechanism",
                "execution_group": "fast",
            }
            for class_name in class_names
        ],
        "parameters": [],
        "dependencies": {"required": [], "optional": []},
    }
    if parameter_namespace is not None:
        _write_extension_parameter_file(root / "parameters.yaml", parameter_namespace)
        manifest["parameters"] = [{"path": "parameters.yaml"}]
    (root / MANIFEST_FILENAME).write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return root


def _write_extension_parameter_file(path: Path, namespace: str) -> None:
    payload = {
        "schema_version": "1.0.0",
        "tier": "extension",
        "namespace": namespace,
        "description": "extension parameter file",
        "parameters": {
            "decomposition_rate": {
                "value": 1.0e-7,
                "units": "1/s",
                "description": "First-order decomposition rate constant",
                "bounds": [0.0, 1.0e-4],
                "dtype": "float32",
                "category": "biological_parameter",
                "references": ["Lloyd and Taylor (1994)"],
                "tags": ["extension"],
                "requires_grad": False,
                "provenance": {"source": "unit_test_extension", "schema_version": "1.0.0"},
            }
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class _RecordingSimulator:
    """Simulator stand-in exposing only the registration surface the hook uses."""

    def __init__(self, *, refusal: Exception | None = None):
        self.refusal = refusal
        self.registered: list[str] = []
        self.functors: list = []

    def register_functor(self, functor, execution_group=None):
        if self.refusal is not None:
            raise self.refusal
        self.registered.append(functor.name)
        self.functors.append(functor)


def _stub_parameter_manager() -> SimpleNamespace:
    return SimpleNamespace(database=SimpleNamespace(cache_dirty=False), parameters={})


@pytest.fixture
def clean_loader_records():
    yield
    # Never leak a half-unloaded record into the next test.
    for name in list(get_loaded_extensions()):
        if name in (EXTENSION_NAME, PY_EXTENSION_NAME, PY_SECOND_EXTENSION_NAME):
            unload_extension(name)


@pytest.mark.usefixtures("clean_loader_records")
class TestUnloadOrder:
    def test_unload_consumes_lifecycle_receipt_then_releases_native_host(self, tmp_path: Path, relay_plugin: Path):
        root = _create_native_extension(tmp_path / EXTENSION_NAME, relay_plugin)
        simulator = RELICSimulator(config={})
        summary = load_and_register_extensions(
            simulator, parameter_manager=get_global_manager(), extra_dirs=[root], validate=True, strict=True
        )
        assert summary["registered_count"] == 1
        functor = simulator.functors[FUNCTOR_NAME]
        functor.initialize({"relay_gain": 2.0})
        assert functor.host.state == "loaded"
        epoch_before = simulator.provider_epoch

        unload_summary = unload_from_runtime(simulator, EXTENSION_NAME, strict=True)

        assert unload_summary["complete"] is True
        assert unload_summary["removed_functors"] == [FUNCTOR_NAME]
        assert FUNCTOR_NAME not in simulator.functors
        receipt = unload_summary["receipts"][FUNCTOR_NAME]
        assert receipt["status"] == "committed" and receipt["epoch_before"] == epoch_before
        assert receipt["epoch_after"] > epoch_before and receipt["receipt_id"]
        native = unload_summary["native"][FUNCTOR_NAME]
        assert native["released"] is True and native["drained"] is True
        assert native["abi"]["native"]["standard"] == "c++23"
        assert functor.host.state == "unloaded"
        # The provider was retired before the library was released: the host
        # receipt trail ends with retire -> unload, both after the load.
        actions = [item["action"] for item in functor.host.receipts]
        assert actions == ["load", "retire", "unload"]
        assert EXTENSION_NAME not in get_loaded_extensions()

    def test_expired_native_drain_keeps_the_record_and_retry_completes(self, tmp_path: Path, relay_plugin: Path):
        root = _create_native_extension(tmp_path / EXTENSION_NAME, relay_plugin)
        simulator = RELICSimulator(config={})
        load_and_register_extensions(
            simulator, parameter_manager=get_global_manager(), extra_dirs=[root], validate=True, strict=True
        )
        functor = simulator.functors[FUNCTOR_NAME]
        functor.initialize()
        started = threading.Event()

        def blocking_exchange() -> None:
            started.set()
            functor.host.exchange(DELAY_PORT, np.array([0.5]), 0.0)

        worker = threading.Thread(target=blocking_exchange, name="native-inflight")
        worker.start()
        started.wait(timeout=5.0)
        deadline = time.perf_counter() + 5.0
        while functor.host.inflight_calls == 0 and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert functor.host.inflight_calls == 1

        partial = unload_from_runtime(simulator, EXTENSION_NAME, strict=False, native_drain_timeout=0.05)
        assert partial["complete"] is False
        assert partial["removed_functors"] == [FUNCTOR_NAME]
        assert FUNCTOR_NAME not in simulator.functors
        assert FUNCTOR_NAME in partial["receipts"]
        assert partial["native"][FUNCTOR_NAME]["released"] is False
        assert "native release incomplete" in partial["failed_functors"][FUNCTOR_NAME]
        assert partial["loader_records_cleared"] == []
        assert get_loaded_extensions()[EXTENSION_NAME] == [FUNCTOR_NAME]
        assert functor.host.state == "retired"

        worker.join(timeout=5.0)
        retry = unload_from_runtime(simulator, EXTENSION_NAME, strict=False)
        assert retry["complete"] is True
        assert retry["missing_functors"] == [FUNCTOR_NAME]
        assert retry["removed_functors"] == []
        assert retry["native"][FUNCTOR_NAME]["released"] is True
        assert retry["loader_records_cleared"] == [FUNCTOR_NAME]
        assert functor.host.state == "unloaded"
        assert EXTENSION_NAME not in get_loaded_extensions()

    def test_unload_without_simulator_releases_the_loader_owned_host(self, tmp_path: Path, relay_plugin: Path):
        root = _create_native_extension(tmp_path / EXTENSION_NAME, relay_plugin)
        loaded = load_extensions(extra_dirs=[root], validate=True, strict=True)
        functor = loaded[FUNCTOR_NAME]
        functor.initialize()
        assert functor.host.state == "loaded"

        summary = unload_extension(EXTENSION_NAME)

        assert summary["complete"] is True
        assert summary["removed_functors"] == [] and summary["missing_functors"] == []
        assert summary["receipts"] == {}
        assert summary["native"][FUNCTOR_NAME]["released"] is True
        assert summary["loader_records_cleared"] == [FUNCTOR_NAME]
        assert functor.host.state == "unloaded"
        assert EXTENSION_NAME not in get_loaded_extensions()

    def test_unload_of_a_never_initialized_host_is_skipped_not_failed(self, tmp_path: Path, relay_plugin: Path):
        root = _create_native_extension(tmp_path / EXTENSION_NAME, relay_plugin)
        loaded = load_extensions(extra_dirs=[root], validate=True, strict=True)
        assert loaded[FUNCTOR_NAME].host.state == "unloaded"

        summary = unload_extension(EXTENSION_NAME)

        assert summary["complete"] is True
        assert summary["native"][FUNCTOR_NAME]["skipped"] is True
        assert summary["failed_functors"] == {}
        assert EXTENSION_NAME not in get_loaded_extensions()

    def test_unknown_extension(self):
        """Unloading a name that was never loaded is an error, never a completed unload."""
        with pytest.raises(KeyError, match="not loaded"):
            unload_extension("no_such_extension")
        with pytest.raises(KeyError, match="not loaded"):
            unload_extension("no_such_extension", strict=True)


class _FailingNativeHost:
    loaded = True
    path = "fake_plugin.dll"
    state = "retired"

    def __init__(self, error: Exception):
        self.error = error

    def unload(self, *, drain_timeout=None):
        raise self.error


def test_native_release_records_lifecycle_failure_and_propagates_other_errors():
    """Only the documented native release failures become an incomplete receipt."""
    lifecycle_failure = SimpleNamespace(host=_FailingNativeHost(PluginLifecycleError("release refused")))
    record = loader_module._release_native_host(lifecycle_failure, strict=False, drain_timeout=None)
    assert record["released"] is False
    assert record["error"] == "PluginLifecycleError: release refused"

    library_failure = SimpleNamespace(host=_FailingNativeHost(OSError("FreeLibrary failed")))
    record = loader_module._release_native_host(library_failure, strict=False, drain_timeout=None)
    assert record["released"] is False and record["error"].startswith("OSError")

    defect = SimpleNamespace(host=_FailingNativeHost(TypeError("host defect")))
    with pytest.raises(TypeError, match="host defect"):
        loader_module._release_native_host(defect, strict=False, drain_timeout=None)


@pytest.mark.usefixtures("clean_loader_records")
class TestLoadFailurePropagation:
    def test_functor_import_failure_raises_and_keeps_constructed_records(self, tmp_path: Path):
        module = "py_ext_partial_functor"
        root = _create_python_extension(
            tmp_path / PY_EXTENSION_NAME,
            module,
            ["GoodTransport", "MissingTransport"],
            ["good_transport"],
        )

        with pytest.raises(ExtensionLoadError, match="MissingTransport") as excinfo:
            load_extensions(extra_dirs=[root], validate=False, strict=False)

        assert isinstance(excinfo.value.__cause__, AttributeError)
        assert get_loaded_extensions()[PY_EXTENSION_NAME] == ["good_transport"]
        assert unload_extension(PY_EXTENSION_NAME)["loader_records_cleared"] == ["good_transport"]

    def test_core_registry_population_failure_raises(self, tmp_path: Path, monkeypatch):
        root = _create_python_extension(
            tmp_path / PY_EXTENSION_NAME,
            "py_ext_core_registry_functor",
            ["GoodTransport"],
            ["good_transport"],
        )

        def _refuse(functor_cls):
            raise ValueError(f"core registry refused {functor_cls.__name__}")

        monkeypatch.setattr(loader_module, "_register_core_functor_class", _refuse)

        with pytest.raises(ExtensionLoadError, match="core registry refused GoodTransport") as excinfo:
            load_extensions(extra_dirs=[root], validate=False, strict=False, populate_core_registry=True)

        assert isinstance(excinfo.value.__cause__, ValueError)
        assert get_loaded_extensions()[PY_EXTENSION_NAME] == ["good_transport"]

    def test_hook_requires_an_explicit_parameter_manager(self, tmp_path: Path, monkeypatch):
        """No implicit global fallback: the caller names the manager that receives the parameters."""
        root = _create_python_extension(
            tmp_path / PY_EXTENSION_NAME,
            "py_ext_manager_functor",
            ["GoodTransport"],
            ["good_transport"],
        )
        monkeypatch.delenv("RELIC_EXTENSION_DIRS", raising=False)
        simulator = _RecordingSimulator()

        with pytest.raises(TypeError, match="parameter_manager"):
            load_and_register_extensions(simulator, extra_dirs=[root], validate=False)

        assert simulator.registered == []

    def test_hook_binds_functors_to_the_injected_parameter_manager(self, tmp_path: Path, monkeypatch):
        root = _create_python_extension(
            tmp_path / PY_EXTENSION_NAME,
            "py_ext_bound_manager_functor",
            ["GoodTransport"],
            ["good_transport"],
        )
        monkeypatch.delenv("RELIC_EXTENSION_DIRS", raising=False)
        manager = _stub_parameter_manager()
        simulator = _RecordingSimulator()

        summary = load_and_register_extensions(simulator, parameter_manager=manager, extra_dirs=[root])

        assert summary["registered_count"] == 1
        assert simulator.functors[0].parameter_manager is manager

    def test_skipped_extension_leaves_no_parameters(self, tmp_path: Path, monkeypatch):
        """Parameters are loaded only for extensions that passed validation."""
        monkeypatch.delenv("RELIC_EXTENSION_DIRS", raising=False)
        valid_root = _create_python_extension(
            tmp_path / "valid" / PY_SECOND_EXTENSION_NAME,
            "py_ext_valid_param_functor",
            ["GoodTransport"],
            ["valid_param_transport"],
            extension_name=PY_SECOND_EXTENSION_NAME,
            parameter_namespace="extensions.valid_param_ext",
        )
        invalid_root = _create_python_extension(
            tmp_path / "invalid" / PY_EXTENSION_NAME,
            "py_ext_invalid_param_functor",
            ["GoodTransport", "MissingTransport"],
            ["invalid_param_transport"],
            parameter_namespace="extensions.invalid_param_ext",
        )
        manager = UnifiedParameterManager(
            database_path=tmp_path / "parameter_db",
            preload_all=False,
            warmup_cache=False,
        )
        manager.initialize()
        simulator = _RecordingSimulator()

        summary = load_and_register_extensions(
            simulator,
            parameter_manager=manager,
            extra_dirs=[valid_root, invalid_root],
            strict=False,
        )

        assert [item["extension"] for item in summary["skipped_extensions"]] == [PY_EXTENSION_NAME]
        assert "extensions.valid_param_ext.decomposition_rate" in manager.parameters
        assert not any(
            key.startswith("extensions.invalid_param_ext") for key in manager.parameters
        )
        assert summary["parameter_source_paths"] == [str((valid_root / "parameters.yaml").resolve())]

    def test_registration_failure_propagates(self, tmp_path: Path, monkeypatch):
        root = _create_python_extension(
            tmp_path / PY_EXTENSION_NAME,
            "py_ext_registration_functor",
            ["GoodTransport"],
            ["good_transport"],
        )
        monkeypatch.delenv("RELIC_EXTENSION_DIRS", raising=False)
        simulator = _RecordingSimulator(
            refusal=ValueError("output 'good_transport_flux' already has a writer"),
        )

        with pytest.raises(ValueError, match="already has a writer"):
            load_and_register_extensions(
                simulator,
                parameter_manager=_stub_parameter_manager(),
                extra_dirs=[root],
                validate=False,
                strict=False,
            )

        assert simulator.registered == []
        assert get_loaded_extensions()[PY_EXTENSION_NAME] == ["good_transport"]


_STRICT_CONFIG_MODULE = """
import torch
from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TransportMechanismFunctor,
    VariableKind,
)


class StrictConfigTransport(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, config=None, name="strict_config_transport", **kwargs):
        if config is not None:
            raise TypeError("rate_constant must be a float, got str")
        metadata = FunctorMetadata(
            name=name,
            category=ProcessCategory.TRANSPORT_MECHANISM,
            description="functor whose declared config is rejected",
            references=["Fick (1855)."],
            input_variables=["gradient"],
            output_variables=["strict_config_flux"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        super().__init__(name=name, config={}, metadata=metadata, **kwargs)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"strict_config_flux": torch.tensor(0.0)}
"""


def _create_strict_config_extension(root: Path) -> Path:
    module = "py_ext_strict_config_functor"
    _create_python_extension(root, module, ["StrictConfigTransport"], [])
    (root / f"{module}.py").write_text(_STRICT_CONFIG_MODULE, encoding="utf-8")
    return root


def _create_invalid_extension(root: Path) -> Path:
    return _create_python_extension(
        root,
        "py_ext_invalid_functor",
        ["GoodTransport", "MissingTransport"],
        ["good_transport"],
    )


@pytest.mark.usefixtures("clean_loader_records")
class TestLoadContract:
    def test_constructor_type_error_is_not_retried_without_config(self, tmp_path: Path):
        root = _create_strict_config_extension(tmp_path / PY_EXTENSION_NAME)

        with pytest.raises(ExtensionLoadError, match="rate_constant must be a float") as excinfo:
            load_extensions(extra_dirs=[root], validate=False)

        assert isinstance(excinfo.value.__cause__, TypeError)
        assert PY_EXTENSION_NAME not in get_loaded_extensions()

    def test_load_extensions_defaults_to_strict(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("RELIC_EXTENSION_DIRS", raising=False)
        root = _create_invalid_extension(tmp_path / PY_EXTENSION_NAME)

        with pytest.raises(ExtensionValidationError, match="MissingTransport"):
            load_extensions(extra_dirs=[root])

        assert PY_EXTENSION_NAME not in get_loaded_extensions()

    def test_non_strict_load_lists_skipped_extensions(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("RELIC_EXTENSION_DIRS", raising=False)
        root = _create_invalid_extension(tmp_path / PY_EXTENSION_NAME)

        loaded = load_extensions(extra_dirs=[root], strict=False)

        assert dict(loaded) == {}
        assert [item["extension"] for item in loaded.skipped_extensions] == [PY_EXTENSION_NAME]
        assert "MissingTransport" in loaded.skipped_extensions[0]["reason"]

    def test_load_single_extension_defaults_to_strict(self, tmp_path: Path):
        root = _create_invalid_extension(tmp_path / PY_EXTENSION_NAME)

        with pytest.raises(ExtensionValidationError, match="MissingTransport"):
            load_single_extension(root)

        explicit = load_single_extension(root, strict=False)
        assert dict(explicit) == {}
        assert explicit.skipped_extensions[0]["extension"] == PY_EXTENSION_NAME

    def test_non_strict_hook_summary_lists_skipped_extensions(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("RELIC_EXTENSION_DIRS", raising=False)
        root = _create_invalid_extension(tmp_path / PY_EXTENSION_NAME)
        simulator = _RecordingSimulator()

        summary = load_and_register_extensions(
            simulator, parameter_manager=_stub_parameter_manager(), extra_dirs=[root], strict=False
        )

        assert summary["registered_count"] == 0
        assert [item["extension"] for item in summary["skipped_extensions"]] == [PY_EXTENSION_NAME]

    def test_duplicate_functor_name_across_extensions_keeps_the_first_record(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("RELIC_EXTENSION_DIRS", raising=False)
        first = _create_python_extension(
            tmp_path / "first" / PY_EXTENSION_NAME,
            "py_ext_first_functor",
            ["GoodTransport"],
            ["shared_transport"],
        )
        second = _create_python_extension(
            tmp_path / "second" / PY_SECOND_EXTENSION_NAME,
            "py_ext_second_functor",
            ["GoodTransport"],
            ["shared_transport"],
            extension_name=PY_SECOND_EXTENSION_NAME,
        )

        with pytest.raises(ExtensionLoadError, match="shared_transport"):
            load_extensions(extra_dirs=[first, second])
        assert get_loaded_extensions() == {PY_EXTENSION_NAME: ["shared_transport"]}
        assert get_extension_registry()["shared_transport"] == "py_ext_first_functor.GoodTransport"
        unload_extension(PY_EXTENSION_NAME)

        loaded = load_extensions(extra_dirs=[first, second], strict=False)
        assert set(loaded) == {"shared_transport"}
        assert get_extension_registry()["shared_transport"] == "py_ext_first_functor.GoodTransport"
        assert [item["extension"] for item in loaded.skipped_extensions] == [PY_SECOND_EXTENSION_NAME]
