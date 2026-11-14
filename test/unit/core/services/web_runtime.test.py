"""Unit tests for the web health payload builder (relic.core.services.web_runtime)."""

import pytest

from relic.core.services.web_runtime import build_health_payload


class _RecordingChecker:
    received: list = []

    def __init__(self, config):
        type(self).received.append(config)

    def check_all(self, detailed=False):
        return {"overall_status": "HEALTHY", "detailed": detailed}


def test_build_health_payload_passes_the_caller_config():
    _RecordingChecker.received.clear()

    payload = build_health_payload(_RecordingChecker, {"enable_gpu_benchmark": False}, detailed=True)

    assert payload == {"overall_status": "HEALTHY", "detailed": True}
    assert _RecordingChecker.received == [{"enable_gpu_benchmark": False}]


def test_build_health_payload_propagates_check_failures():
    class _FailingChecker:
        def __init__(self, config):
            self.config = config

        def check_all(self, detailed=False):
            raise RuntimeError("data layer health probe crashed")

    with pytest.raises(RuntimeError, match="probe crashed"):
        build_health_payload(_FailingChecker, {}, detailed=True)


def test_build_health_payload_rejects_a_non_mapping_payload():
    class _ListChecker:
        def __init__(self, config):
            self.config = config

        def check_all(self, detailed=False):
            return ["HEALTHY"]

    with pytest.raises(TypeError, match="mapping"):
        build_health_payload(_ListChecker, {}, detailed=False)


def test_interface_config_file_that_is_not_a_mapping_is_rejected(monkeypatch, tmp_path):
    import relic.parameters as parameters_package
    from relic.core.services.web_runtime import load_interface_config

    class _Manager:
        def load_yaml_file(self, path, namespace):
            return ["not", "a", "mapping"]

    monkeypatch.setattr(parameters_package, "get_global_manager", lambda: _Manager())

    with pytest.raises(TypeError, match="mapping"):
        load_interface_config(str(tmp_path / "web.yaml"), "interfaces.web.app")


@pytest.mark.parametrize("missing", ["templates", "static"])
def test_missing_packaged_web_assets_raise_naming_the_directory(tmp_path, missing):
    from relic.core.services.web_runtime import ensure_web_assets

    templates_dir = tmp_path / "templates"
    static_dir = tmp_path / "static"
    for directory in (templates_dir, static_dir):
        if directory.name != missing:
            directory.mkdir()

    with pytest.raises(FileNotFoundError, match=missing):
        ensure_web_assets(templates_dir, static_dir)

    assert not (tmp_path / missing).exists()
