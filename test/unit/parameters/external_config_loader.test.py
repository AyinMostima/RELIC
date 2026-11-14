"""L1 tests for the external configuration loaders of the parameter manager.

Module under test: relic/parameters/ops/config_loader.py. A configuration whose
top level is not a mapping is rejected with its source named; it is not wrapped
into {"value": ...}, which callers would accept as a mapping.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from relic.parameters.ops.config_loader import load_omegaconf_config, load_yaml_file


def _manager() -> SimpleNamespace:
    return SimpleNamespace(_external_configs={})


@pytest.mark.parametrize(
    ("text", "kind"),
    [("- n_steps\n- 10\n", "list"), ("42\n", "int"), ("just text\n", "str")],
    ids=["list", "int", "str"],
)
def test_yaml_whose_top_level_is_not_a_mapping_is_rejected(tmp_path: Path, text: str, kind: str) -> None:
    path = tmp_path / "stochastic.yaml"
    path.write_text(text, encoding="utf-8")
    manager = _manager()

    with pytest.raises(TypeError, match=rf"stochastic\.yaml.*mapping.*{kind}"):
        load_yaml_file(manager, path, namespace="cli.stochastic")

    assert manager._external_configs == {}


@pytest.mark.parametrize(
    "text",
    ["pair: !!python/tuple [1, 2]\n", "handler: !!python/object:collections.OrderedDict {}\n"],
    ids=["python-tuple", "python-object"],
)
def test_yaml_with_python_tags_is_rejected_not_rewritten(tmp_path: Path, text: str) -> None:
    """A Python-specific tag is rejected by the safe loader; the content is not
    edited to drop the tag before parsing."""
    path = tmp_path / "tagged.yaml"
    path.write_text(text, encoding="utf-8")
    manager = _manager()

    with pytest.raises(ValueError, match=r"Invalid YAML in .*tagged\.yaml"):
        load_yaml_file(manager, path, namespace="cfg")

    assert manager._external_configs == {}


def test_yaml_mapping_is_returned_and_cached(tmp_path: Path) -> None:
    path = tmp_path / "plan.yaml"
    path.write_text("levels:\n  - name: site\n", encoding="utf-8")
    manager = _manager()

    data = load_yaml_file(manager, path, namespace="coupling.scales.execution_plan")

    assert data == {"levels": [{"name": "site"}]}
    assert manager._external_configs == {"coupling.scales.execution_plan": data}


def test_empty_yaml_declares_no_settings(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    manager = _manager()

    assert load_yaml_file(manager, path, namespace="cfg") == {}
    assert manager._external_configs == {"cfg": {}}


def test_omegaconf_config_whose_top_level_is_not_a_mapping_is_rejected() -> None:
    manager = _manager()

    with pytest.raises(TypeError, match=r"'runtime'.*mapping.*list"):
        load_omegaconf_config(manager, OmegaConf.create([1, 2]), namespace="runtime")

    assert manager._external_configs == {}


def test_omegaconf_namespace_is_named_by_the_caller() -> None:
    """The cache namespace has no default: every caller names its namespace."""
    import inspect

    from relic.parameters.manager import UnifiedParameterManager

    for function in (load_omegaconf_config, UnifiedParameterManager.load_omegaconf_config):
        parameter = inspect.signature(function).parameters["namespace"]
        assert parameter.default is inspect.Parameter.empty, function.__qualname__
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function.__qualname__
    with pytest.raises(TypeError):
        load_omegaconf_config(_manager(), OmegaConf.create({"a": 1}))


def test_a_namespace_never_ingested_is_the_empty_external_configuration() -> None:
    from relic.parameters.ops.config_loader import get_namespace_dict

    manager = _manager()
    assert get_namespace_dict(manager, "never_loaded") == {}
    load_omegaconf_config(manager, OmegaConf.create({"a": 1}), namespace="runtime")
    cached = get_namespace_dict(manager, "runtime")
    cached["a"] = 2
    assert manager._external_configs["runtime"] == {"a": 1}


def test_omegaconf_mapping_is_resolved_and_cached() -> None:
    manager = _manager()

    data = load_omegaconf_config(manager, OmegaConf.create({"a": 1, "b": "${a}"}), namespace="runtime")

    assert data == {"a": 1, "b": 1}
    assert manager._external_configs == {"runtime": {"a": 1, "b": 1}}


def test_loaded_and_cached_configurations_are_separate_objects(tmp_path: Path) -> None:
    """A loader and get_namespace_dict hand out copies at every depth.

    Changing the mapping a loader returned, or a nested part of what
    get_namespace_dict returned, leaves the cached configuration as it was
    loaded (the copy rule of caller_owned_value, architecture overview
    Section 6.1).
    """
    from relic.parameters.ops.config_loader import get_namespace_dict

    path = tmp_path / "operators.yaml"
    path.write_text("gradient:\n  config:\n    dx: 1.0\n", encoding="utf-8")
    manager = _manager()

    loaded = load_yaml_file(manager, path, namespace="computation.operators")
    loaded["gradient"]["config"]["dx"] = 2.0
    resolved = load_omegaconf_config(manager, OmegaConf.create({"a": {"b": [1]}}), namespace="runtime")
    resolved["a"]["b"].append(2)
    view = get_namespace_dict(manager, "computation.operators")
    view["gradient"]["config"]["dx"] = 3.0

    assert manager._external_configs == {
        "computation.operators": {"gradient": {"config": {"dx": 1.0}}},
        "runtime": {"a": {"b": [1]}},
    }
    assert get_namespace_dict(manager, "computation.operators") == {"gradient": {"config": {"dx": 1.0}}}
