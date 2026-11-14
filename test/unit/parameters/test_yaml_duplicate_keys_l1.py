"""L1 checks of the parameter database key structure.

Parameter YAML files contain no duplicate mapping keys, and no database key
is both a parameter and the namespace of another parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that fails on duplicate keys."""


def _construct_mapping(loader: UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line_no = key_node.start_mark.line + 1
            raise ValueError(f"Duplicate key '{key}' at line {line_no}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(  # type: ignore[arg-type]
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _database_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "relic" / "parameters" / "database"


def _yaml_files() -> List[Path]:
    return sorted(_database_root().rglob("*.yaml"))


def test_parameter_yaml_files_have_no_duplicate_keys() -> None:
    errors: List[str] = []
    for path in _yaml_files():
        text = path.read_text(encoding="utf-8")
        try:
            yaml.load(text, Loader=UniqueKeyLoader)
        except Exception as exc:  # noqa: BLE001
            rel_path = str(path).replace(str(Path.cwd()) + "\\", "")
            errors.append(f"{rel_path}: {exc}")

    assert not errors, "Duplicate keys or YAML parse issues detected:\n" + "\n".join(errors)


def test_no_parameter_key_is_the_namespace_of_another_key() -> None:
    """A database key names a parameter or a namespace, never both.

    The nested tree that manager.get("<namespace>") returns has one entry per
    dotted path segment. A key that is also the dot prefix of another key
    would have to be a value and a sub-tree at the same place.
    """
    from relic.parameters.database_manager import ParameterDatabase

    keys = set(ParameterDatabase(_database_root()).load_all())

    clashes: List[str] = []
    for key in sorted(keys):
        parts = key.split(".")
        for depth in range(1, len(parts)):
            prefix = ".".join(parts[:depth])
            if prefix in keys:
                clashes.append(f"{prefix} is a parameter and the namespace of {key}")

    assert not clashes, "Keys that are a parameter and a namespace:\n" + "\n".join(clashes)
