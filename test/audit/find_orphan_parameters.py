from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAM_ROOT = PROJECT_ROOT / "relic" / "parameters" / "database" / "relic"
SEARCH_ROOTS = [
    PROJECT_ROOT / "relic",
    PROJECT_ROOT / "test",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "scripts",
]
OUTPUT_PATH = PROJECT_ROOT / "analysis_results" / "orphan_parameter_candidates.json"


PARAM_CALL_NAMES = {
    "get_parameter_value",
    "_require_parameter",
    "require_parameter_value",
    "require_parameter",
    "_require_param",
    "_require_scalar",
    "_require_scalar_param",
    "get",
    "set",
    "get_parameter",
}

PHYSICAL_CONST_CALLS = {"get_physical_constant", "_get_physical_constant"}


def _flatten(node: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(node, dict):
        return out
    for k, v in node.items():
        cur = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and "value" in v:
            out[cur] = v
        elif isinstance(v, dict):
            out.update(_flatten(v, cur))
    return out


def _iter_python_files() -> Iterable[Path]:
    for root in SEARCH_ROOTS:
        for p in root.rglob("*.py"):
            if p.is_file():
                yield p


def _iter_text_files() -> Iterable[Path]:
    allowed_suffixes = {
        ".py",
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".rst",
        ".toml",
        ".ini",
        ".cfg",
        ".txt",
    }
    for root in SEARCH_ROOTS:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed_suffixes:
                continue
            try:
                p.relative_to(PARAM_ROOT)
                continue
            except Exception:
                pass
            yield p


def _string_arg(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _collect_references() -> tuple[Set[str], Set[Tuple[str, str]], Set[str], Set[str], Set[str]]:
    full_key_refs: Set[str] = set()
    namespace_key_refs: Set[Tuple[str, str]] = set()
    leaf_key_refs: Set[str] = set()
    physical_constant_refs: Set[str] = set()
    prefix_refs: Set[str] = set()

    for p in _iter_python_files():
        try:
            source = p.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except Exception:
            continue

        for match in re.finditer(r"['\"]([a-zA-Z_][\w]*(?:\.[\w]+){1,})\.?['\"]", source):
            candidate = match.group(1).strip(".")
            if candidate.count(".") >= 1:
                prefix_refs.add(candidate)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            call_name = None
            if isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                call_name = node.func.id

            if call_name in PHYSICAL_CONST_CALLS:
                if node.args:
                    const_name = _string_arg(node.args[0])
                    if const_name:
                        physical_constant_refs.add(const_name)
                continue

            if call_name not in PARAM_CALL_NAMES:
                continue

            key: str | None = None
            namespace: str | None = None

            if node.args:
                key = _string_arg(node.args[0])

            for kw in node.keywords:
                if kw.arg in {"name", "key", "param_name"}:
                    candidate = _string_arg(kw.value)
                    if candidate:
                        key = candidate
                elif kw.arg == "namespace":
                    namespace = _string_arg(kw.value)

            if not key:
                continue

            normalized_key = key.strip()
            if "." in normalized_key and not normalized_key.startswith("time."):
                full_key_refs.add(normalized_key)
                leaf_key_refs.add(normalized_key.split(".")[-1])
                if namespace:
                    namespace_key_refs.add((namespace.strip(), normalized_key.split(".")[-1]))
                continue

            leaf_key_refs.add(normalized_key)
            if namespace:
                namespace_key_refs.add((namespace.strip(), normalized_key))

    dotted_pattern = re.compile(r"([a-zA-Z_][\w]*(?:\.[\w]+){1,})")
    for p in _iter_text_files():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for match in dotted_pattern.finditer(text):
            token = match.group(1).strip(".")
            if token.count(".") < 1:
                continue
            full_key_refs.add(token)
            prefix_refs.add(token)
            leaf_key_refs.add(token.split(".")[-1])

    return full_key_refs, namespace_key_refs, leaf_key_refs, physical_constant_refs, prefix_refs


def _collect_runtime_parameter_accesses() -> Set[str]:
    runtime_refs: Set[str] = set()
    try:
        from relic.parameters import get_global_manager
        from relic.core.functor import FunctorBase
        import relic.scientific as scientific
    except Exception:
        return runtime_refs

    try:
        manager = get_global_manager(preload_all=True, warmup_cache=False)
        storage = getattr(manager, "storage", None)
        if storage is not None and hasattr(storage, "_access_count"):
            storage._access_count.clear()
    except Exception:
        return runtime_refs

    try:
        scientific.register_all_functors()
    except Exception:
        pass

    registry = getattr(scientific, "FUNCTOR_REGISTRY", {})
    unique_paths = sorted({path for path in registry.values() if isinstance(path, str) and ":" in path})

    for path in unique_paths:
        module_name, class_name = path.split(":", 1)
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            if not isinstance(cls, type) or not issubclass(cls, FunctorBase):
                continue
            try:
                cls()
            except TypeError:
                cls(name=f"audit_{class_name.lower()}")
            except Exception:
                continue
        except Exception:
            continue

    try:
        access_log = storage.get_access_log() if storage is not None else {}
        runtime_refs.update(access_log.keys())
    except Exception:
        pass
    return runtime_refs


def _possible_namespace_match(full_key: str, namespace_key_refs: Set[Tuple[str, str]]) -> bool:
    if "." not in full_key:
        return False
    leaf = full_key.split(".")[-1]
    parts = full_key.split(".")
    for i in range(1, len(parts)):
        ns = ".".join(parts[:i])
        if (ns, leaf) in namespace_key_refs:
            return True
    return False


def _possible_prefix_match(full_key: str, prefix_refs: Set[str]) -> bool:
    for prefix in prefix_refs:
        if full_key == prefix or full_key.startswith(prefix + "."):
            return True
    return False


def main() -> None:
    keys: Dict[str, str] = {}
    for p in sorted(PARAM_ROOT.rglob("*.yaml")):
        rel = p.relative_to(PROJECT_ROOT).as_posix().replace("/", "\\")
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        namespace = str(data.get("namespace", "") or "").strip()
        flat = _flatten(data.get("parameters", {}))
        for key in flat:
            if namespace and not key.startswith(namespace + "."):
                full_key = f"{namespace}.{key}"
            else:
                full_key = key
            keys[full_key] = rel

    full_refs, ns_key_refs, leaf_refs, physical_refs, prefix_refs = _collect_references()
    runtime_refs = _collect_runtime_parameter_accesses()
    full_refs.update(runtime_refs)

    orphan_candidates: List[Dict[str, Any]] = []
    exact_hits = 0
    namespace_hits = 0
    leaf_hits = 0
    physical_hits = 0
    prefix_hits = 0

    for key, rel in sorted(keys.items()):
        if key in full_refs:
            exact_hits += 1
            continue

        # Physical constants are typically accessed by constant name, not full key.
        if key.startswith("relic.physical.constants.") or "\\physical\\constants.yaml" in rel:
            leaf = key.split(".")[-1]
            if leaf in physical_refs:
                physical_hits += 1
                continue

        if _possible_namespace_match(key, ns_key_refs):
            namespace_hits += 1
            continue

        if _possible_prefix_match(key, prefix_refs):
            prefix_hits += 1
            continue

        leaf = key.split(".")[-1]
        if leaf in leaf_refs:
            leaf_hits += 1
            continue

        orphan_candidates.append({"parameter_key": key, "defined_in": rel})

    report = {
        "total_parameters": len(keys),
        "exact_reference_hits": exact_hits,
        "namespace_reference_hits": namespace_hits,
        "leaf_reference_hits": leaf_hits,
        "physical_constant_hits": physical_hits,
        "prefix_reference_hits": prefix_hits,
        "runtime_reference_hits": len(runtime_refs),
        "orphan_candidates": len(orphan_candidates),
        "items": orphan_candidates,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] report={OUTPUT_PATH}")
    print(
        f"[OK] total={len(keys)} exact={exact_hits} namespace={namespace_hits} "
        f"prefix={prefix_hits} leaf={leaf_hits} physical={physical_hits} runtime={len(runtime_refs)} "
        f"orphan={len(orphan_candidates)}"
    )


if __name__ == "__main__":
    main()
