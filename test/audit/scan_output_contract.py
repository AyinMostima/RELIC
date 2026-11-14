"""Static metadata-to-forward output contract audit.

The module is import-safe. Report generation and filesystem writes occur only
through ``main`` or an explicit ``write_report`` call.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test.audit.collect_functor_inventory import (
    _extract_metadata,
    _is_abstract_class,
    _is_candidate_functor,
    scan as collect_functor_inventory,
)


ROOT = REPO_ROOT / "relic" / "scientific"
OUTPUT_PATH = REPO_ROOT / "analysis_results" / "output_contract_scan.json"


def _node_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _attribute_key(node: ast.AST) -> str | None:
    """Return a stable dotted name for a simple attribute expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _attribute_key(node.value)
        if owner is not None:
            return f"{owner}.{node.attr}"
    return None


def _static_string(node: ast.AST, environment: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return environment.get(node.id)
    if isinstance(node, ast.Subscript):
        # ``self.CONST[i]`` on a class-level tuple of string literals: the
        # indexed elements are registered as ``<dotted name>[<index>]`` when
        # the class constants are collected, so the lookup stays static.
        index = node.slice
        base = _attribute_key(node.value)
        if (
            base is not None
            and isinstance(index, ast.Constant)
            and isinstance(index.value, int)
            and not isinstance(index.value, bool)
        ):
            return environment.get(f"{base}[{index.value}]")
        return None
    if isinstance(node, ast.Attribute):
        key = _attribute_key(node)
        resolved = environment.get(key or "")
        if resolved is not None:
            return resolved
        if node.attr == "value":
            return _static_string(node.value, environment)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                continue
            if isinstance(value, ast.FormattedValue):
                resolved = _static_string(value.value, environment)
                if resolved is not None:
                    parts.append(resolved)
                    continue
            return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left, environment)
        right = _static_string(node.right, environment)
        if left is not None and right is not None:
            return left + right
    return None


def _static_string_sequence(
    node: ast.AST,
    environment: dict[str, tuple[str, ...]],
    string_environment: dict[str, str] | None = None,
) -> tuple[str, ...] | None:
    string_environment = string_environment or {}
    if isinstance(node, ast.Name):
        return environment.get(node.id)
    if isinstance(node, ast.Attribute):
        key = _attribute_key(node)
        return environment.get(key or "")
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: list[str] = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                nested = _static_string_sequence(
                    element.value,
                    environment,
                    string_environment,
                )
                if nested is None:
                    return None
                values.extend(nested)
                continue
            value = _static_string(element, string_environment)
            if value is None:
                return None
            values.append(value)
        return tuple(values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string_sequence(node.left, environment, string_environment)
        right = _static_string_sequence(node.right, environment, string_environment)
        if left is not None and right is not None:
            return left + right
    return None


def _static_string_mapping(
    node: ast.AST,
    environment: dict[str, dict[str, str]],
    string_environment: dict[str, str],
) -> dict[str, str] | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and node.args
    ):
        return _static_string_mapping(node.args[0], environment, string_environment)
    if isinstance(node, ast.Name):
        value = environment.get(node.id)
        return dict(value) if value is not None else None
    if isinstance(node, ast.Attribute):
        key = _attribute_key(node)
        value = environment.get(key or "")
        return dict(value) if value is not None else None
    if not isinstance(node, ast.Dict) or any(key is None for key in node.keys):
        return None
    mapping: dict[str, str] = {}
    for key_node, value_node in zip(node.keys, node.values):
        if key_node is None:
            return None
        key = _static_string(key_node, string_environment)
        value = _static_string(value_node, string_environment)
        if key is None or value is None:
            return None
        mapping[key] = value
    return mapping


def _static_string_value(
    node: ast.AST,
    string_environment: dict[str, str],
    mapping_environment: dict[str, dict[str, str]],
) -> str | None:
    """Resolve a static string, including finite mapping ``get`` calls."""
    resolved = _static_string(node, string_environment)
    if resolved is not None:
        return resolved
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
    ):
        return None
    mapping_key = _attribute_key(node.func.value)
    mapping = mapping_environment.get(mapping_key or "")
    lookup = _static_string(node.args[0], string_environment)
    if mapping is None or lookup is None:
        return None
    default = (
        _static_string(node.args[1], string_environment)
        if len(node.args) > 1
        else None
    )
    return mapping.get(lookup, default)


def _static_output_sequence(
    node: ast.AST,
    sequence_environment: dict[str, tuple[str, ...]],
    mapping_environment: dict[str, dict[str, str]],
    string_environment: dict[str, str],
) -> tuple[str, ...] | None:
    sequence = _static_string_sequence(
        node,
        sequence_environment,
        string_environment,
    )
    if sequence is not None:
        return sequence
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_output_sequence(
            node.left,
            sequence_environment,
            mapping_environment,
            string_environment,
        )
        right = _static_output_sequence(
            node.right,
            sequence_environment,
            mapping_environment,
            string_environment,
        )
        if left is not None and right is not None:
            return left + right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _attribute_key(node.func) == "dict.fromkeys"
        and node.args
    ):
        return _static_output_sequence(
            node.args[0],
            sequence_environment,
            mapping_environment,
            string_environment,
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "tuple", "sorted"}
        and node.args
    ):
        argument = node.args[0]
        key = _attribute_key(argument)
        mapping = mapping_environment.get(key or "")
        if mapping is not None:
            return tuple(mapping.keys())
        return _static_output_sequence(
            argument,
            sequence_environment,
            mapping_environment,
            string_environment,
        )
    return None


class _ForwardOutputCollector:
    """Conservatively resolve keys of dictionaries returned by ``forward``."""

    def __init__(
        self,
        *,
        method_resolver: Any | None = None,
        external_method_resolver: Any | None = None,
        function_resolver: Any | None = None,
        class_string_environment: dict[str, str] | None = None,
        class_sequence_environment: dict[str, tuple[str, ...]] | None = None,
        class_mapping_environment: dict[str, dict[str, str]] | None = None,
        call_stack: set[tuple[str, int]] | None = None,
        initial_dictionaries: dict[str, tuple[set[str], set[str]]] | None = None,
    ) -> None:
        self._method_resolver = method_resolver
        self._external_method_resolver = external_method_resolver
        self._function_resolver = function_resolver
        self._class_string_environment = dict(class_string_environment or {})
        self._class_sequence_environment = dict(class_sequence_environment or {})
        self._class_mapping_environment = {
            name: dict(values)
            for name, values in (class_mapping_environment or {}).items()
        }
        self._call_stack = set(call_stack or set())
        initial_dictionaries = initial_dictionaries or {}
        self._dict_keys = {
            name: set(keys)
            for name, (keys, _) in initial_dictionaries.items()
        }
        self._dict_unresolved = {
            name: set(unresolved)
            for name, (_, unresolved) in initial_dictionaries.items()
        }
        self._proven_dictionary_names = set(initial_dictionaries)
        self._returned_names: set[str] = set()
        self._direct_return_keys: set[str] = set()
        self._direct_return_unresolved: set[str] = set()
        self._returned_dictionary_tuples: list[
            tuple[tuple[set[str], set[str]], ...]
        ] = []
        self._mutated_dict_names: set[str] = set()
        self._last_call_has_dictionary_result = False

    def _dictionary_argument(
        self,
        node: ast.AST,
        sequence_environment: dict[str, tuple[str, ...]],
        string_environment: dict[str, str],
    ) -> tuple[set[str], set[str]] | None:
        dictionary_name = _attribute_key(node)
        if dictionary_name is not None and dictionary_name in self._dict_keys:
            return (
                set(self._dict_keys[dictionary_name]),
                set(self._dict_unresolved[dictionary_name]),
            )
        if isinstance(node, ast.Dict):
            return self._dict_literal(
                node,
                string_environment,
                sequence_environment,
            )
        if isinstance(node, ast.DictComp):
            return self._dict_comprehension(
                node,
                sequence_environment,
                string_environment,
            )
        if isinstance(node, ast.Call):
            return self._resolve_call(
                node,
                sequence_environment,
                string_environment,
            )
        return None

    def _resolve_call(
        self,
        call: ast.Call,
        sequence_environment: dict[str, tuple[str, ...]],
        string_environment: dict[str, str],
    ) -> tuple[set[str], set[str]]:
        self._last_call_has_dictionary_result = False
        if isinstance(call.func, ast.Name) and call.func.id == "cast" and len(call.args) >= 2:
            resolved = self._dictionary_argument(
                call.args[1],
                sequence_environment,
                string_environment,
            )
            if resolved is not None:
                self._last_call_has_dictionary_result = True
                return resolved
        if isinstance(call.func, ast.Name) and call.func.id == "dict" and call.args:
            resolved = self._dictionary_argument(
                call.args[0],
                sequence_environment,
                string_environment,
            )
            if resolved is not None:
                self._last_call_has_dictionary_result = True
                return resolved

        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "finalize_transport_updates"
        ):
            keyword_values = {
                keyword.arg: keyword.value
                for keyword in call.keywords
                if keyword.arg is not None
            }
            outputs_node = keyword_values.get("outputs")
            if outputs_node is None and call.args:
                outputs_node = call.args[0]
            mapping_node = keyword_values.get("mapping")
            drop_node = keyword_values.get("drop_diagnostics")
            incoming = (
                self._dictionary_argument(
                    outputs_node,
                    sequence_environment,
                    string_environment,
                )
                if outputs_node is not None
                else None
            )
            mapping = (
                _static_string_mapping(mapping_node, {}, string_environment)
                if mapping_node is not None
                else None
            )
            if incoming is not None and mapping is not None:
                incoming_keys, unresolved = incoming
                drop_diagnostics = (
                    isinstance(drop_node, ast.Constant)
                    and drop_node.value is True
                )
                output_keys: set[str] = set()
                for key in incoming_keys:
                    if "flux" in key.lower():
                        target = mapping.get(key)
                        if target is None:
                            unresolved.add(f"unmapped transport flux: {key}")
                        else:
                            output_keys.add(target)
                    elif not drop_diagnostics:
                        output_keys.add(key)
                self._last_call_has_dictionary_result = True
                return output_keys, unresolved

        if self._function_resolver is not None and isinstance(call.func, ast.Name):
            function = self._function_resolver(call.func.id)
            if function is not None:
                marker = (call.func.id, function.lineno)
                if marker in self._call_stack:
                    return set(), {f"recursive helper: {call.func.id}"}
                parameters = [argument.arg for argument in function.args.args]
                initial: dict[str, tuple[set[str], set[str]]] = {}
                for parameter, argument in zip(parameters, call.args):
                    resolved = self._dictionary_argument(
                        argument,
                        sequence_environment,
                        string_environment,
                    )
                    if resolved is not None:
                        initial[parameter] = resolved
                for keyword in call.keywords:
                    if keyword.arg is None:
                        continue
                    resolved = self._dictionary_argument(
                        keyword.value,
                        sequence_environment,
                        string_environment,
                    )
                    if resolved is not None:
                        initial[keyword.arg] = resolved
                nested = _ForwardOutputCollector(
                    method_resolver=self._method_resolver,
                    external_method_resolver=self._external_method_resolver,
                    function_resolver=self._function_resolver,
                    class_string_environment=self._class_string_environment,
                    class_sequence_environment=self._class_sequence_environment,
                    class_mapping_environment=self._class_mapping_environment,
                    call_stack=self._call_stack | {marker},
                    initial_dictionaries=initial,
                )
                keys, unresolved = nested.collect(function)
                mutated_parameters = set(initial) & nested._mutated_dict_names
                for parameter in mutated_parameters:
                    keys.update(nested._dict_keys.get(parameter, set()))
                    unresolved.update(
                        nested._dict_unresolved.get(parameter, set())
                    )
                self._last_call_has_dictionary_result = bool(keys) or bool(
                    mutated_parameters
                )
                return keys, unresolved

        if (
            self._method_resolver is not None
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
        ):
            method = self._method_resolver(call.func.attr)
            if method is not None:
                marker = (call.func.attr, method.lineno)
                if marker in self._call_stack:
                    return set(), {f"recursive helper: {call.func.attr}"}
                parameters = [
                    argument.arg
                    for argument in method.args.args
                    if argument.arg != "self"
                ]
                initial: dict[str, tuple[set[str], set[str]]] = {
                    name: (
                        set(self._dict_keys[name]),
                        set(self._dict_unresolved[name]),
                    )
                    for name in self._dict_keys
                    if (
                        name.startswith("self.")
                        and name in self._proven_dictionary_names
                    )
                }
                for parameter, argument in zip(parameters, call.args):
                    resolved = self._dictionary_argument(
                        argument,
                        sequence_environment,
                        string_environment,
                    )
                    if resolved is not None:
                        initial[parameter] = resolved
                for keyword in call.keywords:
                    if keyword.arg is None:
                        continue
                    resolved = self._dictionary_argument(
                        keyword.value,
                        sequence_environment,
                        string_environment,
                    )
                    if resolved is not None:
                        initial[keyword.arg] = resolved
                nested = _ForwardOutputCollector(
                    method_resolver=self._method_resolver,
                    external_method_resolver=self._external_method_resolver,
                    function_resolver=self._function_resolver,
                    class_string_environment=self._class_string_environment,
                    class_sequence_environment=self._class_sequence_environment,
                    class_mapping_environment=self._class_mapping_environment,
                    call_stack=self._call_stack | {marker},
                    initial_dictionaries=initial,
                )
                keys, unresolved = nested.collect(method)
                mutated_parameters = (
                    set(initial) & nested._mutated_dict_names
                ) | {
                    name
                    for name in nested._mutated_dict_names
                    if (
                        name.startswith("self.")
                        and name in nested._proven_dictionary_names
                    )
                }
                for parameter in mutated_parameters:
                    keys.update(nested._dict_keys.get(parameter, set()))
                    unresolved.update(
                        nested._dict_unresolved.get(parameter, set())
                    )
                    if parameter.startswith("self."):
                        self._dict_keys[parameter] = set(
                            nested._dict_keys.get(parameter, set())
                        )
                        self._dict_unresolved[parameter] = set(
                            nested._dict_unresolved.get(parameter, set())
                        )
                        self._proven_dictionary_names.add(parameter)
                self._last_call_has_dictionary_result = bool(keys) or bool(
                    mutated_parameters
                )
                return keys, unresolved

        if (
            self._external_method_resolver is not None
            and isinstance(call.func, ast.Attribute)
        ):
            method = self._external_method_resolver(call.func.attr)
            if method is not None:
                marker = (f"external:{call.func.attr}", method.lineno)
                if marker in self._call_stack:
                    return set(), {f"recursive helper: {call.func.attr}"}
                parameters = [
                    argument.arg
                    for argument in method.args.args
                    if argument.arg not in {"self", "cls"}
                ]
                initial: dict[str, tuple[set[str], set[str]]] = {}
                for parameter, argument in zip(parameters, call.args):
                    resolved = self._dictionary_argument(
                        argument,
                        sequence_environment,
                        string_environment,
                    )
                    if resolved is not None:
                        initial[parameter] = resolved
                for keyword in call.keywords:
                    if keyword.arg is None:
                        continue
                    resolved = self._dictionary_argument(
                        keyword.value,
                        sequence_environment,
                        string_environment,
                    )
                    if resolved is not None:
                        initial[keyword.arg] = resolved
                incoming_key_sets = [
                    set(incoming)
                    for incoming, _ in initial.values()
                    if incoming
                ]
                if incoming_key_sets:
                    nested = _ForwardOutputCollector(
                        method_resolver=self._method_resolver,
                        external_method_resolver=self._external_method_resolver,
                        function_resolver=self._function_resolver,
                        class_string_environment=self._class_string_environment,
                        class_sequence_environment=self._class_sequence_environment,
                        class_mapping_environment=self._class_mapping_environment,
                        call_stack=self._call_stack | {marker},
                        initial_dictionaries=initial,
                    )
                    keys, unresolved = nested.collect(method)
                    if any(keys == incoming for incoming in incoming_key_sets) and not unresolved:
                        self._last_call_has_dictionary_result = True
                        return keys, set()

        return set(), {_node_text(call)}

    def _resolve_dictionary_tuple_call(
        self,
        call: ast.Call,
        sequence_environment: dict[str, tuple[str, ...]],
        string_environment: dict[str, str],
    ) -> list[tuple[set[str], set[str]]] | None:
        """Resolve a helper call that returns a finite tuple of dictionaries."""
        function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        marker_name: str | None = None
        include_instance_dictionaries = False

        if isinstance(call.func, ast.Name) and self._function_resolver is not None:
            function = self._function_resolver(call.func.id)
            marker_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            if (
                isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
                and self._method_resolver is not None
            ):
                function = self._method_resolver(call.func.attr)
                marker_name = call.func.attr
                include_instance_dictionaries = True
            elif self._external_method_resolver is not None:
                function = self._external_method_resolver(call.func.attr)
                marker_name = f"external:{call.func.attr}"

        if function is None or marker_name is None:
            return None
        marker = (marker_name, function.lineno)
        if marker in self._call_stack:
            return None

        parameters = [
            argument.arg
            for argument in function.args.args
            if argument.arg not in {"self", "cls"}
        ]
        initial: dict[str, tuple[set[str], set[str]]] = {}
        if include_instance_dictionaries:
            initial.update(
                {
                    name: (
                        set(self._dict_keys[name]),
                        set(self._dict_unresolved[name]),
                    )
                    for name in self._dict_keys
                    if (
                        name.startswith("self.")
                        and name in self._proven_dictionary_names
                    )
                }
            )
        for parameter, argument in zip(parameters, call.args):
            resolved = self._dictionary_argument(
                argument,
                sequence_environment,
                string_environment,
            )
            if resolved is not None:
                initial[parameter] = resolved
        for keyword in call.keywords:
            if keyword.arg is None:
                continue
            resolved = self._dictionary_argument(
                keyword.value,
                sequence_environment,
                string_environment,
            )
            if resolved is not None:
                initial[keyword.arg] = resolved

        nested = _ForwardOutputCollector(
            method_resolver=self._method_resolver,
            external_method_resolver=self._external_method_resolver,
            function_resolver=self._function_resolver,
            class_string_environment=self._class_string_environment,
            class_sequence_environment=self._class_sequence_environment,
            class_mapping_environment=self._class_mapping_environment,
            call_stack=self._call_stack | {marker},
            initial_dictionaries=initial,
        )
        nested.collect(function)
        if include_instance_dictionaries:
            for name in nested._mutated_dict_names:
                if (
                    not name.startswith("self.")
                    or name not in nested._proven_dictionary_names
                ):
                    continue
                self._dict_keys[name] = set(nested._dict_keys.get(name, set()))
                self._dict_unresolved[name] = set(
                    nested._dict_unresolved.get(name, set())
                )
                self._proven_dictionary_names.add(name)
        returned = nested._returned_dictionary_tuples
        if not returned:
            return None
        if any(unresolved for result in returned for _, unresolved in result):
            return None
        arity = len(returned[0])
        if any(len(result) != arity for result in returned):
            return None
        merged = [(set(), set()) for _ in range(arity)]
        for result in returned:
            for index, (keys, unresolved) in enumerate(result):
                merged[index][0].update(keys)
                merged[index][1].update(unresolved)
        return merged

    @staticmethod
    def _target_name(target: ast.AST) -> str | None:
        return _attribute_key(target)

    @staticmethod
    def _subscript_base_name(target: ast.Subscript) -> str | None:
        return _attribute_key(target.value)

    def _ensure_dictionary(self, name: str) -> None:
        self._dict_keys.setdefault(name, set())
        self._dict_unresolved.setdefault(name, set())
        self._proven_dictionary_names.add(name)

    @staticmethod
    def _is_external_intent(value: ast.expr) -> bool:
        """An ExternalIntent value is a delivery request, not a scientific output.

        The runtime refuses an intent that is also declared as an output
        (``FunctorBase`` partitioning), so such keys are excluded from the
        output contract comparison instead of being reported as undeclared.
        """
        if not isinstance(value, ast.Call):
            return False
        callee = value.func
        name = callee.id if isinstance(callee, ast.Name) else (
            callee.attr if isinstance(callee, ast.Attribute) else None
        )
        return name == "ExternalIntent"

    def _dict_literal(
        self,
        node: ast.Dict,
        string_environment: dict[str, str],
        sequence_environment: dict[str, tuple[str, ...]] | None = None,
    ) -> tuple[set[str], set[str]]:
        keys: set[str] = set()
        unresolved: set[str] = set()
        sequence_environment = sequence_environment or {}
        for key, value in zip(node.keys, node.values):
            if key is not None and self._is_external_intent(value):
                continue
            if key is None:
                resolved = self._dictionary_argument(
                    value,
                    sequence_environment,
                    string_environment,
                )
                if resolved is None:
                    unresolved.add(f"dictionary unpacking: {_node_text(value)}")
                else:
                    unpacked_keys, unpacked_unresolved = resolved
                    keys.update(unpacked_keys)
                    unresolved.update(unpacked_unresolved)
                continue
            resolved = _static_string(key, string_environment)
            if resolved is None:
                unresolved.add(_node_text(key))
            else:
                keys.add(resolved)
        return keys, unresolved

    def _dict_comprehension(
        self,
        node: ast.DictComp,
        sequence_environment: dict[str, tuple[str, ...]],
        string_environment: dict[str, str],
    ) -> tuple[set[str], set[str]]:
        if len(node.generators) != 1:
            return set(), {_node_text(node)}
        generator = node.generators[0]
        bindings = self._loop_bindings(
            generator.target,
            generator.iter,
            sequence_environment,
            self._class_mapping_environment,
            string_environment,
        )
        if bindings is None:
            return set(), {_node_text(node)}
        keys: set[str] = set()
        unresolved: set[str] = set()
        for binding in bindings:
            nested_strings = dict(string_environment)
            nested_strings.update(binding)
            condition_values = [
                self._static_condition(
                    condition,
                    sequence_environment,
                    nested_strings,
                )
                for condition in generator.ifs
            ]
            if any(value is False for value in condition_values):
                continue
            for condition, value in zip(generator.ifs, condition_values):
                if value is None:
                    unresolved.add(f"comprehension condition: {_node_text(condition)}")
            key = _static_string(node.key, nested_strings)
            if key is None:
                unresolved.add(_node_text(node.key))
            else:
                keys.add(key)
        return keys, unresolved

    def _static_condition(
        self,
        node: ast.AST,
        sequence_environment: dict[str, tuple[str, ...]],
        string_environment: dict[str, str],
    ) -> bool | None:
        """Resolve finite string membership filters used by output builders."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self._static_condition(
                node.operand,
                sequence_environment,
                string_environment,
            )
            return None if value is None else not value
        if isinstance(node, ast.BoolOp):
            values = [
                self._static_condition(
                    value,
                    sequence_environment,
                    string_environment,
                )
                for value in node.values
            ]
            if isinstance(node.op, ast.And):
                if any(value is False for value in values):
                    return False
                return True if all(value is True for value in values) else None
            if isinstance(node.op, ast.Or):
                if any(value is True for value in values):
                    return True
                return False if all(value is False for value in values) else None
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
            return None
        left = _static_string(node.left, string_environment)
        if left is None:
            return None
        comparator = node.comparators[0]
        right_string = _static_string(comparator, string_environment)
        operator = node.ops[0]
        if isinstance(operator, (ast.Eq, ast.NotEq)) and right_string is not None:
            equal = left == right_string
            return equal if isinstance(operator, ast.Eq) else not equal
        if isinstance(operator, (ast.In, ast.NotIn)) and right_string is not None:
            contained = left in right_string
            return contained if isinstance(operator, ast.In) else not contained
        right_sequence = _static_string_sequence(
            comparator,
            sequence_environment,
            string_environment,
        )
        if right_sequence is None and isinstance(comparator, ast.Name):
            dictionary_keys = self._dict_keys.get(comparator.id)
            if dictionary_keys is not None:
                right_sequence = tuple(dictionary_keys)
        if right_sequence is None:
            return None
        contained = left in right_sequence
        if isinstance(operator, ast.In):
            return contained
        if isinstance(operator, ast.NotIn):
            return not contained
        return None

    def _assign_dictionary(
        self,
        name: str,
        value: ast.AST,
        sequence_environment: dict[str, tuple[str, ...]],
        string_environment: dict[str, str],
    ) -> None:
        if isinstance(value, ast.Dict):
            keys, unresolved = self._dict_literal(
                value,
                string_environment,
                sequence_environment,
            )
            self._dict_keys[name] = keys
            self._dict_unresolved[name] = unresolved
            self._proven_dictionary_names.add(name)
            return
        source = _attribute_key(value)
        if source is not None and source in self._proven_dictionary_names:
            self._dict_keys[name] = set(self._dict_keys[source])
            self._dict_unresolved[name] = set(self._dict_unresolved[source])
            self._proven_dictionary_names.add(name)
            return
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "copy"
        ):
            source = _attribute_key(value.func.value)
            if source is not None and source in self._proven_dictionary_names:
                self._dict_keys[name] = set(self._dict_keys[source])
                self._dict_unresolved[name] = set(self._dict_unresolved[source])
                self._proven_dictionary_names.add(name)
                return
        if isinstance(value, ast.Call):
            keys, unresolved = self._resolve_call(
                value,
                sequence_environment,
                string_environment,
            )
            self._dict_keys[name] = keys
            self._dict_unresolved[name] = unresolved
            if self._last_call_has_dictionary_result:
                self._proven_dictionary_names.add(name)
            return
        if isinstance(value, ast.DictComp):
            keys, unresolved = self._dict_comprehension(
                value,
                sequence_environment,
                string_environment,
            )
            self._dict_keys[name] = keys
            self._dict_unresolved[name] = unresolved
            self._proven_dictionary_names.add(name)

    def _update_dictionary(
        self,
        name: str,
        value: ast.AST,
        sequence_environment: dict[str, tuple[str, ...]],
        string_environment: dict[str, str],
    ) -> None:
        self._ensure_dictionary(name)
        self._mutated_dict_names.add(name)
        if isinstance(value, ast.Dict):
            keys, unresolved = self._dict_literal(
                value,
                string_environment,
                sequence_environment,
            )
            self._dict_keys[name].update(keys)
            self._dict_unresolved[name].update(unresolved)
            return
        resolved = self._dictionary_argument(
            value,
            sequence_environment,
            string_environment,
        )
        if resolved is not None:
            keys, unresolved = resolved
            self._dict_keys[name].update(keys)
            self._dict_unresolved[name].update(unresolved)
            return
        self._dict_unresolved[name].add(f"update({_node_text(value)})")

    def _loop_bindings(
        self,
        target: ast.AST,
        iterator: ast.AST,
        sequence_environment: dict[str, tuple[str, ...]],
        mapping_environment: dict[str, dict[str, str]],
        string_environment: dict[str, str],
    ) -> list[dict[str, str]] | None:
        target_names = [
            node.id
            for node in (
                target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
            )
            if isinstance(node, ast.Name)
        ]
        if not target_names:
            return None

        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id in {"list", "tuple", "sorted"}
            and iterator.args
        ):
            iterator = iterator.args[0]

        sequence = _static_string_sequence(
            iterator,
            sequence_environment,
            string_environment,
        )
        if sequence is not None and len(target_names) == 1:
            return [{target_names[0]: value} for value in sequence]

        # ``for key, value, unit in ((KEY, tensor, "unit"), ...)``: a literal
        # tuple of literal tuples binds each positional target to the
        # statically resolvable string elements of every row.
        if (
            isinstance(iterator, (ast.Tuple, ast.List))
            and iterator.elts
            and isinstance(target, (ast.Tuple, ast.List))
            and all(isinstance(row, (ast.Tuple, ast.List)) for row in iterator.elts)
        ):
            positional_names = [
                node.id if isinstance(node, ast.Name) else None
                for node in target.elts
            ]
            row_bindings: list[dict[str, str]] = []
            for row in iterator.elts:
                binding: dict[str, str] = {}
                for name, element in zip(positional_names, row.elts):
                    if name is None:
                        continue
                    resolved = _static_string(element, string_environment)
                    if resolved is not None:
                        binding[name] = resolved
                if not binding:
                    return None
                row_bindings.append(binding)
            return row_bindings

        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Attribute)
            and iterator.func.attr == "items"
        ):
            owner_key = _attribute_key(iterator.func.value)
            mapping = mapping_environment.get(owner_key or "")
            if mapping is not None and len(target_names) >= 2:
                return [
                    {target_names[0]: key, target_names[1]: value}
                    for key, value in mapping.items()
                ]
            if (
                isinstance(iterator.func.value, ast.Name)
                and iterator.func.value.id in self._dict_keys
                and len(target_names) >= 1
            ):
                return [
                    {target_names[0]: key}
                    for key in sorted(self._dict_keys[iterator.func.value.id])
                ]

        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id in {"zip", "enumerate"}
        ):
            arguments = (
                iterator.args
                if iterator.func.id == "zip"
                else iterator.args[:1]
            )
            known: list[tuple[int, tuple[str, ...]]] = []
            for index, argument in enumerate(arguments):
                values = _static_string_sequence(
                    argument,
                    sequence_environment,
                    string_environment,
                )
                if values is not None:
                    target_index = index if iterator.func.id == "zip" else 1
                    known.append((target_index, values))
            if known:
                count = max(len(values) for _, values in known)
                bindings: list[dict[str, str]] = []
                for index in range(count):
                    binding: dict[str, str] = {}
                    for target_index, values in known:
                        if target_index < len(target_names) and index < len(values):
                            binding[target_names[target_index]] = values[index]
                    bindings.append(binding)
                return bindings
        return None

    def _visit_block(
        self,
        statements: list[ast.stmt],
        sequence_environment: dict[str, tuple[str, ...]],
        mapping_environment: dict[str, dict[str, str]],
        string_environment: dict[str, str],
    ) -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue

            if isinstance(statement, ast.Assign):
                if (
                    len(statement.targets) == 1
                    and isinstance(statement.targets[0], (ast.Tuple, ast.List))
                    and isinstance(statement.value, ast.Call)
                ):
                    tuple_result = self._resolve_dictionary_tuple_call(
                        statement.value,
                        sequence_environment,
                        string_environment,
                    )
                    target_elements = statement.targets[0].elts
                    if tuple_result is not None and len(tuple_result) == len(
                        target_elements
                    ):
                        for target, (keys, unresolved) in zip(
                            target_elements,
                            tuple_result,
                        ):
                            target_name = self._target_name(target)
                            if target_name is None:
                                continue
                            self._dict_keys[target_name] = set(keys)
                            self._dict_unresolved[target_name] = set(unresolved)
                            self._proven_dictionary_names.add(target_name)
                        continue
                sequence = _static_output_sequence(
                    statement.value,
                    sequence_environment,
                    mapping_environment,
                    string_environment,
                )
                string_value = _static_string_value(
                    statement.value,
                    string_environment,
                    mapping_environment,
                )
                mapping_value = _static_string_mapping(
                    statement.value,
                    mapping_environment,
                    string_environment,
                )
                for target in statement.targets:
                    target_name = self._target_name(target)
                    if target_name is not None:
                        if sequence is not None:
                            sequence_environment[target_name] = sequence
                        if string_value is not None:
                            string_environment[target_name] = string_value
                        if mapping_value is not None:
                            mapping_environment[target_name] = mapping_value
                        self._assign_dictionary(
                            target_name,
                            statement.value,
                            sequence_environment,
                            string_environment,
                        )
                        if (
                            target_name.startswith("self.")
                            and target_name in self._proven_dictionary_names
                        ):
                            self._mutated_dict_names.add(target_name)
                    if isinstance(target, ast.Subscript):
                        base_name = self._subscript_base_name(target)
                        if base_name is not None and self._is_external_intent(statement.value):
                            continue
                        if base_name is not None:
                            self._ensure_dictionary(base_name)
                            self._mutated_dict_names.add(base_name)
                            key = _static_string(target.slice, string_environment)
                            if key is None:
                                self._dict_unresolved[base_name].add(
                                    _node_text(target.slice)
                                )
                            else:
                                self._dict_keys[base_name].add(key)
                continue

            if isinstance(statement, ast.AnnAssign):
                target_name = self._target_name(statement.target)
                if target_name is not None and statement.value is not None:
                    sequence = _static_output_sequence(
                        statement.value,
                        sequence_environment,
                        mapping_environment,
                        string_environment,
                    )
                    if sequence is not None:
                        sequence_environment[target_name] = sequence
                    mapping_value = _static_string_mapping(
                        statement.value,
                        mapping_environment,
                        string_environment,
                    )
                    if mapping_value is not None:
                        mapping_environment[target_name] = mapping_value
                    self._assign_dictionary(
                        target_name,
                        statement.value,
                        sequence_environment,
                        string_environment,
                    )
                    if (
                        target_name.startswith("self.")
                        and target_name in self._proven_dictionary_names
                    ):
                        self._mutated_dict_names.add(target_name)
                continue

            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "update"
                    and isinstance(call.func.value, ast.Name)
                    and call.args
                ):
                    self._update_dictionary(
                        call.func.value.id,
                        call.args[0],
                        sequence_environment,
                        string_environment,
                    )
                else:
                    keys, unresolved = self._resolve_call(
                        call,
                        sequence_environment,
                        string_environment,
                    )
                    if not self._last_call_has_dictionary_result:
                        continue
                    argument_nodes = list(call.args) + [
                        keyword.value
                        for keyword in call.keywords
                        if keyword.arg is not None
                    ]
                    for argument in argument_nodes:
                        if not isinstance(argument, ast.Name):
                            continue
                        if argument.id not in self._dict_keys:
                            continue
                        self._dict_keys[argument.id].update(keys)
                        self._dict_unresolved[argument.id].update(unresolved)
                continue

            if isinstance(statement, (ast.For, ast.AsyncFor)):
                bindings = self._loop_bindings(
                    statement.target,
                    statement.iter,
                    sequence_environment,
                    mapping_environment,
                    string_environment,
                )
                if bindings is not None:
                    for binding in bindings:
                        nested_strings = dict(string_environment)
                        nested_strings.update(binding)
                        self._visit_block(
                            statement.body,
                            dict(sequence_environment),
                            dict(mapping_environment),
                            nested_strings,
                        )
                else:
                    self._visit_block(
                        statement.body,
                        dict(sequence_environment),
                        dict(mapping_environment),
                        dict(string_environment),
                    )
                self._visit_block(
                    statement.orelse,
                    dict(sequence_environment),
                    dict(mapping_environment),
                    dict(string_environment),
                )
                continue

            if isinstance(statement, ast.If):
                self._visit_block(
                    statement.body,
                    dict(sequence_environment),
                    dict(mapping_environment),
                    dict(string_environment),
                )
                self._visit_block(
                    statement.orelse,
                    dict(sequence_environment),
                    dict(mapping_environment),
                    dict(string_environment),
                )
                continue

            if isinstance(statement, ast.Try):
                self._visit_block(
                    statement.body,
                    dict(sequence_environment),
                    dict(mapping_environment),
                    dict(string_environment),
                )
                for handler in statement.handlers:
                    self._visit_block(
                        handler.body,
                        dict(sequence_environment),
                        dict(mapping_environment),
                        dict(string_environment),
                    )
                self._visit_block(
                    statement.orelse,
                    dict(sequence_environment),
                    dict(mapping_environment),
                    dict(string_environment),
                )
                self._visit_block(
                    statement.finalbody,
                    dict(sequence_environment),
                    dict(mapping_environment),
                    dict(string_environment),
                )
                continue

            if isinstance(statement, (ast.With, ast.AsyncWith)):
                self._visit_block(
                    statement.body,
                    dict(sequence_environment),
                    dict(mapping_environment),
                    dict(string_environment),
                )
                continue

            if isinstance(statement, ast.Return):
                if isinstance(statement.value, ast.Name):
                    self._returned_names.add(statement.value.id)
                elif isinstance(statement.value, (ast.Tuple, ast.List)):
                    returned: list[tuple[set[str], set[str]]] = []
                    for element in statement.value.elts:
                        resolved = self._dictionary_argument(
                            element,
                            sequence_environment,
                            string_environment,
                        )
                        if resolved is None:
                            returned = []
                            break
                        returned.append((set(resolved[0]), set(resolved[1])))
                    if returned:
                        self._returned_dictionary_tuples.append(tuple(returned))
                    else:
                        self._direct_return_unresolved.add(
                            f"return {_node_text(statement.value)}"
                        )
                elif isinstance(statement.value, ast.Dict):
                    keys, unresolved = self._dict_literal(
                        statement.value,
                        string_environment,
                        sequence_environment,
                    )
                    self._direct_return_keys.update(keys)
                    self._direct_return_unresolved.update(unresolved)
                elif isinstance(statement.value, ast.DictComp):
                    keys, unresolved = self._dict_comprehension(
                        statement.value,
                        sequence_environment,
                        string_environment,
                    )
                    self._direct_return_keys.update(keys)
                    self._direct_return_unresolved.update(unresolved)
                elif isinstance(statement.value, ast.Call):
                    keys, unresolved = self._resolve_call(
                        statement.value,
                        sequence_environment,
                        string_environment,
                    )
                    self._direct_return_keys.update(keys)
                    self._direct_return_unresolved.update(unresolved)
                elif statement.value is not None:
                    self._direct_return_unresolved.add(
                        f"return {_node_text(statement.value)}"
                    )

    def collect(self, function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[set[str], set[str]]:
        self._visit_block(
            function.body,
            dict(self._class_sequence_environment),
            {
                name: dict(values)
                for name, values in self._class_mapping_environment.items()
            },
            dict(self._class_string_environment),
        )
        keys = set(self._direct_return_keys)
        unresolved = set(self._direct_return_unresolved)
        for name in self._returned_names:
            keys.update(self._dict_keys.get(name, set()))
            unresolved.update(self._dict_unresolved.get(name, set()))
            if name not in self._dict_keys:
                unresolved.add(f"returned dictionary name: {name}")
        return keys, unresolved


ClassEntry = tuple[Path, ast.Module, ast.ClassDef]
ClassIndex = dict[str, list[ClassEntry]]
MethodEntry = tuple[Path, ast.Module, ast.ClassDef, ast.FunctionDef | ast.AsyncFunctionDef]
FunctionEntry = tuple[Path, ast.Module, ast.FunctionDef | ast.AsyncFunctionDef]
FunctionIndex = dict[str, list[FunctionEntry]]


def _base_class_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _find_forward_owner(
    cls: ast.ClassDef,
    path: Path,
    tree: ast.Module,
    class_index: ClassIndex,
    visited: set[tuple[str, str]] | None = None,
) -> ClassEntry | None:
    """Resolve a local or inherited forward implementation conservatively."""
    method_entry = _find_method_owner(
        cls,
        path,
        tree,
        class_index,
        "forward",
        visited,
    )
    if method_entry is None:
        return None
    owner_path, owner_tree, owner_class, _ = method_entry
    return owner_path, owner_tree, owner_class


def _find_method_owner(
    cls: ast.ClassDef,
    path: Path,
    tree: ast.Module,
    class_index: ClassIndex,
    method_name: str,
    visited: set[tuple[str, str]] | None = None,
) -> MethodEntry | None:
    """Resolve a method using the concrete class MRO approximation."""
    marker = (str(path.resolve()), cls.name)
    visited = set(visited or set())
    if marker in visited:
        return None
    visited.add(marker)

    local_method = next(
        (
            node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        ),
        None,
    )
    if local_method is not None:
        return path, tree, cls, local_method

    local_classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    for base in cls.bases:
        base_name = _base_class_name(base)
        if not base_name:
            continue
        local_base = local_classes.get(base_name)
        if local_base is not None:
            resolved = _find_method_owner(
                local_base,
                path,
                tree,
                class_index,
                method_name,
                visited,
            )
            if resolved is not None:
                return resolved
            continue
        candidates = class_index.get(base_name, [])
        if len(candidates) != 1:
            continue
        base_path, base_tree, base_class = candidates[0]
        resolved = _find_method_owner(
            base_class,
            base_path,
            base_tree,
            class_index,
            method_name,
            visited,
        )
        if resolved is not None:
            return resolved
    return None


def _module_constant_environments(
    tree: ast.Module,
) -> tuple[
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, dict[str, str]],
]:
    """Collect finite module constants and string-valued Enum members."""
    strings: dict[str, str] = {}
    sequences: dict[str, tuple[str, ...]] = {}
    mappings: dict[str, dict[str, str]] = {}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_base_class_name(base) == "Enum" for base in node.bases):
            continue
        for statement in node.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            value = statement.value
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                resolved = _static_string(value, strings)
                if resolved is not None:
                    strings[f"{node.name}.{target.id}"] = resolved

    assignments = [
        statement
        for statement in tree.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
    ]
    for _ in range(2):
        for statement in assignments:
            value = statement.value
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                string_value = _static_string(value, strings)
                if string_value is not None:
                    strings[target.id] = string_value
                sequence_value = _static_string_sequence(
                    value,
                    sequences,
                    strings,
                )
                if sequence_value is not None:
                    sequences[target.id] = sequence_value
                mapping_value = _static_string_mapping(value, mappings, strings)
                if mapping_value is not None:
                    mappings[target.id] = mapping_value
    return strings, sequences, mappings


def _class_constant_environments(
    cls: ast.ClassDef,
    path: Path,
    tree: ast.Module,
    class_index: ClassIndex,
    visited: set[tuple[str, str]] | None = None,
) -> tuple[
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, dict[str, str]],
]:
    """Collect finite string constants from base classes and the concrete class."""
    marker = (str(path.resolve()), cls.name)
    visited = set(visited or set())
    if marker in visited:
        return {}, {}, {}
    visited.add(marker)

    strings, sequences, mappings = _module_constant_environments(tree)
    local_classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    for base in cls.bases:
        base_name = _base_class_name(base)
        if not base_name:
            continue
        base_entry: ClassEntry | None = None
        local_base = local_classes.get(base_name)
        if local_base is not None:
            base_entry = (path, tree, local_base)
        else:
            candidates = class_index.get(base_name, [])
            if len(candidates) == 1:
                base_entry = candidates[0]
        if base_entry is None:
            continue
        base_strings, base_sequences, base_mappings = _class_constant_environments(
            base_entry[2],
            base_entry[0],
            base_entry[1],
            class_index,
            visited,
        )
        strings.update(base_strings)
        sequences.update(base_sequences)
        mappings.update(base_mappings)

    for statement in cls.body:
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        if value is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            key = f"self.{target.id}"
            class_key = f"cls.{target.id}"
            string_value = _static_string(value, strings)
            if string_value is not None:
                strings[key] = string_value
                strings[class_key] = string_value
                strings[target.id] = string_value
            sequence_value = _static_string_sequence(value, sequences, strings)
            if sequence_value is not None:
                sequences[key] = sequence_value
                sequences[class_key] = sequence_value
                sequences[target.id] = sequence_value
                # Indexed access to a declared constant tuple is static too.
                for position, element in enumerate(sequence_value):
                    strings[f"{key}[{position}]"] = element
                    strings[f"{class_key}[{position}]"] = element
                    strings[f"{target.id}[{position}]"] = element
            mapping_value = _static_string_mapping(value, mappings, strings)
            if mapping_value is not None:
                mappings[key] = mapping_value
                mappings[class_key] = mapping_value
                mappings[target.id] = mapping_value

    methods = {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def validated_name(node: ast.AST) -> str | None:
        candidate = node
        if (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "tuple"
            and len(candidate.args) == 1
        ):
            candidate = candidate.args[0]
        return _attribute_key(candidate)

    def guarded_sequences(
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, tuple[str, ...]]:
        resolved: dict[str, tuple[str, ...]] = {}
        for guard in [node for node in ast.walk(function) if isinstance(node, ast.If)]:
            test = guard.test
            if (
                not any(isinstance(node, ast.Raise) for node in guard.body)
                or not isinstance(test, ast.Compare)
                or len(test.ops) != 1
                or not isinstance(test.ops[0], ast.NotEq)
                or len(test.comparators) != 1
            ):
                continue
            dynamic_key = validated_name(test.left)
            canonical_node = test.comparators[0]
            canonical = _static_output_sequence(
                canonical_node,
                sequences,
                mappings,
                strings,
            )
            if canonical is None:
                dynamic_key = validated_name(test.comparators[0])
                canonical_node = test.left
                canonical = _static_output_sequence(
                    canonical_node,
                    sequences,
                    mappings,
                    strings,
                )
            if dynamic_key is not None and canonical is not None:
                resolved[dynamic_key] = canonical
        return resolved

    method_return_sequences: dict[str, tuple[str, ...]] = {}
    for method_name, method in methods.items():
        guarded = guarded_sequences(method)
        for returned in [node for node in ast.walk(method) if isinstance(node, ast.Return)]:
            value = returned.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "tuple"
                and len(value.args) == 1
            ):
                continue
            returned_name = _attribute_key(value.args[0])
            if returned_name in guarded:
                method_return_sequences[method_name] = guarded[returned_name]

    initializer = methods.get("__init__")
    if initializer is not None:
        positional_args = [
            *initializer.args.posonlyargs,
            *initializer.args.args,
        ]
        positional_defaults = list(initializer.args.defaults)
        default_bindings = list(
            zip(
                positional_args[-len(positional_defaults) :]
                if positional_defaults
                else [],
                positional_defaults,
            )
        )
        default_bindings.extend(
            (argument, default)
            for argument, default in zip(
                initializer.args.kwonlyargs,
                initializer.args.kw_defaults,
            )
            if default is not None
        )
        for argument, default in default_bindings:
            string_value = _static_string(default, strings)
            if string_value is not None:
                strings[argument.arg] = string_value
            sequence_value = _static_output_sequence(
                default,
                sequences,
                mappings,
                strings,
            )
            if sequence_value is not None:
                sequences[argument.arg] = sequence_value
            mapping_value = _static_string_mapping(default, mappings, strings)
            if mapping_value is not None:
                mappings[argument.arg] = mapping_value

        for dynamic_key, canonical in guarded_sequences(initializer).items():
            if not dynamic_key.startswith("self."):
                continue
            sequences[dynamic_key] = canonical
            sequences[dynamic_key.removeprefix("self.")] = canonical

        for assignment in [
            node
            for node in ast.walk(initializer)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]:
            value = assignment.value
            if value is None:
                continue
            targets = (
                assignment.targets
                if isinstance(assignment, ast.Assign)
                else [assignment.target]
            )
            for target in targets:
                target_key = _attribute_key(target)
                if target_key is None or not target_key.startswith("self."):
                    continue
                string_value = _static_string(value, strings)
                if string_value is not None:
                    strings[target_key] = string_value
                sequence_value = _static_output_sequence(
                    value,
                    sequences,
                    mappings,
                    strings,
                )
                if sequence_value is None and (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == "self"
                ):
                    sequence_value = method_return_sequences.get(value.func.attr)
                if sequence_value is not None:
                    sequences[target_key] = sequence_value
                mapping_value = _static_string_mapping(value, mappings, strings)
                if mapping_value is not None:
                    mappings[target_key] = mapping_value
    return strings, sequences, mappings


def _class_hierarchy_entries(
    cls: ast.ClassDef,
    path: Path,
    tree: ast.Module,
    class_index: ClassIndex,
    visited: set[tuple[str, str]] | None = None,
) -> list[ClassEntry]:
    """Return the concrete class and uniquely resolved bases in MRO order."""
    marker = (str(path.resolve()), cls.name)
    visited = set(visited or set())
    if marker in visited:
        return []
    visited.add(marker)
    entries: list[ClassEntry] = [(path, tree, cls)]
    local_classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    for base in cls.bases:
        base_name = _base_class_name(base)
        if not base_name:
            continue
        local_base = local_classes.get(base_name)
        if local_base is not None:
            base_entry: ClassEntry | None = (path, tree, local_base)
        else:
            candidates = class_index.get(base_name, [])
            base_entry = candidates[0] if len(candidates) == 1 else None
        if base_entry is None:
            continue
        entries.extend(
            _class_hierarchy_entries(
                base_entry[2],
                base_entry[0],
                base_entry[1],
                class_index,
                visited,
            )
        )
    return entries


def _effective_metadata_outputs(
    cls: ast.ClassDef,
    path: Path,
    tree: ast.Module,
    class_index: ClassIndex,
    initial_outputs: list[str],
    strings: dict[str, str],
    sequences: dict[str, tuple[str, ...]],
    mappings: dict[str, dict[str, str]],
) -> list[str]:
    """Apply statically provable metadata output mutations across base initializers."""
    outputs = list(dict.fromkeys(initial_outputs))
    local_sequences = dict(sequences)

    # A class may construct more than one FunctorMetadata (for example one
    # per declared physical boundary). Every statically provable
    # output_variables declaration in the class body is part of the class
    # output contract, in addition to the runtime inventory metadata.
    for node in ast.walk(cls):
        if not (
            isinstance(node, ast.Call)
            and (_attribute_key(node.func) or "").split(".")[-1] == "FunctorMetadata"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg != "output_variables":
                continue
            declared_values = _static_output_sequence(
                keyword.value,
                sequences,
                mappings,
                strings,
            )
            for value in declared_values or ():
                if value not in outputs:
                    outputs.append(value)
            # A local list passed as output_variables may be extended under a
            # configuration branch before the FunctorMetadata call (for example
            # output_variables.extend(self.CONSTANT)); every statically provable
            # append/extend of that local name is part of the class output contract.
            if isinstance(keyword.value, ast.Name):
                for mutation in ast.walk(cls):
                    if not (
                        isinstance(mutation, ast.Call)
                        and isinstance(mutation.func, ast.Attribute)
                        and isinstance(mutation.func.value, ast.Name)
                        and mutation.func.value.id == keyword.value.id
                        and mutation.func.attr in {"append", "extend"}
                        and mutation.args
                    ):
                        continue
                    if mutation.func.attr == "append":
                        single = _static_string(mutation.args[0], strings)
                        mutated = None if single is None else (single,)
                    else:
                        mutated = _static_output_sequence(
                            mutation.args[0],
                            sequences,
                            mappings,
                            strings,
                        )
                    for value in mutated or ():
                        if value not in outputs:
                            outputs.append(value)

    def refresh_declared_outputs() -> None:
        values = tuple(outputs)
        local_sequences["metadata.output_variables"] = values
        local_sequences["self.metadata.output_variables"] = values

    refresh_declared_outputs()
    for _, _, hierarchy_class in _class_hierarchy_entries(
        cls,
        path,
        tree,
        class_index,
    ):
        initializer = next(
            (
                node
                for node in hierarchy_class.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ),
            None,
        )
        if initializer is None:
            continue
        operations = sorted(
            (
                node
                for node in ast.walk(initializer)
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Call))
            ),
            key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
        )
        for node in operations:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = _attribute_key(node.func.value)
                if (
                    not owner
                    or not owner.endswith("metadata.output_variables")
                    or not node.args
                ):
                    continue
                if node.func.attr == "append":
                    value = _static_string(node.args[0], strings)
                    if value is not None and value not in outputs:
                        outputs.append(value)
                        refresh_declared_outputs()
                    continue
                if node.func.attr != "extend":
                    continue
                values = _static_output_sequence(
                    node.args[0],
                    local_sequences,
                    mappings,
                    strings,
                )
                if values is None:
                    continue
                for value in values:
                    if value not in outputs:
                        outputs.append(value)
                refresh_declared_outputs()
                continue

            if isinstance(node, ast.Assign):
                targets = node.targets
                value_node = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value_node = node.value
            else:
                continue
            if value_node is None or not any(
                (_attribute_key(target) or "").endswith("metadata.output_variables")
                for target in targets
            ):
                continue
            values = _static_output_sequence(
                value_node,
                local_sequences,
                mappings,
                strings,
            )
            if values is None:
                continue
            outputs = list(dict.fromkeys(values))
            refresh_declared_outputs()
    return outputs


def _analyze_tree(
    tree: ast.Module,
    path: Path,
    class_index: ClassIndex,
    function_index: FunctionIndex,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        if _is_abstract_class(cls):
            continue
        metadata = _extract_metadata(cls, tree)
        if not _is_candidate_functor(cls) and not metadata:
            continue
        class_strings, class_sequences, class_mappings = (
            _class_constant_environments(
                cls,
                path,
                tree,
                class_index,
            )
        )
        metadata_outputs = [
            value
            for value in metadata.get("output_variables", "").split("|")
            if value
        ]
        metadata_outputs = _effective_metadata_outputs(
            cls,
            path,
            tree,
            class_index,
            metadata_outputs,
            class_strings,
            class_sequences,
            class_mappings,
        )
        if not metadata_outputs:
            continue
        owner = _find_forward_owner(cls, path, tree, class_index)
        if owner is None:
            forward = None
            forward_path = path
            forward_owner = cls.name
            forward_keys: set[str] = set()
            unresolved = {"missing forward implementation"}
            resolution_status = "unresolved_forward"
        else:
            forward_path, _, owner_class = owner
            forward_owner = owner_class.name
            forward = next(
                node
                for node in owner_class.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "forward"
            )
            def resolve_method(
                method_name: str,
            ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
                method_entry = _find_method_owner(
                    cls,
                    path,
                    tree,
                    class_index,
                    method_name,
                )
                return method_entry[3] if method_entry is not None else None

            def resolve_function(
                function_name: str,
            ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
                candidates = function_index.get(function_name, [])
                return candidates[0][2] if len(candidates) == 1 else None

            def resolve_external_method(
                method_name: str,
            ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
                candidates = [
                    method
                    for entries in class_index.values()
                    for _, _, indexed_class in entries
                    for method in indexed_class.body
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and method.name == method_name
                ]
                return candidates[0] if len(candidates) == 1 else None

            forward_sequences = dict(class_sequences)
            forward_sequences["metadata.output_variables"] = tuple(
                metadata_outputs
            )
            forward_sequences["self.metadata.output_variables"] = tuple(
                metadata_outputs
            )
            forward_keys, unresolved = _ForwardOutputCollector(
                method_resolver=resolve_method,
                external_method_resolver=resolve_external_method,
                function_resolver=resolve_function,
                class_string_environment=class_strings,
                class_sequence_environment=forward_sequences,
                class_mapping_environment=class_mappings,
            ).collect(forward)
            resolution_status = (
                "local_static" if owner_class is cls else "inherited_static"
            )
        metadata_set = set(metadata_outputs)
        records.append(
            {
                "file": str(path).replace("/", "\\"),
                "class": cls.name,
                "forward_file": str(forward_path).replace("/", "\\"),
                "forward_owner": forward_owner,
                "forward_line": forward.lineno if forward is not None else None,
                "resolution_status": resolution_status,
                "metadata_outputs": metadata_outputs,
                "forward_literal_keys": sorted(forward_keys),
                "unresolved_dynamic_keys": sorted(unresolved),
                "missing_in_forward_literal": sorted(metadata_set - forward_keys),
                "extra_not_in_metadata": sorted(forward_keys - metadata_set),
            }
        )
    return records


def analyze_source(source: str, path: Path) -> list[dict[str, Any]]:
    """Analyze one source file without importing it or writing any artifact."""
    tree = ast.parse(source, filename=str(path))
    class_index: ClassIndex = {}
    function_index: FunctionIndex = {}
    for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        class_index.setdefault(cls.name, []).append((path, tree, cls))
    for function in [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        function_index.setdefault(function.name, []).append((path, tree, function))
    return _analyze_tree(tree, path, class_index, function_index)


def scan(root: Path = ROOT) -> dict[str, Any]:
    """Scan the existing scientific tree and return an in-memory report."""
    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    parsed_sources: list[tuple[Path, ast.Module]] = []
    class_index: ClassIndex = {}
    function_index: FunctionIndex = {}
    for path in sorted(root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_errors.append({"file": str(path), "error": str(exc)})
            continue
        parsed_sources.append((path, tree))
        for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
            class_index.setdefault(cls.name, []).append((path, tree, cls))
        for function in [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]:
            function_index.setdefault(function.name, []).append((path, tree, function))

    for path, tree in parsed_sources:
        records.extend(_analyze_tree(tree, path, class_index, function_index))

    if root.resolve() == ROOT.resolve():
        formal_process_pairs = {
            (
                str((REPO_ROOT / Path(row["file_path"])).resolve()),
                row["class_name"],
            )
            for row in collect_functor_inventory()
            if row.get("process_status") == "process"
        }
        records = [
            record
            for record in records
            if (
                str(Path(record["file"]).resolve()),
                record["class"],
            )
            in formal_process_pairs
        ]

    issues = [record for record in records if record["extra_not_in_metadata"]]
    missing_only = [
        record
        for record in records
        if record["missing_in_forward_literal"]
        and not record["extra_not_in_metadata"]
        and not record["unresolved_dynamic_keys"]
    ]
    unresolved = [record for record in records if record["unresolved_dynamic_keys"]]
    return {
        "records": records,
        "record_count": len(records),
        "issues": issues,
        "issue_count": len(issues),
        "missing_only": missing_only,
        "missing_only_count": len(missing_only),
        "unresolved": unresolved,
        "unresolved_count": len(unresolved),
        "parse_errors": parse_errors,
        "parse_error_count": len(parse_errors),
    }


def write_report(report: dict[str, Any], output_path: Path = OUTPUT_PATH) -> None:
    """Write one explicitly requested audit artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    report = scan()
    write_report(report)
    print(f"records={report['record_count']}")
    print(f"issues={report['issue_count']}")
    print(f"missing_only={report['missing_only_count']}")
    print(f"unresolved={report['unresolved_count']}")
    print(f"parse_errors={report['parse_error_count']}")
    return 1 if report["parse_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
