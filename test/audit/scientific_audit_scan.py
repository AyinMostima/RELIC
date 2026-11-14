import ast
import io
import json
import re
import sys
import tokenize
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
except Exception as exc:
    raise RuntimeError('PyYAML is required for audit scan') from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relic.data.standards.dimensionality import (
    dimension_category_from_unit,
    is_declared_dimension_compatible,
)

SCIENTIFIC_ROOT = REPO_ROOT / 'relic' / 'scientific'
PARAM_ROOT = REPO_ROOT / 'relic' / 'parameters' / 'database' / 'relic'
CORE_ROOT = REPO_ROOT / 'relic' / 'core'
THEORY_ROOT = REPO_ROOT / 'docs' / 'source' / 'theory'
RESULT_DIR = REPO_ROOT / 'analysis_results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BRIDGE_CLASSES = {
    'PhysicalFieldFunctor',
    'BiologicalProcessFunctor',
    'TransportMechanismFunctor',
    'ScaleCouplingFunctor',
}
ABSTRACTIONS = {
    'PhysicalField',
    'BiologicalProcess',
    'TransportMechanism',
    'ScaleCoupling',
}
FUNCTOR_BASE_NAMES = BRIDGE_CLASSES | {'FunctorBase'}
MANDATORY_METADATA_FIELDS = {
    'name',
    'version',
    'description',
    'category',
    'execution_phase',
    'mathematical_expression',
    'references',
    'parameters',
    'input_variables',
    'output_variables',
    'conservation_laws',
    'spatial_scale',
    'variable_kind',
}

PARAM_REQUIRED_FIELDS = {
    'value',
    'units',
    'dtype',
    'bounds',
    'category',
    'description',
    'references',
    'tags',
    'dimension',
    'requires_grad',
    'provenance',
}

HARD_PATTERNS = {
    'get_parameter_value_default_literal': re.compile(r'get_parameter_value\([^\n\)]*default\s*=\s*[-+]?\d+(?:\.\d+)?'),
    'torch_tensor_literal': re.compile(r'torch\.tensor\(\s*[-+]?\d+(?:\.\d+)?'),
    'state_get_default_literal': re.compile(r'state\.get\([^\n\)]*default\s*=\s*[-+]?\d+(?:\.\d+)?'),
    'manager_get_parameter_value_removed_api': re.compile(
        r'\b(?:manager|param_manager|parameter_manager)\.get_parameter_value\('
    ),
}


def _mask_non_executable_text(text: str) -> str:
    """Mask string literals and comments while preserving source offsets."""
    masked = list(text)
    line_offsets = [0]
    for match in re.finditer(r"\n", text):
        line_offsets.append(match.end())

    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    for token in tokens:
        if token.type not in {tokenize.STRING, tokenize.COMMENT}:
            continue
        start_row, start_col = token.start
        end_row, end_col = token.end
        start = line_offsets[start_row - 1] + start_col
        end = line_offsets[end_row - 1] + end_col
        for index in range(start, end):
            if masked[index] not in "\r\n":
                masked[index] = " "
    return "".join(masked)

REFERENCE_SCAN_ROOTS = [
    REPO_ROOT / 'relic' / 'scientific',
    REPO_ROOT / 'relic' / 'coupling',
    REPO_ROOT / 'relic' / 'core',
    REPO_ROOT / 'scripts',
    REPO_ROOT / 'test',
]

REFERENCE_TEXT_SUFFIXES = {
    '.py',
    '.yaml',
    '.yml',
    '.json',
    '.md',
    '.toml',
    '.ini',
    '.cfg',
    '.txt',
}

PARAM_ACCESS_CALL_NAMES = {
    'get_parameter_value',
    'get',
    'get_parameter',
    'require_parameter_value',
    '_require_parameter',
    '_require_param',
    '_param',
    '_scalar_param',
    'require_param',
    'require_parameter',
    'require_scalar',
}


def node_to_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = node_to_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Subscript):
        return node_to_name(node.value)
    if isinstance(node, ast.Call):
        return node_to_name(node.func)
    return ''


def string_literal(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def iter_scan_files() -> Tuple[List[Path], List[Path]]:
    py_files: List[Path] = []
    text_files: List[Path] = []
    for root in REFERENCE_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == '.py':
                py_files.append(path)
            if suffix in REFERENCE_TEXT_SUFFIXES:
                try:
                    path.relative_to(PARAM_ROOT)
                    continue
                except ValueError:
                    text_files.append(path)
    return sorted(set(py_files)), sorted(set(text_files))


def collect_parameter_reference_sets():
    ast_full_key_refs: Set[str] = set()
    ast_namespace_leaf_refs: Set[Tuple[str, str]] = set()
    ast_leaf_only_refs: Set[str] = set()
    text_full_key_refs: Set[str] = set()
    dynamic_call_sites: List[Dict[str, object]] = []

    py_files, text_files = iter_scan_files()
    dotted_key_pattern = re.compile(r'\b[a-zA-Z_]\w*(?:\.[a-zA-Z0-9_]+){1,}\b')

    for path in py_files:
        rel = str(path.relative_to(REPO_ROOT)).replace('/', '\\')
        try:
            source = path.read_text(encoding='utf-8-sig', errors='ignore')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SyntaxWarning)
                tree = ast.parse(source)
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            call_name = ''
            if isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                call_name = node.func.id
            if call_name not in PARAM_ACCESS_CALL_NAMES:
                continue

            key_literal = None
            namespace_literal = None
            key_is_dynamic = False
            namespace_is_dynamic = False

            if node.args:
                key_literal = string_literal(node.args[0])
                key_is_dynamic = key_literal is None

            for kw in node.keywords:
                if kw.arg in {'name', 'key', 'param_name'}:
                    cand = string_literal(kw.value)
                    if cand is not None:
                        key_literal = cand
                        key_is_dynamic = False
                    elif kw.value is not None:
                        key_is_dynamic = True
                elif kw.arg in {'namespace', 'ns'}:
                    cand = string_literal(kw.value)
                    if cand is not None:
                        namespace_literal = cand
                        namespace_is_dynamic = False
                    elif kw.value is not None:
                        namespace_is_dynamic = True

            if key_literal:
                key = key_literal.strip().strip('.')
                if not key:
                    continue
                if '.' in key:
                    ast_full_key_refs.add(key)
                elif namespace_literal and '.' in namespace_literal:
                    ns = namespace_literal.strip().strip('.')
                    ast_namespace_leaf_refs.add((ns, key))
                else:
                    ast_leaf_only_refs.add(key)
            if key_is_dynamic or namespace_is_dynamic:
                dynamic_call_sites.append(
                    {
                        'file': rel,
                        'line': getattr(node, 'lineno', 0),
                        'call': call_name,
                        'key_literal': key_literal,
                        'namespace_literal': namespace_literal,
                    }
                )

        for match in dotted_key_pattern.finditer(source):
            token = match.group(0).strip('.')
            if token.count('.') >= 1:
                text_full_key_refs.add(token)

    for path in text_files:
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        for match in dotted_key_pattern.finditer(text):
            token = match.group(0).strip('.')
            if token.count('.') >= 1:
                text_full_key_refs.add(token)

    return {
        'python_files_scanned': len(py_files),
        'text_files_scanned': len(text_files),
        'ast_full_key_refs': ast_full_key_refs,
        'ast_namespace_leaf_refs': ast_namespace_leaf_refs,
        'ast_leaf_only_refs': ast_leaf_only_refs,
        'text_full_key_refs': text_full_key_refs,
        'dynamic_call_sites': dynamic_call_sites,
    }


def has_namespace_leaf_hit(full_key: str, namespace_leaf_refs: Set[Tuple[str, str]]) -> bool:
    parts = full_key.split('.')
    if len(parts) < 2:
        return False
    leaf = parts[-1]
    for i in range(1, len(parts)):
        ns = '.'.join(parts[:i])
        if (ns, leaf) in namespace_leaf_refs:
            return True
    return False


def build_parameter_reference_index(param_key_to_files):
    ref_sets = collect_parameter_reference_sets()
    ast_full_key_refs: Set[str] = ref_sets['ast_full_key_refs']
    ast_namespace_leaf_refs: Set[Tuple[str, str]] = ref_sets['ast_namespace_leaf_refs']
    ast_leaf_only_refs: Set[str] = ref_sets['ast_leaf_only_refs']
    text_full_key_refs: Set[str] = ref_sets['text_full_key_refs']

    resolved_exact: Set[str] = set()
    resolved_namespace: Set[str] = set()
    candidates = []
    class_counts = {
        'definitely_unreferenced': 0,
        'possible_dynamic_reference': 0,
        'configuration_only_reference': 0,
    }

    for key, files in sorted(param_key_to_files.items()):
        if key in ast_full_key_refs:
            resolved_exact.add(key)
            continue
        if has_namespace_leaf_hit(key, ast_namespace_leaf_refs):
            resolved_namespace.add(key)
            continue

        classification = 'definitely_unreferenced'
        evidence: List[str] = []
        if key in text_full_key_refs:
            classification = 'configuration_only_reference'
            evidence.append('text_full_key_reference')
        elif key.split('.')[-1] in ast_leaf_only_refs:
            classification = 'possible_dynamic_reference'
            evidence.append('ast_leaf_only_reference')

        class_counts[classification] += 1
        candidates.append(
            {
                'key': key,
                'files': files,
                'classification': classification,
                'evidence': evidence,
            }
        )

    namespace_pairs = [
        {'namespace': ns, 'key': key}
        for ns, key in sorted(ast_namespace_leaf_refs)
    ]
    index = {
        'schema_version': '1.0.0',
        'scan_roots': [
            str(p.relative_to(REPO_ROOT)).replace('/', '\\')
            for p in REFERENCE_SCAN_ROOTS
            if p.exists()
        ],
        'python_files_scanned': ref_sets['python_files_scanned'],
        'text_files_scanned': ref_sets['text_files_scanned'],
        'ast_full_key_reference_count': len(ast_full_key_refs),
        'ast_namespace_key_reference_count': len(namespace_pairs),
        'ast_leaf_only_reference_count': len(ast_leaf_only_refs),
        'text_full_key_reference_count': len(text_full_key_refs),
        'resolved_exact_keys': sorted(resolved_exact),
        'resolved_namespace_keys': sorted(resolved_namespace),
        'ast_namespace_key_references': namespace_pairs,
        'classification_counts': class_counts,
        'dynamic_call_sites': ref_sets['dynamic_call_sites'],
        'orphan_candidates': candidates,
    }
    return index, candidates


def find_functor_metadata_calls(tree):
    calls = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = node_to_name(n.func)
            if fn.endswith('FunctorMetadata'):
                keys = []
                for kw in n.keywords:
                    if kw.arg:
                        keys.append(kw.arg)
                calls.append({'line': n.lineno, 'keys': sorted(keys)})
    return calls


def find_imports(tree):
    imports = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imports.append(a.name)
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ''
            imports.append(mod)
    return imports


def get_class_report(cls):
    bases = [node_to_name(b) for b in cls.bases]
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    is_functor = cls.name.endswith('Functor') or bool(set(bases) & FUNCTOR_BASE_NAMES)

    forward_info = None
    state_direct_write = []
    if 'forward' in methods:
        fn = methods['forward']
        args = [a.arg for a in fn.args.args]
        forward_info = {
            'line': fn.lineno,
            'args': args,
        }
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                targets = n.targets
            elif isinstance(n, ast.AugAssign):
                targets = [n.target]
            else:
                targets = []
            for t in targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'state':
                    state_direct_write.append(getattr(t, 'lineno', fn.lineno))

    # Dependency declarations are not parsed from source literals: overrides
    # route through FunctorBase._merge_declared_dependencies and instance
    # declarations live in config / set_dependencies / add_dependency. The
    # runtime declaration is reported by test.audit.collect_functor_inventory.

    super_init_calls = []
    if '__init__' in methods:
        fn = methods['__init__']
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                fn_name = node_to_name(n.func)
                if fn_name in {'super.__init__', 'super().__init__'} or fn_name.endswith('.__init__'):
                    if 'super' in fn_name:
                        kw = [kw.arg for kw in n.keywords if kw.arg]
                        super_init_calls.append({'line': n.lineno, 'keywords': kw})

    return {
        'name': cls.name,
        'line': cls.lineno,
        'bases': bases,
        'is_functor_candidate': is_functor,
        'forward': forward_info,
        'state_direct_write_lines': state_direct_write,
        'super_init_calls': super_init_calls,
    }


def scan_python_file(path):
    text = path.read_text(encoding='utf-8', errors='ignore')
    executable_text = _mask_non_executable_text(text)
    rel = str(path.relative_to(REPO_ROOT)).replace('/', '\\')
    report = {
        'file': rel,
        'line_count': text.count('\n') + 1,
        'imports_category_theory': False,
        'imports': [],
        'hardcoded_patterns': [],
        'classes': [],
        'functor_metadata_calls': [],
        'uses_get_global_manager': False,
        'uses_require_parameter_value': False,
        'uses_get_physical_constant': False,
    }
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        report['parse_error'] = str(exc)
        return report

    imports = find_imports(tree)
    report['imports'] = imports
    report['imports_category_theory'] = any('category_theory' in i for i in imports)
    report['uses_get_global_manager'] = 'get_global_manager' in text
    report['uses_require_parameter_value'] = 'require_parameter_value' in text
    report['uses_get_physical_constant'] = 'get_physical_constant' in text

    for name, pat in HARD_PATTERNS.items():
        for m in pat.finditer(executable_text):
            line = text.count('\n', 0, m.start()) + 1
            report['hardcoded_patterns'].append({'type': name, 'line': line, 'snippet': text[m.start():m.end()][:160]})

    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    report['classes'] = [get_class_report(c) for c in classes]
    report['functor_metadata_calls'] = find_functor_metadata_calls(tree)
    return report


def expected_dimension_from_units(units):
    return dimension_category_from_unit(units)


def scan_yaml_file(path):
    rel = str(path.relative_to(REPO_ROOT)).replace('/', '\\')
    text = path.read_text(encoding='utf-8', errors='ignore')
    report = {
        'file': rel,
        'parse_error': None,
        'namespace': None,
        'schema_version': None,
        'parameter_count': 0,
        'parameters': [],
        'functor_metadata': None,
    }
    try:
        data = yaml.safe_load(text) or {}
    except Exception as exc:
        report['parse_error'] = str(exc)
        return report

    report['namespace'] = data.get('namespace')
    report['schema_version'] = data.get('schema_version')
    report['functor_metadata'] = data.get('functor_metadata')
    params = data.get('parameters') or {}
    report['parameter_count'] = len(params)

    for key, spec in params.items():
        if not isinstance(spec, dict):
            report['parameters'].append({'key': key, 'error': 'spec_not_mapping'})
            continue
        missing = sorted(list(PARAM_REQUIRED_FIELDS - set(spec.keys())))
        references = spec.get('references')
        refs_empty = isinstance(references, list) and len(references) == 0
        units = spec.get('units')
        dimension = spec.get('dimension')
        expected_dim = expected_dimension_from_units(units)
        dim_mismatch = not is_declared_dimension_compatible(units, dimension)
        bounds_present = 'bounds' in spec

        report['parameters'].append({
            'key': key,
            'missing_fields': missing,
            'references_empty': refs_empty,
            'units': units,
            'dimension': dimension,
            'expected_dimension_from_units': expected_dim,
            'dimension_mismatch': dim_mismatch,
            'uses_fraction_unit_notation': isinstance(units, str) and '/' in units,
            'uses_units_one': str(units).strip() == '1',
            'bounds_present': bounds_present,
        })

    return report


def main(result_dir: Path | None = None):
    active_result_dir = RESULT_DIR if result_dir is None else Path(result_dir)
    active_result_dir.mkdir(parents=True, exist_ok=True)
    py_files = sorted(SCIENTIFIC_ROOT.rglob('*.py'))
    yaml_files = sorted(PARAM_ROOT.rglob('*.yaml'))

    py_reports = [scan_python_file(p) for p in py_files]
    yaml_reports = [scan_yaml_file(p) for p in yaml_files]

    param_key_to_files = {}
    for yr in yaml_reports:
        for p in yr.get('parameters', []):
            key = p.get('key')
            if key:
                param_key_to_files.setdefault(key, []).append(yr['file'])

    duplicate_parameters = {k: v for k, v in param_key_to_files.items() if len(v) > 1}

    parameter_reference_index, orphan_candidates = build_parameter_reference_index(param_key_to_files)
    index_file = active_result_dir / 'parameter_reference_index.json'
    index_file.write_text(
        json.dumps(parameter_reference_index, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    constants_file = PARAM_ROOT / 'physical' / 'constants.yaml'
    constant_keys = set()
    if constants_file.exists():
        c_report = scan_yaml_file(constants_file)
        for p in c_report.get('parameters', []):
            if 'key' in p:
                constant_keys.add(p['key'])

    duplicated_physical_constants = []
    for yr in yaml_reports:
        if yr['file'].endswith('physical\\constants.yaml'):
            continue
        for p in yr.get('parameters', []):
            key = p.get('key')
            if key in constant_keys:
                duplicated_physical_constants.append({'key': key, 'file': yr['file']})

    try:
        index_reference = str(index_file.relative_to(REPO_ROOT)).replace('/', '\\')
    except ValueError:
        index_reference = index_file.name

    result = {
        'repo_root': str(REPO_ROOT),
        'scientific_file_count': len(py_reports),
        'parameter_yaml_count': len(yaml_reports),
        'python_reports': py_reports,
        'yaml_reports': yaml_reports,
        'duplicate_parameters': duplicate_parameters,
        'orphan_parameter_candidates': orphan_candidates,
        'parameter_reference_index_file': index_reference,
        'duplicated_physical_constants': duplicated_physical_constants,
        'theory_docs': sorted(str(p.relative_to(REPO_ROOT)).replace('/', '\\') for p in THEORY_ROOT.rglob('*.md')),
    }

    out_file = active_result_dir / 'scientific_audit_scan_20260213.json'
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(str(out_file))


if __name__ == '__main__':
    main()
