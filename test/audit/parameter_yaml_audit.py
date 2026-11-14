import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relic.data.standards.dimensionality import (
    dimension_category_from_unit,
    is_declared_dimension_compatible,
)

ROOT = REPO_ROOT / 'relic' / 'parameters' / 'database' / 'relic'
OUT_DIR = REPO_ROOT / 'analysis_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FIELDS = [
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
]

def is_param_leaf(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    # Avoid treating metadata/documentation blocks as ParameterSpec leaves.
    # A valid parameter leaf must at least carry explicit numeric/content value
    # and units; optional governance fields are checked later.
    return 'value' in keys and 'units' in keys


def walk(node: Any, path: Tuple[str, ...], file_path: Path, out: Dict[str, Any]) -> None:
    if isinstance(node, dict):
        if 'functor_metadata' in node and isinstance(node['functor_metadata'], dict):
            fm = node['functor_metadata']
            out['functor_metadata_blocks'].append({
                'file': str(file_path).replace('\\\\', '/'),
                'path': '.'.join(path) if path else '<root>',
                'missing_spatial_scale': 'spatial_scale' not in fm,
                'missing_variable_kind': 'variable_kind' not in fm,
                'spatial_scale': fm.get('spatial_scale'),
                'variable_kind': fm.get('variable_kind'),
            })

        if is_param_leaf(node):
            key = '.'.join(path)
            out['parameter_entries'].append({
                'file': str(file_path).replace('\\\\', '/'),
                'key': key,
                'data': node,
            })
            return

        for k, v in node.items():
            if isinstance(k, str):
                walk(v, path + (k,), file_path, out)


def main() -> None:
    audit: Dict[str, Any] = {
        'files_scanned': 0,
        'parameter_entries': [],
        'functor_metadata_blocks': [],
        'issues': {
            'missing_required_fields': [],
            'empty_references': [],
            'provenance_missing_subfields': [],
            'unit_dimension_mismatch': [],
            'duplicate_parameter_keys': [],
            'functor_metadata_missing_fields': [],
        },
    }

    for file_path in sorted(ROOT.rglob('*.yaml')):
        audit['files_scanned'] += 1
        text = file_path.read_text(encoding='utf-8-sig', errors='ignore')
        try:
            data = yaml.safe_load(text)
        except Exception as exc:
            audit['issues']['missing_required_fields'].append({
                'file': str(file_path).replace('\\\\', '/'),
                'key': '<yaml_parse_error>',
                'missing': [str(exc)],
            })
            continue
        if data is None:
            continue
        walk(data, tuple(), file_path, audit)

    # Parameter checks
    key_to_files: Dict[str, List[str]] = {}
    for entry in audit['parameter_entries']:
        file_path = entry['file']
        key = entry['key']
        data = entry['data']

        key_to_files.setdefault(key, []).append(file_path)

        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            audit['issues']['missing_required_fields'].append({
                'file': file_path,
                'key': key,
                'missing': missing,
            })

        refs = data.get('references')
        if refs is None or (isinstance(refs, list) and len(refs) == 0):
            audit['issues']['empty_references'].append({
                'file': file_path,
                'key': key,
            })

        prov = data.get('provenance')
        if isinstance(prov, dict):
            missing_prov = [x for x in ('source', 'schema_version') if x not in prov]
            if missing_prov:
                audit['issues']['provenance_missing_subfields'].append({
                    'file': file_path,
                    'key': key,
                    'missing': missing_prov,
                })
        elif prov is not None:
            audit['issues']['provenance_missing_subfields'].append({
                'file': file_path,
                'key': key,
                'missing': ['source', 'schema_version'],
            })

        units = data.get('units')
        dim = data.get('dimension')
        if not is_declared_dimension_compatible(units, dim):
            audit['issues']['unit_dimension_mismatch'].append({
                'file': file_path,
                'key': key,
                'units': units,
                'dimension': dim,
                'expected_dimension': dimension_category_from_unit(units),
            })

    for key, files in key_to_files.items():
        if len(set(files)) > 1:
            audit['issues']['duplicate_parameter_keys'].append({
                'key': key,
                'files': sorted(set(files)),
            })

    for block in audit['functor_metadata_blocks']:
        if block['missing_spatial_scale'] or block['missing_variable_kind']:
            audit['issues']['functor_metadata_missing_fields'].append(block)

    summary = {
        'files_scanned': audit['files_scanned'],
        'parameter_entries': len(audit['parameter_entries']),
        'functor_metadata_blocks': len(audit['functor_metadata_blocks']),
        'missing_required_fields': len(audit['issues']['missing_required_fields']),
        'empty_references': len(audit['issues']['empty_references']),
        'provenance_missing_subfields': len(audit['issues']['provenance_missing_subfields']),
        'unit_dimension_mismatch': len(audit['issues']['unit_dimension_mismatch']),
        'duplicate_parameter_keys': len(audit['issues']['duplicate_parameter_keys']),
        'functor_metadata_missing_fields': len(audit['issues']['functor_metadata_missing_fields']),
    }

    (OUT_DIR / 'parameter_yaml_audit.json').write_text(
        json.dumps({'summary': summary, 'issues': audit['issues']}, indent=2, ensure_ascii=True),
        encoding='utf-8',
    )

    lines = [f"{k}={v}" for k, v in summary.items()]
    (OUT_DIR / 'parameter_yaml_audit_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
