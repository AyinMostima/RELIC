import ast
import json
from pathlib import Path

ROOT = Path('relic/scientific')
rows = []
for py in ROOT.rglob('*.py'):
    try:
        src = py.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        src = py.read_text(encoding='utf-8', errors='ignore')
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
                else:
                    bases.append(ast.unparse(b) if hasattr(ast, 'unparse') else 'unknown')
            if not any('Functor' in x for x in bases):
                continue
            methods = {m.name: m for m in node.body if isinstance(m, ast.FunctionDef)}
            fwd = methods.get('forward')
            fwd_sig = None
            if fwd is not None:
                fwd_sig = [a.arg for a in fwd.args.args]
            has_get_global_manager = 'get_global_manager(' in src
            has_mgr_get_param_val = 'get_parameter_value(' in src and 'manager.' in src
            metadata_fields = set()
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == '__init__':
                    for n in ast.walk(m):
                        if isinstance(n, ast.Call):
                            fn = n.func
                            fn_name = None
                            if isinstance(fn, ast.Name):
                                fn_name = fn.id
                            elif isinstance(fn, ast.Attribute):
                                fn_name = fn.attr
                            if fn_name == 'FunctorMetadata':
                                for kw in n.keywords:
                                    if kw.arg:
                                        metadata_fields.add(kw.arg)
            rows.append({
                'file': str(py).replace('/', '\\\\'),
                'class': node.name,
                'bases': bases,
                'line': node.lineno,
                'forward_sig': fwd_sig,
                'metadata_fields': sorted(metadata_fields),
                'uses_get_global_manager': has_get_global_manager,
            })

out = Path('analysis_results/functor_scan.json')
out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'count={len(rows)}')
print(str(out))
