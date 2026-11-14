import ast
import yaml
from pathlib import Path

yaml_data = yaml.safe_load(Path('relic/parameters/database/relic/pedosphere/nitrification.yaml').read_text(encoding='utf-8-sig'))
fm = yaml_data.get('functor_metadata', {})
print('YAML_variable_kind=' + str(fm.get('variable_kind')))
print('YAML_outputs=' + ','.join(fm.get('output_variables', [])))

src = Path('relic/scientific/pedosphere/nutrient_cycling/nitrification.py').read_text(encoding='utf-8-sig')
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        fn = ''
        if isinstance(node.func, ast.Name):
            fn = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fn = node.func.attr
        if fn == 'FunctorMetadata':
            fields = {}
            for kw in node.keywords:
                if kw.arg == 'variable_kind':
                    fields['variable_kind'] = ast.unparse(kw.value)
                if kw.arg == 'output_variables':
                    if isinstance(kw.value, (ast.List, ast.Tuple)):
                        vals = []
                        for e in kw.value.elts:
                            if isinstance(e, ast.Constant):
                                vals.append(str(e.value))
                            else:
                                vals.append(ast.unparse(e))
                        fields['output_variables'] = vals
            if fields:
                print('CODE_variable_kind=' + str(fields.get('variable_kind')))
                print('CODE_outputs=' + ','.join(fields.get('output_variables', [])))
                break
