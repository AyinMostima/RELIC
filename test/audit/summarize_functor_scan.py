import json
from pathlib import Path

req = {
    'name','version','description','category','execution_phase',
    'mathematical_expression','references','parameters',
    'input_variables','output_variables','conservation_laws',
    'spatial_scale','variable_kind'
}
rows = json.loads(Path('analysis_results/functor_scan.json').read_text(encoding='utf-8'))
missing = []
sig = []
for r in rows:
    miss = sorted(req - set(r.get('metadata_fields', [])))
    if miss:
        missing.append({'file': r['file'], 'class': r['class'], 'line': r['line'], 'missing': miss})
    fs = r.get('forward_sig')
    if fs is not None and fs != ['self','state','environment','time','dt']:
        sig.append({'file': r['file'], 'class': r['class'], 'line': r['line'], 'sig': fs})
out = {'missing_metadata': missing, 'forward_signature_mismatch': sig}
Path('analysis_results/functor_audit_summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
print('missing_metadata=', len(missing))
print('forward_signature_mismatch=', len(sig))
