"""Conservative CPU compilation gate for the bounded T4 shortlist."""
import argparse
import json
from pathlib import Path
from tp2_kitchen_matrix import configurations


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    lowering=json.loads((a.root/'lowering.json').read_text())['cases']
    compiled={r['source']:r for r in json.loads((a.root/'compilation.json').read_text())['cases']}
    selected={};rejected=[]
    for component in ('te','dit','video'):
        selected[component]=[]
        for cfg in configurations(component):
            rows=[r for r in lowering if r['component']==component and r['name']==cfg['name']]
            if cfg.get('baseline') or cfg.get('native') or cfg['name']=='fused_prepare':
                selected[component].append(cfg);continue
            errors=[]
            for row in rows:
                c=compiled.get(row.get('source'))
                if row['status']!='lowered' or not c or c['exit_code'] or c['spill_store_bytes'] or c['spill_load_bytes'] or c['stack_bytes'] or c['registers_per_thread']>=240:
                    errors.append(dict(lowering=row,compilation=c))
            if errors or len(rows)!=(2 if component=='video' else 1):rejected.append(dict(component=component,config=cfg,reasons=errors))
            else:selected[component].append(cfg)
        prep=[r for r in lowering if r['component']==component and r['name']=='fused_prepare_prep']
        assert prep and all(r['status']=='lowered' and compiled[r['source']]['exit_code']==0 for r in prep),'Fused preparation must compile before submission'
    out=dict(configurations=selected,rejected=rejected,
             scope='Offline CPU compiler resource screen. This is not GPU conformance or performance evidence.')
    (a.root/'shortlist.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
