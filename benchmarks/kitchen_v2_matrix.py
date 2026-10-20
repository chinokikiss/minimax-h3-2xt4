"""Fixed next-pass matrix against the previously measured T4 winners."""
def configurations(component):
    tile,cache,threads,warp=(64,'q',256,'row') if component=='dit' else ((64,'group',128,'square') if component=='video' else (32,'group',128,'square'))
    base=dict(tile=tile,cache=cache,threads=threads,warp=warp)
    rows=[dict(base,name='current',baseline=True,prep='previous'),
          dict(base,name='register_scales',prep='register'),
          dict(base,name='castpad',prep='castpad')]
    if component=='video':
        rows += [dict(base,name='finite_epilogue',prep='previous',finite=True),
                 dict(base,name='combined',prep='castpad',finite=True),dict(name='kitchen',native=True)]
    if component=='dit':
        rows += [dict(base,name='row_scores',prep='previous',row_scores=True),
                 dict(base,name='staged_scales',prep='previous',staged_scales=True)]
    return rows


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--static-gate',action='store_true');a=ap.parse_args()
    rows={c:configurations(c) for c in ('te','dit','video')};rejected=[]
    if a.static_gate:
        lowered=json.loads((a.output.parent/'lowering.json').read_text())['cases']
        compiled={r['source']:r for r in json.loads((a.output.parent/'compilation.json').read_text())['cases']}
        for component,configs in rows.items():
            for cfg in list(configs):
                names=['register_prep'] if cfg.get('prep') in ('register','castpad') else []
                if cfg.get('prep')=='castpad':names+=['castpad']
                if cfg.get('row_scores'):names+=['row_scores']
                if cfg.get('staged_scales'):names+=['staged_scales']
                if cfg.get('finite'):names+=['finite_epilogue']
                tests=[r for r in lowered if r['component']==component and r['name'] in names]
                expect=len(names)*(2 if component=='video' else 1)
                reasons=[]
                if len(tests)!=expect:reasons.append('missing lowering')
                for r in tests:
                    c=compiled.get(r.get('source'),{})
                    if r['status']!='lowered' or c.get('exit_code',1) or c.get('spill_store_bytes',1) or c.get('spill_load_bytes',1) or c.get('stack_bytes',1) or c.get('registers_per_thread',256)>=240:reasons.append(dict(lowering=r,compilation=c))
                if reasons:configs.remove(cfg);rejected.append(dict(component=component,config=cfg,reasons=reasons))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(configurations=rows,rejected=rejected,scope='CPU resource gate only; no runtime conformance or performance inference.'),indent=2))
    print('MATRIX', {k:len(v) for k,v in rows.items()},'rejected',len(rejected))
