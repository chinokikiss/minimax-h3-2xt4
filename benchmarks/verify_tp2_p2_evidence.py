"""CPU-only final integrity gates; avoid upload sidecars and signed download URLs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=Path('D:/attention_eval/tp2_p2_20261018')
VERSIONS={'transport':(1,2),'attention':(1,2,3),'restoration':(1,),
          'transport-confirm':(1,),'scalar-profile':(1,),'counter-probe':(1,)}


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    gates=[];files={};units=0;cross_process=0
    for kind,versions in VERSIONS.items():
        for version in versions:
            p=EVIDENCE/kind/('v'+str(version));provider=p/'provider';runtime=p/'p2'
            manifest=json.loads((p/'download_manifest.json').read_text())
            for row in manifest['files']:
                path=p/row['file'];assert path.resolve().is_relative_to(p.resolve())
                assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
                files[str(path)]=dict(bytes=row['bytes'],sha256=row['sha256'])
            identity=json.loads((provider/'identity.json').read_text())
            notebook=provider/'experiment.ipynb'
            assert hashlib.sha256(notebook.read_text(encoding='utf8').encode()).hexdigest()==identity['provider_notebook_sha256']
            for name,expected_hash in json.loads((provider/'source_hashes.json').read_text()).items():
                path=provider/'submitted_source'/name
                # Windows extraction normalizes physical newlines; runtime hashes refer to UTF-8 source text.
                assert hashlib.sha256(path.read_text(encoding='utf8').encode()).hexdigest()==expected_hash
                files[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
            for path in (provider/'identity.json',provider/'source_hashes.json',notebook):
                files[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
            if kind!='counter-probe':
                reference=json.loads((provider/'source_hashes.json').read_text())
                assert json.loads((runtime/'source_hashes.json').read_text())==reference
                # Source-hash values describe provider-embedded bytes, not current checkout files.
                gates.append(dict(kind=kind,version=version,downloaded_files=len(manifest['files']),provider_runtime_sources='pass'))
                status=json.loads((runtime/'status.json').read_text())
                assert len(status['hardware'])==2 and all(h['name']=='Tesla T4' and h['capability']==[7,5] for h in status['hardware'])
                for unit in runtime.rglob('unit_rank*.json'):
                    data=json.loads(unit.read_text());assert isinstance(data,list)
                    for row in data:
                        assert row.get('status',row.get('gate')) in ('pass','bitwise_pass'),(unit,row)
                        units+=1
                for rank in (0,1):
                    screen=runtime/'screen'/f'rank{rank}.json';repeat=runtime/'repeat'/f'rank{rank}.json'
                    if screen.exists() and repeat.exists():
                        a=json.loads(screen.read_text());b=json.loads(repeat.read_text())
                        original={(r['family'],r['name']):r for r in a['cases'] if r.get('status')=='complete'}
                        for r in b['cases']:
                            key=(r['family'],r['name'])
                            if r.get('status')=='complete' and key in original:
                                assert original[key]['output_sha256']==r['output_sha256'];cross_process+=1
            else:
                status=json.loads((p/'ncu_probe/status.json').read_text())
                assert status['status']=='blocked_ERR_NVGPUCTRPERM'
                assert len(status['hardware'])==2 and all(h['capability']==[7,5] for h in status['hardware'])
    # Confirmation must use the same actual output contract, not a new fixture approximation.
    for rank in (0,1):
        first=json.loads((EVIDENCE/'transport/v2/p2/screen'/f'rank{rank}.json').read_text())
        last=json.loads((EVIDENCE/'transport-confirm/v1/p2/screen'/f'rank{rank}.json').read_text())
        expected={(r['family'],r['name']):r['output_sha256'] for r in first['cases'] if r['status']=='complete'}
        for row in last['cases']:assert row['output_sha256']==expected[(row['family'],row['name'])]
    traces=list(EVIDENCE.rglob('*.nsys-rep'));assert len(traces)==28
    attributed=0
    for path in EVIDENCE.rglob('trace_attribution.json'):
        for row in json.loads(path.read_text()):
            assert row['unattributed']==0;attributed+=1
    assert attributed==len(traces)
    source_names=['patches/tilelang_transport_p2.py','patches/tilelang_attention_p2.py',
        'build_tp2_p2_job.py','build_tp2_p2_ncu_probe.py']
    source_names += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'benchmarks').glob('*tp2_p2*.py')]
    for name in source_names:
        path=ROOT/name;files[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
    output=dict(status='pass',source_identity_gates=gates,independent_rank_unit_records=units,
        fresh_process_output_checks=cross_process,confirmation_primary_output_match=True,
        systems_reports=28,unattributed_kernels=0,artifact_hashes=files,
        interpretation='Integrity and operator arithmetic only; neither statistical generalization nor full-model quality acceptance.')
    (EVIDENCE/'final_verification.json').write_text(json.dumps(output,indent=2))
    print('P2_EVIDENCE_PASS',len(files),'hashed artifacts,',units,'independent rank-unit records,',cross_process,'fresh-process output checks, 28 Systems reports')


if __name__=='__main__':main()
