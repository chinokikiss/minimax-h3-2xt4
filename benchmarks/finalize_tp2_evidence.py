"""Verify version-bound downloads and assemble the completed operator record."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path('D:/attention_eval/operator_optimization_tp2_20261016')
    sessions=[('v1','tp2_operators','chinokiki9/h3-tp2-operator-screen','v1'),
              ('video_v1','video_tp2','chinokiki9/h3-tp2-video-microchain','v1'),
              ('video_v2','video_tp2','chinokiki9/h3-tp2-video-microchain','v2')]
    result={'status':'complete','full_pipeline_runs':0,'production_dispatch_changed':False,'scalar_pooling_retained':True,'sessions':[]}
    stage_seconds=0
    for directory,folder,kernel,version in sessions:
        base=root/directory
        download=json.loads((base/'download_manifest.json').read_text())
        for f in download['files']:
            p=base/f['file'];assert p.stat().st_size==f['bytes']
            assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==f['sha256']
        state=json.loads((base/folder/'status.json').read_text());assert state['status']=='complete'
        assert all(c.get('exit_code')==0 for c in state['commands'])
        assert all(d['capability']==[7,5] and 'T4' in d['name'] for d in state['hardware'])
        seconds=sum(c['seconds'] for c in state['commands']);stage_seconds+=seconds
        result['sessions'].append({'kernel':kernel,'version':version,'status':state['status'],
            'verified_downloads':len(download['files']),'recorded_stage_wall_seconds':seconds})
    result['recorded_stage_wall_seconds']=stage_seconds
    result['counter_blocker']='Previous physical-T4 Nsight Compute ERR_NVGPUCTRPERM; no current counter data, no new attempt.'
    result['video_confirmation']=[]
    for rank in range(2):
        d=json.loads((root/'video_v2/video_tp2/unprofiled'/f'rank{rank}.json').read_text());assert d['status']=='complete'
        for r in d['cases']:
            assert r['microchain']['repeats']==15 and r['microchain']['warmup_actual_calls']==10
            result['video_confirmation'].append({'rank':rank,'candidate':r['candidate'],
                'attention_API_ms':r['attention_API']['gpu_median_ms'],'microchain_ms':r['microchain']['gpu_median_ms']})
    (root/'execution_state.json').write_text(json.dumps(result,indent=2));print('VERIFIED',len(sessions),'completed sessions;',round(stage_seconds/60,2),'recorded stage minutes; no active jobs')


if __name__=='__main__':main()
