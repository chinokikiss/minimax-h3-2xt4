"""Classify downloaded, version-bound T4 evidence without querying Kaggle."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def kernel_totals(path):
    events=read(path).get('traceEvents',[]);duration=Counter();counts=Counter()
    for event in events:
        if event.get('cat')=='kernel' and 'dur' in event:
            name=event.get('name','unknown');duration[name]+=event['dur'];counts[name]+=1
    groups=Counter()
    for name,total in duration.items():
        if 'nccl' in name:group='NCCL_collectives_including_wait'
        elif 'attn_kernel' in name:group='fused_attention'
        elif 'cutlass' in name:group='CUTLASS_GEMM'
        elif 'gemm' in name:group='other_GEMM'
        elif 'quant' in name:group='quantize_or_dequantize'
        elif name=='main_kernel':group='generic_main_kernel_unattributed'
        else:group='other_norm_pointwise_copy'
        groups[group]+=total
    return dict(scope='Sum of recorded kernel durations, not critical-path latency or hardware counters; grouping is kernel-name based',
        total_kernel_duration_us=sum(duration.values()),total_kernel_launches=sum(counts.values()),group_duration_us=dict(groups),
        kernels=[dict(name=name,total_us=total,calls=counts[name]) for name,total in duration.most_common(15)])


def classify(root,cpu):
    results={};files={p.name:p for p in root.rglob('*.json') if not p.name.endswith('_trace.json')}
    batch=read(files['batch_results.json']) if 'batch_results.json' in files else {}
    devices=batch.get('hardware',{}).get('devices',[])
    physical=(len(devices)==2 and len({d.get('uuid') for d in devices})==2
              and all(d.get('uuid') and d.get('capability')==[7,5] and 'T4' in d.get('name','') for d in devices))
    def pair(stem):
        paths=[files.get(stem+f'_rank{rank}.json') for rank in (0,1)]
        records=[read(p) for p in paths if p]
        return records,physical and len(records)==2 and all(r.get('status')=='passed' for r in records)
    def single(name):
        path=files.get(name);record=read(path) if path else {}
        return record,physical and record.get('status')=='passed'
    te,te_ok=pair('te_flow');di,di_ok=pair('dit_full_flow')
    late,late_ok=pair('dit_late_lora_flow');mid,mid_ok=pair('dit_mid_flow')
    def flow_complete(records):
        return all({'native_fp32_flow','fp16_fold_c64'} <= {c.get('name') for c in r.get('cases',[])} for r in records)
    te_ok=te_ok and flow_complete(te);di_ok=di_ok and flow_complete(di)
    late_ok=late_ok and flow_complete(late);mid_ok=mid_ok and flow_complete(mid)
    results['T01']=dict(status='executed_component_scope_accuracy_not_accepted' if all((te_ok,di_ok,late_ok,mid_ok)) else 'partial_or_missing',
        finite_gate_passed=all((te_ok,di_ok,late_ok,mid_ok)),
        cases=[dict(rank=r['rank'],distribution=r.get('distribution'),cases=[
            {key:c[key] for key in ('name','output','vs_native_fp32_flow','steady_state_ms','peak_allocated_bytes') if key in c}
            for c in r.get('cases',[])]) for r in te+di+mid+late],
        scope='Trained block 0 TE; DiT 0 full retained input plus 24/49 reused-input stress; LoRA at 49. No full trajectory or frontend-fold acceptance')
    attention,attention_ok=pair('attention_gates')
    dispatch=[]
    for record in te+di+late+mid:
        for case in record.get('cases',[]):
            if case['name'] not in ('native_fp32_flow','fp16_fold_c64'):continue
            dispatch.append(dict(rank=record['rank'],distribution=record.get('distribution'),flow=case['name'],
                dtype=case['dtype'],divisor=case['divisor'],dispatch=case.get('dispatch',{}),
                int8_calls=case.get('int8_calls',{}),timing=case.get('steady_state_ms'),
                numerical=case.get('vs_native_fp32_flow'),ranges=case.get('ranges')))
    results['T02']=dict(status='passed_component_scope' if attention_ok and te_ok and di_ok else 'partial_or_missing',
        dispatch=dispatch,guard_note='Wrapper selection is not kernel execution. swiglu_pad/_quantize_owned counts identify actual optimized preparation; outer row dispatch can fall back.')
    if attention_ok:
        results['T02']['standalone_attention']=[dict(component=a['component'],scheme=a['scheme'],dtype=a['dtype'],
            q=a['q'],median_dual_device_complete_api_ms=statistics.median(max(x,y) for x,y in zip(
                a['timing']['samples_ms'],b['timing']['samples_ms'])),
            rank_records=[a,b]) for a,b in zip(attention[0]['records'],attention[1]['records'])]
    rows,rows_ok=pair('row_gates')
    projections=[]
    if rows_ok:
        cases=[]
        for record in rows:
            cases.append(next(r['cases'] for r in record['records'] if r['component']=='scaled_fp16_row_deployment_shapes'))
        for first,second in zip(*cases):
            assert first['name']==second['name']
            medians={}
            for backend in ('tilelang_packed','tilelang_scaled_fp16'):
                a=first['steady_state_complete_api'][backend]['samples_ms']
                b=second['steady_state_complete_api'][backend]['samples_ms']
                medians[backend]=statistics.median(max(x,y) for x,y in zip(a,b))
            projections.append(dict(name=first['name'],M=first['M'],K=first['K'],N=first['N'],
                median_dual_rank_complete_api_ms=medians,
                speedup=medians['tilelang_packed']/medians['tilelang_scaled_fp16'],
                numerical=[r.get('scaled_transport_vs_native_int32') for r in (first,second)]))
    results['T03']=dict(status='passed_component_scope' if rows_ok else 'failed_or_missing',projections=projections,
        scope='FP32 eligible row path, physical NCCL four-site independent codec and full TP2 row dimensions; not FP16-default transport activation or generation acceptance')
    refiner,refiner_ok=single('refiner.json')
    lifetime=[c.get('lifetime_ablation') for r in te for c in r.get('cases',[]) if c['name']=='fp16_fold_c64']
    results['T04']=dict(status='passed_component_scope' if refiner_ok and te_ok and len(lifetime)==2 and all(x and x['byte_exact'] for x in lifetime) else 'partial_or_missing',
        refiner=refiner,lifetime=lifetime)
    residency,residency_ok=pair('residency_gates')
    results['T05']=dict(status='passed_component_scope' if residency_ok else 'failed_or_missing',records=residency,
        scope='Two physical device windows, real FP16 workspace and configured headroom thresholds; not full TE throughput')
    bake,bake_ok=single('bake_results.json')
    results['T06']=dict(status='passed_component_scope' if bake_ok else 'failed_or_missing',results=bake,
        missing_measurements=['Peak GPU memory and peak host RSS were not sampled; no full-model throughput claim'])
    vae,vae_ok=single('vae_idle.json');names={r.get('component') for r in vae.get('records',[])}
    seven={'idle_encoder_and_te_coexist','staged_video_encode','staged_audio_encode','idle_cancel_and_pressure_release'}
    eight={'short_query','query_split','temporal_query','video_native_admission_fallback','native_single_gpu_audio'}
    results['T07']=dict(status='passed_component_scope' if vae_ok and seven<=names else 'partial_or_missing',
        scope='Actual parent/worker ownership on two T4s, one TE preload layer, trained bounded VAE operations; Native arithmetic policy isolates lifecycle',
        results=vae)
    results['T08']=dict(status='passed_component_scope' if vae_ok and eight<=names else 'partial_or_missing',
        scope='Trained bounded Video query/native fallback and Native single-GPU Audio; no full video/audio quality claim')
    pread_path=cpu/'native_profile/linux_pread.json'
    pread=read(pread_path) if pread_path.is_file() else {}
    results['T09']=dict(status='passed_cpu_linux_scope' if pread.get('status')=='passed' else 'missing',results=pread)
    traces={p.name:kernel_totals(p) for p in root.rglob('*_trace.json')}
    return dict(schema=1,physical_t4_evidence=physical,batch=batch,validation_dimensions=results,
                profiler_kernel_totals=traces,scope='No full H3, sparse semantic/distribution acceptance or complete FP16 trajectory qualification')


def verify_download(root):
    manifest=root/'download_manifest.json'
    if not manifest.exists():raise RuntimeError('A version-bound download manifest is required: '+str(root))
    downloaded=read(manifest)
    for row in downloaded['files']:
        path=root/row['file']
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=row['sha256']:raise RuntimeError('Downloaded evidence changed: '+str(path))
    return downloaded


def validate_reuse(result,root,original):
    binding=verify_download(root)
    folder=root/'t4_validation';batch=read(folder/'batch_results.json')
    with (original/'t4_validation/batch_results.json').open('rb') as stream:prior=hashlib.file_digest(stream,'sha256').hexdigest()
    if batch.get('qualification_reference',{}).get('receipt_sha256')!=prior:
        raise RuntimeError('Follow-up is not bound to the reviewed original batch')
    devices=batch['hardware']['devices']
    if len(devices)!=2 or len({d['uuid'] for d in devices})!=2 or any(d['capability']!=[7,5] or 'T4' not in d['name'] for d in devices):
        raise RuntimeError('Follow-up lacks distinct physical SM75 devices')
    for key in ('torch','kitchen','tilelang'):
        if batch['hardware'][key]!=result['batch']['hardware'][key]:raise RuntimeError('Follow-up stack drift: '+key)
    original_sources=read(original/'t4_validation/source_hashes.json');sources=read(folder/'source_hashes.json')
    for name,digest in original_sources.items():
        if name.startswith('patches/') and sources.get(name)!=digest:raise RuntimeError('Runtime source changed: '+name)
    return folder,batch,binding


def merge_followup(result,root,original):
    folder,batch,binding=validate_reuse(result,root,original)
    refiner=read(folder/'refiner.json');vae=read(folder/'vae_idle.json');dimensions=result['validation_dimensions']
    if refiner['status']=='passed' and {r['rope'] for r in refiner['records'] if r['byte_exact']}=={False,True}:
        dimensions['T04'].update(refiner=refiner)
        if len(dimensions['T04']['lifetime'])==2 and all(r['byte_exact'] for r in dimensions['T04']['lifetime']):
            dimensions['T04']['status']='passed_component_scope'
    def passed(row):return row.get('status')!='failed' and row.get('byte_exact',row.get('status')=='passed')
    successful={r['component'] for r in vae['records'] if passed(r)}
    required={'idle_encoder_and_te_coexist','staged_video_encode','staged_audio_encode','idle_cancel_and_pressure_release'}
    dimensions['T07'].update(results=vae,status='passed_component_scope' if required<=successful else 'partial_or_missing')
    dimensions['T08'].update(status='failed_video_split_identity' if any(r.get('status')=='failed' for r in vae['records']) else 'partial_or_missing',
        records=vae['records'],native_audio='passed_component_scope' if 'native_single_gpu_audio' in successful else 'missing',
        native_video_fallback='passed_component_scope' if 'video_native_admission_fallback' in successful else 'missing')
    result['component_followup']=dict(batch=batch,download_binding=binding)
    result['nsight_systems_statistics']={}
    for path in folder.glob('v1_te_stats_*.csv'):
        with path.open(newline='',encoding='utf-8') as stream:
            result['nsight_systems_statistics'][path.name]=list(csv.DictReader(stream))


def merge_diagnosis(result,root,original):
    folder,batch,binding=validate_reuse(result,root,original)
    records=[read(folder/f'fp16_lora_diagnosis_rank{rank}.json') for rank in (0,1)]
    expected={(False,'native'),(False,'fp16'),(True,'native'),(True,'fp16'),(True,'fp16_wide_fc2'),(True,'fp16_wide_rows')}
    for record in records:
        if record['status']!='completed_diagnostic_not_quality_acceptance' or {(c['lora'],c['mode']) for c in record['cases']}!=expected:
            raise RuntimeError('Controlled stage diagnosis incomplete')
    result['fp16_stage_diagnosis']=dict(batch=batch,records=records,download_binding=binding,
        interpretation='FP16 SwiGLU range failure localized to FC2; widening the complete FC2 path mitigates it. This ablation also widens rotation/quantization/epilogue/LoRA and is not an activation-only isolation, performance measurement, production fix or trajectory acceptance.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--cpu',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--followup',type=Path)
    parser.add_argument('--diagnosis',type=Path)
    args=parser.parse_args();result=classify(args.data,args.cpu)
    result['download_binding']=verify_download(args.data)
    if args.followup:merge_followup(result,args.followup,args.data)
    if args.diagnosis:merge_diagnosis(result,args.diagnosis,args.data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v['status'] for k,v in result['validation_dimensions'].items()},indent=2))


if __name__=='__main__':main()
