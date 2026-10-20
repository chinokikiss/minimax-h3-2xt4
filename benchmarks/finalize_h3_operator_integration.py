"""CPU-only final source/workflow/evidence manifest; no GPU work or network."""
import ast
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from patches.h3_operator_policy import H3OperatorPolicy

evidence=Path('D:/attention_eval/h3_operator_integration_20261018')
result=json.loads((evidence/'integration_results.json').read_text())
assert result['status']=='pass' and result['provider_status'].endswith('.COMPLETE')
assert len(result['local_pipeline_wiring']['records'])==4
assert all(r['status'] in ('pass','bitwise_pass') for r in result['local_pipeline_wiring']['records'])
assert all(r['status']=='pass' for r in result['local_gates']['records'])
assert version('comfy-kitchen')=='0.2.37'

paths=['patches/h3_operator_policy.py','patches/h3_operator_dependencies.py','patches/tilelang_group32_masked.py',
    'patches/comfy_tp_runtime.py','patches/windowed_tp.py','patches/tp_qwen3vl.py','patches/tp_dit.py',
    'patches/h3_lora.py','patches/multi_gpu_audio_vae.py','patches/parallel_vae.py',
    'patches/tilelang_row_quant.py','patches/tilelang_swiglu_pad.py','patches/tilelang_attention_p2.py',
    'custom_nodes/minimax_h3_tp2/__init__.py','build_comfy_tp2_workflow.py','build_h3_operator_integration_job.py',
    'benchmarks/test_h3_operator_integration.py','benchmarks/test_h3_pipeline_wiring.py',
    'benchmarks/archive_h3_operator_integration.py','benchmarks/finalize_h3_operator_integration.py',
    'requirements-tp2-operators.txt','H3_OPERATOR_INTEGRATION.md','workflows/video_minimax_h3_r2v_tp2_operators.json']
hashes={}
for name in paths:
    path=ROOT/name
    if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'),filename=name)
    hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()

workflow=ROOT/'workflows/video_minimax_h3_r2v_tp2_operators.json'
deployed=ROOT.parent/'ComfyUI-master/user/default/workflows'/workflow.name
assert workflow.read_bytes()==deployed.read_bytes()
node=ROOT.parent/'ComfyUI-master/custom_nodes/minimax_h3_tp2'
assert node.resolve()==(ROOT/'custom_nodes/minimax_h3_tp2').resolve()
result['default_operator_policy']=asdict(H3OperatorPolicy())
result['fixed_defaults']=dict(video_attention='kitchen',audio='channel_parallel',transport='native_int32')
result['final_source_sha256']=hashes
result['deployed_workflow']=str(deployed)
result['deployed_node']=str(node)
result['local_environment']=dict(torch=version('torch'),comfy_kitchen=version('comfy-kitchen'),tilelang=version('tilelang'))
result['final_checks']=dict(syntax='pass',workflow_copy='identical',node_junction='pass',
    source_and_runtime_provider_gate='pass',full_H3_generation='not_run',end_to_end_performance='unmeasured')
(evidence/'integration_results.json').write_text(json.dumps(result,indent=2))
(ROOT/'H3_OPERATOR_INTEGRATION_RESULTS.json').write_text(json.dumps(result,indent=2))
manifest=[]
for p in [ROOT/'H3_OPERATOR_INTEGRATION_RESULTS.json',*[ROOT/n for n in paths]]:
    manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(evidence/'deliverable_manifest.json').write_text(json.dumps(manifest,indent=2))
print('H3_INTEGRATION_FINAL_PASS',len(paths),'source/artifact checks;',sum(len(r['records']) for r in result['ranks']),
    'target rank gates;',len(result['local_gates']['records']),'local gates; 4 pipeline wiring gates')
