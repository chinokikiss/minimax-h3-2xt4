"""Freeze current source only; never upload local model/checkpoint contents."""
import ast
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]
SHORT='''Use <Picture 1> as the exact character identity, costume, and visual style reference.

A cinematic anime scene on a rain-soaked city rooftop at night. The young superhero from <Picture 1> stands near the edge of the roof, his red cape gently fluttering in the wind.

In a single continuous shot, the camera slowly pushes toward the character. He turns his head and looks directly at the camera. Reflections of red neon signs shimmer in puddles behind him. His hair and cape respond naturally to the wind.

Preserve the character's facial features, red costume, black mask, and anime illustration style.

Audio: soft wind blowing across the rooftop, distant city traffic, subtle fabric rustling, and light water dripping. No dialogue, no background music.'''
LONG='''Use <Picture 1> as the exact reference for the young superhero's identity, hairstyle, face, mask, red costume, and flowing cape. Use <Picture 2> as the exact reference for the enormous mechanical dragon's appearance, black metallic armor, glowing red energy core, and electrical effects.

A spectacular cinematic anime action sequence in a futuristic city during a violent nighttime thunderstorm.

0-4 seconds: A wide establishing shot reveals the superhero standing on a skyscraper rooftop. The enormous mechanical dragon rises between distant buildings. Rain falls heavily, and electric-blue lightning illuminates the skyline. The camera slowly moves forward.

4-9 seconds: The dragon spreads its mechanical wings and releases a deafening roar. Its red energy core pulses brightly. The superhero braces against a powerful gust of wind, his cape whipping dramatically. The camera tracks sideways as lightning arcs between the buildings.

9-15 seconds: The superhero runs forward and leaps across the rooftop. The camera follows his movement in a dynamic tracking shot while the dragon turns toward him. A massive lightning flash briefly illuminates both characters. The scene ends with the superhero landing in a determined stance.

Maintain consistent character identity, costume design, mechanical dragon anatomy, lighting, and anime art style throughout the entire sequence. Smooth continuous motion, coherent spatial relationships, detailed background, dramatic cinematic lighting.'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--l1-offline-memory',action='store_true')
    parser.add_argument('--runtime-dataset',help='Private dataset containing the frozen user conda/source bundle and Linux wheelhouse')
    parser.add_argument('--validate-only',action='store_true',help='Check generated source without writing or submitting a notebook')
    options=parser.parse_args()
    if not options.runtime_dataset and not options.validate_only:parser.error('--runtime-dataset is required for a runnable notebook')
    policy=dict(te_attention='tilelang_group32',dit_attention='kitchen',row_linear_backend='tilelang_packed',
        swiglu_padding=True,selective_b=True,scalar_score_workspace_mib=96,fc2_quantizer='owned_fp32')
    short=dict(prompt=SHORT,frames=124,seed=20261019,steps=20,sampler='res_multistep',
        audio_shift=3.,lora=False,reserve_gib=4.5,operators=policy)
    plan=dict(S1=short|dict(name='S1',idle=False),S2=short|dict(name='S2',idle=True),
        L1=dict(name='L1',prompt=LONG,frames=362,seed=20261020,steps=4,sampler='lcm',audio_shift=2.,
            lora=True,idle=False,reserve_gib=9.,operators=policy|dict(dit_attention='tilelang_scalar')))
    if options.l1_offline_memory:
        plan={'L1':plan['L1']|dict(offline_lora=True,memory_profile=True)}
    names=[p.relative_to(ROOT).as_posix() for p in (ROOT/'patches').glob('*.py')]
    names += ['benchmarks/'+n for n in ('ref2va_e2e_run.py','ref2va_e2e_bootstrap.py',
        'pipeline_residency_audit.py','full_models.py','retest.py','fixture_inputs.py','ref2va_workloads.py','frozen_kaggle_setup.py')]
    names+=['semantic_results/pipeline_checkpoint_identity.json','runtime/local_conda.json']
    if options.l1_offline_memory:
        names+=['benchmarks/'+n for n in ('merge_h3_lora.py','offline_merge_reference.py',
            'gpu_memory_overhead.py','analyze_gpu_memory_overhead.py','verify_offline_tp_checkpoint.py')]
    files={name:(ROOT/name).read_bytes() for name in names}
    files['ref2va_e2e_plan.json']=json.dumps(plan,indent=2).encode()
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    files['source_binding.json']=json.dumps(hashes,indent=2).encode()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in files.items():
            if name.endswith('.py'):ast.parse(data.decode('utf-8-sig'),filename=name)
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    code="""import base64,io,pathlib,runpy,tarfile
root=pathlib.Path('/kaggle/working/ref2va_e2e_source');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:archive.extractall(root,filter='data')
runpy.run_path(str(root/'benchmarks/ref2va_e2e_bootstrap.py'),run_name='__main__')
""".replace('PAYLOAD',repr(base64.b64encode(buffer.getvalue()).decode()))
    assert len(code)<1_000_000
    ast.parse(code)
    if options.validate_only:
        print('FROZEN_REF2VA_BUILDER_VALID',len(files),'source files; no notebook or generation was run')
        return
    folder=ROOT/('kaggle_l1_offline_memory_frozen' if options.l1_offline_memory else 'kaggle_ref2va_e2e_frozen');folder.mkdir(exist_ok=True)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(id='full-ref2va',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    (folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf8')
    (folder/'source_binding.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    (folder/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(
        id='ocnsai/h3-l1-offline-lora-memory-frozen' if options.l1_offline_memory else 'ocnsai/h3-ref2va-end-to-end-frozen',
        title='H3 L1 Offline LoRA Memory' if options.l1_offline_memory else 'H3 REF2VA End To End',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_tpu=False,enable_internet=True,dataset_sources=[options.runtime_dataset],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2),encoding='utf8')
    print('SOURCE_ONLY',folder,'compressed_source_bytes',len(buffer.getvalue()),'notebook_code_bytes',len(code))


if __name__=='__main__':main()
