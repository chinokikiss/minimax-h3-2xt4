"""Trained whole-layer LoRA bake on two distinct T4s; no generation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import statistics
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI')]


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--lora',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args();sys.argv=sys.argv[:1]
    import torch
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in range(2)):
        raise RuntimeError('Two physical T4s required')
    from patches.legacy_runtime import validate_source,validate_packages
    validate_source(ROOT/'ComfyUI');validate_packages()
    from patches.offline_lora import convert
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    report=dict(status='running',scope='Three retained trained DiT layers, full-width matrices; no complete model bake or generation',
        devices=[dict(name=torch.cuda.get_device_name(i),uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(2)],
        records=[])
    args.output.mkdir(parents=True,exist_ok=False)
    base=digest(args.checkpoint)
    try:
        for iteration in range(args.repeats+1):
            for name,devices in [('serial',('cuda:0',)),('parallel',('cuda:0','cuda:1'))]:
                start=time.perf_counter();stem=f'{name}_{iteration}'
                path=convert(args.checkpoint,((str(args.lora),1.),),args.output/(stem+'.safetensors'),
                    verify_native=iteration==0,devices=devices)
                record=json.loads(Path(path+'.json').read_text())
                record.update(name=name,iteration=iteration,warmup=iteration==0,
                    seconds=time.perf_counter()-start,sha256=digest(path))
                report['records'].append(record)
                assert record['sha256']==report['records'][0]['sha256']
                Path(path).unlink()
        assert digest(args.checkpoint)==base
        owners={r['device'] for r in report['records'][1]['tensors']}
        assert owners=={'cuda:0','cuda:1'}
        medians={name:statistics.median(r['seconds'] for r in report['records'] if r['name']==name and not r['warmup'])
                 for name in ('serial','parallel')}
        report.update(status='passed',serial_parallel='byte_exact',base_unchanged=True,
            steady_whole_bake_seconds=medians,speedup=medians['serial']/medians['parallel'],
            timing_boundary='Production convert call including I/O and fresh workers; independent verification is in the excluded warmup pair',
            limitation='Three layers partition 2:1; this is not full-model throughput')
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        (args.output/'bake_results.json').write_text(json.dumps(report,indent=2))
        # Receipts and identities suffice; avoid publishing multi-GB baked copies.
        for name in ('serial','parallel'):
            for index in range(args.repeats+1):(args.output/f'{name}_{index}.safetensors').unlink(missing_ok=True)


if __name__=='__main__':main()
