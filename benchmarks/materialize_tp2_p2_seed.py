"""Target-side real fixture reconstruction, before operator timing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import torch
from safetensors import safe_open
from fetch_h3_physical_fixtures import main as fetch_main
from patches import optimized_tp as native,parallel_int8 as pi


def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=Path,required=True);ap.add_argument('--captures',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=True);manifest=json.loads((args.seed/'seed_manifest.json').read_text())
    sys.argv=[sys.argv[0],str(args.seed/'weight_plan.json'),str(args.output)];fetch_main()
    seed_path=args.seed/'fc1_seed.pt';assert digest(seed_path)==manifest['seed_sha256']
    data=torch.load(seed_path,map_location='cpu',weights_only=True);weights={}
    with safe_open(args.output/'transport.safetensors',framework='pt',device='cpu') as f:
        for key in f.keys():weights[key]=f.get_tensor(key)
    def subset(prefix):return {k.removeprefix(prefix):v for k,v in weights.items() if k.startswith(prefix)}
    fw=subset('blocks.0.mlp.fc1.')
    torch.cuda.set_device(0);torch.set_grad_enabled(False)
    x=pi.dequantize(native.native_int_mm(data['q'].cuda(),fw['weight'].cuda()),data['scale'].cuda(),fw['weight_scale'].cuda(),torch.float32).cpu()
    assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==data['expected_x_sha256'],'Real FC2 input does not match original native capture on target hardware'
    records=[]
    for original in manifest['raw_fixtures']['records']:
        if original['name'].startswith('fc2'):xx=x;ww=subset('blocks.0.mlp.fc2.')
        else:
            call=int(original['name'].split('_s')[-1]);path=args.captures/f'dit_layer25_call{call:02d}.pt'
            if not path.exists():
                print('ACCURACY_ENDPOINT_NOT_ATTACHED',original['name'],'not included in the previously published TP2 fixture subset',flush=True)
                continue
            assert digest(path)==original['provenance']['source_sha256']
            y=torch.load(path,map_location='cpu',weights_only=True)['native_output'];xx=y.transpose(1,2).reshape(y.shape[0]*y.shape[2],-1);ww=subset('blocks.25.attn.out_proj.')
        dest=args.output/original['file'];torch.save(dict(x=xx,weights=ww,provenance=original['provenance']),dest)
        record={**original,'sha256':digest(dest),'bytes':dest.stat().st_size,'reconstructed_on_target':True,'original_local_fixture_sha256':original['sha256']}
        records.append(record)
    (args.output/'real_transport_manifest.json').write_text(json.dumps(dict(records=records,seed=manifest,FC2_input_bitwise_hash_gate='pass',
        missing_endpoints=[r['name'] for r in manifest['raw_fixtures']['records'] if r['name'] not in {s['name'] for s in records}]),indent=2))
    print('TARGET_REAL_FIXTURES_RECONSTRUCTED_AND_GATED',len(records),flush=True)


if __name__=='__main__':main()
