"""Plan only missing Native auxiliary ranges; reuse the published base blocks."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import shutil

ROOT=Path(__file__).resolve().parents[1]


def auxiliary(path,kind):
    from patches.windowed_tp import CheckpointReader
    reader=CheckpointReader(path)
    prefix='model.layers.' if kind=='te' else 'blocks.'
    names=sorted(n for n in reader.header if n!='__metadata__' and not n.startswith(prefix))
    header={};offset=0
    if '__metadata__' in reader.header:header['__metadata__']=reader.header['__metadata__']
    for name in names:
        source=reader.header[name];size=source['data_offsets'][1]-source['data_offsets'][0]
        header[name]=dict(dtype=source['dtype'],shape=source['shape'],data_offsets=[offset,offset+size]);offset+=size
    raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
    raw=struct.pack('<Q',len(raw))+raw
    digest=hashlib.sha256(raw);ranges=[]
    with Path(path).open('rb') as stream:
        for name in names:
            source=reader.header[name];start,end=[reader.offset+x for x in source['data_offsets']]
            ranges.append(dict(key=name,start=start,end=end,target=len(raw)+header[name]['data_offsets'][0]))
            stream.seek(start);left=end-start
            while left:
                chunk=stream.read(min(left,8*2**20))
                if not chunk:raise IOError('Incomplete local auxiliary source')
                digest.update(chunk);left-=len(chunk)
    return dict(native_header=reader.header,
        native_header_sha256=hashlib.sha256(json.dumps(reader.header,sort_keys=True).encode()).hexdigest(),
        bytes=len(raw)+offset,sha256=digest.hexdigest(),header_base64=base64.b64encode(raw).decode(),ranges=ranges)


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,default=Path('D:/minimaxH3'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier planned source')
    args.output.mkdir(parents=True);proof=ROOT/'semantic_results/kaggle_hf_publication_v6'
    identity=json.loads((ROOT/'semantic_results/pipeline_checkpoint_identity.json').read_text())
    publication=json.loads((proof/'publication.json').read_text())
    receipts=json.loads((proof/'upload_receipts.json').read_text())
    result=dict(publication=publication,objects=receipts['objects'],auxiliary={},source_identity=identity,
        model_repository='Comfy-Org/MiniMax-H3',model_revision='e5eb578a89295337b8ff433a035929ce0279e0b6')
    target=args.output/'bundle_provenance';target.mkdir()
    for name in ['bundle.json']+[n.removesuffix('.safetensors')+'.json' for n in receipts['objects']]:
        shutil.copyfile(proof/name,target/name)
    for kind in ('te','dit'):
        source=args.models/identity[kind]['file']
        if source.stat().st_size!=identity[kind]['bytes']:raise ValueError('Local model size changed')
        result['auxiliary'][kind]=auxiliary(source,kind)
        print('AUXILIARY_PLAN',kind,result['auxiliary'][kind]['bytes'],flush=True)
    (args.output/'bundle_assets.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
