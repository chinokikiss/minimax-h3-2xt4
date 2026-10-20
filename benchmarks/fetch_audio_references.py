"""Retrieve only the completed audit's small audio evidence, never a GPU job."""
import hashlib
import json
from pathlib import Path
import torch
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'semantic_results/audio_saved_references';out.mkdir(exist_ok=True)
    recorded=json.loads((ROOT/'kaggle_retest_output/cu130/retest_results/round_0/audio/results.json').read_text())
    api=KaggleApi();api.authenticate()
    rows=[]
    with api.build_kaggle_client() as client:
        client._http_client._init_session();session=client._http_client._session
        for seconds in (5,10,20):
            for kind in ('input','native','channel_2gpu'):
                name=f'{kind}_{seconds}s'+('.pt' if kind=='input' else '_output.pt')
                path=out/name;remote='retest_results/round_0/audio/'+name
                if not path.exists():
                    try:
                        response=session.get('https://www.kaggle.com/api/v1/kernels/output/download/chinokiki9/minimax-h3-cuda13-retest/'+remote,timeout=(8,40))
                        response.raise_for_status();path.write_bytes(response.content)
                    except Exception as error:
                        raise RuntimeError('Audio artifact download failed: '+type(error).__name__) from None
                tensor=torch.load(path,weights_only=True,map_location='cpu')
                digest=hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
                if kind=='input':assert digest==recorded['workloads'][str(seconds)+'s']['input_sha256']
                assert torch.isfinite(tensor).all()
                rows.append(dict(file=name,source=remote,tensor_sha256=digest,shape=list(tensor.shape),dtype=str(tensor.dtype)))
                print('VERIFIED',name,flush=True)
    (out/'manifest.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
