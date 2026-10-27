"""CPU-only immutable model/header/reference checks, before any GPU job."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import requests
from PIL import Image


def download_image(row, output):
    # Public input URLs only. No credentials, browser state or mutable aliases.
    response = requests.get(row['url'], timeout=(20, 90),
        headers={'User-Agent': 'H3-REF2VA-evaluation/1.0'})
    response.raise_for_status()
    data = response.content
    if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('Reference bytes changed: ' + row['file'])
    image = Image.open(io.BytesIO(data))
    if list(image.size) != row['size']:
        raise ValueError('Reference dimensions changed')
    image.verify()
    path = output / row['file']
    path.write_bytes(data)
    return dict(file=row['file'], bytes=len(data), sha256=row['sha256'], size=row['size'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from benchmarks.anime_e2e_contract import validate
    from benchmarks.turbo_e2e_contract import validate_header
    plan = json.loads((args.source/'turbo_plan.json').read_text())
    references = json.loads((args.source/'l1_assets_identity.json').read_text())['images']
    result = dict(status='starting', full_pipeline_authorized=True, gpu_used=False)
    try:
        result['workload'] = validate(plan, references)
        assets = json.loads((args.source/'bundle_assets.json').read_text())
        assert assets['publication']['repo'] == 'cnmds/minimax-h3-tp2-int8-561a6211'
        assert assets['publication']['commit'] == 'd3e6a4b131056d2c626a01acf6ba48123a885783'
        header = json.loads((args.source/'turbo_lora_header.json').read_text())
        assert len(validate_header(header, assets['auxiliary']['dit']['native_header'])) == 208
        result['references'] = [download_image(row, args.output) for row in references]
        result.update(status='cpu_input_ready', lora_targets=208)
    except BaseException as exc:
        result.update(status='blocked', error=repr(exc))
        raise
    finally:
        (args.output/'input_qualification.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
