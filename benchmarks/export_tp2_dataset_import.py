"""Export public, immutable base-shard URLs and proofs; no weight/network I/O."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output = Path(output)
    if output.exists(): raise FileExistsError('Preserve earlier import manifests')
    proof = ROOT/'semantic_results/kaggle_hf_publication_v6'
    publication = json.loads((proof/'publication.json').read_text())
    receipts = json.loads((proof/'upload_receipts.json').read_text())
    bundle_bytes = (proof/'bundle.json').read_bytes()
    bundle = json.loads(bundle_bytes)
    plan = json.loads((ROOT/'semantic_results/offline_conversion_plan.json').read_text())
    expected = {f'{kind}.block{i:02d}.rank{rank}' for kind in ('te','dit') for i in range(50) for rank in (0,1)}
    assert publication['complete'] and receipts['complete'] and bundle['complete']
    assert receipts['repo'] == publication['repo']
    assert set(bundle['blocks']) == expected == set(plan['blocks'])
    assert bundle['plan'] == plan
    assert hashlib.sha256(bundle_bytes).hexdigest() == publication['bundle_sha256']
    assert set(receipts['objects']) == {n+'.safetensors' for n in expected}
    prefix = f'https://huggingface.co/{publication["repo"]}/resolve/{publication["commit"]}/'
    files = []; metadata = {}
    for stem in sorted(expected):
        name = stem+'.safetensors'; row = receipts['objects'][name]
        manifest_bytes = (proof/(stem+'.json')).read_bytes(); manifest = json.loads(manifest_bytes)
        assert manifest['native_tensor_bit_exact'] and manifest['file_sha256'] == row['sha256']
        assert manifest['format'] == plan['format'] and manifest['configuration'] == plan['configuration']
        for field in ('kind','index','rank','state_sha256','quant_metadata'):
            assert manifest[field] == plan['blocks'][stem][field]
        files.append(dict(file=name, role='base_weight', bytes=row['bytes'], sha256=row['sha256'], url=prefix+name))
        metadata[stem+'.json'] = manifest_bytes
    metadata['bundle.json'] = bundle_bytes
    for name, raw in sorted(metadata.items()):
        files.append(dict(file=name, role='provenance', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), url=prefix+name))
    record = dict(format='h3-base-tp2-dataset-import-v1', status='prepared_not_uploaded',
        repo=publication['repo'], revision=publication['commit'], weights=200,
        weight_bytes=sum(x['bytes'] for x in files if x['role']=='base_weight'),
        sources=plan['sources'], base_only=True, contains_lora=False,
        contains_baked_weights=False, contains_activations=False, files=files,
        missing_components=['Native TE startup/vision and global tensors',
                            'Native DiT global/refiner/final-head tensors', 'Video and Audio VAEs'],
        qualification='Historical Native tensor-byte proof; current bundle-aware offline LoRA adaptation remains pending')
    output.mkdir(parents=True)
    (output/'import_manifest.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    (output/'all_file_urls.txt').write_text('\n'.join(x['url'] for x in files)+'\n', encoding='utf-8')
    (output/'weight_urls.txt').write_text('\n'.join(x['url'] for x in files if x['role']=='base_weight')+'\n', encoding='utf-8')
    with (output/'files.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file','role','bytes','sha256','url'])
        writer.writeheader(); writer.writerows(files)
    with zipfile.ZipFile(output/'provenance_only.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(metadata.items()): archive.writestr(name, raw)
    text = '''# Base TP2 bundle Dataset import

This import contains 200 unmerged base transformer shards: 50 TE layers and
50 DiT layers, both TP ranks, plus 201 provenance JSON files. Total weight
payload is 44,032,850,400 bytes (41.009 GiB). No LoRA, baked checkpoint,
activation capture, dependency archive or credentials are included.

`all_file_urls.txt` lists the 401 immutable public HF URLs. `files.csv` and
`import_manifest.json` provide SHA256 and byte counts. `provenance_only.zip`
contains the small JSON proofs for convenient upload; it contains no weights.
The URL list is an import aid, not a claim that Kaggle's UI accepts a bulk list.
Do not import the text/CSV alone and expect them to contain model weights.

Preserve the upstream model licenses and notices. A Dataset should keep the
shard basenames and extracted provenance JSON files at its root. The original
bundle covers transformer blocks only; startup/vision/global/refiner/head
weights and VAEs need separate delivery.

The current offline LoRA converter reads a full DiT checkpoint and writes a
complete merged checkpoint. It does not yet consume this base bundle directly.
Its GPU work is layer-parallel and matrix-bounded, while its disk output is a
full baked checkpoint. The proposed bundle adaptation reconstructs only the
matrix being prepared, merges the selected ordered LoRA stack in FP32,
requantizes across the full Native input width, then resplits into TP2 shards.
Temporary prepared weights must follow a bounded, LoRA-keyed lifetime; they
must not become a public per-LoRA model cache. This is a plan, not an integrated
or hardware-qualified feature. The base bundle stays immutable.
'''
    (output/'README.md').write_text(text, encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = build(args.output)
    print(json.dumps(dict(status=result['status'], weights=result['weights'],
                         weight_bytes=result['weight_bytes'], total_files=len(result['files']))))


if __name__ == '__main__': main()
