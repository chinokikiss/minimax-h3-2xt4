"""Extract actual block weights and deterministic inputs for a private GPU run."""
import hashlib
import json
from pathlib import Path
import shutil

from retest import ROOT, digest_tensor, state, write_json
import torch
from safetensors.torch import save_file
from fixture_inputs import block_input


def main():
    model_dir = Path('D:/minimaxH3')
    out = ROOT / 'repaired_fixtures'
    out.mkdir(exist_ok=True)
    manifest = {'checkpoints': {}, 'inputs': {}, 'scope': 'actual trained TE/DiT block 0; complete video decoder'}
    names = {'te': 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
             'dit': 'MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors'}
    for kind, filename in names.items():
        original = model_dir / filename
        prefix = 'model.layers.0.' if kind == 'te' else 'blocks.0.'
        selected = {prefix + k: v.contiguous() for k, v in state(original, prefix).items()}
        if kind == 'dit':
            selected.update({'adaln_t_table' + k: v.contiguous() for k, v in state(original, 'adaln_t_table').items()})
        fixture = out / f'{kind}_block0.safetensors'
        save_file(selected, str(fixture), metadata={'scope': 'unaltered trained block 0 extracted from supplied checkpoint'})
        del selected
        with fixture.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest['checkpoints'][kind] = {'file': fixture.name, 'sha256': digest,
            'bytes': fixture.stat().st_size, 'source_checkpoint': filename,
            'source_checkpoint_bytes': original.stat().st_size, 'block_index': 0}
        rows = []
        curve = state(fixture, 'adaln_t_table')[''] if kind == 'dit' else None
        lengths = [37, 512, 2048] if kind == 'te' else [37, 1024, 4096]
        for i, length in enumerate(lengths):
            shape = (1, length, 5120) if kind == 'te' else (length, 5376)
            data = block_input(kind, length, 704 + i, curve)
            x = data['input']
            filename = f'{kind}_S{length}.pt'
            torch.save(data, out / filename)
            rows.append({'name': f'S{length}', 'file': filename, 'shape': list(shape),
                         'input_sha256': digest_tensor(x), 'seed': 704 + i, 'length': length})
        manifest['inputs'][kind] = rows
    video = model_dir / 'minimax_h3_video_vae_int8_convrot.safetensors'
    with video.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest['checkpoints']['video'] = {'file': video.name, 'sha256': digest, 'bytes': video.stat().st_size,
                                       'source_checkpoint': video.name, 'num_blocks': 36}
    rows = []
    refs = ROOT / 'slab_results/saved_references'
    for name in ['768x768_1F', '768x1344_2F']:
        x = torch.load(refs / f'input_{name}.pt', weights_only=True)
        filename = f'video_{name}.pt'
        torch.save({'input': x}, out / filename)
        rows.append({'name': name, 'file': filename, 'shape': list(x.shape), 'input_sha256': digest_tensor(x)})
        for prefix, suffix in [('input', '.pt'), ('native', '_output.pt')]:
            saved = f'{prefix}_{name}{suffix}'
            shutil.copyfile(refs / saved, out / saved)
    shutil.copyfile(refs / 'manifest.json', out / 'manifest.json')
    manifest['inputs']['video'] = rows
    write_json(out / 'fixture_manifest.json', manifest)
    write_json(out / 'dataset-metadata.json', {
        'title': 'MiniMax H3 Repaired Benchmark Fixtures',
        'id': 'chinokiki9/minimax-h3-repaired-benchmark-fixtures',
        'licenses': [{'name': 'other'}],
        'description': 'Private validation fixtures. Unaltered block 0 from user-supplied MiniMax-H3 INT8 checkpoints, saved benchmark latents and raw references. Original model terms apply. Not full TE/DiT models.'})
    print('Prepared private fixtures, bytes:', sum(p.stat().st_size for p in out.iterdir() if p.is_file()), flush=True)


if __name__ == '__main__':
    main()
