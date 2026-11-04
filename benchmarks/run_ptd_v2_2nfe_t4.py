"""One physical TP2 full-50-layer, first-two-hop PTD v2 Dataset I/O diagnostic.

This is exactly two trained NFE calls over the original eight-hop sigma grid.
It is NOT a valid two-step PTD sampler, video generation, or quality test.
No runtime LoRA baking, source SHA256 sweep or Dataset download/upload occurs.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ComfyUI'), str(ROOT / 'benchmarks')]
FORMAT = 'h3-ptd-v2-t4-two-nfe-mounted-io-v1'
HOPS = 2
TOKENS = 59134


class TwoNFEBoundary(Exception):
    """Intentional diagnostic stop after the second completed PTD denoiser."""


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resolve_mount(dataset):
    slug = dataset.split('/', 1)[1]
    choices = (Path('/kaggle/input') / slug,
               Path('/kaggle/input/datasets') / dataset,
               Path('/kaggle/input/datasets') / slug,
               Path('/kaggle/input') / dataset)
    found = [candidate / 'payload' for candidate in choices if (candidate / 'payload').is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(f'Expected exactly one mounted payload for {dataset}; found {len(found)}')
    return found[0].resolve()


def check_mounts(proofs, stages):
    roots = {}
    stage_start = time.monotonic()
    for part, spec in proofs.items():
        begin = time.monotonic()
        root = resolve_mount(spec['dataset'])
        receipt = root / 'receipt.json'
        if digest(receipt) != spec['receipt_sha256']:
            raise ValueError('Mounted Dataset receipt SHA256 differs from independently verified source: ' + part)
        roots[part] = root
        stages['mount_receipt_' + part] = dict(seconds=time.monotonic() - begin,
                                               receipt_bytes=receipt.stat().st_size,
                                               root=str(root),
                                               remote_proof=spec['published_identity'])
    stages['mount_and_receipt_total'] = dict(seconds=time.monotonic() - stage_start,
                                              scope='Nine mount lookups and receipt-only SHA256; no payload rehash')
    return roots


def assemble_diagnostic_manifest(proofs, roots, destination):
    """Use previously SHA256-proved Dataset receipts, not the 169.5-GB payloads.

    The resulting manifest is ephemeral evidence for this trial only. The
    production assembler's nine fresh readback checks remain untouched.
    """
    from patches.ptd.schedule import hop_sigmas
    shared = json.loads((roots['shared']/'receipt.json').read_text(encoding='utf-8'))
    recipe = shared.get('recipe')
    if not isinstance(recipe, dict):
        raise ValueError('Shared source recipe missing')
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if (shared.get('recipe_sha256') != recipe_sha or recipe.get('layers') != 50
            or recipe.get('hops') != 8 or recipe.get('steps') != 32
            or recipe.get('pruned_adaln_approximation') is not True):
        raise ValueError('Mounted shared recipe differs from qualified full Bank')
    parts = {part: dict(root=part, identity=proof['published_identity'])
             for part, proof in proofs.items()}
    records = []
    for hop in range(8):
        part = f'hop_{hop:02d}'
        receipt = json.loads((roots[part]/'receipt.json').read_text(encoding='utf-8'))
        if (receipt.get('hop') != hop or receipt.get('recipe_sha256') != recipe_sha
                or len(receipt.get('shards', {})) != 100):
            raise ValueError('Mounted Hop manifest differs from trained full Bank')
        shards = {stem: dict(row, part=part, file=stem+'.safetensors',
                             manifest=stem+'.json')
                  for stem, row in receipt['shards'].items()}
        records.append(dict(hop=hop,status='complete',source_recipe_sha256=recipe_sha,
                            auxiliary=dict(receipt['auxiliary'],part=part),shards=shards))
    data = dict(format='h3-ptd-tp2-bank-v2',status='complete',
        online_lora=False,full_model=True,recipe=recipe,recipe_sha256=recipe_sha,
        parts=parts,shared=dict(part='shared',aux=shared['aux'],
            aux_sha256=shared['aux_sha256'],aux_bytes=shared['aux_bytes'],
            native_header=shared['native_header'],block_plan=shared['block_plan']),
        hops=records,sigmas=hop_sigmas(8,32,12.).tolist())
    if destination.exists():
        raise FileExistsError('Preserve existing diagnostic manifest')
    destination.write_text(json.dumps(data,sort_keys=True)+'\n',encoding='utf-8')
    return data


def snapshot_ranks(rows):
    result = []
    for row in rows:
        prep = row.get('bundle_preparation', {}).get('dit', {})
        io = row.get('checkpoint_io', {}).get('dit', {})
        host = row.get('host_cache', {})
        result.append(dict(rank=row['rank'], calls=row.get('calls'),
            prepared_blocks=prep.get('prepared_blocks'),
            rank_read_ms=prep.get('rank_read_ms'),
            io_workers=prep.get('io_workers'),
            logical_read_bytes=io.get('logical_read_bytes'),
            source_read_bytes=io.get('source_read_bytes'),
            range_read_calls=io.get('range_read_calls'),
            host_cache=host, aot_preparation=row.get('aot_preparation')))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--proofs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--io-workers', type=int, default=1)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(format=FORMAT, status='starting', no_media_export=True,
        two_nfe_only=True, full_50_layer_native=True,
        trained_grid_hops=8, measured_hops=2, packed_tokens=TOKENS,
        gpu_execution=False, dataset_mount_includes_platform_prestart=False,
        dataset_full_payload_sha256_repeated=False, dataset_runtime_copy_to_tmp=False,
        no_online_bake=True, stages={}, nfe=[], limitations=[
            'Kaggle provider pre-kernel Dataset staging latency is outside Python-visible timing',
            'Rank read intervals overlap other work and are not additive non-overlap stalls',
            '2 NFE stops after Hop01; no valid completed 8-Hop PTD sample or media'])
    def save():
        (output / 'result.json').write_text(json.dumps(state, indent=2, default=str),
                                            encoding='utf-8')
    def mark(name, action):
        begin = time.monotonic()
        state['status'] = name
        save()
        print('PTD_2NFE_STAGE_START', name, flush=True)
        try:
            result = action()
            state['stages'][name] = dict(status='passed', seconds=time.monotonic()-begin)
            return result
        finally:
            state['stages'].setdefault(name, dict(status='failed',
                                                   seconds=time.monotonic()-begin))
            save()
            print('PTD_2NFE_STAGE_END', name, state['stages'][name], flush=True)

    service = None
    controller = None
    begun = False
    active = False
    save()
    try:
        proofs = json.loads(args.proofs.read_text(encoding='utf-8'))
        if set(proofs) != {'shared'} | {f'hop_{i:02d}' for i in range(8)}:
            raise ValueError('Exactly nine previously verified immutable Dataset parts required')
        roots = mark('mount_receipt_admission', lambda: check_mounts(proofs, state['stages']))
        mark('ephemeral_manifest_from_mounted_receipts',
             lambda: assemble_diagnostic_manifest(proofs, roots, args.manifest))

        import torch
        if args.io_workers not in range(1, 9):
            raise ValueError('PTD v2 io_workers must be 1..8')
        import comfy.options
        sys.argv = ['ptd-v2-2nfe']
        comfy.options.enable_args_parsing()
        import comfy.cli_args
        from safetensors.torch import save_file
        from comfy.nested_tensor import NestedTensor
        from patches.legacy_runtime import validate_packages, validate_source
        from patches.comfy_tp_runtime import RuntimeConfig, PhysicalTPService
        from patches.ptd.bank_v2 import PTDBankV2
        from patches.ptd.runtime import Controller, PTDOptions
        from patches.ptd.sampler import sample_ptd
        from benchmarks.dit_pair_contract import fixture, TOKENS as FIXTURE_TOKENS
        from patches.host_weight_cache import GiB
        from patches.h3_operator_policy import H3OperatorPolicy
        from patches.h3_operator_dependencies import configure_compiler
        validate_packages()
        validate_source(args.source / 'ComfyUI')
        if FIXTURE_TOKENS != TOKENS:
            raise ValueError('Frozen performance fixture token count changed')
        devices = [dict(index=i, name=torch.cuda.get_device_name(i),
                        capability=list(torch.cuda.get_device_capability(i)),
                        uuid=str(torch.cuda.get_device_properties(i).uuid))
                   for i in range(torch.cuda.device_count())]
        if (len(devices) != 2 or any('T4' not in d['name']
                                    or d['capability'] != [7, 5] for d in devices)
                or len({d['uuid'] for d in devices}) != 2):
            raise RuntimeError('Physical 2 x Tesla T4 SM75 GPUs required')
        state['devices'] = devices
        state['gpu_execution'] = True
        torch.set_grad_enabled(False)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Fail before synthetic allocation and rank launch if no qualified nvcc
        # exists; persist exact path/version/qualification in the final receipt.
        state['compiler'] = mark('native_compiler_preflight',
                                 lambda: configure_compiler(H3OperatorPolicy()))
        save()

        bank = mark('bank_metadata_admission', lambda: PTDBankV2(
            args.manifest, part_roots=tuple((part, str(root)) for part, root in sorted(roots.items())),
            verify_payloads=False))
        if bank.layers != 50 or bank.recipe.get('pruned_adaln_approximation') is not True:
            raise ValueError('Expected full 50-layer production bank with explicit pruned AdaLN approximation')
        state['bank'] = dict(format='h3-ptd-tp2-bank-v2',
                             recipe_sha256=bank.recipe_sha,
                             manifest_sha256=digest(args.manifest),
                             layers=bank.layers, mounted_parts=9)
        state['io_policy'] = dict(ptd_io_workers=args.io_workers,
                                  ptd_retire_page_cache=False, direct_mounted_read=True)
        state['stages']['dataset_mount_visibility'] = dict(
            observable='post-kernel filesystem lookup only',
            provider_prestart_staging_seconds=None,
            evidence='Notebook code cannot observe Kaggle dataset preprocessing before cell execution')
        save()

        # A syntactically valid empty TE checkpoint; no TE model is downloaded or executed.
        unused_te = output / 'unused_te.safetensors'
        save_file({}, str(unused_te))
        config = RuntimeConfig(str(unused_te), str(args.manifest),
            layers=50, ptd_part_roots=tuple((part, str(root)) for part, root in sorted(roots.items())),
            ptd_io_workers=args.io_workers, ptd_retire_page_cache=False,
            weight_bytes=6*GiB, pinned_bytes=6*GiB, cache_bytes=4*GiB,
            host_bytes=20*GiB, reserve_bytes=9*GiB,
            timeout_seconds=1800, cross_request_te=False, cross_model_cpu=False,
            debug=False)
        state['config'] = dict(weight_bytes=config.weight_bytes,
            pinned_bytes=config.pinned_bytes, cache_bytes=config.cache_bytes,
            host_bytes=config.host_bytes, reserve_bytes=config.reserve_bytes,
            ptd_part_roots_count=len(config.ptd_part_roots), timeout_seconds=config.timeout_seconds)
        inputs = mark('synthetic_59134_token_ref2va_inputs', fixture)
        noise = NestedTensor(inputs['x'])
        payload = dict(inputs['minimax_payload'], ptd_text_states=inputs['context'],
                       audio_scale=1.0)
        layout = payload['layout']
        if layout.seq_len != TOKENS:
            raise ValueError('Ref2VA tokens mismatch')
        state['synthetic_shapes'] = [list(v.shape) for v in inputs['x']]
        state['synthetic_te_rows'] = int(inputs['context'].shape[1])

        service = mark('physical_tp2_service_startup', lambda: PhysicalTPService(config))
        options = PTDOptions(enabled=True, allow_pruned_adaln=True, allow_fixture=False,
            hop_handoff='queue', future_cache=True, prefetch_refiner=True)
        controller = mark('controller_metadata_admission',
            lambda: Controller(service, args.manifest, options))
        state['full_sigmas'] = controller.sigmas.tolist()
        if controller.hops != 8:
            raise ValueError('PTD sigma grid must retain all eight trained hops')
        aot = mark('native_dit_aot_59134_tokens',
                   lambda: service.call('prepare_dit', layout.seq_len))
        state['aot'] = aot
        mark('controller_begin', controller.begin)
        begun = True
        mark('begin_sample', lambda: service.call('begin_sample'))
        active = True

        class Denoiser:
            latent_image = None
            def __init__(self):
                self.noise = noise
            def __call__(self, latent, sigma, **kwargs):
                hop = len(state['nfe'])
                if hop >= HOPS:
                    raise RuntimeError('Exceeded the explicit two-NFE diagnostic cap')
                begin = time.monotonic()
                result = service.call('denoise', list(latent.unbind()), sigma.float()*1000,
                    context=torch.zeros(1, inputs['context'].shape[1], 5120,
                                        dtype=torch.float16),
                    minimax_payload=payload, transformer_options={})
                denoise_s = time.monotonic() - begin
                if len(result) != 2 or any(not bool(torch.isfinite(v).all()) for v in result):
                    raise FloatingPointError('PTD denoiser returned nonfinite or invalid streams')
                ptd_state = service.call('ptd_state')
                rank_stats = service.call('stats')
                ranks = snapshot_ranks(rank_stats)
                expected_bytes = {}
                receipt = json.loads((roots[f'hop_{hop:02d}'] / 'receipt.json').read_text())
                for rank in (0, 1):
                    expected_bytes[str(rank)] = sum(
                        row['bytes'] for name, row in receipt['shards'].items()
                        if name.endswith(f'rank{rank}'))
                entry = dict(hop=hop, sigma=float(sigma.flatten()[0]), denoise_wall_seconds=denoise_s,
                    ordered_calls=ptd_state['calls'],
                    queue_state=ptd_state['queue'],
                    rank_metrics=ranks, expected_shard_file_bytes=expected_bytes,
                    output_finite=True)
                state['nfe'].append(entry)
                save()
                print('PTD_2NFE_DONE', hop, 'wall_s', round(denoise_s,3),
                      'rank_read_ms', [r['rank_read_ms'] for r in ranks],flush=True)
                return NestedTensor([
                    x - sigma.float().reshape((-1,)+(1,)*(x.ndim-1))*velocity
                    for x, velocity in zip(latent.unbind(),result)])

        callbacks = []
        def on_hop(value):
            callbacks.append(int(value['i']))
            if len(callbacks) == HOPS:
                raise TwoNFEBoundary()

        state['status'] = 'two_nfe_physical'
        save()
        try:
            sample_ptd(Denoiser(), noise, controller.sigmas.float(),
                       controller=controller, callback=on_hop)
        except TwoNFEBoundary:
            pass
        else:
            raise RuntimeError('PTD diagnostic failed to stop after exactly two NFE')
        if callbacks != [0, 1] or len(state['nfe']) != 2:
            raise RuntimeError('PTD did not execute exactly Hop00 and Hop01')
        state['final_rank_stats'] = snapshot_ranks(service.call('stats'))
        state['status'] = 'completed_2nfe_only'
        state['valid_video_generated'] = False
        save()
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__, error_message=str(exc)[:1000])
        (output / 'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        save()
        raise
    finally:
        if active and service is not None and not service.closed:
            try: service.call('end_sample')
            except BaseException as exc: state['end_sample_error'] = repr(exc)
        if begun and controller is not None and service is not None and not service.closed:
            try: controller.end()
            except BaseException as exc: state['controller_end_error'] = repr(exc)
        if service is not None and not service.closed:
            try: service.close()
            except BaseException as exc: state['service_close_error'] = repr(exc)
        save()
        print('PTD_2NFE_FINAL', state['status'], len(state['nfe']), flush=True)


if __name__ == '__main__':
    main()
