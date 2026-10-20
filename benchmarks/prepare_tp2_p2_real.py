"""Real row-linear fixtures. Replay only until the first native FC2 boundary.

Local SM120 is used for capture only, never target performance measurement.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
OUT=Path('D:/attention_eval/tp2_p2_20261018')
DATA=OUT/'real_transport_dataset'
CAPTURE=Path('D:/attention_eval/tokyo_ref2va_20261013')


def main():
    import torch
    DATA.mkdir(parents=True,exist_ok=True)
    from patches.windowed_tp import CheckpointReader
    reader=CheckpointReader('D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')
    records=[]
    def save(name,x,prefix,provenance):
        tensors=reader.read(prefix)
        # Native checkpoint uses Kitchen INT8 ConvRot storage and per-output scales.
        print('WEIGHT_KEYS',prefix,{k:(list(v.shape),str(v.dtype)) for k,v in tensors.items()},flush=True)
        p=DATA/(name+'.pt')
        torch.save(dict(x=x.detach().cpu().contiguous(),weights=tensors,provenance=provenance),p)
        records.append(dict(name=name,file=p.name,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),bytes=p.stat().st_size,
                            x_shape=list(x.shape),weight_prefix=prefix,provenance=provenance))
        (DATA/'real_transport_manifest.json').write_text(json.dumps(dict(records=records,checkpoint=str(reader.path),
            checkpoint_signature=reader.signature,scope='Native real activations, learned INT8 ConvRot weights; TP2 partials must be computed on target hardware.'),indent=2))
    # Three trajectory positions, one representative deep layer; lossless native output.
    for call in (0,9,19):
        p=CAPTURE/'qkv'/f'dit_layer25_call{call:02d}.pt'
        data=torch.load(p,map_location='cpu',weights_only=True)
        y=data['native_output'];x=y.transpose(1,2).reshape(y.shape[0]*y.shape[2],-1)
        save('o_l25_s'+str(call),x,'blocks.25.attn.out_proj.',dict(kind='existing_native_capture',source=str(p),source_sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),metadata=data['metadata']))
        del data,x,y
    if not (DATA/'fc1_seed.pt').exists():
        cfg=json.loads((CAPTURE/'run_state.json').read_text())['config']
        cfg.update(output_directory=str(OUT/'capture_first_fc2'),reuse_native_inputs=str(CAPTURE),capture_layers={},save_denoiser_tensors=False)
        config_path=OUT/'capture_first_fc2.json';config_path.write_text(json.dumps(cfg,indent=2))
        sys.path.insert(0,cfg['comfy_root'])
        import comfy.ops
        import comfy_kitchen as ck
        import retest
        last={};load_original=retest.load_native_block
        def load_block(kind,path,index,device):
            block=load_original(kind,path,index,device)
            if kind=='dit' and index==0:
                block.mlp.fc1.register_forward_pre_hook(lambda module,args:last.update(fc1_input=args[0]))
            return block
        retest.load_native_block=load_block
        original=comfy.ops.linear_input_act
        class CapturedBoundary(Exception):pass
        def capture(linear,x,input_act,*a,**kw):
            # Token-refiner has the same FFN width; it is NOT a DiT block.
            if input_act=='swiglu' and x.shape==torch.Size([6814,28672]):
                from patches import optimized_tp as native,parallel_int8 as pi
                fw=reader.read('blocks.0.mlp.fc1.')
                source=last['fc1_input'];backend=ck.backends.cuda
                # Kitchen .33 native int8_linear takes its fused ConvRot64
                # branch here (the fallback's stale width-band comment is not
                # its actual _fused_convrot_ok predicate).
                q,s=backend.quantize_int8_rowwise_convrot64(source.reshape(6814,5376),256)
                reconstructed=pi.dequantize(native.native_int_mm(q,fw['weight'].cuda()),s,fw['weight_scale'].cuda(),torch.float32)
                assert torch.equal(x,reconstructed),'Compact FC1 replay does not exactly reproduce native FC2 input'
                torch.save(dict(q=q.cpu(),scale=s.cpu(),expected_x_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
                    source='Native first DiT FC1 quantized operand; independent cublas INT32 + native dequant replay bitwise verified',
                    shape=list(x.shape)),DATA/'fc1_seed.pt')
                del reconstructed,fw
                save('fc2_l0_s0',x,'blocks.0.mlp.fc2.',dict(kind='first_native_FC2_boundary',workload_config=cfg,
                    input_hashes={n:hashlib.file_digest((CAPTURE/n).open('rb'),'sha256').hexdigest() for n in ('conditioning.pt','initial_noise.pt','sigmas.pt')},
                    scope='Exact original first denoiser input, native preprocessing and layer0 attention; stopped before first FC2.'))
                raise CapturedBoundary('Expected early stop: native FC2 input saved')
            return original(linear,x,input_act,*a,**kw)
        comfy.ops.linear_input_act=capture
        from capture_real_attention import main as replay
        try:replay(config_path)
        except CapturedBoundary:print('REAL_FC2_CAPTURE_COMPLETE',flush=True)
        finally:comfy.ops.linear_input_act=original;retest.load_native_block=load_original
    assert (DATA/'fc2_l0_s0.pt').exists()
    (DATA/'dataset-metadata.json').write_text(json.dumps(dict(title='H3 TP2 P2 Real Transport Fixtures',id='chinokiki9/h3-tp2-p2-real-transport',licenses=[dict(name='other')]),indent=2))
    print('P2_REAL_DATASET',DATA,flush=True)


if __name__=='__main__':main()
