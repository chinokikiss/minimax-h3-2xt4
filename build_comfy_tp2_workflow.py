"""Create a separate INT8 TP2 variant; preserve the user's original workflow."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    root=Path(__file__).resolve().parent
    local_source=Path(os.environ.get('COMFY_ROOT',str(root/'ComfyUI')))/'user/default/workflows/video_minimax_h3_r2v.json'
    if not local_source.is_file():
        # Workflow data may be imported from the earlier UI; never import its code.
        local_source=Path('C:/Users/Administrator/Documents/project/ComfyUI-master/user/default/workflows/video_minimax_h3_r2v.json')
    parser.add_argument('--source', type=Path, default=local_source)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'workflows/video_minimax_h3_r2v_tp2_int8.json')
    parser.add_argument('--skip-native',action='store_true',help='Preserve an existing Native fallback workflow')
    args = parser.parse_args()
    workflow = json.loads(args.source.read_text(encoding='utf-8'))
    native_workflow = json.loads(json.dumps(workflow))
    native_nodes = {n['id']:n for n in native_workflow['nodes']}
    native_nodes[128]['widgets_values'][0] = 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors'
    native_nodes[128]['widgets_values_named']['clip_name'] = native_nodes[128]['widgets_values'][0]
    native_nodes[128]['properties'] = {'Node name for S&R':'CLIPLoader'}
    native_nodes[119]['widgets_values'][0] = 'minimax_h3_video_vae_int8_convrot.safetensors'
    native_nodes[119]['widgets_values_named']['vae_name'] = native_nodes[119]['widgets_values'][0]
    native_nodes[119]['properties'] = {'Node name for S&R':'VAELoader'}
    native_workflow.setdefault('extra',{})['minimax_tp2_note'] = (
        'Native fallback using the same validated INT8 TE/DiT/video checkpoints; original workflow preserved.')
    nodes = {n['id']:n for n in workflow['nodes']}
    if [nodes[n]['type'] for n in (127,128,119)] != ['UNETLoader','CLIPLoader','VAELoader']:
        raise ValueError('Source loader nodes changed; inspect the graph before converting')
    loader = nodes[127]
    clip_links = nodes[128]['outputs'][0]['links']
    loader.update(type='MiniMaxH3TP2Loader', title='MiniMax H3 TP2 — integrated operators',
        inputs=[dict(name='lora_stack',type='MINIMAX_TP2_LORAS',link=None)],
        size=[640,180], properties={'Node name for S&R':'MiniMaxH3TP2Loader'},
        widgets_values=['D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
            'D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',6.,3.,
            False,'scalemax',4.,False,'tilelang_group32','kitchen','tilelang_packed',True,True,96,False,'owned_fp32','online'])
    loader.pop('widgets_values_named',None)
    loader['outputs'] += [dict(name='CLIP',type='CLIP',links=clip_links),
        dict(name='session',type='MINIMAX_TP2_SESSION',links=[])]
    for link in workflow['links']:
        if link[1] == 128:
            link[1],link[2] = 127,1
    workflow['nodes'].remove(nodes[128])
    link_id = max(x[0] for x in workflow['links'])+1
    vae = nodes[119]
    vae['widgets_values'] = ['minimax_h3_video_vae_int8_convrot.safetensors']
    vae['widgets_values_named'] = {'vae_name':vae['widgets_values'][0]}
    vae['properties'] = {'Node name for S&R':'VAELoader'}
    # Both existing VAE loaders keep their output links and reference-encoding
    # role. Only decoding dispatch changes; no sampler/media node is rewritten.
    for node in workflow['nodes']:
        if node['type'] != 'VAELoader':
            continue
        name = node['widgets_values'][0]
        if not name.startswith('minimax_h3_'):
            continue
        node.update(type='MiniMaxH3ParallelVAELoader',
            title='MiniMax H3 parallel VAE (Native public plan)',
            inputs=[dict(name='session',type='MINIMAX_TP2_SESSION',link=link_id)],
            widgets_values=[name,'auto'], size=[430,110],
            properties={'Node name for S&R':'MiniMaxH3ParallelVAELoader'})
        node.pop('widgets_values_named',None)
        workflow['links'].append([link_id,127,2,node['id'],0,'MINIMAX_TP2_SESSION'])
        loader['outputs'][2]['links'].append(link_id)
        link_id += 1
    workflow['last_link_id'] = link_id-1
    workflow.setdefault('extra',{})['minimax_tp2_note'] = (
        'Separate INT8 TP2 variant. Requires two physical CUDA GPUs and linked custom node. '
        'TE: TileLang Group32 dense; DiT: Kitchen INT8 by default, optional TileLang drop/scalar/dense; '
        'Video attention: Kitchen INT8. Packed row-linear/SwiGLU padding on SM75 FP32 fallback rotation; '
        'Selective B for qualified FC1 rank128 LoRA. Native INT32 reduction retained. '
        'Default video decode: Native-aligned query partitioning inside the Native public plan. '
        'Default audio decode: FP32 Channel Parallel, with small batch-dependent rounding error; '
        'set either VAE decode_mode to native for fallback. Original workflow preserved. '
        'H3 VAE weights load encoder-only for references and decoder-only for outputs; '
        'idle residency also preloads encoders. Optional offline LoRA uses layer-parallel baking. '
        'Component timings do not establish an integrated pipeline speedup.')
    workflow['extra']['minimax_h3_operators']={'te_attention':'tilelang_group32','dit_attention':'kitchen',
        'video_attention':'kitchen','row_linear_backend':'tilelang_packed','swiglu_padding':True,
        'selective_b':True,'scalar_score_workspace_mib':96,'reduction':'native_int32'}
    nodes = {n['id']:n for n in workflow['nodes']}
    for identity,source,source_slot,dest,dest_slot,kind in workflow['links']:
        if nodes[dest]['inputs'][dest_slot]['link'] != identity:
            raise ValueError('Destination input does not match its link')
        if identity not in nodes[source]['outputs'][source_slot]['links']:
            raise ValueError('Source output does not match its link')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(workflow,ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.output)
    if not args.skip_native:
        native_path = args.output.with_name('video_minimax_h3_r2v_native_int8.json')
        native_path.write_text(json.dumps(native_workflow,ensure_ascii=False,indent=2),encoding='utf-8')
        print(native_path)


if __name__ == '__main__':
    main()
