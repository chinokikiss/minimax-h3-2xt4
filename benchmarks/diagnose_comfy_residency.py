"""Reproduce repeated full-depth conditioning admission on the local GPU."""
from qualify_comfy_sampler import *
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer

enable('scalemax');torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
import comfy_kitchen as ck
ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda x:49152
isolate_cublas_workspaces()
config=RuntimeConfig('D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
    'D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',layers=50,
    weight_bytes=4*GiB,reserve_bytes=256*2**20,timeout_seconds=90)
service=LogicalService(config)
tokenizer=MiniMaxH3Tokenizer()
image=torch.linspace(0,1,64*64*3).reshape(1,64,64,3)
requests=[tokenizer.tokenize_with_weights('A small bird moves.',images=[image]),
    tokenizer.tokenize_with_weights('A small bird moves.',minimax_ref_items=[{'type':'image','data':image},
       {'type':'video','data':torch.cat([image,1-image])}]),tokenizer.tokenize_with_weights('blur')]
try:
    for i,tokens in enumerate(requests):
        print('CONDITION',i,flush=True)
        service.call('te',tokens)
        print('GPU',torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),flush=True)
except BaseException:
    for rank in service.ranks:
        w=rank.window
        print('WINDOW',rank.rank,w.cursor,len(w.entries),len(w.retired),w._live_bytes(),w.max_weight_bytes,
            [(w._next_use(k),e.size,e.active) for k,e in w.entries.items()],flush=True)
    raise
finally:
    service.close()
