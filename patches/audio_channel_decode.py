"""FP32 stereo channel ownership inside persistent TP2 ranks.

Independent of the historical temporal/tiled decoder benchmark manager.
"""
import torch


def channel_decode(model,z,comm,rank):
    """Decode a rank-owned stereo channel inside persistent TP2 workers.

    Reuse each worker's decoder replica; do not construct a second two-device
    manager inside a rank. Native public chunking/normalization stays outside.
    """
    if rank not in (0,1) or comm.get_world_size()!=2:raise ValueError('Two audio channel owners required')
    if z.ndim!=4 or z.shape[1]!=32 or z.shape[2]!=2 or z.dtype!=torch.float32:
        raise ValueError('Channel Parallel requires FP32 stereo [B,32,2,T] latents')
    mono=model.decode(z[:,:,rank:rank+1].contiguous()).contiguous()
    channels=[torch.empty_like(mono),torch.empty_like(mono)]
    comm.all_gather(channels,mono)
    return torch.cat(channels,1) if rank==0 else None
