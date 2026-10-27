"""Bounded pairs of Native temporal clips; blending stays in the parent planner."""
import torch


class TemporalClipPairs:
    """Cache at most one decoded neighbour, on CPU, between Native clip calls."""
    def __init__(self, model, z, decode_pair):
        padding, count = model._decode_temporal_chunks(z.shape[2])
        if padding:
            z = torch.cat((z, z[:, :, -1:].repeat(1, 1, padding, 1, 1)), dim=2)
        self.z, self.count = z, count
        self.step, self.span = model.tokens_chunk_size, model.tokens_chunk_size + model.token_overlap
        self.decode_pair, self.index, self.ready = decode_pair, 0, None

    def decode(self, clip):
        index = self.index
        if index >= self.count:
            raise RuntimeError('Native temporal planner requested too many clips')
        self.index += 1
        if self.ready is not None:
            output, self.ready = self.ready, None
        else:
            start = (index + 1) * self.step
            neighbour = self.z[:, :, start:start + self.span] if index + 1 < self.count else None
            output, self.ready = self.decode_pair((clip, neighbour))
            if output is None or (neighbour is None) != (self.ready is None):
                raise RuntimeError('Incomplete temporal clip pair')
        return output.to(device=clip.device)

    def finish(self):
        if self.index != self.count or self.ready is not None:
            raise RuntimeError('Native temporal planner did not consume every clip')

    def close(self):
        self.ready = self.z = None
