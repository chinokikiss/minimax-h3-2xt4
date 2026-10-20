"""Bounded optional detail; aggregate counters remain available in inference."""
import os


class RecentEvents(list):
    def __init__(self,limit=64):
        super().__init__();self.limit=limit;self.dropped=0

    def append(self,item):
        if not self.limit:
            self.dropped+=1;return
        if len(self)>=self.limit:
            del self[0];self.dropped+=1
        super().append(item)


def detail_events():
    # Tracing is a separate opt-in benchmark entry point. This flag requests
    # bounded detail metrics only; it does not start a profiler or dump tensors.
    return RecentEvents(4096 if os.environ.get('MINIMAX_H3_PROFILE_DETAILS')=='1' else 0)
