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


def diagnostics_enabled(enabled=None):
    # Explicit runtime settings override benchmark-only environment defaults.
    return os.environ.get('MINIMAX_H3_PROFILE_DETAILS')=='1' if enabled is None else bool(enabled)


def detail_events(*, enabled=None):
    # Tracing is opt-in; disabled production paths also skip event construction.
    return RecentEvents(4096 if diagnostics_enabled(enabled) else 0)
