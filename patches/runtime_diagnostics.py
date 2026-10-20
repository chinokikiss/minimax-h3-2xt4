"""Read-only worker snapshots, assembled only for explicit diagnostic commands.

No profiler, tensor capture, synchronization, or benchmark dependency is loaded
by this module. Inference scheduling and ownership stay in the runtime.
"""


def idle_snapshot(runtime):
    window = runtime.idle_window
    return dict(
        runtime.idle_stats,
        window=window.stats if window else None,
        live_bytes=window._live_bytes() if window else 0,
    )


def rank_snapshot(runtime):
    config = runtime.config
    return {
        "rank": runtime.rank,
        "calls": runtime.calls,
        "phase": runtime.kind,
        "operators": dict(
            vars(config.operators),
            video_attention="kitchen",
            refiner_attention="native_sdpa",
            audio_decode="channel_parallel",
        ),
        "adaptive_lookahead": config.adaptive_lookahead,
        "sample_active": runtime.sample_active,
        "decode_complete": runtime.completed_decode,
        "vae_phase": runtime.vae_kind,
        "idle": dict(
            runtime.idle_stats,
            live_bytes=runtime.idle_window._live_bytes() if runtime.idle_window else 0,
        ),
        "host_cache": runtime.cache.snapshot(),
        "startup_cache": runtime.startup_cache.snapshot(),
        "events": runtime.events,
    }
