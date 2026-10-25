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
        "bundle_offline_lora": getattr(config,'bundle_offline_lora',False),
        "bundle_preparation": {k:dict(r.stats) for k,r in runtime.readers.items() if hasattr(r,'offline_targets')},
        "checkpoint_io": {k:{name:getattr(r,name,None) for name in
            ('logical_read_bytes','source_read_bytes','range_read_calls')} for k,r in runtime.readers.items()},
        "operators": dict(
            vars(config.operators),
            video_attention="kitchen",
            refiner_attention="native_sdpa",
            audio_decode="native_single_gpu",
        ),
        "adaptive_lookahead": config.adaptive_lookahead,
        "video_decode_reserve_bytes": getattr(config, "video_decode_reserve_bytes", None),
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
