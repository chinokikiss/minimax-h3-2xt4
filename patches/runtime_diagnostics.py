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
        "checkpoint_admission": {k:{
            "verify_payloads": getattr(r,"verify_payloads",None),
            "metadata_only": getattr(r,"verify_payloads",None) is False,
            "identity_has_metadata_marker": "metadata-checked-weights-v1" in tuple(map(str,getattr(r,"identity",()))),
        } for k,r in runtime.readers.items()},
        "operators": dict(
            vars(config.operators),
            dit_attention_schedule=getattr(config,"dit_attention_schedule",()),
            sol_tau=getattr(config,"sol_tau",1.0),
            sol_topology_v2=getattr(config,"sol_topology_v2",False),
            sol_sync_free_lut=getattr(config,"sol_sync_free_lut",False),
            sol_collect_route_stats=getattr(config,"sol_collect_route_stats",False),
            sol_temporal_radius=getattr(config,"sol_temporal_radius",1),
            sol_local_radius_tokens=getattr(config,"sol_local_radius_tokens",64),
            sol_protect_frame_boundaries=getattr(config,"sol_protect_frame_boundaries",True),
            video_attention=getattr(config,"video_attention","kitchen"),
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
        "aot_preparation": dict(getattr(runtime,"aot_receipts",{})),
        "sol_route_stats": __import__('patches.sol_attention',fromlist=['route_stats_snapshot']).route_stats_snapshot()
            if getattr(config,"sol_collect_route_stats",False) else [],
        "events": runtime.events,
    }
