# P0/P1/P2 Kaggle recovery

The queued revision-3 instance was subsequently marked
`CANCEL_ACKNOWLEDGED`. At the user's request, a separate private notebook was
submitted successfully as revision 1:
`chinokiki9/minimax-h3-p0-p1-p2-audit-restart`. Its scoped download links were
renewed before submission. The benchmark source and qualifications are
unchanged. Current restart status/evidence is under
`kaggle_p012_restart_v1_output`; submission provenance is in
`semantic_results/kaggle_p012_restart_v1/submission.json`. The old evidence
watcher was stopped; the new watcher follows only the new notebook.

The Direct Loading Audit revision 2 reports `ERROR`. Kaggle returned no
failure message, an empty persisted log (`[]`), and zero output files. The
locally captured live log ends with a pip dependency-conflict warning. That
warning alone does not establish the failure cause. No performance result
from revision 2 is accepted. Evidence: `semantic_results/kaggle_p012_error_v2/failure.json`.

Revision 3 was accepted for physical dual T4. It reuses the already published
200 Native-proven offline shards at immutable HF commit
`d3e6a4b131056d2c626a01acf6ba48123a885783`. It neither reconverts those shards
nor uploads checkpoint weights from this machine.

The recovery changes the bootstrap, not model operators:

- Download offline shards immediately, concurrently with installation and
  original checkpoint setup, while the one-hour scoped links are fresh.
- Save the manifest before link/disk checks, so those failures have evidence.
- Check disk space for both checkpoint sets plus 12 GiB installation headroom.
- Disable pip's download cache; emit stage starts, exits, and 30-second disk/RAM
  progress diagnostics. Complete downloads before timed GPU work.
- Stream profiler evidence to disk when retrieving results; avoid buffering
  whole trace files in local RAM.

Existing cold/warm local dual-rank qualification passes for all four wrapper
configurations. Numerical production code is unchanged. Qualification is
semantic evidence only, not physical performance evidence.

The physical matrix is fresh Native, direct synchronous, direct P1, direct
P1+P2, and offline P1+P2. Every TP request must match five Native outputs.
`analyze_loading_audit.py` isolates P1, P2, and offline/direct differences and
rejects incomplete, incorrect, or unmatched physical workloads. One cold/warm
pair remains descriptive rather than statistically established.

`watch_loading_audit.py` captures this submission's live logs, downloads saved
evidence at completion/error, and analyzes completed cases. It never submits,
restarts, or cancels benchmarks. Execution status is written to
`kaggle_p012_v3_output/status.json`; submission identity is in
`semantic_results/kaggle_p012_v3/submission.json`.

Native totals include reference artifact IO and different startup boundaries;
they are not directly compared to TP request totals. Profiler traces cover an
extra full-depth DiT evaluation, not TE/VAE. Cache counter snapshots include
validation/background preparation; P2 decode overlap uses explicit monotonic
CPU event intervals. NVML bus bytes are sampled estimates. These limitations
must accompany any latency conclusion.

## October 11 spawn repair

The repaired audit v1 completed Native (608.781 s request; 622.536 s enclosing
process) but failed TP startup after 5.63 s: the loading wrapper had replaced
the `AuditConfig` dataclass symbol with a factory, breaking multiprocessing
pickling. Service construction now applies the policy while retaining the
original importable dataclass. Four compact two-rank cold/warm runs passed all
40 Native bit-exact output checks; eight spawned CPU workers verified the
configuration transport. Model operators are unchanged.

The new continuation is `chinokiki9/minimax-h3-p012-spawn-repaired-audit` v2.
Kaggle refused the failed notebook as a mounted input in v1, so v2 retrieves
its six saved Native artifacts directly with verified byte counts and SHA256.
The live log confirmed all artifacts and unchanged model sources before setup.
At 2026-10-11 04:25 UTC it was RUNNING. Native is not rerun; all four TP cases
remain pending. The watcher PID is 15904, evidence under
`kaggle_p012_spawn_repaired_v2_output`; it does not submit retries.

See `KAGGLE_SPAWN_AND_SHARD_SCALES.md` for the independent activation-scale
proposal, its accuracy implications, and the required separate comparison.

## Completed version 2

Kaggle now reports COMPLETE. All four cold/warm TP configurations finished,
and downloaded evidence passes local revalidation: 40 Native bit-exact output
comparisons, no pending cases. Direct synchronous is fastest cold (392.893 s);
offline P1/P2 is fastest warm (358.179 s), 7.08% below direct synchronous warm.
Only one pair per case was measured. The run uses Native VAEs and does not
qualify the newly integrated parallel VAE default. See [results and next actions](../reports/P012_T4_RESULTS.md).
