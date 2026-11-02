"""Summarize completed physical two-T4 P0 operator receipts; never extrapolate E2E.

Run only after collecting final Kaggle artifacts, e.g.:
    python benchmarks/report_dit_p0_t4.py PATH_TO/dit_p0/tp2
"""
import argparse
import json
from pathlib import Path

GIB = 2**30
VARIANTS = ("baseline", "p0_3", "p0_2", "p0_1_release",
            "p0_1_inplace", "p0_1", "all_p0")
TARGET_STAGE = {
    "p0_3": "fc2",
    "p0_2": "out_projection",
    "p0_1_release": "fc1",
    "p0_1_inplace": "fc1",
    "p0_1": "fc1",
    "all_p0": "fc1",
}

def stage_peak(geometry, variant, stage):
    rows = geometry["stage_peak_allocated"].get(variant, [])
    values = [row["peak_allocated"] for row in rows if row["phase"] == stage]
    return max(values) if values else None

def rank_geometry(r, g):
    base = g["measurements"]["baseline"]
    output = {
        "rank": r,
        "rows": g["rows"],
        "attention": g["attention"],
        "baseline_raw_balance_peak_gib": base["peak_allocated"] / GIB,
        "baseline_balance_initial_gib": base["initial_allocated"] / GIB,
        "baseline_wrapped_stage_peak_gib":
            max(v["peak_allocated"] for v in g["stage_peak_allocated"]["baseline"]) / GIB,
        "baseline_median_cuda_ms": base["median_cuda_ms"],
        "variants": {},
    }
    for name in VARIANTS:
        m = g["measurements"][name]
        p = g["precision"].get(name, {"byte_exact": True, "finite": True,
                                       "max_abs": 0., "mismatch_fraction": 0.,
                                       "relative_rmse": 0.})
        stage = TARGET_STAGE.get(name, "fc1")
        base_stage = stage_peak(g, "baseline", stage)
        candidate_stage = stage_peak(g, name, stage)
        output["variants"][name] = {
            "median_cuda_ms": m["median_cuda_ms"],
            "cuda_latency_vs_baseline_pct":
                100. * (m["median_cuda_ms"] / base["median_cuda_ms"] - 1.),
            "raw_balance_peak_gib": m["peak_allocated"] / GIB,
            "raw_balance_initial_gib": m["initial_allocated"] / GIB,
            "raw_balance_delta_unqualified_gib":
                (base["peak_allocated"] - m["peak_allocated"]) / GIB,
            "wrapped_stage_peak_gib":
                max(v["peak_allocated"] for v in g["stage_peak_allocated"][name]) / GIB,
            "wrapped_stage_peak_reduction_gib":
                (max(v["peak_allocated"] for v in g["stage_peak_allocated"]["baseline"])
                 - max(v["peak_allocated"] for v in g["stage_peak_allocated"][name])) / GIB,
            "stage": stage,
            "stage_peak_reduction_gib": (
                (base_stage - candidate_stage) / GIB
                if base_stage is not None and candidate_stage is not None else None),
            "byte_exact": p["byte_exact"],
            "finite": p["finite"],
            "max_abs": p["max_abs"],
            "mismatch_fraction": p["mismatch_fraction"],
            "relative_rmse": p["relative_rmse"],
        }
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    receipts = [json.loads((args.folder / f"rank{i}" / "result.json").read_text(encoding="utf-8"))
                for i in range(2)]
    if any(row["status"] != "complete" for row in receipts):
        raise RuntimeError("Both physical T4 ranks must complete before summarizing")
    data = [rank_geometry(rank, geometry)
            for rank, receipt in enumerate(receipts)
            for geometry in receipt["per_geometry"]]
    result = {
        "status": "summarized",
        "source": "Trained base DiT block0, generated residual, no LoRA, no full sampling or H3 E2E",
        "hardware": [receipt["hardware"] for receipt in receipts],
        "all_bit_exact": all(v["byte_exact"] and v["finite"]
                            for row in data for v in row["variants"].values()),
        "per_rank_geometry": data,
        "note": "The balance() raw peaks have unstable live initial allocations "
                "(up to 6 GiB drift between variants/ranks), so raw_balance_delta_unqualified_gib "
                "MUST NOT be interpreted as actual savings. Wrapped-stage peaks use a comparable "
                "phase-local allocator ledger, but DO NOT cover uninstrumented sections and are "
                "NOT full-block/global-graph peaks. Stage deltas are not additive.",
    }
    result["strict_no_latency_regression"] = all(
        v["cuda_latency_vs_baseline_pct"] <= 0.
        for row in data for name, v in row["variants"].items() if name != "baseline")
    output = args.folder.parent / "t4_p0_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("All exact", result["all_bit_exact"],
          "strict no latency regression", result["strict_no_latency_regression"])
    for row in data:
        print("RANK", row["rank"], "S", row["rows"], row["attention"],
              "baseline wrapped-stage peak GiB", round(row["baseline_wrapped_stage_peak_gib"], 4),
              "raw balance initial GiB", round(row["baseline_balance_initial_gib"], 4))
        for name, variant in row["variants"].items():
            if name == "baseline":
                continue
            print(name, "exact", variant["byte_exact"],
                  "wrapped_stage_max_saving_GiB", round(variant["wrapped_stage_peak_reduction_gib"], 4),
                  "target_stage_saving_GiB",
                  round(variant["stage_peak_reduction_gib"], 4)
                  if variant["stage_peak_reduction_gib"] is not None else None,
                  "latency_delta_pct", round(variant["cuda_latency_vs_baseline_pct"], 3))
    print("Written", output)


if __name__ == "__main__":
    main()
