"""Reproduce the CPU dataset study summaries from final artifacts, offline only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath
import statistics

MIB = 2**20


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_final(directory, receipt, kind):
    manifest = load(directory / "download_manifest.json")
    for name, expected in manifest.items():
        data = (directory / PureWindowsPath(name).as_posix()).read_bytes()
        assert len(data) == expected["bytes"], name
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], name
    source = load(directory / f"dataset_{kind}" / "source_binding.json")
    assert source == load(receipt)["source_binding"], kind
    code = (directory / f"dataset_{kind}" / source["module"]).read_bytes()
    assert hashlib.sha256(code).hexdigest() == source["source_sha256"], kind
    return {"files_verified": len(manifest), "source_binding": source}


def memory_summary(trials):
    samples = [s for t in trials for s in t["samples"]]
    stats = [dict(line.split() for line in s["cgroup"]["memory.stat"].splitlines())
             for s in samples]
    return {
        "peak_process_rss_bytes": max(t["peak_rss"] for t in trials),
        "peak_trial_rss_growth_bytes": max(t["peak_rss"] - t["baseline_rss"] for t in trials),
        "peak_cgroup_current_bytes": max(int(s["cgroup"]["memory.current"]) for s in samples),
        "peak_cgroup_anon_bytes": max(int(s["anon"]) for s in stats),
        "peak_cgroup_file_bytes": max(int(s["file"]) for s in stats),
        "max_cgroup_pgscan": max(int(s["pgscan"]) for s in stats),
        "max_cgroup_pgsteal": max(int(s["pgsteal"]) for s in stats),
        "memory_pressure_samples": sorted({s["cgroup"]["memory.pressure"] for s in samples}),
    }


def analyze(root, output):
    read_dir = root / "cpu_read/final_v1"
    upload_dir = root / "cpu_upload/retry2/final_v2"
    provenance = {
        "read": verify_final(read_dir, root / "cpu_read/submission.json", "read"),
        "upload": verify_final(upload_dir, root / "cpu_upload/retry2/submission.json", "upload"),
    }
    read = load(read_dir / "dataset_read/result.json")
    upload = load(upload_dir / "dataset_upload/result.json")
    assert read["status"] == upload["status"] == "complete"
    trials = read["trials"]
    assert len(trials) == 74 and not any(t["errors"] for t in trials)
    rows = []
    for count in (1, 2, 4, 6):
        for workers in (1, 2, 4, 8, 16, 32, 64):
            selected = [t for t in trials if t["cohort"] == f"{count}_datasets"
                        and t["workers"] == workers and t["drop_cache_hint"]]
            assert len(selected) == 2
            rates = [t["mib_per_second"] for t in selected]
            value = statistics.median(rates)
            original = next(r for r in read["median_curves"][f"{count}_datasets"]
                            if r["workers"] == workers)
            assert abs(value - original["mib_per_second_median"]) < 1e-8
            rows.append(dict(datasets=count, readers=workers, median_mib_s=value,
                             minimum_mib_s=min(rates), maximum_mib_s=max(rates),
                             repetitions=len(selected)))
    warm = [dict(datasets=t["dataset_count"], readers=t["workers"],
                 mib_s=t["mib_per_second"], seconds=t["seconds"])
            for t in trials if not t["drop_cache_hint"]]
    summary = {
        "scope": "CPU mounted reads and 16 MiB synthetic private upload requests; no T4 result",
        "provenance": provenance,
        "read": {
            "trials": len(trials), "seconds": read["seconds"],
            "cpu_count": read["cpu_count"], "cpu_quota": read["cpu_quota"],
            "curves": rows,
            "best_by_cohort": [max((r for r in rows if r["datasets"] == n),
                                   key=lambda r: r["median_mib_s"]) for n in (1, 2, 4, 6)],
            "normal_cache_controls_single_measurement": warm,
            "memory_all_trials": memory_summary(trials),
            "memory_dontneed_trials": memory_summary([t for t in trials if t["drop_cache_hint"]]),
            "max_payload_buffer_bytes": max(t["buffer_cap_bytes"] for t in trials),
            "inventory": [dict(root=name, files=len(files), bytes=sum(f["bytes"] for f in files))
                          for name, files in zip(read["dataset_roots"], read["datasets"])],
            "dataset_mount_lines": [line for line in read["mountinfo"].splitlines()
                                    if "/melanoma-" in line],
            "caveats": read["caveats"] + [
                "Two repetitions, heterogeneous corpus sizes and 0.2-second sampling limit precision.",
                "Trials stop at 12 seconds, 4 GiB, or corpus exhaustion; not a sustained hard ceiling.",
                "Shared NFS endpoint is observed; precise bottleneck is not isolated.",
            ],
        },
        "upload": {
            "versions": upload["versions"], "payload_bytes": upload["payload_bytes"],
            "payload_sha256": upload["payload_sha256"], "methods": upload["methods"],
            "credential_presence": upload["credential_presence_after_secret_lookup"],
            "interpretation": [
                "API private creation and Hub version upload returned acknowledgements.",
                "Processing completion and round-trip content integrity were not checked.",
                "CLI did not upload: native 2.0.2 has no python -m kaggle entry point.",
                "Console entry-point correction is local and not Kaggle-upload validated.",
                "Create and version requests are different operations; timings do not rank methods.",
            ],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "read_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for count in (1, 2, 4, 6):
        curve = [r for r in rows if r["datasets"] == count]
        x = [r["readers"] for r in curve]
        y = [r["median_mib_s"] for r in curve]
        line, = axes[0].plot(x, y, "o-", label=f"{count} dataset(s)")
        axes[0].fill_between(x, [r["minimum_mib_s"] for r in curve],
                             [r["maximum_mib_s"] for r in curve],
                             alpha=.12, color=line.get_color())
    axes[0].set(xscale="log", xticks=[1, 2, 4, 8, 16, 32, 64], ylim=(0, 1000),
                xlabel="Total readers", ylabel="MiB/s",
                title="DONTNEED hints: median and range of 2 trials")
    axes[0].set_xticklabels([1, 2, 4, 8, 16, 32, 64])
    axes[0].legend()
    for workers, marker in ((1, "o"), (64, "s")):
        selected = [r for r in warm if r["readers"] == workers]
        axes[1].plot([r["datasets"] for r in selected], [r["mib_s"] for r in selected],
                     marker + "-", label=f"{workers} reader" + ("s" if workers != 1 else ""))
    axes[1].set(xticks=[1, 2, 4, 6], ylim=(0, 7000), xlabel="Active datasets", ylabel="MiB/s",
                title="Normal-cache controls: single measurements")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle("Kaggle CPU dataset reads — 2026-10-30\n"
                 "Shared NFS endpoint; advisory hints do not establish cold storage", fontsize=12)
    fig.savefig(output / "read_throughput.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.evidence_root, args.output)
    print(json.dumps({"trials": result["read"]["trials"],
                      "best_by_cohort": result["read"]["best_by_cohort"],
                      "memory_all_trials": result["read"]["memory_all_trials"],
                      "memory_dontneed_trials": result["read"]["memory_dontneed_trials"]}, indent=2))
