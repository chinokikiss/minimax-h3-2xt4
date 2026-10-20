"""Read saved inputs/references from the completed allocation; starts no job."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from kaggle.api.kaggle_api_extended import KaggleApi
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "slab_results/saved_references")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / "kaggle_retest_output/cu130/retest_results/round_0/video_tiling/results.json"
    recorded = json.loads(evidence.read_text())
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    client._http_client._init_session()
    session = client._http_client._session
    records = []
    for name in ["768x768_1F", "768x1344_2F"]:
        kinds = ["input"] + (["native"] if args.include_reference else [])
        for kind in kinds:
            filename = f"{kind}_{name}" + ("_output.pt" if kind == "native" else ".pt")
            target = args.output / filename
            remote_path = f"retest_results/round_0/video_tiling/{filename}"
            if not target.exists():
                url = "https://www.kaggle.com/api/v1/kernels/output/download/chinokiki9/minimax-h3-cuda13-retest/" + remote_path
                for attempt in range(3):
                    try:
                        response = session.get(url, timeout=(8, 30))
                        response.raise_for_status()
                        target.write_bytes(response.content)
                        break
                    except Exception as exc:
                        # Never expose signed URLs or bearer credentials.
                        if attempt == 2:
                            print(f"FAILED {filename}: {type(exc).__name__}", flush=True)
                            sys.exit(1)
            tensor = torch.load(target, map_location="cpu", weights_only=True)
            digest = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
            shape = recorded["workloads"][name]["shape"]
            if kind == "input":
                assert list(tensor.shape) == shape
                assert digest == recorded["workloads"][name]["input_sha256"]
            else:
                assert list(tensor.shape) == [shape[0], 3, shape[2] * 4, shape[3] * 16, shape[4] * 16]
                assert torch.isfinite(tensor).all()
            records.append({"file": filename, "tensor_sha256": digest, "shape": list(tensor.shape),
                            "source": remote_path, "dtype": str(tensor.dtype)})
            print(f"Verified {filename}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
