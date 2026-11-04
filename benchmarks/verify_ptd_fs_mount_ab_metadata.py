"""One-shot authenticated exact-ID metadata verification for the synthetic mount A/B.

No dataset listing, kernel-status queries, live logs or automatic retries.
"""
import argparse
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

HANDLES = [
    "pamcks/h3-ptd-fsmount-ab-private-20261103",
    "pamcks/h3-ptd-fsmount-ab-public-20261103",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "format": "h3-ptd-fsmount-ab-provider-exact-identity-v1",
        "scope": "two exact Dataset metadata reads; no status/list polling",
        "results": [],
    }
    api = KaggleApi()
    try:
        api.authenticate()
        with api.build_kaggle_client() as client:
            for handle in HANDLES:
                owner, slug = handle.split("/")
                request = ApiGetDatasetRequest()
                request.owner_slug, request.dataset_slug = owner, slug
                try:
                    data = client.datasets.dataset_api_client.get_dataset(request)
                    row = {
                        "requested_handle": handle, "status": "found",
                        "dataset_id": getattr(data, "dataset_id", None),
                        "version": getattr(data, "current_version_number", None),
                        "is_private": getattr(data, "is_private", None),
                        "versions_count": len(data.versions) if getattr(data, "versions", None) else None,
                    }
                    if row["dataset_id"] is not None:
                        row["dataset_id"] = str(row["dataset_id"])
                    if row["is_private"] is not None:
                        row["is_private"] = bool(row["is_private"])
                except Exception as exc:
                    row = {"requested_handle": handle, "status": "unverified",
                           "error_type": type(exc).__name__}
                report["results"].append(row)
    except Exception as exc:
        report["auth_error_type"] = type(exc).__name__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
