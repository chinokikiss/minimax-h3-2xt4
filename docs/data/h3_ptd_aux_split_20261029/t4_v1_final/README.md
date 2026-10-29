# PTD P2 final T4 evidence

> Historical reproduction: the analyzer and P2 runtime were removed from active source. The command below refers to [source commit b79b1bc](https://github.com/chinokikiss/minimax-h3-2xt4/tree/b79b1bce1b8450a250e58bb8c8676630e58b4f59). Raw measurements and receipts are unchanged.

Kaggle `pamcks/h3-ptd-aux-split-t4-20261029`, version 1. Four eight-hop cases completed on two distinct SM75 Tesla T4s. Result status is `complete`; final supervisor exit code is 0. The split format is correct but increases this fixture's mean RPC time 16.5%.

- `ptd_result.json.gz`: losslessly compressed original 9,072,697-byte result. Decompressed SHA256 is `812cad379983a8ee181dfd1d24cfb18a79b80e352b0d26d7cc7af6c4481cb2d1`.
- `summary.json`: independently checked per-case metrics and group means, matching all eight durable worker reports and all output hashes.
- `0_monolithic` through `3_monolithic_repeat`: complete worker reports and AOT receipts. Empty `unused_te.safetensors` placeholders are omitted; no model payloads are archived here.
- `setup_receipt.json.gz`, `delivery_binding.json`: exact frozen source/API/environment provenance. The setup receipt is preserved losslessly.
- `supervision.json`, `worker_execution.log`: final experiment exit and execution evidence.
- `partial_transfer*`, `completed_transfer*`: interrupted transfer and successful filtered resume receipts/logs. Previous artifacts were unchanged during resume.
- `download_manifest.json`: hashes of all 32 original collected artifacts. It describes the download directory; this selected archive uses gzip for the large result/setup receipt and omits bootstrap scripts and empty placeholders.
- `comparison.png`: regenerated from the independent summary; nested CPU spans are not additive.

Recheck from the repository root using the `comfyui` conda Python:

```powershell
& 'C:\ProgramData\miniconda3\envs\comfyui\python.exe' benchmarks/analyze_ptd_aux_split.py --root docs/data/h3_ptd_aux_split_20261029/t4_v1_final --submission docs/data/h3_ptd_aux_split_20261029/t4_submission_v1.json --output D:\attention_eval\h3_ptd_aux_split_20261029\rechecked_summary.json
```

The checker uses only the standard library unless `--figure` requests Matplotlib. It makes no network, Kaggle or CUDA calls. It verifies source identity, physical device receipts, supervisor exit, every output-hash comparison, stable queue ownership, once-only CPU preparations, per-rank verified consumption, completed worker spans and teardown.

Original local download: `D:\attention_eval\h3_ptd_aux_split_20261029\gpu_v1\gpu_final_retry1`.

[Results, limitations and next targets](../../../reports/H3_PTD_AUXILIARY_SPLIT_T4_RESULTS_20261029.md).
