# Aggregate evidence records

[Documentation index](../README.md)

These small JSON records support the published reports and PDF charts. Raw
captures, traces, model weights, and private provider capabilities are not included.

| Record | Scope |
|---|---|
| [Operator integration](H3_OPERATOR_INTEGRATION_RESULTS.json) | Preserved integration gate/source record |
| [Pipeline review](H3_PIPELINE_REVIEW_RESULTS.json) | Recorded cleanup and validation checks |
| [TP2 P2 investigation](TP2_P2_RESULTS.json) | Historical attention/transport experiment matrix |
| [FC2 T4 v1 summary](fc2_quantizer_t4_v1_summary.json) | Scoped physical T4 quantizer/FC2/FFN measurements |
| [FC2 T4 v2 summary](fc2_quantizer_t4_v2_summary.json) | Integrated S32768 confirmation |
| [Documentation coverage](h3_documentation_coverage_20261020.json) | Static mapping of 52 implementation items, 85 documents and 119 scanned experiment families; no new hardware result |

The original JSON bytes are preserved. Embedded paths and hashes refer to the
source snapshots recorded at measurement time; moving the file does not rewrite
that history. See the [path index](../path-index.json) for current document locations.
