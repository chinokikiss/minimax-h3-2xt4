# Technical guide authoring

[Documentation index](../README.md) · [Published PDF](../guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf)

`build_h3_optimization_guide.py` creates the updated guide from explanatory text,
vector diagrams, and the small aggregate measurement records in `docs/data/`.
It embeds a JSON source manifest with paths, SHA256 identities and chart data.
The 2026-10-22 revision adds the promoted FP16-fix/fusion/transport default,
physical T4 charts, Video final-projection localization and retired audio
parallelism. The latest 46-page guide also covers FP32 Row-A hidden SUM and
reference-demand encoder loading. The source catalogue includes 66 records; see the [coverage audit](../reports/H3_DOCUMENTATION_COVERAGE_AUDIT.md).
No Torch import, model loading, CUDA work, or Kaggle submission is needed.

Run from the repository root using a **separate document environment** with
ReportLab and pypdf, plus Windows Segoe UI/Consolas fonts:

```powershell
python docs/pdf/build_h3_optimization_guide.py
```

Do not install authoring packages into the frozen inference environment.
The PDF is written to `docs/guides/`; layout and source-manifest sidecars go to
the ignored `output/pdf/` directory. Re-render and visually inspect all affected
pages after editing. Local creation used the bundled document runtime and Poppler.
