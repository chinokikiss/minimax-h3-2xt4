# Technical guide authoring

[Documentation index](../README.md) · [Published PDF](../guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf)

`build_h3_optimization_guide.py` creates the 40-page guide from explanatory text,
vector diagrams, and the small aggregate measurement records in `docs/data/`.
It embeds a JSON source manifest with paths, SHA256 identities and chart data.
The current revision includes nine pages of integration-audit explanations and
56 source records; see the [coverage audit](../reports/H3_DOCUMENTATION_COVERAGE_AUDIT.md).
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
