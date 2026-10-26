# Technical guide authoring

[Documentation index](../README.md) · [Published PDF](../guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf)

`optimization_techniques.py` is the canonical catalogue for the technical guide.
`build_h3_optimization_guide.py` calls `build_technique_guide.py` to generate the
same content as a [Markdown guide](../guides/H3_OPTIMIZATION_TECHNIQUES.md) and
the published PDF. The 2026-10-26 edition has **130 techniques in ten areas,
50 pages and 97 implementation sources**. It describes mechanisms, work/storage
removed and numerical/ownership contracts. It contains no benchmark charts or
phenomena narrative; measurement reports remain separate.

The [coverage matrix](../data/h3_optimization_coverage_20261026.json) maps all
64 runtime modules, all 52 prior audited items and retained native kernels.
Active, alternative, experimental, historical and retired techniques are all
included. The PDF embeds a source-hash manifest and navigation bookmarks.
No Torch import, model loading, CUDA work or Kaggle submission is needed.

Run from the repository root using a **separate document environment** with
ReportLab and pypdf, plus Windows Segoe UI/Consolas fonts:

```powershell
python docs/pdf/build_h3_optimization_guide.py
pdftoppm -r 72 -png docs/guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf output/pdf/technique_page
python docs/pdf/verify_technique_guide.py
```

Do not install authoring packages into the frozen inference environment.
The PDF is written to `docs/guides/`; layout and source-manifest sidecars go to
the ignored `output/pdf/` directory. Re-render and visually inspect all affected
pages after editing. The verifier checks every technique heading, the embedded
manifest and safe text geometry, then produces contact sheets for visual review.
Local authoring uses the bundled document runtime and Poppler.
