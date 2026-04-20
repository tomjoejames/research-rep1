# LaTeX Build Instructions

## Contents

```
latex/
├── paper.tex          # Main LaTeX file (IEEE conference format)
├── figures/           # All 10 publication figures (PNG, 300 DPI)
│   ├── throughput_bar.png
│   ├── acc_vs_tps_bubble.png
│   ├── agent_overhead_bar.png
│   ├── ttft_reduction.png
│   ├── di_heatmap.png
│   ├── per_step_latency.png
│   ├── f1_heatmap.png
│   ├── energy_ppr.png
│   ├── di_floor_comparison.png
│   └── ram_utilization.png
└── README.md          # This file
```

## Building Locally

```bash
# Requires: texlive-full or basictex + IEEEtran
pdflatex paper.tex
pdflatex paper.tex     # Run twice for references
```

## Building on Overleaf

1. Create a new project on [Overleaf](https://www.overleaf.com)
2. Upload `paper.tex` and the entire `figures/` directory
3. Set compiler to **pdfLaTeX**
4. Click **Recompile**

## Package Dependencies

- `IEEEtran` (document class)
- `booktabs`, `multirow`, `array` (tables)
- `graphicx` (figures)
- `amsmath`, `amssymb` (equations)
- `hyperref` (clickable links)
- `cite` (bibliography)

All are included in standard TeX Live distributions.

## Notes

- Figures are NOT embedded via `\includegraphics` in the current version to keep the paper concise for the 8-page IEEE limit. To include figures, add `\includegraphics[width=\columnwidth]{filename}` in the appropriate sections.
- The paper uses inline `thebibliography` (no .bib file needed).
- Target venues: IEEE Access, MDPI Electronics, NeurIPS/EMNLP workshop.
