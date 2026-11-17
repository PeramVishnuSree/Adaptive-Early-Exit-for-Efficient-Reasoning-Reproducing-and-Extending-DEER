# Check-In 4 Overleaf project

This folder is a self-contained IEEE conference draft for CMPE 258 Group 10, Check-In 4.

## Open it on Overleaf

1. Zip this folder (`main.tex`, `figures/`, and this README).
2. In Overleaf, choose **New Project → Upload Project** and upload that zip.
3. Set the main document to `main.tex` if Overleaf does not select it.
4. Set the compiler to **pdfLaTeX**. Overleaf already provides `IEEEtran`.

No extra style file is included. `graphicx`, `amsmath`, `booktabs`, `array`, `url`, `float`, and `dblfloatfix` are part of a standard TeX Live install.

`figures/` holds every project plot used in the draft: the 20-question GSM8K reproduction, the dataset summaries, the Check-In 3 live and answer-fix comparisons, the stopping-method curves, the held-out controller curves, and the AdaDEER-E tradeoffs. `main.pdf` in this folder is the compiled report. The upstream DEER paper's own result images are not reprinted here.
