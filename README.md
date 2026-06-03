# Uncertainty-Aware Machine Learning for Electronic Stopping Cross-Sections

Calibrated probabilistic prediction of electronic stopping cross-sections (ESCS) with
physics-informed descriptors, spanning elemental **and** compound targets over seven
energy decades, with recovery of Z₁-scaling physics and an independent benchmark.

## Key results (held-out test set)

| Model | R² | MAPE | ECE | Uncertainty |
|---|---|---|---|---|
| HistGradientBoosting | 0.994 | 8.8 ± 0.4% | — | none (accuracy ceiling) |
| NGBoost | 0.990 | 12.6 ± 0.5% | **0.006 ± 0.002 (native)** | probabilistic |
| Deep Ensemble | 0.988 | 15.3 ± 1.0% | 0.097 ± 0.010 | overconfident |
| Variational BNN (+temp) | — | 19.3 ± 3.6% | 0.099 → 0.025 | full Bayesian |

- **Calibrated uncertainty:** NGBoost is natively calibrated (ECE 0.006 ± 0.002); the BNN's raw ECE 0.099 falls to 0.025 after one-parameter temperature scaling (T = 0.63). All metrics are mean ± std over 3 seeds.
- **Physics recovery:** the BNN's effective Z₁ exponent rises 0.56 → **2.04** with energy, reproducing the Bethe–Bloch Z₁² limit.
- **Independent benchmark (atomic, 2,376 pts):** ESPNN 6.0% vs our gradient boosting 7.9% MAPE.
- **Robustness:** accuracy is insensitive to the source of mean excitation energies (additivity vs measured ICRU).

## Repository structure
```
src/pipeline.py           end-to-end reproducible pipeline (data → models → calibration → physics)
src/build_composition.py  builds the target composition table
src/espnn_benchmark.py    optional independent benchmark vs ESPNN
data/                     target_composition_table.csv (+ note on obtaining raw IAEA data)
figures/                  six publication figures
manuscript/               Q1 manuscript (.docx) and final research report
models/                   trained weights (download from Zenodo or regenerate)
```

## Reproduce
```bash
pip install -r requirements.txt
# place StoppingPower.csv, StoppingPower_refs.csv in data/ (see data/README.md)
python src/pipeline.py        # writes outputs/ : features, weights, results, alpha curve
python src/espnn_benchmark.py # optional, needs ESPNN (see script header)
```
A GPU is recommended for the Bayesian neural network. See `REPRODUCE.md` for the full step map.

## Data availability
The target composition table, code, figures, and trained models are released here / on Zenodo.
The **raw experimental data** derives from the IAEA stopping-power database and is **not redistributed**;
obtain it from the IAEA and place it in `data/` (see `data/README.md`).

## Citation
See `CITATION.cff`. A Zenodo DOI will be minted on release and added to the manuscript.

## License
Code: MIT (`LICENSE`). Composition table and figures: CC-BY 4.0.
