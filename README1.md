# Trained model artifacts

These binaries are produced by `src/pipeline.py` (and are also archived on Zenodo).
If you did not run the pipeline yourself, download them from the Zenodo release and
place them here:

- `bnn_weights.pt`        — variational Bayesian neural network weights
- `ngboost_model.pkl`     — trained NGBoost model
- `features_full.parquet` — featurised + split dataset (regenerate via pipeline if you have the raw data)
