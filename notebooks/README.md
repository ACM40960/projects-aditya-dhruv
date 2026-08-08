# Notebooks Guide

Run notebooks in the following order:

| Notebook | Description |
|---|---|
| 01_data_collection.ipynb | Fetches player and season data from FPL API |
| 02_eda.ipynb | Exploratory analysis: distributions, correlations, trends |
| 03_feature_engineering.ipynb | Lag features, xG metrics, position encoding |
| 04_xgboost_model.ipynb | XGBoost regressor with SHAP feature importance |
| 05_neural_network.ipynb | MLP Neural Network with training history |
| 06_squad_optimisation.ipynb | ILP squad selection with ensemble predictions |
| 07_evaluation.ipynb | Corrected evaluation: leakage-free, expanding-window CV |
| 08_gameweek_pipeline.ipynb | Full gameweek-level pipeline with FDR and availability |

## Notes
- All notebooks use the venv Python 3.11 kernel
- Run 01 first to generate raw data
- Processed data saved to data/processed/ automatically
- Models saved to models/ after training
- All rolling features shifted by 1 GW to prevent leakage

## Key Results
- Best model: Ensemble Ridge+XGB+NN (R2=0.308, Spearman=0.681)
- ML+ILP squad: 2,415 aggregate actual points vs 2,075 baseline (+16.4%)
