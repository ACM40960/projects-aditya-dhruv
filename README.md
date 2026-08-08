# Fantasy Premier League Team Optimisation
### Using Machine Learning and Integer Linear Programming
**ACM 40960 | University College Dublin | Summer 2026**
**Aditya Upasani (25214055) | Dhruv Mehta (25219708)**

---

## Overview

Fantasy Premier League (FPL) is played by over 11 million managers worldwide. Each manager selects 15 real Premier League players within a £100m budget and earns points based on real match performances each gameweek.

This project builds a **leakage-free end-to-end AI pipeline** that:
1. Fetches 196,538 player-gameweek records across 8 seasons from the FPL API and vaastav GitHub dataset
2. Engineers 71 rolling features (form, xG, xA, fixture difficulty, availability) — all shifted by 1 gameweek to prevent data leakage
3. Trains Ridge Regression, XGBoost and Neural Network models with season-grouped expanding-window cross-validation
4. Selects the optimal 15-player squad using Integer Linear Programming (ILP)
5. Evaluates against a naive previous-season baseline across a full held-out test season (2023/24)

---

## Key Results

| Metric | Value |
|---|---|
| Best model | Ensemble: Ridge + XGBoost + Neural Network |
| R2 (test season 2023/24) | 0.308 |
| MAE | 0.932 pts per gameweek |
| Spearman rank correlation | 0.681 |
| ML+ILP squad aggregate pts | **2,415** |
| Baseline squad aggregate pts | 2,075 |
| Improvement | **+340 pts (+16.4%)** |

> Aggregate actual season points = sum of 2023/24 FPL points across all 15 ILP-selected players accumulated gameweek by gameweek. Starting XI selection, captaincy, chips and transfers are not simulated.

---

## Project Structure
projects-aditya-dhruv/
|
|-- notebooks/
| |-- 01_data_collection.ipynb # FPL API + vaastav data fetching
| |-- 02_eda.ipynb # Exploratory data analysis
| |-- 03_feature_engineering.ipynb # Season-level feature engineering
| |-- 04_xgboost_model.ipynb # XGBoost regression model
| |-- 05_neural_network.ipynb # MLP Neural Network model
| |-- 06_squad_optimisation.ipynb # ILP squad selection
| |-- 07_evaluation.ipynb # Corrected leakage-free evaluation
| -- 08_gameweek_pipeline.ipynb      # Full gameweek-level pipeline (main) | |-- src/ |   |-- features.py                     # Feature engineering helper module |   -- optimiser.py # ILP squad optimiser module
|
|-- data/
| |-- raw/ # Raw API responses (not tracked)
| -- processed/                      # Cleaned CSVs and plots | |-- models/                             # Trained model files |-- requirements.txt                    # Python dependencies -- README.md
---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/ACM40960/projects-aditya-dhruv.git
cd projects-aditya-dhruv
```

### 2. Create virtual environment (Python 3.11 required)
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run notebooks in order

01_data_collection.ipynb -> fetches all data
02_eda.ipynb -> exploratory analysis
03_feature_engineering.ipynb
04_xgboost_model.ipynb
05_neural_network.ipynb
06_squad_optimisation.ipynb
07_evaluation.ipynb -> corrected leakage-free results
08_gameweek_pipeline.ipynb -> main contribution

---

## Methodology

### Data
- **Source:** Official FPL API + vaastav Fantasy-Premier-League historical dataset
- **Coverage:** 8 seasons (2016/17 to 2023/24), 196,538 raw player-gameweek records
- **After cleaning:** 105,859 modelling samples
- **Train:** 2020/21 to 2022/23 | **Test:** 2023/24 (held out throughout)

### Features (71 total)
| Group | Features |
|---|---|
| Player form | Rolling 3 and 5 GW points, xG, xA, minutes |
| Fixture | Opponent attack/defence strength (FDR), home/away |
| Availability | Injury proxy flag, started last GW |
| Team form | Recent goals scored and conceded |
| Static | Price (start_price), position encoding, season number |

All rolling features are **shifted by 1 gameweek** before training to prevent data leakage. Pre-season start_price is used, not end-of-season price.

### Models
| Model | R2 | MAE | RMSE | Spearman |
|---|---|---|---|---|
| Baseline (3GW avg) | 0.130 | 1.008 | 2.130 | 0.677 |
| Ridge Regression | 0.303 | 0.981 | 1.907 | 0.671 |
| XGBoost | 0.295 | 0.957 | 1.918 | 0.683 |
| Neural Network | 0.296 | 0.900 | 1.916 | 0.663 |
| XGB + NN | 0.305 | 0.919 | 1.903 | 0.682 |
| Ridge + XGB | 0.305 | 0.961 | 1.904 | 0.680 |
| **Ridge + XGB + NN** | **0.308** | **0.932** | **1.900** | **0.681** |

### Validation
Season-grouped expanding-window cross-validation. Each fold trains on all seasons before the validation season. 2023/24 held out as final test season throughout.

### Squad Optimisation (ILP)

Maximise: sum(p_hat_i * x_i)
Subject to:
sum(c_i * x_i) <= 100 (budget)
sum(x_i) = 15 (squad size)
2 GKP, 5 DEF, 5 MID, 3 FWD
<= 3 players per club
x_i in {0, 1}

---

## Results

### Full Season Evaluation (GW-by-GW ILP, 2023/24)

| Squad | Aggregate Actual Points |
|---|---|
| ML + ILP (Ensemble Ridge+XGB+NN) | **2,415** |
| Baseline (prev season ILP) | 2,075 |
| **Improvement** | **+340 pts (+16.4%)** |

### Season-Level Baseline
| Model | R2 | MAE | RMSE |
|---|---|---|---|
| Prev season pts | 0.207 | 31.74 | 47.02 |
| **Ridge** | **0.377** | **30.64** | **41.68** |
| XGBoost | 0.320 | 32.15 | 43.55 |
| Neural Network | 0.341 | 31.42 | 42.86 |

Season-level ILP squad: **1,895 pts** vs baseline 1,613 pts (+17.5%)

---

## Limitations

1. Confirmed lineups, late injuries and rotation are not fully observable from historical data
2. Player-gameweek records within a season are temporally and within-player correlated
3. Starting XI selection, captaincy, chips, transfers and automatic substitutions are not simulated
4. Aggregate squad points are not an official FPL manager score

---

## Dependencies

- Python 3.11
- pandas, numpy, scikit-learn
- xgboost, tensorflow/keras
- pulp (ILP solver)
- shap (model interpretability)
- matplotlib, seaborn, requests

See requirements.txt for full list.

---

## Authors

- **Aditya Upasani** (25214055) — data pipeline, feature engineering, model training, ILP optimisation, evaluation
- **Dhruv Mehta** (25219708) — neural network architecture, helper modules, documentation

ACM 40960 | University College Dublin | Summer 2026

