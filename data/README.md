# Data Directory

## Structure

- raw/ — Raw JSON and CSV files from FPL API (not tracked by git)
- processed/ — Cleaned CSVs and visualisation outputs

## Key files in processed/

- players_season.csv — Season-level player stats
- player_past_seasons.csv — Historical season data per player
- features.csv — Engineered feature matrix (season-level)
- player_gameweek_history.csv — Gameweek-level history
- xgb_predictions.csv — XGBoost model predictions
- nn_predictions.csv — Neural Network predictions
- optimal_squad.csv — ILP-selected optimal squad
- metrics_summary.txt — Full evaluation metrics

## Author
Dhruv Mehta - ACM 40960, University College Dublin
