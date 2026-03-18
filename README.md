# ML-FutVE-Predictions

Machine learning system for predicting **Liga FUTVE** (Venezuelan First Division) match outcomes using historical data, Elo ratings, and gradient boosting.

## Overview

This project builds a complete ML pipeline that:

1. **Collects** 6,592 historical matches across 25 seasons (2002–2026) involving 42 teams
2. **Engineers** 46 pre-match features from raw match data (Elo ratings, form, streaks, head-to-head, etc.)
3. **Trains** a CatBoost classifier to predict match outcomes: **Home win (H)**, **Draw (D)**, or **Away win (A)**
4. **Serves** predictions through a FastAPI REST API

## Model Performance

The model is evaluated on a held-out test set of **718 matches** from the 2023–2026 seasons. Training uses all prior seasons (2002–2022), ensuring a strict chronological split with **no data leakage**.

### Accuracy Comparison

| Model | Accuracy |
|---|---|
| **CatBoost** | **47.49%** |
| Logistic Regression | 43.73% |
| Baseline (always predict Home) | 43.45% |
| Random Forest | 42.62% |

CatBoost outperforms all other models by **+3.76 percentage points** over the naive baseline.

### Per-Class Metrics (CatBoost)

| Outcome | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Home win (H)** | 0.57 | 0.54 | 0.55 | 312 |
| **Draw (D)** | 0.39 | 0.28 | 0.32 | 207 |
| **Away win (A)** | 0.41 | 0.59 | 0.49 | 199 |
| **Weighted avg** | **0.48** | **0.47** | **0.47** | **718** |

### Confusion Matrix

```
              Pred H    Pred D    Pred A
Actual H        167        56        89
Actual D         74        57        76
Actual A         50        32       117
```

### How to Read These Results

- **Home wins** are the easiest to predict (0.57 precision, 0.55 F1) because home advantage is a strong and consistent signal in Venezuelan football — 45.4% of all matches end in a home win.
- **Draws** are the hardest to predict (0.32 F1) — this is expected across all football prediction models, since draws are inherently the most random outcome.
- **Away wins** have the highest recall (0.59), meaning the model catches most upsets when they happen.
- An accuracy of ~47% for 3-way football prediction is competitive. For reference, bookmaker-implied models typically achieve 50–55% on major European leagues with far richer data (lineups, injuries, market odds).

### Top 15 Most Important Features

| Rank | Feature | Importance |
|---|---|---|
| 1 | `elo_diff` | 19.98 |
| 2 | `home_elo` | 8.91 |
| 3 | `away_elo` | 6.64 |
| 4 | `goal_diff_diff` | 4.94 |
| 5 | `away_avg_goals_conceded` | 4.22 |
| 6 | `home_last5_losses` | 4.17 |
| 7 | `home_avg_goals_conceded` | 3.64 |
| 8 | `away_form_points` | 3.14 |
| 9 | `home_home_avg_goals` | 2.38 |
| 10 | `away_rank` | 2.34 |
| 11 | `home_loss_streak` | 2.33 |
| 12 | `home_rank` | 2.32 |
| 13 | `home_unbeaten_streak` | 2.21 |
| 14 | `home_home_win_rate` | 2.10 |
| 15 | `home_avg_goals` | 2.07 |

The **Elo difference** between teams alone accounts for ~20% of the model's predictive power, confirming that team strength relative to the opponent is the single most important factor.

## Dataset

| Property | Value |
|---|---|
| Source | [soccerway.com](https://ve.soccerway.com) |
| Matches | 6,592 |
| Seasons | 25 (2002-2003 to 2026) |
| Teams | 42 |
| Outcome distribution | H: 45.4%, D: 29.6%, A: 25.0% |

Raw data is stored in `data/futve_consolidate_results.csv` with columns: `season`, `competition`, `phase`, `round`, `match_id`, `match_date_utc`, `match_date_local`, `home_team`, `away_team`, `home_score`, `away_score`, `result`, `source_url`.

## Engineered Features (46 total)

All features are calculated using **only information available before the match** to avoid data leakage.

| Category | Features |
|---|---|
| **Last 5 matches** | `home/away_last5_wins`, `_draws`, `_losses` |
| **Average goals (all-time)** | `home/away_avg_goals`, `_avg_goals_conceded` |
| **Average goals (last 5)** | `home/away_avg_goals_last5`, `_avg_conceded_last5` |
| **Form** | `home/away_form_points`, `ppg_diff` |
| **Elo rating** | `home/away_elo`, `elo_diff` |
| **Ranking** | `home/away_rank` |
| **Streaks** | `home/away_win_streak`, `_draw_streak`, `_loss_streak`, `_unbeaten_streak` |
| **Head-to-head** | `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`, `h2h_total` |
| **Home advantage** | `home_home_win_rate`, `home_home_avg_goals`, `away_away_win_rate`, `away_away_avg_goals` |
| **Rest days** | `home/away_rest_days`, `rest_days_diff` |
| **Context** | `home/away_matches_played`, `home/away_goal_diff`, `goal_diff_diff` |

## Project Structure

```
ML-FutVE-Predictions/
├── app/
│   └── api.py                  # FastAPI prediction API
├── data/
│   ├── futve_consolidate_results.csv  # Raw match data
│   └── futve_features.csv            # Engineered features dataset
├── features_extractions/
│   ├── extract_features.ipynb  # Feature engineering notebook
│   └── README.md               # Feature documentation
├── models/
│   ├── catboost_model.cbm      # Trained CatBoost model
│   └── model_metadata.json     # Model config and results
├── src/
│   ├── constants.py            # Paths and hyperparameters
│   ├── matchengine.py          # MatchEngine (state + feature computation)
│   └── schemas.py              # Pydantic request/response models
├── training/
│   └── train_model.ipynb       # Model training notebook
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/CarlosASM2001/ML-FutVE-Predictions.git
cd ML-FutVE-Predictions
pip install pandas numpy scikit-learn catboost fastapi uvicorn
```

## Usage

### Running the API

```bash
cd app
set PYTHONPATH=..\src        # Windows
# export PYTHONPATH=../src   # Linux/Mac
uvicorn api:app --host 0.0.0.0 --port 8000
```

The interactive API documentation is available at `http://localhost:8000/docs`.

### API Endpoints

#### `POST /predict` — Predict a match outcome

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Caracas", "away_team": "Dep. Táchira"}'
```

Response:

```json
{
  "home_team": "Caracas",
  "away_team": "Dep. Táchira",
  "prediction": "Draw",
  "probabilities": {
    "Home win": 0.337,
    "Draw": 0.3591,
    "Away win": 0.3039
  },
  "features": { ... }
}
```

#### `GET /teams` — List all teams

```bash
curl http://localhost:8000/teams
```

#### `GET /teams/{team_name}` — Team stats

```bash
curl http://localhost:8000/teams/Caracas
```

Response:

```json
{
  "team": "Caracas",
  "elo": 1605.65,
  "matches_played": 859,
  "form_last5": "DLLDL",
  "form_points_last5": 2,
  "win_streak": 0,
  "unbeaten_streak": 0,
  "loss_streak": 1,
  "avg_goals": 1.52,
  "avg_goals_conceded": 0.92,
  "goal_diff": 517
}
```

#### `GET /ranking` — Full Elo ranking

```bash
curl http://localhost:8000/ranking
```

#### `GET /health` — Server status

```bash
curl http://localhost:8000/health
```

## Tech Stack

- **Python 3.12+**
- **CatBoost** — Gradient boosting classifier (main model)
- **scikit-learn** — Baseline models and evaluation metrics
- **pandas / NumPy** — Data processing
- **FastAPI** — REST API framework
- **Elo rating system** — Custom implementation (K=32, home advantage=100)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
