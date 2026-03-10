"""
FastAPI application for FUTVE match outcome prediction.

On startup, replays all historical matches to build the current engine state,
then loads the trained CatBoost model. Incoming requests compute features in
real time from the engine state and return predictions with probabilities.

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features_extractions.extract_features import (
    FEATURE_COLUMNS,
    MatchEngine,
    build_engine,
    load_data,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "futve_consolidate_results.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "training" / "saved_models" / "catboost_model.cbm"

LABELS = ["A", "D", "H"]

engine: MatchEngine | None = None
model: CatBoostClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, model

    print("Loading historical data and building engine state...")
    df = load_data(INPUT_CSV)
    engine = build_engine(df)
    print(f"  -> {engine.matches_processed} matches processed, {len(engine.known_teams)} teams")

    print(f"Loading CatBoost model from {MODEL_PATH}")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    print("  -> Model loaded")

    yield


app = FastAPI(
    title="FUTVE Match Predictor",
    description="Predicts Liga FUTVE match outcomes (Home win / Draw / Away win) using historical features and CatBoost.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---- Request / Response models ----

class PredictRequest(BaseModel):
    home_team: str
    away_team: str

    model_config = {"json_schema_extra": {"examples": [{"home_team": "Caracas", "away_team": "Dep. Táchira"}]}}


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    prediction: str
    probabilities: dict[str, float]
    features: dict[str, float]


class TeamInfoResponse(BaseModel):
    team: str
    elo: float
    matches_played: int
    form_last5: str
    form_points_last5: int
    win_streak: int
    unbeaten_streak: int
    loss_streak: int
    avg_goals: float
    avg_goals_conceded: float
    goal_diff: int
    last_match: str | None


# ---- Endpoints ----

@app.get("/health")
def health():
    return {
        "status": "ok",
        "matches_processed": engine.matches_processed if engine else 0,
        "teams_loaded": len(engine.known_teams) if engine else 0,
    }


@app.get("/teams", response_model=list[str])
def list_teams():
    """Return all teams known to the engine, sorted alphabetically."""
    return sorted(engine.known_teams)


@app.get("/teams/{team_name}", response_model=TeamInfoResponse)
def get_team(team_name: str):
    """Return current stats for a specific team."""
    info = engine.get_team_info(team_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")
    if info["last_match"] is not None:
        info["last_match"] = info["last_match"].isoformat()
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict the outcome of a match between two teams."""
    if req.home_team not in engine.known_teams:
        raise HTTPException(status_code=404, detail=f"Team '{req.home_team}' not found")
    if req.away_team not in engine.known_teams:
        raise HTTPException(status_code=404, detail=f"Team '{req.away_team}' not found")
    if req.home_team == req.away_team:
        raise HTTPException(status_code=400, detail="Home and away teams must be different")

    now = datetime.now(timezone.utc)
    features = engine.compute_features(req.home_team, req.away_team, now)

    feature_vector = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float64)

    probas = model.predict_proba(feature_vector)[0]
    predicted_class = str(model.predict(feature_vector).flatten()[0])

    label_map = {"H": "Home win", "D": "Draw", "A": "Away win"}

    return PredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        prediction=label_map.get(predicted_class, predicted_class),
        probabilities={
            label_map[label]: round(float(prob), 4)
            for label, prob in zip(LABELS, probas)
        },
        features={k: float(v) for k, v in features.items()},
    )


@app.get("/ranking", response_model=list[dict])
def get_ranking():
    """Return all teams ranked by Elo rating."""
    teams = []
    for team in engine.known_teams:
        info = engine.get_team_info(team)
        teams.append({
            "rank": 0,
            "team": team,
            "elo": info["elo"],
            "matches_played": info["matches_played"],
            "form": info["form_last5"],
            "goal_diff": info["goal_diff"],
        })
    teams.sort(key=lambda x: -x["elo"])
    for i, t in enumerate(teams):
        t["rank"] = i + 1
    return teams
