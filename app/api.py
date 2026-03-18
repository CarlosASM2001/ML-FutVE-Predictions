from contextlib import asynccontextmanager
from pathlib import Path
from constants import Paths as paths
from constants import Constantes as const
import numpy as np
import pandas as pd
from matchengine import MatchEngine
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from schemas import PredictRequest, TeamInfoResponse, PredictResponse
from datetime import datetime, timezone


# --- Data Loading ---

def load_data(csv_path: Path = paths.INPUT_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["match_date_utc"] = pd.to_datetime(df["match_date_utc"], utc=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values(["match_date_utc", "match_id"]).reset_index(drop=True)
    return df


def build_engine(df: pd.DataFrame) -> MatchEngine:
    engine = MatchEngine()
    for _, row in df.iterrows():
        engine.process_match(
            home=row["home_team"],
            away=row["away_team"],
            home_score=row["home_score"],
            away_score=row["away_score"],
            result=row["result"],
            matchdate=row["match_date_utc"],
        )
    return engine


# --- FASTapi aplication ---

engine: MatchEngine | None = None
model: CatBoostClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global engine, model

    print("Loading historical data and building engine state...")
    df = load_data(paths.INPUT_CSV)
    engine = build_engine(df)

    print(
        f"  -> {engine.matches_processed} matches processed, "
        f"{len(engine.known_teams)} teams"
    )
    print("Loading Catboost model")

    model = CatBoostClassifier()
    model.load_model(str(paths.MODEL_PATH))

    print("  -> Model loaded")

    yield


app = FastAPI(
    title= "FUTVE Match Predictor",
    description=(
        "Predicts Liga FUTVE match outcomes (Home win / Draw / Away win) "
        "using historical features and CatBoost."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- EndPoints ---


@app.get("/health")
def health():
    return {
        "status" : "ok",
        "matches_processed": engine.matches_processed if engine else 0,
        "teams_loaded": len(engine.known_teams) if engine else 0
    }

@app.get("/teams" , response_model=list[str])
def list_teams():
    """Return al the teams"""

    return sorted(engine.known_teams)


@app.get("/teams/{team_name}", response_model=TeamInfoResponse)
def get_team(team_name:str):
    """return stats for a specific team"""

    info = engine.get_team_info(team_name)

    if info is None:
        raise HTTPException(
            status_code=404, detail=f"Team '{team_name}' not found"
        )
    if info["last_match"] is not None:
        info["last_match"] = info["last_match"].isoformat()
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict the outcome of a match between two teams."""
    if req.home_team not in engine.known_teams:
        raise HTTPException(
            status_code=404, detail=f"Team '{req.home_team}' not found"
        )
    if req.away_team not in engine.known_teams:
        raise HTTPException(
            status_code=404, detail=f"Team '{req.away_team}' not found"
        )
    if req.home_team == req.away_team:
        raise HTTPException(
            status_code=400, detail="Home and away teams must be different"
        )

    now = datetime.now(timezone.utc)
    features = engine.compute_features(req.home_team, req.away_team, now)

    feature_vector = np.array(
        [[features[col] for col in const.FEATURE_COLUMNS]], dtype=np.float64
    )

    probas = model.predict_proba(feature_vector)[0]
    predicted_class = str(model.predict(feature_vector).flatten()[0])

    label_map = {"H": "Home win", "D": "Draw", "A": "Away win"}

    return PredictResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        prediction=label_map.get(predicted_class, predicted_class),
        probabilities={
            label_map[label]: round(float(prob), 4)
            for label, prob in zip(const.CATBOOST_LABELS, probas)
        },
        features={k: float(v) for k, v in features.items()},
    )
    
@app.get("/ranking", response_model=list[dict])
def get_ranking():
    """Return all teams ranked by Elo rating."""
    teams = []
    for team in engine.known_teams:
        info = engine.get_team_info(team)
        teams.append(
            {
                "rank": 0,
                "team": team,
                "elo": info["elo"],
                "matches_played": info["matches_played"],
                "form": info["form_last5"],
                "goal_diff": info["goal_diff"],
            }
        )
    teams.sort(key=lambda x: -x["elo"])
    for i, t in enumerate(teams):
        t["rank"] = i + 1
    return teams

