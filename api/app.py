"""
FastAPI application for FUTVE match outcome prediction.

Self-contained module: includes the MatchEngine (feature computation and
state management), data loading, and all API endpoints.

On startup, replays all historical matches to build the current engine state,
then loads the trained CatBoost model. Incoming requests compute features in
real time from the engine state and return predictions with probabilities.

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---- Paths ----

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "futve_consolidate_results.csv"
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "training"
    / "saved_models"
    / "catboost_model.cbm"
)

# ---- Constants (must match extract_features.py) ----

INITIAL_ELO = 1500
ELO_K = 32
ELO_HOME_ADVANTAGE = 100
FORM_WINDOW = 5

FEATURE_COLUMNS = [
    "home_last5_wins",
    "home_last5_draws",
    "home_last5_losses",
    "away_last5_wins",
    "away_last5_draws",
    "away_last5_losses",
    "home_avg_goals",
    "home_avg_goals_conceded",
    "away_avg_goals",
    "away_avg_goals_conceded",
    "home_avg_goals_last5",
    "home_avg_conceded_last5",
    "away_avg_goals_last5",
    "away_avg_conceded_last5",
    "home_form_points",
    "away_form_points",
    "home_elo",
    "away_elo",
    "home_rank",
    "away_rank",
    "home_win_streak",
    "home_draw_streak",
    "home_loss_streak",
    "home_unbeaten_streak",
    "away_win_streak",
    "away_draw_streak",
    "away_loss_streak",
    "away_unbeaten_streak",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "h2h_total",
    "home_home_win_rate",
    "home_home_avg_goals",
    "away_away_win_rate",
    "away_away_avg_goals",
    "home_rest_days",
    "away_rest_days",
    "home_matches_played",
    "away_matches_played",
    "home_goal_diff",
    "away_goal_diff",
]

CATBOOST_LABELS = ["A", "D", "H"]


# ---- Helper functions ----


def _expected_elo_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _count_streak(history: list[dict], target_result: str) -> int:
    streak = 0
    for match in reversed(history):
        if match["result"] == target_result:
            streak += 1
        else:
            break
    return streak


def _unbeaten_streak(history: list[dict]) -> int:
    streak = 0
    for match in reversed(history):
        if match["result"] != "L":
            streak += 1
        else:
            break
    return streak


def _form_points(matches: list[dict]) -> int:
    return sum(
        3 if m["result"] == "W" else 1 if m["result"] == "D" else 0
        for m in matches
    )


def _avg_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


# ---- MatchEngine ----


class MatchEngine:
    """
    Maintains the full state of team histories, Elo ratings, H2H records, etc.

    Process historical matches with ``process_match()`` to build up the state,
    then call ``compute_features()`` to get pre-match features for any pair of
    teams without modifying the state.
    """

    def __init__(self):
        self.elo: dict[str, float] = defaultdict(lambda: INITIAL_ELO)
        self.team_history: dict[str, list[dict]] = defaultdict(list)
        self.h2h_records: dict[tuple, list[dict]] = defaultdict(list)
        self.team_last_date: dict[str, datetime] = {}
        self.team_home_stats: dict[str, dict] = defaultdict(
            lambda: {"played": 0, "wins": 0, "goals": 0}
        )
        self.team_away_stats: dict[str, dict] = defaultdict(
            lambda: {"played": 0, "wins": 0, "goals": 0}
        )
        self.matches_processed = 0

    @property
    def known_teams(self) -> set[str]:
        return set(self.team_history.keys())

    def compute_features(
        self, home: str, away: str, match_dt: datetime | None = None
    ) -> dict:
        """
        Compute pre-match features for a given home/away pair using only
        the current state.  Does NOT modify the engine state.
        """
        if match_dt is None:
            match_dt = datetime.now(timezone.utc)

        h_hist = self.team_history[home]
        a_hist = self.team_history[away]
        h_last5 = h_hist[-FORM_WINDOW:]
        a_last5 = a_hist[-FORM_WINDOW:]

        home_last5_wins = sum(1 for m in h_last5 if m["result"] == "W")
        home_last5_draws = sum(1 for m in h_last5 if m["result"] == "D")
        home_last5_losses = sum(1 for m in h_last5 if m["result"] == "L")
        away_last5_wins = sum(1 for m in a_last5 if m["result"] == "W")
        away_last5_draws = sum(1 for m in a_last5 if m["result"] == "D")
        away_last5_losses = sum(1 for m in a_last5 if m["result"] == "L")

        home_avg_goals = _avg_or_zero([m["gf"] for m in h_hist])
        home_avg_goals_conceded = _avg_or_zero([m["ga"] for m in h_hist])
        away_avg_goals = _avg_or_zero([m["gf"] for m in a_hist])
        away_avg_goals_conceded = _avg_or_zero([m["ga"] for m in a_hist])

        home_avg_goals_last5 = _avg_or_zero([m["gf"] for m in h_last5])
        home_avg_conceded_last5 = _avg_or_zero([m["ga"] for m in h_last5])
        away_avg_goals_last5 = _avg_or_zero([m["gf"] for m in a_last5])
        away_avg_conceded_last5 = _avg_or_zero([m["ga"] for m in a_last5])

        home_form_points = _form_points(h_last5)
        away_form_points = _form_points(a_last5)

        home_elo = self.elo[home]
        away_elo = self.elo[away]

        all_elos = dict(self.elo)
        if home not in all_elos:
            all_elos[home] = INITIAL_ELO
        if away not in all_elos:
            all_elos[away] = INITIAL_ELO
        sorted_teams = sorted(all_elos.items(), key=lambda x: -x[1])
        rank_map = {
            team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)
        }
        home_rank = rank_map[home]
        away_rank = rank_map[away]

        home_win_streak = _count_streak(h_hist, "W")
        home_draw_streak = _count_streak(h_hist, "D")
        home_loss_streak = _count_streak(h_hist, "L")
        home_unbeaten = _unbeaten_streak(h_hist)
        away_win_streak = _count_streak(a_hist, "W")
        away_draw_streak = _count_streak(a_hist, "D")
        away_loss_streak = _count_streak(a_hist, "L")
        away_unbeaten = _unbeaten_streak(a_hist)

        h2h_key = tuple(sorted([home, away]))
        h2h_past = self.h2h_records[h2h_key]
        h2h_home_wins = sum(1 for m in h2h_past if m["winner"] == home)
        h2h_away_wins = sum(1 for m in h2h_past if m["winner"] == away)
        h2h_draws = sum(1 for m in h2h_past if m["winner"] is None)
        h2h_total = len(h2h_past)

        hs = self.team_home_stats[home]
        home_home_win_rate = _safe_ratio(hs["wins"], hs["played"])
        home_home_avg_goals = _safe_ratio(hs["goals"], hs["played"])

        aws = self.team_away_stats[away]
        away_away_win_rate = _safe_ratio(aws["wins"], aws["played"])
        away_away_avg_goals = _safe_ratio(aws["goals"], aws["played"])

        home_rest_days = (
            (match_dt - self.team_last_date[home]).days
            if home in self.team_last_date
            else -1
        )
        away_rest_days = (
            (match_dt - self.team_last_date[away]).days
            if away in self.team_last_date
            else -1
        )

        home_matches_played = len(h_hist)
        away_matches_played = len(a_hist)

        home_goal_diff = sum(m["gf"] - m["ga"] for m in h_hist) if h_hist else 0
        away_goal_diff = sum(m["gf"] - m["ga"] for m in a_hist) if a_hist else 0

        return {
            "home_last5_wins": home_last5_wins,
            "home_last5_draws": home_last5_draws,
            "home_last5_losses": home_last5_losses,
            "away_last5_wins": away_last5_wins,
            "away_last5_draws": away_last5_draws,
            "away_last5_losses": away_last5_losses,
            "home_avg_goals": round(home_avg_goals, 4),
            "home_avg_goals_conceded": round(home_avg_goals_conceded, 4),
            "away_avg_goals": round(away_avg_goals, 4),
            "away_avg_goals_conceded": round(away_avg_goals_conceded, 4),
            "home_avg_goals_last5": round(home_avg_goals_last5, 4),
            "home_avg_conceded_last5": round(home_avg_conceded_last5, 4),
            "away_avg_goals_last5": round(away_avg_goals_last5, 4),
            "away_avg_conceded_last5": round(away_avg_conceded_last5, 4),
            "home_form_points": home_form_points,
            "away_form_points": away_form_points,
            "home_elo": round(home_elo, 2),
            "away_elo": round(away_elo, 2),
            "home_rank": home_rank,
            "away_rank": away_rank,
            "home_win_streak": home_win_streak,
            "home_draw_streak": home_draw_streak,
            "home_loss_streak": home_loss_streak,
            "home_unbeaten_streak": home_unbeaten,
            "away_win_streak": away_win_streak,
            "away_draw_streak": away_draw_streak,
            "away_loss_streak": away_loss_streak,
            "away_unbeaten_streak": away_unbeaten,
            "h2h_home_wins": h2h_home_wins,
            "h2h_away_wins": h2h_away_wins,
            "h2h_draws": h2h_draws,
            "h2h_total": h2h_total,
            "home_home_win_rate": round(home_home_win_rate, 4),
            "home_home_avg_goals": round(home_home_avg_goals, 4),
            "away_away_win_rate": round(away_away_win_rate, 4),
            "away_away_avg_goals": round(away_away_avg_goals, 4),
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "home_matches_played": home_matches_played,
            "away_matches_played": away_matches_played,
            "home_goal_diff": home_goal_diff,
            "away_goal_diff": away_goal_diff,
        }

    def process_match(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        result: str,
        match_dt: datetime,
    ):
        """
        Update the engine state after a match has been played.
        Call this in chronological order for each historical match.
        """
        home_elo = self.elo[home]
        away_elo = self.elo[away]

        if result == "H":
            h_res, a_res = "W", "L"
            actual_h, actual_a = 1.0, 0.0
            winner = home
        elif result == "A":
            h_res, a_res = "L", "W"
            actual_h, actual_a = 0.0, 1.0
            winner = away
        else:
            h_res, a_res = "D", "D"
            actual_h, actual_a = 0.5, 0.5
            winner = None

        self.team_history[home].append(
            {"result": h_res, "gf": home_score, "ga": away_score}
        )
        self.team_history[away].append(
            {"result": a_res, "gf": away_score, "ga": home_score}
        )

        exp_h = _expected_elo_score(home_elo + ELO_HOME_ADVANTAGE, away_elo)
        self.elo[home] += ELO_K * (actual_h - exp_h)
        self.elo[away] += ELO_K * (actual_a - (1.0 - exp_h))

        h2h_key = tuple(sorted([home, away]))
        self.h2h_records[h2h_key].append({"winner": winner})

        self.team_home_stats[home]["played"] += 1
        self.team_home_stats[home]["goals"] += home_score
        if result == "H":
            self.team_home_stats[home]["wins"] += 1

        self.team_away_stats[away]["played"] += 1
        self.team_away_stats[away]["goals"] += away_score
        if result == "A":
            self.team_away_stats[away]["wins"] += 1

        self.team_last_date[home] = match_dt
        self.team_last_date[away] = match_dt
        self.matches_processed += 1

    def get_team_info(self, team: str) -> dict | None:
        """Return a summary of a team's current state."""
        if team not in self.known_teams:
            return None

        hist = self.team_history[team]
        last5 = hist[-FORM_WINDOW:]

        return {
            "team": team,
            "elo": round(self.elo[team], 2),
            "matches_played": len(hist),
            "form_last5": "".join(m["result"][0] for m in last5),
            "form_points_last5": _form_points(last5),
            "win_streak": _count_streak(hist, "W"),
            "unbeaten_streak": _unbeaten_streak(hist),
            "loss_streak": _count_streak(hist, "L"),
            "avg_goals": round(_avg_or_zero([m["gf"] for m in hist]), 2),
            "avg_goals_conceded": round(
                _avg_or_zero([m["ga"] for m in hist]), 2
            ),
            "goal_diff": sum(m["gf"] - m["ga"] for m in hist),
            "last_match": self.team_last_date.get(team),
        }


# ---- Data loading ----


def load_data(csv_path: Path = INPUT_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["match_date_utc"] = pd.to_datetime(df["match_date_utc"], utc=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values(["match_date_utc", "match_id"]).reset_index(drop=True)
    return df


def build_engine(df: pd.DataFrame) -> MatchEngine:
    """Replay all historical matches to build the engine state."""
    engine = MatchEngine()
    for _, row in df.iterrows():
        engine.process_match(
            home=row["home_team"],
            away=row["away_team"],
            home_score=row["home_score"],
            away_score=row["away_score"],
            result=row["result"],
            match_dt=row["match_date_utc"],
        )
    return engine


# ---- FastAPI application ----

engine: MatchEngine | None = None
model: CatBoostClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, model

    print("Loading historical data and building engine state...")
    df = load_data(INPUT_CSV)
    engine = build_engine(df)
    print(
        f"  -> {engine.matches_processed} matches processed, "
        f"{len(engine.known_teams)} teams"
    )

    print(f"Loading CatBoost model from {MODEL_PATH}")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    print("  -> Model loaded")

    yield


app = FastAPI(
    title="FUTVE Match Predictor",
    description=(
        "Predicts Liga FUTVE match outcomes (Home win / Draw / Away win) "
        "using historical features and CatBoost."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---- Pydantic models ----


class PredictRequest(BaseModel):
    home_team: str
    away_team: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"home_team": "Caracas", "away_team": "Dep. Táchira"}]
        }
    }


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
        [[features[col] for col in FEATURE_COLUMNS]], dtype=np.float64
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
            for label, prob in zip(CATBOOST_LABELS, probas)
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
