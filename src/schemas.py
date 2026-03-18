from pydantic import BaseModel


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
