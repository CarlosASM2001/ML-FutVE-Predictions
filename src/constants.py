from pathlib import Path


class Paths:

    DATA_DIR = Path(__file__).resolve().parent.parent
    INPUT_CSV = DATA_DIR / "data" / "futve_consolidate_results.csv"
    FEATURES_CSV = DATA_DIR / "data" / "futve_features.csv"
    MODEL_PATH = DATA_DIR / "models" / "catboost_model.cbm"


class Constantes:


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
        "ppg_diff",
        "home_elo",
        "away_elo",
        "elo_diff",
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
        "rest_days_diff",
        "home_matches_played",
        "away_matches_played",
        "home_goal_diff",
        "away_goal_diff",
        "goal_diff_diff",
    ]

    CATBOOST_LABELS = ["H", "D", "A"]