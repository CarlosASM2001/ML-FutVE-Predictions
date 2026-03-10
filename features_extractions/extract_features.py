"""
Feature extraction pipeline for FUTVE match outcome prediction.

Reads the consolidated historical match dataset, sorts matches chronologically,
and generates pre-match features using only past information (no data leakage).

All features are computed BEFORE the match they belong to, using only data
available at that point in time.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

INITIAL_ELO = 1500
ELO_K = 32
ELO_HOME_ADVANTAGE = 100
FORM_WINDOW = 5

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "futve_consolidate_results.csv"
OUTPUT_CSV = DATA_DIR / "futve_features.csv"


def _expected_elo_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _count_streak(history: list[dict], target_result: str) -> int:
    """Count consecutive matches with `target_result` from the most recent backwards."""
    streak = 0
    for match in reversed(history):
        if match["result"] == target_result:
            streak += 1
        else:
            break
    return streak


def _unbeaten_streak(history: list[dict]) -> int:
    """Count consecutive matches without a loss from the most recent backwards."""
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


def load_data(csv_path: Path = INPUT_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["match_date_utc"] = pd.to_datetime(df["match_date_utc"], utc=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values(["match_date_utc", "match_id"]).reset_index(drop=True)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    elo = defaultdict(lambda: INITIAL_ELO)
    team_history = defaultdict(list)
    h2h_records = defaultdict(list)
    team_last_date = {}

    # Separate home/away tracking for home-advantage indicators
    team_home_stats = defaultdict(lambda: {"played": 0, "wins": 0, "goals": 0})
    team_away_stats = defaultdict(lambda: {"played": 0, "wins": 0, "goals": 0})

    rows = []

    for _, match in df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        match_dt = match["match_date_utc"]

        h_hist = team_history[home]
        a_hist = team_history[away]
        h_last5 = h_hist[-FORM_WINDOW:]
        a_last5 = a_hist[-FORM_WINDOW:]

        # ---- Last 5 matches stats ----
        home_last5_wins = sum(1 for m in h_last5 if m["result"] == "W")
        home_last5_draws = sum(1 for m in h_last5 if m["result"] == "D")
        home_last5_losses = sum(1 for m in h_last5 if m["result"] == "L")
        away_last5_wins = sum(1 for m in a_last5 if m["result"] == "W")
        away_last5_draws = sum(1 for m in a_last5 if m["result"] == "D")
        away_last5_losses = sum(1 for m in a_last5 if m["result"] == "L")

        # ---- Average goals (all history) ----
        home_avg_goals = _avg_or_zero([m["gf"] for m in h_hist])
        home_avg_goals_conceded = _avg_or_zero([m["ga"] for m in h_hist])
        away_avg_goals = _avg_or_zero([m["gf"] for m in a_hist])
        away_avg_goals_conceded = _avg_or_zero([m["ga"] for m in a_hist])

        # ---- Average goals last 5 ----
        home_avg_goals_last5 = _avg_or_zero([m["gf"] for m in h_last5])
        home_avg_conceded_last5 = _avg_or_zero([m["ga"] for m in h_last5])
        away_avg_goals_last5 = _avg_or_zero([m["gf"] for m in a_last5])
        away_avg_conceded_last5 = _avg_or_zero([m["ga"] for m in a_last5])

        # ---- Form points ----
        home_form_points = _form_points(h_last5)
        away_form_points = _form_points(a_last5)

        # ---- Elo ratings (pre-match) ----
        home_elo = elo[home]
        away_elo = elo[away]

        # ---- Ranking (by Elo among all known teams) ----
        all_elos = dict(elo)
        if home not in all_elos:
            all_elos[home] = INITIAL_ELO
        if away not in all_elos:
            all_elos[away] = INITIAL_ELO
        sorted_teams = sorted(all_elos.items(), key=lambda x: -x[1])
        rank_map = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}
        home_rank = rank_map[home]
        away_rank = rank_map[away]

        # ---- Streaks ----
        home_win_streak = _count_streak(h_hist, "W")
        home_draw_streak = _count_streak(h_hist, "D")
        home_loss_streak = _count_streak(h_hist, "L")
        home_unbeaten = _unbeaten_streak(h_hist)
        away_win_streak = _count_streak(a_hist, "W")
        away_draw_streak = _count_streak(a_hist, "D")
        away_loss_streak = _count_streak(a_hist, "L")
        away_unbeaten = _unbeaten_streak(a_hist)

        # ---- Head-to-head ----
        h2h_key = tuple(sorted([home, away]))
        h2h_past = h2h_records[h2h_key]
        h2h_home_wins = sum(1 for m in h2h_past if m["winner"] == home)
        h2h_away_wins = sum(1 for m in h2h_past if m["winner"] == away)
        h2h_draws = sum(1 for m in h2h_past if m["winner"] is None)
        h2h_total = len(h2h_past)

        # ---- Home-advantage indicators ----
        hs = team_home_stats[home]
        home_home_win_rate = _safe_ratio(hs["wins"], hs["played"])
        home_home_avg_goals = _safe_ratio(hs["goals"], hs["played"])

        aws = team_away_stats[away]
        away_away_win_rate = _safe_ratio(aws["wins"], aws["played"])
        away_away_avg_goals = _safe_ratio(aws["goals"], aws["played"])

        # ---- Rest days ----
        home_rest_days = (match_dt - team_last_date[home]).days if home in team_last_date else -1
        away_rest_days = (match_dt - team_last_date[away]).days if away in team_last_date else -1

        # ---- Matches played ----
        home_matches_played = len(h_hist)
        away_matches_played = len(a_hist)

        # ---- Goal difference (all-time) ----
        home_goal_diff = sum(m["gf"] - m["ga"] for m in h_hist) if h_hist else 0
        away_goal_diff = sum(m["gf"] - m["ga"] for m in a_hist) if a_hist else 0

        rows.append(
            {
                "match_id": match["match_id"],
                "season": match["season"],
                "match_date_utc": match["match_date_utc"],
                "home_team": home,
                "away_team": away,
                # Last 5
                "home_last5_wins": home_last5_wins,
                "home_last5_draws": home_last5_draws,
                "home_last5_losses": home_last5_losses,
                "away_last5_wins": away_last5_wins,
                "away_last5_draws": away_last5_draws,
                "away_last5_losses": away_last5_losses,
                # Avg goals (all-time)
                "home_avg_goals": round(home_avg_goals, 4),
                "home_avg_goals_conceded": round(home_avg_goals_conceded, 4),
                "away_avg_goals": round(away_avg_goals, 4),
                "away_avg_goals_conceded": round(away_avg_goals_conceded, 4),
                # Avg goals (last 5)
                "home_avg_goals_last5": round(home_avg_goals_last5, 4),
                "home_avg_conceded_last5": round(home_avg_conceded_last5, 4),
                "away_avg_goals_last5": round(away_avg_goals_last5, 4),
                "away_avg_conceded_last5": round(away_avg_conceded_last5, 4),
                # Form
                "home_form_points": home_form_points,
                "away_form_points": away_form_points,
                # Elo
                "home_elo": round(home_elo, 2),
                "away_elo": round(away_elo, 2),
                # Ranking
                "home_rank": home_rank,
                "away_rank": away_rank,
                # Streaks
                "home_win_streak": home_win_streak,
                "home_draw_streak": home_draw_streak,
                "home_loss_streak": home_loss_streak,
                "home_unbeaten_streak": home_unbeaten,
                "away_win_streak": away_win_streak,
                "away_draw_streak": away_draw_streak,
                "away_loss_streak": away_loss_streak,
                "away_unbeaten_streak": away_unbeaten,
                # H2H
                "h2h_home_wins": h2h_home_wins,
                "h2h_away_wins": h2h_away_wins,
                "h2h_draws": h2h_draws,
                "h2h_total": h2h_total,
                # Home advantage
                "home_home_win_rate": round(home_home_win_rate, 4),
                "home_home_avg_goals": round(home_home_avg_goals, 4),
                "away_away_win_rate": round(away_away_win_rate, 4),
                "away_away_avg_goals": round(away_away_avg_goals, 4),
                # Rest days
                "home_rest_days": home_rest_days,
                "away_rest_days": away_rest_days,
                # Context
                "home_matches_played": home_matches_played,
                "away_matches_played": away_matches_played,
                "home_goal_diff": home_goal_diff,
                "away_goal_diff": away_goal_diff,
                # Target
                "result": match["result"],
            }
        )

        # ---- Post-match updates (after feature extraction) ----
        if match["result"] == "H":
            h_res, a_res = "W", "L"
            actual_h, actual_a = 1.0, 0.0
            winner = home
        elif match["result"] == "A":
            h_res, a_res = "L", "W"
            actual_h, actual_a = 0.0, 1.0
            winner = away
        else:
            h_res, a_res = "D", "D"
            actual_h, actual_a = 0.5, 0.5
            winner = None

        team_history[home].append(
            {"result": h_res, "gf": match["home_score"], "ga": match["away_score"]}
        )
        team_history[away].append(
            {"result": a_res, "gf": match["away_score"], "ga": match["home_score"]}
        )

        exp_h = _expected_elo_score(home_elo + ELO_HOME_ADVANTAGE, away_elo)
        elo[home] += ELO_K * (actual_h - exp_h)
        elo[away] += ELO_K * (actual_a - (1.0 - exp_h))

        h2h_records[h2h_key].append({"winner": winner})

        team_home_stats[home]["played"] += 1
        team_home_stats[home]["goals"] += match["home_score"]
        if match["result"] == "H":
            team_home_stats[home]["wins"] += 1

        team_away_stats[away]["played"] += 1
        team_away_stats[away]["goals"] += match["away_score"]
        if match["result"] == "A":
            team_away_stats[away]["wins"] += 1

        team_last_date[home] = match_dt
        team_last_date[away] = match_dt

    return pd.DataFrame(rows)


def main():
    print(f"Loading data from {INPUT_CSV}")
    df = load_data()
    print(f"  -> {len(df)} matches loaded ({df['season'].nunique()} seasons, {df['home_team'].nunique()} teams)")

    print("Extracting features...")
    features_df = extract_features(df)

    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  -> {len(features_df)} rows, {len(features_df.columns)} columns")
    print(f"  -> Saved to {OUTPUT_CSV}")

    print("\nFeature columns:")
    for col in features_df.columns:
        print(f"  {col}")

    print(f"\nTarget distribution:")
    print(features_df["result"].value_counts().to_string())

    print("\nSample (first 3 rows):")
    print(features_df.head(3).T.to_string())


if __name__ == "__main__":
    main()
