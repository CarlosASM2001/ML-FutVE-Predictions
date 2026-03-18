
from constants import Constantes as const
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np

# -- Helper Functions -- 


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



class MatchEngine:


    def __init__(self):
        self.elo : dict[str,float] = defaultdict(lambda: const.INITIAL_ELO)
        self.team_history: dict[str,list[dict]] = defaultdict(list)
        self.h2h_records: dict[tuple,list[dict]] = defaultdict(list)
        self.team_last_date: dict[str,datetime] = {}
        self.team_home_stats: dict[str,dict] = defaultdict(
            lambda: {"played":0,"wins":0,"goals":0}
        )
        self.team_away_stats: dict[str, dict] = defaultdict(
            lambda: {"played": 0, "wins": 0, "goals": 0}
        )
        self.matches_processed: 0


    @property
    def known_teams(self) -> set[str]:
        return set(self.team_history.keys())
    

    def compute_features(self, home:str, away:str, match_dt:datetime | None=None) -> dict:
        if match_dt is None:
            match_dt = datetime.now(timezone.utc)

        h_hist = self.team_history[home]
        a_hist = self.team_history[away]
        h_last5 = h_hist[-const.FORM_WINDOW:]
        a_last5 = a_hist[-const.FORM_WINDOW:]

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
            all_elos[home] = const.INITIAL_ELO
        if away not in all_elos:
            all_elos[away] = const.INITIAL_ELO
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
        home_ppg= _safe_ratio(home_form_points, home_matches_played)
        away_ppg= _safe_ratio(away_form_points, away_matches_played)

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
            "ppg_diff": home_ppg - away_ppg,
            "home_elo": round(home_elo, 2),
            "away_elo": round(away_elo, 2),
            "elo_diff": round((home_elo-away_elo),2),
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
            "rest_days_diff": home_rest_days - away_rest_days,
            "home_matches_played": home_matches_played,
            "away_matches_played": away_matches_played,
            "home_goal_diff": home_goal_diff,
            "away_goal_diff": away_goal_diff,
            "goal_diff_diff": home_goal_diff - away_goal_diff,
        }
    
    def process_match(self,  home:str, away:str, home_score:int, away_score:int, result:str, matchdate: datetime):

        home_elo= self.elo[home]
        away_elo= self.elo[away]

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

        exp_h = _expected_elo_score(home_elo + const.ELO_HOME_ADVANTAGE, away_elo)
        self.elo[home] += const.ELO_K * (actual_h - exp_h)
        self.elo[away] += const.ELO_K * (actual_a - (1.0 - exp_h))

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

        self.team_last_date[home] = matchdate
        self.team_last_date[away] = matchdate
        self.matches_processed += 1

    def get_team_info(self, team: str) -> dict | None:
        if team not in self.known_teams:
            return None

        hist = self.team_history[team]
        last5 = hist[-const.FORM_WINDOW:]

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
            