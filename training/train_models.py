"""
Training pipeline for FUTVE match outcome prediction.

Trains and evaluates multiple classifiers (CatBoost, Random Forest,
Logistic Regression) using a chronological train/test split to
prevent data leakage.

Usage:
    python training/train_models.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_CSV = DATA_DIR / "futve_features.csv"
MODELS_DIR = Path(__file__).resolve().parent / "saved_models"

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

TARGET = "result"
LABELS = ["H", "D", "A"]

# Minimum matches a team must have played before we use that row for training,
# so the features are not all zeros.
MIN_MATCHES_PLAYED = 5

# Chronological split: use last N seasons as test set.
TEST_SEASONS = 3


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    df["match_date_utc"] = pd.to_datetime(df["match_date_utc"], utc=True)
    return df


def filter_cold_start(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where either team has fewer than MIN_MATCHES_PLAYED games."""
    mask = (df["home_matches_played"] >= MIN_MATCHES_PLAYED) & (
        df["away_matches_played"] >= MIN_MATCHES_PLAYED
    )
    filtered = df[mask].reset_index(drop=True)
    print(
        f"  Cold-start filter: {len(df)} -> {len(filtered)} rows "
        f"(removed {len(df) - len(filtered)} with < {MIN_MATCHES_PLAYED} matches played)"
    )
    return filtered


def temporal_split(df: pd.DataFrame):
    """
    Split by season chronologically.

    The last TEST_SEASONS seasons become the test set, everything before
    is training. This mimics real-world usage where the model is trained
    on past data and predicts future matches.
    """
    seasons = sorted(df["season"].unique())
    test_seasons = set(seasons[-TEST_SEASONS:])
    train_seasons = set(seasons) - test_seasons

    train_df = df[df["season"].isin(train_seasons)].reset_index(drop=True)
    test_df = df[df["season"].isin(test_seasons)].reset_index(drop=True)

    print(f"  Train seasons: {sorted(train_seasons)[-3:]} ... ({len(train_seasons)} total)")
    print(f"  Test seasons:  {sorted(test_seasons)}")
    print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")

    return train_df, test_df


def print_evaluation(name: str, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {acc:.4f}")
    print()
    print(classification_report(y_true, y_pred, labels=LABELS, target_names=LABELS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
    print("Confusion Matrix:")
    print(cm_df.to_string())
    print()

    return acc


def print_feature_importance(name: str, feature_names: list[str], importances: np.ndarray):
    pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    print(f"  Top 15 features ({name}):")
    for feat, imp in pairs[:15]:
        bar = "#" * int(imp / pairs[0][1] * 30)
        print(f"    {feat:<30s} {imp:8.4f}  {bar}")
    print()


def train_catboost(X_train, y_train, X_test, y_test):
    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0,
        auto_class_weights="Balanced",
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

    y_pred = model.predict(X_test).flatten()
    acc = print_evaluation("CatBoost", y_test, y_pred)
    print_feature_importance("CatBoost", FEATURE_COLUMNS, model.feature_importances_)

    return model, acc


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = print_evaluation("Random Forest", y_test, y_pred)
    print_feature_importance("Random Forest", FEATURE_COLUMNS, model.feature_importances_)

    return model, acc


def train_logistic_regression(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = print_evaluation("Logistic Regression (baseline)", y_test, y_pred)

    return model, acc, scaler


def baseline_always_home(y_test):
    y_pred = np.full(len(y_test), "H")
    acc = print_evaluation("Baseline: always predict Home", y_test, y_pred)
    return acc


def main():
    print("=" * 60)
    print("  FUTVE Match Prediction - Model Training")
    print("=" * 60)

    print(f"\nLoading features from {FEATURES_CSV}")
    df = load_features()
    print(f"  -> {len(df)} rows, {len(df.columns)} columns")

    print("\nFiltering cold-start matches...")
    df = filter_cold_start(df)

    print("\nSplitting data (temporal split)...")
    train_df, test_df = temporal_split(df)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_train = train_df[TARGET].to_numpy(dtype=str, na_value="")
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_test = test_df[TARGET].to_numpy(dtype=str, na_value="")

    print(f"\nTarget distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Target distribution (test):  {pd.Series(y_test).value_counts().to_dict()}")

    # ---- Baselines ----
    baseline_acc = baseline_always_home(y_test)

    # ---- CatBoost ----
    cb_model, cb_acc = train_catboost(X_train, y_train, X_test, y_test)

    # ---- Random Forest ----
    rf_model, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)

    # ---- Logistic Regression ----
    lr_model, lr_acc, lr_scaler = train_logistic_regression(X_train, y_train, X_test, y_test)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    results = {
        "Baseline (always H)": baseline_acc,
        "Logistic Regression": lr_acc,
        "Random Forest": rf_acc,
        "CatBoost": cb_acc,
    }
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " <-- best" if acc == max(results.values()) else ""
        print(f"  {name:<30s} {acc:.4f}{marker}")

    # ---- Save best model ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cb_model.save_model(str(MODELS_DIR / "catboost_model.cbm"))

    metadata = {
        "features": FEATURE_COLUMNS,
        "target": TARGET,
        "labels": LABELS,
        "test_seasons": sorted(test_df["season"].unique().tolist()),
        "results": {k: round(v, 4) for k, v in results.items()},
        "min_matches_played": MIN_MATCHES_PLAYED,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  CatBoost model saved to {MODELS_DIR / 'catboost_model.cbm'}")
    print(f"  Metadata saved to {MODELS_DIR / 'model_metadata.json'}")
    print()


if __name__ == "__main__":
    main()
