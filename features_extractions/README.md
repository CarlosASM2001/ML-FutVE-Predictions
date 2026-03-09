# ML Project FUTVE Predictions - Feature Engineering

## Overview

This folder contains the scripts, notebooks, and intermediate outputs used to extract and build predictive features from the raw match data.

The main purpose of this stage is to transform basic match information into meaningful variables that can improve the performance of machine learning models for FUTVE match outcome prediction.

## Why Feature Engineering Matters

The raw dataset only includes basic match identifiers such as:

- `home_team`
- `away_team`

While these fields are useful, they are not enough for a model to capture team strength, momentum, or context. To build a stronger predictive system, the project needs historical and performance-based features.

Feature engineering helps convert raw match history into variables that better represent each team's condition before a match.

## Example Features

Some examples of features that can be generated in this folder include:

- Team's last 5 matches performance
- Average goals scored
- Average goals conceded
- Recent form
- Team ranking
- Elo rating
- Home advantage indicators
- Win, draw, and loss streaks
- Head-to-head history

### Sample Engineered Columns

```text
home_last5_wins
away_last5_wins
home_avg_goals
away_avg_goals
home_avg_goals_conceded
away_avg_goals_conceded
home_form_points
away_form_points
home_elo
away_elo
home_rank
away_rank
home_win_streak
away_win_streak
```

## Folder Objective

The goal of this folder is to:

* Read the consolidated historical match dataset
* Sort matches chronologically
* Generate pre-match features using only past information
* Avoid data leakage
* Export a clean feature dataset ready for model training

## Important Principle

All features must be calculated using only information available before the match being predicted.

This is critical to avoid data leakage, which happens when the model accidentally uses future information during training.