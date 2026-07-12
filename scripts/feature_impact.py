#!/usr/bin/env python3
"""Estimate feature impact of enriched weather/moon variables on outcomes."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LAT_COL = "Latitude"
LON_COL = "Longitude"
DATE_COL = "Release_Date"
SPECIES_COL = "Species_Name"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "Latitude",
    "Longitude",
    "Year",
    "Month_Number",
    "Moon_Phase_Sin",
    "Moon_Phase_Cos",
    "Wind_Direction_Sin",
    "Wind_Direction_Cos",
    "Rain_mm",
    "Is_Raining",
]
# Sea_Surface_Temp_C is intentionally excluded: enrichment (scripts/enrich_fish_data.py)
# returned no usable values for it (100% null in every enriched CSV), so it carries no
# information and cannot be imputed. See INTELLIGENT_SYSTEMS_IMPLEMENTATION.md.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate feature impact of enriched inputs on species or catch size."
    )
    parser.add_argument(
        "--input",
        default=os.path.join("newfishdata", "Cleaned_Weather_GameFish_Releases_enriched.csv"),
        help="Path to enriched CSV.",
    )
    parser.add_argument(
        "--target",
        choices=["species", "weight", "length"],
        default="species",
        help="Outcome to model for impact analysis.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV for feature importances (defaults beside input).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on rows for faster runs (0 = no cap).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Permutation importance repeats per feature.",
    )
    return parser.parse_args()


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    sample_value = str(df[DATE_COL].iloc[0]) if not df.empty else ""
    if sample_value.isdigit() and len(sample_value) >= 10:
        df["Date_Obj"] = pd.to_datetime(df[DATE_COL], errors="coerce", unit="ms")
        if df["Date_Obj"].isna().all():
            df["Date_Obj"] = pd.to_datetime(df[DATE_COL], errors="coerce", unit="s")
    else:
        df["Date_Obj"] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)

    df = df.dropna(subset=["Date_Obj"])
    df["Year"] = df["Date_Obj"].dt.year.astype(int)
    df["Month_Number"] = df["Date_Obj"].dt.month.astype(int)
    return df


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Moon_Phase" in df.columns:
        moon = pd.to_numeric(df["Moon_Phase"], errors="coerce")
    else:
        moon = pd.Series(np.nan, index=df.index)
    moon_rad = 2 * np.pi * moon
    df["Moon_Phase_Sin"] = np.sin(moon_rad)
    df["Moon_Phase_Cos"] = np.cos(moon_rad)

    if "Wind_Direction_10m" in df.columns:
        wind = pd.to_numeric(df["Wind_Direction_10m"], errors="coerce")
    else:
        wind = pd.Series(np.nan, index=df.index)
    wind_rad = np.deg2rad(wind)
    df["Wind_Direction_Sin"] = np.sin(wind_rad)
    df["Wind_Direction_Cos"] = np.cos(wind_rad)

    if "Rain_mm" in df.columns:
        df["Rain_mm"] = pd.to_numeric(df["Rain_mm"], errors="coerce")
    else:
        df["Rain_mm"] = np.nan

    if "Is_Raining" in df.columns:
        df["Is_Raining"] = pd.to_numeric(df["Is_Raining"], errors="coerce")
    else:
        df["Is_Raining"] = np.nan

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")
    df = df.dropna(subset=[LAT_COL, LON_COL])
    df = add_date_parts(df)
    df = add_cyclic_features(df)
    return df


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    df = prepare_features(df)

    if args.target == "species":
        target_col = SPECIES_COL
        df = df.dropna(subset=[target_col])
        y = df[target_col].astype(str).str.strip()
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        scoring = "accuracy"
    elif args.target == "weight":
        target_col = "Weight"
        df[target_col] = pd.to_numeric(df.get(target_col), errors="coerce")
        df = df.dropna(subset=[target_col])
        y = df[target_col].astype(float)
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        scoring = "r2"
    else:
        target_col = "Length"
        df[target_col] = pd.to_numeric(df.get(target_col), errors="coerce")
        df = df.dropna(subset=[target_col])
        y = df[target_col].astype(float)
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        scoring = "r2"

    X = df[FEATURE_COLUMNS].astype(float)

    stratify_target = y if args.target == "species" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if args.target == "species":
        score = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {score:.3f}")
    else:
        score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"R2: {score:.3f}")
        print(f"MAE: {mae:.3f}")

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=args.repeats,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    output_path = args.output or os.path.join(
        os.path.dirname(args.input), f"feature_importance_{args.target}.csv"
    )
    importance.to_csv(output_path, index=False)

    print(f"Saved feature importances to: {output_path}")
    print("Top 8 features:")
    print(importance.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
