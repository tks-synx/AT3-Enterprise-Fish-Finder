"""Rule-based certainty-factor expert system for fishing recommendations.

This module implements the expert system described in the planning documents:
- scripts/CertaintyFactors.md  (rule -> certainty factor point values, final score bands)
- scripts/IF-THEN.md           (the IF-THEN rule logic and recommendation thresholds)
- scripts/Flowchart.md         (the decision flow: species check -> season -> catch
                                 density -> location -> user preference -> final score)

It is intentionally kept separate from fish_app.py (no Streamlit/UI code here) so the
rule logic can be read, explained, and even run on its own, independently of the
neural-network model in train_species_nn.py. The two intelligent systems are meant to
complement each other:
  - train_species_nn.py / fish_species_nn.joblib -> data-driven predictive modelling
  - certainty_factor.py                          -> rule-based expert-system reasoning

Rule set and points (from scripts/CertaintyFactors.md):
    Species exists          +10
    Season match             +25
    High catch density       +35
    Medium catch density     +20
    Low catch density        +5
    Location match           +25
    User preference match    +15

Recommendation thresholds (from scripts/IF-THEN.md, which gives exact operators where
scripts/CertaintyFactors.md only gives a rounded summary table):
    score >= 75          -> Strong recommendation
    45 <= score < 75      -> Moderate recommendation
    0 < score < 45        -> Weak recommendation / advise changing species, season or location
    species not in data   -> Insufficient data (score forced to 0, per the IF-THEN "ELSE" branch)

Note: because several rules can fire together, the maximum achievable score is
10 + 25 + 35 + 25 + 15 = 110, which is slightly above the "75-100" wording used in the
summary table in CertaintyFactors.md. The IF-THEN.md thresholds (">= 75", "45-74", "< 45")
are used directly here since they are unambiguous at any score, including above 100.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

SPECIES_COL = 'Species_Name'
LAT_COL = 'Latitude'
LON_COL = 'Longitude'
MONTH_COL = 'Month_Name'
MOON_PHASE_COL = 'Moon_Phase'
WIND_DIRECTION_COL = 'Wind_Direction_10m'
RAIN_COL = 'Rain_mm'
IS_RAINING_COL = 'Is_Raining'

RULE_POINTS = {
    'species_exists': 10,
    'season_match': 25,
    'catch_density_high': 35,
    'catch_density_medium': 20,
    'catch_density_low': 5,
    'location_match': 25,
    'user_preference_match': 15,
}

# Maximum possible score if every rule fires at its best tier (10 + 25 + 35 + 25 + 15).
MAX_SCORE = (
    RULE_POINTS['species_exists']
    + RULE_POINTS['season_match']
    + RULE_POINTS['catch_density_high']
    + RULE_POINTS['location_match']
    + RULE_POINTS['user_preference_match']
)

STRONG_THRESHOLD = 75
MODERATE_THRESHOLD = 45

TOP_SEASON_MONTHS = 4
NEARBY_WIND_DEGREES = 45.0
NEARBY_MOON_PHASE_FRACTION = 0.1
HIGH_DENSITY_COUNT = 20
MEDIUM_DENSITY_COUNT = 5
LOCATION_PERCENTILE_LOW = 5
LOCATION_PERCENTILE_HIGH = 95


def _rule(rule_id: str, description: str, passed: bool, points: int) -> dict:
    return {
        'rule': rule_id,
        'description': description,
        'passed': passed,
        'points': points if passed else 0,
    }


def _season_match(species_subset: pd.DataFrame, month_name: Optional[str]) -> dict:
    if not month_name or MONTH_COL not in species_subset.columns:
        return _rule(
            'season_match',
            'No month selected, so season match could not be checked.',
            False,
            RULE_POINTS['season_match'],
        )

    month_counts = species_subset[MONTH_COL].value_counts()
    if month_counts.empty:
        return _rule('season_match', 'No historical month data for this species.', False, RULE_POINTS['season_match'])

    top_months = month_counts.head(TOP_SEASON_MONTHS).index.tolist()
    passed = month_name in top_months
    description = (
        f"{month_name} is one of this species' top {len(top_months)} historical months "
        f"({', '.join(top_months)})."
        if passed
        else f"{month_name} is not among this species' top {len(top_months)} historical months "
        f"({', '.join(top_months)})."
    )
    return _rule('season_match', description, passed, RULE_POINTS['season_match'])


def _catch_density(
    species_subset: pd.DataFrame,
    latitude: Optional[float],
    longitude: Optional[float],
    wind_direction: Optional[float],
    rain_mm: Optional[float],
    is_raining: Optional[bool],
    moon_phase: Optional[float],
    grid_size: float = 0.1667,
) -> dict:
    if latitude is None or longitude is None:
        return _rule(
            'catch_density',
            'No location context available to check historical catch density.',
            False,
            0,
        )

    nearby = species_subset.copy()
    nearby['_lat_diff'] = (nearby[LAT_COL] - latitude).abs()
    nearby['_lon_diff'] = (nearby[LON_COL] - longitude).abs()
    nearby = nearby[(nearby['_lat_diff'] <= grid_size) & (nearby['_lon_diff'] <= grid_size)]

    condition_notes = ['within the local area (~10nm grid)']

    if moon_phase is not None and MOON_PHASE_COL in nearby.columns:
        moon_values = pd.to_numeric(nearby[MOON_PHASE_COL], errors='coerce')
        nearby = nearby[(moon_values - moon_phase).abs() <= NEARBY_MOON_PHASE_FRACTION]
        condition_notes.append('similar moon phase')

    if wind_direction is not None and WIND_DIRECTION_COL in nearby.columns:
        wind_values = pd.to_numeric(nearby[WIND_DIRECTION_COL], errors='coerce')
        angular_diff = (wind_values - wind_direction).abs() % 360
        angular_diff = np.minimum(angular_diff, 360 - angular_diff)
        nearby = nearby[angular_diff <= NEARBY_WIND_DEGREES]
        condition_notes.append('similar wind direction')

    if is_raining is not None and IS_RAINING_COL in nearby.columns:
        rain_flag_values = pd.to_numeric(nearby[IS_RAINING_COL], errors='coerce').fillna(0) > 0
        nearby = nearby[rain_flag_values == bool(is_raining)]
        condition_notes.append('matching rain conditions')

    count = len(nearby)
    conditions_text = ', '.join(condition_notes)

    if count >= HIGH_DENSITY_COUNT:
        return _rule(
            'catch_density_high',
            f'High historical catch density: {count} historical catches match {conditions_text}.',
            True,
            RULE_POINTS['catch_density_high'],
        )
    if count >= MEDIUM_DENSITY_COUNT:
        return _rule(
            'catch_density_medium',
            f'Medium historical catch density: {count} historical catches match {conditions_text}.',
            True,
            RULE_POINTS['catch_density_medium'],
        )
    if count >= 1:
        return _rule(
            'catch_density_low',
            f'Low historical catch density: only {count} historical catches match {conditions_text}.',
            True,
            RULE_POINTS['catch_density_low'],
        )
    return _rule(
        'catch_density_low',
        f'No historical catches found that match {conditions_text}.',
        False,
        0,
    )


def _location_match(species_subset: pd.DataFrame, latitude: Optional[float], longitude: Optional[float]) -> dict:
    if latitude is None or longitude is None:
        return _rule('location_match', 'No location context available to check.', False, RULE_POINTS['location_match'])

    lat_values = pd.to_numeric(species_subset[LAT_COL], errors='coerce').dropna()
    lon_values = pd.to_numeric(species_subset[LON_COL], errors='coerce').dropna()
    if lat_values.empty or lon_values.empty:
        return _rule('location_match', 'No historical location data for this species.', False, RULE_POINTS['location_match'])

    lat_low, lat_high = np.percentile(lat_values, [LOCATION_PERCENTILE_LOW, LOCATION_PERCENTILE_HIGH])
    lon_low, lon_high = np.percentile(lon_values, [LOCATION_PERCENTILE_LOW, LOCATION_PERCENTILE_HIGH])

    passed = (lat_low <= latitude <= lat_high) and (lon_low <= longitude <= lon_high)
    description = (
        f"Location ({latitude:.2f}, {longitude:.2f}) is inside this species' typical "
        f"{LOCATION_PERCENTILE_LOW}-{LOCATION_PERCENTILE_HIGH} percentile range."
        if passed
        else f"Location ({latitude:.2f}, {longitude:.2f}) is outside this species' typical "
        f"{LOCATION_PERCENTILE_LOW}-{LOCATION_PERCENTILE_HIGH} percentile range "
        f"(lat {lat_low:.2f} to {lat_high:.2f}, lon {lon_low:.2f} to {lon_high:.2f})."
    )
    return _rule('location_match', description, passed, RULE_POINTS['location_match'])


def _user_preference_match(species: str, month_name: Optional[str], user_profile: Optional[dict]) -> dict:
    if not user_profile:
        return _rule('user_preference_match', 'No user feedback history available yet.', False, RULE_POINTS['user_preference_match'])

    filter_stats = user_profile.get('filter_stats', {})
    species_stats = filter_stats.get('species', {}).get(species)
    month_stats = filter_stats.get('months', {}).get(month_name) if month_name else None

    species_liked = bool(species_stats) and species_stats.get('positive', 0) > species_stats.get('negative', 0)
    month_liked = bool(month_stats) and month_stats.get('positive', 0) > month_stats.get('negative', 0)

    passed = species_liked or month_liked
    if passed:
        liked_parts = []
        if species_liked:
            liked_parts.append(f"species '{species}'")
        if month_liked:
            liked_parts.append(f"month '{month_name}'")
        description = f"User has previously given positive feedback for {' and '.join(liked_parts)}."
    else:
        description = 'No prior positive user feedback recorded for this species or month.'

    return _rule('user_preference_match', description, passed, RULE_POINTS['user_preference_match'])


def classify_score(score: int) -> str:
    if score <= 0:
        return 'insufficient'
    if score >= STRONG_THRESHOLD:
        return 'strong'
    if score >= MODERATE_THRESHOLD:
        return 'moderate'
    return 'weak'


BAND_LABELS = {
    'insufficient': 'Insufficient data',
    'strong': 'Strong recommendation',
    'moderate': 'Moderate recommendation',
    'weak': 'Weak recommendation',
}


def evaluate_certainty_factor(
    df: pd.DataFrame,
    species: str,
    month_name: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    wind_direction: Optional[float] = None,
    rain_mm: Optional[float] = None,
    is_raining: Optional[bool] = None,
    moon_phase: Optional[float] = None,
    user_profile: Optional[dict] = None,
) -> dict:
    """Evaluate the certainty-factor rules for one species/context and return a
    plain-dict result: score, confidence band, rules fired/evaluated, and a short
    plain-English recommendation. Mirrors the flow in scripts/Flowchart.md.
    """

    species_subset = df[df[SPECIES_COL] == species]

    species_exists_rule = _rule(
        'species_exists',
        f"'{species}' appears {len(species_subset)} time(s) in the historical dataset."
        if not species_subset.empty
        else f"'{species}' does not appear in the historical dataset.",
        not species_subset.empty,
        RULE_POINTS['species_exists'],
    )

    if species_subset.empty:
        return {
            'species': species,
            'score': 0,
            'max_score': MAX_SCORE,
            'band': 'insufficient',
            'band_label': BAND_LABELS['insufficient'],
            'rules_fired': [species_exists_rule],
            'recommendation': (
                f"Insufficient data: '{species}' does not appear in the historical dataset, "
                "so no expert-system recommendation can be made (see scripts/IF-THEN.md, rule 1)."
            ),
        }

    rules = [
        species_exists_rule,
        _season_match(species_subset, month_name),
        _catch_density(species_subset, latitude, longitude, wind_direction, rain_mm, is_raining, moon_phase),
        _location_match(species_subset, latitude, longitude),
        _user_preference_match(species, month_name, user_profile),
    ]

    score = sum(rule['points'] for rule in rules)
    band = classify_score(score)
    band_label = BAND_LABELS[band]

    fired_descriptions = [rule['description'] for rule in rules if rule['passed']]
    if fired_descriptions:
        explanation = (
            f"{band_label} ({score}/{MAX_SCORE}) for {species}"
            + (f" in {month_name}" if month_name else "")
            + ". " + ' '.join(fired_descriptions)
        )
    else:
        explanation = (
            f"{band_label} ({score}/{MAX_SCORE}) for {species}"
            + (f" in {month_name}" if month_name else "")
            + ". None of the supporting rules fired for the current context; "
            "consider changing species, month, or location (see scripts/IF-THEN.md)."
        )

    return {
        'species': species,
        'score': score,
        'max_score': MAX_SCORE,
        'band': band,
        'band_label': band_label,
        'rules_fired': rules,
        'recommendation': explanation,
    }


if __name__ == '__main__':
    # Small standalone demo so the rule engine can be explained/run without Streamlit.
    demo_df = pd.DataFrame({
        SPECIES_COL: ['YELLOWFIN TUNA'] * 30,
        LAT_COL: np.random.normal(-33.5, 0.2, 30),
        LON_COL: np.random.normal(151.2, 0.2, 30),
        MONTH_COL: ['January'] * 20 + ['July'] * 10,
        MOON_PHASE_COL: np.random.uniform(0, 1, 30),
        WIND_DIRECTION_COL: np.random.uniform(0, 360, 30),
        RAIN_COL: np.random.uniform(0, 5, 30),
        IS_RAINING_COL: [0] * 25 + [1] * 5,
    })
    result = evaluate_certainty_factor(
        demo_df,
        species='YELLOWFIN TUNA',
        month_name='January',
        latitude=-33.5,
        longitude=151.2,
        wind_direction=180.0,
        rain_mm=0.0,
        is_raining=False,
        moon_phase=0.5,
    )
    print(result['recommendation'])
    for fired_rule in result['rules_fired']:
        print(f"  [{'x' if fired_rule['passed'] else ' '}] {fired_rule['rule']}: {fired_rule['description']} (+{fired_rule['points']})")
