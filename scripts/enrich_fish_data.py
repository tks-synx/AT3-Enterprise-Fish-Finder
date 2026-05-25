#!/usr/bin/env python3
"""Enrich fisheries records with moon phase, wind, rain, and sea-surface temperature.

This script does not alter existing columns. It appends new fields and writes
an updated CSV. Weather and SST are fetched at local midday using Open-Meteo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from timezonefinder import TimezoneFinder
except ImportError:  # Optional dependency
    TimezoneFinder = None

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

LAT_COL_DEFAULT = "Latitude"
LON_COL_DEFAULT = "Longitude"
DATE_COL_DEFAULT = "Release_Date"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

SYNODIC_MONTH = 29.53058867
KNOWN_NEW_MOON = dt.datetime(2000, 1, 6, 18, 14, tzinfo=dt.timezone.utc)


@dataclass(frozen=True)
class CacheKey:
    grid_lat: float
    grid_lon: float
    local_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich fisheries records with weather + moon data.")
    parser.add_argument(
        "--input",
        default=os.path.join("newfishdata", "GameFish_Releases_Master.csv"),
        help="Path to the source CSV.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("newfishdata", "GameFish_Releases_Master_enriched.csv"),
        help="Output CSV with appended columns.",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=0.1667,
        help="Grid size in degrees for batching weather calls (default ~10 nm).",
    )
    parser.add_argument(
        "--cache",
        default=os.path.join("newfishdata", "enrichment_cache.csv"),
        help="CSV cache for previously fetched grid/date values.",
    )
    parser.add_argument(
        "--timezone-mode",
        choices=["auto", "longitude"],
        default="auto",
        help="Use timezonefinder when available, or approximate from longitude.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between API calls.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Cap the number of new API requests (0 = no cap).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of HTTP retries for transient errors.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=0.3,
        help="Backoff factor between retries.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=200,
        help="Save cache every N new API results (0 = only at end).",
    )
    parser.add_argument(
        "--refresh-missing",
        action="store_true",
        help="Re-fetch cache rows where any weather/SST field is missing.",
    )
    return parser.parse_args()


def round_to_grid(values: pd.Series, grid_size: float) -> pd.Series:
    return (values / grid_size).round() * grid_size


def normalize_longitude(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def clamp_latitude(lat: float) -> float:
    return max(-90.0, min(90.0, lat))


def approx_timezone_offset_hours(lon: float) -> int:
    return int(round(lon / 15.0))


def get_timezone_name(
    lat: float,
    lon: float,
    tz_finder: Optional[TimezoneFinder],
    tz_cache: Dict[Tuple[float, float], Optional[str]],
) -> Optional[str]:
    key = (lat, lon)
    if key in tz_cache:
        return tz_cache[key]

    tz_name = None
    if tz_finder is not None:
        # Ensure longitude is within [-180, 180] for timezonefinder
        try:
            safe_lon = float(normalize_longitude(lon))
        except Exception:
            safe_lon = lon
        tz_name = tz_finder.timezone_at(lat=lat, lng=safe_lon)
    tz_cache[key] = tz_name
    return tz_name


def compute_local_date(
    utc_dt: pd.Timestamp,
    lat: float,
    lon: float,
    tz_mode: str,
    tz_finder: Optional[TimezoneFinder],
    tz_cache: Dict[Tuple[float, float], Optional[str]],
) -> Optional[str]:
    if pd.isna(utc_dt):
        return None

    if tz_mode == "auto" and tz_finder is not None and ZoneInfo is not None:
        tz_name = get_timezone_name(lat, lon, tz_finder, tz_cache)
        if tz_name:
            try:
                local_dt = utc_dt.tz_convert(ZoneInfo(tz_name))
                return local_dt.date().isoformat()
            except Exception:
                pass

    offset_hours = approx_timezone_offset_hours(lon)
    local_dt = (utc_dt.to_pydatetime() + dt.timedelta(hours=offset_hours)).date()
    return local_dt.isoformat()


def moon_phase_fraction(date_str: str) -> float:
    date_value = dt.date.fromisoformat(date_str)
    midpoint = dt.datetime.combine(date_value, dt.time(12, 0), tzinfo=dt.timezone.utc)
    days_since = (midpoint - KNOWN_NEW_MOON).total_seconds() / 86400.0
    phase = (days_since % SYNODIC_MONTH) / SYNODIC_MONTH
    return float(phase)


def pick_hour_value(times: Iterable[str], values: Iterable[float], target_hour: int = 12) -> Optional[float]:
    pairs = [(t, v) for t, v in zip(times, values) if t and v is not None]
    if not pairs:
        return None

    for t, v in pairs:
        if t.endswith(f"T{target_hour:02d}:00"):
            return float(v)

    closest = None
    min_diff = None
    for t, v in pairs:
        try:
            hour = int(t[11:13])
        except Exception:
            continue
        diff = abs(hour - target_hour)
        if min_diff is None or diff < min_diff:
            min_diff = diff
            closest = float(v)
    return closest


def fetch_open_meteo(
    session: requests.Session,
    url: str,
    lat: float,
    lon: float,
    date_str: str,
    fields: str,
    timeout: float = 30.0,
) -> Dict[str, Optional[float]]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": fields,
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "auto",
    }
    try:
        resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.HTTPError as e:
        try:
            body = resp.text
            u = resp.url
        except Exception:
            body = str(e)
            u = f"{url}?{params}"
        print(f"Warning: Open-Meteo HTTP error for {lat},{lon} on {date_str}: {resp.status_code} {u}")
        print("Response body:", body)
        # Return None for each requested (canonicalized) field so the caller can continue
        out = {}
        for f in fields.split(","):
            key = f
            if key.startswith("winddirection_"):
                key = key.replace("winddirection_", "wind_direction_")
            out[key] = None
        return out
    except requests.RequestException as e:
        print(f"Warning: Open-Meteo request failed for {lat},{lon} on {date_str}: {e}")
        out = {}
        for f in fields.split(","):
            key = f
            if key.startswith("winddirection_"):
                key = key.replace("winddirection_", "wind_direction_")
            out[key] = None
        return out

    payload = resp.json()
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])

    results = {}
    for field in fields.split(","):
        values = hourly.get(field, [])
        results[field] = pick_hour_value(times, values, target_hour=12)
    # Normalize some Open-Meteo field keys (e.g. winddirection_10m -> wind_direction_10m)
    normalized: Dict[str, Optional[float]] = {}
    for field, val in results.items():
        key = field
        if key.startswith("winddirection_"):
            key = key.replace("winddirection_", "wind_direction_")
        normalized[key] = val
    return normalized


def build_cache_frame(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "grid_lat",
            "grid_lon",
            "local_date",
            "wind_direction_10m",
            "precipitation",
            "sea_surface_temperature",
        ])
    cache = pd.read_csv(path)
    if "wind_speed_10m" in cache.columns and "wind_direction_10m" not in cache.columns:
        cache = cache.rename(columns={"wind_speed_10m": "wind_direction_10m"})
    for required_col in [
        "grid_lat",
        "grid_lon",
        "local_date",
        "wind_direction_10m",
        "precipitation",
        "sea_surface_temperature",
    ]:
        if required_col not in cache.columns:
            cache[required_col] = np.nan
    cache = cache[[
        "grid_lat",
        "grid_lon",
        "local_date",
        "wind_direction_10m",
        "precipitation",
        "sea_surface_temperature",
    ]]
    return cache


def save_cache(path: str, cache: pd.DataFrame) -> None:
    cache.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    for col in (LAT_COL_DEFAULT, LON_COL_DEFAULT, DATE_COL_DEFAULT):
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

    work = df[[LAT_COL_DEFAULT, LON_COL_DEFAULT, DATE_COL_DEFAULT]].copy()
    work["row_index"] = work.index
    work["lat_num"] = pd.to_numeric(work[LAT_COL_DEFAULT], errors="coerce")
    work["lon_num"] = pd.to_numeric(work[LON_COL_DEFAULT], errors="coerce")
    work["utc_dt"] = pd.to_datetime(work[DATE_COL_DEFAULT], errors="coerce", unit="ms", utc=True)
    work = work.dropna(subset=["lat_num", "lon_num", "utc_dt"])

    work["grid_lat"] = round_to_grid(work["lat_num"], args.grid_size).round(4)
    work["grid_lon"] = round_to_grid(work["lon_num"], args.grid_size).round(4)
    work["grid_lat"] = work["grid_lat"].map(clamp_latitude)
    work["grid_lon"] = work["grid_lon"].map(normalize_longitude)

    tz_finder = TimezoneFinder() if args.timezone_mode == "auto" and TimezoneFinder else None
    tz_cache: Dict[Tuple[float, float], Optional[str]] = {}

    work["local_date"] = work.apply(
        lambda row: compute_local_date(
            row["utc_dt"],
            row["grid_lat"],
            row["grid_lon"],
            args.timezone_mode,
            tz_finder,
            tz_cache,
        ),
        axis=1,
    )
    work = work.dropna(subset=["local_date"])

    cache = build_cache_frame(args.cache)
    if args.refresh_missing and not cache.empty:
        missing_mask = cache[[
            "wind_direction_10m",
            "precipitation",
            "sea_surface_temperature",
        ]].isna().any(axis=1)
        if missing_mask.any():
            cache = cache.loc[~missing_mask].copy()
            save_cache(args.cache, cache)

    cached_keys = set(
        (float(row["grid_lat"]), float(row["grid_lon"]), str(row["local_date"]))
        for _, row in cache.iterrows()
    )

    requests_session = requests.Session()
    # Configure retries/backoff for transient errors if requested
    try:
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
    except Exception:
        Retry = None
        HTTPAdapter = None

    if args.retries and HTTPAdapter is not None and Retry is not None:
        retry_strategy = Retry(
            total=args.retries,
            backoff_factor=args.backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        requests_session.mount("https://", adapter)
        requests_session.mount("http://", adapter)

    unique_keys = (
        work[["grid_lat", "grid_lon", "local_date"]]
        .drop_duplicates()
        .sort_values(["local_date", "grid_lat", "grid_lon"])
    )

    new_rows = []
    request_count = 0
    flush_every = max(0, int(args.flush_every))
    interrupted = False

    try:
        for _, row in unique_keys.iterrows():
            key = (float(row["grid_lat"]), float(row["grid_lon"]), str(row["local_date"]))
            if key in cached_keys:
                continue

            if args.max_requests and request_count >= args.max_requests:
                break

            lat = key[0]
            lon = key[1]
            date_str = key[2]

            # Choose the appropriate Open-Meteo endpoint: use the archive API for historical dates
            try:
                date_obj = dt.date.fromisoformat(date_str)
            except Exception:
                date_obj = None
            if date_obj is not None and date_obj < dt.date.today():
                weather_url = WEATHER_ARCHIVE_URL
            else:
                weather_url = WEATHER_URL

            weather = fetch_open_meteo(
                requests_session,
                weather_url,
                lat,
                lon,
                date_str,
                "winddirection_10m,precipitation",
                timeout=args.timeout,
            )
            marine = fetch_open_meteo(
                requests_session,
                MARINE_URL,
                lat,
                lon,
                date_str,
                "sea_surface_temperature",
                timeout=args.timeout,
            )

            new_rows.append({
                "grid_lat": lat,
                "grid_lon": lon,
                "local_date": date_str,
                "wind_direction_10m": weather.get("wind_direction_10m"),
                "precipitation": weather.get("precipitation"),
                "sea_surface_temperature": marine.get("sea_surface_temperature"),
            })
            request_count += 1

            if flush_every and len(new_rows) >= flush_every:
                cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
                save_cache(args.cache, cache)
                cached_keys.update(
                    (float(r["grid_lat"]), float(r["grid_lon"]), str(r["local_date"]))
                    for r in new_rows
                )
                timestamp = dt.datetime.now().isoformat(timespec="seconds")
                print(
                    f"[{timestamp}] Checkpoint saved: {len(new_rows)} rows (total cache {len(cache)})."
                )
                new_rows = []

            if args.sleep:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted: saving progress so you can resume later...")

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        save_cache(args.cache, cache)
        cached_keys.update(
            (float(r["grid_lat"]), float(r["grid_lon"]), str(r["local_date"]))
            for r in new_rows
        )

    merged = work.merge(
        cache,
        on=["grid_lat", "grid_lon", "local_date"],
        how="left",
    )

    moon_lookup = {
        date_str: moon_phase_fraction(date_str)
        for date_str in merged["local_date"].dropna().unique()
    }
    merged["Moon_Phase"] = merged["local_date"].map(moon_lookup)

    merged = merged.set_index("row_index")

    df_out = df.copy()
    df_out["Moon_Phase"] = merged["Moon_Phase"]
    df_out["Wind_Direction_10m"] = merged["wind_direction_10m"]
    df_out["Rain_mm"] = merged["precipitation"]
    df_out["Is_Raining"] = merged["precipitation"].fillna(0) > 0
    df_out["Sea_Surface_Temp_C"] = merged["sea_surface_temperature"]

    df_out.to_csv(args.output, index=False)

    print(f"Enriched CSV written to: {args.output}")
    print(f"Cache updated: {args.cache}")
    if interrupted:
        print("Run again to continue fetching remaining grid/date keys.")


if __name__ == "__main__":
    main()
