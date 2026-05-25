#!/usr/bin/env python3
"""Convert a shapefile export into the Fisheries app CSV schema.

Expected output columns:
Tag_Number,Release_Da,Latitude,Longitude,Species_Na,Length,Weight,Latitude_F,Longitude_

Usage:
  python scripts/import_shapefile_to_fish_csv.py \
    --input-dir /path/to/export/folder \
    --basename GameFish_Tagging_Releases \
    --output Fish_Data_2.csv \
    --backup
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import shapefile  # pyshp


TARGET_COLUMNS = [
    "Tag_Number",
    "Release_Da",
    "Latitude",
    "Longitude",
    "Species_Na",
    "Length",
    "Weight",
    "Latitude_F",
    "Longitude_",
]


def normalize_col(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def build_alias_map(columns: list[str]) -> dict[str, str]:
    normalized = {normalize_col(col): col for col in columns}

    def first_match(*aliases: str) -> str | None:
        for alias in aliases:
            if alias in normalized:
                return normalized[alias]
        return None

    return {
        "Tag_Number": first_match("tagnumber", "tagno", "tagid", "tag"),
        "Release_Da": first_match("releaseda", "releasedate", "rel_date", "date", "datereleased"),
        "Latitude": first_match("latitude", "lat", "lattext", "latstr"),
        "Longitude": first_match("longitude", "lon", "long", "lng", "lontext", "lonstr"),
        "Species_Na": first_match("speciesna", "species", "speciesname", "specname"),
        "Length": first_match("length", "len", "lengthcm", "len_cm"),
        "Weight": first_match("weight", "wt", "weightkg", "wtkg"),
        "Latitude_F": first_match("latitudef", "latf", "latdec", "latdecimal"),
        "Longitude_": first_match("longitudef", "lonf", "londec", "longdec", "londecimal"),
    }


def parse_lat_lon_text(value: str, is_lat: bool) -> float | None:
    """Parse values like '33.52S' or '151.27E' into signed decimal degrees."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None

    hemisphere = ""
    if text[-1:] in {"N", "S", "E", "W"}:
        hemisphere = text[-1]
        text = text[:-1]

    if "." in text:
        parts = text.split(".", 1)
        if parts[0].isdigit() and parts[1].isdigit() and len(parts[1]) == 2:
            deg = float(parts[0])
            mins = float(parts[1])
            number = deg + (mins / 60.0)
        else:
            try:
                number = float(text)
            except ValueError:
                return None
    else:
        try:
            number = float(text)
        except ValueError:
            return None

    if hemisphere in {"S", "W"}:
        number = -abs(number)
    elif hemisphere in {"N", "E"}:
        number = abs(number)

    if is_lat and not (-90 <= number <= 90):
        return None
    if (not is_lat) and not (-180 <= number <= 180):
        return None
    return number


def clean_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    # Keep existing app style: month/day/2-digit-year (no leading zero month/day on macOS)
    formatted = parsed.dt.strftime("%-m/%-d/%y")
    return formatted.fillna("")


def load_shapefile_as_df(shp_path: Path) -> pd.DataFrame:
    reader = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
    rows = [list(rec) for rec in reader.records()]
    df = pd.DataFrame(rows, columns=fields)

    # Add geometry-derived lon/lat for point shapefiles where needed.
    if reader.shapeType in {
        shapefile.POINT,
        shapefile.POINTM,
        shapefile.POINTZ,
    }:
        lons = []
        lats = []
        for shape in reader.shapes():
            if shape.points:
                lon, lat = shape.points[0]
                lons.append(lon)
                lats.append(lat)
            else:
                lons.append(None)
                lats.append(None)
        df["_geom_lon"] = lons
        df["_geom_lat"] = lats

    return df


def transform_to_target(df: pd.DataFrame) -> pd.DataFrame:
    aliases = build_alias_map(df.columns.tolist())

    out = pd.DataFrame(index=df.index)

    # Core string columns
    out["Tag_Number"] = df[aliases["Tag_Number"]].astype(str).str.strip() if aliases["Tag_Number"] else ""
    out["Species_Na"] = (
        df[aliases["Species_Na"]].astype(str).str.strip().str.upper()
        if aliases["Species_Na"]
        else ""
    )

    # Date column
    if aliases["Release_Da"]:
        out["Release_Da"] = clean_date(df[aliases["Release_Da"]])
    else:
        out["Release_Da"] = ""

    # Raw text lat/lon if available
    out["Latitude"] = df[aliases["Latitude"]].astype(str).str.strip() if aliases["Latitude"] else ""
    out["Longitude"] = df[aliases["Longitude"]].astype(str).str.strip() if aliases["Longitude"] else ""

    # Numeric optionals
    out["Length"] = pd.to_numeric(df[aliases["Length"]], errors="coerce") if aliases["Length"] else 0
    out["Weight"] = pd.to_numeric(df[aliases["Weight"]], errors="coerce") if aliases["Weight"] else 0

    lat_numeric = None
    lon_numeric = None

    if aliases["Latitude_F"]:
        lat_numeric = pd.to_numeric(df[aliases["Latitude_F"]], errors="coerce")
    elif "_geom_lat" in df.columns:
        lat_numeric = pd.to_numeric(df["_geom_lat"], errors="coerce")
    elif aliases["Latitude"]:
        lat_numeric = df[aliases["Latitude"]].apply(lambda v: parse_lat_lon_text(v, is_lat=True))

    if aliases["Longitude_"]:
        lon_numeric = pd.to_numeric(df[aliases["Longitude_"]], errors="coerce")
    elif "_geom_lon" in df.columns:
        lon_numeric = pd.to_numeric(df["_geom_lon"], errors="coerce")
    elif aliases["Longitude"]:
        lon_numeric = df[aliases["Longitude"]].apply(lambda v: parse_lat_lon_text(v, is_lat=False))

    out["Latitude_F"] = lat_numeric if lat_numeric is not None else pd.Series([None] * len(out))
    out["Longitude_"] = lon_numeric if lon_numeric is not None else pd.Series([None] * len(out))

    # Fill app-expected numeric defaults and drop unusable rows
    out["Length"] = out["Length"].fillna(0.0).astype(float)
    out["Weight"] = out["Weight"].fillna(0.0).astype(float)

    out = out.dropna(subset=["Latitude_F", "Longitude_"])
    out = out[out["Species_Na"].astype(str).str.len() > 0]
    out = out[out["Release_Da"].astype(str).str.len() > 0]

    # Force the exact column order used by the existing project.
    out = out[TARGET_COLUMNS]
    return out


def resolve_shapefile_path(input_dir: Path, basename: str) -> Path:
    shp_path = input_dir / f"{basename}.shp"
    if shp_path.exists():
        return shp_path

    shps = sorted(input_dir.glob("*.shp"))
    if len(shps) == 1:
        return shps[0]

    if not shps:
        raise FileNotFoundError(f"No .shp file found in {input_dir}")

    raise FileNotFoundError(
        "Multiple .shp files found. Provide --basename to choose one."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Import shapefile data into Fish_Data_2.csv format.")
    parser.add_argument("--input-dir", required=True, help="Folder containing .shp/.dbf/.shx files")
    parser.add_argument("--basename", default="", help="Base file name without extension")
    parser.add_argument("--output", default="Fish_Data_2.csv", help="Output CSV path")
    parser.add_argument("--backup", action="store_true", help="Backup existing output file before overwrite")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    basename = args.basename.strip()
    if basename:
        shp_path = resolve_shapefile_path(input_dir, basename)
    else:
        shp_path = resolve_shapefile_path(input_dir, "")

    df_raw = load_shapefile_as_df(shp_path)
    df_clean = transform_to_target(df_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.backup and output_path.exists():
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_suffix(f".backup_{timestamp}.csv")
        shutil.copy2(output_path, backup_path)
        print(f"Backup created: {backup_path}")

    df_clean.to_csv(output_path, index=False)
    print(f"Wrote {len(df_clean)} rows to {output_path}")
    print(f"Columns: {', '.join(df_clean.columns.tolist())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
