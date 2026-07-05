import json
import hashlib
import os
import re
from datetime import datetime, timezone

import joblib
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import Draw, HeatMap, MeasureControl
from streamlit_folium import st_folium
import math
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from certainty_factor import evaluate_certainty_factor

# Try to import ollama - if not installed, show setup message
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ==== How To Run ====
# python3 -m streamlit run BasicProduction/fish_app.py


# ================= CONFIGURATION =================
INPUT_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'newfishdata', 'Cleaned_Weather_GameFish_Releases_enriched.csv'
)
LAT_COL = 'Latitude'      
LON_COL = 'Longitude'      
SPECIES_COL = 'Species_Name'  
DATE_COL = 'Release_Date'     
MODEL_NAME = 'tinyllama'
PROFILE_FILE = os.path.join(os.path.dirname(__file__), 'user_profile.json')
FEEDBACK_CATEGORIES = [
    'Relevancy',
    'Clarity',
    'Specificity',
    'Map usefulness',
    'Overall quality',
]
VESSEL_TYPES = ['Displacement', 'Semi-planing', 'Planing']
DEFAULT_SYDNEY_SPEED_LIMIT_KN = 4.0
NN_FEATURE_COLUMNS = [
    'Latitude',
    'Longitude',
    'Year',
    'Month_Number',
    'Moon_Phase_Sin',
    'Moon_Phase_Cos',
    'Wind_Direction_Sin',
    'Wind_Direction_Cos',
    'Rain_mm',
    'Is_Raining',
]
NN_TOP_SPECIES_LIMIT = 12
NN_MIN_CLASS_SAMPLES = 75
NN_TEST_SIZE = 0.2
NN_RANDOM_STATE = 42
NN_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'fish_species_nn.joblib')
# =================================================

st.set_page_config(layout="wide", page_title="Fisheries Heatmap")

# --- CSS: HIDE HEADER & REDUCE WHITESPACE ---
st.markdown("""
    <style>
        header[data-testid="stHeader"] { display: none; }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
        div.stButton > button {
            width: 100%;
            margin-top: 0px; 
            margin-bottom: 0px;
        }
    </style>
""", unsafe_allow_html=True)

# --- AI SETUP ---
def init_ollama():
    """Check if Ollama server is running"""
    try:
        tags = ollama.list()
        return True
    except Exception:
        return False

def get_data_summary(df, species_filter, year_filter, month_filter):
    """Generate a text summary of the current data for AI context"""
    total_records = len(df)
    unique_species = len(df[SPECIES_COL].unique())
    date_range = f"{df['Year'].min()}-{df['Year'].max()}"
    lat_range = f"{df[LAT_COL].min():.2f} to {df[LAT_COL].max():.2f}"
    lon_range = f"{df[LON_COL].min():.2f} to {df[LON_COL].max():.2f}"
    
    return f"""Current Dataset Context:
- Total Records: {total_records}
- Unique Species: {unique_species}
- Years Available: {date_range}
- Latitude Range: {lat_range}
- Longitude Range: {lon_range}
- Selected Species: {', '.join(species_filter) if species_filter else 'All'}
- Selected Years: {', '.join(map(str, year_filter)) if year_filter else 'All'}
- Selected Months: {', '.join(month_filter) if month_filter else 'All'}"""

def chat_with_ai(user_message, data_summary):
    """Send a message to TinyLlama and get a response"""
    try:
        preference_context = build_user_memory_context(user_profile)
        prompt = f"""You are an expert fisheries analyst.

{data_summary}

{preference_context}

User question: {user_message}

Provide a concise, helpful response about fisheries patterns, fish behavior, or data insights.
If a user preference memory exists, prioritize those patterns when the question is ambiguous."""
        
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            stream=False
        )
        return response['response']
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- 1. DATA LOADER ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(INPUT_FILE)
        
        found_cols = df.columns.tolist()
        missing = [c for c in [LAT_COL, LON_COL, SPECIES_COL, DATE_COL] if c not in found_cols]
        if missing:
            return None, f"MISSING COLUMNS: {missing}"

        df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors='coerce')
        df[LON_COL] = pd.to_numeric(df[LON_COL], errors='coerce')
        df = df.dropna(subset=[LAT_COL, LON_COL])

        # Handle date parsing for both old (string) and new (Unix timestamp) formats
        # First, try to detect if this is a Unix timestamp (very large number) or string date
        sample_value = str(df[DATE_COL].iloc[0]) if not df.empty else ''
        
        if sample_value.isdigit() and len(sample_value) >= 10:
            # Likely Unix timestamp (milliseconds or seconds)
            df['Date_Obj'] = pd.to_datetime(df[DATE_COL], errors='coerce', unit='ms')
            # If all NaT, try seconds instead
            if df['Date_Obj'].isna().all():
                df['Date_Obj'] = pd.to_datetime(df[DATE_COL], errors='coerce', unit='s')
        else:
            # String date format - try m/d/yy first, then fallback to dayfirst
            df['Date_Obj'] = pd.to_datetime(df[DATE_COL], errors='coerce', format='%m/%d/%y')
            if df['Date_Obj'].isna().any():
                fallback_mask = df['Date_Obj'].isna()
                df.loc[fallback_mask, 'Date_Obj'] = pd.to_datetime(
                    df.loc[fallback_mask, DATE_COL], errors='coerce', dayfirst=True
                )
        
        df = df.dropna(subset=['Date_Obj'])

        df['Year'] = df['Date_Obj'].dt.year.astype(int)
        df['Month_Number'] = df['Date_Obj'].dt.month.astype(int)
        df['Month_Name'] = df['Date_Obj'].dt.month_name()
        df = df.dropna(subset=[SPECIES_COL])
        df[SPECIES_COL] = df[SPECIES_COL].astype(str).str.strip()
        df = add_model_features(df)
        
        return df, "Success"
    except Exception as e:
        return None, f"Critical Error: {str(e)}"


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'Moon_Phase' in df.columns:
        moon = pd.to_numeric(df['Moon_Phase'], errors='coerce')
    else:
        moon = pd.Series(np.nan, index=df.index)
    moon_rad = 2 * np.pi * moon
    df['Moon_Phase_Sin'] = np.sin(moon_rad)
    df['Moon_Phase_Cos'] = np.cos(moon_rad)

    if 'Wind_Direction_10m' in df.columns:
        wind = pd.to_numeric(df['Wind_Direction_10m'], errors='coerce')
    else:
        wind = pd.Series(np.nan, index=df.index)
    wind_rad = np.deg2rad(wind)
    df['Wind_Direction_Sin'] = np.sin(wind_rad)
    df['Wind_Direction_Cos'] = np.cos(wind_rad)

    if 'Rain_mm' in df.columns:
        df['Rain_mm'] = pd.to_numeric(df['Rain_mm'], errors='coerce')
    else:
        df['Rain_mm'] = np.nan

    if 'Is_Raining' in df.columns:
        df['Is_Raining'] = pd.to_numeric(df['Is_Raining'], errors='coerce')
    else:
        df['Is_Raining'] = np.nan

    # Sea_Surface_Temp_C is intentionally excluded because enrichment
    # produced no usable values (100% null), so it is not a model feature.
    return df

df, status_msg = load_data()

if df is None or df.empty:
    st.error("Data Load Failed")
    st.stop()

all_species = sorted(df[SPECIES_COL].unique())
all_years = sorted(df['Year'].unique())
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
available_months = sorted(
    df['Month_Name'].unique(),
    key=lambda x: month_order.index(x) if x in month_order else 99
)


def create_grid(label, options, num_columns, key_prefix):
    st.caption(f"**{label}** (Leave empty to select all)")
    selected = []
    cols = st.columns(num_columns)
    for i, opt in enumerate(options):
        with cols[i % num_columns]:
            if st.checkbox(str(opt), key=f"{key_prefix}_{opt}"):
                selected.append(opt)
    return selected


def default_boat_profile():
    return {
        'tank_capacity_l': 300.0,
        'vessel_length_m': 7.0,
        'fuel_burn_lph': 30.0,
        'cruise_speed_kn': 18.0,
        'vessel_type': 'Planing',
    }


def ensure_boat_profile_state():
    base = default_boat_profile()
    profile = st.session_state.get('boat_profile', {})
    if not isinstance(profile, dict):
        profile = {}
    for key, value in base.items():
        profile.setdefault(key, value)
    st.session_state['boat_profile'] = profile


def suggest_cruise_speed_kn(vessel_length_m, vessel_type):
    # Hull speed baseline in knots: 1.34 * sqrt(LWL in feet).
    length_ft = max(float(vessel_length_m), 0.1) * 3.28084
    hull_speed = 1.34 * (length_ft ** 0.5)
    type_multiplier = {
        'Displacement': 0.85,
        'Semi-planing': 1.2,
        'Planing': 1.8,
    }
    suggested = hull_speed * type_multiplier.get(vessel_type, 1.0)
    return round(max(suggested, 3.0), 1)


def render_boat_profile_editor():
    ensure_boat_profile_state()
    profile = st.session_state['boat_profile']
    current_type = profile.get('vessel_type', VESSEL_TYPES[-1])
    if current_type not in VESSEL_TYPES:
        current_type = VESSEL_TYPES[-1]
    type_index = VESSEL_TYPES.index(current_type)
    suggested = suggest_cruise_speed_kn(profile.get('vessel_length_m', 7.0), current_type)

    with st.container(border=True):
        st.markdown("### Boat Details")
        st.caption("Used to estimate travel time and fuel after you measure distance on the map ruler.")
        with st.form("boat_profile_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                tank_capacity_l = st.number_input(
                    "Fuel tank capacity (L)",
                    min_value=1.0,
                    value=float(profile.get('tank_capacity_l', 300.0)),
                    step=10.0,
                )
                vessel_length_m = st.number_input(
                    "Vessel length (m)",
                    min_value=1.0,
                    value=float(profile.get('vessel_length_m', 7.0)),
                    step=0.1,
                )
            with col_b:
                fuel_burn_lph = st.number_input(
                    "Average fuel burn (L/hour)",
                    min_value=0.1,
                    value=float(profile.get('fuel_burn_lph', 30.0)),
                    step=0.5,
                )
                vessel_type = st.selectbox("Vessel type", VESSEL_TYPES, index=type_index)

            suggested_speed = suggest_cruise_speed_kn(vessel_length_m, vessel_type)
            auto_fill_speed = st.checkbox(
                "Auto-fill cruise speed from vessel length + type",
                value=True,
                help="Uses a hull-speed-based estimate. You can turn this off and enter your own speed.",
            )
            cruise_speed_kn = st.number_input(
                "Average cruise speed (knots)",
                min_value=1.0,
                value=float(profile.get('cruise_speed_kn', suggested_speed)),
                step=0.5,
            )
            st.caption(f"Suggested cruise speed: {suggested_speed} kn")

            save_profile = st.form_submit_button("Save boat details")

        if save_profile:
            st.session_state['boat_profile'] = {
                'tank_capacity_l': float(tank_capacity_l),
                'vessel_length_m': float(vessel_length_m),
                'fuel_burn_lph': float(fuel_burn_lph),
                'cruise_speed_kn': float(suggested_speed if auto_fill_speed else cruise_speed_kn),
                'vessel_type': vessel_type,
            }
            st.success("Boat details saved.")

        summary = st.session_state['boat_profile']
        st.caption(
            "Current profile: "
            f"{summary['vessel_length_m']:.1f} m {summary['vessel_type']} vessel, "
            f"{summary['tank_capacity_l']:.0f} L tank, "
            f"{summary['fuel_burn_lph']:.1f} L/h, "
            f"{summary['cruise_speed_kn']:.1f} kn cruise"
        )


def estimate_trip(distance_nm, profile, restricted_pct, restricted_speed_kn):
    distance_nm = max(float(distance_nm), 0.0)
    cruise_speed_kn = max(float(profile.get('cruise_speed_kn', 1.0)), 0.1)
    fuel_burn_lph = max(float(profile.get('fuel_burn_lph', 0.1)), 0.1)
    tank_capacity_l = max(float(profile.get('tank_capacity_l', 1.0)), 0.1)
    restricted_speed_kn = max(float(restricted_speed_kn), 0.1)
    restricted_pct = min(max(float(restricted_pct), 0.0), 100.0)

    restricted_nm = distance_nm * (restricted_pct / 100.0)
    open_nm = max(distance_nm - restricted_nm, 0.0)
    total_hours = (open_nm / cruise_speed_kn) + (restricted_nm / restricted_speed_kn)
    average_speed_kn = (distance_nm / total_hours) if total_hours > 0 else 0.0
    fuel_used_l = total_hours * fuel_burn_lph
    fuel_remaining_l = tank_capacity_l - fuel_used_l
    max_range_nm = (tank_capacity_l / fuel_burn_lph) * cruise_speed_kn

    return {
        'hours': total_hours,
        'average_speed_kn': average_speed_kn,
        'fuel_used_l': fuel_used_l,
        'fuel_remaining_l': fuel_remaining_l,
        'max_range_nm': max_range_nm,
    }


def render_trip_estimator(panel_key):
    ensure_boat_profile_state()
    route_key = ensure_route_state(panel_key)
    profile = st.session_state['boat_profile']
    distance_state_key = f"{panel_key}_distance_nm"
    distance_default = float(st.session_state.get(distance_state_key, 0.0))
    route_segments = st.session_state.get(route_key, [])

    with st.container(border=True):
        st.markdown("### Trip Estimator")
        st.caption("Draw route segments on the map. Tag any segment as restricted below.")
        if distance_default > 0:
            st.caption(f"Map measurement detected: {distance_default:.1f} nautical miles")
        distance_nm = st.number_input(
            "Measured distance (nautical miles)",
            min_value=0.0,
            value=distance_default,
            step=0.5,
            key=f"{panel_key}_distance_nm",
        )

        restricted_speed_kn = st.number_input(
            "Restricted speed (knots)",
            min_value=1.0,
            value=DEFAULT_SYDNEY_SPEED_LIMIT_KN,
            step=0.5,
            key=f"{panel_key}_restricted_speed",
        )

        if route_segments:
            st.markdown("#### Route segments")
            updated_segments = []
            for idx, segment in enumerate(route_segments, start=1):
                left_col, right_col = st.columns([3, 1])
                with left_col:
                    st.caption(f"Segment {idx}: {segment.get('label', 'Line')} - {segment['distance_nm']:.1f} nm")
                with right_col:
                    restricted = st.checkbox(
                        "Restricted",
                        value=bool(segment.get('restricted', False)),
                        key=f"{panel_key}_restricted_{segment['key']}",
                    )
                updated_segments.append({**segment, 'restricted': restricted})

            if st.button("Update segment types", key=f"{panel_key}_update_segments"):
                st.session_state[route_key] = updated_segments
                st.rerun()

            route_segments = updated_segments

            route_name = st.text_input(
                "Route name for export",
                value=st.session_state.get(f"{panel_key}_route_name", "Fish Route"),
                key=f"{panel_key}_route_name_input",
            )
            st.session_state[f"{panel_key}_route_name"] = route_name

            gpx_data = export_route_as_gpx(route_segments, route_name=route_name)
            kml_data = export_route_as_kml(route_segments, route_name=route_name)
            export_col_1, export_col_2 = st.columns(2)
            with export_col_1:
                if gpx_data:
                    st.download_button(
                        "Download GPX for Garmin",
                        data=gpx_data,
                        file_name=f"{route_name.replace(' ', '_').lower()}.gpx",
                        mime="application/gpx+xml",
                        key=f"{panel_key}_download_gpx",
                    )
            with export_col_2:
                if kml_data:
                    st.download_button(
                        "Download KML",
                        data=kml_data,
                        file_name=f"{route_name.replace(' ', '_').lower()}.kml",
                        mime="application/vnd.google-earth.kml+xml",
                        key=f"{panel_key}_download_kml",
                    )

        if route_segments:
            metrics = estimate_trip_from_segments(route_segments, profile, restricted_speed_kn)
            total_minutes = int(round(metrics['hours'] * 60))
            hours_component = total_minutes // 60
            minutes_component = total_minutes % 60
            st.caption(
                f"Estimated travel time: {metrics['hours']:.2f} h "
                f"(~{hours_component}h {minutes_component}m)"
            )
            st.caption(f"Effective average speed: {metrics['average_speed_kn']:.1f} kn")
            st.caption(f"Estimated fuel use: {metrics['fuel_used_l']:.1f} L")
            st.caption(f"Approx max range at cruise: {metrics['max_range_nm']:.1f} nm")
            st.caption(
                f"Route total: {metrics['total_nm']:.1f} nm | restricted: {metrics['restricted_nm']:.1f} nm | open water: {metrics['open_nm']:.1f} nm"
            )
            if metrics['fuel_remaining_l'] < 0:
                st.error(
                    f"Fuel shortfall: {abs(metrics['fuel_remaining_l']):.1f} L. "
                    "Reduce distance or refuel plan."
                )
            else:
                st.success(f"Estimated fuel remaining after trip: {metrics['fuel_remaining_l']:.1f} L")
        elif distance_nm > 0:
            metrics = estimate_trip(distance_nm, profile, 0, restricted_speed_kn)
            total_minutes = int(round(metrics['hours'] * 60))
            hours_component = total_minutes // 60
            minutes_component = total_minutes % 60
            st.caption(
                f"Estimated travel time: {metrics['hours']:.2f} h "
                f"(~{hours_component}h {minutes_component}m)"
            )
            st.caption(f"Effective average speed: {metrics['average_speed_kn']:.1f} kn")
            st.caption(f"Estimated fuel use: {metrics['fuel_used_l']:.1f} L")
            st.caption(f"Approx max range at cruise: {metrics['max_range_nm']:.1f} nm")
            if metrics['fuel_remaining_l'] < 0:
                st.error(
                    f"Fuel shortfall: {abs(metrics['fuel_remaining_l']):.1f} L. "
                    "Reduce distance or refuel plan."
                )
            else:
                st.success(f"Estimated fuel remaining after trip: {metrics['fuel_remaining_l']:.1f} L")
        else:
            st.info("Enter a measured distance or draw route segments to compute travel time and fuel.")


def haversine_nm(lat1, lon1, lat2, lon2):
    radius_nm = 3440.065
    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_nm * math.asin(min(1.0, math.sqrt(a)))


def route_distance_nm_from_drawings(drawings):
    if not drawings:
        return None

    for feature in reversed(drawings):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get('geometry', {})
        if geometry.get('type') != 'LineString':
            continue
        coordinates = geometry.get('coordinates', [])
        if len(coordinates) < 2:
            continue
        total_distance = 0.0
        for idx in range(len(coordinates) - 1):
            lon1, lat1 = coordinates[idx]
            lon2, lat2 = coordinates[idx + 1]
            total_distance += haversine_nm(lat1, lon1, lat2, lon2)
        return round(total_distance, 1)

    return None


def route_segments_from_drawings(drawings):
    segments = []
    if not drawings:
        return segments

    for feature in drawings:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get('geometry', {})
        if geometry.get('type') != 'LineString':
            continue
        coordinates = geometry.get('coordinates', [])
        if len(coordinates) < 2:
            continue

        for idx in range(len(coordinates) - 1):
            lon1, lat1 = coordinates[idx]
            lon2, lat2 = coordinates[idx + 1]
            distance_nm = haversine_nm(lat1, lon1, lat2, lon2)
            segment_payload = {
                'start': [lon1, lat1],
                'end': [lon2, lat2],
                'edge_index': idx,
            }
            segment_key = hashlib.md5(json.dumps(segment_payload, sort_keys=True).encode('utf-8')).hexdigest()[:12]
            segments.append({
                'key': segment_key,
                'distance_nm': round(distance_nm, 1),
                'restricted': bool(feature.get('properties', {}).get('restricted', False)),
                'coordinates': [segment_payload['start'], segment_payload['end']],
                'label': f"Point {idx + 1} -> {idx + 2}",
            })

    return segments


def ensure_route_state(panel_key):
    route_key = f'{panel_key}_route_segments'
    if route_key not in st.session_state:
        st.session_state[route_key] = []
    return route_key


def update_route_segments_from_map(panel_key, drawings):
    route_key = ensure_route_state(panel_key)
    parsed_segments = route_segments_from_drawings(drawings)
    existing = st.session_state.get(route_key, [])

    existing_by_key = {segment.get('key'): segment for segment in existing if isinstance(segment, dict)}
    updated_segments = []
    for segment in parsed_segments:
        previous = existing_by_key.get(segment['key'], {})
        segment['restricted'] = bool(previous.get('restricted', segment['restricted']))
        updated_segments.append(segment)

    st.session_state[route_key] = updated_segments


def estimate_trip_from_segments(segments, profile, restricted_speed_kn):
    cruise_speed_kn = max(float(profile.get('cruise_speed_kn', 1.0)), 0.1)
    fuel_burn_lph = max(float(profile.get('fuel_burn_lph', 0.1)), 0.1)
    tank_capacity_l = max(float(profile.get('tank_capacity_l', 1.0)), 0.1)
    restricted_speed_kn = max(float(restricted_speed_kn), 0.1)

    total_nm = 0.0
    total_hours = 0.0
    restricted_nm = 0.0

    for segment in segments:
        distance_nm = float(segment.get('distance_nm', 0.0))
        if distance_nm <= 0:
            continue
        total_nm += distance_nm
        if segment.get('restricted', False):
            restricted_nm += distance_nm
            total_hours += distance_nm / restricted_speed_kn
        else:
            total_hours += distance_nm / cruise_speed_kn

    average_speed_kn = (total_nm / total_hours) if total_hours > 0 else 0.0
    fuel_used_l = total_hours * fuel_burn_lph
    fuel_remaining_l = tank_capacity_l - fuel_used_l
    max_range_nm = (tank_capacity_l / fuel_burn_lph) * cruise_speed_kn

    return {
        'total_nm': total_nm,
        'restricted_nm': restricted_nm,
        'open_nm': max(total_nm - restricted_nm, 0.0),
        'hours': total_hours,
        'average_speed_kn': average_speed_kn,
        'fuel_used_l': fuel_used_l,
        'fuel_remaining_l': fuel_remaining_l,
        'max_range_nm': max_range_nm,
    }


def build_route_points_from_segments(segments):
    points = []
    for segment in segments:
        coordinates = segment.get('coordinates', [])
        if len(coordinates) != 2:
            continue
        start, end = coordinates
        if not points:
            points.append(start)
        if points[-1] != start:
            points.append(start)
        points.append(end)
    return points


def export_route_as_gpx(segments, route_name='Fish Route'):
    points = build_route_points_from_segments(segments)
    if len(points) < 2:
        return None

    track_name = route_name.replace('&', '&amp;')
    gpx = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="Fisheries Heatmap" xmlns="http://www.topografix.com/GPX/1/1">',
        f'  <trk><name>{track_name}</name><trkseg>',
    ]
    for lon, lat in points:
        gpx.append(f'    <trkpt lat="{lat:.8f}" lon="{lon:.8f}" />')
    gpx.extend(['  </trkseg></trk>', '</gpx>'])
    return '\n'.join(gpx)


def export_route_as_kml(segments, route_name='Fish Route'):
    points = build_route_points_from_segments(segments)
    if len(points) < 2:
        return None

    coord_text = ' '.join(f'{lon:.8f},{lat:.8f},0' for lon, lat in points)
    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{route_name}</name>
    <Placemark>
      <name>{route_name}</name>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{coord_text}</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
'''
    return kml


def default_user_profile():
    return {
        'feedback': [],
        'request_stats': {},
        'filter_stats': {
            'species': {},
            'years': {},
            'months': {},
        },
        'favorite_requests': [],
    }


def load_user_profile():
    try:
        if os.path.exists(PROFILE_FILE):
            with open(PROFILE_FILE, 'r', encoding='utf-8') as profile_file:
                profile = json.load(profile_file)
                if isinstance(profile, dict):
                    base_profile = default_user_profile()
                    base_profile.update(profile)
                    base_profile.setdefault('filter_stats', default_user_profile()['filter_stats'])
                    base_profile.setdefault('feedback', [])
                    base_profile.setdefault('request_stats', {})
                    base_profile.setdefault('favorite_requests', [])
                    return base_profile
    except Exception:
        pass
    return default_user_profile()


def save_user_profile(profile):
    def to_json_safe(value):
        if isinstance(value, dict):
            return {str(key): to_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [to_json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    with open(PROFILE_FILE, 'w', encoding='utf-8') as profile_file:
        json.dump(to_json_safe(profile), profile_file, indent=2, ensure_ascii=True)


def canonical_filter_signature(filters):
    species = '|'.join(sorted(str(item) for item in filters.get('species', []))) or 'all_species'
    years = '|'.join(str(item) for item in sorted(filters.get('years', []))) or 'all_years'
    months = '|'.join(filters.get('months', [])) or 'all_months'
    return f'{species}::{years}::{months}'


def increment_counter(container, values, positive=True):
    for value in values:
        key = str(value)
        if key not in container:
            container[key] = {'asked': 0, 'positive': 0, 'negative': 0}
        container[key]['asked'] += 1
        if positive:
            container[key]['positive'] += 1
        else:
            container[key]['negative'] += 1


def summarize_top_items(counter, limit=3):
    ranked = sorted(
        counter.items(),
        key=lambda item: (
            item[1].get('positive', 0) - item[1].get('negative', 0),
            item[1].get('positive', 0),
            item[1].get('asked', 0),
        ),
        reverse=True,
    )
    return [name for name, stats in ranked[:limit] if stats.get('positive', 0) > 0 or stats.get('asked', 0) > 0]


def build_user_memory_context(profile):
    top_requests = sorted(
        profile.get('request_stats', {}).items(),
        key=lambda item: (
            item[1].get('positive', 0) - item[1].get('negative', 0),
            item[1].get('asked', 0),
        ),
        reverse=True,
    )[:3]
    favorite_requests = [request for request, stats in top_requests if stats.get('positive', 0) > 0]
    favorite_species = summarize_top_items(profile.get('filter_stats', {}).get('species', {}), limit=3)
    favorite_years = summarize_top_items(profile.get('filter_stats', {}).get('years', {}), limit=3)
    favorite_months = summarize_top_items(profile.get('filter_stats', {}).get('months', {}), limit=4)

    context_lines = []
    if favorite_requests:
        context_lines.append(f"- Frequent high-rated map requests: {', '.join(favorite_requests)}")
    if favorite_species:
        context_lines.append(f"- Frequently preferred species: {', '.join(favorite_species)}")
    if favorite_years:
        context_lines.append(f"- Frequently preferred years: {', '.join(favorite_years)}")
    if favorite_months:
        context_lines.append(f"- Frequently preferred months: {', '.join(favorite_months)}")

    if not context_lines:
        return "User preference memory: none yet."

    return "User preference memory:\n" + "\n".join(context_lines)


def get_favorite_request_suggestions(profile):
    top_requests = sorted(
        profile.get('request_stats', {}).items(),
        key=lambda item: (
            item[1].get('positive', 0) - item[1].get('negative', 0),
            item[1].get('asked', 0),
        ),
        reverse=True,
    )
    suggestions = []
    for request_text, stats in top_requests:
        if stats.get('positive', 0) > 0:
            suggestions.append(request_text)
        if len(suggestions) == 3:
            break
    return suggestions


def record_map_feedback(profile, request_text, filters, rating_value, note_text=''):
    request_key = request_text.strip() or 'manual selection'
    signature = canonical_filter_signature(filters)
    feedback_entry = {
        'request': request_key,
        'signature': signature,
        'filters': filters,
        'rating': rating_value,
        'note': note_text.strip(),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    profile.setdefault('feedback', []).append(feedback_entry)

    request_stats = profile.setdefault('request_stats', {})
    if request_key not in request_stats:
        request_stats[request_key] = {
            'asked': 0,
            'positive': 0,
            'negative': 0,
            'quality_total': 0.0,
            'quality_count': 0,
            'last_signature': signature,
        }
    request_stats[request_key]['asked'] += 1
    request_stats[request_key]['last_signature'] = signature
    if rating_value > 0:
        request_stats[request_key]['positive'] += 1
    else:
        request_stats[request_key]['negative'] += 1
    if 'overall_score' in feedback_entry:
        request_stats[request_key]['quality_total'] += feedback_entry['overall_score']
        request_stats[request_key]['quality_count'] += 1

    filter_stats = profile.setdefault('filter_stats', {
        'species': {},
        'years': {},
        'months': {},
    })
    increment_counter(filter_stats['species'], filters.get('species', []), positive=rating_value > 0)
    increment_counter(filter_stats['years'], filters.get('years', []), positive=rating_value > 0)
    increment_counter(filter_stats['months'], filters.get('months', []), positive=rating_value > 0)

    favorite_requests = []
    ranked_requests = sorted(
        request_stats.items(),
        key=lambda item: (
            item[1].get('positive', 0) - item[1].get('negative', 0),
            item[1].get('positive', 0),
            item[1].get('asked', 0),
        ),
        reverse=True,
    )
    for request_name, stats in ranked_requests:
        if stats.get('positive', 0) > 0:
            favorite_requests.append({
                'request': request_name,
                'positive': stats.get('positive', 0),
                'negative': stats.get('negative', 0),
                'asked': stats.get('asked', 0),
            })
        if len(favorite_requests) == 5:
            break
    profile['favorite_requests'] = favorite_requests
    save_user_profile(profile)


def get_average_feedback_score(profile):
    request_stats = profile.get('request_stats', {})
    weighted_total = 0.0
    weighted_count = 0
    for stats in request_stats.values():
        weighted_total += stats.get('quality_total', 0.0)
        weighted_count += stats.get('quality_count', 0)
    if weighted_count == 0:
        return None
    return round(weighted_total / weighted_count, 2)


user_profile = load_user_profile()


@st.cache_data(show_spinner=False)
def load_species_neural_network():
    if not os.path.exists(NN_MODEL_FILE):
        return {
            'status': 'missing',
            'source': 'missing',
            'message': (
                'No trained neural-network model was found. Run '
                'BasicProduction/train_species_nn.py in VS Code or the terminal to build it.'
            ),
        }

    try:
        cached_model = joblib.load(NN_MODEL_FILE)
    except Exception as exc:
        return {
            'status': 'error',
            'source': 'error',
            'message': f'Could not load the saved neural-network model: {exc}',
        }

    if not isinstance(cached_model, dict) or cached_model.get('model') is None:
        return {
            'status': 'error',
            'source': 'error',
            'message': 'The saved neural-network model file is not in the expected format.',
        }

    cached_model['source'] = 'loaded'
    return cached_model


def plot_network_architecture():
    """Visualize the neural network architecture."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Neural Network Architecture', ha='center', fontsize=14, fontweight='bold')
    
    # Input layer
    ax.text(1, 8.5, 'Input Layer', ha='center', fontsize=10, fontweight='bold')
    input_features = ['Latitude', 'Longitude', 'Year', 'Month']
    for i, feature in enumerate(input_features):
        y_pos = 7.5 - i * 0.6
        circle = plt.Circle((1, y_pos), 0.25, color='lightblue', ec='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(1.6, y_pos, feature, fontsize=8, va='center')
    
    # Hidden layer 1
    ax.text(4, 8.5, 'Hidden 1\n(64 neurons)', ha='center', fontsize=9, fontweight='bold')
    for i in range(5):
        y_pos = 7.8 - i * 0.5
        circle = plt.Circle((4, y_pos), 0.2, color='lightgreen', ec='black', linewidth=1)
        ax.add_patch(circle)
    ax.text(4, 4.8, '...', ha='center', fontsize=10, fontweight='bold')
    
    # Hidden layer 2
    ax.text(7, 8.5, 'Hidden 2\n(32 neurons)', ha='center', fontsize=9, fontweight='bold')
    for i in range(3):
        y_pos = 7.8 - i * 0.5
        circle = plt.Circle((7, y_pos), 0.2, color='lightyellow', ec='black', linewidth=1)
        ax.add_patch(circle)
    ax.text(7, 5.3, '...', ha='center', fontsize=10, fontweight='bold')
    
    # Output layer
    ax.text(9.5, 8.5, 'Output\n(Species)', ha='center', fontsize=9, fontweight='bold')
    output_neurons = 3
    for i in range(output_neurons):
        y_pos = 7.5 - i * 0.8
        circle = plt.Circle((9.5, y_pos), 0.25, color='lightcoral', ec='black', linewidth=1.5)
        ax.add_patch(circle)
    ax.text(9.5, 5.5, '...', ha='center', fontsize=10, fontweight='bold')
    
    # Info box
    info_text = 'Activation: ReLU\nOptimizer: Adam\nLoss: Cross-entropy'
    ax.text(5, 1.5, info_text, ha='center', fontsize=8, bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def compute_feature_importance(model, X_train, y_train):
    """Compute feature importance using permutation importance."""
    try:
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X_train, y_train, n_repeats=10, random_state=NN_RANDOM_STATE, n_jobs=-1
        )
        return result.importances_mean
    except Exception as e:
        st.warning(f"Could not compute feature importance: {e}")
        return None


def plot_feature_importance(importances, feature_names):
    """Plot feature importance as a bar chart."""
    if importances is None or len(importances) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
    ax.barh(sorted_features, sorted_importances, color=colors)
    ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax.set_title('Feature Importance (Permutation-based)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    for i, v in enumerate(sorted_importances):
        ax.text(v, i, f' {v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_data_distributions(data_frame, species_col):
    """Plot species distribution across features."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Species Distribution Across Features', fontsize=13, fontweight='bold')
    
    # Latitude distribution
    ax = axes[0, 0]
    for species in data_frame[species_col].unique()[:5]:  # Top 5 species
        species_data = data_frame[data_frame[species_col] == species]['Latitude']
        ax.hist(species_data, alpha=0.5, label=species, bins=20)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Count')
    ax.set_title('Latitude Distribution by Species')
    ax.legend(fontsize=8, loc='best')
    
    # Longitude distribution
    ax = axes[0, 1]
    for species in data_frame[species_col].unique()[:5]:
        species_data = data_frame[data_frame[species_col] == species]['Longitude']
        ax.hist(species_data, alpha=0.5, label=species, bins=20)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Count')
    ax.set_title('Longitude Distribution by Species')
    ax.legend(fontsize=8, loc='best')
    
    # Year distribution
    ax = axes[1, 0]
    year_species_counts = data_frame.groupby(['Year', species_col]).size().unstack(fill_value=0)
    year_species_counts.plot(ax=ax, marker='o', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title('Species Catches Over Years')
    ax.legend(fontsize=8, loc='best')
    
    # Month distribution
    ax = axes[1, 1]
    month_species_counts = data_frame.groupby(['Month_Number', species_col]).size().unstack(fill_value=0)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_species_counts.plot(ax=ax, marker='s', alpha=0.7)
    ax.set_xlabel('Month')
    ax.set_ylabel('Count')
    ax.set_title('Species Catches by Month')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, rotation=45, fontsize=8)
    ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    return fig


def plot_loss_curve_enhanced(loss_curve):
    """Plot enhanced training loss curve."""
    if not loss_curve or len(loss_curve) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(loss_curve) + 1)
    ax.plot(epochs, loss_curve, marker='o', linewidth=2, markersize=4, color='#2E86AB')
    ax.fill_between(epochs, loss_curve, alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    ax.set_title('Neural Network Training Loss Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add final loss annotation
    final_loss = loss_curve[-1]
    ax.annotate(f'Final: {final_loss:.4f}', 
                xy=(len(loss_curve), final_loss),
                xytext=(-40, -20), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(confusion_matrix_values, class_labels):
    matrix = np.asarray(confusion_matrix_values)
    fig, ax = plt.subplots(figsize=(max(7, len(class_labels) * 0.75), max(5, len(class_labels) * 0.6)))
    image = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_yticklabels(class_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Neural network confusion matrix')

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                int(matrix[row_index, col_index]),
                ha='center',
                va='center',
                color='black',
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def build_nn_ollama_prompt(model_info):
    return f"""You are explaining a small fisheries neural network to a student.

Model details:
- Task: predict fish species from latitude, longitude, year, and month
- Training rows: {model_info['train_rows']}
- Test rows: {model_info['test_rows']}
- Accuracy: {model_info['accuracy']:.3f}
- Macro F1: {model_info['macro_f1']:.3f}
- Weighted F1: {model_info['weighted_f1']:.3f}
- Classes: {', '.join(model_info['classes'])}

Keep the answer short, practical, and honest.
Explain what the model is useful for, what its limits are, and whether the metrics look good enough for a demo or light analysis.
"""


def run_nn_ollama_explanation(model_info):
    if not OLLAMA_AVAILABLE or not init_ollama():
        return 'Ollama is not available right now, so the app is showing the neural network metrics directly.'

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=build_nn_ollama_prompt(model_info),
            stream=False,
        )
        return response.get('response', '').strip() or 'Ollama returned an empty explanation.'
    except Exception as exc:
        return f'Ollama explanation failed: {exc}'


def get_model_feature_columns(model_info):
    configured = model_info.get('feature_columns')
    if isinstance(configured, list) and configured and all(isinstance(item, str) for item in configured):
        return configured
    return list(NN_FEATURE_COLUMNS)


def build_prediction_features_from_row(row, feature_columns):
    payload = {}
    for feature_name in feature_columns:
        value = row.get(feature_name, np.nan)
        numeric_value = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
        if pd.isna(numeric_value):
            payload[feature_name] = np.nan
        elif feature_name in ['Year', 'Month_Number']:
            payload[feature_name] = int(numeric_value)
        else:
            payload[feature_name] = float(numeric_value)
    return pd.DataFrame([payload], columns=feature_columns)


def build_context_prediction_features(data_frame, feature_columns):
    if data_frame is None or data_frame.empty:
        return None

    working_frame = data_frame.copy()
    for feature_name in feature_columns:
        if feature_name not in working_frame.columns:
            working_frame[feature_name] = np.nan

    required_context_features = [LAT_COL, LON_COL, 'Year', 'Month_Number']
    if any(feature_name not in working_frame.columns for feature_name in required_context_features):
        return None
    if working_frame.dropna(subset=required_context_features).empty:
        return None

    payload = {}
    for feature_name in feature_columns:
        column = pd.to_numeric(working_frame[feature_name], errors='coerce')
        if feature_name in ['Year', 'Month_Number']:
            modes = column.dropna().mode()
            payload[feature_name] = int(modes.iloc[0]) if not modes.empty else np.nan
        else:
            payload[feature_name] = float(column.mean()) if column.notna().any() else np.nan

    return pd.DataFrame([payload], columns=feature_columns)


def month_name_to_number(month_name):
    if month_name in month_order:
        return month_order.index(month_name) + 1
    return None


def cyclical_to_degrees(sin_value, cos_value):
    if pd.isna(sin_value) or pd.isna(cos_value):
        return 0.0
    radians = np.arctan2(float(sin_value), float(cos_value))
    return (np.degrees(radians) + 360.0) % 360.0


def cyclical_to_fraction(sin_value, cos_value):
    if pd.isna(sin_value) or pd.isna(cos_value):
        return 0.5
    radians = np.arctan2(float(sin_value), float(cos_value))
    return ((radians / (2 * np.pi)) + 1.0) % 1.0


def build_probability_frame(model_info, feature_frame):
    probabilities = model_info['model'].predict_proba(feature_frame)[0]
    return pd.DataFrame({
        'Species': model_info['classes'],
        'Probability': probabilities,
    }).sort_values('Probability', ascending=False)


def build_adjusted_what_if_features(
    baseline_features,
    model_feature_columns,
    selected_month_name,
    selected_wind_direction,
    selected_rain_mm,
    selected_is_raining,
    selected_moon_phase,
    selected_latitude,
    selected_longitude,
):
    adjusted = baseline_features.copy()
    month_number = month_name_to_number(selected_month_name)

    for feature_name in model_feature_columns:
        if feature_name == LAT_COL:
            adjusted.at[0, LAT_COL] = float(selected_latitude)
        elif feature_name == LON_COL:
            adjusted.at[0, LON_COL] = float(selected_longitude)
        elif feature_name == 'Month_Number' and month_number is not None:
            adjusted.at[0, 'Month_Number'] = int(month_number)
        elif feature_name == 'Wind_Direction_Sin':
            adjusted.at[0, 'Wind_Direction_Sin'] = float(np.sin(np.deg2rad(selected_wind_direction)))
        elif feature_name == 'Wind_Direction_Cos':
            adjusted.at[0, 'Wind_Direction_Cos'] = float(np.cos(np.deg2rad(selected_wind_direction)))
        elif feature_name == 'Rain_mm':
            adjusted.at[0, 'Rain_mm'] = float(selected_rain_mm)
        elif feature_name == 'Is_Raining':
            adjusted.at[0, 'Is_Raining'] = 1.0 if selected_is_raining else 0.0
        elif feature_name == 'Moon_Phase_Sin':
            adjusted.at[0, 'Moon_Phase_Sin'] = float(np.sin(2 * np.pi * selected_moon_phase))
        elif feature_name == 'Moon_Phase_Cos':
            adjusted.at[0, 'Moon_Phase_Cos'] = float(np.cos(2 * np.pi * selected_moon_phase))

    return adjusted[model_feature_columns]


def render_neural_network_dashboard(context_df=None):
    with st.container(border=True):
        st.markdown('### Neural Network Explorer')
        st.caption('Loads a pre-trained species classifier and shows the results with simple charts.')
        model_info = load_species_neural_network()
        if model_info.get('status') != 'ok':
            st.warning(model_info.get('message', 'Neural network model could not be loaded.'))
            st.info('To train the model, run BasicProduction/train_species_nn.py in VS Code or the terminal, then reload the app.')
            return

        model_feature_columns = get_model_feature_columns(model_info)
        st.caption(f"Model source: {model_info.get('source', 'trained')}")

        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
        metric_col_1.metric('Accuracy', f"{model_info['accuracy']:.1%}")
        metric_col_2.metric('Macro F1', f"{model_info['macro_f1']:.1%}")
        metric_col_3.metric('Weighted F1', f"{model_info['weighted_f1']:.1%}")
        metric_col_4.metric('Classes', str(len(model_info['classes'])))

        st.caption(
            f"Training set: {model_info['train_rows']} rows | Test set: {model_info['test_rows']} rows | "
            f"Rare species grouped into Other: {model_info['other_count']} records"
        )

        # Create tabs for different visualizations
        viz_tabs = st.tabs(['Training Progress', 'Network Architecture', 'Feature Importance', 'Data Insights', 'Performance Details'])
        
        # Tab 1: Training Progress
        with viz_tabs[0]:
            st.markdown('#### Training Loss Over Time')
            if model_info['loss_curve']:
                loss_fig = plot_loss_curve_enhanced(model_info['loss_curve'])
                st.pyplot(loss_fig, clear_figure=True)
            else:
                st.info('No loss curve data available.')
        
        # Tab 2: Network Architecture
        with viz_tabs[1]:
            st.markdown('#### Neural Network Structure')
            st.info(f"This neural network uses {len(model_feature_columns)} input features (location, time, moon phase, wind direction, and rainfall), two hidden layers with 64 and 32 neurons respectively, and outputs one probability per species class.")
            arch_fig = plot_network_architecture()
            st.pyplot(arch_fig, clear_figure=True)
        
        # Tab 3: Feature Importance
        with viz_tabs[2]:
            st.markdown('#### Which Features Matter Most?')
            st.caption('Shows how important each input feature is for making predictions (higher = more important).')
            try:
                # Get training features and target
                train_data = df.copy()
                for feature_name in model_feature_columns:
                    if feature_name not in train_data.columns:
                        train_data[feature_name] = np.nan
                train_data = train_data[model_feature_columns + [SPECIES_COL]].dropna(subset=[SPECIES_COL]).copy()
                X_importance = train_data[model_feature_columns].astype(float)
                y_importance = train_data[SPECIES_COL]
                
                if len(X_importance) > 0 and len(y_importance.unique()) > 1:
                    importances = compute_feature_importance(model_info['model'], X_importance, y_importance)
                    if importances is not None:
                        fig = plot_feature_importance(importances, model_feature_columns)
                        st.pyplot(fig, clear_figure=True)
                    else:
                        st.warning('Could not compute feature importance.')
                else:
                    st.info('Not enough data to compute feature importance.')
            except Exception as e:
                st.warning(f'Feature importance analysis unavailable: {e}')
        
        # Tab 4: Data Insights
        with viz_tabs[3]:
            st.markdown('#### Species Distribution Across Features')
            st.caption('Visualizes how different fish species are distributed geographically (latitude/longitude) and temporally (year/month).')
            try:
                viz_data = df[[LAT_COL, LON_COL, 'Year', 'Month_Number', SPECIES_COL]].dropna().copy()
                if len(viz_data) > 100:  # Only show if we have enough data
                    dist_fig = plot_data_distributions(viz_data, SPECIES_COL)
                    st.pyplot(dist_fig, clear_figure=True)
                else:
                    st.info('Not enough data points to show distributions.')
            except Exception as e:
                st.warning(f'Data distribution visualization unavailable: {e}')
        
        # Tab 5: Performance Details
        with viz_tabs[4]:
            st.markdown('#### Confusion Matrix & Per-Class Metrics')
            chart_col, report_col = st.columns([3, 2])
            with chart_col:
                st.pyplot(
                    plot_confusion_matrix(model_info['confusion_matrix'], model_info['classes']),
                    clear_figure=True,
                )
            with report_col:
                report_frame = pd.DataFrame(model_info['classification_report']).T
                report_view = report_frame.loc[
                    [label for label in model_info['classes'] if label in report_frame.index],
                    ['precision', 'recall', 'f1-score', 'support'],
                ]
                st.dataframe(report_view, use_container_width=True)
                st.caption('This is the easiest place to check whether the neural network is learning useful patterns.')

        sample_source = context_df if isinstance(context_df, pd.DataFrame) and not context_df.empty else df
        sample_source = sample_source.dropna(subset=[SPECIES_COL, LAT_COL, LON_COL, 'Year', 'Month_Number'])
        if sample_source.empty:
            st.warning('No sample row is available for prediction preview.')
            return

        sample_row = sample_source.sample(1, random_state=NN_RANDOM_STATE).iloc[0]
        sample_features = build_prediction_features_from_row(sample_row, model_feature_columns)
        prediction = model_info['model'].predict(sample_features)[0]
        probabilities = model_info['model'].predict_proba(sample_features)[0]
        probability_frame = pd.DataFrame({
            'Species': model_info['classes'],
            'Probability': probabilities,
        }).sort_values('Probability', ascending=False)

        prediction_col_1, prediction_col_2 = st.columns([2, 3])
        with prediction_col_1:
            st.markdown('#### Example prediction')
            st.write(f"Actual species: **{sample_row[SPECIES_COL]}**")
            st.write(f"Predicted species: **{prediction}**")
            st.caption(
                f"Example input: {sample_row[LAT_COL]:.2f}, {sample_row[LON_COL]:.2f}, "
                f"year {int(sample_row['Year'])}, month {int(sample_row['Month_Number'])}"
            )
        with prediction_col_2:
            st.bar_chart(probability_frame.head(8).set_index('Species'))

        st.caption(f"Artifact loaded from: {NN_MODEL_FILE}")

        st.markdown('#### Predict from current map context')
        st.caption('Uses the active filtered map if one exists, otherwise it falls back to the full dataset.')
        if st.button('Predict current context', key='nn_context_predict'):
            context_features = build_context_prediction_features(
                context_df if isinstance(context_df, pd.DataFrame) else df,
                model_feature_columns,
            )
            if context_features is None:
                st.warning('No usable filtered rows are available for a context prediction.')
            else:
                context_prediction = model_info['model'].predict(context_features)[0]
                context_probabilities = model_info['model'].predict_proba(context_features)[0]
                context_probability_frame = pd.DataFrame({
                    'Species': model_info['classes'],
                    'Probability': context_probabilities,
                }).sort_values('Probability', ascending=False)

                st.success(f"Predicted dominant species for the current context: {context_prediction}")
                st.bar_chart(context_probability_frame.head(8).set_index('Species'))

        explanation_key = 'nn_ollama_explanation'
        if explanation_key not in st.session_state:
            st.session_state[explanation_key] = ''

        if st.button('Ask Ollama to explain the network', key='nn_explain_button'):
            with st.spinner('Getting Ollama explanation...'):
                st.session_state[explanation_key] = run_nn_ollama_explanation(model_info)

        if st.session_state[explanation_key]:
            st.markdown('#### Ollama explanation')
            st.write(st.session_state[explanation_key])


def render_what_if_analysis_panel(context_df=None):
    with st.container(border=True):
        st.markdown('### What-If Analysis (Neural Network)')
        st.caption(
            'Adjust context inputs and compare baseline vs adjusted species probabilities. '
            'This demonstrates what-if analysis for assessment evidence.'
        )

        model_info = load_species_neural_network()
        if model_info.get('status') != 'ok':
            st.warning(model_info.get('message', 'Neural network model could not be loaded.'))
            return

        model_feature_columns = get_model_feature_columns(model_info)
        context_source = context_df if isinstance(context_df, pd.DataFrame) and not context_df.empty else df
        baseline_features = build_context_prediction_features(context_source, model_feature_columns)

        if baseline_features is None:
            st.warning('No usable baseline context is available for what-if analysis.')
            return

        baseline_row = baseline_features.iloc[0]
        baseline_month_number = int(pd.to_numeric(pd.Series([baseline_row.get('Month_Number', 1)]), errors='coerce').iloc[0] or 1)
        baseline_month_number = min(max(baseline_month_number, 1), 12)
        baseline_month_name = month_order[baseline_month_number - 1]

        baseline_wind_direction = cyclical_to_degrees(
            baseline_row.get('Wind_Direction_Sin', np.nan),
            baseline_row.get('Wind_Direction_Cos', np.nan),
        )
        baseline_moon_phase = cyclical_to_fraction(
            baseline_row.get('Moon_Phase_Sin', np.nan),
            baseline_row.get('Moon_Phase_Cos', np.nan),
        )
        baseline_rain_mm = float(pd.to_numeric(pd.Series([baseline_row.get('Rain_mm', 0.0)]), errors='coerce').fillna(0.0).iloc[0])
        baseline_is_raining = bool(pd.to_numeric(pd.Series([baseline_row.get('Is_Raining', 0.0)]), errors='coerce').fillna(0.0).iloc[0] >= 0.5)
        baseline_latitude = float(pd.to_numeric(pd.Series([baseline_row.get(LAT_COL, 0.0)]), errors='coerce').fillna(0.0).iloc[0])
        baseline_longitude = float(pd.to_numeric(pd.Series([baseline_row.get(LON_COL, 0.0)]), errors='coerce').fillna(0.0).iloc[0])

        st.caption(
            f'Baseline context: month {baseline_month_name}, wind {baseline_wind_direction:.0f}°, '
            f'rain {baseline_rain_mm:.1f} mm, raining={baseline_is_raining}, '
            f'moon phase {baseline_moon_phase:.2f}, lat/lon ({baseline_latitude:.2f}, {baseline_longitude:.2f}).'
        )

        with st.form('what_if_analysis_form'):
            control_col_1, control_col_2, control_col_3 = st.columns(3)
            with control_col_1:
                what_if_month = st.selectbox(
                    'Month',
                    options=month_order,
                    index=month_order.index(baseline_month_name),
                    key='what_if_month',
                )
                what_if_wind = st.slider(
                    'Wind direction (degrees)',
                    min_value=0,
                    max_value=359,
                    value=int(round(baseline_wind_direction)),
                    key='what_if_wind',
                )
            with control_col_2:
                what_if_rain = st.number_input(
                    'Rain amount (mm)',
                    min_value=0.0,
                    value=float(round(baseline_rain_mm, 2)),
                    step=0.1,
                    key='what_if_rain',
                )
                what_if_is_raining = st.checkbox(
                    'Is raining',
                    value=baseline_is_raining,
                    key='what_if_is_raining',
                )
            with control_col_3:
                what_if_moon_phase = st.slider(
                    'Moon phase (0-1)',
                    min_value=0.0,
                    max_value=1.0,
                    value=float(round(baseline_moon_phase, 2)),
                    step=0.01,
                    key='what_if_moon_phase',
                )
                what_if_latitude = st.number_input(
                    'Latitude',
                    value=float(round(baseline_latitude, 5)),
                    format='%.5f',
                    key='what_if_latitude',
                )
                what_if_longitude = st.number_input(
                    'Longitude',
                    value=float(round(baseline_longitude, 5)),
                    format='%.5f',
                    key='what_if_longitude',
                )

            compare_submitted = st.form_submit_button('Run what-if comparison')

        if compare_submitted:
            # Rebuild the baseline through the same rounding/reconstruction path as the
            # adjusted inputs (rounded degrees/fraction -> sin/cos), rather than using the
            # raw averaged sin/cos vector directly. Without this, leaving every control at
            # its displayed default would still produce a different result to "baseline",
            # since averaging sin and cos across many rows does not reproduce a clean
            # unit-circle value the way a single rounded angle does.
            reconstructed_baseline_features = build_adjusted_what_if_features(
                baseline_features,
                model_feature_columns,
                baseline_month_name,
                int(round(baseline_wind_direction)),
                round(baseline_rain_mm, 2),
                baseline_is_raining,
                round(baseline_moon_phase, 2),
                round(baseline_latitude, 5),
                round(baseline_longitude, 5),
            )
            adjusted_features = build_adjusted_what_if_features(
                baseline_features,
                model_feature_columns,
                what_if_month,
                what_if_wind,
                what_if_rain,
                what_if_is_raining,
                what_if_moon_phase,
                what_if_latitude,
                what_if_longitude,
            )
            baseline_probability_frame = build_probability_frame(model_info, reconstructed_baseline_features)
            adjusted_probability_frame = build_probability_frame(model_info, adjusted_features)

            baseline_top = baseline_probability_frame.iloc[0]
            adjusted_top = adjusted_probability_frame.iloc[0]
            comparison_frame = (
                baseline_probability_frame.rename(columns={'Probability': 'Baseline_Probability'})
                .merge(
                    adjusted_probability_frame.rename(columns={'Probability': 'Adjusted_Probability'}),
                    on='Species',
                    how='outer',
                )
                .fillna(0.0)
            )
            comparison_frame['Change'] = comparison_frame['Adjusted_Probability'] - comparison_frame['Baseline_Probability']
            comparison_frame = comparison_frame.sort_values('Adjusted_Probability', ascending=False)

            st.session_state['what_if_result'] = {
                'baseline_top_species': baseline_top['Species'],
                'baseline_top_probability': float(baseline_top['Probability']),
                'adjusted_top_species': adjusted_top['Species'],
                'adjusted_top_probability': float(adjusted_top['Probability']),
                'comparison_frame': comparison_frame,
            }

        result = st.session_state.get('what_if_result')
        if not result:
            st.info('Set values above and click "Run what-if comparison" to compare predictions.')
            return

        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric(
            'Baseline top prediction',
            f"{result['baseline_top_species']}",
            f"{result['baseline_top_probability']:.1%}",
        )
        metric_col_2.metric(
            'Adjusted top prediction',
            f"{result['adjusted_top_species']}",
            f"{result['adjusted_top_probability']:.1%}",
        )

        st.markdown('#### Probability comparison')
        comparison_frame = result['comparison_frame'].copy()
        bar_view = comparison_frame.head(8).set_index('Species')[['Baseline_Probability', 'Adjusted_Probability']]
        st.bar_chart(bar_view)

        table_view = comparison_frame.head(12).copy()
        table_view['Baseline_Probability'] = (table_view['Baseline_Probability'] * 100).round(2)
        table_view['Adjusted_Probability'] = (table_view['Adjusted_Probability'] * 100).round(2)
        table_view['Change'] = (table_view['Change'] * 100).round(2)
        table_view = table_view.rename(columns={
            'Baseline_Probability': 'Baseline %',
            'Adjusted_Probability': 'Adjusted %',
            'Change': 'Change (pp)',
        })
        st.dataframe(table_view, use_container_width=True, hide_index=True)


def render_certainty_factor_panel(context_df=None):
    """Rule-based expert-system panel, separate from the neural network above.

    Implements the certainty-factor engine described in scripts/CertaintyFactors.md,
    scripts/IF-THEN.md and scripts/Flowchart.md (see certainty_factor.py for the rule
    logic). This is a second, independent intelligent system: instead of a trained
    model, it reasons with explicit IF-THEN rules and a running certainty score.
    """
    with st.container(border=True):
        header_col, toggle_col = st.columns([3, 1])
        with header_col:
            st.markdown('### Certainty-Factor Expert System')
            st.caption(
                'Rule-based recommendation engine (IF-THEN rules + certainty scoring). '
                'Complements the neural network above rather than replacing it.'
            )
        with toggle_col:
            show_cf = st.checkbox('Show', value=st.session_state.get('show_certainty_factor', True), key='show_certainty_factor')

        if not show_cf:
            st.info('Certainty-factor panel hidden. Tick "Show" to run the rule-based expert system.')
            return

        context_source = context_df if isinstance(context_df, pd.DataFrame) and not context_df.empty else df

        default_species_list = sorted(context_source[SPECIES_COL].dropna().unique()) or all_species
        default_species = default_species_list[0] if default_species_list else all_species[0]

        input_col_1, input_col_2 = st.columns(2)
        with input_col_1:
            cf_species = st.selectbox('Species to evaluate', options=all_species, index=all_species.index(default_species) if default_species in all_species else 0, key='cf_species')
        with input_col_2:
            month_options = ['(none)'] + available_months
            default_month = context_source['Month_Name'].mode().iloc[0] if 'Month_Name' in context_source.columns and not context_source['Month_Name'].mode().empty else '(none)'
            cf_month = st.selectbox('Month to evaluate', options=month_options, index=month_options.index(default_month) if default_month in month_options else 0, key='cf_month')

        # Location and weather context are auto-filled from the current map context
        # (or the whole dataset if no map filter is active), matching how the
        # neural network's "Predict from current map context" button behaves.
        numeric_context = context_source.dropna(subset=[LAT_COL, LON_COL])
        has_location_context = not numeric_context.empty

        cf_latitude = float(numeric_context[LAT_COL].mean()) if has_location_context else None
        cf_longitude = float(numeric_context[LON_COL].mean()) if has_location_context else None
        cf_wind_direction = (
            float(pd.to_numeric(numeric_context['Wind_Direction_10m'], errors='coerce').mean())
            if has_location_context and 'Wind_Direction_10m' in numeric_context.columns
            else None
        )
        cf_rain_mm = (
            float(pd.to_numeric(numeric_context['Rain_mm'], errors='coerce').mean())
            if has_location_context and 'Rain_mm' in numeric_context.columns
            else None
        )
        cf_is_raining = (
            bool(pd.to_numeric(numeric_context['Is_Raining'], errors='coerce').fillna(0).mean() >= 0.5)
            if has_location_context and 'Is_Raining' in numeric_context.columns
            else None
        )
        cf_moon_phase = (
            float(pd.to_numeric(numeric_context['Moon_Phase'], errors='coerce').mean())
            if has_location_context and 'Moon_Phase' in numeric_context.columns
            else None
        )

        if has_location_context:
            st.caption(
                f"Context auto-filled from {'the current map filter' if isinstance(context_df, pd.DataFrame) and not context_df.empty else 'the full dataset'}: "
                f"location ~({cf_latitude:.2f}, {cf_longitude:.2f}), wind ~{cf_wind_direction:.0f}°, "
                f"rain ~{cf_rain_mm:.1f}mm, raining: {cf_is_raining}, moon phase ~{cf_moon_phase:.2f}."
            )
        else:
            st.caption('No location/weather context available yet; location and catch-density rules will be skipped.')

        if st.button('Evaluate certainty factor', key='cf_evaluate_button'):
            result = evaluate_certainty_factor(
                df,
                species=cf_species,
                month_name=cf_month if cf_month != '(none)' else None,
                latitude=cf_latitude,
                longitude=cf_longitude,
                wind_direction=cf_wind_direction,
                rain_mm=cf_rain_mm,
                is_raining=cf_is_raining,
                moon_phase=cf_moon_phase,
                user_profile=user_profile,
            )
            st.session_state['cf_last_result'] = result

        result = st.session_state.get('cf_last_result')
        if not result:
            st.info('Choose a species and month above, then click "Evaluate certainty factor".')
            return

        band_icon = {'strong': '🟢', 'moderate': '🟡', 'weak': '🟠', 'insufficient': '🔴'}.get(result['band'], '')
        score_col, band_col = st.columns(2)
        score_col.metric('Certainty score', f"{result['score']} / {result['max_score']}")
        band_col.metric('Confidence band', f"{band_icon} {result['band_label']}")

        st.write(result['recommendation'])

        st.markdown('#### Rules evaluated')
        rules_frame = pd.DataFrame([
            {
                'Rule': rule['rule'].replace('_', ' ').title(),
                'Fired?': 'Yes' if rule['passed'] else 'No',
                'Points': f"+{rule['points']}" if rule['passed'] else '0',
                'Explanation': rule['description'],
            }
            for rule in result['rules_fired']
        ])
        st.dataframe(rules_frame, use_container_width=True, hide_index=True)

        st.caption(
            'Rule definitions: scripts/CertaintyFactors.md | Rule logic: scripts/IF-THEN.md | '
            'Decision flow: scripts/Flowchart.md'
        )


def normalize_text(value):
    return re.sub(r'[^a-z0-9]+', ' ', str(value).lower()).strip()


def unique_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def parse_years_from_text(request_text, years_available):
    lower_text = request_text.lower()
    if any(term in lower_text for term in ['all years', 'every year', 'any year']):
        return list(years_available)

    found_years = []
    for match in re.findall(r'\b(?:19|20)\d{2}\b', request_text):
        year_value = int(match)
        if year_value in years_available:
            found_years.append(year_value)

    return unique_preserve_order(found_years)


def parse_months_from_text(request_text, months_available):
    lower_text = normalize_text(request_text)
    if any(term in lower_text for term in ['all seasons', 'all months', 'every season', 'every month']):
        return list(months_available)

    season_months = {
        'spring': ['March', 'April', 'May'],
        'summer': ['June', 'July', 'August'],
        'autumn': ['September', 'October', 'November'],
        'fall': ['September', 'October', 'November'],
        'winter': ['December', 'January', 'February'],
    }

    selected_months = []
    for month in months_available:
        if month.lower() in lower_text:
            selected_months.append(month)

    for season, season_month_list in season_months.items():
        if season in lower_text:
            for month in season_month_list:
                if month in months_available:
                    selected_months.append(month)

    return unique_preserve_order(selected_months)


def parse_species_from_text(request_text, species_available):
    lower_text = normalize_text(request_text)
    if any(term in lower_text for term in ['all species', 'every species', 'any species']):
        return list(species_available)

    matches = []
    for species in species_available:
        normalized_species = normalize_text(species)
        if normalized_species in lower_text or lower_text in normalized_species:
            matches.append(species)

    if matches:
        return unique_preserve_order(matches)

    request_tokens = set(lower_text.split())
    for species in species_available:
        species_tokens = set(normalize_text(species).split())
        if len(species_tokens & request_tokens) >= 2:
            matches.append(species)

    return unique_preserve_order(matches)


def apply_filters_to_map(species_filter, years_filter, months_filter, request_text='Manual selection'):
    mask = (
        df[SPECIES_COL].isin(species_filter) &
        df['Year'].isin(years_filter) &
        df['Month_Name'].isin(months_filter)
    )
    filtered_df = df[mask]

    st.session_state['map_df'] = filtered_df
    st.session_state['map_active'] = True
    st.session_state['current_filters'] = {
        'species': species_filter,
        'years': years_filter,
        'months': months_filter,
    }
    st.session_state['current_map_request'] = request_text
    st.session_state['current_map_signature'] = canonical_filter_signature(st.session_state['current_filters'])

    if not filtered_df.empty:
        df_math = filtered_df.copy()
        df_math['lat_bin'] = df_math[LAT_COL].round(2)
        df_math['lon_bin'] = df_math[LON_COL].round(2)
        grid_counts = df_math.groupby(['lat_bin', 'lon_bin']).size()
        st.session_state['max_catch_count'] = int(grid_counts.max())
    else:
        st.session_state['max_catch_count'] = 0


def safe_ai_filter_parse(request_text, species_available, years_available, months_available):
    fallback = {
        'species': parse_species_from_text(request_text, species_available) or list(species_available),
        'years': parse_years_from_text(request_text, years_available) or list(years_available),
        'months': parse_months_from_text(request_text, months_available) or list(months_available),
    }

    try:
        if not OLLAMA_AVAILABLE or not init_ollama():
            return fallback

        preference_context = build_user_memory_context(user_profile)

        prompt = f"""You convert user map requests into JSON filters for a fisheries app.
Return JSON only, with exactly these keys: species, years, months.

Rules:
- species must be a list of species names chosen from the available species list or [] for all species
- years must be a list of integers or [] for all years
- months must be a list of month names or [] for all months
- If the user asks for seasons, expand them to month names
- If the user asks for all seasons or all months, return [] for months
- If nothing is specified for a field, return [] for that field

Available species: {species_available}
Available years: {years_available}
Available months: {months_available}

{preference_context}

User request: {request_text}"""

        response = ollama.generate(model=MODEL_NAME, prompt=prompt, stream=False)
        raw_response = response.get('response', '').strip()
        raw_response = raw_response.removeprefix('```json').removeprefix('```').removesuffix('```').strip()

        start_index = raw_response.find('{')
        end_index = raw_response.rfind('}')
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return fallback

        parsed = json.loads(raw_response[start_index:end_index + 1])
        parsed_species = parsed.get('species') or fallback['species']
        parsed_years = parsed.get('years') or fallback['years']
        parsed_months = parsed.get('months') or fallback['months']

        parsed_species = [s for s in species_available if s in parsed_species] or fallback['species']
        parsed_years = [int(y) for y in parsed_years if int(y) in years_available] or fallback['years']
        parsed_months = [m for m in months_available if m in parsed_months] or fallback['months']

        return {
            'species': unique_preserve_order(parsed_species),
            'years': unique_preserve_order(parsed_years),
            'months': unique_preserve_order(parsed_months),
        }
    except Exception:
        return fallback


def render_map(data_to_map, measurement_state_key=None):
    if data_to_map.empty:
        st.warning("No data found for the selected filters.")
        return

    st.caption(f"Displaying **{len(data_to_map)}** records.")

    center_lat = data_to_map[LAT_COL].mean()
    center_lon = data_to_map[LON_COL].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")

    m.add_child(MeasureControl(
        position='topright',
        primary_length_unit='nauticalmiles',
        secondary_length_unit='kilometers',
        active_color='#2563eb',
        completed_color='#38bdf8'
    ))

    Draw(
        export=False,
        position='topleft',
        draw_options={
            'polyline': True,
            'polygon': False,
            'rectangle': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
        },
        edit_options={'edit': True, 'remove': True},
    ).add_to(m)

    heat_data = data_to_map[[LAT_COL, LON_COL]].values.tolist()
    HeatMap(
        heat_data,
        radius=18,
        blur=22,
        min_opacity=0.25,
        gradient={
            0.15: '#dbeafe',
            0.35: '#93c5fd',
            0.55: '#60a5fa',
            0.75: '#f59e0b',
            1.0: '#ef4444',
        }
    ).add_to(m)

    map_result = st_folium(
        m,
        height=560,
        use_container_width=True,
        returned_objects=['all_drawings', 'last_active_drawing'],
    )

    drawings = (map_result or {}).get('all_drawings')
    update_route_segments_from_map(measurement_state_key, drawings)

    route_distance_nm = route_distance_nm_from_drawings(drawings)
    if measurement_state_key and route_distance_nm is not None:
        st.session_state[f'{measurement_state_key}_distance_nm'] = route_distance_nm

    return route_distance_nm


def apply_request_to_map(request_text, allow_ai_parse=True):
    if allow_ai_parse:
        request_filters = safe_ai_filter_parse(request_text, all_species, all_years, available_months)
    else:
        request_filters = {
            'species': parse_species_from_text(request_text, all_species) or all_species,
            'years': parse_years_from_text(request_text, all_years) or all_years,
            'months': parse_months_from_text(request_text, available_months) or available_months,
        }

    apply_filters_to_map(
        request_filters['species'],
        request_filters['years'],
        request_filters['months'],
        request_text=request_text,
    )
    st.session_state['ai_request_submitted'] = True
    st.session_state['ai_request_editor_visible'] = False
    st.session_state['ai_request_text'] = request_text
    st.session_state['ai_request_processed'] = request_text


def render_landing_page():
    st.markdown(
        '<div style="text-align:center; padding: 3rem 0 1rem 0;"><h1>Fisheries Heatmap</h1><p style="font-size:1.05rem; opacity:0.8;">Choose how you want to build your map.</p></div>',
        unsafe_allow_html=True,
    )

    render_boat_profile_editor()

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        with st.container(border=True):
            st.markdown("<div style='text-align:center; padding: 2rem 0 1rem 0;'><h3>Manual Entry</h3></div>", unsafe_allow_html=True)
            st.write("Use the original species, year, and month filters.")
            if st.button("Manual Entry", use_container_width=True, type="primary", key="landing_manual"):
                st.session_state['app_mode'] = 'manual'
                st.rerun()

    with right_col:
        with st.container(border=True):
            st.markdown("<div style='text-align:center; padding: 2rem 0 1rem 0;'><h3>AI Assisted</h3></div>", unsafe_allow_html=True)
            st.write("Describe the map you want and let the AI choose the filters.")
            if st.button("AI Assisted", use_container_width=True, type="primary", key="landing_ai"):
                st.session_state['app_mode'] = 'ai'
                st.rerun()


def render_manual_page():
    top_left, top_right = st.columns([1, 4])
    with top_left:
        if st.button("← Home", key="manual_home"):
            st.session_state['app_mode'] = 'landing'
            st.rerun()
    with top_right:
        st.markdown("### Manual Entry")

    control_left, control_right = st.columns([3, 1])

    with control_right:
        with st.container(border=True):
            selected_years = create_grid("Years", all_years, 3, "year")
        with st.container(border=True):
            selected_months = create_grid("Months", available_months, 3, "month")

        btn_generate = st.button("Generate Map", type="primary", use_container_width=True, key="manual_generate")

        max_val = st.session_state.get('max_catch_count', 0)
        with st.container(border=True):
            st.caption("**Density Legend**")
            if max_val > 0:
                st.caption(f"Red: ~{max_val} | Blue: 1")
                st.markdown(
                    '<div style="background: linear-gradient(to right, blue, lime, orange, red); height: 10px; width: 100%; border-radius: 2px;"></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("(Map not generated)")

        st.info("💡 Select filters, then generate the map.")

    with control_left:
        with st.container(border=True):
            selected_species = create_grid("Select Species", all_species, 6, "spec")

        if btn_generate:
            s_filter = selected_species if selected_species else all_species
            y_filter = selected_years if selected_years else all_years
            m_filter = selected_months if selected_months else available_months
            apply_filters_to_map(s_filter, y_filter, m_filter, request_text='Manual selection')

        if st.session_state.get('map_active'):
            data_to_map = st.session_state.get('map_df', pd.DataFrame())
            render_map(data_to_map, measurement_state_key='manual')
            render_trip_estimator('manual')
        else:
            st.info("Select filters and click Generate Map to start.")


def render_ai_page():
    top_left, top_right = st.columns([1, 4])
    with top_left:
        if st.button("← Home", key="ai_home"):
            st.session_state['app_mode'] = 'landing'
            st.rerun()
    with top_right:
        st.markdown("### AI Assisted")

    main_col, spacer_col = st.columns([4, 1])
    with main_col:
        if st.session_state.get('ai_request_editor_visible', True):
            with st.container(border=True):
                st.markdown(
                    "<div style='text-align:center; font-size:1.2rem; padding: 0.5rem 0 1rem 0;'>Type what you want and press Enter to generate the map.</div>",
                    unsafe_allow_html=True,
                )
                if 'ai_request_draft' not in st.session_state:
                    st.session_state['ai_request_draft'] = st.session_state.get('ai_request_text', '')

                with st.form("ai_request_form", clear_on_submit=False):
                    st.text_input(
                        "Describe your ideal map",
                        key="ai_request_draft",
                        placeholder="Example: show all seasons for bluefin tuna",
                        label_visibility="collapsed",
                    )
                    submitted = st.form_submit_button("Go")

                if submitted:
                    request_text = st.session_state.get('ai_request_draft', '').strip()
                    if request_text:
                        with st.spinner("Building your map..."):
                            apply_request_to_map(request_text, allow_ai_parse=True)
                        st.rerun()

                if st.session_state.get('ai_request_draft'):
                    st.caption(f"Draft: {st.session_state.get('ai_request_draft', '')}")
        else:
            current_request_text = st.session_state.get('ai_request_text', '')
            request_card_html = (
                '<div style="padding: 0.75rem 1rem; border: 1px solid rgba(128,128,128,0.25); '
                'border-radius: 12px; margin-bottom: 1rem;">'
                '<strong>Current request:</strong> '
                + current_request_text +
                '<div style="margin-top: 0.75rem;"><em>Your map is shown below. Use Edit request to change it.</em></div>'
                '</div>'
            )
            st.markdown(
                request_card_html,
                unsafe_allow_html=True,
            )
            edit_col, refresh_col = st.columns([1, 1])
            with edit_col:
                if st.button("Edit request", key="ai_edit_request"):
                    st.session_state['ai_request_draft'] = st.session_state.get('ai_request_text', '')
                    st.session_state['ai_request_editor_visible'] = True
                    st.rerun()
            with refresh_col:
                if st.button("Rebuild map", key="ai_rebuild_request"):
                    apply_request_to_map(st.session_state.get('ai_request_text', ''), allow_ai_parse=True)
                    st.rerun()

            if st.session_state.get('map_active'):
                render_map(st.session_state.get('map_df', pd.DataFrame()), measurement_state_key='ai')
                render_trip_estimator('ai')

    render_neural_network_dashboard(st.session_state.get('map_df', pd.DataFrame()))
    render_what_if_analysis_panel(st.session_state.get('map_df', pd.DataFrame()))
    render_certainty_factor_panel(st.session_state.get('map_df', pd.DataFrame()))

    with spacer_col:
        st.empty()


def render_ai_feedback_sidebar():
    with st.sidebar:
        st.header("Map Rating")
        st.caption("Rate the current AI-generated map so future suggestions improve.")

        current_request = st.session_state.get('ai_request_text', 'AI-assisted map')
        current_filters = st.session_state.get('current_filters', {
            'species': all_species,
            'years': all_years,
            'months': available_months,
        })

        default_scores = st.session_state.get('ai_feedback_scores', {
            'Relevancy': 5,
            'Clarity': 5,
            'Specificity': 5,
            'Map usefulness': 5,
            'Overall quality': 5,
        })

        with st.form("ai_feedback_form"):
            feedback_scores = {}
            for category in FEEDBACK_CATEGORIES:
                feedback_scores[category] = st.select_slider(
                    category,
                    options=[1, 2, 3, 4, 5],
                    value=int(default_scores.get(category, 5)),
                    format_func=lambda value: '★' * int(value),
                    key=f"feedback_{normalize_text(category)}",
                )

            feedback_note = st.text_input(
                "Optional note",
                placeholder="What should be better next time?",
            )
            save_feedback = st.form_submit_button("Save rating")

        if save_feedback:
            overall_score = round(sum(feedback_scores.values()) / len(feedback_scores), 2)
            st.session_state['ai_feedback_scores'] = feedback_scores
            record_map_feedback(
                user_profile,
                current_request,
                current_filters,
                1 if overall_score >= 4 else -1,
                feedback_note,
            )
            user_profile['feedback'][-1]['category_scores'] = feedback_scores
            user_profile['feedback'][-1]['overall_score'] = overall_score
            save_user_profile(user_profile)
            st.success("Saved rating.")

        average_score = get_average_feedback_score(user_profile)
        if average_score is not None:
            st.caption(f"Average learned score: {average_score}/5")

        favorite_requests = get_favorite_request_suggestions(user_profile)
        if favorite_requests:
            st.divider()
            st.caption("Favorite prompts")
            for favorite_request in favorite_requests:
                if st.button(favorite_request, key=f"favorite_request_{normalize_text(favorite_request)}"):
                    st.session_state['ai_request_text'] = favorite_request
                    st.session_state['ai_request_editor_visible'] = True
                    st.rerun()


if 'app_mode' not in st.session_state:
    st.session_state['app_mode'] = 'landing'

if 'ai_request_editor_visible' not in st.session_state:
    st.session_state['ai_request_editor_visible'] = True

ensure_boat_profile_state()

if st.session_state['app_mode'] == 'landing':
    render_landing_page()
elif st.session_state['app_mode'] == 'manual':
    render_manual_page()
elif st.session_state['app_mode'] == 'ai':
    render_ai_feedback_sidebar()
    render_ai_page()
else:
    st.session_state['app_mode'] = 'landing'
    st.rerun()
