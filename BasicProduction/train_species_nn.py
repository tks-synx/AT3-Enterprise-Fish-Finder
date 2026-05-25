import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Change Wind Speed to Wind Direction 

INPUT_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'newfishdata', 'GameFish_Releases_Master_enriched.csv'
)
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'fish_species_nn.joblib')
LAT_COL = 'Latitude'
LON_COL = 'Longitude'
SPECIES_COL = 'Species_Name'
DATE_COL = 'Release_Date'
FEATURE_COLUMNS = [
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
    'Sea_Surface_Temp_C',
]
TOP_SPECIES_LIMIT = 12
MIN_CLASS_SAMPLES = 75
TEST_SIZE = 0.2
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description='Train species classifier with enriched data.')
    parser.add_argument('--input', default=INPUT_FILE, help='Path to enriched CSV input.')
    parser.add_argument('--model-out', default=MODEL_FILE, help='Path to save the trained model.')
    parser.add_argument(
        '--max-rows',
        type=int,
        default=0,
        help='Optional cap on rows to speed up training (0 = no cap).',
    )
    return parser.parse_args()


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
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

    if 'Sea_Surface_Temp_C' in df.columns:
        df['Sea_Surface_Temp_C'] = pd.to_numeric(df['Sea_Surface_Temp_C'], errors='coerce')
    else:
        df['Sea_Surface_Temp_C'] = np.nan
    return df


def load_data(input_path: str, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors='coerce')
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors='coerce')
    df = df.dropna(subset=[LAT_COL, LON_COL])

    # Handle both Unix timestamps (new data) and string dates (old data)
    sample_value = str(df[DATE_COL].iloc[0]) if not df.empty else ''
    
    if sample_value.isdigit() and len(sample_value) >= 10:
        # Unix timestamp (milliseconds or seconds)
        df['Date_Obj'] = pd.to_datetime(df[DATE_COL], errors='coerce', unit='ms')
        if df['Date_Obj'].isna().all():
            df['Date_Obj'] = pd.to_datetime(df[DATE_COL], errors='coerce', unit='s')
    else:
        # String date format
        df['Date_Obj'] = pd.to_datetime(df[DATE_COL], errors='coerce', dayfirst=True)
    
    df = df.dropna(subset=['Date_Obj'])

    df['Year'] = df['Date_Obj'].dt.year.astype(int)
    df['Month_Number'] = df['Date_Obj'].dt.month.astype(int)
    df['Month_Name'] = df['Date_Obj'].dt.month_name()
    df = df.dropna(subset=[SPECIES_COL])
    df[SPECIES_COL] = df[SPECIES_COL].astype(str).str.strip()
    return add_cyclic_features(df)


def train_model(df: pd.DataFrame, model_path: str):
    model_frame = df[[LAT_COL, LON_COL, 'Year', 'Month_Number', SPECIES_COL]].dropna().copy()
    model_frame = model_frame.join(df[FEATURE_COLUMNS], how='left')
    class_counts = model_frame[SPECIES_COL].value_counts()
    kept_species = class_counts[class_counts >= MIN_CLASS_SAMPLES].head(TOP_SPECIES_LIMIT).index.tolist()
    if len(kept_species) < 2:
        kept_species = class_counts.head(min(TOP_SPECIES_LIMIT, len(class_counts))).index.tolist()

    model_frame['Species_Model'] = model_frame[SPECIES_COL].where(
        model_frame[SPECIES_COL].isin(kept_species),
        'Other',
    )

    model_class_counts = model_frame['Species_Model'].value_counts()
    if len(model_class_counts) < 2:
        raise RuntimeError('Need at least two target classes to train the neural network.')

    features = model_frame[FEATURE_COLUMNS].astype(float)
    target = model_frame['Species_Model']
    stratify_target = target if target.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    classifier = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=220,
            early_stopping=False,  # Disabled due to compatibility with some Python/NumPy environments
            random_state=RANDOM_STATE,
        )),
    ])
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    labels = list(classifier.named_steps['classifier'].classes_)
    confusion = confusion_matrix(y_test, y_pred, labels=labels)
    loss_curve = getattr(classifier.named_steps['classifier'], 'loss_curve_', [])

    model_info = {
        'status': 'ok',
        'model': classifier,
        'source': 'trained',
        'classes': labels,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(report.get('macro avg', {}).get('f1-score', 0.0)),
        'weighted_f1': float(report.get('weighted avg', {}).get('f1-score', 0.0)),
        'classification_report': report,
        'confusion_matrix': confusion,
        'loss_curve': [float(value) for value in loss_curve],
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'class_counts': {str(key): int(value) for key, value in model_class_counts.items()},
        'kept_species': [str(item) for item in kept_species],
        'other_count': int(model_class_counts.get('Other', 0)),
    }

    model_info['feature_columns'] = list(FEATURE_COLUMNS)
    joblib.dump(model_info, model_path)
    return model_info


def main():
    args = parse_args()
    print(f'Loading data from: {args.input}')
    df = load_data(args.input, args.max_rows)
    print(f'Rows available for training: {len(df):,}')
    model_info = train_model(df, args.model_out)
    print(f"Saved model to: {args.model_out}")
    print(f"Accuracy: {model_info['accuracy']:.3f}")
    print(f"Macro F1: {model_info['macro_f1']:.3f}")
    print(f"Weighted F1: {model_info['weighted_f1']:.3f}")
    print(f"Classes: {', '.join(model_info['classes'])}")


if __name__ == '__main__':
    main()
