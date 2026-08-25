import numpy as np
import pandas as pd
import streamlit as st

REQUIRED_MIN_ROWS = 10


def load_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def validate_dataset(df: pd.DataFrame) -> list:
    issues = []
    if df.empty:
        issues.append("Dataset is empty.")
    if len(df) < REQUIRED_MIN_ROWS:
        issues.append(f"Dataset has fewer than {REQUIRED_MIN_ROWS} rows.")
    if df.columns.duplicated().any():
        issues.append("Dataset contains duplicate column names.")
    return issues


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=np.number).columns.tolist(),
    }


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.drop_duplicates().copy()

    numeric_cols = clean.select_dtypes(include=np.number).columns
    categorical_cols = clean.select_dtypes(exclude=np.number).columns

    for col in numeric_cols:
        if clean[col].isna().any():
            clean[col] = clean[col].fillna(clean[col].median())

    for col in categorical_cols:
        if clean[col].isna().any():
            mode = clean[col].mode()
            clean[col] = clean[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return clean.reset_index(drop=True)


def get_dataframe():
    """Returns the preprocessed dataset if available, else the raw upload, else None."""
    if st.session_state.get("df_clean") is not None:
        return st.session_state["df_clean"]
    return st.session_state.get("df")
