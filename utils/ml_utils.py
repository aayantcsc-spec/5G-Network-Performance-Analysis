import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

MODEL_FACTORY = {
    "Linear Regression": lambda: LinearRegression(),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=12, random_state=42),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=100, max_depth=18, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
    ),
}

DEFAULT_TARGET = "Download Speed (Mbps)"
PREDICTION_TARGETS = [
    "Download Speed (Mbps)",
    "Upload Speed (Mbps)",
    "Latency (ms)",
    "Jitter (ms)",
]

# Values that are outcomes rather than useful predictors for a network-performance model.
TARGET_LIKE = {
    "Download Speed (Mbps)",
    "Upload Speed (Mbps)",
    "Latency (ms)",
    "Jitter (ms)",
    "Ping to Google (ms)",
    "Video Streaming Quality",
    "Dropped Connection",
}


def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    X = df[feature_cols].copy()
    if "Timestamp" in X.columns:
        ts = pd.to_datetime(X["Timestamp"], errors="coerce")
        X["Hour"] = ts.dt.hour
        X["Day of Week"] = ts.dt.dayofweek
        X["Month"] = ts.dt.month
        X = X.drop(columns=["Timestamp"])
    # sklearn handles numeric bool poorly in some versions; make booleans explicit categories.
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(str)
    return X


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.2,
) -> dict:
    X = prepare_features(df, feature_cols)
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    results = {}
    for name, factory in MODEL_FACTORY.items():
        preprocessor = make_preprocessor(X_train)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", factory()),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        results[name] = {
            "model": pipeline,
            "feature_cols": list(feature_cols),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, y_pred)),
            "y_test": y_test.reset_index(drop=True),
            "y_pred": y_pred,
        }

    return results


def best_model_name(results: dict) -> str:
    return max(results, key=lambda name: results[name]["r2"])


def predict(model_bundle: dict, input_values: dict) -> float:
    row = pd.DataFrame([input_values])
    return float(model_bundle["model"].predict(row)[0])


def default_feature_columns(df: pd.DataFrame, target_col: str) -> list:
    """Choose sensible prediction inputs while avoiding other outcome columns."""
    cols = []
    for c in df.columns:
        if c == target_col or c in TARGET_LIKE:
            continue
        cols.append(c)
    return cols


def feature_importance(model_bundle: dict) -> pd.DataFrame:
    """Return feature importance for tree models after one-hot preprocessing."""
    pipeline = model_bundle["model"]
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocessor"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])
    names = pre.get_feature_names_out()
    values = model.feature_importances_
    out = pd.DataFrame({"Feature": names, "Importance": values})
    out["Feature"] = out["Feature"].str.replace(r"^(numeric|categorical)__", "", regex=True)
    return out.sort_values("Importance", ascending=False).head(20).reset_index(drop=True)
