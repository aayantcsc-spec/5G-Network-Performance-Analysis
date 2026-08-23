import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

MODEL_FACTORY = {
    "Linear Regression": lambda: LinearRegression(),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
}


def train_and_evaluate(df: pd.DataFrame, feature_cols: list, target_col: str, test_size: float = 0.2) -> dict:
    X = df[feature_cols].select_dtypes(include=np.number)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    for name, factory in MODEL_FACTORY.items():
        model = factory()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        results[name] = {
            "model": model,
            "scaler": scaler,
            "feature_cols": list(X.columns),
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": r2_score(y_test, y_pred),
            "y_test": y_test.reset_index(drop=True),
            "y_pred": y_pred,
        }

    return results


def best_model_name(results: dict) -> str:
    return max(results, key=lambda name: results[name]["r2"])


def predict(model_bundle: dict, input_values: dict) -> float:
    row = pd.DataFrame([input_values])[model_bundle["feature_cols"]]
    scaled = model_bundle["scaler"].transform(row)
    return float(model_bundle["model"].predict(scaled)[0])
