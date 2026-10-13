import joblib
import pandas as pd


def predict_rain(
    input_features: pd.DataFrame, model_path: str = "models/rain_prediction_rf.pkl"
):
    """Memuat model terelatih dan memprediksi apakah besok akan hujan."""
    model = joblib.load(model_path)
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)[:, 1]
    return prediction, probability
