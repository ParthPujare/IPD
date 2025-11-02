"""
Simple and robust ensemble prediction module.
Works even if one or more models fail — defaults to TFT if only one succeeds.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predict import predict_next_day


def ensemble_predict(weights=None):
    """
    Combine predictions from multiple models using weighted average.
    Falls back gracefully if some models fail.
    """
    if weights is None:
        # Default equal weights for LSTM and TFT
        weights = {"LSTM": 0.5, "TFT": 0.5}

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    predictions = {}
    failed = []

    print("\n🔮 Running Ensemble Prediction...\n")

    for model_name, w in weights.items():
        try:
            pred = predict_next_day(model_name)
            predictions[model_name] = pred
            print(f"✅ {model_name} prediction successful: {pred['predicted_price']:.2f}")
        except Exception as e:
            print(f"⚠️ {model_name} prediction failed: {e}")
            failed.append(model_name)

    # Remove failed models
    for model in failed:
        weights.pop(model, None)

    # If no models succeeded, raise error
    if not predictions:
        raise RuntimeError("❌ No model predictions available for ensemble.")

    # If only one model succeeded, just use that one
    if len(predictions) == 1:
        model_name = list(predictions.keys())[0]
        print(f"\n⚠️ Only {model_name} succeeded. Returning its prediction directly.\n")
        return predictions[model_name]

    # Renormalize weights after removing failed models
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Weighted average of predictions
    ensemble_predicted_price = sum(
        pred["predicted_price"] * weights[m]
        for m, pred in predictions.items()
    )

    # Weighted average of confidence
    ensemble_confidence = sum(
        pred.get("confidence", 50) * weights[m]
        for m, pred in predictions.items()
    )

    last_price = predictions[list(predictions.keys())[0]]["last_price"]
    ensemble_change_pct = (ensemble_predicted_price - last_price) / last_price * 100

    ensemble_uncertainty = sum(
        pred.get("uncertainty", 10) * weights[m]
        for m, pred in predictions.items()
    )

    result = {
        "predicted_price": ensemble_predicted_price,
        "last_price": last_price,
        "predicted_change_pct": ensemble_change_pct,
        "confidence": ensemble_confidence,
        "uncertainty": ensemble_uncertainty,
        "model_predictions": predictions,
        "weights": weights,
    }

    print("\n=== 📊 Ensemble Result ===")
    print(f"Predicted Price: ₹{ensemble_predicted_price:.2f}")
    print(f"Last Price: ₹{last_price:.2f}")
    print(f"Predicted Change: {ensemble_change_pct:.2f}%")
    print(f"Confidence: {ensemble_confidence:.2f}%")
    print(f"Uncertainty: {ensemble_uncertainty:.2f}")
    print(f"Models used: {list(predictions.keys())}")
    print(f"Weights: {weights}")
    print("===========================\n")

    return result


if __name__ == "__main__":
    print("Testing Ensemble Prediction...\n")
    result = ensemble_predict()
    print(f"Final Ensemble Prediction: ₹{result['predicted_price']:.2f}")
