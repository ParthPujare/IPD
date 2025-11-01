"""
Ensemble prediction module.
Combines predictions from multiple models with weighted averaging.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predict import predict_next_day


def ensemble_predict(weights=None):
    """
    Combine predictions from LSTM and TFT models.
    
    Args:
        weights (dict, optional): Weights for each model. Default: {'LSTM': 0.5, 'TFT': 0.5}
    
    Returns:
        dict: Ensemble prediction results
    """
    if weights is None:
        weights = {"LSTM": 0.5, "TFT": 0.5}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    predictions = {}
    errors = []
    
    # Get predictions from each model
    for model_name in weights.keys():
        try:
            pred = predict_next_day(model_name)
            predictions[model_name] = pred
        except Exception as e:
            print(f"Warning: {model_name} prediction failed: {e}")
            errors.append(model_name)
    
    if not predictions:
        raise ValueError("No model predictions available")
    
    # Remove failed models from weights
    for model in errors:
        weights.pop(model, None)
    
    # Renormalize weights
    if weights:
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
    
    # Weighted average of predictions
    ensemble_predicted_price = sum(
        pred["predicted_price"] * weights[model_name]
        for model_name, pred in predictions.items()
    )
    
    # Weighted average of confidence
    ensemble_confidence = sum(
        pred.get("confidence", 50) * weights[model_name]
        for model_name, pred in predictions.items()
    )
    
    # Average last price (should be same for all models)
    last_price = predictions[list(predictions.keys())[0]]["last_price"]
    
    # Calculate ensemble change percentage
    ensemble_change_pct = (ensemble_predicted_price - last_price) / last_price * 100
    
    # Uncertainty as weighted average
    ensemble_uncertainty = sum(
        pred.get("uncertainty", 10) * weights[model_name]
        for model_name, pred in predictions.items()
    )
    
    result = {
        "predicted_price": ensemble_predicted_price,
        "last_price": last_price,
        "predicted_change_pct": ensemble_change_pct,
        "confidence": ensemble_confidence,
        "uncertainty": ensemble_uncertainty,
        "model_predictions": predictions,
        "weights": weights
    }
    
    return result


if __name__ == "__main__":
    # Test ensemble prediction
    print("Testing ensemble prediction...")
    try:
        result = ensemble_predict()
        print(f"Ensemble Prediction: {result['predicted_price']:.2f}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Model predictions:")
        for model, pred in result["model_predictions"].items():
            print(f"  {model}: {pred['predicted_price']:.2f}")
    except Exception as e:
        print(f"Ensemble prediction failed: {e}")

