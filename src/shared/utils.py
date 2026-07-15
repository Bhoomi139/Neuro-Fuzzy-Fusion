import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any


def save_model_bundle(model: Any, scaler: Any, imputer: Any, threshold: float, metrics: dict, output_dir: str, model_name: str, fold: int) -> str:
    bundle_data = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "threshold": threshold,
        "metrics": metrics,
        "params": model.get_params() if hasattr(model, 'get_params') else {}
    }
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"best_{model_name.lower()}_fold_{fold}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "wb") as f:
        pickle.dump(bundle_data, f)
    
    return filepath


def save_predictions(pred_df: pd.DataFrame, output_dir: str, model_name: str, fold: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{model_name.lower()}_predictions_fold_{fold}.csv")
    pred_df.to_csv(filepath, index=False)
    return filepath


def create_prediction_dataframe(test_idx: np.ndarray, y_true: np.ndarray, y_proba: np.ndarray, 
                                y_pred: np.ndarray, threshold: float, fold: int, model_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_index": test_idx,
        "y_true": y_true,
        "probability": y_proba,
        "prediction": y_pred,
        "threshold": threshold,
        "correct": (y_pred == y_true).astype(int),
        "fold": fold,
        "model": model_name
    })


def print_results_summary(results: Dict[str, list], title: str = "RESULTS"):
    print("\n" + "="*70)
    print(f"FINAL {title}")
    print("="*70)
    
    for model_name, metrics_list in results.items():
        df = pd.DataFrame(metrics_list)
        print(f"\n--- {model_name} ---")
        for col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"  Mean {col}: {mean_val:.4f} +/- {std_val:.4f}")
