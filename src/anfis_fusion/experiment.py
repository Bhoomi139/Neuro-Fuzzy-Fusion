import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import DEFAULT_SEED, VALIDATION_SPLIT
from shared.data_loader import load_raw_data, get_high_auc_lomo_splits, perfectly_balance_train_data, split_train_val
from shared.metrics import find_best_threshold, compute_metrics
from shared.utils import save_predictions, create_prediction_dataframe, print_results_summary

from .anfis_models import tune_anfis, build_anfis


def save_anfis_bundle(model: Any, scaler: Any, imputer: Any, threshold: float, best_params: dict, 
                      output_dir: str, fold: int) -> str:
    bundle = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "threshold": threshold,
        "params": best_params
    }
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"anfis_bundle_fold_{fold}.pkl")
    
    with open(filepath, "wb") as f:
        pickle.dump(bundle, f)
    
    return filepath


def run_experiment(data_dir: str, output_dir: str, seed: int = DEFAULT_SEED):
    os.makedirs(output_dir, exist_ok=True)
    
    X, y, groups = load_raw_data(data_dir)
    metrics_list = []
    all_predictions = []

    print("\nStarting ANFIS Leave-One-Model-Out (LOMO) Validation...")

    for fold, (train_idx, test_idx, test_fake_models) in enumerate(get_high_auc_lomo_splits(y, groups, seed=seed), 1):
        print(f"\n--- FOLD {fold}/4 --- Held-out: {test_fake_models}")
        
        X_train_raw = X[train_idx]
        y_train_raw = y[train_idx]
        groups_train = groups[train_idx]
        X_test_raw = X[test_idx]
        y_test = y[test_idx]

        X_balanced, y_balanced, _ = perfectly_balance_train_data(X_train_raw, y_train_raw, groups_train, seed)
        X_train_85, X_val_15, y_train_85, y_val_15 = split_train_val(X_balanced, y_balanced, test_size=VALIDATION_SPLIT, seed=seed)

        imputer = SimpleImputer(strategy='mean')
        X_train_85_imp = imputer.fit_transform(X_train_85)
        X_val_15_imp = imputer.transform(X_val_15)
        X_test_imp = imputer.transform(X_test_raw)

        scaler = StandardScaler()
        X_train_85_scaled = scaler.fit_transform(X_train_85_imp)
        X_val_15_scaled = scaler.transform(X_val_15_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        print("    Tuning ANFIS...")
        best_params = tune_anfis(X_train_85_imp, y_train_85, seed=seed, n_trials=15, fold_idx=fold)
        print(f"    Best params: {best_params}")
        
        model = build_anfis(best_params, seed=seed)
        model.fit(X_train_85_scaled, y_train_85)
        
        val_proba = np.ravel(model.predict_proba(X_val_15_scaled))
        best_threshold = find_best_threshold(y_val_15, val_proba)
        
        test_proba = np.ravel(model.predict_proba(X_test_scaled))
        y_pred = (test_proba >= best_threshold).astype(int)
        
        metrics = compute_metrics(y_test, test_proba, best_threshold)
        
        model_path = save_anfis_bundle(model, scaler, imputer, best_threshold, best_params, output_dir, fold)
        print(f"    Bundle saved -> {model_path}")
        
        pred_df = create_prediction_dataframe(test_idx, y_test, test_proba, y_pred, best_threshold, fold, 'ANFIS')
        all_predictions.append(pred_df)
        save_predictions(pred_df, output_dir, 'anfis', fold)
        
        metrics_list.append(metrics)
        print(f"    Fold Results | AUC: {metrics['AUC']:.4f} | EER: {metrics['EER']:.4f} | Acc: {metrics['Acc']:.4f} | F1: {metrics['F1']:.4f}")

    final_df = pd.concat(all_predictions, ignore_index=True)
    final_df.to_csv(os.path.join(output_dir, "all_anfis_predictions.csv"), index=False)
    print("\nSaved combined ANFIS predictions.")

    results = {'ANFIS': metrics_list}
    print_results_summary(results, title="ANFIS EVALUATION RESULTS")
