import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import COMPARISON_MODELS, DEFAULT_SEED, VALIDATION_SPLIT
from shared.data_loader import load_raw_data, get_high_auc_lomo_splits, perfectly_balance_train_data, split_train_val
from shared.metrics import find_best_threshold, compute_metrics
from shared.utils import save_model_bundle, save_predictions, create_prediction_dataframe, print_results_summary

from .models import tune_model
from .ensembles import get_ensemble


def run_experiment(data_dir: str, output_dir: str, seed: int = DEFAULT_SEED):
    os.makedirs(output_dir, exist_ok=True)
    
    X, y, groups = load_raw_data(data_dir)
    results = {model_name: [] for model_name in COMPARISON_MODELS}
    all_predictions = []

    print("\nStarting Cross-Model Evaluation (LR, XGB, Ensemble Methods)...")

    for fold, (train_idx, test_idx, test_fakes) in enumerate(get_high_auc_lomo_splits(y, groups, seed=seed), 1):
        print(f"\n--- FOLD {fold}/4 --- Held-out: {test_fakes}")
        
        X_train_raw = X[train_idx]
        y_train_raw = y[train_idx]
        groups_train = groups[train_idx]
        X_test_raw = X[test_idx]
        y_test = y[test_idx]

        X_balanced, y_balanced, _ = perfectly_balance_train_data(X_train_raw, y_train_raw, groups_train, seed)
        X_train_85, X_val_15, y_train_85, y_val_15 = split_train_val(X_balanced, y_balanced, test_size=VALIDATION_SPLIT, seed=seed)

        imputer = SimpleImputer(strategy='mean')
        X_train_85_imputed = imputer.fit_transform(X_train_85)
        X_val_15_imputed = imputer.transform(X_val_15)
        X_test_imputed = imputer.transform(X_test_raw)

        lr_model = tune_model(X_train_85_imputed, y_train_85, 'LR', seed, fold)
        xgb_model = tune_model(X_train_85_imputed, y_train_85, 'XGB', seed, fold)

        scaler = StandardScaler()
        X_train_85_scaled = scaler.fit_transform(X_train_85_imputed)
        X_val_15_scaled = scaler.transform(X_val_15_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        models = {
            'Mean_Averaging': None,
            'Weighted_Averaging': None,
            'Majority_Voting': None,
            'Logistic_Regression': lr_model,
            'XGBoost': xgb_model
        }

        for model_name in COMPARISON_MODELS:
            model = models[model_name]

            if model_name in ['Mean_Averaging', 'Weighted_Averaging', 'Majority_Voting']:
                ensemble = get_ensemble(model_name)
                ensemble.fit(X_train_85_imputed, y_train_85)
                val_proba = ensemble.predict_proba(X_val_15_imputed)
                test_proba = ensemble.predict_proba(X_test_imputed)
            else:
                model.fit(X_train_85_scaled, y_train_85)
                val_proba = model.predict_proba(X_val_15_scaled)[:, 1]
                test_proba = model.predict_proba(X_test_scaled)[:, 1]

            best_threshold = find_best_threshold(y_val_15, val_proba)
            y_pred = (test_proba >= best_threshold).astype(int)

            metrics = compute_metrics(y_test, test_proba, best_threshold)
            
            if model_name in ['Logistic_Regression', 'XGBoost']:
                model_path = save_model_bundle(model, scaler, imputer, best_threshold, metrics, output_dir, model_name, fold)
                print(f"    [{model_name}] Saved bundle -> {model_path}")

            pred_df = create_prediction_dataframe(test_idx, y_test, test_proba, y_pred, best_threshold, fold, model_name)
            all_predictions.append(pred_df)
            save_predictions(pred_df, output_dir, model_name, fold)

            results[model_name].append(metrics)
            print(f"    [{model_name}] AUC: {metrics['AUC']:.4f} | EER: {metrics['EER']:.4f} | Acc: {metrics['Acc']:.4f} | F1: {metrics['F1']:.4f}")

    final_df = pd.concat(all_predictions, ignore_index=True)
    final_df.to_csv(os.path.join(output_dir, "all_model_predictions.csv"), index=False)
    print("\nSaved combined prediction file.")

    print_results_summary(results, title="CROSS-MODEL COMPARISON RESULTS")
