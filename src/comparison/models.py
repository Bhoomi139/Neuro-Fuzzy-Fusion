import numpy as np
import optuna
import xgboost as xgb
from typing import Any
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_model(X_train_raw: np.ndarray, y_train: np.ndarray, model_type: str, seed: int, fold_idx: int, n_trials: int = 15) -> Any:
    def objective(trial):
        skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        fold_scores = []
        
        for train_idx, val_idx in skfold.split(X_train_raw, y_train):
            X_train_fold = X_train_raw[train_idx]
            X_val_fold = X_train_raw[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)

            if model_type == 'LR':
                model = LogisticRegression(
                    C=trial.suggest_float("C", 1e-4, 10.0, log=True),
                    random_state=seed,
                    max_iter=500,
                    class_weight='balanced'
                )
            elif model_type == 'XGB':
                params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                    "eval_metric": "auc",
                    "random_state": seed,
                    "n_jobs": -1
                }
                model = xgb.XGBClassifier(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.fit(X_train_scaled, y_train_fold)
            fold_scores.append(roc_auc_score(y_val_fold, model.predict_proba(X_val_scaled)[:, 1]))
        
        return np.mean(fold_scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed + fold_idx))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if model_type == 'LR':
        return LogisticRegression(**study.best_params, random_state=seed, max_iter=500, class_weight='balanced')
    elif model_type == 'XGB':
        return xgb.XGBClassifier(**study.best_params, random_state=seed, eval_metric="auc", n_jobs=1)
