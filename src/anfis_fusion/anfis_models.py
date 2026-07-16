import os
import numpy as np
import optuna
from typing import Any, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from xanfis import GdAnfisClassifier
except ImportError:
    print("Installing xanfis...")
    os.system('pip install xanfis')
    from xanfis import GdAnfisClassifier


def tune_anfis(X_train_raw: np.ndarray, y_train: np.ndarray, 
               seed: int = 42, n_trials: int = 15, fold_idx: int = 1) -> Dict[str, Any]:
    def objective(trial):
        num_rules = trial.suggest_int("num_rules", 2, 5)
        learning_rate = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        epochs = trial.suggest_int("epochs", 50, 150, step=25)
        membership_type = trial.suggest_categorical("mf_class", ["Gaussian", "GBell"])
        vanish_type = trial.suggest_categorical("vanishing", ["blend", "mean", "prod"])
        batch = trial.suggest_categorical("batch_size", [64, 128])

        fold_scores = []
        skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

        for train_fold_idx, val_fold_idx in skfold.split(X_train_raw, y_train):
            X_train_fold = X_train_raw[train_fold_idx]
            X_val_fold = X_train_raw[val_fold_idx]
            y_train_fold = y_train[train_fold_idx]
            y_val_fold = y_train[val_fold_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)

            model = GdAnfisClassifier(
                num_rules=num_rules,
                mf_class=membership_type,
                act_output=None,
                vanishing_strategy=vanish_type,
                reg_lambda=1e-4,
                epochs=epochs,
                batch_size=batch,
                optim="Adam",
                optim_params={"lr": learning_rate},
                early_stopping=True,
                n_patience=5,
                epsilon=0.001,
                valid_rate=0.1,
                seed=seed,
                verbose=False
            )
            
            model.fit(X_train_scaled, y_train_fold)
            
            y_proba = np.ravel(model.predict_proba(X_val_scaled))
            
            if not np.isnan(y_proba).any():
                fold_scores.append(roc_auc_score(y_val_fold, y_proba))

        if not fold_scores:
            raise optuna.TrialPruned("Found NaNs in predictions.")
        
        return np.mean(fold_scores)

    dynamic_seed = seed + fold_idx
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=dynamic_seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params


def build_anfis(best_params: Dict[str, Any], seed: int = 42) -> GdAnfisClassifier:
    model = GdAnfisClassifier(
        num_rules=best_params["num_rules"],
        mf_class=best_params["mf_class"],
        act_output=None,
        vanishing_strategy=best_params["vanishing"],
        reg_lambda=1e-4,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        optim="Adam",
        optim_params={"lr": best_params["lr"]},
        early_stopping=True,
        n_patience=5,
        epsilon=0.001,
        valid_rate=0.1,
        seed=seed,
        verbose=False
    )
    return model
