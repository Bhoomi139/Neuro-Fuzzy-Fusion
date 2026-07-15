import numpy as np
from sklearn.metrics import roc_auc_score


class Ensemble:
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MeanEnsemble(Ensemble):
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean(X, axis=1)


class WeightedEnsemble(Ensemble):
    def __init__(self):
        self.weights = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        scores = [max(roc_auc_score(y_train, X_train[:, i]), 0.001) for i in range(X_train.shape[1])]
        self.weights = np.array(scores) / np.sum(scores)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Call fit() before predict_proba()")
        return np.dot(X, self.weights)


class MajorityEnsemble(Ensemble):
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean((X >= 0.5).astype(int), axis=1)


def get_ensemble(method_name: str) -> Ensemble:
    methods = {
        'Mean_Averaging': MeanEnsemble,
        'Weighted_Averaging': WeightedEnsemble,
        'Majority_Voting': MajorityEnsemble
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown ensemble: {method_name}")
    
    return methods[method_name]()
