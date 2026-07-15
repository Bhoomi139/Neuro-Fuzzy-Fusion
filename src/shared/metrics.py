import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, precision_recall_curve


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = 'f1') -> float:
    prec, rec, ths = precision_recall_curve(y_true, y_proba)
    
    if metric == 'f1':
        f1_scores = (2 * prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_scores)
    elif metric == 'youden':
        fpr, tpr, ths = roc_curve(y_true, y_proba)
        best_idx = np.argmax(tpr - fpr)
        return ths[best_idx]
    else:
        f1_scores = (2 * prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_scores)
    
    return ths[best_idx] if len(ths) > 0 else 0.5


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        'AUC': roc_auc_score(y_true, y_proba),
        'EER': calculate_eer(y_true, y_proba),
        'Acc': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, pos_label=1)
    }
