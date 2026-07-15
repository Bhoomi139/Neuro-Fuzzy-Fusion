import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import KFold, train_test_split
from .config import REAL_DATASETS, FAKE_FOLDS, FEATURE_DROP_COLS


def load_raw_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    data_frames = []
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            df['source_dataset'] = os.path.basename(filepath).replace('.csv', '')
            data_frames.append(df)
        except Exception as e:
            print(f"Warning: Error reading {filepath}: {e}")
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    feature_cols = [col for col in combined_df.columns if col not in FEATURE_DROP_COLS]
    
    return combined_df[feature_cols].values, combined_df['label'].values.astype(int), combined_df['source_dataset'].values


def get_high_auc_lomo_splits(y: np.ndarray, groups: np.ndarray, seed: int = 42):
    real_indices = np.where((y == 0) & np.isin(groups, REAL_DATASETS))[0]
    kfold = KFold(n_splits=4, shuffle=True, random_state=seed)
    real_splits = list(kfold.split(real_indices))

    for fold_idx, test_fake_models in enumerate(FAKE_FOLDS):
        train_real_idx, test_real_idx = real_splits[fold_idx]
        fold_train_reals = real_indices[train_real_idx]
        fold_test_reals = real_indices[test_real_idx]
        fold_train_fakes = np.where((y == 1) & (~np.isin(groups, test_fake_models)))[0]
        fold_test_fakes = np.where((y == 1) & (np.isin(groups, test_fake_models)))[0]

        train_idx = np.concatenate([fold_train_reals, fold_train_fakes])
        test_idx = np.concatenate([fold_test_reals, fold_test_fakes])
        np.random.seed(seed + fold_idx)
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        yield train_idx, test_idx, test_fake_models


def perfectly_balance_train_data(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    real_indices = np.where(y == 0)[0]
    fake_indices = np.where(y == 1)[0]
    num_reals = len(real_indices)
    
    if len(fake_indices) <= num_reals:
        return X, y, groups
    
    try:
        sampled_fake_indices, _ = train_test_split(fake_indices, train_size=num_reals, stratify=groups[fake_indices], random_state=seed)
    except ValueError:
        np.random.seed(seed)
        sampled_fake_indices = np.random.choice(fake_indices, num_reals, replace=False)
    
    balanced_indices = np.concatenate([real_indices, sampled_fake_indices])
    np.random.shuffle(balanced_indices)
    
    return X[balanced_indices], y[balanced_indices], groups[balanced_indices]


def split_train_val(X: np.ndarray, y: np.ndarray, test_size: float = 0.15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
