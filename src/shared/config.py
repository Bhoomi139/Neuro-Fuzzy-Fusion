REAL_DATASETS = ['deepfacelab_logits_merged', 'CollabDiff_logits_merged']

FAKE_FOLDS = [
    ['CollabDiff_logits_merged', 'danet_logits_merged', 'inswap_logits_merged', 'blendface_logits_merged', 'facevid2vid_logits_merged'],
    ['ddim_logits_merged', 'fsgan_logits_merged', 'pirender_logits_merged', 'deepfacelab_logits_merged', 'fomm_logits_merged'],
    ['pixart_logits_merged', 'simswap_logits_merged', 'uniface_logits_merged', 'faceswap_logits_merged', 'mcnet_logits_merged'],
    ['rddm_logits_merged', 'vqgan_logits_merged', 'MRAA_logits_merged', 'sadtalker_logits_merged', 'tpsm_logits_merged', 'wav2lip_logits_merged']
]

COMPARISON_MODELS = ['Mean_Averaging', 'Weighted_Averaging', 'Majority_Voting', 'Logistic_Regression', 'XGBoost']

DEFAULT_SEED = 42
N_SPLITS_OUTER = 4
N_SPLITS_INNER = 3
N_TRIALS_OPTUNA = 15
VALIDATION_SPLIT = 0.15

LR_PARAMS = {
    'C': (1e-4, 10.0),
    'max_iter': 500,
    'class_weight': 'balanced'
}

XGB_PARAMS = {
    'max_depth': (2, 8),
    'learning_rate': (1e-3, 0.3),
    'n_estimators': (50, 300),
    'eval_metric': 'auc'
}

ANFIS_PARAMS = {
    'num_rules': (2, 5),
    'lr': (1e-4, 5e-2),
    'epochs': (50, 150),
    'mf_class': ['Gaussian', 'GBell'],
    'vanishing_strategy': ['blend', 'mean', 'prod'],
    'batch_size': [64, 128],
    'reg_lambda': 1e-4,
    'early_stopping': True,
    'n_patience': 5,
    'epsilon': 0.001,
    'valid_rate': 0.1,
    'optim': 'Adam'
}

IMPUTATION_STRATEGY = 'mean'
FEATURE_DROP_COLS = ['image', 'label', 'source_dataset']
