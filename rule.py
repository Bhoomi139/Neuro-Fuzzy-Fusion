import pickle
import numpy as np
import warnings
import torch.nn as nn

warnings.filterwarnings("ignore")

original_getattr = nn.Module.__getattr__

def patched_getattr(self, name):
    if name in ["__get_strength_by_mean", "__get_strength_by_prod", "__get_strength_by_blend"]:
        return lambda *args, **kwargs: None
    return original_getattr(self, name)

nn.Module.__getattr__ = patched_getattr

def to_numpy(val):
    if hasattr(val, 'detach'):
        return val.detach().cpu().numpy()
    return val

def extract_fuzzy_rules(model_path, feature_names):
    print(f"Loading bundle from: {model_path}")
    
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
        
    # Unpatch immediately
    nn.Module.__getattr__ = original_getattr
    
    wrapper_model = bundle["model"]
    scaler = bundle["scaler"]
    inner_model = getattr(wrapper_model, 'network', None)
            
    if inner_model is None:
        raise AttributeError("Could not locate the 'network' attribute inside the wrapper!")

    num_rules = wrapper_model.num_rules
    mf_type = wrapper_model.mf_class
    
    
    print(f"ANFIS ARCHITECTURE (FOLD 2)")
    print(f"Number of Rules: {num_rules}")
    print(f"Membership Function Used: {mf_type}")

    print("\n LINGUISTIC TERMS (Real-World Logit Boundaries)")
    term_names = ["Very Low", "Low", "Medium", "High", "Very High"] if num_rules == 5 else [f"MF{i+1}" for i in range(num_rules)]

    feature_term_mappings = np.zeros((len(feature_names), num_rules), dtype=int)

    for i, feature in enumerate(feature_names):
        # We need the '.c' parameter for the center of a GBell curve
        # If using Gaussian, we would need '.mu' or '.center' depending on how the developer named it
        param_name = f"memberships.{i}.c" if mf_type == "GBell" else f"memberships.{i}.mu"
        
        feat_centers_scaled = to_numpy(dict(inner_model.named_parameters())[param_name])
        
        dummy_array = np.zeros((len(feat_centers_scaled), len(feature_names)))
        dummy_array[:, i] = feat_centers_scaled
        feat_centers_real = scaler.inverse_transform(dummy_array)[:, i]
        
        sorted_idx = np.argsort(feat_centers_real)
        
        for term_rank, original_mf_idx in enumerate(sorted_idx):
             feature_term_mappings[i, original_mf_idx] = term_rank
        
        print(f"\nDetector: {feature}")
        for j, order_idx in enumerate(sorted_idx):
            label = term_names[j]
            val = feat_centers_real[order_idx]
            print(f"  {label:<10} -> Center: {val:.4f} (Internal MF Index: {order_idx})")

    print("--- EXTRACTED FUZZY RULES ---")
    
    con_weights = to_numpy(dict(inner_model.named_parameters())["coeffs"]).squeeze()
    
    for rule_idx in range(num_rules):
        rule_weights = con_weights[rule_idx]
    
        bias_weight = rule_weights[-1]
        
        conclusion = "FAKE" if bias_weight > 0 else "REAL"
    
        if_conditions = []
        for feat_idx, feature in enumerate(feature_names):
             term_rank = feature_term_mappings[feat_idx, rule_idx]
             term_label = term_names[term_rank]
             if_conditions.append(f"{feature} is {term_label}")
             
        if_string = " AND ".join(if_conditions)
        
        print(f"\nRule {rule_idx + 1}:")
        print(f"  IF {if_string}")
        print(f"  THEN weight is {bias_weight:>8.4f} --> Suggests: {conclusion}")

if __name__ == "__main__":
    my_detectors = ["FIRE_Logit", "FreqNet_Logit", "F3Net_Logit", "SPSL_Logit", "SRM_Logit"]
    model_file = r"C:\Users\Bhoomi Priya\Documents\ANFIS\output_final\anfis_bundle_fold_2.pkl" 
    extract_fuzzy_rules(model_file, my_detectors)