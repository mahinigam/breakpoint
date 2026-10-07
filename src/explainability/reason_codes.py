import numpy as np

def generate_reason_codes(feature_names, customer_shap_values, top_k=4):
    """
    Generates plain-English reason codes for a customer based on their SHAP values.
    We only look at positive SHAP values (features pushing risk higher).
    """
    top_indices = np.argsort(customer_shap_values)[-top_k:][::-1]
    top_features = [feature_names[k] for k in top_indices]
    
    reasons = []
    for feat in top_features:
        # In this dataset, features are anonymized but prefixed by family
        if feat.startswith('D_'): reasons.append(f"Delinquency indicator ({feat})")
        elif feat.startswith('P_'): reasons.append(f"Payment metric ({feat})")
        elif feat.startswith('B_'): reasons.append(f"Balance metric ({feat})")
        elif feat.startswith('S_'): reasons.append(f"Spend behavior ({feat})")
        elif feat.startswith('R_'): reasons.append(f"Risk metric ({feat})")
        else: reasons.append(feat)
        
    return reasons

def print_reason_codes_for_top_risks(preds, shap_values_pos, feature_names, n_customers=3):
    """
    Finds the n highest risk customers and prints their reason codes.
    """
    print(f"\n--- Reason Codes for Top {n_customers} High-Risk Customers ---")
    top_indices = np.argsort(preds)[-n_customers:][::-1] 
    
    for i, idx in enumerate(top_indices):
        customer_shap = shap_values_pos[idx]
        reasons = generate_reason_codes(feature_names, customer_shap, top_k=4)
            
        print(f"Customer {i+1} (Risk Score: {preds[idx]:.3f})")
        print(f"Action: Restrict/Decline")
        print(f"Top reasons: 1. {reasons[0]}, 2. {reasons[1]}, 3. {reasons[2]}, 4. {reasons[3]}\n")
