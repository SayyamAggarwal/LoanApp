
import pandas as pd
import joblib
import os
import sys
import numpy as np

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.ml_pipeline.preprocessing import LoanPreprocessor

def investigate_impact():
    print("Investigating Impact of Home Ownership vs Income...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Load model and preprocessor
    preprocessor = LoanPreprocessor()
    preprocessor.load(os.path.join(project_root, 'app', 'models', 'preprocessor.pkl'))
    # Try loading XGB model as it was the best model
    try:
        model_path = os.path.join(project_root, 'app', 'models', 'xgb_model.pkl')
        model = joblib.load(model_path)
        print(f"Loaded XGB model from {model_path}")
        
        if hasattr(model, 'get_booster'):
           feats = model.get_booster().feature_names
        else:
           feats = model.feature_names_in_
            
    except:
        model_path = os.path.join(project_root, 'app', 'models', 'lgb_model.pkl')
        model = joblib.load(model_path)
        print(f"Loaded LGB model from {model_path}")
        feats = model.feature_name_
    
    # Base High Income Profile
    base_profile = {
        'person_age': 35.0,
        'person_education': 'Master',
        'person_income': 120000.0,  # High Income
        'person_emp_exp': 8,
        'loan_amnt': 15000.0,
        'loan_intent': 'PERSONAL',
        'loan_percent_income': 0.125,
        'cb_person_cred_hist_length': 6.0,
        'credit_score': 720,
        'previous_loan_defaults_on_file': 'No'
    }
    
    # Compare Loan Default History
    defaults = ['No', 'Yes']
    
    print(f"\n{'PREV DEFAULT':<15} | {'PROB (DEFAUT/REJECT)':<20} | {'PREDICTION'}")
    print("-" * 60)
    
    for default_status in defaults:
        profile = base_profile.copy()
        profile['previous_loan_defaults_on_file'] = default_status
        
        df = pd.DataFrame([profile])
        X = preprocessor.transform(df)
        
        # Prob of Class 1 (Default/Reject)
        prob_default = model.predict_proba(X)[0][1] 
        pred = model.predict(X)[0]
        
        print(f"{default_status:<15} | {prob_default:.4f}               | {pred} ({'REJECT' if pred==1 else 'APPROVE'})")

    # Feature Importances TO see what's driving it
    print("\nTop 10 Feature Importances:")
    importances = model.feature_importances_
    # feats already defined above
    feat_imp = pd.DataFrame({'feature': feats, 'importance': importances})
    print(feat_imp.sort_values('importance', ascending=False).head(10))

if __name__ == "__main__":
    investigate_impact()
