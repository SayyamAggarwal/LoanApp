
import sys
import os
import pandas as pd
import joblib
import traceback

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml_pipeline.preprocessing import LoanPreprocessor

def test_prediction():
    print("Testing model prediction...")
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Load preprocessor
        preprocessor = LoanPreprocessor()
        preprocessor_path = os.path.join(project_root, 'app', 'models', 'preprocessor.pkl')
        if not os.path.exists(preprocessor_path):
            print(f"Preprocessor not found at {preprocessor_path}")
            return
            
        preprocessor.load(preprocessor_path)
        print("Preprocessor loaded")
        
        # Load model
        model_paths = [
            os.path.join(project_root, 'app', 'models', 'xgb_model.pkl'),
            os.path.join(project_root, 'app', 'models', 'lgb_model.pkl'),
            os.path.join(project_root, 'app', 'models', 'cat_model.pkl')
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("No model found")
            return
            
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        
        # Create dummy data
        features = {
            'person_age': 30.0,
            'person_education': 'Bachelor',
            'person_income': 50000.0,
            'person_emp_exp': 5,
            'person_home_ownership': 'RENT',
            'loan_amnt': 10000.0,
            'loan_intent': 'PERSONAL',
            'loan_percent_income': 0.2,
            'cb_person_cred_hist_length': 3.0,
            'credit_score': 700,
            'previous_loan_defaults_on_file': 'No'
        }
        
        df = pd.DataFrame([features])
        print("Dataframe created")
        
        # Transform
        X = preprocessor.transform(df)
        print(f"Transformed shape: {X.shape}")
        
        # Predict
        prediction = model.predict(X)[0]
        print(f"Prediction: {prediction}")
        
        try:
            prediction_proba = model.predict_proba(X)[0]
            print(f"Prediction Proba: {prediction_proba}")
        except Exception as e:
            print(f"predict_proba failed: {e}")
            
        print("Test successful")
        
    except Exception as e:
        print("Test failed with error:")
        traceback.print_exc()
        with open('error_msg.txt', 'w') as f:
            f.write(str(e))

if __name__ == "__main__":
    test_prediction()
