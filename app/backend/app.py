"""
Flask Backend Application for Loan Eligibility & Advisory System
"""
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import json
from dotenv import load_dotenv
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    print("Warning: LangChain dependencies not found. Chatbot will not work.")

# Load environment variables
load_dotenv()
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from app.ml_pipeline.preprocessing import LoanPreprocessor
from app.ml_pipeline.shap_explainer import SHAPExplainer
from app.utils.pdf_generator import generate_loan_report_pdf

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
CORS(app)

# Global variables for model and preprocessor
model = None
preprocessor = None
shap_explainer = None
X_train_sample = None

def load_model_and_preprocessor():
    """
    Load the trained model and preprocessor
    """
    global model, preprocessor, shap_explainer, X_train_sample
    
    try:
        # Load preprocessor
        preprocessor = LoanPreprocessor()
        preprocessor_path = os.path.join(project_root, 'app', 'models', 'preprocessor.pkl')
        preprocessor.load(preprocessor_path)
        
        # Try to load the best model (check which one exists)
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
        
        if model_path is None:
            raise FileNotFoundError("No trained model found. Please train the model first.")
        
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        
        # Load sample training data for SHAP
        try:
            data_path = os.path.join(project_root, 'loan_dataS.csv')
            df = pd.read_csv(data_path)
            # Transform using already-fitted preprocessor
            X_sample = preprocessor.transform(df.sample(n=min(1000, len(df)), random_state=42))
            X_train_sample = X_sample
            shap_explainer = SHAPExplainer(model, preprocessor, X_train_sample)
            print("SHAP explainer initialized")
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            shap_explainer = SHAPExplainer(model, preprocessor)
        
        print("Model and preprocessor loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    """
    Serve the main page
    """
    return render_template('index.html')

from flask import request, jsonify
import pandas as pd

# ---------- SAFE CAST HELPERS ----------
def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict loan eligibility (Production-safe)
    """
    try:
        # ---------- VALIDATE JSON ----------
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid or empty JSON payload"}), 400

        # ---------- REQUIRED FIELD CHECK ----------
        required_fields = [
            'person_age',
            'person_income',
            'loan_amnt',
            'credit_score'
        ]

        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing)}"
            }), 400

        # ---------- FEATURE EXTRACTION ----------
        features = {
            'person_age': safe_float(data.get('person_age')),
            'person_education': data.get('person_education'),
            'person_income': safe_float(data.get('person_income')),
            'person_emp_exp': safe_int(data.get('person_emp_exp')),
            'person_home_ownership': data.get('person_home_ownership'),
            'loan_amnt': safe_float(data.get('loan_amnt')),
            'loan_intent': data.get('loan_intent'),
            'loan_percent_income': safe_float(data.get('loan_percent_income')),
            'cb_person_cred_hist_length': safe_float(
                data.get('cb_person_cred_hist_length')
            ),
            'credit_score': safe_int(data.get('credit_score')),
            'previous_loan_defaults_on_file': data.get(
                'previous_loan_defaults_on_file'
            )
        }

        # ---------- DERIVED FEATURE ----------
        if (
            features['loan_percent_income'] == 0
            and features['person_income'] > 0
        ):
            features['loan_percent_income'] = (
                features['loan_amnt'] / features['person_income']
            )

        # ---------- DATAFRAME ----------
        df = pd.DataFrame([features])

        # ---------- PREPROCESS ----------
        X = preprocessor.transform(df)

        # ---------- PREDICTION ----------
        prediction = int(model.predict(X)[0])
        prediction_proba = model.predict_proba(X)[0]

        # ---------- RESPONSE ----------
        result = {
            "eligible": 1 if prediction == 0 else 0,
            "status": "Eligible" if prediction == 0 else "Not Eligible",
            "probability": float(prediction_proba[0]),  # Probability of 0 (Non-Default/Eligible)
            "confidence": float(max(prediction_proba)),
            "message": (
                "Congratulations! You are eligible for the loan."
                if prediction == 0
                else "Unfortunately, you are not eligible for the loan at this time."
            )
        }

        return jsonify(result), 200

    except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": "Prediction failed",
                "details": str(e)
            }), 500



@app.route('/api/shap', methods=['POST'])
def get_shap_explanation():
    """
    Get SHAP explanation for a prediction
    """
    try:
        data = request.json
        
        # Extract features
        features = {
            'person_age': float(data.get('person_age')),
            'person_education': data.get('person_education'),
            'person_income': float(data.get('person_income')),
            'person_emp_exp': int(data.get('person_emp_exp')),
            'person_home_ownership': data.get('person_home_ownership'),
            'loan_amnt': float(data.get('loan_amnt')),
            'loan_intent': data.get('loan_intent'),
            'loan_percent_income': float(data.get('loan_percent_income', 0)),
            'cb_person_cred_hist_length': float(data.get('cb_person_cred_hist_length')),
            'credit_score': int(data.get('credit_score')),
            'previous_loan_defaults_on_file': data.get('previous_loan_defaults_on_file')
        }
        
        # Calculate loan_percent_income if not provided
        if features['loan_percent_income'] == 0:
            features['loan_percent_income'] = features['loan_amnt'] / features['person_income']
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Get SHAP explanation
        explanation = shap_explainer.explain_instance(X)
        
        # Add original features for display
        explanation['original_features'] = features
        
        return jsonify(explanation)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    """
    Generate PDF report
    """
    try:
        data = request.json
        
        # Get prediction and SHAP explanation
        features = {
            'person_age': float(data.get('person_age')),
            'person_education': data.get('person_education'),
            'person_income': float(data.get('person_income')),
            'person_emp_exp': int(data.get('person_emp_exp')),
            'person_home_ownership': data.get('person_home_ownership'),
            'loan_amnt': float(data.get('loan_amnt')),
            'loan_intent': data.get('loan_intent'),
            'loan_percent_income': float(data.get('loan_percent_income', 0)),
            'cb_person_cred_hist_length': float(data.get('cb_person_cred_hist_length')),
            'credit_score': int(data.get('credit_score')),
            'previous_loan_defaults_on_file': data.get('previous_loan_defaults_on_file')
        }
        
        # Calculate loan_percent_income if not provided
        if features['loan_percent_income'] == 0:
            features['loan_percent_income'] = features['loan_amnt'] / features['person_income']
        
        # Create DataFrame and predict
        df = pd.DataFrame([features])
        X = preprocessor.transform(df)
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Get SHAP explanation
        shap_explanation = shap_explainer.explain_instance(X)
        
        # Generate PDF
        output_dir = os.path.join(project_root, 'app', 'static', 'reports')
        pdf_path = generate_loan_report_pdf(
            features,
            1 if prediction == 0 else 0, # eligible if 0
            prediction_proba[0], # probability of 0 (eligible)
            shap_explanation,
            output_dir
        )
        
        return send_file(pdf_path, as_attachment=True, 
                       download_name=f'loan_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """
    Chatbot endpoint using Groq API
    """
    try:
        data = request.json
        message = data.get('message', '')
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
             return jsonify({
                'response': "Setup Error: GROQ_API_KEY is missing from environment variables. Please check your .env file.",
                'timestamp': datetime.now().isoformat()
            })

        # Initialize ChatGroq
        try:
            chat = ChatGroq(
                temperature=0.3, 
                groq_api_key=api_key, 
                model_name="llama-3.3-70b-versatile"
            )
            
            # Simple prompt template for loan advisory
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful and knowledgeable banking assistant for LoanIQ. "
                           "You help users understand loan eligibility, credit scores, and financial planning. "
                           "Keep your answers concise, professional, and friendly. "
                           "Do not provide specific financial advice or guarantee loan approval."),
                ("user", "{input}")
            ])
            
            chain = prompt | chat | StrOutputParser()
            
            # Invoke chain
            response_text = chain.invoke({"input": message})
            
            response = {
                'response': response_text,
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Groq API Error: {e}")
            return jsonify({
                'response': "I am currently experiencing connection issues. Please try again later.",
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'shap_loaded': shap_explainer is not None
    })

if __name__ == '__main__':
    print("Loading model and preprocessor...")
    if load_model_and_preprocessor():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train the model first.")
        print("Run: python app/ml_pipeline/train_model.py")

