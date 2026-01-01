# Loan Eligibility & Advisory Web App

A production-ready, AI-powered loan eligibility assessment system with transparent explainability using SHAP. This system acts as a reliable pre-screening tool to reduce manual loan processing.

## Features

- **ML Pipeline**: Comprehensive EDA, feature engineering, and model training with XGBoost, LightGBM, and CatBoost
- **Explainability**: SHAP integration for global and per-user explanations
- **Web Application**: Modern, bank-grade UI with Flask backend
- **PDF Reports**: Professional PDF generation with user inputs, eligibility results, and SHAP explanations
- **AI Chatbot**: Financial chatbot powered by Groq and Google Gemma model for answering loan and financial queries

## Project Structure

```
LoanIQ/
├── app/
│   ├── backend/
│   │   └── app.py                 # Flask application
│   ├── frontend/
│   │   └── (static files)
│   ├── ml_pipeline/
│   │   ├── eda.py                 # Exploratory Data Analysis
│   │   ├── preprocessing.py       # Feature engineering & preprocessing
│   │   ├── train_model.py         # Model training script
│   │   └── shap_explainer.py      # SHAP explainability module
│   ├── utils/
│   │   ├── pdf_generator.py       # PDF report generator
│   │   └── chatbot.py            # Financial chatbot with Groq & Gemma
│   ├── models/                    # Saved models and preprocessors
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css          # Modern UI styles
│   │   ├── js/
│   │   │   └── main.js            # Frontend JavaScript
│   │   └── reports/               # Generated PDF reports
│   └── templates/
│       └── index.html             # Main HTML template
├── loan_dataS.csv                 # Dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### 2. Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd LoanIQ
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Groq API Key for Chatbot (Optional but recommended):**
   
   Get your free API key from [Groq Console](https://console.groq.com/):
   
   Create a `.env` file in the project root:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   The chatbot will work without the API key but will return placeholder responses.

### 3. Train the Model

Before running the web application, you need to train the ML model:

```bash
# Run EDA (optional but recommended)
python app/ml_pipeline/eda.py

# Train the model (this will save the best model)
python app/ml_pipeline/train_model.py
```

This will:
- Perform data preprocessing and feature engineering
- Train XGBoost, LightGBM, and CatBoost models
- Select the best model based on ROC-AUC score
- Save the model and preprocessor to `app/models/`

### 4. Run the Application

```bash
# Navigate to the backend directory
cd app/backend

# Run the Flask application
python app.py
```

Or from the project root:

```bash
python app/backend/app.py
```

The application will be available at: `http://localhost:5000`

### 5. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Fill out the loan application form** with the required information:
   - Personal information (age, education, income, employment experience)
   - Home ownership status
   - Loan details (amount, purpose, percentage of income)
   - Credit information (credit score, history length, previous defaults)

2. **Submit the form** to get instant eligibility prediction

3. **View SHAP explanations** to understand which factors influenced the decision

4. **Download PDF report** containing:
   - User inputs
   - Eligibility result
   - SHAP explanations
   - Recommendations

5. **Use the AI chatbot** for loan-related questions and financial guidance

## API Endpoints

- `GET /` - Main web interface
- `POST /api/predict` - Get loan eligibility prediction
- `POST /api/shap` - Get SHAP explanation for a prediction
- `POST /api/generate-pdf` - Generate and download PDF report
- `POST /api/chatbot` - Financial chatbot endpoint (Groq + Gemma model)
- `POST /api/chatbot/reset` - Reset chatbot conversation memory
- `GET /api/health` - Health check endpoint

## Model Information

The system trains three models and selects the best one:
- **XGBoost**: Gradient boosting with XGBoost
- **LightGBM**: Microsoft's gradient boosting framework
- **CatBoost**: Yandex's gradient boosting library

The best model is selected based on ROC-AUC score on the test set.

## SHAP Explainability

The system uses SHAP (SHapley Additive exPlanations) TreeExplainer to provide:
- **Global explanations**: Feature importance across all predictions
- **Per-user explanations**: Detailed breakdown of factors affecting individual predictions

## Technical Details

- **Framework**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **ML Models**: XGBoost, LightGBM, CatBoost
- **Explainability**: SHAP TreeExplainer
- **PDF Generation**: ReportLab
- **Data Processing**: Pandas, NumPy, Scikit-learn

## Troubleshooting

### Model Not Found Error
If you see "No trained model found", make sure you've run the training script:
```bash
python app/ml_pipeline/train_model.py
```

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Port Already in Use
If port 5000 is already in use, modify `app/backend/app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port number
```

## Production Deployment

For production deployment:

1. Set `debug=False` in `app.py`
2. Use a production WSGI server (e.g., Gunicorn, uWSGI)
3. Configure proper security settings
4. Set up environment variables for sensitive data
5. Use a reverse proxy (e.g., Nginx)
6. Enable HTTPS

## License

This project is developed for educational and demonstration purposes.

## Support

For issues or questions, please refer to the project documentation or contact the development team.



