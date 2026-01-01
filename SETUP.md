# Quick Setup Guide

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train XGBoost, LightGBM, and CatBoost models
- Select the best model
- Save the model and preprocessor

**Expected time:** 5-10 minutes depending on your system

### 3. Run the Application

```bash
python run_app.py
```

The application will start at: `http://localhost:5000`

### 4. Access the Web Interface

Open your browser and go to: `http://localhost:5000`

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Make sure you're in the project root directory and all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Model not found

**Solution:** Run the training script first:
```bash
python train_model.py
```

### Issue: Port already in use

**Solution:** Change the port in `run_app.py` or `app/backend/app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port
```

## Optional: Run EDA

To generate exploratory data analysis plots:

```bash
python app/ml_pipeline/eda.py
```

Plots will be saved to `app/ml_pipeline/eda_output/`



