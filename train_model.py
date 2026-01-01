"""
Main entry point for training the loan eligibility model
"""
import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml_pipeline.train_model import train_models, save_model
from app.ml_pipeline.preprocessing import prepare_data

if __name__ == "__main__":
    print("=" * 60)
    print("Loan Eligibility Model Training")
    print("=" * 60)
    
    # Prepare data with SMOTE balancing
    print("\n1. Preparing data with SMOTE balancing...")
    print("   (SMOTE will balance the imbalanced dataset)")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(apply_smote=True)
    
    # Train models
    print("\n2. Training models...")
    best_model, best_model_name, results = train_models(X_train, y_train, X_test, y_test)
    
    # Save best model
    print("\n3. Saving model...")
    model_path = save_model(best_model, best_model_name)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nBest model: {best_model_name.upper()}")
    print(f"Model saved to: {model_path}")
    print("\nYou can now run the web application:")
    print("  python run_app.py")
    print("=" * 60)

