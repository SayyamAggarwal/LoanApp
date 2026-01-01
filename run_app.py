"""
Main entry point for the Loan Eligibility Web Application
"""
import os
import sys

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.backend.app import app, load_model_and_preprocessor

if __name__ == '__main__':
    print("=" * 60)
    print("Loan Eligibility & Advisory System")
    print("=" * 60)
    print("\nLoading model and preprocessor...")
    
    if load_model_and_preprocessor():
        print("\n" + "=" * 60)
        print("Starting Flask server...")
        print("Application will be available at: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n" + "=" * 60)
        print("ERROR: Failed to load model and preprocessor!")
        print("=" * 60)
        print("\nPlease train the model first by running:")
        print("  python app/ml_pipeline/train_model.py")
        print("\n" + "=" * 60)
        sys.exit(1)



