"""
Feature Engineering and Preprocessing for Loan Eligibility Model
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

class LoanPreprocessor:
    """
    Preprocessing pipeline for loan eligibility data
    Uses OneHotEncoding for categorical features to prevent ordinal bias
    """
    
    def __init__(self):
        self.label_encoders = {}  # Keep for backward compatibility if needed
        self.onehot_encoders = {}  # New: OneHot encoders for categorical features
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_features = ['person_education', 'person_home_ownership', 
                                    'loan_intent', 'previous_loan_defaults_on_file']
        
    def fit_transform(self, df, target_col='loan_status'):
        """
        Fit and transform the training data
        Uses OneHotEncoding for categorical features to ensure fair representation
        """
        df = df.copy()
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Separate numerical and categorical features
        numerical_features = [col for col in X.columns if col not in self.categorical_features]
        
        # Apply OneHotEncoding to categorical features (prevents ordinal bias)
        X_encoded = X.copy()
        onehot_feature_names = []
        
        for col in self.categorical_features:
            if col in X.columns:
                # Create OneHotEncoder
                ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded = ohe.fit_transform(X[[col]])
                
                # Get feature names
                feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]  # Skip first category (drop='first')
                onehot_feature_names.extend(feature_names)
                
                # Store encoder
                self.onehot_encoders[col] = ohe
                
                # Replace original column with encoded columns
                X_encoded = X_encoded.drop(columns=[col])
                for i, feat_name in enumerate(feature_names):
                    X_encoded[feat_name] = encoded[:, i]
        
        # Feature Engineering (on numerical features)
        X_encoded = self._feature_engineering(X_encoded)
        
        # Store feature names
        self.feature_names = X_encoded.columns.tolist()
        
        # Scale all features (including one-hot encoded)
        X_scaled = self.scaler.fit_transform(X_encoded)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled, y
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessors
        """
        df = df.copy()
        
        # Apply OneHotEncoding to categorical features
        X_encoded = df.copy()
        
        for col in self.categorical_features:
            if col in df.columns and col in self.onehot_encoders:
                ohe = self.onehot_encoders[col]
                encoded = ohe.transform(df[[col]])
                
                # Get feature names
                feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                
                # Remove original column and add encoded columns
                X_encoded = X_encoded.drop(columns=[col])
                for i, feat_name in enumerate(feature_names):
                    X_encoded[feat_name] = encoded[:, i]
        
        # Feature Engineering
        X_encoded = self._feature_engineering(X_encoded)
        
        # Ensure all columns are present (fill missing one-hot columns with 0)
        for col in self.feature_names:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        # Reorder columns to match training
        X_encoded = X_encoded[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X_encoded)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled
    
    def _feature_engineering(self, df):
        """
        Create engineered features
        """
        # Income to Loan Ratio
        df['income_to_loan_ratio'] = df['person_income'] / (df['loan_amnt'] + 1)
        
        # Credit Score Categories (normalized)
        df['credit_score_normalized'] = df['credit_score'] / 850.0
        
        # Age Groups (normalized)
        df['age_normalized'] = df['person_age'] / 100.0
        
        # Employment Experience Ratio
        df['emp_exp_ratio'] = df['person_emp_exp'] / (df['person_age'] + 1)
        
        # Debt to Income (already have loan_percent_income, but create additional)
        df['debt_income_ratio'] = df['loan_percent_income']
        
        # Credit History Ratio
        df['cred_hist_ratio'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)
        
        # Risk Score (composite)
        df['risk_score'] = (
            (1 - df['credit_score_normalized']) * 0.4 +
            df['loan_percent_income'] * 0.3 +
            (1 - df['income_to_loan_ratio'] / df['income_to_loan_ratio'].max()) * 0.3
        )
        
        return df
    
    def save(self, filepath=None):
        """
        Save the preprocessor
        """
        if filepath is None:
            # Get project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            filepath = os.path.join(project_root, 'app', 'models', 'preprocessor.pkl')
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'label_encoders': self.label_encoders,  # Keep for backward compatibility
            'onehot_encoders': self.onehot_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features
        }, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath=None):
        """
        Load the preprocessor
        """
        if filepath is None:
            # Get project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            filepath = os.path.join(project_root, 'app', 'models', 'preprocessor.pkl')
        
        data = joblib.load(filepath)
        self.label_encoders = data.get('label_encoders', {})  # Backward compatibility
        self.onehot_encoders = data.get('onehot_encoders', {})
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.categorical_features = data.get('categorical_features', 
                                            ['person_education', 'person_home_ownership', 
                                             'loan_intent', 'previous_loan_defaults_on_file'])
        print(f"Preprocessor loaded from {filepath}")

def prepare_data(data_path=None, test_size=0.2, random_state=42, apply_smote=True, smote_k_neighbors=5):
    """
    Prepare train-test split with optional SMOTE balancing
    
    Args:
        data_path: Path to the dataset CSV file
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        apply_smote: Whether to apply SMOTE for balancing (default: True)
        smote_k_neighbors: Number of nearest neighbors for SMOTE (default: 5)
    """
    # Get project root and data path
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_path = os.path.join(project_root, 'loan_dataS.csv')
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor
    preprocessor = LoanPreprocessor()
    
    # Fit and transform
    X, y = preprocessor.fit_transform(df)
    
    # Train-test split (before SMOTE to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply SMOTE to training data only (not test data)
    if apply_smote:
        print("\n" + "=" * 60)
        print("APPLYING SMOTE FOR DATA BALANCING")
        print("=" * 60)
        print(f"Before SMOTE - Training set distribution:")
        print(y_train.value_counts().sort_index())
        print(f"Class ratio: {y_train.value_counts()[0] / y_train.value_counts()[1]:.2f}:1")
        
        # Initialize SMOTE
        # Calculate k_neighbors safely (must be less than minority class samples)
        minority_count = y_train.value_counts().min()
        safe_k_neighbors = min(smote_k_neighbors, max(1, minority_count - 1))
        
        smote = SMOTE(
            random_state=random_state,
            k_neighbors=safe_k_neighbors,
            sampling_strategy=1.0  # Balance to 1:1 ratio (equal to majority class)
        )
        
        # Apply SMOTE
        try:
            # Convert to numpy for SMOTE (SMOTE works better with numpy arrays)
            X_train_np = X_train.values
            y_train_np = y_train.values
            
            # Apply SMOTE
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_np, y_train_np)
            
            # Convert back to DataFrame/Series
            X_train = pd.DataFrame(X_train_balanced, columns=X_train.columns)
            y_train = pd.Series(y_train_balanced, name=y_train.name)
            
            print(f"\nAfter SMOTE - Training set distribution:")
            print(y_train.value_counts().sort_index())
            print(f"Class ratio: {y_train.value_counts()[0] / y_train.value_counts()[1]:.2f}:1")
            print(f"Training set size increased from {len(X_train_np)} to {len(X_train)} samples")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"Warning: SMOTE failed with error: {e}")
            print("Continuing with original imbalanced data...")
    
    # Save preprocessor
    preprocessor.save()
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Train target distribution:\n{y_train.value_counts()}")
    print(f"Test target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    print("\nData preprocessing completed successfully!")

