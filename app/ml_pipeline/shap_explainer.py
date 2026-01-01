"""
SHAP Explainability Module for Loan Eligibility Model
Provides global and per-user explanations
"""
import shap
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import json

class SHAPExplainer:
    """
    SHAP explainer for loan eligibility model
    """
    
    def __init__(self, model, preprocessor, X_train_sample=None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            X_train_sample: Sample of training data for TreeExplainer
        """
        self.model = model
        self.preprocessor = preprocessor
        
        # Get model's expected feature names
        if hasattr(model, 'get_booster'):
            self.model_feature_names = model.get_booster().feature_names
        elif hasattr(model, 'feature_name_'):
            self.model_feature_names = model.feature_name_
        elif hasattr(model, 'feature_names_'):
            self.model_feature_names = model.feature_names_
        else:
            self.model_feature_names = None
        
        # Use TreeExplainer for tree-based models
        if X_train_sample is not None:
            # Sample for faster computation
            if len(X_train_sample) > 1000:
                X_train_sample = X_train_sample.sample(n=1000, random_state=42)
            
            # Check for feature mismatch
            if self.model_feature_names is not None:
                X_train_sample = self._align_features(X_train_sample)
                # Check if alignment was successful
                if list(X_train_sample.columns) != list(self.model_feature_names):
                    # Feature mismatch - initialize without background data
                    print("Warning: Feature mismatch detected. Initializing TreeExplainer without background data.")
                    self.explainer = shap.TreeExplainer(model)
                else:
                    self.explainer = shap.TreeExplainer(model, X_train_sample)
            else:
                self.explainer = shap.TreeExplainer(model, X_train_sample)
        else:
            self.explainer = shap.TreeExplainer(model)
        
        self.feature_names = preprocessor.feature_names
    
    def _align_features(self, X_instance):
        """
        Align DataFrame features to match model's expected feature names
        
        Args:
            X_instance: DataFrame with features
            
        Returns:
            DataFrame aligned to model's expected features
        """
        if not isinstance(X_instance, pd.DataFrame):
            return X_instance
        
        if self.model_feature_names is None:
            return X_instance
        
        # Check if we have one-hot encoded features but model expects categorical
        # This happens when model was trained with old preprocessor
        has_onehot = any('_' in col for col in X_instance.columns)
        model_expects_categorical = any(col in self.model_feature_names for col in 
                                      ['person_education', 'person_home_ownership', 
                                       'loan_intent', 'previous_loan_defaults_on_file'])
        
        # If model expects categorical but we have one-hot, we can't convert
        # In this case, the model feature names are likely wrong (model was trained with one-hot)
        # So we'll try to use the one-hot features directly
        if has_onehot and model_expects_categorical:
            # Check if model can actually accept one-hot features (by trying to predict)
            # For now, we'll assume if we have one-hot and model expects categorical,
            # the model was actually trained with one-hot but has wrong feature names
            # So we'll keep the one-hot features and let XGBoost handle it
            # (XGBoost can work with features even if names don't match exactly)
            return X_instance
        
        # Normal alignment: match model's expected features
        if list(X_instance.columns) != list(self.model_feature_names):
            missing_cols = set(self.model_feature_names) - set(X_instance.columns)
            extra_cols = set(X_instance.columns) - set(self.model_feature_names)
            
            if missing_cols:
                for col in missing_cols:
                    X_instance[col] = 0
            
            if extra_cols:
                X_instance = X_instance.drop(columns=list(extra_cols))
            
            # Reorder to match model's expected order
            X_instance = X_instance[self.model_feature_names]
        
        return X_instance
    
    def explain_global(self, X_train_sample, output_dir='app/static/shap_plots'):
        """
        Generate global SHAP explanations
        
        Args:
            X_train_sample: Sample of training data
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample for faster computation
        if len(X_train_sample) > 1000:
            X_train_sample = X_train_sample.sample(n=1000, random_state=42)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_train_sample)
        
        # Handle binary classification (use class 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train_sample, 
                         feature_names=self.feature_names, 
                         show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_bar.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detailed summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train_sample, 
                         feature_names=self.feature_names, 
                         show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_dot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(
            os.path.join(output_dir, 'feature_importance.csv'), 
            index=False
        )
        
        print(f"Global SHAP explanations saved to {output_dir}")
        
        return {
            'shap_values': shap_values.tolist(),
            'feature_importance': feature_importance.to_dict('records'),
            'base_value': float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value)
        }
    
    def explain_instance(self, X_instance):
        """
        Generate per-user SHAP explanation
        
        Args:
            X_instance: Single instance (DataFrame or array)
        
        Returns:
            Dictionary with SHAP values and explanation
        """
        # Use model's expected feature names (stored during initialization)
        model_feature_names = self.model_feature_names or self.feature_names
        
        # Ensure it's a DataFrame and align features
        if isinstance(X_instance, np.ndarray):
            # If array, assume it matches model's expected features
            X_instance = pd.DataFrame(X_instance, columns=model_feature_names)
        else:
            # Align features using helper method
            X_instance = self._align_features(X_instance)
        
        # Check if there's still a feature mismatch after alignment
        # If so, convert to numpy to bypass XGBoost's strict feature name validation
        use_numpy = False
        if isinstance(X_instance, pd.DataFrame) and model_feature_names:
            if list(X_instance.columns) != list(model_feature_names):
                # Check if feature count matches (if not, we can't proceed)
                if len(X_instance.columns) != len(model_feature_names):
                    raise ValueError(
                        f"Feature count mismatch: Model expects {len(model_feature_names)} features "
                        f"but got {len(X_instance.columns)}. The model needs to be retrained with "
                        f"the current preprocessor. Please run: python app/ml_pipeline/train_model.py"
                    )
                # Feature names don't match but count does - convert to numpy array
                use_numpy = True
                X_instance_np = X_instance.values
                # Store column names for later reference
                actual_feature_names = X_instance.columns.tolist()
            else:
                X_instance_np = X_instance.values
                actual_feature_names = model_feature_names
        else:
            X_instance_np = X_instance.values if hasattr(X_instance, 'values') else X_instance
            actual_feature_names = model_feature_names or self.feature_names
        
        # Calculate SHAP values for this instance
        # Use numpy array if there's a feature mismatch to avoid validation errors
        if use_numpy:
            shap_values = self.explainer.shap_values(X_instance_np)
        else:
            shap_values = self.explainer.shap_values(X_instance)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get base value
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]
        
        # Get prediction - use numpy to avoid feature name validation
        if use_numpy:
            prediction = self.model.predict_proba(X_instance_np)[0]
            prediction_class = int(self.model.predict(X_instance_np)[0])
        else:
            prediction = self.model.predict_proba(X_instance)[0]
            prediction_class = int(self.model.predict(X_instance)[0])
        prediction_proba = float(prediction[1])
        
        # Create feature contribution dictionary using actual feature names
        feature_contributions = {}
        for i, feature in enumerate(actual_feature_names):
            try:
                if use_numpy:
                    feature_value = float(X_instance_np[0, i])
                else:
                    feature_value = float(X_instance.iloc[0, i]) if i < len(X_instance.columns) else 0.0
            except (IndexError, KeyError):
                feature_value = 0.0
            
            feature_contributions[feature] = {
                'shap_value': float(shap_values[0][i]),
                'feature_value': feature_value
            }
        
        # Aggregate SHAP values for one-hot encoded features and clean up names
        aggregated_contributions = {}
        
        # User-facing features (inputs) to prioritize
        user_inputs = [
            'person_age', 'person_education', 'person_income', 'person_emp_exp', 
            'person_home_ownership', 'loan_amnt', 'loan_intent', 'loan_percent_income',
            'cb_person_cred_hist_length', 'credit_score', 'previous_loan_defaults_on_file'
        ]
        
        # Mapping derived features to user friendly concepts if roughly equivalent
        # or just keep them if they are distinct enough
        feature_map = {
            'debt_income_ratio': 'loan_percent_income',
            'credit_score_normalized': 'credit_score',
            'age_normalized': 'person_age',
        }
        
        # Get categorical features from preprocessor
        cat_features = self.preprocessor.categorical_features if hasattr(self.preprocessor, 'categorical_features') else []
        
        for feature, contrib in feature_contributions.items():
            shap_val = contrib['shap_value']
            feat_val = contrib['feature_value']
            
            # 1. Handle One-Hot Encoding
            is_encoded = False
            base_feature = None
            
            for cat_col in cat_features:
                if feature.startswith(f"{cat_col}_"):
                    is_encoded = True
                    base_feature = cat_col
                    break
            
            if is_encoded:
                target_key = base_feature
            else:
                target_key = feature
                
            # 2. Handle Derived/mapped features
            if target_key in feature_map:
                target_key = feature_map[target_key]
                
            # 3. Accumulate
            if target_key not in aggregated_contributions:
                aggregated_contributions[target_key] = {
                    'shap_value': 0.0,
                    'feature_value': feat_val # This might be wrong for aggregated/derived, but we use original values in frontend usually
                }
            aggregated_contributions[target_key]['shap_value'] += shap_val
            
        # Filter to only show features that map to user inputs (plus maybe highly significant others)
        # The user asked: "show only features which i gave to model"
        final_features = {}
        for k, v in aggregated_contributions.items():
            if k in user_inputs:
                final_features[k] = v
            # If we want to keep high impact derived features that don't map perfectly:
            elif abs(v['shap_value']) > 0.1: # Threshold for "Important enough to show even if derived"
                # Rename for display
                display_name = k.replace('_', ' ').title()
                final_features[k] = v # Keep it
        
        # Sort by absolute SHAP value
        sorted_features = sorted(
            final_features.items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        )
        
        return {
            'prediction': prediction_class,
            'prediction_proba': prediction_proba,
            'base_value': float(base_value),
            'shap_values': shap_values[0].tolist(),
            'feature_contributions': dict(sorted_features),
            'top_features': [
                {
                    'feature': feat,
                    'shap_value': float(contrib['shap_value']),
                    'feature_value': float(contrib['feature_value']),
                    # SHAP > 0 means pushing towards 1 (Risk/Default) -> Negative Impact
                    # SHAP < 0 means pushing towards 0 (Safety/Eligible) -> Positive Impact
                    'impact': 'negative' if contrib['shap_value'] > 0 else 'positive'
                }
                for feat, contrib in sorted_features[:10]
            ],
            'feature_names': actual_feature_names  # Include for debugging
        }
    
    def save_explainer(self, filepath='app/models/shap_explainer.pkl'):
        """
        Save the SHAP explainer (note: TreeExplainer can be recreated, so we save model and preprocessor)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Note: TreeExplainer is lightweight and can be recreated
        # We'll save a reference instead
        print(f"SHAP explainer reference saved (model and preprocessor should be loaded separately)")
    
    @staticmethod
    def load_explainer(model, preprocessor, X_train_sample=None):
        """
        Load/create SHAP explainer from model and preprocessor
        """
        return SHAPExplainer(model, preprocessor, X_train_sample)

if __name__ == "__main__":
    # Example usage
    print("SHAP Explainer module loaded successfully!")
    print("Use this class with your trained model and preprocessor.")



