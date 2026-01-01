"""
Exploratory Data Analysis for Loan Eligibility Dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir=None):
    """
    Perform comprehensive EDA on the loan dataset
    """
    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'eda_output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Basic Info
    print("\n1. Dataset Shape:", df.shape)
    print("\n2. Column Names:", df.columns.tolist())
    print("\n3. Data Types:\n", df.dtypes)
    print("\n4. Missing Values:\n", df.isnull().sum())
    print("\n5. Basic Statistics:\n", df.describe())
    
    # Target Distribution
    print("\n6. Target Distribution (loan_status):")
    print(df['loan_status'].value_counts())
    print("\nTarget Distribution (%):")
    print(df['loan_status'].value_counts(normalize=True) * 100)
    
    # Categorical Features Analysis
    categorical_features = ['person_education', 'person_home_ownership', 
                           'loan_intent', 'previous_loan_defaults_on_file']
    
    print("\n7. Categorical Features Analysis:")
    for col in categorical_features:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    # Numerical Features Distribution
    numerical_features = ['person_age', 'person_income', 'person_emp_exp',
                         'loan_amnt', 'loan_percent_income', 
                         'cb_person_cred_hist_length', 'credit_score']
    
    # Correlation Analysis
    print("\n8. Correlation Matrix:")
    corr_matrix = df[numerical_features + ['loan_status']].corr()
    print(corr_matrix['loan_status'].sort_values(ascending=False))
    
    # Visualizations
    # Try different matplotlib styles for compatibility
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Target Distribution
    df['loan_status'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['#e74c3c', '#2ecc71'])
    axes[0, 0].set_title('Loan Status Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Loan Status (0=Not Eligible, 1=Eligible)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticklabels(['Not Eligible', 'Eligible'], rotation=0)
    
    # Credit Score Distribution
    df['credit_score'].hist(bins=50, ax=axes[0, 1], color='#3498db', edgecolor='black')
    axes[0, 1].set_title('Credit Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Credit Score')
    axes[0, 1].set_ylabel('Frequency')
    
    # Loan Amount vs Income
    axes[1, 0].scatter(df['person_income'], df['loan_amnt'], 
                      alpha=0.5, c=df['loan_status'], cmap='RdYlGn')
    axes[1, 0].set_title('Loan Amount vs Person Income', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Person Income')
    axes[1, 0].set_ylabel('Loan Amount')
    
    # Correlation Heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes[1, 1], square=True)
    axes[1, 1].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_summary.png'), dpi=300, bbox_inches='tight')
    print(f"\nEDA plots saved to {output_dir}/eda_summary.png")
    
    # Additional Analysis
    print("\n9. Key Insights:")
    print(f"   - Average Credit Score: {df['credit_score'].mean():.2f}")
    print(f"   - Average Loan Amount: ${df['loan_amnt'].mean():,.2f}")
    print(f"   - Average Person Income: ${df['person_income'].mean():,.2f}")
    print(f"   - Average Loan Percent Income: {df['loan_percent_income'].mean():.2%}")
    
    return {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'target_distribution': df['loan_status'].value_counts().to_dict(),
        'correlations': corr_matrix['loan_status'].to_dict()
    }

if __name__ == "__main__":
    # Get project root and data path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, 'loan_dataS.csv')
    
    # Load data
    df = pd.read_csv(data_path)
    eda_results = perform_eda(df)
    print("\nEDA completed successfully!")

