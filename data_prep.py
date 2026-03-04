""" 
Data preparation for churn prediction models.
Handles loading, encoding, and splitting the data for training and testing.
"""
from os import path

from importlib.resources import path

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os

class ChurnDataPrep:
    """Preparing data for churn prodecition models with consistent preprocessing steps."""

    def __init__(self, data_path='data/ml_features.csv', test_size=0.2, random_state=42):
        """Initialize data preparation

        Args:
            data_path: Path to the cleaned features CSV
            test_size: Proportion of test set (default 0.2 = 20%)
            random_state: Random seed for reproducibility (default 42)
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        # These will be set during preprocessing
        self.churn_df = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.X_encoded = None
        self.feature_names = None
        self.categorical_cols = None

    def load_data(self):
        """Loading the cleaned feature data."""
        print(f"📊Loading data from {self.data_path}...")
        self.churn_df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.churn_df):,} customers")
        print(f"✓ Churn rate: {self.churn_df['churn'].mean()*100:.2f}%")
        return self 
    
    def encode_features(self):
        """Encoding categorical features using one-hot encoding"""
        # Separating features and target
        X = self.churn_df.drop(columns=['churn'])
        Y = self.churn_df['churn']

        # Finding categorical columns
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        print(f"\n✓ Encoding {len(self.categorical_cols)} categorical features:")
        for col in self.categorical_cols:
            print(f"  • {col}")

        # One-hot encoding 
        self.X_encoded = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
        self.feature_names = self.X_encoded.columns.tolist()

        print(f"✓ Total features after encoding: {len(self.feature_names)}")

        # Storing Y for splitting
        self.Y = Y

        return self
    
    def split_data(self):
        """Splitting data into training and testing sets."""
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_encoded,
            self.Y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.Y
        )

        print(f"\n✓ Train set: {len(self.X_train):,} customers ({self.Y_train.mean()*100:.1f}% churn)")
        print(f"✓ Test set:  {len(self.X_test):,} customers ({self.Y_test.mean()*100:.1f}% churn)")
        
        return self
    
    def prepare(self):
        """ Running full preparation pipeline: load → encode → split."""
        print("\n" + "="*70)
        print("🔧 PREPARING DATA FOR MODELING")
        print("="*70)
        
        self.load_data()
        self.encode_features()
        self.split_data()

        print("\n✓ Data preparation complete!")
        return self
    
    def get_train_test_data(self):
        """Returning train/test splits."""
        if self.X_train is None:
            raise ValueError("Data not prepared yet. Call prepare() first.")
        
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    def save_preprocessor(self, path='data/preprocessor.pkl'):
        """Saving the preprocessor for future use (feature names, categorical cols, etc.)"""
        preprocessor_info = {
           'feature_names': self.feature_names,
            'categorical_cols': self.categorical_cols,
            'test_size': self.test_size,
            'random_state': self.random_state
       }
        
        with open(path, 'wb')as f:
            pickle.dump(preprocessor_info, f)
        
        print(f"✓ Preprocessor saved to: {path}")
    
    @staticmethod
    def load_preprocessor(path='data/preprocessor.pkl'):
        """Loading a saved preprocessor."""
        with open(path, 'rb') as f:
            preprocessor_info = pickle.load(f)
        
        print(f"✓ Loaded preprocessor from {path}")
        return preprocessor_info
    
# Convenience funtion for quick use
def prepare_data(data_pat='data/ml_features.csv'):
    """Quick data preparation with default settings."""
    prep = ChurnDataPrep(data_path=data_pat)
    prep.prepare()
    return prep.get_train_test_data()
