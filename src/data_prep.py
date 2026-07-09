""" 
Data preparation for churn prediction models.
Loads engineered features from dbt (fct_customer_churn_features)
and prepares train/test splits for modeling. 

This module only handles: loading -> encoding -> splitting. 
"""
import pandas as pd
import duckdb
import pickle
import os
from sklearn.model_selection import train_test_split

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(_THIS_DIR, '..', 'data')

DEFAULT_DUCKDB_PATH      = os.path.join(_DATA_DIR, 'churn_dev.duckdb')
DEFAULT_PREPROCESSOR_PATH = os.path.join(_DATA_DIR, 'preprocessor.pkl')


class ChurnDataPrep:
    """Preparing data for churn prediction using dbt-engineered features."""

    def __init__(
            self, 
            duckdb_path=None, 
            test_size=0.2, 
            random_state=42):
        
        """Initialize data preparation

        Args:
            duckdb_path: Path to the DuckDB database (dbt output)
            test_size: Proportion of test set (default 0.2 = 20%)
            random_state: Random seed for reproducibility (default 42)
        """
        self.duckdb_path = duckdb_path or DEFAULT_DUCKDB_PATH
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

    # STEP 1: LOADING DATA FROM DBT

    def load_data(self):
        """Loading engineered features from dbt fct_customer_churn_features."""
        print(f"📊Loading data from DuckDB{self.duckdb_path}...")

        conn = duckdb.connect(self.duckdb_path)
        self.churn_df = conn.execute("""
            SELECT * FROM fct_customer_churn_features
        """).df()
        conn.close()

        print(f"✓ Loaded {len(self.churn_df):,} customers")
        print(f"✓ Churn rate: {self.churn_df['churn'].mean()*100:.2f}%")
        print(f"✓ Features available: {len(self.churn_df.columns)}")
        return self 
    
    # STEP 2: SELECTING FEATURES

    def select_features(self):
        """
        Selecting modeling features from dbt output.
        Excludes: primary key, metadata, dashboard-only columns.
        """
    
        # Columns to exclude from modeling
        exclude_cols = [
            'customer_id',                      # primary key - not a feature
            'churn',                            # target variable - handled separately
            'loaded_at',                        # metadata - not a feature
            'features_generated_at',            # metadata - not a feature
            'tenure_segment',                   # dashboard-only - lifecycle_stage covers this
            'recency_bin',
            'demographic_stability',
            'is_single',                         # keeping it as OHE
            'is_married',                        # keeping it as OHE
            'is_divorced',                       # keeping it as OHE
        ]

        # Adding any timestamp/datetime columns automatically
        timestamp_cols = self.churn_df.select_dtypes(
            include=['datetime64']
        ).columns.tolist()

        exclude_cols = list(set(exclude_cols + timestamp_cols))

        print(f"Excluding: {exclude_cols}")

        # Categorical columns - need encoding
        # Raw strings kept in fct for dashboard use
        self.categorical_cols = [
            'preferred_order_category',         # will be target encoded
            'marital_status',                   # will be OHE
            'lifecycle_stage',                  # will be OHE
            'recency_risk',                     # will be OHE
            'product_risk_category',            # will be OHE
            'value_tier',                       # will be OHE
        ]

        feature_cols = [
            c for c in self.churn_df.columns 
            if c not in exclude_cols
        ]

        self.X = self.churn_df[feature_cols]
        self.Y = self.churn_df['churn']

        print(f"\n✓ Features selected: {len(feature_cols)}")
        print(f"    Numeric:       {len([c for c in feature_cols if c not in self.categorical_cols])}")
        print(f"    Categorical:   {len(self.categorical_cols)}")
        return self
    
    # STEP 3: ENCODING FEATURES

    def encode_features(self):
        """
        Encoding categorical features.
        - preferred_order_category: target encoding (churn rate per category)
        - all other: one-hot encoding
        
        IMPORTANT: target encoding computed on full dataset here for EDA.
        During production modeling it should be computed on train set only to avoid data leakage.
        """
        X = self.X.copy()

        # Target encoding preferred_order_category
        # Uses churn rate per category as numeric signal
        print("\n✓ Target encoding: preferred_order_category")
        self.target_encodings = (
            self.churn_df.groupby('preferred_order_category')['churn']
            .mean()
            .to_dict()
        )
        X['preferred_order_category'] = (
            X['preferred_order_category'].map(self.target_encodings)
        )
        X = X.drop(columns=['preferred_order_category'])

        # OHE remaining categoricals
        remaining_cats = [
            c for c in self.categorical_cols
            if c != 'preferred_order_category'
        ]
        print(f"✓ One-hot encoding: {len(remaining_cats)}")
        self.X_encoded = pd.get_dummies(
            X,
            columns=remaining_cats,
            drop_first=False,                  # tree models do not need drop_first
            dtype=int
        )

        self.feature_names = self.X_encoded.columns.tolist()
        print(f"✓ Total features after encoding: {len(self.feature_names)}")
        return self

    # STEP 4: SPLITTING INTO TRAIN/TEST
    
    def split_data(self):
        """Splitting data into train/test with stratification."""
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_encoded,
            self.Y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.Y                # ensures same churn rate in train and test sets
        )

        print(f"\n✓ Train set: {len(self.X_train):,} customers "
              f"({self.Y_train.mean()*100:.1f}% churn)")
        print(f"✓ Test set:  {len(self.X_test):,} customers "
              f"({self.Y_test.mean()*100:.1f}% churn)")
        return self
    
    # FULL PREPARATION PIPELINE
    
    def prepare(self):
        """ Running full preparation pipeline: load → select → encode → split."""
        print("\n" + "="*70)
        print("🔧 PREPARING DATA FOR MODELING")
        print("="*70)
        
        self.load_data()
        self.select_features()
        self.encode_features()
        self.split_data()

        print("\n✓ Data preparation complete!")
        return self
    
    def get_train_test_data(self):
        """Returning train/test splits."""
        if self.X_train is None:
            raise ValueError("Data not prepared yet. Call prepare() first.")
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    # SAVE / LOAD PREPROCESSOR 

    def save_preprocessor(self, path=None):
        """Saving the preprocessor for future use (feature names, categorical cols, etc.)"""
        path = path or DEFAULT_PREPROCESSOR_PATH
        preprocessor_info = {
           'feature_names': self.feature_names,
            'categorical_cols': self.categorical_cols,
            'target_encodings': self.target_encodings,
            'test_size': self.test_size,
            'random_state': self.random_state
       }
        
        with open(path, 'wb')as f:
            pickle.dump(preprocessor_info, f)
        print(f"✓ Preprocessor saved to: {path}")
    
    @staticmethod
    def load_preprocessor(path=None):
        """Loading a saved preprocessor."""
        path = path or DEFAULT_PREPROCESSOR_PATH
        with open(path, 'rb') as f:
            preprocessor_info = pickle.load(f)
        print(f"✓ Loaded preprocessor from {path}")
        return preprocessor_info
    
    # TRANSFORMING NEW DATA (INFERENCE)
    
    @staticmethod
    def transform_new_data(new_data, preprocessor_path=None):
        """
        Transform new data using saved preprocessor settings.
        Applies SAME encoding as training data.
        
        Args:
            new_data: DataFrame or dict with raw customer features
            preprocessor_path: Path to saved preprocessor.pkl
        
        Returns:
            DataFrame with encoded features matching training data
        
        Note: Feature engineering (lifecycle_stage, recency_risk, etc.)
        should come from dbt fct_customer_churn_feature.
        This method handles encoding only.
        """
        preprocessor_path = preprocessor_path or DEFAULT_PREPROCESSOR_PATH
        # Loading preprocessor info
        with open(preprocessor_path, 'rb') as f:
            preprocessor_info = pickle.load(f)
        
        # Converting dict to DataFrame if needed
        if isinstance(new_data, dict):
            df = pd.DataFrame([new_data])
        else:
            df = new_data.copy()

        # Target encoding preferred_order_category
        target_encodings = preprocessor_info['target_encodings']
        df['preferred_order_category'] = (
            df['preferred_order_category']
            .map(target_encodings)
            .fillna(target_encodings.get('Others', 0.07))  # fallback for unknown
        )
        df = df.drop(columns=['preferred_order_category'])

        # OHE remaining categoricals
        remaining_cats = [
            c for c in preprocessor_info['categorical_cols']
            if c != 'preferred_order_category'
        ]
        df_encoded = pd.get_dummies(
            df,
            columns=remaining_cats,
            drop_first=False,                  
            dtype=int
        )
        
        # Aligning columns with training data
        expected = preprocessor_info['feature_names']

        # Adding missing columns with 0
        for col in expected:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Removing extra columns
        df_encoded = df_encoded[expected]

        return df_encoded
        
    
# Convenience funtion for quick use
def prepare_data(duckdb_path=None):
    """Quick data preparation with default settings."""
    prep = ChurnDataPrep(duckdb_path=duckdb_path)
    prep.prepare()
    return prep.get_train_test_data()
