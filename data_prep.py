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
    
    @staticmethod
    def transform_new_data(new_data, preprocessor_path='data/preprocessor.pkl'):
        """
        Transform new data using saved preprocessor settings.
        
        This applies the SAME one-hot encoding as training data and ensures
        all features match exactly.
        
        Args:
            new_data: DataFrame or dict with raw customer features
            preprocessor_path: Path to saved preprocessor.pkl
        
        Returns:
            DataFrame with encoded features matching training data
        
        Required input features:
        - tenure
        - warehouse_to_home
        - num_devices_registered
        - satisfaction_score
        - num_addresses
        - days_since_last_order
        - cashback_amount
        - preferred_order_category
        - marital_status
        - has_complained
        - tenure_missing_flag (optional, defaults to 0)
        """
        # Loading preprocessor info
        with open(preprocessor_path, 'rb') as f:
            preprocessor_info = pickle.load(f)
        
        # Converting dict to DataFrame if needed
        if isinstance(new_data, dict):
            df = pd.DataFrame([new_data])
        else:
            df = new_data.copy()

        # Ensure tenure_missing_flag exists
        if 'tenure_missing_flag' not in df.columns:
            df['tenure_missing_flag'] = 0
        
        # ========================================
        # FEATURE ENGINEERING (matching dbt models!)
        # ========================================
        
        # 1. days_missing_flag (if days_since_last_order is missing)
        df['days_missing_flag'] = df['days_since_last_order'].isna().astype(int)
        
        # Fill missing days_since_last_order if needed
        if df['days_since_last_order'].isna().any():
            df['days_since_last_order'] = df['days_since_last_order'].fillna(
                df['days_since_last_order'].median()
            )
        
        # 2. lifecycle_stage (based on tenure)
        def get_lifecycle_stage(tenure):
            if tenure == 0:
                return 'brand_new'
            elif tenure <= 3:
                return 'new'
            elif tenure <= 6:
                return 'growing'
            elif tenure <= 12:
                return 'established'
            else:
                return 'loyal'
        
        df['lifecycle_stage'] = df['tenure'].apply(get_lifecycle_stage)
        
        # 3. is_new_customer (tenure <= 3 months)
        df['is_new_customer'] = (df['tenure'] <= 3).astype(int)
        
        # 4. is_established (tenure >= 7 months)
        df['is_established'] = (df['tenure'] >= 7).astype(int)
        
        # 5. tenure_segment (categorical binning)
        def get_tenure_segment(tenure):
            if tenure == 0:
                return 'brand_new_0mo'
            elif tenure <= 3:
                return '1-3_months'
            elif tenure <= 6:
                return '4-6_months'
            elif tenure <= 12:
                return '7-12_months'
            elif tenure <= 24:
                return '1-2_years'
            else:
                return '2+_years'
        
        df['tenure_segment'] = df['tenure'].apply(get_tenure_segment)
        
        # 6. recency_risk (based on days_since_last_order)
        def get_recency_risk(days):
            if days <= 7:
                return 'very_recent'
            elif days <= 15:
                return 'recent'
            elif days <= 30:
                return 'moderate'
            else:
                return 'dormant'
        
        df['recency_risk'] = df['days_since_last_order'].apply(get_recency_risk)
        
        # 7. is_very_recent (0-7 days)
        df['is_very_recent'] = (df['days_since_last_order'] <= 7).astype(int)
        
        # 8. is_dormant (31+ days)
        df['is_dormant'] = (df['days_since_last_order'] > 30).astype(int)
        
        # 9. is_active_stable (15-30 days sweet spot)
        df['is_active_stable'] = (
            (df['days_since_last_order'] >= 15) & 
            (df['days_since_last_order'] <= 30)
        ).astype(int)
        
        # 10. recency_bin (for U-shaped pattern)
        def get_recency_bin(days):
            if days <= 7:
                return '0-7_days'
            elif days <= 14:
                return '8-14_days'
            elif days <= 30:
                return '15-30_days'
            else:
                return '31+_days'
        
        df['recency_bin'] = df['days_since_last_order'].apply(get_recency_bin)
        
        # 11. product_risk_category (based on preferred_order_category)
        product_risk_map = {
            'Mobile Phone': 'high_risk',
            'Laptop & Accessory': 'high_risk',
            'Fashion': 'medium_risk',
            'Others': 'medium_risk',
            'Grocery': 'low_risk'
        }
        df['product_risk_category'] = df['preferred_order_category'].map(
            product_risk_map
        ).fillna('medium_risk')
        
        # 12. is_tech_buyer (Mobile or Laptop)
        df['is_tech_buyer'] = df['preferred_order_category'].isin(
            ['Mobile Phone', 'Laptop & Accessory']
        ).astype(int)
        
        # 13. is_grocery_buyer
        df['is_grocery_buyer'] = (
            df['preferred_order_category'] == 'Grocery'
        ).astype(int)
        
        # 14. is_fashion_buyer
        df['is_fashion_buyer'] = (
            df['preferred_order_category'] == 'Fashion'
        ).astype(int)
        
        # 15. demographic_stability (based on marital_status)
        df['demographic_stability'] = df['marital_status'].apply(
            lambda x: 'stable' if x == 'Married' else 'unstable'
        )
        
        # 16. is_single
        df['is_single'] = (df['marital_status'] == 'Single').astype(int)
        
        # 17. is_married
        df['is_married'] = (df['marital_status'] == 'Married').astype(int)
        
        # 18. is_divorced
        df['is_divorced'] = (df['marital_status'] == 'Divorced').astype(int)
        
        # 19. value_tier (based on cashback_amount)
        def get_value_tier(cashback):
            if cashback < 100:
                return 'minimal_value'
            elif cashback < 200:
                return 'low_value'
            elif cashback < 300:
                return 'medium_value'
            else:
                return 'high_value'
        
        df['value_tier'] = df['cashback_amount'].apply(get_value_tier)
        
        # 20. cashback_per_month (TOP FEATURE! 24% importance)
        df['cashback_per_month'] = df['cashback_amount'] / (df['tenure'] + 1)
        
        # 21. is_high_value (cashback >= 300)
        df['is_high_value'] = (df['cashback_amount'] >= 300).astype(int)
        
        # 22. is_low_spender (cashback < 100)
        df['is_low_spender'] = (df['cashback_amount'] < 100).astype(int)
        
        # 23. composite_risk_score (0-5 scale based on 5 risk factors)
        risk_score = 0
        
        # Risk factor 1: New customer (tenure <= 3)
        risk_score += (df['tenure'] <= 3).astype(int)
        
        # Risk factor 2: High-risk product (tech)
        risk_score += df['is_tech_buyer']
        
        # Risk factor 3: Has complained
        risk_score += df['has_complained']
        
        # Risk factor 4: Single
        risk_score += df['is_single']
        
        # Risk factor 5: Dormant (31+ days)
        risk_score += df['is_dormant']
        
        df['composite_risk_score'] = risk_score
            
        # Getting settings from preprocessor
        categorical_cols = preprocessor_info['categorical_cols']
        expected_features = preprocessor_info['feature_names']
        
        # Applying one-hot encoding (SAME as training!)
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Adding missing columns with 0 (categories that didn't appear in new data)
        for col in expected_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Removing extra columns (categories that appeared in new data but not training)
        df_encoded = df_encoded[expected_features]
        
        # Ensuring correct dtypes (all numeric)
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                # This shouldn't happen, but just in case
                df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        
        return df_encoded
    
# Convenience funtion for quick use
def prepare_data(data_pat='data/ml_features.csv'):
    """Quick data preparation with default settings."""
    prep = ChurnDataPrep(data_path=data_pat)
    prep.prepare()
    return prep.get_train_test_data()
