"""
Test script to verify prediction pipeline works correctly.
Uses dbt feature engineering logic replicated in Python
for real-time inference without requiring dbt pipeline.
"""
print("Script started...")
import joblib
import pandas as pd
import numpy as np
import os
import sys

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.append(BASE_DIR)

# Importing from src
from src.data_prep import ChurnDataPrep
from src.features import engineer_features

print("\n" + "="*70)
print("🧪 TESTING PREDICTION PIPELINE")
print("="*70)

# LOAD MODEL & PREPROCESSOR
print("\n1️⃣ Loading model and preprocessor...")
model = joblib.load(os.path.join(DATA_DIR, 'best_model.pkl'))
preprocessor = ChurnDataPrep.load_preprocessor(
    path=os.path.join(DATA_DIR, 'preprocessor.pkl')
)
print("✓ Model and preprocessor loaded successfully!")
print(f"✓ Expected features: {len(preprocessor['feature_names'])}")


def predict_churn(customer: dict, model, preprocessor, threshold=0.3) -> dict:
    """
    Lower threshold catches more churners (higher recall)
    Full prediction pipeline for a single customer.
    1. Engineer features
    2. Preprocess data
    3. Predict churn and probabilities
    """
    # Step 1: Feature engineering
    engineered = engineer_features(customer)

    # Step 2: Convert to DataFrame
    df = pd.DataFrame([engineered])

    # Step 3: Preprocess
    X = ChurnDataPrep.transform_new_data(
        df,
        preprocessor_path=os.path.join(DATA_DIR, 'preprocessor.pkl')
    )

    # Step 4: Predict - using custom threshold for classification 
    probabilities = model.predict_proba(X)[0]
    churn_prob = float(probabilities[1])

    prediction = int(churn_prob >= threshold)

    return {
        'prediction': int(prediction),
        'churn_probability': round(float(probabilities[1]) * 100, 2),
        'retention_probability': round(float(probabilities[0]) * 100, 2),
        'threshold_used': threshold,
        'lifecycle_stage': engineered['lifecycle_stage'],
        'recency_risk': engineered['recency_risk'],
        'product_risk': engineered['product_risk_category'],
        'value_tier': engineered['value_tier'],
        'composite_risk_score': engineered['composite_risk_score']
    }

# TEST PROFILES
print("\n2️⃣ Testing customer profiles...")

test_profiles = [
    {
     'name': '🚨 Brand New High Risk',
        'expected': 'CHURN',
        'data': {
            'tenure': 0,
            'warehouse_to_home': 25.0,
            'num_devices_registered': 2,
            'preferred_order_category': 'Mobile Phone',
            'satisfaction_score': 2,
            'marital_status': 'Single',
            'num_addresses': 1,
            'has_complained': 1,
            'days_since_last_order': 30.0,
            'cashback_amount': 50.0,
            'tenure_missing_flag': 0,
            'warehouse_missing_flag': 0,
            'days_missing_flag': 0
        }
},
    {
        'name': '⚠️ At-Risk Customer',
        'expected': 'CHURN',
        'data': {
            'tenure': 2,
            'warehouse_to_home': 30.0,
            'num_devices_registered': 3,
            'preferred_order_category': 'Fashion',
            'satisfaction_score': 3,
            'marital_status': 'Single',
            'num_addresses': 2,
            'has_complained': 1,
            'days_since_last_order': 45.0,
            'cashback_amount': 100.0,
            'tenure_missing_flag': 0,
            'warehouse_missing_flag': 0,
            'days_missing_flag': 0
        }
    },
    {
           'name': '❓ Missing Tenure',
        'expected': 'CHURN',
        'data': {
            'tenure': 0,
            'warehouse_to_home': 15.0,
            'num_devices_registered': 3,
            'preferred_order_category': 'Mobile Phone',
            'satisfaction_score': 3,
            'marital_status': 'Single',
            'num_addresses': 2,
            'has_complained': 0,
            'days_since_last_order': 5.0,
            'cashback_amount': 120.0,
            'tenure_missing_flag': 1,   
            'warehouse_missing_flag': 0,
            'days_missing_flag': 0
        }
    },
    {
    'name': '✅ Established Low Risk',
    'expected': 'NO CHURN',
    'data': {
        'tenure': 24,
        'warehouse_to_home': 10.0,
        'num_devices_registered': 5,
        'preferred_order_category': 'Grocery',
        'satisfaction_score': 5,
        'marital_status': 'Married',
        'num_addresses': 3,
        'has_complained': 0,
        'days_since_last_order': 3.0,
        'cashback_amount': 300.0,
        'tenure_missing_flag': 0,
        'warehouse_missing_flag': 0,
        'days_missing_flag': 0
    }
}
]

# RUN PREDICTIONS
print("\n" + "="*70)
print("🎯 PREDICTION RESULTS")
print("="*70)

all_passed = True

for profile in test_profiles:
    result = predict_churn(profile['data'], model, preprocessor)

    predicted = 'CHURN' if result['prediction'] == 1 else 'NO CHURN'
    expected = profile['expected']
    passed = predicted == expected

    if not passed:
        all_passed = False

    status = "✅ PASSED" if passed else "❌ FAILED"

    print(f"\n{profile['name']}")
    print(f"   {status} Prediction: {predicted} | Expected: {expected}")
    print(f"   📊 Churn Probability: {result['churn_probability']}%")
    print(f"   🔄 Lifecycle Stage: {result['lifecycle_stage']}")
    print(f"   ⏰ Recency Risk: {result['recency_risk']}")
    print(f"   🛍️ Product Risk: {result['product_risk']}")
    print(f"   💰 Value Tier: {result['value_tier']}")
    print(f"   ⚠️ Risk Score: {result['composite_risk_score']}/5")

# FINAL SUMMARY
print("\n" + "="*70)
if all_passed:
    print("✅ ALL TESTS PASSED — Pipeline working correctly!")
else:
    print("⚠️  SOME TESTS FAILED — Review predictions above")
print("="*70)
print("\n💡 Ready for GCP deployment!")