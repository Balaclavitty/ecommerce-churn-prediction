"""
Test script to verify prediction pipeline works correctly.
Uses dbt feature engineering logic replicated in Python
for real-time inference without requiring dbt pipeline.
"""
import joblib
import pandas as pd
import numpy as np
import os
import sys

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.append(BASE_DIR)

from data_prep import ChurnDataPrep

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

# FEATURE ENGINEERING FUNCTION
def engineer_features(customer: dict) -> dict:
    """
    Replicate dbt feature engineering for real-time inference.
    Mirrors logic from:
    - int_customer_lifecycle.sql
    - int_recency_risk.sql
    - int_product_risk.sql
    - int_customer_value.sql
    """
    c = customer.copy()

    # Ensuring missing flags exist
    c.setdefault('tenure_missing_flag', 0)
    c.setdefault('warehouse_missing_flag', 0)
    c.setdefault('days_missing_flag', 0)

    # int_customer_lifecycle logic
    if c['tenure_missing_flag'] == 1:
        c['lifecycle_stage'] = 'missing_tenure'
    elif c['tenure'] == 0:
        c['lifecycle_stage'] = 'brand_new'
    elif c['tenure'] <= 3:
        c['lifecycle_stage'] = 'new_high_risk'
    elif c['tenure'] <= 6:
        c['lifecycle_stage'] = 'growing_moderate'
    elif c['tenure'] <= 12:
        c['lifecycle_stage'] = 'maturing_stable'
    else:
        c['lifecycle_stage'] = 'established_loyal'

    c['is_new_customer'] = int(
        c['tenure_missing_flag'] == 0 and c['tenure'] <= 3
    )
    c['is_missing_tenure'] = int(c['tenure_missing_flag'] == 1)
    c['is_established'] = int(
        c['tenure'] > 12 and c['tenure_missing_flag'] == 0
    )

    # int_recency_risk logic
    days = c['days_since_last_order']
    if days <= 7:
        c['recency_risk'] = 'recent_new_customer'
    elif days <= 14:
        c['recency_risk'] = 'active_low_risk'
    elif days <= 30:
        c['recency_risk'] = 'stable_lowest_risk'
    else:
        c['recency_risk'] = 'dormant_high_risk'

    c['is_very_recent'] = int(days <= 7)
    c['is_dormant'] = int(days > 30)
    c['is_active_stable'] = int(8 <= days <= 30)

    # int_product_risk logic
    product_risk_map = {
        'Mobile Phone': 'high_risk_tech',
        'Fashion': 'moderate_risk_fashion',
        'Laptop & Accessory': 'low_risk_electronics',
        'Others': 'low_risk_other',
        'Grocery': 'lowest_risk_grocery'
    }
    c['product_risk_category'] = product_risk_map.get(
        c['preferred_order_category'], 'low_risk_other'
    )
    c['is_tech_buyer'] = int(c['preferred_order_category'] == 'Mobile Phone')
    c['is_grocery_buyer'] = int(c['preferred_order_category'] == 'Grocery')
    c['is_fashion_buyer'] = int(c['preferred_order_category'] == 'Fashion')

    # int_customer_value logic
    cashback = c['cashback_amount']
    tenure = c['tenure']

    if cashback >= 200:
        c['value_tier'] = 'high_value'
    elif cashback >= 145:
        c['value_tier'] = 'medium_value'
    else:
        c['value_tier'] = 'low_value'

    c['cashback_per_month'] = round(
        cashback / tenure if tenure > 0 else cashback, 2
    )
    c['is_high_value'] = int(cashback >= 200)
    c['is_low_spender'] = int(cashback < 75)

    # composite_risk_score
    c['composite_risk_score'] = int(
        int(c['is_new_customer'] == 1 or c['is_missing_tenure'] == 1) +
        int(c['is_tech_buyer']) +
        int(c['has_complained']) +
        int(c.get('marital_status') == 'Single') +
        int(c['is_dormant'])
    )

    return c

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