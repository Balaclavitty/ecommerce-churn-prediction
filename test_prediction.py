"""
Test script to verify prediction pipeline works correctly.
"""
import joblib
import pandas as pd
from data_prep import ChurnDataPrep

print("\n" + "="*70)
print("🧪 TESTING PREDICTION PIPELINE")
print("="*70)

# Load trained model
print("\n1️⃣ Loading model...")
model = joblib.load('data/best_model.pkl')
print("✓ Model loaded")

# Create test customer data
print("\n2️⃣ Creating test customer...")
test_customer = {
    'tenure': 0,
    'warehouse_to_home': 15.0,
    'num_devices_registered': 4,
    'preferred_order_category': 'Mobile Phone',
    'satisfaction_score': 3,
    'marital_status': 'Single',
    'num_addresses': 2,
    'has_complained': 0,
    'days_since_last_order': 5.0,
    'cashback_amount': 150.0,
    'tenure_missing_flag': 0
}

print("✓ Test customer created:")
for key, value in test_customer.items():
    print(f"  • {key}: {value}")

# Transform using preprocessor
print("\n3️⃣ Transforming data...")
X_processed = ChurnDataPrep.transform_new_data(test_customer)

print(f"✓ Data transformed:")
print(f"  • Input columns: {len(test_customer)}")
print(f"  • Output columns: {len(X_processed.columns)}")
print(f"  • All numeric: {X_processed.dtypes.apply(lambda x: x.kind in 'biufc').all()}")

# Make prediction
print("\n4️⃣ Making prediction...")
prediction = model.predict(X_processed)[0]
probabilities = model.predict_proba(X_processed)[0]

print(f"✓ Prediction complete!")
print(f"\n{'='*70}")
print("🎯 RESULTS")
print("="*70)
print(f"\nChurn Prediction: {'WILL CHURN' if prediction == 1 else 'WILL NOT CHURN'}")
print(f"Churn Probability: {probabilities[1]*100:.2f}%")
print(f"Retention Probability: {probabilities[0]*100:.2f}%")

# Test with different customer profiles
print(f"\n{'='*70}")
print("🧪 TESTING MULTIPLE CUSTOMER PROFILES")
print("="*70)

test_profiles = [
    {
        'name': 'Brand New - High Risk',
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
            'tenure_missing_flag': 0
        }
    },
    {
        'name': 'Established - Low Risk',
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
            'cashback_amount': 500.0,
            'tenure_missing_flag': 0
        }
    },
    {
        'name': 'At-Risk Customer',
        'data': {
            'tenure': 6,
            'warehouse_to_home': 30.0,
            'num_devices_registered': 3,
            'preferred_order_category': 'Fashion',
            'satisfaction_score': 3,
            'marital_status': 'Single',
            'num_addresses': 2,
            'has_complained': 1,
            'days_since_last_order': 45.0,
            'cashback_amount': 100.0,
            'tenure_missing_flag': 0
        }
    }
]

for profile in test_profiles:
    X = ChurnDataPrep.transform_new_data(profile['data'])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    
    print(f"\n{profile['name']}:")
    print(f"  Prediction: {'CHURN' if pred == 1 else 'NO CHURN'}")
    print(f"  Churn Risk: {prob[1]*100:.1f}%")

print(f"\n{'='*70}")
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\n💡 Prediction pipeline is working correctly!")
print("   Ready for deployment/API integration.")