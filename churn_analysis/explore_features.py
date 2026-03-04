import duckdb
import pandas as pd

# Connecting to dbt's DuckDB database
conn = duckdb.connect('dev.duckdb')

# Querying the final feature table
query = """
SELECT * 
FROM fct_customer_churn_features 
LIMIT 10
"""

df = conn.execute(query).fetchdf()

print("\n" + "="*70)
print("FEATURE TABLE OVERVIEW")
print("="*70)

print(f"\nTotal rows in table:")
total = conn.execute("SELECT COUNT(*) FROM fct_customer_churn_features").fetchone()[0]
print(f"  {total:,} customers")

print(f"\nTotal columns: {len(df.columns)}")

print("\n" + "="*70)
print("ALL FEATURES")
print("="*70)

# Grouping columns by type
raw_features = [col for col in df.columns if col in [
    'tenure', 'warehouse_to_home', 'num_devices_registered',
    'satisfaction_score', 'num_addresses', 'days_since_last_order',
    'cashback_amount', 'preferred_order_category', 'marital_status',
    'has_complained', 'days_missing_flag', 'tenure_missing_flag'
]]

engineered_features = [col for col in df.columns if col not in raw_features + ['customer_id', 'churn', 'loaded_at', 'features_generated_at']]

print("\n📊 RAW FEATURES:")
for col in raw_features:
    dtype = df[col].dtype
    print(f"  • {col:30} ({dtype})")

print(f"\n  Total raw features: {len(raw_features)}")

print("\n🔧 ENGINEERED FEATURES:")
for col in engineered_features:
    dtype = df[col].dtype
    print(f"  • {col:30} ({dtype})")

print(f"\n  Total engineered features: {len(engineered_features)}")

print("\n🎯 TARGET:")
print(f"  • churn")

print("\n📋 METADATA:")
print(f"  • customer_id, loaded_at, features_generated_at")

print("\n" + "="*70)
print("SAMPLE DATA (First 5 customers)")
print("="*70)

# Showing a subset of important columns
important_cols = [
    'customer_id', 'tenure', 'days_since_last_order',
    'lifecycle_stage', 'recency_risk', 'product_risk_category',
    'demographic_stability', 'value_tier', 'composite_risk_score',
    'has_complained', 'churn'
]

print(df[important_cols].head().to_string())

print("\n" + "="*70)
print("FEATURE DISTRIBUTIONS & VALIDATION")
print("="*70)

# Checking categorical features
print("\n📊 Lifecycle Stage Distribution:")
lifecycle_dist = conn.execute("""
    SELECT lifecycle_stage, COUNT(*) as count, 
           ROUND(AVG(churn) * 100, 1) as churn_pct
    FROM fct_customer_churn_features
    GROUP BY lifecycle_stage
    ORDER BY churn_pct DESC
""").fetchdf()
print(lifecycle_dist.to_string(index=False))

print("\n📊 Recency Risk Distribution:")
recency_dist = conn.execute("""
    SELECT recency_risk, COUNT(*) as count,
           ROUND(AVG(churn) * 100, 1) as churn_pct
    FROM fct_customer_churn_features
    GROUP BY recency_risk
    ORDER BY churn_pct DESC
""").fetchdf()
print(recency_dist.to_string(index=False))

print("\n📊 Product Risk Distribution:")
product_dist = conn.execute("""
    SELECT product_risk_category, COUNT(*) as count,
           ROUND(AVG(churn) * 100, 1) as churn_pct
    FROM fct_customer_churn_features
    GROUP BY product_risk_category
    ORDER BY churn_pct DESC
""").fetchdf()
print(product_dist.to_string(index=False))

print("\n📊 Demographic Stability Distribution:")
demo_dist = conn.execute("""
    SELECT demographic_stability, COUNT(*) as count,
           ROUND(AVG(churn) * 100, 1) as churn_pct
    FROM fct_customer_churn_features
    GROUP BY demographic_stability
    ORDER BY churn_pct DESC
""").fetchdf()
print(demo_dist.to_string(index=False))

print("\n📊 Composite Risk Score Distribution:")
risk_dist = conn.execute("""
    SELECT composite_risk_score, COUNT(*) as count,
           ROUND(AVG(churn) * 100, 1) as churn_pct
    FROM fct_customer_churn_features
    GROUP BY composite_risk_score
    ORDER BY composite_risk_score
""").fetchdf()
print(risk_dist.to_string(index=False))

print("\n" + "="*70)
print("VALIDATION: Do churn rates match EDA?")
print("="*70)

print("\n✓ Expected from EDA:")
print("  • new_high_risk:        35.1% churn")
print("  • recent_new_customer:  16.8% churn")
print("  • dormant_high_risk:    16.9%+ churn")
print("  • high_risk_tech:       27.1% churn")
print("  • unstable_high_risk:   25.9% churn")

print("\n✓ Check if dbt output matches these numbers above!")

print("\n" + "="*70)
print("✅ FEATURE TABLE READY FOR MODELING!")
print("="*70)

print(f"\nTotal features: {len(df.columns)}")
print(f"  • Raw features: {len(raw_features)}")
print(f"  • Engineered features: {len(engineered_features)}")
print(f"  • Target: 1 (churn)")
print(f"  • Metadata: 3")

conn.close()