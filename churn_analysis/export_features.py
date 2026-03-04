import duckdb
import pandas as pd

print("\n" + "="*70)
print("EXPORTING FEATURE TABLE FOR MODELING")
print("="*70)

# Connecting to dbt's database
conn = duckdb.connect('dev.duckdb')

# Exporting all features
query = """
SELECT *
FROM fct_customer_churn_features
"""
churn_df = conn.execute(query).fetchdf()

print(f"\n✓ Loaded {len(churn_df):,} rows, {len(churn_df.columns)} columns")

# Dropping metadata columns (not needed for modeling)
churn_df = churn_df.drop(['customer_id', 'loaded_at', 'features_generated_at'], axis=1)

print(f"✓ Dropped metadata columns")
print(f"✓ Remaining columns: {len(churn_df.columns)} features + 1 target")

# Saving to csv
output_path = '../data/ml_features.csv'
churn_df.to_csv(output_path, index=False)

print(f"✓ Saved features to {output_path}")

print("\n" + "="*70)
print("FEATURE SUMMARY")
print("="*70)

print(f"\nRows: {len(churn_df):,}") 
print(f"Total columns: {len(churn_df.columns)}") # total = 36
print(f"Features: {len(churn_df.columns) - 1}")  # features = 35 (36- 1 target)
print(f"Target: 'churn'")                        # target = 1
print(f"Churn rate: {churn_df['churn'].mean()*100:.2f}%")

# Showing column names
print(f"\nAll columns:")
for i, col in enumerate(churn_df.columns, 1):
    print(f" {i:2}. {col}")

conn.close()

print("\n✅ Ready for modeling!")