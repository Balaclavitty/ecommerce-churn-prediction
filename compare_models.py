"""
Model comparison: Random Forest vs XGBoost for churn prediction.
Loads engineered features from dbt via data_prep.py.
Handles class imbalance, evaluates and saves best model.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import sys

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DUCKDB_PATH = os.path.join(DATA_DIR, 'churn_dev.duckdb') 

# IMPORTS 
sys.path.append(BASE_DIR)
from src.data_prep import ChurnDataPrep

print("\n" + "="*70)
print("🤖 MODEL COMPARISON: RANDOM FOREST vs XGBOOST")
print("="*70)

# DATA PREPARATION
data_prep = ChurnDataPrep(duckdb_path=DUCKDB_PATH)
data_prep.prepare()

X_train, X_test, Y_train, Y_test = data_prep.get_train_test_data()

# Defining feature names immediately after split
feature_names = X_train.columns.tolist()
print(f"✓ Feature names defined: {len(feature_names)} features")

# Saving preprocessor for future use
data_prep.save_preprocessor(
    path=os.path.join(DATA_DIR, 'preprocessor.pkl')
)

# CLASS IMBALANCE
negative = (Y_train == 0).sum()
positive = (Y_train == 1).sum()
scale_pos_weight = negative / positive

print(f"\n📊 Class distribution:")
print(f"     Non-churn: {negative:,} ({negative/(negative+positive)*100:.1f}%)")
print(f"     Churn:     {positive:,} ({positive/(negative+positive)*100:.1f}%)")
print(f" scale_pos_weight: {scale_pos_weight:.2f}")

# ============================================================
# RANDOM FOREST
# ============================================================

rf_model_path = os.path.join(DATA_DIR, 'random_forest_model.pkl')

if os.path.exists(rf_model_path):
    print("\n" + "="*70)
    print("🌲 RANDOM FOREST: Loading existing model")
    print("="*70)

    rf_model = joblib.load(rf_model_path)
    print(f"✓ Loaded existing Random Forest model")

else:
    print("\n" + "="*70)
    print("🌲 RANDOM FOREST: Training new model")
    print("="*70)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight= 'balanced',     # handles 16.3% churn imbalance
        random_state=42,
        n_jobs=-1
    )

    print("Training...")
    rf_model.fit(X_train, Y_train)
    print("✓ Random Forest training complete!")

    joblib.dump(rf_model, rf_model_path)
    print(f"✓ Saved: {rf_model_path}")

# Predictions
rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_roc_auc = roc_auc_score(Y_test, rf_pred_proba)
rf_cm = confusion_matrix(Y_test, rf_pred)

print(f"\n📊 Random Forest ROC-AUC: {rf_roc_auc:.4f}")

# ============================================================
# XGBOOST
# ============================================================

xgb_model_path  = os.path.join(DATA_DIR, 'xgboost_model.pkl')

if os.path.exists(xgb_model_path):
    print("\n" + "="*70)
    print("⚡ XGBOOST: Loading existing model")
    print("="*70)
    xgb_model = joblib.load(xgb_model_path)
    print(f"✓ Loaded existing XGBoost model")

else:
    print("\n" + "="*70)
    print("⚡ XGBOOST: Training new model")
    print("="*70)

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
    )

    print("Training...")
    xgb_model.fit(X_train, Y_train)
    print("✓ XGBoost training complete!")

    joblib.dump(xgb_model, xgb_model_path)
    print(f"✓ Saved: {xgb_model_path}")

# Predictions
xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_roc_auc = roc_auc_score(Y_test, xgb_pred_proba)
xgb_cm = confusion_matrix(Y_test, xgb_pred)

print(f"\n📊 XGBoost ROC-AUC: {xgb_roc_auc:.4f}")

# ============================================================
# COMPARISON
# ============================================================

improvement = ((xgb_roc_auc - rf_roc_auc) / rf_roc_auc) * 100

print("\n" + "="*70)
print("📊 HEAD-TO-HEAD COMPARISON")
print("="*70)

print(f"\n{'Metric':<30} {'Random Forest':<20} {'XGBoost':<20} {'Winner'}")
print("-" * 85)

winner = "🏆 XGBoost" if xgb_roc_auc > rf_roc_auc else "🏆 Random Forest" if rf_roc_auc > xgb_roc_auc else "Tie"
print(f"{'ROC-AUC':<30} {rf_roc_auc:<20.4f} {xgb_roc_auc:<20.4f} {winner}")

if abs(improvement) > 0.1:
    better = "XGBoost" if  improvement > 0 else "Random Forest"
    print(f"\n💡 {better} is {abs(improvement):.2f}% better")
else:
    print(f"\n💡 Models perform equally (difference < 0.1%)")

# ============================================================
# DETAILED REPORTS
# ============================================================

print("\n" + "="*70)
print("CLASSIFICATION REPORTS")
print("="*70)

print("\n🌲 RANDOM FOREST:")
print(classification_report(Y_test, rf_pred, target_names=['No Churn', 'Churn']))
print("Confusion Matrix:", rf_cm.tolist())

print("\n⚡ XGBOOST:")
print(classification_report(Y_test, xgb_pred, target_names=['No Churn', 'Churn']))
print("Confusion Matrix:", xgb_cm.tolist())

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "="*70)
print("🔥 TOP 10 FEATURES")
print("="*70)

rf_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

xgb_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🌲 Random Forest:")
print(rf_importance.head(10).to_string(index=False))

print("\n⚡ XGBoost:")
print(xgb_importance.head(10).to_string(index=False))

# ============================================================
# VISUALIZATIONS
# ============================================================

print("\n" + "="*70)
print("📊 Creating visualizations...")
print("="*70)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. ROC Curves
ax1 = fig.add_subplot(gs[0, :])
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_pred_proba)
xgb_fpr, xgb_tpr, _ = roc_curve(Y_test, xgb_pred_proba)

ax1.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_roc_auc:.4f})', 
         linewidth=2.5, color='#2ecc71')
ax1.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_roc_auc:.4f})', 
         linewidth=2.5, color='#e74c3c')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve Comparison', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# 2. Confusion Matrix - RF
ax2 = fig.add_subplot(gs[1, 0])
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
ax2.set_title(f'Random Forest (AUC={rf_roc_auc:.4f})', fontweight='bold')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')

# 3. Confusion Matrix - XGB
ax3 = fig.add_subplot(gs[1, 1])
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Reds', ax=ax3,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
ax3.set_title(f'XGBoost (AUC={xgb_roc_auc:.4f})', fontweight='bold')
ax3.set_ylabel('Actual')
ax3.set_xlabel('Predicted')

# 4. Feature Importance Comparison
ax4 = fig.add_subplot(gs[2, :])

importance_merged = rf_importance.merge(
    xgb_importance, on='feature', suffixes=('_rf', '_xgb')
)
top_15 = importance_merged.nlargest(15, 'importance_rf')

x = np.arange(len(top_15))
width = 0.35

ax4.barh(x - width/2, top_15['importance_rf'], width, 
         label='Random Forest', color='#2ecc71', alpha=0.8)
ax4.barh(x + width/2, top_15['importance_xgb'], width, 
         label='XGBoost', color='#e74c3c', alpha=0.8)

ax4.set_yticks(x)
ax4.set_yticklabels(top_15['feature'], fontsize=9)
ax4.invert_yaxis()
ax4.set_xlabel('Importance', fontsize=12)
ax4.set_title('Top 15 Feature Importance', fontweight='bold', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(axis='x', alpha=0.3)

plt.savefig(os.path.join(DATA_DIR, 'model_comparison.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Saved: data/model_comparison.png")
plt.show()

# ============================================================
# RECOMMENDATION
# ============================================================

print("\n" + "="*70)
print("🎯 RECOMMENDATION")
print("="*70)

if xgb_roc_auc > rf_roc_auc + 0.005:
    recommendation = "XGBoost"
    best_model = xgb_model
    best_auc = xgb_roc_auc
elif rf_roc_auc > xgb_roc_auc + 0.005:
    recommendation = "Random Forest"
    best_model = rf_model
    best_auc = rf_roc_auc
else:
    recommendation = "XGBoost (tie)"
    best_model = xgb_model
    best_auc = xgb_roc_auc

print(f"\n✅ RECOMMENDED: {recommendation}")
print(f"   ROC-AUC: {best_auc:.4f}")
print(f"   Improvement: {improvement:+.2f}%")

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

print("\n" + "="*70)
print("🔍 SHAP EXPLAINABILITY ANALYSIS")
print(f"   Model: {recommendation}")
print("="*70)

import shap

# 1. Compute SHAP values for best model
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Handling RF which returns list of arrays (one per class)
# XGBoost returns single array
if isinstance(shap_values, list):
    shap_values = shap_values[1]   # class 1 churn

print(f"✓ SHAP values computed")
print(f"Customer explained: {shap_values.shape[0]}")
print(f"Features explained: {shap_values.shape[1]}")

# 2. Global Feature Importance (Summary Plot)
print("\n📊 Generating Summary Plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    plot_type='bar',
    max_display=20,
    show=False
)
plt.title(f"SHAP Feature Importance - {recommendation}\n" 
          f"(Mean absolute impact on churn probability)",
          fontweight='bold', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'shap_summary_bar.png'),
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: shap_summary_bar.png")

# 3. Beeswarm plot (direction of impact)
print("\n📊 Generating Beeswarm Plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.title(f"SHAP Beeswarm - {recommendation}\n"
          f"(Red = increases churn, Blue = decreases churn)",
          fontweight='bold', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'shap_beeswarm.png'),
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: shap_beeswarm.png")

# 4. Top SHAP Features Table
print("\n📊 Top 15 Features by Mean SHAP Value:")
shap_importance = pd.DataFrame({
    'feature': feature_names,
    'shap_value': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_value', ascending=False)

print(shap_importance.head(15).to_string(index=False))

# 5. Individual Customer Explanation
print(f"\n📊 Individual Customer Explanations...")

# Using correct probabilities based on winning model
best_pred_proba = (
    xgb_pred_proba if 'XGBoost' in recommendation
    else rf_pred_proba
)

highest_risk_idx = np.argmax(best_pred_proba)
lowest_risk_idx = np.argmin(best_pred_proba)

for label, idx in [
    ('🚨 Highest Risk', highest_risk_idx),
    ('✅ Lowest Risk', lowest_risk_idx)
]:
    print(f"\n{label} Customer")
    print(f"   Churn probability: {best_pred_proba[idx]*100:.1f}%")
    print(f"   Actual churn: {Y_test.iloc[idx]} ")

    customer_shap = shap_values[idx]
    feature_contributions = pd.DataFrame({
        'feature': feature_names,
        'shap_value': customer_shap,
        'feature_value': X_test.iloc[idx].values
    }).sort_values('shap_value', key=abs, ascending=False).head(5)

    print(f" Top 5 drivers:")
    for _, row in feature_contributions.iterrows():
        direction = "↑ increases" if row['shap_value'] > 0 else "↓ decreases"
        print(f" • {row['feature']:35}"
              f"value={row['feature_value']:.2f}"
              f"SHAP={row['shap_value']:+.3f} ({direction} churn)")

# 6. Waterfall Plot - Highest Risk Customer
print("\n📊 Generating Waterfall Plot for highest risk customer...")

shap_explanation = shap.Explanation(
    values=shap_values[highest_risk_idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[highest_risk_idx].values,
    feature_names=feature_names
)

plt.figure(figsize=(14, 8))
shap.waterfall_plot(shap_explanation, max_display=15, show=False)
plt.title(
    f"Why is this customer "
    f"{best_pred_proba[highest_risk_idx]*100:.1f}% likely to churn?",
    fontweight='bold', fontsize=14
)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'shap_waterfall_high_risk.png'),
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: shap_waterfall_high_risk.png")

# 7. SHAP vs Model Feature Importance Comparison
print("\n📊 SHAP vs Model Feature Importance — Top 10:")
print(f"\n{'Rank':<6} {'Feature':<40} {'SHAP rank':>10} {'Model rank':>10} {'Match'}")
print("-" * 75)

shap_top10 = shap_importance.head(10)['feature'].tolist()
model_top10 = (xgb_importance if 'XGBoost' in recommendation
               else rf_importance).head(10)['feature'].tolist()

for rank, feat in enumerate(shap_top10, 1):
    model_rank = model_top10.index(feat) + 1 if feat in model_top10 else '>10'
    agreement = "✅" if feat in model_top10 else "⚠️"
    shap_rank_str = f"#{rank}"
    model_rank_str = f"#{model_rank}"
    print(f"#{rank:<5} {feat:<40} {shap_rank_str:>10} {model_rank_str:>10} {agreement}")
    
# 8. Save SHAP values for dashboard use
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df.to_csv(
    os.path.join(DATA_DIR, 'shap_values.csv'),
    index=False
)
print(f"\n✓ Saved: shap_values.csv")
print(f"  Rows: {shap_df.shape[0]} customers")
print(f"  Cols: {shap_df.shape[1]} features")
print(f"  Use for: Streamlit dashboard, PowerBI analysis")

print("\n" + "="*70)
print("✅ SHAP ANALYSIS COMPLETE!")
print("="*70)
print(f"""
Key SHAP Findings:
  • Model explained:    {recommendation}
  • Top predictor:      {shap_importance.iloc[0]['feature']}
  • Customers analyzed: {shap_values.shape[0]}
  • Files saved:        shap_summary_bar.png
                        shap_beeswarm.png
                        shap_waterfall_high_risk.png
                        shap_values.csv
""")
# ============================================================
# SAVE RESULTS
# ============================================================

print("\n💾 Saving results...")

joblib.dump(best_model, 
            os.path.join(DATA_DIR, 'best_model.pkl'))
joblib.dump(rf_model, 
            os.path.join(DATA_DIR, 'random_forest_model.pkl'))
joblib.dump(xgb_model, 
            os.path.join(DATA_DIR, 'xgboost_model.pkl'))

comparison_results = {
    'random_forest_auc': float(rf_roc_auc),
    'xgboost_auc': float(xgb_roc_auc),
    'recommendation': recommendation,
    'improvement_pct': float(improvement),
    'scale_pos_weight': float(scale_pos_weight),
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'n_features': int(len(feature_names)),
    'churn_rate_train': float(Y_train.mean()),
    'churn_rate_test': float(Y_test.mean())
}

with open(os.path.join(DATA_DIR, 'model_comparison_results.json'),
           'w') as f:
    json.dump(comparison_results, f, indent=2)

print("✓ Saved: best_model.pkl")
print("✓ Saved: random_forest_model.pkl")
print("✓ Saved: xgboost_model.pkl")
print("✓ Saved: model_comparison_results.json")
print("✓ Saved: model_comparison.png")

print("\n" + "="*70)
print("✅ COMPARISON COMPLETE!")
print("="*70)

print(f"""
Summary:
  • Best Model: {recommendation}
  • ROC-AUC:    {best_auc:.4f}
  • RF:         {rf_roc_auc:.4f}
  • XGB:        {xgb_roc_auc:.4f}

Next: dbt tests & documentation!
""")