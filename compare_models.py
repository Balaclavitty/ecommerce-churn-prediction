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

# Importing data prep module
from data_prep import ChurnDataPrep

print("\n" + "="*70)
print("🤖 MODEL COMPARISON: RANDOM FOREST vs XGBOOST")
print("="*70)

# Creating data prep object
data_prep = ChurnDataPrep(data_path='data/ml_features.csv')

# Preparing data
data_prep.prepare()

# Getting train/test splits
X_train, X_test, Y_train, Y_test = data_prep.get_train_test_data()

# Saving preprocessor for future use
data_prep.save_preprocessor(path='data/preprocessor.pkl')

# ============================================================
# RANDOM FOREST
# ============================================================

rf_model_path = 'data/rf_baseline_model.pkl'

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

print("\n" + "="*70)
print("⚡ XGBOOST: Training new model")
print("="*70)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

print("Training...")
xgb_model.fit(X_train, Y_train)
print("✓ XGBoost training complete!")

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
    'feature': data_prep.feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

xgb_importance = pd.DataFrame({
    'feature': data_prep.feature_names,
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

plt.savefig('data/model_comparison.png', dpi=300, bbox_inches='tight')
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
# SAVE RESULTS
# ============================================================

print("\n💾 Saving results...")

joblib.dump(best_model, 'data/best_model.pkl')
joblib.dump(xgb_model, 'data/xgboost_model.pkl')

comparison_results = {
    'random_forest_auc': float(rf_roc_auc),
    'xgboost_auc': float(xgb_roc_auc),
    'recommendation': recommendation,
    'improvement_pct': float(improvement)
}

with open('data/model_comparison_results.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

print("✓ Saved best_model.pkl")
print("✓ Saved xgboost_model.pkl")
print("✓ Saved model_comparison_results.json")

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