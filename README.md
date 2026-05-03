# E-Commerce Customer Churn Prediction

End-to-end churn prediction pipeline combining analytics engineering
(dbt) with machine learning (XGBoost) to identify at-risk customers
and enable proactive retention strategies.

## 🎯 Business Problem

An e-commerce platform loses revenue when customers churn silently.
This project predicts churn probability for each customer and explains
**why** they are at risk — enabling targeted retention campaigns.

**Key result:** XGBoost model achieving **96% ROC-AUC** with 81%
recall on churners, deployed on a dataset of 3,269 customers.

---

## 🏗️ Architecture

```
Raw Data (CSV)
     ↓
dbt Pipeline (DuckDB)
     ├── stg_customers          ← standardize & validate
     ├── int_customer_lifecycle ← tenure segments
     ├── int_recency_risk       ← U-shaped recency pattern
     ├── int_product_risk       ← category risk scoring
     ├── int_demographic_stability ← marital status risk
     ├── int_customer_value     ← spending velocity
     └── fct_customer_churn_features ← ML-ready features
          ↓
Python ML Pipeline
     ├── data_prep.py     ← load from DuckDB, encode, split
     ├── compare_models.py ← RF vs XGBoost + SHAP
     └── test_prediction.py ← inference pipeline
```

---

## 📊 Key Findings

| Segment | Churn Rate | Insight |
|---|---|---|
| Brand new (0 months) | **54.3%** | Critical intervention window |
| 1-3 months tenure | **35.1%** | High-risk early stage |
| Missing tenure | **30.0%** | Data gap = disengaged customer |
| Mobile Phone buyers | **26.9%** | Price-driven, easy to switch |
| Grocery buyers | **4.1%** | Habitual, most loyal |
| 2+ year customers | **0.0%** | Fully retained |

**Composite risk score validation:**
| Score | Churn Rate |
|---|---|
| 0/5 | 3.1% |
| 1/5 | 6.4% |
| 2/5 | 21.5% |
| 3/5 | 50.7% |
| 4/5 | **79.2%** |

---

## 🤖 Model Results

| Model | ROC-AUC | Churn Recall | Churn Precision |
|---|---|---|---|
| Random Forest | 0.9203 | 81% | 52% |
| **XGBoost** ✅ | **0.9595** | **81%** | **77%** |

XGBoost wins on every metric — same recall but dramatically
fewer false positives (26 vs 79).

---

## 🔍 SHAP Feature Importance

Top predictors by mean absolute SHAP value:
1. `cashback_per_month` — spending velocity (engineered)
2. `tenure` — months as customer
3. `has_complained` — complaint behaviour
4. `composite_risk_score` — engineered risk score
5. `num_addresses` — platform engagement proxy

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data transformation | dbt + DuckDB |
| EDA & statistics | Python, pandas, scipy |
| ML modeling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn, Plotly |
| Version control | Git + GitHub |
| Cloud (WIP) | Google Cloud Platform |

---

## 📁 Project Structure

```
Churn_Project/
├── churn_analysis/     ← dbt project
│   ├── models/         ← SQL transformation models
│   ├── seeds/          ← source data
│   └── schema.yaml     ← tests & documentation
├── notebooks/
│   └── 01_eda.ipynb    ← full exploratory analysis
├── data_prep.py        ← data loading & encoding
├── compare_models.py   ← model training & SHAP
├── test_prediction.py  ← inference pipeline
└── explore_features.py ← dbt output validation
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install dbt-duckdb xgboost scikit-learn shap
pip install pandas numpy matplotlib seaborn joblib
```

### Run dbt Pipeline
```bash
cd churn_analysis
dbt deps
dbt seed
dbt run
dbt test
```

### Train Models
```bash
python compare_models.py
```

### Test Predictions
```bash
python test_prediction.py
```

---

## 📈 Business Recommendations

1. **First 90-day onboarding program** — target 35-54% churn segment
2. **24-hour complaint response SLA** — complainers have 3x higher churn
3. **Mobile Phone buyer retention** — 26.9% churn vs 4.1% grocery
4. **Win-back campaign** — customers dormant 15-30 days (sweet spot)
5. **Single customer targeting** — 25.8% churn vs 10.9% married

---

## 🔮 Next Steps

- [ ] GCP Cloud Run API deployment
- [ ] Streamlit interactive dashboard
- [ ] Power BI stakeholder dashboard
- [ ] Hyperparameter tuning with Optuna
- [ ] Real-time prediction pipeline

---

## 👩‍💻 Author

Data Analyst with 3+ years experience in DACH market operations.
Combining business domain knowledge with modern data engineering
and machine learning practices.

*Built as part of continuous learning in analytics engineering
and data science.*
