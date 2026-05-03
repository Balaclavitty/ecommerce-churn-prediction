
# E-Commerce Churn Analysis - EDA Summary
Date: 2026-04-23

## Dataset Overview
- Total customers: 3,268
- Churn rate: 16.31%
- Features: 14 (excluding target)

## Top 5 Churn Drivers

### 1. TENURE - STRONGEST PREDICTOR
- New customers (0-3 months): 39.4% churn
- Established customers (7+ months): 4.8% churn
- Correlation: r = -0.344

### 2. PRODUCT CATEGORY
- Mobile Phone: 26.9% churn (highest risk)
- Grocery: 4.1% churn (lowest risk)

### 3. COMPLAINTS
- Complained: 31.8% churn
- No complaint: 10.2% churn

### 4. MARITAL STATUS
- Single: 25.8% churn
- Married: 10.9% churn

### 5. RECENCY (Non-linear U-shaped)
- Recent (0-7d): 18.5% churn
- Sweet spot (8-30d): 9.0% churn
- Note: 31+ day bucket only 2 customers — unreliable sample

## Missing Value Patterns (MNAR — Not Missing At Random)

### Warehouse Distance Missing
- Missing: 33.3% churn
- Present: 15.6% churn
- Interpretation: incomplete profile = disengaged customer
- Action: warehouse_missing_flag kept as predictive feature

### Tenure Missing
- Missing: 29.6% churn
- Present: 15.6% churn
- Interpretation: unregistered/dormant customers = elevated risk
- Action: tenure_missing_flag kept as predictive feature

## Features for Modeling

### Tier 1 - Strong Predictors (Bonferroni survivors, |r| > 0.1):
1. tenure (r=-0.344)
2. has_complained (r=0.263)
3. preferred_order_category_encoded (r=0.228)
4. marital_status_Single (r=0.172)
5. cashback_amount (r=-0.151)
6. days_since_last_order (r=-0.143)
7. num_devices_registered (r=0.110)

### Tier 2 - Moderate Predictors (Bonferroni survivors, |r| 0.05-0.1):
8. satisfaction_score (r=0.099) — paradoxical, use cautiously
9. warehouse_missing_flag (r=0.096)
10. tenure_missing_flag (r=0.081)
11. warehouse_to_home (r=0.066)

### Tier 3 - Weak but kept:
12. num_addresses (p=0.007, borderline)
13. marital_status_Married (r=-0.151)
14. marital_status_Divorced (r=-0.011)

### Dropped Features:
- days_missing_flag: p=0.919, not significant
- is_early_stage: r=-0.700 with tenure, redundant
- tenure_segment: derived from tenure, redundant
- preferred_order_category: replaced by encoded version
- marital_status: replaced by dummy columns

## Statistical Findings
- Bonferroni correction applied across 14 features (α=0.003)
- U-shaped pattern confirmed: days_since_last_order
- Satisfaction paradox: higher score → more churn (survey timing bias)
- Outlier impact: minimal across all features (max impact 0.012)
- No severe multicollinearity detected (all |r| < 0.7)

## Business Recommendations
1. First 90-day onboarding program — target 39% churn segment
2. 24-hour complaint response SLA — complainers have 32% churn rate
3. Win-back campaign for 15-30 days inactive (sweet spot before churn spikes)
4. Product-specific retention: Mobile Phone vs Grocery customers
5. Demographic targeting: Single customers have 26% churn rate

---
Analysis completed: 2026-04-23 10:23
