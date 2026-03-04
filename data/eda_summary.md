
# E-Commerce Churn Analysis - EDA Summary
Date: 2026-02-25

## Dataset Overview
- Total customers: 3,270
- Churn rate: 16.33%
- Features: 14

## Top 5 Churn Drivers (Ranked by Effect Size)

### 1. TENURE (7x impact) - STRONGEST PREDICTOR
- New customers (0-3 months): 35.1% churn
- Established customers (12+ months): ~5% churn
- Correlation: r = -0.344 (p < 0.000001)

### 2. PRODUCT CATEGORY (6.5x impact)
- Mobile/Mobile Phone: 27.1% churn
- Grocery: 4.1% churn
- Chi-square: p < 0.000001

### 3. COMPLAINTS (3x impact)
- Complained: 31.8% churn
- No complaint: 10.3% churn
- Phi: 0.261 (p < 0.000001)

### 4. MARITAL STATUS (2.4x impact)
- Single: 25.9% churn
- Married: 10.9% churn
- Chi-square: p < 0.000001

### 5. RECENCY (Non-linear, U-shaped)
- Recent (0-7d): 16.8% churn
- Sweet spot (15-30d): 2.6% churn
- Dormant (47+d): 31.6% churn

## Features for dbt

### Tier 1 - Must Include:
1. tenure → lifecycle_stage
2. preferred_order_category → product_risk
3. has_complained → service_quality
4. marital_status → demographic_stability
5. days_since_last_order → recency_risk (binned)

### Tier 2 - Include:
6. cashback_amount → value_tier
7. num_devices_registered
8. tenure_missing_flag

### Drop:
- days_missing_flag (multicollinear, r=0.94)
- warehouse_to_home (weak, r=0.068)
- num_addresses (weak, r=0.046)

## Statistical Findings
- Bonferroni survivors: tenure, has_complained, cashback_amount
- U-shaped pattern: days_since_last_order (linear corr misleading)
- Paradox: satisfaction_score (higher = more churn, survivorship bias)
- Outlier impact: 184 customers (5.6%) affecting correlation

## Business Recommendations
1. First 90-day onboarding program (reduce 35% → 15% churn)
2. 24-hour complaint response SLA
3. Win-back campaign for 30+ days inactive
4. Product-specific retention (tech vs grocery)
5. Demographic targeting (singles vs married)

---
Analysis completed: 2026-02-25 15:24
