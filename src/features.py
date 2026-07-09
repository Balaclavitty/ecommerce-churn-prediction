"""
Feature engineering logic that mirrors dbt models.
Single source of truth used by:
- test_predicton.py (local testing)
- deployment.app.py (GCP Cloud Run API)

Mirrors logic from:
- int_customer_lifecycle.sql
- int_recency_risk.sql
- int_product_risk.sql
- int_customer_value.sql
"""

def engineer_features(customer: dict) -> dict:
    """
    Replicate dbt fearure engineering for real-time inference.
    Takes raw customer dict, returns fully engineered feature dict.
    """

    c = customer.copy()

    c.setdefault('tenure_missing_flag', 0)
    c.setdefault('warehouse_missing_flag', 0)
    c.setdefault('days_missing_flag', 0)

    # ── int_customer_lifecycle ────────────────────────────
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

    c['is_new_customer']   = int(c['tenure_missing_flag'] == 0 and c['tenure'] <= 3)
    c['is_missing_tenure'] = int(c['tenure_missing_flag'] == 1)
    c['is_established']    = int(c['tenure'] > 12 and c['tenure_missing_flag'] == 0)

    # ── int_recency_risk ──────────────────────────────────
    days = c['days_since_last_order']
    if days <= 7:
        c['recency_risk'] = 'recent_new_customer'
    elif days <= 14:
        c['recency_risk'] = 'active_low_risk'
    elif days <= 30:
        c['recency_risk'] = 'stable_lowest_risk'
    else:
        c['recency_risk'] = 'dormant_high_risk'

    c['is_very_recent']   = int(days <= 7)
    c['is_dormant']       = int(days > 30)
    c['is_active_stable'] = int(8 <= days <= 30)

    # ── int_product_risk ──────────────────────────────────
    product_risk_map = {
        'Mobile Phone':       'high_risk_tech',
        'Fashion':            'moderate_risk_fashion',
        'Laptop & Accessory': 'low_risk_electronics',
        'Others':             'low_risk_other',
        'Grocery':            'lowest_risk_grocery'
    }
    c['product_risk_category'] = product_risk_map.get(
        c['preferred_order_category'], 'low_risk_other'
    )
    c['is_tech_buyer']    = int(c['preferred_order_category'] == 'Mobile Phone')
    c['is_grocery_buyer'] = int(c['preferred_order_category'] == 'Grocery')
    c['is_fashion_buyer'] = int(c['preferred_order_category'] == 'Fashion')

    # ── int_customer_value ────────────────────────────────
    cashback = c['cashback_amount']
    tenure   = c['tenure']

    if cashback >= 200:
        c['value_tier'] = 'high_value'
    elif cashback >= 145:
        c['value_tier'] = 'medium_value'
    else:
        c['value_tier'] = 'low_value'

    c['cashback_per_month'] = round(
        cashback / tenure if tenure > 0 else cashback, 2
    )
    c['is_high_value']  = int(cashback >= 200)
    c['is_low_spender'] = int(cashback < 75)

    # ── composite_risk_score ──────────────────────────────
    c['composite_risk_score'] = int(
        int(c['is_new_customer'] == 1 or c['is_missing_tenure'] == 1) +
        int(c['is_tech_buyer']) +
        int(c['has_complained']) +
        int(c.get('marital_status') == 'Single') +
        int(c['is_dormant'])
    )

    return c