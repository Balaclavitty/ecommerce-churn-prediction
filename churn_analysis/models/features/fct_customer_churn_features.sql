-- Final ML-ready feature table combining all intermediate models

WITH customers AS (
    SELECT * FROM {{ ref('stg_customers') }}
),

lifecycle AS (
    SELECT * FROM {{ ref('int_customer_lifecycle') }}
),

recency AS (
    SELECT * FROM {{ ref('int_recency_risk') }}
),

product AS (
    SELECT * FROM {{ ref('int_product_risk') }}
),

demographic AS (
    SELECT * FROM {{ ref('int_demographic_stability') }}
),

value AS (
    SELECT * FROM {{ ref('int_customer_value') }}
)

SELECT 
    -- Primary key
    c.customer_id,
    
    -- Raw numeric features
    c.tenure,
    c.warehouse_to_home,
    c.num_devices_registered,
    c.satisfaction_score,
    c.num_addresses,
    c.days_since_last_order,
    c.cashback_amount,
    
    -- Raw categorical features
    c.preferred_order_category,
    c.marital_status,
    
    -- Raw binary features
    c.has_complained,
    c.days_missing_flag,
    c.tenure_missing_flag,
    c.warehouse_missing_flag,
    
    -- Lifecycle features
    l.lifecycle_stage,
    l.is_new_customer,
    l.is_missing_tenure,
    l.is_established,
    l.tenure_segment,
    
    -- Recency features
    r.recency_risk,
    r.is_very_recent,
    r.is_dormant,
    r.is_active_stable,
    r.recency_bin,
    
    -- Product features
    p.product_risk_category,
    p.is_tech_buyer,
    p.is_grocery_buyer,
    p.is_fashion_buyer,
    
    -- Demographic features
    d.demographic_stability,
    d.is_single,
    d.is_married,
    d.is_divorced,
    
    -- Value features
    v.value_tier,
    v.cashback_per_month,
    v.is_high_value,
    v.is_low_spender,
    
    -- Composite risk score (0-5, higher = more risk)
    (
        CASE WHEN l.is_new_customer = 1
                OR l.is_missing_tenure = 1
                THEN 1 ELSE 0 END +   -- new OR missing tenure risk
        p.is_tech_buyer +             -- Tech buyer risk
        c.has_complained +            -- Complaint risk
        d.is_single +                 -- Single status risk
        r.is_dormant                  -- Dormant risk
    ) AS composite_risk_score,
    
    -- Target variable
    c.churn,
    
    -- Metadata
    c.loaded_at,
    CURRENT_TIMESTAMP AS features_generated_at

-- All models derive from stg_customers
-- they should ALL have the same customer_ids

FROM customers c
INNER JOIN lifecycle l ON c.customer_id = l.customer_id
INNER JOIN recency r ON c.customer_id = r.customer_id
INNER JOIN product p ON c.customer_id = p.customer_id
INNER JOIN demographic d ON c.customer_id = d.customer_id
INNER JOIN value v ON c.customer_id = v.customer_id
