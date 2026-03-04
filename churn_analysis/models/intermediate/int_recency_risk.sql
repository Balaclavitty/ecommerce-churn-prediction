-- Recency-based risk segmentation (handles U-shaped pattern)

WITH customers AS (
    SELECT * FROM {{ ref('stg_customers')}}
)

SELECT 
    customer_id,
    days_since_last_order,

    -- Recent (0-7d): 16.8% churn - new customers, trial period
    -- Active (8-14d): 9.3% churn - engaged but recent
    -- Sweet spot (15-30d): 2.6% churn - LOWEST risk, regular customers
    -- Dormant (31+d): 16.9%+ churn - inactive, high risk
    CASE
        WHEN days_since_last_order <= 7 THEN 'recent_new_customer'          -- 16.8% churn
        WHEN days_since_last_order <= 14 THEN 'active_low_risk'             -- 9.3% churn
        WHEN days_since_last_order <= 30 THEN 'stable_lowest_risk'          -- 2.6% churn
        ELSE 'dormant_high_risk'                                            -- 16.9-31.6% churn
    END AS recency_risk,

    -- Binary flags
    CASE WHEN days_since_last_order <= 7 THEN 1 ELSE 0 END AS is_very_recent,
    CASE WHEN days_since_last_order > 30 THEN 1 ELSE 0 END AS is_dormant,
    CASE WHEN days_since_last_order BETWEEN 8 AND 30 THEN 1 ELSE 0 END AS is_active_stable,

    -- Days bins
    CASE
        WHEN days_since_last_order <= 7 THEN '0-7 days'
        WHEN days_since_last_order <= 14 THEN '8-14 days'
        WHEN days_since_last_order <= 30 THEN '15-30 days'
        ELSE '31+ days'
    END AS recency_bin

FROM customers

