-- Customer value segmentation based on spending

WITH customers AS (
    SELECT * FROM {{ ref('stg_customers') }}
)

SELECT 
    customer_id,
    cashback_amount,
    tenure,
    
    -- Value tier (based on cashback as proxy for spending)
    -- Thresholds derived from EDA data distribution (max=324.99, median=~175)
    CASE 
        WHEN cashback_amount >= 200 THEN 'high_value'      -- 25%, 4.7% churn -> top spenders
        WHEN cashback_amount >= 145 THEN 'medium_value'    -- 50%, ~15% churn -> mid-tier spenders
        ELSE 'low_value'                                   -- 25%, ~25% churn -> low spenders
    END AS value_tier,
    
    -- Spending velocity (cashback per month of tenure)
    -- Note: tenure=0 (brand new + missing tenure) uses raw cashback
    -- as 1-month proxy to avoid division by zero
    CASE 
        WHEN tenure > 0 THEN ROUND(cashback_amount / tenure, 2)
        ELSE ROUND(cashback_amount, 2)
    END AS cashback_per_month,
    
    -- Binary flags
    CASE WHEN cashback_amount >= 200 THEN 1 ELSE 0 END AS is_high_value,
    CASE WHEN cashback_amount < 145 THEN 1 ELSE 0 END AS is_low_spender
    
FROM customers