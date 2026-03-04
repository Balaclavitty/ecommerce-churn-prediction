-- Customer value segmentation based on spending

WITH customers AS (
    SELECT * FROM {{ ref('stg_customers') }}
)

SELECT 
    customer_id,
    cashback_amount,
    tenure,
    
    -- Value tier (based on cashback as proxy for spending)
    CASE 
        WHEN cashback_amount >= 300 THEN 'high_value'
        WHEN cashback_amount >= 150 THEN 'medium_value'
        WHEN cashback_amount >= 50 THEN 'low_value'
        ELSE 'minimal_value'
    END AS value_tier,
    
    -- Spending velocity (cashback per month of tenure)
    CASE 
        WHEN tenure > 0 THEN cashback_amount / tenure 
        ELSE cashback_amount 
    END AS cashback_per_month,
    
    -- Binary flags
    CASE WHEN cashback_amount >= 300 THEN 1 ELSE 0 END AS is_high_value,
    CASE WHEN cashback_amount < 100 THEN 1 ELSE 0 END AS is_low_spender
    
FROM customers