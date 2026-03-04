-- Marital status stability indicator

WITH customers AS (
    SELECT * FROM {{ ref('stg_customers') }}
)

SELECT 
    customer_id,
    marital_status,
    
    -- Demographic stability (Single = 25.9%, Married = 10.9%)
    CASE 
        WHEN marital_status = 'Single' THEN 'unstable_high_risk'    -- 25.9% churn
        WHEN marital_status = 'Divorced' THEN 'moderate_risk'       -- 15.5% churn
        WHEN marital_status = 'Married' THEN 'stable_low_risk'      -- 10.9% churn
        ELSE 'unknown'
    END AS demographic_stability,
    
    -- Binary flags
    CASE WHEN marital_status = 'Single' THEN 1 ELSE 0 END AS is_single,
    CASE WHEN marital_status = 'Married' THEN 1 ELSE 0 END AS is_married,
    CASE WHEN marital_status = 'Divorced' THEN 1 ELSE 0 END AS is_divorced
    
FROM customers