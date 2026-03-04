-- Customer lifecycle segmentation based on tenure
WITH customers AS (
    SELECT * FROM {{ ref('stg_customers')}}
)

SELECT 
    customer_id,
    tenure,
    
    -- Lifecycle stage (based on EDA: 0-3 months = 35% churn, 12mo+ = 5% churn)
    CASE
        WHEN tenure <= 3 THEN 'new_high_risk'              -- 35% churn
        WHEN tenure <= 6 THEN 'growing_moderate'           -- 7% churn
        WHEN tenure <= 12 THEN 'maturing_stable'           -- 5.5% churn
        ELSE 'established_loyal'                           -- 5-6% churn
    END AS lifecycle_stage,

    -- Binary flags for modeling
    CASE WHEN tenure <= 3 THEN 1 ELSE 0 END AS is_new_customer,
    CASE WHEN tenure > 12 THEN 1 ELSE 0 END AS is_established,

    -- Tenure bins (for analysis)
    CASE
        WHEN tenure <= 3 THEN '0-3 months'
        WHEN tenure <= 6 THEN '4-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '1-2 years'
        ELSE '2+ years'
    END AS tenure_segment

FROM customers