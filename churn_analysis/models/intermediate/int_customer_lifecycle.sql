-- Customer lifecycle segmentation based on tenure
-- Key EDA finding: missing tenure (30% churn) does not equal brand_new (54.3% churn)

WITH customers AS (
    SELECT * FROM {{ ref('stg_customers')}}
)

SELECT 
    customer_id,
    tenure,
    tenure_missing_flag,
    
    -- Lifecycle stage
    -- IMPORTANT: check missing flag first - filled with 0 
    -- but behaviorally different from genuine brand new customers
    CASE
        WHEN tenure_missing_flag = 1 THEN 'missing_tenure'     -- 30% churn
        WHEN tenure = 0 THEN 'brand_new'                       -- 54.3% churn
        WHEN tenure <= 3 THEN 'new_high_risk'                  -- 35.1% churn
        WHEN tenure <= 6 THEN 'growing_moderate'               -- 7% churn
        WHEN tenure <= 12 THEN 'maturing_stable'               -- 5.5% churn
        ELSE 'established_loyal'                               -- ~5% churn
    END AS lifecycle_stage,

    -- Binary flags for modeling
    CASE
        WHEN tenure_missing_flag = 0
        AND tenure <=3 THEN 1
        ELSE 0
    END AS is_new_customer,

    CASE
        WHEN tenure_missing_flag = 1 THEN 1
        ELSE 0
    END AS is_missing_tenure,

    CASE WHEN tenure > 12
        AND tenure_missing_flag = 0 THEN 1
        ELSE 0
    END AS is_established,

    -- Tenure bins (for dashboard/analysis)
    CASE
        WHEN tenure_missing_flag = 1 THEN 'Missing Tenure'
        WHEN tenure = 0 THEN 'Brand New'
        WHEN tenure <= 3 THEN '1-3 months'
        WHEN tenure <= 6 THEN '4-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '1-2 years'
        ELSE '2+ years'
    END AS tenure_segment

FROM customers