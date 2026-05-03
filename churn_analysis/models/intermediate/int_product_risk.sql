-- Product category risk segmentation
-- Note: 'Mobile'merged to 'Mobile Phone' in seed cleaning

WITH customers AS (
    SELECT * FROM {{ ref('stg_customers') }}
)

SELECT
    customer_id,
    preferred_order_category,

    -- Product risk based on EDA churn rates
    CASE
        WHEN preferred_order_category = 'Mobile Phone'
            THEN 'high_risk_tech'                          -- 26.9% churn
        WHEN preferred_order_category = 'Fashion'
            THEN 'moderate_risk_fashion'                   -- 15.5% churn
        WHEN preferred_order_category = 'Laptop & Accessory'
            THEN 'low_risk_electronics'                    -- 9.8% churn
        WHEN preferred_order_category = 'Grocery'
            THEN 'lowest_risk_grocery'                     -- 4.1% churn
        ELSE 'low_risk_other'                              -- 7.4% churn
    END AS product_risk_category,

    -- Binary flags
        CASE WHEN preferred_order_category = 'Mobile Phone' 
         THEN 1 ELSE 0 END AS is_tech_buyer,
    CASE WHEN preferred_order_category = 'Grocery' 
         THEN 1 ELSE 0 END AS is_grocery_buyer,
    CASE WHEN preferred_order_category = 'Fashion' 
         THEN 1 ELSE 0 END AS is_fashion_buyer
    
FROM customers