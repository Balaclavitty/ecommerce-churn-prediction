-- models/staging/stg_customers.sql
-- Clean and standardize raw customer data

WITH source AS (
    SELECT * FROM {{ ref('ecommerce_churn_seed')}}
)

SELECT 
-- Using row_number as customer_id 
    ROW_NUMBER() OVER (ORDER BY tenure DESC, cashback_amount DESC,
                       days_since_last_order ASC) AS customer_id,

    -- Numeric features
    tenure,
    warehouse_to_home,
    num_devices_registered,
    satisfaction_score,
    num_addresses,
    days_since_last_order,
    cashback_amount,

    -- Categorical features
    preferred_order_category,
    marital_status,

    -- Binary features
    has_complained,
    churn,

    -- Missing value flags
    days_missing_flag,
    tenure_missing_flag,
    warehouse_missing_flag,

    -- Metadata
    CURRENT_TIMESTAMP AS loaded_at

FROM source