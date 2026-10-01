
import streamlit as st
import requests


# =========================================================
# 1. PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Support Ticket Priority Predictor",
    page_icon="🎫"
)


# =========================================================
# 2. TITLE
# =========================================================

st.title("🎫 Support Ticket Priority Predictor")

st.write(
    "Enter the support ticket information below "
    "and click Predict."
)


# =========================================================
# 3. FASTAPI ADDRESS
# =========================================================

API_URL = "http://localhost:8000"


# =========================================================
# 4. INPUT FORM
# =========================================================

with st.form("prediction_form"):

    st.subheader("Ticket Information")


    # -------------------------
    # Numeric features
    # -------------------------

    customers_affected = st.number_input(
        "Customers affected",
        min_value=0,
        value=850
    )

    downtime_min = st.number_input(
        "Downtime (minutes)",
        min_value=0.0,
        value=120.0
    )

    error_rate_pct = st.number_input(
        "Error rate (%)",
        min_value=0.0,
        value=5.0
    )

    org_users = st.number_input(
        "Organization users",
        min_value=0,
        value=1000
    )

    past_30d_tickets = st.number_input(
        "Tickets in past 30 days",
        min_value=0,
        value=20
    )

    past_90d_incidents = st.number_input(
        "Incidents in past 90 days",
        min_value=0,
        value=5
    )

    description_length = st.number_input(
        "Description length",
        min_value=0,
        value=200
    )


    # -------------------------
    # Yes / No features
    # -------------------------

    payment_impact = st.checkbox(
        "Payment impacted?"
    )

    security_incident = st.checkbox(
        "Security incident?"
    )

    data_loss = st.checkbox(
        "Data loss?"
    )

    has_runbook_input = st.checkbox(
        "Runbook available?"
    )


    # -------------------------
    # Categorical features
    # -------------------------

    company_size = st.selectbox(
        "Company size",
        [
            "small",
            "medium",
            "large"
        ],
        index=2
    )

    industry = st.text_input(
        "Industry",
        value="finance"
    )

    customer_tier = st.text_input(
        "Customer tier",
        value="enterprise"
    )

    region = st.text_input(
        "Region",
        value="north america"
    )

    product_area = st.text_input(
        "Product area",
        value="payments"
    )

    booking_channel = st.text_input(
        "Booking channel",
        value="web"
    )

    reported_by_role = st.text_input(
        "Reported by role",
        value="customer"
    )

    day_of_week = st.selectbox(
        "Day of week",
        [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday"
        ]
    )

    customer_sentiment = st.selectbox(
        "Customer sentiment",
        [
            "positive",
            "neutral",
            "negative"
        ],
        index=2
    )


    # -------------------------
    # Submit button
    # -------------------------

    submitted = st.form_submit_button(
        "Predict Priority"
    )


# =========================================================
# 5. AFTER USER CLICKS PREDICT
# =========================================================

if submitted:

    # Create data dictionary
    data = {

        "customers_affected":
            customers_affected,

        "downtime_min":
            downtime_min,

        "error_rate_pct":
            error_rate_pct,

        "org_users":
            org_users,

        "past_30d_tickets":
            past_30d_tickets,

        "past_90d_incidents":
            past_90d_incidents,

        "payment_impact_flag":
            int(payment_impact),

        "security_incident_flag":
            int(security_incident),

        "data_loss_flag":
            int(data_loss),

        "has_runbook":
            int(has_runbook_input),

        "description_length":
            description_length,

        "company_size":
            company_size,

        "industry":
            industry,

        "customer_tier":
            customer_tier,

        "region":
            region,

        "product_area":
            product_area,

        "booking_channel":
            booking_channel,

        "reported_by_role":
            reported_by_role,

        "day_of_week":
            day_of_week,

        "customer_sentiment":
            customer_sentiment
    }


    # Show data sent to FastAPI
    st.subheader("Data sent to API")

    st.json(data)


    # =====================================================
    # 6. SEND DATA TO FASTAPI
    # =====================================================

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=30
        )


        # If API returns an error
        response.raise_for_status()


        # Convert API response to Python dictionary
        result = response.json()


        # Get prediction
        prediction = result[
            "predicted_priority"
        ]


        # =================================================
        # 7. SHOW PREDICTION
        # =================================================

        st.subheader("Prediction Result")


        if prediction.lower() == "high":

            st.error(
                f"🔴 Predicted Priority: {prediction}"
            )


        elif prediction.lower() == "medium":

            st.warning(
                f"🟠 Predicted Priority: {prediction}"
            )


        else:

            st.success(
                f"🟢 Predicted Priority: {prediction}"
            )


    except requests.exceptions.RequestException as error:

        st.error(
            f"Could not connect to the API: {error}"
        )
