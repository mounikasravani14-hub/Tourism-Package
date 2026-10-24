import os
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Visit with Us - Tourism Prediction",
    page_icon="✈️",
    layout="centered"
)


# Application title
st.title("✈️ Visit with Us")
st.subheader("Tourism Package Purchase Prediction")

st.write(
    "Enter the customer's details below to predict whether "
    "they are likely to purchase a tourism package."
)

# Customer details
st.header("Customer Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=0,
    value=15
)

persons_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=3
)

number_of_followups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    value=3
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    value=3
)

passport = st.selectbox(
    "Has Passport?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Owns a Car?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

number_of_children = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    value=0
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=25000
)

# Categorical features
type_of_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced", "Unmarried"]
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)


# Predict
if st.button("Predict Package Purchase", type="primary"):

    input_data = pd.DataFrame([{
        "Age": age,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "NumberOfPersonVisiting": persons_visiting,
        "NumberOfFollowups": number_of_followups,
        "PreferredPropertyStar": preferred_property_star,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children,
        "MonthlyIncome": monthly_income,
        "TypeofContact": type_of_contact,
        "Occupation": occupation,
        "Gender": gender,
        "ProductPitched": product_pitched,
        "MaritalStatus": marital_status,
        "Designation": designation
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "The customer is predicted to purchase the tourism package."
            )
        else:
            st.info(
                "The customer is predicted not to purchase the tourism package."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
