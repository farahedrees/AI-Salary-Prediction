import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Salary Range Predictor", layout="centered")

# ---------- Load model artifacts (cached so this runs once) ----------
@st.cache_resource
def load_artifacts():
    model_point = joblib.load("model_point.joblib")
    model_low = joblib.load("model_low.joblib")
    model_high = joblib.load("model_high.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    with open("ui_options.json") as f:
        ui_options = json.load(f)
    with open("numeric_defaults.json") as f:
        numeric_defaults = json.load(f)
    return model_point, model_low, model_high, feature_columns, ui_options, numeric_defaults

model_point, model_low, model_high, feature_columns, ui_options, numeric_defaults = load_artifacts()

experience_level_map = {"Entry": 0, "Mid": 1, "Senior": 2, "Lead": 3, "Executive": 4}
company_size_map = {"S": 0, "M": 1, "L": 2}
education_level_map = {"Self-taught": 0, "Bootcamp": 1, "Bachelors": 2, "Masters": 3, "PhD": 4}

remaining_cat_cols = ["job_title", "employment_type", "company_location",
                       "employee_residence", "industry", "primary_language"]


def build_feature_row(user_input: dict) -> pd.DataFrame:
    """Turn raw user selections into the one-hot encoded row the models expect."""
    row = {}

    # Ordinal encodings
    row["experience_level"] = experience_level_map[user_input["experience_level"]]
    row["company_size"] = company_size_map[user_input["company_size"]]
    row["education_level"] = education_level_map[user_input["education_level"]]

    # Direct numeric / boolean fields the user provided
    row["years_experience"] = user_input["years_experience"]
    row["manages_people"] = int(user_input["manages_people"])

    # Numeric fields not asked in the UI -> sensible dataset defaults
    for k, v in numeric_defaults.items():
        if k not in row:
            row[k] = v

    df_row = pd.DataFrame([row])

    # One-hot encode the nominal fields exactly like training, then align columns
    for col in remaining_cat_cols:
        df_row[col] = user_input[col]
    df_row = pd.get_dummies(df_row, columns=remaining_cat_cols)

    # Reindex to the exact training column order; missing dummy cols -> 0
    df_row = df_row.reindex(columns=feature_columns, fill_value=0)
    return df_row


st.title("Salary Range Predictor")

with st.form("salary_form"):
    col1, col2 = st.columns(2)

    with col1:
        job_title = st.selectbox("Job title", ui_options["job_title"])
        experience_level = st.selectbox("Experience level", ui_options["experience_level"])
        years_experience = st.slider("Years of experience", 0.0, 25.0, 5.0, step=0.5)
        education_level = st.selectbox("Education level", ui_options["education_level"])
        employment_type = st.selectbox("Employment type", ui_options["employment_type"])

    with col2:
        company_location = st.selectbox("Company location", ui_options["company_location"])
        employee_residence = st.selectbox("Your residence", ui_options["employee_residence"])
        company_size = st.selectbox("Company size", ["S", "M", "L"],
                                     format_func=lambda x: {"S": "Small", "M": "Medium", "L": "Large"}[x])
        industry = st.selectbox("Industry", ui_options["industry"])
        primary_language = st.selectbox("Primary language", ui_options["primary_language"])

    manages_people = st.checkbox("Manages people")

    submitted = st.form_submit_button("Predict salary range")

if submitted:
    user_input = {
        "job_title": job_title,
        "experience_level": experience_level,
        "years_experience": years_experience,
        "education_level": education_level,
        "employment_type": employment_type,
        "company_location": company_location,
        "employee_residence": employee_residence,
        "company_size": company_size,
        "industry": industry,
        "primary_language": primary_language,
        "manages_people": manages_people,
    }

    X_row = build_feature_row(user_input)

    point = model_point.predict(X_row)[0]
    low = model_low.predict(X_row)[0]
    high = model_high.predict(X_row)[0]

    # Guard against rare quantile-model inversions
    low, high = min(low, point), max(high, point)

    st.subheader("Estimated salary range")
    c1, c2, c3 = st.columns(3)
    c1.metric("Low", f"${low:,.0f}")
    c2.metric("Expected", f"${point:,.0f}")
    c3.metric("High", f"${high:,.0f}")
