"""
Reproduces the preprocessing + tuned XGBoost + quantile models from the
notebook, then saves everything needed for the Streamlit app.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import xgboost as xgb

RANDOM_STATE = 42

# ---------- Load & preprocess (mirrors the notebook) ----------
df = pd.read_csv("Salaries.csv")
df.drop_duplicates(inplace=True)
df = df.drop(columns="salary_currency")

bool_cols = ["has_ml_in_title", "manages_people", "uses_ai_tools_daily", "switched_jobs_last_year"]
for col in bool_cols:
    df[col] = df[col].astype(int)

experience_level_map = {"Entry": 0, "Mid": 1, "Senior": 2, "Lead": 3, "Executive": 4}
company_size_map = {"S": 0, "M": 1, "L": 2}
education_level_map = {"Self-taught": 0, "Bootcamp": 1, "Bachelors": 2, "Masters": 3, "PhD": 4}
df["experience_level"] = df["experience_level"].map(experience_level_map)
df["company_size"] = df["company_size"].map(company_size_map)
df["education_level"] = df["education_level"].map(education_level_map)

remaining_cat_cols = ["job_title", "employment_type", "company_location",
                       "employee_residence", "industry", "primary_language"]
df = pd.get_dummies(df, columns=remaining_cat_cols, drop_first=True)

y = df["salary_usd"]
X = df.drop(columns=["salary_usd", "equity_offered_pct"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# ---------- Tuned XGBoost (best params from the notebook's GridSearchCV) ----------
best_params = {
    "colsample_bytree": 0.9,
    "learning_rate": 0.1,
    "max_depth": 3,
    "n_estimators": 200,
    "subsample": 0.9,
}
best_xgb_model = XGBRegressor(random_state=RANDOM_STATE, **best_params)
best_xgb_model.fit(X_train, y_train)

y_pred = best_xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Point model — RMSE: {rmse:.2f}, R2: {r2:.4f}")

# ---------- Quantile models for the range ----------
xgb_low = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.1,
                            random_state=RANDOM_STATE, **best_params)
xgb_high = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.9,
                             random_state=RANDOM_STATE, **best_params)
xgb_low.fit(X_train, y_train)
xgb_high.fit(X_train, y_train)

preds_low = xgb_low.predict(X_test)
preds_high = xgb_high.predict(X_test)
coverage = ((y_test >= preds_low) & (y_test <= preds_high)).mean()
print(f"80% interval empirical coverage: {coverage:.1%}")

# ---------- Save everything the app needs ----------
joblib.dump(best_xgb_model, "model_point.joblib")
joblib.dump(xgb_low, "model_low.joblib")
joblib.dump(xgb_high, "model_high.joblib")

# Exact column order the models expect
joblib.dump(list(X_train.columns), "feature_columns.joblib")

# Raw category options for building the UI dropdowns (from before encoding)
raw = pd.read_csv("Salaries.csv")
raw.drop_duplicates(inplace=True)
options = {
    "job_title": sorted(raw["job_title"].unique().tolist()),
    "experience_level": ["Entry", "Mid", "Senior", "Lead", "Executive"],
    "employment_type": sorted(raw["employment_type"].unique().tolist()),
    "company_size": ["S", "M", "L"],
    "company_location": sorted(raw["company_location"].unique().tolist()),
    "employee_residence": sorted(raw["employee_residence"].unique().tolist()),
    "industry": sorted(raw["industry"].unique().tolist()),
    "education_level": ["Self-taught", "Bootcamp", "Bachelors", "Masters", "PhD"],
    "primary_language": sorted(raw["primary_language"].unique().tolist()),
}
with open("ui_options.json", "w") as f:
    json.dump(options, f, indent=2)

# Defaults for numeric fields the user won't be asked about directly (median values)
numeric_defaults = {
    "remote_ratio": int(raw["remote_ratio"].median()),
    "team_size": int(raw["team_size"].median()),
    "certifications_count": int(raw["certifications_count"].median()),
    "weekly_hours": float(raw["weekly_hours"].median()),
    "uses_ai_tools_daily": int(raw["uses_ai_tools_daily"].mode()[0]),
    "ai_tools_hours_per_week": float(raw["ai_tools_hours_per_week"].median()),
    "bonus_pct": float(raw["bonus_pct"].median()),
    "job_satisfaction_score": float(raw["job_satisfaction_score"].median()),
    "interviews_to_offer": int(raw["interviews_to_offer"].median()),
    "switched_jobs_last_year": int(raw["switched_jobs_last_year"].mode()[0]),
    "upskilling_hours_per_month": float(raw["upskilling_hours_per_month"].median()),
    "fears_ai_automation_score": float(raw["fears_ai_automation_score"].median()),
    "has_ml_in_title": 0,
}
with open("numeric_defaults.json", "w") as f:
    json.dump(numeric_defaults, f, indent=2)

print("Saved: model_point.joblib, model_low.joblib, model_high.joblib, "
      "feature_columns.joblib, ui_options.json, numeric_defaults.json")
