# AI and Data Science Job Salaries Prediction
Predicts a salary range for tech/data roles (Data Scientist, ML Engineer, Analyst, etc.) based on job title, experience, education, and company details. Built with XGBoost and deployed as an interactive Streamlit app.

Live app: https://ai-salary-prediction-byfarah.streamlit.app

Overview

Given a few inputs — job title, experience level, years of experience, education, employment type, company location, residence, company size, and industry — the app returns a predicted salary range (low / expected / high) rather than a single number, since real salaries vary even for similar profiles.

How it works
Data: 5,000 rows of tech/data-job salary records with fields like job title, experience level, location, company size, industry, and salary in USD.
Preprocessing:
Dropped duplicate rows and low-signal columns (salary_currency, primary_language, manages_people — confirmed via feature importance to contribute negligibly)
Ordinal encoding for ordered categories (experience_level, company_size, education_level)
One-hot encoding for nominal categories (job_title, employment_type, company_location, employee_residence, industry)
Modeling:
Tuned XGBoost via GridSearchCV → R² = 0.84, RMSE ≈ $21.8k
After dropping the two low-value features above: R² = 0.84, RMSE ≈ $21.9k (negligible change, confirming those features weren't adding value)
Salary range: trained two additional XGBoost models with quantile loss (reg:quantileerror, alpha = 0.1 and 0.9) alongside the point-estimate model, giving an 80% prediction interval. Empirical coverage on the test set: 80.9%.
App: a Streamlit form collects the fields a real user would know; less-informative fields not exposed in the UI are filled with dataset medians so the model still receives a complete feature vector.
Project structure
├── app.py                  # Streamlit app
├── train_and_save.py       # Reproduces preprocessing + trains/saves all models
├── requirements.txt        # Dependencies for deployment
├── model_point.joblib      # Tuned XGBoost point-estimate model
├── model_low.joblib        # Quantile model, 10th percentile
├── model_high.joblib       # Quantile model, 90th percentile
├── feature_columns.joblib  # Exact column order models expect (for encoding new input)
├── ui_options.json         # Dropdown values sourced from the training data
├── numeric_defaults.json   # Median/mode defaults for fields not exposed in the UI
└── .streamlit/config.toml  # App theme
Running locally
bash
pip install -r requirements.txt
streamlit run app.py
Retraining

To regenerate the models after changing the dataset or feature set, run:

bash
python train_and_save.py

This reproduces the full preprocessing pipeline and overwrites the .joblib/.json artifacts the app depends on.

Results summary
Model	R²	RMSE
XGBoost (default)	0.82	$23,605
XGBoost (tuned)	0.85	$21,569
XGBoost (tuned, trimmed features)	0.84	$21,859
