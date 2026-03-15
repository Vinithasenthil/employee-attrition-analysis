import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
# -------------------------------
# FILE PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "employee_attrition.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.append(BASE_DIR)

from src.feature_engineering import engineer_features

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Employee Attrition Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# CUSTOM CSS — BLUE PROFESSIONAL
# -------------------------------
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #2d6a9f 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }

    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #d1e3f0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }

    h1 { color: #1a3a5c !important; }
    h2 { color: #2d6a9f !important; }
    h3 { color: #1a3a5c !important; }

    .stButton > button {
        background-color: #2d6a9f;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1a3a5c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM employees", conn)
    conn.close()
    return df

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    attrition_model   = joblib.load(os.path.join(MODELS_DIR, "attrition_model.pkl"))
    performance_model = joblib.load(os.path.join(MODELS_DIR, "performance_model.pkl"))
    scaler            = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return attrition_model, performance_model, scaler

# -------------------------------
# PREPROCESS SINGLE INPUT
# -------------------------------
def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])

    # Fixed mappings — same as LabelEncoder learned during training
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    df["OverTime"] = df["OverTime"].map({"No": 0, "Yes": 1})
    df["MaritalStatus"] = df["MaritalStatus"].map({
        "Divorced": 0, "Married": 1, "Single": 2
    })
    df["BusinessTravel"] = df["BusinessTravel"].map({
        "Non-Travel": 0, "Travel_Frequently": 1, "Travel_Rarely": 2
    })
    df["Department"] = df["Department"].map({
        "Human Resources": 0, "Research & Development": 1, "Sales": 2
    })
    df["EducationField"] = df["EducationField"].map({
        "Human Resources": 0, "Life Sciences": 1, "Marketing": 2,
        "Medical": 3, "Other": 4, "Technical Degree": 5
    })
    df["JobRole"] = df["JobRole"].map({
        "Healthcare Representative": 0, "Human Resources": 1,
        "Laboratory Technician": 2, "Manager": 3,
        "Manufacturing Director": 4, "Research Director": 5,
        "Research Scientist": 6, "Sales Executive": 7,
        "Sales Representative": 8
    })

    # Feature engineering
    df["StressRisk"]       = df["OverTime"] * (df["MaritalStatus"] == 2).astype(int)
    df["IncomeLevelRatio"] = df["MonthlyIncome"] / (df["JobLevel"] + 1)
    df["ExperienceRate"]   = df["TotalWorkingYears"] / (df["Age"] + 1)
    df["StagnationScore"]  = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["LoyaltyScore"]     = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)

    return df

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
def sidebar():
    st.sidebar.markdown("## 👥 HR Analytics")
    st.sidebar.markdown("### Employee Attrition")
    st.sidebar.markdown("---")
    pages = {
        "📊 EDA Dashboard"       : "eda",
        "🤖 Predict Attrition"   : "predict_attrition",
        "📈 Predict Performance" : "predict_performance",
        "📂 Bulk Prediction"     : "bulk",
        "📉 Model Performance"   : "metrics",
        "⚠️ At Risk Employees"   : "atrisk"
    }
    choice = st.sidebar.radio("Navigate", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset:** IBM HR Analytics")
    st.sidebar.markdown("**Records:** 1,470 employees")
    st.sidebar.markdown("**Models:** Random Forest")
    return pages[choice]

# ================================
# PAGE 1 — EDA DASHBOARD
# ================================
def page_eda(df):
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Deep dive into employee attrition patterns and key insights.")
    st.markdown("---")

    total  = len(df)
    left   = df[df["Attrition"] == "Yes"].shape[0]
    stayed = df[df["Attrition"] == "No"].shape[0]
    rate   = round((left / total) * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", f"{total:,}")
    col2.metric("Employees Left", f"{left:,}")
    col3.metric("Employees Stayed", f"{stayed:,}")
    col4.metric("Attrition Rate", f"{rate}%")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        df["Attrition"].value_counts().plot.pie(
            autopct="%1.1f%%", colors=["#2d6a9f", "#ff4444"],
            startangle=90, ax=ax, labels=["Stayed", "Left"]
        )
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close()
        st.info("💡 83.9% employees stay, 16.1% leave")

    with col2:
        st.subheader("Attrition by Department")
        fig, ax = plt.subplots(figsize=(5, 4))
        dept_attr = df.groupby("Department")["Attrition"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).reset_index()
        dept_attr.columns = ["Department", "Attrition Rate %"]
        sns.barplot(data=dept_attr, x="Department", y="Attrition Rate %",
                    hue="Department", palette="Blues_d", legend=False, ax=ax)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)
        plt.close()
        st.info("💡 Sales has highest attrition rate")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by OverTime")
        fig, ax = plt.subplots(figsize=(5, 4))
        ot_attr = df.groupby("OverTime")["Attrition"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).reset_index()
        ot_attr.columns = ["OverTime", "Attrition Rate %"]
        sns.barplot(data=ot_attr, x="OverTime", y="Attrition Rate %",
                    hue="OverTime", palette=["#2d6a9f", "#ff4444"],
                    legend=False, ax=ax)
        st.pyplot(fig)
        plt.close()
        st.info("💡 Overtime employees leave 3x more")

    with col2:
        st.subheader("Monthly Income vs Attrition")
        fig, ax = plt.subplots(figsize=(5, 4))
        df.boxplot(column="MonthlyIncome", by="Attrition", ax=ax)
        plt.suptitle("")
        ax.set_xlabel("Attrition")
        ax.set_ylabel("Monthly Income")
        st.pyplot(fig)
        plt.close()
        st.info("💡 Leavers earn significantly less on average")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution by Attrition")
        fig, ax = plt.subplots(figsize=(5, 4))
        df[df["Attrition"] == "Yes"]["Age"].hist(
            bins=20, alpha=0.7, color="#ff4444", label="Left", ax=ax)
        df[df["Attrition"] == "No"]["Age"].hist(
            bins=20, alpha=0.7, color="#2d6a9f", label="Stayed", ax=ax)
        ax.legend()
        ax.set_xlabel("Age")
        st.pyplot(fig)
        plt.close()
        st.info("💡 Younger employees (25-35) leave more")

    with col2:
        st.subheader("Attrition by Marital Status")
        fig, ax = plt.subplots(figsize=(5, 4))
        ms_attr = df.groupby("MaritalStatus")["Attrition"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).reset_index()
        ms_attr.columns = ["MaritalStatus", "Attrition Rate %"]
        sns.barplot(data=ms_attr, x="MaritalStatus", y="Attrition Rate %",
                    hue="MaritalStatus", palette="Blues_d", legend=False, ax=ax)
        st.pyplot(fig)
        plt.close()
        st.info("💡 Single employees have 25% attrition rate")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 5))
        num_cols = ["Age", "MonthlyIncome", "TotalWorkingYears",
                    "YearsAtCompany", "JobSatisfaction",
                    "WorkLifeBalance", "DistanceFromHome"]
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",
                    cmap="Blues", ax=ax, linewidths=0.5)
        st.pyplot(fig)
        plt.close()
        st.info("💡 Income and experience are strongly correlated")

    with col2:
        st.subheader("Top Factors — Feature Importance")
        try:
            model = joblib.load(os.path.join(MODELS_DIR, "attrition_model.pkl"))
            from src.feature_engineering import feature_pipeline
            X, _ = feature_pipeline()
            feat_imp = pd.Series(model.feature_importances_,
                                 index=X.columns).nlargest(10)
            fig, ax = plt.subplots(figsize=(6, 5))
            feat_imp.plot(kind="barh", color="#2d6a9f", ax=ax)
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)
            plt.close()
            st.info("💡 Monthly income and overtime are top predictors")
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")

# ================================
# PAGE 2 — PREDICT ATTRITION
# ================================
def page_predict_attrition(attrition_model, scaler):
    st.title("🤖 Predict Employee Attrition")
    st.markdown("Enter employee details to predict if they are likely to leave.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Info")
        age            = st.slider("Age", 18, 60, 30)
        gender         = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance       = st.slider("Distance From Home (km)", 1, 29, 5)

    with col2:
        st.subheader("Job Info")
        department       = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role         = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative",
            "Manager", "Sales Representative", "Research Director", "Human Resources"
        ])
        job_level        = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_satisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
        overtime         = st.selectbox("OverTime", ["Yes", "No"])

    with col3:
        st.subheader("Experience & Pay")
        monthly_income        = st.number_input("Monthly Income", 1000, 20000, 5000)
        total_working_years   = st.slider("Total Working Years", 0, 40, 5)
        years_at_company      = st.slider("Years at Company", 0, 40, 3)
        years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_manager    = st.slider("Years With Current Manager", 0, 17, 2)

    st.markdown("---")

    with st.expander("Additional Details (optional)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            education       = st.selectbox("Education", [1, 2, 3, 4, 5])
            education_field = st.selectbox("Education Field", [
                "Life Sciences", "Medical", "Marketing",
                "Technical Degree", "Human Resources", "Other"
            ])
            business_travel = st.selectbox("Business Travel", [
                "Travel_Rarely", "Travel_Frequently", "Non-Travel"
            ])
        with col2:
            daily_rate               = st.number_input("Daily Rate", 100, 1500, 800)
            hourly_rate              = st.number_input("Hourly Rate", 30, 100, 65)
            monthly_rate             = st.number_input("Monthly Rate", 2000, 27000, 14000)
            environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        with col3:
            job_involvement           = st.selectbox("Job Involvement", [1, 2, 3, 4])
            num_companies_worked      = st.slider("Num Companies Worked", 0, 9, 2)
            percent_salary_hike       = st.slider("Percent Salary Hike", 11, 25, 14)
            relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
            stock_option_level        = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            training_times            = st.slider("Training Times Last Year", 0, 6, 3)
            work_life_balance         = st.selectbox("Work Life Balance", [1, 2, 3, 4])
            years_in_role             = st.slider("Years In Current Role", 0, 18, 2)

    if st.button("🔍 Predict Attrition"):
        input_data = {
            "Age": age, "BusinessTravel": business_travel,
            "DailyRate": daily_rate, "Department": department,
            "DistanceFromHome": distance, "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": environment_satisfaction,
            "Gender": gender, "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement, "JobLevel": job_level,
            "JobRole": job_role, "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "OverTime": overtime,
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promotion,
            "YearsWithCurrManager": years_with_manager
        }

        df_input = preprocess_input(input_data)
        scaled   = scaler.transform(df_input)
        prob     = attrition_model.predict_proba(scaled)[0][1]

        # Lowered threshold from 0.5 to 0.3
        # Base attrition rate = 16.1%, so 30%+ is genuinely high risk
        pred = 1 if prob >= 0.3 else 0

        if prob >= 0.5:
            risk  = "HIGH RISK"
            color = "🔴"
        elif prob >= 0.25:
            risk  = "MEDIUM RISK"
            color = "🟡"
        else:
            risk  = "LOW RISK"
            color = "🟢"

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", "Will Leave ❌" if pred == 1 else "Will Stay ✅")
        col2.metric("Probability of Leaving", f"{prob*100:.1f}%")
        col3.metric("Risk Level", f"{color} {risk}")

        if pred == 1:
            st.error("⚠️ This employee is at risk of leaving. Consider retention strategies.")
        else:
            st.success("✅ This employee is likely to stay with the company.")

# ================================
# PAGE 3 — PREDICT PERFORMANCE
# ================================
def page_predict_performance(performance_model, scaler):
    st.title("📈 Predict Performance Rating")
    st.markdown("Predict whether an employee will have Rating 3 (Good) or Rating 4 (Excellent).")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        age             = st.slider("Age", 18, 60, 30)
        education       = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        job_level       = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])

    with col2:
        monthly_income      = st.number_input("Monthly Income", 1000, 20000, 5000)
        years_at_company    = st.slider("Years at Company", 0, 40, 3)
        years_in_role       = st.slider("Years in Current Role", 0, 18, 2)
        total_working_years = st.slider("Total Working Years", 0, 40, 5)

    with col3:
        percent_salary_hike   = st.slider("Percent Salary Hike", 11, 25, 14)
        training_times        = st.slider("Training Times Last Year", 0, 6, 3)
        years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_manager    = st.slider("Years With Current Manager", 0, 17, 2)

    if st.button("📈 Predict Performance"):
        input_data = {
            "Age": age, "BusinessTravel": "Travel_Rarely",
            "DailyRate": 800, "Department": "Research & Development",
            "DistanceFromHome": 5, "Education": education,
            "EducationField": "Life Sciences",
            "EnvironmentSatisfaction": 3,
            "Gender": "Male", "HourlyRate": 65,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level, "JobRole": "Research Scientist",
            "JobSatisfaction": 3, "MaritalStatus": "Married",
            "MonthlyIncome": monthly_income, "MonthlyRate": 14000,
            "NumCompaniesWorked": 2, "OverTime": "No",
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": 3,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promotion,
            "YearsWithCurrManager": years_with_manager
        }

        df_input = preprocess_input(input_data)
        scaled   = scaler.transform(df_input)
        pred     = performance_model.predict(scaled)[0]

        st.markdown("---")
        rating_text = "⭐⭐⭐ Good (Rating 3)" if pred == 3 else "⭐⭐⭐⭐ Excellent (Rating 4)"
        st.metric("Predicted Performance Rating", rating_text)

        if pred == 4:
            st.success("🌟 Excellent performer! Consider for promotion or leadership roles.")
        else:
            st.info("👍 Good performer. Provide training opportunities to reach excellence.")

# ================================
# PAGE 4 — BULK PREDICTION
# ================================
def page_bulk(attrition_model, scaler):
    st.title("📂 Bulk Attrition Prediction")
    st.markdown("Upload a CSV file with employee data to predict attrition for multiple employees at once.")
    st.markdown("---")

    st.subheader("Step 1 — Download Template")
    sample = pd.DataFrame([{
        "Age": 35, "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800, "Department": "Sales",
        "DistanceFromHome": 10, "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male", "HourlyRate": 65,
        "JobInvolvement": 3, "JobLevel": 2,
        "JobRole": "Sales Executive", "JobSatisfaction": 3,
        "MaritalStatus": "Single", "MonthlyIncome": 5000,
        "MonthlyRate": 14000, "NumCompaniesWorked": 2,
        "OverTime": "Yes", "PercentSalaryHike": 14,
        "PerformanceRating": 3, "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1, "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 3, "WorkLifeBalance": 3,
        "YearsAtCompany": 5, "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2
    }])
    csv_template = sample.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Sample Template",
                       csv_template, "employee_template.csv", "text/csv")

    st.markdown("---")
    st.subheader("Step 2 — Upload Your File")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.success(f"✅ File uploaded: {len(df_upload)} employees found")
        st.dataframe(df_upload.head(5))

        if st.button("🚀 Run Bulk Prediction"):
            results = []
            for _, row in df_upload.iterrows():
                try:
                    df_input = preprocess_input(row.to_dict())
                    scaled   = scaler.transform(df_input)
                    prob     = attrition_model.predict_proba(scaled)[0][1]
                    pred     = 1 if prob >= 0.3 else 0
                    risk     = "High" if prob >= 0.5 else "Medium" if prob >= 0.25 else "Low"
                    results.append({
                        "Attrition_Prediction": "Yes" if pred == 1 else "No",
                        "Probability_%": round(prob * 100, 1),
                        "Risk_Level": risk
                    })
                except:
                    results.append({
                        "Attrition_Prediction": "Error",
                        "Probability_%": 0,
                        "Risk_Level": "Unknown"
                    })

            df_results = pd.concat([
                df_upload.reset_index(drop=True),
                pd.DataFrame(results)
            ], axis=1)

            st.markdown("---")
            st.subheader("Step 3 — Results")
            st.dataframe(df_results)

            col1, col2, col3 = st.columns(3)
            r = pd.DataFrame(results)
            col1.metric("🔴 High Risk",   (r["Risk_Level"] == "High").sum())
            col2.metric("🟡 Medium Risk", (r["Risk_Level"] == "Medium").sum())
            col3.metric("🟢 Low Risk",    (r["Risk_Level"] == "Low").sum())

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_results.to_excel(writer, index=False, sheet_name="Predictions")
            st.download_button(
                "📥 Download Results as Excel",
                output.getvalue(),
                "attrition_predictions.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ================================
# PAGE 5 — MODEL PERFORMANCE
# ================================
def page_metrics():
    st.title("📉 Model Performance Metrics")
    st.markdown("Evaluation results of all trained models saved during training.")
    st.markdown("---")

    conn = sqlite3.connect(DB_PATH)
    df_metrics = pd.read_sql("SELECT * FROM model_metrics", conn)
    conn.close()

    if df_metrics.empty:
        st.warning("No metrics found. Run train.py first.")
        return

    st.subheader("🤖 Attrition Prediction Models")
    df_att = df_metrics[df_metrics["target"] == "Attrition"].drop(
        columns=["id", "target", "created_at"], errors="ignore"
    )
    st.dataframe(df_att, use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Performance Rating Models")
    df_perf = df_metrics[df_metrics["target"] == "PerformanceRating"].drop(
        columns=["id", "target", "created_at"], errors="ignore"
    )
    st.dataframe(df_perf, use_container_width=True)

    st.markdown("---")

    st.subheader("📊 Evaluation Plots")
    col1, col2 = st.columns(2)
    plots = {
        col1: [
            ("confusion_matrix_attrition.png", "Confusion Matrix — Attrition"),
            ("roc_curve_attrition.png",         "ROC Curve — Attrition"),
        ],
        col2: [
            ("confusion_matrix_performance.png", "Confusion Matrix — Performance"),
            ("feature_importance_attrition.png", "Feature Importance — Attrition"),
        ]
    }
    for col, items in plots.items():
        for filename, title in items:
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                col.image(path, caption=title, use_container_width=True)

# ================================
# PAGE 6 — AT RISK EMPLOYEES
# ================================
def page_atrisk(df, attrition_model, scaler):
    st.title("⚠️ At Risk Employees")
    st.markdown("Employees with highest probability of leaving based on model predictions.")
    st.markdown("---")

    from sklearn.preprocessing import LabelEncoder

    df_copy = df.copy()
    df_copy["Attrition_Actual"] = df_copy["Attrition"]

    cat_cols = ["BusinessTravel", "Department", "EducationField",
                "Gender", "JobRole", "MaritalStatus", "OverTime", "Attrition"]
    le = LabelEncoder()
    for col in cat_cols:
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))

    drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    df_copy = df_copy.drop(columns=drop_cols, errors="ignore")

    df_copy["StressRisk"]       = df_copy["OverTime"] * (df_copy["MaritalStatus"] == 2).astype(int)
    df_copy["IncomeLevelRatio"] = df_copy["MonthlyIncome"] / (df_copy["JobLevel"] + 1)
    df_copy["ExperienceRate"]   = df_copy["TotalWorkingYears"] / (df_copy["Age"] + 1)
    df_copy["StagnationScore"]  = df_copy["YearsSinceLastPromotion"] / (df_copy["YearsAtCompany"] + 1)
    df_copy["LoyaltyScore"]     = df_copy["YearsWithCurrManager"] / (df_copy["YearsAtCompany"] + 1)

    X = df_copy.drop(columns=["Attrition", "Attrition_Actual"], errors="ignore")
    X_scaled = scaler.transform(X)

    probs = attrition_model.predict_proba(X_scaled)[:, 1]
    df["Attrition_Probability_%"] = (probs * 100).round(1)
    df["Risk_Level"] = pd.cut(
        probs, bins=[0, 0.25, 0.5, 1.0],
        labels=["🟢 Low", "🟡 Medium", "🔴 High"]
    )

    col1, col2 = st.columns(2)
    risk_filter = col1.selectbox("Filter by Risk Level", ["All", "🔴 High", "🟡 Medium", "🟢 Low"])
    dept_filter = col2.selectbox("Filter by Department", ["All"] + df["Department"].unique().tolist())

    df_show = df.copy()
    if risk_filter != "All":
        df_show = df_show[df_show["Risk_Level"] == risk_filter]
    if dept_filter != "All":
        df_show = df_show[df_show["Department"] == dept_filter]

    df_show = df_show.sort_values("Attrition_Probability_%", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Risk",   (df["Risk_Level"] == "🔴 High").sum())
    col2.metric("🟡 Medium Risk", (df["Risk_Level"] == "🟡 Medium").sum())
    col3.metric("🟢 Low Risk",    (df["Risk_Level"] == "🟢 Low").sum())

    st.markdown("---")
    st.dataframe(
        df_show[[
            "Age", "Department", "JobRole", "MonthlyIncome",
            "OverTime", "MaritalStatus", "YearsAtCompany",
            "Attrition_Probability_%", "Risk_Level"
        ]].head(50),
        use_container_width=True
    )

# ================================
# MAIN
# ================================
def main():
    df = load_data()
    attrition_model, performance_model, scaler = load_models()
    page = sidebar()

    if page == "eda":
        page_eda(df)
    elif page == "predict_attrition":
        page_predict_attrition(attrition_model, scaler)
    elif page == "predict_performance":
        page_predict_performance(performance_model, scaler)
    elif page == "bulk":
        page_bulk(attrition_model, scaler)
    elif page == "metrics":
        page_metrics()
    elif page == "atrisk":
        page_atrisk(df, attrition_model, scaler)

if __name__ == "__main__":
    main()
