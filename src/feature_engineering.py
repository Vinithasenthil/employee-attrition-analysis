import pandas as pd
import os
import sys

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.preprocess import preprocess_pipeline

# -------------------------------
# FEATURE 1 — STRESS RISK SCORE
# -------------------------------
def add_stress_risk(df):
    """
    High stress = OverTime=Yes AND MaritalStatus=Single
    OverTime:      Yes=1, No=0
    MaritalStatus: Single=2, Married=1, Divorced=0
    
    Logic: If employee works overtime AND is single
           → highest stress → more likely to leave
    """
    df["StressRisk"] = df["OverTime"] * (df["MaritalStatus"] == 2).astype(int)
    print("Added: StressRisk")
    return df

# -------------------------------
# FEATURE 2 — INCOME TO LEVEL RATIO
# -------------------------------
def add_income_level_ratio(df):
    """
    Is the employee underpaid for their job level?
    
    Low ratio → underpaid → likely to leave
    High ratio → fairly paid → likely to stay
    """
    df["IncomeLevelRatio"] = df["MonthlyIncome"] / (df["JobLevel"] + 1)
    print("Added: IncomeLevelRatio")
    return df

# -------------------------------
# FEATURE 3 — EXPERIENCE RATE
# -------------------------------
def add_experience_rate(df):
    """
    How much experience relative to age?
    
    TotalWorkingYears / Age
    
    Low ratio → started working late or gaps → less stable
    High ratio → experienced for their age → more stable
    """
    df["ExperienceRate"] = df["TotalWorkingYears"] / (df["Age"] + 1)
    print("Added: ExperienceRate")
    return df

# -------------------------------
# FEATURE 4 — STAGNATION SCORE
# -------------------------------
def add_stagnation_score(df):
    """
    Has the employee been stuck without promotion?
    
    YearsSinceLastPromotion / (YearsAtCompany + 1)
    
    High score → long time without promotion → frustrated → likely to leave
    Low score  → recently promoted → happy → likely to stay
    """
    df["StagnationScore"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    print("Added: StagnationScore")
    return df

# -------------------------------
# FEATURE 5 — LOYALTY SCORE
# -------------------------------
def add_loyalty_score(df):
    """
    How long has employee been with current manager?
    Relative to total years at company.
    
    High score → stable relationship → likely to stay
    Low score  → frequent manager changes → unsettled → may leave
    """
    df["LoyaltyScore"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
    print("Added: LoyaltyScore")
    return df

# -------------------------------
# MASTER FUNCTION
# -------------------------------
def engineer_features(df):
    """
    Adds all new derived features to DataFrame.
    Call this after preprocess_pipeline().
    """
    df = add_stress_risk(df)
    df = add_income_level_ratio(df)
    df = add_experience_rate(df)
    df = add_stagnation_score(df)
    df = add_loyalty_score(df)
    print(f"\nFeature engineering done!")
    print(f"New shape: {df.shape}")
    print(f"New columns added: StressRisk, IncomeLevelRatio, ExperienceRate, StagnationScore, LoyaltyScore")
    return df

# -------------------------------
# FULL PIPELINE
# -------------------------------
def feature_pipeline():
    """
    Full pipeline:
    preprocess → feature engineering → return X, y
    """
    X, y = preprocess_pipeline()

    # Combine X and y temporarily to add features
    df = X.copy()
    df["Attrition"] = y.values

    # Add new features
    df = engineer_features(df)

    # Split back
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    return X, y

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    X, y = feature_pipeline()
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nAll columns:\n{X.columns.tolist()}")
    print(f"\nSample new features:")
    print(X[["StressRisk", "IncomeLevelRatio", "ExperienceRate",
             "StagnationScore", "LoyaltyScore"]].head(5))