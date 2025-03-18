import pandas as pd
import numpy as np


# Load the dataset (correct method for .xlsx files)
df = pd.read_excel("data/raw_data.xlsx")


# Drop LoanID as it's not a useful predictor
df.drop(columns=['LoanID'], inplace=True)

# ------------------------------
# FEATURE ENGINEERING
# ------------------------------

# 1. Create Debt-to-Income Ratio Buckets
def categorize_dti(dti):
    if dti < 0.2:
        return "Low Risk"
    elif dti < 0.4:
        return "Medium Risk"
    elif dti < 0.6:
        return "High Risk"
    else:
        return "Very High Risk"

df['DTI_Category'] = df['DTIRatio'].apply(categorize_dti)

# 2. Categorize Credit Score into Risk Buckets
def categorize_credit(score):
    if score < 580:
        return "Poor"
    elif score < 670:
        return "Fair"
    elif score < 740:
        return "Good"
    else:
        return "Excellent"

df['CreditScore_Category'] = df['CreditScore'].apply(categorize_credit)

# 3. Loan Amount to Income Ratio
df['Loan_Income_Ratio'] = df['LoanAmount'] / df['Income']

# 4. Employment Stability Feature (Years Employed Categories)
def categorize_employment(months):
    if months < 12:
        return "Less than 1 year"
    elif months < 36:
        return "1-3 years"
    elif months < 60:
        return "3-5 years"
    else:
        return "5+ years"

df['Employment_Stability'] = df['MonthsEmployed'].apply(categorize_employment)

# 5. Loan Term Categories (Short, Medium, Long Term)
def categorize_loan_term(term):
    if term <= 24:
        return "Short-Term"
    elif term <= 48:
        return "Medium-Term"
    else:
        return "Long-Term"

df['LoanTerm_Category'] = df['LoanTerm'].apply(categorize_loan_term)

# ------------------------------
# SAVE INTERMEDIATE DATA FOR PREPROCESSING
# ------------------------------
df.to_csv("data/feature_engineered_data.csv", index=False)

print("Feature Engineering Completed. Data saved as 'feature_engineered_data.csv' for preprocessing.")
