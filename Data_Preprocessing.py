import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the feature-engineered dataset
df = pd.read_csv("data/feature_engineered_data.csv")

# Convert 'Yes/No' categorical columns to binary (0/1)
yes_no_columns = ['HasMortgage', 'HasDependents', 'HasCoSigner']
for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Define numerical and categorical features
num_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Loan_Income_Ratio']
cat_features = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 
                'DTI_Category', 'CreditScore_Category', 'Employment_Stability', 'LoanTerm_Category']

target = 'Default'

# Define transformers
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers in a column transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Split dataset before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply preprocessing (standardize numerical values and encode categorical variables)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Apply SMOTE to the **already encoded numerical dataset**
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Save preprocessed data
np.save("data/X_train.npy", X_train_resampled)
np.save("data/X_test.npy", X_test_preprocessed)
np.save("data/y_train.npy", y_train_resampled)
np.save("data/y_test.npy", y_test)

print("✅ Data Preprocessing Completed Successfully! Preprocessed data saved in 'data/' directory.")
