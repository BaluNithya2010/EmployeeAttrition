import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Handle missing values for numerical and categorical columns
def handle_missing_values(df, numerical_columns, categorical_columns):
    # Convert numerical columns to numeric (if not already)
    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
    
    # Handle missing numerical columns (replace NaN with column mean)
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())  # Fill missing numerical with mean
    
    # Handle missing categorical columns (replace NaN with mode)
    if categorical_columns:
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])  # Fill categorical with mode
    
    return df


# Encode categorical columns using Label Encoding (or one-hot encoding)
def encode_categorical_data(df, categorical_columns):
    # Check if categorical_columns is not empty
    if not categorical_columns:
        print("No categorical columns to encode.")
        return df

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # You can save the encoder and reuse it in inference if needed
    return df

# Scale numerical features
def scale_numerical_features(df, numerical_columns, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])  # Fit and transform during training
    else:
        df[numerical_columns] = scaler.transform(df[numerical_columns])  # Transform using saved scaler during inference
    return df, scaler

# Preprocessing for liver, kidney, and Parkinson's disease data
def preprocess_data(df, inp_type, scaler=None):
    if inp_type == 'EmpAttr':
        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 
                       'MaritalStatus', 'Over18', 'OverTime']
        numerical_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 
                     'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
                     'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
                     'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                     'StandardHours','StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                     'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'WorkLifeBalance']
    # Handle missing values (if any)
    df = handle_missing_values(df, numerical_columns, categorical_columns)

    # Encode categorical columns using Label Encoding
    df = encode_categorical_data(df, categorical_columns)

    # Scale numerical features
    df, scaler = scale_numerical_features(df, numerical_columns, scaler)

    return df, scaler


