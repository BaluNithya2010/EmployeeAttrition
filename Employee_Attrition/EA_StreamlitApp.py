import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from EA_Preprocess import preprocess_data
from streamlit_option_menu import option_menu
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load pre-trained models and scalers
model_EmpAttr = joblib.load('models/EmpAttr_model.pkl')
scaler_EmpAttr = joblib.load('models/scaler_EmpAttr.pkl')

# Load model metrics from the saved CSV
model_EmpAttr_metrics_df = pd.read_csv('models/model_EmpAttr_metrics.csv')


# Streamlit app
st.title('Employee Attrition Prediction')

st.write("""
### Predict whether an employee is at risk of attrition based on various factors.
""")

# Collect user input for prediction
Age = st.slider('Age', 20, 60, 35)
BusinessTravel = st.selectbox('BusinessTravel', ['Travel_Rarely','Travel_Frequently','Non-Travel'])
DailyRate = st.number_input('DailyRate', min_value=0, max_value=5000, value=1000)
Department = st.selectbox('Department', ['Sales','human Resources','Research & Development'])
DistanceFromHome = st.number_input('DistanceFromHome', min_value=0, max_value=100, value=0)
Education = st.number_input('Education', min_value=0, max_value=10, value=0)
EducationField = st.selectbox('EducationField', ['Human Resources','Life Sciences','Marketing', 'Medical', 'Other', 'Technical Degree'])
EmployeeCount = st.number_input('EmployeeCount', min_value=0, max_value=1, value=1)
#EmployeeNumber = st.number_input('EmployeeNumber', min_value=0, max_value=10000, value=0)
EnvironmentSatisfaction = st.selectbox('EnvironmentSatisfaction', [1, 2, 3, 4])
Gender = st.selectbox('Gender', ['Male','Female'])
HourlyRate = st.number_input('HourlyRate', min_value=0, max_value=110, value=0)
JobInvolvement = st.selectbox('JobInvolvement', [1, 2, 3, 4])
JobLevel = st.selectbox('JobLevel', [1, 2, 3, 4, 5])
JobRole = st.selectbox('JobRole', ['Healthcare Representative','Human Resources','Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director','Research Scientist','Sales Executive', 'Sales Representative'])
JobSatisfaction = st.selectbox('JobSatisfaction', [1, 2, 3, 4])
MaritalStatus = st.selectbox('MaritalStatus', ['Single','Married','Divorced'])
MonthlyIncome = st.number_input('MonthlyIncome', min_value=0, max_value=100000, value=0)
MonthlyRate = st.number_input('MonthlyRate', min_value=0, max_value=100000, value=0)
NumCompaniesWorked = st.number_input('NumCompaniesWorked', min_value=0, max_value=20, value=0)
Over18 = st.selectbox('Over18', ['Y','N'])
OverTime = st.selectbox('OverTime', ['Yes', 'No'])
PercentSalaryHike = st.number_input('PercentSalaryHike', min_value=0, max_value=100, value=0)
PerformanceRating = st.selectbox('PerformanceRating', [1, 2, 3, 4])
RelationshipSatisfaction = st.selectbox('RelationshipSatisfaction', [1, 2, 3, 4])
StandardHours = st.number_input('StandardHours', min_value=0, max_value=100, value=80)
StockOptionLevel = st.selectbox('StockOptionLevel', [0, 1, 2, 3])
TotalWorkingYears = st.number_input('TotalWorkingYears', min_value=0, max_value=100, value=0)
TrainingTimesLastYear = st.number_input('TrainingTimesLastYear', min_value=0, max_value=10, value=0)
WorkLifeBalance = st.selectbox('WorkLifeBalance', [1, 2, 3, 4])
YearsAtCompany = st.number_input('YearsAtCompany', min_value=0, max_value=100, value=0)
YearsInCurrentRole = st.number_input('YearsInCurrentRole', min_value=0, max_value=50, value=0)
YearsSinceLastPromotion = st.number_input('YearsSinceLastPromotion', min_value=0, max_value=50, value=0)
YearsWithCurrManager = st.number_input('YearsWithCurrManager', min_value=0, max_value=50, value=0)

   
# Create input dataframe for prediction
input_data = pd.DataFrame([[Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField,
EmployeeCount, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement,
JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear,
WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager]], 
                            columns=[ "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education", "EducationField",
                                "EmployeeCount", "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement",
                                "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
                                "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
                                "StandardHours", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
                                "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"])

# Preprocessing for liver disease
input_data_EmpAttr, scaler_EmpAttr = preprocess_data(input_data, 'EmpAttr', scaler_EmpAttr)

# Make prediction for EmpAttr disease
pred_EmpAttr = model_EmpAttr.predict(input_data_EmpAttr)
# Instead of the usual threshold of 0.5
#pred_EmpAttr = (model_EmpAttr.predict_proba(input_data_EmpAttr)[:, 1] > 0.4).astype(int)
st.write("pred_EmpAttr :", pred_EmpAttr[0])
# Display prediction result
#st.subheader(f"Employee Attrition Prediction:",  pred_EmpAttr[0] )
st.subheader(f"Employee Attrition Prediction: {'Yes' if pred_EmpAttr[0] == 'Yes' else 'No'}")


# Interactive Plot: Probability of Employee Attrition
proba_EmpAttr = model_EmpAttr.predict_proba(input_data_EmpAttr)[0]
reshaped_probabilities = proba_EmpAttr.reshape(1, 2)
proba_df_EmpAttr = pd.DataFrame(reshaped_probabilities, columns=["No", "Yes"], index=["Probability"])
fig_EmpAttr = px.bar(proba_df_EmpAttr.T, title="Employee Attrition Prediction Probability", labels={'value': 'Probability', 'variable': 'Prediction'})
st.plotly_chart(fig_EmpAttr)


# Display liver Model Metrics
st.subheader('Model Evaluation Metrics')
st.table(model_EmpAttr_metrics_df)

    

    
