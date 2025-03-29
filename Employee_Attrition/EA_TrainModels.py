import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from EA_Preprocess import load_data, preprocess_data
from EA_HelperFunctions import save_model
import joblib

# Load datasets and preprocess
print("before load the dataset")
df_EmpAttr = load_data("data/Employee_Attrition.csv")
df_EmpAttr, scaler_EmpAttr = preprocess_data(df_EmpAttr, 'EmpAttr')

# Train Employee Attrition model
X_EmpAttr = df_EmpAttr.drop(columns=['Attrition', 'EmployeeNumber'])
y_EmpAttr = df_EmpAttr['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X_EmpAttr, y_EmpAttr, test_size=0.3, random_state=42)

#model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#model_xgb.fit(X_train, y_train)
#model_xgb = xgb.XGBClassifier(scale_pos_weight=len(y_train) / sum(y_train), 
#                               use_label_encoder=False, 
#                               eval_metric='logloss')
#model_xgb.fit(X_train, y_train)

#model_lgb = lgb.LGBMClassifier(class_weight='balanced')
#model_lgb.fit(X_train, y_train)

#logreg_model_EmpAttr = LogisticRegression(class_weight='balanced')
#logreg_model_EmpAttr.fit(X_train, y_train)

svm_model = SVC(class_weight='balanced', probability=True)
svm_model.fit(X_train, y_train)

#rf_model_EmpAttr = RandomForestClassifier()
#rf_model_EmpAttr.fit(X_train, y_train)

# Evaluate Employee Attrition model
y_pred_EmpAttr = svm_model.predict(X_test)
accuracy_EmpAttr = accuracy_score(y_test, y_pred_EmpAttr)
precision_EmpAttr = precision_score(y_test, y_pred_EmpAttr, average='weighted')
recall_EmpAttr = recall_score(y_test, y_pred_EmpAttr, average='weighted')
f1_EmpAttr = f1_score(y_test, y_pred_EmpAttr, average='weighted')


print("before save the model")
# Save models and scalers
save_model(svm_model, 'models/EmpAttr_model.pkl')

joblib.dump(scaler_EmpAttr, 'models/scaler_EmpAttr.pkl')


# Save metrics to display later in the Streamlit app
model_EmpAttr_metrics = {
    "Model": ["Employee Attrition"],
    "Accuracy": [accuracy_EmpAttr],
    "Precision": [precision_EmpAttr],
    "Recall": [recall_EmpAttr],
    "F1-Score": [f1_EmpAttr]
}

model_EmpAttr_metrics_df = pd.DataFrame(model_EmpAttr_metrics)
model_EmpAttr_metrics_df.to_csv('models/model_EmpAttr_metrics.csv', index=False)  # Save metrics to a CSV for later use in Streamlit

