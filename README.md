# Healthcare-Analytics.github.io
Abstract  
This project tackles the problem of overcrowding in emergency rooms (ER) using healthcare analytics. By examining past patient admission data, we aim to find trends, predict future surges, and create strategies based on data to improve hospital resource planning and patient care efficiency.

Introduction  
ER overcrowding poses serious risks to patient outcomes and the efficiency of healthcare delivery. Long wait times, lack of staff, and poor resource distribution often occur. This project uses data analytics and machine learning techniques to study a real-world dataset of 55,500 hospital admission records. We aim to predict overcrowding patterns and suggest ways to improve operations.

Objectives of Data Analysis   
Predict busy periods in the ER to manage patient flow effectively.  
Examine factors that lead to long stays or high costs.  
Visualize patterns across hospitals, doctors, and medical conditions.  
Suggest policy and operational strategies for better resource allocation.

Data Overview  
The dataset has 16 features and 55,500 records related to patient admissions. Key columns include:  
- Age, Gender, Blood Type  
- Medical Condition, Admission Type, Date of Admission, Discharge Date  
- Hospital, Doctor, Insurance Provider  
- Billing Amount, Room Number, Length of Stay, Medication, Test Results  

Admission types include Emergency, Elective, and Routine.

Dataset Cleaning Workflow  
We converted dates (Date of Admission, Discharge Date) into datetime format.  
We dealt with outliers in Billing Amount and Length of Stay.  
Categorical variables were encoded for modeling (like Gender and Medical Condition).  
We confirmed there were no missing values in key columns.  
New features were created such as Admission Month, Weekday, and Stay Duration.

Exploratory Data Analysis (EDA)  
Age Distribution: Patients range from children to seniors, with a median age of about 45.  
Top Medical Conditions: Flu, Asthma, Heart Disease, Infections.  
Admission Trends: Peaks occur in winter months, likely during flu season.  
Hospital Load: "UI Health" has the highest number of patients.  
Length of Stay: Most stays are under 10 days, but some exceed 100 days.

Prediction Models  
We used models to predict potential overcrowding and long hospital stays:  
- Time Series Models (like ARIMA) for daily/weekly admission counts.  
- Classification Models to predict overcrowding (Logistic Regression, Random Forest).  
- Regression Models for predicting Length of Stay or Billing Amount.  

Model evaluation included metrics such as accuracy, RMSE, and AUC based on the model.

Data Visualization and Dashboards  
Interactive visuals include:  
- Heatmaps of admissions by day and hour.  
- Trendlines of admission rates by hospital and condition.  
- Distribution plots for age, billing, and stay duration.  

Dashboards were created using Python (Plotly, Seaborn) or Power BI.

Insights and Recommendations  
Overcrowding Periods: High in winter and during public holidays.  
Insurance Impact: Some providers are associated with longer hospital stays or higher bills.  
Medical Conditions: Patients with cancer and heart disease usually have longer stays.  
Operational: Shift staff and open triage zones during predicted peak times.

Future Work and Enhancements  
Integrate real-time data feeds from hospital ER systems.  
Use weather, flu outbreaks, and local event data to improve predictions.  
Apply newer models (like LSTM, XGBoost) for better forecasting.  
Include patient satisfaction and outcome data to improve quality.

Conclusion  
This project shows how healthcare data analytics can effectively identify patterns in ER admissions and help manage hospital resources. Predictive insights can reduce overcrowding, shorten wait times, and ultimately improve the quality of patient care.
