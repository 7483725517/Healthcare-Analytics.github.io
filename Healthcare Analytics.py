import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Healthcare Analytics Prediction.csv")

print(df.head())
df.shape
print(df.shape)
df.info()
print(df.describe())
print(df.columns.tolist())
print(df.isnull().sum())
print(df.nunique())

# Convert to datetime
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], dayfirst=True, errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], dayfirst=True, errors='coerce')

# Feature Engineering
df['Admission Hour'] = df['Date of Admission'].dt.hour
df['Admission Day'] = df['Date of Admission'].dt.dayofweek
df['Month'] = df['Date of Admission'].dt.month
df['Season'] = df['Month'] % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall

# ----------------------------------
# 📊 EDA Visualizations
# ----------------------------------

# 1. Heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 2. Violin plot of Length of Stay by Medical Condition
plt.figure(figsize=(14, 6))
sns.violinplot(x='Medical Condition', y='Length of Stay', data=df)
plt.title("Length of Stay by Condition")
plt.xticks(rotation=45)
plt.show()

# 3. Count of admissions by hour (High-traffic hours)
plt.figure(figsize=(12, 6))
sns.countplot(x='Admission Hour', data=df)
plt.title("ER Admissions by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Admissions")
plt.show()

# 4. Admissions by Season
plt.figure(figsize=(8, 5))
sns.countplot(x='Season', data=df)
plt.title("Admissions by Season")
plt.xlabel("Season (1=Winter, 4=Fall)")
plt.ylabel("Count")
plt.show()

# 5. Line Plot - Admissions over Time
daily = df.groupby('Date of Admission').size().reset_index(name='Count')
plt.figure(figsize=(14, 5))
plt.plot(daily['Date of Admission'], daily['Count'])
plt.title("Daily ER Admissions Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Admissions")
plt.tight_layout()
plt.show()

# ----------------------------------
# 🤖 Predictive Modeling
# ----------------------------------

# Time Series Forecast with Prophet
prophet_df = daily.rename(columns={'Date of Admission': 'ds', 'Count': 'y'})
model = Prophet()
model.fit(prophet_df)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Forecast plot
model.plot(forecast)
plt.title("30-Day ER Admission Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Admissions")
plt.show()

# Classification: Predicting High-Risk Patients
df['High_Risk'] = (df['Length of Stay'] > 7).astype(int)
features = ['Age', 'Gender', 'Blood Type', 'Admission Type', 'Medical Condition']
df_model = df[features + ['High_Risk']].dropna()

# Encode categorical variables
for col in ['Gender', 'Blood Type', 'Admission Type', 'Medical Condition']:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])

X = df_model[features]
y = df_model['High_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {clf.score(X_test, y_test):.2f}")
