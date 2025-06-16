import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet

df = pd.read_csv('Healthcare Analytics Prediction.csv')
print(df.head())
df.shape
print(df.shape)
df.info()
print(df.describe())
print(df.columns.tolist())
print(df.isnull().sum())
print(df.nunique())

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], dayfirst=True)
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], dayfirst=True)

daily_admissions = df.groupby('Date of Admission').size().reset_index(name='y')
daily_admissions.columns = ['ds', 'y']
model = Prophet()
model.fit(daily_admissions)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig = model.plot(forecast)
plt.title("Predicted ER Admissions for Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Number of Admissions")
plt.tight_layout()
plt.show()

emergency_df = df[df['Admission Type'] == 'Emergency']
admission_counts = emergency_df['Date of Admission'].dt.to_period('M').value_counts().sort_index()

plt.figure(figsize=(12, 6))
admission_counts.plot(kind='line', marker='o')
plt.title("Monthly ER Admissions Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Admissions")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

sns.scatterplot(data=df, x='Length of Stay', y='Billing Amount', hue='Admission Type')
plt.title("Stay Duration vs Billing")
plt.show()

df.groupby('Medical Condition')['Length of Stay'].mean().sort_values(ascending=False).plot(kind='bar', figsize=(12, 5))
plt.title("Average Stay Duration by Condition")
plt.ylabel("Average Days")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

ts_df = df.groupby('Date of Admission').size().reset_index(name='Count')
ts_df.columns = ['ds', 'y']

model = Prophet()
model.fit(ts_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

model.plot(forecast)
plt.title("Forecast of ER Admissions")
plt.show()

df['High_Risk'] = (df['Length of Stay'] > 7).astype(int)

features = ['Age', 'Gender', 'Blood Type', 'Admission Type', 'Medical Condition']
df_model = df[features + ['High_Risk']].copy()

for col in ['Gender', 'Blood Type', 'Admission Type', 'Medical Condition']:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])

X = df_model[features]
y = df_model['High_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
print("Model accuracy:", model_rf.score(X_test, y_test))


