import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("Healthcare Analytics Prediction.csv")
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], dayfirst=True, errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], dayfirst=True, errors='coerce')

print(df.head())
df.shape
print(df.shape)
df.info()
print(df.describe())
print(df.columns.tolist())
print(df.isnull().sum())
print(df.nunique())

# Feature Engineering
df['Admission Hour'] = df['Date of Admission'].dt.hour
df['Admission Day'] = df['Date of Admission'].dt.dayofweek
df['Month'] = df['Date of Admission'].dt.month
df['Season'] = df['Month'] % 12 // 3 + 1

# -----------------------------------
#  Extended EDA: More Visualizations
# -----------------------------------

# 1. Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 2. Violin plot
plt.figure(figsize=(14, 6))
sns.violinplot(x='Medical Condition', y='Length of Stay', data=df)
plt.title("Length of Stay by Condition")
plt.xticks(rotation=45)
plt.show()

# 3. Countplot of ER traffic by hour
plt.figure(figsize=(12, 6))
sns.countplot(x='Admission Hour', data=df, palette='mako')
plt.title("ER Admissions by Hour")
plt.show()

# 4. Countplot of ER traffic by season
plt.figure(figsize=(10, 5))
sns.countplot(x='Season', data=df, palette='Set2')
plt.title("ER Admissions by Season (1=Winter to 4=Fall)")
plt.show()

# 5. Line plot - Daily admission trends
daily = df.groupby('Date of Admission').size().reset_index(name='Count')
plt.figure(figsize=(14, 5))
plt.plot(daily['Date of Admission'], daily['Count'], color='teal')
plt.title("Daily ER Admissions Over Time")
plt.show()

# 6. Pie chart - Admission Types
plt.figure(figsize=(6, 6))
df['Admission Type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title("Admission Type Distribution")
plt.ylabel("")
plt.show()

# 7. Box plot - Billing Amount by Admission Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Admission Type', y='Billing Amount', palette='pastel')
plt.title("Billing Amount by Admission Type")
plt.show()

# 8. Histogram - Patient Age
plt.figure(figsize=(10, 5))
df['Age'].hist(bins=30, color='skyblue')
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 9. Pairplot - Age, Billing, Stay
sns.pairplot(df[['Age', 'Billing Amount', 'Length of Stay']], diag_kind='kde')
plt.suptitle("Pairwise Feature Analysis", y=1.02)
plt.show()

# 10. Bar chart - Avg. stay by condition
plt.figure(figsize=(14, 6))
df.groupby('Medical Condition')['Length of Stay'].mean().sort_values().plot(kind='bar', color='coral')
plt.title("Avg. Length of Stay by Condition")
plt.ylabel("Days")
plt.xticks(rotation=45)
plt.show()

# -----------------------------------
#  Time Series Forecast (Prophet)
# -----------------------------------

ts = df.groupby('Date of Admission').size().reset_index(name='y')
ts.columns = ['ds', 'y']
model = Prophet()
model.fit(ts)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

model.plot(forecast)
plt.title("30-Day Forecast of ER Admissions")
plt.xlabel("Date")
plt.ylabel("Predicted Volume")
plt.show()

# -----------------------------------
# Classification: High-Risk Patients
# -----------------------------------

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
print(f"✅ Random Forest Classifier Accuracy: {clf.score(X_test, y_test):.2f}")
