# Task 1: Data Preprocessing & Exploration

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Step 2: Load dataset
file_path = "/content/Face Recognition Image.xlsx"  # <-- upload this file in Colab
df = pd.read_excel(file_path)

# Step 3: Basic info about dataset
print("Shape of dataset:", df.shape)
print("\nColumns:\n", df.columns)
print("\nFirst 5 rows:\n", df.head())
print("\nDataset Info:\n")
df.info()

# Step 4: Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing numeric values with mean, categorical with mode
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Step 5: Descriptive statistics
print("\nSummary Statistics:\n", df.describe(include="all"))

# Step 6: Correlation Heatmap (only numeric features)
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Step 7: Distribution of each numeric feature
df.hist(figsize=(12,8), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Step 8: Normalize numeric data
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nNormalized Dataset (first 5 rows):\n", df.head())
