# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load Excel file
file_path = "Face Recognition Image.xlsx"   # Update path if needed
df = pd.read_excel(file_path)

# -------------------------
# 1. Basic Info
# -------------------------
print("Shape of dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Handle Missing Values

# Example approaches (choose as per your need)
df = df.drop_duplicates()  # Remove duplicate rows

# Fill numeric columns with mean, categorical with mode
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -------------------------
# 3. Normalization / Scaling
# -------------------------
# Example: MinMax scaling numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nFirst 5 rows after normalization:")
print(df.head())

# -------------------------
# 4. Basic EDA
# -------------------------
# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Distribution of numeric columns
df[num_cols].hist(figsize=(10,8), bins=20)
plt.suptitle("Distribution of Numeric Features")
plt.show()

# Count plot for categorical columns (if any)
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[col])
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.show()
