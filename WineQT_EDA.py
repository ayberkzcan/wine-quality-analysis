import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("WineQT.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df["quality"].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(x="quality", data=df)
plt.title("Distribution of quality levels")
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(12,10))
sns.boxplot(x="quality", y="alcohol", data=df)
plt.title("Quality vs Alcohol")
plt.xlabel("Quality")
plt.ylabel("Alcohol")
plt.show()
# Alcohol arttıkça quality artma eğiliminde.

plt.figure(figsize=(12,10))
sns.boxplot(x="quality", y="volatile acidity", data=df)
plt.title("Quality vs Volatile")
plt.xlabel("Quality")
plt.ylabel("Volatile")
plt.show()
# Volatile acidity arttıkça quality düşme eğiliminde.

plt.figure(figsize=(12,10))
sns.histplot(x="alcohol", data=df)
plt.title("Histogram of Alcohol")
plt.xlabel("Alcohol")
plt.ylabel("Number of Alcohol")
plt.show()
# Dataset orta kalite (5-6) şaraplarda yoğunlaşmış.

plt.figure(figsize=(12,10))
sns.scatterplot(x="alcohol", y="quality", data=df)
plt.title("Scatter plot of Alcohol vs Quality")
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.show()
