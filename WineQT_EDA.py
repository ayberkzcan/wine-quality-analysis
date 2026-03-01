import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("WineQT.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

#Target Distribution
print(df["quality"].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(x="quality", data=df)
plt.title("Distribution of quality levels")
plt.show()

#For Coorelation
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
# As alcohol content increases, quality tends to improve.

plt.figure(figsize=(12,10))
sns.boxplot(x="quality", y="volatile acidity", data=df)
plt.title("Quality vs Volatile")
plt.xlabel("Quality")
plt.ylabel("Volatile")
plt.show()
# As volatile acidity increases, quality tends to decrease.

plt.figure(figsize=(12,10))
sns.histplot(x="alcohol", data=df)
plt.title("Histogram of Alcohol")
plt.xlabel("Alcohol")
plt.ylabel("Number of Alcohol")
plt.show()
# Dataset focus the medium levels(5-6) wines.

plt.figure(figsize=(12,10))
sns.scatterplot(x="alcohol", y="quality", data=df)
plt.title("Scatter plot of Alcohol vs Quality")
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.show()

# Feature engineering
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
df["sulfur_ratio"] = df["free sulfur dioxide"] / df["total sulfur dioxide"]

# Drop the used colums
df = df.drop(["fixed acidity", "volatile acidity", "citric acid", "free sulfur dioxide", "total sulfur dioxide"], axis=1)

print(df.isnull().sum())

y= df["quality"]
x = df.drop(["quality"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)
#stratify=y for balance, because the quality values are not of the same size and not stable. Class imbalance

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))
