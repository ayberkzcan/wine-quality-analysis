# 🍷 Wine Quality Prediction

## 📌 Project Overview

This project focuses on analyzing and predicting wine quality based on physicochemical properties.
The goal is to explore the dataset, extract meaningful insights, and build a classification model.

## 📊 Dataset

	•	Dataset: Wine Quality Dataset
	•	Number of samples: 1143
	•	Features: 11 numerical features + target (quality)

## 🔍 Exploratory Data Analysis (EDA)

	•	Analyzed distribution of wine quality
	•	Visualized relationships using:
	•	Histograms
	•	Boxplots
	•	Scatter plots
	•	Correlation heatmap

Key Insights:

	•	Higher alcohol content tends to increase quality
	•	Higher volatile acidity tends to decrease quality
	•	Dataset is imbalanced (mostly quality 5–6 wines)

## ⚙️ Feature Engineering

Created new features to improve model performance:

	•	total_acidity = fixed acidity + volatile acidity + citric acid
	•	sulfur_ratio = free sulfur dioxide / total sulfur dioxide

Removed original correlated columns to reduce multicollinearity.

## 🤖 Model

	•	Model: Logistic Regression
	•	Applied:
	•	StandardScaler
	•	Stratified train-test split
	•	Class balancing (class_weight="balanced")

## 📈 Evaluation

	•	Accuracy
	•	Confusion Matrix
	•	Classification Report

### ⚠️ Due to class imbalance, accuracy is limited and minority classes are harder to predict.

## 🚀 Future Improvements

	•	Apply advanced models (Random Forest, XGBoost)
	•	Hyperparameter tuning
	•	Improve class imbalance handling

## 🛠️ Technologies

	•	Python
	•	Pandas, NumPy
	•	Matplotlib, Seaborn
	•	Scikit-learn

## 📂 Project Status

✅ Completed (basic version)
🔄 Open for improvements
