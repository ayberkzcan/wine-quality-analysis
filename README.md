# 🍷 Wine Quality Prediction

## 📌 Project Overview

This project aims to explore the Wine Quality Dataset and build a predictive model to understand the factors affecting wine quality.
The focus is on data analysis, feature engineering, and model building, with continuous improvements planned.

## 📊 Dataset
	•	Source: Wine Quality Dataset￼
	•	Samples: 1143
	•	Features: 11 numerical variables + target (quality)

## 🔍 Exploratory Data Analysis (EDA)

We are actively analyzing the dataset to extract meaningful insights:
	•	Quality distribution visualization
	•	Relationships between features and target using histograms, boxplots, scatter plots, and correlation heatmaps

Initial Insights:
	•	Higher alcohol content tends to correlate with higher quality
	•	Higher volatile acidity tends to reduce quality
	•	Dataset is imbalanced (mostly quality 5–6 wines)

## ⚙️ Feature Engineering

New features added to enhance predictive power:
	•	total_acidity = fixed acidity + volatile acidity + citric acid
	•	sulfur_ratio = free sulfur dioxide / total sulfur dioxide

We removed original correlated columns to reduce redundancy and improve model efficiency.

## 🤖 Current Model
	•	Logistic Regression (with balanced class weights)
	•	Scaled features using StandardScaler
	•	Stratified train-test split to preserve class distribution

Evaluation:
	•	Accuracy, confusion matrix, and classification report
	•	Current results show room for improvement on minority classes

## 🚀 Next Steps
	•	Explore advanced models (Random Forest, XGBoost)
	•	Hyperparameter tuning for better performance
	•	Improved handling of class imbalance
	•	Continuous updates on GitHub as new insights and models are developed

## 🛠️ Technologies

	•	Python: Pandas, NumPy
	•	Visualization: Matplotlib, Seaborn
	•	Modeling: Scikit-learn

## 📂 Project Status

Ongoing – actively exploring, engineering features, and improving models
