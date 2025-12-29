# Project Overview
This project focuses on predicting flight ticket prices based on various flight-related features such as airline, source, destination, journey date, duration, and total stops.
The goal is to build a reliable machine learning model and deploy it as an interactive web application using Streamlit.

After performing Exploratory Data Analysis (EDA), feature engineering, and hyperparameter tuning, multiple models were evaluated.
Among them, XGBoost (XGBRegressor) delivered the best performance and was selected for deployment.

# Data Cleaning & Feature Engineering

The columns Route and Additional_Info were removed due to low predictive value, and extensive feature engineering was performed by extracting day and month from Date_of_Journey, hour and minute from departure and arrival times, converting Duration into total minutes, encoding categorical variables, handling missing values, and preparing the data for modeling.

# Exploratory Data Analysis (EDA)
EDA was conducted to understand how flight prices vary with airlines, routes, number of stops, journey dates, and duration, helping identify important patterns, detect skewness and outliers, and select meaningful features for model training.

# Hyperparameter Tuning
Hyperparameter tuning was applied using GridSearchCV and RandomizedSearchCV to optimize parameters such as number of estimators, maximum depth, learning rate, and subsampling, improving model generalization and reducing overfitting.🏆 Final Model Selection

After evaluation, XGBoost Regressor was selected as the final model due to its superior performance, achieving approximately 93% accuracy on training data and 87% accuracy on test data, indicating strong predictive capability on unseen flights.

# Challenges Faced
Key challenges included extracting meaningful features from time and duration columns, handling high-cardinality categorical variables, tuning complex models without overfitting, and maintaining consistent performance between training and test datasets.

# Demo

# Conclusion

This project demonstrates a complete real-world machine learning workflow, covering data preprocessing, exploratory analysis, model tuning, evaluation, and deployment, resulting in a reliable flight price prediction system with practical usability.
