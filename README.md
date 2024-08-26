# 🏡 Ames Housing Dataset Project


<div align="center">
  <img src="https://cdn.pixabay.com/animation/2023/03/26/21/52/21-52-36-662_512.gif" alt="Ames Housing Project">
</div>

### Predicting House Prices with Machine Learning & Deep Learning



This project aims to predict house prices using the Ames Housing dataset, which contains detailed information on property sales. Through a comprehensive analysis, I explore various machine learning algorithms, data preprocessing techniques, and deep learning methods to develop accurate predictive models.

## 📋 Project Overview

In this project, I applied several machine learning models such as:

- **Linear Regression**
- **Random Forest Regressor**
- **Decision Tree Regressor**
- **XGBoost**
- **LightGBM**
- **CatBoost**

I also implemented a deep learning model using **TensorFlow** and **Keras**. Each model was trained, evaluated, and compared to determine the best-performing algorithm for house price prediction.

## 🚀 Key Steps

### 1. Data Preprocessing
- **Data Loading:** The dataset was loaded from a zipped file and inspected for missing values and outliers.
- **Missing Value Imputation:** Numerical features were imputed using the mean strategy, while categorical features were imputed using the most frequent strategy.
- **One-Hot Encoding:** Categorical variables were transformed into dummy variables suitable for machine learning algorithms.
- **Feature Scaling:** Numerical features were standardized using `StandardScaler` to normalize the input data.
- **Handling Ordinal Variables:** Ordinal features like quality ratings were encoded using `LabelEncoder` to preserve their ranked nature.

### 2. Model Implementation
- **Linear Regression:** A basic linear model was used as a baseline, evaluated with Mean Squared Error (MSE) and R² metrics.
- **Random Forest Regressor:** An ensemble method was used to capture more complex patterns in the data.
- **Decision Tree Regressor:** A single decision tree model was implemented to explore non-linear relationships.
- **XGBoost:** An advanced gradient boosting model with hyperparameter tuning was used to achieve enhanced performance.
- **LightGBM:** Another gradient boosting model, optimized for speed and performance, was applied.
- **CatBoost:** This model, known for handling categorical data efficiently, was also tested for comparison.

### 3. Deep Learning Model (TensorFlow + Keras)
- **Neural Network Architecture:** A sequential neural network with several layers, including dropout and L2 regularization, was developed to prevent overfitting.
- **Learning Rate Schedule:** Exponential learning rate decay was implemented to achieve efficient learning.
- **Early Stopping:** Training was monitored with early stopping to halt when validation loss stopped improving.

### 4. Model Evaluation
- **Regression Metrics:** All models were evaluated using MSE, RMSE, and R². Training and validation loss were tracked during deep learning training.
- **Training and Validation Plots:** For the deep learning model, loss plots over epochs were generated to assess model convergence and overfitting.

## 📊 Results

The performance of each model was compared in terms of MSE, RMSE, and R² metrics:

- **XGBoost** and **CatBoost** emerged as the top performers, demonstrating superior accuracy.
- **Random Forest Regressor** also showed competitive results.
- The **deep learning model** provided insights into the application of neural networks for this problem and enabled experimentation with different architectures.

## 🏆 Conclusion

This project successfully explored a range of machine learning techniques and deep learning methods for predicting house prices using the Ames dataset. The experience underscored the importance of data preprocessing, feature engineering, and model evaluation when building robust predictive models.

