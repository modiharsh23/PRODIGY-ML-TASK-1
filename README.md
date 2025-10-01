# **House Price Prediction Project**

This repository contains a machine learning project that predicts house prices using linear regression. The model is built based on the Ames Housing dataset available on Kaggle.

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Libraries Used](#libraries-used)

## **Project Overview**
The objective of this project is to create a simple yet effective linear regression model to predict the sale price of houses. The notebook covers all the essential steps of a machine learning workflow, including data loading, exploratory data analysis, feature engineering, model training, evaluation, and prediction.

## **Dataset**
The dataset used is the **Ames Housing dataset**, sourced from the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) competition on Kaggle. It contains various features describing residential homes in Ames, Iowa.

## **Features Used**
For this simple model, the following features were selected as predictors:
- **`GrLivArea`**: Above ground living area square feet.
- **`BedroomAbvGr`**: Number of bedrooms above ground.
- **`TotalBath`**: A custom-engineered feature combining full and half bathrooms.

The target variable is:
- **`SalePrice`**: The property's sale price in dollars.

## **Methodology**
The project follows these steps:
1. **Data Loading and Inspection**: The `train.csv` and `test.csv` files are loaded using pandas.
2. **Exploratory Data Analysis (EDA)**: The relationships between the features and the target variable (`SalePrice`) are visualized using histograms, scatter plots, and heatmaps to identify trends and correlations.
3. **Feature Engineering**: `FullBath` and `HalfBath` are combined to create a single `TotalBath` feature.
4. **Data Splitting**: The training data is split into an 80% training set and a 20% testing set.
5. **Model Training**: A `LinearRegression` model from scikit-learn is trained on the training data.
6. **Model Evaluation**: The model's performance is assessed on the testing set using **Root Mean Squared Error (RMSE)** and the **R-squared (R²)** score.
7. **Prediction**: The final model is used to generate predictions on the `test.csv` dataset, which are then saved to a `submission.csv` file.

## **Results**
The simple linear regression model achieved the following performance on the test set:
- **Root Mean Squared Error (RMSE)**: ~$53,371.56
- **R-squared (R²)**: 0.63

The R² score indicates that the model can explain approximately **63%** of the variance in house prices, which is a reasonable result for a model with only three features.

## **How to Run**
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/modiharsh23/PRODIGY-ML-TASK-1

2. Navigate to the project directory:
   ```Bash
    cd https://github.com/modiharsh23/PRODIGY-ML-TASK-1

3. Install the required libraries:
   ```Bash
   pip install numpy pandas matplotlib seaborn scikit-learn

4. Download the `train.csv` and `test.csv` files from the Kaggle competition page and place them in a DATA subfolder within the project directory.

5. Open and run the `TASK-1.ipynb` notebook in Jupyter Notebook or JupyterLab.

## **Libraries Used**
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`