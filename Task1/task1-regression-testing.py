import mlflow
from mlflow.models import infer_signature

import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import KBinsDiscretizer


from ucimlrepo import fetch_ucirepo


def main():


    # Fetch dataset
    print("Fetching dataset...")
    wine_quality = fetch_ucirepo(id=186)

    # Data (as pandas dataframes)
    print("Data fetched successfully, formatting data...")
    x = wine_quality.data.features
    y = wine_quality.data.targets

     # Preprocessing
    ## Remove the useless features
    """
    To select the features that are useful to the model, you need to know the features that make the model more accurate. 
    In this case, the feutures that tell us the how good is the wine. These are:
    - volatile acidity
    - density
    - pH
    - alcohol

    NOTE: The features that are not the best, use all the features give us a better result. Choose other features to see the difference.
    """

    print ("Selecting the useful features...")
    print(x.head())
    print(x.columns.tolist())
    x = x[['volatile_acidity', 'density', 'pH', 'alcohol']]
    print(x.head())
    print(x.columns.tolist())
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

   


    # REGRESSION
    """
    For regression, evaluate the model using metrics like Mean Absolute
    Error Mean Absolute Error (MAE), Mean Square Error (MSE), or R².
    """

    ## LINEAR REGRESSION
    print("Running Linear Regression...")
    # Select only the first column of the dataset and save it as dataframe

    # for i in range(x.shape[1]):
    #     print(f"Running Linear Regression for feature {i} , {x.columns[i]}")
    #     x_for_lineal = x.iloc[:, [i]] 
    #     x_train_lineal, x_test_lineal, y_train_lineal, y_test_lineal = train_test_split(x_for_lineal, y, test_size=0.2, random_state=42)

    #     model = LinearRegression()
    #     model.fit (x_train_lineal, y_train_lineal)
    #     y_pred = model.predict(x_test_lineal)
    #     mae = mean_absolute_error(y_test_lineal, y_pred)
    #     mse = mean_squared_error(y_test_lineal, y_pred)
    #     r2 = r2_score(y_test_lineal, y_pred)
    #     print(f"Mean Absolute Error: {mae}")
    #     print(f"Mean Squared Error: {mse}")
    #     print(f"R²: {r2}")

    #      # Representation of the data in a graph
    #     plt.scatter(x_test_lineal, y_test_lineal, color = 'red')
    #     plt.plot(x_test_lineal, model.predict(x_test_lineal), color = 'blue')
    #     plt.title(f'Wine Quality vs {x.columns[i]} (Test set)')
    #     plt.xlabel('Alcohol')
    #     plt.ylabel('Wine Quality')
    #     plt.savefig(f'plots/plot_feature_{x.columns[i]}.png')
    #     plt.close()

    """ 
    R² values are generally low, indicating that individual features do not explain a large proportion of the variability in wine quality. 
    This suggests that wine quality is a complex phenomenon influenced by multiple factors and possibly by interactions between them.
    """

    ## MULTIPLE LINEAR REGRESSION
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R²: {r2}")

    ## LASSO REGRESSION
    print("Running Lasso Regression...")
    model = Lasso()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R²: {r2}")
    
    ## RIDGE REGRESSION
    print("Running Ridge Regression...")
    model = Ridge()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R²: {r2}")

    # DECISION TREE - REGRESSION
    print("Running Decision Tree Regression...")
    model = DecisionTreeRegressor(criterion='squared_error', max_depth=4, random_state=0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R²: {r2}")


if __name__ == "__main__":
    main()