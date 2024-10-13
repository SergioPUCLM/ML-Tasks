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


from sklearn.preprocessing import KBinsDiscretizer


from ucimlrepo import fetch_ucirepo


def main():
    # Set up MLflow
    print("Setting up MLflow...")
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("WINE QUALITY")


    # Fetch dataset
    print("Fetching dataset...")
    wine_quality = fetch_ucirepo(id=186)

    # Data (as pandas dataframes)
    print("Data fetched successfully, formatting data...")
    x = wine_quality.data.features
    y = wine_quality.data.targets

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    # REGRESSION
    # For regression, evaluate the model using metrics like Mean Absolute
    # Error Mean Absolute Error (MAE), Mean Square Error (MSE), or R².
    ## LINEAR REGRESSION
    print("Running Linear Regression...")
    # Select only the first column of the dataset and save it as dataframe
    """
    NOTE: Erase the comment to run the linear regression for each feature
    for i in range(x.shape[1]):
        print(f"Running Linear Regression for feature {i}")
        x_for_lineal = x.iloc[:, [i]] 
        x_train_lineal, x_test_lineal, y_train_lineal, y_test_lineal = train_test_split(x_for_lineal, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit (x_train_lineal, y_train_lineal)
        y_pred = model.predict(x_test_lineal)
        mae = mean_absolute_error(y_test_lineal, y_pred)
        mse = mean_squared_error(y_test_lineal, y_pred)
        r2 = r2_score(y_test_lineal, y_pred)
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        print(f"R²: {r2}")
    
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



def mlflow_run(accuracy, report, x_train, model, model_name, artifact_path="wine_quality_model"):
    print("Logging metrics and model...")
    with mlflow.start_run():
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report['weighted avg']['precision'])
        mlflow.log_metric("recall", report['weighted avg']['recall'])
        mlflow.log_metric("f1", report['weighted avg']['f1-score'])
        mlflow.set_tag("Training Info", model_name)
        signature = infer_signature(x_train, model.predict(x_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=x_train,
            registered_model_name=model_name,
        )
    print(f"Model logged sucessfully")
 

if __name__ == "__main__":
    main()