import mlflow
from mlflow.models import infer_signature

import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB

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

    MIN_BINS = 5
    MAX_BINS = 15
    SLEEP_TIME = 1


    for bins in range(MIN_BINS, MAX_BINS+1):
        print(f"Running with {bins} bins...")
        x_train, x_test, y_train, y_test = discretize_data(bins, x, y)

        # Grid hyperparameters
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 3, 4]
        }

        param_grid_svc = {
            'C': [0.1, 1, 10, 100],
            'tol': [1e-3, 1e-4],
        }

        param_grid_k = {
            'n_neighbors': [3, 5, 7, 9], 
            'weights': ['uniform', 'distance']
        }

        param_grid_nb_gaussian = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }

        param_grid_nb = {
            'alpha': [0.1, 0.5, 1.0]
        }


        # CLASSIFICATION
        ## RANDOM FOREST
        print("Running Random Forest Classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = grid_search.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, best_model, best_params, f"Random Forest Classifier Bins {bins}", f"random-forest-classifier-bins-{bins}")
        time.sleep(SLEEP_TIME)

        ## GRADIENT BOOSTING
        print("Running Gradient Boosting Classifier...")
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = grid_search.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, best_model, best_params, f"Gradient Boosting Classifier Bins {bins}", f"gradient-boosting-classifier-bins-{bins}")
        time.sleep(SLEEP_TIME)

        ## SUPPORT VECTOR MACHINE
        print("Running Support Vector Machine Classifier...")
        model = SVC(kernel='linear', random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_svc, cv=10, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = grid_search.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, best_model, best_params, f"Support Vector Machine Classifier Bins {bins}", f"support-vector-machine-classifier-bins-{bins}")
        time.sleep(SLEEP_TIME)

        ## K-NEAREST NEIGHBORS
        print("Running K-Nearest Neighbors Classifier...")
        model = KNeighborsClassifier(n_neighbors=3)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_k, cv=10, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = grid_search.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, best_model, best_params, f"K-Nearest Neighbors Classifier Bins {bins}", f"k-nearest-neighbors-classifier-bins-{bins}")
        time.sleep(SLEEP_TIME)

        ## GAUSSIAN NAIVE BAYES
        print("Running Gaussian Naive Bayes Classifier...")
        model = GaussianNB()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_nb_gaussian, cv=10, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = grid_search.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, best_model, best_params, f"Gaussian Naive Bayes Classifier Bins {bins}", f"gaussian-naive-bayes-classifier-bins-{bins}")
        time.sleep(SLEEP_TIME)

        ## MULTINOMIAL NAIVE BAYES
        print("Running Multinomial Naive Bayes Classifier...")
        model = MultinomialNB()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_nb, cv=10, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = grid_search.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, best_model, best_params, f"Multinomial Naive Bayes Classifier Bins {bins}", f"multinomial-naive-bayes-classifier-bins-{bins}")
        time.sleep(SLEEP_TIME)

        ## COMPLEMENT NAIVE BAYES
        print("Running Complement Naive Bayes Classifier...")
        model = ComplementNB()
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid_nb, cv=10, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = grid_search.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, best_model, best_params, f"Complement Naive Bayes Classifier Bins {bins}", f"complement-naive-bayes-classifier-bins-{bins}")
        time.sleep(SLEEP_TIME)


def mlflow_run(accuracy, report, x_train, model, best_params, model_name, artifact_path="wine_quality_model"):
    print("Logging metrics and model...")
    with mlflow.start_run():
        mlflow.log_params(best_params)
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


def discretize_data(bins, x, y):
    # Discretize the data
    print("Discretizing data...")
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    X_disc = discretizer.fit_transform(x)
    y_disc = discretizer.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(X_disc, y_disc, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    main()