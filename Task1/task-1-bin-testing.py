import mlflow
from mlflow.models import infer_signature

import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
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

    MIN_BINS = 3
    MAX_BINS = 20
    SLEEP_TIME = 5


    for bins in range(MIN_BINS, MAX_BINS+1):
        print(f"Running with {bins} bins...")
        x_train, x_test, y_train, y_test = discretize_data(bins, x, y)

        # CLASSIFICATION
        ## RANDOM FOREST
        print("Running Random Forest Classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, model, f"Random Forest Classifier Bins {bins}", f"random-forest-classifier-bins-{bins}")
        time.sleep(SLEEP_TIME)


        ## GRADIENT BOOSTING
        print("Running Gradient Boosting Classifier...")
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, model, f"Gradient Boosting Classifier Bins {bins}", f"gradient-boosting-classifier-bins-{bins}")
        time.sleep(SLEEP_TIME)


        ## SUPPORT VECTOR MACHINE
        print("Running Support Vector Machine...")
        model = SVC(kernel='linear', random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, model, f"Support Vector Machine Bins {bins}", f"support-vector-machine-bins-{bins}")
        time.sleep(SLEEP_TIME)


        ## K-NEAREST NEIGHBORS
        print("Running K-Nearest Neighbors...")
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, model, f"K-Nearest Neighbors Bins {bins}", f"k-nearest-neighbors-bins-{bins}")
        time.sleep(SLEEP_TIME)


        ## NAIVE BAYES - GAUSSIAN
        print("Running Naive Bayes - Gaussian...")
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, model, f"Naive Bayes - Gaussian Bins {bins}", f"naive-bayes-gaussian-bins-{bins}")
        time.sleep(SLEEP_TIME)


        ## NAIVE BAYES - MULTINOMIAL
        print("Running Naive Bayes - Multinomial...")
        model = MultinomialNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, model, f"Naive Bayes - Multinomial Bins {bins}", f"naive-bayes-multinomial-bins-{bins}")
        time.sleep(SLEEP_TIME)


        ## NAIVE BAYES - COMPLEMENT
        print("Running Naive Bayes - Complement...")
        model = ComplementNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow_run(accuracy, report, x_train, model, f"Naive Bayes - Complement Bins {bins}", f"naive-bayes-complement-bins-{bins}")
        time.sleep(SLEEP_TIME)


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