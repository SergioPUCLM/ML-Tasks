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

    # Discretize the data
    print("Discretizing data...")
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    X_disc = discretizer.fit_transform(x)
    y_disc = discretizer.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(X_disc, y_disc, test_size=0.2, random_state=42)


    # CLASSIFICATION
    ## RANDOM FOREST
    print("Running Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow_run(accuracy, report, x_train, model, "Random Forest Classifier", "random-forest-classifier")
    time.sleep(1)


    ## GRADIENT BOOSTING
    print("Running Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow_run(accuracy, report, x_train, model, "Gradient Boosting Classifier", "gradient-boosting-classifier")
    time.sleep(1)


    ## SUPPORT VECTOR MACHINE
    print("Running Support Vector Machine...")
    model = SVC(kernel='linear', random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow_run(accuracy, report, x_train, model, "Support Vector Machine", "support-vector-machine")
    time.sleep(1)


    ## K-NEAREST NEIGHBORS
    print("Running K-Nearest Neighbors...")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow_run(accuracy, report, x_train, model, "K-Nearest Neighbors", "k-nearest-neighbors")
    time.sleep(1)


    ## NAIVE BAYES - GAUSSIAN
    print("Running Naive Bayes - Gaussian...")
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow_run(accuracy, report, x_train, model, "Naive Bayes - Gaussian", "naive-bayes-gaussian")
    time.sleep(1)


    ## NAIVE BAYES - MULTINOMIAL
    print("Running Naive Bayes - Multinomial...")
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow_run(accuracy, report, x_train, model, "Naive Bayes - Multinomial", "naive-bayes-multinomial")
    time.sleep(1)


    ## NAIVE BAYES - COMPLEMENT
    print("Running Naive Bayes - Complement...")
    model = ComplementNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow_run(accuracy, report, x_train, model, "Naive Bayes - Complement", "naive-bayes-complement")
    time.sleep(1)


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