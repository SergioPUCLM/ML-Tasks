import mlflow
from mlflow.models import infer_signature

import pandas as pd
import numpy as np
import seaborn as sns
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

# Set up MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("WINE QUALITY")

# Fetch dataset
wine_quality = fetch_ucirepo(id=186)

# Data (as pandas dataframes)
x = wine_quality.data.features
y = wine_quality.data.targets

# Discretize the data
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
X_disc = discretizer.fit_transform(x)
y_disc = discretizer.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(X_disc, y_disc, test_size=0.2, random_state=42)

# RANDOM FOREST
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report['weighted avg']['precision'])
    mlflow.log_metric("recall", report['weighted avg']['recall'])
    mlflow.log_metric("f1", report['weighted avg']['f1-score'])
    mlflow.set_tag("Training Info", "Random Forest Classifier")
    signature = infer_signature(x_train, model.predict(x_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="wine_quality_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="random-forest-classifier",
    )

# GRADIENT BOOSTING
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report['weighted avg']['precision'])
    mlflow.log_metric("recall", report['weighted avg']['recall'])
    mlflow.log_metric("f1", report['weighted avg']['f1-score'])
    mlflow.set_tag("Training Info", "Gradient Boosting Classifier")
    signature = infer_signature(x_train, model.predict(x_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="wine_quality_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="gradient-boosting-classifier",
    )

# SUPPORT VECTOR MACHINE
model = SVC(kernel='linear', random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report['weighted avg']['precision'])
    mlflow.log_metric("recall", report['weighted avg']['recall'])
    mlflow.log_metric("f1", report['weighted avg']['f1-score'])
    mlflow.set_tag("Training Info", "Support Vector Machine")
    signature = infer_signature(x_train, model.predict(x_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="wine_quality_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="support-vector-machine",
    )

# K-NEAREST NEIGHBORS
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report['weighted avg']['precision'])
    mlflow.log_metric("recall", report['weighted avg']['recall'])
    mlflow.log_metric("f1", report['weighted avg']['f1-score'])
    mlflow.set_tag("Training Info", "K-Nearest Neighbors")
    signature = infer_signature(x_train, model.predict(x_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="wine_quality_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="k-nearest-neighbors",
    )

# NAIVE BAYES - GAUSSIAN
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report['weighted avg']['precision'])
    mlflow.log_metric("recall", report['weighted avg']['recall'])
    mlflow.log_metric("f1", report['weighted avg']['f1-score'])
    mlflow.set_tag("Training Info", "Naive Bayes - Gaussian")
    signature = infer_signature(x_train, model.predict(x_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="wine_quality_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="naive-bayes-gaussian",
    )

# NAIVE BAYES - MULTINOMIAL
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report['weighted avg']['precision'])
    mlflow.log_metric("recall", report['weighted avg']['recall'])
    mlflow.log_metric("f1", report['weighted avg']['f1-score'])
    mlflow.set_tag("Training Info", "Naive Bayes - Multinomial")
    signature = infer_signature(x_train, model.predict(x_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="wine_quality_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="naive-bayes-multinomial",
    )

# NAIVE BAYES - COMPLEMENT
model = ComplementNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report['weighted avg']['precision'])
    mlflow.log_metric("recall", report['weighted avg']['recall'])
    mlflow.log_metric("f1", report['weighted avg']['f1-score'])
    mlflow.set_tag("Training Info", "Naive Bayes - Complement")
    signature = infer_signature(x_train, model.predict(x_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="wine_quality_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="naive-bayes-complement",
    )
