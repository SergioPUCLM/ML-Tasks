import mlflow
import json


experiment_name = "WINE QUALITY"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    all_run_data = []

    

    for _, run in runs.iterrows():
        parameters = {}
        for param_key, param_value in run.items():
            if param_key.startswith("params."):  # Only capture parameter keys
                param_name = param_key.replace("params.", "")  # Remove the "params." prefix
                if param_value is not None:  # Skip None values
                    parameters[f"_{param_name}"] = param_value  # Store in the desired format

        training_info_tag = run["tags.Training Info"]
        feature_columns = run.get("metrics.feature_columns", None)  # Safely retrieve feature columns
        run_data = {
            training_info_tag: {
                "accuracy": run["metrics.accuracy"],
                "f1_score": run["metrics.f1"],
                "precision": run["metrics.precision"],
                "recall": run["metrics.recall"],
                "parameters": parameters
            }
        }
        all_run_data.append(run_data)

    json_data = json.dumps(all_run_data, indent=4)

    with open("task1-classification-models-performance.json", "w") as f:
        f.write(json_data)

    with mlflow.start_run():
        mlflow.log_artifact("task1-classification-models-performance.json")

    print("Metrics logged as task1-classification-models-performance.json")

else:
    print(f"Experiment '{experiment_name}' not found.")