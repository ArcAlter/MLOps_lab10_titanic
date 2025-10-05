import pandas as pd
import mlflow

def validate_data():
    mlflow.set_experiment("Titanic - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load data
        df = pd.read_csv("data/titanic.csv")
        print("Data loaded successfully.")

        # 2. Basic validation
        num_rows, num_cols = df.shape
        missing_values = df.isnull().sum().sum()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Missing values: {missing_values}")

        # Log
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)

        validation_status = "Success" if missing_values < 100 else "Failed"
        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

if __name__ == "__main__":
    validate_data()
