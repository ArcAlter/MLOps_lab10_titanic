import sys, os, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts

def train_evaluate_register(preprocessing_run_id, C=1.0):
    ACCURACY_THRESHOLD = 0.78
    mlflow.set_experiment("Titanic - Model Training")

    with mlflow.start_run(run_name=f"logistic_regression_C_{C}"):
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        local_artifact_path = download_artifacts(run_id=preprocessing_run_id, artifact_path="processed_data")
        train_df = pd.read_csv(os.path.join(local_artifact_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(local_artifact_path, "test.csv"))

        X_train = train_df.drop('Survived', axis=1)
        y_train = train_df['Survived']
        X_test = test_df.drop('Survived', axis=1)
        y_test = test_df['Survived']

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=C, random_state=42, max_iter=10000))
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "Titanic_classifier_pipeline")

        if acc >= ACCURACY_THRESHOLD:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/Titanic_classifier_pipeline"
            registered_model = mlflow.register_model(model_uri, "Titanic-classifier-prod")
            print(f"Model registered as '{registered_model.name}' version {registered_model.version}")
        else:
            print(f"Model accuracy ({acc:.4f}) below threshold")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <run_id>")
        sys.exit(1)
    run_id = sys.argv[1]
    c_value = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    train_evaluate_register(run_id, C=c_value)
