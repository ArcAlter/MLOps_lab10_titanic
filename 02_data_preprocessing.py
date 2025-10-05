import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def preprocess_data(test_size=0.25, random_state=42):
    mlflow.set_experiment("Titanic - Data Preprocessing")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        # Load dataset
        df = pd.read_csv("data/titanic.csv")

        # Handle missing values
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

        # Encode categorical features
        label_cols = ['Sex', 'Embarked']
        df[label_cols] = df[label_cols].apply(LabelEncoder().fit_transform)

        # Select features
        X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        y = df['Survived']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Save processed data
        processed_data_dir = "processed_data"
        os.makedirs(processed_data_dir, exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(processed_data_dir, "train.csv"), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(processed_data_dir, "test.csv"), index=False)

        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("train_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))
        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")

        print(f"Preprocessing Run ID: {run_id}")

        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={run_id}", file=f)

if __name__ == "__main__":
    preprocess_data()
