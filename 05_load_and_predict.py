import mlflow
import pandas as pd

def load_and_predict():
    MODEL_NAME = "Titanic-classifier-prod"
    MODEL_STAGE = "Staging"

    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

    # Sample data (ใช้ข้อมูลจริง 1 แถว)
    sample = pd.DataFrame([{
        "Pclass": 3, "Sex": 1, "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": 0
    }])

    prediction = model.predict(sample)
    print(f"Predicted survival: {prediction[0]}")

if __name__ == "__main__":
    load_and_predict()
