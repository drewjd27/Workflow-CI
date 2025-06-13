import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():

    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("modelling-Andrew")
    mlflow.sklearn.autolog()

    train_df = pd.read_csv('penguins_train_preprocessing.csv')
    test_df = pd.read_csv('penguins_test_preprocessing.csv')

    X_train = train_df.drop(columns='species')
    y_train = train_df['species']
    X_test = test_df.drop(columns='species')
    y_test = test_df['species']

    with mlflow.start_run(run_name="RandomForest"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Log model secara eksplisit (untuk kebutuhan Docker image)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
