import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("modelling-Andrew")

# Aktifkan autolog
mlflow.sklearn.autolog()

# Load dataset preprocessed
train_df = pd.read_csv('penguins_train_preprocessing.csv')
test_df = pd.read_csv('penguins_test_preprocessing.csv')

# Pisahkan fitur dan target
X_train = train_df.drop(columns='species')
y_train = train_df['species']
X_test = test_df.drop(columns='species')
y_test = test_df['species']

# Start MLflow experiment run
with mlflow.start_run(run_name="RandomForest"):
    # Inisialisasi dan latih model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))