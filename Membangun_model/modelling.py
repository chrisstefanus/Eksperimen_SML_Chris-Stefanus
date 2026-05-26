import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# load dataset
df = pd.read_csv("../preprocessing/titanic_preprocessed.csv")

# fitur & target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# mlflow experiment
mlflow.set_experiment("Titanic Survival Prediction")

with mlflow.start_run():

    # parameter model
    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # train
    model.fit(X_train, y_train)

    # prediksi
    y_pred = model.predict(X_test)

    # metric
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # log parameter
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # log metric
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # log model
    mlflow.sklearn.log_model(
        model,
        "random_forest_model"
    )

print("Training selesai")
print("Accuracy:", accuracy)
