import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import mlflow
import mlflow.sklearn

mlflow.set_experiment("water_exp2")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
data = pd.read_csv('C:/Users/user/exp-mlflow/data/water_potability.csv')

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].dtype == 'float64':
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df

# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

x_train = train_processed_data.iloc[:, 0:-1].values
y_train = train_processed_data.iloc[:, -1].values

n_estimator = 500

with mlflow.start_run():

    clf = GradientBoostingClassifier(n_estimators=n_estimator)
    clf.fit(x_train, y_train)

    #save
    pickle.dump(clf, open('model.pkl', 'wb'))

    x_test = test_processed_data.iloc[:, 0:-1].values
    y_test = test_processed_data.iloc[:, -1].values

    from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix, recall_score, f1_score

    model = pickle.load(open('model.pkl', 'rb'))

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("F1 Score", f1_score)

    mlflow.log_param("n_estimators", n_estimator)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predicred")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(clf, "GradientBoostingClassifier")

    mlflow.log_artifact(__file__)

    mlflow.set_tag("Author", "Faruq")

    mlflow.set_tag("model", "GB")

    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1_score)