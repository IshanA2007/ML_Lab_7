import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report



def classify_instances(data_location):
    df = pd.read_csv(data_location)


    df['Glucose'] = pd.cut(df['Glucose'], bins=5, labels=False)
    df['BloodPressure'] = pd.cut(df['BloodPressure'], bins=10, labels=False)
    df['SkinThickness'] = pd.cut(df['SkinThickness'], bins=5, labels=False)
    df['BMI'] = pd.cut(df['BMI'], bins=5, labels=False)
    df['Age'] = pd.cut(df['Age'], bins=5, labels=False)
    df['Insulin'] = pd.cut(df['Insulin'], bins=5, labels=False)
    df['DiabetesPedigreeFunction'] = pd.cut(df['DiabetesPedigreeFunction'], bins=5, labels=False)


    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    clf = DecisionTreeClassifier(max_depth=1, random_state=1)

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Predict the outcomes on the test data
    y_pred = clf.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    classification_repo = classification_report(y_test, y_pred)

    print(classification_repo)

    print(f"Total Test Instances: {X_test.shape[0]}")
    print(f"Accuracy: {clf.score(X_test, y_test) * 100:.2f}%")
