from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd

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


    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    classification_repo = classification_report(y_test, y_pred)

    print(classification_repo)

    # Output the results
    print(f"Total Test Instances: {X_test.shape[0]}")
    print(f"Accuracy: {gnb.score(X_test, y_test) * 100:.2f}%")
    #print("Number of mislabeled points out of a total %d points : %d"
        #% (X_test.shape[0], (y_test != y_pred).sum()))

classify_instances("../data/diabetes.csv")