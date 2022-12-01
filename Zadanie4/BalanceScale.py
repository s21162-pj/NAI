import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings

"""
* Balans obiektu *

Dane dotyczą balansu przedmiotu zależnie od tego jak przeciążona jest strona lewa i prawa.
Program przewiduje i oblicza przeciążenie przy pomocy danych.

https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/

Autorzy:
- Bartosz Krystowski s19545
- Robert Brzoskowski s21162

Przygotowanie środowiska:
Instalacja bibliotek: pandas, sklearn, warnings
"""


warnings.filterwarnings('ignore')


"""Funkcja importująca zestaw danych"""
def importdata():
    balance_data = pd.read_csv(
        'data/balance-scale.data',
        sep=',', header=None)

    """Wyświetlanie długości zestawu danych oraz kształtu"""
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)

    """Wyświetlanie obserwacji zbiory danych"""
    print("Dataset: ", balance_data.head())
    return balance_data


"""Funkcja do podziału zestawu danych"""
def splitdataset(balance_data):
    """Oddzielenie zmiennej docelowej"""
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]

    """Podział zestawu danych na train i test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


"""funkcja do przeprowadzania treningu z giniIndex"""
def train_using_gini(X_train, X_test, y_train):
    """Tworzenie klasyfikatora"""
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    """Wykonanie treningu"""
    clf_gini.fit(X_train, y_train)
    return clf_gini


"""Funkcja do wykonywania treningu z entropią"""
def tarin_using_entropy(X_train, X_test, y_train):
    """Drzewo decyzyjne z entropy"""
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    """Wykonanie treningu"""
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


"""Funkcja do przewidywania"""
def prediction(X_test, clf_object):
    """Przewidywanie na teście z giniIndex"""
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


"""Funkcja do obliczania dokładności"""
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


"""Main code"""
def main():
    """Faza budowy, import danych"""
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

    """Wypisanie wyników"""
    print("Results Using Gini Index:")

    """Przewidywanie za pomocą Gini"""
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    """Przewidywanie przy użyciu entropy"""
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


if __name__ == "__main__":
    main()
