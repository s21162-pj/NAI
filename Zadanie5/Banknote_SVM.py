import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
"""
* klasyfikacja autentyczności banknotów SVM *

Dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

Autorzy:
- Bartosz Krystowski s19545
- Robert Brzoskowski s21162

Przygotowanie środowiska:
Instalacja bibliotek: pandas, seaborn, matplotlib, sklearn
"""

"""Kolumny:
1. Variance - wariancja obrazu poddanego transformacji falkowej
2. Skewness - Skośność obrazu poddanego transformacji falkowej
3. Curtosis - kurtoza obrazu poddanego transformacji falkowej
4. Entropy - entropia obrazu
Class - klasa
"""

columns = ["Variance", "Skewness", "Curtosis", "Entropy", "Class"]
"""Import zestawu danych"""
banknote = pd.read_csv('data/data_banknote_authentication.txt', names=columns)

"""Wypisanie 5 pierwszych wierszy zestawu danych"""
print(banknote.head(5))

"""Utworzenie domyślnych wykresów par oraz pokazanie go na ekranie"""
sns.pairplot(banknote, hue='Class')

plt.show()
"""Usunięcie class z wykresu"""
x, y = banknote.drop('Class', axis=1), banknote['Class']

"""Określenie danych treningowych i testowych 
oraz tego że rozmiar testu wynosi 20% całości danych"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

"""Wypisanie ile dokładnie jest danych treningowych i testowych"""
print("x train shape ", x_train.shape)
print("x test shape ", x_test.shape)

"""Określenie klasyfikatora svc"""
linear_svc_classifier = SVC(kernel="linear")
linear_svc_classifier.fit(x_train, y_train)

"""Przewidywany wynik"""
linear_svc_classifier_prediction = linear_svc_classifier.predict(x_test)

"""Wypisanie oceny przebiegu algorytmu"""
print(confusion_matrix(y_test, linear_svc_classifier_prediction))
print(classification_report(y_test, linear_svc_classifier_prediction))
"""Wypisanie w procentach dokładności wyniku"""
print("accuracy of linear svm", accuracy_score(y_test, linear_svc_classifier_prediction) * 100, "%")
