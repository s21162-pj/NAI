import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_graphviz

"""
* Swedish Auto Insurance Dataset *

The Swedish Auto Insurance Dataset involves predicting the total payment for all claims in thousands of Swedish Kronor, given the total number of claims.

It is a regression problem. It is comprised of 63 observations with 1 input variable and one output variable. The variable names are as follows:

X = Number of claims.
Y = Total payment for all claims in thousands of Swedish Kronor.

link:
https://machinelearningmastery.com/standard-machine-learning-datasets/

Autorzy:
- Bartosz Krystowski s19545
- Robert Brzoskowski s21162

Przygotowanie środowiska:
Instalacja bibliotek: pandas, numpy, matplotlib, sklearn.tree

"""

"""Odczyt danych z pliku z danymi w formacie csv wraz z przekazaniem ich do biblioteki numpy"""
pf = pd.read_csv("data/data.csv")

df = pf.to_numpy()

"""Wypisanie tych danych"""
print(df)
"""Zmienne trzymające wartości tablic z danych kolumn"""
X = df[:, 0:1].astype(int)
y = df[:, 1].astype(float)

print(X)
print(y)
"""W związku z tym że ten akurat problem jest problemem regresywności
 to tworzymy potem obiekt z biblioteki DecisionTreeRegressor,
 następnie każemy mu wpasować te dane przez fit(X, y).
 Dzielone są one następnie na dane treningowe itd"""
regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X, y)
"""
Co przewidzieć ma program
w tym akurat przypadku przewiduje jaka będzie wartość w odniesieniu do tego co wpisze użytkownik
"""
input_pred = input("Enter number of claims for predicted payment: ")

y_pred = regressor.predict([[input_pred]])

print('Predicted payment for {0} claims is {1}'.format(input_pred, y_pred))


"""
Utworzenie zakresu wartości od minimalnej wartości X do maksymalnej wartości X
z różnicą 0.01 między dwoma kolejnymi wartościami
"""
X_grid = np.arange(min(X), max(X), 0.1)

"""
Przekształcenie danych do postaci tablicy a len(X_grid)*1,
aby utworzyć kolumnę z wartości X_grid
"""
X_grid = X_grid.reshape((len(X_grid), 1))

# scatter plot for original data
plt.scatter(X, y, color='red')

"""Wykres punktowy dla oryginalnych danych"""
plt.plot(X_grid, regressor.predict(X_grid), color='blue')

"""Tytuł wykresu"""
plt.title('Claims to Total payment (Decision Tree Regression)')

"""Nazwa osi X"""
plt.xlabel('Number of claims')

"""Nazwa osi Y"""
plt.ylabel('Total payment for claims')

"""Wyświetlenie wykresu"""
plt.show()

export_graphviz(regressor, out_file='tree.dot', feature_names=['Claims'])
