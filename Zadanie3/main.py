import csv

import pandas as pd
from recommender import recommend_movie

"""
* Dopasowywanie filmów do użytkownika, na zasadzie ocen filmów innych użytkowników *

Program polega na wprowadzeniu w pierwszym inpucie
imienia i nazwiska użytkownika dla którego chcemy wyszukać polecany lub odradzany film.
W kolejnych zaś inputach, wpisujemy nazwę filmu z polecanych lub odradzanych dla którego chcemy pokazać listę filmów podobnych.

Autorzy:
- Bartosz Krystowski s19545
- Robert Brzoskowski s21162

Przygotowanie środowiska:
Instalacja bibliotek: pandas, tmdbsimple, difflib, seaborn

"""

"""
Odczyt danych z pliku excel'a .xlsx
Fill "None" jeśli komurka jest pusta
"""
dataframe1 = pd.read_excel('./data/datasheet.xlsx')
dataframe1.fillna("None", inplace=True)

"""Zamiana na pandas"""
df = pd.DataFrame(dataframe1)

"""Utworzenie tablic na dane"""
dataset1 = []
movie = []
movies = []

"""Wypisanie wartości bazy"""
print(df.values[0][3])

"""Przypisywanie użytkowników do filmów"""
for i in range(len(df.values)):
    movies = []
    username = df.values[i][0]
    for j in range(len(df.values[i])):
        if j != 0 and df.values[i][j] != "None" and isinstance(df.values[i][j], str):
            movie = [df.values[i][j], df.values[i][j + 1]]
            movies.append(movie)
    dataset1.append([username, movies])
"""Wypisanie przypisanych wyżej danych"""
for t in dataset1:
    print(t)


"""Metoda sprawdzająca czy dany użytkownik istnieje w bazie"""
def choose_name():
    result = []
    while len(result) != 1:
        name = input("Input name you want to check\n")
        for l in dataset1:
            if name in l[0]:
                result.append(l)
        if len(result) > 1:
            print("Choose full name")
            for i in result:
                print(i[0])
            result = []
    print("Got it")
    print(result)
    return result


"""Utworzenie pliku .csv z pliku excela wraz z wypisaniem na ekranie"""
def create_csv():
    header = ['username', 'title', 'rating']

    temp_dataset = []
    temp_ratings = []
    for m in range(len(dataset1)):
        for p in range(len(dataset1[m][1])):
            if dataset1[m][1][p][1] != "None":
                temp_ratings.append(dataset1[m][0])
                temp_ratings.append(dataset1[m][1][p][0])
                temp_ratings.append(float((dataset1[m][1][p][1])) / 2)
                temp_dataset.append(temp_ratings)
                temp_ratings = []

    with open('data/data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(temp_dataset)


"""Metoda służąca zebraniu informacji o wybranym filmie"""
def choose_movie(movies_array):
    result = []
    while len(result) != 1:
        name = input("Input name you want to check\n")
        for l in range(len(movies_array)):
            for c in movies_array[l]['movie']:
                print(c)
                if name in c:
                    result.append(c)
        if len(result) > 1:
            print("Choose full name")
            for i in result:
                print(i)
            result = []
    print("Got it")
    print(result[0])
    movies = pd.read_csv('data/movies_data.csv')
    for p in range(len(movies['title'])):
        if result[0] == movies['title'][p]:
            result = movies.iloc[p]
            break
    return result


if __name__ == "__main__":
    create_csv()
    name = choose_name()[0]
    result = recommend_movie(name[0])
    print(choose_movie(result))
