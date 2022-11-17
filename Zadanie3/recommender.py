from os.path import exists

import matplotlib.pyplot
import pandas as pd
import tmdbsimple as tmdb
from difflib import SequenceMatcher

import seaborn as sns


"""Metoda zbierająca rekomendacje dla użytkowników w bazie"""
def recommend_movie(picked_username):
    """Metoda szukająca filmy użytkowników w bazie"""
    def search_in_api(ratings):
        """Token wymagany do połączenia z baza danych"""
        tmdb.API_KEY = "5d7af906802f4ffd3fbdb7c1d9b25b68"


        """Tablice do przechowywania wyszukanych filmów i seriali"""
        movies_arr = []
        tv_series_array = []

        """Call method do szukania w bazie danych"""
        movie_tmdb = tmdb.Search()

        """Tymczasowa kopia użytkowników, filmów i ocen"""
        movies_temp = ratings

        """Pętla służąca do wyszukiwania tytułu każdego filmu podanego przez użytkownika w bazie danych filmów,
            a następnie przekazująca informacje o nim do tablicy tymczasowej"""
        for r in ratings['title'].unique():
            response = movie_tmdb.movie(query=r)
            test_matcher_last = 0
            for c in movie_tmdb.results:
                test_matcher = SequenceMatcher(a=r.lower(), b=c['title'].lower()).ratio()
                if test_matcher > 0.8:
                    if test_matcher > test_matcher_last:
                        movie_result = c
                        test_matcher_last = test_matcher
            if test_matcher_last > 0.9:
                print(movie_result['title'] + " == " + r)
                movies_temp = movies_temp[movies_temp.title != r]
                movies_arr.append(movie_result)

        """Pętla służąca do wyszukiwania tytułu każdego serialu podanego przez użytkownika w bazie danych seriali,
                    a następnie przekazująca informacje o nim do tablicy tymczasowej"""
        for r in movies_temp['title'].unique():
            response = movie_tmdb.tv(query=r)
            test_matcher_last = 0
            for c in movie_tmdb.results:
                test_matcher = SequenceMatcher(a=r.lower(), b=c['name'].lower()).ratio()
                if test_matcher > 0.8:
                    if test_matcher > test_matcher_last:
                        movie_result = c
                        test_matcher_last = test_matcher
            if test_matcher_last > 0.9:
                print(movie_result['name'] + " == " + r)
                tv_series_array.append(movie_result)

        """Przeformatowanie tabeli seriali, tak aby pasowała do rablicy filmów"""
        test_dataframe = pd.DataFrame(tv_series_array, columns=tv_series_array[0].keys())
        test_dataframe_tv = test_dataframe.rename(
            columns={'first_air_date': 'release_date', 'original_name': 'original_title', 'name': 'title'},
            errors="raise")
        test_dataframe_tv = test_dataframe_tv.drop(columns=['origin_country'])
        test_dataframe = pd.DataFrame(movies_arr, columns=movies_arr[0].keys())

        frames = [test_dataframe, test_dataframe_tv]

        movies = pd.concat(frames)

        return movies

    """Metoda do tworzenia bazy danych filmów w pliku csv, jeśli do tej pory jeszcze jej nie było"""
    def create_csv(movies):
        if not exists('data/movies_data.csv'):
            with open('data/movies_data.csv', 'w', encoding='UTF8', newline='') as f:
                pass

        movies.to_csv('data/movies_data.csv', index=False)


    """
    Wczytywanie danych o użytkownikach, filmach i ocenach
    jeśli baza filmów nie instaje, wyszukuje nazwy filmów
    podane przez użytkownika i przechowuje je w pliku csv
    """
    ratings = pd.read_csv('data/data.csv')
    if not exists('data/movies_data.csv'):
        movies = search_in_api(ratings)
        create_csv(movies)

    """Odczyt bazy danych filmów z pliku csv """
    movies = pd.read_csv('data/movies_data.csv')

    print(ratings)

    """
    Pętla do wyszukiwania nazw filmów wybranych przez użytkownika w bazie filmów,
    wybiera te o największym prawdopodbieństwie (90% w górę)
    """
    for p in movies['title']:
        for t in range(len(ratings['title'])):
            test_matcher = SequenceMatcher(a=p.lower().replace(" ", ""),
                                           b=ratings['title'][t].lower().replace(" ", "")).ratio()
            if test_matcher > 0.9:
                ratings['title'][t] = p

    print(ratings)

    """Połączenie zbiorów ocen i filmów"""
    df = pd.merge(ratings, movies, on='title', how='inner')

    """Baza zawiera X unikalnych użytkowników"""
    print('The ratings dataset has', df['username'].nunique(), 'unique users')

    """Baza zawiera X unikalnych filmów"""
    print('The ratings dataset has', df['title'].nunique(), 'unique movies')

    """Baza zawiera X unikalnych ocen"""
    print('The ratings dataset has', df['rating'].nunique(), 'unique ratings')

    """Lista unikalnych ocen"""
    print('The unique ratings are', sorted(df['rating'].unique()))

    """Agregacja wedle filmu"""
    agg_ratings = df.groupby('title').agg(mean_rating=('rating', 'mean'),
                                          number_of_ratings=('rating', 'count')).reset_index()

    """Zachowaj filmy z oceną powyżej 1"""
    agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings'] > 0]
    print(agg_ratings_GT100['title'])

    """Sprawdza popularne filmy"""
    agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=False).head()

    """Wizualizacja"""
    sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)

    """Pokazuje zwizualizowany wykres"""
    """matplotlib.pyplot.show()"""

    """Połączenie danych"""
    df_GT100 = pd.merge(df, agg_ratings_GT100[['title']], on='title', how='inner')
    df_GT100.info()

    """Baza zawiera X unikalnych użytkowników"""
    print('The ratings dataset has', df_GT100['username'].nunique(), 'unique users')

    """Baza zawiera X unikalnych filmów"""
    print('The ratings dataset has', df_GT100['title'].nunique(), 'unique movies')

    """Baza zawiera X unikalnych filmów"""
    print('The ratings dataset has', df_GT100['rating'].nunique(), 'unique ratings')

    """Lista unikalnych ocen"""
    print('The unique ratings are', sorted(df_GT100['rating'].unique()))

    """ustawienie bazy aby nie ograniczać maksymalnej ilości kolumn"""
    pd.set_option('display.max_columns', None)
    """Tworzenie macierzy elementów użytkownika"""
    matrix = df_GT100.pivot_table(index='username', columns='title', values='rating')
    print(matrix.head())

    """Normalizacja macierzy elementów użytkownika"""
    matrix_norm = matrix.subtract(matrix.mean(axis=1), axis='rows')
    print(matrix_norm.head())

    """Macierz podobieństw użytkowników z wykorzystaniem korelacji Pearsona"""
    user_similarity = matrix_norm.T.corr()
    user_similarity.head()

    """Usuń wybrany identyfikator użytkownika z listy kandydatów"""
    user_similarity.drop(index=picked_username, inplace=True)

    """Wypisanie danych"""
    print(user_similarity)

    """Numer podobnych użytkowników"""
    n = 5

    """Próg podobieństwa użytkownika"""
    user_similarity_threshold = 0.3

    """Wyberz "n" pierwszych użytkowników"""
    similar_users = user_similarity[user_similarity[picked_username] > user_similarity_threshold][
                        picked_username].sort_values(
        ascending=False)[:n]

    """Wpisz n pierwszych użutkowników"""
    print(f'The similar users for user {picked_username} are', similar_users)

    """Filmy, które oglądał docelowy użytkownik"""
    picked_userid_watched = matrix_norm[matrix_norm.index == picked_username].dropna(axis=1, how='all')
    print(picked_userid_watched)

    """Filmy, które oglądali podobni użytkownicy. Usuń filmy, których żaden z podobnych użytkowników nie oglądał"""
    similar_user_movies = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')
    print(similar_user_movies)

    """Usuń oglądany film z listy filmów"""
    similar_user_movies.drop(picked_userid_watched.columns, axis=1, inplace=True, errors='ignore')

    """Wypisanie danych"""
    print(similar_user_movies)

    """Słownik do przechowywania wyników"""
    item_score = {}

    """Pętla elementów"""
    for i in similar_user_movies.columns:
        """Bierze oceny dla filmu I"""
        movie_rating = similar_user_movies[i]
        """Utwórz zmienną do przechowywania wyniku"""
        total = 0
        """Utwórz zmienną do przechowywania liczby wyników"""
        count = 0
        """Pętla podobnych użytkowników"""
        for u in similar_users.index:
            """Jeśli film ma ocenę"""
            if pd.isna(movie_rating[u]) is False:
                """Wynik to suma wyniku podobieństwa użytkowników pomnożona przez ocenę filmu"""
                score = similar_users[u] * movie_rating[u]
                """Dodaj wynik do łącznej punktacji filmu do tej pory"""
                total += score
                """Dodaj 1 do licznika"""
                count += 1
        """Uzyskaj średni wynik dla elementu"""
        item_score[i] = total / count

    """Zamiana słownika na ramkę danych pandas"""
    item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])

    """Sortuj dobre propozycje filmów"""
    ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)

    """Sortuj złe propozycje filmów"""
    deranked_item_score = item_score.sort_values(by='movie_score', ascending=True)

    """Wybierz górne m filmów"""
    m = 5

    print(ranked_item_score.head(m))
    print(deranked_item_score.head(m))

    return [ranked_item_score.head(m), deranked_item_score.head(m)]
