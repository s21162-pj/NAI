from easyAI import TwoPlayerGame
"""
* Game of NIM *

https://en.wikipedia.org/wiki/Nim

Nim to gra, w której dwóch graczy na zmianę bierze obiekty z kilku stosów.
Jedyną zasadą jest to, że każdy gracz musi wziąć minimum jeden lub więcej obiektór, 
o ile wszystkie pochodzą z tego samego stosu.

Autorzy:
- Bartosz Krystowski s19545
- Robert Brzoskowski s21162

Przygotowanie środowiska:
Instalacja EasyAI - "pip install easyai"
"""

class Nim(TwoPlayerGame):

    def __init__(self, players = None, max_remove = None, piles=(5,5,5,5)):
        """Inicjalizacja gracza, tego ile obiektów zostaje usuniętych oraz stosów
        Parameters:
            players: zmienna z TwoPlayerGAme z EasyAI, będąca listą dwóch graczy
            max_remove: usunięte obiekty
            piles: stosy z obiektami
        Returns:
            self.players = players: wybór gracza pomiędzy człowiekiem, a AI (linijka 74)
            self.max_remove = max_remove: Ile obietków zostaje usuniętych z danego stosu
            self.piles = list(piles): Parametr na ten moment ustawiony na piles=(5,5,5,5), czyli 4 stosy po 5 obiektów w każdym
            self.current_player = 1: aktualny gracz = 1, czyli człowiek - określa, który gracz rozpoczyna grę
        """
        self.players = players
        self.piles = list(piles)
        self.max_remove = max_remove
        self.current_player = 1

    def possible_moves(self):
        """Możliwe do wykonania ruchy, po przecinku - stos,ile obiektów z niego usuwamy"""
        return [
            "%d,%d" % (i + 1, j)
            for i in range(len(self.piles))
            for j in range(
                1,
                self.piles[i] + 1
                if self.max_remove is None
                else min(self.piles[i] + 1, self.max_remove),
            )
        ]

    def make_move(self, move):
        """Funkcja odpowiedzialna za wykonanie ruchu, separatorem jest przecinek w formacie
                - stos,ile obiektów z niego usuwamy (np. 1,3)
            a następnie usuwa wskazane obiekty z podanego stosu
            Parameters:
                move: lista możliwych ruchów
            Returns:
                self.piles: aktualizacja pola gry"""
        move = list(map(int, move.split(',')))
        self.piles[move[0] - 1] -= move[1]

    def show(self): print(" ".join(map(str, self.piles)))
    """Pokazanie zaktualizowanego pola gry"""

    def win(self): return max(self.piles) == 0
    """Wygrana po ostanim ruchu"""

    def is_over(self): return self.win()
    """Zakończenie gry gdy win = true"""

    def scoring(self): return 100 if self.win() else 0
    """Śledzenie wyniku gry"""

if __name__ == "__main__":

    from easyAI import AI_Player, Human_Player, Negamax

    ai = Negamax(3)
    """Ile ruchów do przodu obmyślać ma AI"""
    game = Nim([Human_Player(), AI_Player(ai)])
    """Instancja tworzy grę z dwoma graczami - człowiekiem oraz AI z algorytmem Negamax podanym w argumencie AI_Player """
    game.play()
    """Start gry
    W grze uczestniczymy poprzez wpisanie po przecinku najpierw z którego stosu
    chcemy zabrać obiekt następnie ile obiektów zabieramy
    Przykład: 2,3"""
    print("player %d wins" % game.current_player)
    """Wypisanie na ekranie, który gracz okazał się zwycięzcą batalii"""