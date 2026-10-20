# Konfiguracja eksperymentow

# instancje testowe
INSTANCES = [
    "data/instances/berlin52_n51_uncorr_09.ttp",
    "data/instances/berlin52_n51_bounded-strongly-corr_10.ttp",
    "data/instances/berlin52_n153_bounded-strongly-corr_07.ttp",
    "data/instances/berlin52_n255_bounded-strongly-corr_07.ttp",
    "data/instances/berlin52_n510_uncorr-similar-weights_09.ttp",
]

# powtarzalnosc
SEED = 42

# parametry EA (wartosci startowe z tresci zadania)
POP_SIZE = 100       # rozmiar populacji
GENERATIONS = 100    # liczba pokolen
PX = 0.7             # prawdopodobienstwo krzyzowania
PM = 0.1             # prawdopodobienstwo mutacji (inwersja, na osobnika)
TOURNAMENT = 5       # rozmiar turnieju
N_RUNS = 10          # liczba uruchomien metod niedeterministycznych

# algorytm losowy: tyle ocen, ile wykonuje EA (pokolenia x osobniki)
N_RANDOM = POP_SIZE * GENERATIONS