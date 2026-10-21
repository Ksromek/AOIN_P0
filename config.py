from pathlib import Path

PROJECT_DIR = Path(__file__).parent
INSTANCES_DIR = PROJECT_DIR / "data" / "instances"
RESULTS_DIR = PROJECT_DIR / "results"

# instancje testowe
INSTANCES = [
    INSTANCES_DIR / "berlin52_n51_uncorr_09.ttp",
    INSTANCES_DIR / "berlin52_n51_bounded-strongly-corr_10.ttp",
    INSTANCES_DIR / "berlin52_n153_bounded-strongly-corr_07.ttp",
    INSTANCES_DIR / "berlin52_n255_bounded-strongly-corr_07.ttp",
    INSTANCES_DIR / "berlin52_n510_uncorr-similar-weights_09.ttp",
]

# powtarzalnosc
SEED = 42
N_RUNS = 10          # liczba uruchomien metod niedeterministycznych

# parametry EA (wartosci startowe z tresci zadania)
POP_SIZE = 100       # rozmiar populacji
GENERATIONS = 100    # liczba pokolen
PX = 0.7             # prawdopodobienstwo krzyzowania
PM = 0.1             # prawdopodobienstwo mutacji: inwersja - na osobnika, swap - na gen (start od 0.01)
TOURNAMENT = 5       # rozmiar turnieju

# operatory EA
INIT_METHOD = "random"           # "random" / "greedy"
SELECTION_METHOD = "tournament"  # "tournament" / "roulette"
CROSSOVER_METHOD = "ox"          # "ox" / "pmx" / "cx"
MUTATION_METHOD = "inversion"    # "inversion" / "swap"

# algorytm losowy: tyle ocen, ile wykonuje EA (pokolenia x osobniki)
N_RANDOM = POP_SIZE * GENERATIONS

# parametry SA
SA_START_TEMP = 1000.0             # temperatura poczatkowa
SA_ALPHA = 0.999                   # wspolczynnik chlodzenia (tylko dla "geometric")
SA_COOLING_SCHEMA = "geometric"    # "geometric" / "linear" / "logarithmic"
SA_MIN_TEMP = 0.0                  # stop, gdy temperatura spadnie ponizej (0 = wylaczone)
SA_MAX_ITERATIONS = N_RANDOM       # ten sam budzet ocen co EA
SA_LOG_EVERY = 100                 # co ile iteracji zapisywac historie SA do CSV