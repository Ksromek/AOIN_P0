from pathlib import Path

PROJECT_DIR = Path(__file__).parent
INSTANCES_DIR = PROJECT_DIR / "data" / "instances"
RESULTS_DIR = PROJECT_DIR / "results"

# instancje testowe
INSTANCES = [
    INSTANCES_DIR / "berlin52_n153_uncorr_06.ttp",
    INSTANCES_DIR / "berlin52_n510_uncorr_07.ttp",
]

# powtarzalnosc
SEED = 42
N_RUNS = 10          # liczba uruchomien metod niedeterministycznych

# parametry EA (wartosci startowe z tresci zadania)
POP_SIZE = 100       # rozmiar populacji
GENERATIONS = 100    # liczba pokolen
PX = 0.6             # prawdopodobienstwo krzyzowania
PM = 0.1             # prawdopodobienstwo mutacji: inwersja - na osobnika, swap - na gen (start od 0.01)
TOURNAMENT = 5       # rozmiar turnieju

# operatory EA
INIT_METHOD = "random"           # "random" / "greedy"
SELECTION_METHOD = "tournament"  # "tournament" / "roulette"
CROSSOVER_METHOD = "ox"          # "ox" / "pmx" / "cx"
MUTATION_METHOD = "inversion"    # "inversion" / "swap"

# algorytm losowy: tyle ocen, ile wykonuje EA (pokolenia x osobniki)
N_RANDOM = POP_SIZE * GENERATIONS

# parametry SA (stop tylko po temperaturze: SA dziala, dopoki T > SA_MIN_TEMP)
SA_START_TEMP = 2000.0             # temperatura poczatkowa
SA_MIN_TEMP = 0.01                # temperatura koncowa - warunek stopu (musi byc > 0)
SA_COOLING_SCHEMA = "geometric"    # "geometric" (T = alpha * T) / "linear" (T = T - beta)
SA_ALPHA = 0.999                   # dla "geometric", z przedzialu (0, 1)
SA_BETA = 0.1                      # dla "linear", > 0
SA_LOG_EVERY = 100                 # co ile iteracji zapisywac historie SA do CSV