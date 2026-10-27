from pathlib import Path

PROJECT_DIR = Path(__file__).parent
INSTANCES_DIR = PROJECT_DIR / "data" / "instances"
RESULTS_DIR = PROJECT_DIR / "results"

INSTANCES = [
    # latwe
    INSTANCES_DIR / "berlin52_n153_uncorr_06.ttp",
    INSTANCES_DIR / "berlin52_n510_uncorr_07.ttp",
    # srednie
    INSTANCES_DIR / "ch150_n149_uncorr-similar-weights_09.ttp",
    INSTANCES_DIR / "ch150_n447_uncorr-similar-weights_06.ttp",
    # trudne
    INSTANCES_DIR / "a280_n279_bounded-strongly-corr_08.ttp",
    INSTANCES_DIR / "a280_n837_bounded-strongly-corr_04.ttp",
    # extra
    INSTANCES_DIR / "d493_n2460_uncorr-similar-weights_10.ttp",
]

# powtarzalnosc
SEED = 2000
N_RUNS = 10          # liczba uruchomien

POP_SIZE = 50        # rozmiar populacji
GENERATIONS = 200    # liczba pokolen
PX = 0.7             # prawdopodobienstwo krzyzowania
PM = 0.7             # prawdopodobienstwo mutacji: inwersja - na osobnika, swap - na gen
TOURNAMENT = 10      # rozmiar turnieju

# operatory EA
INIT_METHOD = "random"           # "random" / "greedy"
SELECTION_METHOD = "tournament"  # "tournament" / "roulette"
CROSSOVER_METHOD = "ox"          # "ox" / "pmx" / "cx"
MUTATION_METHOD = "inversion"    # "inversion" / "swap"

# drugi wariant EA
EA_GREEDY = {"INIT_METHOD": "greedy", "POP_SIZE": 100, "GENERATIONS": 100,
             "TOURNAMENT": 5, "PX": 0.7, "PM": 0.5}

# parametry SA
SA_START_TEMP = 1000.0             # temperatura poczatkowa
SA_MIN_TEMP = 0.05                 # temperatura koncowa - warunek stopu (musi byc > 0)
SA_COOLING_SCHEMA = "geometric"    # "geometric" (T = alpha * T) / "linear" (T = T - beta)
SA_ALPHA = 0.999                   # dla "geometric", z przedzialu (0, 1)
SA_BETA = 0.1                      # dla "linear", > 0
SA_LOG_EVERY = 100                 # co ile iteracji zapisywac historie SA do CSV