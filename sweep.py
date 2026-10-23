import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import pandas as pd

from config import INSTANCES_DIR, RESULTS_DIR
from experiments import (default_params, config_tag, sa_config_tag, ea_csv_params, sa_csv_params,
                         run_ea, run_sa, EA_COLUMNS, SA_COLUMNS)
from ttp import TTPInstance
from utils.log_to_csv import save_history_csv

SWEEP_INSTANCES = [INSTANCES_DIR / "berlin52_n153_uncorr_06.ttp",
                   INSTANCES_DIR / "berlin52_n510_uncorr_07.ttp"]

# punkt odniesienia badania: wartosci startowe z tresci zadania
SWEEP_BASE = {"SEED": 42, "N_RUNS": 10,
              "POP_SIZE": 100, "GENERATIONS": 100, "PX": 0.7, "PM": 0.1, "TOURNAMENT": 5,
              "INIT_METHOD": "random", "SELECTION_METHOD": "tournament",
              "CROSSOVER_METHOD": "ox", "MUTATION_METHOD": "inversion",
              "SA_START_TEMP": 1000.0, "SA_MIN_TEMP": 0.05, "SA_COOLING_SCHEMA": "geometric",
              "SA_ALPHA": 0.999, "SA_BETA": 0.1}

N_WORKERS = max(1, (os.cpu_count() or 2) - 4)  # kilka rdzeni dla systemu

SWEEP_DIR = RESULTS_DIR / "sweep"

# SA przy stalym budzecie: alfa / beta dobierane tak, zeby liczba iteracji wynosila ok. SA_BUDGET
SA_BUDGET = 9900


def alpha_for(t0, tmin):
    return round((tmin / t0) ** (1 / SA_BUDGET), 6)


def beta_for(t0, tmin):
    return round((t0 - tmin) / SA_BUDGET, 6)


# wartosci Px wspolne dla wszystkich siatek EA
PX_GRID = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

# walidacja najlepszych konfiguracji: inne seedy (1000..1029) i wiecej uruchomien
VALIDATION = {"SEED": 1000, "N_RUNS": 30}

EA_CANDIDATES = [
    {},
    {"TOURNAMENT": 75, "CROSSOVER_METHOD": "pmx", "PX": 1.0, "PM": 1.0},
    {"TOURNAMENT": 50, "CROSSOVER_METHOD": "pmx", "PX": 1.0, "PM": 1.0},
    {"TOURNAMENT": 75, "CROSSOVER_METHOD": "pmx", "PX": 0.3, "PM": 1.0},
    {"TOURNAMENT": 10, "PX": 0.7, "PM": 0.7, "POP_SIZE": 50, "GENERATIONS": 200},
    {"TOURNAMENT": 20, "PX": 1.0, "PM": 0.7, "POP_SIZE": 50, "GENERATIONS": 200},
    {"TOURNAMENT": 50, "PX": 1.0, "PM": 0.7},
    {"MUTATION_METHOD": "swap", "TOURNAMENT": 10, "PX": 0.9, "PM": 0.02, "POP_SIZE": 50, "GENERATIONS": 200},
    {"INIT_METHOD": "greedy"},
    {"INIT_METHOD": "greedy", "TOURNAMENT": 10, "PX": 0.5, "PM": 0.5},
    {"INIT_METHOD": "greedy", "TOURNAMENT": 5, "PX": 0.7, "PM": 0.5},
]

SA_CANDIDATES = [
    {},
    {"SA_START_TEMP": 1000.0, "SA_MIN_TEMP": 1.0, "SA_ALPHA": alpha_for(1000.0, 1.0)},
    {"SA_START_TEMP": 3000.0, "SA_MIN_TEMP": 0.1, "SA_ALPHA": alpha_for(3000.0, 0.1)},
    {"SA_COOLING_SCHEMA": "linear", "SA_START_TEMP": 100.0, "SA_MIN_TEMP": 1.0, "SA_BETA": beta_for(100.0, 1.0)},
    {"SA_COOLING_SCHEMA": "linear", "SA_START_TEMP": 10.0, "SA_MIN_TEMP": 0.1, "SA_BETA": beta_for(10.0, 0.1)},
]


EA_SWEEP = {
    # jeden parametr naraz
    "PM": [{"PM": v} for v in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]],
    "PX": [{"PX": v} for v in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]],
    "TOUR": [{"TOURNAMENT": v} for v in [1, 2, 3, 5, 10, 20, 50, 100]],
    "SELEKCJA": [{"SELECTION_METHOD": v} for v in ["tournament", "roulette"]],
    "KRZYZOWANIE": [{"CROSSOVER_METHOD": v} for v in ["ox", "pmx", "cx"]],

    "MUTACJA": [{"MUTATION_METHOD": "inversion"}]
               + [{"MUTATION_METHOD": "swap", "PM": v} for v in [0.005, 0.01, 0.02, 0.05]],
    "INICJALIZACJA": [{"INIT_METHOD": v} for v in ["random", "greedy"]],

    "POP_SIZE": [{"POP_SIZE": v} for v in [10, 20, 50, 100, 200, 500]],
    "GENERATIONS": [{"GENERATIONS": v} for v in [10, 20, 50, 100, 200, 500]],

    # ten sam budzet ocen (pop_size x gen = 10 000), rozny podzial
    "POP_GEN": [{"POP_SIZE": pop, "GENERATIONS": gen}
                for pop, gen in [(20, 500), (50, 200), (100, 100), (200, 50), (500, 20)]],

    # siatki kombinacji
    # Pm x Tour x Px x krzyzowanie x podzial budzetu (turniej nie wiekszy niz populacja)
    "SIATKA": [{"PM": pm, "TOURNAMENT": tour, "PX": px, "CROSSOVER_METHOD": cx,
                "POP_SIZE": pop, "GENERATIONS": gen}
               for pm, tour, px, cx, (pop, gen) in product(
                   [0.1, 0.3, 0.5, 0.7, 1.0], [3, 5, 10, 20, 50, 75], PX_GRID, ["ox", "pmx"],
                   [(50, 200), (100, 100), (200, 50)])
               if tour <= pop],
    # to samo dla swapa
    "SIATKA_SWAP": [{"MUTATION_METHOD": "swap", "PM": pm, "TOURNAMENT": tour, "PX": px,
                     "POP_SIZE": pop, "GENERATIONS": gen}
                    for pm, tour, px, (pop, gen) in product(
                        [0.005, 0.01, 0.02], [5, 10, 20, 50], PX_GRID,
                        [(50, 200), (100, 100), (200, 50)])
                    if tour <= pop],
    # inicjalizacja zachlanna - czy wymaga innych Pm / Tour niz losowa
    "SIATKA_GREEDY": [{"INIT_METHOD": "greedy", "PM": pm, "TOURNAMENT": tour, "PX": px}
                      for pm, tour, px in product([0.1, 0.3, 0.5, 0.7, 1.0], [3, 5, 10, 20, 50],
                                                  PX_GRID)],

    # walidacja kandydatow na innych seedach
    "WALIDACJA": [{**VALIDATION, **c} for c in EA_CANDIDATES],
}


SA_SWEEP = {
    # jeden parametr naraz
    "T0": [{"SA_START_TEMP": v} for v in [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]],
    "ALFA": [{"SA_ALPHA": v} for v in [0.99, 0.995, 0.999, 0.9995, 0.9999]],
    "TMIN": [{"SA_MIN_TEMP": v} for v in [1.0, 0.1, 0.05, 0.01, 0.001]],
    "CHLODZENIE": [{"SA_COOLING_SCHEMA": "geometric"},
                   {"SA_COOLING_SCHEMA": "linear", "SA_BETA": 0.1}],

    # staly budzet ~SA_BUDGET iteracji, rozny zakres temperatur
    "SIATKA_GEOMETRYCZNE": [{"SA_START_TEMP": t0, "SA_MIN_TEMP": tmin, "SA_ALPHA": alpha_for(t0, tmin)}
                            for t0, tmin in product(
                                [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0],
                                [0.001, 0.01, 0.1, 1.0])
                            if t0 > tmin],
    "SIATKA_LINIOWE": [{"SA_COOLING_SCHEMA": "linear", "SA_START_TEMP": t0, "SA_MIN_TEMP": tmin,
                        "SA_BETA": beta_for(t0, tmin)}
                       for t0, tmin in product(
                           [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0],
                           [0.01, 0.1, 1.0])
                       if t0 > tmin],

    # walidacja kandydatow na innych seedach
    "WALIDACJA": [{**VALIDATION, **c} for c in SA_CANDIDATES],
}

# kolumny tabel zbiorczych (parametry kazdej konfiguracji w osobnych kolumnach)
EA_SUMMARY_COLUMNS = ["instancja", "badanie", "init", "selekcja", "tour", "krzyzowanie", "px",
                      "mutacja", "pm", "pop_size", "gen", "best", "worst", "avg", "std", "oceny", "plik"]
SA_SUMMARY_COLUMNS = ["instancja", "badanie", "chlodzenie", "t0", "t_min", "alfa", "beta",
                      "best", "worst", "avg", "std", "oceny", "plik"]



# lista zadan: (metoda, instancja, nazwa badania, pelne parametry, sciezka CSV)
def build_jobs():
    base = {**default_params(), **SWEEP_BASE}
    jobs = []
    for path in SWEEP_INSTANCES:
        for method, sweep in [("EA", EA_SWEEP), ("SA", SA_SWEEP)]:
            for study, changes in sweep.items():
                for change in changes:
                    p = {**base, **change}
                    tag = config_tag(p) if method == "EA" else sa_config_tag(p)
                    csv_path = SWEEP_DIR / f"{path.stem}_{tag}_s{p['SEED']}.csv"
                    jobs.append((method, path, study, p, csv_path))
    return jobs


# jedno zadanie = N_RUNS uruchomien jednej konfiguracji na jednej instancji (liczone w osobnym procesie)
def run_job(method, path, p, csv_path):
    inst = TTPInstance(path)
    if method == "EA":
        _, _, history_rows = run_ea(inst, p)
        save_history_csv(csv_path, EA_COLUMNS, ea_csv_params(p), history_rows)
    else:
        _, _, history_rows = run_sa(inst, p)
        save_history_csv(csv_path, SA_COLUMNS, sa_csv_params(p), history_rows)
    return csv_path


# statystyki z gotowego pliku CSV: najlepszy wynik kazdego uruchomienia -> best / worst / avg / std
def stats_from_csv(method, csv_path):
    df = pd.read_csv(csv_path, sep=";")
    finals = df.groupby("uruchomienie")["najlepsza_ocena"].max()
    if method == "EA":
        evals = (df["nr_pokolenia"].max() + 1) * df["pop_size"].iloc[0]
    else:
        evals = (df.groupby("uruchomienie")["iteracja"].max() + 2).mean()  # +1 start, +1 iteracje od 0
    return finals.max(), finals.min(), finals.mean(), finals.std(ddof=1), int(evals)


# parametry jako tekst, zeby zapis CSV nie zaokraglil np. alfa = 0.999302 do 1.00
def ea_param_cells(p):
    tour = str(p["TOURNAMENT"]) if p["SELECTION_METHOD"] == "tournament" else "-"
    return [p["INIT_METHOD"], p["SELECTION_METHOD"], tour, p["CROSSOVER_METHOD"], str(p["PX"]),
            p["MUTATION_METHOD"], str(p["PM"]), str(p["POP_SIZE"]), str(p["GENERATIONS"])]


def sa_param_cells(p):
    geometric = p["SA_COOLING_SCHEMA"] == "geometric"
    return [p["SA_COOLING_SCHEMA"], str(p["SA_START_TEMP"]), str(p["SA_MIN_TEMP"]),
            str(p["SA_ALPHA"]) if geometric else "-", "-" if geometric else str(p["SA_BETA"])]

def main():
    jobs = build_jobs()

    # ta sama konfiguracja moze wystapic w kilku badaniach, ale liczymy ja raz
    unique = {}
    for method, path, study, p, csv_path in jobs:
        unique.setdefault(csv_path, (method, path, p))
    todo = {csv_path: job for csv_path, job in unique.items() if not csv_path.exists()}

    print(f"Konfiguracji: {len(unique)}, juz policzonych: {len(unique) - len(todo)}, "
          f"do policzenia: {len(todo)}, procesow: {N_WORKERS}")

    start = time.perf_counter()
    done = 0
    errors = 0
    if todo:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(run_job, method, path, p, csv_path): csv_path
                       for csv_path, (method, path, p) in todo.items()}
            for future in as_completed(futures):
                csv_path = futures[future]
                done += 1
                try:
                    future.result()
                    status = "OK"
                except Exception:
                    errors += 1
                    status = "BLAD"
                    print(traceback.format_exc())
                elapsed = time.perf_counter() - start
                eta = elapsed / done * (len(todo) - done)
                print(f"[{done}/{len(todo)}] {status} {csv_path.name}  "
                      f"(uplynelo {elapsed / 60:.1f} min, zostalo ok. {eta / 60:.1f} min)")

    # tabele zbiorcze ze wszystkich gotowych plikow (osobno EA i SA)
    ea_rows, sa_rows = [], []
    for method, path, study, p, csv_path in jobs:
        if not csv_path.exists():
            continue
        best, worst, avg, std, evals = stats_from_csv(method, csv_path)
        stats = [best, worst, avg, std, evals, csv_path.name]
        if method == "EA":
            ea_rows.append([path.stem, study, *ea_param_cells(p), *stats])
        else:
            sa_rows.append([path.stem, study, *sa_param_cells(p), *stats])
    save_history_csv(SWEEP_DIR / "podsumowanie_ea.csv", EA_SUMMARY_COLUMNS, {}, ea_rows)
    save_history_csv(SWEEP_DIR / "podsumowanie_sa.csv", SA_SUMMARY_COLUMNS, {}, sa_rows)

    print(f"\nGotowe w {(time.perf_counter() - start) / 60:.1f} min, bledow: {errors}")
    print(f"Podsumowania -> {SWEEP_DIR / 'podsumowanie_ea.csv'}, {SWEEP_DIR / 'podsumowanie_sa.csv'}")


if __name__ == "__main__":
    main()