import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from config import INSTANCES, RESULTS_DIR
from experiments import default_params
from algorithms import evolutionary_algorithm, simulated_annealing
from ttp import TTPInstance
from utils.log_to_csv import save_history_csv

MULTIPLIERS = [1, 3, 10]
METHODS = ["EA", "SA"]

N_WORKERS = max(1, (os.cpu_count() or 2) - 4)  # zostawiamy kilka rdzeni dla systemu


# jedno uruchomienie jednej metody z danym mnoznikiem budzetu -> (najlepsza ocena, liczba ocen)
def run_once(path, method, mult, run, p):
    inst = TTPInstance(path)
    rng = np.random.default_rng(p["SEED"] + run)  # te same seedy co w experiments.py
    if method == "EA":
        _, best, _, n_evals = evolutionary_algorithm(
            inst, rng, p["POP_SIZE"], p["GENERATIONS"] * mult, p["PX"], p["PM"], p["TOURNAMENT"],
            p["INIT_METHOD"], p["SELECTION_METHOD"], p["CROSSOVER_METHOD"], p["MUTATION_METHOD"])
    else:
        alpha = p["SA_ALPHA"] ** (1 / mult)
        _, best, _, n_evals = simulated_annealing(
            inst, rng, p["SA_START_TEMP"], p["SA_MIN_TEMP"], p["SA_COOLING_SCHEMA"], alpha, p["SA_BETA"])
    return best, n_evals


def main():
    p = default_params()

    # lista wszystkich uruchomien: (instancja, metoda, mnoznik, nr uruchomienia)
    jobs = [(path, method, mult, run)
            for path in INSTANCES for method in METHODS for mult in MULTIPLIERS for run in range(p["N_RUNS"])]

    # uruchomienia liczone rownolegle; wyniki wracaja w tej samej kolejnosci co jobs
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(run_once, *job, p) for job in jobs]
        results = {job: future.result() for job, future in zip(jobs, futures)}

    # statystyki z 10 uruchomien dla kazdej trojki (instancja, metoda, mnoznik)
    rows = []
    for path in INSTANCES:
        for method in METHODS:
            for mult in MULTIPLIERS:
                res = [results[(path, method, mult, run)] for run in range(p["N_RUNS"])]
                values = np.array([best for best, _ in res])
                n_evals = res[0][1]
                rows.append((path.stem, method, mult, n_evals,
                             values.max(), values.min(), values.mean(), values.std(ddof=1)))
                print(f"{path.stem:<40} {method} x{mult:<3} avg={values.mean():12.1f}  "
                      f"std={values.std(ddof=1):10.1f}  [{n_evals} ocen]")

    save_history_csv(RESULTS_DIR / "budzet" / "podsumowanie_budzet.csv",
                     ["instancja", "metoda", "mnoznik", "oceny", "best", "worst", "avg", "std"],
                     {}, rows)


if __name__ == "__main__":
    main()