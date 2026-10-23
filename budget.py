"""Wplyw budzetu ocen na EA i SA (konfiguracje z config.py, budzet x1 / x3 / x10).

EA: wiecej pokolen (gen = GENERATIONS * mnoznik), ta sama populacja.
SA: wolniejsze chlodzenie (alfa = SA_ALPHA ** (1 / mnoznik)), wiec liczba iteracji rosnie mnoznik razy.
Mnoznik 1 to dokladnie konfiguracja z Tabeli 1 (te same seedy), wiec wyniki x1 musza sie z nia zgadzac.
Wyniki: results/budzet/podsumowanie_budzet.csv i budzet_uruchomienia.csv.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from config import INSTANCES, RESULTS_DIR
from experiments import default_params
from algorithms import evolutionary_algorithm, simulated_annealing
from ttp import TTPInstance
from utils.log_to_csv import save_history_csv

MULTIPLIERS = [1, 3, 10]
METHODS = ["EA", "SA"]

N_WORKERS = max(1, (os.cpu_count() or 2) - 4)  # zostawiamy kilka rdzeni dla systemu

BUDGET_DIR = RESULTS_DIR / "budzet"


# jedno uruchomienie jednej metody z danym mnoznikiem budzetu -> (najlepsza ocena, liczba ocen, czas)
def run_once(path, method, mult, run, p):
    inst = TTPInstance(path)
    rng = np.random.default_rng(p["SEED"] + run)  # te same seedy co w experiments.py
    t0 = time.perf_counter()
    if method == "EA":
        _, best, _, n_evals = evolutionary_algorithm(
            inst, rng, p["POP_SIZE"], p["GENERATIONS"] * mult, p["PX"], p["PM"], p["TOURNAMENT"],
            p["INIT_METHOD"], p["SELECTION_METHOD"], p["CROSSOVER_METHOD"], p["MUTATION_METHOD"])
    else:
        alpha = p["SA_ALPHA"] ** (1 / mult)
        _, best, _, n_evals = simulated_annealing(
            inst, rng, p["SA_START_TEMP"], p["SA_MIN_TEMP"], p["SA_COOLING_SCHEMA"], alpha, p["SA_BETA"])
    return best, n_evals, time.perf_counter() - t0


def main():
    p = default_params()
    jobs = [(path, method, mult, run)
            for path in INSTANCES for method in METHODS for mult in MULTIPLIERS for run in range(p["N_RUNS"])]
    # najdluzsze zadania najpierw (duza instancja, duzy budzet), zeby procesy konczyly rowno
    n_items = {path: TTPInstance(path).n_items for path in INSTANCES}
    jobs.sort(key=lambda job: n_items[job[0]] * job[2], reverse=True)

    print(f"Zadan: {len(jobs)}, procesow: {N_WORKERS}")
    results = {}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_once, *job, p): job for job in jobs}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done = len(results)
            elapsed = time.perf_counter() - start
            if done % 10 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] uplynelo {elapsed / 60:.1f} min, "
                      f"zostalo ok. {elapsed / done * (len(jobs) - done) / 60:.1f} min")

    # wszystkie uruchomienia + statystyki (instancje jak w config.py, potem metoda, potem mnoznik)
    run_rows, summary_rows = [], []
    for path in INSTANCES:
        for method in METHODS:
            for mult in MULTIPLIERS:
                res = [results[(path, method, mult, run)] for run in range(p["N_RUNS"])]
                values = np.array([r[0] for r in res])
                evals = int(np.mean([r[1] for r in res]))
                elapsed = float(np.mean([r[2] for r in res]))
                for run, (best, n_evals, t) in enumerate(res):
                    run_rows.append((path.stem, method, mult, run, p["SEED"] + run, best, n_evals, t))
                summary_rows.append((path.stem, method, mult, evals, values.max(), values.min(),
                                     values.mean(), values.std(ddof=1), elapsed))
                print(f"{path.stem:<40} {method} x{mult:<3} avg={values.mean():12.1f}  "
                      f"std={values.std(ddof=1):10.1f}  [{evals} ocen]")

    save_history_csv(BUDGET_DIR / "budzet_uruchomienia.csv",
                     ["instancja", "metoda", "mnoznik", "uruchomienie", "seed", "best", "oceny", "czas_s"],
                     {}, run_rows)
    save_history_csv(BUDGET_DIR / "podsumowanie_budzet.csv",
                     ["instancja", "metoda", "mnoznik", "oceny", "best", "worst", "avg", "std", "czas_s"],
                     {}, summary_rows)
    print(f"\nGotowe w {(time.perf_counter() - start) / 60:.1f} min -> {BUDGET_DIR / 'podsumowanie_budzet.csv'}")


if __name__ == "__main__":
    main()