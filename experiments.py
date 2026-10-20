import time

import numpy as np

from config import (INSTANCES, RESULTS_DIR, SEED, N_RUNS, N_RANDOM, POP_SIZE, GENERATIONS, PX, PM, TOURNAMENT, INIT_METHOD, SELECTION_METHOD, CROSSOVER_METHOD, MUTATION_METHOD)
from ttp import TTPInstance
from algorithms import random_search, greedy, evolutionary_algorithm
from utils.log_to_csv import save_history_csv


def summary(name, values, elapsed, n_evals):
    print(f"  {name:<10} best={values.max():12.1f}  worst={values.min():12.1f}  "
          f"avg={values.mean():12.1f}  std={values.std():10.1f}  "
          f"[{n_evals} ocen, {elapsed:.3f} s]")


def run_ea(inst, instance_name, all_history):
    best_scores = np.zeros(N_RUNS)
    for run in range(N_RUNS):
        rng = np.random.default_rng(SEED + run)  # kazde uruchomienie z innym seedem
        _, best_scores[run], history, n_evals = evolutionary_algorithm(
            inst, rng, POP_SIZE, GENERATIONS, PX, PM, TOURNAMENT,
            INIT_METHOD, SELECTION_METHOD, CROSSOVER_METHOD, MUTATION_METHOD)
        for gen, best, avg, worst in history:
            all_history.append((instance_name, run, gen, best, avg, worst))
    return best_scores, n_evals


def main():
    print(f"EA: init={INIT_METHOD}, sel={SELECTION_METHOD}, Tour={TOURNAMENT}, "
          f"cx={CROSSOVER_METHOD}, Px={PX}, mut={MUTATION_METHOD}, Pm={PM}, "
          f"pop_size={POP_SIZE}, gen={GENERATIONS}, runs={N_RUNS}")

    all_history = []  # wszystkie przebiegi EA, zapisane do jednego CSV

    for path in INSTANCES:
        rng = np.random.default_rng(SEED)
        inst = TTPInstance(path)
        instance_name = path.stem
        print(f"\n{path.name}  (n={inst.n_cities}, m={inst.n_items})")

        t0 = time.perf_counter()
        _, _, values, best_history = random_search(inst, N_RANDOM, rng)
        summary("losowy", values, time.perf_counter() - t0, N_RANDOM)

        t0 = time.perf_counter()
        _, _, values = greedy(inst)
        summary("zachlanny", values, time.perf_counter() - t0, inst.n_cities)

        t0 = time.perf_counter()
        values, n_evals = run_ea(inst, instance_name, all_history)
        summary("EA", values, (time.perf_counter() - t0) / N_RUNS, n_evals)

    save_history_csv(RESULTS_DIR / "ea_history.csv", all_history)
    print(f"\nHistoria EA zapisana do {RESULTS_DIR / 'ea_history.csv'}")


if __name__ == "__main__":
    main()