import time

import numpy as np

from config import (INSTANCES, RESULTS_DIR, SEED, N_RUNS, N_RANDOM,
                    POP_SIZE, GENERATIONS, PX, PM, TOURNAMENT,
                    INIT_METHOD, SELECTION_METHOD, CROSSOVER_METHOD, MUTATION_METHOD,
                    SA_START_TEMP, SA_MIN_TEMP, SA_COOLING_SCHEMA, SA_ALPHA, SA_BETA,
                    SA_LOG_EVERY)
from ttp import TTPInstance
from algorithms import random_search, greedy, evolutionary_algorithm, simulated_annealing
from utils.log_to_csv import save_history_csv


def summary(name, values, elapsed, n_evals, instance_name, summary_rows):
    std = values.std(ddof=1)  # odchylenie z proby (dzielenie przez n-1)
    print(f"  {name:<10} best={values.max():12.1f}  worst={values.min():12.1f}  "
          f"avg={values.mean():12.1f}  std={std:10.1f}  "
          f"[{n_evals} ocen, {elapsed:.3f} s]")
    summary_rows.append((instance_name, name, values.max(), values.min(), values.mean(), std,
                         n_evals, elapsed))


# opis konfiguracji EA do nazwy pliku, np. random_tour5_ox_px0.7_inversion_pm0.1_pop100_gen100
def config_tag():
    selection = f"tour{TOURNAMENT}" if SELECTION_METHOD == "tournament" else SELECTION_METHOD
    return (f"{INIT_METHOD}_{selection}_{CROSSOVER_METHOD}_px{PX}_"
            f"{MUTATION_METHOD}_pm{PM}_pop{POP_SIZE}_gen{GENERATIONS}")


# opis konfiguracji SA do nazwy pliku, np. sa_geometric_a0.999_t1000.0_min0.05
def sa_config_tag():
    step = f"a{SA_ALPHA}" if SA_COOLING_SCHEMA == "geometric" else f"b{SA_BETA}"
    return f"sa_{SA_COOLING_SCHEMA}_{step}_t{SA_START_TEMP}_min{SA_MIN_TEMP}"


def run_ea(inst):
    best_scores = np.zeros(N_RUNS)
    history_rows = []  # historia wszystkich uruchomien na tej instancji
    for run in range(N_RUNS):
        seed = SEED + run  # kazde uruchomienie z innym seedem
        rng = np.random.default_rng(seed)
        _, best_scores[run], history, n_evals = evolutionary_algorithm(
            inst, rng, POP_SIZE, GENERATIONS, PX, PM, TOURNAMENT,
            INIT_METHOD, SELECTION_METHOD, CROSSOVER_METHOD, MUTATION_METHOD)
        for gen, best, avg, worst in history:
            history_rows.append((run, seed, gen, best, avg, worst))
    return best_scores, n_evals, history_rows


def run_sa(inst):
    best_scores = np.zeros(N_RUNS)
    evals = np.zeros(N_RUNS, dtype=int)
    history_rows = []  # historia wszystkich uruchomien na tej instancji
    for run in range(N_RUNS):
        seed = SEED + run  # kazde uruchomienie z innym seedem
        rng = np.random.default_rng(seed)
        _, best_scores[run], history, evals[run] = simulated_annealing(
            inst, rng, SA_START_TEMP, SA_MIN_TEMP, SA_COOLING_SCHEMA, SA_ALPHA, SA_BETA)
        for it, current, best, temp in history:
            if it % SA_LOG_EVERY == 0 or it == len(history) - 1:
                history_rows.append((run, seed, it, current, best, temp))
    return best_scores, evals, history_rows


def main():
    print(f"EA: init={INIT_METHOD}, sel={SELECTION_METHOD}, Tour={TOURNAMENT}, "
          f"cx={CROSSOVER_METHOD}, Px={PX}, mut={MUTATION_METHOD}, Pm={PM}, "
          f"pop_size={POP_SIZE}, gen={GENERATIONS}, runs={N_RUNS}")
    print(f"SA: cooling={SA_COOLING_SCHEMA}, T0={SA_START_TEMP}, min_temp={SA_MIN_TEMP}, "
          f"alpha={SA_ALPHA}, beta={SA_BETA}, runs={N_RUNS}")

    # parametry do CSV
    ea_params = {
        "init": INIT_METHOD, "selection": SELECTION_METHOD, "tournament": TOURNAMENT,
        "crossover": CROSSOVER_METHOD, "px": PX, "mutation": MUTATION_METHOD, "pm": PM,
        "pop_size": POP_SIZE, "generations": GENERATIONS,
    }
    sa_params = {
        "cooling": SA_COOLING_SCHEMA, "start_temp": SA_START_TEMP, "min_temp": SA_MIN_TEMP,
        "alpha": SA_ALPHA, "beta": SA_BETA,
    }

    summary_rows = []  # tabela podsumowania

    for path in INSTANCES:
        rng = np.random.default_rng(SEED)
        inst = TTPInstance(path)
        instance_name = path.stem
        print(f"\n{path.name}  (n={inst.n_cities}, m={inst.n_items})")

        t0 = time.perf_counter()
        _, _, values, best_history = random_search(inst, N_RANDOM, rng)
        summary("losowy", values, time.perf_counter() - t0, N_RANDOM, instance_name, summary_rows)

        t0 = time.perf_counter()
        _, _, values = greedy(inst)
        summary("zachlanny", values, time.perf_counter() - t0, inst.n_cities, instance_name,
                summary_rows)

        t0 = time.perf_counter()
        values, n_evals, history_rows = run_ea(inst)
        summary("EA", values, (time.perf_counter() - t0) / N_RUNS, n_evals, instance_name,
                summary_rows)

        csv_path = RESULTS_DIR / f"{instance_name}_{config_tag()}.csv"
        save_history_csv(csv_path, ["uruchomienie", "seed", "nr_pokolenia", "najlepsza_ocena",
                                    "srednia_ocen", "najgorsza_ocena"], ea_params, history_rows)
        print(f"  historia EA -> {csv_path.name}")

        t0 = time.perf_counter()
        values, evals, history_rows = run_sa(inst)
        summary("SA", values, (time.perf_counter() - t0) / N_RUNS, int(evals.mean()), instance_name,
                summary_rows)

        csv_path = RESULTS_DIR / f"{instance_name}_{sa_config_tag()}.csv"
        save_history_csv(csv_path, ["uruchomienie", "seed", "iteracja", "biezaca_ocena",
                                    "najlepsza_ocena", "temperatura"], sa_params, history_rows)
        print(f"  historia SA -> {csv_path.name}")

    # tabela podsumowania
    csv_path = RESULTS_DIR / "podsumowanie" / f"{config_tag()}__{sa_config_tag()}.csv"
    save_history_csv(csv_path, ["instancja", "metoda", "best", "worst", "avg", "std", "oceny", "czas_s"],
                     {}, summary_rows)
    print(f"\nPodsumowanie -> podsumowanie/{csv_path.name}")


if __name__ == "__main__":
    main()