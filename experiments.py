import time

import numpy as np

import config
from config import INSTANCES, RESULTS_DIR
from ttp import TTPInstance
from algorithms import random_search, greedy, evolutionary_algorithm, simulated_annealing
from utils.log_to_csv import save_history_csv

EA_COLUMNS = ["uruchomienie", "seed", "nr_pokolenia", "najlepsza_ocena", "srednia_ocen", "najgorsza_ocena"]
SA_COLUMNS = ["uruchomienie", "seed", "iteracja", "biezaca_ocena", "najlepsza_ocena", "temperatura"]


# wszystkie parametry eksperymentu
def default_params():
    names = ["SEED", "N_RUNS",
             "POP_SIZE", "GENERATIONS", "PX", "PM", "TOURNAMENT",
             "INIT_METHOD", "SELECTION_METHOD", "CROSSOVER_METHOD", "MUTATION_METHOD",
             "SA_START_TEMP", "SA_MIN_TEMP", "SA_COOLING_SCHEMA", "SA_ALPHA", "SA_BETA", "SA_LOG_EVERY"]
    return {name: getattr(config, name) for name in names}


# statystyki metody
def summary(name, values, elapsed, n_evals, instance_name):
    std = values.std(ddof=1)  # odchylenie z proby (dzielenie przez n-1)
    line = (f"{instance_name:<40} {name:<10} best={values.max():12.1f}  worst={values.min():12.1f}  "
            f"avg={values.mean():12.1f}  std={std:10.1f}  [{n_evals} ocen, {elapsed:.3f} s]")
    row = (instance_name, name, values.max(), values.min(), values.mean(), std, n_evals, elapsed)
    return row, line


# opis konfiguracji EA do nazwy pliku, np. random_tour5_ox_px0.7_inversion_pm0.1_pop100_gen100
def config_tag(p):
    selection = f"tour{p['TOURNAMENT']}" if p["SELECTION_METHOD"] == "tournament" else p["SELECTION_METHOD"]
    return (f"{p['INIT_METHOD']}_{selection}_{p['CROSSOVER_METHOD']}_px{p['PX']}_"
            f"{p['MUTATION_METHOD']}_pm{p['PM']}_pop{p['POP_SIZE']}_gen{p['GENERATIONS']}")


# opis konfiguracji SA do nazwy pliku, np. sa_geometric_a0.999_t1000.0_min0.05
def sa_config_tag(p):
    step = f"a{p['SA_ALPHA']}" if p["SA_COOLING_SCHEMA"] == "geometric" else f"b{p['SA_BETA']}"
    return f"sa_{p['SA_COOLING_SCHEMA']}_{step}_t{p['SA_START_TEMP']}_min{p['SA_MIN_TEMP']}"


# parametry zapisywane w kazdym wierszu CSV
def ea_csv_params(p):
    return {
        "init": p["INIT_METHOD"], "selection": p["SELECTION_METHOD"], "tournament": p["TOURNAMENT"],
        "crossover": p["CROSSOVER_METHOD"], "px": p["PX"], "mutation": p["MUTATION_METHOD"],
        "pm": p["PM"], "pop_size": p["POP_SIZE"], "generations": p["GENERATIONS"],
    }


def sa_csv_params(p):
    return {
        "cooling": p["SA_COOLING_SCHEMA"], "start_temp": p["SA_START_TEMP"], "min_temp": p["SA_MIN_TEMP"],
        "alpha": p["SA_ALPHA"], "beta": p["SA_BETA"],
    }


def run_ea(inst, p):
    best_scores = np.zeros(p["N_RUNS"])
    history_rows = []  # historia wszystkich uruchomien na tej instancji
    for run in range(p["N_RUNS"]):
        seed = p["SEED"] + run  # kazde uruchomienie z innym seedem
        rng = np.random.default_rng(seed)
        _, best_scores[run], history, n_evals = evolutionary_algorithm(
            inst, rng, p["POP_SIZE"], p["GENERATIONS"], p["PX"], p["PM"], p["TOURNAMENT"],
            p["INIT_METHOD"], p["SELECTION_METHOD"], p["CROSSOVER_METHOD"], p["MUTATION_METHOD"])
        for gen, best, avg, worst in history:
            history_rows.append((run, seed, gen, best, avg, worst))
    return best_scores, n_evals, history_rows


def run_sa(inst, p):
    best_scores = np.zeros(p["N_RUNS"])
    evals = np.zeros(p["N_RUNS"], dtype=int)
    history_rows = []  # historia wszystkich uruchomien na tej instancji (co SA_LOG_EVERY iteracji)
    for run in range(p["N_RUNS"]):
        seed = p["SEED"] + run  # kazde uruchomienie z innym seedem
        rng = np.random.default_rng(seed)
        _, best_scores[run], history, evals[run] = simulated_annealing(
            inst, rng, p["SA_START_TEMP"], p["SA_MIN_TEMP"], p["SA_COOLING_SCHEMA"],
            p["SA_ALPHA"], p["SA_BETA"])
        for it, current, best, temp in history:
            if it % p["SA_LOG_EVERY"] == 0 or it == len(history) - 1:
                history_rows.append((run, seed, it, current, best, temp))
    return best_scores, evals, history_rows


METHODS = ["losowy", "zachlanny", "EA", "EA_greedy", "SA"]


# jedna metoda na jednej instancji
def run_method(path, method, p):
    inst = TTPInstance(path)
    instance_name = path.stem

    if method == "losowy":
        n_random = p["POP_SIZE"] * p["GENERATIONS"]  # tyle ocen, ile EA (pokolenia x osobniki)
        rng = np.random.default_rng(p["SEED"])
        t0 = time.perf_counter()
        _, _, values, _ = random_search(inst, n_random, rng)
        return summary(method, values, time.perf_counter() - t0, n_random, instance_name)

    if method == "zachlanny":
        t0 = time.perf_counter()
        _, _, values = greedy(inst)
        return summary(method, values, time.perf_counter() - t0, inst.n_cities, instance_name)

    if method in ("EA", "EA_greedy"):
        p_ea = p if method == "EA" else {**p, **config.EA_GREEDY}
        t0 = time.perf_counter()
        values, n_evals, history_rows = run_ea(inst, p_ea)
        result = summary(method, values, (time.perf_counter() - t0) / p["N_RUNS"], n_evals, instance_name)
        save_history_csv(RESULTS_DIR / f"{instance_name}_{config_tag(p_ea)}.csv",
                         EA_COLUMNS, ea_csv_params(p_ea), history_rows)
        return result

    if method == "SA":
        t0 = time.perf_counter()
        values, evals, history_rows = run_sa(inst, p)
        result = summary(method, values, (time.perf_counter() - t0) / p["N_RUNS"], int(evals.mean()),
                         instance_name)
        save_history_csv(RESULTS_DIR / f"{instance_name}_{sa_config_tag(p)}.csv",
                         SA_COLUMNS, sa_csv_params(p), history_rows)
        return result

    raise ValueError(f"Nieznana metoda: {method}")


def main():
    p = default_params()

    print(f"EA: init={p['INIT_METHOD']}, sel={p['SELECTION_METHOD']}, Tour={p['TOURNAMENT']}, "
          f"cx={p['CROSSOVER_METHOD']}, Px={p['PX']}, mut={p['MUTATION_METHOD']}, Pm={p['PM']}, "
          f"pop_size={p['POP_SIZE']}, gen={p['GENERATIONS']}, runs={p['N_RUNS']}")
    print(f"EA_greedy: zmiany wzgledem EA: {config.EA_GREEDY}")
    print(f"SA: cooling={p['SA_COOLING_SCHEMA']}, T0={p['SA_START_TEMP']}, min_temp={p['SA_MIN_TEMP']}, "
          f"alpha={p['SA_ALPHA']}, beta={p['SA_BETA']}, runs={p['N_RUNS']}")

    summary_rows = []
    for path in INSTANCES:
        for method in METHODS:
            row, line = run_method(path, method, p)
            summary_rows.append(row)
            print(line)

    # tabela podsumowania (konfiguracja EA + SA w nazwie)
    csv_path = RESULTS_DIR / "podsumowanie" / f"{config_tag(p)}__{sa_config_tag(p)}.csv"
    save_history_csv(csv_path, ["instancja", "metoda", "best", "worst", "avg", "std", "oceny", "czas_s"],
                     {}, summary_rows)
    print(f"\nPodsumowanie -> podsumowanie/{csv_path.name}")


if __name__ == "__main__":
    main()