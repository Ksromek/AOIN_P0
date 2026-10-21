import numpy as np

from ttp.evaluation import evaluate_tour
from algorithms.random_algorithm import random_tour
from algorithms.operators import inversion_mutation

def make_neighbour(current, rng):
    n = len(current)
    pos_1, pos_2 = np.sort(rng.choice(np.arange(1, n), 2, replace=False))
    return inversion_mutation(current, pos_1, pos_2)


def cool(temp, start_temp, alpha, step, max_iterations, cooling_schema):
    if cooling_schema == "geometric":
        return temp * alpha                                  # T_k = T0 * alpha^k
    if cooling_schema == "linear":
        return start_temp * (1 - step / max_iterations)      # T_k = T0 * (1 - k/K)
    if cooling_schema == "logarithmic":
        return start_temp / np.log(step + np.e)              # T_k = T0 / ln(k + e)
    raise ValueError(f"Nieznany schemat chlodzenia: {cooling_schema}")


def simulated_annealing(inst, rng, start_temp, alpha, max_iterations, min_temp, cooling_schema):
    n = inst.n_cities

    current = random_tour(n, rng)
    current_value, _ = evaluate_tour(inst, current)
    best_tour, best_value = current, current_value
    temp = start_temp
    history = []

    for it in range(max_iterations):
        neighbour = make_neighbour(current, rng)

        neighbour_value, _ = evaluate_tour(inst, neighbour)

        delta = neighbour_value - current_value
        if delta > 0 or rng.random() < np.exp(delta / temp):
            current = neighbour
            current_value = neighbour_value

        if current_value > best_value:
            best_tour = current
            best_value = current_value

        temp = cool(temp, start_temp, alpha, it + 1, max_iterations, cooling_schema)

        history.append((it, current_value, best_value, temp))

        if temp < min_temp:
            break

    n_evals = len(history) + 1  # +1 to ocena trasy startowej
    return best_tour, best_value, history, n_evals