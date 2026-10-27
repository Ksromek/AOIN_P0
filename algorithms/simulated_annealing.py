import numpy as np

from ttp.evaluation import evaluate_tour
from algorithms.random_algorithm import random_tour
from algorithms.operators import inversion_mutation


# sasiad = biezaca trasa z jedna inwersja
def make_neighbour(current, rng):
    n = len(current)
    pos_1, pos_2 = np.sort(rng.choice(np.arange(1, n), 2, replace=False))
    return inversion_mutation(current, pos_1, pos_2)


# schematy chłodzenia
def cool(temp, cooling_schema, alpha, beta):
    if cooling_schema == "geometric":
        return temp * alpha        # T_k+1 = alpha * T_k
    if cooling_schema == "linear":
        return temp - beta         # T_k+1 = T_k - beta
    raise ValueError(f"Nieznany schemat chlodzenia: {cooling_schema}")


def simulated_annealing(inst, rng, start_temp, min_temp, cooling_schema, alpha, beta):

    if min_temp <= 0:
        raise ValueError("min_temp musi byc > 0")
    if cooling_schema == "geometric" and not 0 < alpha < 1:
        raise ValueError("alpha musi byc z przedzialu (0, 1)")
    if cooling_schema == "linear" and beta <= 0:
        raise ValueError("beta musi byc > 0")

    n = inst.n_cities

    current = random_tour(n, rng)
    current_value, _ = evaluate_tour(inst, current)
    best_tour, best_value = current, current_value
    temp = start_temp
    history = []
    it = 0

    # stop: temperatura < min_temp
    while temp > min_temp:
        neighbour = make_neighbour(current, rng)

        neighbour_value, _ = evaluate_tour(inst, neighbour)

        delta = neighbour_value - current_value
        if delta > 0 or rng.random() < np.exp(delta / temp):
            current = neighbour
            current_value = neighbour_value

        if current_value > best_value:
            best_tour = current
            best_value = current_value

        temp = cool(temp, cooling_schema, alpha, beta)

        history.append((it, current_value, best_value, temp))
        it += 1

    n_evals = len(history) + 1  # +1 za ocene trasy startowej
    return best_tour, best_value, history, n_evals