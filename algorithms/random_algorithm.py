import numpy as np

from ttp.evaluation import evaluate_tour


def random_tour(n, rng):

    return np.concatenate(([0], rng.permutation(np.arange(1, n))))


def random_search(inst, n_samples, rng):
    
    values = np.zeros(n_samples)
    best_tour = None
    best_value = -np.inf

    best_history = np.zeros(n_samples)

    for s in range(n_samples):
        tour = random_tour(inst.n_cities, rng)
        value, _ = evaluate_tour(inst, tour)
        values[s] = value
        if value > best_value:
            best_value = value
            best_tour = tour
        best_history[s] = best_value

    return best_tour, best_value, values, best_history