import numpy as np

from ttp.evaluation import evaluate_tour


def nearest_neighbour_tour(inst, start):
    n = inst.n_cities
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = tour[-1]
        dist = inst.dist[current].copy()
        dist[visited] = np.inf  # odwiedzonych nie bierzemy pod uwage
        nearest = int(np.argmin(dist))
        tour.append(nearest)
        visited[nearest] = True

    tour = np.array(tour)
    shift = int(np.where(tour == 0)[0][0])
    return np.roll(tour, -shift)  # obrot: miasto 0 na poczatek


def greedy(inst):
    n = inst.n_cities
    values = np.zeros(n)
    best_tour = None
    best_value = -np.inf

    for start in range(n):
        tour = nearest_neighbour_tour(inst, start)
        value, _ = evaluate_tour(inst, tour)
        values[start] = value
        if value > best_value:
            best_value = value
            best_tour = tour

    return best_tour, best_value, values