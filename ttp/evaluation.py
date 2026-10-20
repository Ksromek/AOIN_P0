import numpy as np


# g(y) - suma wartości zabranych przedmiotów
def g_profit(inst, picked):
    total_profit = 0.0
    for i in range(inst.n_items):
        if picked[i]:
            total_profit += inst.profit[i]
    return total_profit


# f(x, y) - czas podróży trasą x z przedmiotami y
def f_travel_time(inst, tour, picked):
    n = inst.n_cities  # liczba miast
    speed_loss_per_weight = (inst.v_max - inst.v_min) / inst.capacity  # spadek prędkości na jednostkę wagi

    # waga podnoszona w każdym mieście
    weight_at_city = np.zeros(n)
    for i in range(inst.n_items):
        if picked[i]:
            weight_at_city[inst.item_city[i]] += inst.weight[i]

    # przejście po miastach
    current_weight = 0.0
    total_time = 0.0
    for k in range(n):
        city = tour[k]
        next_city = tour[(k + 1) % n]  # po ostatnim mieście wracamy do startu
        current_weight += weight_at_city[city]
        speed = inst.v_max - current_weight * speed_loss_per_weight
        total_time += inst.dist[city, next_city] / speed

    return total_time


# G(x, y) = g(y) - f(x, y), przeładowany plecak -> -inf
def evaluate(inst, tour, picked):
    total_weight = 0.0
    for i in range(inst.n_items):
        if picked[i]:
            total_weight += inst.weight[i]

    if total_weight > inst.capacity:
        return -np.inf

    return g_profit(inst, picked) - f_travel_time(inst, tour, picked)


# heurystyka zachłanna - dobiera przedmioty do trasy
def pack_items(inst, tour):
    n = inst.n_cities  # liczba miast
    speed_loss_per_weight = (inst.v_max - inst.v_min) / inst.capacity  # spadek prędkości na jednostkę wagi

    # dystans od każdego miasta na trasie do końca trasy
    dist_to_end = np.zeros(n)
    remaining = 0.0
    for k in range(n - 1, -1, -1):
        city = tour[k]
        next_city = tour[(k + 1) % n]
        remaining += inst.dist[city, next_city]
        dist_to_end[city] = remaining

    item_dist = dist_to_end[inst.item_city]

    # strata czasu na niesieniu przedmiotu (gdyby był jedynym w plecaku)
    carry_penalty = item_dist / (inst.v_max - inst.weight * speed_loss_per_weight) - item_dist / inst.v_max
    worth_it = inst.profit > carry_penalty

    # ranking: cenne, lekkie i niesione krótko na początek
    score = inst.profit / (inst.weight * item_dist)
    order = np.argsort(-score)

    picked = np.zeros(inst.n_items, dtype=np.int8)
    current_weight = 0.0
    for i in order:
        if worth_it[i] and current_weight + inst.weight[i] <= inst.capacity:
            picked[i] = 1
            current_weight += inst.weight[i]

    return picked


# pakowanie + ocena, zwraca (G, picked)
def evaluate_tour(inst, tour):
    picked = pack_items(inst, tour)
    return evaluate(inst, tour, picked), picked


def is_valid_tour(inst, tour):
    if len(tour) != inst.n_cities:  # czy długość trasy równa liczbie miast?
        return False
    if tour[0] != 0:  # czy zaczynamy od miasta 0?
        return False
    if not np.array_equal(np.sort(tour), np.arange(inst.n_cities)):  # czy każde miasto występuje raz?
        return False
    return True