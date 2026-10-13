import numpy as np


def evaluate(inst, tour, picked):
    n = inst.n_cities
    nu = (inst.v_max - inst.v_min) / inst.capacity

    # weight and profit
    weight_at_city = np.zeros(n)
    total_profit = 0.0
    total_weight = 0.0
    for i in range(inst.n_items):
        if picked[i]:
            weight_at_city[inst.item_city[i]] += inst.weight[i]
            total_profit += inst.profit[i]
            total_weight += inst.weight[i]

    if total_weight > inst.capacity:
        return -np.inf

    # przejście po miastach
    current_weight = 0.0
    total_time = 0.0
    for k in range(n):
        city = tour[k]
        next_city = tour[(k + 1) % n]   # po ostatnim miescie wracamy do startu
        current_weight += weight_at_city[city]
        speed = inst.v_max - current_weight * nu
        total_time += inst.dist[city, next_city] / speed

    return total_profit - total_time


def pack_items(inst, tour):
    n = inst.n_cities
    nu = (inst.v_max - inst.v_min) / inst.capacity

    # dystans od kazdego miasta na trasie do konca trasy
    dist_to_end = np.zeros(n)
    remaining = 0.0
    for k in range(n - 1, -1, -1):
        city = tour[k]
        next_city = tour[(k + 1) % n]
        remaining += inst.dist[city, next_city]
        dist_to_end[city] = remaining

    item_dist = dist_to_end[inst.item_city]

    # czas straty na niesieniu przedmiotu
    carry_penalty = item_dist / (inst.v_max - inst.weight * nu) - item_dist / inst.v_max
    worth_it = inst.profit > carry_penalty

    score = inst.profit / (inst.weight * item_dist)
    order = np.argsort(-score)

    picked = np.zeros(inst.n_items, dtype=np.int8)
    current_weight = 0.0
    for i in order:
        if worth_it[i] and current_weight + inst.weight[i] <= inst.capacity:
            picked[i] = 1
            current_weight += inst.weight[i]

    return picked


def evaluate_tour(inst, tour):

    picked = pack_items(inst, tour)
    return evaluate(inst, tour, picked), picked