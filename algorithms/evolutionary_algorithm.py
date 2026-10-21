import numpy as np

from ttp.evaluation import evaluate_tour
from algorithms.random_algorithm import random_tour
from algorithms.greedy_algorithm import nearest_neighbour_tour
from algorithms.operators import (OX_crossover, PMX_crossover, CX_crossover, swap_mutation, inversion_mutation, tournament_selection, roulette_selection)


def evaluate_population(inst, population):
    evals = np.zeros(len(population))
    for i in range(len(population)):
        evals[i], _ = evaluate_tour(inst, population[i])
    return evals


def init_population(inst, pop_size, init, rng):
    n = inst.n_cities
    population = []

    if init == "greedy":
        starts = rng.permutation(n)[:min(pop_size, n)]
        for start in starts:
            population.append(nearest_neighbour_tour(inst, start))

    # random
    while len(population) < pop_size:
        population.append(random_tour(n, rng))

    return population


def select(population, evals, selection, tournament, rng):
    if selection == "tournament":
        return tournament_selection(population, evals, tournament, rng)
    if selection == "roulette":
        return roulette_selection(population, evals, rng)
    raise ValueError(f"Nieznana selekcja: {selection}")


def crossover_parents(parent_1, parent_2, crossover, position, length):
    if crossover == "ox":
        return OX_crossover(parent_1, parent_2, position, length)
    if crossover == "pmx":
        child_1, _ = PMX_crossover(parent_1, parent_2, position, length)
        return child_1
    if crossover == "cx":
        return CX_crossover(parent_1, parent_2, position)
    raise ValueError(f"Nieznane krzyzowanie: {crossover}")


def mutate(child, mutation, pm, rng):
    n = len(child)

    if mutation == "inversion":
        pos_1, pos_2 = np.sort(rng.choice(np.arange(1, n), 2, replace=False))
        if rng.random() < pm:
            child = inversion_mutation(child, pos_1, pos_2)
        return child

    if mutation == "swap":
        for i in range(1, n):
            if rng.random() < pm:
                j = rng.integers(1, n)
                child = swap_mutation(child, i, j)
        return child

    raise ValueError(f"Nieznana mutacja: {mutation}")


def evolutionary_algorithm(inst, rng, pop_size, generations, px, pm, tournament, init="random", selection="tournament", crossover="ox", mutation="inversion"):
    n = inst.n_cities

    # pierwsza populacja i jej oceny
    population = init_population(inst, pop_size, init, rng)
    evals = evaluate_population(inst, population)

    best_index = np.argmax(evals)
    best_score = evals[best_index]
    best_tour = population[best_index]

    # historia populacji: (nr pokolenia, best, avg, worst); pokolenie 0 = populacja startowa
    history = [(0, evals.max(), evals.mean(), evals.min())]

    for gen in range(1, generations + 1):
        new_population = []

        for _ in range(pop_size):
            parent_1 = select(population, evals, selection, tournament, rng)
            parent_2 = select(population, evals, selection, tournament, rng)

            position = rng.integers(1, n)
            length = rng.integers(0, n - position + 1)

            if rng.random() < px:
                child = crossover_parents(parent_1, parent_2, crossover, position, length)
            else:
                child = parent_1.copy()

            child = mutate(child, mutation, pm, rng)
            new_population.append(child)

        population = new_population
        evals = evaluate_population(inst, population)

        best_index = np.argmax(evals)
        if evals[best_index] > best_score:
            best_score = evals[best_index]
            best_tour = population[best_index]

        history.append((gen, evals.max(), evals.mean(), evals.min()))

    n_evals = pop_size * (generations + 1)
    return best_tour, best_score, history, n_evals