import numpy as np


def OX_crossover(parent_1, parent_2, position, length):
    child = np.full(len(parent_1), -1)
    common = []
    for i in range(position, position + length):
        child[i] = parent_1[i]
        common.append(parent_1[i])

    missing = []
    for city in parent_2:
        if city not in common:
            missing.append(city)

    j = 0
    for i in range(len(parent_1)):
        if child[i] == -1:
            child[i] = missing[j]
            j += 1

    return child


def PMX_crossover(parent_1, parent_2, position, length):
    child_1 = np.full(len(parent_1), -1)
    child_2 = np.full(len(parent_2), -1)

    for i in range(position, position + length):
        child_1[i] = parent_2[i]
        child_2[i] = parent_1[i]

    segment_1 = list(parent_2[position:position + length])
    segment_2 = list(parent_1[position:position + length])

    for i in range(len(parent_1)):
        if child_1[i] == -1:
            city = parent_1[i]
            while city in segment_1:
                city = segment_2[segment_1.index(city)]
            child_1[i] = city
        if child_2[i] == -1:
            city = parent_2[i]
            while city in segment_2:
                city = segment_1[segment_2.index(city)]
            child_2[i] = city

    return child_1, child_2


def CX_crossover(parent_1, parent_2, position):
    child = np.full(len(parent_1), -1)

    while child[position] == -1:
        child[position] = parent_1[position]
        position = list(parent_2).index(parent_1[position])

    for i in range(len(child)):
        if child[i] == -1:
            child[i] = parent_2[i]

    return child


def swap_mutation(instance, pos_1, pos_2):
    new_instance = np.copy(instance)
    temp = new_instance[pos_1]
    new_instance[pos_1] = new_instance[pos_2]
    new_instance[pos_2] = temp

    return new_instance


def inversion_mutation(instance, pos_1, pos_2):
    new_instance = np.copy(instance)
    left = pos_1
    right = pos_2

    while left < right:
        temp = new_instance[left]
        new_instance[left] = new_instance[right]
        new_instance[right] = temp
        left += 1
        right -= 1

    return new_instance


def tournament_selection(population, eval_array, size_of_tournament, rng):

    participants = rng.choice(len(population), size_of_tournament, replace=False)

    best_index = participants[0]
    for i in participants:
        if eval_array[i] > eval_array[best_index]:
            best_index = i

    winner = population[best_index]

    return winner


def roulette_selection(population, eval_array, rng):
    fitness = eval_array - eval_array.min() + 1
    probabilities = fitness / fitness.sum()
    boundaries = np.cumsum(probabilities)

    spin = rng.random()

    winner_index = 0
    
    for i in range(len(boundaries)):
        if boundaries[i] > spin:
            winner_index = i
            break

    return population[winner_index]