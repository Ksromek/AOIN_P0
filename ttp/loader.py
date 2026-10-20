import numpy as np


class TTPInstance:
    def __init__(self, path):

        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]

        cities_start = 0
        items_start = 0
        for i in range(len(lines)):
            if lines[i].startswith("NODE_COORD_SECTION"):
                cities_start = i
            elif lines[i].startswith("ITEMS SECTION"):
                items_start = i

        header_lines = lines[:cities_start]
        city_lines = lines[cities_start + 1:items_start]
        item_lines = lines[items_start + 1:]

        self.params = {}
        for line in header_lines:
            key, value = line.split(":", 1)
            self.params[key.strip()] = value.strip()

        coords = []
        for line in city_lines:
            _, x, y = line.split()
            coords.append([float(x), float(y)])

        items = []
        for line in item_lines:
            _, profit, weight, city = line.split()
            items.append([float(profit), float(weight), int(city) - 1])

        self.n_cities = len(coords)
        self.n_items = len(items)

        expected_cities = int(self.params["DIMENSION"])
        expected_items = int(self.params["NUMBER OF ITEMS"])
        if self.n_cities != expected_cities:
            raise ValueError(f"Oczekiwano {expected_cities} miast, wczytano {self.n_cities}")
        if self.n_items != expected_items:
            raise ValueError(f"Oczekiwano {expected_items} przedmiotow, wczytano {self.n_items}")

        self.capacity = float(self.params["CAPACITY OF KNAPSACK"])
        self.v_min = float(self.params["MIN SPEED"])
        self.v_max = float(self.params["MAX SPEED"])

        items = np.array(items)
        self.profit = items[:, 0]
        self.weight = items[:, 1]
        self.item_city = items[:, 2].astype(np.int64)

        # macierz CEIL_2D
        coords = np.array(coords)
        self.dist = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                self.dist[i, j] = np.ceil(np.sqrt(dx * dx + dy * dy))
