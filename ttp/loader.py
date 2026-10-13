import numpy as np


class TTPInstance:
    def __init__(self, path):
        self.params = {}   # parametry z naglowka, np. "CAPACITY OF KNAPSACK" -> "40469"
        coords = []        # [x, y] dla kazdego miasta
        items = []         # [profit, weight, miasto] dla kazdego przedmiotu
        section = "header"

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("NODE_COORD_SECTION"):
                    section = "cities"
                elif line.startswith("ITEMS SECTION"):
                    section = "items"
                elif section == "header":
                    key, value = line.split(":", 1)
                    self.params[key.strip()] = value.strip()
                elif section == "cities":
                    _, x, y = line.split()
                    coords.append([float(x), float(y)])
                else:
                    _, profit, weight, city = line.split()
                    items.append([float(profit), float(weight), int(city) - 1])  # miasta od 0

        self.n_cities = len(coords)
        self.n_items = len(items)
        self.capacity = float(self.params["CAPACITY OF KNAPSACK"])
        self.v_min = float(self.params["MIN SPEED"])
        self.v_max = float(self.params["MAX SPEED"])

        items = np.array(items)
        self.profit = items[:, 0]
        self.weight = items[:, 1]
        self.item_city = items[:, 2].astype(np.int64)

        # macierz odleglosci CEIL_2D
        coords = np.array(coords)
        self.dist = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                self.dist[i, j] = np.ceil(np.sqrt(dx * dx + dy * dy))