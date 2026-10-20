import time

import numpy as np

from config import INSTANCES, N_RANDOM, SEED
from ttp import TTPInstance
from algorithms import random_search, greedy


def summary(name, values, elapsed, n_evals):
    print(f"  {name:<10} best={values.max():12.1f}  worst={values.min():12.1f}  "
          f"avg={values.mean():12.1f}  std={values.std():10.1f}  "
          f"[{n_evals} ocen, {elapsed:.3f} s]")


def main():
    for path in INSTANCES:
        rng = np.random.default_rng(SEED)
        inst = TTPInstance(path)
        print(f"\n{path}  (n={inst.n_cities}, m={inst.n_items})")

        t0 = time.perf_counter()
        _, _, values, best_history = random_search(inst, N_RANDOM, rng)
        summary("losowy", values, time.perf_counter() - t0, N_RANDOM)

        t0 = time.perf_counter()
        _, _, values = greedy(inst)
        summary("zachlanny", values, time.perf_counter() - t0, inst.n_cities)


if __name__ == "__main__":
    main()