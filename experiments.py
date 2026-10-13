import time

import numpy as np

from ttp import TTPInstance
from algorithms import random_search, greedy

INSTANCES = [
    "data/instances/berlin52_n51_uncorr_09.ttp",
    "data/instances/berlin52_n51_bounded-strongly-corr_10.ttp",
    "data/instances/berlin52_n153_bounded-strongly-corr_07.ttp",
    "data/instances/berlin52_n255_bounded-strongly-corr_07.ttp",
    "data/instances/berlin52_n510_uncorr-similar-weights_09.ttp",
]
N_RANDOM = 10_000
SEED = 42


def summary(name, values, elapsed):
    print(f"  {name:<10} best={values.max():12.1f}  worst={values.min():12.1f}  "
          f"avg={values.mean():12.1f}  std={values.std():10.1f}  "
          f"[{len(values)} ocen, {elapsed:.2f} s]")


def main():
    for path in INSTANCES:
        rng = np.random.default_rng(SEED)
        inst = TTPInstance(path)
        print(f"\n{path}  (n={inst.n_cities}, m={inst.n_items})")

        t0 = time.time()
        _, _, values = random_search(inst, N_RANDOM, rng)
        summary("losowy", values, time.time() - t0)

        t0 = time.time()
        _, _, values = greedy(inst)
        summary("zachlanny", values, time.time() - t0)


if __name__ == "__main__":
    main()