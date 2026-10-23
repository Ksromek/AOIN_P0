import matplotlib

matplotlib.use("Agg")  # zapis do plikow, bez otwierania okien

import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"

# przebiegi dla jednej instancji z kazdej grupy (latwa, srednia, trudna, extra)
PLOT_INSTANCES = [
    "berlin52_n510_uncorr_07",
    "ch150_n447_uncorr-similar-weights_06",
    "a280_n837_bounded-strongly-corr_04",
    "d493_n2460_uncorr-similar-weights_10",
]

# kolory serii (stala kolejnosc) i kolory tekstu/siatki
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.figsize": (9, 5),
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
})


# odczyt CSV
def load_history(path):
    return pd.read_csv(path, sep=";")


# opis konfiguracji na podstawie kolumn z parametrami w CSV
def describe(df):
    p = df.iloc[0]
    if is_ea(df):
        init = {"random": "losowa", "greedy": "zachłanna"}[p["init"]]
        if p["selection"] == "tournament":
            selection = f"turniej (Tour = {p['tournament']})"
        else:
            selection = "ruletka"
        crossover = p["crossover"].upper()
        mutation = {"inversion": "inwersja", "swap": "swap"}[p["mutation"]]
        return (f"EA: inicjalizacja {init}, selekcja {selection}, krzyżowanie {crossover} (Px = {p['px']}), "
                f"mutacja {mutation} (Pm = {p['pm']}), pop_size = {p['pop_size']}, gen = {p['generations']}")
    if p["cooling"] == "geometric":
        cooling = f"geometryczne (alfa = {p['alpha']})"
    else:
        cooling = f"liniowe (beta = {p['beta']})"
    return f"SA: chłodzenie {cooling}, T0 = {p['start_temp']}, T_min = {p['min_temp']}"


def is_ea(df):
    return "nr_pokolenia" in df.columns


def save(fig, name):
    file_path = PLOTS_DIR / f"{name}.png"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)
    plt.close(fig)


# EA: srednia z uruchomien - best / avg / worst w kolejnych pokoleniach
def plot_ea_generations(path):
    df = load_history(path)
    mean = df.groupby("nr_pokolenia")[["najlepsza_ocena", "srednia_ocen", "najgorsza_ocena"]].mean()

    fig, ax = plt.subplots()
    ax.plot(mean.index, mean["najlepsza_ocena"], color=COLORS[0], label="best")
    ax.plot(mean.index, mean["najgorsza_ocena"], color=COLORS[1], label="worst")
    ax.plot(mean.index, mean["srednia_ocen"], color=COLORS[2], label="avg")
    ax.set_xlabel("pokolenie")
    ax.set_ylabel("ocena G(x, y)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)  # nad wykresem, nie zaslania danych
    fig.text(0.01, -0.02, describe(df), color=TEXT_SECONDARY, fontsize=8)
    save(fig, path.stem)


# EA: wszystkie uruchomienia po kolei na jednej osi - best / avg / worst kazdego pokolenia, kolejne uruchomienia jedno za drugim
def plot_ea_runs(path):
    df = load_history(path).sort_values(["uruchomienie", "nr_pokolenia"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    # kazde uruchomienie rysowane osobno, zeby linie nie laczyly konca jednego z poczatkiem nastepnego
    for run, run_df in df.groupby("uruchomienie"):
        first = run == 0
        ax.plot(run_df.index, run_df["najlepsza_ocena"], color=COLORS[0], label="best" if first else None)
        ax.plot(run_df.index, run_df["najgorsza_ocena"], color=COLORS[1], label="worst" if first else None)
        ax.plot(run_df.index, run_df["srednia_ocen"], color=COLORS[2], label="avg" if first else None)

    # granice miedzy uruchomieniami
    starts = df.index[df["nr_pokolenia"] == 0]
    for start in starts[1:]:
        ax.axvline(start - 0.5, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")

    ax.set_xlabel("kolejne pokolenia wszystkich uruchomień")
    ax.set_ylabel("ocena G(x, y)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)  # nad wykresem, nie zaslania danych
    fig.text(0.01, -0.02, describe(df), color=TEXT_SECONDARY, fontsize=8)
    save(fig, f"{path.stem}_uruchomienia")


# SA: najlepsza dotad ocena z uruchomien - najlepsze uruchomienie / srednia / najgorsze uruchomienie oraz temperatura
def plot_sa(path):
    df = load_history(path)
    stats = df.groupby("iteracja")["najlepsza_ocena"].agg(["max", "mean", "min"])
    temp = df.groupby("iteracja")["temperatura"].mean()

    fig, (ax_value, ax_temp) = plt.subplots(2, 1, sharex=True, figsize=(9, 6),
                                            gridspec_kw={"height_ratios": [3, 1]})
    ax_value.plot(stats.index, stats["max"], color=COLORS[0], label="najlepsze uruchomienie")
    ax_value.plot(stats.index, stats["min"], color=COLORS[1], label="najgorsze uruchomienie")
    ax_value.plot(stats.index, stats["mean"], color=COLORS[2], label="średnia z uruchomień")
    ax_value.set_ylabel("ocena G(x, y)")
    ax_value.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)  # nad wykresem, nie zaslania danych

    ax_temp.plot(temp.index, temp.values, color=TEXT_SECONDARY)
    ax_temp.set_xlabel("iteracja")
    ax_temp.set_ylabel("temperatura")

    fig.text(0.01, -0.02, describe(df), color=TEXT_SECONDARY, fontsize=8)
    save(fig, path.stem)


def main():
    csv_paths = sorted(path for path in RESULTS_DIR.glob("*.csv")
                       if any(path.stem.startswith(f"{inst}_") for inst in PLOT_INSTANCES))
    if not csv_paths:
        print(f"Brak plikow CSV dla instancji z PLOT_INSTANCES w {RESULTS_DIR}")
        return

    # EA: dwa wykresy na plik, SA: jeden
    for path in csv_paths:
        if is_ea(load_history(path)):
            plot_ea_generations(path)
            plot_ea_runs(path)
        else:
            plot_sa(path)

    print(f"Wykresy zapisane w {PLOTS_DIR}")


if __name__ == "__main__":
    main()