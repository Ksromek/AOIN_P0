import matplotlib

matplotlib.use("Agg")  # zapis do plikow, bez otwierania okien

import matplotlib.pyplot as plt
import pandas as pd

from config import INSTANCES, RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / "plots"

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


# odczyt CSV z experiments.py
def load_history(path):
    return pd.read_csv(path, sep=";")


# podzial nazwy pliku na nazwe instancji i opis konfiguracji
def split_name(path):
    for instance in INSTANCES:
        prefix = instance.stem + "_"
        if path.stem.startswith(prefix):
            return instance.stem, path.stem[len(prefix):]
    return path.stem, ""


def is_ea(df):
    return "nr_pokolenia" in df.columns


def save(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{name}.png")
    plt.close(fig)


# EA: srednia z uruchomien - best / avg / worst w kolejnych pokoleniach (jak Rysunek 5 w tresci)
def plot_ea(path):
    df = load_history(path)
    instance, tag = split_name(path)
    n_runs = df["uruchomienie"].nunique()
    mean = df.groupby("nr_pokolenia")[["najlepsza_ocena", "srednia_ocen", "najgorsza_ocena"]].mean()

    fig, ax = plt.subplots()
    ax.plot(mean.index, mean["najlepsza_ocena"], color=COLORS[0], label="best")
    ax.plot(mean.index, mean["najgorsza_ocena"], color=COLORS[1], label="worst")
    ax.plot(mean.index, mean["srednia_ocen"], color=COLORS[2], label="avg")
    ax.set_title(f"EA - {instance}\nśrednia z {n_runs} uruchomień")
    ax.set_xlabel("pokolenie")
    ax.set_ylabel("ocena G(x, y)")
    ax.legend(loc="lower right")
    fig.text(0.01, -0.02, tag, color=TEXT_SECONDARY, fontsize=8)
    save(fig, path.stem)


# SA: srednia z uruchomien - biezaca i najlepsza ocena (gora) oraz temperatura (dol)
def plot_sa(path):
    df = load_history(path)
    instance, tag = split_name(path)
    n_runs = df["uruchomienie"].nunique()
    mean = df.groupby("iteracja")[["biezaca_ocena", "najlepsza_ocena", "temperatura"]].mean()

    fig, (ax_value, ax_temp) = plt.subplots(2, 1, sharex=True, figsize=(9, 6),
                                            gridspec_kw={"height_ratios": [3, 1]})
    ax_value.plot(mean.index, mean["biezaca_ocena"], color=COLORS[1], label="bieżąca")
    ax_value.plot(mean.index, mean["najlepsza_ocena"], color=COLORS[0], label="najlepsza")
    ax_value.set_title(f"SA - {instance}\nśrednia z {n_runs} uruchomień")
    ax_value.set_ylabel("ocena G(x, y)")
    ax_value.legend(loc="lower right")

    ax_temp.plot(mean.index, mean["temperatura"], color=COLORS[2])
    ax_temp.set_xlabel("iteracja")
    ax_temp.set_ylabel("temperatura")

    fig.text(0.01, -0.02, tag, color=TEXT_SECONDARY, fontsize=8)
    save(fig, path.stem)


# box-plot w czasie: rozklad najlepszej oceny z uruchomien w wybranych momentach przebiegu
def plot_box(path, step_column, steps, xlabel, title):
    df = load_history(path)
    instance, tag = split_name(path)
    n_runs = df["uruchomienie"].nunique()

    data = [df.loc[df[step_column] == s, "najlepsza_ocena"].values for s in steps]

    fig, ax = plt.subplots()
    ax.boxplot(data, widths=0.6, patch_artist=True,
               boxprops={"facecolor": COLORS[0], "edgecolor": TEXT_SECONDARY},
               medianprops={"color": TEXT, "linewidth": 1.5},
               whiskerprops={"color": TEXT_SECONDARY}, capprops={"color": TEXT_SECONDARY},
               flierprops={"marker": "D", "markersize": 4, "markerfacecolor": TEXT_SECONDARY,
                           "markeredgecolor": TEXT_SECONDARY})
    ax.set_xticks(range(1, len(steps) + 1), steps)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{title} - {instance}\nnajlepsza ocena z {n_runs} uruchomień")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ocena G(x, y)")
    fig.text(0.01, -0.02, tag, color=TEXT_SECONDARY, fontsize=8)
    save(fig, f"{path.stem}_box")


# EA: pudelko co 10 pokolen
def plot_ea_box(path):
    generations = sorted(load_history(path)["nr_pokolenia"].unique())
    steps = [g for g in generations if g % 10 == 0]
    plot_box(path, "nr_pokolenia", steps, "pokolenie", "EA")


# SA: pudelko co 1000 iteracji + ostatnia iteracja
def plot_sa_box(path):
    iterations = sorted(load_history(path)["iteracja"].unique())
    steps = [it for it in iterations if it % 1000 == 0]
    if iterations[-1] not in steps:
        steps.append(iterations[-1])
    plot_box(path, "iteracja", steps, "iteracja", "SA")


def main():
    csv_paths = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"Brak plikow CSV w {RESULTS_DIR}")
        return

    # dla kazdego pliku: wykres przebiegu (srednia z uruchomien) i box-plot w czasie
    for path in csv_paths:
        if is_ea(load_history(path)):
            plot_ea(path)
            plot_ea_box(path)
        else:
            plot_sa(path)
            plot_sa_box(path)

    print(f"Wykresy zapisane w {PLOTS_DIR}")


if __name__ == "__main__":
    main()

