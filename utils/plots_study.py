import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR
from experiments import default_params, config_tag, sa_config_tag
from utils.plots import COLORS, TEXT, TEXT_SECONDARY  # import ustawia tez wspolny styl (rcParams)

SWEEP_DIR = RESULTS_DIR / "sweep"
BUDGET_FILE = RESULTS_DIR / "budzet" / "podsumowanie_budzet.csv"
OUT_DIR = RESULTS_DIR / "wykresy"

Y_LABEL = "ocena G(x, y): avg ± std"
EA_COLOR, SA_COLOR, GREEDY_COLOR = COLORS[0], COLORS[1], COLORS[2]

plt.rcParams["axes.axisbelow"] = True  # siatka pod danymi



# odczyt CSV; kolumny, w ktorych czesc wierszy ma "-"
def read(path):
    df = pd.read_csv(path, sep=";")
    for column in ["tour", "alfa", "beta"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


# berlin52_n153_uncorr_06 -> berlin52_n153
def short(instance):
    return "_".join(instance.split("_")[:2])


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()  # odstepy miedzy panelami, zeby opisy nie nachodzily na siebie
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


# linia = avg, zacieniowany pas = avg +- std
def band(ax, x, avg, std, color, label=None):
    ax.plot(x, avg, color=color, marker="o", markersize=4, label=label)
    ax.fill_between(x, avg - std, avg + std, color=color, alpha=0.15, linewidth=0)


# wartosci kategoryczne: punkt = avg, slupek bledu = +- std
def points(ax, labels, avg, std, color):
    x = np.arange(len(labels))
    ax.errorbar(x, avg, yerr=std, color=color, marker="o", markersize=6, capsize=4, linestyle="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)


# os X w skali logarytmicznej, podpisy tylko przy badanych wartosciach
def log_ticks(ax, values):
    ax.set_xscale("log")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.minorticks_off()


# mapa ciepla z wartoscia w kazdej komorce (bialy tekst na ciemnym tle, czarny na jasnym)
def heatmap(ax, table, xlabel, ylabel):
    values = table.to_numpy(dtype=float)
    ax.imshow(values, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f"{c:g}" for c in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"{r:g}" for r in table.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)

    middle = (np.nanmin(values) + np.nanmax(values)) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):  # pusta komorka (np. T0 = T_min) zostaje bez napisu
                color = "white" if values[i, j] > middle else TEXT
                ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=7, color=color)


# EA: wplyw jednego parametru liczbowego (Pm / Px / Tour); pionowa linia - wartosc z tresci zadania
def plot_ea_param(ea, study, column, xlabel, base, name, log_x=False):
    data = ea[ea["badanie"] == study]
    for inst in data["instancja"].unique():
        d = data[data["instancja"] == inst].sort_values(column)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        band(ax, d[column], d["avg"], d["std"], EA_COLOR)
        ax.axvline(base, color=TEXT_SECONDARY, linestyle=":", linewidth=1)
        if log_x:
            log_ticks(ax, d[column])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(Y_LABEL)
        save(fig, f"{name}_{short(inst)}")


# EA: pop_size (przy gen = 100) i gen (przy pop_size = 100) na jednym wykresie
def plot_ea_pop_gen(ea):
    for inst in ea["instancja"].unique():
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for study, column, color, label in [("POP_SIZE", "pop_size", EA_COLOR, "pop_size (gen = 100)"),
                                            ("GENERATIONS", "gen", SA_COLOR, "gen (pop_size = 100)")]:
            d = ea[(ea["badanie"] == study) & (ea["instancja"] == inst)].sort_values(column)
            band(ax, d[column], d["avg"], d["std"], color, label)
        log_ticks(ax, d[column])
        ax.set_xlabel("wartość parametru")
        ax.set_ylabel(Y_LABEL)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        save(fig, f"A4_pop_size_gen_{short(inst)}")


# EA: rozny podzial tego samego budzetu (~10 000 ocen) na pop_size i gen
def plot_ea_pop_gen_budget(ea):
    data = ea[ea["badanie"] == "POP_GEN"]
    for inst in data["instancja"].unique():
        d = data[data["instancja"] == inst].sort_values("pop_size")
        labels = [f"{p}/{g}" for p, g in zip(d["pop_size"], d["gen"])]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        points(ax, labels, d["avg"], d["std"], EA_COLOR)
        ax.set_xlabel("pop_size / gen")
        ax.set_ylabel(Y_LABEL)
        save(fig, f"A5_pop_gen_staly_budzet_{short(inst)}")


# EA: selekcja, krzyzowanie, mutacja i inicjalizacja - cztery panele obok siebie
def plot_ea_categorical(ea):
    names = {"tournament": "turniej", "roulette": "ruletka", "ox": "OX", "pmx": "PMX", "cx": "CX",
             "random": "losowa", "greedy": "zachłanna"}
    panels = [("SELEKCJA", "selekcja", "selekcja"),
              ("KRZYZOWANIE", "krzyzowanie", "krzyżowanie"),
              ("MUTACJA", "mutacja", "mutacja (operator, Pm)"),
              ("INICJALIZACJA", "init", "inicjalizacja")]
    for inst in ea["instancja"].unique():
        fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.8))
        for ax, (study, column, xlabel) in zip(axes, panels):
            d = ea[(ea["badanie"] == study) & (ea["instancja"] == inst)]
            if study == "MUTACJA":
                # kilka wartosci Pm dla swapa - podpis: operator + Pm
                d = d.sort_values(["mutacja", "pm"])
                labels = [f"inw.\n{pm}" if m == "inversion" else f"swap\n{pm}"
                          for m, pm in zip(d["mutacja"], d["pm"])]
            else:
                labels = [names[v] for v in d[column]]
            points(ax, labels, d["avg"], d["std"], EA_COLOR)
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel(Y_LABEL)
        save(fig, f"A6_operatory_{short(inst)}")


# SA: T0, alfa i T_min - trzy panele; pod kazda wartoscia liczba iteracji, ktora z niej wynika
def plot_sa_params(sa):
    panels = [("T0", "t0", "T0"), ("ALFA", "alfa", "alfa"), ("TMIN", "t_min", "T_min")]
    for inst in sa["instancja"].unique():
        fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.8))
        for ax, (study, column, xlabel) in zip(axes, panels):
            d = sa[(sa["badanie"] == study) & (sa["instancja"] == inst)].sort_values(column)
            x = np.arange(len(d))  # rowne odstepy miedzy wartosciami
            band(ax, x, d["avg"].to_numpy(), d["std"].to_numpy(), SA_COLOR)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}\n({e} it.)" for v, e in zip(d[column], d["oceny"])], fontsize=8)
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel(Y_LABEL)
        save(fig, f"A7_sa_parametry_{short(inst)}")



# EA: srednie avg dla pary parametrow; tylko pop_size = 100, zeby kazda komorka miala te same warianty
def plot_ea_interaction(ea, rows, cols, row_label, col_label, name):
    data = ea[(ea["badanie"] == "SIATKA") & (ea["pop_size"] == 100)]
    for inst in data["instancja"].unique():
        d = data[data["instancja"] == inst]
        table = d.pivot_table(index=rows, columns=cols, values="avg", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        heatmap(ax, table, col_label, row_label)
        save(fig, f"{name}_{short(inst)}")


# SA przy stalym budzecie: geometryczne (mapa T0 x T_min) i liniowe (avg w funkcji T0)
def plot_sa_budget_grid(sa):
    for inst in sa["instancja"].unique():
        fig, (ax_geo, ax_lin) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 2]})

        geo = sa[(sa["badanie"] == "SIATKA_GEOMETRYCZNE") & (sa["instancja"] == inst)]
        heatmap(ax_geo, geo.pivot_table(index="t_min", columns="t0", values="avg"), "T0", "T_min")
        ax_geo.set_title("chłodzenie geometryczne", fontsize=10, color=TEXT)

        # liniowe: dla kazdego T0 srednia (i std) z kilku wartosci T_min
        lin = sa[(sa["badanie"] == "SIATKA_LINIOWE") & (sa["instancja"] == inst)]
        lin = lin.groupby("t0")["avg"].agg(["mean", "std"]).reset_index()
        band(ax_lin, lin["t0"], lin["mean"], lin["std"].fillna(0), SA_COLOR, "liniowe")
        ax_lin.axhline(geo["avg"].mean(), color=TEXT_SECONDARY, linestyle="--", linewidth=1,
                       label="geometryczne (średnia z siatki)")
        log_ticks(ax_lin, lin["t0"])
        ax_lin.tick_params(axis="x", labelsize=7)
        ax_lin.set_xlabel("T0")
        ax_lin.set_ylabel("ocena G(x, y): avg")
        ax_lin.set_title("chłodzenie liniowe", fontsize=10, color=TEXT)
        ax_lin.legend(loc="lower left", fontsize=8)
        save(fig, f"B3_sa_staly_budzet_{short(inst)}")


# opis kandydata na osi Y, np. "EA losowa, Tour=10, OX, Px=0.7, inw. Pm=0.7, 50/200"
def ea_label(r):
    init = "greedy" if r["init"] == "greedy" else "losowa"
    mutation = "inw." if r["mutacja"] == "inversion" else "swap"
    return (f"EA {init}, Tour={int(r['tour'])}, {r['krzyzowanie'].upper()}, Px={r['px']}, "
            f"{mutation} Pm={r['pm']}, {r['pop_size']}/{r['gen']}")


def sa_label(r):
    cooling = "geom." if r["chlodzenie"] == "geometric" else "lin."
    return f"SA {cooling} T0={r['t0']:g} Tmin={r['t_min']:g}"


# wszyscy kandydaci na jednym wykresie, posortowani wg avg; kolor = rodzaj metody
def plot_validation(ea, sa):
    ea_val = ea[ea["badanie"] == "WALIDACJA"].copy()
    ea_val["etykieta"] = ea_val.apply(ea_label, axis=1)
    ea_val["rodzaj"] = np.where(ea_val["init"] == "greedy", "EA greedy", "EA")

    sa_val = sa[sa["badanie"] == "WALIDACJA"].copy()
    sa_val["etykieta"] = sa_val.apply(sa_label, axis=1)
    sa_val["rodzaj"] = "SA"

    columns = ["instancja", "etykieta", "rodzaj", "avg", "std"]
    data = pd.concat([ea_val[columns], sa_val[columns]])
    colors = {"EA": EA_COLOR, "EA greedy": GREEDY_COLOR, "SA": SA_COLOR}

    for inst in data["instancja"].unique():
        d = data[data["instancja"] == inst].sort_values("avg")
        y = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for kind, color in colors.items():
            mask = (d["rodzaj"] == kind).to_numpy()
            ax.errorbar(d["avg"][mask], y[mask], xerr=d["std"][mask], color=color, marker="o",
                        markersize=6, capsize=3, linestyle="none", label=kind)
        ax.set_yticks(y)
        ax.set_yticklabels(d["etykieta"])
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel(Y_LABEL)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
        save(fig, f"C1_walidacja_{short(inst)}")


# EA i SA w funkcji budzetu; dla porownania zachlanny (linie poziome)
def plot_budget(budget, table1):
    for inst in budget["instancja"].unique():
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for method, color, label in [("EA", EA_COLOR, "EA"), ("EA_greedy", GREEDY_COLOR, "EA greedy"),
                                     ("SA", SA_COLOR, "SA")]:
            d = budget[(budget["instancja"] == inst) & (budget["metoda"] == method)].sort_values("oceny")
            band(ax, d["oceny"], d["avg"], d["std"], color, label)

        t = table1[table1["instancja"] == inst].set_index("metoda")  # wiersze Tabeli 1 dla tej instancji
        ax.axhline(t.loc["zachlanny", "avg"], color=TEXT_SECONDARY, linestyle="--", linewidth=1,
                   label="zachłanny (avg)")
        ax.axhline(t.loc["zachlanny", "best"], color=TEXT_SECONDARY, linestyle=":", linewidth=1,
                   label="zachłanny (best)")

        log_ticks(ax, [10_000, 30_000, 100_000])
        ax.set_xticklabels(["10k", "30k", "100k"])
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_xlabel("liczba ocen")
        ax.set_ylabel(Y_LABEL)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
        save(fig, f"E1_budzet_{short(inst)}")


def main():
    ea = read(SWEEP_DIR / "podsumowanie_ea.csv")
    sa = read(SWEEP_DIR / "podsumowanie_sa.csv")

    plot_ea_param(ea, "PM", "pm", "Pm (inwersja, na osobnika)", 0.1, "A1_pm")
    plot_ea_param(ea, "PX", "px", "Px", 0.7, "A2_px")
    plot_ea_param(ea, "TOUR", "tour", "Tour", 5, "A3_tour", log_x=True)
    plot_ea_pop_gen(ea)
    plot_ea_pop_gen_budget(ea)
    plot_ea_categorical(ea)
    plot_sa_params(sa)

    plot_ea_interaction(ea, "pm", "tour", "Pm", "Tour", "B1_pm_x_tour")
    plot_ea_interaction(ea, "pm", "px", "Pm", "Px", "B2_pm_x_px")
    plot_sa_budget_grid(sa)

    plot_validation(ea, sa)

    # Tabela 1 (z experiments.py) - stad zachlanny na wykresie budzetu
    p = default_params()
    table1_file = RESULTS_DIR / "podsumowanie" / f"{config_tag(p)}__{sa_config_tag(p)}.csv"
    if BUDGET_FILE.exists() and table1_file.exists():
        plot_budget(read(BUDGET_FILE), read(table1_file))
    else:
        print("Pomijam wykres budzetu: brak podsumowania budget.py albo Tabeli 1")

    print(f"Wykresy zapisane w {OUT_DIR}")


if __name__ == "__main__":
    main()