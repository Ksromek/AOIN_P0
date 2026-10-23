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

# opis wartosci bazowych (do stopek)
EA_BASE = ("turniej Tour = 5, OX, Px = 0.7, inwersja Pm = 0.1, inicjalizacja losowa, "
           "pop_size = 100, gen = 100")
SA_BASE = "chłodzenie geometryczne, T0 = 1000, alfa = 0.999, T_min = 0.05"

plt.rcParams["axes.axisbelow"] = True  # siatka pod danymi



# pomocnicze

def read(path):
    df = pd.read_csv(path, sep=";")
    if "tour" in df.columns:
        df["tour"] = pd.to_numeric(df["tour"], errors="coerce")  # ruletka ma "-" zamiast rozmiaru turnieju
    return df


# berlin52_n153_uncorr_06 -> berlin52_n153
def short(instance):
    return "_".join(instance.split("_")[:2])


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()  # odstepy miedzy panelami, zeby tytuly nie nachodzily na opisy osi
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def footnote(fig, text):
    fig.text(0.01, -0.03, text, color=TEXT_SECONDARY, fontsize=8)


# linia avg z +- std
def band(ax, x, avg, std, color, label=None, marker="o"):
    ax.plot(x, avg, color=color, marker=marker, markersize=4, label=label)
    ax.fill_between(x, avg - std, avg + std, color=color, alpha=0.15, linewidth=0)


# wartosci kategoryczne: punkt = avg, +- std
def points(ax, labels, avg, std, color):
    x = np.arange(len(labels))
    ax.errorbar(x, avg, yerr=std, color=color, marker="o", markersize=6, capsize=4, linestyle="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)


def log_ticks(ax, values):
    ax.set_xscale("log")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.minorticks_off()


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
        footnote(fig, f"{short(inst)}; pozostałe parametry: {EA_BASE}.\n"
                      f"Linia kropkowana: wartość z treści ({base}). 10 uruchomień na punkt.")
        save(fig, f"{name}_{short(inst)}")


# EA: pop_size (gen = 100) i gen (pop_size = 100) na jednej osi
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
        footnote(fig, f"{short(inst)}; pozostałe parametry: {EA_BASE}.\n"
                      f"Liczba ocen = pop_size × (gen + 1). 10 uruchomień na punkt.")
        save(fig, f"A4_pop_size_gen_{short(inst)}")


# EA: rozny podzial tego samego budzetu (~10 000 ocen)
def plot_ea_pop_gen_budget(ea):
    data = ea[ea["badanie"] == "POP_GEN"]
    for inst in data["instancja"].unique():
        d = data[data["instancja"] == inst].sort_values("pop_size")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        points(ax, [f"{p}/{g}" for p, g in zip(d["pop_size"], d["gen"])], d["avg"], d["std"], EA_COLOR)
        ax.set_xlabel("pop_size / gen")
        ax.set_ylabel(Y_LABEL)
        footnote(fig, f"{short(inst)}; budżet ~10 000 ocen; pozostałe parametry jak w treści zadania. "
                      f"10 uruchomień na punkt.")
        save(fig, f"A5_pop_gen_staly_budzet_{short(inst)}")


# EA: parametry kategoryczne (selekcja, krzyzowanie, mutacja, inicjalizacja)
def plot_ea_categorical(ea):
    studies = [("SELEKCJA", "selekcja", {"tournament": "turniej", "roulette": "ruletka"}, "selekcja"),
               ("KRZYZOWANIE", "krzyzowanie", {"ox": "OX", "pmx": "PMX", "cx": "CX"}, "krzyżowanie"),
               ("MUTACJA", None, None, "mutacja (operator, Pm)"),
               ("INICJALIZACJA", "init", {"random": "losowa", "greedy": "zachłanna"}, "inicjalizacja")]
    for inst in ea["instancja"].unique():
        fig, axes = plt.subplots(1, len(studies), figsize=(13, 3.8))
        for ax, (study, column, names, xlabel) in zip(axes, studies):
            d = ea[(ea["badanie"] == study) & (ea["instancja"] == inst)]
            if study == "MUTACJA":
                d = d.sort_values(["mutacja", "pm"])
                labels = [f"inw.\n{pm}" if m == "inversion" else f"swap\n{pm}"
                          for m, pm in zip(d["mutacja"], d["pm"])]
            else:
                labels = [names[v] for v in d[column]]
            points(ax, labels, d["avg"], d["std"], EA_COLOR)
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel(Y_LABEL)
        footnote(fig, f"{short(inst)}; pozostałe parametry: {EA_BASE}. Swap: Pm na gen, inwersja: Pm na osobnika. "
                      f"10 uruchomień na punkt.")
        save(fig, f"A6_operatory_{short(inst)}")


# SA: T0, alfa, T_min - os X kategoryczna, pod wartoscia liczba iteracji
def plot_sa_params(sa):
    studies = [("T0", "t0", "T0"), ("ALFA", "alfa", "alfa"), ("TMIN", "t_min", "T_min")]
    for inst in sa["instancja"].unique():
        fig, axes = plt.subplots(1, len(studies), figsize=(13, 3.8))
        for ax, (study, column, xlabel) in zip(axes, studies):
            d = sa[(sa["badanie"] == study) & (sa["instancja"] == inst)].copy()
            d["_x"] = d[column].astype(float)
            d = d.sort_values("_x")
            x = np.arange(len(d))
            band(ax, x, d["avg"].to_numpy(), d["std"].to_numpy(), SA_COLOR)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}\n({e} it.)" for v, e in zip(d["_x"], d["oceny"])], fontsize=8)
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel(Y_LABEL)
        footnote(fig, f"{short(inst)}; pozostałe parametry: {SA_BASE}. 10 uruchomień na punkt.")
        save(fig, f"A7_sa_parametry_{short(inst)}")


def heatmap(ax, table, xlabel, ylabel, fmt="{:.0f}"):
    values = table.to_numpy(dtype=float)
    ax.imshow(values, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f"{c:g}" if isinstance(c, (int, float)) else c for c in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"{r:g}" if isinstance(r, (int, float)) else r for r in table.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    # wartosci w komorkach (kolor tekstu zalezny od tla)
    finite = values[np.isfinite(values)]
    middle = (finite.min() + finite.max()) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if values[i, j] > middle else TEXT)


# EA: srednia avg z siatki dla pary parametrow (pop_size = 100, zeby kazda komorka miala te same warianty)
def plot_ea_interaction(ea, rows, cols, row_label, col_label, name):
    data = ea[(ea["badanie"] == "SIATKA") & (ea["pop_size"] == 100)]
    others = {"pm": "Pm", "tour": "Tour", "px": "Px"}
    averaged = [others[c] for c in others if c not in (rows, cols)]
    for inst in data["instancja"].unique():
        table = data[data["instancja"] == inst].pivot_table(index=rows, columns=cols, values="avg", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        heatmap(ax, table, col_label, row_label)
        footnote(fig, f"{short(inst)}; wartość w komórce: średnia avg z siatki (inwersja, inicjalizacja losowa, "
                      f"pop_size = 100, gen = 100)\nuśredniona po: {', '.join(averaged)}, krzyżowanie OX i PMX.")
        save(fig, f"{name}_{short(inst)}")


# SA przy stalym budzecie: geometryczne (T0 x T_min) i liniowe (avg w funkcji T0)
def plot_sa_budget_grid(sa):
    for inst in sa["instancja"].unique():
        fig, (ax_geo, ax_lin) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 2]})

        geo = sa[(sa["badanie"] == "SIATKA_GEOMETRYCZNE") & (sa["instancja"] == inst)]
        table = geo.pivot_table(index="t_min", columns="t0", values="avg")
        heatmap(ax_geo, table, "T0", "T_min")
        ax_geo.set_title("chłodzenie geometryczne", fontsize=10, color=TEXT)

        lin = sa[(sa["badanie"] == "SIATKA_LINIOWE") & (sa["instancja"] == inst)]
        lin = lin.groupby("t0")["avg"].agg(["mean", "std"]).reset_index()
        band(ax_lin, lin["t0"], lin["mean"], lin["std"].fillna(0), SA_COLOR, "liniowe")
        ax_lin.axhline(geo["avg"].mean(), color=TEXT_SECONDARY, linestyle="--", linewidth=1,
                       label="geometryczne (średnia z mapy)")
        log_ticks(ax_lin, lin["t0"])
        ax_lin.tick_params(axis="x", labelsize=7)
        ax_lin.set_xlabel("T0")
        ax_lin.set_ylabel("ocena G(x, y): avg")
        ax_lin.set_title("chłodzenie liniowe", fontsize=10, color=TEXT)
        ax_lin.legend(loc="lower left", fontsize=8)
        footnote(fig, f"{short(inst)}; budżet ~9900 iteracji (alfa i beta dobrane do T0 i T_min). "
                      f"Mapa: avg z 10 uruchomień. Chłodzenie liniowe: średnia i rozrzut avg po T_min ∈ {{0.01, 0.1, 1}}.")
        save(fig, f"B3_sa_staly_budzet_{short(inst)}")


def candidate_label(row, method):
    if method == "SA":
        cooling = "geom." if row["chlodzenie"] == "geometric" else "lin."
        return f"SA {cooling} T0={float(row['t0']):g} Tmin={float(row['t_min']):g}"
    init = "greedy" if row["init"] == "greedy" else "losowa"
    mutation = "inw." if row["mutacja"] == "inversion" else "swap"
    return (f"EA {init}, Tour={int(row['tour'])}, {row['krzyzowanie'].upper()}, Px={row['px']}, "
            f"{mutation} Pm={row['pm']}, {row['pop_size']}/{row['gen']}")


def plot_validation(ea, sa):
    rows = []
    for method, df in [("EA", ea), ("SA", sa)]:
        for _, r in df[df["badanie"] == "WALIDACJA"].iterrows():
            kind = "SA" if method == "SA" else ("EA greedy" if r["init"] == "greedy" else "EA")
            rows.append((r["instancja"], candidate_label(r, method), kind, r["avg"], r["std"]))
    data = pd.DataFrame(rows, columns=["instancja", "etykieta", "rodzaj", "avg", "std"])
    colors = {"EA": EA_COLOR, "EA greedy": GREEDY_COLOR, "SA": SA_COLOR}
    for inst in data["instancja"].unique():
        d = data[data["instancja"] == inst].sort_values("avg")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        y = np.arange(len(d))
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
        footnote(fig, f"{short(inst)}; seedy 1000-1029 (30 uruchomień na konfigurację).")
        save(fig, f"C1_walidacja_{short(inst)}")


def plot_budget(budget, table1):
    for inst in budget["instancja"].unique():
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for method, color in [("EA", EA_COLOR), ("SA", SA_COLOR)]:
            d = budget[(budget["instancja"] == inst) & (budget["metoda"] == method)].sort_values("oceny")
            band(ax, d["oceny"], d["avg"], d["std"], color, method)
        t = table1[table1["instancja"] == inst].set_index("metoda")
        ax.axhline(t.loc["zachlanny", "avg"], color=TEXT_SECONDARY, linestyle="--", linewidth=1,
                   label="zachłanny (avg)")
        ax.axhline(t.loc["zachlanny", "best"], color=TEXT_SECONDARY, linestyle=":", linewidth=1,
                   label="zachłanny (best)")
        ax.errorbar(t.loc["EA_greedy", "oceny"], t.loc["EA_greedy", "avg"], yerr=t.loc["EA_greedy", "std"],
                    color=GREEDY_COLOR, marker="*", markersize=9, capsize=3, linestyle="none",
                    label="EA greedy")
        log_ticks(ax, [10_000, 30_000, 100_000])
        ax.set_xticklabels(["10k", "30k", "100k"])
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_xlabel("liczba ocen")
        ax.set_ylabel(Y_LABEL)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
        footnote(fig, f"{short(inst)}; EA: pop_size = 50, gen = 200 × mnożnik; SA: alfa = 0.999^(1/mnożnik).\n"
                      f"Mnożnik 1 / 3 / 10, seedy 2000-2009 (10 uruchomień). Zachłanny i EA greedy z Tabeli 1.")
        save(fig, f"E1_budzet_{short(inst)}")


def table1_path():
    p = default_params()
    return RESULTS_DIR / "podsumowanie" / f"{config_tag(p)}__{sa_config_tag(p)}.csv"


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

    if BUDGET_FILE.exists() and table1_path().exists():
        plot_budget(read(BUDGET_FILE), read(table1_path()))
    else:
        print("Pomijam wykres budzetu: brak podsumowania budget.py albo Tabeli 1")

    print(f"Wykresy zapisane w {OUT_DIR}")


if __name__ == "__main__":
    main()