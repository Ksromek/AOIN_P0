import csv


# jeden plik na instancje, w kazdym wierszu: dane z historii + parametry metody
def save_history_csv(path, columns, params, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([*columns, *params.keys()])
        for row in rows:
            values = [f"{v:.2f}" if isinstance(v, float) else v for v in row]
            writer.writerow([*values, *params.values()])