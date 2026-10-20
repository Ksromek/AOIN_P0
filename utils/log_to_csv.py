import csv

# instancja; nr_uruchomienia; nr_pokolenia; najlepsza; srednia; najgorsza
def save_history_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["instancja", "uruchomienie", "nr_pokolenia",
                         "najlepsza_ocena", "srednia_ocen", "najgorsza_ocena"])
        for instance_name, run, gen, best, avg, worst in rows:
            writer.writerow([instance_name, run, gen, f"{best:.2f}", f"{avg:.2f}", f"{worst:.2f}"])