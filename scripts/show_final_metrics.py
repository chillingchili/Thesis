import csv

for name, p in [
    ("BALL", r"runs\detect\ball_yolo26s\results.csv"),
    ("COURT", r"runs\pose\court_yolo26s\results.csv"),
]:
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    last = rows[-1]
    keys = {k.strip(): k for k in last}
    print(f"--- {name} ({len(rows)} epochs) ---")
    mkeys = [k for k in keys if k.startswith("metrics/")]
    for k in ["epoch"] + mkeys:
        print(f"  {k}: {float(last[keys[k]]):.4f}")
    for mk in mkeys:
        best = max(rows, key=lambda r: float(r[keys[mk]]))
        be = best[keys["epoch"]].strip()
        print(f"  best[{mk}] = {float(best[keys[mk]]):.4f} @ epoch {be}")
        break
