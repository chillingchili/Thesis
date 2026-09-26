"""Quick wins: TTA (moment/mean matching) + probability ensembles."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

CLASSES = ["drive", "lob", "topspin"]

def load(p):
    d = np.load(p, allow_pickle=True)
    return d["X"].astype(np.float64), d["y"].astype(str)

def make(kind, seed=42):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=8,
                                      min_samples_leaf=5, random_state=seed, n_jobs=-1)
    if kind == "gb":
        return GradientBoostingClassifier(random_state=seed)
    if kind == "svm":
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=1.0, probability=True, random_state=seed))
    if kind == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=seed))
    if kind == "knn5":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    if kind == "knn7s":
        class W:
            def __init__(self):
                self.p = make_pipeline(StandardScaler(),
                                       KNeighborsClassifier(n_neighbors=7, metric="manhattan"))
            def fit(self, X, y):
                self.classes_ = np.unique(y)
                self.m = {c: i for i, c in enumerate(self.classes_)}
                self.p.fit(X, [self.m[v] for v in y])
                return self
            def predict(self, X):
                return self.classes_[self.p.predict(X)]
            def predict_proba(self, X):
                return self.p.predict_proba(X)
        return W()
    raise ValueError(kind)

def align_mean(Xtr, Xh):
    return Xh - Xh.mean(0) + Xtr.mean(0)

def align_moment(Xtr, Xh):
    return (Xh - Xh.mean(0)) / (Xh.std(0) + 1e-8) * Xtr.std(0) + Xtr.mean(0)

def acc(clf, X, y):
    return accuracy_score(y, clf.predict(X))

def probs3(clf, X):
    P = clf.predict_proba(X)
    # map to CLASSES order; handle int or str classes
    cls = [str(c) for c in clf.classes_]
    col = {c: i for i, c in enumerate(cls)}
    return P[:, [col[c] for c in CLASSES]]

def main(use_enriched=False):
    if use_enriched:
        Xtr, ytr = load("features/timing_enriched_train.npz")
        holds = {"Beg3": load("features/timing_enriched_beg3.npz"),
                 "Beg4": load("features/timing_enriched_beg4.npz")}
        tag = "[enriched] "
    else:
        Xtr, ytr = load("features/timing_train.npz")
        holds = {"Beg3": load("features/timing_test.npz"), "Beg4": load("features/timing_beg4.npz")}
        tag = ""

    kinds = ["knn5", "knn7s", "svm", "lr", "rf", "gb"]
    models = {}
    for k in kinds:
        c = make(k)
        c.fit(Xtr, ytr)
        models[k] = c

    print("=" * 72)
    print(f"{tag}PART A: TTA — moment matching (test -> train mean/std)")
    print("=" * 72)
    print(f"{'model':<10} {'B3 base':>8} {'B3 TTA':>8} {'B4 base':>8} {'B4 TTA':>8} "
          f"{'mean base':>10} {'mean TTA':>9}")
    tta_best = {}
    for n, c in models.items():
        r = {}
        for h, (Xh, yh) in holds.items():
            b = acc(c, Xh, yh)
            t = acc(c, align_moment(Xtr, Xh), yh)
            r[h] = (b, t)
        mb = (r["Beg3"][0] + r["Beg4"][0]) / 2
        mt = (r["Beg3"][1] + r["Beg4"][1]) / 2
        best_mode = "moment" if mt > mb else "base"
        tta_best[n] = (best_mode, max(mb, mt))
        print(f"{n:<10} {r['Beg3'][0]:>8.3f} {r['Beg3'][1]:>8.3f} "
              f"{r['Beg4'][0]:>8.3f} {r['Beg4'][1]:>8.3f} {mb:>10.3f} {mt:>9.3f}")

    print("\n" + "=" * 72)
    print(f"{tag}PART A2: TTA — mean matching only")
    print("=" * 72)
    for n, c in models.items():
        accs_b, accs_t = [], []
        for h, (Xh, yh) in holds.items():
            accs_b.append(acc(c, Xh, yh))
            accs_t.append(acc(c, align_mean(Xtr, Xh), yh))
        print(f"{n:<10} base={np.mean(accs_b):.3f}  meanMM={np.mean(accs_t):.3f}")

    print("\n" + "=" * 72)
    print(f"{tag}PART B: Ensembles (base TTA)")
    print("=" * 72)
    ens = {
        "knn5+svm": ["knn5", "svm"],
        "knn5+svm+lr": ["knn5", "svm", "lr"],
        "knn5+svm+rf": ["knn5", "svm", "rf"],
        "svm+rf+gb": ["svm", "rf", "gb"],
        "knn5+rf": ["knn5", "rf"],
        "all6": kinds,
        "bestTTA_mix": None,  # filled below: per-model TTA choice then average
    }

    for ename, members in ens.items():
        if members is None:
            # mixed TTA: apply each model's better mode
            accs = []
            for h, (Xh, yh) in holds.items():
                P = []
                for m in kinds:
                    mode = tta_best[m][0]
                    Xa = align_moment(Xtr, Xh) if mode == "moment" else Xh
                    P.append(probs3(models[m], Xa))
                pred = np.array([CLASSES[i] for i in np.mean(P, 0).argmax(1)])
                accs.append(accuracy_score(yh, pred))
            print(f"{ename:<16} Beg3={accs[0]:.3f} Beg4={accs[1]:.3f} mean={np.mean(accs):.3f}")
            continue
        accs = []
        for h, (Xh, yh) in holds.items():
            P = np.mean([probs3(models[m], Xh) for m in members], 0)
            pred = np.array([CLASSES[i] for i in P.argmax(1)])
            accs.append(accuracy_score(yh, pred))
        print(f"{ename:<16} Beg3={accs[0]:.3f} Beg4={accs[1]:.3f} mean={np.mean(accs):.3f}")

    print("\n" + "=" * 72)
    print(f"{tag}PART C: Ensembles + moment-matching TTA")
    print("=" * 72)
    for ename, members in ens.items():
        if members is None:
            continue
        accs = []
        for h, (Xh, yh) in holds.items():
            Xa = align_moment(Xtr, Xh)
            P = np.mean([probs3(models[m], Xa) for m in members], 0)
            pred = np.array([CLASSES[i] for i in P.argmax(1)])
            accs.append(accuracy_score(yh, pred))
        print(f"{ename:<16} Beg3={accs[0]:.3f} Beg4={accs[1]:.3f} mean={np.mean(accs):.3f}")

    # single best per model summary
    print("\nBest single-model means (base vs TTA):")
    for n, c in models.items():
        b = np.mean([acc(c, Xh, yh) for Xh, yh in holds.values()])
        t = np.mean([acc(c, align_moment(Xtr, Xh), yh) for Xh, yh in holds.values()])
        print(f"  {n:<8} base={b:.3f} TTA={t:.3f} best={max(b,t):.3f}")

if __name__ == "__main__":
    import sys
    main(use_enriched="--enriched" in sys.argv)
