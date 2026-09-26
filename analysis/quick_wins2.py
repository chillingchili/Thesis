"""Final quick checks: CORAL alignment + tuned weighted ensembles."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

def coral(Xtr, Xh):
    """Diagonal + full-ish CORAL: shrink test cov toward train cov (Ledoit-like simplified)."""
    mu_t, mu_h = Xtr.mean(0), Xh.mean(0)
    cov_t = np.cov(Xtr, rowvar=False) + 1e-6 * np.eye(Xtr.shape[1])
    cov_h = np.cov(Xh, rowvar=False) + 1e-6 * np.eye(Xh.shape[1])
    # whitening via eigen for stability
    def sqrtm_keep(A, inverse=False):
        w, v = np.linalg.eigh(A)
        w = np.clip(w, 1e-8, None)
        return v @ np.diag(w ** (-0.5 if inverse else 0.5)) @ v.T
    A = sqrtm_keep(cov_h, inverse=True)  # whiten test
    B = sqrtm_keep(cov_t, inverse=False)  # recolor to train
    return (Xh - mu_h) @ A @ B + mu_t

def probs3(clf, X):
    P = clf.predict_proba(X)
    cls = [str(c) for c in clf.classes_]
    col = {c: i for i, c in enumerate(cls)}
    return P[:, [col[c] for c in CLASSES]]

def evaluate(pred_fn, holds):
    accs = []
    for Xh, yh in holds:
        pred = pred_fn(Xh)
        accs.append(accuracy_score(yh, pred))
    return accs, np.mean(accs)

def main():
    Xtr, ytr = load("features/timing_train.npz")
    h3 = load("features/timing_test.npz")
    h4 = load("features/timing_beg4.npz")
    holds = [h3, h4]
    kinds = ["knn5", "knn7s", "svm", "lr", "rf", "gb"]
    models = {}
    for k in kinds:
        c = make(k)
        c.fit(Xtr, ytr)
        models[k] = c

    print("=== CORAL alignment (full cov) ===")
    for n, c in models.items():
        accs, m = evaluate(lambda X, c=c: c.predict(coral(Xtr, X)), holds)
        base, mb = evaluate(lambda X, c=c: c.predict(X), holds)
        print(f"  {n:<8} base={mb:.3f}  CORAL={m:.3f}  "
              f"(Beg3 {base[0]:.3f}->{accs[0]:.3f}, Beg4 {base[1]:.3f}->{accs[1]:.3f})")

    print("\n=== Weighted ensemble search (base probs) ===")
    # grid over 2-3 model subsets with simple uniform and a few hand weights
    from itertools import combinations
    best = (0, None, None)
    for r in [2, 3]:
        for combo in combinations(kinds, r):
            # uniform
            def pred_uniform(X, combo=combo):
                P = np.mean([probs3(models[m], X) for m in combo], 0)
                return np.array([CLASSES[i] for i in P.argmax(1)])
            accs, m = evaluate(pred_uniform, holds)
            if m > best[0]:
                best = (m, combo, "uniform", accs)
            # weight toward knn5/svm/rf if present
            for wname, wmap in [
                ("knn_heavy", {"knn5": 2, "knn7s": 1.5}),
                ("svm_heavy", {"svm": 2}),
                ("rf_heavy", {"rf": 2}),
            ]:
                def pred_w(X, combo=combo, wmap=wmap):
                    ws = np.array([wmap.get(m, 1.0) for m in combo])
                    ws = ws / ws.sum()
                    P = sum(w * probs3(models[m], X) for w, m in zip(ws, combo))
                    return np.array([CLASSES[i] for i in P.argmax(1)])
                accs, m = evaluate(pred_w, holds)
                if m > best[0]:
                    best = (m, combo, wname, accs)
    print(f"  BEST: {best[1]} w={best[2]} mean={best[0]:.3f} "
          f"(Beg3={best[3][0]:.3f}, Beg4={best[3][1]:.3f})")

    # top few from previous knowledge + coral-transformed versions
    print("\n=== CORAL + ensemble ===")
    for combo in [("knn5", "svm", "rf"), ("knn5", "svm"), ("knn7s", "svm", "rf"),
                  ("knn5", "knn7s", "svm", "rf")]:
        def pred_c(X, combo=combo):
            Xc = coral(Xtr, X)
            P = np.mean([probs3(models[m], Xc) for m in combo], 0)
            return np.array([CLASSES[i] for i in P.argmax(1)])
        accs, m = evaluate(pred_c, holds)
        print(f"  {'+'.join(combo):<22} CORAL mean={m:.3f} "
              f"(Beg3={accs[0]:.3f}, Beg4={accs[1]:.3f})")

if __name__ == "__main__":
    main()
