"""Complete results table: fit each sklearn model on all train, eval Beg3 + Beg4 holdouts."""
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

CLASSES = ["drive", "lob", "topspin"]

def make_clf(kind, seed=42):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                      random_state=seed, n_jobs=-1)
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
        return make_pipeline(StandardScaler(),
                             KNeighborsClassifier(n_neighbors=7, metric="manhattan"))
    raise ValueError(kind)

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"].astype(str)

class LabelWrap:
    """Wrap a pipeline needing int labels (sklearn manhattan argkmin)."""
    def __init__(self, pipe):
        self.pipe = pipe
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.map_ = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([self.map_[v] for v in y])
        self.pipe.fit(X, yi)
        return self
    def predict(self, X):
        pi = self.pipe.predict(X)
        return self.classes_[pi]


def main():
    X, y = load_npz("features/timing_train.npz")
    holdouts = {"Beg3": "features/timing_test.npz", "Beg4": "features/timing_beg4.npz"}
    H = {k: load_npz(v) for k, v in holdouts.items()}

    results = []
    for kind in ["svm", "lr", "rf", "gb", "knn5", "knn7s"]:
        base = make_clf(kind)
        clf = LabelWrap(base) if kind == "knn7s" else base
        clf.fit(X, y)
        # save
        with open(f"models/timing_{kind}.pkl", "wb") as f:
            pickle.dump({"clf": clf, "classes": CLASSES, "kind": kind}, f)
        row = {"model": kind}
        for name, (Xh, yh) in H.items():
            pred = clf.predict(Xh)
            acc = accuracy_score(yh, pred)
            f1 = f1_score(yh, pred, average="macro")
            row[f"{name}_acc"] = acc
            row[f"{name}_f1"] = f1
            if kind in ("svm", "rf", "gb", "knn7s"):
                print(f"\n=== {kind} on {name} (n={len(yh)}) acc={acc:.3f} f1={f1:.3f} ===")
                print(classification_report(yh, pred, labels=CLASSES,
                                            target_names=CLASSES, zero_division=0, digits=3))
        results.append(row)
        print(f"[saved] models/timing_{kind}.pkl  Beg3={row['Beg3_acc']:.3f}  Beg4={row['Beg4_acc']:.3f}")

    print("\n" + "=" * 60)
    print(f"{'model':<10} {'Beg3 acc':>10} {'Beg3 f1':>10} {'Beg4 acc':>10} {'Beg4 f1':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<10} {r['Beg3_acc']:>10.3f} {r['Beg3_f1']:>10.3f} "
              f"{r['Beg4_acc']:>10.3f} {r['Beg4_f1']:>10.3f}")

if __name__ == "__main__":
    main()
