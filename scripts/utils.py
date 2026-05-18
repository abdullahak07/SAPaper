"""utils.py — shared data loading, models, evaluation helpers."""
import time, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def load_config(path="configs/config.json"):
    with open(path) as f:
        return json.load(f)


# ── Data loaders ───────────────────────────────────────────────────────────
def load_armd(cfg, forget_n=500, seed=42):
    retain = pd.read_csv(cfg["armd"]["retain_csv"])
    test   = pd.read_csv(cfg["armd"]["test_csv"])
    forget_pool = pd.read_csv(cfg["armd"]["forget_csv"])

    with open(cfg["armd"]["feature_cols_txt"]) as f:
        fc = [l.strip() for l in f if l.strip()]

    def clean(df):
        return (df[fc]
                .replace("Null", np.nan)
                .replace("NULL", np.nan)
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0).values)

    # Scale forget set size
    if forget_n <= len(forget_pool):
        forget = forget_pool.sample(n=forget_n, random_state=seed)
    else:
        extra  = retain.sample(n=forget_n - len(forget_pool), random_state=seed)
        forget = pd.concat([forget_pool, extra])
        retain = retain.drop(extra.index)

    return (clean(retain), retain["target"].values,
            clean(test),   test["target"].values,
            clean(forget), forget["target"].values)


def load_patric(cfg, forget_n=500, seed=42):
    df = pd.read_csv(cfg["patric"]["raw_csv"])
    df["target"] = (df["resistant_phenotype"] == "Resistant").astype(int)

    cat_cols = ["antibiotic", "species", "laboratory_typing_method",
                "host_name", "isolation_country"]
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    df["measurement_value"] = np.log1p(
        pd.to_numeric(df["measurement_value"], errors="coerce").fillna(0))

    fc = cat_cols + ["measurement_value"]
    df = df[fc + ["target"]].dropna().reset_index(drop=True)

    # Stratified forget set
    n_each = forget_n // 2
    fg0 = df[df.target == 0].sample(n_each, random_state=seed)
    fg1 = df[df.target == 1].sample(n_each, random_state=seed)
    forget = pd.concat([fg0, fg1])
    remain = df.drop(forget.index).reset_index(drop=True)

    tr_i, te_i = train_test_split(
        remain.index, test_size=0.2, random_state=seed,
        stratify=remain["target"])

    return (remain.loc[tr_i, fc].values, remain.loc[tr_i, "target"].values,
            remain.loc[te_i, fc].values, remain.loc[te_i, "target"].values,
            forget[fc].values, forget["target"].values)


# ── Model factory ──────────────────────────────────────────────────────────
def get_model(model_type, cfg, seed=42):
    n = cfg["n_estimators"]
    if model_type == "RF":
        p = cfg["rf_params"]
        return RandomForestClassifier(
            n_estimators=n, random_state=seed, **p)
    elif model_type == "XGB":
        p = cfg["xgb_params"]
        return XGBClassifier(
            n_estimators=n, random_state=seed, **p)
    raise ValueError(f"Unknown model type: {model_type}")


# ── Safe predict_proba (handles single-class edge cases) ──────────────────
def safe_proba(model, X):
    p = model.predict_proba(X)
    if p.shape[1] == 1:
        return np.column_stack([1 - p[:, 0], p[:, 0]])
    return p


# ── MIA gap ───────────────────────────────────────────────────────────────
def mia_gap(model, X_forget, X_test):
    c_f = safe_proba(model, X_forget)[:, 1].mean()
    c_t = safe_proba(model, X_test)[:, 1].mean()
    return abs(c_f - c_t)


# ── Evaluate ──────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, X_forget, time_sec, original_acc):
    p   = safe_proba(model, X_test)
    acc = accuracy_score(y_test, (p[:, 1] >= 0.5).astype(int))
    auc = roc_auc_score(y_test, p[:, 1])
    gap = mia_gap(model, X_forget, X_test)
    return dict(accuracy=acc, auc=auc, mia_gap=gap,
                time_sec=time_sec, acc_drop=original_acc - acc)


# ── SISA ensemble ─────────────────────────────────────────────────────────
class SISAEnsemble:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([safe_proba(m, X) for m in self.models], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── Unlearning methods ────────────────────────────────────────────────────
def method_full_retrain(model_type, cfg, X_r, y_r, X_t, y_t,
                         X_f, seed, orig_acc):
    m = get_model(model_type, cfg, seed)
    t0 = time.time()
    m.fit(X_r, y_r)
    return evaluate(m, X_t, y_t, X_f, time.time() - t0, orig_acc)


def method_sisa(model_type, cfg, X_r, y_r, X_t, y_t,
                X_f, seed, orig_acc, k=5):
    n   = len(X_r)
    sz  = n // k
    n_s = cfg["n_estimators"]

    # Train all shards (for ensemble prediction quality)
    subs = []
    for i in range(k):
        sm = get_model(model_type, cfg, seed + i)
        sm.n_estimators = n_s
        sm.fit(X_r[i*sz:(i+1)*sz], y_r[i*sz:(i+1)*sz])
        subs.append(sm)

    # Time = retrain one shard only (best-case deletion scenario)
    t0 = time.time()
    s0 = get_model(model_type, cfg, seed)
    s0.n_estimators = n_s
    s0.fit(X_r[:sz], y_r[:sz])
    sisa_t = time.time() - t0
    subs[0] = s0

    ens = SISAEnsemble(subs)
    return evaluate(ens, X_t, y_t, X_f, sisa_t, orig_acc)


def method_label_flip(model_type, cfg, X_r, y_r, X_t, y_t,
                       X_f, y_f, seed, orig_acc):
    X_all = np.vstack([X_r, X_f])
    y_all = np.concatenate([y_r, 1 - y_f])
    # Ensure both classes present
    if len(np.unique(y_all)) < 2:
        y_all[-1] = 1 - y_all[-1]
    m = get_model(model_type, cfg, seed)
    t0 = time.time()
    m.fit(X_all, y_all)
    return evaluate(m, X_t, y_t, X_f, time.time() - t0, orig_acc)


def method_influence_rewt(model_type, cfg, X_r, y_r, X_t, y_t,
                           X_f, y_f, seed, orig_acc):
    X_all = np.vstack([X_r, X_f])
    y_all = np.concatenate([y_r, y_f])
    w = np.ones(len(y_all))
    w[len(y_r):] = 1e-6
    m = get_model(model_type, cfg, seed)
    t0 = time.time()
    m.fit(X_all, y_all, sample_weight=w)
    return evaluate(m, X_t, y_t, X_f, time.time() - t0, orig_acc)


def method_tree_pruning(model_type, cfg, X_r, y_r, X_t, y_t,
                         X_f, y_f, seed, orig_acc):
    """Only meaningful for RF. XGB returns None (skip in tables)."""
    if model_type != "RF":
        return None
    m = get_model("RF", cfg, seed)
    m.fit(X_r, y_r)
    t0 = time.time()
    errors = [np.mean(e.predict(X_f) != y_f) for e in m.estimators_]
    threshold = np.percentile(errors, 40)
    m.estimators_ = [e for e, err in zip(m.estimators_, errors)
                     if err >= threshold]
    m.n_estimators = len(m.estimators_)
    tp_t = time.time() - t0
    return evaluate(m, X_t, y_t, X_f, tp_t, orig_acc)


def run_all_methods(model_type, cfg, X_r, y_r, X_t, y_t,
                    X_f, y_f, seed, dataset, k=5):
    """Run all 5 methods, return list of result dicts."""
    m_orig = get_model(model_type, cfg, seed)
    m_orig.fit(X_r, y_r)
    orig_acc = accuracy_score(y_t,
        (safe_proba(m_orig, X_t)[:, 1] >= 0.5).astype(int))

    methods = {
        "Full Retrain":          lambda: method_full_retrain(
            model_type, cfg, X_r, y_r, X_t, y_t, X_f, seed, orig_acc),
        "SISA":                  lambda: method_sisa(
            model_type, cfg, X_r, y_r, X_t, y_t, X_f, seed, orig_acc, k),
        "Label-Flip Retraining": lambda: method_label_flip(
            model_type, cfg, X_r, y_r, X_t, y_t, X_f, y_f, seed, orig_acc),
        "Influence Reweighting": lambda: method_influence_rewt(
            model_type, cfg, X_r, y_r, X_t, y_t, X_f, y_f, seed, orig_acc),
        "Tree Pruning":          lambda: method_tree_pruning(
            model_type, cfg, X_r, y_r, X_t, y_t, X_f, y_f, seed, orig_acc),
    }

    rows = []
    retrain_time = None
    for name, fn in methods.items():
        print(f"    [{model_type}] {name}...", end=" ", flush=True)
        result = fn()
        if result is None:
            print("N/A (XGB)")
            continue
        if name == "Full Retrain":
            retrain_time = result["time_sec"]
        speedup = (retrain_time / result["time_sec"]
                   if retrain_time and result["time_sec"] > 0 else 0)
        rows.append({
            "dataset": dataset, "model": model_type, "seed": seed,
            "method": name, "original_acc": orig_acc,
            "speedup": speedup, **result
        })
        print(f"acc={result['accuracy']:.4f} "
              f"time={result['time_sec']:.3f}s "
              f"speedup={speedup:.1f}×")

    return rows


def add_speedup(df):
    """Recalculate speedup from Full Retrain time per group."""
    for _, grp in df.groupby(["dataset", "model", "seed"]):
        rt_rows = grp[grp["method"] == "Full Retrain"]
        if len(rt_rows) == 0:
            continue
        rt = rt_rows["time_sec"].values[0]
        df.loc[grp.index, "speedup"] = rt / df.loc[grp.index, "time_sec"]
    return df
