# modelling_ui_extended_importance_fixed.py
# Full UI with rich outputs, corrected VIF, proper permutation importance wrappers, and LIME background matching.
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import traceback

import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

# Optional deps (graceful fallbacks)
try:
    from pyearth import Earth
    HAS_EARTH = True
except Exception:
    HAS_EARTH = False

try:
    from skgarden import RandomForestQuantileRegressor
    HAS_QRF = True
except Exception:
    HAS_QRF = False

# === SciPy: now import everything we need for KS + manual LP/CC ===
try:
    from scipy.stats import ks_2samp, ranksums, ansari, chi2, norm, rankdata
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# hyppo (optional; we won't rely on it for LP/CC)
try:
    from hyppo.ksample import Lepage, Cucconi
    HAS_HYPP0 = True
except Exception:
    HAS_HYPP0 = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except Exception:
    HAS_LIME = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_VIF = True
except Exception:
    HAS_VIF = False


# ---------- manual LP & CC (no hyppo) ----------
# Permutations for Cucconi p-value (increase for more accuracy, slower runtime)
CUCCONI_PERMUTATIONS = 2000
SCORER_R2 = make_scorer(r2_score)

def _clean1d(a):
    a = np.asarray(a, float).ravel()
    return a[np.isfinite(a)]

def lepage_test_manual(x, y):
    """
    Lepage = sum of squares of:
      - Wilcoxon rank-sum z (location)
      - Ansari–Bradley z (scale; reconstructed from two-sided p)
    Asymptotic p from chi-square with 2 df.
    """
    if not HAS_SCIPY:
        raise RuntimeError("SciPy required for Lepage (ranksums/ansari). Install with: pip install scipy")
    x = _clean1d(x); y = _clean1d(y)
    z_loc, _ = ranksums(x, y)
    _, p_ab = ansari(x, y)            # two-sided p
    z_scale = norm.isf(p_ab / 2.0)    # |z| from p
    L = float(z_loc**2 + z_scale**2)
    p = float(1.0 - chi2.cdf(L, df=2))
    return L, p, float(z_loc), float(z_scale)

def cucconi_stat(x, y):
    """
    Cucconi statistic based on ranks of the FIRST sample (x) in pooled sample.
    """
    if not HAS_SCIPY:
        raise RuntimeError("SciPy required for Cucconi (rankdata). Install with: pip install scipy")
    x = _clean1d(x); y = _clean1d(y)
    m, n = len(x), len(y); N = m + n
    z = np.concatenate([x, y])
    R = rankdata(z, method="average")[:m]  # ranks for x only
    denom = np.sqrt(m * n * (N + 1) * (2 * N + 1) * (8 * N + 11) / 5.0)
    U = (6.0 * np.sum(R**2) - m * (N + 1) * (2 * N + 1)) / denom
    V = (6.0 * np.sum((N + 1 - R)**2) - m * (N + 1) * (2 * N + 1)) / denom
    rho = (2.0 * (N**2 - 4.0)) / ((2 * N + 1) * (8 * N + 11)) - 1.0
    C = (U**2 + V**2 - 2.0 * rho * U * V) / (2.0 * (1.0 - rho**2))
    return float(C)

def cucconi_permutation_test(x, y, n_perm=CUCCONI_PERMUTATIONS, seed=42):
    """
    Permutation p-value for Cucconi statistic.
    """
    x = _clean1d(x); y = _clean1d(y)
    rng = np.random.default_rng(seed)
    obs = cucconi_stat(x, y)
    z = np.concatenate([x, y])
    m = len(x)
    ge = 0
    for _ in range(int(n_perm)):
        rng.shuffle(z)
        c = cucconi_stat(z[:m], z[m:])
        if c >= obs:
            ge += 1
    p = (ge + 1.0) / (n_perm + 1.0)
    return obs, float(p)


# ---------- metrics & importance helpers ----------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return 100 * np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask])

def safe_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

def pls_vip(pls: PLSRegression, X: np.ndarray, y: np.ndarray):
    # VIP for PLS1
    T = pls.x_scores_; W = pls.x_weights_; Q = pls.y_loadings_
    p, a = W.shape
    SSY = np.sum((T ** 2), axis=0) * (Q.ravel() ** 2)
    total_SSY = np.sum(SSY)
    vip = np.zeros((p,))
    for j in range(p):
        weight = np.sum(SSY * (W[j, :] ** 2) / np.sum(W ** 2, axis=0))
        vip[j] = np.sqrt(p * weight / total_SSY) if total_SSY > 0 else np.nan
    return vip

def std_coefs_from_standardized(model, scaler, feature_names):
    coef_std = getattr(model, "coef_", None)
    if coef_std is None:
        return None
    mu = scaler.mean_; sig = scaler.scale_
    beta_orig = coef_std / sig
    intercept_orig = getattr(model, "intercept_", 0.0) - np.sum(coef_std * mu / sig)
    df = pd.DataFrame({"variable": feature_names, "coef_orig": beta_orig, "coef_std": coef_std})
    return df, intercept_orig

# lightweight wrappers so permutation_importance always gets a .predict(X)
class ScaledPredictorWrapper:
    def __init__(self, scaler: StandardScaler, estimator):
        self.scaler = scaler
        self.estimator = estimator
    def predict(self, X):
        return self.estimator.predict(self.scaler.transform(np.asarray(X)))

class QuantilePredictorWrapper:
    def __init__(self, qrf, quantile):
        self.qrf = qrf
        self.quantile = float(quantile)
    def predict(self, X):
        return self.qrf.predict(np.asarray(X), quantile=self.quantile*100)

# --- Comparable (R²-drop) permutation-importance, normalized to sum=1 ---
def perm_importance_share(estimator, X, y, feature_names, n_repeats=15):
    """
    Computes permutation importance using R² (so higher = bigger drop in R²),
    clips negatives to 0, then normalizes to sum=1 for comparability.
    Works with wrappers that only implement .predict because we pass scoring=SCORER_R2.
    """
    if X is None or y is None or len(feature_names) == 0:
        return []
    X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
    y_arr = np.asarray(y)
    try:
        perm = permutation_importance(
            estimator, X_arr, y_arr,
            n_repeats=n_repeats, random_state=42, n_jobs=-1,
            scoring=SCORER_R2
        )
        imp = np.maximum(0.0, np.nan_to_num(perm.importances_mean, nan=0.0))
        s = imp.sum()
        share = imp / (s if s > 0 else 1.0)
        return sorted(list(zip(feature_names, share)), key=lambda t: t[1], reverse=True)
    except Exception:
        return []

def _print_importance_block(lines, pairs, title):
    if not pairs:
        lines.append(f"{title}: N/A")
        lines.append("")
        return
    lines.append(f"{title} (sum=1):")
    header = f"{'feature':36} {'share':>12}"
    lines.append(header); lines.append("-"*len(header))
    for f, s in pairs[:25]:
        lines.append(f"{f[:36]:36} {s:12.6f}")
    lines.append("")

# --- Standardized-beta share (for linear models) ---
def std_beta_share_from_ols_z(X_train_df, y_train_series):
    """
    Refit OLS on z-scored X and z-scored y (statsmodels) and return abs(std-beta) shares.
    """
    if X_train_df.shape[1] == 0:
        return []
    Xz = (X_train_df - X_train_df.mean())/X_train_df.std(ddof=0)
    yz = (y_train_series - y_train_series.mean())/y_train_series.std(ddof=0)
    fitz = sm.OLS(yz, sm.add_constant(Xz, has_constant='add')).fit()
    betaz = fitz.params.drop(labels=['const'], errors='ignore')
    vals = np.abs(betaz.reindex(X_train_df.columns).fillna(0.0).values)
    s = vals.sum()
    share = vals / (s if s > 0 else 1.0)
    return list(zip(X_train_df.columns, share))

def std_beta_share_from_sklearn_linear(coef_vec, y_train_series, feature_names):
    """
    For sklearn linear models trained on standardized X (but not y):
    std-beta_i = coef_i / std(y).  We then take abs and L1-normalize to sum=1.
    """
    coef_vec = np.asarray(coef_vec).ravel()
    ysd = float(np.std(y_train_series.values, ddof=0)) if len(y_train_series) else 1.0
    betas = np.abs(coef_vec / (ysd if ysd > 0 else 1.0))
    s = betas.sum()
    share = betas / (s if s > 0 else 1.0)
    return sorted(list(zip(feature_names, share)), key=lambda t: t[1], reverse=True)

# --- Minimal wrapper so permutation_importance works with statsmodels OLS ---
class _SMWrapper:
    def __init__(self, fit, columns):
        self._fit = fit
        self._cols = list(columns)
    def predict(self, X):
        # accept array or DataFrame; ensure DataFrame with correct cols
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._cols)
        else:
            X = X[self._cols]
        return self._fit.predict(sm.add_constant(X, has_constant='add'))


class KNNBoostRegressor:
    def __init__(self, n_estimators=3, n_neighbors=10, learning_rate=0.5, metric="minkowski"):
        self.n_estimators = int(n_estimators); self.n_neighbors = int(n_neighbors)
        self.learning_rate = float(learning_rate); self.metric = metric
        self.models_ = []; self.init_ = None
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).ravel()
        self.models_ = []; self.init_ = np.mean(y); residual = y - self.init_
        for _ in range(self.n_estimators):
            knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, metric=self.metric)
            knn.fit(X, residual); pred = knn.predict(X)
            self.models_.append(knn); residual = residual - self.learning_rate * pred
        return self
    def predict(self, X):
        X = np.asarray(X)
        pred = np.full(X.shape[0], self.init_, dtype=float)
        for knn in self.models_:
            pred += self.learning_rate * knn.predict(X)
        return pred


class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression Workbench (Extended + Importance, Fixed)")

        self.train_path = tk.StringVar(); self.test_path = tk.StringVar()
        self.model_choice = tk.StringVar(value="OLS (All)")
        self.robust_var = tk.BooleanVar(value=True)
        self.use_all_features_var = tk.BooleanVar(value=True)

        # params
        self.alpha_var = tk.StringVar(value="0.1")
        self.l1ratio_var = tk.StringVar(value="0.5")
        self.pls_comp_var = tk.StringVar(value="4")
        self.n_estimators_var = tk.StringVar(value="300")
        self.max_depth_var = tk.StringVar(value="")
        self.min_leaf_var = tk.StringVar(value="1")
        self.quantiles_var = tk.StringVar(value="0.1,0.5,0.9")
        self.svr_c_var = tk.StringVar(value="10"); self.svr_eps_var = tk.StringVar(value="0.01")
        self.knn_k_var = tk.StringVar(value="15"); self.knn_T_var = tk.StringVar(value="3"); self.knn_eta_var = tk.StringVar(value="0.3")
        self.mlp_hidden_var = tk.StringVar(value="64,32"); self.mlp_alpha_var = tk.StringVar(value="0.0005")

        self._excluded_cols = []; self.df_train = None; self.df_test = None
        
        # --- UI ---
        top = tk.Frame(root, pady=4); top.pack(fill="x")
        tk.Button(top, text="Select TRAIN CSV", command=self.load_train).pack(side="left", padx=4)
        tk.Label(top, textvariable=self.train_path, fg="gray").pack(side="left", padx=8)
        tk.Button(top, text="Select TEST CSV", command=self.load_test).pack(side="left", padx=12)
        tk.Label(top, textvariable=self.test_path, fg="gray").pack(side="left", padx=8)

        opts = tk.Frame(root, pady=4); opts.pack(fill="x")
        tk.Label(opts, text="Target:").grid(row=0, column=0, sticky="w")
        self.target_dropdown = tk.StringVar()
        self.target_menu = tk.OptionMenu(opts, self.target_dropdown, ()); self.target_menu.grid(row=0, column=1, sticky="w", padx=6)

        self.use_all_chk = tk.Checkbutton(opts, text="Use all non-target columns (auto-exclude A–E, IDs/time names)", variable=self.use_all_features_var, command=self.toggle_feature_list)
        self.use_all_chk.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        tk.Label(opts, text="(Optional) Select features:").grid(row=2, column=0, sticky="w")
        self.feature_list = tk.Listbox(opts, selectmode="multiple", width=44, height=8, exportselection=False)
        self.feature_list.grid(row=2, column=1, sticky="w", padx=6)

        self.excl_hint = tk.Label(opts, text="Auto-excluding: (pick TRAIN to show)", fg="gray")
        self.excl_hint.grid(row=3, column=0, columnspan=2, sticky="w")

        mdl = tk.Frame(root, pady=4); mdl.pack(fill="x")
        tk.Label(mdl, text="Model:").grid(row=0, column=0, sticky="w")
        model_menu = tk.OptionMenu(
            mdl, self.model_choice,
            "OLS (All)", "OLS (Stepwise AIC)", "OLS (Stepwise BIC)",
            "Lasso", "Elastic Net", "PLS Regression",
            "Bayesian Ridge", "Random Forest", "Bagged CART", "Bagged MARS",
            "Quantile Random Forest", "Boosted KNN", "SVR (RBF)", "Bayesian-Reg. NN",
            command=lambda _: self.toggle_param_fields()
        )
        model_menu.grid(row=0, column=1, sticky="w", padx=6)

        self.robust_chk = tk.Checkbutton(mdl, text="Use robust SEs (HC1) for OLS", variable=self.robust_var)
        self.robust_chk.grid(row=0, column=2, sticky="w", padx=12)

        # dynamic param widgets
        self.alpha_lbl = tk.Label(mdl, text="alpha:"); self.alpha_ent = tk.Entry(mdl, width=8, textvariable=self.alpha_var)
        self.l1ratio_lbl = tk.Label(mdl, text="l1_ratio:"); self.l1ratio_ent = tk.Entry(mdl, width=8, textvariable=self.l1ratio_var)
        self.pls_lbl = tk.Label(mdl, text="PLS components:"); self.pls_ent = tk.Entry(mdl, width=8, textvariable=self.pls_comp_var)
        self.nest_lbl = tk.Label(mdl, text="n_estimators:"); self.nest_ent = tk.Entry(mdl, width=8, textvariable=self.n_estimators_var)
        self.maxd_lbl = tk.Label(mdl, text="max_depth:"); self.maxd_ent = tk.Entry(mdl, width=8, textvariable=self.max_depth_var)
        self.minl_lbl = tk.Label(mdl, text="min_samples_leaf:"); self.minl_ent = tk.Entry(mdl, width=8, textvariable=self.min_leaf_var)
        self.q_lbl = tk.Label(mdl, text="quantiles (csv):"); self.q_ent = tk.Entry(mdl, width=12, textvariable=self.quantiles_var)
        self.svr_c_lbl = tk.Label(mdl, text="SVR C:"); self.svr_c_ent = tk.Entry(mdl, width=8, textvariable=self.svr_c_var)
        self.svr_e_lbl = tk.Label(mdl, text="epsilon:"); self.svr_e_ent = tk.Entry(mdl, width=8, textvariable=self.svr_eps_var)
        self.knn_k_lbl = tk.Label(mdl, text="k (neighbors):"); self.knn_k_ent = tk.Entry(mdl, width=8, textvariable=self.knn_k_var)
        self.knn_T_lbl = tk.Label(mdl, text="boost rounds:"); self.knn_T_ent = tk.Entry(mdl, width=8, textvariable=self.knn_T_var)
        self.knn_eta_lbl = tk.Label(mdl, text="learning_rate:"); self.knn_eta_ent = tk.Entry(mdl, width=8, textvariable=self.knn_eta_var)
        self.mlp_h_lbl = tk.Label(mdl, text="MLP hidden (csv):"); self.mlp_h_ent = tk.Entry(mdl, width=14, textvariable=self.mlp_hidden_var)
        self.mlp_a_lbl = tk.Label(mdl, text="alpha (L2):"); self.mlp_a_ent = tk.Entry(mdl, width=8, textvariable=self.mlp_alpha_var)

        self.toggle_param_fields()

        runbar = tk.Frame(root, pady=4); runbar.pack(fill="x")
        tk.Button(runbar, text="Run", command=self.run_model, width=14).pack(side="left", padx=4)

        self.output = tk.Text(root, wrap="word", height=36)
        self.output.pack(fill="both", expand=True, padx=6, pady=6)
        self.output.configure(font=("Consolas", 10))

    # -------- UI helpers --------
    def compute_excluded(self, df: pd.DataFrame):
        if df is None or df.empty: return []
        by_pos = list(df.columns[:5])
        by_name = [c for c in df.columns if c in {"A","B","C","D","E"}]
        name_map = {"TIME PERIOD","YEAR","QUARTER","COUNTRY","BANK NAME","ID","BANK","DATE"}
        by_name2 = [c for c in df.columns if c.upper() in name_map]
        seen, out = set(), []
        for c in by_pos + by_name + by_name2:
            if c in df.columns and c not in seen:
                out.append(c); seen.add(c)
        return out

    def load_train(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if not path: return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read TRAIN CSV:\n{e}"); return
        self.df_train = df; self.train_path.set(path); self.populate_columns()

    def load_test(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if not path: return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read TEST CSV:\n{e}"); return
        self.df_test = df; self.test_path.set(path)

    def populate_columns(self):
        if self.df_train is None: return
        cols = list(self.df_train.columns)
        menu = self.target_menu["menu"]; menu.delete(0,"end")
        for c in cols: menu.add_command(label=c, command=lambda v=c: self.target_dropdown.set(v))
        if cols: self.target_dropdown.set(cols[-1])
        self._excluded_cols = self.compute_excluded(self.df_train)
        hint = "Auto-excluding: " + (", ".join(self._excluded_cols[:28]) + (" ..." if len(self._excluded_cols)>28 else ""))
        self.excl_hint.config(text=hint)
        self.feature_list.delete(0,"end")
        for c in cols:
            if c not in self._excluded_cols:
                self.feature_list.insert("end", c)
        self.toggle_feature_list()

    def toggle_feature_list(self):
        self.feature_list.configure(state=("disabled" if self.use_all_features_var.get() else "normal"))

    def toggle_param_fields(self):
        for w in [self.alpha_lbl, self.alpha_ent, self.l1ratio_lbl, self.l1ratio_ent,
                  self.pls_lbl, self.pls_ent, self.nest_lbl, self.nest_ent,
                  self.maxd_lbl, self.maxd_ent, self.minl_lbl, self.minl_ent,
                  self.q_lbl, self.q_ent, self.svr_c_lbl, self.svr_c_ent,
                  self.svr_e_lbl, self.svr_e_ent, self.knn_k_lbl, self.knn_k_ent,
                  self.knn_T_lbl, self.knn_T_ent, self.knn_eta_lbl, self.knn_eta_ent,
                  self.mlp_h_lbl, self.mlp_h_ent, self.mlp_a_lbl, self.mlp_a_ent]:
            w.grid_forget()
        m = self.model_choice.get(); row = 1
        if m in ["Lasso","Elastic Net"]:
            self.alpha_lbl.grid(row=row, column=0, sticky="w", pady=2); self.alpha_ent.grid(row=row, column=1, sticky="w", padx=6)
        if m=="Elastic Net":
            self.l1ratio_lbl.grid(row=row, column=2, sticky="w"); self.l1ratio_ent.grid(row=row, column=3, sticky="w", padx=6)
        if m=="PLS Regression":
            self.pls_lbl.grid(row=row, column=2, sticky="w"); self.pls_ent.grid(row=row, column=3, sticky="w", padx=6)
        if m in ["Random Forest","Bagged CART","Bagged MARS","Quantile Random Forest"]:
            self.nest_lbl.grid(row=row, column=0, sticky="w"); self.nest_ent.grid(row=row, column=1, sticky="w", padx=6)
            self.maxd_lbl.grid(row=row, column=2, sticky="w"); self.maxd_ent.grid(row=row, column=3, sticky="w", padx=6)
            self.minl_lbl.grid(row=row+1, column=0, sticky="w"); self.minl_ent.grid(row=row+1, column=1, sticky="w", padx=6)
        if m=="Quantile Random Forest":
            self.q_lbl.grid(row=row+1, column=2, sticky="w"); self.q_ent.grid(row=row+1, column=3, sticky="w", padx=6)
        if m=="SVR (RBF)":
            self.svr_c_lbl.grid(row=row, column=0, sticky="w"); self.svr_c_ent.grid(row=row, column=1, sticky="w", padx=6)
            self.svr_e_lbl.grid(row=row, column=2, sticky="w"); self.svr_e_ent.grid(row=row, column=3, sticky="w", padx=6)
        if m=="Boosted KNN":
            self.knn_k_lbl.grid(row=row, column=0, sticky="w"); self.knn_k_ent.grid(row=row, column=1, sticky="w", padx=6)
            self.knn_T_lbl.grid(row=row, column=2, sticky="w"); self.knn_T_ent.grid(row=row, column=3, sticky="w", padx=6)
            self.knn_eta_lbl.grid(row=row+1, column=0, sticky="w"); self.knn_eta_ent.grid(row=row+1, column=1, sticky="w", padx=6)
        if m=="Bayesian-Reg. NN":
            self.mlp_h_lbl.grid(row=row, column=0, sticky="w"); self.mlp_h_ent.grid(row=row, column=1, sticky="w", padx=6)
            self.mlp_a_lbl.grid(row=row, column=2, sticky="w"); self.mlp_a_ent.grid(row=row, column=3, sticky="w", padx=6)

    # -------- core --------
    def forward_stepwise(self, X: pd.DataFrame, y: pd.Series, criterion="aic", max_vars=None):
        remaining = list(X.columns); selected = []; current_score = np.inf
        if max_vars is None: max_vars = len(remaining)
        for _ in range(max_vars):
            scores = []
            for cand in remaining:
                feats = selected + [cand]
                Xc = sm.add_constant(X[feats], has_constant='add')
                res = sm.OLS(y, Xc).fit()
                score = res.aic if criterion.lower()=="aic" else res.bic
                scores.append((score, cand))
            if not scores: break
            scores.sort(key=lambda t: t[0])
            best_score, best_cand = scores[0]
            if best_score + 1e-9 < current_score:
                selected.append(best_cand); remaining.remove(best_cand); current_score = best_score
            else:
                break
        return selected

    def eval_and_print(self, lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                       predictor=None, importance=None, extra_info=None, lime_on=True,
                       comp_importance=None, linear_importance=None):
        tr_r2 = r2_score(y_train, yhat_tr); tr_rmse = safe_rmse(y_train, yhat_tr); tr_mae = mean_absolute_error(y_train, yhat_tr)
        te_r2 = te_rmse = te_mae = te_smape = np.nan
        if y_test is not None and yhat_te is not None and len(yhat_te)==len(y_test):
            te_r2 = r2_score(y_test, yhat_te); te_rmse = safe_rmse(y_test, yhat_te)
            te_mae = mean_absolute_error(y_test, yhat_te); te_smape = smape(y_test, yhat_te)
        lines.append(f"Train  R²: {tr_r2:.4f} | RMSE: {tr_rmse:.6g} | MAE: {tr_mae:.6g}")
        if not np.isnan(te_r2):
            lines.append(f"Test   R²: {te_r2:.4f} | RMSE: {te_rmse:.6g} | MAE: {te_mae:.6g} | sMAPE(%): {te_smape:.3f}")
            lines.append(f"Gaps   ΔRMSE(test-train): {te_rmse - tr_rmse:.6g} | ΔR²(train-test): {tr_r2 - te_r2:.6g}")
        else:
            lines.append("Test metrics: N/A (no target in TEST)")
        lines.append("")

        # residual distribution tests
        if y_test is not None and not np.isnan(te_r2) and HAS_SCIPY:
            res_tr = (y_train.values - yhat_tr).astype(float)
            res_te = (y_test.values  - yhat_te).astype(float)
            ks_stat, ks_p = ks_2samp(res_tr, res_te, alternative="two-sided", method="auto")
            lines.append(f"KS test on residuals (train vs test): stat={ks_stat:.4f}, p={ks_p:.4f}")

            # Manual LP & CC (no hyppo)
            try:
                L, pL, z_loc, z_scale = lepage_test_manual(res_tr, res_te)
                lines.append(f"Lepage (LP, manual):           stat={L:.4f}, p={pL:.4f}  | z_loc={z_loc:.3f}, z_scale={z_scale:.3f}")
            except Exception as e:
                lines.append(f"(Lepage manual failed: {e})")

            try:
                C, pC = cucconi_permutation_test(res_tr, res_te, n_perm=CUCCONI_PERMUTATIONS, seed=42)
                lines.append(f"Cucconi (CC, manual perm {CUCCONI_PERMUTATIONS}): stat={C:.4f}, p~{pC:.4f}")
            except Exception as e:
                lines.append(f"(Cucconi manual failed: {e})")

            # Optional: hyppo if installed (kept for users who want to compare)
            if HAS_HYPP0:
                try:
                    lp = Lepage().test(res_tr.reshape(-1,1), res_te.reshape(-1,1))
                    cc = Cucconi().test(res_tr.reshape(-1,1), res_te.reshape(-1,1))
                    lines.append(f"[hyppo] Lepage:                 stat={lp[0]:.4f}, p={lp[1]:.4f}")
                    lines.append(f"[hyppo] Cucconi:                stat={cc[0]:.4f}, p={cc[1]:.4f}")
                except Exception as e:
                    lines.append(f"(hyppo LP/CC failed: {e})")
            else:
                lines.append("(Manual LP/CC shown; install 'hyppo' if you also want its versions)")

            lines.append("Note: p < 0.05 → distributions differ (potential overfit); p ≥ 0.05 → similar.")
            lines.append("")
        elif y_test is not None:
            lines.append("(Install SciPy for KS/LP/CC tests: pip install scipy)")
            lines.append("")

        # (legacy) model-specific importance (e.g., VIP, etc.)
        if importance is not None and len(importance):
            lines.append("Top feature importance (model-specific):")
            header = f"{'feature':36} {'score':>12}"
            lines.append(header); lines.append("-"*len(header))
            for f, s in importance[:25]:
                lines.append(f"{f[:36]:36} {s:12.6g}")
            lines.append("")

        # NEW: comparable importance (R²-drop permutation, sum=1)
        if comp_importance is not None:
            _print_importance_block(lines, comp_importance, "Comparable importance (R²-drop)")

        # NEW: linear standardized-beta share (sum=1) for linear models
        if linear_importance is not None:
            _print_importance_block(lines, linear_importance, "Linear (std-beta) importance")

        # LIME preview
        if lime_on and HAS_LIME and predictor is not None and predictor.get("background") is not None and predictor.get("X_test_for_lime") is not None:
            try:
                explainer = LimeTabularExplainer(
                    predictor["background"],
                    feature_names=list(X_train.columns),
                    verbose=False,
                    mode='regression'
                )
                X0 = predictor["X_test_for_lime"][0]
                exp = explainer.explain_instance(X0, predictor["predict"], num_features=min(10, X_train.shape[1]))
                lines.append("LIME (test row 0) top factors: " + ", ".join([f"{k}:{v:+.3f}" for k,v in exp.as_list()]))
                lines.append("")
            except Exception as e:
                lines.append(f"(LIME preview skipped: {e})")
                lines.append("")
        if extra_info:
            lines.extend(extra_info)

    def run_model(self):
        self.output.delete("1.0","end")
        try:
            if self.df_train is None or self.df_test is None:
                messagebox.showwarning("Missing files", "Please select both training and test CSVs."); return
            target = self.target_dropdown.get()
            if not target or target not in self.df_train.columns:
                messagebox.showwarning("Target", "Please select a valid target column (from training file)."); return

            if self.use_all_features_var.get():
                features = [c for c in self.df_train.columns if c != target and c not in self._excluded_cols]
            else:
                sel = self.feature_list.curselection()
                features = [self.feature_list.get(i) for i in sel if self.feature_list.get(i) != target]
            if not features:
                messagebox.showwarning("Features", "No features selected after auto-exclusion."); return

            train_cols = features + [target]
            test_cols  = [c for c in features if c in self.df_test.columns] + ([target] if target in self.df_test.columns else [])
            dftr = self.df_train[train_cols].copy().dropna()
            dfts = self.df_test[test_cols].copy().dropna()

            y_train = dftr[target]
            X_train = pd.get_dummies(dftr.drop(columns=[target]), drop_first=True)
            if target in dfts.columns:
                y_test = dfts[target]
                X_test = pd.get_dummies(dfts.drop(columns=[target]), drop_first=True)
            else:
                y_test = None
                X_test = pd.get_dummies(dfts, drop_first=True)

            # remove zero-variance
            nunique = X_train.nunique()
            keep = list(nunique[nunique > 1].index)
            X_train = X_train[keep] if keep else pd.DataFrame(index=X_train.index)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            model_name = self.model_choice.get()
            lines = []
            lines.append(f"=== {model_name} ===")
            lines.append(f"Rows used (train/test): {len(X_train)}/{len(X_test)}")
            lines.append(f"Features after dummies/filters: {X_train.shape[1]}")
            self._excluded_cols = self.compute_excluded(self.df_train)
            lines.append(f"Auto-excluded: {', '.join(self._excluded_cols) if self._excluded_cols else '(none)'}")
            lines.append("")

            # ----- OLS (All) -----
            if model_name == "OLS (All)":
                Xc_tr = sm.add_constant(X_train, has_constant='add')
                Xc_te = sm.add_constant(X_test,  has_constant='add')
                fit = sm.OLS(y_train, Xc_tr).fit(cov_type='HC1') if self.robust_var.get() else sm.OLS(y_train, Xc_tr).fit()
                yhat_tr = fit.predict(Xc_tr); yhat_te = fit.predict(Xc_te)

                # Full coefficient table with variance + standardized betas
                cov = fit.cov_params()
                var = np.diag(cov.values) if hasattr(cov, "values") else np.diag(cov)
                coef_df = pd.DataFrame({
                    "variable": fit.params.index,
                    "coef": fit.params.values,
                    "var(coef)": var,
                    "std_err": fit.bse.values,
                    "t": fit.tvalues.values,
                    "p>|t|": fit.pvalues.values,
                })
                conf = fit.conf_int(alpha=0.05); coef_df["CI_low"] = conf[0].values; coef_df["CI_high"] = conf[1].values

                # standardized betas (re-fit on z-scored X & y)
                if X_train.shape[1] > 0:
                    Xz = (X_train - X_train.mean())/X_train.std(ddof=0)
                    yz = (y_train - y_train.mean())/y_train.std(ddof=0)
                    betaz = sm.OLS(yz, sm.add_constant(Xz, has_constant='add')).fit().params
                    std_beta = betaz.reindex(coef_df["variable"]).fillna(np.nan).values
                else:
                    std_beta = np.full(coef_df.shape[0], np.nan)
                coef_df["std_beta"] = std_beta

                inter = coef_df[coef_df["variable"]=="const"]; others = coef_df[coef_df["variable"]!="const"].sort_values("p>|t|")
                lines.append("Coefficients (incl. variance & std. beta):")
                header = f"{'variable':35} {'coef':>12} {'var(coef)':>12} {'std_err':>12} {'t':>9} {'p':>8} {'[0.025':>12} {'0.975]':>12} {'std_beta':>12}"
                lines.append(header); lines.append("-"*len(header))
                for _, r in pd.concat([inter, others]).iterrows():
                    lines.append(f"{r['variable'][:35]:35} {r['coef']:12.6g} {r['var(coef)']:12.6g} {r['std_err']:12.6g} {r['t']:9.3f} {r['p>|t|']:8.4f} {r['CI_low']:12.6g} {r['CI_high']:12.6g} {r['std_beta']:12.6g}")
                lines.append("")

                # Correct VIF: compute on X with constant, but report only features
                if HAS_VIF and X_train.shape[1] > 0:
                    lines.append("VIF (multicollinearity check):")
                    Xc = sm.add_constant(X_train, has_constant='add').values
                    cols = ["const"] + list(X_train.columns)
                    vif_vals = []
                    for i in range(1, Xc.shape[1]):  # skip intercept index 0
                        try:
                            vif_vals.append((cols[i], variance_inflation_factor(Xc, i)))
                        except Exception:
                            vif_vals.append((cols[i], np.nan))
                    for col, v in sorted(vif_vals, key=lambda t: (np.nan_to_num(t[1], nan=0.0)), reverse=True)[:25]:
                        lines.append(f"{col[:35]:35} VIF={v:.3f}")
                    lines.append("")
                else:
                    lines.append("(VIF unavailable or no features)")
                    lines.append("")

                # NEW comparable & linear importance
                lin_imp = std_beta_share_from_ols_z(X_train, y_train)
                sm_wrap = _SMWrapper(fit, X_train.columns)
                comp_imp = perm_importance_share(sm_wrap, X_test, y_test.values if y_test is not None else None, list(X_train.columns), n_repeats=15)

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={
                        "predict": (lambda A: fit.predict(sm.add_constant(pd.DataFrame(A, columns=X_train.columns), has_constant='add'))),
                        "background": sm.add_constant(X_train, has_constant='add').values,  # same space as predict
                        "X_test_for_lime": sm.add_constant(X_test, has_constant='add').values
                    },
                    importance=None,
                    comp_importance=comp_imp,
                    linear_importance=lin_imp,
                    lime_on=False
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- Stepwise OLS -----
            if model_name in ["OLS (Stepwise AIC)", "OLS (Stepwise BIC)"]:
                crit = "aic" if "AIC" in model_name else "bic"
                selected = self.forward_stepwise(X_train, y_train, criterion=crit)
                lines.append(f"Selected ({crit.upper()}): {', '.join(selected) if selected else '(none)'}")
                Xs_tr = X_train[selected] if selected else pd.DataFrame(index=X_train.index)
                Xs_te = X_test[selected] if selected else pd.DataFrame(index=X_test.index)
                Xc_tr = sm.add_constant(Xs_tr, has_constant='add'); Xc_te = sm.add_constant(Xs_te, has_constant='add')
                fit = sm.OLS(y_train, Xc_tr).fit(cov_type='HC1') if self.robust_var.get() else sm.OLS(y_train, Xc_tr).fit()
                yhat_tr = fit.predict(Xc_tr); yhat_te = fit.predict(Xc_te)

                cov = fit.cov_params(); var = np.diag(cov.values) if hasattr(cov, "values") else np.diag(cov)
                coef_df = pd.DataFrame({"variable": fit.params.index, "coef": fit.params.values, "var(coef)": var,
                                        "std_err": fit.bse.values, "t": fit.tvalues.values, "p": fit.pvalues.values})
                conf = fit.conf_int(alpha=0.05); coef_df["CI_low"]=conf[0].values; coef_df["CI_high"]=conf[1].values
                lines.append("\nCoefficients (selected):")
                header = f"{'variable':35} {'coef':>12} {'var(coef)':>12} {'std_err':>12} {'t':>9} {'p':>8} {'[0.025':>12} {'0.975]':>12}"
                lines.append(header); lines.append("-"*len(header))
                for _, r in coef_df.iterrows():
                    lines.append(f"{r['variable'][:35]:35} {r['coef']:12.6g} {r['var(coef)']:12.6g} {r['std_err']:12.6g} {r['t']:9.3f} {r['p']:8.4f} {r['CI_low']:12.6g} {r['CI_high']:12.6g}")
                lines.append("")

                # NEW comparable & linear importance
                lin_imp = std_beta_share_from_ols_z(Xs_tr, y_train) if selected else []
                sm_wrap = _SMWrapper(fit, Xs_tr.columns if selected else [])
                comp_imp = perm_importance_share(sm_wrap, Xs_te, y_test.values if y_test is not None else None,
                                                 list(Xs_tr.columns) if selected else [], n_repeats=15)

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={
                        "predict": (lambda A: fit.predict(sm.add_constant(pd.DataFrame(A, columns=Xs_tr.columns), has_constant='add'))),
                        "background": sm.add_constant(Xs_tr, has_constant='add').values if Xs_tr.shape[1]>0 else sm.add_constant(pd.DataFrame(index=Xs_tr.index), has_constant='add').values,
                        "X_test_for_lime": sm.add_constant(Xs_te, has_constant='add').values if Xs_te.shape[1]>0 else sm.add_constant(pd.DataFrame(index=Xs_te.index), has_constant='add').values
                    },
                    importance=None,
                    comp_importance=comp_imp,
                    linear_importance=lin_imp,
                    lime_on=False
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- Lasso / Elastic Net (scaled) -----
            if model_name in ["Lasso","Elastic Net"]:
                try: alpha = float(self.alpha_var.get())
                except: messagebox.showwarning("Param","alpha must be numeric."); return
                if model_name=="Lasso":
                    reg = Lasso(alpha=alpha, max_iter=20000)
                else:
                    try: l1r = float(self.l1ratio_var.get())
                    except: messagebox.showwarning("Param","l1_ratio must be numeric."); return
                    reg = ElasticNet(alpha=alpha, l1_ratio=l1r, max_iter=20000)

                scaler = StandardScaler(with_mean=True, with_std=True)
                Xs_tr = scaler.fit_transform(X_train.values); Xs_te = scaler.transform(X_test.values)
                reg.fit(Xs_tr, y_train.values)
                yhat_tr = reg.predict(Xs_tr); yhat_te = reg.predict(Xs_te)

                # NEW comparable & linear importance
                wrapper = ScaledPredictorWrapper(scaler, reg)
                comp_imp = perm_importance_share(wrapper, X_test, y_test.values if y_test is not None else None,
                                                 list(X_train.columns), n_repeats=15)
                lin_imp = std_beta_share_from_sklearn_linear(reg.coef_, y_train, list(X_train.columns))

                lines.append(f"Non-zero coefficients: {int(np.sum(reg.coef_!=0))}/{len(reg.coef_)}")
                lines.append("")

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={"predict": (lambda A: reg.predict(scaler.transform(np.asarray(A))))},
                    importance=None,
                    comp_importance=comp_imp,
                    linear_importance=lin_imp
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- PLS -----
            if model_name=="PLS Regression":
                try: ncomp = int(float(self.pls_comp_var.get()))
                except: messagebox.showwarning("Param","PLS components must be integer."); return
                ncomp = max(1, min(ncomp, max(1, min(X_train.shape))))
                pls = PLSRegression(n_components=ncomp, scale=True)
                pls.fit(X_train.values, y_train.values)
                yhat_tr = pls.predict(X_train.values).ravel(); yhat_te = pls.predict(X_test.values).ravel()

                vip = pls_vip(pls, X_train.values, y_train.values)
                vip_df = pd.DataFrame({"feature": X_train.columns, "VIP": vip, "coef": pls.coef_.ravel()}).sort_values("VIP", ascending=False)
                imp_vip = list(zip(vip_df["feature"], vip_df["VIP"]))

                # NEW comparable importance
                comp_imp = perm_importance_share(pls, X_test.values, y_test.values if y_test is not None else None,
                                                 list(X_train.columns), n_repeats=15)

                lines.append(f"Components: {ncomp}")
                lines.append("")
                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={"predict": (lambda A: pls.predict(np.asarray(A)).ravel())},
                    importance=imp_vip,
                    comp_importance=comp_imp
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- Bayesian Ridge (scaled) -----
            if model_name=="Bayesian Ridge":
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xs_tr = scaler.fit_transform(X_train.values); Xs_te = scaler.transform(X_test.values)
                br = BayesianRidge()
                br.fit(Xs_tr, y_train.values)
                yhat_tr = br.predict(Xs_tr); yhat_te = br.predict(Xs_te)

                # NEW comparable & linear importance
                wrapper = ScaledPredictorWrapper(scaler, br)
                comp_imp = perm_importance_share(wrapper, X_test, y_test.values if y_test is not None else None,
                                                 list(X_train.columns), n_repeats=15)
                lin_imp = std_beta_share_from_sklearn_linear(br.coef_, y_train, list(X_train.columns))

                extra = [f"(BR alpha={getattr(br,'alpha_',np.nan):.6g}, lambda={getattr(br,'lambda_',np.nan):.6g})"]

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={"predict": (lambda A: br.predict(scaler.transform(np.asarray(A))))},
                    importance=None,
                    comp_importance=comp_imp,
                    linear_importance=lin_imp,
                    extra_info=extra
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- Random Forest -----
            if model_name=="Random Forest":
                print("y (train) min/mean/max:", float(np.nanmin(y_train)), float(np.nanmean(y_train)), float(np.nanmax(y_train)))
                try: n_est = int(float(self.n_estimators_var.get()))
                except: messagebox.showwarning("Param","n_estimators must be int."); return
                max_depth = int(float(self.max_depth_var.get())) if self.max_depth_var.get() else None
                min_leaf = int(float(self.min_leaf_var.get())) if self.min_leaf_var.get() else 1
                rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, min_samples_leaf=min_leaf,
                                           n_jobs=-1, random_state=42, oob_score=True, bootstrap=True)
                rf.fit(X_train, y_train)
                yhat_tr = rf.predict(X_train); yhat_te = rf.predict(X_test)

                # NEW comparable importance
                comp_imp = perm_importance_share(rf, X_test.values, y_test.values if y_test is not None else None,
                                                 list(X_train.columns), n_repeats=15)
                extra = [f"OOB R² (internal): {getattr(rf,'oob_score_',np.nan):.4f}"]

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={"predict": rf.predict},
                    importance=None,
                    comp_importance=comp_imp,
                    extra_info=extra
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- Bagged CART -----
            if model_name=="Bagged CART":
                try: n_est = int(float(self.n_estimators_var.get()))
                except: messagebox.showwarning("Param","n_estimators must be int."); return
                max_depth = int(float(self.max_depth_var.get())) if self.max_depth_var.get() else None
                min_leaf = int(float(self.min_leaf_var.get())) if self.min_leaf_var.get() else 1
                base = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42)
                bag = BaggingRegressor(base_estimator=base, n_estimators=n_est, n_jobs=-1, random_state=42)
                bag.fit(X_train, y_train)
                yhat_tr = bag.predict(X_train); yhat_te = bag.predict(X_test)

                # NEW comparable importance
                comp_imp = perm_importance_share(bag, X_test.values, y_test.values if y_test is not None else None,
                                                 list(X_train.columns), n_repeats=15)

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={"predict": bag.predict},
                    importance=None,
                    comp_importance=comp_imp
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- Bagged MARS -----
            if model_name=="Bagged MARS":
                if not HAS_EARTH:
                    lines.append("(Install 'sklearn-contrib-py-earth' to enable MARS: pip install sklearn-contrib-py-earth)")
                    self.output.insert("1.0", "\n".join(lines)); return
                try: n_est = int(float(self.n_estimators_var.get()))
                except: messagebox.showwarning("Param","n_estimators must be int."); return
                base = Earth()
                bag = BaggingRegressor(base_estimator=base, n_estimators=n_est, n_jobs=-1, random_state=42)
                bag.fit(X_train.values, y_train.values)
                yhat_tr = bag.predict(X_train.values); yhat_te = bag.predict(X_test.values)

                # NEW comparable importance
                comp_imp = perm_importance_share(bag, X_test.values, y_test.values if y_test is not None else None,
                                                 list(X_train.columns), n_repeats=15)

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={"predict": (lambda A: bag.predict(np.asarray(A)))},
                    importance=None,
                    comp_importance=comp_imp
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- Quantile Random Forest (median prediction) -----
            if model_name=="Quantile Random Forest":
                try:
                    qs = [float(q.strip()) for q in self.quantiles_var.get().split(",") if q.strip()!=""]
                except:
                    messagebox.showwarning("Param","quantiles must be numbers, e.g. 0.1,0.5,0.9"); return
                qs = sorted([q for q in qs if 0 < q < 1]); qs = qs if qs else [0.5]
                median_q = 0.5 if 0.5 in qs else qs[len(qs)//2]

                if HAS_QRF:
                    qrf = RandomForestQuantileRegressor(
                        n_estimators=int(float(self.n_estimators_var.get())),
                        random_state=42, n_jobs=-1,
                        min_samples_leaf=int(float(self.min_leaf_var.get() or 1))
                    )
                    qrf.fit(X_train, y_train)
                    yhat_tr = qrf.predict(X_train, quantile=median_q*100); yhat_te = qrf.predict(X_test, quantile=median_q*100)
                    low_q, high_q = qs[0], qs[-1]
                    qlow = qrf.predict(X_test, quantile=low_q*100); qhigh = qrf.predict(X_test, quantile=high_q*100)

                    # NEW comparable importance via wrapper
                    wrapper = QuantilePredictorWrapper(qrf, median_q)
                    comp_imp = perm_importance_share(wrapper, X_test, y_test.values if y_test is not None else None,
                                                     list(X_train.columns), n_repeats=15)
                else:
                    # Fallback: quantile GBMs
                    def fit_q(q): return GradientBoostingRegressor(loss="quantile", alpha=q, random_state=42).fit(X_train, y_train)
                    g_m = fit_q(median_q); yhat_tr = g_m.predict(X_train); yhat_te = g_m.predict(X_test)
                    low_q, high_q = qs[0], qs[-1]; qlow = fit_q(low_q).predict(X_test); qhigh = fit_q(high_q).predict(X_test)

                    # NEW comparable importance
                    comp_imp = perm_importance_share(g_m, X_test.values, y_test.values if y_test is not None else None,
                                                     list(X_train.columns), n_repeats=15)
                    lines.append("(Using quantile GradientBoostingRegressor fallback; install 'scikit-garden' for QRF)")
                interval_width = float(np.median(qhigh - qlow))
                extra = [f"Predictive interval median width (q{low_q:.2f}–q{high_q:.2f}): {interval_width:.6g}"]

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={"predict": (lambda A: yhat_te)},  # LIME not meaningful here
                    importance=None,
                    comp_importance=comp_imp,
                    extra_info=extra, lime_on=False
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- SVR (RBF, scaled) -----
            if model_name=="SVR (RBF)":
                try: C = float(self.svr_c_var.get()); eps = float(self.svr_eps_var.get())
                except: messagebox.showwarning("Param","SVR C/epsilon must be numeric."); return
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xs_tr = scaler.fit_transform(X_train.values); Xs_te = scaler.transform(X_test.values)
                svr = SVR(C=C, epsilon=eps, kernel="rbf")
                svr.fit(Xs_tr, y_train.values)
                yhat_tr = svr.predict(Xs_tr); yhat_te = svr.predict(Xs_te)

                # NEW comparable importance (use scaled X for permutation)
                comp_imp = perm_importance_share(svr, Xs_te, y_test.values if y_test is not None else None,
                                                 list(X_train.columns), n_repeats=15)
                extra = [f"Support vectors: {getattr(svr,'support_',np.array([])).shape[0]}"]

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={"predict": (lambda A: svr.predict(A))},  # predictor expects scaled arrays
                    importance=None,
                    comp_importance=comp_imp,
                    extra_info=extra
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- Boosted KNN (scaled) -----
            if model_name=="Boosted KNN":
                try: k = int(float(self.knn_k_var.get())); T = int(float(self.knn_T_var.get())); eta = float(self.knn_eta_var.get())
                except: messagebox.showwarning("Param","k, rounds, learning_rate must be numeric."); return
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xs_tr = scaler.fit_transform(X_train.values); Xs_te = scaler.transform(X_test.values)
                bk = KNNBoostRegressor(n_estimators=T, n_neighbors=k, learning_rate=eta)
                bk.fit(Xs_tr, y_train.values)
                yhat_tr = bk.predict(Xs_tr); yhat_te = bk.predict(Xs_te)

                # NEW comparable importance via wrapper
                wrapper = ScaledPredictorWrapper(scaler, bk)
                comp_imp = perm_importance_share(wrapper, X_test, y_test.values if y_test is not None else None,
                                                 list(X_train.columns), n_repeats=15)

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={"predict": (lambda A: bk.predict(A))},  # expects scaled arrays
                    importance=None,
                    comp_importance=comp_imp
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # ----- Bayesian-regularized NN -----
            if model_name=="Bayesian-Reg. NN":
                try:
                    hidden = tuple(int(s.strip()) for s in self.mlp_hidden_var.get().split(",") if s.strip()!="")
                    alpha = float(self.mlp_alpha_var.get())
                except:
                    messagebox.showwarning("Param","Hidden layers must be csv ints; alpha numeric."); return
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xs_tr = scaler.fit_transform(X_train.values); Xs_te = scaler.transform(X_test.values)
                mlp = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, random_state=42,
                                   activation="relu", solver="adam", max_iter=2000, early_stopping=True)
                mlp.fit(Xs_tr, y_train.values)
                yhat_tr = mlp.predict(Xs_tr); yhat_te = mlp.predict(Xs_te)

                # NEW comparable importance
                comp_imp = perm_importance_share(mlp, Xs_te, y_test.values if y_test is not None else None,
                                                 list(X_train.columns), n_repeats=15)
                extra = [f"MLP iters: {getattr(mlp,'n_iter_',np.nan)} | hidden: {hidden} | L2: {alpha}"]

                self.eval_and_print(
                    lines, y_train, y_test, yhat_tr, yhat_te, X_train,
                    predictor={"predict": (lambda A: mlp.predict(A))},  # expects scaled arrays
                    importance=None,
                    comp_importance=comp_imp,
                    extra_info=extra
                )
                self.output.insert("1.0", "\n".join(lines)); return

            # Fallback
            lines.append("Model not recognized.")
            self.output.insert("1.0", "\n".join(lines))

        except Exception as e:
            tb = traceback.format_exc()
            self.output.insert("1.0", f"Error:\n{e}\n\nTraceback:\n{tb}")

    @property
    def excluded_cols(self):
        return self._excluded_cols
    @excluded_cols.setter
    def excluded_cols(self, val):
        self._excluded_cols = val

def main():
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
