"""
091 계열 실험험
"""
import os, re, warnings, traceback, shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Paths / Const --------------------
DATA_DIR   = Path("data")
PRO_DIR    = DATA_DIR / "processed"
REP_DIR    = Path("reports")
SUM_DIR    = REP_DIR / "summaries"
SCR_DIR    = REP_DIR / "scripts"
MET_DIR    = REP_DIR / "metrics"
MODELS_DIR = Path("models")
for d in (PRO_DIR, REP_DIR, SUM_DIR, SCR_DIR, MET_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET_COL = "warping_index_target"            # t+1
CUR_WARP   = "warping_index_current_pass"      # t
PLATE_COL  = "FM_날판번호"
PASS_COL   = "FM_PASS NO N"
MONTH_COL  = "FM_압연월"

EXCLUDE = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL, "_target"}  # <- _target도 제외

SEED       = 42
TEST_SIZE  = 0.2
CV_FOLDS   = 3
SELECT_K   = 120

# -------------------- Env flags --------------------
def _env_on(name: str, default="0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1","true","yes","y"}

RUN_NAME         = os.environ.get("RUN_NAME", "pass_t_plus_1_cross")
PERMUTE_TARGET   = _env_on("PERMUTE_TARGET","0")
LEAK_AUTOCHECK   = _env_on("LEAK_AUTOCHECK","1")
USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET","0")
USE_PAIR_TE      = _env_on("USE_PAIR_TE","0")
USE_HETERO_W     = _env_on("USE_HETERO_W","0")
USE_MONO_CUR     = _env_on("USE_MONO_CUR","0")
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY","0")

# 모델 선택: E2X=고정 XGB, X2E=환경변수로 선택 가능
X2E_ALGO = os.environ.get("X2E_ALGO", "XGBoost").strip() or "XGBoost"
E2X_ALGO = "XGBoost"

# -------------------- Leak guards --------------------
_BAD_KW = ("target","_target","label","pred","_hat","oof","te_","_te","y_")

def _hard_leak_guard_cols(cols: list[str]) -> list[str]:
    safe = []
    for c in cols:
        low = c.lower()
        if any(k in low for k in _BAD_KW):
            continue
        safe.append(c)
    return safe

def _ensure_numeric_X(df: pd.DataFrame):
    X_all = df.drop(columns=[c for c in EXCLUDE if c in df.columns], errors="ignore")
    X_all = X_all[_hard_leak_guard_cols(list(X_all.columns))]
    X = X_all.select_dtypes(include=[np.number]).copy()
    med = X.median(numeric_only=True)
    return X.fillna(med), med.to_dict(), list(X.columns)

# -------------------- Data load/merge (t→t+1) --------------------
def load_and_merge_tplus1():
    ent = pd.read_csv(DATA_DIR/"entry_direction_results.csv")
    ext = pd.read_csv(DATA_DIR/"exit_direction_results.csv")
    gt  = pd.concat([ent, ext], ignore_index=True)
    fm  = pd.read_csv(DATA_DIR/"posco2_105190.csv")

    gt["extracted_plate"] = gt["filename"].str.extract(r"(PB\d+)")
    gt["extracted_pass"]  = gt["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    merged = (pd.merge(gt, fm, left_on=["extracted_plate","extracted_pass"],
                       right_on=[PLATE_COL, PASS_COL], how="inner")
                .sort_values([PLATE_COL, PASS_COL]).reset_index(drop=True))

    # t+1 target
    merged[TARGET_COL] = merged.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged = merged.rename(columns={"warping_index": CUR_WARP})
    # 불연속 pass 드랍
    nxt = merged.groupby(PLATE_COL)[PASS_COL].shift(-1)
    nonconsec = (nxt.notna()) & ((nxt - merged[PASS_COL]) != 1)
    merged.loc[nonconsec, TARGET_COL] = np.nan
    before = len(merged)
    merged = merged.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # 정리
    merged = merged.drop(columns=["extracted_plate","extracted_pass","filename",
                                  "quality_grade","quality_grade_current_pass","direction"],
                         errors="ignore")

    PRO_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PRO_DIR/"final_merged_data_regression_tplus1.csv", index=False)
    return merged, int(nonconsec.sum()), before - len(merged)

def split_e2x_x2e(df: pd.DataFrame):
    e2x = df[df[PASS_COL] % 2 == 1].copy()
    x2e = df[df[PASS_COL] % 2 == 0].copy()
    e2x.to_csv(PRO_DIR/"e2x_raw_tplus1.csv", index=False)
    x2e.to_csv(PRO_DIR/"x2e_raw_tplus1.csv", index=False)
    return e2x, x2e

# -------------------- Anchor split (group) --------------------
def _anchor_path(prefix: str) -> Path:
    return MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl"

def split_by_anchor(df: pd.DataFrame, prefix: str, seed=SEED, test_size=TEST_SIZE):
    anc = _anchor_path(prefix)
    plates_all = df[PLATE_COL].astype(str).unique().tolist()
    if anc.exists():
        test_plates = set(pd.read_pickle(anc))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        Xtmp = df.drop(columns=[TARGET_COL], errors="ignore")
        ytmp = df[TARGET_COL].values
        tr_i, te_i = next(gss.split(Xtmp, ytmp, groups=df[PLATE_COL].astype(str).values))
        test_plates = set(df.iloc[te_i][PLATE_COL].astype(str).unique().tolist())
        pd.to_pickle(sorted(list(test_plates)), anc)

    tr = df[~df[PLATE_COL].astype(str).isin(test_plates)].copy()
    te = df[df[PLATE_COL].astype(str).isin(test_plates)].copy()
    # 비정상 분할 복구
    if len(tr) == 0 or len(te) == 0:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed+777)
        tr_i, te_i = next(gss.split(df.drop(columns=[TARGET_COL], errors="ignore"),
                                    df[TARGET_COL].values, groups=df[PLATE_COL].astype(str).values))
        test_plates = set(df.iloc[te_i][PLATE_COL].astype(str).unique().tolist())
        pd.to_pickle(sorted(list(test_plates)), anc)
        tr = df.iloc[tr_i].copy(); te = df.iloc[te_i].copy()
    return tr, te, set(test_plates)

# -------------------- Feature engineering --------------------
def base_fe(df: pd.DataFrame, pass_max: float) -> pd.DataFrame:
    d = df.copy()
    fm_cols = [c for c in d.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if fm_cols:
        blk = d[fm_cols]
        d["FM_mean"]  = blk.mean(axis=1)
        d["FM_std"]   = blk.std(axis=1)
        d["FM_max"]   = blk.max(axis=1)
        d["FM_min"]   = blk.min(axis=1)
        d["FM_range"] = d["FM_max"] - d["FM_min"]
        d["FM_cv"]    = d["FM_std"] / (d["FM_mean"].abs() + 1e-8)
    d["FM_PASS_squared"]  = d[PASS_COL] ** 2
    d["FM_PASS_progress"] = d[PASS_COL] / (pass_max if pass_max > 0 else 1.0)
    return d

def apply_fe_train_test(tr_raw: pd.DataFrame, te_raw: pd.DataFrame):
    pm = float(tr_raw[PASS_COL].max()) if len(tr_raw) else 1.0
    return base_fe(tr_raw, pm), base_fe(te_raw, pm), pm

# -------------------- Outlier guard --------------------
def _compute_train_outlier_bounds(Xtr_df: pd.DataFrame, ytr: np.ndarray,
                                  qx=(0.01, 0.99), qy=(0.005, 0.995)):
    if len(Xtr_df) < 5 or len(ytr) < 5:
        # 표본 부족 → 필터 미적용
        return {}, None, None
    bounds = {}
    for c in Xtr_df.columns:
        arr = Xtr_df[c].values
        if np.isfinite(arr).sum() < max(5, int(0.1*len(arr))):
            continue
        lo, hi = np.quantile(arr, qx)
        bounds[c] = (float(lo), float(hi))
    ylo, yhi = np.quantile(ytr, qy)
    return bounds, float(ylo), float(yhi)

def _apply_train_outlier_filter(tr_df: pd.DataFrame, y: np.ndarray, bounds: dict, ylo, yhi):
    if not bounds or ylo is None or yhi is None:
        return tr_df.copy(), np.arange(len(tr_df))  # no filter
    keep = (y >= ylo) & (y <= yhi)
    for c,(lo,hi) in bounds.items():
        if c in tr_df.columns:
            v = tr_df[c].values
            keep &= (v >= lo) & (v <= hi)
    idx = np.where(keep)[0]
    if len(idx) < max(10, int(0.2*len(tr_df))):  # 과도 필터 → 미적용
        idx = np.arange(len(tr_df))
    return tr_df.iloc[idx].reset_index(drop=True), idx

def _clip_test_to_bounds(te_df: pd.DataFrame, bounds: dict):
    if not bounds:
        return te_df.copy()
    d = te_df.copy()
    for c,(lo,hi) in bounds.items():
        if c in d.columns:
            d[c] = d[c].clip(lower=lo, upper=hi)
    return d

# -------------------- Pair Δ-TE / Hetero-W (OOF-safe) --------------------
def _pair_str(df: pd.DataFrame) -> pd.Series:
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

def add_pair_delta_te(train_df: pd.DataFrame, test_df: pd.DataFrame, n0=50):
    tr = train_df.copy(); te = test_df.copy()
    tr["pair"] = _pair_str(tr); te["pair"] = _pair_str(te)
    tr["delta"] = tr[TARGET_COL] - tr[CUR_WARP]
    gkf = GroupKFold(n_splits=max(2, min(CV_FOLDS, tr[PLATE_COL].nunique())))
    plates = tr[PLATE_COL].astype(str).values
    oof = np.zeros(len(tr))
    for tr_i, va_i in gkf.split(tr, tr["delta"].values, groups=plates):
        sub = tr.iloc[tr_i]
        gmean = sub["delta"].mean()
        stat = sub.groupby("pair")["delta"].agg(n="size", mean="mean").reset_index()
        stat["te"] = (stat["n"]/(stat["n"]+n0))*stat["mean"] + (n0/(stat["n"]+n0))*gmean
        m = dict(zip(stat["pair"], stat["te"]))
        oof[va_i] = tr.iloc[va_i]["pair"].map(m).fillna(gmean).values
    gmean = tr["delta"].mean()
    stat = tr.groupby("pair")["delta"].agg(n="size", mean="mean").reset_index()
    stat["te"] = (stat["n"]/(stat["n"]+n0))*stat["mean"] + (n0/(stat["n"]+n0))*gmean
    m_full = dict(zip(stat["pair"], stat["te"]))
    te_feat = te["pair"].map(m_full).fillna(gmean).values
    tr = tr.drop(columns=["pair","delta"])
    te = te.drop(columns=["pair"])
    tr["pair_delta_te"] = oof
    te["pair_delta_te"] = te_feat
    return tr, te

def compute_hetero_weights(train_df: pd.DataFrame, test_df: pd.DataFrame, n0=50, clip=(0.25,4.0)):
    tr = train_df.copy(); te = test_df.copy()
    tr["pair"] = _pair_str(tr)
    tr["delta"] = tr[TARGET_COL] - tr[CUR_WARP]
    gkf = GroupKFold(n_splits=max(2, min(CV_FOLDS, tr[PLATE_COL].nunique())))
    plates = tr[PLATE_COL].astype(str).values
    oof_std = np.zeros(len(tr))
    for tr_i, va_i in gkf.split(tr, tr["delta"].values, groups=plates):
        sub = tr.iloc[tr_i]
        gs  = sub["delta"].std(ddof=1)
        stat = sub.groupby("pair")["delta"].agg(n="size", std="std").reset_index()
        w = stat["n"]/(stat["n"]+n0)
        stat["std_sh"] = w*stat["std"].fillna(gs) + (1-w)*gs
        m = dict(zip(stat["pair"], stat["std_sh"]))
        pva = tr.iloc[va_i]["pair"].values
        oof_std[va_i] = [m.get(p, gs) for p in pva]
    gs = tr["delta"].std(ddof=1)
    w_tr = (float(gs)/(oof_std + 1e-6))
    w_tr = np.clip(w_tr, clip[0], clip[1])
    return w_tr / (np.mean(w_tr) + 1e-8)

# -------------------- Select-K (corr-based) --------------------
def select_kbest_corr(Xtr_df: pd.DataFrame, ytr: np.ndarray, Xte_df: pd.DataFrame,
                      mandatory=None, k=SELECT_K):
    mandatory = list(mandatory or [])
    cols_all = list(Xtr_df.columns)

    # finite / non-constant
    vals = Xtr_df.values
    if len(Xtr_df) == 0 or vals.size == 0:
        chosen = mandatory[:1] if mandatory else cols_all[:1]
        return chosen, Xtr_df.get(chosen, pd.DataFrame()).values, Xte_df.get(chosen, pd.DataFrame()).values

    finite_mask = np.isfinite(vals).all(axis=0)
    var = Xtr_df.var(numeric_only=True).reindex(cols_all).fillna(0.0).values
    keep_mask = finite_mask & (var > 1e-12)
    keep_cols = [c for c,m in zip(cols_all, keep_mask) if m]
    for c in mandatory:
        if c not in keep_cols and c in cols_all:
            keep_cols.append(c)
    if not keep_cols:
        keep_cols = mandatory[:1] if mandatory else cols_all[:1]

    Xtr = Xtr_df[keep_cols].fillna(Xtr_df[keep_cols].median(numeric_only=True))
    Xte = Xte_df.reindex(columns=keep_cols, fill_value=np.nan).fillna(Xtr.median(numeric_only=True))

    # corr-based score
    y = ytr.astype(float)
    if len(y) < 3:
        chosen = list(dict.fromkeys(mandatory + keep_cols))[:max(1, min(k, len(keep_cols)))]
        return chosen, Xtr[chosen].values, Xte[chosen].values

    x = Xtr.values
    y0 = y - y.mean()
    ys = y0.std()
    ys = ys if ys >= 1e-12 else 1e-12
    xm = x.mean(axis=0); xs = x.std(axis=0); xs[xs < 1e-12] = 1e-12
    xr = (x - xm)/xs
    yr = y0/ys
    r  = (xr.T @ yr) / (len(y) - 1)
    r2 = np.clip(r**2, 0.0, 1.0 - 1e-12)

    mand = [c for c in mandatory if c in keep_cols]
    others = [c for c in keep_cols if c not in mand]
    k_rem = max(1, min(int(k) - len(mand), len(others)))
    idx_map = {c:i for i,c in enumerate(keep_cols)}
    order = np.argsort([r2[idx_map[c]] for c in others])[::-1] if others else []
    chosen = mand + [others[i] for i in order[:k_rem]]
    return chosen, Xtr[chosen].values, Xte[chosen].values

# -------------------- Models --------------------
def _build_model(name: str, n_features: int, mono_idx: int | None = None, seed=SEED):
    if name == "XGBoost":
        params = dict(
            random_state=seed, tree_method="hist", eval_metric="rmse",
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0, n_jobs=0
        )
        if mono_idx is not None and (0 <= mono_idx < n_features):
            v = [0]*n_features; v[mono_idx] = 1
            params["monotone_constraints"] = tuple(v)
        return XGBRegressor(objective="reg:squarederror", **params)
    elif name == "RandomForest":
        return RandomForestRegressor(
            random_state=seed, n_estimators=600, n_jobs=-1,
            max_depth=20, min_samples_leaf=5, max_features="sqrt"
        )
    else:
        raise ValueError(f"Unknown model: {name}")

def _fit_simple(model, X, y, sample_weight=None):
    if len(X) == 0:
        raise ValueError("Empty design matrix.")
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    return model

# -------------------- Train one direction --------------------
def train_one_direction(tr_raw: pd.DataFrame, te_raw: pd.DataFrame,
                        direction_key: str, algo_name: str):
    tr = tr_raw.copy(); te = te_raw.copy()
    if tr.empty or te.empty:
        return {"train_r2":np.nan,"test_r2":np.nan,"rmse_tr":np.nan,"rmse_te":np.nan,
                "yte":np.array([]),"yhat":np.array([]),
                "model":None,"sel_cols":[],"med":{}, "pass_max":float(tr_raw[PASS_COL].max() if len(tr_raw) else 1.0)}

    # (옵션) pair Δ-TE
    if USE_PAIR_TE:
        tr, te = add_pair_delta_te(tr, te)

    # 타깃
    if USE_DELTA_TARGET:
        tr["_target"] = tr[TARGET_COL] - tr[CUR_WARP]
        te["_target"] = te[TARGET_COL] - te[CUR_WARP]
        mandatory = [CUR_WARP] + (["pair_delta_te"] if USE_PAIR_TE else [])
    else:
        tr["_target"] = tr[TARGET_COL]
        te["_target"] = te[TARGET_COL]
        mandatory = [CUR_WARP] + (["pair_delta_te"] if USE_PAIR_TE else [])

    # FE (train 기준 스케일)
    tr_fe, te_fe, pass_max = apply_fe_train_test(tr, te)

    # train-only 중앙값 대치
    Xtr_df, med, cols = _ensure_numeric_X(tr_fe)
    Xte_df = te_fe.drop(columns=[c for c in EXCLUDE if c in te_fe.columns], errors="ignore")
    Xte_df = Xte_df[_hard_leak_guard_cols(list(Xte_df.columns))].select_dtypes(include=[np.number]) \
                     .reindex(columns=cols, fill_value=np.nan).fillna(med)
    ytr = tr_fe["_target"].values
    yte = te_fe["_target"].values

    # 이상치: train만 필터, test는 clip
    bnds, ylo, yhi = _compute_train_outlier_bounds(Xtr_df, ytr)
    Xtr_df_f, keep_idx = _apply_train_outlier_filter(Xtr_df, ytr, bnds, ylo, yhi)
    ytr_f = ytr[keep_idx]
    Xte_df_c = _clip_test_to_bounds(Xte_df, bnds)

    # 가중치 (OOF-safe)
    sample_weight = None
    if USE_HETERO_W:
        w = compute_hetero_weights(tr.iloc[keep_idx], te)
        if len(w) == len(keep_idx):
            sample_weight = w / (np.mean(w) + 1e-8)

    # Select-K
    chosen, Xtr, Xte = select_kbest_corr(Xtr_df_f, ytr_f, Xte_df_c, mandatory=mandatory, k=SELECT_K)
    mono_idx = chosen.index(CUR_WARP) if (USE_MONO_CUR and CUR_WARP in chosen and algo_name=="XGBoost") else None

    # Model
    model = _build_model(algo_name, n_features=len(chosen), mono_idx=mono_idx, seed=SEED)
    model = _fit_simple(model, Xtr, ytr_f, sample_weight=sample_weight)
    ytr_hat = model.predict(Xtr)
    yte_hat = model.predict(Xte)

    if USE_DELTA_TARGET:
        ytr_hat = tr_fe.iloc[keep_idx][CUR_WARP].values + ytr_hat
        yte_hat = te_fe[CUR_WARP].values + yte_hat
        ytr_true = tr_fe.iloc[keep_idx][TARGET_COL].values
        yte_true = te_fe[TARGET_COL].values
    else:
        ytr_true = ytr_f
        yte_true = yte

    r2_tr = float(r2_score(ytr_true, ytr_hat)) if len(ytr_true) else np.nan
    r2_te = float(r2_score(yte_true, yte_hat)) if len(yte_true) else np.nan
    rmse_tr = float(np.sqrt(mean_squared_error(ytr_true, ytr_hat))) if len(ytr_true) else np.nan
    rmse_te = float(np.sqrt(mean_squared_error(yte_true, yte_hat))) if len(yte_true) else np.nan

    return {"train_r2":r2_tr,"test_r2":r2_te,"rmse_tr":rmse_tr,"rmse_te":rmse_te,
            "yte":yte_true,"yhat":yte_hat, "model":model,
            "sel_cols":chosen, "med":med, "pass_max":pass_max}

# -------------------- RM train-only attach (optional) --------------------
def attach_train_only_rm_stats(tr_df: pd.DataFrame, te_df: pd.DataFrame):
    rm = pd.read_csv(DATA_DIR/"posco1_105190.csv")
    plates_tr = set(tr_df[PLATE_COL].astype(str).unique())
    rm = rm[rm["RM_날판번호"].astype(str).isin(plates_tr)].copy()
    if rm.empty:
        return tr_df, te_df
    num_cols = rm.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"RM_날판번호","RM_압연Pass번호","warping_index"}
    stat_cols = [c for c in num_cols if c not in drop_like and not c.endswith("_target")]
    rm_stats = rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"])
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    rm_stats = rm_stats.reset_index()
    tr = pd.merge(tr_df, rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left").drop(columns=["RM_날판번호"])
    te = pd.merge(te_df, rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left").drop(columns=["RM_날판번호"])
    # fill test holes with train-global medians
    med = rm_stats.drop(columns=["RM_날판번호"], errors="ignore").median(numeric_only=True)
    for c in med.index:
        if c in tr.columns: tr[c] = tr[c].fillna(float(med[c]))
        if c in te.columns: te[c] = te[c].fillna(float(med[c]))
    return tr, te

# -------------------- Quick permute audit --------------------
def quick_perm_audit(tr_df: pd.DataFrame, te_df: pd.DataFrame, algo_name: str):
    if not LEAK_AUTOCHECK:
        return np.nan
    rng = np.random.RandomState(SEED+9)
    tr = tr_df.copy(); te = te_df.copy()
    ytr = tr[TARGET_COL].values.copy()
    rng.shuffle(ytr)  # permute train target
    Xtr_df, med, cols = _ensure_numeric_X(tr)
    Xte_df = te.drop(columns=[c for c in EXCLUDE if c in te.columns], errors="ignore")
    Xte_df = Xte_df[_hard_leak_guard_cols(list(Xte_df.columns))].select_dtypes(include=[np.number]) \
                   .reindex(columns=cols, fill_value=np.nan).fillna(med)
    chosen, Xtr, Xte = select_kbest_corr(Xtr_df, ytr, Xte_df, mandatory=[CUR_WARP] if CUR_WARP in Xtr_df.columns else [], k=min(60, SELECT_K))
    mdl = _build_model(algo_name, n_features=len(chosen), mono_idx=None, seed=SEED+13)
    mdl = _fit_simple(mdl, Xtr, ytr)
    yhat = mdl.predict(Xte)
    return float(r2_score(te[TARGET_COL].values, yhat)) if len(te) else np.nan

# -------------------- Strict Test-only Rollforward R² --------------------
def strict_rollforward_r2(final_df: pd.DataFrame,
                          e2x_res: dict, x2e_res: dict,
                          e2x_tests: set, x2e_tests: set):
    inter = set(map(str, e2x_tests)) & set(map(str, x2e_tests))
    if not inter:
        return np.nan
    # FE 재구성(각 방향 train pass_max 사용)
    all_e2x = base_fe(final_df, e2x_res.get("pass_max", float(final_df[PASS_COL].max())))
    all_x2e = base_fe(final_df, x2e_res.get("pass_max", float(final_df[PASS_COL].max())))

    rows=[]
    for plate, sub in final_df.groupby(PLATE_COL):
        if str(plate) not in inter:
            continue
        idx = sub.sort_values(PASS_COL).set_index(PASS_COL, drop=False)
        idx_e2x = all_e2x[all_e2x[PLATE_COL]==plate].set_index(PASS_COL, drop=False)
        idx_x2e = all_x2e[all_x2e[PLATE_COL]==plate].set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index:  # need t+1
                continue
            if p % 2 == 1:
                # E2X
                src = idx_e2x
                sel = e2x_res["sel_cols"]; med = e2x_res["med"]; mdl = e2x_res["model"]
            else:
                src = idx_x2e
                sel = x2e_res["sel_cols"]; med = x2e_res["med"]; mdl = x2e_res["model"]

            if (p in src.index) and (mdl is not None) and len(sel)>0:
                row = src.loc[p].drop(labels=[TARGET_COL], errors="ignore")
                Xrow = pd.DataFrame([row.reindex(sel)], columns=sel).fillna(med)
                try:
                    pred = float(mdl.predict(Xrow.values)[0])
                    gt   = float(idx.loc[p+1].get(CUR_WARP))
                    rows.append({"pred":pred, "gt":gt})
                except Exception:
                    continue
    if not rows:
        return np.nan
    df = pd.DataFrame(rows)
    return float(r2_score(df["gt"], df["pred"]))

# -------------------- Main --------------------
def main():
    # experiment id
    def _next_id():
        ex = []
        for f in SUM_DIR.glob("*.txt"):
            m = re.match(r"(\d{3})_", f.name)
            if m:
                try: ex.append(int(m.group(1)))
                except: pass
        return f"{(max(ex)+1) if ex else 1:03d}"

    experiment_id = _next_id()
    ts = datetime.now()
    summary_path = SUM_DIR / f"{experiment_id}_test_summary.txt"  # 지정 경로

    try:
        final_df, n_nonconsec, n_lastdrop = load_and_merge_tplus1()

        if PERMUTE_TARGET:
            rng = np.random.RandomState(SEED)
            final_df[TARGET_COL] = rng.permutation(final_df[TARGET_COL].values)

        e2x_raw, x2e_raw = split_e2x_x2e(final_df)

        # anchor split
        e2x_tr_raw, e2x_te_raw, e2x_tests = split_by_anchor(e2x_raw, "e2x", seed=SEED, test_size=TEST_SIZE)
        x2e_tr_raw, x2e_te_raw, x2e_tests = split_by_anchor(x2e_raw, "x2e", seed=SEED, test_size=TEST_SIZE)

        # (옵션) RM train-only
        if USE_RM_TRAINONLY:
            e2x_tr_raw, e2x_te_raw = attach_train_only_rm_stats(e2x_tr_raw, e2x_te_raw)
            x2e_tr_raw, x2e_te_raw = attach_train_only_rm_stats(x2e_tr_raw, x2e_te_raw)

        # train both dirs
        e2x_res = train_one_direction(e2x_tr_raw, e2x_te_raw, "E2X", algo_name=E2X_ALGO)
        x2e_res = train_one_direction(x2e_tr_raw, x2e_te_raw, "X2E", algo_name=X2E_ALGO)

        avg_test_r2 = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))

        # quick permute audit (속도 절약: 간단 피처/모델)
        e2x_perm = quick_perm_audit(e2x_tr_raw, e2x_te_raw, E2X_ALGO)
        x2e_perm = quick_perm_audit(x2e_tr_raw, x2e_te_raw, X2E_ALGO)

        # strict rollforward on test-only plates (교집합)
        rf_strict = strict_rollforward_r2(final_df, e2x_res, x2e_res, e2x_tests, x2e_tests)

        # write metrics csv (for shell aggregation)
        MET_DIR.mkdir(parents=True, exist_ok=True)
        met_path = MET_DIR / "last_run_metrics.csv"
        pd.DataFrame([{
            "experiment_id": experiment_id,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "e2x_train_r2": e2x_res["train_r2"], "e2x_test_r2": e2x_res["test_r2"],
            "x2e_train_r2": x2e_res["train_r2"], "x2e_test_r2": x2e_res["test_r2"],
            "avg_test_r2": avg_test_r2,
            "rollforward_strict_r2": rf_strict,
            "summary_path": str(summary_path),
        }]).to_csv(met_path, index=False)

        # write summary (txt)
        lines=[]
        lines.append("Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
        lines.append("="*72)
        lines.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} 작성")
        lines.append(f"실험 ID: {experiment_id}\n")

        lines.append("[모델 성능 - R2]")
        lines.append(f"* E2X | Algo: {E2X_ALGO} | Train R2: {e2x_res['train_r2']:.4f} | Test R2: {e2x_res['test_r2']:.4f} "
                     f"(RMSE {e2x_res['rmse_tr']:.4f}/{e2x_res['rmse_te']:.4f})")
        lines.append(f"* X2E | Algo: {X2E_ALGO} | Train R2: {x2e_res['train_r2']:.4f} | Test R2: {x2e_res['test_r2']:.4f} "
                     f"(RMSE {x2e_res['rmse_tr']:.4f}/{x2e_res['rmse_te']:.4f})")
        lines.append(f"\n[평균 Test R2] {avg_test_r2:.4f}")
        lines.append(f"[Strict Test‑only Rollforward R²] {rf_strict:.4f}")

        if LEAK_AUTOCHECK:
            lines.append("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
            lines.append(f"- E2X permuted R²: {e2x_perm:.4f}")
            lines.append(f"- X2E permuted R²: {x2e_perm:.4f}")

        lines.append("\n[DEBUG] 설정/분할")
        lines.append(f"- PERMUTE_TARGET: {int(PERMUTE_TARGET)}")
        lines.append(f"- USE_DELTA_TARGET={int(USE_DELTA_TARGET)} | USE_PAIR_TE={int(USE_PAIR_TE)} "
                     f"| USE_HETERO_W={int(USE_HETERO_W)} | USE_MONO_CUR={int(USE_MONO_CUR)} | USE_RM_TRAINONLY={int(USE_RM_TRAINONLY)}")
        lines.append(f"- 불연속 PASS 드롭 수: {n_nonconsec}")
        lines.append(f"- t+1 생성 시 삭제 수(마지막 PASS 등): {n_lastdrop}")

        summary_path.write_text("\n".join(lines), encoding="utf-8")

        # 코드 아카이브
        try:
            shutil.copy2(Path(__file__).resolve(), SCR_DIR / f"{experiment_id}_{Path(__file__).name}")
        except Exception:
            pass

        print("\n" + "\n".join(lines))

    except Exception as e:
        tb = traceback.format_exc()
        (REP_DIR/"last_error_traceback.txt").write_text(tb, encoding="utf-8")
        print(f"[실패] {type(e).__name__}: {e}")
        print(tb)

if __name__ == "__main__":
    main()
