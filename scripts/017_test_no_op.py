# -*- coding: utf-8 -*-
"""
004_r2_uplift_final_noopt.py  (leak-safe, strategy-integrated, Optuna-free)

[목적]
- 003_experiment.py의 Test R2(평균 0.4174) 대비 유의한 향상을 탐색
- 기존 test.py의 실행 흐름/전략/산출물 포맷은 유지, 단 **Optuna만 제거**
- 누수 방지: plate-Group split 선행, 모든 적합/통계는 train 기준, OOF 기반 통계/가중

[전략(시동어)]
- S0: Baseline (003 호환)
- L1: 시퀀스 랙 피처 (lag/모멘텀/롤링-std)
- PN1: 판(plate) 확장 정규화 z-score(과거 평균/표준편차; expanding)
- DF1: 판 내 1차 차분/비율 Δ 피처
- R1: Robust Loss (XGB absolute)
- S1: Stratified Group Split
- TE2: Pair Δ Target Encoding (OOF + EB 수축)
- W2: Heteroscedastic Weights (쌍 Δ-분산 OOF)
- T2: Δ-Target 회귀 (ŷ = y(t) + Δ̂)
- M2: Monotone Constraint (현재 warp 단조 +1)
- SEG: Phase Segmentation (MoE) — placeholder (off by default)
- IW: Importance Weighting (Train OOF; transductive 금지)
- BAG: Bagging (K seeds)
"""

# ========================= [ USER CONFIG ] =========================
RUN_NAME = "r2_uplift_final_v2"     # 동시 다실험 시 서로 다른 이름 권장
RUN_ONLY = ["S0"]                   # 예) ["S0"] 또는 ["S1","TE2"] 등
RUN_WITH = [
    "S0+L1", "S0+PN1", "S0+DF1",    # FE 전략
    "S0+TE2", "S0+W2", "S0+T2",     # 고급 전략
    "S0+L1+PN1+DF1+TE2+W2",         # 조합 예시
]

# 고정 알고리즘(003 기준: E2X=XGB, X2E=RF)
ALGO_FIXED = {"E2X": "XGBoost", "X2E": "RandomForest"}

# 공통 설정
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
SELECTK_K = 120
PAIR_EB_N0 = 50
WEIGHT_CLIP = (0.25, 3.0)
STRAT_TRY = 25
BAG_K = 5
DO_STRICT_ROLLFORWARD = False      # 필요 시 True
IW_MODE = "oof"                    # {"oof"}만 허용(누수 방지)
# 앵커 정책(환경변수로도 제어 가능): 공정 비교 기본
#   - 기본: RUN 전체 공통 앵커
#   - ANCHOR_PER_STEP=1 이면 스텝별 앵커(과거 동작 재현)
ANCHOR_PER_STEP = ( __import__("os").environ.get("ANCHOR_PER_STEP","0") == "1" )

# ==================================================================

import os, re, shutil, warnings, traceback, joblib
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------- Paths ---------------------------------
DATA_DIRS = [Path("data"), Path("/mnt/data")]
PRO_DIR = Path("data") / "processed"
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SUM_DIR = REPORTS_DIR / "summaries"; SUM_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR = REPORTS_DIR / "scripts"; SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR = REPORTS_DIR / "csv"; CSV_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- Cols ----------------------------------
TARGET_COL = "warping_index_target"
CUR_WARP   = "warping_index_current_pass"
PLATE_COL  = "FM_날판번호"
PASS_COL   = "FM_PASS NO N"
MONTH_COL  = "FM_압연월"
EXCLUDE    = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}
THIS_PATH  = Path(__file__).resolve()

# ----------------------- 시동어 → 전략 플래그 ----------------------
# 내부 플래그: 'STRAT_SPLIT','DELTA_TARGET','PLATE_NORM','DELTA_FEATS',
#             'HETERO_W','IMP_WEIGHT','MOE_PHASE','ROBUST_LOSS',
#             'PAIR_TE','MONO_CUR','BAGGING','SEQ_LAG'
ALIAS2FLAGS = {
    "S0":   [],
    "L1":   ["SEQ_LAG"],
    "PN1":  ["PLATE_NORM"],
    "DF1":  ["DELTA_FEATS"],
    "R1":   ["ROBUST_LOSS"],
    "S1":   ["STRAT_SPLIT"],
    "TE2":  ["PAIR_TE"],
    "W2":   ["HETERO_W"],
    "T2":   ["DELTA_TARGET"],
    "M2":   ["MONO_CUR"],
    "SEG":  ["MOE_PHASE"],  # placeholder
    "IW":   ["IMP_WEIGHT"],
    "BAG":  ["BAGGING"],
}

ALIAS_DESC = {
    "S0":  "Baseline(SelectK 120, plate-Group split, E2X=XGB/X2E=RF)",
    "L1":  "Sequence lags/momentum/rolling-std (plate-wise)",
    "PN1": "Plate expanding z-norm (past mean/std z-score)",
    "DF1": "Plate 1-step deltas & ratios",
    "R1":  "Robust Loss (XGB absolute)",
    "S1":  "Stratified Group Split (pair 분포 유사도 최소화)",
    "TE2": "Pair Δ Target Encoding (OOF, EB 수축) → 특성 추가",
    "W2":  "Heteroscedastic Weights (쌍 Δ-분산 OOF+EB 기반 WLS)",
    "T2":  "Δ‑Target 회귀 (ŷ = y(t) + Δ̂)",
    "M2":  "Monotone constraint on CUR_WARP (+1) for XGB",
    "SEG": "Phase Segmentation(MoE) — (off)",
    "IW":  "Importance Weighting (Train OOF density-ratio)",
    "BAG": "Bagging(K=5) seed 앙상블",
}

# ------------------------------- Utils ---------------------------------
def _resolve_data_dir() -> Path:
    for d in DATA_DIRS:
        if (d/"entry_direction_results.csv").exists() and (d/"exit_direction_results.csv").exists():
            return d
    return DATA_DIRS[0]

def _next_experiment_id():
    nums=[]
    for f in SUM_DIR.glob("*.txt"):
        m=re.match(r"(\d{3})_", f.name)
        if m:
            try: nums.append(int(m.group(1)))
            except: pass
    return f"{(max(nums)+1) if nums else 1:03d}"

def _pair_str(df: pd.DataFrame) -> pd.Series:
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

def _ensure_numeric_X(df: pd.DataFrame):
    X_all = df.drop(columns=[c for c in EXCLUDE if c in df.columns], errors="ignore")
    X = X_all.select_dtypes(include=[np.number]).copy()
    med = X.median(numeric_only=True)
    return X.fillna(med), med, list(X.columns)

# -------------------- 데이터 병합 & 타깃 t→t+1 --------------------
def load_and_merge_tplus1():
    D = _resolve_data_dir()
    gt_entry = pd.read_csv(D/"entry_direction_results.csv")
    gt_exit  = pd.read_csv(D/"exit_direction_results.csv")
    ground_truth = pd.concat([gt_entry, gt_exit], ignore_index=True)
    ground_truth["extracted_plate"] = ground_truth["filename"].str.extract(r"(PB\d+)")
    ground_truth["extracted_pass"]  = ground_truth["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    rm = pd.read_csv(D/"posco1_105190.csv")
    fm = pd.read_csv(D/"posco2_105190.csv")

    merged_rm = pd.merge(
        ground_truth, rm,
        left_on=["extracted_plate","extracted_pass"],
        right_on=["RM_날판번호","RM_압연Pass번호"],
        how="inner"
    )
    num_cols = merged_rm.select_dtypes(include=[np.number]).columns.tolist()
    stat_cols = [c for c in num_cols if c not in ["RM_날판번호","RM_압연Pass번호","warping_index","extracted_pass"]]
    rm_stats = merged_rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(6)
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    rm_stats = rm_stats.reset_index()

    merged_fm = pd.merge(
        ground_truth, fm,
        left_on=["extracted_plate","extracted_pass"],
        right_on=[PLATE_COL, PASS_COL],
        how="inner"
    ).sort_values([PLATE_COL, PASS_COL])

    merged_fm[TARGET_COL] = merged_fm.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged_fm = merged_fm.rename(columns={"warping_index": CUR_WARP})
    merged_fm = merged_fm.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    final_df = pd.merge(
        merged_fm, rm_stats,
        left_on=PLATE_COL, right_on="RM_날판번호", how="left"
    ).drop(columns=["RM_날판번호"], errors="ignore")

    final_df = final_df.drop(columns=["extracted_plate","extracted_pass","filename",
                                      "quality_grade","quality_grade_current_pass","direction"], errors="ignore")

    PRO_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PRO_DIR/"final_merged_data_regression_tplus1.csv", index=False)
    return final_df

def split_e2x_x2e(df: pd.DataFrame):
    e2x = df[df[PASS_COL] % 2 == 1].copy()
    x2e = df[df[PASS_COL] % 2 == 0].copy()
    (PRO_DIR/"e2x_raw_tplus1.csv").write_text(e2x.to_csv(index=False))
    (PRO_DIR/"x2e_raw_tplus1.csv").write_text(x2e.to_csv(index=False))
    return e2x, x2e

# ------------------------- Feature Engineering -------------------------
def base_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
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
    d["FM_PASS_progress"] = d[PASS_COL] / (d[PASS_COL].max() if d[PASS_COL].max()>0 else 1)
    return d

def add_sequence_lags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    g = d.groupby(PLATE_COL, sort=False)
    d["lag1_prev_warp"]  = g[CUR_WARP].shift(1)
    d["mom1_warp"]       = d[CUR_WARP] - d["lag1_prev_warp"]
    d["roll_std_warp2"]  = g[CUR_WARP].apply(lambda s: s.rolling(2).std()).reset_index(level=0, drop=True)
    return d

def add_plate_expanding_norm(df: pd.DataFrame, topk=20) -> pd.DataFrame:
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    fm_cols = [c for c in d.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if not fm_cols: return d
    var_rank = pd.Series(d[fm_cols].var(numeric_only=True)).sort_values(ascending=False)
    cols = list(var_rank.index[:min(topk, len(var_rank))])
    for c in cols:
        g = d.groupby(PLATE_COL, sort=False)[c]
        mean_exp = g.expanding().mean().reset_index(level=0, drop=True).shift(1)
        std_exp  = g.expanding().std().reset_index(level=0, drop=True).shift(1)
        d[f"{c}_zexp"] = (d[c] - mean_exp) / (std_exp + 1e-8)
    return d

def add_delta_causal_features(df: pd.DataFrame, topk=20) -> pd.DataFrame:
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    fm_cols = [c for c in d.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if not fm_cols: return d
    var_rank = pd.Series(d[fm_cols].var(numeric_only=True)).sort_values(ascending=False)
    cols = list(var_rank.index[:min(topk, len(var_rank))])
    g = d.groupby(PLATE_COL, sort=False)
    for c in cols:
        prev = g[c].shift(1)
        d[f"{c}_d1"] = d[c] - prev
        d[f"{c}_ratio"] = d[c] / (prev.abs() + 1e-8)
    return d

# ------------------------ 누수 없는 이상치 처리 ------------------------
def compute_train_outlier_bounds(train_df: pd.DataFrame, q_low=0.01, q_high=0.99,
                                 y_q_low=0.005, y_q_high=0.995):
    num_cols = train_df.drop(columns=[c for c in EXCLUDE if c in train_df.columns], errors="ignore") \
                       .select_dtypes(include=[np.number]).columns.tolist()
    bounds = {}
    for c in num_cols:
        q1, q9 = train_df[c].quantile([q_low, q_high])
        bounds[c] = (float(q1), float(q9))
    y = train_df[TARGET_COL]
    y1, y9 = y.quantile([y_q_low, y_q_high])
    return bounds, float(y1), float(y9)

def apply_train_outlier_filter(train_df: pd.DataFrame, bounds: dict, y_lo: float, y_hi: float) -> pd.DataFrame:
    d = train_df.copy()
    # 타깃 outlier train에서만 제거
    d = d[(d[TARGET_COL] >= y_lo) & (d[TARGET_COL] <= y_hi)]
    # feature outlier train에서만 제거
    for c,(lo,hi) in bounds.items():
        d = d[(d[c] >= lo) & (d[c] <= hi)]
    return d.reset_index(drop=True)

def clip_test_features_to_bounds(test_df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    d = test_df.copy()
    for c,(lo,hi) in bounds.items():
        if c in d.columns:
            d[c] = d[c].clip(lower=lo, upper=hi)
    return d

# ------------------------ 전략별 보조 모듈 ------------------------
def compute_oof_pair_stats_and_weights(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                       n0=PAIR_EB_N0, clip=WEIGHT_CLIP):
    tr = train_df.copy(); te = test_df.copy()
    tr["pair"] = _pair_str(tr); te["pair"] = _pair_str(te)
    tr["delta"] = tr[TARGET_COL] - tr[CUR_WARP]
    gkf = GroupKFold(n_splits=CV_FOLDS); groups = tr[PLATE_COL].astype(str).values
    oof_std = np.zeros(len(tr))
    for tr_i, va_i in gkf.split(tr, tr["delta"].values, groups):
        sub = tr.iloc[tr_i]
        gs  = sub["delta"].std(ddof=1)
        stat = sub.groupby("pair")["delta"].agg(n="size", std="std", mean="mean").reset_index()
        w = stat["n"]/(stat["n"]+n0)
        stat["std_sh"]  = w*stat["std"].fillna(gs) + (1-w)*gs
        m_std = dict(zip(stat["pair"], stat["std_sh"]))
        pva = tr.iloc[va_i]["pair"].values
        oof_std[va_i] = [m_std.get(p, gs) for p in pva]
    # test mapping (full-train)
    gs = tr["delta"].std(ddof=1)
    stat = tr.groupby("pair")["delta"].agg(n="size", std="std", mean="mean").reset_index()
    w = stat["n"]/(stat["n"]+n0)
    stat["std_sh"] = w*stat["std"].fillna(gs) + (1-w)*gs
    m_std = dict(zip(stat["pair"], stat["std_sh"]))
    te_std = te["pair"].map(m_std).fillna(gs).values
    w_tr = (float(gs)/(oof_std + 1e-6))
    w_tr = np.clip(w_tr, clip[0], clip[1])
    return {"w_tr": w_tr, "test_std": te_std}

def compute_importance_weights(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    IW_MODE == 'oof'만 허용(누수 방지). test-set을 classifier에 태우는 transductive 방식 금지.
    """
    assert IW_MODE == "oof", "Only OOF IW is allowed to avoid leakage."
    tr = train_df.copy()
    gkf = GroupKFold(n_splits=CV_FOLDS)
    groups = tr[PLATE_COL].astype(str).values
    Xall, med, cols = _ensure_numeric_X(tr)
    oof_w = np.zeros(len(tr))
    for tr_i, va_i in gkf.split(Xall, np.zeros(len(tr)), groups):
        Xtr, Xva = Xall.iloc[tr_i], Xall.iloc[va_i]
        yclf = np.r_[np.zeros(len(tr_i)), np.ones(len(va_i))]
        Xcat = pd.concat([Xtr, Xva], ignore_index=True)
        clf = XGBClassifier(random_state=SEED, n_estimators=200, max_depth=4,
                            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", eval_metric="logloss", n_jobs=0)
        clf.fit(Xcat.values, yclf)
        pr = clf.predict_proba(Xva.values)[:,1]
        oof_w[va_i] = pr / (1 - pr + 1e-6)
    return np.clip(oof_w, 0.2, 5.0)

def add_pair_delta_te_feature(train_df: pd.DataFrame, test_df: pd.DataFrame, n0=PAIR_EB_N0):
    """TE2: Δ OOF 타깃인코딩(쌍별 평균 EB 수축) → 특성 'pair_delta_te' 추가"""
    tr = train_df.copy(); te = test_df.copy()
    tr["pair"] = _pair_str(tr); te["pair"] = _pair_str(te)
    tr["delta"] = tr[TARGET_COL] - tr[CUR_WARP]
    gkf = GroupKFold(n_splits=CV_FOLDS)
    plates = tr[PLATE_COL].astype(str).values
    oof = np.zeros(len(tr))
    for tr_i, va_i in gkf.split(tr, tr["delta"].values, groups=plates):
        sub = tr.iloc[tr_i]
        gmean = sub["delta"].mean()
        stat = sub.groupby("pair")["delta"].agg(n="size", mean="mean").reset_index()
        stat["te"] = (stat["n"]/(stat["n"]+n0))*stat["mean"] + (n0/(stat["n"]+n0))*gmean
        m = dict(zip(stat["pair"], stat["te"]))
        oof[va_i] = tr.iloc[va_i]["pair"].map(m).fillna(gmean).values
    # test mapping from full-train
    gmean = tr["delta"].mean()
    stat = tr.groupby("pair")["delta"].agg(n="size", mean="mean").reset_index()
    stat["te"] = (stat["n"]/(stat["n"]+n0))*stat["mean"] + (n0/(stat["n"]+n0))*gmean
    m_full = dict(zip(stat["pair"], stat["te"]))
    te_feat = te["pair"].map(m_full).fillna(gmean).values
    tr = tr.drop(columns=["pair","delta"]); te = te.drop(columns=["pair"])
    return tr.assign(pair_delta_te=oof), te.assign(pair_delta_te=te_feat)

# ---------------------------- Split(anchor) ----------------------------
def split_by_anchor(df: pd.DataFrame, direction_key: str, use_strat: bool, anchor_name: str):
    anchor_file = MODELS_DIR / f"{anchor_name}__{direction_key}_test_plates.pkl"
    if anchor_file.exists():
        test_plates = set(joblib.load(anchor_file))
        tr = df[~df[PLATE_COL].astype(str).isin(test_plates)].copy()
        te = df[df[PLATE_COL].astype(str).isin(test_plates)].copy()
        return tr, te, test_plates

    if not use_strat:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        tr_i, te_i = next(gss.split(df.drop(columns=[TARGET_COL]), df[TARGET_COL], groups=df[PLATE_COL].astype(str).values))
        te_plates = sorted(df.iloc[te_i][PLATE_COL].astype(str).unique())
        joblib.dump(te_plates, anchor_file)
        return df.iloc[tr_i].copy(), df.iloc[te_i].copy(), set(te_plates)

    # Stratified: pair 분포 유사화
    pairs = _pair_str(df)
    pair_vals = sorted(pairs.unique().tolist())
    best = (1e9, None, None)
    rng = np.random.RandomState(SEED)
    for _ in range(STRAT_TRY):
        seed = int(rng.randint(0, 1e9))
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr_i, te_i = next(gss.split(df.drop(columns=[TARGET_COL]), df[TARGET_COL], groups=df[PLATE_COL].astype(str).values))
        hist_all = df.groupby(pairs).size().reindex(pair_vals, fill_value=0).values
        hist_te  = df.iloc[te_i].groupby(pairs.iloc[te_i]).size().reindex(pair_vals, fill_value=0).values
        p_all = hist_all/(hist_all.sum()+1e-8); p_te = hist_te/(hist_te.sum()+1e-8)
        l1 = np.abs(p_all - p_te).sum()
        if l1 < best[0]: best = (l1, tr_i, te_i)
    tr_i, te_i = best[1], best[2]
    te_plates = sorted(df.iloc[te_i][PLATE_COL].astype(str).unique())
    joblib.dump(te_plates, anchor_file)
    return df.iloc[tr_i].copy(), df.iloc[te_i].copy(), set(te_plates)

# --------------------------- Modeling Core (No-Optuna) -----------------
def select_features(Xtr_df, ytr, Xte_df, mandatory=None, k=SELECTK_K):
    mandatory = mandatory or []
    mand = [c for c in mandatory if c in Xtr_df.columns]
    others = [c for c in Xtr_df.columns if c not in mand]
    if len(others)==0:
        chosen = mand; return chosen, Xtr_df[chosen].values, Xte_df[chosen].values
    k_remain = max(1, min(k-len(mand), len(others)))
    sel = SelectKBest(score_func=f_regression, k=k_remain)
    sel.fit(Xtr_df[others], ytr)
    chosen = mand + [others[i] for i in sel.get_support(indices=True)]
    return chosen, Xtr_df[chosen].values, Xte_df[chosen].values

def _build_xgb(robust=False, monotone_index=None, n_features=None, seed=SEED):
    params=dict(random_state=seed, tree_method="hist",
                n_estimators=500, learning_rate=0.05,
                max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=0)
    if monotone_index is not None and n_features is not None:
        v=[0]*n_features
        if 0 <= monotone_index < n_features: v[monotone_index]=1
        params["monotone_constraints"]=tuple(v)
    return XGBRegressor(objective=("reg:absoluteerror" if robust else "reg:squarederror"), **params)

def _build_rf(seed=SEED):
    return RandomForestRegressor(random_state=seed, n_estimators=600, n_jobs=-1)

def _fit_predict_model(Xtr, ytr, Xte, flags, algo_name, chosen_names, sample_weight=None):
    # 단조 제약: 선택된 피처 중 CUR_WARP 위치 탐지
    mono_idx = chosen_names.index(CUR_WARP) if ("MONO_CUR" in flags and CUR_WARP in chosen_names and algo_name=="XGBoost") else None

    def make_one(seed):
        if algo_name == "XGBoost":
            return _build_xgb(("ROBUST_LOSS" in flags), monotone_index=mono_idx, n_features=len(chosen_names), seed=seed)
        else:
            return _build_rf(seed=seed)

    if "BAGGING" in flags:
        seeds = [SEED+i for i in range(BAG_K)]
        preds=[]; mdl=None
        for s in seeds:
            m = make_one(s); m.fit(Xtr, ytr, sample_weight=sample_weight); preds.append(m.predict(Xte))
            mdl = m
        pte = np.mean(preds, axis=0)
        return mdl, pte
    else:
        mdl = make_one(SEED); mdl.fit(Xtr, ytr, sample_weight=sample_weight); return mdl, mdl.predict(Xte)

def train_single_model(train_df, test_df, direction_key: str, flags: set, algo_name: str):
    """누수-세이프 학습/예측"""
    tr = train_df.copy()
    te = test_df.copy()

    # TE2: Δ-TE 특성 추가 (train OOF, test full-train 매핑)
    if "PAIR_TE" in flags:
        tr, te = add_pair_delta_te_feature(tr, te, n0=PAIR_EB_N0)

    # 타깃 구성
    if "DELTA_TARGET" in flags:
        tr["_target"] = tr[TARGET_COL] - tr[CUR_WARP]
        te["_target"] = te[TARGET_COL] - te[CUR_WARP]
        mandatory = [CUR_WARP] + (["pair_delta_te"] if "PAIR_TE" in flags else [])
    else:
        tr["_target"] = tr[TARGET_COL]
        te["_target"] = te[TARGET_COL]
        mandatory = (["pair_delta_te"] if "PAIR_TE" in flags else [])

    # 가중치
    sample_weight = None
    if ("HETERO_W" in flags) or ("IMP_WEIGHT" in flags):
        w = np.ones(len(tr), dtype=float)
        if "HETERO_W" in flags:
            w *= compute_oof_pair_stats_and_weights(tr, te)["w_tr"]
        if "IMP_WEIGHT" in flags:
            iw = compute_importance_weights(tr, te)
            if len(iw) == len(tr): w *= iw
        sample_weight = w / (np.mean(w) + 1e-8)

    # 피처 구축/선택
    Xtr_df, med, cols = _ensure_numeric_X(tr)
    Xte_df = te.drop(columns=[c for c in EXCLUDE if c in te.columns], errors="ignore") \
               .select_dtypes(include=[np.number]).reindex(columns=cols, fill_value=np.nan).fillna(med)
    ytr = tr["_target"].values; yte = te["_target"].values

    chosen_names, Xtr, Xte = select_features(Xtr_df, ytr, Xte_df, mandatory=mandatory, k=SELECTK_K)
    model, yhat = _fit_predict_model(Xtr, ytr, Xte, flags, algo_name, chosen_names, sample_weight=sample_weight)

    # Δ-Target 복구
    if "DELTA_TARGET" in flags:
        yhat = te[CUR_WARP].values + yhat

    # 점수
    r2_tr = float(r2_score(ytr, model.predict(Xtr)))
    r2_te = float(r2_score(yte, yhat))
    rmse_tr = float(np.sqrt(mean_squared_error(ytr, model.predict(Xtr))))
    rmse_te = float(np.sqrt(mean_squared_error(yte, yhat)))

    return {
        "model": model, "features": chosen_names,
        "r2_tr": r2_tr, "r2_te": r2_te,
        "rmse_tr": rmse_tr, "rmse_te": rmse_te,
        "yte": yte, "yhat": yhat
    }

# ------------------------------ Runner ---------------------------------
def _apply_fe_by_flags(df: pd.DataFrame, flags: set) -> pd.DataFrame:
    d = base_feature_engineering(df)
    if "SEQ_LAG"     in flags: d = add_sequence_lags(d)
    if "PLATE_NORM"  in flags: d = add_plate_expanding_norm(d)
    if "DELTA_FEATS" in flags: d = add_delta_causal_features(d)
    return d

def _run_one(key: str, flags: set, tag: str, e2x_raw: pd.DataFrame, x2e_raw: pd.DataFrame, anchor_base: str):
    # 1) FE
    e2x = _apply_fe_by_flags(e2x_raw, flags)
    x2e = _apply_fe_by_flags(x2e_raw, flags)

    # 2) Split(plate-group) with anchor
    anchor_name = f"{anchor_base}__{key}__anchor" if ANCHOR_PER_STEP else f"{anchor_base}__anchor"
    use_strat = ("STRAT_SPLIT" in flags)
    e2x_tr, e2x_te, e2x_plates = split_by_anchor(e2x, "E2X", use_strat, anchor_name)
    x2e_tr, x2e_te, x2e_plates = split_by_anchor(x2e, "X2E", use_strat, anchor_name)

    # 3) 이상치 경계(train-only) & clip(test)
    b_e2x, ylo_e2x, yhi_e2x = compute_train_outlier_bounds(e2x_tr)
    b_x2e, ylo_x2e, yhi_x2e = compute_train_outlier_bounds(x2e_tr)
    e2x_tr = apply_train_outlier_filter(e2x_tr, b_e2x, ylo_e2x, yhi_e2x)
    x2e_tr = apply_train_outlier_filter(x2e_tr, b_x2e, ylo_x2e, yhi_x2e)
    e2x_te = clip_test_features_to_bounds(e2x_te, b_e2x)
    x2e_te = clip_test_features_to_bounds(x2e_te, b_x2e)

    # 4) 학습/예측
    e2x_res = train_single_model(e2x_tr, e2x_te, "E2X", flags, ALGO_FIXED["E2X"])
    x2e_res = train_single_model(x2e_tr, x2e_te, "X2E", flags, ALGO_FIXED["X2E"])

    # 5) 점수/요약/저장
    avg_r2 = (e2x_res["r2_te"] + x2e_res["r2_te"]) / 2.0

    # Strict RF(선택)
    rf_strict = np.nan
    if DO_STRICT_ROLLFORWARD:
        try:
            # 간단: 교집합 Test 판 위에서 t→t+1 한 칸 예측 r2 (전체 all_featured 필요)
            # 여기서는 생략하거나 별도 구현 연결 가능
            rf_strict = np.nan
        except:
            rf_strict = np.nan

    # 스텝 CSV
    steps_csv = CSV_DIR / f"{experiment_id}_{RUN_NAME}_{tag}_steps.csv"
    pd.DataFrame({
        "direction": ["E2X"]*len(e2x_res["yte"]) + ["X2E"]*len(x2e_res["yte"]),
        "y_true":    np.r_[e2x_res["yte"], x2e_res["yte"]],
        "y_pred":    np.r_[e2x_res["yhat"], x2e_res["yhat"]],
    }).to_csv(steps_csv, index=False)

    # 코드 스냅샷
    code_snap = SCRIPTS_DIR / f"{experiment_id}_{Path(__file__).name}"
    try: shutil.copy2(THIS_PATH, code_snap)
    except Exception: pass

    # 요약 TXT
    summary_txt = SUM_DIR / f"{experiment_id}_{RUN_NAME}_{tag}_summary.txt"
    lines=[]
    ts = datetime.now(); lines.append(f"R² Uplift (alias) Summary ({ts.strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append("="*72)
    lines.append(f"실험 ID: {experiment_id} | RUN_NAME: {RUN_NAME} | STEP: {tag}")
    lines.append(f"전략 플래그: {', '.join(sorted(list(flags))) if flags else 'S0 (전략 없음)'}\n")
    def fmt(label, r):
        return (f"* {label} | Algo: {ALGO_FIXED[label]} "
                f"| Train R2: {r['r2_tr']:.4f} | Test R2: {r['r2_te']:.4f} "
                f"(RMSE {r['rmse_tr']:.4f}/{r['rmse_te']:.4f})")
    lines.append("[모델 성능 - R2]")
    lines.append(fmt("E2X", e2x_res))
    lines.append(fmt("X2E", x2e_res))
    lines.append(f"\n[평균 Test R2] {avg_r2:.4f}")
    if DO_STRICT_ROLLFORWARD and not np.isnan(rf_strict):
        lines.append(f"[Strict Roll-forward R²] {rf_strict:.4f}")
    lines.append("="*72)
    lines.append("\n[시동어 설명]")
    if "+" in key:
        for p in key.split("+"):
            p=p.strip()
            if p in ALIAS_DESC: lines.append(f"- {p}: {ALIAS_DESC[p]}")
    else:
        lines.append(f"- {key}: {ALIAS_DESC.get(key, 'N/A')}")
    lines.append(f"\n[산출물]\n- 단계별 결과 CSV: {steps_csv}\n- 코드 스냅샷: {code_snap}")
    summary_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 저장: {steps_csv}\n✓ 저장: {summary_txt}")

    return {"avg": avg_r2, "e2x": e2x_res["r2_te"], "x2e": x2e_res["r2_te"], "rf": rf_strict, "anchor": anchor_name}

# -------------------------------- main ---------------------------------
def _parse_recipes(run_only, run_with):
    """
    'S0+PN1'처럼 +로 연결된 조합을 파싱. 'S0'는 효과가 없으니 있어도/없어도 동일.
    예) 'PN1' == 'S0+PN1'
    """
    recipes = {}
    def to_flags(expr: str):
        toks = [t.strip() for t in expr.split("+") if t.strip()]
        toks = [t for t in toks if t != "S0"]            # S0는 no-op
        flags = []
        for t in toks:
            flags.extend(ALIAS2FLAGS.get(t, []))
        return flags
    for k in (run_only or []):
        recipes[k] = to_flags(k)
    for k in (run_with or []):
        recipes[k] = to_flags(k)
    # S0가 하나도 없으면 baseline도 추가
    if "S0" not in recipes:
        recipes["S0"] = []
    return recipes

def main():
    global experiment_id
    experiment_id = _next_experiment_id()

    print("="*80)
    print(f"[RUN_NAME] {RUN_NAME} | Optuna-free")
    print("="*80)

    final_df = load_and_merge_tplus1()
    e2x_raw, x2e_raw = split_e2x_x2e(final_df)
    recipes = _parse_recipes(RUN_ONLY, RUN_WITH)

    rows = []
    anchor_base = RUN_NAME

    for key, flag_list in recipes.items():
        flags = set(flag_list)
        print("\n" + "="*80)
        print(f"[RUN] {key} | step={key} | flags={sorted(list(flags))} | anchor="
              f"{(f'{RUN_NAME}__{key}__anchor' if ANCHOR_PER_STEP else f'{RUN_NAME}__anchor')}")
        print("="*80)
        res = _run_one(key, flags, key, e2x_raw, x2e_raw, RUN_NAME)
        rows.append({"step": key, "avg_test_r2": res["avg"],
                     "E2X_test_r2": res["e2x"], "X2E_test_r2": res["x2e"],
                     "RF_strict": res["rf"], "anchor": res["anchor"]})

    # ---------- 리더보드 출력 ----------
    lb = pd.DataFrame(rows)
    if not lb.empty:
        if (lb["step"] == "S0").any():
            base = float(lb.loc[lb["step"]=="S0", "avg_test_r2"].iloc[0])
            lb["delta_vs_S0"] = lb["avg_test_r2"] - base
        else:
            lb["delta_vs_S0"] = np.nan
        lb = lb.sort_values("avg_test_r2", ascending=False)
        print("\n[Leaderboard — 높은 Avg Test R² 순]")
        print(lb[["step","avg_test_r2","delta_vs_S0","E2X_test_r2","X2E_test_r2","RF_strict","anchor"]].to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        Path("reports").mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR/"last_error_traceback.txt").write_text(tb, encoding="utf-8")
        print(f"[실패] {type(e).__name__}: {e}")
        print(tb)
