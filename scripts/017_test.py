# -*- coding: utf-8 -*-
"""
004_r2_uplift_final.py  (leak-safe, Optuna-enabled, strategy-integrated)

[목적]
- 003_experiment.py의 Test R2 (평균 0.4174)를 초과하는 모델 개발
- 누수 방지 강화: RM 통계 Train-only, 이상치 제거 Train-only
- 003과 공정 비교를 위해 Optuna 사용
- update2.py의 FE 전략(L1, PN1, DF1) 및 모든 고급 전략 통합

[누수 방지 원칙]
  (1) 반드시 'plate-Group split'을 먼저 수행 → 이후 모든 적합/통계는 train 기준
  (2) RM 통계는 Train plate만으로 계산, Test에는 Train 통계 매핑
  (3) 이상치 처리는 Train 기준으로만 경계 계산, Test는 clip만 적용
  (4) Δ-TE/분산가중/중요도가중은 OOF 또는 train-only 적합 후 test에 매핑
  (5) test 행은 삭제하지 않음(분포 보존). 필요 시 feature만 train 범위로 clip

[전략]
- S0: Baseline (003 호환)
- L1: 시퀀스 랙 피처
- PN1: 판별 확장 정규화
- DF1: 판별 차분/비율 피처
- R1: Robust Loss
- S1: Stratified Group Split
- TE2: Pair Δ Target Encoding
- W2: Heteroscedastic Weights
- T2: Δ-Target 회귀
- M2: Monotone Constraint
- SEG: Phase Segmentation (MoE)
- IW: Importance Weighting
- BAG: Bagging
"""

# ========================= [ USER CONFIG ] =========================
RUN_NAME = "r2_uplift_final_v1"   # 동시 다실험 시 서로 다른 이름으로 변경
RUN_ONLY = ["S0"]  # 기본 베이스라인
RUN_WITH = [
    "S0+L1", "S0+PN1", "S0+DF1",  # FE 전략
    "S0+TE2", "S0+W2", "S0+T2",   # 고급 전략
    "S0+L1+PN1+DF1+TE2+W2",       # 조합
]

# 고정 알고리즘(003 기준과 동일: E2X=XGB, X2E=RF)
ALGO_FIXED = {"E2X": "XGBoost", "X2E": "RandomForest"}

# 공통 설정
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
SELECTK_K = 120
# EB/가중 안정화
PAIR_EB_N0 = 50
WEIGHT_CLIP = (0.25, 3.0)
# Stratified split 탐색 시도
STRAT_TRY = 25
# Bagging 크기
BAG_K = 5
# Strict roll-forward 계산 여부
DO_STRICT_ROLLFORWARD = False

IW_MODE = "oof"   # {"oof", "test"}; "oof" 권장 (누수 방지)
# ==================================================================

import os, re, shutil, warnings, traceback, joblib
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from xgboost import XGBRegressor, XGBClassifier
import optuna
from optuna.trial import TrialState
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------- Paths ---------------------------------
DATA_DIRS = [Path("data"), Path("/mnt/data")]
PRO_DIR = Path("data") / "processed"; PRO_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SUM_DIR = REPORTS_DIR / "summaries"; SUM_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR = REPORTS_DIR / "scripts"; SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR = REPORTS_DIR / "csv"; CSV_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
THIS_PATH  = Path(__file__).resolve()

# ------------------------------- Cols ----------------------------------
TARGET_COL = "warping_index_target"
CUR_WARP   = "warping_index_current_pass"
PLATE_COL  = "FM_날판번호"
PASS_COL   = "FM_PASS NO N"
MONTH_COL  = "FM_압연월"
EXCLUDE    = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}

# ----------------------- 시동어 → 전략 플래그 ----------------------
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
    "SEG":  ["MOE_PHASE"],
    "IW":   ["IMP_WEIGHT"],
    "BAG":  ["BAGGING"],
}

ALIAS_DESC = {
    "S0":  "Baseline(SelectK 120, plate-Group split, E2X=XGB/X2E=RF, Optuna)",
    "L1":  "Sequential Lags (lag1_prev_warp, mom1_warp, etc.)",
    "PN1": "Plate-wise Expanding Z-Score Norm (Top 20 vars)",
    "DF1": "Plate-wise Delta/Ratio Features (Top 20 vars)",
    "R1":  "Robust Loss (XGB absolute)",
    "S1":  "Stratified Group Split (쌍(pair) 분포 유사도 최소화)",
    "TE2": "Pair Δ Target Encoding(OOF, EB 수축) → 특성 추가",
    "W2":  "Heteroscedastic Weights (쌍 Δ-분산 OOF+EB 기반 WLS)",
    "T2":  "Δ‑Target 회귀 (ŷ= y(t) + Δ̂)",
    "M2":  "Monotone constraint on CUR_WARP (+1) for XGB",
    "SEG": "Phase Segmentation(MoE) — 초/중/후 소전문가 + 글로벌 백업",
    "IW":  "Importance Weighting (Train↔Test 로지스틱 밀도비, OOF)",
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

def _save_code_snapshot(exp_id: str):
    dst = SCRIPTS_DIR / f"{exp_id}_{Path(__file__).name}"
    try:
        shutil.copy2(THIS_PATH, dst)
    except Exception:
        pass
    return dst

# -------------------- 데이터 병합 & 타깃 t→t+1 (누수 방지) --------------------
def load_and_merge_base():
    """RM 통계 없이 GT+FM만 병합 (누수 방지)"""
    D = _resolve_data_dir()
    gt_entry = pd.read_csv(D/"entry_direction_results.csv")
    gt_exit  = pd.read_csv(D/"exit_direction_results.csv")
    ground_truth = pd.concat([gt_entry, gt_exit], ignore_index=True)
    ground_truth["extracted_plate"] = ground_truth["filename"].str.extract(r"(PB\d+)")
    ground_truth["extracted_pass"]  = ground_truth["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    fm = pd.read_csv(D/"posco2_105190.csv")

    merged_fm = pd.merge(
        ground_truth, fm,
        left_on=["extracted_plate","extracted_pass"],
        right_on=[PLATE_COL, PASS_COL],
        how="inner"
    ).sort_values([PLATE_COL, PASS_COL])

    merged_fm[TARGET_COL] = merged_fm.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged_fm = merged_fm.rename(columns={"warping_index": CUR_WARP})
    merged_fm = merged_fm.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    final_df = merged_fm.drop(columns=["extracted_plate","extracted_pass","filename",
                                      "quality_grade","quality_grade_current_pass","direction"], errors="ignore")

    PRO_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PRO_DIR/"final_merged_base_tplus1.csv", index=False)
    return final_df, ground_truth

def compute_rm_stats_leak_safe(merged_rm_train: pd.DataFrame) -> pd.DataFrame:
    """Train 데이터로만 RM 통계 계산 (누수 방지)"""
    num_cols = merged_rm_train.select_dtypes(include=[np.number]).columns.tolist()
    stat_cols = [c for c in num_cols if c not in ["RM_날판번호","RM_압연Pass번호","warping_index","extracted_pass"]]
    rm_stats = merged_rm_train.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(6)
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    return rm_stats.reset_index()

def add_rm_stats_leak_safe(df: pd.DataFrame, train_plates: set, rm_data: pd.DataFrame, gt: pd.DataFrame):
    """Train plate 기준 RM 통계 계산 후 전체에 매핑"""
    # Train plate만 추출
    train_gt = gt[gt["extracted_plate"].isin(train_plates)].copy()
    merged_rm_train = pd.merge(
        train_gt, rm_data,
        left_on=["extracted_plate","extracted_pass"],
        right_on=["RM_날판번호","RM_압연Pass번호"],
        how="inner"
    )
    
    # Train 통계 계산
    rm_stats = compute_rm_stats_leak_safe(merged_rm_train)
    
    # 전체 데이터에 매핑
    result = pd.merge(
        df, rm_stats,
        left_on=PLATE_COL, right_on="RM_날판번호", how="left"
    ).drop(columns=["RM_날판번호"], errors="ignore")
    return result

def split_e2x_x2e(df: pd.DataFrame):
    e2x = df[df[PASS_COL] % 2 == 1].copy()
    x2e = df[df[PASS_COL] % 2 == 0].copy()
    (PRO_DIR/"e2x_raw_tplus1.csv").write_text(e2x.to_csv(index=False))
    (PRO_DIR/"x2e_raw_tplus1.csv").write_text(x2e.to_csv(index=False))
    return e2x, x2e

# ------------------------- Base FE (S0) -------------------------
def base_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """003_experiment.py의 feature_engineering() 함수"""
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

# ------------------------- 신규 FE 전략 (L1, PN1, DF1) -------------------------
def add_sequence_lags(df: pd.DataFrame) -> pd.DataFrame:
    """전략 L1: 시퀀스 랙 피처"""
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    g = d.groupby(PLATE_COL, sort=False)
    d["lag1_prev_warp"]  = g[CUR_WARP].shift(1)
    d["mom1_warp"]       = d[CUR_WARP] - d["lag1_prev_warp"]
    d["roll_std_warp2"]  = g[CUR_WARP].apply(lambda s: s.rolling(2).std()).reset_index(level=0, drop=True)
    return d

def add_plate_expanding_norm(df: pd.DataFrame, topk=20) -> pd.DataFrame:
    """전략 PN1: 판별 확장 정규화 (누적 Z-score)"""
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    fm_cols = [c for c in d.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if not fm_cols: return d
    num_fm_cols = d[fm_cols].select_dtypes(include=[np.number]).columns
    if not len(num_fm_cols): return d
    var_rank = pd.Series(d[num_fm_cols].var(numeric_only=True)).sort_values(ascending=False)
    cols = list(var_rank.index[:min(topk, len(var_rank))])
    for c in cols:
        g = d.groupby(PLATE_COL, sort=False)[c]
        mean_exp = g.expanding().mean().reset_index(level=0, drop=True).shift(1)
        std_exp  = g.expanding().std().reset_index(level=0, drop=True).shift(1)
        d[f"{c}_zexp"] = (d[c] - mean_exp) / (std_exp + 1e-8)
    return d

def add_delta_causal_features(df: pd.DataFrame, topk=20) -> pd.DataFrame:
    """전략 DF1: 판별 차분/비율 피처 (이전 패스 대비)"""
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    fm_cols = [c for c in d.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if not fm_cols: return d
    num_fm_cols = d[fm_cols].select_dtypes(include=[np.number]).columns
    if not len(num_fm_cols): return d
    var_rank = pd.Series(d[num_fm_cols].var(numeric_only=True)).sort_values(ascending=False)
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
    """Train 기준으로만 이상치 경계 계산"""
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
    """Train에서만 이상치 행 '제거'"""
    d = train_df.copy()
    d = d[(d[TARGET_COL] >= y_lo) & (d[TARGET_COL] <= y_hi)]
    for c,(lo,hi) in bounds.items():
        if c in d.columns:
            d = d[(d[c] >= lo) & (d[c] <= hi)]
    return d.reset_index(drop=True)

def clip_test_features_to_bounds(test_df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Test는 행을 삭제하지 않고 Train 경계로 'Clip'"""
    d = test_df.copy()
    for c,(lo,hi) in bounds.items():
        if c in d.columns:
            d[c] = d[c].clip(lower=lo, upper=hi)
    return d

# ------------------------ 전략별 보조 모듈 ------------------------
def compute_oof_pair_stats_and_weights(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                       n0=PAIR_EB_N0, clip=WEIGHT_CLIP):
    """전략 W2: OOF 기반 이분산성 가중치 계산 (Leak-safe)"""
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
    """전략 IW: 중요도 가중치 계산 (Leak-safe OOF mode)"""
    tr = train_df.copy(); te = test_df.copy()
    if IW_MODE == "test":
        tr["_label"] = 0; te["_label"] = 1
        data = pd.concat([tr, te], ignore_index=True)
        X, med, cols = _ensure_numeric_X(data)
        y = data["_label"].values
        clf = XGBClassifier(random_state=SEED, n_estimators=200, max_depth=4,
                            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", eval_metric="logloss", n_jobs=0)
        clf.fit(X.values, y)
        pr = clf.predict_proba(X.values)[:,1][:len(tr)]
        w = pr / (1 - pr + 1e-6)
        return np.clip(w, 0.2, 5.0)
    # OOF mode (leak-safe)
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
    """전략 TE2: Δ OOF 타깃인코딩(쌍별 평균 EB 수축) → 특성 'pair_delta_te' 추가 (Leak-safe)"""
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
    """Plate-Group 분할 수행"""
    anchor_file = MODELS_DIR / f"{anchor_name}__{direction_key}_test_plates.pkl"
    
    if anchor_file.exists():
        test_plates = set(joblib.load(anchor_file))
        tr = df[~df[PLATE_COL].astype(str).isin(test_plates)].copy()
        te = df[df[PLATE_COL].astype(str).isin(test_plates)].copy()
        return tr, te, test_plates

    # 기본: plate-group split
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

# --------------------------- Modeling Core (Optuna 사용) ---------------------------
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

def _optuna_tune_model(X_tr, y_tr, algorithm: str, groups=None, seed=SEED):
    """Optuna로 하이퍼파라미터 튜닝 (update1.py 기반)"""
    def build_model(trial):
        if algorithm == "XGBoost":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            }
            return XGBRegressor(random_state=seed, tree_method="hist", n_jobs=0, **params)
        else:  # RandomForest
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 8, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt","log2",None]),
            }
            return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    
    gkf = GroupKFold(n_splits=CV_FOLDS) if groups is not None else None
    def score(trial):
        model = build_model(trial)
        if gkf is not None:
            sc = []
            for tr_i, va_i in gkf.split(X_tr, y_tr, groups):
                sc.append(r2_score(y_tr[va_i], model.fit(X_tr[tr_i], y_tr[tr_i]).predict(X_tr[va_i])))
            return float(np.mean(sc))
        else:
            return float(cross_val_score(model, X_tr, y_tr, cv=CV_FOLDS, scoring="r2").mean())
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(score, n_trials=50, show_progress_bar=False)
    
    if any(t.state == TrialState.COMPLETE for t in study.trials):
        return build_model(optuna.trial.FixedTrial(study.best_trial.params))
    else:
        # Fallback to default
        if algorithm == "XGBoost":
            return XGBRegressor(random_state=seed, tree_method="hist", n_estimators=500, learning_rate=0.05,
                               max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=0)
        else:
            return RandomForestRegressor(random_state=seed, n_estimators=600, n_jobs=-1)

def _fit_predict_model(Xtr, ytr, Xte, flags, algo_name, chosen_names, sample_weight=None, groups=None):
    """단일/Bagging/Monotone/Robust 옵션 반영해 학습 및 예측"""
    # 단조 제약: 선택된 피처 중 CUR_WARP 위치 탐지
    mono_idx = chosen_names.index(CUR_WARP) if ("MONO_CUR" in flags and CUR_WARP in chosen_names and algo_name=="XGBoost") else None

    # Optuna 튜닝
    tuned = _optuna_tune_model(Xtr, ytr, algo_name, groups=groups)
    
    def make_one(seed):
        m = clone(tuned) if hasattr(tuned, 'get_params') else tuned
        if hasattr(m, "random_state"):
            m.set_params(random_state=seed)
        # R1 적용
        if "ROBUST_LOSS" in flags and algo_name == "XGBoost" and hasattr(m, "set_params"):
            m.set_params(objective="reg:absoluteerror")
        # M2 적용
        if "MONO_CUR" in flags and mono_idx is not None and algo_name == "XGBoost" and hasattr(m, "set_params"):
            constraints = [0] * len(chosen_names)
            constraints[mono_idx] = 1
            m.set_params(monotone_constraints=tuple(constraints))
        return m

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

    # --- TE2: Δ-타깃 인코딩 (train=OOF, test=full-train 매핑) ---
    if "PAIR_TE" in flags:
        tr, te = add_pair_delta_te_feature(tr, te, n0=PAIR_EB_N0)

    # --- 타깃 구성(지역 변수만) ---
    use_delta = ("DELTA_TARGET" in flags)
    if use_delta:
        y_tr = tr[TARGET_COL].values - tr[CUR_WARP].values
        mandatory = [CUR_WARP] + (["pair_delta_te"] if "PAIR_TE" in flags else [])
    else:
        y_tr = tr[TARGET_COL].values
        mandatory = (["pair_delta_te"] if "PAIR_TE" in flags else [])

    # --- 가중치(OOF 계열만 허용) ---
    sample_weight = None
    if ("HETERO_W" in flags) or ("IMP_WEIGHT" in flags):
        w = np.ones(len(tr), dtype=float)
        if "HETERO_W" in flags:
            w *= compute_oof_pair_stats_and_weights(tr, te)["w_tr"]
        if "IMP_WEIGHT" in flags:
            if IW_MODE == "test":
                raise RuntimeError("IMP_WEIGHT with IW_MODE='test' is disallowed (leak risk). Set IW_MODE='oof'.")
            iw = compute_importance_weights(tr, te)
            if len(iw) == len(tr): w *= iw
        sample_weight = w / (np.mean(w) + 1e-8)

    # --- 피처 행렬(금지 컬럼 제거) ---
    FORBIDDEN = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL,
                 "_target", "delta", "pair"}
    trX_df = tr.drop(columns=[c for c in FORBIDDEN if c in tr.columns], errors="ignore")
    teX_df = te.drop(columns=[c for c in FORBIDDEN if c in te.columns], errors="ignore")

    # 수치형만, 같은 중앙값/열순서 사용
    Xtr_all, med, cols = _ensure_numeric_X(trX_df)
    Xte_all = teX_df[cols].fillna(med)

    # --- 피처 선택(필수 피처 강제 포함) ---
    chosen, Xtr, Xte = select_features(Xtr_all, y_tr, Xte_all,
                                       mandatory=mandatory, k=SELECTK_K)

    # 가드: 금지 컬럼이 선택되었는지 확인
    bad = [c for c in chosen if c in FORBIDDEN or c == TARGET_COL]
    if bad:
        raise RuntimeError(f"Leaky features found in 'chosen': {bad}")

    # --- 학습/예측 ---
    groups = tr[PLATE_COL].astype(str).values
    model, yte_hat_core = _fit_predict_model(Xtr, y_tr, Xte, flags, algo_name, chosen,
                                             sample_weight=sample_weight, groups=groups)
    ytr_hat_core = model.predict(Xtr)

    # Δ 복원
    if use_delta:
        ytr_hat = tr[CUR_WARP].values + ytr_hat_core
        yte_hat = te[CUR_WARP].values + yte_hat_core
    else:
        ytr_hat = ytr_hat_core
        yte_hat = yte_hat_core

    # --- 스코어 ---
    r2_tr = float(r2_score(tr[TARGET_COL].values, ytr_hat))
    r2_te = float(r2_score(te[TARGET_COL].values, yte_hat))
    rmse_tr = float(np.sqrt(mean_squared_error(tr[TARGET_COL].values, ytr_hat)))
    rmse_te = float(np.sqrt(mean_squared_error(te[TARGET_COL].values, yte_hat)))

    # 키(plate, pass) – Strict roll-forward용
    k_tr = list(zip(tr[PLATE_COL].astype(str).values, tr[PASS_COL].astype(int).values))
    k_te = list(zip(te[PLATE_COL].astype(str).values, te[PASS_COL].astype(int).values))

    bundle = dict(algo=algo_name, med=med, cols=cols, chosen=chosen, model=model, flags=sorted(list(flags)))
    return dict(bundle=bundle, r2_tr=r2_tr, r2_te=r2_te, rmse_tr=rmse_tr, rmse_te=rmse_te,
                pred_tr=ytr_hat, pred_te=yte_hat, keys_tr=k_tr, keys_te=k_te)

def train_moe_by_phase(train_df, test_df, direction_key: str, flags: set, algo_name: str):
    """전략 SEG: 구간별 전문가 모델"""
    def phase_bucket(p):
        if p <= 2: return "early"
        elif p <= 6: return "mid"
        else: return "late"
    tr = train_df.copy(); te = test_df.copy()
    tr["_phase"] = tr[PASS_COL].apply(phase_bucket)
    te["_phase"] = te[PASS_COL].apply(phase_bucket)

    base = train_single_model(tr, te, direction_key, flags - {"MOE_PHASE"}, algo_name)
    preds_tr = np.zeros(len(tr)); preds_te = np.zeros(len(te)); used=set()
    for ph in ["early","mid","late"]:
        tr_ph = tr[tr["_phase"]==ph]; te_ph = te[te["_phase"]==ph]
        if len(tr_ph) < max(100, 0.02*len(tr)):
            preds_tr[tr_ph.index] = base["pred_tr"][tr_ph.index] if isinstance(base["pred_tr"], np.ndarray) else base["pred_tr"]
            preds_te[te_ph.index] = base["pred_te"][te_ph.index] if isinstance(base["pred_te"], np.ndarray) else base["pred_te"]
            continue
        sub = train_single_model(tr_ph, te_ph, direction_key, flags - {"MOE_PHASE"}, algo_name)
        preds_tr[tr_ph.index] = sub["pred_tr"]; preds_te[te_ph.index] = sub["pred_te"]; used.add(ph)

    r2_tr = float(r2_score(tr[TARGET_COL].values, preds_tr))
    r2_te = float(r2_score(te[TARGET_COL].values, preds_te))
    rmse_tr = float(np.sqrt(mean_squared_error(tr[TARGET_COL].values, preds_tr)))
    rmse_te = float(np.sqrt(mean_squared_error(te[TARGET_COL].values, preds_te)))
    bundle = dict(algo=algo_name, flags=sorted(list(flags)), moe_used=sorted(list(used)))
    return dict(bundle=bundle, r2_tr=r2_tr, r2_te=r2_te, rmse_tr=rmse_tr, rmse_te=rmse_te,
                pred_tr=preds_tr, pred_te=preds_te, keys_tr=None, keys_te=None)

# ----------------------------- Runner --------------------------------
def prepare_train_test_leak_safe(df_base: pd.DataFrame, use_strat: bool, direction_key: str, split_anchor: str, flags: set,
                                 rm_data: pd.DataFrame, gt: pd.DataFrame):
    """누수 방지 파이프라인"""
    tr_raw, te_raw, test_plates = split_by_anchor(df_base, direction_key, use_strat=use_strat, anchor_name=split_anchor)
    
    # RM 통계 추가 (Train-only)
    tr = add_rm_stats_leak_safe(tr_raw, set(tr_raw[PLATE_COL]), rm_data, gt)
    te = add_rm_stats_leak_safe(te_raw, set(tr_raw[PLATE_COL]), rm_data, gt)  # Train 통계 사용
    
    # FE → lag → (선택) expanding/delta
    tr = base_feature_engineering(tr)
    te = base_feature_engineering(te)
    if "SEQ_LAG" in flags:
        tr = add_sequence_lags(tr)
        te = add_sequence_lags(te)
    if "PLATE_NORM" in flags:
        tr = add_plate_expanding_norm(tr)
        te = add_plate_expanding_norm(te)
    if "DELTA_FEATS" in flags:
        tr = add_delta_causal_features(tr)
        te = add_delta_causal_features(te)
    
    # 이상치: train 기준
    bounds, y_lo, y_hi = compute_train_outlier_bounds(tr)
    tr = apply_train_outlier_filter(tr, bounds, y_lo, y_hi)
    te = clip_test_features_to_bounds(te, bounds)
    return tr, te, test_plates

def run_one_direction(df_base: pd.DataFrame, direction_key: str, flags: set, split_anchor: str,
                     rm_data: pd.DataFrame, gt: pd.DataFrame):
    tr, te, test_plates = prepare_train_test_leak_safe(
        df_base, use_strat=("STRAT_SPLIT" in flags), direction_key=direction_key, split_anchor=split_anchor, flags=flags,
        rm_data=rm_data, gt=gt
    )
    algo = ALGO_FIXED.get(direction_key, "XGBoost")
    res = train_moe_by_phase(tr, te, direction_key, flags, algo) if "MOE_PHASE" in flags else \
          train_single_model(tr, te, direction_key, flags, algo)
    res["test_plates"] = sorted(list(test_plates))
    return res

def strict_rollforward_r2(all_df: pd.DataFrame, e2x_test_plates: set, x2e_test_plates: set,
                          e2x_pred_map: dict, x2e_pred_map: dict):
    """교집합 Test 판만 roll-forward R² 계산"""
    test_plates = {str(p) for p in e2x_test_plates}.intersection({str(p) for p in x2e_test_plates})
    if not test_plates: return float("nan")
    rows=[]
    df = all_df[all_df[PLATE_COL].astype(str).isin(test_plates)].copy().sort_values([PLATE_COL, PASS_COL])
    for plate, sub in df.groupby(PLATE_COL):
        sub = sub.sort_values(PASS_COL)
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index: continue
            yhat = e2x_pred_map.get((str(plate), p), np.nan) if p%2==1 else x2e_pred_map.get((str(plate), p), np.nan)
            gt_next = float(idx.loc[p+1][CUR_WARP])
            if not (np.isnan(yhat) or np.isnan(gt_next)): rows.append({"pred": yhat, "gt": gt_next})
    if not rows: return float("nan")
    dfp = pd.DataFrame(rows)
    return float(r2_score(dfp["gt"], dfp["pred"]))

# ---------------------- Recipe Builder ----------------------
def build_recipes_from_user():
    recipes = {}
    # 1) RUN_ONLY 단건들
    for key in RUN_ONLY:
        key = key.strip()
        if not key: continue
        if key not in ALIAS2FLAGS:
            print(f"[WARN] 알 수 없는 시동어: {key} (무시)")
            continue
        recipes[key] = list(ALIAS2FLAGS[key])

    # 2) RUN_WITH 조합들
    for combo in RUN_WITH:
        combo = combo.strip()
        if not combo: continue
        parts = [p.strip() for p in combo.split("+") if p.strip()]
        flags = []
        ok = True
        for p in parts:
            if p not in ALIAS2FLAGS:
                print(f"[WARN] 조합 '{combo}' 중 알 수 없는 시동어: {p} (무시)")
                ok = False; break
            flags += ALIAS2FLAGS[p]
        if ok:
            uniq_flags = sorted(set(flags))
            recipes[combo] = uniq_flags

    # 3) 아무것도 지정 안 했으면 S0만
    if not recipes:
        recipes = {"S0": []}
    return recipes

# --------------------------------- Main ---------------------------------
def main():
    # -------------------- 데이터 -----------------------
    final_df_base, ground_truth = load_and_merge_base()
    e2x_raw, x2e_raw = split_e2x_x2e(final_df_base)
    
    # RM 데이터 로드
    D = _resolve_data_dir()
    rm_data = pd.read_csv(D/"posco1_105190.csv")

    exp_id = _next_experiment_id()
    code_snap = _save_code_snapshot(exp_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recipes = build_recipes_from_user()

    # 결과 누적(리더보드)
    rows = []

    def _run_one(key: str, flags: set, tag: str,
                 e2x_src: pd.DataFrame, x2e_src: pd.DataFrame, anchor_base: str):
        """단일 러닝 + 요약/가드"""
        anchor_name = f"{RUN_NAME}__anchor"

        print("\n" + "="*80)
        print(f"[RUN] {tag} | step={key} | flags={sorted(list(flags))} | anchor={anchor_name}")
        print("="*80)

        e2x_res = run_one_direction(e2x_src, "E2X", flags, anchor_name, rm_data, ground_truth)
        x2e_res = run_one_direction(x2e_src, "X2E", flags, anchor_name, rm_data, ground_truth)
        avg_r2 = float(np.mean([e2x_res["r2_te"], x2e_res["r2_te"]]))

        # --- Strict Roll-forward (교집합 Test) ---
        e2x_pred_map = { (p,pa): float(v) for (p,pa), v in zip(e2x_res["keys_te"], e2x_res["pred_te"]) }
        x2e_pred_map = { (p,pa): float(v) for (p,pa), v in zip(x2e_res["keys_te"], x2e_res["pred_te"]) }
        rf_strict = strict_rollforward_r2(final_df_base,
                                          set(e2x_res["test_plates"]), set(x2e_res["test_plates"]),
                                          e2x_pred_map, x2e_pred_map)

        # --- 저장 ---
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        steps_csv   = CSV_DIR / f"{exp_id}_{RUN_NAME}_{tag}_steps.csv"
        summary_txt = SUM_DIR / f"{exp_id}_{RUN_NAME}_{tag}_summary.txt"
        pd.DataFrame([{"step": tag, "E2X": e2x_res["r2_te"], "X2E": x2e_res["r2_te"], "Avg": avg_r2,
                       "RF_strict": rf_strict, "anchor": anchor_name}]).to_csv(steps_csv, index=False)

        lines=[]
        lines.append(f"R² Uplift Final Summary ({timestamp})")
        lines.append("="*72)
        lines.append(f"실험 ID: {exp_id} | RUN_NAME: {RUN_NAME} | STEP: {tag}")
        lines.append(f"전략 플래그: {', '.join(sorted(list(flags))) if flags else 'S0 (전략 없음)'}\n")
        def fmt(tag2, r):
            return (f"* {tag2} | Algo: {ALGO_FIXED[tag2]} (Optuna) "
                    f"| Train R2: {r['r2_tr']:.4f} | Test R2: {r['r2_te']:.4f} "
                    f"(RMSE {r['rmse_tr']:.4f}/{r['rmse_te']:.4f})")
        lines.append("[모델 성능 - R2]")
        lines.append(fmt("E2X", e2x_res))
        lines.append(fmt("X2E", x2e_res))
        lines.append(f"\n[평균 Test R2] {avg_r2:.4f}")
        if DO_STRICT_ROLLFORWARD and not np.isnan(rf_strict):
            lines.append(f"[Strict Roll-forward R²] {rf_strict:.4f}")
        lines.append("="*72)
        if "+" in key:
            lines.append("\n[시동어 설명]")
            for p in key.split("+"):
                p=p.strip()
                if p in ALIAS_DESC: lines.append(f"- {p}: {ALIAS_DESC[p]}")
        else:
            lines.append("\n[시동어 설명]")
            lines.append(f"- {key}: {ALIAS_DESC.get(key, 'N/A')}")
        lines.append(f"\n[산출물]\n- 단계별 결과 CSV: {steps_csv}\n- 코드 스냅샷: {code_snap}")
        summary_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"✓ 저장: {steps_csv}\n✓ 저장: {summary_txt}")

        # 리더보드용 누적
        rows.append({"step": tag, "avg_test_r2": avg_r2,
                     "E2X_test_r2": e2x_res["r2_te"], "X2E_test_r2": x2e_res["r2_te"],
                     "RF_strict": rf_strict, "anchor": anchor_name})

    # ---------- 정상 러닝 ----------
    for key, flag_list in recipes.items():
        flags = set(flag_list)
        _run_one(key, flags, key, e2x_raw, x2e_raw, RUN_NAME)

    # ---------- 리더보드 출력 ----------
    lb = pd.DataFrame(rows)
    if not lb.empty:
        # S0 기준 Δ
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

