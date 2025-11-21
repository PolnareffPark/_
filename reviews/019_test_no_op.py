# -*- coding: utf-8 -*-
"""
test_no_op.py  (leak-safe, optuna-free, warning-safe, auto leak-audit)

[핵심]
- 003 기준선 파이프라인 호환 + 누수 방지 + 경고 제거 + 과적합 완화
- PASS(t)→PASS(t+1) 불연속은 assert 중단 대신 '타깃=NaN 후 제거'로 안전 처리(카운트 로그)
- XGBoost: eval_metric은 생성자에 설정(버전 호환), fit()에는 전달하지 않음
- SelectKBest 대체: 상수/비유효 피처 제거 + 안전 F-score(상관계수 기반) → sqrt 경고 제거
- RM plate 통계(train-only)는 기본 OFF(환경변수 USE_RM_TRAINONLY=1일 때만 사용)
- 누수 자동 점검: 빠른 퍼뮤테이션 학습으로 평균 R²≈0 확인(LEAK_AUTOCHECK=1 기본 ON)
"""

# ========================= [ USER CONFIG ] =========================
RUN_NAME = "r2_uplift_final_v2"
RUN_ONLY = ["S0"]
RUN_WITH = [
    "S0+L1", "S0+PN1", "S0+DF1",
    "S0+TE2", "S0+W2", "S0+T2",
    "S0+L1+PN1+DF1+TE2+W2",
]

ALGO_FIXED = {"E2X": "XGBoost", "X2E": "RandomForest"}

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
SELECTK_K = 120
PAIR_EB_N0 = 50
WEIGHT_CLIP = (0.25, 3.0)
STRAT_TRY = 25
BAG_K = 5
DO_STRICT_ROLLFORWARD = False
IW_MODE = "oof"

# 앵커/검증 옵션
import os as _os
ANCHOR_PER_STEP   = (_os.environ.get("ANCHOR_PER_STEP","0") == "1")
PERMUTE_TARGET    = (_os.environ.get("PERMUTE_TARGET","0") == "1")
ANCHOR_FROM_CROSS = (_os.environ.get("ANCHOR_FROM_CROSS","0") == "1")
USE_RM_TRAINONLY  = (_os.environ.get("USE_RM_TRAINONLY","0") == "1")  # 기본 OFF
LEAK_AUTOCHECK    = (_os.environ.get("LEAK_AUTOCHECK","1") == "1")    # 기본 ON

# ==================================================================

import os, re, shutil, warnings, traceback, joblib
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
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
    "S0":  "Baseline(SelectK 120, plate-Group split, E2X=XGB/X2E=RF)",
    "L1":  "Sequence lags/momentum/rolling-std (plate-wise)",
    "PN1": "Plate expanding z-norm (past mean/std z-score)",
    "DF1": "Plate 1-step deltas & ratios",
    "R1":  "Robust Loss (XGB absolute)",
    "S1":  "Stratified Group Split",
    "TE2": "Pair Δ Target Encoding (OOF, EB)",
    "W2":  "Heteroscedastic Weights (OOF Δ-분산 EB)",
    "T2":  "Δ‑Target 회귀 (ŷ = y(t) + Δ̂)",
    "M2":  "Monotone constraint on CUR_WARP (+1)",
    "SEG": "Phase Segmentation(MoE) — off",
    "IW":  "Importance Weighting (Train OOF)",
    "BAG":  "Bagging(K=5)",
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
    """
    Entry/Exit → [plate, pass] 단일화 → FM 조인 → t→t+1 생성.
    - [연속성 처리] next_pass - pass != 1 인 행은 타깃을 제거(drop)하며, 개수를 로그로 남김.
    - RM plate 통계는 여기서 계산하지 않음(누수 방지: split 후 train-only에서 선택적으로 매핑).
    """
    D = _resolve_data_dir()

    # 1) GT 로드 & [plate, pass] 단일화
    gt_entry = pd.read_csv(D/"entry_direction_results.csv")
    gt_exit  = pd.read_csv(D/"exit_direction_results.csv")
    gt_all = pd.concat([gt_entry, gt_exit], ignore_index=True)
    gt_all["extracted_plate"] = gt_all["filename"].str.extract(r"(PB\d+)")
    gt_all["extracted_pass"]  = gt_all["filename"].str.extract(r"_\d+_(\d+)").astype(int)
    gt_one = (gt_all.groupby(["extracted_plate","extracted_pass"], as_index=False)
                    .agg(warping_index=("warping_index","mean")))

    # 2) FM 조인
    fm = pd.read_csv(D/"posco2_105190.csv")
    merged = (pd.merge(gt_one, fm,
                       left_on=["extracted_plate","extracted_pass"],
                       right_on=[PLATE_COL, PASS_COL],
                       how="inner")
                .sort_values([PLATE_COL, PASS_COL])
                .reset_index(drop=True))

    # 3) 유일성 가드
    if merged.duplicated([PLATE_COL, PASS_COL]).any():
        dup = int(merged.duplicated([PLATE_COL, PASS_COL]).sum())
        raise AssertionError(f"(PLATE, PASS) 중복 {dup}행 존재 — Entry/Exit 축약/조인 로직 점검 필요")

    # 4) t→t+1 타깃 & 연속성 필터
    merged[TARGET_COL] = merged.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged = merged.rename(columns={"warping_index": CUR_WARP})
    merged["_next_pass"] = merged.groupby(PLATE_COL)[PASS_COL].shift(-1)
    diff = merged["_next_pass"] - merged[PASS_COL]
    nonconsec = (diff.notna()) & (diff != 1)
    n_bad = int(nonconsec.sum())
    if n_bad > 0:
        print(f"[WARN] PASS(t)→PASS(t+1) 불연속 {n_bad}행: 타깃 제거 후 드롭")
        # 타깃을 제거해 drop 단계에서 필터되도록 함
        merged.loc[nonconsec, TARGET_COL] = np.nan
    merged = merged.drop(columns=["_next_pass"])

    # 5) 마지막 PASS 및 불연속 표본 제거
    merged = merged.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # 6) 불필요 열 정리 & 저장
    final_df = merged.drop(columns=["extracted_plate","extracted_pass","filename",
                                    "quality_grade","quality_grade_current_pass","direction"],
                           errors="ignore")
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
def base_feature_engineering(df: pd.DataFrame, pass_max: int | float | None = None) -> pd.DataFrame:
    """
    - FM_* 요약 + PASS 파생
    - PASS_PROGRESS는 train의 pass_max를 test에도 재사용(일관 스케일)
    """
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
    d["FM_PASS_squared"] = d[PASS_COL] ** 2
    if pass_max is None:
        pass_max = float(d[PASS_COL].max()) if len(d) else 1.0
    if pass_max <= 0: pass_max = 1.0
    d["FM_PASS_progress"] = d[PASS_COL] / pass_max
    return d

def add_sequence_lags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    g = d.groupby(PLATE_COL, sort=False)
    d["lag1_prev_warp"]  = g[CUR_WARP].shift(1)
    d["mom1_warp"]       = d[CUR_WARP] - d["lag1_prev_warp"]
    d["roll_std_warp2"]  = g[CUR_WARP].apply(lambda s: s.rolling(2).std()).reset_index(level=0, drop=True)
    return d

def add_plate_expanding_norm_train(df: pd.DataFrame, topk=20):
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    fm_cols = [c for c in d.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if not fm_cols: return d, []
    var_rank = pd.Series(d[fm_cols].var(numeric_only=True)).sort_values(ascending=False)
    topk_cols = list(var_rank.index[:min(topk, len(var_rank))])
    for c in topk_cols:
        g = d.groupby(PLATE_COL, sort=False)[c]
        mean_exp = g.expanding().mean().reset_index(level=0, drop=True).shift(1)
        std_exp  = g.expanding().std().reset_index(level=0, drop=True).shift(1)
        d[f"{c}_zexp"] = (d[c] - mean_exp) / (std_exp + 1e-8)
    return d, topk_cols

def add_plate_expanding_norm_test(df: pd.DataFrame, topk_cols: list):
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    for c in topk_cols:
        if c not in d.columns: continue
        g = d.groupby(PLATE_COL, sort=False)[c]
        mean_exp = g.expanding().mean().reset_index(level=0, drop=True).shift(1)
        std_exp  = g.expanding().std().reset_index(level=0, drop=True).shift(1)
        d[f"{c}_zexp"] = (d[c] - mean_exp) / (std_exp + 1e-8)
    return d

def add_delta_causal_features_train(df: pd.DataFrame, topk=20):
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    fm_cols = [c for c in d.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if not fm_cols: return d, []
    var_rank = pd.Series(d[fm_cols].var(numeric_only=True)).sort_values(ascending=False)
    topk_cols = list(var_rank.index[:min(topk, len(var_rank))])
    g = d.groupby(PLATE_COL, sort=False)
    for c in topk_cols:
        prev = g[c].shift(1)
        d[f"{c}_d1"] = d[c] - prev
        d[f"{c}_ratio"] = d[c] / (prev.abs() + 1e-8)
    return d, topk_cols

def add_delta_causal_features_test(df: pd.DataFrame, topk_cols: list):
    d = df.copy().sort_values([PLATE_COL, PASS_COL])
    g = d.groupby(PLATE_COL, sort=False)
    for c in topk_cols:
        if c not in d.columns: continue
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
    d = d[(d[TARGET_COL] >= y_lo) & (d[TARGET_COL] <= y_hi)]
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

    # 003 앵커 강제(선택)
    if ANCHOR_FROM_CROSS:
        if direction_key == "E2X":
            cross_anchor = MODELS_DIR / "e2x_test_plates_tplus1.pkl"
        else:
            cross_anchor = MODELS_DIR / "x2e_test_plates_tplus1.pkl"
        if cross_anchor.exists():
            test_plates = set(joblib.load(cross_anchor))
            tr = df[~df[PLATE_COL].astype(str).isin(test_plates)].copy()
            te = df[df[PLATE_COL].astype(str).isin(test_plates)].copy()
            return tr, te, test_plates

    if anchor_file.exists():
        test_plates = set(joblib.load(anchor_file))
        tr = df[~df[PLATE_COL].astype(str).isin(test_plates)].copy()
        te = df[df[PLATE_COL].astype(str).isin(test_plates)].copy()
        return tr, te, test_plates

    if not use_strat:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        tr_i, te_i = next(gss.split(df.drop(columns=[TARGET_COL]), df[TARGET_COL],
                                    groups=df[PLATE_COL].astype(str).values))
        te_plates = sorted(df.iloc[te_i][PLATE_COL].astype(str).unique())
        joblib.dump(te_plates, anchor_file)
        return df.iloc[tr_i].copy(), df.iloc[te_i].copy(), set(te_plates)

    # Stratified by pair
    pairs = _pair_str(df)
    pair_vals = sorted(pairs.unique().tolist())
    best = (1e9, None, None)
    rng = np.random.RandomState(SEED)
    for _ in range(STRAT_TRY):
        seed = int(rng.randint(0, 1e9))
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr_i, te_i = next(gss.split(df.drop(columns=[TARGET_COL]), df[TARGET_COL],
                                    groups=df[PLATE_COL].astype(str).values))
        hist_all = df.groupby(pairs).size().reindex(pair_vals, fill_value=0).values
        hist_te  = df.iloc[te_i].groupby(pairs.iloc[te_i]).size().reindex(pair_vals, fill_value=0).values
        p_all = hist_all/(hist_all.sum()+1e-8); p_te = hist_te/(hist_te.sum()+1e-8)
        l1 = np.abs(p_all - p_te).sum()
        if l1 < best[0]: best = (l1, tr_i, te_i)
    tr_i, te_i = best[1], best[2]
    te_plates = sorted(df.iloc[te_i][PLATE_COL].astype(str).unique())
    joblib.dump(te_plates, anchor_file)
    return df.iloc[tr_i].copy(), df.iloc[te_i].copy(), set(te_plates)

# --------------------------- Modeling Core -----------------------------
def select_features(Xtr_df: pd.DataFrame, ytr: np.ndarray,
                    Xte_df: pd.DataFrame, mandatory=None, k=SELECTK_K):
    """
    안전 K-best:
    - 상수/비유효/비유한 피처 제거
    - 상관계수 기반 F-score(수치 안정) → 상위 K 선택
    """
    mandatory = list(mandatory or [])
    all_cols  = list(Xtr_df.columns)

    # 1) 사전 정화
    finite_mask = np.isfinite(Xtr_df.values).all(axis=0)
    var = Xtr_df.var(numeric_only=True).reindex(all_cols).fillna(0.0).values
    nonconst_mask = var > 1e-12
    keep_mask = finite_mask & nonconst_mask
    keep_cols = [c for c, m in zip(all_cols, keep_mask) if m]
    for c in mandatory:
        if c not in keep_cols and c in all_cols:
            keep_cols.append(c)

    Xtr = Xtr_df[keep_cols].copy()
    Xte = Xte_df.reindex(columns=keep_cols, fill_value=np.nan).copy()
    Xtr = Xtr.fillna(Xtr.median(numeric_only=True))
    Xte = Xte.fillna(Xtr.median(numeric_only=True))

    # 2) 안전 F-score
    x = Xtr.values
    y = ytr - ytr.mean()
    y_std = y.std();  y_std = y_std if y_std >= 1e-12 else 1e-12
    x_mean = x.mean(axis=0); x_std = x.std(axis=0); x_std[x_std < 1e-12] = 1e-12
    xr = (x - x_mean) / x_std;  yr = y / y_std
    r  = (xr.T @ yr) / (len(y) - 1)
    r2 = np.clip(r**2, 0.0, 1.0 - 1e-12)
    F  = (r2 / (1.0 - r2)) * (len(y) - 2)

    mand = [c for c in mandatory if c in keep_cols]
    others = [c for c in keep_cols if c not in mand]
    k_remain = max(1, min(int(k) - len(mand), len(others)))
    idx_map = {c:i for i,c in enumerate(keep_cols)}
    order = np.argsort([F[idx_map[c]] for c in others])[::-1]
    chosen = mand + [others[i] for i in order[:k_remain]]
    return chosen, Xtr[chosen].values, Xte[chosen].values

def _build_xgb(robust=False, monotone_index=None, n_features=None, seed=SEED):
    params = dict(
        random_state=seed, tree_method="hist",
        n_estimators=800, learning_rate=0.05,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0,
        n_jobs=0,
        eval_metric="rmse",   # 생성자에 지정(버전 호환)
    )
    if monotone_index is not None and n_features is not None:
        v = [0]*n_features
        if 0 <= monotone_index < n_features: v[monotone_index] = 1
        params["monotone_constraints"] = tuple(v)
    return XGBRegressor(objective=("reg:absoluteerror" if robust else "reg:squarederror"), **params)

def _build_rf(seed=SEED):
    return RandomForestRegressor(
        random_state=seed, n_estimators=600, n_jobs=-1,
        max_depth=20, min_samples_leaf=5, max_features="sqrt"
    )

def _fit_predict_model(Xtr, ytr, Xte, flags, algo_name, chosen_names,
                       sample_weight=None, groups: np.ndarray | None = None):
    mono_idx = chosen_names.index(CUR_WARP) if ("MONO_CUR" in flags and CUR_WARP in chosen_names and algo_name=="XGBoost") else None

    def make_one(seed):
        if algo_name == "XGBoost":
            return _build_xgb(("ROBUST_LOSS" in flags), monotone_index=mono_idx, n_features=len(chosen_names), seed=seed)
        else:
            return _build_rf(seed=seed)

    # 내부 홀드아웃 (plate group) — early stopping용
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr_i, va_i = next(gss.split(Xtr, ytr, groups=groups))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr_i, va_i = next(gss.split(Xtr, ytr))

    Xfit, yfit = Xtr[tr_i], ytr[tr_i]
    Xval, yval = Xtr[va_i], ytr[va_i]
    sw_fit = None if sample_weight is None else sample_weight[tr_i]

    if "BAGGING" in flags:
        preds=[]; mdl=None
        for s in [SEED+i for i in range(BAG_K)]:
            m = make_one(s)
            if algo_name == "XGBoost":
                try:
                    m.fit(Xfit, yfit, sample_weight=sw_fit,
                          eval_set=[(Xval, yval)],
                          early_stopping_rounds=50, verbose=False)
                except TypeError:
                    m.fit(Xfit, yfit, sample_weight=sw_fit)
            else:
                m.fit(Xfit, yfit, sample_weight=sw_fit)
            preds.append(m.predict(Xte)); mdl = m
        return mdl, np.mean(preds, axis=0)

    m = make_one(SEED)
    if algo_name == "XGBoost":
        try:
            m.fit(Xfit, yfit, sample_weight=sw_fit,
                  eval_set=[(Xval, yval)],
                  early_stopping_rounds=50, verbose=False)
        except TypeError:
            m.fit(Xfit, yfit, sample_weight=sw_fit)
    else:
        m.fit(Xfit, yfit, sample_weight=sw_fit)
    return m, m.predict(Xte)

def train_single_model(train_df, test_df, direction_key: str, flags: set, algo_name: str):
    """
    - 모든 모드에서 CUR_WARP를 필수 피처로 포함(003 호환성)
    - XGB: group holdout early stopping(버전 호환 안전구문)
    """
    tr = train_df.copy(); te = test_df.copy()

    # TE2: Δ-TE 특성
    if "PAIR_TE" in flags:
        tr, te = add_pair_delta_te_feature(tr, te, n0=PAIR_EB_N0)

    # 타깃
    if "DELTA_TARGET" in flags:
        tr["_target"] = tr[TARGET_COL] - tr[CUR_WARP]
        te["_target"] = te[TARGET_COL] - te[CUR_WARP]
        mandatory = [CUR_WARP] + (["pair_delta_te"] if "PAIR_TE" in flags else [])
    else:
        tr["_target"] = tr[TARGET_COL]
        te["_target"] = te[TARGET_COL]
        mandatory = [CUR_WARP] + (["pair_delta_te"] if "PAIR_TE" in flags else [])

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

    # X, y
    Xtr_df, med, cols = _ensure_numeric_X(tr)
    Xte_df = te.drop(columns=[c for c in EXCLUDE if c in te.columns], errors="ignore") \
               .select_dtypes(include=[np.number]).reindex(columns=cols, fill_value=np.nan).fillna(med)
    ytr = tr["_target"].values; yte = te["_target"].values

    # 선택 & 학습
    chosen_names, Xtr, Xte = select_features(Xtr_df, ytr, Xte_df, mandatory=mandatory, k=SELECTK_K)
    groups = tr[PLATE_COL].astype(str).values
    model, yhat = _fit_predict_model(Xtr, ytr, Xte, flags, algo_name, chosen_names,
                                     sample_weight=sample_weight, groups=groups)

    if "DELTA_TARGET" in flags:
        yhat = te[CUR_WARP].values + yhat

    r2_tr = float(r2_score(ytr, model.predict(Xtr)))
    r2_te = float(r2_score(yte, yhat))
    rmse_tr = float(np.sqrt(mean_squared_error(ytr, model.predict(Xtr))))
    rmse_te = float(np.sqrt(mean_squared_error(yte, yhat)))

    return {"model": model, "features": chosen_names,
            "r2_tr": r2_tr, "r2_te": r2_te,
            "rmse_tr": rmse_tr, "rmse_te": rmse_te,
            "yte": yte, "yhat": yhat}

# ----------------------- 빠른 퍼뮤테이션 누수 감사 ----------------------
def _quick_perm_leak_r2(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        algo_name: str, seed: int = SEED+7) -> float:
    """가벼운 모델로 타깃 퍼뮤테이션 R² 측정(속도 절약)."""
    rng = np.random.RandomState(seed)
    tr = train_df.copy(); te = test_df.copy()
    ytr = rng.permutation(tr[TARGET_COL].values)
    yte = te[TARGET_COL].values

    # 간단 피처(모든 수치, CUR_WARP는 반드시 포함)
    Xtr_df, med, cols = _ensure_numeric_X(tr)
    Xte_df = te.drop(columns=[c for c in EXCLUDE if c in te.columns], errors="ignore") \
               .select_dtypes(include=[np.number]).reindex(columns=cols, fill_value=np.nan).fillna(med)

    mandatory = [CUR_WARP] if CUR_WARP in Xtr_df.columns else []
    chosen_names, Xtr, Xte = select_features(Xtr_df, ytr, Xte_df, mandatory=mandatory, k=min(60, SELECTK_K))

    if algo_name == "XGBoost":
        m = XGBRegressor(
            objective="reg:squarederror", random_state=seed, tree_method="hist",
            n_estimators=120, learning_rate=0.08, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, n_jobs=0, eval_metric="rmse"
        )
        m.fit(Xtr, ytr)
        yhat = m.predict(Xte)
    else:
        m = RandomForestRegressor(random_state=seed, n_estimators=200,
                                  max_depth=12, min_samples_leaf=5, n_jobs=-1)
        m.fit(Xtr, ytr)
        yhat = m.predict(Xte)
    return float(r2_score(yte, yhat))

# ------------------------------ Runner ---------------------------------
def _apply_fe_by_flags_safe(train_df: pd.DataFrame, test_df: pd.DataFrame, flags: set):
    """
    FE는 모두 train 기준 → test 동일 규칙 매핑.
    PASS_PROGRESS의 분모(pass_max)는 train에서 구해 test에 재사용.
    """
    pass_max_tr = float(train_df[PASS_COL].max()) if len(train_df) else 1.0
    tr = base_feature_engineering(train_df, pass_max=pass_max_tr)
    te = base_feature_engineering(test_df,  pass_max=pass_max_tr)

    if "SEQ_LAG" in flags:
        tr = add_sequence_lags(tr); te = add_sequence_lags(te)
    if "PLATE_NORM" in flags:
        tr, topk_cols = add_plate_expanding_norm_train(tr); te = add_plate_expanding_norm_test(te, topk_cols)
    if "DELTA_FEATS" in flags:
        tr, topk_cols = add_delta_causal_features_train(tr); te = add_delta_causal_features_test(te, topk_cols)
    return tr, te

def compute_rm_stats_for_plates(plates: set) -> pd.DataFrame:
    D = _resolve_data_dir()
    rm = pd.read_csv(D/"posco1_105190.csv")
    rm = rm[rm["RM_날판번호"].astype(str).isin({str(p) for p in plates})].copy()
    if rm.empty: return pd.DataFrame({"RM_날판번호": []})
    num_cols = rm.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"RM_날판번호", "RM_압연Pass번호", "warping_index"}
    stat_cols = [c for c in num_cols if c not in drop_like and not c.endswith("_target")]
    rm_stats = rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(6)
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    return rm_stats.reset_index()

def attach_train_only_rm_stats(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tr = train_df.copy(); te = test_df.copy()
    plates_tr = set(tr[PLATE_COL].astype(str).unique().tolist())
    rm_stats_tr = compute_rm_stats_for_plates(plates_tr)
    tr = pd.merge(tr, rm_stats_tr, left_on=PLATE_COL, right_on="RM_날판번호", how="left")
    te = pd.merge(te, rm_stats_tr, left_on=PLATE_COL, right_on="RM_날판번호", how="left")
    tr = tr.drop(columns=["RM_날판번호"], errors="ignore")
    te = te.drop(columns=["RM_날판번호"], errors="ignore")
    if not rm_stats_tr.empty:
        global_med = rm_stats_tr.drop(columns=["RM_날판번호"], errors="ignore").median(numeric_only=True)
        for c in global_med.index:
            if c in te.columns: te[c] = te[c].fillna(float(global_med[c]))
            if c in tr.columns: tr[c] = tr[c].fillna(float(global_med[c]))
    return tr, te

def _run_one(key: str, flags: set, tag: str, e2x_raw: pd.DataFrame, x2e_raw: pd.DataFrame, anchor_base: str):
    # 1) Split 먼저 (FE 전에!) - 누수 방지
    anchor_name = f"{anchor_base}__{key}__anchor" if ANCHOR_PER_STEP else f"{anchor_base}__anchor"
    use_strat = ("STRAT_SPLIT" in flags)
    e2x_tr, e2x_te, _ = split_by_anchor(e2x_raw, "E2X", use_strat, anchor_name)
    x2e_tr, x2e_te, _ = split_by_anchor(x2e_raw, "X2E", use_strat, anchor_name)

    # 2) RM train-only plate 통계 — 기본 OFF (필요시 환경변수로 ON)
    if USE_RM_TRAINONLY:
        e2x_tr, e2x_te = attach_train_only_rm_stats(e2x_tr, e2x_te)
        x2e_tr, x2e_te = attach_train_only_rm_stats(x2e_tr, x2e_te)

    # 3) FE (train 기준 → test 매핑)
    e2x_tr, e2x_te = _apply_fe_by_flags_safe(e2x_tr, e2x_te, flags)
    x2e_tr, x2e_te = _apply_fe_by_flags_safe(x2e_tr, x2e_te, flags)

    # 4) 이상치 경계(train) & clip(test)
    b_e2x, ylo_e2x, yhi_e2x = compute_train_outlier_bounds(e2x_tr)
    b_x2e, ylo_x2e, yhi_x2e = compute_train_outlier_bounds(x2e_tr)
    e2x_tr = apply_train_outlier_filter(e2x_tr, b_e2x, ylo_e2x, yhi_e2x)
    x2e_tr = apply_train_outlier_filter(x2e_tr, b_x2e, ylo_x2e, yhi_x2e)
    e2x_te = clip_test_features_to_bounds(e2x_te, b_e2x)
    x2e_te = clip_test_features_to_bounds(x2e_te, b_x2e)

    # 5) 학습/예측
    e2x_res = train_single_model(e2x_tr, e2x_te, "E2X", flags, ALGO_FIXED["E2X"])
    x2e_res = train_single_model(x2e_tr, x2e_te, "X2E", flags, ALGO_FIXED["X2E"])

    # 6) 저장/요약
    avg_r2 = (e2x_res["r2_te"] + x2e_res["r2_te"]) / 2.0

    steps_csv = CSV_DIR / f"{experiment_id}_{RUN_NAME}_{tag}_steps.csv"
    pd.DataFrame({
        "direction": ["E2X"]*len(e2x_res["yte"]) + ["X2E"]*len(x2e_res["yte"]),
        "y_true":    np.r_[e2x_res["yte"], x2e_res["yte"]],
        "y_pred":    np.r_[e2x_res["yhat"], x2e_res["yhat"]],
    }).to_csv(steps_csv, index=False)

    code_snap = SCRIPTS_DIR / f"{experiment_id}_{Path(__file__).name}"
    try: shutil.copy2(THIS_PATH, code_snap)
    except Exception: pass

    summary_txt = SUM_DIR / f"{experiment_id}_{RUN_NAME}_{tag}_summary.txt"
    lines=[]
    ts = datetime.now(); lines.append(f"R² Uplift (alias) Summary ({ts.strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append("="*72)
    lines.append(f"실험 ID: {experiment_id} | RUN_NAME: {RUN_NAME} | STEP: {tag}")
    lines.append(f"전략 플래그: {', '.join(sorted(list(flags)))}" if flags else "전략 플래그: S0 (전략 없음)")

    def fmt(label, r):
        return (f"* {label} | Algo: {ALGO_FIXED[label]} "
                f"| Train R2: {r['r2_tr']:.4f} | Test R2: {r['r2_te']:.4f} "
                f"(RMSE {r['rmse_tr']:.4f}/{r['rmse_te']:.4f})")
    lines.append("\n[모델 성능 - R2]")
    lines.append(fmt("E2X", e2x_res))
    lines.append(fmt("X2E", x2e_res))
    lines.append(f"\n[평균 Test R2] {avg_r2:.4f}")

    # 7) 빠른 퍼뮤테이션 누수 감사 (자동)
    if LEAK_AUTOCHECK:
        e2x_perm = _quick_perm_leak_r2(e2x_tr, e2x_te, ALGO_FIXED["E2X"])
        x2e_perm = _quick_perm_leak_r2(x2e_tr, x2e_te, ALGO_FIXED["X2E"])
        lines.append("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        lines.append(f"- E2X permuted R²: {e2x_perm:.4f}")
        lines.append(f"- X2E permuted R²: {x2e_perm:.4f}")
        if max(e2x_perm, x2e_perm) > 0.05:
            lines.append("[LEAK WARNING] permuted R² > 0.05 → 파이프라인/스플릿 경계 점검 필요")

    lines.append("="*72)
    lines.append(f"\n[산출물]\n- 단계별 결과 CSV: {steps_csv}\n- 코드 스냅샷: {code_snap}")
    summary_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 저장: {steps_csv}\n✓ 저장: {summary_txt}")

    return {"avg": avg_r2, "e2x": e2x_res["r2_te"], "x2e": x2e_res["r2_te"], "rf": np.nan, "anchor": anchor_name}

# -------------------------------- main ---------------------------------
def _parse_recipes(run_only, run_with):
    recipes = {}
    def to_flags(expr: str):
        toks = [t.strip() for t in expr.split("+") if t.strip()]
        toks = [t for t in toks if t != "S0"]
        flags = []
        for t in toks:
            flags.extend(ALIAS2FLAGS.get(t, []))
        return flags
    for k in (run_only or []):
        recipes[k] = to_flags(k)
    for k in (run_with or []):
        recipes[k] = to_flags(k)
    if "S0" not in recipes:
        recipes["S0"] = []
    return recipes

def main():
    global experiment_id
    experiment_id = _next_experiment_id()

    print("="*80)
    print(f"[RUN_NAME] {RUN_NAME} | Optuna-free")
    if PERMUTE_TARGET:    print("[WARNING] PERMUTE_TARGET=1 → 타깃 무작위화로 누수 검사 모드")
    if ANCHOR_FROM_CROSS: print("[INFO] ANCHOR_FROM_CROSS=1 → 003의 test plate 사용")
    if USE_RM_TRAINONLY:  print("[INFO] USE_RM_TRAINONLY=1 → train-only RM plate 통계 사용")
    if LEAK_AUTOCHECK:    print("[INFO] LEAK_AUTOCHECK=1 → 빠른 퍼뮤테이션 감사 자동 수행")
    print("="*80)

    final_df = load_and_merge_tplus1()

    if PERMUTE_TARGET:
        rng = np.random.RandomState(SEED)
        final_df[TARGET_COL] = rng.permutation(final_df[TARGET_COL].values)
        print(f"[PERMUTE] Target shuffled with seed={SEED}")

    e2x_raw, x2e_raw = split_e2x_x2e(final_df)
    recipes = _parse_recipes(RUN_ONLY, RUN_WITH)

    rows = []; anchor_base = RUN_NAME
    for key, flag_list in recipes.items():
        flags = set(flag_list)
        print("\n" + "="*80)
        print(f"[RUN] {key} | step={key} | flags={sorted(list(flags))} | anchor="
              f"{(f'{RUN_NAME}__{key}__anchor' if ANCHOR_PER_STEP else f'{RUN_NAME}__anchor')}")
        print("="*80)
        res = _run_one(key, flags, key, e2x_raw, x2e_raw, RUN_NAME)
        rows.append({"step": key, "avg_test_r2": res["avg"],
                     "E2X_test_r2": res["e2x"], "X2E_test_r2": res["x2e"],
                     "RF_strict": np.nan, "anchor": res["anchor"]})

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
