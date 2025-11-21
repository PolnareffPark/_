# -*- coding: utf-8 -*-
"""
test_no_op.py  (leak-hard-guard, optuna-free, warning-safe, auto leak-audit)

- 라벨·잠재 누수 전면 차단:
  1) 이름 기반 블랙리스트로 라벨 의심 열 제거
  2) 타깃과 완전/준완전 일치하는 피처 자동 탐지·제거(|r|>=0.99999 또는 값 동일)
  3) FE/통계/선택/정규화는 split 이후 train-only 적합 → test 매핑
  4) PERMUTE_TARGET=1 시 앵커 파일명을 별도로 써서 캐시 간섭 차단
  5) 요약 .txt에 제거 목록/분할 전·후 상관 Top‑k 기록
"""

# ========================= [ USER CONFIG ] =========================
import os as _os

RUN_NAME = "r2_uplift_final_v3"
RUN_ONLY = ["S0"]
RUN_WITH = []

# 기본 알고리즘(020 재현): 두 방향 모두 XGBoost
ALGO_FIXED = {"E2X": "XGBoost", "X2E": "XGBoost"}

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

def _env_on(name: str, default="0") -> bool:
    return str(_os.environ.get(name, default)).strip().lower() in {"1","true","yes","y"}

ANCHOR_PER_STEP   = _env_on("ANCHOR_PER_STEP","0")
PERMUTE_TARGET    = _env_on("PERMUTE_TARGET","0")
ANCHOR_FROM_CROSS = _env_on("ANCHOR_FROM_CROSS","0")
USE_RM_TRAINONLY  = _env_on("USE_RM_TRAINONLY","0")
LEAK_AUTOCHECK    = _env_on("LEAK_AUTOCHECK","1")
STRICT_NO_TRANSD  = _env_on("STRICT_NO_TRANSD","0")

# X2E 강화 전략 환경 플래그(선택)
X2E_EXPERTS       = _env_on("X2E_EXPERTS","0")
X2E_RESIDUAL      = _env_on("X2E_RESIDUAL","0")
X2E_MONO          = _env_on("X2E_MONO","0")

# 하드 가드(기본 ON 권장): 이름/값 기반 라벨 차단
HARD_LEAK_GUARD   = _env_on("HARD_LEAK_GUARD","1")

if _os.environ.get("RUN_NAME"): RUN_NAME = _os.environ["RUN_NAME"]

def _env_list(varname: str, default_list: list[str]) -> list[str]:
    s = _os.environ.get(varname)
    if s is None or not str(s).strip(): return default_list
    return [x.strip() for x in str(s).split(",") if x.strip()]

# ==================================================================

import os, re, shutil, warnings, traceback, joblib
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
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
    "W2":  ["HETERO_W"],
    "T2":  ["DELTA_TARGET"],
    "M2":  ["MONO_CUR"],
    "SEG": ["MOE_PHASE"],
    "IW":  ["IMP_WEIGHT"],
    "BAG": ["BAGGING"],
    # X2E 전용
    "X2E_MOE": ["X2E_MOE"],
    "X2E_YJ":  ["X2E_YJ"],
    "X2E_RES": ["X2E_RES"],
}

ALIAS_DESC = {
    "S0":  "Baseline(SelectK 120, plate-Group split, E2X=XGB/X2E=XGB)",
    "L1":  "Sequence lags/momentum/rolling-std (plate-wise)",
    "PN1": "Plate expanding z-norm (past mean/std z-score)",
    "DF1": "Plate 1-step deltas & ratios",
    "R1":  "Robust Loss (XGB absolute)",
    "S1":  "Stratified Group Split",
    "TE2": "Pair Δ Target Encoding (OOF, EB)",
    "W2":  "Heteroscedastic Weights",
    "T2":  "Δ-Target 회귀 (ŷ = y(t) + Δ̂)",
    "M2":  "Monotone(+1) on CUR_WARP",
    "SEG": "Phase Segmentation(MoE)",
    "IW":  "Importance Weighting (OOF)",
    "BAG": "Bagging",
    "X2E_MOE": "X2E Experts",
    "X2E_YJ":  "X2E Yeo-Johnson",
    "X2E_RES": "X2E 2-stage Residual",
}

# --------------------------- Globals (Debug) ---------------------------
GLOBAL_DIAG = {
    "permute": int(PERMUTE_TARGET),
    "permute_seed": SEED if PERMUTE_TARGET else None,
    "anchor_from_cross": int(ANCHOR_FROM_CROSS),
    "strict_no_transd": int(STRICT_NO_TRANSD),
    "leak_autocheck": int(LEAK_AUTOCHECK),
    "hard_leak_guard": int(HARD_LEAK_GUARD),
    "nonconsec_dropped": 0,
    "name_drop_cols": [],
    "eq_drop_cols": [],
    "near1_drop_cols": [],
}

# ------------------------------- Utils ---------------------------------
LEAKY_PATTERNS = [
    r"(^|[_\-])target($|[_\-])",
    r"(^|[_\-])label($|[_\-])",
    r"(^|[_\-])y_true($|[_\-])",
    r"(^|[_\-])y_pred($|[_\-])",
    r"(^|[_\-])oof($|[_\-])",
    r"(^|[_\-])pred($|[_\-])",
    r"warping_index_target",
]
LEAKY_REGEX = re.compile("|".join(LEAKY_PATTERNS), re.IGNORECASE)

WHITELIST_EXACT = {CUR_WARP}  # 타깃이 아님(현재 pass), 반드시 허용

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

def _drop_name_leaky_cols(df: pd.DataFrame) -> pd.DataFrame:
    if not HARD_LEAK_GUARD or df.empty: return df
    keep = []
    dropped=[]
    for c in df.columns:
        if c in EXCLUDE or c in WHITELIST_EXACT:  # 타깃/키/화이트리스트 별도 처리
            keep.append(c); continue
        if LEAKY_REGEX.search(c):
            dropped.append(c)
        else:
            keep.append(c)
    if dropped:
        GLOBAL_DIAG["name_drop_cols"].extend(dropped)
    return df[keep].copy()

def _ensure_numeric_X(df: pd.DataFrame):
    # 1) 이름 기반 1차 차단
    X_all = _drop_name_leaky_cols(df.drop(columns=[c for c in EXCLUDE if c in df.columns], errors="ignore"))
    # 2) 수치형만 사용 + 결측 대치
    X = X_all.select_dtypes(include=[np.number]).copy()
    med = X.median(numeric_only=True)
    return X.fillna(med), med, list(X.columns)

def _assert_and_drop_value_leak(Xtr_df: pd.DataFrame, ytr: np.ndarray,
                                Xte_df: pd.DataFrame | None = None,
                                yte: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame|None]:
    """
    2차 방어막: (train 기준)
      - 피처==타깃(정확히 동일) → 드롭
      - |corr|>=0.99999 (준완전 일치) → 드롭
    test에도 동일 열이 있으면 함께 드롭.
    """
    if not HARD_LEAK_GUARD or Xtr_df.empty:
        return Xtr_df, Xte_df

    drop_cols = []
    # 값 동일 검사
    for c in Xtr_df.columns:
        xc = Xtr_df[c].values
        # 길이 동일 & 모두 유한 & 완전 동일 여부
        if xc.shape == ytr.shape and np.all(np.isfinite(xc)) and np.all(np.isfinite(ytr)):
            if np.array_equal(xc, ytr):
                drop_cols.append(c)
                continue
        # 상관(준완전)
        sx = np.std(xc); sy = np.std(ytr)
        if sx >= 1e-12 and sy >= 1e-12:
            r = float(np.corrcoef(xc, ytr)[0,1])
            if np.isfinite(r) and abs(r) >= 0.99999:
                drop_cols.append(c)

    drop_cols = sorted(set([c for c in drop_cols if c not in WHITELIST_EXACT]))
    if drop_cols:
        GLOBAL_DIAG["eq_drop_cols"].extend(drop_cols)
        Xtr_df = Xtr_df.drop(columns=drop_cols, errors="ignore")
        if Xte_df is not None:
            Xte_df = Xte_df.drop(columns=drop_cols, errors="ignore")
    return Xtr_df, Xte_df

# -------------------- 데이터 병합 & 타깃 t→t+1 --------------------
def load_and_merge_tplus1():
    D = _resolve_data_dir()

    gt_entry = pd.read_csv(D/"entry_direction_results.csv")
    gt_exit  = pd.read_csv(D/"exit_direction_results.csv")
    gt_all = pd.concat([gt_entry, gt_exit], ignore_index=True)
    gt_all["extracted_plate"] = gt_all["filename"].str.extract(r"(PB\d+)")
    gt_all["extracted_pass"]  = gt_all["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    gt_one = (gt_all.groupby(["extracted_plate","extracted_pass"], as_index=False)
                    .agg(warping_index=("warping_index","mean")))

    fm = pd.read_csv(D/"posco2_105190.csv")
    merged = (pd.merge(gt_one, fm,
                       left_on=["extracted_plate","extracted_pass"],
                       right_on=[PLATE_COL, PASS_COL],
                       how="inner")
                .sort_values([PLATE_COL, PASS_COL])
                .reset_index(drop=True))

    # 유일성
    if merged.duplicated([PLATE_COL, PASS_COL]).any():
        dup = int(merged.duplicated([PLATE_COL, PASS_COL]).sum())
        raise AssertionError(f"(PLATE, PASS) 중복 {dup}행 존재")

    # 타깃 생성 및 연속성
    merged[TARGET_COL] = merged.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged = merged.rename(columns={"warping_index": CUR_WARP})
    merged["_next_pass"] = merged.groupby(PLATE_COL)[PASS_COL].shift(-1)
    diff = merged["_next_pass"] - merged[PASS_COL]
    nonconsec = (diff.notna()) & (diff != 1)
    n_bad = int(nonconsec.sum())
    if n_bad > 0:
        print(f"[WARN] PASS(t)→PASS(t+1) 불연속 {n_bad}행: 타깃 제거 후 드롭")
        merged.loc[nonconsec, TARGET_COL] = np.nan
    GLOBAL_DIAG["nonconsec_dropped"] = n_bad
    merged = merged.drop(columns=["_next_pass"])

    # 마지막 PASS, 불연속 제거
    merged = merged.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # 불필요 열 제거
    final_df = merged.drop(columns=["extracted_plate","extracted_pass","filename",
                                    "quality_grade","quality_grade_current_pass","direction"],
                           errors="ignore")
    PRO_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PRO_DIR/"final_merged_data_regression_tplus1.csv", index=False)
    return final_df

def split_e2x_x2e(df: pd.DataFrame):
    e2x = df[df[PASS_COL] % 2 == 1].copy()
    x2e = df[df[PASS_COL] % 2 == 0].copy()
    # 참고용 저장(파이프라인은 메모리 객체 사용)
    (PRO_DIR/"e2x_raw_tplus1.csv").write_text(e2x.to_csv(index=False))
    (PRO_DIR/"x2e_raw_tplus1.csv").write_text(x2e.to_csv(index=False))
    return e2x, x2e

# ------------------------- Feature Engineering -------------------------
def base_feature_engineering(df: pd.DataFrame, pass_max: float | None = None) -> pd.DataFrame:
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

# ------------------------ 이상치 처리 ------------------------
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

# ------------------------ 전략별 보조 ------------------------
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
    # PERMUTE_TARGET=1인 경우 앵커 캐시 간섭을 피하기 위해 접미어 추가
    anchor_key = f"{anchor_name}__perm" if PERMUTE_TARGET else anchor_name
    anchor_file = MODELS_DIR / f"{anchor_key}__{direction_key}_test_plates.pkl"

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

    # 최초 생성
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    # (라벨 분포 고려 split이지만, 라벨을 모델에 사용하진 않음 → 데이터 분할 메타정보로만 사용)
    tr_i, te_i = next(gss.split(df.drop(columns=[TARGET_COL]), df[TARGET_COL],
                                groups=df[PLATE_COL].astype(str).values))
    te_plates = sorted(df.iloc[te_i][PLATE_COL].astype(str).unique())
    joblib.dump(te_plates, anchor_file)
    return df.iloc[tr_i].copy(), df.iloc[te_i].copy(), set(te_plates)

# --------------------------- Modeling Core -----------------------------
def select_features(Xtr_df: pd.DataFrame, ytr: np.ndarray,
                    Xte_df: pd.DataFrame, mandatory=None, k=SELECTK_K):
    mandatory = list(mandatory or [])
    all_cols  = list(Xtr_df.columns)

    # 사전 정화
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

    # F-score(상관 기반)
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
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0,
        n_jobs=0,
        eval_metric="rmse",
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

    # 내부 홀드아웃(plate group)
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
                          eval_set=[(Xval, yval)], early_stopping_rounds=50, verbose=False)
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
                  eval_set=[(Xval, yval)], early_stopping_rounds=50, verbose=False)
        except TypeError:
            m.fit(Xfit, yfit, sample_weight=sw_fit)
    else:
        m.fit(Xfit, yfit, sample_weight=sw_fit)
    return m, m.predict(Xte)

# ----------- Yeo-Johnson 변환 유틸 ---------
def fit_yj_transformer(y_tr: np.ndarray):
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(y_tr.reshape(-1,1))
    return pt

def yj_transform(pt: PowerTransformer, y: np.ndarray) -> np.ndarray:
    return pt.transform(y.reshape(-1,1)).ravel()

def yj_inverse(pt: PowerTransformer, y_t: np.ndarray) -> np.ndarray:
    return pt.inverse_transform(y_t.reshape(-1,1)).ravel()

# ----------- OOF XGB 유틸 ---------
def _oof_predict_xgb(X: np.ndarray, y: np.ndarray, groups: np.ndarray, chosen_names: list[str],
                     flags: set, seed=SEED):
    gkf = GroupKFold(n_splits=CV_FOLDS)
    oof = np.zeros(len(y), dtype=float); models=[]
    mono_idx = (chosen_names.index(CUR_WARP) if ("MONO_CUR" in flags and CUR_WARP in chosen_names) else None)
    for k,(tr_i,va_i) in enumerate(gkf.split(X, y, groups)):
        m = _build_xgb(("ROBUST_LOSS" in flags), monotone_index=mono_idx, n_features=len(chosen_names), seed=seed+k)
        try: m.fit(X[tr_i], y[tr_i], eval_set=[(X[va_i], y[va_i])], early_stopping_rounds=50, verbose=False)
        except TypeError: m.fit(X[tr_i], y[tr_i])
        oof[va_i] = m.predict(X[va_i]); models.append(m)
    mf = _build_xgb(("ROBUST_LOSS" in flags), monotone_index=mono_idx, n_features=len(chosen_names), seed=seed+777)
    try: mf.fit(X, y, eval_set=[(X, y)], early_stopping_rounds=10, verbose=False)
    except TypeError: mf.fit(X, y)
    return oof, mf

# ----------- X2E 강화 트레이너 ---------
def train_x2e_boosted(train_df: pd.DataFrame, test_df: pd.DataFrame, flags: set):
    base_flags = set(flags) | ({"MONO_CUR"} if X2E_MONO else set())

    # 숫자/대치/이름 차단
    Xtr_df, med, cols = _ensure_numeric_X(train_df)
    Xte_df = test_df.drop(columns=[c for c in EXCLUDE if c in test_df.columns], errors="ignore") \
                    .select_dtypes(include=[np.number]).reindex(columns=cols, fill_value=np.nan).fillna(med)

    # 값 기반 누수 차단(완전/준완전)
    Xtr_df, Xte_df = _assert_and_drop_value_leak(Xtr_df, train_df[TARGET_COL].values, Xte_df, test_df[TARGET_COL].values)

    y_raw_tr = train_df[TARGET_COL].values
    y_raw_te = test_df[TARGET_COL].values

    use_yj = ("X2E_YJ" in flags)
    if use_yj:
        pt = fit_yj_transformer(y_raw_tr); y_tr = yj_transform(pt, y_raw_tr)
    else:
        pt = None; y_tr = y_raw_tr

    chosen, Xtr, Xte = select_features(Xtr_df, y_tr, Xte_df, mandatory=[CUR_WARP], k=SELECTK_K)
    groups = train_df[PLATE_COL].astype(str).values

    if "X2E_MOE" in flags:
        tr_pair = _pair_str(train_df); te_pair = _pair_str(test_df)
        yhat_te_t = np.zeros(len(test_df), dtype=float)
        vc = tr_pair.value_counts()
        good_pairs = vc[vc >= max(40, Xtr.shape[1]*3)].index.tolist()
        for p in te_pair.unique().tolist():
            te_mask = (te_pair==p).values
            tr_mask = (tr_pair==p).values
            if p not in good_pairs:
                oof_all, mf_all = _oof_predict_xgb(Xtr, y_tr, groups, chosen, base_flags, seed=SEED+31)
                yhat_te_t[te_mask] = mf_all.predict(Xte[te_mask])
            else:
                Xtr_p, y_tr_p, grp_p = Xtr[tr_mask], y_tr[tr_mask], groups[tr_mask]
                oof_p, mf_p = _oof_predict_xgb(Xtr_p, y_tr_p, grp_p, chosen, base_flags, seed=SEED+33)
                yhat_te_t[te_mask] = mf_p.predict(Xte[te_mask])

        mf_rep = _build_xgb(("ROBUST_LOSS" in base_flags),
                            monotone_index=(chosen.index(CUR_WARP) if ("MONO_CUR" in base_flags and CUR_WARP in chosen) else None),
                            n_features=len(chosen), seed=SEED+99)
        try: mf_rep.fit(Xtr, y_tr, eval_set=[(Xtr, y_tr)], early_stopping_rounds=10, verbose=False)
        except TypeError: mf_rep.fit(Xtr, y_tr)
        yhat_tr_t = mf_rep.predict(Xtr)

        if use_yj:
            yhat_te = yj_inverse(pt, yhat_te_t); yhat_tr = yj_inverse(pt, yhat_tr_t)
        else:
            yhat_te, yhat_tr = yhat_te_t, yhat_tr_t

        return {"model":"X2E_EXPERTS","features":chosen,
                "r2_tr":float(r2_score(y_raw_tr,yhat_tr)),"r2_te":float(r2_score(y_raw_te,yhat_te)),
                "rmse_tr":float(np.sqrt(mean_squared_error(y_raw_tr,yhat_tr))),
                "rmse_te":float(np.sqrt(mean_squared_error(y_raw_te,yhat_te))),
                "yte":y_raw_te,"yhat":yhat_te}

    if "X2E_RES" in flags:
        oof1_t, mf1 = _oof_predict_xgb(Xtr, y_tr, groups, chosen, base_flags, seed=SEED+11)
        yhat1_te_t = mf1.predict(Xte)
        yres_t = y_tr - oof1_t
        Xtr_res = np.c_[Xtr, oof1_t]; Xte_res = np.c_[Xte, yhat1_te_t]
        chosen2 = chosen + ["oof_pred_stage1"]
        oof2_t, mf2 = _oof_predict_xgb(Xtr_res, yres_t, groups, chosen2, base_flags, seed=SEED+22)
        yhat_te_t = yhat1_te_t + mf2.predict(Xte_res)
        yhat_tr_t = (oof1_t + oof2_t)
        if use_yj:
            yhat_te = yj_inverse(pt, yhat_te_t); yhat_tr = yj_inverse(pt, yhat_tr_t)
        else:
            yhat_te, yhat_tr = yhat_te_t, yhat_tr_t

        return {"model":"X2E_RESIDUAL","features":chosen2,
                "r2_tr":float(r2_score(y_raw_tr,yhat_tr)),"r2_te":float(r2_score(y_raw_te,yhat_te)),
                "rmse_tr":float(np.sqrt(mean_squared_error(y_raw_tr,yhat_tr))),
                "rmse_te":float(np.sqrt(mean_squared_error(y_raw_te,yhat_te))),
                "yte":y_raw_te,"yhat":yhat_te}

    # 기본 경로(YJ 옵션 포함)
    tr = train_df.copy(); te = test_df.copy()
    tr["_target"] = tr[TARGET_COL]; te["_target"] = te[TARGET_COL]
    if use_yj:
        pt2 = fit_yj_transformer(tr["_target"].values)
        ytr_t = yj_transform(pt2, tr["_target"].values)
        chosen_p, Xtr_p, Xte_p = select_features(Xtr_df, ytr_t, Xte_df, mandatory=[CUR_WARP], k=SELECTK_K)
        groups = tr[PLATE_COL].astype(str).values
        model, yhat_t = _fit_predict_model(Xtr_p, ytr_t, Xte_p, base_flags, "XGBoost", chosen_p,
                                           sample_weight=None, groups=groups)
        yhat = yj_inverse(pt2, yhat_t)
        return {"model": model, "features": chosen_p,
                "r2_tr": float(r2_score(tr[TARGET_COL].values, yj_inverse(pt2, model.predict(Xtr_p)))),
                "r2_te": float(r2_score(te[TARGET_COL].values, yhat)),
                "rmse_tr": float(np.sqrt(mean_squared_error(tr[TARGET_COL].values, yj_inverse(pt2, model.predict(Xtr_p))))),
                "rmse_te": float(np.sqrt(mean_squared_error(te[TARGET_COL].values, yhat))),
                "yte": te[TARGET_COL].values, "yhat": yhat}

    return train_single_model(train_df, test_df, "X2E", base_flags, "XGBoost")

def train_single_model(train_df, test_df, direction_key: str, flags: set, algo_name: str):
    tr = train_df.copy(); te = test_df.copy()

    if "PAIR_TE" in flags:
        tr, te = add_pair_delta_te_feature(tr, te, n0=PAIR_EB_N0)

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

    # X, y (이름 기반 차단)
    Xtr_df, med, cols = _ensure_numeric_X(tr)
    Xte_df = te.drop(columns=[c for c in EXCLUDE if c in te.columns], errors="ignore") \
               .select_dtypes(include=[np.number]).reindex(columns=cols, fill_value=np.nan).fillna(med)

    # 값 기반 차단(완전/준완전)
    Xtr_df, Xte_df = _assert_and_drop_value_leak(Xtr_df, tr["_target"].values, Xte_df, te["_target"].values)

    ytr = tr["_target"].values; yte = te["_target"].values
    chosen_names, Xtr, Xte = select_features(Xtr_df, ytr, Xte_df, mandatory=mandatory, k=SELECTK_K)
    groups = tr[PLATE_COL].astype(str).values
    model, yhat = _fit_predict_model(Xtr, ytr, Xte, flags, algo_name, chosen_names,
                                     sample_weight=sample_weight, groups=groups)

    if "DELTA_TARGET" in flags:
        yhat = te[CUR_WARP].values + yhat

    return {"model": model, "features": chosen_names,
            "r2_tr": float(r2_score(ytr, model.predict(Xtr))),
            "r2_te": float(r2_score(yte, yhat)),
            "rmse_tr": float(np.sqrt(mean_squared_error(ytr, model.predict(Xtr)))),
            "rmse_te": float(np.sqrt(mean_squared_error(yte, yhat))),
            "yte": yte, "yhat": yhat}

# ----------------------- 빠른 퍼뮤테이션 누수 감사 ----------------------
def _quick_perm_leak_r2(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        algo_name: str, seed: int = SEED+7) -> float:
    rng = np.random.RandomState(seed)
    tr = train_df.copy(); te = test_df.copy()
    ytr = rng.permutation(tr[TARGET_COL].values)
    yte = te[TARGET_COL].values

    Xtr_df, med, cols = _ensure_numeric_X(tr)
    Xte_df = te.drop(columns=[c for c in EXCLUDE if c in te.columns], errors="ignore") \
               .select_dtypes(include=[np.number]).reindex(columns=cols, fill_value=np.nan).fillna(med)

    # 값 기반 차단(안전)
    Xtr_df, Xte_df = _assert_and_drop_value_leak(Xtr_df, ytr, Xte_df, yte)

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

# ------------------------ 디버그: 상관 리포트 ---------------------
def _corr_topk(df: pd.DataFrame, k=12):
    if df.empty or TARGET_COL not in df.columns: return []
    X = df.drop(columns=[c for c in EXCLUDE if c in df.columns], errors="ignore") \
          .select_dtypes(include=[np.number]).copy()
    X = _drop_name_leaky_cols(X)
    med = X.median(numeric_only=True); X = X.fillna(med)
    y = df[TARGET_COL].values
    out=[]
    for c in X.columns:
        if c == CUR_WARP:  # 디버그 보고에 포함은 허용
            pass
        xc = X[c].values
        sx = np.std(xc); sy = np.std(y)
        if sx < 1e-12 or sy < 1e-12: continue
        r = float(np.corrcoef(xc, y)[0,1])
        if np.isfinite(r):
            out.append((c, r, abs(r)))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:k]

def _append_corr_section(lines: list[str], label: str, df_full: pd.DataFrame, df_train_raw: pd.DataFrame, k=12):
    top_full = _corr_topk(df_full, k=k)
    top_tr   = _corr_topk(df_train_raw, k=k)
    lines.append(f"\n[DEBUG:{label}] 분할 전 타깃-특징 상관 Top-{k} (abs)")
    for c, r, _ in top_full: lines.append(f"- {c}: r={r:.4f}")
    lines.append(f"\n[DEBUG:{label}] (train-raw) 상관 Top-{k}")
    for c, r, _ in top_tr: lines.append(f"- {c}: r={r:.4f}")

# ------------------------------ Runner ---------------------------------
def _apply_fe_by_flags_safe(train_df: pd.DataFrame, test_df: pd.DataFrame, flags: set):
    pass_max_tr = float(train_df[PASS_COL].max()) if len(train_df) else 1.0
    tr = base_feature_engineering(train_df, pass_max=pass_max_tr)
    te = base_feature_engineering(test_df,  pass_max=pass_max_tr)

    if STRICT_NO_TRANSD:
        return tr, te

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

def _run_one(key: str, flags: set, tag: str, e2x_raw_all: pd.DataFrame, x2e_raw_all: pd.DataFrame, anchor_base: str):
    anchor_name = f"{anchor_base}__{key}__anchor" if ANCHOR_PER_STEP else f"{anchor_base}__anchor"
    use_strat = ("STRAT_SPLIT" in flags)
    e2x_tr_raw, e2x_te_raw, _ = split_by_anchor(e2x_raw_all, "E2X", use_strat, anchor_name)
    x2e_tr_raw, x2e_te_raw, _ = split_by_anchor(x2e_raw_all, "X2E", use_strat, anchor_name)

    if USE_RM_TRAINONLY:
        e2x_tr_raw, e2x_te_raw = attach_train_only_rm_stats(e2x_tr_raw, e2x_te_raw)
        x2e_tr_raw, x2e_te_raw = attach_train_only_rm_stats(x2e_tr_raw, x2e_te_raw)

    e2x_tr, e2x_te = _apply_fe_by_flags_safe(e2x_tr_raw, e2x_te_raw, flags)
    x2e_tr, x2e_te = _apply_fe_by_flags_safe(x2e_tr_raw, x2e_te_raw, flags)

    b_e2x, ylo_e2x, yhi_e2x = compute_train_outlier_bounds(e2x_tr)
    b_x2e, ylo_x2e, yhi_x2e = compute_train_outlier_bounds(x2e_tr)
    e2x_tr = apply_train_outlier_filter(e2x_tr, b_e2x, ylo_e2x, yhi_e2x)
    x2e_tr = apply_train_outlier_filter(x2e_tr, b_x2e, ylo_x2e, yhi_x2e)
    e2x_te = clip_test_features_to_bounds(e2x_te, b_e2x)
    x2e_te = clip_test_features_to_bounds(x2e_te, b_x2e)

    e2x_res = train_single_model(e2x_tr, e2x_te, "E2X", flags, ALGO_FIXED["E2X"])
    use_boost = (X2E_EXPERTS or X2E_RESIDUAL or X2E_MONO or ("X2E_YJ" in flags) or ("X2E_MOE" in flags) or ("X2E_RES" in flags))
    if use_boost:
        x2e_res = train_x2e_boosted(x2e_tr, x2e_te, flags)
    else:
        x2e_res = train_single_model(x2e_tr, x2e_te, "X2E", flags, ALGO_FIXED["X2E"])

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
        return (f"* {label} | Algo: {ALGO_FIXED.get(label,'XGBoost')} "
                f"| Train R2: {r['r2_tr']:.4f} | Test R2: {r['r2_te']:.4f} "
                f"(RMSE {r['rmse_tr']:.4f}/{r['rmse_te']:.4f})")
    lines.append("\n[모델 성능 - R2]")
    lines.append(fmt("E2X", e2x_res))
    lines.append(fmt("X2E", x2e_res))
    lines.append(f"\n[평균 Test R2] {avg_r2:.4f}")

    # 누수 Quick Audit
    if LEAK_AUTOCHECK:
        e2x_perm = _quick_perm_leak_r2(e2x_tr, e2x_te, ALGO_FIXED["E2X"])
        x2e_perm = _quick_perm_leak_r2(x2e_tr, x2e_te, ALGO_FIXED["X2E"])
        lines.append("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        lines.append(f"- E2X permuted R²: {e2x_perm:.4f}")
        lines.append(f"- X2E permuted R²: {x2e_perm:.4f}")
        if max(e2x_perm, x2e_perm) > 0.05:
            lines.append("[LEAK WARNING] permuted R² > 0.05 → 파이프라인/스플릿 경계 점검 필요")

    # 디버그/설정
    lines.append("\n[DEBUG] 실행 설정")
    lines.append(f"- PERMUTE_TARGET: {GLOBAL_DIAG['permute']} (seed={GLOBAL_DIAG['permute_seed']})")
    lines.append(f"- ANCHOR_FROM_CROSS: {GLOBAL_DIAG['anchor_from_cross']}")
    lines.append(f"- STRICT_NO_TRANSD: {GLOBAL_DIAG['strict_no_transd']}")
    lines.append(f"- LEAK_AUTOCHECK: {GLOBAL_DIAG['leak_autocheck']}")
    lines.append(f"- HARD_LEAK_GUARD: {GLOBAL_DIAG['hard_leak_guard']}")
    lines.append(f"- PASS 불연속 드롭 수: {GLOBAL_DIAG['nonconsec_dropped']}")

    if GLOBAL_DIAG["name_drop_cols"]:
        lines.append("\n[LEAK-GUARD] 이름 패턴으로 제거된 열")
        for c in sorted(set(GLOBAL_DIAG["name_drop_cols"])):
            lines.append(f"- {c}")

    if GLOBAL_DIAG["eq_drop_cols"]:
        lines.append("\n[LEAK-GUARD] 타깃과 동일/준동일로 제거된 열")
        for c in sorted(set(GLOBAL_DIAG["eq_drop_cols"])):
            lines.append(f"- {c}")

    # 분할 통계
    lines.append("\n[DEBUG] 분할 통계")
    lines.append(f"- E2X: train={len(e2x_tr_raw):5d} (plates={e2x_tr_raw[PLATE_COL].nunique()}), test={len(e2x_te_raw):5d} (plates={e2x_te_raw[PLATE_COL].nunique()})")
    lines.append(f"- X2E: train={len(x2e_tr_raw):5d} (plates={x2e_tr_raw[PLATE_COL].nunique()}), test={len(x2e_te_raw):5d} (plates={x2e_te_raw[PLATE_COL].nunique()})")

    # 분할 전/후(raw) 상관 Top-k (보고용)
    _append_corr_section(lines, "E2X", e2x_raw_all, e2x_tr_raw, k=12)
    _append_corr_section(lines, "X2E", x2e_raw_all, x2e_tr_raw, k=12)

    lines.append("="*72)
    lines.append(f"\n[산출물]\n- 단계별 결과 CSV: {steps_csv}\n- 코드 스냅샷: {code_snap}")
    summary_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 저장: {steps_csv}\n✓ 저장: {summary_txt}")

    return {"avg": avg_r2, "e2x": e2x_res["r2_te"], "x2e": x2e_res["r2_te"], "rf": np.nan, "anchor": anchor_name}

def _parse_recipes(run_only, run_with):
    recipes = {}
    def to_flags(expr: str):
        toks = [t.strip() for t in expr.split("+") if t.strip()]
        toks = [t for t in toks if t != "S0"]
        flags = []
        for t in toks: flags.extend(ALIAS2FLAGS.get(t, []))
        return flags
    for k in (run_only or []): recipes[k] = to_flags(k)
    for k in (run_with or []): recipes[k] = to_flags(k)
    if "S0" not in recipes: recipes["S0"] = []
    return recipes

def main():
    global experiment_id
    experiment_id = _next_experiment_id()

    print("="*80)
    print(f"[RUN_NAME] {RUN_NAME} | Optuna-free")
    print("="*80)

    final_df = load_and_merge_tplus1()

    if PERMUTE_TARGET:
        rng = np.random.RandomState(SEED)
        final_df[TARGET_COL] = rng.permutation(final_df[TARGET_COL].values)  # 음성 대조군
        # 콘솔 미출력(요약 .txt에 기록)

    e2x_raw, x2e_raw = split_e2x_x2e(final_df)
    run_only_list = _env_list("RUN_ONLY_EXPR", RUN_ONLY)
    run_with_list = _env_list("RUN_WITH_EXPR", RUN_WITH)
    recipes = _parse_recipes(run_only_list, run_with_list)

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
