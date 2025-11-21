# -*- coding: utf-8 -*-
"""
003_r2_uplift_suite.py  (leak-safe)
- RUN_ONLY / RUN_WITH 로 S0, R1, S1, TE2, W2, T2, M2, SEG, IW, BAG 시동
- XGBoost / RandomForest only (LightGBM/Optuna 없음)
- 누수 방지:
  (1) 반드시 'plate-Group split'을 먼저 수행한 뒤(train/test 분할) 모든 특성/정규화/이상치처리/OOF/통계 적합을 train 기준으로만 산출
  (2) Δ-TE/분산가중/중요도가중은 OOF 또는 train-only 적합 후 test에 매핑
  (3) test 행은 삭제하지 않음(분포 보존). 필요 시 feature만 train 범위로 clip
- 저장 충돌 방지: RUN_NAME+STEP 접두사 및 자동 experiment_id
"""

# ========================= [ USER CONFIG ] =========================
RUN_NAME = "r2_uplift_alias_v3"   # 동시 다실험 시 서로 다른 이름으로 변경
RUN_ONLY = ["S0"] #, "R1", "S1", "TE2", "W2", "T2", "M2", "SEG", "IW", "BAG"]  # 예: ["S0","W2","TE2"]
RUN_WITH = []                     # 예: ["S0+R1","R1+S1"]

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
# (선택) Strict roll-forward
DO_STRICT_ROLLFORWARD = False

IW_MODE = "oof"   # {"oof", "test"}; "test" reproduces prior behavior
# ==================================================================

import os, re, shutil, warnings, traceback, joblib
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
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
#             'PAIR_TE','MONO_CUR','BAGGING'
ALIAS2FLAGS = {
    "S0":   [],
    "L1":   ["SEQ_LAG"],   # ← 랙을 켜고 싶을 때 조합 사용(S0+L1)
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
    "R1":  "Robust Loss (XGB absolute)",
    "S1":  "Stratified Group Split (쌍(pair) 분포 유사도 최소화)",
    "TE2": "Pair Δ Target Encoding(OOF, EB 수축) → 특성 추가",
    "W2":  "Heteroscedastic Weights (쌍 Δ-분산 OOF+EB 기반 WLS)",
    "T2":  "Δ‑Target 회귀 (ŷ= y(t) + Δ̂)",
    "M2":  "Monotone constraint on CUR_WARP (+1) for XGB",
    "SEG": "Phase Segmentation(MoE) — 초/중/후 소전문가 + 글로벌 백업",
    "IW":  "Importance Weighting (Train↔Test 로지스틱 밀도비)",
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

def _build_xgb(robust=False, monotone_index=None, n_features=None, seed=SEED):
    params=dict(random_state=seed, tree_method="hist",
                n_estimators=500, learning_rate=0.05,
                max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=0)
    if monotone_index is not None and n_features is not None:
        v=[0]*n_features
        if 0 <= monotone_index < n_features: v[monotone_index]=1
        params["monotone_constraints"]=tuple(v)
    # robust는 안전하게 absoluteerror만 사용(이전 pseudo-huber 발산 방지)
    return XGBRegressor(objective=("reg:absoluteerror" if robust else "reg:squarederror"), **params)

def _build_rf(seed=SEED):
    return RandomForestRegressor(random_state=seed, n_estimators=600, n_jobs=-1)

def _save_code_snapshot(exp_id: str):
    dst = SCRIPTS_DIR / f"{exp_id}_{Path(__file__).name}"
    try:
        shutil.copy2(THIS_PATH, dst)
    except Exception:
        pass
    return dst

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
    # RM의 'warping_index'는 제외
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

# ------------------------- Base FE -------------------------
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
    IW_MODE == "oof": train만으로 OOF 로지스틱 분류 → 밀도비 근사(누수 엄격).
    IW_MODE == "test": 과거 방식(train+test 입력 사용; transductive).
    """
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
    # (옵션) 003 실험 앵커 강제 재현
    if os.environ.get("ANCHOR_FROM_CROSS") == "1":
        cross_file = MODELS_DIR / f"{direction_key.lower()}_test_plates_tplus1.pkl"
        if cross_file.exists():
            test_plates = set(joblib.load(cross_file))
            tr = df[~df[PLATE_COL].astype(str).isin(test_plates)].copy()
            te = df[df[PLATE_COL].astype(str).isin(test_plates)].copy()
            return tr, te, test_plates
    anchor_file = MODELS_DIR / f"{anchor_name}__{direction_key}_test_plates.pkl"

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

# --------------------------- Modeling Core ---------------------------
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

def _fit_predict_model(Xtr, ytr, Xte, flags, algo_name, chosen_names, sample_weight=None):
    """단일/Bagging/Monotone/Robust 옵션 반영해 학습 및 예측"""
    # 단조 제약: 선택된 피처 중 CUR_WARP 위치 탐지
    mono_idx = chosen_names.index(CUR_WARP) if ("MONO_CUR" in flags and CUR_WARP in chosen_names and algo_name=="XGBoost") else None

    def make_one(seed):
        return _build_xgb("ROBUST_LOSS" in flags, monotone_index=mono_idx, n_features=len(chosen_names), seed=seed) \
               if algo_name=="XGBoost" else _build_rf(seed=seed)

    if "BAGGING" in flags:
        seeds = [SEED+i for i in range(BAG_K)]
        preds=[]; mdl=None
        for s in seeds:
            m = make_one(s); m.fit(Xtr, ytr, sample_weight=sample_weight); preds.append(m.predict(Xte))
            mdl = m  # 마지막 모델 메타 저장용(대표)
        pte = np.mean(preds, axis=0)
        return mdl, pte
    else:
        mdl = make_one(SEED); mdl.fit(Xtr, ytr, sample_weight=sample_weight); return mdl, mdl.predict(Xte)

def train_single_model(train_df, test_df, direction_key: str, flags: set, algo_name: str):
    """
    누수-세이프 학습/예측:
      - test_df에는 어떤 형태로도 라벨/파생 라벨 컬럼을 만들지 않는다.
      - 피처 행렬 구성 시 금지 컬럼(_target, delta, pair 등)을 강제로 제외한다.
      - 선택 피처에 금지 컬럼이 섞이면 즉시 예외를 발생(가드).
    """
    tr = train_df.copy()
    te = test_df.copy()

    # --- TE2: Δ-타깃 인코딩 (train=OOF, test=full-train 매핑) ---
    if "PAIR_TE" in flags:
        tr, te = add_pair_delta_te_feature(tr, te, n0=PAIR_EB_N0)  # 안전 구현(OOF) 유지

    # --- 타깃 구성(지역 변수만) ---
    use_delta = ("DELTA_TARGET" in flags)
    if use_delta:
        y_tr = tr[TARGET_COL].values - tr[CUR_WARP].values
        # te에서는 라벨이 필요 없다(평가 때만 사용). 절대 te에 _target을 만들지 않음.
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
            # transductive 모드 금지(IW_MODE == "test"면 누수 위험)
            if IW_MODE == "test":
                raise RuntimeError("IMP_WEIGHT with IW_MODE='test' is disallowed (leak risk). Set IW_MODE='oof'.")
            iw = compute_importance_weights(tr, te)
            if len(iw) == len(tr): w *= iw
        sample_weight = w / (np.mean(w) + 1e-8)

    # --- 피처 행렬(금지 컬럼 제거) ---
    FORBIDDEN = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL,
                 "_target", "delta", "pair"}  # 내부 보조 라벨/키
    trX_df = tr.drop(columns=[c for c in FORBIDDEN if c in tr.columns], errors="ignore")
    teX_df = te.drop(columns=[c for c in FORBIDDEN if c in te.columns], errors="ignore")

    # 수치형만, 같은 중앙값/열순서 사용
    Xtr_all, med, cols = _ensure_numeric_X(trX_df)
    Xte_all = teX_df[cols].fillna(med)

    # --- 피처 선택(필수 피처 강제 포함) ---
    chosen, Xtr, Xte = select_features(Xtr_all, y_tr, Xte_all,
                                       mandatory=mandatory, k=SELECTK_K)

    # 가드: 금지 컬럼이 선택되었는지 확인(이론상 위에서 제거했지만 2중 안전)
    bad = [c for c in chosen if c in FORBIDDEN or c == TARGET_COL]
    if bad:
        raise RuntimeError(f"Leaky features found in 'chosen': {bad}")

    # --- 학습/예측 ---
    model, yte_hat_core = _fit_predict_model(Xtr, y_tr, Xte, flags, algo_name, chosen,
                                             sample_weight=sample_weight)
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

    # --- 누수 가드(추가) : 선택 피처와 라벨 동등/과상관 탐지 ---
    def _post_leak_checks():
        # 1) 테스트 라벨과 동일/거의 동일한 피처가 있는지(완전 일치)
        Xte_df_chosen = teX_df[chosen].fillna(med)
        yte = te[TARGET_COL].values
        for c in chosen:
            v = Xte_df_chosen[c].values
            if len(v) == len(yte) and np.allclose(v, yte, atol=1e-9, rtol=0.0):
                raise RuntimeError(f"Leak detected: feature '{c}' equals test target exactly.")
        # 2) 상관 상위 피처 로깅(디버깅용)
        try:
            corr = pd.DataFrame({"feat": chosen})
            corr["pearson_r"] = [np.corrcoef(Xte_df_chosen[c].values, yte)[0,1] for c in chosen]
            corr = corr.replace([np.inf, -np.inf], np.nan).dropna().sort_values("pearson_r", ascending=False)
            top_corr = corr.head(5)
            (REPORTS_DIR / "leak_checks").mkdir(parents=True, exist_ok=True)
            top_corr.to_csv(REPORTS_DIR / "leak_checks" / f"top_corr_{direction_key}.csv", index=False)
        except Exception:
            pass

    _post_leak_checks()

    # 키(plate, pass) – Strict roll-forward용
    k_tr = list(zip(tr[PLATE_COL].astype(str).values, tr[PASS_COL].astype(int).values))
    k_te = list(zip(te[PLATE_COL].astype(str).values, te[PASS_COL].astype(int).values))

    bundle = dict(algo=algo_name, med=med, cols=cols, chosen=chosen, model=model, flags=sorted(list(flags)))
    return dict(bundle=bundle, r2_tr=r2_tr, r2_te=r2_te, rmse_tr=rmse_tr, rmse_te=rmse_te,
                pred_tr=ytr_hat, pred_te=yte_hat, keys_tr=k_tr, keys_te=k_te)

def train_moe_by_phase(train_df, test_df, direction_key: str, flags: set, algo_name: str):
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
                pred_tr=preds_tr, pred_te=preds_te)

# ----------------------------- Runner --------------------------------
def prepare_train_test_leak_safe(df_base: pd.DataFrame, use_strat: bool, direction_key: str, split_anchor: str, flags: set):
    """
    [중요] 누수 방지 파이프라인:
      1) 먼저 plate-group split
      2) 이후에 train/test 각각 FE/lag/expanding/delta 파생
      3) 이상치는 train에서만 제거, test는 feature clip만 적용
    """
    tr_raw, te_raw, test_plates = split_by_anchor(df_base, direction_key, use_strat=use_strat, anchor_name=split_anchor)
    # FE → lag → (선택) expanding/delta
    # 시퀀스 랙 사용 여부를 플래그로 제어
    USE_LAG = ("SEQ_LAG" in flags)
    tr = base_feature_engineering(tr_raw)
    te = base_feature_engineering(te_raw)
    if USE_LAG:
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

def run_one_direction(df_base: pd.DataFrame, direction_key: str, flags: set, split_anchor: str):
    tr, te, test_plates = prepare_train_test_leak_safe(
        df_base, use_strat=("STRAT_SPLIT" in flags), direction_key=direction_key, split_anchor=split_anchor, flags=flags
    )
    algo = ALGO_FIXED.get(direction_key, "XGBoost")
    res = train_moe_by_phase(tr, te, direction_key, flags, algo) if "MOE_PHASE" in flags else \
          train_single_model(tr, te, direction_key, flags, algo)
    res["test_plates"] = sorted(list(test_plates))
    return res

def strict_rollforward_r2(all_df: pd.DataFrame, e2x_test_plates: set, x2e_test_plates: set,
                          e2x_pred_map: dict, x2e_pred_map: dict):
    """(옵션) 교집합 Test 판만 roll-forward R² 계산 — 필요 시 구현/활성화"""
    test_plates = {str(p) for p in e2x_test_plates}.intersection({str(p) for p in x2e_test_plates})
    if not test_plates: return float("nan")
    rows=[]
    df = all_df[all_df[PLATE_COL].astype(str).isin(test_plates)].copy().sort_values([PLATE_COL, PASS_COL])
    for plate, sub in df.groupby(PLATE_COL):
        sub = sub.sort_values(PASS_COL)
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index: continue
            yhat = e2x_pred_map.get((plate, p), np.nan) if p%2==1 else x2e_pred_map.get((plate, p), np.nan)
            gt_next = float(idx.loc[p+1][CUR_WARP])
            if not (np.isnan(yhat) or np.isnan(gt_next)): rows.append({"pred": yhat, "gt": gt_next})
    if not rows: return float("nan")
    dfp = pd.DataFrame(rows)
    return float(r2_score(dfp["gt"], dfp["pred"]))

# ---------------------- Recipe Builder (RUN_ONLY/RUN_WITH) ----------------------
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
    # -------------------- 환경 토글 --------------------
    per_step_anchor = (os.environ.get("ANCHOR_PER_STEP", "0") == "1") # 스텝별 앵커(각 모드마다 다른 test plate)
    force_cross     = (os.environ.get("ANCHOR_FROM_CROSS", "0") == "1")  # 003 저장 테스트판을 강제 사용(apple‑to‑apple)
    do_permute      = (os.environ.get("PERMUTE_TARGET", "1") == "1") # plate 내부 라벨 셔플을 사용한 누수 자가진단 러닝 추가

    # -------------------- 데이터 -----------------------
    final_df = load_and_merge_tplus1()
    e2x_raw, x2e_raw = split_e2x_x2e(final_df)

    # (옵션) 타깃 퍼뮤테이션용 데이터 준비
    if do_permute:
        rng = np.random.RandomState(SEED)
        perm = final_df.copy()
        perm[TARGET_COL] = perm.groupby(PLATE_COL)[TARGET_COL] \
                               .transform(lambda s: s.sample(frac=1.0, random_state=int(rng.randint(1, 10**9))).values)
        e2x_perm, x2e_perm = split_e2x_x2e(perm)
    else:
        e2x_perm = x2e_perm = None

    exp_id = _next_experiment_id()
    code_snap = _save_code_snapshot(exp_id)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recipes = build_recipes_from_user()

    # 결과 누적(리더보드)
    rows = []

    def _run_one(key: str, flags: set, tag: str,
                 e2x_src: pd.DataFrame, x2e_src: pd.DataFrame, anchor_base: str):
        """단일 러닝 + 요약/가드"""
        # 공정 비교: 기본은 RUN 전체 공통 앵커, 옵션으로 스텝별
        anchor_name = f"{RUN_NAME}__{key}__anchor" if per_step_anchor else f"{RUN_NAME}__anchor"

        print("\n" + "="*80)
        print(f"[RUN] {tag} | step={key} | flags={sorted(list(flags))} | anchor={anchor_name}")
        print("="*80)

        e2x_res = run_one_direction(e2x_src, "E2X", flags, anchor_name)
        x2e_res = run_one_direction(x2e_src, "X2E", flags, anchor_name)
        avg_r2 = float(np.mean([e2x_res["r2_te"], x2e_res["r2_te"]]))

        # --- 분할 가드: 학습/검증 plate 완전 분리 확인 ---
        def _check_disjoint(tr_res, te_res):
            # prepare_train_test_leak_safe()의 내부 분할 결과는 반환치로 직접 오지 않으므로,
            # test plate만 검출 가능. 대신 앵커 기반 저장 파일을 신뢰(plate 단위 split).
            # 교집합이 있으면 split 함수 오류/캐시 오염 가능성.
            pass  # plate 기반 Group split로 설계, anchor 파일이 test plate 전체를 규정

        # --- Strict Roll-forward (교집합 Test) ---
        e2x_pred_map = { (p,pa): float(v) for (p,pa), v in zip(e2x_res["keys_te"], e2x_res["pred_te"]) }
        x2e_pred_map = { (p,pa): float(v) for (p,pa), v in zip(x2e_res["keys_te"], x2e_res["pred_te"]) }
        rf_strict = strict_rollforward_r2(final_df,
                                          set(e2x_res["test_plates"]), set(x2e_res["test_plates"]),
                                          e2x_pred_map, x2e_pred_map)

        # --- 저장 ---
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        steps_csv   = CSV_DIR / f"{exp_id}_{RUN_NAME}_{tag}_steps.csv"
        summary_txt = SUM_DIR / f"{exp_id}_{RUN_NAME}_{tag}_summary.txt"
        pd.DataFrame([{"step": tag, "E2X": e2x_res["r2_te"], "X2E": x2e_res["r2_te"], "Avg": avg_r2,
                       "RF_strict": rf_strict, "anchor": anchor_name}]).to_csv(steps_csv, index=False)

        lines=[]
        lines.append(f"R² Uplift (alias) Summary ({timestamp})")
        lines.append("="*72)
        lines.append(f"실험 ID: {exp_id} | RUN_NAME: {RUN_NAME} | STEP: {tag}")
        lines.append(f"전략 플래그: {', '.join(sorted(list(flags))) if flags else 'S0 (전략 없음)'}\n")
        def fmt(tag2, r):
            return (f"* {tag2} | Algo: {ALGO_FIXED[tag2]} "
                    f"| Train R2: {r['r2_tr']:.4f} | Test R2: {r['r2_te']:.4f} "
                    f"(RMSE {r['rmse_tr']:.4f}/{r['rmse_te']:.4f})")
        lines.append("[모델 성능 - R2]")
        lines.append(fmt("E2X", e2x_res))
        lines.append(fmt("X2E", x2e_res))
        lines.append(f"\n[평균 Test R2] {avg_r2:.4f}")
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

        # (옵션) 타깃 퍼뮤테이션 – 같은 플래그로 한 번 더
        if do_permute:
            _run_one(key, flags, f"{key}__PERMUTE", e2x_perm, x2e_perm, RUN_NAME)

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

    # ---------- 퍼뮤테이션 경보 ----------
    if do_permute:
        red = lb[lb["step"].str.endswith("__PERMUTE")]
        if not red.empty:
            suspicious = red[(red["avg_test_r2"] > 0.1) | (red["E2X_test_r2"] > 0.1) | (red["X2E_test_r2"] > 0.1)]
            if len(suspicious):
                warn_path = REPORTS_DIR / "leak_checks" / f"{exp_id}_{RUN_NAME}_permute_warning.csv"
                warn_path.parent.mkdir(parents=True, exist_ok=True)
                suspicious.to_csv(warn_path, index=False)
                print(f"\n[경보] 퍼뮤테이션에서도 R²가 높습니다 → 누수 의심: {warn_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        Path("reports").mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR/"last_error_traceback.txt").write_text(tb, encoding="utf-8")
        print(f"[실패] {type(e).__name__}: {e}")
        print(tb)
