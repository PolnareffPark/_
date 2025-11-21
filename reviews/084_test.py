# -*- coding: utf-8 -*-
"""
065_test_no_op.py  (clean, leak-safe, flag-aware, anchor-fixed)

요구사항 요약
- 기존 베이스라인 흐름 유지(병합→분할→FE→옵션 플래그→학습/평가→요약/메트릭)
- 라벨 누수 차단: 하드 가드(컬럼명), OOF 계산, Train-only 집계/경계, test에는 clip/매핑만
- 플래그: Δ-Target(DT), Pair Δ-TE(TE), 이분산 가중(W), 단조(MONO), RM Train-only(RM)
- 앵커 분할 고정: plate-group split 결과 파일 저장/재사용
- X2E 알고리즘 선택: env X2E_ALGO in {"XGBoost","RandomForest"} (E2X는 기본 XGBoost, env로 재정의 가능)
- 메트릭 CSV: reports/metrics/last_run_metrics.csv (summary_path가 끝에서 2번째, avg_test_r2가 마지막 열)
"""

import os, re, warnings, shutil, traceback, joblib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------- Paths & Consts ----------------------------
DATA_DIR      = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR   = Path("reports")
SUM_DIR       = REPORTS_DIR / "summaries"
SCRIPTS_DIR   = REPORTS_DIR / "scripts"
METRICS_DIR   = REPORTS_DIR / "metrics"
MODELS_DIR    = Path("models")

for d in [PROCESSED_DIR, REPORTS_DIR, SUM_DIR, SCRIPTS_DIR, METRICS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_COL = "warping_index_target"      # t+1
CUR_WARP   = "warping_index_current_pass" # t
PLATE_COL  = "FM_날판번호"
PASS_COL   = "FM_PASS NO N"
MONTH_COL  = "FM_압연월"

# 베이스라인 설정
SEED       = int(os.environ.get("SEED", "42"))
TEST_SIZE  = float(os.environ.get("TEST_SIZE", "0.2"))
CV_FOLDS   = int(os.environ.get("CV_FOLDS", "3"))
SELECT_K   = int(os.environ.get("SELECT_K", "120"))

RUN_NAME   = os.environ.get("RUN_NAME", "pass_t_plus_1_cross")
ANCHOR_TAG = os.environ.get("ANCHOR_TAG", "default_anchor")

# 플래그(환경변수)
def _env_on(name: str, default="0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1","true","yes","y"}

PERMUTE_TARGET   = _env_on("PERMUTE_TARGET","0")
LEAK_AUTOCHECK   = _env_on("LEAK_AUTOCHECK","1")
USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET","0")    # DT
USE_PAIR_TE      = _env_on("USE_PAIR_TE","0")         # TE
USE_HETERO_W     = _env_on("USE_HETERO_W","0")        # W
USE_MONO_CUR     = _env_on("USE_MONO_CUR","0")        # MONO
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY","0")    # RM

# 알고리즘 선택 — E2X는 기본 XGB, X2E는 env로 선택(필요시 E2X_ALGO로 override 가능)
E2X_ALGO = os.environ.get("E2X_ALGO", "XGBoost")
X2E_ALGO = os.environ.get("X2E_ALGO", "XGBoost")  # "RandomForest" or "XGBoost"

# ------------------------------- IO Utils --------------------------------
def _next_experiment_id() -> str:
    nums = []
    for f in SUM_DIR.glob("*.txt"):
        m = re.match(r"(\d{3})_", f.name)
        if m:
            try: nums.append(int(m.group(1)))
            except: pass
    return f"{(max(nums)+1) if nums else 1:03d}"

def _hard_leak_guard(cols: list[str]) -> list[str]:
    """컬럼명으로 라벨/예측 스멜 전부 제거."""
    bad_kw = ("target","label","oof","te_","_te","pred","_hat","_target")
    out=[]
    for c in cols:
        low = c.lower()
        if any(k in low for k in bad_kw): 
            continue
        out.append(c)
    return out

# ----------------------- Data merge & target(t+1) ------------------------
def load_and_merge_tplus1() -> pd.DataFrame:
    gt_e = pd.read_csv(DATA_DIR/"entry_direction_results.csv")
    gt_x = pd.read_csv(DATA_DIR/"exit_direction_results.csv")
    gt = pd.concat([gt_e, gt_x], ignore_index=True)
    gt["extracted_plate"] = gt["filename"].str.extract(r"(PB\d+)")
    gt["extracted_pass"]  = gt["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    fm = pd.read_csv(DATA_DIR/"posco2_105190.csv")
    df = pd.merge(
        gt, fm,
        left_on=["extracted_plate","extracted_pass"],
        right_on=[PLATE_COL, PASS_COL],
        how="inner"
    ).sort_values([PLATE_COL, PASS_COL]).reset_index(drop=True)

    # t+1 target & 불연속 pass 제거
    df[TARGET_COL] = df.groupby(PLATE_COL)["warping_index"].shift(-1)
    df = df.rename(columns={"warping_index": CUR_WARP})
    df["_next_pass"] = df.groupby(PLATE_COL)[PASS_COL].shift(-1)
    # 불연속: next != pass+1 은 제거
    bad = df["_next_pass"].notna() & (df["_next_pass"] != df[PASS_COL] + 1)
    df.loc[bad, TARGET_COL] = np.nan
    df = df.drop(columns=["_next_pass"])

    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # 정리
    drop_cols = ["extracted_plate","extracted_pass","filename",
                 "quality_grade","quality_grade_current_pass","direction"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR/"final_merged_data_regression_tplus1.csv").write_text(df.to_csv(index=False))
    print(f"[MERGE] t→t+1 rows: {before} → {len(df)} (dropped {before-len(df)})")
    return df

def split_e2x_x2e(df: pd.DataFrame):
    e2x = df[df[PASS_COL] % 2 == 1].copy()
    x2e = df[df[PASS_COL] % 2 == 0].copy()
    e2x.to_csv(PROCESSED_DIR/"e2x_raw_tplus1.csv", index=False)
    x2e.to_csv(PROCESSED_DIR/"x2e_raw_tplus1.csv", index=False)
    return e2x, x2e

# ----------------------------- Feature Eng. ------------------------------
def _base_fe_train_test(tr: pd.DataFrame, te: pd.DataFrame):
    """기본 FE — PASS 진행도는 train의 max로 고정해 test에 재사용."""
    tr = tr.copy(); te = te.copy()
    fm_cols = [c for c in tr.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    def _apply(d: pd.DataFrame, pass_max: float):
        if fm_cols:
            blk = d[fm_cols]
            d["FM_mean"]  = blk.mean(axis=1)
            d["FM_std"]   = blk.std(axis=1)
            d["FM_max"]   = blk.max(axis=1)
            d["FM_min"]   = blk.min(axis=1)
            d["FM_range"] = d["FM_max"] - d["FM_min"]
            d["FM_cv"]    = d["FM_std"] / (d["FM_mean"].abs() + 1e-8)
        d["FM_PASS_squared"] = d[PASS_COL] ** 2
        pm = float(pass_max) if pass_max and pass_max > 0 else 1.0
        d["FM_PASS_progress"] = d[PASS_COL] / pm
        return d
    pmax = float(tr[PASS_COL].max()) if len(tr) else 1.0
    tr = _apply(tr, pmax); te = _apply(te, pmax)
    return tr, te

def _ensure_numeric_X(df: pd.DataFrame):
    """수치형만 + 하드 누수 가드 + 중앙값 대치 정보."""
    Xall = df.drop(columns=[c for c in [TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL] if c in df.columns], errors="ignore")
    cols = _hard_leak_guard(list(Xall.columns))
    X = Xall[cols].select_dtypes(include=[np.number]).copy()
    med = X.median(numeric_only=True).to_dict()
    return X.fillna(med), med, list(X.columns)

# ------------------------ Train-only augments ---------------------------
def add_pair_delta_te(train_df: pd.DataFrame, test_df: pd.DataFrame, n0=50):
    """OOF 안전 Pair Δ-TE: y−cur 평균을 EB 수축해 특성으로."""
    tr = train_df.copy(); te = test_df.copy()
    tr["pair"] = tr[PASS_COL].astype(int).astype(str) + "→" + (tr[PASS_COL]+1).astype(int).astype(str)
    te["pair"] = te[PASS_COL].astype(int).astype(str) + "→" + (te[PASS_COL]+1).astype(int).astype(str)
    tr["delta"] = tr[TARGET_COL] - tr[CUR_WARP]

    gkf = GroupKFold(n_splits=CV_FOLDS)
    groups = tr[PLATE_COL].astype(str).values
    oof = np.zeros(len(tr), dtype=float)
    for tr_i, va_i in gkf.split(tr, tr["delta"].values, groups):
        sub = tr.iloc[tr_i]
        gmean = float(sub["delta"].mean())
        stat = sub.groupby("pair")["delta"].agg(n="size", mean="mean").reset_index()
        w = stat["n"]/(stat["n"]+n0)
        stat["te"] = w*stat["mean"] + (1-w)*gmean
        m = dict(zip(stat["pair"], stat["te"]))
        oof[va_i] = tr.iloc[va_i]["pair"].map(m).fillna(gmean).values
    # test mapping(full-train)
    gmean = float(tr["delta"].mean())
    stat = tr.groupby("pair")["delta"].agg(n="size", mean="mean").reset_index()
    w = stat["n"]/(stat["n"]+n0)
    stat["te"] = w*stat["mean"] + (1-w)*gmean
    m_full = dict(zip(stat["pair"], stat["te"]))

    tr = tr.drop(columns=["pair","delta"], errors="ignore")
    te = te.copy()
    tr["pair_delta_te"] = oof
    te["pair_delta_te"] = te["pair"].map(m_full).fillna(gmean).values
    te = te.drop(columns=["pair"], errors="ignore")
    return tr, te

def compute_hetero_weights(train_df: pd.DataFrame, n0=50, clip=(0.25, 3.0)):
    """OOF 안전 쌍별 Δ 분산 → WLS 가중치(1/std)."""
    tr = train_df.copy()
    tr["pair"] = tr[PASS_COL].astype(int).astype(str) + "→" + (tr[PASS_COL]+1).astype(int).astype(str)
    tr["delta"] = tr[TARGET_COL] - tr[CUR_WARP]
    gkf = GroupKFold(n_splits=CV_FOLDS)
    groups = tr[PLATE_COL].astype(str).values
    oof_std = np.zeros(len(tr), dtype=float)

    for tr_i, va_i in gkf.split(tr, tr["delta"].values, groups):
        sub = tr.iloc[tr_i]
        gs = float(sub["delta"].std(ddof=1)) if len(sub) > 1 else 1.0
        stat = sub.groupby("pair")["delta"].agg(n="size", std="std").reset_index()
        w = stat["n"]/(stat["n"]+n0)
        stat["std_sh"] = w*stat["std"].fillna(gs) + (1-w)*gs
        m = dict(zip(stat["pair"], stat["std_sh"]))
        oof_std[va_i] = [m.get(p, gs) for p in tr.iloc[va_i]["pair"].values]

    gs = float(tr["delta"].std(ddof=1)) if len(tr) > 1 else 1.0
    w_tr = np.clip(gs/(oof_std + 1e-6), clip[0], clip[1])
    return w_tr / (w_tr.mean() + 1e-8)

def attach_train_only_rm_stats(train_df: pd.DataFrame, test_df: pd.DataFrame):
    D = pd.read_csv(DATA_DIR/"posco1_105190.csv")
    plates_tr = set(train_df[PLATE_COL].astype(str).unique())
    rm = D[D["RM_날판번호"].astype(str).isin(plates_tr)].copy()
    if rm.empty:
        return train_df.copy(), test_df.copy()
    num_cols = rm.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"RM_날판번호", "RM_압연Pass번호", "warping_index"}
    stat_cols = [c for c in num_cols if c not in drop_like]
    rm_stats = rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"])
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    rm_stats = rm_stats.reset_index()

    tr = pd.merge(train_df, rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left")
    te = pd.merge(test_df,  rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left")
    tr = tr.drop(columns=["RM_날판번호"], errors="ignore")
    te = te.drop(columns=["RM_날판번호"], errors="ignore")

    if not rm_stats.empty:
        med = rm_stats.drop(columns=["RM_날판번호"]).median(numeric_only=True)
        for c in med.index:
            if c in te.columns: te[c] = te[c].fillna(float(med[c]))
            if c in tr.columns: tr[c] = tr[c].fillna(float(med[c]))
    return tr, te

# ------------------------------ Split(Anchor) ---------------------------
def _anchor_split(df: pd.DataFrame, prefix: str):
    """plate-group split 고정. 파일 존재 시 재사용."""
    anchor_file = MODELS_DIR / f"{prefix}__{ANCHOR_TAG}__test_plates.pkl"
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    y = df[TARGET_COL].values
    groups = df[PLATE_COL].astype(str).values

    if anchor_file.exists():
        test_plates = set(joblib.load(anchor_file))
        tr = df[~df[PLATE_COL].astype(str).isin(test_plates)].copy()
        te = df[ df[PLATE_COL].astype(str).isin(test_plates)].copy()
        return tr, te, test_plates

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_i, te_i = next(gss.split(X, y, groups=groups))
    te_plates = sorted(df.iloc[te_i][PLATE_COL].astype(str).unique())
    joblib.dump(te_plates, anchor_file)

    return df.iloc[tr_i].copy(), df.iloc[te_i].copy(), set(te_plates)

# ------------------------------ SelectKBest -----------------------------
def select_kbest_by_corr(Xtr_df: pd.DataFrame, ytr: np.ndarray,
                         Xte_df: pd.DataFrame, mandatory=None, k=SELECT_K):
    """상관 기반 K-best (수치 안정)."""
    mandatory = list(mandatory or [])
    all_cols = list(Xtr_df.columns)

    # 사전 정화
    finite_mask = np.isfinite(Xtr_df.values).all(axis=0)
    var = Xtr_df.var(numeric_only=True).reindex(all_cols).fillna(0.0).values
    keep_mask = finite_mask & (var > 1e-12)
    keep_cols = [c for c,m in zip(all_cols, keep_mask) if m]
    for c in mandatory:
        if c in all_cols and c not in keep_cols:
            keep_cols.append(c)

    Xtr = Xtr_df[keep_cols].copy()
    Xte = Xte_df.reindex(columns=keep_cols, fill_value=np.nan).copy()

    Xmed = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(Xmed); Xte = Xte.fillna(Xmed)

    # 안전 상관 F-score
    x = Xtr.values
    y = ytr - ytr.mean()
    y_std = y.std() or 1e-12
    xc = (x - x.mean(axis=0))
    xs = x.std(axis=0); xs[xs < 1e-12] = 1e-12
    xr = xc / xs; yr = y / y_std
    r  = (xr.T @ yr) / (len(y) - 1)
    r2 = np.clip(r**2, 0.0, 1.0-1e-12)
    F  = (r2/(1.0-r2)) * (len(y)-2)

    mand = [c for c in mandatory if c in keep_cols]
    others = [c for c in keep_cols if c not in mand]
    k_rem = max(1, min(int(k)-len(mand), len(others)))
    idx_map = {c:i for i,c in enumerate(keep_cols)}
    order = np.argsort([F[idx_map[c]] for c in others])[::-1]
    chosen = mand + [others[i] for i in order[:k_rem]]
    return chosen, Xtr[chosen].values, Xte[chosen].values

# ------------------------------- Models ---------------------------------
def _build_xgb(monotone_index=None, n_features=None, robust=False, seed=SEED):
    from xgboost import XGBRegressor
    params = dict(
        random_state=seed, tree_method="hist",
        n_estimators=800, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0, n_jobs=0,
        objective=("reg:absoluteerror" if robust else "reg:squarederror"),
        eval_metric="rmse",   # 생성자에만 지정(버전 호환)
    )
    if (monotone_index is not None) and (n_features is not None):
        v = [0]*n_features
        if 0 <= monotone_index < n_features: v[monotone_index] = 1
        params["monotone_constraints"] = tuple(v)
    return XGBRegressor(**params)

def _build_rf(seed=SEED):
    return RandomForestRegressor(
        random_state=seed, n_estimators=600, n_jobs=-1,
        max_depth=20, min_samples_leaf=5, max_features="sqrt",
    )

# ------------------------------ Train core ------------------------------
def train_one(train_df: pd.DataFrame, test_df: pd.DataFrame,
              direction_key: str, algo_name: str):
    """플래그 반영하여 한 방향(E2X/X2E) 학습."""
    tr = train_df.copy(); te = test_df.copy()

    # --- 플래그 전처리 ---
    # TE: Pair Δ-TE
    if USE_PAIR_TE:
        tr, te = add_pair_delta_te(tr, te)

    # 타깃 설정(Δ‑Target이면 y=target−cur_warp, 추론 시 복원)
    if USE_DELTA_TARGET:
        tr["_y"] = tr[TARGET_COL] - tr[CUR_WARP]
        te["_y"] = te[TARGET_COL] - te[CUR_WARP]
        mandatory = [CUR_WARP] + (["pair_delta_te"] if USE_PAIR_TE else [])
    else:
        tr["_y"] = tr[TARGET_COL]
        te["_y"] = te[TARGET_COL]
        mandatory = [CUR_WARP] + (["pair_delta_te"] if USE_PAIR_TE else [])

    # W: 이분산 가중치
    sample_weight = None
    if USE_HETERO_W:
        w = compute_hetero_weights(tr)
        sample_weight = w / (w.mean() + 1e-8)

    # RM: Train‑only 집계 부착
    if USE_RM_TRAINONLY:
        tr, te = attach_train_only_rm_stats(tr, te)

    # --- 수치 / 누수 가드 / SelectK ---
    Xtr_df, med, cols = _ensure_numeric_X(tr)
    Xte_df = te.drop(columns=[c for c in [TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL] if c in te.columns], errors="ignore") \
               .select_dtypes(include=[np.number]).reindex(columns=cols, fill_value=np.nan).fillna(med)

    ytr = tr["_y"].values; yte = te["_y"].values
    chosen, Xtr, Xte = select_kbest_by_corr(Xtr_df, ytr, Xte_df, mandatory=mandatory, k=SELECT_K)

    # 단조 제약(+1 on CUR_WARP) — XGB에서만
    mono_idx = (chosen.index(CUR_WARP) if (USE_MONO_CUR and (CUR_WARP in chosen) and (algo_name=="XGBoost")) else None)

    # 모델 생성/학습
    if algo_name == "XGBoost":
        model = _build_xgb(monotone_index=mono_idx, n_features=len(chosen), robust=False, seed=SEED)
        # 주의: 일부 xgboost 버전은 fit()의 early_stopping_rounds 미지원 → 사용 안 함
        model.fit(Xtr, ytr, sample_weight=sample_weight)
    else:
        model = _build_rf(seed=SEED)
        model.fit(Xtr, ytr, sample_weight=sample_weight)

    # 예측/복원
    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)
    if USE_DELTA_TARGET:
        # 평가/저장 시에는 실제 타깃 스케일로 환원
        yhat_tr = tr[CUR_WARP].values + yhat_tr
        yhat_te = te[CUR_WARP].values + yhat_te
        ytr_eval = tr[TARGET_COL].values
        yte_eval = te[TARGET_COL].values
    else:
        ytr_eval = ytr
        yte_eval = yte

    r2_tr = float(r2_score(ytr_eval, yhat_tr))
    r2_te = float(r2_score(yte_eval, yhat_te))
    rmse_tr = float(np.sqrt(mean_squared_error(ytr_eval, yhat_tr)))
    rmse_te = float(np.sqrt(mean_squared_error(yte_eval, yhat_te)))

    # 산출물 저장(선택 컬럼/중앙값/모델/앵커)
    prefix = "e2x" if direction_key == "E2X" else "x2e"
    joblib.dump(model,         MODELS_DIR / f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(chosen,        MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl")
    joblib.dump(med,           MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl")

    return {
        "algorithm": algo_name,
        "features": chosen,
        "r2_tr": r2_tr, "r2_te": r2_te,
        "rmse_tr": rmse_tr, "rmse_te": rmse_te,
        "yte": yte_eval, "yhat": yhat_te,
    }

# ------------------------------ Permute Audit ---------------------------
def quick_perm_r2(train_df: pd.DataFrame, test_df: pd.DataFrame, algo="XGBoost", seed=SEED+7):
    rng = np.random.RandomState(seed)
    tr = train_df.copy(); te = test_df.copy()
    ytr = rng.permutation(tr[TARGET_COL].values)
    yte = te[TARGET_COL].values
    Xtr_df, med, cols = _ensure_numeric_X(tr)
    Xte_df = te.drop(columns=[c for c in [TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL] if c in te.columns], errors="ignore") \
               .select_dtypes(include=[np.number]).reindex(columns=cols, fill_value=np.nan).fillna(med)
    chosen, Xtr, Xte = select_kbest_by_corr(Xtr_df, ytr, Xte_df, mandatory=[CUR_WARP] if CUR_WARP in Xtr_df.columns else [], k=min(60, SELECT_K))
    if algo == "XGBoost":
        m = _build_xgb(monotone_index=None, n_features=None, robust=False, seed=seed)
    else:
        m = _build_rf(seed=seed)
    m.fit(Xtr, ytr)
    return float(r2_score(yte, m.predict(Xte)))

# -------------------------------- Runner --------------------------------
def main():
    success = False
    experiment_id = _next_experiment_id()
    ts = datetime.now()
    summary_path = SUM_DIR / f"{experiment_id}_{RUN_NAME}_summary.txt"
    archived_script_path = SCRIPTS_DIR / f"{experiment_id}_{Path(__file__).name}"

    try:
        print("="*80)
        print(f"[RUN] {RUN_NAME} | ANCHOR_TAG={ANCHOR_TAG} | SEED={SEED}")
        print("="*80)

        df = load_and_merge_tplus1()
        if PERMUTE_TARGET:
            rng = np.random.RandomState(SEED)
            df[TARGET_COL] = rng.permutation(df[TARGET_COL].values)
            print("[PERMUTE] Target shuffled for leak audit.")

        e2x_raw, x2e_raw = split_e2x_x2e(df)

        # --- 앵커 분할 ---
        e2x_tr_raw, e2x_te_raw, e2x_plates = _anchor_split(e2x_raw, "e2x")
        x2e_tr_raw, x2e_te_raw, x2e_plates = _anchor_split(x2e_raw, "x2e")
        # 앵커 plate 목록 저장(재확인)
        joblib.dump(sorted(list(e2x_plates)), MODELS_DIR / f"e2x__{ANCHOR_TAG}__test_plates.pkl")
        joblib.dump(sorted(list(x2e_plates)), MODELS_DIR / f"x2e__{ANCHOR_TAG}__test_plates.pkl")

        # --- FE (train 기준 → test 매핑) ---
        e2x_tr, e2x_te = _base_fe_train_test(e2x_tr_raw, e2x_te_raw)
        x2e_tr, x2e_te = _base_fe_train_test(x2e_tr_raw, x2e_te_raw)

        # --- Train/test 이상치 가드: train 분위 경계 → train 필터, test는 clip ---
        def _train_bounds(train_df):
            X, _, _ = _ensure_numeric_X(train_df)
            bounds = {c: tuple(np.quantile(X[c].values, [0.01,0.99])) for c in X.columns}
            y = train_df[TARGET_COL].values
            ylo, yhi = np.quantile(y, [0.005, 0.995])
            return bounds, ylo, yhi

        def _apply_bounds(train_df, test_df, bounds, ylo, yhi):
            tr = train_df.copy(); te = test_df.copy()
            keep = (tr[TARGET_COL] >= ylo) & (tr[TARGET_COL] <= yhi)
            X,_,_ = _ensure_numeric_X(tr)
            for c,(lo,hi) in bounds.items():
                if c in X.columns:
                    keep &= (X[c].values >= lo) & (X[c].values <= hi)
            tr = tr.loc[keep].reset_index(drop=True)
            # test clip
            for c,(lo,hi) in bounds.items():
                if c in te.columns:
                    te[c] = te[c].clip(lower=lo, upper=hi)
            return tr, te

        b, ylo, yhi = _train_bounds(e2x_tr)
        e2x_tr, e2x_te = _apply_bounds(e2x_tr, e2x_te, b, ylo, yhi)
        b, ylo, yhi = _train_bounds(x2e_tr)
        x2e_tr, x2e_te = _apply_bounds(x2e_tr, x2e_te, b, ylo, yhi)

        # --- 학습 ---
        e2x_algo = os.environ.get("E2X_ALGO", E2X_ALGO)
        x2e_algo = os.environ.get("X2E_ALGO", X2E_ALGO)
        e2x_res = train_one(e2x_tr, e2x_te, "E2X", e2x_algo)
        x2e_res = train_one(x2e_tr, x2e_te, "X2E", x2e_algo)

        avg_r2 = float(np.nanmean([e2x_res["r2_te"], x2e_res["r2_te"]]))

        # --- 빠른 퍼뮤 감사 ---
        if LEAK_AUTOCHECK:
            e2x_perm = quick_perm_r2(e2x_tr, e2x_te, algo=e2x_algo)
            x2e_perm = quick_perm_r2(x2e_tr, x2e_te, algo=x2e_algo)
            print(f"[Leak Audit] E2X perm R²={e2x_perm:.4f} | X2E perm R²={x2e_perm:.4f} (≈0 기대)")

        # --- 요약 TXT ---
        lines = []
        lines.append(f"Pass t → Pass t+1 Cross-Transition Summary")
        lines.append("="*72)
        lines.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')}  | ID: {experiment_id}")
        lines.append(f"RUN_NAME={RUN_NAME} | ANCHOR_TAG={ANCHOR_TAG}")
        lines.append(f"Flags: DT={int(USE_DELTA_TARGET)} TE={int(USE_PAIR_TE)} W={int(USE_HETERO_W)} MONO={int(USE_MONO_CUR)} RM={int(USE_RM_TRAINONLY)}")
        lines.append("")
        def fmt(label, r):
            return (f"* {label} | Algo: {r['algorithm']} | "
                    f"Train R2: {r['r2_tr']:.4f} | Test R2: {r['r2_te']:.4f} "
                    f"(RMSE {r['rmse_tr']:.4f}/{r['rmse_te']:.4f})")
        lines.append("[R2]")
        lines.append(fmt("E2X", e2x_res))
        lines.append(fmt("X2E", x2e_res))
        lines.append(f"\n[Average Test R2] {avg_r2:.4f}")
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"✓ Summary: {summary_path}")

        # --- 메트릭 CSV (집계 스크립트에서 읽기 쉽게: summary_path가 끝-1, avg가 마지막) ---
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_csv = METRICS_DIR / "last_run_metrics.csv"
        header = ["experiment_id","run_name","timestamp","anchor_tag",
                  "e2x_algo","x2e_algo",
                  "dt","te","w","mono","rm",
                  "e2x_train_r2","e2x_test_r2","x2e_train_r2","x2e_test_r2",
                  "summary_path","average_test_r2"]
        row = [experiment_id, RUN_NAME, ts.strftime("%Y-%m-%d %H:%M:%S"), ANCHOR_TAG,
               e2x_res["algorithm"], x2e_res["algorithm"],
               int(USE_DELTA_TARGET), int(USE_PAIR_TE), int(USE_HETERO_W), int(USE_MONO_CUR), int(USE_RM_TRAINONLY),
               f"{e2x_res['r2_tr']:.6f}", f"{e2x_res['r2_te']:.6f}", f"{x2e_res['r2_tr']:.6f}", f"{x2e_res['r2_te']:.6f}",
               str(summary_path), f"{avg_r2:.6f}"]
        pd.DataFrame([row], columns=header).to_csv(metrics_csv, index=False)
        print(f"✓ Metrics: {metrics_csv}")

        # --- 코드 스냅샷 ---
        try: shutil.copy2(Path(__file__).resolve(), SCRIPTS_DIR / f"{experiment_id}_{Path(__file__).name}")
        except Exception: pass

        success = True

    except Exception as e:
        tb = traceback.format_exc()
        (REPORTS_DIR/"last_error_traceback.txt").write_text(tb, encoding="utf-8")
        print(f"[실패] {type(e).__name__}: {e}")
        print(tb)

    if not success:
        pass

if __name__ == "__main__":
    main()
