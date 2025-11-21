# SVR 고정 + 누수차단형 향상전략 (250 베이스 확장)
import os, re, shutil, warnings
from datetime import datetime
from pathlib import Path
from inspect import signature

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)

# ====== (고정) 허용 PASS 집합 ======
ALLOWED_E2X_PASSES = {5, 7, 9}   # Entry -> Exit (5->6, 7->8, 9->10)
ALLOWED_X2E_PASSES = {6, 8}      # Exit  -> Entry (6->7, 8->9)

# ====== 경로/상수 ======
DATA_DIR     = Path("data")
PROCESSED    = DATA_DIR / "processed"
REPORTS_DIR  = Path("reports")
SUM_DIR      = REPORTS_DIR / "summaries"
SCRIPTS_DIR  = REPORTS_DIR / "scripts"
MODELS_DIR   = Path("models")
METRICS_DIR  = REPORTS_DIR / "metrics"
for d in [PROCESSED, REPORTS_DIR, SUM_DIR, SCRIPTS_DIR, MODELS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 컬럼명
PLATE_COL = "FM_날판번호"
PASS_COL  = "FM_PASS NO N"
MONTH_COL = "FM_압연월"

CUR_WARP  = "warping_index_current_pass"  # t
TGT_WARP  = "warping_index_target"        # t+1
TARGET_COL = "_target"                    # 내부 학습 타깃명

# 명시 제외 + 이름 스멜
EXCLUDE_COLS_BASE = {TARGET_COL, TGT_WARP, PLATE_COL, PASS_COL, MONTH_COL}
EXCLUDE_NAME_SMELLS = ("target", "label", "oof", "pred", "_hat")

SEED = int(os.getenv("SEED", "42"))

# ====== ENV 플래그 / 하이퍼 ======
def _env_on(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() not in {"0", "false", "", "none"}

RUN_NAME       = os.getenv("RUN_NAME", "svr_pass_tplus1")
ANCHOR_TAG     = os.getenv("ANCHOR_TAG", "anchor_default")
PERMUTE_TARGET = _env_on("PERMUTE_TARGET", "0")
LEAK_AUTOCHECK = _env_on("LEAK_AUTOCHECK", "1")

# 선택/전처리 전략 (250 기본 유지)
SELECT_MODE   = os.getenv("SELECT_MODE", "union")   # ["kbest","mi","union","stability"]
SELECT_K      = int(os.getenv("SELECT_K", "30"))    # 250 베이스라인: 30
USE_CV_KSELECT = _env_on("USE_CV_KSELECT", "0")
K_GRID        = [int(x) for x in os.getenv("K_GRID", "15,30,45,60").split(",") if x.strip()]

USE_DROP_NEAR_CONST = _env_on("USE_DROP_NEAR_CONST", "1")
NEAR_CONST_THR      = float(os.getenv("NEAR_CONST_THR", "1e-12"))

USE_DROP_CORR  = _env_on("USE_DROP_CORR", "1")
DROP_CORR_THR  = float(os.getenv("DROP_CORR_THR", "0.995"))

# y 스케일링(250은 0이 더 좋았음)
USE_Y_SCALE    = _env_on("USE_Y_SCALE", "0")       # train 통계 기반
Y_SCALE_MODE   = os.getenv("Y_SCALE_MODE", "mad")  # ["std","mad"]

# 새 전략들
USE_RESIDUAL_PAIR_Y = _env_on("USE_RESIDUAL_PAIR_Y", "0")   # pair별 y 평균으로 잔차학습
USE_RIDGE_META      = _env_on("USE_RIDGE_META", "0")        # Ridge OOF 1D 메타특성
RIDGE_META_ALPHA    = float(os.getenv("RIDGE_META_ALPHA", "1.0"))

# RM 사용 방식: none(미사용)/plate(기존 plate-통집계)/causal(≤t 누적)
RM_MODE = os.getenv("RM_MODE", "none").strip().lower()

# SVR 하이퍼 (250 기준값)
SVR_C       = float(os.getenv("SVR_C", "10.0"))
SVR_EPS     = float(os.getenv("SVR_EPS", "0.1"))
SVR_KERNEL  = os.getenv("SVR_KERNEL", "rbf")
SVR_GAMMA   = os.getenv("SVR_GAMMA", "scale")

# 분위 클리핑
Q_LOW   = float(os.getenv("Q_LOW", "0.01"))
Q_HIGH  = float(os.getenv("Q_HIGH", "0.99"))

# ====== 유틸 ======
def _next_experiment_id() -> str:
    existing = []
    for f in SUM_DIR.glob("*.txt"):
        m = re.match(r"(\d{3})_", f.name)
        if m:
            try: existing.append(int(m.group(1)))
            except: pass
    return f"{(max(existing)+1) if existing else 1:03d}"

def _hard_leak_guard(cols):
    safe = []
    for c in cols:
        low = c.lower()
        if any(k in low for k in EXCLUDE_NAME_SMELLS):
            continue
        safe.append(c)
    return safe

def _rm_cols(df: pd.DataFrame):
    return [c for c in df.columns if str(c).startswith("RM_") or str(c).startswith("RMc_")]

def _save_metrics(experiment_id, avg_r2, e2x_r2, x2e_r2, summary_path: Path):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "last_run_metrics.csv"
    row = {
        "experiment_id": experiment_id,
        "avg_test_r2": float(avg_r2) if pd.notna(avg_r2) else np.nan,
        "e2x_test_r2": float(e2x_r2) if pd.notna(e2x_r2) else np.nan,
        "x2e_test_r2": float(x2e_r2) if pd.notna(x2e_r2) else np.nan,
        "summary_path": str(summary_path),
    }
    pd.DataFrame([row]).to_csv(out, index=False)

def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ====== RM 특성 병합 유틸 ======
def _merge_rm_plate_stats(base_df: pd.DataFrame, rm: pd.DataFrame) -> pd.DataFrame:
    rm = rm.copy()
    num_cols = rm.select_dtypes(include=[np.number]).columns.tolist()
    stat_cols = [c for c in num_cols if c not in ["RM_날판번호","RM_압연Pass번호"]]
    agg = rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(4)
    agg.columns = [f"RM_{c[0]}_{c[1]}" for c in agg.columns]
    agg = agg.reset_index()
    merged = pd.merge(
        base_df, agg, left_on=PLATE_COL, right_on="RM_날판번호", how="left"
    ).drop(columns=["RM_날판번호"], errors="ignore")
    return merged

def _merge_rm_causal_stats(base_df: pd.DataFrame, rm: pd.DataFrame) -> pd.DataFrame:
    # plate 내 pass 오름차순 누적(≤t)
    rm_s = rm.sort_values(["RM_날판번호","RM_압연Pass번호"]).copy()
    num_cols = rm_s.select_dtypes(include=[np.number]).columns.tolist()
    stat_cols = [c for c in num_cols if c not in ["RM_날판번호","RM_압연Pass번호"]]
    def _expanding(g):
        exp = g[stat_cols].expanding()
        df = pd.concat([
            exp.mean().add_prefix("RMc_").add_suffix("_mean"),
            exp.std().add_prefix("RMc_").add_suffix("_std"),
            exp.min().add_prefix("RMc_").add_suffix("_min"),
            exp.max().add_prefix("RMc_").add_suffix("_max"),
            exp.median().add_prefix("RMc_").add_suffix("_median"),
        ], axis=1)
        df["RM_날판번호"] = g["RM_날판번호"].values
        df["RM_압연Pass번호"] = g["RM_압연Pass번호"].values
        return df
    stats = rm_s.groupby("RM_날판번호", group_keys=False).apply(_expanding).reset_index(drop=True)
    merged = pd.merge(
        base_df, stats,
        left_on=[PLATE_COL, PASS_COL],
        right_on=["RM_날판번호","RM_압연Pass번호"], how="left"
    ).drop(columns=["RM_날판번호","RM_압연Pass번호"], errors="ignore")
    return merged

# ====== 1) 병합/타깃 ======
def load_and_merge_tplus1():
    gt_entry = pd.read_csv(DATA_DIR / "entry_direction_results.csv")
    gt_exit  = pd.read_csv(DATA_DIR / "exit_direction_results.csv")
    gt = pd.concat([gt_entry, gt_exit], ignore_index=True)

    fm = pd.read_csv(DATA_DIR / "posco2_105190.csv")
    rm = pd.read_csv(DATA_DIR / "posco1_105190.csv")

    gt["extracted_plate"] = gt["filename"].str.extract(r"(PB\d+)")
    gt["extracted_pass"]  = gt["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    # FM 병합
    df = pd.merge(
        gt, fm,
        left_on=["extracted_plate","extracted_pass"],
        right_on=[PLATE_COL, PASS_COL],
        how="inner"
    ).sort_values([PLATE_COL, PASS_COL])

    # 타깃: t→t+1
    df[TGT_WARP] = df.groupby(PLATE_COL)["warping_index"].shift(-1)
    df = df.rename(columns={"warping_index": CUR_WARP})
    before = len(df)
    df = df.dropna(subset=[TGT_WARP]).reset_index(drop=True)

    # 내부 표준 타깃명
    df[TARGET_COL] = df[TGT_WARP].astype(float)

    # RM 결합 방식 선택
    rm = rm.rename(columns={"RM_압연Pass번호":"RM_압연Pass번호"})  # 명시적
    if RM_MODE == "plate":
        df = _merge_rm_plate_stats(df, rm)
    elif RM_MODE == "causal":
        df = _merge_rm_causal_stats(df, rm)
    else:
        # none: RM 미사용
        pass

    # 불필요 제거
    df = df.drop(columns=[
        "extracted_plate","extracted_pass","filename",
        "quality_grade","quality_grade_current_pass","direction"
    ], errors="ignore")

    if PERMUTE_TARGET:
        df[TARGET_COL] = np.random.permutation(df[TARGET_COL].values)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "final_merged_data_regression_tplus1.csv", index=False)
    dropped_shift = before - len(df)
    return df, dropped_shift

# ====== 2) 분리(허용 패스만) ======
def split_e2x_x2e(final_df: pd.DataFrame):
    e2x_df = final_df[(final_df[PASS_COL] % 2 == 1) & (final_df[PASS_COL].isin(ALLOWED_E2X_PASSES))].copy()
    x2e_df = final_df[(final_df[PASS_COL] % 2 == 0) & (final_df[PASS_COL].isin(ALLOWED_X2E_PASSES))].copy()
    e2x_df.to_csv(PROCESSED / "e2x_raw_tplus1.csv", index=False)
    x2e_df.to_csv(PROCESSED / "x2e_raw_tplus1.csv", index=False)
    return e2x_df, x2e_df

# ====== 3) 간단 FE ======
def basic_fe(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    fm_cols = [c for c in d.columns if str(c).startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if fm_cols:
        blk = d[fm_cols]
        d["FM_mean"]  = blk.mean(axis=1)
        d["FM_std"]   = blk.std(axis=1)
        d["FM_max"]   = blk.max(axis=1)
        d["FM_min"]   = blk.min(axis=1)
        d["FM_range"] = d["FM_max"] - d["FM_min"]
        d["FM_cv"]    = d["FM_std"] / (d["FM_mean"].abs() + 1e-8)
    d["FM_PASS_squared"] = d[PASS_COL] ** 2
    d.to_csv(PROCESSED / "all_featured_tplus1.csv", index=False)
    return d

# ====== 전처리 보조 ======
def _drop_near_constant(X: pd.DataFrame, thr: float = 1e-12) -> pd.DataFrame:
    std = X.std(numeric_only=True).replace(0, 0.0)
    keep = std[std > thr].index.tolist()
    return X[keep]

def _drop_high_corr(X: pd.DataFrame, thr: float = 0.995) -> pd.DataFrame:
    if X.shape[1] <= 1:
        return X
    corr = X.corr().abs()
    cols = X.columns.tolist()
    to_drop = set()
    for i in range(len(cols)):
        if cols[i] in to_drop: 
            continue
        for j in range(i+1, len(cols)):
            if corr.iat[i, j] >= thr:
                to_drop.add(cols[j])
    return X.drop(columns=list(to_drop), errors="ignore")

def _pair_key(df):
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)

# ====== 설계행렬 + 누수 스크리너 ======
def _prepare_design(tr_df: pd.DataFrame, te_df: pd.DataFrame, direction: str):
    tr = tr_df.copy(); te = te_df.copy()

    # PASS 진행도: train 의 max 로 고정
    pmax = float(tr[PASS_COL].max()) if len(tr) else 1.0
    scale = pmax if pmax > 0 else 1.0
    tr["FM_PASS_progress"] = tr[PASS_COL] / scale
    te["FM_PASS_progress"] = te[PASS_COL] / scale

    # 제외/스멜 제거
    Xtr_all = tr.drop(columns=[c for c in tr.columns if c in EXCLUDE_COLS_BASE], errors="ignore")
    Xte_all = te.drop(columns=[c for c in te.columns if c in EXCLUDE_COLS_BASE], errors="ignore")
    Xtr_all = Xtr_all[_hard_leak_guard(Xtr_all.columns)]
    Xte_all = Xte_all[[c for c in Xte_all.columns if c in Xtr_all.columns]]

    # 수치형만
    Xtr = Xtr_all.select_dtypes(include=[np.number]).copy()
    Xte = Xte_all.select_dtypes(include=[np.number]).copy()

    # 중앙값(train) 대치
    med = Xtr.median(numeric_only=True).to_dict()
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)

    # 값기반 누수 스크리너: |r|>=0.999 제거(학습에서만 결정)
    ytr = tr[TARGET_COL].astype(float).values
    if len(Xtr) and len(ytr):
        Xc = (Xtr - Xtr.mean()) / (Xtr.std().replace(0, 1))
        yc = ytr - ytr.mean()
        ys = np.std(yc) if np.std(yc) > 1e-12 else 1e-12
        r = (Xc.values.T @ (yc / ys)) / max(1, (len(ytr) - 1))
        bad_idx = np.where(np.abs(r) >= 0.999)[0].tolist()
        if bad_idx:
            bad_cols = [Xtr.columns[i] for i in bad_idx]
            Xtr = Xtr.drop(columns=bad_cols, errors="ignore")
            Xte = Xte.drop(columns=[c for c in bad_cols if c in Xte.columns], errors="ignore")

    # 상수/상관 정리 (train에서만 결정)
    if USE_DROP_NEAR_CONST:
        Xtr = _drop_near_constant(Xtr, NEAR_CONST_THR)
        Xte = Xte[[c for c in Xte.columns if c in Xtr.columns]]
    if USE_DROP_CORR:
        Xtr = _drop_high_corr(Xtr, DROP_CORR_THR)
        Xte = Xte[[c for c in Xte.columns if c in Xtr.columns]]

    # 이상치 가드: train 분위 경계 → train filter, test clip
    bounds = {}
    for c in Xtr.columns:
        lo, hi = np.quantile(Xtr[c].values, [Q_LOW, Q_HIGH])
        bounds[c] = (float(lo), float(hi))
    keep = np.ones(len(Xtr), dtype=bool)
    for c,(lo,hi) in bounds.items():
        v = Xtr[c].values
        keep &= (v >= lo) & (v <= hi)
    Xtr2 = Xtr.loc[Xtr.index[keep]].copy()
    tr2  = tr.loc[Xtr2.index].copy()
    for c,(lo,hi) in bounds.items():
        if c in Xte.columns:
            Xte[c] = Xte[c].clip(lower=lo, upper=hi)

    return tr2, Xtr2, Xte, med, list(Xtr2.columns)

# ====== 피처 선택 ======
class IndexSelector:
    def __init__(self, idx, all_cols):
        self.idx = np.array(sorted(set(idx)), dtype=int)
        self.all_cols = list(all_cols)
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.idx] if X.ndim==2 else X
    def get_support(self, indices=False):
        return self.idx if indices else np.isin(np.arange(len(self.all_cols)), self.idx)

def _select_kbest_indices(Xtr: pd.DataFrame, ytr: np.ndarray, k: int):
    k = max(1, min(k, Xtr.shape[1]))
    sel = SelectKBest(score_func=f_regression, k=k).fit(Xtr.values, ytr)
    return sel.get_support(indices=True)

def _select_mi_indices(Xtr: pd.DataFrame, ytr: np.ndarray, k: int, seed: int):
    k = max(1, min(k, Xtr.shape[1]))
    mi = mutual_info_regression(Xtr.values, ytr, random_state=seed)
    order = np.argsort(mi)[::-1]
    return order[:k]

def _select_union_indices(Xtr: pd.DataFrame, ytr: np.ndarray, k: int, seed: int):
    k = max(1, min(k, Xtr.shape[1]))
    k1 = max(1, k//2); k2 = k - k1
    idx_a = set(_select_kbest_indices(Xtr, ytr, k1))
    idx_b = set(_select_mi_indices(Xtr, ytr, k2, seed))
    idx = list(idx_a.union(idx_b))
    if len(idx) > k:
        f = f_regression(Xtr.values, ytr)[0]
        order = np.argsort(f)[::-1]
        picked = []
        for j in order:
            if j in idx:
                picked.append(j)
            if len(picked) >= k: break
        idx = picked
    return np.array(idx, dtype=int)

def _choose_k_by_cv(Xtr: pd.DataFrame, ytr: np.ndarray, groups: np.ndarray, mode: str, k_grid, seed: int):
    best_k, best_r2 = None, -1e9
    gkf = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    for k in k_grid:
        if mode == "kbest":
            idx = _select_kbest_indices(Xtr, ytr, k)
        elif mode == "mi":
            idx = _select_mi_indices(Xtr, ytr, k, seed)
        else:
            idx = _select_union_indices(Xtr, ytr, k, seed)
        Xs = Xtr.values[:, idx]
        mdl = Pipeline([("scale", StandardScaler()), ("reg", SVR(C=SVR_C, epsilon=SVR_EPS, kernel=SVR_KERNEL, gamma=SVR_GAMMA))])
        cv_r2 = []
        for tr_i, va_i in gkf.split(Xs, groups=groups):
            mdl.fit(Xs[tr_i], ytr[tr_i])
            yhat = mdl.predict(Xs[va_i])
            cv_r2.append(r2_score(ytr[va_i], yhat))
        m = float(np.mean(cv_r2)) if cv_r2 else -1e9
        if m > best_r2:
            best_r2, best_k = m, k
    return int(best_k or SELECT_K)

# ====== SVR 파이프라인 / 학습 ======
def _build_svr_pipeline():
    steps = [("scale", StandardScaler(with_mean=True, with_std=True))]
    svr = SVR(C=SVR_C, epsilon=SVR_EPS, kernel=SVR_KERNEL, gamma=SVR_GAMMA)
    steps.append(("reg", svr))
    return Pipeline(steps)

def _fit_model(model, X, y, sample_weight=None):
    if sample_weight is None:
        return model.fit(X, y)
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        pass
    if isinstance(model, Pipeline):
        last_name, last_est = model.steps[-1]
        if "sample_weight" in signature(last_est.fit).parameters:
            return model.fit(X, y, **{f"{last_name}__sample_weight": sample_weight})
    print("[WARN] sample_weight 미지원 추정기 → 가중치 미적용 학습")
    return model.fit(X, y)

def _y_scale_fit(y: np.ndarray, mode: str="mad"):
    y = y.astype(float)
    mu = float(np.median(y)) if mode=="mad" else float(np.mean(y))
    if mode == "mad":
        s = 1.4826 * float(np.median(np.abs(y - mu))) + 1e-9
    else:
        s = float(np.std(y)) + 1e-9
    return mu, s

def _ridge_oof_feature(Xtr_sel: np.ndarray, Xte_sel: np.ndarray, ytr_model: np.ndarray, groups: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof = np.full(len(Xtr_sel), np.nan, dtype=float)
    for tr_i, va_i in gkf.split(Xtr_sel, groups=groups):
        mdl = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=SEED))])
        mdl.fit(Xtr_sel[tr_i], ytr_model[tr_i])
        oof[va_i] = mdl.predict(Xtr_sel[va_i])
    oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
    mdl_full = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=SEED))])
    mdl_full.fit(Xtr_sel, ytr_model)
    te_pred = mdl_full.predict(Xte_sel)
    # 1열 추가
    Xtr_aug = np.concatenate([Xtr_sel, oof.reshape(-1,1)], axis=1)
    Xte_aug = np.concatenate([Xte_sel, te_pred.reshape(-1,1)], axis=1)
    return Xtr_aug, Xte_aug

# ====== 단방향 학습 ======
def _fit_one_direction(df: pd.DataFrame, direction: str):
    d0 = df.copy()
    if direction.upper() == "E2X":
        d0 = d0[(d0[PASS_COL] % 2 == 1) & (d0[PASS_COL].isin(ALLOWED_E2X_PASSES))].copy()
        prefix = f"e2x_{ANCHOR_TAG}"
    else:
        d0 = d0[(d0[PASS_COL] % 2 == 0) & (d0[PASS_COL].isin(ALLOWED_X2E_PASSES))].copy()
        prefix = f"x2e_{ANCHOR_TAG}"

    if d0.empty:
        raise RuntimeError(f"{direction}: 허용 전이에 해당하는 표본이 없습니다.")

    # plate‑group 앵커 분할
    anchor_path = MODELS_DIR / f"{prefix}_test_plates.pkl"
    groups_all = d0[PLATE_COL].astype(str).values
    n_plates = int(pd.unique(groups_all).size)
    test_size = 0.3 if n_plates < 10 else 0.2
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)

    if anchor_path.exists():
        test_plates = set(pd.read_pickle(anchor_path))
        te_mask = d0[PLATE_COL].astype(str).isin(test_plates).values
        tr_mask = ~te_mask
        tr_df_raw = d0.loc[tr_mask].copy()
        te_df_raw = d0.loc[te_mask].copy()
    else:
        tr_i, te_i = next(gss.split(d0, groups=groups_all))
        tr_df_raw = d0.iloc[tr_i].copy()
        te_df_raw = d0.iloc[te_i].copy()
        pd.to_pickle(sorted(te_df_raw[PLATE_COL].astype(str).unique().tolist()), anchor_path)

    if tr_df_raw.empty or te_df_raw.empty:
        raise RuntimeError(f"{direction}: 앵커 분할 실패(train/test 중 하나가 비어 있음).")

    # 타깃 정의(그대로: t+1)
    tr_df_raw[TARGET_COL] = tr_df_raw[TGT_WARP].astype(float)
    te_df_raw[TARGET_COL] = te_df_raw[TGT_WARP].astype(float)

    # 설계행렬 + 누수스크리너
    tr_df2, Xtr, Xte, med, pre_cols = _prepare_design(tr_df_raw, te_df_raw, direction)
    ytr = tr_df2[TARGET_COL].to_numpy(dtype=float)
    groups = tr_df2[PLATE_COL].astype(str).to_numpy()

    # --- (새) PAIR-Y Residualization ---
    mu_tr_pair = mu_te_pair = None
    if USE_RESIDUAL_PAIR_Y:
        pair_tr = _pair_key(tr_df2)
        pair_te = _pair_key(te_df_raw)
        g_mean = pd.DataFrame({"pair": pair_tr, "y": ytr}).groupby("pair")["y"].mean()
        g_mu = g_mean.to_dict()
        global_mu = float(np.mean(ytr)) if len(ytr) else 0.0
        mu_tr_pair = pair_tr.map(g_mu).fillna(global_mu).to_numpy()
        mu_te_pair = pair_te.map(g_mu).fillna(global_mu).to_numpy()
        ytr_model = ytr - mu_tr_pair
    else:
        ytr_model = ytr

    # --- y 스케일링(모델용 표적) ---
    y_mu = y_sc = None
    if USE_Y_SCALE:
        y_mu, y_sc = _y_scale_fit(ytr_model, Y_SCALE_MODE)
        ytr_model = (ytr_model - y_mu) / y_sc

    # --- K 선택 ---
    if USE_CV_KSELECT:
        k_used = _choose_k_by_cv(Xtr, ytr_model, groups, SELECT_MODE, K_GRID, SEED)
    else:
        k_used = max(1, min(SELECT_K, Xtr.shape[1]))

    # --- 선택 인덱스 ---
    if SELECT_MODE == "kbest":
        idx = _select_kbest_indices(Xtr, ytr_model, k_used)
    elif SELECT_MODE == "mi":
        idx = _select_mi_indices(Xtr, ytr_model, k_used, SEED)
    elif SELECT_MODE == "union":
        idx = _select_union_indices(Xtr, ytr_model, k_used, SEED)
    else:
        # stability는 250/270에서 이득 적었음 → 필요 시 ENV로 켜서 사용
        idx = _select_union_indices(Xtr, ytr_model, k_used, SEED)

    selector = IndexSelector(idx, pre_cols)
    Xtr_sel = selector.transform(Xtr.values)
    Xte_sel = selector.transform(Xte.values)

    # --- (새) Ridge OOF 메타특성 ---
    if USE_RIDGE_META:
        Xtr_sel, Xte_sel = _ridge_oof_feature(Xtr_sel, Xte_sel, ytr_model, groups, alpha=RIDGE_META_ALPHA)

    # --- SVR 학습 ---
    model = _build_svr_pipeline()
    _fit_model(model, Xtr_sel, ytr_model)

    # --- 예측/복원 ---
    ytr_hat = model.predict(Xtr_sel)
    yte_hat = model.predict(Xte_sel)
    if USE_Y_SCALE:
        ytr_hat = ytr_hat * y_sc + y_mu
        yte_hat = yte_hat * y_sc + y_mu
    if USE_RESIDUAL_PAIR_Y:
        ytr_hat = ytr_hat + mu_tr_pair
        yte_hat = yte_hat + mu_te_pair

    ytr_true = tr_df2[TARGET_COL].values
    yte_true = te_df_raw[TARGET_COL].values

    train_r2  = float(r2_score(ytr_true, ytr_hat)) if len(ytr_true) else np.nan
    test_r2   = float(r2_score(yte_true, yte_hat)) if len(yte_true) else np.nan
    train_rmse = _rmse(ytr_true, ytr_hat)
    test_rmse  = _rmse(yte_true, yte_hat)

    # (참고 저장 — 재현용)
    pd.to_pickle(model,            MODELS_DIR / f"{prefix}_reg.pkl")
    pd.to_pickle(selector,         MODELS_DIR / f"{prefix}_selector.pkl")
    pd.to_pickle(pre_cols,         MODELS_DIR / f"{prefix}_pre_cols.pkl")
    pd.to_pickle([pre_cols[i] for i in idx], MODELS_DIR / f"{prefix}_sel_cols.pkl")
    pd.to_pickle(med,              MODELS_DIR / f"{prefix}_med.pkl")

    return {
        "direction": direction,
        "algorithm": "SVR",
        "train_r2": train_r2, "test_r2": test_r2,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "k_used": int(k_used),
        "select_mode": SELECT_MODE,
        "best_params": None
    }

# ====== Permutation 감사 (간이) ======
def permutation_r2_once(tr_df_raw: pd.DataFrame, te_df_raw: pd.DataFrame, direction: str):
    tr_df = tr_df_raw.copy(); te_df = te_df_raw.copy()
    tr_df[TARGET_COL] = tr_df[TGT_WARP].astype(float)
    te_df[TARGET_COL] = te_df[TGT_WARP].astype(float)

    tr_df2, Xtr, Xte, med, cols = _prepare_design(tr_df, te_df, direction)
    ytr = tr_df2[TARGET_COL].to_numpy(dtype=float)

    # residualization 동일 적용
    if USE_RESIDUAL_PAIR_Y:
        pair_tr = _pair_key(tr_df2)
        g_mean = pd.DataFrame({"pair": pair_tr, "y": ytr}).groupby("pair")["y"].mean()
        global_mu = float(np.mean(ytr)) if len(ytr) else 0.0
        mu_tr_pair = pair_tr.map(g_mean).fillna(global_mu).to_numpy()
        ytr = ytr - mu_tr_pair

    if USE_Y_SCALE:
        y_mu, y_sc = _y_scale_fit(ytr, Y_SCALE_MODE)
        ytr = (ytr - y_mu) / y_sc

    ytr_perm = np.random.permutation(ytr)
    k = max(1, min(min(30, SELECT_K), Xtr.shape[1]))
    idx = _select_union_indices(Xtr, ytr_perm, k, SEED)
    Xtr_sel = Xtr.values[:, idx]
    Xte_sel = Xte.values[:, idx]

    mdl = _build_svr_pipeline()
    mdl.fit(Xtr_sel, ytr_perm)
    yhat = mdl.predict(Xte_sel)
    # 역변환은 permute 목적상 불필요 (R²≈0 기대)
    y_true = te_df[TARGET_COL].values
    return float(r2_score(y_true, yhat)) if len(y_true) else np.nan

# ====== 요약 저장 ======
def write_summary(experiment_id: str, results: dict, dropped_shift: int, summary_path: Path):
    lines = []
    lines.append("Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
    lines.append("="*72)
    lines.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 작성")
    lines.append(f"실험 ID: {experiment_id}\n")

    lines.append("[모델 성능 - R2]")
    for tag in ("E2X","X2E"):
        r = results[tag]
        lines.append(f"* {tag} | Algo: {r['algorithm']} | "
                     f"Train R2: {r['train_r2']:.4f} | Test R2: {r['test_r2']:.4f} "
                     f"(RMSE {r['train_rmse']:.4f}/{r['test_rmse']:.4f})")
    avg = float(np.nanmean([results['E2X']['test_r2'], results['X2E']['test_r2']]))
    lines.append(f"\n[평균 Test R2] {avg:.4f}")

    if "perm_e2x" in results and "perm_x2e" in results:
        lines.append("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        lines.append(f"- E2X permuted R²: {results['perm_e2x']:.4f}")
        lines.append(f"- X2E permuted R²: {results['perm_x2e']:.4f}")

    lines.append("\n[DEBUG] 설정/분할/선택")
    lines.append(f"- PERMUTE_TARGET={1 if PERMUTE_TARGET else 0} | USE_Y_SCALE={1 if USE_Y_SCALE else 0} | "
                 f"USE_DROP_NEAR_CONST={1 if USE_DROP_NEAR_CONST else 0} | USE_DROP_CORR={1 if USE_DROP_CORR else 0} (thr={DROP_CORR_THR}) | "
                 f"SELECT_MODE={SELECT_MODE} | USE_CV_KSELECT={1 if USE_CV_KSELECT else 0} | SELECT_K={SELECT_K} | K_GRID={','.join(map(str,K_GRID))}")
    lines.append(f"- 새 전략 | USE_RESIDUAL_PAIR_Y={1 if USE_RESIDUAL_PAIR_Y else 0} | USE_RIDGE_META={1 if USE_RIDGE_META else 0} | RM_MODE={RM_MODE}")
    lines.append(f"- 허용 E2X 패스: {sorted(ALLOWED_E2X_PASSES)} | 허용 X2E 패스: {sorted(ALLOWED_X2E_PASSES)}")
    lines.append(f"- t+1 생성 시 삭제 수(마지막 PASS 등): {dropped_shift}")
    for tag in ("E2X","X2E"):
        r = results[tag]
        lines.append(f"- [{tag}] k_used={r.get('k_used','?')} | select_mode={r.get('select_mode','?')}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

# ====== main ======
def main():
    experiment_id = _next_experiment_id()
    summary_path = SUM_DIR / f"{experiment_id}_test_summary.txt"

    # 코드 스냅샷
    try:
        shutil.copy2(Path(__file__).resolve(), SCRIPTS_DIR / f"{experiment_id}_test.py")
    except Exception:
        pass

    final_df, dropped_shift = load_and_merge_tplus1()
    all_fe = basic_fe(final_df)
    e2x_raw, x2e_raw = split_e2x_x2e(all_fe)

    # 학습 (E2X, X2E)
    e2x_res = _fit_one_direction(all_fe, "E2X")
    x2e_res = _fit_one_direction(all_fe, "X2E")

    # 간이 permutation 감사(≈0 기대)
    perm_e2x = permutation_r2_once(e2x_raw, x2e_raw, "E2X") if LEAK_AUTOCHECK else np.nan
    perm_x2e = permutation_r2_once(x2e_raw, e2x_raw, "X2E") if LEAK_AUTOCHECK else np.nan

    results = {"E2X": e2x_res, "X2E": x2e_res, "perm_e2x": perm_e2x, "perm_x2e": perm_x2e}
    write_summary(experiment_id, results, dropped_shift, summary_path)

    avg_r2 = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))
    _save_metrics(experiment_id, avg_r2, e2x_res["test_r2"], x2e_res["test_r2"], summary_path)

    # 콘솔 요약
    print("\n" + "Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
    print("="*72)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 작성")
    print(f"실험 ID: {experiment_id}\n")
    for tag in ("E2X","X2E"):
        r = results[tag]
        print(f"* {tag} | Algo: {r['algorithm']} | Train R2: {r['train_r2']:.4f} | "
              f"Test R2: {r['test_r2']:.4f} (RMSE {r['train_rmse']:.4f}/{r['test_rmse']:.4f})")
    print(f"\n[평균 Test R2] {avg_r2:.4f}")
    if LEAK_AUTOCHECK:
        print("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        print(f"- E2X permuted R²: {perm_e2x:.4f}")
        print(f"- X2E permuted R²: {perm_x2e:.4f}")

if __name__ == "__main__":
    main()
