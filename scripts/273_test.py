# SVR (고정) + 누수차단형 R² 향상 전략 (실험 250 베이스라인 확장)
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

# 선택/전처리 전략
SELECT_MODE   = os.getenv("SELECT_MODE", "union")   # ["kbest","mi","union","stability"]
SELECT_K      = int(os.getenv("SELECT_K", "30"))    # 250 베이스라인: 30
USE_CV_KSELECT = _env_on("USE_CV_KSELECT", "0")
K_GRID        = [int(x) for x in os.getenv("K_GRID", "15,30,45,60").split(",") if x.strip()]

USE_DROP_NEAR_CONST = _env_on("USE_DROP_NEAR_CONST", "1")
NEAR_CONST_THR      = float(os.getenv("NEAR_CONST_THR", "1e-12"))

USE_DROP_CORR  = _env_on("USE_DROP_CORR", "1")
DROP_CORR_THR  = float(os.getenv("DROP_CORR_THR", "0.995"))

USE_Y_SCALE    = _env_on("USE_Y_SCALE", "1")       # y 스케일링(훈련 통계 기반)
Y_SCALE_MODE   = os.getenv("Y_SCALE_MODE", "mad")  # ["std","mad"]

USE_PAIR_OHE   = _env_on("USE_PAIR_OHE", "0")      # 0/1 — pass쌍 OHE

# STABILITY 선택
USE_STABILITY_KSELECT = _env_on("USE_STABILITY_KSELECT", "1")
STAB_N       = int(os.getenv("STAB_N", "12"))      # 반복 횟수
STAB_SUBRATE = float(os.getenv("STAB_SUBRATE", "0.85"))  # 각 반복의 샘플 비율(그룹 보존)

# SVR 하이퍼 (베이스라인 값)
SVR_C       = float(os.getenv("SVR_C", "10.0"))
SVR_EPS     = float(os.getenv("SVR_EPS", "0.1"))
SVR_KERNEL  = os.getenv("SVR_KERNEL", "rbf")
SVR_GAMMA   = os.getenv("SVR_GAMMA", "scale")

# 소형 HPO
USE_HPO     = _env_on("USE_HPO", "0")
HPO_C_GRID      = [float(x) for x in os.getenv("HPO_C_GRID", "5.0,10.0,20.0").split(",")]
HPO_EPS_GRID    = [float(x) for x in os.getenv("HPO_EPS_GRID", "0.05,0.1,0.2").split(",")]
HPO_GAMMA_GRID  = [x for x in os.getenv("HPO_GAMMA_GRID", "scale,auto").split(",")]

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
    return [c for c in df.columns if str(c).startswith("RM_")]

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

# ====== 데이터 병합/타깃 ======
def load_and_merge_tplus1():
    gt_entry = pd.read_csv(DATA_DIR / "entry_direction_results.csv")
    gt_exit  = pd.read_csv(DATA_DIR / "exit_direction_results.csv")
    gt = pd.concat([gt_entry, gt_exit], ignore_index=True)

    fm = pd.read_csv(DATA_DIR / "posco2_105190.csv")
    rm = pd.read_csv(DATA_DIR / "posco1_105190.csv")

    gt["extracted_plate"] = gt["filename"].str.extract(r"(PB\d+)")
    gt["extracted_pass"]  = gt["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    # RM 집계(판 단위 5통계)
    merged_rm = pd.merge(
        gt, rm,
        left_on=["extracted_plate","extracted_pass"],
        right_on=["RM_날판번호","RM_압연Pass번호"],
        how="inner"
    )
    num_cols = merged_rm.select_dtypes(include=[np.number]).columns.tolist()
    stat_cols = [c for c in num_cols if c not in ["RM_날판번호","RM_압연Pass번호","warping_index","extracted_pass"]]
    rm_stats = merged_rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(4)
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    rm_stats = rm_stats.reset_index()

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

    # RM 집계 결합
    final_df = pd.merge(df, rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left") \
                  .drop(columns=["RM_날판번호"], errors="ignore")

    # 불필요 제거
    final_df = final_df.drop(columns=[
        "extracted_plate","extracted_pass","filename",
        "quality_grade","quality_grade_current_pass","direction"
    ], errors="ignore")

    if PERMUTE_TARGET:
        final_df[TARGET_COL] = np.random.permutation(final_df[TARGET_COL].values)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(PROCESSED / "final_merged_data_regression_tplus1.csv", index=False)
    dropped_shift = before - len(final_df)
    return final_df, dropped_shift

# ====== 분리(허용 패스만) ======
def split_e2x_x2e(final_df: pd.DataFrame):
    e2x_df = final_df[(final_df[PASS_COL] % 2 == 1) & (final_df[PASS_COL].isin(ALLOWED_E2X_PASSES))].copy()
    x2e_df = final_df[(final_df[PASS_COL] % 2 == 0) & (final_df[PASS_COL].isin(ALLOWED_X2E_PASSES))].copy()
    e2x_df.to_csv(PROCESSED / "e2x_raw_tplus1.csv", index=False)
    x2e_df.to_csv(PROCESSED / "x2e_raw_tplus1.csv", index=False)
    return e2x_df, x2e_df

# ====== 간단 FE ======
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
def _append_pair_ohe(df_base: pd.DataFrame, Xdf: pd.DataFrame, direction: str) -> pd.DataFrame:
    """허용된 pass쌍(one-hot), 카테고리 집합은 코드에 고정 → 누수 없음."""
    pair = df_base[PASS_COL].astype(int).astype(str) + "→" + (df_base[PASS_COL] + 1).astype(int).astype(str)
    X = Xdf.copy()
    allowed = ["5→6","7→8","9→10"] if direction.upper()=="E2X" else ["6→7","8→9"]
    for cat in allowed:
        X[f"PAIR_{cat}"] = (pair == cat).astype(int)
    return X

def _drop_near_constant(X: pd.DataFrame, thr: float = 1e-12) -> pd.DataFrame:
    std = X.std(numeric_only=True).replace(0, 0.0)
    keep = std[std > thr].index.tolist()
    return X[keep]

def _drop_high_corr(X: pd.DataFrame, thr: float = 0.995) -> pd.DataFrame:
    if X.shape[1] <= 1:
        return X
    corr = X.corr().abs()
    upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    to_drop = set()
    cols = X.columns.tolist()
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i+1, len(cols)):
            if corr.iat[i, j] >= thr:
                to_drop.add(cols[j])
    return X.drop(columns=list(to_drop), errors="ignore")

# ====== 설계행렬 + 누수 스크리너 ======
def _prepare_design(tr_df: pd.DataFrame, te_df: pd.DataFrame, direction: str):
    tr = tr_df.copy()
    te = te_df.copy()

    # PASS 진행도: train의 max로 고정
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

    # pair OHE (고정 카테고리)
    if USE_PAIR_OHE:
        Xtr = _append_pair_ohe(tr, Xtr, direction)
        Xte = _append_pair_ohe(te, Xte, direction)

    # 중앙값(train) 대치
    med = Xtr.median(numeric_only=True).to_dict()
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)

    # 값기반 누수 스크리너: |r|>=0.999 → 제거(학습에서만 결정)
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

    # 상수/상관 정리 (train에서만 의사결정)
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
        # 초과 시 f_regression 점수 순으로 상위 k
        f = f_regression(Xtr.values, ytr)[0]
        order = np.argsort(f)[::-1]
        picked = []
        for j in order:
            if j in idx:
                picked.append(j)
            if len(picked) >= k: break
        idx = picked
    return np.array(idx, dtype=int)

def _select_stability_indices(Xtr: pd.DataFrame, ytr: np.ndarray, k: int, groups: np.ndarray, seed: int):
    rng = np.random.RandomState(seed)
    counts = np.zeros(Xtr.shape[1], dtype=int)
    g_unique = np.unique(groups)
    for t in range(STAB_N):
        # 그룹 단위 서브샘플링 (비율 유지)
        keep_groups = rng.choice(g_unique, size=max(1, int(len(g_unique)*STAB_SUBRATE)), replace=False)
        mask = np.isin(groups, keep_groups)
        Xs = Xtr.loc[Xtr.index[mask]]
        ys = ytr[mask]
        # f_reg 기반 k/2, mi 기반 k/2의 union
        idx = _select_union_indices(Xs, ys, k, seed + 31*t)
        counts[idx] += 1
    # 빈도 높은 순으로 상위 k
    order = np.argsort(counts)[::-1]
    order = order[counts[order] > 0]
    if len(order) < k:
        # 부족하면 f_reg 점수로 보충
        f = f_regression(Xtr.values, ytr)[0]
        tail = [j for j in np.argsort(f)[::-1] if j not in order]
        order = list(order) + tail[:(k-len(order))]
    return np.array(order[:k], dtype=int)

def _choose_k_by_cv(Xtr: pd.DataFrame, ytr: np.ndarray, groups: np.ndarray, mode: str, k_grid, seed: int):
    best_k, best_r2 = None, -1e9
    gkf = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    for k in k_grid:
        # 선택 인덱스
        if mode == "kbest":
            idx = _select_kbest_indices(Xtr, ytr, k)
        elif mode == "mi":
            idx = _select_mi_indices(Xtr, ytr, k, seed)
        elif mode == "union":
            idx = _select_union_indices(Xtr, ytr, k, seed)
        elif mode == "stability":
            idx = _select_stability_indices(Xtr, ytr, k, groups, seed)
        else:
            idx = _select_kbest_indices(Xtr, ytr, k)

        Xs = Xtr.values[:, idx]
        # 간단 파이프라인으로 CV R²
        pipe = _build_svr_pipeline()
        cv_r2 = []
        for tr_i, va_i in gkf.split(Xs, groups=groups):
            Xtr_cv, Xva_cv = Xs[tr_i], Xs[va_i]
            ytr_cv, yva_cv = ytr[tr_i], ytr[va_i]
            _fit_model(pipe, Xtr_cv, ytr_cv)
            yhat = pipe.predict(Xva_cv)
            cv_r2.append(r2_score(yva_cv, yhat))
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
    # 직접 전달 시도
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        pass
    # Pipeline -> 마지막 스텝으로 위임
    if isinstance(model, Pipeline):
        last_name, last_est = model.steps[-1]
        if "sample_weight" in signature(last_est.fit).parameters:
            return model.fit(X, y, **{f"{last_name}__sample_weight": sample_weight})
    # 미지원: 경고 후 미적용
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

def _svr_cv_hpo(X: np.ndarray, y: np.ndarray, groups: np.ndarray, sample_weight=None):
    params = []
    for c in HPO_C_GRID:
        for e in HPO_EPS_GRID:
            for g in HPO_GAMMA_GRID:
                params.append((c, e, g))
    gkf = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    best, best_r2 = (SVR_C, SVR_EPS, SVR_GAMMA), -1e9
    for (c, e, g) in params:
        cv = []
        for tr_i, va_i in gkf.split(X, groups=groups):
            mdl = Pipeline([("scale", StandardScaler()), ("reg", SVR(C=c, epsilon=e, gamma=g, kernel=SVR_KERNEL))])
            _fit_model(mdl, X[tr_i], y[tr_i], sample_weight=None if sample_weight is None else sample_weight[tr_i])
            pred = mdl.predict(X[va_i])
            cv.append(r2_score(y[va_i], pred))
        m = float(np.mean(cv)) if cv else -1e9
        if m > best_r2:
            best_r2 = m
            best = (c, e, g)
    return best  # (C, eps, gamma)

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

    # y 스케일링(훈련 통계 기반)
    y_mu = y_sc = None
    if USE_Y_SCALE:
        y_mu, y_sc = _y_scale_fit(ytr, Y_SCALE_MODE)
        ytr_s = (ytr - y_mu) / y_sc
    else:
        ytr_s = ytr

    # K 선택
    if USE_CV_KSELECT:
        k_used = _choose_k_by_cv(Xtr, ytr_s, groups, SELECT_MODE, K_GRID, SEED)
    else:
        k_used = max(1, min(SELECT_K, Xtr.shape[1]))

    # 선택 인덱스
    if SELECT_MODE == "kbest":
        idx = _select_kbest_indices(Xtr, ytr_s, k_used)
    elif SELECT_MODE == "mi":
        idx = _select_mi_indices(Xtr, ytr_s, k_used, SEED)
    elif SELECT_MODE == "stability" and USE_STABILITY_KSELECT:
        idx = _select_stability_indices(Xtr, ytr_s, k_used, groups, SEED)
    else:
        idx = _select_union_indices(Xtr, ytr_s, k_used, SEED)

    selector = IndexSelector(idx, pre_cols)
    Xtr_sel = selector.transform(Xtr.values)
    Xte_sel = selector.transform(Xte.values)

    # HPO (선택)
    best_params = None
    if USE_HPO:
        best_params = _svr_cv_hpo(Xtr_sel, ytr_s, groups, sample_weight=None)
        c, e, g = best_params
        mdl = Pipeline([("scale", StandardScaler()), ("reg", SVR(C=c, epsilon=e, gamma=g, kernel=SVR_KERNEL))])
    else:
        mdl = _build_svr_pipeline()

    _fit_model(mdl, Xtr_sel, ytr_s)

    # 예측/복원/스코어
    ytr_hat = mdl.predict(Xtr_sel)
    yte_hat = mdl.predict(Xte_sel)
    if USE_Y_SCALE:
        ytr_hat = ytr_hat * y_sc + y_mu
        yte_hat = yte_hat * y_sc + y_mu

    ytr_true = tr_df2[TARGET_COL].values
    yte_true = te_df_raw[TARGET_COL].values

    train_r2  = float(r2_score(ytr_true, ytr_hat)) if len(ytr_true) else np.nan
    test_r2   = float(r2_score(yte_true, yte_hat)) if len(yte_true) else np.nan
    train_rmse = _rmse(ytr_true, ytr_hat)
    test_rmse  = _rmse(yte_true, yte_hat)

    # (참고 저장 — 재현용)
    pd.to_pickle(mdl,            MODELS_DIR / f"{prefix}_reg.pkl")
    pd.to_pickle(selector,       MODELS_DIR / f"{prefix}_selector.pkl")
    pd.to_pickle(pre_cols,       MODELS_DIR / f"{prefix}_pre_cols.pkl")
    pd.to_pickle([pre_cols[i] for i in idx], MODELS_DIR / f"{prefix}_sel_cols.pkl")
    pd.to_pickle(med,            MODELS_DIR / f"{prefix}_med.pkl")

    return {
        "direction": direction,
        "algorithm": "SVR",
        "train_r2": train_r2, "test_r2": test_r2,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "k_used": int(k_used),
        "select_mode": SELECT_MODE,
        "best_params": None if best_params is None else {"C":best_params[0], "eps":best_params[1], "gamma":best_params[2]},
    }

# ====== Permutation 감사 (간이) ======
def permutation_r2_once(tr_df_raw: pd.DataFrame, te_df_raw: pd.DataFrame, direction: str):
    tr_df = tr_df_raw.copy(); te_df = te_df_raw.copy()
    tr_df[TARGET_COL] = tr_df[TGT_WARP].astype(float)
    te_df[TARGET_COL] = te_df[TGT_WARP].astype(float)

    tr_df2, Xtr, Xte, med, cols = _prepare_design(tr_df, te_df, direction)
    ytr = tr_df2[TARGET_COL].to_numpy(dtype=float)
    if USE_Y_SCALE:
        y_mu, y_sc = _y_scale_fit(ytr, Y_SCALE_MODE)
        ytr = (ytr - y_mu) / y_sc

    # 타깃 셔플
    ytr_perm = np.random.permutation(ytr)
    k = max(1, min(min(30, SELECT_K), Xtr.shape[1]))
    idx = _select_union_indices(Xtr, ytr_perm, k, SEED)
    Xtr_sel = Xtr.values[:, idx]
    Xte_sel = Xte.values[:, idx]

    mdl = _build_svr_pipeline()
    _fit_model(mdl, Xtr_sel, ytr_perm)
    yhat = mdl.predict(Xte_sel)
    if USE_Y_SCALE:
        yhat = yhat * y_sc + y_mu
        y_true = te_df[TARGET_COL].values
    else:
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

    # 디버그 플래그/설정
    lines.append("\n[DEBUG] 설정/분할/선택")
    lines.append(f"- PERMUTE_TARGET={1 if PERMUTE_TARGET else 0} | USE_Y_SCALE={1 if USE_Y_SCALE else 0} | "
                 f"USE_DROP_NEAR_CONST={1 if USE_DROP_NEAR_CONST else 0} | USE_DROP_CORR={1 if USE_DROP_CORR else 0} (thr={DROP_CORR_THR}) | "
                 f"SELECT_MODE={SELECT_MODE} | USE_STABILITY_KSELECT={1 if USE_STABILITY_KSELECT else 0} | "
                 f"USE_CV_KSELECT={1 if USE_CV_KSELECT else 0} | SELECT_K={SELECT_K} | K_GRID={','.join(map(str,K_GRID))} | USE_HPO={1 if USE_HPO else 0}")
    lines.append(f"- 허용 E2X 패스: {sorted(ALLOWED_E2X_PASSES)} | 허용 X2E 패스: {sorted(ALLOWED_X2E_PASSES)}")
    lines.append(f"- t+1 생성 시 삭제 수(마지막 PASS 등): {dropped_shift}")
    for tag in ("E2X","X2E"):
        r = results[tag]
        lines.append(f"- [{tag}] k_used={r.get('k_used','?')} | select_mode={r.get('select_mode','?')} | best_params={r.get('best_params',None)}")

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

    # 학습
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
