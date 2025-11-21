# 032_experiment.py 에다가 추가 기술들을 더합니다.
# 062_experiment.py 기본으로 추가된 기술들을 더합니다.

import re
import shutil
import warnings, os
from datetime import datetime
from pathlib import Path

import joblib, json
import numpy as np
import pandas as pd
import traceback
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from xgboost import XGBRegressor

# (선택) SciPy가 있으면 검정 수행
try:
    from scipy.stats import kruskal, levene, spearmanr
except Exception:
    kruskal = levene = spearmanr = None


warnings.filterwarnings("ignore", category=UserWarning)


# -------- Env flags / Leak guard / Numeric utils --------
def _env_on(name: str, default: str = "0") -> int:
    """환경변수 플래그 ON/OFF (문자 '1'→1, 그 외 0)."""
    return 1 if str(os.getenv(name, default)).strip() == "1" else 0

def _env_get(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()

def _hard_leak_guard(cols: list[str]) -> list[str]:
    bad_kw = ("target", "label", "_y", "oof", "pred", "_hat", "te_")
    safe = []
    for c in cols:
        low = c.lower()
        if any(k in low for k in bad_kw):
            continue
        safe.append(c)
    return safe

def _ensure_numeric_X(df: pd.DataFrame, exclude: set) -> tuple[pd.DataFrame, dict, list[str]]:
    """
    exclude 컬럼을 제거하고, 수치형만 남겨 DataFrame으로 반환.
    med(중앙값 dict), base_cols(열 순서)도 함께 돌려줌.
    """
    X_all = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")
    # 이름 기반 누수 가드
    base_cols = _hard_leak_guard(list(X_all.columns))
    X_all = X_all[base_cols].select_dtypes(include=[np.number]).copy()
    med = X_all.median(numeric_only=True).to_dict()
    X_all = X_all.fillna(med)
    return X_all, med, base_cols


# ------------------------- 경로/상수 --------------------------
DATA_DIR       = globals().get("DATA_DIR", Path("data"))
IMAGES_DIR = Path("images")
SCRIPT_PATH = Path(__file__).resolve()
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SUM_DIR = REPORTS_DIR / "summaries"; SUM_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR    = REPORTS_DIR / "metrics" ; METRICS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR  = globals().get("PROCESSED_DIR", DATA_DIR / "processed")
REPORTS_SUMMARY_DIR = REPORTS_DIR / "summaries"
REPORTS_CODE_DIR = REPORTS_DIR / "scripts"
EXPERIMENT_LOG = REPORTS_DIR / "experiments_log_tplus1_cross.csv"
REPORTS_DIR    = globals().get("REPORTS_DIR", Path("reports"))




PERMUTE_TARGET     = _env_on("PERMUTE_TARGET", "0")     # 전체 타깃 무작위화
LEAK_AUTOCHECK     = _env_on("LEAK_AUTOCHECK", "1")     # 빠른 퍼뮤 감사(선택)
USE_DELTA_TARGET   = _env_on("USE_DELTA_TARGET", "0")   # DT: Δ-타깃 (y - cur_warp)
USE_PAIR_TE        = _env_on("USE_PAIR_TE", "0")        # TE: 페어 Δ-타깃 인코딩(O O F)
USE_HETERO_W       = _env_on("USE_HETERO_W", "0")       # W : 이분산 가중(WLS; O O F std^-1)
USE_MONO_CUR       = _env_on("USE_MONO_CUR", "0")       # MONO: CUR_WARP 단조(+1; XGB 전용)
USE_RM_TRAINONLY   = _env_on("USE_RM_TRAINONLY", "0")   # RM : Train plate만으로 RM 통계 결합
ANCHOR_TAG         = os.environ.get("ANCHOR_TAG", "")   # 앵커 이름 강제(모든 조합 동일 분할)
RUN_NAME           = os.environ.get("RUN_NAME", "pass_t_plus_1_cross")

# 알고리즘 선택 (E2X는 XGB로 고정, X2E는 환경변수로 XGBoost|RandomForest)
X2E_ALGO = os.environ.get("X2E_ALGO", "XGBoost")  # "XGBoost" or "RandomForest"

# ===== [CONSTANTS] =====
TARGET_COL = "warping_index_target"          # t+1
CUR_WARP   = "warping_index_current_pass"    # t
PLATE_COL  = "FM_날판번호"
PASS_COL   = "FM_PASS NO N"
MONTH_COL  = "FM_압연월"
EXCLUDE    = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_FROM_X = set(globals().get("EXCLUDE_FROM_X", {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}))

REPORTS_METRICS_DIR = REPORTS_DIR / "metrics"
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------- 유틸 -----------------------------
def _hard_leak_guard(cols: list[str]) -> list[str]:
    bad_kw = ("target","label","_y","y_","oof","pred","_hat","_te","te_")
    out=[]
    for c in cols:
        low = c.lower()
        if any(k in low for k in bad_kw):
            continue
        out.append(c)
    return out

def _ensure_dirs():
    for d in [PROCESSED_DIR, REPORTS_DIR, REPORTS_SUMMARY_DIR, REPORTS_CODE_DIR, MODELS_DIR, IMAGES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def _next_experiment_id() -> str:
    existing = []
    for f in REPORTS_SUMMARY_DIR.glob("*.txt"):
        m = re.match(r"(\d{3})_", f.name)
        if m:
            try: existing.append(int(m.group(1)))
            except: pass
    return f"{(max(existing)+1) if existing else 1:03d}"

def _append_success_log(row: dict) -> pd.DataFrame:
    cols = [
        "experiment_id","timestamp",
        "e2x_algorithm","x2e_algorithm",
        "e2x_test_r2","x2e_test_r2","average_test_r2",
        "e2x_train_r2","x2e_train_r2",
        "rollforward_r2",
        "e2x_kw_p","x2e_kw_p","e2x_lev_p","x2e_lev_p",
    ]
    if EXPERIMENT_LOG.exists():
        log = pd.read_csv(EXPERIMENT_LOG)
    else:
        log = pd.DataFrame(columns=cols)
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log = log.drop_duplicates(subset="experiment_id", keep="last")
    log["experiment_numeric"] = pd.to_numeric(log["experiment_id"], errors="coerce")
    log = log.sort_values("experiment_numeric").drop(columns="experiment_numeric")
    log.to_csv(EXPERIMENT_LOG, index=False)
    return log

def _base_fe_train_test(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    tr = tr.copy(); te = te.copy()
    fm_cols = [c for c in tr.columns if c.startswith("FM_") and c not in {PLATE_COL,PASS_COL,MONTH_COL}]
    def _fe(d: pd.DataFrame, pass_max: float):
        if fm_cols:
            blk = d[fm_cols]
            d["FM_mean"]  = blk.mean(axis=1)
            d["FM_std"]   = blk.std(axis=1)
            d["FM_max"]   = blk.max(axis=1)
            d["FM_min"]   = blk.min(axis=1)
            d["FM_range"] = d["FM_max"] - d["FM_min"]
            d["FM_cv"]    = d["FM_std"] / (d["FM_mean"].abs() + 1e-8)
        d["FM_PASS_squared"]  = d[PASS_COL] ** 2
        pm = float(pass_max) if pass_max and pass_max>0 else 1.0
        d["FM_PASS_progress"] = d[PASS_COL] / pm
        return d
    pass_max_tr = float(tr[PASS_COL].max()) if len(tr) else 1.0
    return _fe(tr, pass_max_tr), _fe(te, pass_max_tr)

def _train_outlier_bounds(Xtr: pd.DataFrame, ytr: np.ndarray, qx=(0.01,0.99), qy=(0.005,0.995)):
    b = {c: tuple(np.quantile(Xtr[c].values, qx)) for c in Xtr.columns}
    y_lo, y_hi = np.quantile(ytr, qy)
    return b, float(y_lo), float(y_hi)

def _apply_train_outlier_filter(Xtr: pd.DataFrame, ytr: np.ndarray, bounds: dict, y_lo: float, y_hi: float):
    mask = (ytr >= y_lo) & (ytr <= y_hi)
    for c,(lo,hi) in bounds.items():
        xv = Xtr[c].values
        mask &= (xv >= lo) & (xv <= hi)
    return Xtr.loc[mask].reset_index(drop=True), ytr[mask]

def _clip_test_by_bounds(Xte: pd.DataFrame, bounds: dict):
    Xte = Xte.copy()
    for c,(lo,hi) in bounds.items():
        if c in Xte.columns:
            Xte[c] = Xte[c].clip(lower=lo, upper=hi)
    return Xte

def _get_algo_overrides_from_env():
    """
    Return (e2x_algo or None, x2e_algo or None).
    Recognizes both new 'FORCE_*' and legacy '*_ALGO' names.
    """
    import os
    e2x = os.environ.get("FORCE_E2X_ALGO") or os.environ.get("E2X_ALGO")
    x2e = os.environ.get("FORCE_X2E_ALGO") or os.environ.get("X2E_ALGO")
    e2x = str(e2x) if e2x else None
    x2e = str(x2e) if x2e else None
    return e2x, x2e

def write_last_metrics(run_label: str,
                       summary_path: str,
                       e2x_test_r2: float,
                       x2e_test_r2: float,
                       avg_test_r2: float):
    """
    Always overwrite 'reports/metrics/last_run_metrics.csv'
    with a single line having FIXED column order, so that
    shell scripts can reliably read the last two fields.
    """
    from pathlib import Path
    import csv

    metrics_dir = Path("reports") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "last_run_metrics.csv"

    header = ["run_label", "e2x_test_r2", "x2e_test_r2", "average_test_r2", "summary_path"]
    row = [run_label,
           f"{e2x_test_r2:.6f}" if e2x_test_r2 is not None else "",
           f"{x2e_test_r2:.6f}" if x2e_test_r2 is not None else "",
           f"{avg_test_r2:.6f}" if avg_test_r2 is not None else "",
           str(summary_path)]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)

LEAKY_PATTERNS = [
    r"(^|[_\-])target($|[_\-])",
    r"(^|[_\-])label($|[_\-])",
    r"(^|[_\-])y_true($|[_\-])",
    r"(^|[_\-])y_pred($|[_\-])",
    r"(^|[_\-])oof($|[_\-])",
    r"(^|[_\-])pred($|[_\-])",
    r"warping_index_target",           # explicit
]
LEAKY_REGEX = re.compile("|".join(LEAKY_PATTERNS), re.IGNORECASE)
WHITELIST_EXACT = { "warping_index_current_pass" }

def _drop_name_leaky_cols(df: pd.DataFrame,
                          exclude: set[str]) -> pd.DataFrame:
    keep = []
    dropped = []
    for c in df.columns:
        if c in exclude or c in WHITELIST_EXACT:
            keep.append(c); continue
        if LEAKY_REGEX.search(c):
            dropped.append(c)
        else:
            keep.append(c)
    if dropped:
        # optional: collect to GLOBAL_DIAG["name_drop_cols"]
        pass
    return df[keep].copy()

def _ensure_numeric_X(
    df: pd.DataFrame,
    exclude: set | list | tuple | None = None,
    ref_cols: list[str] | None = None,
    fillmed: dict[str, float] | None = None
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    """
    수치 피처만 추출 + 결측 중앙값 대치.
    - exclude: X에서 제외할 컬럼들. 미지정 시 EXCLUDE_FROM_X 사용.
    - ref_cols: (테스트/검증용) 열 순서를 train과 동일하게 강제할 때 사용.
    - fillmed : (테스트/검증용) train 중앙값 딕셔너리를 그대로 재사용할 때 전달.
    return: (X_df, medians_dict, col_list)
    """
    if exclude is None:
        exclude = EXCLUDE_FROM_X
    # 제외 컬럼 드롭 → 수치만
    X_all = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")
    X = X_all.select_dtypes(include=[np.number]).copy()

    # 참조 열 강제(테스트/검증 단계)
    if ref_cols is not None:
        X = X.reindex(columns=ref_cols, fill_value=np.nan)

    # 중앙값 계산/재사용
    if fillmed is None:
        med = X.median(numeric_only=True).to_dict()
    else:
        med = dict(fillmed)

    X = X.fillna(med)
    return X, med, list(X.columns)

def _assert_and_drop_value_leak(Xtr_df: pd.DataFrame, ytr: np.ndarray,
                                Xte_df: pd.DataFrame | None = None
                                ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Drop features that are exactly equal to y, or nearly perfectly correlated (|r|>=0.99999)
    computed on TRAIN ONLY; drop same columns from test if present.
    """
    drop_cols = []
    y = ytr
    for c in Xtr_df.columns:
        xc = Xtr_df[c].values
        if xc.shape == y.shape and np.all(np.isfinite(xc)) and np.all(np.isfinite(y)):
            if np.array_equal(xc, y):
                drop_cols.append(c); continue
        sx = np.std(xc); sy = np.std(y)
        if sx >= 1e-12 and sy >= 1e-12:
            r = float(np.corrcoef(xc, y)[0, 1])
            if np.isfinite(r) and abs(r) >= 0.99999:
                drop_cols.append(c)
    if drop_cols:
        drop_cols = sorted(set([c for c in drop_cols if c not in WHITELIST_EXACT]))
        Xtr_df = Xtr_df.drop(columns=drop_cols, errors="ignore")
        if Xte_df is not None:
            Xte_df = Xte_df.drop(columns=drop_cols, errors="ignore")
    return Xtr_df, Xte_df

def _write_last_run_metrics(avg_test_r2: float, summary_path: Path):
    """
    쉘 스크립트가 읽는 metrics 파일을 항상 남긴다.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    mpath = METRICS_DIR / "last_run_metrics.csv"
    row = pd.DataFrame([{
        "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_path": str(summary_path),
        "avg_test_r2": float(avg_test_r2),
    }])
    row.to_csv(mpath, index=False)
    print(f"✓ 메트릭 저장: {mpath}")

# ---------- 1) 병합 & 타깃 생성: t → t+1 (shift -1) ----------

def load_and_merge_data_tplus1() -> pd.DataFrame:
    print("\n" + "="*80)
    print("[Step 1] 데이터 병합 및 Target 변환 (Pass t → Pass t+1)")
    print("="*80)

    gt_entry = pd.read_csv(DATA_DIR / "entry_direction_results.csv")
    gt_exit  = pd.read_csv(DATA_DIR / "exit_direction_results.csv")
    ground_truth = pd.concat([gt_entry, gt_exit], ignore_index=True)

    rm_data = pd.read_csv(DATA_DIR / "posco1_105190.csv")  # RM
    fm_data = pd.read_csv(DATA_DIR / "posco2_105190.csv")  # FM

    # 원본 코드의 키 추출 규칙 재사용
    ground_truth["extracted_plate"] = ground_truth["filename"].str.extract(r"(PB\d+)")
    ground_truth["extracted_pass"]  = ground_truth["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    # RM 병합 + 판단위 통계(평균/최대/최소/표준편차/중앙값)
    merged_rm = pd.merge(
        ground_truth, rm_data,
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
    merged_fm = pd.merge(
        ground_truth, fm_data,
        left_on=["extracted_plate","extracted_pass"],
        right_on=[PLATE_COL, PASS_COL],
        how="inner"
    ).sort_values([PLATE_COL, PASS_COL])

    # 핵심: t+1 타깃
    merged_fm[TARGET_COL] = merged_fm.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged_fm = merged_fm.rename(columns={"warping_index": CUR_WARP})
    before = len(merged_fm)
    merged_fm = merged_fm.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"Target shift(-1): {before} → {len(merged_fm)} (삭제 {before-len(merged_fm)})")

    # RM 집계 결합 + 보조 컬럼 정리
    final_df = pd.merge(
        merged_fm, rm_stats,
        left_on=PLATE_COL, right_on="RM_날판번호", how="left"
    ).drop(columns=["RM_날판번호"], errors="ignore")
    final_df = final_df.drop(columns=[
        "extracted_plate","extracted_pass","filename","quality_grade","quality_grade_current_pass","direction"
    ], errors="ignore")

    out_path = PROCESSED_DIR / "final_merged_data_regression_tplus1.csv"
    final_df.to_csv(out_path, index=False)
    print(f"✓ 저장: {out_path}")
    return final_df

def _pair_str(df: pd.DataFrame) -> pd.Series:
    return (df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str))

def _safe_median_fill(X_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    med = X_df.median(numeric_only=True).to_dict()
    return X_df.fillna(med), med

class SimpleKBest:  # 상관 기반 K-best (joblib 덤프 불필요: 선택 컬럼만 저장)
    def __init__(self, k: int, mandatory: list[str] | None = None):
        self.k = int(max(1, k)); self.mandatory = list(mandatory or []); self.selected_cols_ = []

    def fit(self, X_df: pd.DataFrame, y: np.ndarray):
        X = X_df.select_dtypes(include=[np.number]).copy()
        cols_all = X.columns.tolist()
        finite_mask = np.isfinite(X.values).all(axis=0)
        var = X.var(numeric_only=True).reindex(cols_all).fillna(0.0).values
        keep = [c for c,m in zip(cols_all, (finite_mask & (var > 1e-12))) if m]
        Xc = X[keep].fillna(X[keep].median(numeric_only=True))
        y0 = y - y.mean(); ys = y0.std() if y0.std() >= 1e-12 else 1e-12
        xr = (Xc.values - Xc.values.mean(axis=0)) / np.clip(Xc.values.std(axis=0), 1e-12, None)
        yr = y0 / ys
        r = (xr.T @ yr) / (len(y) - 1)
        r2 = np.clip(r**2, 0.0, 1.0 - 1e-12)
        F = (r2 / (1.0 - r2)) * (len(y) - 2)
        mand = [c for c in self.mandatory if c in keep]
        others = [c for c in keep if c not in mand]
        order = np.argsort([F[others.index(c)] for c in others])[::-1] if others else []
        k_remain = max(1, min(self.k - len(mand), len(others)))
        self.selected_cols_ = mand + [others[i] for i in order[:k_remain]]
        return self

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        d = X_df.copy()
        # 누락 컬럼은 NaN (이전 단계에서 median 대치가 끝나있음)
        for c in self.selected_cols_:
            if c not in d.columns: d[c] = np.nan
        return d[self.selected_cols_].values

@dataclass
class BaggedRegressor:
    """여러 개 개별 모델의 예측 평균을 내는 래퍼(roll-forward에서 predict 보장)."""
    models: list

    def predict(self, X):
        if not self.models:
            return np.zeros((len(X),), dtype=float)
        preds = [m.predict(X) for m in self.models]
        return np.mean(np.vstack(preds), axis=0)

def xgb_fit_safe(model: XGBRegressor,
                 X, y,
                 eval_set=None,
                 sample_weight=None,
                 verbose=False):
    """
    xgboost 버전별 호환 안전 fit. early_stopping_rounds/ eval_set 지원 여부에 따라 단계적 시도.
    """
    try:
        model.fit(X, y,
                  sample_weight=sample_weight,
                  eval_set=eval_set,
                  early_stopping_rounds=50,
                  verbose=verbose)
        return
    except TypeError:
        pass
    try:
        model.fit(X, y,
                  sample_weight=sample_weight,
                  eval_set=eval_set,
                  verbose=verbose)
        return
    except TypeError:
        pass
    model.fit(X, y, sample_weight=sample_weight)

def _select_kbest(Xtr_df: pd.DataFrame, ytr: np.ndarray, k: int, mandatory: list[str]|None=None) -> list[str]:
    k = int(max(1, k)); man = [c for c in (mandatory or []) if c in Xtr_df.columns]
    cols = [c for c in Xtr_df.columns if c not in man]
    if not cols: return man
    X = Xtr_df[cols].values
    y = ytr - ytr.mean()
    ys = y.std() or 1e-12
    xr = (X - X.mean(axis=0)) / np.clip(X.std(axis=0), 1e-12, None)
    r = (xr.T @ (y/ys)) / (len(y)-1)
    r2 = np.clip(r**2, 0.0, 1.0-1e-12)
    F = (r2/(1.0-r2)) * (len(y)-2)
    order = np.argsort(F)[::-1]
    rest = [cols[i] for i in order[:max(0, k-len(man))]]
    chosen = man + rest
    # 다시 한 번 하드 가드
    assert all("target" not in c.lower() for c in chosen), "LeakGuard: selected_cols contains target-like name"
    return chosen

def _build_model(algorithm: str, seed: int = 42, n_features: int | None = None,
                 monotone_idx: int | None = None):
    if algorithm == "XGBoost":
        from xgboost import XGBRegressor
        params = dict(
            objective="reg:squarederror", random_state=seed, tree_method="hist",
            n_estimators=800, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.0, n_jobs=0,
            eval_metric="rmse"
        )
        if monotone_idx is not None and n_features is not None:
            v = [0]*n_features; 
            if 0 <= monotone_idx < n_features: v[monotone_idx] = 1
            params["monotone_constraints"] = tuple(v)
        return XGBRegressor(**params)
    elif algorithm == "RandomForest":
        return RandomForestRegressor(
            random_state=seed, n_estimators=600, n_jobs=-1,
            max_depth=20, min_samples_leaf=5, max_features="sqrt"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def _safe_fit_xgb(model, Xfit, yfit, Xval=None, yval=None, sample_weight=None):
    """xgboost 버전별 early_stopping 인자 호환."""
    try:
        if Xval is not None and yval is not None:
            model.fit(Xfit, yfit, sample_weight=sample_weight,
                      eval_set=[(Xval, yval)], early_stopping_rounds=50, verbose=False)
        else:
            model.fit(Xfit, yfit, sample_weight=sample_weight)
    except TypeError:  # 구버전: early_stopping 미지원
        model.fit(Xfit, yfit, sample_weight=sample_weight)

# --------- 2) 전이 세트(E→X / X→E) 분리: 홀/짝 pass ----------
def split_cross_transitions(final_df: pd.DataFrame):
    print("\n" + "="*80)
    print("[Step 2] 전이(E→X / X→E) 세트 분리 (t → t+1)")
    print("="*80)
    e2x_df = final_df[final_df[PASS_COL] % 2 == 1].copy()  # Entry t → Exit t+1
    x2e_df = final_df[final_df[PASS_COL] % 2 == 0].copy()  # Exit  t → Entry t+1

    e2x_path = PROCESSED_DIR / "e2x_raw_tplus1.csv"
    x2e_path = PROCESSED_DIR / "x2e_raw_tplus1.csv"
    e2x_df.to_csv(e2x_path, index=False)
    x2e_df.to_csv(x2e_path, index=False)
    print(f"✓ 저장: {e2x_path}")
    print(f"✓ 저장: {x2e_path}")
    return e2x_df, x2e_df

# ---------------- 3) Feature Engineering (동일 규칙) ---------------
def feature_engineering(df: pd.DataFrame, dataset_name: str, output_path: Path):
    print(f"\n[Step 3] Feature Engineering - {dataset_name}")
    df = df.copy()

    fm_cols = [c for c in df.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if fm_cols:
        fm_blk = df[fm_cols]
        df["FM_mean"]  = fm_blk.mean(axis=1)
        df["FM_std"]   = fm_blk.std(axis=1)
        df["FM_max"]   = fm_blk.max(axis=1)
        df["FM_min"]   = fm_blk.min(axis=1)
        df["FM_range"] = df["FM_max"] - df["FM_min"]
        df["FM_cv"]    = df["FM_std"] / (df["FM_mean"].abs() + 1e-8)

    if PASS_COL in df.columns:
        df["FM_PASS_squared"]  = df[PASS_COL] ** 2
        df["FM_PASS_progress"] = df[PASS_COL] / df[PASS_COL].max()

    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path}")
    return df

# -------------------- 4) 이상치 제거 (동일 규칙) -------------------
def remove_outliers(input_path: Path, output_path: Path, dataset_name: str):
    """
    순수 비감독(IQR) 기반 필터.
    - y/라벨 불참조 → 누수 없음
    - 학습 전 공통 클린으로 쓰되, 공격적인 제거는 피함(1%~99% clip)
    """
    print(f"\n[Step 4] 이상치 제거 - {dataset_name}")
    df = pd.read_csv(input_path)

    features_all = [c for c in df.columns if c not in EXCLUDE_FROM_X]
    num_features = df[features_all].select_dtypes(include=[np.number]).columns.tolist()
    X = df[num_features].copy()

    # IQR 기반 soft mask (전역·비감독)
    q1 = X.quantile(0.01)
    q99 = X.quantile(0.99)
    mask = ((X >= q1) & (X <= q99)).all(axis=1)

    cleaned = df.loc[mask].reset_index(drop=True)
    cleaned.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path} (최종 {cleaned.shape[0]}개)")
    return cleaned

# ------------------- 5) 알고리즘 비교 (동일 규칙) -------------------
def compare_algorithms(df: pd.DataFrame, dataset_name: str):
    """
    plate-group CV로 XGB/RF 비교.
    단, 환경변수(E2X, X2E)로 강제할 수 있음.
    """
    print("\n" + "="*80)
    print(f"[Step 3] 알고리즘 비교 - {dataset_name} (plate-group CV)")
    print("="*80)

    # 숫자만, 누수 가드
    X, _, _ = _ensure_numeric_X(df, EXCLUDE_FROM_X)
    y = df[TARGET_COL].values
    groups = df[PLATE_COL].astype(str).values

    gkf = GroupKFold(n_splits=5)

    algos = {
        "XGBoost": XGBRegressor(
            random_state=42, n_estimators=400, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist"
        ),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=600, n_jobs=-1)
    }

    # CV 평가
    scores = {}
    for name, model in algos.items():
        s = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="r2", n_jobs=-1)
        scores[name] = (float(s.mean()), float(s.std()))
        print(f"  {name:12s}: R2 = {s.mean():.4f} ± {s.std():.4f}")

    # 선택(기본: CV 최고)
    best = max(scores.items(), key=lambda kv: kv[1][0])[0]

    # 환경변수 강제 (E2X는 고정 XGB, X2E만 외부에서 바꿀 수도 있음)
    if dataset_name.upper() == "E2X":
        forced = _env_get("E2X_ALGO", "XGBoost") or "XGBoost"
        if forced in algos:
            best = forced
    else:
        forced = _env_get("X2E_ALGO", "").strip()
        if forced in algos:
            best = forced

    print(f"\n  → 최적 알고리즘: {best} (CV R2 = {scores[best][0]:.4f})")
    return best, scores[best][0]
# ------------ 6) 학습/튜닝 + 전처리 메타 저장(중앙값/열순서) -----------
# ---------- (A) OOF 쌍-Δ Target Encoding(EB 축소) ----------
def _build_pair_key(s: pd.Series) -> str:
    return f"{int(s[PASS_COL])}→{int(s[PASS_COL])+1}"

def _oof_pair_te(train_df: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> tuple[pd.Series, dict]:
    """
    plate GroupKFold OOF로 pair 기반 Δ-타깃 인코딩.
    반환: oof_te(train), te_map(global)
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = train_df[PLATE_COL].astype(str).values
    pair_col = train_df.apply(_build_pair_key, axis=1)

    oof = np.zeros(len(train_df), dtype=float)
    for tr, va in gkf.split(train_df, y, groups=groups):
        kv = pair_col.iloc[tr].values
        yt = y[tr]
        # fold 내부 통계
        dfm = pd.DataFrame({"k": kv, "y": yt}).groupby("k")["y"].mean()
        # validation 변환
        oof[va] = pair_col.iloc[va].map(dfm).fillna(dfm.mean()).values

    # 전체 train으로 global map
    full_map = pd.DataFrame({"k": pair_col.values, "y": y}).groupby("k")["y"].mean().to_dict()
    return pd.Series(oof, index=train_df.index, name="te_pair_delta"), full_map

def _compute_hetero_weights(train_df: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """
    plate OOF 잔차 분산 기반 EB‑shrink 가중치.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = train_df[PLATE_COL].astype(str).values
    pair_col = train_df.apply(_build_pair_key, axis=1)

    # 간단 베이스: y_hat := pair별 평균(OOB)
    oof = np.zeros(len(train_df), dtype=float)
    for tr, va in gkf.split(train_df, y, groups=groups):
        kv = pair_col.iloc[tr].values
        yt = y[tr]
        dfm = pd.DataFrame({"k": kv, "y": yt}).groupby("k")["y"].mean()
        oof[va] = pair_col.iloc[va].map(dfm).fillna(dfm.mean()).values

    resid = y - oof
    by_pair = pd.DataFrame({"k": pair_col.values, "r2": resid**2})
    stat = by_pair.groupby("k")["r2"].agg(["mean", "count"]).reset_index()
    global_var = float(resid.var()) if len(resid) else 1.0
    lam = 10.0  # shrink 강도(작을수록 pair 분산 신뢰)
    stat["v_shrunk"] = ((stat["count"] - 1) * stat["mean"] + lam * global_var) / (stat["count"] - 1 + lam)
    v_map = dict(zip(stat["k"], stat["v_shrunk"]))

    w = 1.0 / (pd.Series(pair_col).map(v_map).fillna(global_var).values + 1e-6)
    return w


def _base_feat_by_train_scale(tr_df: pd.DataFrame, te_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = tr_df.copy(); te = te_df.copy()
    fm_cols = [c for c in tr.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    def add_block(d: pd.DataFrame, pass_max: float):
        if fm_cols:
            blk = d[fm_cols]
            d["FM_mean"]  = blk.mean(axis=1); d["FM_std"] = blk.std(axis=1)
            d["FM_max"]   = blk.max(axis=1);  d["FM_min"] = blk.min(axis=1)
            d["FM_range"] = d["FM_max"] - d["FM_min"]
            d["FM_cv"]    = d["FM_std"] / (d["FM_mean"].abs() + 1e-8)
        d["FM_PASS_squared"]  = d[PASS_COL] ** 2
        d["FM_PASS_progress"] = d[PASS_COL] / (pass_max if pass_max > 0 else 1.0)
        return d
    pmax = float(tr[PASS_COL].max()) if len(tr) else 1.0
    return add_block(tr, pmax), add_block(te, pmax)

def add_pair_delta_te_feature(train_df: pd.DataFrame, test_df: pd.DataFrame, n0=50):
    """OOF(plate-group) Δ-타깃의 EB 수축 평균을 TE로 추가. (누수 방지)"""
    tr = train_df.copy(); te = test_df.copy()
    tr["pair"] = _pair_str(tr); te["pair"] = _pair_str(te)
    tr["delta"] = tr[TARGET_COL] - tr[CUR_WARP]
    gkf = GroupKFold(n_splits=3)
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

def compute_hetero_weights(train_df: pd.DataFrame, test_df: pd.DataFrame, clip=(0.25, 3.0)):
    """OOF로 pair별 Δ의 표준편차를 추정 → 표준편차의 역수 비례 가중."""
    tr = train_df.copy(); te = test_df.copy()
    tr["pair"] = _pair_str(tr)
    tr["delta"] = tr[TARGET_COL] - tr[CUR_WARP]
    gkf = GroupKFold(n_splits=3); plates = tr[PLATE_COL].astype(str).values
    oof_std = np.zeros(len(tr))
    for tr_i, va_i in gkf.split(tr, tr["delta"].values, groups=plates):
        sub = tr.iloc[tr_i]
        gs = sub["delta"].std(ddof=1)
        stat = sub.groupby("pair")["delta"].agg(n="size", std="std").reset_index()
        w = stat["n"]/(stat["n"]+50)
        stat["std_sh"] = w*stat["std"].fillna(gs) + (1-w)*gs
        m = dict(zip(stat["pair"], stat["std_sh"]))
        oof_std[va_i] = tr.iloc[va_i]["pair"].map(m).fillna(gs).values
    gs = tr["delta"].std(ddof=1); w_tr = np.clip(float(gs)/(oof_std + 1e-6), clip[0], clip[1])
    return w_tr

def attach_train_only_rm_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train 판만으로 RM 집계 생성 → train/test 둘에 매핑 (누수 방지)"""
    D = Path("data")
    src = D/"posco1_105190.csv"
    if not src.exists():
        return train_df, test_df  # 원천 데이터 없으면 skip
    rm = pd.read_csv(src)
    plates_tr = set(train_df[PLATE_COL].astype(str).unique().tolist())
    rm = rm[rm["RM_날판번호"].astype(str).isin({str(p) for p in plates_tr})].copy()
    if rm.empty: return train_df, test_df
    num_cols = rm.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"RM_날판번호","RM_압연Pass번호","warping_index"}
    stat_cols = [c for c in num_cols if c not in drop_like and not c.endswith("_target")]
    rm_stats = rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(6)
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    rm_stats = rm_stats.reset_index()
    tr = pd.merge(train_df, rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left")
    te = pd.merge(test_df,  rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left")
    tr = tr.drop(columns=["RM_날판번호"], errors="ignore")
    te = te.drop(columns=["RM_날판번호"], errors="ignore")
    # 결측은 train 기준 전역 중앙값으로 채움
    gmed = rm_stats.drop(columns=["RM_날판번호"], errors="ignore").median(numeric_only=True)
    for c in gmed.index:
        if c in te.columns: te[c] = te[c].fillna(float(gmed[c]))
        if c in tr.columns: tr[c] = tr[c].fillna(float(gmed[c]))
    return tr, te
# ---------- (B) OOF 쌍-표준편차 기반 가중치(WLS) ----------
def compute_oof_pair_std_weights(train_df, test_df, PASS_COL, PLATE_COL, CUR_WARP, TARGET_COL,
                                 n0=50, clip=(0.25, 3.0)):
    tr = train_df.copy(); te = test_df.copy()
    tr["pair"] = tr[PASS_COL].astype(int).astype(str) + "→" + (tr[PASS_COL]+1).astype(int).astype(str)
    te["pair"] = te[PASS_COL].astype(int).astype(str) + "→" + (te[PASS_COL]+1).astype(int).astype(str)
    tr["delta"] = tr[TARGET_COL] - tr[CUR_WARP]

    gkf = GroupKFold(n_splits=3)
    groups = tr[PLATE_COL].astype(str).values
    oof_std = np.zeros(len(tr))
    for tr_i, va_i in gkf.split(tr, tr["delta"].values, groups):
        sub = tr.iloc[tr_i]
        gs = sub["delta"].std(ddof=1)
        stat = sub.groupby("pair")["delta"].agg(n="size", std="std").reset_index()
        w = stat["n"]/(stat["n"]+n0)
        stat["std_sh"] = w*stat["std"].fillna(gs) + (1-w)*gs
        m = dict(zip(stat["pair"], stat["std_sh"]))
        pva = tr.iloc[va_i]["pair"].values
        oof_std[va_i] = [m.get(p, gs) for p in pva]

    # test mapping
    gs = tr["delta"].std(ddof=1)
    stat = tr.groupby("pair")["delta"].agg(n="size", std="std").reset_index()
    w = stat["n"]/(stat["n"]+n0)
    stat["std_sh"] = w*stat["std"].fillna(gs) + (1-w)*gs
    m_full = dict(zip(stat["pair"], stat["std_sh"]))
    te_std = te["pair"].map(m_full).fillna(gs).values

    w_tr = np.clip(float(gs)/(oof_std + 1e-6), clip[0], clip[1])
    return w_tr, te_std

# ---------- (C) 쌍-기준선 2-Stage Residual 학습 ----------
def train_model(df: pd.DataFrame,
                dataset_name: str,           # "E2X" or "X2E"
                algorithm: str = "XGBoost",  # "XGBoost" | "RandomForest"
                seed: int = 42,
                select_k: int = 120):
    """
    통합본의 핵심 학습 함수.
    - plate-group split 이후에만 통계/TE/WLS를 '학습'
    - Δ-타깃/TE/W/Mono 플래그를 모두 적용
    - XGB monotone(+1 on CUR_WARP) 옵션 지원
    - 모델/메타 저장 (003/032/088 계열과 동일 포맷)
    """
    rng = np.random.RandomState(seed)

    # ---- Flags ----
    USE_DT   = _env_on("USE_DELTA_TARGET", "0")
    USE_TE   = _env_on("USE_PAIR_TE", "0")
    USE_W    = _env_on("USE_HETERO_W", "0")
    USE_MONO = _env_on("USE_MONO_CUR", "0")

    # ---- 데이터 세분화(E2X 홀수 / X2E 짝수) ----
    data = df.copy()
    if dataset_name.upper() == "E2X":
        data = data[data[PASS_COL] % 2 == 1].copy()
        prefix = "e2x"
    else:
        data = data[data[PASS_COL] % 2 == 0].copy()
        prefix = "x2e"

    # ---- Permutation leak audit (선택) ----
    if _env_on("PERMUTE_TARGET", "0") == 1:
        data[TARGET_COL] = rng.permutation(data[TARGET_COL].values)
        print("[PERMUTE] Target shuffled for leak sanity check.")

    # ---- plate-group split ----
    groups_all = data[PLATE_COL].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gss.split(data, data[TARGET_COL].values, groups=groups_all))

    tr_df_raw = data.iloc[tr_idx].copy()
    te_df_raw = data.iloc[te_idx].copy()

    # ---- Feature engineering (PASS scaling: train max 기준) ----
    def _fe(d: pd.DataFrame, pass_max: float):
        d = d.copy()
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

    pass_max_train = float(tr_df_raw[PASS_COL].max()) if len(tr_df_raw) else 1.0
    tr_fe = _fe(tr_df_raw, pass_max_train)
    te_fe = _fe(te_df_raw, pass_max_train)

    # ---- 기본 X, y 구성 (train 중앙값으로 impute) ----
    Xtr_df, med, base_cols = _ensure_numeric_X(tr_fe, EXCLUDE_FROM_X)
    Xte_df = te_fe[base_cols].select_dtypes(include=[np.number]).copy().fillna(med)

    y_tr = tr_fe[TARGET_COL].values.astype(float)
    y_te = te_fe[TARGET_COL].values.astype(float)

    # ---- Δ-타깃 옵션 ----
    if USE_DT == 1 and CUR_WARP in tr_fe.columns:
        y_tr = (tr_fe[TARGET_COL] - tr_fe[CUR_WARP]).values.astype(float)
        y_te = (te_fe[TARGET_COL] - te_fe[CUR_WARP]).values.astype(float)

    # ---- Pair Δ‑TE(OOF) 옵션 ----
    te_train = None
    te_map = None
    if USE_TE == 1:
        te_train, te_map = _oof_pair_te(tr_fe, y_tr, n_splits=5)
        # train에 붙이고, test는 full_map으로 변환
        Xtr_df = Xtr_df.copy()
        Xtr_df["te_pair_delta"] = te_train.values
        pair_te_test_key = te_fe.apply(_build_pair_key, axis=1)
        Xte_df = Xte_df.copy()
        Xte_df["te_pair_delta"] = pair_te_test_key.map(te_map).fillna(np.nanmean(te_train.values)).values

    # ---- 간단 K-best (이름 유지) ----
    # 상관 기반 스코어 → 상위 select_k, 반드시 CUR_WARP 포함(있다면)
    must = [CUR_WARP] if CUR_WARP in Xtr_df.columns else []
    cols = list(Xtr_df.columns)
    # corr score (수치 안정화)
    Xm = Xtr_df.fillna(0.0).values
    ym = (y_tr - y_tr.mean()) / (y_tr.std() if y_tr.std() > 1e-12 else 1.0)
    xs = Xtr_df.std().replace(0, 1.0).values
    xr = (Xm - Xm.mean(axis=0)) / xs
    r = (xr.T @ ym) / max(1, len(y_tr) - 1)
    r2 = np.nan_to_num(r * r, nan=0.0, posinf=0.0, neginf=0.0)
    score_map = dict(zip(cols, r2))
    # 필수 + 상위
    others = [c for c in cols if c not in must]
    order = sorted(others, key=lambda c: score_map.get(c, 0.0), reverse=True)
    take = must + order[: max(1, min(select_k - len(must), len(order)))]
    Xtr_sel = Xtr_df[take].copy()
    Xte_sel = Xte_df[take].copy()

    # ---- Hetero‑W (OOF) 옵션 ----
    sample_weight = None
    if USE_W == 1:
        sample_weight = _compute_hetero_weights(tr_fe, y_tr, n_splits=5)

    # ---- 모델 구성 (E2X = XGB 고정 / X2E = 외부지정 또는 인자) ----
    algo = algorithm
    if dataset_name.upper() == "E2X":
        algo = "XGBoost"

    if algo == "XGBoost":
        # monotone(+1 on CUR_WARP) 지원
        mono = None
        if USE_MONO == 1:
            mono = ["1" if c == CUR_WARP else "0" for c in take]
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=seed, tree_method="hist",
            n_estimators=400, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.0,
            monotone_constraints=f"({','.join(mono)})" if mono is not None else None,
        )
    elif algo == "RandomForest":
        model = RandomForestRegressor(
            random_state=seed, n_estimators=600, n_jobs=-1,
            max_depth=None, min_samples_leaf=1, max_features="sqrt"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # ---- 내부 홀드아웃(plate)로 과적합 가드 (early stopping 미사용) ----
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    grp_tr = tr_fe[PLATE_COL].astype(str).values
    tr_i, va_i = next(gss_inner.split(Xtr_sel, y_tr, groups=grp_tr))
    Xfit, yfit = Xtr_sel.iloc[tr_i], y_tr[tr_i]
    Xval, yval = Xtr_sel.iloc[va_i], y_tr[va_i]
    if sample_weight is not None:
        sw_fit = sample_weight[tr_i]
        model.fit(Xfit, yfit, sample_weight=sw_fit)
    else:
        model.fit(Xfit, yfit)

    # ---- 평가 ----
    y_tr_pred = model.predict(Xtr_sel)
    y_te_pred = model.predict(Xte_sel)

    results = {
        "train_r2":  float(r2_score(y_tr, y_tr_pred)),
        "test_r2":   float(r2_score(y_te, y_te_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_tr, y_tr_pred))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te, y_te_pred))),
        "train_samples": int(len(y_tr)), "test_samples": int(len(y_te)),
        "best_params": getattr(model, "get_params", lambda: {})(),
        "algorithm": algo,
    }

    # ---- 산출물 저장(모델/열/중앙값/테스트판목록) ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,                MODELS_DIR / f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(take,                 MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl")
    joblib.dump(med,                  MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl")

    # test plate anchor 저장(재현성 목적)
    test_plates = sorted(te_df_raw[PLATE_COL].astype(str).unique().tolist())
    joblib.dump(test_plates,          MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl")

    # (선택) TE 맵 저장(추후 재사용)
    if te_map is not None:
        joblib.dump(te_map,           MODELS_DIR / f"{prefix}_pair_te_map.pkl")

    return results, set(test_plates)

# ----------------- 7) 번갈아 예측(roll-forward) -------------------
def _load_bundle(prefix: str):
    return {
        "model":    joblib.load(MODELS_DIR / f"{prefix}_regressor_tplus1.pkl"),
        "cols":     joblib.load(MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl"),
    }

def _predict_next_from_row(row: pd.Series, bundle: dict) -> float:
    # 열 이름 기준으로 재구성하고, 누락은 medians로 채움
    desired = list(bundle["cols"])
    df1 = pd.DataFrame([ {c: row.get(c, np.nan) for c in desired} ])[desired]
    df1 = df1.fillna(pd.Series(bundle["medians"]))
    return float(bundle["model"].predict(df1)[0])

# --- rollforward 예측: '교집합 Test 판'만 평가하는 함수 추가 ---
def rollforward_r2_on_test_only(all_featured_df: pd.DataFrame) -> float:
    from sklearn.metrics import r2_score
    e2x_bundle = {
        "model":    joblib.load(Path("models")/"e2x_regressor_tplus1.pkl"),
        "cols":     joblib.load(Path("models")/"e2x_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(Path("models")/"e2x_feature_medians_tplus1.pkl"),
    }
    x2e_bundle = {
        "model":    joblib.load(Path("models")/"x2e_regressor_tplus1.pkl"),
        "cols":     joblib.load(Path("models")/"x2e_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(Path("models")/"x2e_feature_medians_tplus1.pkl"),
    }
    e2x_test = set(joblib.load(Path("models")/"e2x_test_plates_tplus1.pkl"))
    x2e_test = set(joblib.load(Path("models")/"x2e_test_plates_tplus1.pkl"))
    test_plates = e2x_test.intersection(x2e_test)
    if not test_plates:
        print("[Strict RF] 교집합 Test 판이 없습니다.")
        return float("nan")

    rows, row_errors = [], []
    for plate, sub in all_featured_df.groupby("FM_날판번호"):
        if str(plate) not in test_plates: 
            continue
        sub = sub.sort_values("FM_PASS NO N")
        idx = sub.set_index("FM_PASS NO N", drop=False)
        for p in idx.index:
            if (p + 1) not in idx.index:
                continue
            try:
                row_t: pd.Series = idx.loc[p].drop(labels=["warping_index_target"], errors="ignore")
                bundle = e2x_bundle if (p % 2 == 1) else x2e_bundle
                cols = list(bundle["cols"]); med = bundle["medians"]
                X_row = pd.DataFrame([row_t.reindex(cols)], columns=cols).fillna(med)
                pred = float(bundle["model"].predict(X_row.values)[0])
                gt_next = float(idx.loc[p+1].get("warping_index_current_pass"))
                rows.append({"plate": plate, "from": p, "to": p+1, "pred": pred, "gt_next": gt_next})
            except Exception as ex:
                row_errors.append({"plate": plate, "from": p, "to": p+1, "error": repr(ex)})
                continue

    if row_errors:
        err_csv = REPORTS_DIR / "rollforward_test_only_row_errors.csv"
        pd.DataFrame(row_errors).to_csv(err_csv, index=False)
        print(f"[Strict RF] 행 단위 에러 {len(row_errors)}건 → {err_csv}")

    if not rows:
        print("[Strict RF] 유효한 예측 행이 없습니다.")
        return float("nan")

    df = pd.DataFrame(rows)
    return float(r2_score(df["gt_next"], df["pred"]))

def rollforward_predict_all_plates(all_featured_df: pd.DataFrame, e2x_bundle: dict, x2e_bundle: dict) -> pd.DataFrame:
    outs=[]
    for plate in sorted(all_featured_df[PLATE_COL].dropna().unique()):
        sub = all_featured_df[all_featured_df[PLATE_COL]==plate].sort_values(PASS_COL)
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index: 
                continue
            row_t = idx.loc[p].drop(labels=[TARGET_COL], errors="ignore")
            bundle = e2x_bundle if (p % 2 == 1) else x2e_bundle
            pred = _predict_next_from_row(row_t, bundle)
            gt_next = idx.loc[p+1].get(CUR_WARP, None)
            outs.append({PLATE_COL: plate, "from_pass": p, "to_pass": p+1, "pred": pred, "gt_next": gt_next})
    return pd.DataFrame(outs) if outs else pd.DataFrame(columns=[PLATE_COL,"from_pass","to_pass","pred","gt_next"])

# ------------- 8) Δ-분석: 패스쌍별 분포/검정/리포트 ----------------
def analyze_transition_variability(final_df: pd.DataFrame) -> dict:
    df = final_df.copy()
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["pair"] = df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

    def _stats(sub):
        g = sub.groupby("pair")["delta"]
        stat = g.agg(count="count", mean="mean", std="std", median="median",
                     q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
                     min_="min", max_="max").reset_index()
        # 변동성 비교 지표
        if stat["std"].notna().any():
            std_ratio = (stat["std"].max() / stat["std"].replace(0, np.nan).min())
        else:
            std_ratio = np.nan
        # 검정
        kw_p = lev_p = np.nan
        if kruskal is not None:
            groups = [g.get_group(k).values for k in g.groups if len(g.get_group(k)) >= 2]
            if len(groups) >= 2:
                kw_p = float(kruskal(*groups).pvalue)
        if levene is not None:
            groups = [g.get_group(k).values for k in g.groups if len(g.get_group(k)) >= 2]
            if len(groups) >= 2:
                lev_p = float(levene(*groups).pvalue)
        return stat, std_ratio, kw_p, lev_p

    e2x = df[df[PASS_COL] % 2 == 1]
    x2e = df[df[PASS_COL] % 2 == 0]

    e2x_stat, e2x_std_ratio, e2x_kw_p, e2x_lev_p = _stats(e2x)
    x2e_stat, x2e_std_ratio, x2e_kw_p, x2e_lev_p = _stats(x2e)

    e2x_stat.to_csv(REPORTS_DIR / "e2x_delta_stats.csv", index=False)
    x2e_stat.to_csv(REPORTS_DIR / "x2e_delta_stats.csv", index=False)
    print(f"✓ Δ 통계 저장: {REPORTS_DIR/'e2x_delta_stats.csv'}, {REPORTS_DIR/'x2e_delta_stats.csv'}")

    return {
        "e2x": {"stats": e2x_stat, "std_ratio": e2x_std_ratio, "kw_p": e2x_kw_p, "lev_p": e2x_lev_p},
        "x2e": {"stats": x2e_stat, "std_ratio": x2e_std_ratio, "kw_p": x2e_kw_p, "lev_p": x2e_lev_p},
    }

# --- 모든 쌍의 Δ 통계(수치 풀세트) CSV로 내보내기 ---
def export_pairwise_delta_stats(final_df: pd.DataFrame):
    import numpy as np, pandas as pd
    from pathlib import Path

    PLATE_COL, PASS_COL = "FM_날판번호", "FM_PASS NO N"
    TARGET_COL, CUR_WARP = "warping_index_target", "warping_index_current_pass"

    df = final_df.copy()
    # 쌍/전이/델타 생성
    df["pair"] = df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["transition"] = np.where(df[PASS_COL] % 2 == 1, "E2X", "X2E")

    g = df.groupby(["transition", "pair"])["delta"]

    # ★ 명명된 집계로 'n' 컬럼을 확정 생성
    stats = g.agg(
        n="size",
        mean="mean",
        std="std",
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    ).reset_index()

    # SE/95% CI 및 품질 플래그
    stats["se"] = stats["std"] / np.sqrt(stats["n"])
    # n<=1이면 CI 계산 불가 → NaN
    stats.loc[stats["n"] <= 1, ["se"]] = np.nan
    stats["ci95_lo"] = stats["mean"] - 1.96 * stats["se"]
    stats["ci95_hi"] = stats["mean"] + 1.96 * stats["se"]

    # 샘플 수 적음 경고 플래그
    stats["n_flag"] = np.where(stats["n"] < 30, "small-n(<30)", "ok")

    out = Path("reports") / "pair_delta_stats.csv"
    stats.to_csv(out, index=False)
    print(f"✓ 모든 쌍 Δ 통계 저장: {out}")
    return stats

# ------------------------ 9) 요약 저장 ---------------------------
def write_last_metrics(experiment_id: str, summary_path: Path,
                       e2x_res: dict, x2e_res: dict) -> Path:
    avg = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))
    out_csv = METRICS_DIR / "last_run_metrics.csv"
    out_json = METRICS_DIR / "last_run_metrics.json"
    row = {
        "experiment_id": experiment_id,
        "summary_path": str(summary_path),
        "run_name": RUN_NAME,
        "anchor_tag": ANCHOR_TAG,
        "E2X_algo": e2x_res.get("algorithm"),
        "X2E_algo": x2e_res.get("algorithm"),
        "USE_DT": int(USE_DELTA_TARGET),
        "USE_TE": int(USE_PAIR_TE),
        "USE_W": int(USE_HETERO_W),
        "USE_MONO": int(USE_MONO_CUR),
        "USE_RM": int(USE_RM_TRAINONLY),
        "E2X_test_r2": float(e2x_res["test_r2"]),
        "X2E_test_r2": float(x2e_res["test_r2"]),
        "avg_test_r2": avg
    }
    # CSV (머리글 유지; 없으면 생성)
    if not out_csv.exists():
        out_csv.write_text(",".join(row.keys()) + "\n", encoding="utf-8")
    with out_csv.open("a", encoding="utf-8") as f:
        f.write(",".join([str(row[k]) for k in row.keys()]) + "\n")
    out_json.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_csv

def write_summary(summary_data, summary_path: Path, timestamp: datetime, experiment_id: str, best_record: dict):
    lines = []
    lines.append("Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
    lines.append("=" * 72)
    lines.append(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} 작성")
    lines.append(f"실험 ID: {experiment_id}\n")

    lines.append("[데이터 산출물]")
    for it in summary_data["data_artifacts"]:
        lines.append(f"- {it}")

    lines.append("\n[모델 성능 - R2]")
    for key in ["E2X","X2E"]:
        res = summary_data["models"].get(key, {})
        lines.append(f"* {key} | Algo: {res.get('algorithm','N/A')} | "
                     f"Train R2: {res.get('train_r2',np.nan):.4f} | Test R2: {res.get('test_r2',np.nan):.4f} "
                     f"(RMSE {res.get('train_rmse',np.nan):.4f}/{res.get('test_rmse',np.nan):.4f})")
    lines.append(f"\n[평균 Test R2] {float(summary_data.get('average_test_r2', float('nan'))):.4f}")
    rf_r2 = summary_data.get("rollforward_r2", None)
    if rf_r2 is not None:
        lines.append(f"[롤포워드 R2(전체 판, t→t+1)] {rf_r2:.4f}")

    # Δ-분석 요약
    v = summary_data.get("variability", {})
    for tag in ["e2x","x2e"]:
        vv = v.get(tag, {})
        lines.append(f"\n[Δ-분석 {tag.upper()}] std_max/min={vv.get('std_ratio', np.nan):.3f} | "
                     f"Kruskal p={vv.get('kw_p', np.nan)} | Levene p={vv.get('lev_p', np.nan)}")
        # 상위 3개 |mean| 큰 패스쌍
        st = vv.get("stats")
        if isinstance(st, pd.DataFrame) and "mean" in st.columns:
            top3 = st.assign(absmean=st["mean"].abs()).nlargest(3, "absmean")[["pair","count","mean","std","median"]]
            lines.append("  - |Δ| 상위 3쌍:")
            for _, r in top3.iterrows():
                lines.append(f"    · {r['pair']}: n={int(r['count'])}, mean={r['mean']:.3f}, std={r['std']:.3f}, median={r['median']:.3f}")

    best_id = best_record.get("experiment_id")
    if best_id:
        best_avg = float(best_record.get("average_test_r2", float("nan")))
        lines.append(f"\n[베스트] 실험 {best_id} 평균 Test R2 = {best_avg:.4f}")
        if best_id == experiment_id:
            lines.append("  → 이번 실험이 현재 최고 기록")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 요약 저장: {summary_path}")

def dump_last_run_metrics(average_test_r2: float, summary_path: Path | str):
    """
    쉘 스크립트가 집계하는 metrics/last_run_metrics.csv를 항상 생성.
    """
    REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    met = REPORTS_METRICS_DIR / "last_run_metrics.csv"
    with open(met, "w", encoding="utf-8") as f:
        f.write("average_test_r2,summary_path\n")
        f.write(f"{float(average_test_r2):.4f},{str(summary_path)}\n")

# ----------------------------- main ------------------------------
def main():
    _ensure_dirs()
    success = False
    experiment_id = _next_experiment_id()
    timestamp = datetime.now()
    summary_path = REPORTS_SUMMARY_DIR / f"{experiment_id}_test_no_op_summary.txt"
    archived_script_path = REPORTS_CODE_DIR / f"{experiment_id}_experiment.py"

    try:
        # 1) 병합/타깃(t+1)
        final_df = load_and_merge_data_tplus1()

        # (A) Δ-분석: 패스쌍별 변화량이 제각각인지 확인
        variability = analyze_transition_variability(final_df)

        # 2) 전이 세트 분리
        e2x_df, x2e_df = split_cross_transitions(final_df)

        # 3) 피처 엔지니어링
        e2x_fe = feature_engineering(e2x_df, "E2X", PROCESSED_DIR / "e2x_featured_tplus1.csv")
        x2e_fe = feature_engineering(x2e_df, "X2E", PROCESSED_DIR / "x2e_featured_tplus1.csv")

        # 4) 이상치 제거
        e2x_clean = remove_outliers(PROCESSED_DIR / "e2x_featured_tplus1.csv", PROCESSED_DIR / "e2x_cleaned_tplus1.csv", "E2X")
        x2e_clean = remove_outliers(PROCESSED_DIR / "x2e_featured_tplus1.csv", PROCESSED_DIR / "x2e_cleaned_tplus1.csv", "X2E")

        # 5) 알고리즘 비교
        # [알고리즘 결정] E2X는 XGB 고정, X2E는 env(X2E_ALGO) 또는 CV
        e2x_algo = "XGBoost"   # 요청대로 고정
        x2e_forced = _env_get("X2E_ALGO", "").strip()
        if x2e_forced in ("XGBoost", "RandomForest"):
            x2e_algo = x2e_forced
        else:
            # 필요 시 비교(보고용); 하지만 강제값 있으면 그 값을 우선합니다.
            _, _ = compare_algorithms(e2x_clean, "E2X")  # 로깅 용도
            x2e_algo, _ = compare_algorithms(x2e_clean, "X2E")

        print(f"[ALG] E2X={e2x_algo} | X2E={x2e_algo}")
        print(f"[FLAGS] DT={_env_on('USE_DELTA_TARGET')} TE={_env_on('USE_PAIR_TE')} "
            f"W={_env_on('USE_HETERO_W')} MONO={_env_on('USE_MONO_CUR')} RM={_env_on('USE_RM_TRAINONLY')}")

        # [학습]
        e2x_res, e2x_test_plates = train_model(e2x_clean, "E2X", algorithm=e2x_algo)
        x2e_res, x2e_test_plates = train_model(x2e_clean, "X2E", algorithm=x2e_algo)

        average_test_r2 = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))

        # 7) 번갈아 예측 및 R2
        all_fe = feature_engineering(final_df, "All", PROCESSED_DIR / "all_featured_tplus1.csv")
        e2x_bundle = _load_bundle("e2x")
        x2e_bundle = _load_bundle("x2e")
        rf_pred = rollforward_predict_all_plates(all_fe, e2x_bundle, x2e_bundle)
        rf_out = PROCESSED_DIR / "rollforward_predictions_tplus1_cross.csv"
        rf_pred.to_csv(rf_out, index=False)
        print(f"✓ 번갈아 예측 결과 저장: {rf_out}")

        rf_mask = rf_pred["gt_next"].notna()
        rollforward_r2 = float(r2_score(rf_pred.loc[rf_mask, "gt_next"], rf_pred.loc[rf_mask, "pred"])) if rf_mask.any() else np.nan

        # (1) 모든 쌍 Δ 통계 CSV
        _ = export_pairwise_delta_stats(final_df)

        # (2) 교집합 Test 판만으로 엄격 rollforward R²
        all_fe = feature_engineering(final_df, "All", PROCESSED_DIR / "all_featured_tplus1.csv")
        strict_rollforward_r2 = rollforward_r2_on_test_only(all_fe)
        print(f"[Strict Test‑only Rollforward R²] {strict_rollforward_r2:.4f}")

        # 요약 준비(이번 run 포함 가상의 베스트)
        temp_log = pd.read_csv(EXPERIMENT_LOG) if EXPERIMENT_LOG.exists() else pd.DataFrame(columns=["experiment_id","average_test_r2"])
        temp_row = {"experiment_id": experiment_id, "average_test_r2": average_test_r2}
        temp_log = pd.concat([temp_log, pd.DataFrame([temp_row])], ignore_index=True)
        idx = pd.to_numeric(temp_log["average_test_r2"], errors="coerce").idxmax() if len(temp_log) else None
        best_record = temp_log.loc[idx].to_dict() if idx is not None else {}

        summary = {
            "data_artifacts": [
                f"Final merged(t+1): {PROCESSED_DIR / 'final_merged_data_regression_tplus1.csv'}",
                f"E2X raw/clean: {PROCESSED_DIR / 'e2x_raw_tplus1.csv'}, {PROCESSED_DIR / 'e2x_cleaned_tplus1.csv'}",
                f"X2E raw/clean: {PROCESSED_DIR / 'x2e_raw_tplus1.csv'}, {PROCESSED_DIR / 'x2e_cleaned_tplus1.csv'}",
                f"All featured: {PROCESSED_DIR / 'all_featured_tplus1.csv'}",
                f"Rollforward preds: {rf_out}",
                f"Archived script: {archived_script_path}",
            ],
            "models": {"E2X": e2x_res, "X2E": x2e_res},
            "average_test_r2": average_test_r2,
            "rollforward_r2": rollforward_r2,
            "variability": variability,
        }
        write_summary(summary, summary_path, timestamp, experiment_id, best_record)
        
        # ★ 메트릭 CSV 강제 생성 (쉘이 읽어감)
        dump_last_run_metrics(average_test_r2, summary_path)

        # 성공 결과만 로그/코드 스냅샷 기록
        log_row = {
            "experiment_id": experiment_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "e2x_algorithm": e2x_res["algorithm"], "x2e_algorithm": x2e_res["algorithm"],
            "e2x_test_r2": e2x_res["test_r2"], "x2e_test_r2": x2e_res["test_r2"],
            "average_test_r2": average_test_r2,
            "e2x_train_r2": e2x_res["train_r2"], "x2e_train_r2": x2e_res["train_r2"],
            "rollforward_r2": rollforward_r2,
            "e2x_kw_p": variability["e2x"]["kw_p"], "x2e_kw_p": variability["x2e"]["kw_p"],
            "e2x_lev_p": variability["e2x"]["lev_p"], "x2e_lev_p": variability["x2e"]["lev_p"],
        }
        _append_success_log(log_row)
        shutil.copy2(SCRIPT_PATH, archived_script_path)
        print(f"✓ 코드 아카이브 저장: {archived_script_path}")

        # Top 5 출력
        log_df = pd.read_csv(EXPERIMENT_LOG)
        top = log_df.sort_values("average_test_r2", ascending=False).head(5)
        print("\n[상위 5 실험] 평균 Test R2 내림차순")
        print(top[["experiment_id","average_test_r2","e2x_test_r2","x2e_test_r2","rollforward_r2"]].to_string(index=False))

        success = True

        MET_DIR = REPORTS_DIR / "metrics"
        MET_DIR.mkdir(parents=True, exist_ok=True)
        last_metrics = pd.DataFrame([{
            "summary_path": str(summary_path),
            "avg_test_r2":  average_test_r2
        }])
        last_metrics.to_csv(MET_DIR / "last_run_metrics.csv", index=False)
        print("✓ 메트릭 저장: reports/metrics/last_run_metrics.csv")

    except Exception as e:
        tb = traceback.format_exc()
        print("\n[실험 실패] 예외가 발생했습니다. 로그/코드 스냅샷은 저장되지 않습니다.")
        print(f"Exception: {type(e).__name__}: {e}")
        print(tb)
        err_path = REPORTS_DIR / "last_error_traceback.txt"
        err_path.write_text(tb, encoding="utf-8")
        print(f"↳ 전체 traceback을 {err_path} 에 저장했습니다.")

    if not success:
        pass

if __name__ == "__main__":
    main()
