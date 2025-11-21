# 032_experiment.py 에다가 추가 기술들을 더합니다.

import re
import shutil
import warnings, os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import traceback
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from dataclasses import dataclass

from xgboost import XGBRegressor

# (선택) SciPy가 있으면 검정 수행
try:
    from scipy.stats import kruskal, levene, spearmanr
except Exception:
    kruskal = levene = spearmanr = None


warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- 경로/상수 --------------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
REPORTS_SUMMARY_DIR = REPORTS_DIR / "summaries"
REPORTS_CODE_DIR = REPORTS_DIR / "scripts"
EXPERIMENT_LOG = REPORTS_DIR / "experiments_log_tplus1_cross.csv"
MODELS_DIR = Path("models")
IMAGES_DIR = Path("images")
SCRIPT_PATH = Path(__file__).resolve()

TARGET_COL = "warping_index_target"   # t+1 타깃
PASS_COL   = "FM_PASS NO N"
PLATE_COL  = "FM_날판번호"
MONTH_COL  = "FM_압연월"
CUR_WARP   = "warping_index_current_pass"

EXCLUDE_FROM_X = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}

# --------------------------- 유틸 -----------------------------
def _env_on(k, default="0"):
    return str(os.environ.get(k, default)).strip().lower() in {"1","true","yes","y"}

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

class SimpleKBest:
    """
    상관 기반 K-best(수치 안정). 모듈 최상단에 정의하여 joblib으로 안전 저장.
    .fit(X_df, y) 후 .transform(X_df) -> np.ndarray
    """
    def __init__(self, k: int, mandatory: list[str] | None = None):
        self.k = int(max(1, k))
        self.mandatory = list(mandatory or [])
        self.selected_cols_: list[str] = []

    def fit(self, X_df: pd.DataFrame, y: np.ndarray):
        X = X_df.select_dtypes(include=[np.number]).copy()
        cols_all = X.columns.tolist()

        # 사전 정리: 비유한/상수/NaN-only 제거
        finite_mask = np.isfinite(X.values).all(axis=0)
        var = X.var(numeric_only=True).reindex(cols_all).fillna(0.0).values
        keep_mask = finite_mask & (var > 1e-12)
        cols = [c for c, m in zip(cols_all, keep_mask) if m]

        if not cols:
            # 안전장치: 모든 컬럼이 탈락하면 그대로 종료
            self.selected_cols_ = []
            return self

        Xc = X[cols].fillna(X[cols].median(numeric_only=True))
        y0 = y - y.mean()
        ys = y0.std()
        ys = ys if ys >= 1e-12 else 1e-12

        xv = Xc.values
        xm = xv.mean(axis=0)
        xs = xv.std(axis=0)
        xs[xs < 1e-12] = 1e-12

        xr = (xv - xm) / xs
        yr = y0 / ys
        r = (xr.T @ yr) / (len(y) - 1)
        r2 = np.clip(r**2, 0.0, 1.0 - 1e-12)
        F = (r2 / (1.0 - r2)) * (len(y) - 2)

        mand = [c for c in self.mandatory if c in cols]
        others = [c for c in cols if c not in mand]
        k_remain = max(1, min(self.k - len(mand), len(others)))

        # others의 F-score를 정렬
        idx_map = {c: i for i, c in enumerate(cols)}
        order = np.argsort([F[idx_map[c]] for c in others])[::-1] if others else []
        self.selected_cols_ = mand + [others[i] for i in order[:k_remain]]
        return self

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        if not self.selected_cols_:
            # 아무 것도 없으면 빈 배열 반환(호출부에서 대비)
            return np.empty((len(X_df), 0), dtype=float)
        Z = X_df.copy()
        for c in self.selected_cols_:
            if c not in Z.columns:
                Z[c] = np.nan
        return Z[self.selected_cols_].values

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

def _build_model(name: str, seed: int):
    if name == "XGBoost":
        return XGBRegressor(
            random_state=seed, tree_method="hist", n_jobs=0,
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=1.0,
            objective="reg:squarederror", eval_metric="rmse"
        )
    elif name == "RandomForest":
        return RandomForestRegressor(
            random_state=seed, n_estimators=600, n_jobs=-1,
            max_depth=20, min_samples_leaf=5, max_features="sqrt"
        )
    else:
        raise ValueError(f"Unsupported algo: {name}")

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
    print(f"\n[Step 4] 이상치 제거 - {dataset_name}")
    df = pd.read_csv(input_path)

    # 수치형 피처만 사용
    features_all = [c for c in df.columns if c not in EXCLUDE_FROM_X]
    num_features = df[features_all].select_dtypes(include=[np.number]).columns.tolist()
    X = df[num_features].fillna(df[num_features].median(numeric_only=True))
    y = df[TARGET_COL]

    iso = IsolationForest(contamination=0.05, random_state=42)
    mask = iso.fit_predict(X) == 1

    Xs = StandardScaler().fit_transform(X)
    if len(Xs) > 2:
        nn = NearestNeighbors(n_neighbors=min(20, len(Xs)-1))
        nn.fit(Xs)
        _, idx = nn.kneighbors(Xs)
        neighbor_mean = y.values[idx[:,1:]].mean(axis=1)
        inconsistency = np.abs(y.values - neighbor_mean)
        thr = np.percentile(inconsistency, 95)
        mask &= inconsistency <= thr

    cleaned = df.loc[mask].reset_index(drop=True)
    cleaned.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path} (최종 {cleaned.shape[0]}개)")
    return cleaned

# ------------------- 5) 알고리즘 비교 (동일 규칙) -------------------
def compare_algorithms(df: pd.DataFrame, dataset_name: str):
    print("\n" + "="*80)
    print(f"[Step 5] 알고리즘 비교 - {dataset_name}")
    print("="*80)

    X_all = df.drop(columns=[c for c in EXCLUDE_FROM_X if c in df.columns], errors="ignore")
    X = X_all.select_dtypes(include=[np.number]).fillna(X_all.median(numeric_only=True))
    y = df[TARGET_COL]

    algos = {
        "XGBoost": XGBRegressor(random_state=42, n_estimators=400, learning_rate=0.05,
                                max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist"),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=500, n_jobs=-1),
    }

    best_name, best_score = None, -np.inf
    for name, model in algos.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        print(f"  {name:12s}: R2 = {scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_score, best_name = scores.mean(), name
    print(f"\n  → 최적 알고리즘: {best_name} (CV R2 = {best_score:.4f})")
    return best_name, best_score

# ------------ 6) 학습/튜닝 + 전처리 메타 저장(중앙값/열순서) -----------
# ---------- (A) OOF 쌍-Δ Target Encoding(EB 축소) ----------
def add_pair_delta_te_feature(train_df, test_df, PASS_COL, PLATE_COL, CUR_WARP, TARGET_COL, n0=50):
    tr = train_df.copy(); te = test_df.copy()
    tr["pair"] = tr[PASS_COL].astype(int).astype(str) + "→" + (tr[PASS_COL]+1).astype(int).astype(str)
    te["pair"] = te[PASS_COL].astype(int).astype(str) + "→" + (te[PASS_COL]+1).astype(int).astype(str)
    tr["delta"] = tr[TARGET_COL] - tr[CUR_WARP]

    gkf = GroupKFold(n_splits=3)
    groups = tr[PLATE_COL].astype(str).values
    oof = np.zeros(len(tr))
    for tr_i, va_i in gkf.split(tr, tr["delta"].values, groups):
        sub = tr.iloc[tr_i]
        gmean = sub["delta"].mean()
        stat = sub.groupby("pair")["delta"].agg(n="size", mean="mean").reset_index()
        w = stat["n"]/(stat["n"]+n0)
        stat["te"] = w*stat["mean"] + (1-w)*gmean
        m = dict(zip(stat["pair"], stat["te"]))
        oof[va_i] = tr.iloc[va_i]["pair"].map(m).fillna(gmean).values

    # full-train mapping → test
    gmean = tr["delta"].mean()
    stat = tr.groupby("pair")["delta"].agg(n="size", mean="mean").reset_index()
    w = stat["n"]/(stat["n"]+n0)
    stat["te"] = w*stat["mean"] + (1-w)*gmean
    m_full = dict(zip(stat["pair"], stat["te"]))
    te_feat = te["pair"].map(m_full).fillna(gmean).values

    tr = tr.drop(columns=["pair","delta"])
    te = te.drop(columns=["pair"])
    return tr.assign(pair_delta_te=oof), te.assign(pair_delta_te=te_feat)

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
                test_size: float = 0.2,
                select_k: int = 120,
                bag_k: int = 3):
    """
    003 플로우 유지:
      - plate-group 분할(앵커 저장/재사용)
      - train-only 중앙값/분위 경계 → train 필터, test는 clip만
      - SimpleKBest(모듈 최상단 정의)로 K-best
      - XGB/RandomForest 지원(LightGBM 제거)
      - 배깅 평균(BaggedRegressor 저장) → roll-forward에서 predict 보장
    저장 산출물(003 호환):
      models/{prefix}_regressor_tplus1.pkl
      models/{prefix}_selector_tplus1.pkl
      models/{prefix}_feature_cols_tplus1.pkl
      models/{prefix}_feature_medians_tplus1.pkl
      models/{prefix}_test_plates_tplus1.pkl
    """
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import r2_score, mean_squared_error

    # ---- 상수/경로 ----
    prefix = "e2x" if dataset_name.upper() == "E2X" else "x2e"

    # E2X/X2E 패스 필터(홀/짝)
    data = df.copy()
    if prefix == "e2x":
        data = data[data[PASS_COL] % 2 == 1].copy()
    else:
        data = data[data[PASS_COL] % 2 == 0].copy()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    anchor_path = MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl"

    # ---- 전체 X/y/g ----
    X_all = data.drop(columns=[c for c in EXCLUDE_FROM_X if c in data.columns], errors="ignore") \
               .select_dtypes(include=[np.number]).copy()
    y_all = data[TARGET_COL].values
    g_all = data[PLATE_COL].astype(str).values

    # ---- plate-group 분할(앵커 재사용) ----
    if anchor_path.exists():
        test_plates = set(joblib.load(anchor_path))
        tr_mask = ~data[PLATE_COL].astype(str).isin(test_plates)
        te_mask =  data[PLATE_COL].astype(str).isin(test_plates)
        tr_idx = np.where(tr_mask.values)[0]
        te_idx = np.where(te_mask.values)[0]
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(X_all, y_all, groups=g_all))
        test_plates = sorted(data.iloc[te_idx][PLATE_COL].astype(str).unique())
        joblib.dump(test_plates, anchor_path)

    # ---- train/test 추출 & 결측/중앙값 ----
    X_tr_raw = X_all.iloc[tr_idx].copy()
    X_te_raw = X_all.iloc[te_idx].copy()
    y_tr_raw = y_all[tr_idx].copy()
    y_te = y_all[te_idx].copy()

    med = X_tr_raw.median(numeric_only=True).to_dict()
    X_tr_raw = X_tr_raw.fillna(med)
    X_te_raw = X_te_raw.reindex(columns=X_tr_raw.columns, fill_value=np.nan).fillna(med)

    # ---- train-only 분위 경계로 필터, test는 clip만 ----
    q_low, q_high = 0.01, 0.99
    # y 필터 (극단 라벨 제거 — 과적합/불안정 완화)
    y_lo, y_hi = np.quantile(y_tr_raw, [q_low, q_high])
    keep = (y_tr_raw >= y_lo) & (y_tr_raw <= y_hi)

    # X 필터
    bounds = {}
    for c in X_tr_raw.columns:
        lo, hi = np.quantile(X_tr_raw[c].values, [q_low, q_high])
        bounds[c] = (float(lo), float(hi))
        keep &= (X_tr_raw[c].values >= lo) & (X_tr_raw[c].values <= hi)

    X_tr = X_tr_raw.loc[keep].reset_index(drop=True)
    y_tr = y_tr_raw[keep]

    # test clip
    X_te = X_te_raw.copy()
    for c, (lo, hi) in bounds.items():
        if c in X_te.columns:
            X_te[c] = X_te[c].clip(lower=lo, upper=hi)

    # ---- K-best (CUR_WARP 강제 포함) ----
    mandatory = [CUR_WARP] if CUR_WARP in X_tr.columns else []
    selector = SimpleKBest(k=min(select_k, X_tr.shape[1]), mandatory=mandatory).fit(X_tr, y_tr)
    sel_cols = selector.selected_cols_
    Xtr = selector.transform(X_tr)
    Xte = selector.transform(X_te)

    # 빈 특성 예외 방지
    if Xtr.shape[1] == 0:
        # 최소한 CUR_WARP만이라도 쓰도록 리カ버리
        if CUR_WARP in X_tr.columns:
            sel_cols = [CUR_WARP]
            Xtr = X_tr[[CUR_WARP]].values
            Xte = X_te[[CUR_WARP]].values
        else:
            # 정말 없으면 상수 0
            sel_cols = []
            Xtr = np.zeros((len(X_tr), 1), dtype=float)
            Xte = np.zeros((len(X_te), 1), dtype=float)

    # ---- 내부 홀드아웃(plate-group) 정의용 그룹 (필터 후) ----
    grp_tr_full = data.iloc[tr_idx][keep].loc[:, PLATE_COL].astype(str).values

    # ---- 내부 fit 함수(배깅 1회) ----
    def _fit_predict_one(run_seed: int):
        mdl = _build_model(algorithm, run_seed)
        # group holdout for early stop (XGB only)
        from sklearn.model_selection import GroupShuffleSplit
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=run_seed)
        tr_i, va_i = next(gss_inner.split(Xtr, y_tr, groups=grp_tr_full))
        Xfit, yfit = Xtr[tr_i], y_tr[tr_i]
        Xval, yval = Xtr[va_i], y_tr[va_i]

        if isinstance(mdl, XGBRegressor):
            xgb_fit_safe(mdl, Xfit, yfit, eval_set=[(Xval, yval)], sample_weight=None, verbose=False)
        else:
            mdl.fit(Xfit, yfit)

        p_te = mdl.predict(Xte)
        p_tr = mdl.predict(Xtr)
        return p_tr, p_te, mdl

    # ---- 배깅 학습 ----
    models = []
    preds_tr = []
    preds_te = []
    for b in range(int(max(1, bag_k))):
        p_tr, p_te, mdl = _fit_predict_one(seed + 17 * b)
        models.append(mdl)
        preds_tr.append(p_tr)
        preds_te.append(p_te)

    y_tr_hat = np.mean(np.vstack(preds_tr), axis=0)
    y_te_hat = np.mean(np.vstack(preds_te), axis=0)

    bagged = BaggedRegressor(models=models)

    res = {
        "train_r2":  float(r2_score(y_tr, y_tr_hat)),
        "test_r2":   float(r2_score(y_te, y_te_hat)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_tr, y_tr_hat))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te, y_te_hat))),
        "train_samples": int(len(y_tr)), "test_samples": int(len(y_te)),
        "best_params": getattr(models[0], "get_params", lambda: {})() if models else {},
    }

    # ---- 저장 (003 호환) ----
    joblib.dump(bagged,               MODELS_DIR / f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(selector,             MODELS_DIR / f"{prefix}_selector_tplus1.pkl")
    joblib.dump(sel_cols,             MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl")
    joblib.dump(med,                  MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl")
    joblib.dump(sorted(list(test_plates)), MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl")

    return res, set(test_plates)

# ----------------- 7) 번갈아 예측(roll-forward) -------------------
def _load_bundle(prefix: str):
    """
    다양한 저장 포맷(모델이 tuple/dict/list로 저장된 경우, selector가 객체/리스트인 경우)을
    모두 흡수하는 안전 로더.
    반환: {"model": estimator, "selector": selector_like, "cols": list[str], "medians": dict}
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import joblib

    def _unwrap_model(obj):
        # estimator 바로 주어진 경우
        if hasattr(obj, "predict"):
            return obj
        # (model, ...) tuple/list인 경우 첫 번째로 predict 있는 객체 선택
        if isinstance(obj, (list, tuple)):
            for o in obj:
                if hasattr(o, "predict"):
                    return o
        # {"model": estimator, ...} dict로 저장된 경우
        if isinstance(obj, dict) and "model" in obj and hasattr(obj["model"], "predict"):
            return obj["model"]
        raise TypeError("저장된 모델 객체에서 predict를 찾지 못했습니다. 모델 파일을 재생성해주세요.")

    def _coerce_selector(sel, cols_hint):
        """
        selector가
          - 객체이고 transform 지원 → 그대로 사용
          - 리스트/튜플/ndarray(= 컬럼 리스트) → 간이 selector로 래핑
          - None → cols_hint로 간이 selector 생성
        반환 객체는 transform(DataFrame) -> ndarray 를 보장
        """
        class _ListSelector:
            def __init__(self, cols):
                self.cols = list(cols or [])
            def transform(self, X_df):
                X_df = X_df.reindex(columns=self.cols, fill_value=np.nan)
                return X_df.values

        if sel is None:
            return _ListSelector(cols_hint or [])
        if hasattr(sel, "transform"):
            return sel
        if isinstance(sel, (list, tuple, np.ndarray)):
            return _ListSelector(sel)
        # 혹시 dict 등 특수 포맷이면 cols_hint로 대체
        return _ListSelector(cols_hint or [])

    # --- 로드 ---
    model_obj    = joblib.load(MODELS_DIR / f"{prefix}_regressor_tplus1.pkl")
    selector_obj = None
    try:
        selector_obj = joblib.load(MODELS_DIR / f"{prefix}_selector_tplus1.pkl")
    except Exception:
        selector_obj = None

    try:
        cols = joblib.load(MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl")
        if isinstance(cols, np.ndarray): cols = cols.tolist()
    except Exception:
        cols = None

    try:
        medians = joblib.load(MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl")
    except Exception:
        medians = {}

    # --- 언래핑/보정 ---
    model = _unwrap_model(model_obj)
    selector = _coerce_selector(selector_obj, cols)

    # cols가 비어있으면 selector 내부에서 유추
    if not cols:
        if hasattr(selector, "selected_features_"):
            cols = list(selector.selected_features_)
        elif hasattr(selector, "selected_cols_"):
            cols = list(selector.selected_cols_)
        elif hasattr(selector, "get_feature_names_out"):
            try:
                cols = list(selector.get_feature_names_out())
            except Exception:
                cols = list(getattr(selector, "feature_names_in_", []))
        elif hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
        else:
            cols = list(medians.keys())

    # medians를 dict로 보장
    if not isinstance(medians, dict):
        try:
            medians = dict(medians)
        except Exception:
            medians = {}

    return {"model": model, "selector": selector, "cols": list(cols or []), "medians": medians}

def _predict_next_from_row(row: pd.Series, bundle: dict) -> float:
    """
    - bundle["model"]가 tuple/dict였던 예전 산출물을 자동 언래핑
    - bundle["selector"]가 객체 또는 리스트인 경우 모두 호환
    - 누락 컬럼은 NaN 채운 뒤 medians로 대치
    """
    import numpy as np
    import pandas as pd

    # 1) 입력 행을 훈련 시 사용한 컬럼 순서로 재배열
    desired_cols = list(bundle["cols"])
    X_row = pd.DataFrame([row.reindex(desired_cols)], columns=desired_cols)
    # 결측을 훈련 중앙값으로 보정
    if isinstance(bundle.get("medians", {}), dict) and bundle["medians"]:
        X_row = X_row.fillna(bundle["medians"])
    else:
        X_row = X_row.fillna(0.0)

    # 2) 선택자 적용(객체/리스트 모두 처리)
    selector = bundle.get("selector", None)
    if selector is not None and hasattr(selector, "transform"):
        X_in = selector.transform(X_row)
        # 일부 selector가 DataFrame을 반환하는 경우
        if hasattr(X_in, "values"):
            X_in = X_in.values
    else:
        X_in = X_row.values

    # 3) 모델 언래핑 & 예측
    model = bundle.get("model")
    if not hasattr(model, "predict"):
        # 로더에서 언래핑했지만 혹시 몰라 2차 방어
        if isinstance(model, (list, tuple)):
            model = next((m for m in model if hasattr(m, "predict")), None)
        elif isinstance(model, dict):
            model = model.get("model", model)
    if not hasattr(model, "predict"):
        raise TypeError("bundle['model']에서 예측기를 찾지 못했습니다. 모델 파일을 재생성하세요.")

    yhat = model.predict(X_in)
    return float(np.asarray(yhat).ravel()[0])


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
    outs = []
    for plate in sorted(all_featured_df[PLATE_COL].dropna().unique()):
        sub = all_featured_df[all_featured_df[PLATE_COL] == plate].sort_values(PASS_COL)
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index:  # t+1 없으면 skip
                continue
            row_t = idx.loc[p]
            feat_row = row_t.drop(labels=[TARGET_COL], errors="ignore")
            bundle = e2x_bundle if (p % 2 == 1) else x2e_bundle
            pred = _predict_next_from_row(feat_row, bundle)
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

# ----------------------------- main ------------------------------
def main():
    _ensure_dirs()
    success = False
    experiment_id = _next_experiment_id()
    timestamp = datetime.now()
    summary_path = REPORTS_SUMMARY_DIR / f"{experiment_id}_pass_t_plus_1_cross_summary.txt"
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
        e2x_algo, _ = compare_algorithms(e2x_clean, "E2X")
        x2e_algo, _ = compare_algorithms(x2e_clean, "X2E")

        # 6) 학습/튜닝
        e2x_ret = train_model(e2x_clean, "E2X", algorithm=e2x_algo)
        x2e_ret = train_model(x2e_clean, "X2E", algorithm=x2e_algo)

        # train_model 이 (results, test_plates) 튜플을 돌려주는 버전과
        # results(dict)만 돌려주는 예전 버전 모두 호환되게 처리
        if isinstance(e2x_ret, tuple):
            e2x_res, e2x_test_plates = e2x_ret
        else:
            e2x_res, e2x_test_plates = e2x_ret, None

        if isinstance(x2e_ret, tuple):
            x2e_res, x2e_test_plates = x2e_ret
        else:
            x2e_res, x2e_test_plates = x2e_ret, None

        # 이후 딕셔너리에 안전하게 메타 추가
        e2x_res["algorithm"] = e2x_algo
        x2e_res["algorithm"] = x2e_algo

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
