"""
119_test_no_op.py — rollforward 제거 + SVM 고정 + R² 전략 스위치 유지 (누수 가드 엄격)

핵심:
- 알고리즘 SVM 고정(SVR + 표준화). E2X/X2E 모두 SVM.
- GroupShuffleSplit(plate 단위)로 분할 → 모든 통계/대치/특수특성은 Train 기준으로만 계산 후 Test에 적용
- USE_* 스위치:
  * USE_DELTA_TARGET=1 : y=(t+1)-cur 로 학습, 추정값은 다시 cur 더해 복원해 평가
  * USE_PAIR_TE=1      : pair( p→p+1 ) OOF Δ‑mean 인코딩(Train OOF / Test는 Train 평균 주입)
  * USE_HETERO_W=1     : pair별 Δ 분산 기반 inverse-variance 가중(sample_weight)으로 SVM 학습
  * USE_RM_TRAINONLY=1 : RM_* 컬럼은 Train 기준으로만 사용(Test는 train median으로 채워 실질 배제)
- SelectKBest(f_regression)로 피처 선택(k는 env로 조절), 표준화(StandardScaler)는 Train 기준
- 라벨/타깃/예측/TE 등 이름에 의한 누수 방지 필터링 포함
- summary_path 규격: reports/summaries/{experiment_id}_test_no_op_summary.txt
- last_run_metrics.csv 열 순서 고정: experiment_id, e2x_test_r2, x2e_test_r2, average_test_r2, summary_path
"""

import os
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import traceback

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- 경로/상수 --------------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
REPORTS_SUMMARY_DIR = REPORTS_DIR / "summaries"
REPORTS_CODE_DIR = REPORTS_DIR / "scripts"
MODELS_DIR = Path("models")  # (저장은 최소화하지만 디렉토리는 유지)
IMAGES_DIR = Path("images")  # (미사용)

TARGET_COL = "warping_index_target"   # t+1 타깃
PASS_COL   = "FM_PASS NO N"
PLATE_COL  = "FM_날판번호"
MONTH_COL  = "FM_압연월"
CUR_WARP   = "warping_index_current_pass"

# 모델 입력에서 절대 배제해야 할 컬럼(타깃/식별자/시계열키 등)
EXCLUDE_FROM_X = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}

# --------------------------- ENV 스위치/하이퍼 -----------------------------
def _env_on(name: str, default: str = "0") -> int:
    v = os.getenv(name, default).strip().lower()
    return 1 if v in ("1", "true", "yes", "y", "on") else 0

RUN_NAME         = os.getenv("RUN_NAME", "pass_t_plus_1_cross")
PERMUTE_TARGET   = _env_on("PERMUTE_TARGET", "0")
LEAK_AUTOCHECK   = _env_on("LEAK_AUTOCHECK", "1")  # 빠른 퍼뮤 확인(기본 켜짐)

USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET", "0")
USE_PAIR_TE      = _env_on("USE_PAIR_TE", "0")
USE_HETERO_W     = _env_on("USE_HETERO_W", "0")
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY", "0")

# === Scaling / Selection / Correlation Pruning ENV ===
from typing import Optional

def _parse_corr_thr(raw: str) -> Optional[float]:
    if raw is None:
        return 0.995  # 기본값
    s = raw.strip().lower()
    # "none", "", "off", "false" -> 사용 안 함
    if s in ("", "none", "off", "false"):
        return None
    return float(s)

SCALE_METHOD = os.getenv("SCALE_METHOD", "robust").strip().lower()  # 'standard' | 'robust' | 'none'
SELECT_K     = int(os.getenv("SELECT_K", "120"))
QCLIP        = int(os.getenv("QCLIP", "0"))  # 1이면 훈련분위수로 clip 적용
QCLIP_LO     = float(os.getenv("QCLIP_LO", "0.01"))
QCLIP_HI     = float(os.getenv("QCLIP_HI", "0.99"))
CORR_THR     = _parse_corr_thr(os.getenv("CORR_THR", "0.995"))

# SVM 하이퍼파라미터
SVR_C     = float(os.getenv("SVR_C", "10.0"))
SVR_EPS   = float(os.getenv("SVR_EPS", "0.2"))
SVR_GAMMA = os.getenv("SVR_GAMMA", "scale").strip()  # 'scale' | 'auto' | float 문자열도 허용
try:
    # 숫자로 넘겼다면 float 변환 허용 (예: '0.05')
    SVR_GAMMA = float(SVR_GAMMA) if SVR_GAMMA not in ("scale", "auto") else SVR_GAMMA
except Exception:
    SVR_GAMMA = "scale"

# --------------------------- 유틸 -----------------------------
def _ensure_dirs():
    for d in [PROCESSED_DIR, REPORTS_DIR, REPORTS_SUMMARY_DIR, REPORTS_CODE_DIR, MODELS_DIR, REPORTS_DIR/"metrics"]:
        d.mkdir(parents=True, exist_ok=True)

def _next_experiment_id() -> str:
    existing = []
    for f in REPORTS_SUMMARY_DIR.glob("*.txt"):
        m = re.match(r"(\d{3})_", f.name)
        if m:
            try:
                existing.append(int(m.group(1)))
            except:
                pass
    return f"{(max(existing)+1) if existing else 1:03d}"

def _hard_leak_guard(cols: list[str]) -> list[str]:
    """이름에 타깃/예측 스멜이 있는 컬럼 전부 배제"""
    bad_kw = ("target", "label", "oof", "pred", "_hat", "y_", "_y", "te_")
    safe = []
    for c in cols:
        low = c.lower()
        if any(k in low for k in bad_kw):
            continue
        safe.append(c)
    return safe

def _save_last_metrics(experiment_id: str, e2x_test_r2: float, x2e_test_r2: float, avg_test_r2: float, summary_path: Path):
    met_dir = REPORTS_DIR / "metrics"
    met_dir.mkdir(parents=True, exist_ok=True)
    csv = met_dir / "last_run_metrics.csv"
    df = pd.DataFrame([{
        "experiment_id": experiment_id,
        "e2x_test_r2": float(e2x_test_r2),
        "x2e_test_r2": float(x2e_test_r2),
        "average_test_r2": float(avg_test_r2),
        "summary_path": str(summary_path),
    }])
    df = df[["experiment_id","e2x_test_r2","x2e_test_r2","average_test_r2","summary_path"]]  # 열 순서 고정
    df.to_csv(csv, index=False)
    print(f"✓ 메트릭 저장: {csv}")

def _fit_clip_params(X: pd.DataFrame, q: float = 0.01) -> dict:
    params = {}
    for c in X.columns:
        col = X[c].values
        lo = np.nanquantile(col, q)
        hi = np.nanquantile(col, 1.0 - q)
        if not np.isfinite(lo): lo = np.nanmin(col)
        if not np.isfinite(hi): hi = np.nanmax(col)
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = 0.0
        if hi < lo: lo, hi = hi, lo
        params[c] = (lo, hi)
    return params

def _apply_clip(X: pd.DataFrame, clip_params: dict) -> pd.DataFrame:
    Xc = X.copy()
    for c, (lo, hi) in clip_params.items():
        if c in Xc.columns:
            Xc[c] = np.clip(Xc[c].values, lo, hi)
    return Xc

def _fit_scaler(X: pd.DataFrame, kind: str):
    if kind == "standard":
        sc = StandardScaler()
    elif kind == "robust":
        sc = RobustScaler(quantile_range=(25, 75))
    else:
        return None
    sc.fit(X.values)
    return sc

def _apply_scaler(X: pd.DataFrame, sc):
    if sc is None:
        return X.values
    return sc.transform(X.values)

def _corr_prune_cols(X: pd.DataFrame, thr: float = 0.995) -> list:
    if X.shape[1] <= 1: 
        return list(X.columns)
    cmat = X.corr(numeric_only=True).abs()
    upper = np.triu(np.ones(cmat.shape), k=1).astype(bool)
    drop = set()
    cols = list(cmat.columns)
    for i in range(len(cols)):
        if cols[i] in drop: 
            continue
        for j in range(i+1, len(cols)):
            if cols[j] in drop: 
                continue
            if upper[i, j] and cmat.iat[i, j] >= thr:
                drop.add(cols[j])
    keep = [c for c in cols if c not in drop]
    return keep

def _split_by_anchor(df: pd.DataFrame, prefix: str, seed: int, test_size: float):
    """ANCHOR_TAG가 있으면 models/{ANCHOR_TAG}_{prefix}_test_plates.pkl을 사용/생성."""
    groups = df[PLATE_COL].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    anchor = os.getenv("ANCHOR_TAG", "").strip()
    path = MODELS_DIR / (f"{anchor}_{prefix}_test_plates.pkl" if anchor else f"{prefix}_test_plates.pkl")

    if path.exists():
        test_plates = set(joblib.load(path))
        te_mask = df[PLATE_COL].astype(str).isin(test_plates).values
        if te_mask.any() and (~te_mask).any():
            tr_idx = np.where(~te_mask)[0]; te_idx = np.where(te_mask)[0]
            return tr_idx, te_idx, sorted(list(test_plates))
        # 무효하면 새로 생성

    tr_idx, te_idx = next(gss.split(df, df[TARGET_COL].values, groups=groups))
    test_plates = sorted(df.iloc[te_idx][PLATE_COL].astype(str).unique())
    joblib.dump(test_plates, path)
    return tr_idx, te_idx, test_plates
 
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

    # 키 추출
    ground_truth["extracted_plate"] = ground_truth["filename"].str.extract(r"(PB\d+)")
    ground_truth["extracted_pass"]  = ground_truth["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    # RM 통계(전량) — 누수 방지 위해 Test에는 실질 반영 차단(USE_RM_TRAINONLY로 제어)
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

    # 타깃 생성
    merged_fm[TARGET_COL] = merged_fm.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged_fm = merged_fm.rename(columns={"warping_index": CUR_WARP})
    before = len(merged_fm)
    merged_fm = merged_fm.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"Target shift(-1): {before} → {len(merged_fm)} (삭제 {before-len(merged_fm)})")

    # PERMUTE (sanity)
    if PERMUTE_TARGET:
        merged_fm[TARGET_COL] = merged_fm[TARGET_COL].sample(frac=1.0, random_state=42).values
        print("[PERMUTE] Target shuffled for leak sanity check.")

    # RM 집계 결합 + 보조 컬럼 정리
    final_df = pd.merge(
        merged_fm, rm_stats,
        left_on=PLATE_COL, right_on="RM_날판번호", how="left"
    ).drop(columns=["RM_날판번호"], errors="ignore")
    final_df = final_df.drop(columns=["extracted_plate","extracted_pass","filename",
                                      "quality_grade","quality_grade_current_pass","direction"], errors="ignore")

    out_path = PROCESSED_DIR / "final_merged_data_regression_tplus1.csv"
    final_df.to_csv(out_path, index=False)
    print(f"✓ 저장: {out_path}")
    return final_df

# --------- 2) 전이 세트(E→X / X→E) 분리 ----------
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

# ---------------- 3) Feature Engineering (경량; 누수 안전) ---------------
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

    # 단조 제약(MONO)은 SVM에선 실제 거는 게 불가 → 로깅만 함
    if PASS_COL in df.columns:
        df["FM_PASS_squared"] = df[PASS_COL] ** 2

    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path}")
    return df

# -------------------- 4) 이상치 제거 (NO-OP; train-only 처리) -------------------
def remove_outliers(input_path: Path, output_path: Path, dataset_name: str):
    print(f"\n[Step 4] 이상치 제거 - {dataset_name}")
    df = pd.read_csv(input_path)
    # 실제 이상치 처리는 train 내에서만 해야 누수 방지 → 여기서는 NO-OP
    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path} (최종 {df.shape[0]}개)")
    return df

# -------------------- 학습 보조 (OOF TE / 가중치) --------------------
def _make_pair_id(df: pd.DataFrame) -> pd.Series:
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)

def _oof_pair_te_smooth(tr_df: pd.DataFrame, y_tr: np.ndarray, use_delta: bool, alpha: float = 10.0, n_splits=5):
    """plate-group OOF로 pair 평균을 추정하되, 소표본 안정화를 위해 전역평균으로 EB 수축.
       반환: te_tr(OOB), te_map(test용 pair→값 dict), global_mean(default)"""
    gkf = GroupKFold(n_splits=n_splits)
    groups = tr_df[PLATE_COL].astype(str).values
    pair   = _make_pair_id(tr_df).values
    cur    = tr_df[CUR_WARP].values

    target = y_tr.copy()
    if use_delta:
        target = y_tr - cur

    global_mean = float(np.nanmean(target))
    te_tr = np.zeros(len(tr_df), dtype=float)

    for tr_idx, va_idx in gkf.split(tr_df, target, groups):
        # fold‑train에서 pair별 통계
        p_tr = pair[tr_idx]
        y_tr_fold = target[tr_idx]
        df_te = pd.DataFrame({"pair": p_tr, "y": y_tr_fold})
        agg = df_te.groupby("pair")["y"].agg(["mean", "count"])
        # EB 수축
        smoothed = (agg["count"] * agg["mean"] + alpha * global_mean) / (agg["count"] + alpha)
        smap = smoothed.to_dict()
        te_tr[va_idx] = np.fromiter((smap.get(p, global_mean) for p in pair[va_idx]), dtype=float, count=len(va_idx))

    # test 주입용(전체 train에서 다시 적합)
    df_all = pd.DataFrame({"pair": pair, "y": target})
    agg_all = df_all.groupby("pair")["y"].agg(["mean", "count"])
    smoothed_all = (agg_all["count"] * agg_all["mean"] + alpha * global_mean) / (agg_all["count"] + alpha)
    te_map = smoothed_all.to_dict()
    return te_tr, te_map, global_mean


def _pair_weights(tr_df: pd.DataFrame, y_tr: np.ndarray, use_delta: bool):
    pair  = _make_pair_id(tr_df).values
    cur   = tr_df[CUR_WARP].values
    target = y_tr.copy()
    if use_delta:
        target = y_tr - cur  # Δ
    df = pd.DataFrame({"pair": pair, "y": target})
    s = df.groupby("pair")["y"].std().replace(0, np.nan).fillna(df["y"].std())
    w = 1.0 / (s**2 + 1e-6)
    w_map = w.to_dict()
    return np.array([w_map.get(p, 1.0) for p in pair], dtype=float)

def _quantile_clip_train_apply(Xtr: pd.DataFrame, Xte: pd.DataFrame, qlo: float, qhi: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 열별 훈련 분위수 경계 계산 후 train/test 동일 적용
    lo = Xtr.quantile(qlo)
    hi = Xtr.quantile(qhi)
    Xtr_c = Xtr.clip(lower=lo, upper=hi, axis=1)
    Xte_c = Xte.clip(lower=lo, upper=hi, axis=1)
    return Xtr_c, Xte_c

def _corr_prune_train_apply(Xtr: pd.DataFrame, Xte: pd.DataFrame, thr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 훈련셋 상관계수만으로 가지치기 → 테스트에도 동일 열 서브셋 적용
    corr = Xtr.corr().abs()
    upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
    to_drop = set()
    cols = Xtr.columns.tolist()
    for i, ci in enumerate(cols):
        if ci in to_drop:
            continue
        for j, cj in enumerate(cols):
            if j <= i:
                continue
            if upper[i, j] and corr.iloc[i, j] >= thr:
                # 더 "덜 정보적인" 쪽을 버리는 간단한 규칙: 분산이 작은 쪽 삭제
                vi, vj = Xtr[ci].var(ddof=0), Xtr[cj].var(ddof=0)
                drop = cj if vj <= vi else ci
                to_drop.add(drop)
    keep = [c for c in cols if c not in to_drop]
    return Xtr[keep].copy(), Xte[keep].copy()

def _scale_train_apply(Xtr: pd.DataFrame, Xte: pd.DataFrame, method: str):
    if method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        scaler = None
    if scaler is None:
        # 스케일링 OFF
        return Xtr.values, Xte.values, None
    scaler.fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), scaler

# ---------------------------- 학습 본체 (SVM 고정) ----------------------------
def train_model(
    df: pd.DataFrame,
    dataset_name: str,             # "E2X" | "X2E" (로깅 용도)
    seed: int = 42,
    test_size: float = 0.2,
    select_k: int = 120,
    scale_method: str = "robust",  # 'robust' | 'standard' | 'none'
    corr_thr: float | None = 0.995,
    qclip: int = 0,
    qclip_lo: float = 0.01,
    qclip_hi: float = 0.99,
    use_pair_te: int = 0,
    use_hetero_w: int = 0,
):
    """
    - plate GroupShuffleSplit
    - 중앙값/분위수/상관/스케일/선택 모든 통계는 **train-only**, test에는 train 파라미터만 적용 (누수 차단)
    - pair TE는 plate-group OOF로 생성 (누수 차단), test에는 train의 pair 평균 주입
    - SVR 고정(SVR_C, SVR_EPS, SVR_GAMMA는 ENV로 전달)
    """
    rng = np.random.RandomState(seed)

    # 1) 사용 열 결정 및 split
    feat_cols = [c for c in df.columns if c not in EXCLUDE_FROM_X]
    num_cols  = [c for c in feat_cols if np.issubdtype(df[c].dtype, np.number)]
    y_all     = df[TARGET_COL].values
    groups    = df[PLATE_COL].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(df[num_cols], y_all, groups=groups))

    tr_df = df.iloc[tr_idx].copy()
    te_df = df.iloc[te_idx].copy()

    X_tr = tr_df[num_cols].copy()
    X_te = te_df[num_cols].copy()

    # 2) 중앙값 대치(train-only)
    med = X_tr.median(numeric_only=True).to_dict()
    X_tr = X_tr.fillna(med)
    X_te = X_te.fillna(med)

    # 3) (옵션) Pair TE (OOF, plate-group)
    if use_pair_te:
        te_tr, te_map, te_def = _oof_pair_te(tr_df, tr_df[TARGET_COL].values, use_delta=False, n_splits=5)
        # test pair → train map
        te_pairs = _make_pair_id(te_df).values
        te_te = np.array([te_map.get(p, te_def) for p in te_pairs], dtype=float)

        # fragmentation 방지: concat 1회
        X_tr = pd.concat([X_tr, pd.Series(te_tr, index=X_tr.index, name="pair_te")], axis=1)
        X_te = pd.concat([X_te, pd.Series(te_te, index=X_te.index, name="pair_te")], axis=1)

    # 4) (옵션) 분위수 clip (train 경계로만)
    if qclip:
        X_tr, X_te = _quantile_clip_train_apply(X_tr, X_te, qclip_lo, qclip_hi)

    # 5) (옵션) 상관 가지치기 (train 기준)
    if corr_thr is not None:
        X_tr, X_te = _corr_prune_train_apply(X_tr, X_te, float(corr_thr))

    # 6) 스케일링 (train 기준)
    Xtr_s, Xte_s, _ = _scale_train_apply(X_tr, X_te, scale_method)

    # 7) SelectKBest (train 기준)
    k = max(1, min(select_k, Xtr_s.shape[1]))
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(Xtr_s, tr_df[TARGET_COL].values)
    Xtr_sel = selector.transform(Xtr_s)
    Xte_sel = selector.transform(Xte_s)

    # 8) (옵션) 가중치 (SVR sample_weight 지원)
    sample_weight = None
    if use_hetero_w:
        sample_weight = _pair_weights(tr_df, tr_df[TARGET_COL].values, use_delta=False).astype(float)

    # 9) SVR 학습
    svr = SVR(C=SVR_C, epsilon=SVR_EPS, gamma=SVR_GAMMA)
    svr.fit(Xtr_sel, tr_df[TARGET_COL].values, sample_weight=sample_weight)

    # 10) 평가
    y_tr_pred = svr.predict(Xtr_sel)
    y_te_pred = svr.predict(Xte_sel)

    res = {
        "algorithm":      "SVM",
        "train_r2":       float(r2_score(tr_df[TARGET_COL].values, y_tr_pred)),
        "test_r2":        float(r2_score(te_df[TARGET_COL].values, y_te_pred)),
        "train_rmse":     float(np.sqrt(mean_squared_error(tr_df[TARGET_COL].values, y_tr_pred))),
        "test_rmse":      float(np.sqrt(mean_squared_error(te_df[TARGET_COL].values, y_te_pred))),
        "train_samples":  int(len(tr_df)),
        "test_samples":   int(len(te_df)),
        "used_k":         int(k),
        "scale_method":   scale_method,
        "corr_thr":       None if corr_thr is None else float(corr_thr),
        "qclip":          int(qclip),
        "qclip_lo":       float(qclip_lo),
        "qclip_hi":       float(qclip_hi),
        "use_pair_te":    int(use_pair_te),
        "use_hetero_w":   int(use_hetero_w),
        "svr_C":          float(SVR_C),
        "svr_eps":        float(SVR_EPS),
        "svr_gamma":      SVR_GAMMA,
    }
    return res

# ------------- Δ-분석/요약 ----------------
def analyze_transition_variability(final_df: pd.DataFrame) -> dict:
    try:
        from scipy.stats import kruskal, levene
    except Exception:
        kruskal = levene = None

    df = final_df.copy()
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["pair"] = df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

    def _stats(sub):
        g = sub.groupby("pair")["delta"]
        stat = g.agg(count="count", mean="mean", std="std", median="median",
                     q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index()
        std_ratio = (stat["std"].max() / stat["std"].replace(0, np.nan).min()) if stat["std"].notna().any() else np.nan
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

def write_summary(summary_data, summary_path: Path, timestamp: datetime, experiment_id: str):
    lines = []
    lines.append("Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
    lines.append("="*72)
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

    v = summary_data.get("variability", {})
    for tag in ["e2x","x2e"]:
        vv = v.get(tag, {})
        lines.append(f"\n[Δ-분석 {tag.upper()}] std_max/min={vv.get('std_ratio', np.nan):.3f} | "
                     f"Kruskal p={vv.get('kw_p', np.nan)} | Levene p={vv.get('lev_p', np.nan)}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 요약 저장: {summary_path}")

# ----------------------------- main ------------------------------
def main():
    _ensure_dirs()
    success = False
    experiment_id = _next_experiment_id()
    timestamp = datetime.now()
    summary_path = REPORTS_SUMMARY_DIR / f"{experiment_id}_test_no_op_summary.txt"
    archived_script_path = REPORTS_CODE_DIR / f"{experiment_id}_test_no_op.py"

    print("\n[DEBUG] FLAGS",
          f"DT={USE_DELTA_TARGET} TE={USE_PAIR_TE} W={USE_HETERO_W} RM={USE_RM_TRAINONLY} "
          f"| PERMUTE={PERMUTE_TARGET} LEAK_AUTOCHECK={LEAK_AUTOCHECK} "
          f"| SELECT_K={SELECT_K} SVR(C={SVR_C},eps={SVR_EPS},gamma={SVR_GAMMA})")

    try:
        # 1) 병합/타깃(t+1)
        final_df = load_and_merge_data_tplus1()

        # Δ‑변동성 분석
        variability = analyze_transition_variability(final_df)

        # 2) 전이 세트
        e2x_df, x2e_df = split_cross_transitions(final_df)

        # 3) FE
        e2x_fe = feature_engineering(e2x_df, "E2X", PROCESSED_DIR / "e2x_featured_tplus1.csv")
        x2e_fe = feature_engineering(x2e_df, "X2E", PROCESSED_DIR / "x2e_featured_tplus1.csv")

        # 4) 이상치 (분할 전 NO-OP; 내부 train-only 처리)
        e2x_clean = remove_outliers(PROCESSED_DIR / "e2x_featured_tplus1.csv",
                                    PROCESSED_DIR / "e2x_cleaned_tplus1.csv", "E2X")
        x2e_clean = remove_outliers(PROCESSED_DIR / "x2e_featured_tplus1.csv",
                                    PROCESSED_DIR / "x2e_cleaned_tplus1.csv", "X2E")

        # 5) 학습(SVM 고정)
        # E2X
        e2x_res = train_model(
            e2x_clean, "E2X",
            seed=42, test_size=0.2,
            select_k=SELECT_K,
            scale_method=SCALE_METHOD,
            corr_thr=CORR_THR,
            qclip=QCLIP, qclip_lo=QCLIP_LO, qclip_hi=QCLIP_HI,
            use_pair_te=USE_PAIR_TE,
            use_hetero_w=USE_HETERO_W,
        )
        # X2E
        x2e_res = train_model(
            x2e_clean, "X2E",
            seed=42, test_size=0.2,
            select_k=SELECT_K,
            scale_method=SCALE_METHOD,
            corr_thr=CORR_THR,
            qclip=QCLIP, qclip_lo=QCLIP_LO, qclip_hi=QCLIP_HI,
            use_pair_te=USE_PAIR_TE,
            use_hetero_w=USE_HETERO_W,
        )

        average_test_r2 = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))

        # 요약/산출물
        summary = {
            "data_artifacts": [
                f"Final merged(t+1): {PROCESSED_DIR / 'final_merged_data_regression_tplus1.csv'}",
                f"E2X raw/clean: {PROCESSED_DIR / 'e2x_raw_tplus1.csv'}, {PROCESSED_DIR / 'e2x_cleaned_tplus1.csv'}",
                f"X2E raw/clean: {PROCESSED_DIR / 'x2e_raw_tplus1.csv'}, {PROCESSED_DIR / 'x2e_cleaned_tplus1.csv'}",
                f"All featured: {PROCESSED_DIR / 'all_featured_tplus1.csv'}",
                f"Archived script: {archived_script_path}",
            ],
            "models": {"E2X": e2x_res, "X2E": x2e_res},
            "average_test_r2": average_test_r2,
            "variability": variability,
        }
        write_summary(summary, summary_path, timestamp, experiment_id)

        shutil.copy2(Path(__file__).resolve(), archived_script_path)
        print(f"✓ 코드 아카이브 저장: {archived_script_path}")

        # 메트릭 파일 (sh 집계 호환)
        _save_last_metrics(experiment_id,
                           e2x_res["test_r2"], x2e_res["test_r2"],
                           average_test_r2, summary_path)
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
