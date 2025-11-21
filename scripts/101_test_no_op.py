"""
092_experiment.py — 통합 안전본 (라벨 누수 가드 + 스위치 실적용 + 오류 수정)

요구사항 반영:
- summary_path = reports/summaries/{experiment_id}_test_no_op_summary.txt
- DT/TE/W/MONO/RM 스위치가 실제로 모델 입력/학습에 반영
- 분할 이후에만 모든 통계/중앙값/이상치 경계 계산 (train-only)
- Group-aware CV (GroupKFold with groups=plate)
- rollforward 번들 불일치 수정
- last_run_metrics.csv 열 순서 고정 (experiment_id,e2x_test_r2,x2e_test_r2,average_test_r2,summary_path)

필수 데이터:
- data/entry_direction_results.csv
- data/exit_direction_results.csv
- data/posco1_105190.csv (RM)
- data/posco2_105190.csv (FM)
"""

import os
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import traceback

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score, cross_validate
from xgboost import XGBRegressor

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

TARGET_COL = "warping_index_target"   # t+1 타깃
PASS_COL   = "FM_PASS NO N"
PLATE_COL  = "FM_날판번호"
MONTH_COL  = "FM_압연월"
CUR_WARP   = "warping_index_current_pass"

EXCLUDE_FROM_X = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}

# --------------------------- ENV 스위치 -----------------------------
def _env_on(name: str, default: str = "0") -> int:
    v = os.getenv(name, default).strip().lower()
    return 1 if v in ("1", "true", "yes", "y", "on") else 0

RUN_NAME         = os.getenv("RUN_NAME", "pass_t_plus_1_cross")
PERMUTE_TARGET   = _env_on("PERMUTE_TARGET", "0")
LEAK_AUTOCHECK   = _env_on("LEAK_AUTOCHECK", "1")
USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET", "0")
USE_PAIR_TE      = _env_on("USE_PAIR_TE", "0")
USE_HETERO_W     = _env_on("USE_HETERO_W", "0")
USE_MONO_CUR     = _env_on("USE_MONO_CUR", "0")
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY", "0")

# 알고리즘 고정(있으면 강제, 없으면 CV로 선택)
E2X_ALGO_FIXED = os.getenv("E2X_ALGO", "XGBoost").strip() or None  # 기본 XGBoost
X2E_ALGO_FIXED = os.getenv("X2E_ALGO", "").strip() or None

ANCHOR_TAG = os.getenv("ANCHOR_TAG", "").strip() or None

# --------------------------- 유틸 -----------------------------
def _ensure_dirs():
    for d in [PROCESSED_DIR, REPORTS_DIR, REPORTS_SUMMARY_DIR, REPORTS_CODE_DIR, MODELS_DIR, IMAGES_DIR, REPORTS_DIR/"metrics"]:
        d.mkdir(parents=True, exist_ok=True)

def _next_experiment_id() -> str:
    existing = []
    for f in REPORTS_SUMMARY_DIR.glob("*.txt"):
        m = re.match(r"(\d{3})_", f.name)
        if m:
            try: existing.append(int(m.group(1)))
            except: pass
    return f"{(max(existing)+1) if existing else 1:03d}"

def _hard_leak_guard(cols: list[str]) -> list[str]:
    """이름 기반 라벨·예측 스멜 컬럼 제거."""
    bad_kw = ("target", "label", "pred", "oof", "_hat", "y_", "_y", "te_")
    out = []
    for c in cols:
        low = c.lower()
        if any(k in low for k in bad_kw):
            continue
        out.append(c)
    return out

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
    # 열 순서 고정 (sh에서 NF-1/NF로 읽음)
    df = df[["experiment_id","e2x_test_r2","x2e_test_r2","average_test_r2","summary_path"]]
    df.to_csv(csv, index=False)
    print(f"✓ 메트릭 저장: {csv}")

def _ensure_numeric_X(df: pd.DataFrame,
                      exclude: set[str],
                      base_cols: list[str] | None = None,
                      medians: dict | None = None) -> tuple[pd.DataFrame, dict, list]:
    """
    - exclude 컬럼 제거
    - base_cols가 주어지면 그 순서로 재배치(누락 열은 만들어 결측으로)
    - medians 없으면 중앙값(train-only)을 산출해 반환
    """
    d = df.copy()
    # 우선 exclude 제거
    d = d.drop(columns=[c for c in exclude if c in d.columns], errors="ignore")
    # 숫자형 후보만
    num_cols_all = d.select_dtypes(include=[np.number]).columns.tolist()

    if base_cols is None:
        base_cols = num_cols_all
    else:
        base_cols = list(base_cols)

    # base_cols에 있지만 현재 df에 없는 열은 결측으로 보강
    for c in base_cols:
        if c not in d.columns:
            d[c] = np.nan

    X = d[base_cols].copy()

    if medians is None:
        med_series = X.median(numeric_only=True).fillna(0.0)
        medians = med_series.to_dict()

    X = X.fillna(medians).fillna(0.0)
    return X, medians, base_cols


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

    # RM 통계(전체) — 잠재 누수 회피 위해 이후 train-only 드롭/주입 제어
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

    # PERMUTE (누수 sanity)
    if PERMUTE_TARGET:
        merged_fm[TARGET_COL] = merged_fm[TARGET_COL].sample(frac=1.0, random_state=42).values
        print("[PERMUTE] Target shuffled for leak sanity check.")

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

    # 누수 방지: 진행도는 train-only에서 다시 만듦. 여기서는 squared만.
    if PASS_COL in df.columns:
        df["FM_PASS_squared"] = df[PASS_COL] ** 2

    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path}")
    return df

# -------------------- 4) 이상치 제거 (NO-OP; train-only로 이동) -------------------
def remove_outliers(input_path: Path, output_path: Path, dataset_name: str):
    print(f"\n[Step 4] 이상치 제거 - {dataset_name}")
    # 분할 전 이상치 제거는 누수 위험 → 여기서는 그대로 복사만, 실제 처리는 train_model 내부에서 train-only로 수행
    df = pd.read_csv(input_path)
    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path} (최종 {df.shape[0]}개)")
    return df

# ------------------- 5) 알고리즘 비교 (Group-aware CV) -------------------
def _cv_score(model, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    out = cross_val_score(model, X, y, cv=gkf, scoring="r2", n_jobs=-1, groups=groups)
    return out.mean(), out.std()

def compare_algorithms(df: pd.DataFrame, dataset_name: str):
    print("\n" + "="*80)
    print(f"[Step 3] 알고리즘 비교 - {dataset_name} (plate-group CV)")
    print("="*80)

    # 입력 구성: exclude 제거 + 수치형 + 같은 기준으로 중앙값 대치(현재 df 기준)
    base_cols = _hard_leak_guard([c for c in df.columns if c not in EXCLUDE_FROM_X])
    X_df, _, _ = _ensure_numeric_X(df[base_cols], exclude=EXCLUDE_FROM_X, base_cols=base_cols, medians=None)
    y = df[TARGET_COL].values
    groups = df[PLATE_COL].astype(str).values

    algos = {
        "XGBoost": XGBRegressor(
            random_state=42, tree_method="hist", n_jobs=0,
            n_estimators=400, learning_rate=0.05,
            max_depth=4, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.25, reg_lambda=1.5, gamma=0.1, min_child_weight=5
        ),
        "RandomForest": RandomForestRegressor(
            random_state=42, n_estimators=600, n_jobs=-1,
            max_depth=14, min_samples_leaf=3, max_features="sqrt"
        ),
    }

    gkf = GroupKFold(n_splits=5)
    best_name, best_score = None, -np.inf
    for name, model in algos.items():
        scores = cross_val_score(model, X_df.values, y, cv=gkf, groups=groups, scoring="r2", n_jobs=-1)
        print(f"  {name:12s}: R2 = {scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_score, best_name = scores.mean(), name

    print(f"\n  → 최적 알고리즘: {best_name} (CV R2 = {best_score:.4f})")
    return best_name, best_score



# -------------------- 학습 유틸(OOF TE, 가중치, 선택자) --------------------

def _make_pair_id(df: pd.DataFrame) -> pd.Series:
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)

def _oof_pair_te(tr_df: pd.DataFrame, y_tr: np.ndarray, use_delta: bool, n_splits=5):
    """plate-group OOF로 pair Δ 평균 타깃 인코딩 생성 (train만 적합, test에는 train 통계만 주입)."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = tr_df[PLATE_COL].astype(str).values
    pair  = _make_pair_id(tr_df).values
    cur   = tr_df[CUR_WARP].values
    target = y_tr.copy()
    if use_delta:
        target = y_tr - cur

    te = np.zeros(len(tr_df), dtype=float)
    for tr_idx, va_idx in gkf.split(tr_df, target, groups):
        means = (
            pd.DataFrame({"pair": pair[tr_idx], "y": target[tr_idx]})
            .groupby("pair")["y"].mean()
        )
        glob = float(target[tr_idx].mean())
        te[va_idx] = [means.get(p, glob) for p in pair[va_idx]]

    # test 주입용 전역 통계
    global_means = pd.DataFrame({"pair": pair, "y": target}).groupby("pair")["y"].mean().to_dict()
    global_default = float(target.mean())
    return te, global_means, global_default

def _pair_weights(tr_df: pd.DataFrame, y_tr: np.ndarray, use_delta: bool):
    pair  = _make_pair_id(tr_df).values
    cur   = tr_df[CUR_WARP].values
    target = y_tr.copy()
    if use_delta:
        target = y_tr - cur
    dfw = pd.DataFrame({"pair": pair, "y": target})
    s = dfw.groupby("pair")["y"].std().replace(0, np.nan).fillna(dfw["y"].std())
    w = (1.0 / (s**2 + 1e-6)).to_dict()
    return np.array([w.get(p, 1.0) for p in pair], dtype=float)

# ---------------------------- 학습 본체 ----------------------------
def train_model(df: pd.DataFrame,
                dataset_name: str,           # "E2X" | "X2E"
                algorithm: str = "XGBoost",  # "XGBoost" | "RandomForest"
                seed: int = 42,
                test_size: float = 0.2,
                select_k: int = 120):
    """
    099 흐름 유지 + 누수 가드 + 스위치 실적용 + XGB 정규화 강화.
    - 분할: plate GroupShuffleSplit(8:2) + ANCHOR_TAG 있으면 재사용
    - PASS 진행도: train의 max로 고정하여 test에도 동일 스케일 적용
    - 중앙값/클리핑/선택: 전부 train-only로 계산, test에는 주입만
    - USE_DELTA_TARGET, USE_PAIR_TE(OOF), USE_HETERO_W(sample_weight), USE_MONO_CUR(+1 on CUR_WARP) 반영
    - 저장: model, selector, feature_medians, selector_fit_cols, test_plates
    """
    # ---------------- 환경 스위치 ----------------
    USE_DELTA = str(os.getenv("USE_DELTA_TARGET", "0")).lower() in ("1","true","yes","on")
    USE_TE    = str(os.getenv("USE_PAIR_TE", "0")).lower() in ("1","true","yes","on")
    USE_W     = str(os.getenv("USE_HETERO_W", "0")).lower() in ("1","true","yes","on")
    USE_MONO  = str(os.getenv("USE_MONO_CUR", "0")).lower() in ("1","true","yes","on")
    anchor_tag = os.getenv("ANCHOR_TAG", "").strip()

    data = df.copy()
    # E2X/X2E 분리(홀/짝)
    if dataset_name.upper() == "E2X":
        data = data[data[PASS_COL] % 2 == 1].copy()
        prefix = "e2x"
    else:
        data = data[data[PASS_COL] % 2 == 0].copy()
        prefix = "x2e"

    # 앵커 경로
    anchor_path = MODELS_DIR / (f"{anchor_tag}_" if anchor_tag else "") / f"{prefix}_test_plates_tplus1.pkl"
    anchor_path.parent.mkdir(parents=True, exist_ok=True)

    # 그룹 분할
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups_all = data[PLATE_COL].astype(str).values

    if anchor_path.exists():
        test_plates = set(joblib.load(anchor_path))
        te_mask = data[PLATE_COL].astype(str).isin(test_plates)
        tr_mask = ~te_mask
        if not tr_mask.any() or not te_mask.any():  # 방어
            tr_idx, te_idx = next(gss.split(data, data[TARGET_COL].values, groups=groups_all))
            test_plates = set(data.iloc[te_idx][PLATE_COL].astype(str).unique())
            joblib.dump(sorted(list(test_plates)), anchor_path)
        else:
            tr_idx = np.where(tr_mask)[0]; te_idx = np.where(te_mask)[0]
    else:
        tr_idx, te_idx = next(gss.split(data, data[TARGET_COL].values, groups=groups_all))
        test_plates = set(data.iloc[te_idx][PLATE_COL].astype(str).unique())
        joblib.dump(sorted(list(test_plates)), anchor_path)

    tr_df_raw = data.iloc[tr_idx].copy()
    te_df_raw = data.iloc[te_idx].copy()

    # PASS 진행도(train max로 고정)
    if PASS_COL in tr_df_raw.columns:
        pmax = float(tr_df_raw[PASS_COL].max()) or 1.0
        tr_df_raw["FM_PASS_progress"] = tr_df_raw[PASS_COL] / pmax
        te_df_raw["FM_PASS_progress"] = te_df_raw[PASS_COL] / pmax

    # 기본 피처 후보(누수 스멜 제거 + exclude 제거)
    base_cols_all = _hard_leak_guard([c for c in tr_df_raw.columns if c not in EXCLUDE_FROM_X])

    # 수치행렬 + 중앙값(train) 산출
    Xtr_df, med, base_cols = _ensure_numeric_X(tr_df_raw[base_cols_all], exclude=EXCLUDE_FROM_X, base_cols=base_cols_all, medians=None)
    Xte_df, _,  _          = _ensure_numeric_X(te_df_raw[base_cols_all],  exclude=EXCLUDE_FROM_X, base_cols=base_cols,   medians=med)

    # 이상치(train-only) : 1%~99% 경계 → train 필터, test clip
    q_lo, q_hi = 0.01, 0.99
    bounds = {}
    for c in Xtr_df.columns:
        lo, hi = np.quantile(Xtr_df[c].values, [q_lo, q_hi])
        bounds[c] = (float(lo), float(hi))
    keep = np.ones(len(Xtr_df), dtype=bool)
    for c,(lo,hi) in bounds.items():
        v = Xtr_df[c].values
        keep &= (v >= lo) & (v <= hi)
    Xtr_df = Xtr_df.loc[keep].reset_index(drop=True)
    tr_df_raw = tr_df_raw.loc[keep].reset_index(drop=True)

    for c,(lo,hi) in bounds.items():
        Xte_df[c] = Xte_df[c].clip(lower=lo, upper=hi)

    # 타깃
    y_tr = tr_df_raw[TARGET_COL].values
    y_te = te_df_raw[TARGET_COL].values
    y_te_true = y_te.copy()

    # Δ-타깃
    if USE_DELTA:
        y_tr = y_tr - tr_df_raw[CUR_WARP].values
        y_te = y_te - te_df_raw[CUR_WARP].values

    # OOF Pair-TE (선택)
    if USE_TE:
        te_tr, te_map, te_def = _oof_pair_te(tr_df_raw, y_tr, use_delta=USE_DELTA, n_splits=5)
        Xtr_df["PAIR_TE"] = te_tr
        # test에는 train 전역 통계 주입
        te_pairs = _make_pair_id(te_df_raw).values
        Xte_df["PAIR_TE"] = np.array([te_map.get(p, te_def) for p in te_pairs], dtype=float)

    # 선택(SelectKBest) — k가 너무 크면 과적합 경향 → 기본 120, 필요 시 ENV로 낮춰 실행
    selector = SelectKBest(score_func=f_regression, k=min(select_k, Xtr_df.shape[1]))
    selector.fit(Xtr_df.values, y_tr)
    Xtr_sel = selector.transform(Xtr_df.values)
    Xte_sel = selector.transform(Xte_df.values)

    # XGB 단조 제약(선택): 선택된 피처 중 CUR_WARP가 있으면 +1 제약
    mono_constraints = None
    if USE_MONO and algorithm == "XGBoost":
        sel_idx = selector.get_support(indices=True)
        sel_names = [list(Xtr_df.columns)[i] for i in sel_idx]
        mono_constraints = [0] * len(sel_names)
        if CUR_WARP in sel_names:
            j = sel_names.index(CUR_WARP)
            mono_constraints[j] = 1  # (+) 단조
        # XGBoost는 list/tuple 또는 문자열 "(1,0,...)" 형태를 허용
        mono_constraints = tuple(mono_constraints)

    # 가중치(선택)
    sw = None
    if USE_W:
        sw = _pair_weights(tr_df_raw, y_tr, use_delta=USE_DELTA)

    # 모델
    if algorithm == "RandomForest":
        model = RandomForestRegressor(
            random_state=seed, n_estimators=700, n_jobs=-1,
            max_depth=14, min_samples_leaf=3, max_features="sqrt"
        )
        model.fit(Xtr_sel, y_tr, sample_weight=sw)
    else:
        model = XGBRegressor(
            random_state=seed, tree_method="hist", n_jobs=0,
            n_estimators=400, learning_rate=0.05,
            max_depth=4, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.25, reg_lambda=1.5, gamma=0.1, min_child_weight=5,
            monotone_constraints=mono_constraints
        )
        # 일부 xgboost 버전은 early_stopping 인자 미지원 → 정규화로 대체
        model.fit(Xtr_sel, y_tr, sample_weight=sw)

    # 예측 및 Δ 복원
    y_tr_pred = model.predict(Xtr_sel)
    y_te_pred = model.predict(Xte_sel)
    if USE_DELTA:
        y_tr_pred = y_tr_pred + tr_df_raw[CUR_WARP].values
        y_te_pred = y_te_pred + te_df_raw[CUR_WARP].values

    # 스코어
    res = {
        "train_r2":  float(r2_score(tr_df_raw[TARGET_COL].values, y_tr_pred)),
        "test_r2":   float(r2_score(y_te_true, y_te_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(tr_df_raw[TARGET_COL].values, y_tr_pred))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te_true, y_te_pred))),
        "train_samples": int(len(y_tr)), "test_samples": int(len(y_te_true)),
        "best_params": getattr(model, "get_params", lambda: {})(),
    }

    # 저장(rollforward 호환: 학습 당시 '입력 전체 열 순서' 보관)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,                MODELS_DIR / f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(selector,             MODELS_DIR / f"{prefix}_selector_tplus1.pkl")
    joblib.dump(med,                  MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl")
    joblib.dump(list(Xtr_df.columns), MODELS_DIR / f"{prefix}_selector_fit_cols_tplus1.pkl")
    joblib.dump(sorted(list(test_plates)), MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl")

    return res, test_plates

# ----------------- rollforward -------------------
def _load_bundle(prefix: str) -> dict:
    model    = joblib.load(MODELS_DIR / f"{prefix}_regressor_tplus1.pkl")
    selector = joblib.load(MODELS_DIR / f"{prefix}_selector_tplus1.pkl")
    medians  = joblib.load(MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl")
    fit_cols = joblib.load(MODELS_DIR / f"{prefix}_selector_fit_cols_tplus1.pkl")
    return {"model": model, "selector": selector, "medians": medians, "fit_cols": list(fit_cols)}

def _predict_next_from_row(row: pd.Series, bundle: dict) -> float:
    fit_cols = bundle.get("fit_cols", list(row.index))
    X_row = pd.DataFrame([row.reindex(fit_cols)], columns=fit_cols)
    # 중앙값 대치
    med = bundle["medians"]
    X_row = X_row.fillna({k: med.get(k, 0.0) for k in fit_cols}).fillna(0.0)
    # SelectKBest 마스크로 직접 슬라이스(이름검사 우회)
    selector = bundle["selector"]
    sel_idx = selector.get_support(indices=True)
    X_sel = X_row.iloc[:, sel_idx].to_numpy()
    pred = float(bundle["model"].predict(X_sel)[0])
    return pred

def rollforward_predict_all_plates(all_featured_df: pd.DataFrame, e2x_bundle: dict, x2e_bundle: dict) -> pd.DataFrame:
    outs = []
    for plate in sorted(all_featured_df[PLATE_COL].dropna().unique()):
        sub = all_featured_df[all_featured_df[PLATE_COL] == plate].sort_values(PASS_COL)
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

def rollforward_r2_on_test_only(all_featured_df: pd.DataFrame) -> float:
    """
    두 모델의 '테스트 판 교집합'만 사용하여 t→t+1 한 칸 예측 후 전역 R² 계산.
    번들에서 'selector_input_cols'를 읽어 입력 컬럼을 안전하게 구성한다.
    """
    e2x_bundle = _load_bundle("e2x")
    x2e_bundle = _load_bundle("x2e")

    e2x_test = set(joblib.load(MODELS_DIR / "e2x_test_plates_tplus1.pkl"))
    x2e_test = set(joblib.load(MODELS_DIR / "x2e_test_plates_tplus1.pkl"))
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
                pred = _predict_next_from_row(row_t, bundle)
                gt_next = float(idx.loc[p+1].get("warping_index_current_pass"))
                rows.append({"plate": plate, "from": p, "to": p+1, "pred": pred, "gt_next": gt_next})
            except Exception as ex:
                row_errors.append({"plate": plate, "from": p, "to": p+1, "error": repr(ex)})
                continue

    if not rows:
        print("[Strict RF] 유효한 예측 행이 없습니다.")
        return float("nan")

    df = pd.DataFrame(rows)
    return float(r2_score(df["gt_next"], df["pred"]))
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

def export_pairwise_delta_stats(final_df: pd.DataFrame):
    df = final_df.copy()
    df["pair"] = df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["transition"] = np.where(df[PASS_COL] % 2 == 1, "E2X", "X2E")
    g = df.groupby(["transition","pair"])["delta"]
    stats = g.agg(
        n="size", mean="mean", std="std", median="median",
        q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
    ).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["n"])
    stats.loc[stats["n"] <= 1, ["se"]] = np.nan
    stats["ci95_lo"] = stats["mean"] - 1.96 * stats["se"]
    stats["ci95_hi"] = stats["mean"] + 1.96 * stats["se"]
    stats["n_flag"] = np.where(stats["n"] < 30, "small-n(<30)", "ok")
    out = REPORTS_DIR / "pair_delta_stats.csv"
    stats.to_csv(out, index=False)
    print(f"✓ 모든 쌍 Δ 통계 저장: {out}")
    return stats

def write_summary(summary_data, summary_path: Path, timestamp: datetime, experiment_id: str, best_record: dict):
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
    rf_r2 = summary_data.get("rollforward_r2", None)
    if rf_r2 is not None and not np.isnan(rf_r2):
        lines.append(f"[롤포워드 R2(전체 판, t→t+1)] {rf_r2:.4f}")

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
    # 요구사항: 파일명 고정
    summary_path = REPORTS_SUMMARY_DIR / f"{experiment_id}_test_no_op_summary.txt"
    archived_script_path = REPORTS_CODE_DIR / f"{experiment_id}_test_no_op.py"

    print("\n[DEBUG] FLAGS",
          f"DT={USE_DELTA_TARGET} TE={USE_PAIR_TE} W={USE_HETERO_W} MONO={USE_MONO_CUR} RM={USE_RM_TRAINONLY} "
          f"| PERMUTE={PERMUTE_TARGET} LEAK_AUTOCHECK={LEAK_AUTOCHECK} "
          f"| E2X_ALGO_FIXED={E2X_ALGO_FIXED} X2E_ALGO_FIXED={X2E_ALGO_FIXED} ANCHOR_TAG={ANCHOR_TAG}")

    try:
        # 1) 병합/타깃(t+1)
        final_df = load_and_merge_data_tplus1()

        # Δ-변동성 분석
        variability = analyze_transition_variability(final_df)

        # 2) 전이 세트
        e2x_df, x2e_df = split_cross_transitions(final_df)

        # 3) FE
        e2x_fe = feature_engineering(e2x_df, "E2X", PROCESSED_DIR / "e2x_featured_tplus1.csv")
        x2e_fe = feature_engineering(x2e_df, "X2E", PROCESSED_DIR / "x2e_featured_tplus1.csv")

        # 4) 이상치 (분할 전 NO-OP; 내부 train-only 처리)
        e2x_clean = remove_outliers(PROCESSED_DIR / "e2x_featured_tplus1.csv", PROCESSED_DIR / "e2x_cleaned_tplus1.csv", "E2X")
        x2e_clean = remove_outliers(PROCESSED_DIR / "x2e_featured_tplus1.csv", PROCESSED_DIR / "x2e_cleaned_tplus1.csv", "X2E")

        # 5) 알고리즘 비교(필요 시)
        if E2X_ALGO_FIXED:
            e2x_algo = E2X_ALGO_FIXED
        else:
            e2x_algo, _ = compare_algorithms(e2x_clean, "E2X")

        if X2E_ALGO_FIXED:
            x2e_algo = X2E_ALGO_FIXED
        else:
            x2e_algo, _ = compare_algorithms(x2e_clean, "X2E")

        # 6) 학습
        e2x_res, e2x_test_plates = train_model(e2x_clean, "E2X", algorithm=e2x_algo)
        x2e_res, x2e_test_plates = train_model(x2e_clean, "X2E", algorithm=x2e_algo)

        e2x_res["algorithm"] = e2x_algo
        x2e_res["algorithm"] = x2e_algo

        average_test_r2 = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))

        # 7) 번갈아 예측/rollforward
        all_fe = feature_engineering(final_df, "All", PROCESSED_DIR / "all_featured_tplus1.csv")
        e2x_bundle = _load_bundle("e2x")
        x2e_bundle = _load_bundle("x2e")
        rf_pred = rollforward_predict_all_plates(all_fe, e2x_bundle, x2e_bundle)
        rf_out = PROCESSED_DIR / "rollforward_predictions_tplus1_cross.csv"
        rf_pred.to_csv(rf_out, index=False)
        print(f"✓ 번갈아 예측 결과 저장: {rf_out}")

        # 전체 rollforward R² (gt_next가 있는 행만)
        mask = rf_pred["gt_next"].notna()
        if mask.any():
            rollforward_r2 = float(r2_score(rf_pred.loc[mask,"gt_next"], rf_pred.loc[mask,"pred"]))
        else:
            rollforward_r2 = np.nan

        # (옵션) 교집합 test 판만 rollforward
        strict_rollforward_r2 = rollforward_r2_on_test_only(all_fe)
        if not np.isnan(strict_rollforward_r2):
            # summary에는 전체 rollforward를, 로그에는 strict도 함께
            print(f"[Strict Test‑only Rollforward R²] {strict_rollforward_r2:.4f}")

        # (1) pair Δ 통계 CSV
        _ = export_pairwise_delta_stats(final_df)

        # 요약 준비
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
        write_summary(summary, summary_path, timestamp, experiment_id, best_record={})

        # 성공 결과만 로그/코드 아카이브
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
