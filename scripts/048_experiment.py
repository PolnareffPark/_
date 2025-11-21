# -*- coding: utf-8 -*-
"""
032_experiment.py  (032 기준 강화판)
- 누수 차단: Group Split 이후 전처리, Train-only 통계/스케일, Hard Leak Guard
- 전략 플래그: Δ-타깃(USE_DELTA_TARGET), Pair Δ-TE(USE_PAIR_TE), 이분산 가중(USE_HETERO_W), 단조 제약(USE_MONO_CUR)
- 퍼뮤테이션 모드(PERMUTE_TARGET) + 빠른 퍼뮤테이션 감사(LEAK_AUTOCHECK)
- 셀렉터 피클 오류 방지: selector 객체 저장 대신 feature_cols/medians만 저장
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

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- 환경 플래그 ------------------------------
def _env_on(name: str, default="0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1","true","yes","y"}

# 전략/검증 플래그
PERMUTE_TARGET     = _env_on("PERMUTE_TARGET", "0")     # 전체 타깃 무작위화 모드
LEAK_AUTOCHECK     = _env_on("LEAK_AUTOCHECK", "1")     # 빠른 퍼뮤 감사
USE_DELTA_TARGET   = _env_on("USE_DELTA_TARGET", "0")   # Δ-타깃
USE_PAIR_TE        = _env_on("USE_PAIR_TE", "0")        # Pair Δ-TE 특성
USE_HETERO_W       = _env_on("USE_HETERO_W", "0")       # 이분산 가중(WLS)
USE_MONO_CUR       = _env_on("USE_MONO_CUR", "0")       # 단조 제약(+1 on CUR_WARP, XGB)
USE_RM_TRAINONLY   = _env_on("USE_RM_TRAINONLY", "0")   # RM 집계 Train-only 부착

SEED = 42
TEST_SIZE = 0.2
SELECT_K = 120
CV_FOLDS = 3
PAIR_EB_N0 = 50
WEIGHT_CLIP = (0.25, 3.0)

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
        "e2x_perm_r2","x2e_perm_r2"
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

def _hard_leak_guard(cols: list[str]) -> list[str]:
    """타깃/라벨 스멜 컬럼 전부 제거(이름기반)."""
    bad_kw = ("target", "label", "y_", "_y", "oof", "te_", "_te", "pred", "_hat")
    safe = []
    for c in cols:
        low = c.lower()
        if any(k in low for k in bad_kw):
            continue
        safe.append(c)
    return safe

def _pair_str(df: pd.DataFrame) -> pd.Series:
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)

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

    ground_truth["extracted_plate"] = ground_truth["filename"].str.extract(r"(PB\d+)")
    ground_truth["extracted_pass"]  = ground_truth["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    # RM 병합 + 판단위 통계(평균/최대/최소/표준편차/중앙값) — (초기 결합; 필요시 train-only로 대체 부착)
    merged_rm = pd.merge(
        ground_truth, rm_data,
        left_on=["extracted_plate","extracted_pass"],
        right_on=["RM_날판번호","RM_압연Pass번호"],
        how="inner"
    )
    num_cols = merged_rm.select_dtypes(include=[np.number]).columns.tolist()
    stat_cols = [c for c in num_cols if c not in ["RM_날판번호","RM_압연Pass번호","warping_index","extracted_pass"]]
    rm_stats = merged_rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(6)
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    rm_stats = rm_stats.reset_index()

    # FM 병합
    merged_fm = pd.merge(
        ground_truth, fm_data,
        left_on=["extracted_plate","extracted_pass"],
        right_on=[PLATE_COL, PASS_COL],
        how="inner"
    ).sort_values([PLATE_COL, PASS_COL])

    # t+1 타깃
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

# -------------------- 보조: Train-only RM 통계 부착 --------------------
def _attach_train_only_rm_stats(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """RM 통계는 train 판에서만 계산 → train/test에 매핑(누수 방지)"""
    try:
        rm = pd.read_csv(DATA_DIR / "posco1_105190.csv")
    except Exception:
        return train_df.copy(), test_df.copy()
    tr = train_df.copy(); te = test_df.copy()
    plates_tr = set(tr[PLATE_COL].astype(str).unique().tolist())
    rm = rm[rm["RM_날판번호"].astype(str).isin(plates_tr)].copy()
    num_cols = rm.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"RM_날판번호","RM_압연Pass번호","warping_index"}
    stat_cols = [c for c in num_cols if c not in drop_like]
    rm_stats = rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(6)
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    rm_stats = rm_stats.reset_index()
    tr = pd.merge(tr, rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left").drop(columns=["RM_날판번호"], errors="ignore")
    te = pd.merge(te, rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left").drop(columns=["RM_날판번호"], errors="ignore")
    # 결측은 train 통계의 전역 중앙값으로 채움
    if not rm_stats.empty:
        med = rm_stats.drop(columns=["RM_날판번호"], errors="ignore").median(numeric_only=True)
        for c in med.index:
            if c in tr.columns: tr[c] = tr[c].fillna(float(med[c]))
            if c in te.columns: te[c] = te[c].fillna(float(med[c]))
    return tr, te

# -------------------- FE: Train 기준 스케일/요약 ----------------------
def _base_fe_train_test(tr: pd.DataFrame, te: pd.DataFrame):
    """PASS 진행도는 train의 max로 고정해 test에도 동일 스케일 적용."""
    tr = tr.copy(); te = te.copy()
    fm_cols = [c for c in tr.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
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
        pass_max = pass_max if pass_max > 0 else 1.0
        d["FM_PASS_progress"] = d[PASS_COL] / pass_max
        return d
    pass_max_train = float(tr[PASS_COL].max()) if len(tr) else 1.0
    tr = _fe(tr, pass_max_train)
    te = _fe(te, pass_max_train)
    return tr, te

# -------------------- Pair Δ-TE(OOF EB) -----------------------
def _add_pair_delta_te(tr: pd.DataFrame, te: pd.DataFrame, n0=PAIR_EB_N0):
    tr = tr.copy(); te = te.copy()
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
    tr["pair_delta_te"] = oof; te["pair_delta_te"] = te_feat
    return tr, te

# -------------------- 이분산 가중(OOF EB) ---------------------
def _oof_pair_std_weights(tr: pd.DataFrame, te: pd.DataFrame, n0=PAIR_EB_N0, clip=WEIGHT_CLIP):
    tr = tr.copy(); te = te.copy()
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
    # test mapping
    gs = tr["delta"].std(ddof=1)
    stat = tr.groupby("pair")["delta"].agg(n="size", std="std", mean="mean").reset_index()
    w = stat["n"]/(stat["n"]+n0)
    stat["std_sh"] = w*stat["std"].fillna(gs) + (1-w)*gs
    m_std = dict(zip(stat["pair"], stat["std_sh"]))
    te_std = te["pair"].map(m_std).fillna(gs).values
    w_tr = (float(gs)/(oof_std + 1e-6))
    w_tr = np.clip(w_tr, clip[0], clip[1])
    return w_tr, te_std

# -------------------- 알고리즘 빌더 ---------------------------
def _build_model(name: str, seed=SEED, mono_cur_idx=None, n_features=None):
    if name == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise RuntimeError("xgboost가 설치되어 있지 않습니다.") from e
        params = dict(
            objective="reg:squarederror",
            random_state=seed, tree_method="hist",
            n_estimators=800, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.0,
            n_jobs=0, eval_metric="rmse"
        )
        if USE_MONO_CUR and (mono_cur_idx is not None) and (n_features is not None):
            v = [0]*n_features
            if 0 <= mono_cur_idx < n_features:
                v[mono_cur_idx] = 1
            params["monotone_constraints"] = tuple(v)
        return XGBRegressor(**params)
    elif name == "RandomForest":
        return RandomForestRegressor(
            random_state=seed, n_estimators=600, n_jobs=-1,
            max_depth=20, min_samples_leaf=5, max_features="sqrt"
        )
    else:
        raise ValueError(f"알 수 없는 알고리즘: {name}")

# ------------------- 알고리즘 비교(plate-group CV)----------------
def compare_algorithms(df: pd.DataFrame, dataset_name: str):
    from sklearn.model_selection import cross_val_score
    print("\n" + "="*80)
    print(f"[Step 3] 알고리즘 비교 - {dataset_name} (plate-group CV)")
    print("="*80)
    data = df.copy()
    # 홀/짝 필터
    if dataset_name.upper() == "E2X":
        data = data[data[PASS_COL] % 2 == 1].copy()
    else:
        data = data[data[PASS_COL] % 2 == 0].copy()

    X_all = data.drop(columns=[c for c in EXCLUDE_FROM_X if c in data.columns], errors="ignore")
    cols = _hard_leak_guard(X_all.columns.tolist())
    X = X_all[cols].select_dtypes(include=[np.number]).fillna(X_all.median(numeric_only=True))
    y = data[TARGET_COL].values
    groups = data[PLATE_COL].astype(str).values

    algos = {
        "XGBoost": _build_model("XGBoost"),
        "RandomForest": _build_model("RandomForest"),
    }

    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
    cv_splits = list(gss.split(X, y, groups=groups))

    best_name, best_score = None, -np.inf
    for name, model in algos.items():
        scores = cross_val_score(model, X.values, y, cv=cv_splits, scoring="r2", n_jobs=-1)
        print(f"  {name:12s}: R2 = {scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_score, best_name = scores.mean(), name
    print(f"\n  → 최적 알고리즘: {best_name} (CV R2 = {best_score:.4f})")
    return best_name, best_score

# ------------------- 학습/평가(누수-안전 파이프라인) -------------------
def train_model(df: pd.DataFrame, dataset_name: str, algorithm: str, seed: int = SEED):
    """
    분할(plate group) → Train-only FE/통계 → (옵션) RM train-only → (옵션) Pair Δ-TE, 이분산 가중
    → 상관 기반 K-best → 학습(내부 plate holdout) → 평가 및 저장(feature_cols/medians만 저장)
    """
    data = df.copy()
    if dataset_name.upper() == "E2X":
        data = data[data[PASS_COL] % 2 == 1].copy()
        prefix = "e2x"
    else:
        data = data[data[PASS_COL] % 2 == 0].copy()
        prefix = "x2e"

    # Group Split
    X_all = data.drop(columns=[c for c in EXCLUDE_FROM_X if c in data.columns], errors="ignore")
    cols_all = _hard_leak_guard(X_all.columns.tolist())
    X_all = X_all[cols_all].select_dtypes(include=[np.number]).copy()
    y_all = data[TARGET_COL].values
    groups_all = data[PLATE_COL].astype(str).values

    anchor_path = MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl"
    if anchor_path.exists():
        test_plates = set(joblib.load(anchor_path))
        tr_mask = ~data[PLATE_COL].astype(str).isin(test_plates)
        te_mask =  data[PLATE_COL].astype(str).isin(test_plates)
        tr_idx = np.where(tr_mask.values)[0]; te_idx = np.where(te_mask.values)[0]
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr_idx, te_idx = next(gss.split(X_all, y_all, groups=groups_all))
        test_plates = sorted(data.iloc[te_idx][PLATE_COL].astype(str).unique())
        joblib.dump(test_plates, anchor_path)

    tr_raw = data.iloc[tr_idx].copy()
    te_raw = data.iloc[te_idx].copy()

    # (선택) RM train-only 통계 부착
    if USE_RM_TRAINONLY:
        tr_raw, te_raw = _attach_train_only_rm_stats(tr_raw, te_raw)

    # Train-only FE(스케일·요약)
    tr_fe, te_fe = _base_fe_train_test(tr_raw, te_raw)

    # (옵션) Pair Δ-TE 특성
    if USE_PAIR_TE:
        tr_fe, te_fe = _add_pair_delta_te(tr_fe, te_fe, n0=PAIR_EB_N0)

    # 숫자 피처 + 중앙값 대치
    feats_tr = tr_fe.drop(columns=[c for c in EXCLUDE_FROM_X if c in tr_fe.columns], errors="ignore")
    feats_te = te_fe.drop(columns=[c for c in EXCLUDE_FROM_X if c in te_fe.columns], errors="ignore")
    feat_cols = _hard_leak_guard(feats_tr.columns.tolist())
    feats_tr = feats_tr[feat_cols].select_dtypes(include=[np.number]).copy()
    feats_te = feats_te[feat_cols].select_dtypes(include=[np.number]).copy()
    med = feats_tr.median(numeric_only=True).to_dict()
    feats_tr = feats_tr.fillna(med); feats_te = feats_te.fillna(med)

    # 이상치 가드: train 분위 경계 필터 / test clip
    q_low, q_high = 0.01, 0.99
    bounds = {}
    for c in feats_tr.columns:
        lo, hi = np.quantile(feats_tr[c].values, [q_low, q_high])
        bounds[c] = (float(lo), float(hi))
    y_tr_raw = tr_fe[TARGET_COL].values
    y_lo, y_hi = np.quantile(y_tr_raw, [0.005, 0.995])
    keep = (y_tr_raw >= y_lo) & (y_tr_raw <= y_hi)
    for c,(lo,hi) in bounds.items():
        keep &= (feats_tr[c].values >= lo) & (feats_tr[c].values <= hi)
    feats_tr = feats_tr.loc[keep].reset_index(drop=True)
    tr_fe = tr_fe.loc[keep].reset_index(drop=True)
    y_tr_raw = tr_fe[TARGET_COL].values
    # test clip
    for c,(lo,hi) in bounds.items():
        if c in feats_te.columns:
            feats_te[c] = feats_te[c].clip(lower=lo, upper=hi)

    # 타깃 설정(Δ-타깃 옵션)
    if USE_DELTA_TARGET:
        y_tr = tr_fe[TARGET_COL].values - tr_fe[CUR_WARP].values
        y_te = te_fe[TARGET_COL].values  # 평가 시 원공간에서 비교
        reconstruct_te = te_fe[CUR_WARP].values
        reconstruct_tr = tr_fe[CUR_WARP].values
    else:
        y_tr = tr_fe[TARGET_COL].values
        y_te = te_fe[TARGET_COL].values
        reconstruct_te = None
        reconstruct_tr = None

    # 상관 기반 K-best(필수: CUR_WARP)
    mand = [CUR_WARP] if CUR_WARP in feats_tr.columns else []
    # 상관계수 계산
    Xc = feats_tr.copy()
    x = Xc.values
    y0 = y_tr - y_tr.mean()
    ys = y0.std(); ys = ys if ys >= 1e-12 else 1e-12
    xm = x.mean(axis=0); xs = x.std(axis=0); xs[xs < 1e-12] = 1e-12
    xr = (x - xm) / xs; yr = y0 / ys
    r = (xr.T @ yr) / (len(y_tr) - 1)
    r2 = np.clip(r**2, 0.0, 1.0 - 1e-12)
    F = (r2 / (1.0 - r2)) * (len(y_tr) - 2)
    cols_all = list(feats_tr.columns)
    F_map = {c: F[i] for i,c in enumerate(cols_all)}
    others = [c for c in cols_all if c not in mand]
    k_remain = max(1, min(SELECT_K - len(mand), len(others)))
    order = sorted(others, key=lambda c: F_map[c], reverse=True)[:k_remain]
    selected_cols = mand + order

    Xtr = feats_tr[selected_cols].values
    Xte = feats_te[selected_cols].values

    # 이분산 가중(옵션)
    sample_weight = None
    if USE_HETERO_W:
        w_tr, _ = _oof_pair_std_weights(tr_fe, te_fe, n0=PAIR_EB_N0, clip=WEIGHT_CLIP)
        sample_weight = w_tr / (np.mean(w_tr) + 1e-8)

    # 모델 구성(단조 제약은 CUR_WARP 위치 기반)
    mono_idx = selected_cols.index(CUR_WARP) if (USE_MONO_CUR and CUR_WARP in selected_cols) else None
    model = _build_model(algorithm, seed=seed, mono_cur_idx=mono_idx, n_features=len(selected_cols))

    # 내부 plate holdout로 early stopping
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    grp_tr = tr_fe[PLATE_COL].astype(str).values
    tr_i, va_i = next(gss_inner.split(Xtr, y_tr, groups=grp_tr))
    Xfit, yfit = Xtr[tr_i], y_tr[tr_i]
    Xval, yval = Xtr[va_i], y_tr[va_i]
    sw_fit = None if sample_weight is None else sample_weight[tr_i]

    if algorithm == "XGBoost":
        try:
            model.fit(Xfit, yfit, sample_weight=sw_fit,
                      eval_set=[(Xval, yval)], early_stopping_rounds=50, verbose=False)
        except TypeError:
            model.fit(Xfit, yfit, sample_weight=sw_fit)
    else:
        model.fit(Xfit, yfit, sample_weight=sw_fit)

    # 예측/복원
    y_tr_pred = model.predict(Xtr)
    y_te_pred = model.predict(Xte)
    if USE_DELTA_TARGET:
        y_tr_pred = reconstruct_tr + y_tr_pred
        y_te_pred = reconstruct_te + y_te_pred
        y_tr_eval = tr_fe[TARGET_COL].values
        y_te_eval = te_fe[TARGET_COL].values
    else:
        y_tr_eval = y_tr
        y_te_eval = y_te

    res = {
        "train_r2":  float(r2_score(y_tr_eval, y_tr_pred)),
        "test_r2":   float(r2_score(y_te_eval, y_te_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_tr_eval, y_tr_pred))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te_eval, y_te_pred))),
        "train_samples": int(len(y_tr_eval)), "test_samples": int(len(y_te_eval)),
    }

    # 저장: 모델 + 선택컬럼/중앙값 + 테스트 판(앵커)
    joblib.dump(model,                MODELS_DIR / f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(selected_cols,        MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl")
    joblib.dump(med,                  MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl")
    joblib.dump(sorted(list(test_plates)), MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl")

    # 감사용 패키지(빠른 퍼뮤테이션에 활용)
    audit = {
        "Xtr": Xtr, "y_tr_eval": y_tr_eval if not USE_DELTA_TARGET else tr_fe[TARGET_COL].values,
        "Xte": Xte, "y_te_eval": y_te_eval if not USE_DELTA_TARGET else te_fe[TARGET_COL].values,
        "selected_cols": selected_cols, "algorithm": algorithm
    }
    return res, set(test_plates), audit

# ---------------- 빠른 퍼뮤테이션 누수 감사 --------------------
def quick_perm_leak_r2(audit_pack: dict, seed: int = SEED+7) -> float:
    """Train y를 무작위로 섞어 학습 → Test에서 R² 측정(≈0 기대)."""
    rng = np.random.RandomState(seed)
    Xtr = audit_pack["Xtr"]; Xte = audit_pack["Xte"]
    ytr_perm = rng.permutation(audit_pack["y_tr_eval"])
    yte = audit_pack["y_te_eval"]
    algo = audit_pack["algorithm"]
    # 가벼운 설정
    if algo == "XGBoost":
        from xgboost import XGBRegressor
        m = XGBRegressor(
            objective="reg:squarederror", random_state=seed, tree_method="hist",
            n_estimators=200, learning_rate=0.08, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, n_jobs=0, eval_metric="rmse"
        )
    else:
        m = RandomForestRegressor(random_state=seed, n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1)
    m.fit(Xtr, ytr_perm)
    yhat = m.predict(Xte)
    return float(r2_score(yte, yhat))

# ----------------- rollforward 유틸(셀렉터 無) ------------------
def _load_bundle(prefix: str):
    return {
        "model":    joblib.load(MODELS_DIR / f"{prefix}_regressor_tplus1.pkl"),
        "cols":     joblib.load(MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl"),
    }

def _predict_next_from_row(feat_row: pd.Series, bundle: dict) -> float:
    # 저장된 feature_cols 순서대로 재배열 후 중앙값 대치
    desired = list(bundle["cols"])
    X_row = pd.DataFrame([feat_row.reindex(desired)], columns=desired).fillna(bundle["medians"])
    return float(bundle["model"].predict(X_row.values)[0])

def rollforward_predict_all_plates(all_df: pd.DataFrame, e2x_bundle: dict, x2e_bundle: dict) -> pd.DataFrame:
    outs = []
    for plate in sorted(all_df[PLATE_COL].dropna().astype(str).unique()):
        sub = all_df[all_df[PLATE_COL].astype(str) == plate].sort_values(PASS_COL)
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index:
                continue
            row_t = idx.loc[p]
            feat_row = row_t.drop(labels=[TARGET_COL], errors="ignore")
            bundle = e2x_bundle if (p % 2 == 1) else x2e_bundle
            pred = _predict_next_from_row(feat_row, bundle)
            gt_next = idx.loc[p+1].get(CUR_WARP, None)
            outs.append({PLATE_COL: plate, "from_pass": p, "to_pass": p+1, "pred": pred, "gt_next": gt_next})
    return pd.DataFrame(outs) if outs else pd.DataFrame(columns=[PLATE_COL,"from_pass","to_pass","pred","gt_next"])

# --- (리포팅) 간단 Δ-분석 요약 --------------------------------
def analyze_transition_variability(final_df: pd.DataFrame) -> dict:
    try:
        from scipy.stats import kruskal, levene
    except Exception:
        kruskal = levene = None
    df = final_df.copy()
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["pair"] = _pair_str(df)

    def _stats(sub):
        g = sub.groupby("pair")["delta"]
        stat = g.agg(count="count", mean="mean", std="std", median="median",
                     q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index()
        if stat["std"].notna().any():
            std_ratio = (stat["std"].max() / stat["std"].replace(0, np.nan).min())
        else:
            std_ratio = np.nan
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

# --- 모든 쌍 Δ 통계 CSV(참고) -----------------------------------
def export_pairwise_delta_stats(final_df: pd.DataFrame):
    df = final_df.copy()
    df["pair"] = _pair_str(df)
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["transition"] = np.where(df[PASS_COL] % 2 == 1, "E2X", "X2E")
    g = df.groupby(["transition","pair"])["delta"]
    stats = g.agg(
        n="size",
        mean="mean",
        std="std",
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    ).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["n"])
    stats.loc[stats["n"] <= 1, ["se"]] = np.nan
    stats["ci95_lo"] = stats["mean"] - 1.96*stats["se"]
    stats["ci95_hi"] = stats["mean"] + 1.96*stats["se"]
    stats["n_flag"]  = np.where(stats["n"] < 30, "small-n(<30)", "ok")
    out = REPORTS_DIR / "pair_delta_stats.csv"
    stats.to_csv(out, index=False)
    print(f"✓ 모든 쌍 Δ 통계 저장: {out}")
    return stats

# ------------------------ 요약 저장 ---------------------------
def write_summary(summary_data, summary_path: Path, timestamp: datetime, experiment_id: str):
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
    if rf_r2 is not None and np.isfinite(rf_r2):
        lines.append(f"[롤포워드 R2(전체 판, t→t+1)] {rf_r2:.4f}")

    # 전략/검증 플래그
    lines.append("\n[실행 설정/플래그]")
    lines.append(f"- PERMUTE_TARGET: {int(PERMUTE_TARGET)}")
    lines.append(f"- LEAK_AUTOCHECK: {int(LEAK_AUTOCHECK)}")
    lines.append(f"- USE_DELTA_TARGET: {int(USE_DELTA_TARGET)} | USE_PAIR_TE: {int(USE_PAIR_TE)} | USE_HETERO_W: {int(USE_HETERO_W)} | USE_MONO_CUR: {int(USE_MONO_CUR)} | USE_RM_TRAINONLY: {int(USE_RM_TRAINONLY)}")

    # 빠른 퍼뮤 감사
    perm_e2x = summary_data.get("perm_e2x", None)
    perm_x2e = summary_data.get("perm_x2e", None)
    if (perm_e2x is not None) or (perm_x2e is not None):
        lines.append("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        if perm_e2x is not None: lines.append(f"- E2X permuted R²: {perm_e2x:.4f}")
        if perm_x2e is not None: lines.append(f"- X2E permuted R²: {perm_x2e:.4f}")

    # Δ-분석 요약
    v = summary_data.get("variability", {})
    for tag in ["e2x","x2e"]:
        vv = v.get(tag, {})
        lines.append(f"\n[Δ-분석 {tag.upper()}] std_max/min={vv.get('std_ratio', np.nan):.3f} | "
                     f"Kruskal p={vv.get('kw_p', np.nan)} | Levene p={vv.get('lev_p', np.nan)}")
        st = vv.get("stats")
        if isinstance(st, pd.DataFrame) and "mean" in st.columns and "count" in st.columns:
            top3 = st.assign(absmean=st["mean"].abs()).nlargest(3, "absmean")[["pair","count","mean","std","median"]]
            lines.append("  - |Δ| 상위 3쌍:")
            for _, r in top3.iterrows():
                lines.append(f"    · {r['pair']}: n={int(r['count'])}, mean={r['mean']:.3f}, std={r['std']:.3f}, median={r['median']:.3f}")

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

        # (옵션) 퍼뮤테이션 모드: 전체 타깃 무작위화
        if PERMUTE_TARGET:
            rng = np.random.RandomState(SEED)
            final_df[TARGET_COL] = rng.permutation(final_df[TARGET_COL].values)
            print("[PERMUTE] Target shuffled for leak sanity check.")

        # 2) 전이 세트 분리
        e2x_df, x2e_df = split_cross_transitions(final_df)

        # 리포팅용 간단 Δ-분석
        variability = analyze_transition_variability(final_df)

        # 3) 알고리즘 비교(plate-group CV)
        e2x_algo, _ = compare_algorithms(e2x_df, "E2X")
        x2e_algo, _ = compare_algorithms(x2e_df, "X2E")

        # 4) 학습/평가/저장
        e2x_res, e2x_test_plates, e2x_audit = train_model(e2x_df, "E2X", e2x_algo)
        x2e_res, x2e_test_plates, x2e_audit = train_model(x2e_df, "X2E", x2e_algo)

        e2x_res["algorithm"] = e2x_algo
        x2e_res["algorithm"] = x2e_algo

        average_test_r2 = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))

        # 5) 롤포워드(전체 판): 저장된 feature_cols/medians 사용
        all_fe = final_df.copy()
        e2x_bundle = _load_bundle("e2x")
        x2e_bundle = _load_bundle("x2e")
        rf_pred = rollforward_predict_all_plates(all_fe, e2x_bundle, x2e_bundle)
        rf_out = PROCESSED_DIR / "rollforward_predictions_tplus1_cross.csv"
        rf_pred.to_csv(rf_out, index=False)
        print(f"✓ 번갈아 예측 결과 저장: {rf_out}")

        rf_mask = rf_pred["gt_next"].notna()
        rollforward_r2 = float(r2_score(rf_pred.loc[rf_mask, "gt_next"], rf_pred.loc[rf_mask, "pred"])) if rf_mask.any() else np.nan

        # 6) 빠른 퍼뮤테이션 감사(기본 ON)
        perm_e2x = perm_x2e = None
        if LEAK_AUTOCHECK and not PERMUTE_TARGET:
            perm_e2x = quick_perm_leak_r2(e2x_audit)
            perm_x2e = quick_perm_leak_r2(x2e_audit)
            print(f"[Leak Audit] E2X perm R²={perm_e2x:.4f} | X2E perm R²={perm_x2e:.4f} (≃0 기대)")

        # 7) 요약/로그/코드 스냅샷
        summary = {
            "data_artifacts": [
                f"Final merged(t+1): {PROCESSED_DIR / 'final_merged_data_regression_tplus1.csv'}",
                f"E2X raw: {PROCESSED_DIR / 'e2x_raw_tplus1.csv'}",
                f"X2E raw: {PROCESSED_DIR / 'x2e_raw_tplus1.csv'}",
                f"Rollforward preds: {rf_out}",
                f"Archived script: {archived_script_path}",
            ],
            "models": {"E2X": e2x_res, "X2E": x2e_res},
            "average_test_r2": average_test_r2,
            "rollforward_r2": rollforward_r2,
            "variability": variability,
            "perm_e2x": perm_e2x, "perm_x2e": perm_x2e
        }
        write_summary(summary, summary_path, timestamp, experiment_id)

        log_row = {
            "experiment_id": experiment_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "e2x_algorithm": e2x_algo, "x2e_algorithm": x2e_algo,
            "e2x_test_r2": e2x_res["test_r2"], "x2e_test_r2": x2e_res["test_r2"],
            "average_test_r2": average_test_r2,
            "e2x_train_r2": e2x_res["train_r2"], "x2e_train_r2": x2e_res["train_r2"],
            "rollforward_r2": rollforward_r2,
            "e2x_perm_r2": perm_e2x, "x2e_perm_r2": perm_x2e
        }
        _append_success_log(log_row)
        shutil.copy2(SCRIPT_PATH, archived_script_path)
        print(f"✓ 코드 아카이브 저장: {archived_script_path}")

        success = True

        # 상위 5 실험 출력
        log_df = pd.read_csv(EXPERIMENT_LOG)
        top = log_df.sort_values("average_test_r2", ascending=False).head(5)
        print("\n[상위 5 실험] 평균 Test R2 내림차순")
        print(top[["experiment_id","average_test_r2","e2x_test_r2","x2e_test_r2","rollforward_r2"]].to_string(index=False))

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
