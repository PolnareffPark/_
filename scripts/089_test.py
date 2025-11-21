# -*- coding: utf-8 -*-
"""
084 실험 개선본
- 누수 안전(t→t+1) 교차 전이 회귀 파이프라인 (E→X / X→E)
- Baseline 흐름 유지 + 불필요 코드 정리 + 플래그(DT/TE/W/MONO/RM) 완전 반영
- E2X는 XGBoost 고정, X2E는 XGBoost/RandomForest 선택 가능 (X2E_ALGO)
- 앵커 고정 분할, 메트릭 CSV/요약 저장, 퍼뮤테이션 누수 감사, Strict RF R² 저장
"""

import os
import re
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import traceback

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------- 경로/상수 ----------------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
REPORTS_SUMMARY_DIR = REPORTS_DIR / "summaries"
REPORTS_CODE_DIR = REPORTS_DIR / "scripts"
REPORTS_METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = Path("models")

for d in [PROCESSED_DIR, REPORTS_DIR, REPORTS_SUMMARY_DIR, REPORTS_CODE_DIR, REPORTS_METRICS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_COL = "warping_index_target"   # t+1
CUR_WARP   = "warping_index_current_pass"
PASS_COL   = "FM_PASS NO N"
PLATE_COL  = "FM_날판번호"
MONTH_COL  = "FM_압연월"

EXCLUDE_BASE = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}

# ---------------------------- 환경 플래그 ----------------------------
def _env_on(name: str, default="0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1","true","yes","y"}

RUN_NAME         = os.environ.get("RUN_NAME", "_V2_pass_t_plus_1_cross")
ANCHOR_TAG       = os.environ.get("ANCHOR_TAG", RUN_NAME)  # 앵커 파일명 prefix
SEED             = int(os.environ.get("SEED", "42"))

# 옵션 플래그
PERMUTE_TARGET   = _env_on("PERMUTE_TARGET", "0")
LEAK_AUTOCHECK   = _env_on("LEAK_AUTOCHECK", "1")

USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET", "0")  # DT
USE_PAIR_TE      = _env_on("USE_PAIR_TE", "0")       # TE
USE_HETERO_W     = _env_on("USE_HETERO_W", "0")      # W
USE_MONO_CUR     = _env_on("USE_MONO_CUR", "0")      # MONO(+1 on CUR_WARP, XGB만 해당)
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY", "0")  # RM

# X2E 알고리즘 선택 ("XGBoost" | "RandomForest")
X2E_ALGO         = os.environ.get("X2E_ALGO", "XGBoost").strip()

# ---------------------------- 유틸/가드 ----------------------------
def _hard_leak_guard(cols: list[str]) -> list[str]:
    """타깃/예측 냄새 열 제거. (pair_delta_te는 제외해야 하므로 'te_'는 제거하지 않음)"""
    bad_kw = ("target", "_target", "label", "_y", "oof", "pred", "_hat")
    safe=[]
    for c in cols:
        low=c.lower()
        if any(k in low for k in bad_kw):
            continue
        safe.append(c)
    return safe

def _pair_str(df: pd.DataFrame) -> pd.Series:
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

def _next_experiment_id() -> str:
    nums=[]
    for f in REPORTS_SUMMARY_DIR.glob("*.txt"):
        m=re.match(r"(\d{3})_", f.name)
        if m:
            try: nums.append(int(m.group(1)))
            except: pass
    return f"{(max(nums)+1) if nums else 1:03d}"

# --------------------------- 데이터 병합 ---------------------------
def load_and_merge_data_tplus1() -> pd.DataFrame:
    gt_entry = pd.read_csv(DATA_DIR/"entry_direction_results.csv")
    gt_exit  = pd.read_csv(DATA_DIR/"exit_direction_results.csv")
    gt = pd.concat([gt_entry, gt_exit], ignore_index=True)

    fm = pd.read_csv(DATA_DIR/"posco2_105190.csv")

    gt["extracted_plate"] = gt["filename"].str.extract(r"(PB\d+)")
    gt["extracted_pass"]  = gt["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    merged = pd.merge(
        gt, fm,
        left_on=["extracted_plate","extracted_pass"],
        right_on=[PLATE_COL, PASS_COL],
        how="inner"
    ).sort_values([PLATE_COL, PASS_COL]).reset_index(drop=True)

    # t+1 타깃
    merged[TARGET_COL] = merged.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged = merged.rename(columns={"warping_index": CUR_WARP})

    # PASS(t)→(t+1) 불연속은 타깃 제거
    next_pass = merged.groupby(PLATE_COL)[PASS_COL].shift(-1)
    nonconsec = (next_pass.notna()) & (next_pass != merged[PASS_COL] + 1)
    merged.loc[nonconsec, TARGET_COL] = np.nan

    before = len(merged)
    merged = merged.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # 정리
    merged = merged.drop(columns=[
        "filename","quality_grade","quality_grade_current_pass","direction",
        "extracted_plate","extracted_pass"
    ], errors="ignore")

    PROCESSED_DIR.joinpath("final_merged_data_regression_tplus1.csv").write_text(
        merged.to_csv(index=False), encoding="utf-8"
    )
    return merged

def split_cross_transitions(df: pd.DataFrame):
    e2x = df[df[PASS_COL] % 2 == 1].copy()
    x2e = df[df[PASS_COL] % 2 == 0].copy()
    PROCESSED_DIR.joinpath("e2x_raw_tplus1.csv").write_text(e2x.to_csv(index=False))
    PROCESSED_DIR.joinpath("x2e_raw_tplus1.csv").write_text(x2e.to_csv(index=False))
    return e2x, x2e

# ------------------------ Train-only RM 집계 ------------------------
def compute_rm_stats_for_plates(plates: set) -> pd.DataFrame:
    rm = pd.read_csv(DATA_DIR/"posco1_105190.csv")
    rm = rm[rm["RM_날판번호"].astype(str).isin({str(p) for p in plates})].copy()
    if rm.empty:
        return pd.DataFrame({"RM_날판번호":[]})
    num_cols = rm.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"RM_날판번호","RM_압연Pass번호","warping_index"}
    stat_cols = [c for c in num_cols if c not in drop_like and not c.endswith("_target")]
    agg = rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(6)
    agg.columns = [f"RM_{c[0]}_{c[1]}" for c in agg.columns]
    return agg.reset_index()

def attach_train_only_rm_stats(tr: pd.DataFrame, te: pd.DataFrame):
    if not USE_RM_TRAINONLY:
        return tr, te
    plates_tr = set(tr[PLATE_COL].astype(str).unique())
    rm_stats = compute_rm_stats_for_plates(plates_tr)
    tr = pd.merge(tr, rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left").drop(columns=["RM_날판번호"], errors="ignore")
    te = pd.merge(te, rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left").drop(columns=["RM_날판번호"], errors="ignore")
    if not rm_stats.empty:
        med = rm_stats.drop(columns=["RM_날판번호"], errors="ignore").median(numeric_only=True)
        for c,v in med.items():
            if c in te.columns: te[c] = te[c].fillna(float(v))
            if c in tr.columns: tr[c] = tr[c].fillna(float(v))
    return tr, te

# ----------------------- 베이스 FE(train 기준) -----------------------
def base_feature_engineering(tr_raw: pd.DataFrame, te_raw: pd.DataFrame):
    tr = tr_raw.copy(); te = te_raw.copy()
    fm_cols_tr = [c for c in tr.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    fm_cols_te = [c for c in te.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    fm_cols = sorted(set(fm_cols_tr) | set(fm_cols_te))

    def _fe(d: pd.DataFrame, pass_max: float):
        if fm_cols:
            blk = d.reindex(columns=fm_cols, fill_value=np.nan)
            d["FM_mean"]  = blk.mean(axis=1)
            d["FM_std"]   = blk.std(axis=1)
            d["FM_max"]   = blk.max(axis=1)
            d["FM_min"]   = blk.min(axis=1)
            d["FM_range"] = d["FM_max"] - d["FM_min"]
            d["FM_cv"]    = d["FM_std"] / (d["FM_mean"].abs() + 1e-8)
        d["FM_PASS_squared"]  = d[PASS_COL] ** 2
        d["FM_PASS_progress"] = d[PASS_COL] / (pass_max if pass_max>0 else 1.0)
        return d

    pass_max_tr = float(tr[PASS_COL].max()) if len(tr) else 1.0
    tr = _fe(tr, pass_max_tr)
    te = _fe(te, pass_max_tr)
    return tr, te

# ---------------------- OOF TE / OOF Weights ----------------------
def add_pair_delta_te_oof(tr: pd.DataFrame, te: pd.DataFrame, n0=50):
    if not USE_PAIR_TE:
        return tr.copy(), te.copy()
    dtr = tr.copy(); dte = te.copy()
    dtr["pair"] = _pair_str(dtr)
    gkf = GroupKFold(n_splits=3)
    groups = dtr[PLATE_COL].astype(str).values
    delta = (dtr[TARGET_COL] - dtr[CUR_WARP]).values
    oof = np.zeros(len(dtr))
    for tr_i, va_i in gkf.split(dtr, delta, groups):
        sub = dtr.iloc[tr_i]
        gmean = sub[TARGET_COL].sub(sub[CUR_WARP]).mean()
        stat = sub.groupby("pair")[TARGET_COL].agg(n="size", mean=lambda s: (s - sub.loc[s.index, CUR_WARP]).mean()).reset_index()
        stat["te"] = (stat["n"]/(stat["n"]+n0))*stat["mean"] + (n0/(stat["n"]+n0))*gmean
        m = dict(zip(stat["pair"], stat["te"]))
        oof[va_i] = dtr.iloc[va_i]["pair"].map(m).fillna(gmean).values
    # test 매핑: full-train
    gmean = (dtr[TARGET_COL] - dtr[CUR_WARP]).mean()
    stat = dtr.groupby("pair")[TARGET_COL].agg(n="size", mean=lambda s: (s - dtr.loc[s.index, CUR_WARP]).mean()).reset_index()
    stat["te"] = (stat["n"]/(stat["n"]+n0))*stat["mean"] + (n0/(stat["n"]+n0))*gmean
    m_full = dict(zip(stat["pair"], stat["te"]))
    dte["pair_delta_te"] = _pair_str(dte).map(m_full).fillna(gmean).values
    dtr["pair_delta_te"] = oof
    dtr = dtr.drop(columns=["pair"])
    return dtr, dte

def compute_hetero_weights_oof(tr: pd.DataFrame, te: pd.DataFrame, n0=50, clip=(0.25, 3.0)):
    if not USE_HETERO_W:
        return None
    dtr = tr.copy()
    dtr["pair"] = _pair_str(dtr)
    delta = dtr[TARGET_COL] - dtr[CUR_WARP]
    gkf = GroupKFold(n_splits=3)
    groups = dtr[PLATE_COL].astype(str).values
    oof_std = np.zeros(len(dtr))
    for tr_i, va_i in gkf.split(dtr, delta.values, groups):
        sub = dtr.iloc[tr_i]
        gs = sub["pair"].map(sub.groupby("pair")["pair"].size()).std(ddof=1)  # fallback에 쓸 전역 표준편차 proxy
        stat = sub.groupby("pair")["pair"].agg(n="size").reset_index()
        # Δ 표준편차 계산
        sub_delta = sub[TARGET_COL] - sub[CUR_WARP]
        std_tbl = sub.join(sub_delta.rename("delta")).groupby("pair")["delta"].std().rename("std").reset_index()
        stat = pd.merge(stat, std_tbl, on="pair", how="left")
        w = stat["n"]/(stat["n"]+n0)
        stat["std_sh"] = w*stat["std"].fillna(sub_delta.std(ddof=1)) + (1-w)*sub_delta.std(ddof=1)
        m = dict(zip(stat["pair"], stat["std_sh"]))
        oof_std[va_i] = dtr.iloc[va_i]["pair"].map(m).fillna(sub_delta.std(ddof=1)).values
    gs = (dtr[TARGET_COL] - dtr[CUR_WARP]).std(ddof=1)
    w_tr = np.clip((float(gs)/(oof_std + 1e-6)), clip[0], clip[1])
    return w_tr / (w_tr.mean() + 1e-8)

# ---------------------- 피처 선택(SimpleKBest) ----------------------
class SimpleKBest:
    """상관 기반 K-best (상수/비유한 제거, 안전). transform()은 선택 열 순서대로 ndarray 반환."""
    def __init__(self, k=120, mandatory=None):
        self.k = int(max(1, k))
        self.mandatory = list(mandatory or [])
        self.selected_cols_: list[str] = []

    def fit(self, X_df: pd.DataFrame, y: np.ndarray):
        X = X_df.select_dtypes(include=[np.number]).copy()
        cols_all = X.columns.tolist()
        # 상수/비유한 제거
        finite_mask = np.isfinite(X.values).all(axis=0)
        var = X.var(numeric_only=True).reindex(cols_all).fillna(0.0).values
        keep_mask = finite_mask & (var > 1e-12)
        cols = [c for c,m in zip(cols_all, keep_mask) if m]
        if not cols:
            self.selected_cols_ = []
            return self
        Xc = X[cols].fillna(X[cols].median(numeric_only=True))
        y0 = y - y.mean()
        ys = max(1e-12, y0.std())
        xm = Xc.values.mean(axis=0)
        xs = Xc.values.std(axis=0); xs[xs<1e-12]=1e-12
        xr = (Xc.values - xm)/xs; yr = y0/ys
        r = (xr.T @ yr) / (len(y)-1)
        r2 = np.clip(r**2, 0.0, 1.0-1e-12)
        F = (r2/(1.0-r2)) * (len(y)-2)

        mand = [c for c in self.mandatory if c in cols]
        others = [c for c in cols if c not in mand]
        k_rem = max(1, min(self.k - len(mand), len(others)))
        # others 인덱스 매핑
        idx = {c:i for i,c in enumerate(others)}
        order = np.argsort([F[idx[c]] for c in others])[::-1] if others else []
        self.selected_cols_ = mand + [others[i] for i in order[:k_rem]]
        return self

    def transform(self, X_df: pd.DataFrame):
        X_df = X_df.copy()
        for c in self.selected_cols_:
            if c not in X_df.columns:
                X_df[c] = np.nan
        return X_df[self.selected_cols_].values

# ----------------------- 앵커 고정 분할 -----------------------
def split_by_anchor(df: pd.DataFrame, direction_key: str, test_size=0.2, seed=SEED):
    assert direction_key in {"E2X","X2E"}
    key = "e2x" if direction_key=="E2X" else "x2e"
    anchor_file = MODELS_DIR / f"{ANCHOR_TAG}__{key}_test_plates_tplus1.pkl"
    groups = df[PLATE_COL].astype(str).values
    if anchor_file.exists():
        te_plates = set(joblib.load(anchor_file))
        m_te = df[PLATE_COL].astype(str).isin(te_plates)
        return df[~m_te].copy(), df[m_te].copy(), te_plates
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    X_dummy = df.drop(columns=[TARGET_COL], errors="ignore")
    y_dummy = df[TARGET_COL].values
    tr_i, te_i = next(gss.split(X_dummy, y_dummy, groups))
    te_plates = sorted(df.iloc[te_i][PLATE_COL].astype(str).unique().tolist())
    joblib.dump(te_plates, anchor_file)
    return df.iloc[tr_i].copy(), df.iloc[te_i].copy(), set(te_plates)

# ----------------------- 학습 루틴(한 방향) -----------------------
@dataclass
class TrainResult:
    model: object
    selector: SimpleKBest
    feature_cols: list
    medians: dict
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float
    test_plates: set

def _build_model(name: str, n_features: int, mono_idx: int | None):
    if name == "RandomForest":
        return RandomForestRegressor(
            random_state=SEED, n_estimators=600, n_jobs=-1,
            max_depth=20, min_samples_leaf=5, max_features="sqrt"
        )
    # XGBoost(default)
    params = dict(
        random_state=SEED, tree_method="hist", n_estimators=700, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
        n_jobs=0, eval_metric="rmse",
        objective=("reg:squarederror")
    )
    if USE_MONO_CUR and (mono_idx is not None):
        cons = [0]*n_features
        if 0 <= mono_idx < n_features: cons[mono_idx]=1
        params["monotone_constraints"] = tuple(cons)
    return XGBRegressor(**params)

def _ensure_numeric_X(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in EXCLUDE_BASE]
    cols = _hard_leak_guard(cols)
    X = df[cols].select_dtypes(include=[np.number]).copy()
    return X

def train_one_direction(raw_df: pd.DataFrame, direction_key: str, algo_name: str) -> TrainResult:
    # 1) 앵커 분할
    tr_raw, te_raw, te_plates = split_by_anchor(raw_df, direction_key)

    # 2) RM train-only
    tr_raw, te_raw = attach_train_only_rm_stats(tr_raw, te_raw)

    # 3) FE (train 기준 PASS 스케일)
    tr, te = base_feature_engineering(tr_raw, te_raw)

    # 4) TE/Weights (OOF)
    tr, te = add_pair_delta_te_oof(tr, te)
    sample_weight = compute_hetero_weights_oof(tr, te)

    # 5) X, y 구축 + 결측 대치(Train 중앙값)
    ytr = tr[TARGET_COL].values
    yte = te[TARGET_COL].values

    # Δ-타깃
    if USE_DELTA_TARGET:
        tr["_target"] = tr[TARGET_COL] - tr[CUR_WARP]
        te["_target"] = te[TARGET_COL] - te[CUR_WARP]
        ytr = tr["_target"].values
        yte = te["_target"].values
        mandatory = [CUR_WARP] + (["pair_delta_te"] if USE_PAIR_TE else [])
    else:
        mandatory = [CUR_WARP] + (["pair_delta_te"] if USE_PAIR_TE else [])

    Xtr_df = _ensure_numeric_X(tr)
    Xte_df = _ensure_numeric_X(te)
    med = Xtr_df.median(numeric_only=True).to_dict()
    Xtr_df = Xtr_df.fillna(med)
    Xte_df = Xte_df.reindex(columns=Xtr_df.columns, fill_value=np.nan).fillna(med)

    # 6) 이상치 가드: train 분위 경계로 train filter, test clip
    q_low, q_high = 0.01, 0.99
    keep = np.ones(len(tr), dtype=bool)
    # y 범위 제한(Δ/원타깃 어느쪽이든)
    y_q_lo, y_q_hi = np.quantile(ytr, [0.005, 0.995])
    keep &= (ytr >= y_q_lo) & (ytr <= y_q_hi)
    bounds={}
    for c in Xtr_df.columns:
        lo,hi = np.quantile(Xtr_df[c].values, [q_low, q_high])
        bounds[c]=(float(lo), float(hi))
        keep &= (Xtr_df[c].values>=lo) & (Xtr_df[c].values<=hi)
    Xtr_df = Xtr_df.loc[keep].reset_index(drop=True)
    ytr = ytr[keep]
    if sample_weight is not None:
        sample_weight = sample_weight[keep]
    # test clip
    for c,(lo,hi) in bounds.items():
        Xte_df[c] = Xte_df[c].clip(lower=lo, upper=hi)

    # 7) 피처 선택
    selector = SimpleKBest(k=min(120, Xtr_df.shape[1]), mandatory=[c for c in mandatory if c in Xtr_df.columns]).fit(Xtr_df, ytr)
    chosen = selector.selected_cols_
    Xtr = selector.transform(Xtr_df)
    Xte = selector.transform(Xte_df)

    mono_idx = (chosen.index(CUR_WARP) if (USE_MONO_CUR and CUR_WARP in chosen and algo_name=="XGBoost") else None)
    model = _build_model(algo_name, n_features=len(chosen), mono_idx=mono_idx)

    # 8) 학습(조용히, early stopping 미사용: 버전 호환)
    if sample_weight is not None:
        model.fit(Xtr, ytr, sample_weight=sample_weight)
    else:
        model.fit(Xtr, ytr)

    # 9) 평가
    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)
    # Δ-타깃이면 복원(보고용)
    if USE_DELTA_TARGET:
        # 보고용으로 복원한 R2는 별도 의미가 다르지만, 비교 일관성을 위해 Δ 그대로의 R²를 저장
        pass

    tr_r2 = float(r2_score(ytr, yhat_tr))
    te_r2 = float(r2_score(yte, yhat_te))
    tr_rmse = float(np.sqrt(mean_squared_error(ytr, yhat_tr)))
    te_rmse = float(np.sqrt(mean_squared_error(yte, yhat_te)))

    # 10) 저장(번들)
    prefix = "e2x" if direction_key=="E2X" else "x2e"
    joblib.dump(model, MODELS_DIR/f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(selector, MODELS_DIR/f"{prefix}_selector_tplus1.pkl")
    joblib.dump(chosen, MODELS_DIR/f"{prefix}_feature_cols_tplus1.pkl")
    joblib.dump(med, MODELS_DIR/f"{prefix}_feature_medians_tplus1.pkl")
    # 앵커 plate 저장은 split_by_anchor에서 이미 수행

    return TrainResult(
        model=model, selector=selector, feature_cols=chosen, medians=med,
        train_r2=tr_r2, test_r2=te_r2, train_rmse=tr_rmse, test_rmse=te_rmse,
        test_plates=te_plates
    )

# --------------------- 롤포워드(Strict Test Only) ---------------------
def _load_bundle(prefix: str):
    return {
        "model": joblib.load(MODELS_DIR/f"{prefix}_regressor_tplus1.pkl"),
        "selector": joblib.load(MODELS_DIR/f"{prefix}_selector_tplus1.pkl"),
        "cols": joblib.load(MODELS_DIR/f"{prefix}_feature_cols_tplus1.pkl"),
        "med": joblib.load(MODELS_DIR/f"{prefix}_feature_medians_tplus1.pkl"),
    }

def _predict_next_from_row(row: pd.Series, bundle: dict) -> float:
    # selector는 SimpleKBest: DataFrame → transform → ndarray
    desired = bundle["cols"]
    X_row = pd.DataFrame([row.reindex(desired)], columns=desired).fillna(bundle["med"])
    X_sel = bundle["selector"].transform(X_row)
    return float(bundle["model"].predict(X_sel)[0])

def rollforward_r2_strict(all_df: pd.DataFrame) -> float:
    e2x = _load_bundle("e2x")
    x2e = _load_bundle("x2e")
    e2x_test = set(joblib.load(MODELS_DIR/f"{ANCHOR_TAG}__e2x_test_plates_tplus1.pkl"))
    x2e_test = set(joblib.load(MODELS_DIR/f"{ANCHOR_TAG}__x2e_test_plates_tplus1.pkl"))
    test_inter = e2x_test & x2e_test
    if not test_inter:
        return float("nan")

    rows=[]
    for plate, sub in all_df.groupby(PLATE_COL):
        if str(plate) not in test_inter:
            continue
        sub = sub.sort_values(PASS_COL).copy()
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index: continue
            row_t = idx.loc[p].drop(labels=[TARGET_COL], errors="ignore")
            bundle = e2x if (p%2==1) else x2e
            pred = _predict_next_from_row(row_t, bundle)
            gt   = float(idx.loc[p+1].get(CUR_WARP, np.nan))
            if np.isfinite(gt):
                rows.append({"pred": pred, "gt": gt})
    if not rows:
        return float("nan")
    df = pd.DataFrame(rows)
    return float(r2_score(df["gt"], df["pred"]))

# --------------------- 퍼뮤테이션 누수 감사(빠름) ---------------------
def quick_perm_r2(tr_df: pd.DataFrame, te_df: pd.DataFrame, algo_name: str) -> float:
    rng = np.random.RandomState(SEED+17)
    tr = tr_df.copy(); te = te_df.copy()
    ytr = rng.permutation(tr[TARGET_COL].values)
    yte = te[TARGET_COL].values
    Xtr = _ensure_numeric_X(tr).fillna(_ensure_numeric_X(tr).median(numeric_only=True).to_dict())
    med = Xtr.median(numeric_only=True).to_dict()
    Xte = _ensure_numeric_X(te).reindex(columns=Xtr.columns, fill_value=np.nan).fillna(med)
    sel = SimpleKBest(k=min(60, Xtr.shape[1]), mandatory=[c for c in [CUR_WARP] if c in Xtr.columns]).fit(Xtr, ytr)
    Xt = sel.transform(Xtr); Xe = sel.transform(Xte)

    mdl = _build_model(algo_name, n_features=Xt.shape[1], mono_idx=None)
    mdl.fit(Xt, ytr)
    yhat = mdl.predict(Xe)
    return float(r2_score(yte, yhat))

# --------------------------- 요약/메트릭 ---------------------------
def write_summary_and_metrics(exp_id: str,
                              e2x_res: TrainResult, x2e_res: TrainResult,
                              avg_test_r2: float, rf_r2_strict: float,
                              permutation_note: str, summary_path: Path):
    lines=[]
    ts = datetime.now()
    lines.append("Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
    lines.append("="*72)
    lines.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} 작성")
    lines.append(f"실험 ID: {exp_id}\n")

    lines.append("[모델 성능 - R2]")
    lines.append(f"* E2X | Algo: XGBoost | Train R2: {e2x_res.train_r2:.4f} | Test R2: {e2x_res.test_r2:.4f} "
                 f"(RMSE {e2x_res.train_rmse:.4f}/{e2x_res.test_rmse:.4f})")
    lines.append(f"* X2E | Algo: {'RandomForest' if isinstance(x2e_res.model, RandomForestRegressor) else 'XGBoost'} "
                 f"| Train R2: {x2e_res.train_r2:.4f} | Test R2: {x2e_res.test_r2:.4f} "
                 f"(RMSE {x2e_res.train_rmse:.4f}/{x2e_res.test_rmse:.4f})")
    lines.append(f"\n[평균 Test R2] {avg_test_r2:.4f}")
    lines.append(f"[Strict Test‑only Rollforward R²] {rf_r2_strict:.4f}")

    if permutation_note:
        lines.append("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        lines.append(permutation_note)

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # 메트릭 CSV(항상 동일 컬럼)
    REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    met_csv = REPORTS_METRICS_DIR/"last_run_metrics.csv"
    row = {
        "experiment_id": exp_id,
        "label": RUN_NAME,
        "anchor_name": ANCHOR_TAG,
        "e2x_algo": "XGBoost",
        "x2e_algo": ("RandomForest" if isinstance(x2e_res.model, RandomForestRegressor) else "XGBoost"),
        "DT": int(USE_DELTA_TARGET), "TE": int(USE_PAIR_TE), "W": int(USE_HETERO_W),
        "MONO": int(USE_MONO_CUR), "RM": int(USE_RM_TRAINONLY),
        "e2x_train_r2": e2x_res.train_r2, "e2x_test_r2": e2x_res.test_r2,
        "x2e_train_r2": x2e_res.train_r2, "x2e_test_r2": x2e_res.test_r2,
        "avg_test_r2": avg_test_r2,
        "rollforward_r2_strict": rf_r2_strict,
        "summary_path": str(summary_path),
    }
    df = pd.DataFrame([row])
    if met_csv.exists():
        old = pd.read_csv(met_csv)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(met_csv, index=False)

# ------------------------------ main ------------------------------
def main():
    exp_id = _next_experiment_id()
    summary_path = REPORTS_SUMMARY_DIR/f"{exp_id}_{RUN_NAME}_summary.txt"
    archived_script = REPORTS_CODE_DIR/f"{exp_id}_{Path(__file__).name}"

    try:
        # 0) 데이터 병합
        final_df = load_and_merge_data_tplus1()

        # (옵션) 타깃 퍼뮤테이션 (전체 df에 대해) — 진짜 학습 전에 수행
        if PERMUTE_TARGET:
            rng = np.random.RandomState(SEED)
            final_df[TARGET_COL] = rng.permutation(final_df[TARGET_COL].values)
            print("[PERMUTE] Target shuffled for leak sanity check.")

        # 1) 전이 분리
        e2x_raw, x2e_raw = split_cross_transitions(final_df)

        # 2) 방향별 학습
        #    - E2X: XGBoost 고정
        #    - X2E: 환경변수로 선택 (XGBoost/RandomForest)
        e2x_res = train_one_direction(e2x_raw, "E2X", "XGBoost")
        x2e_res = train_one_direction(x2e_raw, "X2E", ("RandomForest" if X2E_ALGO=="RandomForest" else "XGBoost"))

        avg = (e2x_res.test_r2 + x2e_res.test_r2)/2.0

        # 3) Strict Rollforward (교집합 Test 판 전용)
        all_fe_tr, all_fe_te = base_feature_engineering(final_df, final_df)  # PASS_PROGRESS 분모만 필요
        rf_strict = rollforward_r2_strict(all_fe_tr)

        # 4) 누수 퍼뮤 감사(빠른) — 요약 기록용
        perm_note = ""
        if LEAK_AUTOCHECK:
            e2x_perm = quick_perm_r2(e2x_raw, e2x_raw.sample(frac=0.2, random_state=SEED), "XGBoost")
            x2e_perm = quick_perm_r2(x2e_raw, x2e_raw.sample(frac=0.2, random_state=SEED), "XGBoost" if X2E_ALGO!="RandomForest" else "RandomForest")
            perm_note = f"- E2X permuted R²: {e2x_perm:.4f}\n- X2E permuted R²: {x2e_perm:.4f}"
            # 기대치는 ≈0, 0.05 이상이면 의심

        # 5) 요약/메트릭 저장
        write_summary_and_metrics(exp_id, e2x_res, x2e_res, avg, rf_strict, perm_note, summary_path)

        # 6) 코드 스냅샷
        try:
            shutil.copy2(Path(__file__).resolve(), archived_script)
        except Exception:
            pass

        print(f"✓ 요약 저장: {summary_path}")
        print(f"[평균 Test R2] {avg:.4f} | [Strict RF R²] {rf_strict:.4f}")

    except Exception as e:
        tb = traceback.format_exc()
        (REPORTS_DIR/"last_error_traceback.txt").write_text(tb, encoding="utf-8")
        print(f"[실패] {type(e).__name__}: {e}")
        print(tb)

if __name__ == "__main__":
    main()
