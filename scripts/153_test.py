# 134_test.py 에서 시작
# SVR 고정 + 누수 차단 + 전략 플래그 ON/OFF + 자동 메트릭 저장 (rollforward 제거)
import os, re, shutil, warnings
from datetime import datetime
from pathlib import Path
from inspect import signature

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
PERMUTE_TARGET = _env_on("PERMUTE_TARGET", "0")   # 전체 타깃 셔플(감사용)
LEAK_AUTOCHECK = _env_on("LEAK_AUTOCHECK", "1")   # 간이 permutation 감사

# === 전략 플래그(ON/OFF) ===
USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET", "0")   # Δ-타깃 (t+1 - t)
USE_PAIR_TE      = _env_on("USE_PAIR_TE", "0")        # Pair Δ-TE (OOF, EB)
USE_HETERO_W     = _env_on("USE_HETERO_W", "0")       # 이분산 가중치
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY", "0")   # test의 RM_* 직접사용 금지
USE_POLY_FEATS   = _env_on("USE_POLY_FEATS", "0")     # 다항특성(파이프라인 내부)

# === 선택/전처리 하이퍼 ===
SELECT_K     = int(os.getenv("SELECT_K", "60"))       # K-best 수
Q_LOW        = float(os.getenv("Q_LOW", "0.01"))      # train 분위 하한
Q_HIGH       = float(os.getenv("Q_HIGH", "0.99"))     # train 분위 상한
PAIR_MIN_N   = int(os.getenv("PAIR_MIN_N", "1"))      # TE/Weight 최소 표본수 (안정성)

# === SVR 하이퍼 ===
SVR_C       = float(os.getenv("SVR_C", "10.0"))
SVR_EPS     = float(os.getenv("SVR_EPS", "0.1"))
SVR_KERNEL  = os.getenv("SVR_KERNEL", "rbf")          # "rbf" | "linear" | "poly"
SVR_GAMMA   = os.getenv("SVR_GAMMA", "scale")         # "scale" | "auto" | float

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

# ====== SVR 파이프라인 ======
def _build_svr_pipeline():
    steps = [("scale", StandardScaler(with_mean=True, with_std=True))]
    if USE_POLY_FEATS:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    svr = SVR(C=SVR_C, epsilon=SVR_EPS, kernel=SVR_KERNEL, gamma=SVR_GAMMA)
    steps.append(("reg", svr))
    return Pipeline(steps)

def _fit_model(model, X, y, sample_weight=None):
    # 1) 직통 시도
    if sample_weight is None:
        return model.fit(X, y)
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        pass
    # 2) Pipeline이면 마지막 스텝으로 위임
    if isinstance(model, Pipeline):
        last_name, last_est = model.steps[-1]
        if "sample_weight" in signature(last_est.fit).parameters:
            return model.fit(X, y, **{f"{last_name}__sample_weight": sample_weight})
    # 3) 미지원 → 경고 후 무시
    print("[WARN] sample_weight 미지원 추정기 → 가중치 미적용 학습")
    return model.fit(X, y)

# ====== 1) 병합/타깃 ======
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

# ====== 4) 설계행렬 + 누수 스크리너 ======
def _prepare_design(tr_df: pd.DataFrame, te_df: pd.DataFrame):
    tr = tr_df.copy()
    te = te_df.copy()

    # PASS 진행도: train의 max 로 고정
    pmax = float(tr[PASS_COL].max()) if len(tr) else 1.0
    scale = pmax if pmax > 0 else 1.0
    tr["FM_PASS_progress"] = tr[PASS_COL] / scale
    te["FM_PASS_progress"] = te[PASS_COL] / scale

    # RM train‑only (test의 RM_* 직접사용 금지)
    if USE_RM_TRAINONLY:
        rm_cols = _rm_cols(tr)
        if rm_cols:
            tr_rm_med = tr[rm_cols].median(numeric_only=True).to_dict()
            te[rm_cols] = np.nan
            te = te.fillna(tr_rm_med)

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

# ====== 5) TE / Weights ======
def _pair_key(df):
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)

def _add_pair_te_oof(tr: pd.DataFrame, te: pd.DataFrame):
    """
    Pair(p→p+1) Δ-타깃 OOF 인코딩(EB 수축).
    - 포지션 인덱스로만 OOF 할당(넘파이)
    - 각 fold의 검증행은 fold-train 평균으로만 채움
    - test는 train 전체 평균으로만 매핑
    """
    tr = tr.copy()
    te = te.copy()

    pair_tr = _pair_key(tr)
    pair_te = _pair_key(te)
    delta_tr = (tr[TARGET_COL] - tr[CUR_WARP]).to_numpy(dtype=float)

    n = len(tr)
    oof = np.full(n, np.nan, dtype=float)

    n_groups = int(tr[PLATE_COL].astype(str).nunique())
    if n_groups >= 2 and n >= 2:
        gkf = GroupKFold(n_splits=min(5, n_groups))
        groups = tr[PLATE_COL].astype(str).to_numpy()
        for tr_i, va_i in gkf.split(np.zeros(n), groups=groups):
            g_mean = (
                pd.DataFrame({"pair": pair_tr.iloc[tr_i].to_numpy(),
                              "delta": delta_tr[tr_i]})
                .groupby("pair", sort=False)["delta"]
                .mean()
            )
            mapped = pair_tr.iloc[va_i].map(g_mean).to_numpy()
            oof[va_i] = mapped
    # EB 수축
    global_mean = float(np.nanmean(delta_tr)) if n else 0.0
    oof = np.where(np.isfinite(oof), oof, global_mean)
    tr["pair_te_delta_mean"] = oof

    # test: train 전체 평균으로만
    g_all = (
        pd.DataFrame({"pair": pair_tr.to_numpy(), "delta": delta_tr})
        .groupby("pair", sort=False)["delta"]
        .mean()
    )
    te["pair_te_delta_mean"] = pair_te.map(g_all).fillna(global_mean).to_numpy()
    return tr, te

def _calc_wls_weight(tr: pd.DataFrame):
    """쌍 Δ 분산 기반 가중: w=1/(eps+std^2)."""
    delta = (tr[TARGET_COL] - tr[CUR_WARP]).values
    pair = _pair_key(tr)
    dfm = pd.DataFrame({"pair": pair, "delta": delta})
    g = dfm.groupby("pair")["delta"]
    std = g.std()
    m = std.mean()
    std = std.fillna(m if pd.notna(m) else 1.0)
    s = pair.map(std)
    w = 1.0 / (1e-6 + (s.values ** 2))
    return w

# ====== 6) 단방향 학습 ======
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

    # 타깃 정의(Δ 여부)
    if USE_DELTA_TARGET:
        tr_df_raw[TARGET_COL] = (tr_df_raw[TGT_WARP] - tr_df_raw[CUR_WARP]).astype(float)
        te_df_raw[TARGET_COL] = (te_df_raw[TGT_WARP] - te_df_raw[CUR_WARP]).astype(float)
    else:
        tr_df_raw[TARGET_COL] = tr_df_raw[TGT_WARP].astype(float)
        te_df_raw[TARGET_COL] = te_df_raw[TGT_WARP].astype(float)

    # TE (누수 없이 OOF)
    tr_df = tr_df_raw.copy(); te_df = te_df_raw.copy()
    if USE_PAIR_TE:
        tr_df, te_df = _add_pair_te_oof(tr_df, te_df)

    # 설계행렬 + 누수스크리너
    tr_df2, Xtr, Xte, med, pre_cols = _prepare_design(tr_df, te_df)
    ytr = tr_df2[TARGET_COL].to_numpy(dtype=float)

    # K‑best
    k = max(1, min(SELECT_K, Xtr.shape[1]))
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(Xtr.values, ytr)
    Xtr_sel = selector.transform(Xtr.values)
    Xte_sel = selector.transform(Xte.values)

    # 모델 구성/학습
    model = _build_svr_pipeline()
    sw = _calc_wls_weight(tr_df2) if USE_HETERO_W else None
    _fit_model(model, Xtr_sel, ytr, sample_weight=sw)

    # 예측/복원/스코어
    ytr_hat = model.predict(Xtr_sel)
    yte_hat = model.predict(Xte_sel)
    if USE_DELTA_TARGET:
        ytr_hat = ytr_hat + tr_df2[CUR_WARP].values
        yte_hat = yte_hat + te_df[CUR_WARP].values
        ytr_true = tr_df_raw[TGT_WARP].loc[tr_df2.index].values
        yte_true = te_df_raw[TGT_WARP].values
    else:
        ytr_true = tr_df2[TARGET_COL].values
        yte_true = te_df[TARGET_COL].values

    train_r2   = float(r2_score(ytr_true, ytr_hat)) if len(ytr_true) else np.nan
    test_r2    = float(r2_score(yte_true, yte_hat)) if len(yte_true) else np.nan
    train_rmse = _rmse(ytr_true, ytr_hat)
    test_rmse  = _rmse(yte_true, yte_hat)

    # (참고용 저장: 재현을 위해)
    pd.to_pickle(model,    MODELS_DIR / f"{prefix}_reg.pkl")
    pd.to_pickle(selector, MODELS_DIR / f"{prefix}_selector.pkl")
    pd.to_pickle(pre_cols, MODELS_DIR / f"{prefix}_pre_cols.pkl")
    pd.to_pickle(med,      MODELS_DIR / f"{prefix}_med.pkl")

    return {
        "direction": direction,
        "algorithm": "SVR",
        "train_r2": train_r2, "test_r2": test_r2,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
    }, (tr_df_raw, te_df_raw)

# ====== 7) Permutation leak audit (간이) ======
def permutation_r2_once(tr_df_raw: pd.DataFrame, te_df_raw: pd.DataFrame):
    tr_df = tr_df_raw.copy(); te_df = te_df_raw.copy()
    if USE_DELTA_TARGET:
        tr_df[TARGET_COL] = (tr_df[TGT_WARP] - tr_df[CUR_WARP]).astype(float)
        te_df[TARGET_COL] = (te_df[TGT_WARP] - te_df[CUR_WARP]).astype(float)
    else:
        tr_df[TARGET_COL] = tr_df[TGT_WARP].astype(float)
        te_df[TARGET_COL] = te_df[TGT_WARP].astype(float)

    if USE_PAIR_TE:
        tr_df, te_df = _add_pair_te_oof(tr_df, te_df)

    tr_df2, Xtr, Xte, med, cols = _prepare_design(tr_df, te_df)
    ytr_perm = np.random.permutation(tr_df2[TARGET_COL].values)

    k = max(1, min(min(30, SELECT_K), Xtr.shape[1]))
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(Xtr.values, ytr_perm)
    Xtr_sel = selector.transform(Xtr.values)
    Xte_sel = selector.transform(Xte.values)

    model = _build_svr_pipeline()
    _fit_model(model, Xtr_sel, ytr_perm)
    yte_hat = model.predict(Xte_sel)

    if USE_DELTA_TARGET:
        yte_hat = yte_hat + te_df[CUR_WARP].values
        yte_true = te_df_raw[TGT_WARP].values
    else:
        yte_true = te_df[TARGET_COL].values
    return float(r2_score(yte_true, yte_hat)) if len(yte_true) else np.nan

# ====== 8) 요약 저장 ======
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

    lines.append("\n[DEBUG] 설정/분할")
    lines.append(f"- PERMUTE_TARGET: {1 if PERMUTE_TARGET else 0}")
    lines.append(f"- USE_DELTA_TARGET={1 if USE_DELTA_TARGET else 0} | "
                 f"USE_PAIR_TE={1 if USE_PAIR_TE else 0} | "
                 f"USE_HETERO_W={1 if USE_HETERO_W else 0} | "
                 f"USE_RM_TRAINONLY={1 if USE_RM_TRAINONLY else 0} | "
                 f"USE_POLY_FEATS={1 if USE_POLY_FEATS else 0}")
    lines.append(f"- 허용 E2X 패스: {sorted(ALLOWED_E2X_PASSES)} | 허용 X2E 패스: {sorted(ALLOWED_X2E_PASSES)}")
    lines.append(f"- t+1 생성 시 삭제 수(마지막 PASS 등): {dropped_shift}")

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
    _ = split_e2x_x2e(all_fe)  # 파일 저장용

    # 학습
    e2x_res, (e2x_tr_raw, e2x_te_raw) = _fit_one_direction(all_fe, "E2X")
    x2e_res, (x2e_tr_raw, x2e_te_raw) = _fit_one_direction(all_fe, "X2E")

    # 간이 permutation 감사(≈0 기대)
    perm_e2x = permutation_r2_once(e2x_tr_raw, e2x_te_raw) if LEAK_AUTOCHECK else np.nan
    perm_x2e = permutation_r2_once(x2e_tr_raw, x2e_te_raw) if LEAK_AUTOCHECK else np.nan

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
