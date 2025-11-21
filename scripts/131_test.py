# 104_test 에서 시작
# SVR/Ridge/Lasso/Linear 전용, 허용 패스 제한 + 누수 가드 + 고정 앵커 + 자동 요약/메트릭 저장

import os, re, shutil, warnings
from datetime import datetime
from pathlib import Path
from inspect import signature

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# === 허용되는 Pass 집합 ===
ALLOWED_E2X_PASSES = {5, 7, 9}   # Entry -> Exit (5->6, 7->8, 9->10)
ALLOWED_X2E_PASSES = {6, 8}      # Exit  -> Entry (6->7, 8->9)

# -------------------- 경로/상수 --------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
SUM_DIR = REPORTS_DIR / "summaries"
SCRIPTS_DIR = REPORTS_DIR / "scripts"
MODELS_DIR = Path("models")
METRICS_DIR = REPORTS_DIR / "metrics"

for d in [PROCESSED_DIR, REPORTS_DIR, SUM_DIR, SCRIPTS_DIR, MODELS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 컬럼 상수
PLATE_COL = "FM_날판번호"
PASS_COL  = "FM_PASS NO N"
MONTH_COL = "FM_압연월"

CUR_WARP = "warping_index_current_pass"   # t에서의 warping
TGT_WARP = "warping_index_target"         # t+1 타깃
TARGET_COL = "_target"                    # 내부 표준 타깃명 (t+1)

EXCLUDE_COLS_BASE = {
    TARGET_COL, TGT_WARP, PLATE_COL, PASS_COL, MONTH_COL,  # 명시적 제외
}
# 이름 스멜(타깃/예측/레이블 등)
EXCLUDE_NAME_SMELLS = ("target", "label", "oof", "pred", "_hat")

SEED = int(os.getenv("SEED", "42"))

# -------------------- ENV 플래그 --------------------
def _env_on(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() not in {"0", "false", "", "none"}

RUN_NAME         = os.getenv("RUN_NAME", "pass_t_plus_1_cross")
ANCHOR_TAG       = os.getenv("ANCHOR_TAG", "anchor_default")
PERMUTE_TARGET   = _env_on("PERMUTE_TARGET", "0")     # 전체 타깃 무작위화
LEAK_AUTOCHECK   = _env_on("LEAK_AUTOCHECK", "1")     # 퍼뮤 감사 수행

# R² 전략 플래그(원클릭 실험용)
USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET", "0")
USE_PAIR_TE      = _env_on("USE_PAIR_TE", "0")
USE_HETERO_W     = _env_on("USE_HETERO_W", "0")
USE_MONO_CUR     = _env_on("USE_MONO_CUR", "0")       # (선형/커널에선 의미 없음: 자리만 유지)
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY", "0")   # test의 RM_* 직접 사용 금지

# 알고리즘(기본 Linear)
E2X_ALGO = os.getenv("E2X_ALGO", "Linear")     # "SVR" | "Ridge" | "Lasso" | "Linear"
X2E_ALGO = os.getenv("X2E_ALGO", "Linear")     # "SVR" | "Ridge" | "Lasso" | "Linear"

# -------------------- 유틸 --------------------
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

def _rm_cols(df):
    return [c for c in df.columns if c.startswith("RM_")]

def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _save_metrics(experiment_id, avg_r2, e2x_r2, x2e_r2, strict_rf_r2, summary_path: Path):
    METRICS_DIR.mkdir(exist_ok=True, parents=True)
    out = METRICS_DIR / "last_run_metrics.csv"
    row = {
        "experiment_id": experiment_id,
        "avg_test_r2": float(avg_r2) if pd.notna(avg_r2) else np.nan,
        "e2x_test_r2": float(e2x_r2) if pd.notna(e2x_r2) else np.nan,
        "x2e_test_r2": float(x2e_r2) if pd.notna(x2e_r2) else np.nan,
        "strict_rollforward_r2": float(strict_rf_r2) if pd.notna(strict_rf_r2) else np.nan,
        "summary_path": str(summary_path),
    }
    pd.DataFrame([row]).to_csv(out, index=False)

def _build_model(name: str) -> Pipeline:
    """name in {"SVR","Ridge","Lasso","Linear"} — 마지막 스텝명은 'reg' 고정"""
    key = str(name).strip().lower()
    if key == "svr":
        est = SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
    elif key == "ridge":
        est = Ridge(alpha=1.0, tol=1e-4, max_iter=10000, random_state=SEED)
    elif key == "lasso":
        est = Lasso(alpha=0.001, tol=1e-4, max_iter=10000, random_state=SEED)
    elif key == "linear":
        est = LinearRegression()
    else:
        raise ValueError(f"Unknown algorithm name: {name}")
    return Pipeline([("scale", StandardScaler(with_mean=True)), ("reg", est)])

def _fit_model(model: Pipeline, X, y, sample_weight=None):
    """Pipeline/추정기별 sample_weight 안전 전달."""
    if sample_weight is None:
        return model.fit(X, y)
    # 1) 직접 전달 시도
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        pass
    # 2) Pipeline → 마지막 스텝으로 전달
    if isinstance(model, Pipeline):
        last_name, last_est = model.steps[-1]
        try:
            if "sample_weight" in signature(last_est.fit).parameters:
                return model.fit(X, y, **{f"{last_name}__sample_weight": sample_weight})
        except Exception:
            pass
    # 3) 미지원 → 경고 후 무시
    print("[WARN] sample_weight 미지원 추정기 → 미적용 학습")
    return model.fit(X, y)

# -------------------- 1) 데이터 병합/타깃 --------------------
def load_and_merge_tplus1():
    gt_entry = pd.read_csv(DATA_DIR / "entry_direction_results.csv")
    gt_exit  = pd.read_csv(DATA_DIR / "exit_direction_results.csv")
    gt = pd.concat([gt_entry, gt_exit], ignore_index=True)

    fm = pd.read_csv(DATA_DIR / "posco2_105190.csv")
    rm = pd.read_csv(DATA_DIR / "posco1_105190.csv")

    gt["extracted_plate"] = gt["filename"].str.extract(r"(PB\d+)")
    gt["extracted_pass"]  = gt["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    # RM 집계 (판 단위 5통계)
    merged_rm = pd.merge(
        gt, rm, left_on=["extracted_plate","extracted_pass"],
        right_on=["RM_날판번호","RM_압연Pass번호"], how="inner"
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

    # 타깃 생성: t→t+1
    df[TGT_WARP] = df.groupby(PLATE_COL)["warping_index"].shift(-1)
    df = df.rename(columns={"warping_index": CUR_WARP})
    before = len(df)
    df = df.dropna(subset=[TGT_WARP]).reset_index(drop=True)
    dropped_shift = before - len(df)

    # 내부 표준 타깃명
    df[TARGET_COL] = df[TGT_WARP].astype(float)

    # RM 집계 조인
    final_df = pd.merge(
        df, rm_stats, left_on=PLATE_COL, right_on="RM_날판번호", how="left"
    ).drop(columns=["RM_날판번호"], errors="ignore")

    # 군더더기 제거
    final_df = final_df.drop(columns=[
        "extracted_plate","extracted_pass","filename","quality_grade","quality_grade_current_pass","direction"
    ], errors="ignore")

    if PERMUTE_TARGET:
        final_df[TARGET_COL] = np.random.permutation(final_df[TARGET_COL].values)

    out_path = PROCESSED_DIR / "final_merged_data_regression_tplus1.csv"
    final_df.to_csv(out_path, index=False)
    return final_df, dropped_shift

# -------------------- 2) 전이 분리(허용 패스만) --------------------
def split_e2x_x2e(final_df: pd.DataFrame):
    e2x_df = final_df[(final_df[PASS_COL] % 2 == 1) & (final_df[PASS_COL].isin(ALLOWED_E2X_PASSES))].copy()
    x2e_df = final_df[(final_df[PASS_COL] % 2 == 0) & (final_df[PASS_COL].isin(ALLOWED_X2E_PASSES))].copy()
    e2x_df.to_csv(PROCESSED_DIR / "e2x_raw_tplus1.csv", index=False)
    x2e_df.to_csv(PROCESSED_DIR / "x2e_raw_tplus1.csv", index=False)
    return e2x_df, x2e_df

# -------------------- 3) 간단 FE --------------------
def basic_fe(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    fm_cols = [c for c in d.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if fm_cols:
        blk = d[fm_cols]
        d["FM_mean"]  = blk.mean(axis=1)
        d["FM_std"]   = blk.std(axis=1)
        d["FM_max"]   = blk.max(axis=1)
        d["FM_min"]   = blk.min(axis=1)
        d["FM_range"] = d["FM_max"] - d["FM_min"]
        d["FM_cv"]    = d["FM_std"] / (d["FM_mean"].abs() + 1e-8)
    d["FM_PASS_squared"] = d[PASS_COL] ** 2
    d.to_csv(PROCESSED_DIR / "all_featured_tplus1.csv", index=False)
    return d

# -------------------- 4) 설계행렬 준비/누수 스크리너 --------------------
def _prepare_design(tr_df: pd.DataFrame, te_df: pd.DataFrame):
    tr = tr_df.copy(); te = te_df.copy()

    # PASS 진행도: train max로 고정 → test 동일 스케일
    pass_max = float(tr[PASS_COL].max()) if len(tr) else 1.0
    tr["FM_PASS_progress"] = tr[PASS_COL] / (pass_max if pass_max > 0 else 1.0)
    te["FM_PASS_progress"] = te[PASS_COL] / (pass_max if pass_max > 0 else 1.0)

    # RM train-only 옵션: test의 RM_* 사용 금지(보수적 처리)
    if USE_RM_TRAINONLY:
        rm_cols = _rm_cols(tr)
        if rm_cols:
            tr_rm_meds = tr[rm_cols].median(numeric_only=True).to_dict()
            te[rm_cols] = np.nan
            te = te.fillna(tr_rm_meds)

    # 피처 선택(수치형)
    exclude_cols = set(EXCLUDE_COLS_BASE)
    Xtr_all = tr.drop(columns=[c for c in tr.columns if c in exclude_cols], errors="ignore")
    Xte_all = te.drop(columns=[c for c in te.columns if c in exclude_cols], errors="ignore")

    # 이름 스멜 제거
    Xtr_all = Xtr_all[_hard_leak_guard(Xtr_all.columns)]
    Xte_all = Xte_all[[c for c in Xte_all.columns if c in Xtr_all.columns]]

    # 수치형만
    Xtr = Xtr_all.select_dtypes(include=[np.number]).copy()
    Xte = Xte_all.select_dtypes(include=[np.number]).copy()

    # 중앙값(train) 대치
    med = Xtr.median(numeric_only=True).to_dict()
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)

    # 값 기반 누수 스크리너: |r|>=0.999 제거(학습 기준)
    ytr = tr[TARGET_COL].astype(float).values
    if len(Xtr) and len(ytr):
        Xc = (Xtr - Xtr.mean()) / (Xtr.std().replace(0, 1))
        yc = ytr - ytr.mean()
        ys = np.std(yc) if np.std(yc) > 1e-12 else 1e-12
        r = (Xc.values.T @ (yc / ys)) / max(1, (len(ytr) - 1))
        bad_idx = np.where(np.abs(r) >= 0.999)[0].tolist()
        bad_cols = [Xtr.columns[i] for i in bad_idx]
        if bad_cols:
            Xtr = Xtr.drop(columns=bad_cols, errors="ignore")
            Xte = Xte.drop(columns=[c for c in bad_cols if c in Xte.columns], errors="ignore")

    # 이상치 가드: train 분위 경계 산출 → train 필터, test clip
    bounds = {}
    for c in Xtr.columns:
        lo, hi = np.quantile(Xtr[c].values, [0.01, 0.99])
        bounds[c] = (float(lo), float(hi))
    keep = np.ones(len(Xtr), dtype=bool)
    for c,(lo,hi) in bounds.items():
        v = Xtr[c].values
        keep &= (v >= lo) & (v <= hi)

    Xtr2 = Xtr.loc[Xtr.index[keep]].copy()
    tr2  = tr.loc[Xtr2.index].copy()  # 인덱스 정합 유지
    for c,(lo,hi) in bounds.items():
        if c in Xte.columns:
            Xte[c] = Xte[c].clip(lower=lo, upper=hi)

    return tr2, Xtr2, Xte, med, list(Xtr2.columns)

# -------------------- 5) TE/WLS 유틸 --------------------
def _pair_key(df):
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)

def _add_pair_te_oof(tr: pd.DataFrame, te: pd.DataFrame):
    tr = tr.copy(); te = te.copy()
    pair_tr = _pair_key(tr)
    pair_te = _pair_key(te)
    delta_tr = (tr[TARGET_COL] - tr[CUR_WARP]).values

    oof = pd.Series(index=tr.index, dtype=float)
    gkf = GroupKFold(n_splits=min(5, tr[PLATE_COL].nunique()))
    for tr_i, va_i in gkf.split(tr, groups=tr[PLATE_COL].astype(str).values):
        sub = tr.iloc[tr_i]
        g_mean = pd.DataFrame({"pair": _pair_key(sub), "delta": delta_tr[tr_i]}).groupby("pair")["delta"].mean()
        oof.iloc[tr.index[va_i]] = _pair_key(tr.iloc[va_i]).map(g_mean)

    global_mean = float(np.nanmean(delta_tr)) if len(delta_tr) else 0.0
    oof = oof.fillna(global_mean)
    g_all = pd.DataFrame({"pair": pair_tr, "delta": delta_tr}).groupby("pair")["delta"].mean()
    te_te = pair_te.map(g_all).fillna(global_mean)

    tr["pair_te_delta_mean"] = oof.values
    te["pair_te_delta_mean"] = te_te.values
    return tr, te

def _calc_wls_weight(tr: pd.DataFrame):
    delta = (tr[TARGET_COL] - tr[CUR_WARP]).values
    pair = _pair_key(tr)
    g = pd.DataFrame({"pair": pair, "delta": delta}).groupby("pair")["delta"]
    std = g.std().fillna(g.mean()*0.0 + 1.0)
    s = pair.map(std)
    w = 1.0 / (1e-6 + (s.values ** 2))
    return w

# -------------------- 6) 단방향 학습 --------------------
def _fit_one_direction(df: pd.DataFrame, direction: str, algo_name: str):
    d0 = df.copy()
    if direction.upper() == "E2X":
        d0 = d0[(d0[PASS_COL] % 2 == 1) & (d0[PASS_COL].isin(ALLOWED_E2X_PASSES))].copy()
        prefix = f"e2x_{ANCHOR_TAG}"
    else:
        d0 = d0[(d0[PASS_COL] % 2 == 0) & (d0[PASS_COL].isin(ALLOWED_X2E_PASSES))].copy()
        prefix = f"x2e_{ANCHOR_TAG}"

    if d0.empty:
        raise RuntimeError(f"{direction}: 허용된 전이에 해당하는 표본이 없습니다.")

    # 앵커 분할(plate 기반 고정)
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
        raise RuntimeError(f"{direction}: 앵커 분할 결과 train/test 중 한쪽이 비었습니다.")

    # 타깃 정의(Δ 여부)
    if USE_DELTA_TARGET:
        tr_df_raw[TARGET_COL] = (tr_df_raw[TGT_WARP] - tr_df_raw[CUR_WARP]).astype(float)
        te_df_raw[TARGET_COL] = (te_df_raw[TGT_WARP] - te_df_raw[CUR_WARP]).astype(float)
    else:
        tr_df_raw[TARGET_COL] = tr_df_raw[TGT_WARP].astype(float)
        te_df_raw[TARGET_COL] = te_df_raw[TGT_WARP].astype(float)

    # (옵션) TE
    tr_df = tr_df_raw.copy()
    te_df = te_df_raw.copy()
    if USE_PAIR_TE:
        tr_df, te_df = _add_pair_te_oof(tr_df, te_df)

    # 설계행렬 + 누수 스크리너
    tr_df2, Xtr, Xte, med, pre_cols = _prepare_design(tr_df, te_df)
    ytr = tr_df2[TARGET_COL].to_numpy(dtype=float)

    # 안정적인 K-best
    k = max(1, min(50, Xtr.shape[1]))
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(Xtr.values, ytr)
    Xtr_sel = selector.transform(Xtr.values)
    Xte_sel = selector.transform(Xte.values)
    sel_idx = selector.get_support(indices=True)
    sel_cols = [pre_cols[i] for i in sel_idx]

    # 모델 학습
    model = _build_model(algo_name)
    _fit_model(model, Xtr_sel, ytr, sample_weight=_calc_wls_weight(tr_df2) if USE_HETERO_W else None)

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

    train_r2 = float(r2_score(ytr_true, ytr_hat)) if len(ytr_true) else np.nan
    test_r2  = float(r2_score(yte_true, yte_hat)) if len(yte_true) else np.nan
    train_rmse = _rmse(ytr_true, ytr_hat) if len(ytr_true) else np.nan
    test_rmse  = _rmse(yte_true, yte_hat) if len(yte_true) else np.nan

    # 번들 저장(롤포워드 호환)
    pd.to_pickle(model,            MODELS_DIR / f"{prefix}_reg.pkl")
    pd.to_pickle(selector,         MODELS_DIR / f"{prefix}_selector.pkl")
    pd.to_pickle(pre_cols,         MODELS_DIR / f"{prefix}_pre_cols.pkl")
    pd.to_pickle(sel_cols,         MODELS_DIR / f"{prefix}_sel_cols.pkl")
    pd.to_pickle(med,              MODELS_DIR / f"{prefix}_med.pkl")

    return {
        "direction": direction,
        "algorithm": algo_name,
        "train_r2": train_r2, "test_r2": test_r2,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "bundle": {
            "model": model,
            "selector": selector,
            "pre_cols": pre_cols,
            "sel_cols": sel_cols,
            "medians": med,
        },
        "test_plates_path": MODELS_DIR / f"{prefix}_test_plates.pkl",
    }

# -------------------- 7) Strict roll-forward --------------------
def _load_bundle(prefix: str):
    return {
        "model":     pd.read_pickle(MODELS_DIR / f"{prefix}_reg.pkl"),
        "selector":  pd.read_pickle(MODELS_DIR / f"{prefix}_selector.pkl"),
        "pre_cols":  pd.read_pickle(MODELS_DIR / f"{prefix}_pre_cols.pkl"),
        "sel_cols":  pd.read_pickle(MODELS_DIR / f"{prefix}_sel_cols.pkl"),
        "medians":   pd.read_pickle(MODELS_DIR / f"{prefix}_med.pkl"),
        "tests":     set(pd.read_pickle(MODELS_DIR / f"{prefix}_test_plates.pkl")),
    }

def strict_rollforward_r2(all_df: pd.DataFrame):
    # 저장된 번들 로드
    try:
        e2x = _load_bundle(f"e2x_{ANCHOR_TAG}")
        x2e = _load_bundle(f"x2e_{ANCHOR_TAG}")
    except FileNotFoundError:
        return np.nan

    test_plates = set(e2x["tests"]).intersection(set(x2e["tests"]))
    if not test_plates:
        return np.nan

    rows = []
    for plate, sub in all_df.groupby(PLATE_COL):
        if str(plate) not in test_plates:
            continue
        sub = sub.sort_values(PASS_COL).copy()
        idx = sub.set_index(PASS_COL, drop=False)

        for p in idx.index:
            if (p + 1) not in idx.index:
                continue

            # 허용된 전이만 평가 (학습 분포와 일치)
            if p % 2 == 1 and p not in ALLOWED_E2X_PASSES:
                continue
            if p % 2 == 0 and p not in ALLOWED_X2E_PASSES:
                continue

            row_t = idx.loc[p].copy()
            b = e2x if (p % 2 == 1) else x2e

            # 학습 당시 '선택된 피처' 순서대로 행 구성 + 중앙값 대치
            cols = list(b["sel_cols"])            # ★ 여기서 'cols' → 'sel_cols'로 고정
            med  = b["medians"]
            X_row_df = pd.DataFrame([row_t.reindex(cols)], columns=cols).fillna(med)
            X_row = X_row_df.values               # selector.transform() 금지 (차원불일치 방지)

            pred = float(b["model"].predict(X_row)[0])
            if USE_DELTA_TARGET:
                pred = pred + float(row_t[CUR_WARP])

            gt_next = float(idx.loc[p+1][CUR_WARP]) if pd.notna(idx.loc[p+1][CUR_WARP]) else np.nan
            rows.append({"plate": plate, "from": p, "to": p+1, "pred": pred, "gt_next": gt_next})

    if not rows:
        return np.nan
    df = pd.DataFrame(rows).dropna(subset=["gt_next"])
    if not len(df):
        return np.nan
    return float(r2_score(df["gt_next"], df["pred"]))

# -------------------- 8) Permutation leak audit --------------------
def permutation_r2_once(tr_df_raw: pd.DataFrame, te_df_raw: pd.DataFrame, algo_name: str):
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

    # 타깃 셔플
    if len(tr_df2) == 0:
        return np.nan
    ytr_perm = np.random.permutation(tr_df2[TARGET_COL].values)

    k = min(30, Xtr.shape[1]) if Xtr.shape[1] else 1
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(Xtr.values, ytr_perm)
    Xtr_sel = selector.transform(Xtr.values)
    Xte_sel = selector.transform(Xte.values)

    model = _build_model(algo_name)
    _fit_model(model, Xtr_sel, ytr_perm)
    yte_hat = model.predict(Xte_sel)

    if USE_DELTA_TARGET:
        yte_hat = yte_hat + te_df[CUR_WARP].values
        yte_true = te_df_raw[TGT_WARP].values
    else:
        yte_true = te_df[TARGET_COL].values
    return float(r2_score(yte_true, yte_hat)) if len(yte_true) else np.nan

# -------------------- 9) 요약 저장 --------------------
def write_summary(experiment_id: str, results: dict, strict_rf_r2: float, dropped_shift: int, summary_path: Path):
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
    avg = float(np.nanmean([results["E2X"]["test_r2"], results["X2E"]["test_r2"]]))
    lines.append(f"\n[평균 Test R2] {avg:.4f}")
    if pd.notna(strict_rf_r2):
        lines.append(f"[Strict Test‑only Rollforward R²] {strict_rf_r2:.4f}")

    # 퍼뮤 감사 결과
    if "perm_e2x" in results and "perm_x2e" in results:
        lines.append("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        lines.append(f"- E2X permuted R²: {results['perm_e2x']:.4f}")
        lines.append(f"- X2E permuted R²: {results['perm_x2e']:.4f}")

    # 디버그 플래그
    lines.append("\n[DEBUG] 설정/분할")
    lines.append(f"- PERMUTE_TARGET: {1 if PERMUTE_TARGET else 0}")
    lines.append(f"- USE_DELTA_TARGET={1 if USE_DELTA_TARGET else 0} | "
                 f"USE_PAIR_TE={1 if USE_PAIR_TE else 0} | "
                 f"USE_HETERO_W={1 if USE_HETERO_W else 0} | "
                 f"USE_MONO_CUR={1 if USE_MONO_CUR else 0} | "
                 f"USE_RM_TRAINONLY={1 if USE_RM_TRAINONLY else 0}")
    lines.append(f"- 허용 E2X 패스: {sorted(ALLOWED_E2X_PASSES)} | 허용 X2E 패스: {sorted(ALLOWED_X2E_PASSES)}")
    lines.append(f"- t+1 생성 시 삭제 수(마지막 PASS 등): {dropped_shift}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

# -------------------- main --------------------
def main():
    experiment_id = _next_experiment_id()
    summary_path = SUM_DIR / f"{experiment_id}_test_summary.txt"
    # 스크립트 스냅샷 저장
    try:
        this_py = Path(__file__).resolve()
        shutil.copy2(this_py, SCRIPTS_DIR / f"{experiment_id}_test.py")
    except Exception:
        pass

    final_df, dropped_shift = load_and_merge_tplus1()
    all_fe = basic_fe(final_df)
    _ = split_e2x_x2e(all_fe)  # 산출물 저장 겸용

    # 학습/평가
    e2x_res = _fit_one_direction(all_fe, "E2X", E2X_ALGO)
    x2e_res = _fit_one_direction(all_fe, "X2E", X2E_ALGO)

    strict_rf = strict_rollforward_r2(all_fe)

    # 퍼뮤 감사(옵션)
    perm_e2x = perm_x2e = np.nan
    if LEAK_AUTOCHECK:
        # E2X
        d0 = all_fe[(all_fe[PASS_COL] % 2 == 1) & (all_fe[PASS_COL].isin(ALLOWED_E2X_PASSES))].copy()
        e2x_test_plates = set(pd.read_pickle(MODELS_DIR / f"e2x_{ANCHOR_TAG}_test_plates.pkl"))
        e2x_tr = d0[~d0[PLATE_COL].astype(str).isin(e2x_test_plates)].copy()
        e2x_te = d0[ d0[PLATE_COL].astype(str).isin(e2x_test_plates)].copy()
        perm_e2x = permutation_r2_once(e2x_tr, e2x_te, E2X_ALGO)

        # X2E
        d1 = all_fe[(all_fe[PASS_COL] % 2 == 0) & (all_fe[PASS_COL].isin(ALLOWED_X2E_PASSES))].copy()
        x2e_test_plates = set(pd.read_pickle(MODELS_DIR / f"x2e_{ANCHOR_TAG}_test_plates.pkl"))
        x2e_tr = d1[~d1[PLATE_COL].astype(str).isin(x2e_test_plates)].copy()
        x2e_te = d1[ d1[PLATE_COL].astype(str).isin(x2e_test_plates)].copy()
        perm_x2e = permutation_r2_once(x2e_tr, x2e_te, X2E_ALGO)

    results = {
        "E2X": e2x_res,
        "X2E": x2e_res,
        "perm_e2x": perm_e2x,
        "perm_x2e": perm_x2e,
    }
    write_summary(experiment_id, results, strict_rf, dropped_shift, summary_path)

    avg_r2 = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))
    _save_metrics(experiment_id, avg_r2, e2x_res["test_r2"], x2e_res["test_r2"], strict_rf, summary_path)

    # 콘솔 출력
    print("\n" + "Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
    print("="*72)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 작성")
    print(f"실험 ID: {experiment_id}\n")
    for tag in ("E2X","X2E"):
        r = results[tag]
        print(f"* {tag} | Algo: {r['algorithm']} | Train R2: {r['train_r2']:.4f} | "
              f"Test R2: {r['test_r2']:.4f} (RMSE {r['train_rmse']:.4f}/{r['test_rmse']:.4f})")
    print(f"\n[평균 Test R2] {avg_r2:.4f}")
    print(f"[Strict Test‑only Rollforward R²] {strict_rf if pd.notna(strict_rf) else np.nan}")
    if LEAK_AUTOCHECK:
        print("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        print(f"- E2X permuted R²: {perm_e2x:.4f}")
        print(f"- X2E permuted R²: {perm_x2e:.4f}")

if __name__ == "__main__":
    main()
