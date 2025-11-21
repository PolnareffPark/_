# 095 버전 회귀
import os, re, shutil, warnings, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

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

# R² 상승 전략 플래그
USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET", "0")
USE_PAIR_TE      = _env_on("USE_PAIR_TE", "0")
USE_HETERO_W     = _env_on("USE_HETERO_W", "0")
USE_MONO_CUR     = _env_on("USE_MONO_CUR", "0")
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY", "0")   # test의 RM_* 직접 사용 금지

# 알고리즘 선택 (E2X는 XGB 고정, X2E는 선택)
E2X_ALGO = "XGBoost"
X2E_ALGO = os.getenv("X2E_ALGO", "XGBoost")  # "XGBoost" or "RandomForest"

# -------------------- 유틸 --------------------
def _next_experiment_id() -> str:
    existing = []
    for f in SUM_DIR.glob("*.txt"):
        m = re.match(r"(\d{3})_", f.name)
        if m:
            try:
                existing.append(int(m.group(1)))
            except:
                pass
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

def _save_metrics(experiment_id, avg_r2, e2x_r2, x2e_r2, strict_rf_r2, summary_path):
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

def _algo_build(name: str, monotone_vec=None):
    if name == "XGBoost":
        params = dict(
            objective="reg:squarederror",
            random_state=SEED,
            tree_method="hist",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            n_jobs=0,
        )
        if monotone_vec is not None:
            # xgboost는 list/tuple 또는 문자열 "(0,1,0,...)" 모두 허용되는 버전이 있음
            try:
                return XGBRegressor(**params, monotone_constraints=tuple(monotone_vec))
            except TypeError:
                return XGBRegressor(**params, monotone_constraints="(" + ",".join(map(str, monotone_vec)) + ")")
        return XGBRegressor(**params)
    elif name == "RandomForest":
        return RandomForestRegressor(
            random_state=SEED, n_estimators=600, n_jobs=-1,
            max_depth=20, min_samples_leaf=4, max_features="sqrt"
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

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

# -------------------- 2) 전이 분리 --------------------
def split_e2x_x2e(final_df: pd.DataFrame):
    e2x_df = final_df[final_df[PASS_COL] % 2 == 1].copy()  # Entry t → Exit t+1
    x2e_df = final_df[final_df[PASS_COL] % 2 == 0].copy()  # Exit  t → Entry t+1

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
    # PASS 파생(최종 스케일은 train 분할 뒤 적용)
    d["FM_PASS_squared"] = d[PASS_COL] ** 2
    d.to_csv(PROCESSED_DIR / "all_featured_tplus1.csv", index=False)
    return d

# -------------------- 4) 설계행렬 준비/누수 스크리너 --------------------
def _prepare_design(tr_df: pd.DataFrame, te_df: pd.DataFrame):
    tr = tr_df.copy()
    te = te_df.copy()

    # PASS 진행도: train max로 고정 → test 동일 스케일
    pass_max = float(tr[PASS_COL].max()) if len(tr) else 1.0
    tr["FM_PASS_progress"] = tr[PASS_COL] / (pass_max if pass_max > 0 else 1.0)
    te["FM_PASS_progress"] = te[PASS_COL] / (pass_max if pass_max > 0 else 1.0)

    # RM train-only 옵션: test의 RM_*는 사용하지 않음(누수 보수)
    if USE_RM_TRAINONLY:
        rm_cols = _rm_cols(tr)
        if rm_cols:
            # train 중앙값 저장, test는 NaN으로 비워 train 중앙값으로만 채우기
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

    # 값 기반 누수 스크리너: |r|>=0.999이면 타깃 스멜로 간주하여 제거(학습에서만 결정)
    ytr = tr[TARGET_COL].astype(float).values
    if len(Xtr) and len(ytr):
        # 상관 계산 안전화
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
    # train 필터
    Xtr2 = Xtr.loc[Xtr.index[keep]].copy()
    tr2  = tr.loc[Xtr2.index].copy()  # 인덱스 정합 유지 (중요)
    # test clip
    for c,(lo,hi) in bounds.items():
        if c in Xte.columns:
            Xte[c] = Xte[c].clip(lower=lo, upper=hi)

    return tr2, Xtr2, Xte, med, list(Xtr2.columns)

# -------------------- 5) TE/WLS 유틸 --------------------
def _pair_key(df):
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)

def _add_pair_te_oof(tr: pd.DataFrame, te: pd.DataFrame):
    """pair(p→p+1) Δ-타깃 OOF 인코딩(EB 수축)."""
    tr = tr.copy(); te = te.copy()
    pair_tr = _pair_key(tr)
    pair_te = _pair_key(te)

    # Δ-타깃
    delta_tr = (tr[TARGET_COL] - tr[CUR_WARP]).values

    oof = pd.Series(index=tr.index, dtype=float)
    gkf = GroupKFold(n_splits=min(5, tr[PLATE_COL].nunique()))
    for tr_i, va_i in gkf.split(tr, groups=tr[PLATE_COL].astype(str).values):
        sub = tr.iloc[tr_i]
        # 그룹 평균
        g_mean = pd.DataFrame({"pair": _pair_key(sub), "delta": delta_tr[tr_i]}).groupby("pair")["delta"].mean()
        oof.iloc[tr.index[va_i]] = _pair_key(tr.iloc[va_i]).map(g_mean)

    # EB 수축: 전체 평균 towards
    global_mean = float(np.nanmean(delta_tr)) if len(delta_tr) else 0.0
    oof = oof.fillna(global_mean)
    # test는 train 통계로만
    g_all = pd.DataFrame({"pair": pair_tr, "delta": delta_tr}).groupby("pair")["delta"].mean()
    te_te = pair_te.map(g_all).fillna(global_mean)

    tr["pair_te_delta_mean"] = oof.values
    te["pair_te_delta_mean"] = te_te.values
    return tr, te

def _calc_wls_weight(tr: pd.DataFrame):
    """쌍 Δ-분산 기반 가중치: w = 1/(eps + std^2)."""
    delta = (tr[TARGET_COL] - tr[CUR_WARP]).values
    pair = _pair_key(tr)
    g = pd.DataFrame({"pair": pair, "delta": delta}).groupby("pair")["delta"]
    std = g.std().fillna(g.mean()*0.0 + 1.0)
    std_map = std.reindex(pair.unique()).to_dict()
    # 매 표본의 pair std
    s = pair.map(std)
    w = 1.0 / (1e-6 + (s.values ** 2))
    return w

# -------------------- 6) 단방향 학습 --------------------
def _fit_one_direction(df: pd.DataFrame, direction: str, algo_name: str):
    """direction: 'E2X' or 'X2E'"""
    d0 = df.copy()
    if direction.upper() == "E2X":
        d0 = d0[d0[PASS_COL] % 2 == 1].copy()
        prefix = f"e2x_{ANCHOR_TAG}"
    else:
        d0 = d0[d0[PASS_COL] % 2 == 0].copy()
        prefix = f"x2e_{ANCHOR_TAG}"

    # 그룹 분할(plate 기준), 앵커 고정 재사용
    anchor_path = MODELS_DIR / f"{prefix}_test_plates.pkl"
    groups_all = d0[PLATE_COL].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)

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

    # Δ-타깃 여부에 따라 y 정의
    if USE_DELTA_TARGET:
        tr_df_raw[TARGET_COL] = (tr_df_raw[TGT_WARP] - tr_df_raw[CUR_WARP]).astype(float)
        te_df_raw[TARGET_COL] = (te_df_raw[TGT_WARP] - te_df_raw[CUR_WARP]).astype(float)
    else:
        tr_df_raw[TARGET_COL] = tr_df_raw[TGT_WARP].astype(float)
        te_df_raw[TARGET_COL] = te_df_raw[TGT_WARP].astype(float)

    # TE(누수 없이)
    tr_df = tr_df_raw.copy()
    te_df = te_df_raw.copy()
    if USE_PAIR_TE:
        tr_df, te_df = _add_pair_te_oof(tr_df, te_df)

    # 설계행렬 준비(누수 스크리너/클리핑 포함)
    tr_df2, Xtr, Xte, med, pre_cols = _prepare_design(tr_df, te_df)  # pre_cols = 선택 전 피처 목록
    ytr = tr_df2[TARGET_COL].values

    # SelectKBest
    k = min(120, Xtr.shape[1]) if Xtr.shape[1] else 1
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(Xtr.values, ytr)
    Xtr_sel = selector.transform(Xtr.values)
    Xte_sel = selector.transform(Xte.values)
    sel_idx = selector.get_support(indices=True)
    sel_cols = [pre_cols[i] for i in sel_idx]  # 선택 후 피처명

    # 단조 제약 벡터(선택 후 피처 순서 기준)
    mono_vec = None
    if (algo_name == "XGBoost") and USE_MONO_CUR:
        mono_vec = [0] * len(sel_cols)
        if CUR_WARP in sel_cols:
            mono_vec[sel_cols.index(CUR_WARP)] = 1

    # 가중치
    sample_weight = None
    if USE_HETERO_W:
        sample_weight = _calc_wls_weight(tr_df2)

    # 모델 적합
    model = _algo_build(algo_name, monotone_vec=mono_vec)
    model.fit(Xtr_sel, ytr, sample_weight=sample_weight)

    # 예측/스코어
    ytr_hat = model.predict(Xtr_sel)
    yte_hat = model.predict(Xte_sel)

    if USE_DELTA_TARGET:
        # Δ→t+1 복원
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

    # 번들 저장(rollforward용) — 선택 전/후 피처 모두 저장
    pd.to_pickle(model,            MODELS_DIR / f"{prefix}_reg.pkl")
    pd.to_pickle(selector,         MODELS_DIR / f"{prefix}_selector.pkl")
    pd.to_pickle(pre_cols,         MODELS_DIR / f"{prefix}_pre_cols.pkl")  # NEW
    pd.to_pickle(sel_cols,         MODELS_DIR / f"{prefix}_sel_cols.pkl")  # NEW
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
        "pre_cols":  pd.read_pickle(MODELS_DIR / f"{prefix}_pre_cols.pkl"),  # 선택 전 피처
        "sel_cols":  pd.read_pickle(MODELS_DIR / f"{prefix}_sel_cols.pkl"),  # 선택 후 피처(참고)
        "medians":   pd.read_pickle(MODELS_DIR / f"{prefix}_med.pkl"),
        "tests":     set(pd.read_pickle(MODELS_DIR / f"{prefix}_test_plates.pkl")),
    }


def strict_rollforward_r2(all_df: pd.DataFrame):
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
            row_t = idx.loc[p].copy()

            # 전이 방향에 맞는 번들 선택
            b = e2x if (p % 2 == 1) else x2e

            pre_cols = list(b["pre_cols"])
            med      = b["medians"]

            # 1행 설계행렬: 반드시 '선택 전(pre_cols) 피처 순서'로 구성 후 selector 적용
            X_row_full = pd.DataFrame([row_t.reindex(pre_cols)], columns=pre_cols).fillna(med)
            X_sel = b["selector"].transform(X_row_full.values)  # 차원 일치

            pred = float(b["model"].predict(X_sel)[0])
            if USE_DELTA_TARGET:
                pred = pred + float(row_t.get(CUR_WARP, np.nan))

            gt_next = float(idx.loc[p+1].get(CUR_WARP, np.nan))
            rows.append({"plate": plate, "from": p, "to": p+1, "pred": pred, "gt_next": gt_next})

    if not rows:
        return np.nan

    df = pd.DataFrame(rows).dropna(subset=["gt_next"])
    if not len(df):
        return np.nan
    return float(r2_score(df["gt_next"], df["pred"]))


# -------------------- 8) Permutation leak audit --------------------
def quick_permutation_audit(tr_res, direction: str, algo_name: str, te_df_for_design: pd.DataFrame):
    """train만 섞고, 동일 변환으로 빠르게 1회 학습/평가."""
    # tr_res에는 학습에 사용한 tr_df2, Xtr, med, selected_cols 등이 없으므로
    # 간소화: 동일 _prepare_design을 다시 돌려 구함
    return_val = 0.0
    try:
        # direction에 맞는 원본 다시 준비
        # 실제 누수 감지는 대략적이면 충분 → 작은 모델/적은 나무 수로 구성
        # 여기서는 test 설계행렬만 te_df_for_design에서 재구성
        pass
    except Exception:
        return_val = np.nan
    # 본 함수는 간소화하여 생략 가능 → 아래 본문에서 별도 구현
    return return_val

def permutation_r2_once(tr_df_raw: pd.DataFrame, te_df_raw: pd.DataFrame, algo_name: str):
    """동일 파이프라인으로 train y만 섞어서 R² 산출(≈0 기대)."""
    # 플래그 반영 동일
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

    # 셔플 타깃
    ytr_perm = np.random.permutation(tr_df2[TARGET_COL].values)
    k = min(60, Xtr.shape[1]) if Xtr.shape[1] else 1  # 가볍게
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(Xtr.values, ytr_perm)
    Xtr_sel = selector.transform(Xtr.values)
    Xte_sel = selector.transform(Xte.values)

    model = _algo_build(algo_name)
    model.fit(Xtr_sel, ytr_perm)
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
    e2x_raw, x2e_raw = split_e2x_x2e(all_fe)

    # 학습/평가
    e2x_res = _fit_one_direction(all_fe, "E2X", E2X_ALGO)
    x2e_res = _fit_one_direction(all_fe, "X2E", X2E_ALGO)

    strict_rf = strict_rollforward_r2(all_fe)

    # 퍼뮤 감사(옵션)
    perm_e2x = perm_x2e = np.nan
    if LEAK_AUTOCHECK:
        # 앵커 재사용 분할로 다시 분할 구성
        # E2X
        d0 = all_fe[all_fe[PASS_COL]%2==1].copy()
        e2x_test_plates = set(pd.read_pickle(MODELS_DIR / f"e2x_{ANCHOR_TAG}_test_plates.pkl"))
        e2x_tr = d0[~d0[PLATE_COL].astype(str).isin(e2x_test_plates)].copy()
        e2x_te = d0[ d0[PLATE_COL].astype(str).isin(e2x_test_plates)].copy()
        perm_e2x = permutation_r2_once(e2x_tr, e2x_te, E2X_ALGO)

        # X2E
        d1 = all_fe[all_fe[PASS_COL]%2==0].copy()
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
