# 093 테스트 수정본

import os
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- 경로/상수 --------------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
SUM_DIR = REPORTS_DIR / "summaries"
SCRIPTS_DIR = REPORTS_DIR / "scripts"
METRICS_DIR = REPORTS_DIR / "metrics"
ANCHOR_DIR = REPORTS_DIR / "anchors"
MODELS_DIR = Path("models")

SCRIPT_PATH = Path(__file__).resolve()

TARGET_COL = "warping_index_target"       # t+1 타깃
CUR_WARP   = "warping_index_current_pass" # 현재 패스의 warping_index
PASS_COL   = "FM_PASS NO N"
PLATE_COL  = "FM_날판번호"
MONTH_COL  = "FM_압연월"

# 이름 스멜 기반 제외(피처에서 제거)
NAME_LEAK_KWS = ("target", "label", "oof", "pred", "_hat")

# 환경 플래그
def _env_on(k: str, default: str = "0") -> int:
    v = os.environ.get(k, default).strip()
    return 1 if v in ("1", "true", "True", "YES", "yes", "on", "ON") else 0

RUN_NAME        = os.environ.get("RUN_NAME", "pass_t_plus_1_cross")
ANCHOR_TAG      = os.environ.get("ANCHOR_TAG", f"anchor_{datetime.now():%Y%m%d_%H%M%S}")
PERMUTE_TARGET  = _env_on("PERMUTE_TARGET", "0")
LEAK_AUTOCHECK  = _env_on("LEAK_AUTOCHECK", "1")

USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET", "0")
USE_PAIR_TE      = _env_on("USE_PAIR_TE", "0")
USE_HETERO_W     = _env_on("USE_HETERO_W", "0")
USE_MONO_CUR     = _env_on("USE_MONO_CUR", "0")
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY", "0")

E2X_ALGO = os.environ.get("E2X_ALGO", "XGBoost")            # 고정 가능
X2E_ALGO = os.environ.get("X2E_ALGO", os.environ.get("X2E_ALGO", "XGBoost"))  # XGB 또는 RandomForest
SEED     = int(os.environ.get("SEED", "42"))

# 선택 피처 개수
SELECT_K = int(os.environ.get("SELECT_K", "120"))

# --------------------------- 유틸 -----------------------------
def _ensure_dirs():
    for d in [PROCESSED_DIR, REPORTS_DIR, SUM_DIR, SCRIPTS_DIR, METRICS_DIR, ANCHOR_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def _next_experiment_id() -> str:
    existing = []
    for f in SUM_DIR.glob("*.txt"):
        m = re.match(r"(\d{3})_", f.name)
        if m:
            try: existing.append(int(m.group(1)))
            except: pass
    return f"{(max(existing)+1) if existing else 1:03d}"

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _hard_leak_guard(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        low = c.lower()
        if any(kw in low for kw in NAME_LEAK_KWS):
            # 단, CUR_WARP는 허용
            if c == CUR_WARP:
                out.append(c)
            continue
        if c == TARGET_COL:
            continue
        out.append(c)
    return out

def _num_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()

def _save_metrics_row(experiment_id: str,
                      e2x_res: dict, x2e_res: dict,
                      avg_test_r2: float,
                      summary_path: Path):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    last_csv = METRICS_DIR / "last_run_metrics.csv"
    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "e2x_algo": e2x_res.get("algo"),
        "x2e_algo": x2e_res.get("algo"),
        "e2x_train_r2": e2x_res.get("train_r2"),
        "e2x_test_r2": e2x_res.get("test_r2"),
        "x2e_train_r2": x2e_res.get("train_r2"),
        "x2e_test_r2": x2e_res.get("test_r2"),
        "average_test_r2": avg_test_r2,
        "summary_path": str(summary_path),
    }
    df = pd.DataFrame([row])
    df.to_csv(last_csv, index=False)

# ------------------- 1) 데이터 병합/타깃(t+1) -------------------
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

    # RM 집계(판 단위 5통계)
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

    # 타깃 생성(t+1)
    merged_fm[TARGET_COL] = merged_fm.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged_fm = merged_fm.rename(columns={"warping_index": CUR_WARP})
    before = len(merged_fm)
    merged_fm = merged_fm.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    dropped = before - len(merged_fm)
    print(f"Target shift(-1): {before} → {len(merged_fm)} (삭제 {dropped})")

    # RM 집계 병합
    final_df = pd.merge(
        merged_fm, rm_stats,
        left_on=PLATE_COL, right_on="RM_날판번호", how="left"
    ).drop(columns=["RM_날판번호"], errors="ignore")

    # 불필요 컬럼 제거
    final_df = final_df.drop(columns=["extracted_plate","extracted_pass","filename",
                                      "quality_grade","quality_grade_current_pass",
                                      "direction"], errors="ignore")

    # 저장
    out_path = PROCESSED_DIR / "final_merged_data_regression_tplus1.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"✓ 저장: {out_path}")
    return final_df

# --------- 2) 세트 분리(E→X / X→E) + 앵커 분할 고정 ----------
def split_cross_sets(final_df: pd.DataFrame):
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

# -------------------- 3) Feature Engineering -------------------
def _fe_pass_block(d: pd.DataFrame, pass_max: float) -> pd.DataFrame:
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
    # pair id (p→p+1)
    d["pair"] = d[PASS_COL].astype(int).astype(str) + "→" + (d[PASS_COL] + 1).astype(int).astype(str)
    return d

def fe_train_test(tr: pd.DataFrame, te: pd.DataFrame):
    pass_max_train = float(tr[PASS_COL].max()) if len(tr) else 1.0
    tr_fe = _fe_pass_block(tr, pass_max_train)
    te_fe = _fe_pass_block(te, pass_max_train)
    return tr_fe, te_fe

# -------------------- 4) 디자인 행렬/선택/가드 -------------------
def _prepare_design(tr_fe: pd.DataFrame, te_fe: pd.DataFrame):
    # 숫자형만 + 이름 스멜 제거 + 타깃 제거
    drop_candidates = [TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL]
    trX = tr_fe.drop(columns=[c for c in drop_candidates if c in tr_fe.columns], errors="ignore")
    teX = te_fe.drop(columns=[c for c in drop_candidates if c in te_fe.columns], errors="ignore")

    trX = _num_df(trX)
    teX = _num_df(teX)

    keep_cols = _hard_leak_guard(trX.columns.tolist())
    trX = trX[keep_cols]
    teX = teX[[c for c in keep_cols if c in teX.columns]]

    # train 중앙값 대치
    med = trX.median(numeric_only=True).to_dict()
    trX = trX.fillna(med)
    teX = teX.fillna(med)

    # 분위 clip(양끝 1%): train 기준
    qlow, qhi = 0.01, 0.99
    bounds = {}
    for c in trX.columns:
        lo, hi = np.quantile(trX[c].values, [qlow, qhi])
        bounds[c] = (float(lo), float(hi))
        trX[c] = np.clip(trX[c].values, lo, hi)
        if c in teX.columns:
            teX[c] = np.clip(teX[c].values, lo, hi)

    return trX, teX, med, bounds

def _simple_kbest_columns(X: pd.DataFrame, y: np.ndarray, k: int, mandatory: list[str] | None = None):
    # 안정적인 피어슨 기반 K-best
    mandatory = list(mandatory or [])
    cols = X.columns.tolist()
    # 표준화(수치 안정)
    Xm = X.values - np.nanmean(X.values, axis=0, keepdims=True)
    Xs = np.nanstd(X.values, axis=0, ddof=0, keepdims=True); Xs[Xs < 1e-12] = 1e-12
    xr = Xm / Xs
    y0 = y - y.mean()
    ys = y0.std() if y0.std() >= 1e-12 else 1e-12
    r = (xr.T @ (y0 / ys)) / (len(y) - 1)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).flatten()
    abs_r = np.abs(r)

    # mandatory 우선 포함
    rest = [c for c in cols if c not in mandatory]
    order_idx = np.argsort([abs_r[cols.index(c)] for c in rest])[::-1] if rest else []
    k_remain = max(0, min(k - len(mandatory), len(order_idx)))
    selected = mandatory + [rest[i] for i in order_idx[:k_remain]]
    return selected

# -------------------- 5) Pair Δ-TE (OOF + EB) -------------------
def _pair_te_oof(train_df: pd.DataFrame, ytr: np.ndarray, full_df_for_test: pd.DataFrame):
    # group=plate 기준 OOF
    groups = train_df[PLATE_COL].astype(str).values
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))) if len(np.unique(groups)) >= 3 else 3)
    oof_te = np.zeros(len(train_df), dtype=float)
    global_mean = float(np.mean(ytr))
    # EB 수축 파라미터 m: train에서 pair별 n의 중앙값
    pair_counts = train_df.groupby("pair").size()
    m = float(np.median(pair_counts.values)) if len(pair_counts) else 10.0

    for tr_idx, va_idx in gkf.split(train_df, ytr, groups):
        sub = train_df.iloc[tr_idx]
        y_sub = ytr[tr_idx]
        stat = sub.groupby("pair")[[]].size().to_frame("n")
        stat["mean"] = sub.groupby("pair")["_tmp_y"].mean() if "_tmp_y" in sub.columns else \
                       pd.Series(y_sub, index=sub.index).groupby(sub["pair"]).mean()
        # EB
        stat["te"] = (stat["n"] * stat["mean"] + m * global_mean) / (stat["n"] + m)
        te_map = stat["te"].to_dict()
        oof_te[va_idx] = [te_map.get(p, global_mean) for p in train_df.iloc[va_idx]["pair"].values]

    # test용 train-full 통계
    stat_full = train_df.copy()
    stat_full["_ty"] = ytr
    statf = stat_full.groupby("pair")["_ty"].agg(["count","mean"]).rename(columns={"count":"n"})
    statf["te"] = (statf["n"] * statf["mean"] + m * global_mean) / (statf["n"] + m)
    te_map_full = statf["te"].to_dict()

    test_te = [te_map_full.get(p, global_mean) for p in full_df_for_test["pair"].values]
    return oof_te, np.array(test_te, dtype=float), te_map_full, global_mean, m

# -------------------- 6) 이분산 가중(쌍별 분산) -------------------
def _hetero_weights(train_df: pd.DataFrame, ytr: np.ndarray):
    # pair별 분산 추정 → w = 1/(eps + var)
    eps = 1e-6
    g = pd.DataFrame({"pair": train_df["pair"].values, "y": ytr})
    stat = g.groupby("pair")["y"].agg(["count","var"]).rename(columns={"count":"n"})
    stat["var"] = stat["var"].fillna(stat["var"].median() if stat["var"].notna().any() else 1.0)
    var_map = stat["var"].to_dict()
    w = np.array([1.0 / (eps + var_map.get(p, stat["var"].median())) for p in train_df["pair"].values], dtype=float)
    # 스케일 안정화
    w = w / np.mean(w)
    return w

# -------------------- 7) 모델 구성 -------------------
def _build_model(name: str, monotone_list: list[int] | None = None):
    if name == "XGBoost":
        params = dict(
            objective="reg:squarederror",
            random_state=SEED,
            tree_method="hist",
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            n_jobs=0
        )
        if monotone_list is not None:
            try:
                params["monotone_constraints"] = tuple(monotone_list)
            except Exception:
                pass
        return XGBRegressor(**params)
    elif name == "RandomForest":
        return RandomForestRegressor(
            random_state=SEED, n_estimators=700, n_jobs=-1,
            max_depth=20, min_samples_leaf=5, max_features="sqrt"
        )
    else:
        raise ValueError(f"알 수 없는 알고리즘: {name}")

# -------------------- 8) 단일 방향 학습 -------------------
def train_one_direction(df_dir: pd.DataFrame, direction: str, algo_name: str):
    # 8.1 앵커 기반 plate 분할 고정
    anchor_path = ANCHOR_DIR / f"{ANCHOR_TAG}_{direction.lower()}_plates.txt"
    plates_all = df_dir[PLATE_COL].astype(str).unique()
    if anchor_path.exists():
        test_plates = [ln.strip() for ln in anchor_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        _, te_idx = next(gss.split(df_dir, df_dir[TARGET_COL].values, groups=df_dir[PLATE_COL].astype(str).values))
        test_plates = sorted(df_dir.iloc[te_idx][PLATE_COL].astype(str).unique().tolist())
        anchor_path.write_text("\n".join(test_plates), encoding="utf-8")

    tr_raw = df_dir[~df_dir[PLATE_COL].astype(str).isin(test_plates)].copy()
    te_raw = df_dir[df_dir[PLATE_COL].astype(str).isin(test_plates)].copy()

    # 8.2 FE (train pass max 기준)
    tr_fe, te_fe = fe_train_test(tr_raw, te_raw)

    # 8.3 타깃 준비(Δ-타깃 옵션)
    if USE_DELTA_TARGET:
        if CUR_WARP not in tr_fe.columns or CUR_WARP not in te_fe.columns:
            raise RuntimeError("CUR_WARP가 누락되었습니다.")
        tr_fe["_target"] = tr_fe[TARGET_COL].values - tr_fe[CUR_WARP].values
        te_fe["_target"] = te_fe[TARGET_COL].values - te_fe[CUR_WARP].values
    else:
        tr_fe["_target"] = tr_fe[TARGET_COL].values
        te_fe["_target"] = te_fe[TARGET_COL].values

    # 8.4 PERMUTE(누수 자가 점검용)
    if PERMUTE_TARGET:
        rng = np.random.default_rng(SEED + (1 if direction == "E2X" else 2))
        perm = rng.permutation(len(tr_fe))
        tr_fe["_target"] = tr_fe["_target"].values[perm]

    # 8.5 디자인 행렬
    Xtr_df, Xte_df, med, _ = _prepare_design(tr_fe, te_fe)

    # 8.6 RM train-only 옵션: RM_* 컬럼은 train plate만 유지하고 test plate에는 누락(=NaN→중앙값 대치)
    if USE_RM_TRAINONLY:
        rm_cols = [c for c in Xtr_df.columns if c.startswith("RM_")]
        if rm_cols:
            # train에는 그대로 두고, test에서는 RM_*을 NaN으로 클리핑 후 median 대치(상동)
            for c in rm_cols:
                if c in Xte_df.columns:
                    Xte_df[c] = np.nan
            Xte_df = Xte_df.fillna(med)

    # 8.7 Pair Δ-TE 추가(OOF + EB) — train 기준으로만 학습/적용
    te_bundle = None
    if USE_PAIR_TE:
        # y는 현재 학습 타깃(Δ 또는 원타깃)
        ytr_tmp = tr_fe["_target"].to_numpy(dtype=float)
        tr_te_vals, te_te_vals, te_map, te_global, te_m = _pair_te_oof(tr_fe, ytr_tmp, te_fe)
        Xtr_df["pair_te"] = tr_te_vals
        Xte_df["pair_te"] = te_te_vals
        te_bundle = {"map": te_map, "global": te_global, "m": te_m}
        # 중앙값 갱신(안정성)
        med = Xtr_df.median(numeric_only=True).to_dict()
        Xtr_df = Xtr_df.fillna(med)
        Xte_df = Xte_df.fillna(med)

    # 8.8 Hetero 가중치
    sample_weight = None
    if USE_HETERO_W:
        sample_weight = _hetero_weights(tr_fe, tr_fe["_target"].to_numpy(dtype=float))

    # 8.9 필수 피처(CUR_WARP)는 가능하면 포함
    mandatory = [CUR_WARP] if CUR_WARP in Xtr_df.columns else []
    ytr = tr_fe["_target"].to_numpy(dtype=float)
    yte = te_fe["_target"].to_numpy(dtype=float)
    selected_cols = _simple_kbest_columns(Xtr_df, ytr, k=min(SELECT_K, Xtr_df.shape[1]), mandatory=mandatory)
    Xtr2 = Xtr_df[selected_cols].to_numpy(dtype=float)
    Xte2 = Xte_df[[c for c in selected_cols if c in Xte_df.columns]].reindex(columns=selected_cols, fill_value=np.nan).to_numpy(dtype=float)
    # 결측 안전 대치
    Xte2 = np.where(np.isnan(Xte2), np.array([med.get(c, 0.0) for c in selected_cols])[None, :], Xte2)

    # 8.10 단조 제약(선택) — XGB 에서만, CUR_WARP 컬럼에 +1
    mono = None
    if algo_name == "XGBoost" and USE_MONO_CUR:
        mono = [0] * len(selected_cols)
        if CUR_WARP in selected_cols:
            idx = selected_cols.index(CUR_WARP)
            mono[idx] = 1

    # 8.11 모델 학습
    model = _build_model(algo_name, monotone_list=mono)
    if sample_weight is not None:
        model.fit(Xtr2, ytr, sample_weight=sample_weight)
    else:
        model.fit(Xtr2, ytr)

    # 8.12 예측/지표(Δ‑타깃이면 복원)
    pred_tr = model.predict(Xtr2)
    pred_te = model.predict(Xte2)
    if USE_DELTA_TARGET:
        pred_tr = pred_tr + tr_fe[CUR_WARP].values
        pred_te = pred_te + te_fe[CUR_WARP].values
        ytr_eval = tr_fe[TARGET_COL].values
        yte_eval = te_fe[TARGET_COL].values
    else:
        ytr_eval = ytr
        yte_eval = yte

    res = {
        "algo": algo_name,
        "train_r2": float(r2_score(ytr_eval, pred_tr)),
        "test_r2": float(r2_score(yte_eval, pred_te)),
        "train_rmse": _rmse(ytr_eval, pred_tr),
        "test_rmse": _rmse(yte_eval, pred_te),
        "train_samples": int(len(ytr)),
        "test_samples": int(len(yte)),
        "selected_cols": selected_cols,
        "medians": med,
        "test_plates": sorted(test_plates),
        "te_bundle": te_bundle
    }
    # 추후 rollforward 계산을 위한 번들(메모리)
    bundle = {"model": model, "cols": selected_cols, "med": med, "te": te_bundle}
    return res, bundle

# -------------------- 9) Strict rollforward (교집합 test판) ------
def strict_rollforward_r2(final_df: pd.DataFrame, e2x_bundle: dict, x2e_bundle: dict,
                          e2x_res: dict, x2e_res: dict) -> float:
    e2x_test = set(e2x_res["test_plates"])
    x2e_test = set(x2e_res["test_plates"])
    inter = e2x_test.intersection(x2e_test)
    if not inter:
        return float("nan")

    # 전체에 공통 FE (pass_max는 전체가 아닌 'plate 로컬값'이 아니므로 여기서는 단순 진행도 재계산)
    # rollforward에서는 학습 시 썼던 중앙값/컬럼 순서로 1행씩 구성
    rows = []
    for plate, sub in final_df.groupby(PLATE_COL):
        if str(plate) not in inter:
            continue
        sub = sub.sort_values(PASS_COL)
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index:
                continue
            row_t = idx.loc[p].copy()
            if p % 2 == 1:
                # E2X
                bundle = e2x_bundle
            else:
                bundle = x2e_bundle

            # pair_te가 필요하면 학습 시 te_map 으로 계산
            feat = {}
            for c in bundle["cols"]:
                if c == "pair_te":
                    if bundle["te"] is not None:
                        pair_id = f"{p}→{p+1}"
                        feat[c] = bundle["te"]["map"].get(pair_id, bundle["te"]["global"])
                    else:
                        feat[c] = np.nan
                else:
                    feat[c] = row_t.get(c, np.nan)

            # 결측 대치
            for c in bundle["cols"]:
                if pd.isna(feat[c]):
                    feat[c] = bundle["med"].get(c, 0.0)

            X1 = np.array([[feat[c] for c in bundle["cols"]]], dtype=float)
            yhat = float(bundle["model"].predict(X1)[0])
            # Δ‑타깃 사용 여부는 알 수 없지만, rollforward는 항상 "다음 패스의 CUR_WARP"를 비교하므로
            # 학습에서 Δ를 썼다면 CUR_WARP를 더해 복원
            if USE_DELTA_TARGET:
                yhat = yhat + float(row_t.get(CUR_WARP, 0.0))
            gt_next = float(idx.loc[p+1].get(CUR_WARP, np.nan))
            if not np.isnan(gt_next):
                rows.append({"pred": yhat, "gt": gt_next})

    if not rows:
        return float("nan")
    df = pd.DataFrame(rows)
    return float(r2_score(df["gt"].values, df["pred"].values))

# -------------------- 10) 요약 저장 -------------------
def write_summary(experiment_id: str,
                  e2x_res: dict, x2e_res: dict,
                  avg_test_r2: float,
                  strict_rf_r2: float,
                  summary_path: Path):
    lines = []
    lines.append("Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
    lines.append("="*72)
    lines.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 작성")
    lines.append(f"실험 ID: {experiment_id}\n")
    lines.append("[모델 성능 - R2]")
    lines.append(f"* E2X | Algo: {e2x_res['algo']} | Train R2: {e2x_res['train_r2']:.4f} | "
                 f"Test R2: {e2x_res['test_r2']:.4f} (RMSE {e2x_res['train_rmse']:.4f}/{e2x_res['test_rmse']:.4f})")
    lines.append(f"* X2E | Algo: {x2e_res['algo']} | Train R2: {x2e_res['train_r2']:.4f} | "
                 f"Test R2: {x2e_res['test_r2']:.4f} (RMSE {x2e_res['train_rmse']:.4f}/{x2e_res['test_rmse']:.4f})")
    lines.append(f"\n[평균 Test R2] {avg_test_r2:.4f}")
    lines.append(f"[Strict Test‑only Rollforward R²] {strict_rf_r2 if not np.isnan(strict_rf_r2) else 'nan'}")
    lines.append("\n[DEBUG] 설정/분할")
    lines.append(f"- PERMUTE_TARGET: {PERMUTE_TARGET}")
    lines.append(f"- USE_DELTA_TARGET={USE_DELTA_TARGET} | USE_PAIR_TE={USE_PAIR_TE} | "
                 f"USE_HETERO_W={USE_HETERO_W} | USE_MONO_CUR={USE_MONO_CUR} | USE_RM_TRAINONLY={USE_RM_TRAINONLY}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 요약 저장: {summary_path}")

# -------------------- 11) Permutation Audit -------------------
def quick_permute_audit(e2x_df: pd.DataFrame, x2e_df: pd.DataFrame) -> tuple[float,float]:
    # PERMUTE_TARGET=1 로 E2X/X2E 각 1회
    global PERMUTE_TARGET
    old = PERMUTE_TARGET
    PERMUTE_TARGET = 1
    try:
        e2x_res, _ = train_one_direction(e2x_df, "E2X", E2X_ALGO)
        x2e_res, _ = train_one_direction(x2e_df, "X2E", X2E_ALGO)
        return float(e2x_res["test_r2"]), float(x2e_res["test_r2"])
    finally:
        PERMUTE_TARGET = old

# ----------------------------- main ------------------------------
def main():
    _ensure_dirs()
    experiment_id = _next_experiment_id()
    summary_path = SUM_DIR / f"{experiment_id}_test_summary.txt"
    archived_script_path = SCRIPTS_DIR / f"{experiment_id}_test.py"

    # 1) 병합
    final_df = load_and_merge_data_tplus1()

    # 2) 전이 세트 분리
    e2x_df, x2e_df = split_cross_sets(final_df)

    # 3) 학습(E2X/X2E)
    e2x_res, e2x_bundle = train_one_direction(e2x_df, "E2X", E2X_ALGO)
    x2e_res, x2e_bundle = train_one_direction(x2e_df, "X2E", X2E_ALGO)

    # 4) 평균 Test R2
    avg_test_r2 = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))

    # 5) Strict rollforward R²(교집합 test판)
    strict_rf_r2 = strict_rollforward_r2(final_df, e2x_bundle, x2e_bundle, e2x_res, x2e_res)
    print(f"[Strict Test‑only Rollforward R²] {strict_rf_r2 if not np.isnan(strict_rf_r2) else 'nan'}")

    # 6) Permutation 빠른 감사(원하면 끌 수 있게)
    if LEAK_AUTOCHECK:
        pe, px = quick_permute_audit(e2x_df, x2e_df)
        print("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        print(f"- E2X permuted R²: {pe:.4f}")
        print(f"- X2E permuted R²: {px:.4f}")

    # 7) 요약/메트릭/스크립트 스냅샷 저장
    write_summary(experiment_id, e2x_res, x2e_res, avg_test_r2, strict_rf_r2, summary_path)
    _save_metrics_row(experiment_id, e2x_res, x2e_res, avg_test_r2, summary_path)
    shutil.copy2(SCRIPT_PATH, archived_script_path)
    print(f"✓ 코드 스냅샷 저장: {archived_script_path}")

if __name__ == "__main__":
    main()
