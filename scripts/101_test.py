# 095 테스트 수정본

import os, re, shutil, warnings, traceback
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- 경로/상수 --------------------
DATA_DIR     = Path("data")
PROCESSED    = DATA_DIR / "processed"
REPORTS      = Path("reports")
SUM_DIR      = REPORTS / "summaries"
SCRIPT_DIR   = REPORTS / "scripts"
METRICS_DIR  = REPORTS / "metrics"
MODELS_DIR   = Path("models")

for d in [PROCESSED, REPORTS, SUM_DIR, SCRIPT_DIR, METRICS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PLATE_COL = "FM_날판번호"
PASS_COL  = "FM_PASS NO N"
MONTH_COL = "FM_압연월"
TARGET_COL = "warping_index_target"          # t+1
CUR_WARP   = "warping_index_current_pass"    # t

EXCLUDE_FROM_X = {PLATE_COL, PASS_COL, MONTH_COL, TARGET_COL}

# -------------------- ENV 플래그 --------------------
def _env_on(k: str, default="0") -> int:
    v = os.environ.get(k, default).strip()
    return 1 if v in ("1", "true", "True", "YES", "yes", "on", "ON") else 0

RUN_NAME         = os.environ.get("RUN_NAME", "pass_t_plus_1_cross")
ANCHOR_TAG       = os.environ.get("ANCHOR_TAG", "r2_anchor")
PERMUTE_TARGET   = _env_on("PERMUTE_TARGET", "0")
LEAK_AUTOCHECK   = _env_on("LEAK_AUTOCHECK", "1")

# R² 상승 전략 플래그
USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET", "0")
USE_PAIR_TE      = _env_on("USE_PAIR_TE", "0")
USE_HETERO_W     = _env_on("USE_HETERO_W", "0")
USE_MONO_CUR     = _env_on("USE_MONO_CUR", "0")
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY", "0")

# 알고리즘 선택: E2X는 XGBoost 고정, X2E는 env로 선택 가능 (XGBoost|RandomForest)
E2X_ALGO = "XGBoost"
X2E_ALGO = os.environ.get("X2E_ALGO", "XGBoost").strip()

SEED = int(os.environ.get("SEED", "42"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.2"))
SELECT_K  = int(os.environ.get("SELECT_K", "120"))

# -------------------- 실험 ID --------------------
def _next_experiment_id() -> str:
    existing = []
    for f in SUM_DIR.glob("*.txt"):
        m = re.match(r"(\d{3})_", f.name)
        if m:
            try: existing.append(int(m.group(1)))
            except: pass
    return f"{(max(existing)+1) if existing else 1:03d}"

# -------------------- 데이터 병합 --------------------
def load_and_merge_data_tplus1() -> pd.DataFrame:
    gt_entry = pd.read_csv(DATA_DIR / "entry_direction_results.csv")
    gt_exit  = pd.read_csv(DATA_DIR / "exit_direction_results.csv")
    ground_truth = pd.concat([gt_entry, gt_exit], ignore_index=True)

    rm = pd.read_csv(DATA_DIR / "posco1_105190.csv")   # RM
    fm = pd.read_csv(DATA_DIR / "posco2_105190.csv")   # FM

    ground_truth["extracted_plate"] = ground_truth["filename"].str.extract(r"(PB\d+)")
    ground_truth["extracted_pass"]  = ground_truth["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    # RM merge & 판 단위 통계
    merged_rm = pd.merge(
        ground_truth, rm,
        left_on=["extracted_plate","extracted_pass"],
        right_on=["RM_날판번호","RM_압연Pass번호"],
        how="inner"
    )
    num_cols = merged_rm.select_dtypes(include=[np.number]).columns.tolist()
    stat_cols = [c for c in num_cols if c not in ["RM_날판번호","RM_압연Pass번호","warping_index","extracted_pass"]]
    rm_stats = merged_rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(4)
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    rm_stats = rm_stats.reset_index()

    # FM merge
    merged_fm = pd.merge(
        ground_truth, fm,
        left_on=["extracted_plate","extracted_pass"],
        right_on=[PLATE_COL, PASS_COL],
        how="inner"
    ).sort_values([PLATE_COL, PASS_COL])

    # t+1 target
    merged_fm[TARGET_COL] = merged_fm.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged_fm = merged_fm.rename(columns={"warping_index": CUR_WARP})
    before = len(merged_fm)
    merged_fm = merged_fm.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    # RM 통계 결합
    final_df = pd.merge(
        merged_fm, rm_stats,
        left_on=PLATE_COL, right_on="RM_날판번호", how="left"
    ).drop(columns=["RM_날판번호"], errors="ignore")
    # 불필요 드롭
    final_df = final_df.drop(columns=[
        "extracted_plate","extracted_pass","filename","quality_grade","quality_grade_current_pass","direction"
    ], errors="ignore")

    out_path = PROCESSED / "final_merged_data_regression_tplus1.csv"
    final_df.to_csv(out_path, index=False)
    return final_df

# -------------------- 전이 세트 분리 --------------------
def split_cross(final_df: pd.DataFrame):
    e2x = final_df[final_df[PASS_COL] % 2 == 1].copy()
    x2e = final_df[final_df[PASS_COL] % 2 == 0].copy()
    (PROCESSED / "e2x_raw_tplus1.csv").write_text("")  # 마커
    (PROCESSED / "x2e_raw_tplus1.csv").write_text("")
    return e2x, x2e

# -------------------- FE --------------------
def feature_engineering(df: pd.DataFrame, train_pass_max=None) -> pd.DataFrame:
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
    # PASS 진행도: train max 기준으로만 스케일
    if train_pass_max is None:
        pm = float(d[PASS_COL].max()) if PASS_COL in d.columns else 1.0
    else:
        pm = float(train_pass_max)
    d["FM_PASS_squared"]  = d[PASS_COL] ** 2
    d["FM_PASS_progress"] = d[PASS_COL] / (pm if pm > 0 else 1.0)
    return d

def _load_bundle(prefix: str) -> dict:
    """
    models/{prefix}_regressor_tplus1.pkl
    models/{prefix}_selector_tplus1.pkl           (없어도 동작)
    models/{prefix}_feature_cols_tplus1.pkl
    models/{prefix}_feature_medians_tplus1.pkl
    models/{prefix}_test_plates_tplus1.pkl
    """
    import joblib, os
    from pathlib import Path

    def _take_model(m):
        # 실수로 (model, meta) 형태가 저장되는 경우 방어
        if hasattr(m, "predict"):
            return m
        if isinstance(m, (list, tuple)):
            for x in m:
                if hasattr(x, "predict"):
                    return x
        if isinstance(m, dict):
            for x in m.values():
                if hasattr(x, "predict"):
                    return x
        raise TypeError("로딩된 객체에서 predict 가능한 모델을 찾지 못했습니다.")

    b = {}
    mpath = MODELS_DIR / f"{prefix}_regressor_tplus1.pkl"
    selpath = MODELS_DIR / f"{prefix}_selector_tplus1.pkl"
    colspath = MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl"
    medpath = MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl"
    testpath = MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl"

    b["model"] = _take_model(joblib.load(mpath))
    # selector는 없을 수 있음(항등 변환 제공)
    try:
        b["selector"] = joblib.load(selpath)
        # scikit-learn 호환: transform이 없으면 항등 처리
        if not hasattr(b["selector"], "transform"):
            b["selector"] = None
    except Exception:
        b["selector"] = None

    b["cols"] = list(joblib.load(colspath))
    b["medians"] = joblib.load(medpath)
    # 테스트 판 목록(교집합 계산에 사용)
    try:
        b["test_plates"] = set(joblib.load(testpath))
    except Exception:
        b["test_plates"] = None

    return b

def _predict_next_from_row(row: pd.Series, bundle: dict) -> float:
    """
    row: t 시점의 한 행(Series). 여기에는 TARGET_COL이 없어야 하며, 있어도 drop됨.
    bundle: _load_bundle 로딩 결과.
    """
    # 1) 대상 열을 훈련시 순서대로 강제 재구성
    desired_cols = list(bundle["cols"])
    # Series → dict → 1행 DataFrame, 누락 컬럼은 NaN으로 채친 뒤 medians로 대치
    # (DataFrame([...])로 감싸면 Series.select_dtypes 오류가 원천 차단됨)
    X_row = pd.DataFrame([{c: row.get(c, np.nan) for c in desired_cols}], columns=desired_cols)

    # 2) 훈련 중앙값으로 결측 대치
    if isinstance(bundle["medians"], dict):
        X_row = X_row.fillna(bundle["medians"])
    else:
        # dict가 아닐 경우를 대비한 안전장치
        X_row = X_row.fillna(0.0)

    # 3) 선택기 적용(없으면 항등)
    if bundle["selector"] is not None:
        X_sel = bundle["selector"].transform(X_row)
    else:
        # 모델이 scikit-learn 계열이면 넘파이 배열/DF 모두 허용
        X_sel = X_row.values

    # 4) 예측
    pred = bundle["model"].predict(X_sel)
    # xgboost/sklearn 호환: 1차원 배열 보장
    return float(np.asarray(pred).ravel()[0])

# -------------------- 안전한 설계행렬 준비 --------------------
def _prepare_design(tr_df: pd.DataFrame, te_df: pd.DataFrame):
    # train 기준 PASS max로 FE
    tr_pass_max = tr_df[PASS_COL].max()
    tr_fe = feature_engineering(tr_df, train_pass_max=tr_pass_max)
    te_fe = feature_engineering(te_df, train_pass_max=tr_pass_max)

    # RM train-only 모드: test의 RM_*는 train 중앙값으로만 채움(누수 차단)
    if USE_RM_TRAINONLY:
        rm_cols = [c for c in tr_fe.columns if c.startswith("RM_")]
        if rm_cols:
            med_rm = tr_fe[rm_cols].median(numeric_only=True).to_dict()
            te_fe[rm_cols] = np.nan
            te_fe[rm_cols] = te_fe[rm_cols].fillna(med_rm)

    # 타깃 구성(+ 퍼뮤트)
    ytr = tr_fe[TARGET_COL].to_numpy(float)
    yte = te_fe[TARGET_COL].to_numpy(float)
    if PERMUTE_TARGET:
        rng = np.random.default_rng(SEED)
        rng.shuffle(ytr)

    # Δ-타깃
    if USE_DELTA_TARGET:
        if CUR_WARP not in tr_fe.columns or CUR_WARP not in te_fe.columns:
            raise RuntimeError("Δ-타깃을 쓰려면 warping_index_current_pass 컬럼이 필요합니다.")
        ytr = (ytr - tr_fe[CUR_WARP].to_numpy(float))
        yte = (yte - te_fe[CUR_WARP].to_numpy(float))

    # X 준비(train med/열순서)
    def to_X(d: pd.DataFrame):
        X_all = d.drop(columns=[c for c in EXCLUDE_FROM_X if c in d.columns], errors="ignore")
        X_num = X_all.select_dtypes(include=[np.number]).copy()
        return X_num

    Xtr_raw = to_X(tr_fe)
    Xte_raw = to_X(te_fe)

    med = Xtr_raw.median(numeric_only=True).to_dict()
    Xtr = Xtr_raw.fillna(med).copy()
    Xte = Xte_raw.fillna(med).copy()
    cols = Xtr.columns.tolist()
    Xte = Xte.reindex(columns=cols, fill_value=np.nan).fillna(med)

    return tr_fe, te_fe, Xtr, Xte, ytr, yte, med, cols

# -------------------- 값 기반 누수 스크리너 --------------------
def drop_value_leaks(X: pd.DataFrame, y: np.ndarray, corr_th=0.995):
    # (train 전용) y와 거의 동일한 컬럼/상관 과대 컬럼 제거
    dropped = []
    y0 = y - y.mean()
    ys = y0.std()
    if ys < 1e-12:
        return X, dropped
    for c in list(X.columns):
        xc = X[c].to_numpy(float)
        xs = np.std(xc)
        if xs < 1e-12:
            continue
        r = np.corrcoef(xc, y0)[0,1]
        if not np.isfinite(r):
            continue
        if abs(r) >= corr_th:
            dropped.append(c)
    if dropped:
        X = X.drop(columns=dropped, errors="ignore")
    return X, dropped

# -------------------- Pair OOF Target Encoding(Δ용) --------------------
def pair_id(df: pd.DataFrame) -> pd.Series:
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

def oof_pair_te(tr_fe: pd.DataFrame, te_fe: pd.DataFrame, ytr: np.ndarray, m_smooth=10.0):
    # Δ-타깃 기준으로 인코딩(Δ가 아니면 y 그대로)
    base = ytr.copy()
    gkf = GroupKFold(n_splits=5)
    groups = tr_fe[PLATE_COL].astype(str).values
    pid_tr = pair_id(tr_fe).values
    pid_te = pair_id(te_fe).values

    oof = np.zeros_like(base, dtype=float)
    glob_mean = base.mean()
    for tr_idx, va_idx in gkf.split(np.zeros_like(base), base, groups):
        pid_tr_fold = pid_tr[tr_idx]
        y_tr_fold = base[tr_idx]
        # fold 내 pair 통계
        df_tmp = pd.DataFrame({"pid": pid_tr_fold, "y": y_tr_fold})
        stat = df_tmp.groupby("pid")["y"].agg(["mean","count"]).reset_index()
        stat["enc"] = (stat["mean"]*stat["count"] + m_smooth*glob_mean)/(stat["count"]+m_smooth)
        enc_map = dict(zip(stat["pid"], stat["enc"]))
        oof[va_idx] = np.array([enc_map.get(k, glob_mean) for k in pid_tr[va_idx]])

    # test 인코딩: 전체 train으로 스무딩
    df_all = pd.DataFrame({"pid": pid_tr, "y": base})
    stat_all = df_all.groupby("pid")["y"].agg(["mean","count"]).reset_index()
    stat_all["enc"] = (stat_all["mean"]*stat_all["count"] + m_smooth*glob_mean)/(stat_all["count"]+m_smooth)
    enc_map_all = dict(zip(stat_all["pid"], stat_all["enc"]))
    te_enc = np.array([enc_map_all.get(k, glob_mean) for k in pid_te])

    return oof, te_enc

# -------------------- Hetero Weights (pair 분산 역가중) --------------------
def hetero_weights(tr_fe: pd.DataFrame, ytr: np.ndarray):
    pid = pair_id(tr_fe)
    df = pd.DataFrame({"pid": pid, "y": ytr})
    stat = df.groupby("pid")["y"].agg(["var","count"]).reset_index()
    stat["w"] = 1.0 / (stat["var"].replace(0.0, np.nan).fillna(stat["var"].median() if stat["var"].notna().any() else 1.0) + 1e-3)
    w_map = dict(zip(stat["pid"], stat["w"]))
    w = np.array([w_map.get(k, 1.0) for k in pid])
    return w

# -------------------- 안전 K-best --------------------
def safe_kbest(Xtr: pd.DataFrame, ytr: np.ndarray, k: int, mandatory=None):
    mandatory = list(mandatory or [])
    cols = Xtr.columns.tolist()
    if not cols:
        return Xtr, list(cols)
    # 상관 기반 점수
    y0 = ytr - ytr.mean()
    ys = y0.std()
    scores = []
    for c in cols:
        xc = Xtr[c].to_numpy(float)
        xs = xc.std()
        if xs < 1e-12 or ys < 1e-12:
            scores.append((c, 0.0))
            continue
        r = np.corrcoef((xc - xc.mean())/max(xs,1e-12), y0/ys)[0,1]
        s = float(np.nan_to_num(r*r, nan=0.0, posinf=0.0, neginf=0.0))
        scores.append((c, s))
    scores.sort(key=lambda z: z[1], reverse=True)
    ranked = [c for c,_ in scores if c not in mandatory]
    keep = mandatory + ranked[:max(1, min(k - len(mandatory), len(ranked)))]
    Xtr2 = Xtr[keep].copy()
    return Xtr2, keep

# -------------------- 모델 생성 --------------------
def build_model(name: str, seed: int):
    if name == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=seed, tree_method="hist",
            n_estimators=1200, learning_rate=0.05,
            max_depth=4, min_child_weight=6,
            gamma=1.0, subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.05, reg_lambda=2.0,
            n_jobs=0, eval_metric="rmse"
        )
    elif name == "RandomForest":
        return RandomForestRegressor(
            random_state=seed, n_estimators=800, n_jobs=-1,
            max_depth=18, min_samples_leaf=5, max_features="sqrt"
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")

# -------------------- group holdout fit --------------------
def fit_with_group_holdout(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           seed: int, sample_weight=None, algo_name="XGBoost"):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_i, va_i = next(gss.split(X, y, groups))
    Xfit, yfit = X[tr_i], y[tr_i]
    Xval, yval = X[va_i], y[va_i]
    sw_fit = sample_weight[tr_i] if sample_weight is not None else None
    if algo_name == "XGBoost":
        try:
            model.fit(Xfit, yfit, sample_weight=sw_fit,
                      eval_set=[(Xval, yval)], early_stopping_rounds=50, verbose=False)
        except TypeError:
            model.fit(Xfit, yfit, sample_weight=sw_fit)
    else:
        model.fit(Xfit, yfit, sample_weight=sw_fit)
    return model

# -------------------- 한 전이 방향 학습 --------------------
def train_one_direction(tr_df_raw: pd.DataFrame, te_df_raw: pd.DataFrame, tag: str, algo_name: str):
    prefix = "e2x" if tag.upper()=="E2X" else "x2e"

    # 앵커 분할: test plate 고정
    anchor_path = MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl"
    if anchor_path.exists():
        test_plates = set(joblib.load(anchor_path))
        tr_mask = ~tr_df_raw[PLATE_COL].astype(str).isin(test_plates)
        te_mask =  tr_df_raw[PLATE_COL].astype(str).isin(test_plates)
        # 재사용이 의미 없으므로, 없는 경우 새로 분할
        if not te_mask.any():
            gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
            tr_idx, te_idx = next(gss.split(tr_df_raw, tr_df_raw[TARGET_COL], tr_df_raw[PLATE_COL].astype(str)))
            tr_df = tr_df_raw.iloc[tr_idx].copy()
            te_df = tr_df_raw.iloc[te_idx].copy()
            test_plates = set(te_df[PLATE_COL].astype(str).unique())
            joblib.dump(sorted(list(test_plates)), anchor_path)
        else:
            tr_df = tr_df_raw[tr_mask].copy()
            te_df = tr_df_raw[te_mask].copy()
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        tr_idx, te_idx = next(gss.split(tr_df_raw, tr_df_raw[TARGET_COL], tr_df_raw[PLATE_COL].astype(str)))
        tr_df = tr_df_raw.iloc[tr_idx].copy()
        te_df = tr_df_raw.iloc[te_idx].copy()
        test_plates = set(te_df[PLATE_COL].astype(str).unique())
        joblib.dump(sorted(list(test_plates)), anchor_path)

    # 설계행렬 준비
    tr_fe, te_fe, Xtr, Xte, ytr, yte, med, cols = _prepare_design(tr_df, te_df)

    # 값 기반 누수 스크리너(train only)
    Xtr, dropped = drop_value_leaks(Xtr, ytr, corr_th=0.995)
    if dropped:
        Xte = Xte.drop(columns=[c for c in dropped if c in Xte.columns], errors="ignore")

    # Pair TE(선택)
    if USE_PAIR_TE:
        oof_enc, te_enc = oof_pair_te(tr_fe, te_fe, ytr if USE_DELTA_TARGET else ytr)
        Xtr["pair_te_mean"] = oof_enc
        Xte["pair_te_mean"] = te_enc

    # 안전 K-best (mandatory 최소화: Δ-타깃이면 CUR_WARP는 굳이 강제X)
    mandatory = []
    Xtr2, keep_cols = safe_kbest(Xtr, ytr, k=min(SELECT_K, Xtr.shape[1]), mandatory=mandatory)
    Xte2 = Xte.reindex(columns=keep_cols, fill_value=np.nan).fillna({c: Xtr2[c].median() for c in Xtr2.columns})

    # 이분산 가중(선택)
    sw = None
    if USE_HETERO_W:
        sw = hetero_weights(tr_fe.loc[Xtr2.index], ytr)

    # Monotone (CUR_WARP 단조 +)
    monotone_constraints = None
    if USE_MONO_CUR and algo_name == "XGBoost":
        monotone_constraints = []
        for c in keep_cols:
            if c == CUR_WARP:
                monotone_constraints.append(1)
            else:
                monotone_constraints.append(0)

    # 모델 생성/학습
    model = build_model(algo_name, SEED)
    if (monotone_constraints is not None) and hasattr(model, "set_params"):
        model.set_params(monotone_constraints=monotone_constraints)

    groups = tr_fe.loc[Xtr2.index, PLATE_COL].astype(str).values
    model = fit_with_group_holdout(model, Xtr2.to_numpy(float), ytr, groups, SEED, sample_weight=sw, algo_name=algo_name)

    # 예측/복원(Δ 사용 시 +CUR_WARP)
    ytr_pred = model.predict(Xtr2.to_numpy(float))
    yte_pred = model.predict(Xte2.to_numpy(float))
    if USE_DELTA_TARGET:
        ytr_pred = ytr_pred + tr_fe.loc[Xtr2.index, CUR_WARP].to_numpy(float)
        yte_pred = yte_pred + te_fe.loc[Xte2.index, CUR_WARP].to_numpy(float)
        ytr_true = tr_fe.loc[Xtr2.index, TARGET_COL].to_numpy(float)
        yte_true = te_fe.loc[Xte2.index, TARGET_COL].to_numpy(float)
    else:
        ytr_true, yte_true = ytr, yte

    train_r2 = float(r2_score(ytr_true, ytr_pred))
    test_r2  = float(r2_score(yte_true, yte_pred))
    train_rmse = float(np.sqrt(mean_squared_error(ytr_true, ytr_pred)))
    test_rmse  = float(np.sqrt(mean_squared_error(yte_true, yte_pred)))

    # 산출물 저장
    joblib.dump(model,      MODELS_DIR / f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(keep_cols,  MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl")
    joblib.dump(med,        MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl")
    joblib.dump(sorted(list(test_plates)), MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl")

    return {
        "algorithm": algo_name,
        "train_r2": train_r2, "test_r2": test_r2,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "train_n": int(len(ytr_true)), "test_n": int(len(yte_true)),
        "keep_cols": keep_cols
    }

# -------------------- Strict Test-only Rollforward R² --------------------
def strict_rollforward_r2(final_df: pd.DataFrame) -> float:
    """
    - 훈련 시 저장된 번들(e2x/x2e)을 사용하여,
      두 모델의 '테스트 판 교집합'에 대해서만 t→t+1 예측 R²을 계산.
    - 내부적으로 최소한의 FE를 보정(모델이 기대하는 열이 없을 때 대비).
    - TARGET(=t+1)이나 누수 위험 열은 절대 feature로 넣지 않음.
    """
    # 번들 로드
    e2x_b = _load_bundle("e2x")
    x2e_b = _load_bundle("x2e")

    # 교집합 테스트 판
    e_set = e2x_b.get("test_plates") or set()
    x_set = x2e_b.get("test_plates") or set()
    test_plates = e_set.intersection(x_set) if e_set and x_set else set()
    if not test_plates:
        print("[Strict RF] 교집합 Test 판이 없습니다. (저장된 테스트 판 정보가 없거나 교집합이 비어있음)")
        return float("nan")

    # 최소 FE 보정: 모델이 기대하는 열이 일부 없더라도 중앙값으로 대치 가능하지만,
    # 기본적인 파생(예: PASS 기반)은 생성해 예측 안정성 확보
    def _ensure_min_fe(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # PASS 파생
        if PASS_COL in d.columns:
            if "FM_PASS_squared" not in d.columns:
                d["FM_PASS_squared"] = d[PASS_COL] ** 2
            if "FM_PASS_progress" not in d.columns:
                # train의 max 기준이 가장 안전하지만, 여기서는 전체 max로 근사
                pmax = float(d[PASS_COL].max()) if len(d) else 1.0
                d["FM_PASS_progress"] = d[PASS_COL] / (pmax if pmax > 0 else 1.0)
        # FM_* 집계 요약(없으면 생성)
        fm_cols = [c for c in d.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
        if fm_cols:
            blk = d[fm_cols]
            if "FM_mean" not in d.columns:   d["FM_mean"]  = blk.mean(axis=1)
            if "FM_std"  not in d.columns:   d["FM_std"]   = blk.std(axis=1)
            if "FM_max"  not in d.columns:   d["FM_max"]   = blk.max(axis=1)
            if "FM_min"  not in d.columns:   d["FM_min"]   = blk.min(axis=1)
            if "FM_range" not in d.columns:  d["FM_range"] = d["FM_max"] - d["FM_min"]
            if "FM_cv" not in d.columns:     d["FM_cv"]    = d["FM_std"] / (d["FM_mean"].abs() + 1e-8)
        return d

    df = _ensure_min_fe(final_df)

    rows = []
    # 판별 → PASS 오름차순 정렬 → t와 t+1이 모두 존재하는 페어만 예측
    for plate, sub in df.groupby(PLATE_COL):
        if str(plate) not in test_plates:
            continue
        sub = sub.sort_values(PASS_COL)
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p + 1) not in idx.index:
                continue
            # t시점 행에서 타깃 제거
            row_t = idx.loc[p].drop(labels=[TARGET_COL], errors="ignore")
            # 전이 방향에 맞는 번들 선택
            bundle = e2x_b if (p % 2 == 1) else x2e_b

            try:
                pred = _predict_next_from_row(row_t, bundle)
                gt_next = idx.loc[p + 1].get(CUR_WARP, None)
                if pd.notna(gt_next):
                    rows.append({"plate": plate, "from": p, "to": p + 1, "pred": float(pred), "gt_next": float(gt_next)})
            except Exception as ex:
                # 행 단위 예외는 스킵 (전체 중단 방지)
                # print(f"[Strict RF] skip {plate} {p}->{p+1}: {ex}")
                continue

    if not rows:
        print("[Strict RF] 유효한 예측 페어가 없습니다.")
        return float("nan")

    dfp = pd.DataFrame(rows)
    return float(r2_score(dfp["gt_next"], dfp["pred"]))

# -------------------- Permutation audit --------------------
def quick_permutation_r2(tr_df: pd.DataFrame, tag: str, algo_name: str):
    # train을 다시 분할해 permute-y에 대한 OOS R²를 추정(대략 0 근처 기대)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED+7)
    tr_idx, te_idx = next(gss.split(tr_df, tr_df[TARGET_COL], tr_df[PLATE_COL].astype(str)))
    tr_sub = tr_df.iloc[tr_idx].copy()
    te_sub = tr_df.iloc[te_idx].copy()

    tr_fe, te_fe, Xtr, Xte, ytr, yte, med, cols = _prepare_design(tr_sub, te_sub)

    rng = np.random.default_rng(SEED+123)
    rng.shuffle(ytr)

    # 누수 스크리너
    Xtr, dropped = drop_value_leaks(Xtr, ytr, corr_th=0.995)
    if dropped:
        Xte = Xte.drop(columns=[c for c in dropped if c in Xte.columns], errors="ignore")

    # TE
    if USE_PAIR_TE:
        oof_enc, te_enc = oof_pair_te(tr_fe, te_fe, ytr)
        Xtr["pair_te_mean"] = oof_enc
        Xte["pair_te_mean"] = te_enc

    Xtr2, keep = safe_kbest(Xtr, ytr, k=min(SELECT_K, Xtr.shape[1]))
    Xte2 = Xte.reindex(columns=keep, fill_value=np.nan).fillna({c: Xtr2[c].median() for c in Xtr2.columns})

    model = build_model(algo_name, SEED+9)
    groups = tr_fe.loc[Xtr2.index, PLATE_COL].astype(str).values
    model = fit_with_group_holdout(model, Xtr2.to_numpy(float), ytr, groups, SEED+9, None, algo_name)
    y_pred = model.predict(Xte2.to_numpy(float))
    return float(r2_score(yte, y_pred))

# -------------------- 요약/메트릭 저장 --------------------
def write_summary(experiment_id: str, ts: datetime, e2x_res: dict, x2e_res: dict, strict_rf_r2: float,
                  perm_e2x: float, perm_x2e: float, summary_path: Path):
    lines = []
    lines.append("Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
    lines.append("="*72)
    lines.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} 작성")
    lines.append(f"실험 ID: {experiment_id}\n")

    lines.append("[모델 성능 - R2]")
    lines.append(f"* E2X | Algo: {e2x_res['algorithm']} | Train R2: {e2x_res['train_r2']:.4f} | Test R2: {e2x_res['test_r2']:.4f} (RMSE {e2x_res['train_rmse']:.4f}/{e2x_res['test_rmse']:.4f})")
    lines.append(f"* X2E | Algo: {x2e_res['algorithm']} | Train R2: {x2e_res['train_r2']:.4f} | Test R2: {x2e_res['test_r2']:.4f} (RMSE {x2e_res['train_rmse']:.4f}/{x2e_res['test_rmse']:.4f})\n")
    avg = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))
    lines.append(f"[평균 Test R2] {avg:.4f}")
    lines.append(f"[Strict Test‑only Rollforward R²] {strict_rf_r2 if np.isfinite(strict_rf_r2) else float('nan')}")
    if LEAK_AUTOCHECK:
        lines.append("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        lines.append(f"- E2X permuted R²: {perm_e2x:.4f}")
        lines.append(f"- X2E permuted R²: {perm_x2e:.4f}")
    lines.append("\n[DEBUG] 설정/분할")
    lines.append(f"- PERMUTE_TARGET: {PERMUTE_TARGET}")
    lines.append(f"- USE_DELTA_TARGET={USE_DELTA_TARGET} | USE_PAIR_TE={USE_PAIR_TE} | USE_HETERO_W={USE_HETERO_W} | USE_MONO_CUR={USE_MONO_CUR} | USE_RM_TRAINONLY={USE_RM_TRAINONLY}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # metrics CSV(집계용)
    MET = METRICS_DIR / "last_run_metrics.csv"
    MET.parent.mkdir(parents=True, exist_ok=True)
    MET.write_text("experiment_id,avg_test_r2,summary_path\n", encoding="utf-8")
    with MET.open("a", encoding="utf-8") as f:
        f.write(f"{experiment_id},{avg},{summary_path}\n")

# -------------------- main --------------------
def main():
    ts = datetime.now()
    experiment_id = _next_experiment_id()
    summary_path = SUM_DIR / f"{experiment_id}_test_summary.txt"
    script_archive = SCRIPT_DIR / f"{experiment_id}_test.py"

    try:
        final_df = load_and_merge_data_tplus1()
        e2x_raw, x2e_raw = split_cross(final_df)

        # 각 방향 학습
        e2x_res = train_one_direction(e2x_raw, e2x_raw, "E2X", algo_name=E2X_ALGO)
        x2e_res = train_one_direction(x2e_raw, x2e_raw, "X2E", algo_name=X2E_ALGO)

        # Strict rollforward R²
        strict_rf = strict_rollforward_r2(final_df)

        # Permutation audit
        perm_e2x = quick_permutation_r2(e2x_raw, "E2X", algo_name=E2X_ALGO) if LEAK_AUTOCHECK else float("nan")
        perm_x2e = quick_permutation_r2(x2e_raw, "X2E", algo_name=X2E_ALGO) if LEAK_AUTOCHECK else float("nan")

        # 요약/메트릭 저장
        write_summary(experiment_id, ts, e2x_res, x2e_res, strict_rf, perm_e2x, perm_x2e, summary_path)

        # 스크립트 스냅샷 저장
        try:
            this = Path(__file__).resolve()
            shutil.copy2(this, script_archive)
        except Exception:
            pass

        print(summary_path.read_text(encoding="utf-8"))

    except Exception as e:
        print(f"[실패] {type(e).__name__}: {e}")
        print(traceback.format_exc())
        err_path = REPORTS / "last_error_traceback.txt"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise

if __name__ == "__main__":
    main()
