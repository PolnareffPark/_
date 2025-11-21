# -*- coding: utf-8 -*-
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import traceback
from optuna.trial import TrialState
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# (선택) SciPy가 있으면 검정 수행
try:
    from scipy.stats import kruskal, levene
except Exception:
    kruskal = levene = None

# (선택) LightGBM
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

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

# 컬럼 기본 키
TARGET_COL = "warping_index_target"   # t+1 타깃(절대 레벨)
PASS_COL   = "FM_PASS NO N"
PLATE_COL  = "FM_날판번호"
MONTH_COL  = "FM_압연월"
CUR_WARP   = "warping_index_current_pass"

# X에서 제외할 기본 컬럼
EXCLUDE_FROM_X = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL, "pair_id", "delta_target"}

# ---- 전략 스위치(필요에 맞게 on/off 가능) ----
CFG = dict(
    use_delta_target = True,          # Δ-타깃 회귀: y = target - current; 추론 시 current + pred로 복원
    use_pair_te      = True,          # 쌍(pair=p→p+1) 타깃 인코딩(훈련 평균 Δ/타깃)
    use_pass_onehot  = True,          # PASS 원-핫(feat)
    use_interactions = True,          # 진행도×요약치/현재값 상호작용 feat
    use_prev_feats   = True,          # prev_warping, delta_prev, plate-relative feat
    use_pair_weights = True,          # 쌍별 이분산 가중치(분산 역수)
    robust_objective = "mae",         # "none" | "mae"(권장) | "huber"(xgb만)
    use_pair_biascal = True,          # 쌍별 잔차 바이어스 보정(2단계 캘리브레이션)
)

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
        "strict_rollforward_r2",
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

def _make_pair_series(df: pd.DataFrame) -> pd.Series:
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

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

    # ---- (추가) plate 내부 시계열 기반 피처 ----
    final_df = final_df.sort_values([PLATE_COL, PASS_COL]).reset_index(drop=True)
    # 1) 이전 값/증분
    final_df["prev_warping"] = final_df.groupby(PLATE_COL)[CUR_WARP].shift(1)
    final_df["delta_prev"]   = final_df[CUR_WARP] - final_df["prev_warping"]
    # 2) plate 상대값(오프셋 완화)
    plate_stats = final_df.groupby(PLATE_COL)[CUR_WARP].agg(plate_mean_CUR="mean", plate_std_CUR="std").reset_index()
    final_df = final_df.merge(plate_stats, on=PLATE_COL, how="left")
    final_df["CUR_minus_plate_mean"] = final_df[CUR_WARP] - final_df["plate_mean_CUR"]
    final_df["CUR_div_plate_mean"]   = final_df[CUR_WARP] / (final_df["plate_mean_CUR"].abs() + 1e-8)
    # 3) Δ-타깃(옵션): y_delta = target - current
    final_df["delta_target"] = final_df[TARGET_COL] - final_df[CUR_WARP]

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

# ---------------- 3) Feature Engineering ------------------------
def feature_engineering(df: pd.DataFrame, dataset_name: str, output_path: Path):
    print(f"\n[Step 3] Feature Engineering - {dataset_name}")
    df = df.copy()

    # FM_* 행요약
    fm_cols = [c for c in df.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    if fm_cols:
        fm_blk = df[fm_cols]
        df["FM_mean"]  = fm_blk.mean(axis=1)
        df["FM_std"]   = fm_blk.std(axis=1)
        df["FM_max"]   = fm_blk.max(axis=1)
        df["FM_min"]   = fm_blk.min(axis=1)
        df["FM_range"] = df["FM_max"] - df["FM_min"]
        df["FM_cv"]    = df["FM_std"] / (df["FM_mean"].abs() + 1e-8)

    # PASS 파생
    if PASS_COL in df.columns:
        df["FM_PASS_squared"]  = df[PASS_COL] ** 2
        df["FM_PASS_progress"] = df[PASS_COL] / df[PASS_COL].max()

    # 단계(Stage) 강화
    if PASS_COL in df.columns:
        p = df[PASS_COL].astype(int)
        df["stage_early"] = (p <= 4).astype(int)
        df["stage_mid"]   = ((p >= 5) & (p <= 8)).astype(int)
        df["stage_late"]  = (p >= 9).astype(int)
        prog = df["FM_PASS_progress"].clip(0,1)
        df["pass_sin"] = np.sin(2*np.pi*prog)
        df["pass_cos"] = np.cos(2*np.pi*prog)

    if CFG["use_pass_onehot"] and PASS_COL in df.columns:
        dmy = pd.get_dummies(df[PASS_COL].astype(int), prefix="pass", drop_first=False)
        df = pd.concat([df, dmy], axis=1)

    # 상호작용
    if CFG["use_interactions"]:
        if CUR_WARP in df.columns and "FM_PASS_progress" in df.columns:
            df["progress_x_current"] = df["FM_PASS_progress"] * df[CUR_WARP]
        if "FM_range" in df.columns and "FM_PASS_progress" in df.columns:
            df["range_x_progress"] = df["FM_range"] * df["FM_PASS_progress"]
        if "FM_cv" in df.columns and CUR_WARP in df.columns:
            df["cv_x_current"] = df["FM_cv"] * df[CUR_WARP]

    # pair id (모델 입력엔 제외, TE/보정에서만 사용)
    df["pair_id"] = _make_pair_series(df)

    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path}")
    return df

# -------------------- 4) 이상치 제거 ----------------------------
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

# ------------------- 5) 알고리즘 비교 ---------------------------
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
    if lgb is not None:
        algos["LightGBM"] = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=64)

    best_name, best_score = None, -np.inf
    for name, model in algos.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        print(f"  {name:12s}: R2 = {scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_score, best_name = scores.mean(), name
    print(f"\n  → 최적 알고리즘: {best_name} (CV R2 = {best_score:.4f})")
    return best_name, best_score

# ------------ 6) 학습/튜닝 + 전처리/보정 메타 저장 ---------------
def train_model(df: pd.DataFrame, dataset_name: str, algorithm: str, seed: int = 42):
    print("\n" + "="*80)
    print(f"[Step 6] {dataset_name} 모델 학습 - {algorithm}")
    print("="*80)

    data = df.copy()
    data["pair_id"] = data.get("pair_id", _make_pair_series(data))

    # --- y 설정: Δ-타깃 or 절대 레벨 ---
    if CFG["use_delta_target"] and "delta_target" in data.columns:
        y = data["delta_target"].copy()
        target_mode = "delta"
    else:
        y = data[TARGET_COL].copy()
        target_mode = "level"

    # --- X 원천: 제외 컬럼 제거 후 수치형만 ---
    X_all = data.drop(columns=[c for c in EXCLUDE_FROM_X if c in data.columns], errors="ignore")
    X_all = X_all.select_dtypes(include=[np.number])

    # 그룹 분할(판 기준)
    groups = data.get(PLATE_COL, pd.Series(np.arange(len(data)), index=data.index)).values
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X_all, y, groups))
    X_tr_raw, X_te_raw = X_all.iloc[tr_idx].copy(), X_all.iloc[te_idx].copy()
    y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
    pair_tr, pair_te = data.iloc[tr_idx]["pair_id"], data.iloc[te_idx]["pair_id"]

    # --- (옵션) 쌍 타깃 인코딩: 훈련 평균(Δ 또는 레벨) ---
    if CFG["use_pair_te"]:
        te_map = y_tr.groupby(pair_tr).mean()
        global_mean = float(y_tr.mean())
        pair_te_all = data["pair_id"].map(te_map).fillna(global_mean)
        X_tr_raw["pair_te"] = pair_te_all.iloc[tr_idx].values
        X_te_raw["pair_te"] = pair_te_all.iloc[te_idx].values
        X_all["pair_te"] = pair_te_all.values
    else:
        te_map = pd.Series(dtype=float)
        global_mean = float("nan")

    # 안전한 결측 대체(평균/중앙값 등)
    med_all = X_all.median(numeric_only=True)
    X_tr_raw = X_tr_raw.fillna(med_all)
    X_te_raw = X_te_raw.fillna(med_all)

    # --- 피처 선택 ---
    selector = SelectKBest(score_func=f_regression, k=min(120, X_tr_raw.shape[1]))
    selector.fit(X_tr_raw, y_tr)
    X_tr = selector.transform(X_tr_raw); X_te = selector.transform(X_te_raw)

    # --- (옵션) 쌍별 가중치 ---
    fit_kwargs = {}
    if CFG["use_pair_weights"]:
        s = y_tr.groupby(pair_tr).std().fillna(y_tr.std())
        n = y_tr.groupby(pair_tr).size()
        w_pair = 1.0 / (s**2 + 1.0)  # 안정화용 +1.0
        w_pair *= np.minimum(1.0, n / 30.0)  # 소표본 완충
        w_tr = pair_tr.map(w_pair).fillna(float(w_pair.mean())).values
        fit_kwargs["sample_weight"] = w_tr

    # --- 모델/옵티마이저 ---
    def build_model(trial):
        if algorithm == "XGBoost":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "tree_method": "hist",
            }
            if CFG["robust_objective"] == "mae":
                params["objective"] = "reg:absoluteerror"
            elif CFG["robust_objective"] == "huber":
                params["objective"] = "reg:pseudohubererror"
            return XGBRegressor(random_state=seed, n_jobs=1, **params)
        elif algorithm == "LightGBM" and lgb is not None:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "objective": "regression",
            }
            if CFG["robust_objective"] == "mae":
                params["objective"] = "quantile"; params["alpha"] = 0.5  # Median 회귀
            return lgb.LGBMRegressor(random_state=seed, n_jobs=1, **params)
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 8, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt","log2",None]),
            }
            return RandomForestRegressor(random_state=seed, n_jobs=1, **params)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: cross_val_score(build_model(t), X_tr, y_tr, cv=3, scoring="r2", n_jobs=1).mean(),
                   n_trials=50, show_progress_bar=False)
    best_params = study.best_trial.params if any(t.state==TrialState.COMPLETE for t in study.trials) else {}
    model = build_model(optuna.trial.FixedTrial(best_params))
    model.fit(X_tr, y_tr, **fit_kwargs)

    # ---- 예측/복원 ----
    y_tr_pred = model.predict(X_tr); y_te_pred = model.predict(X_te)
    if target_mode == "delta":
        cur_tr = data.iloc[tr_idx][CUR_WARP].values
        cur_te = data.iloc[te_idx][CUR_WARP].values
        y_tr_pred = cur_tr + y_tr_pred
        y_te_pred = cur_te + y_te_pred
        y_tr_true = data.iloc[tr_idx][TARGET_COL].values
        y_te_true = data.iloc[te_idx][TARGET_COL].values
    else:
        y_tr_true, y_te_true = y_tr.values, y_te.values

    # ---- (옵션) 2단계 캘리브레이션(쌍별 잔차 평균 보정) ----
    if CFG["use_pair_biascal"]:
        res_tr = y_tr_true - y_tr_pred
        pair_bias = pd.Series(res_tr, index=pair_tr).groupby(pair_tr).mean().to_dict()
    else:
        pair_bias = {}

    # 테스트에 보정 적용
    y_te_pred_cal = y_te_pred + pd.Series(pair_te).map(pair_bias).fillna(0.0).values

    results = {
        "train_r2": float(r2_score(y_tr_true, y_tr_pred)),
        "test_r2":  float(r2_score(y_te_true, y_te_pred_cal)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_tr_true, y_tr_pred))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te_true, y_te_pred_cal))),
        "train_samples": int(len(y_tr_true)), "test_samples": int(len(y_te_true)),
        "best_params": model.get_params(),
    }
    print(f"  Train R2: {results['train_r2']:.4f} | Test R2: {results['test_r2']:.4f}")

    # --- 아티팩트 저장 ---
    prefix = dataset_name.lower()
    joblib.dump(model,    MODELS_DIR / f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(selector, MODELS_DIR / f"{prefix}_selector_tplus1.pkl")
    joblib.dump(best_params, MODELS_DIR / f"{prefix}_regressor_params_tplus1.pkl")
    joblib.dump(med_all,  MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl")
    joblib.dump(list(X_all.columns), MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl")
    # 훈련 메타
    joblib.dump(CFG,      MODELS_DIR / f"{prefix}_training_cfg_tplus1.pkl")
    joblib.dump({"target_mode": target_mode}, MODELS_DIR / f"{prefix}_meta_tplus1.pkl")
    # pair TE/바이어스 맵
    joblib.dump({"pair_te_map": (te_map.to_dict() if len(te_map)>0 else {}), "global_mean": float(global_mean if not np.isnan(global_mean) else 0.0)},
                MODELS_DIR / f"{prefix}_pair_te_map_tplus1.pkl")
    joblib.dump({"pair_bias": pair_bias, "bias_global": float(0.0)}, MODELS_DIR / f"{prefix}_pair_bias_tplus1.pkl")
    # test plate 목록
    test_plates = set(data.iloc[te_idx][PLATE_COL].astype(str).unique())
    joblib.dump(sorted(list(test_plates)), MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl")

    return results, test_plates

# ----------------- 7) 번갈아 예측(roll-forward) -------------------
def _load_bundle(prefix: str):
    bundle = {
        "model":    joblib.load(MODELS_DIR / f"{prefix}_regressor_tplus1.pkl"),
        "selector": joblib.load(MODELS_DIR / f"{prefix}_selector_tplus1.pkl"),
        "cols":     joblib.load(MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl"),
        "meta":     joblib.load(MODELS_DIR / f"{prefix}_meta_tplus1.pkl") if (MODELS_DIR / f"{prefix}_meta_tplus1.pkl").exists() else {"target_mode":"delta"},
        "cfg":      joblib.load(MODELS_DIR / f"{prefix}_training_cfg_tplus1.pkl") if (MODELS_DIR / f"{prefix}_training_cfg_tplus1.pkl").exists() else CFG,
        "pair_bias":joblib.load(MODELS_DIR / f"{prefix}_pair_bias_tplus1.pkl") if (MODELS_DIR / f"{prefix}_pair_bias_tplus1.pkl").exists() else {"pair_bias":{}, "bias_global":0.0},
        "pair_te":  joblib.load(MODELS_DIR / f"{prefix}_pair_te_map_tplus1.pkl") if (MODELS_DIR / f"{prefix}_pair_te_map_tplus1.pkl").exists() else {"pair_te_map":{}, "global_mean":0.0},
    }
    return bundle

def _predict_next_from_row(row: pd.Series, bundle: dict, transition_prefix: str) -> float:
    """
    단일 행(row)로부터 다음 패스(t+1)를 예측.
    - Δ 타깃이면 current를 더해 레벨 복원
    - pair TE(훈련 평균)와 pair 잔차 보정 적용
    """
    cols = bundle["cols"]; med = bundle["medians"]
    cfg  = bundle["cfg"]; meta = bundle["meta"]
    pair_bias = bundle["pair_bias"]; te_map = bundle["pair_te"]

    # 원본 row에서 필요한 열을 정렬/보강
    # pair_te가 feature로 사용되었다면, 훈련 map으로 채워 넣는다.
    row_dict = row.reindex(cols).to_dict()
    if "pair_te" in cols:
        p = int(row.get(PASS_COL))
        key = f"{p}→{p+1}"
        row_dict["pair_te"] = float(te_map.get("pair_te_map", {}).get(key, te_map.get("global_mean", 0.0)))

    X_row = pd.DataFrame([row_dict], columns=cols)
    X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(med.reindex(cols))
    X_sel = bundle["selector"].transform(X_row)
    y_hat = float(bundle["model"].predict(X_sel)[0])

    # Δ면 레벨 복원
    if meta.get("target_mode","delta") == "delta":
        y_hat = float(row.get(CUR_WARP)) + y_hat

    # pair 보정
    try:
        p = int(row.get(PASS_COL))
        key = f"{p}→{p+1}"
        y_hat += float(pair_bias.get("pair_bias", {}).get(key, pair_bias.get("bias_global", 0.0)))
    except Exception:
        pass
    return y_hat

def rollforward_predict_all_plates(all_featured_df: pd.DataFrame, e2x_bundle: dict, x2e_bundle: dict, restrict_plates: set = None) -> pd.DataFrame:
    outs = []
    df = all_featured_df
    if restrict_plates is not None and len(restrict_plates) > 0:
        df = df[df[PLATE_COL].astype(str).isin({str(p) for p in restrict_plates})]

    for plate in sorted(df[PLATE_COL].dropna().unique()):
        sub = df[df[PLATE_COL] == plate].sort_values(PASS_COL)
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index:  # t+1 없으면 skip
                continue
            row_t = idx.loc[p]
            feat_row = row_t.drop(labels=[TARGET_COL], errors="ignore")
            if p % 2 == 1:
                bundle = e2x_bundle; prefix = "e2x"
            else:
                bundle = x2e_bundle; prefix = "x2e"
            pred = _predict_next_from_row(feat_row, bundle, prefix)
            gt_next = idx.loc[p+1].get(CUR_WARP, None)
            outs.append({PLATE_COL: plate, "from_pass": p, "to_pass": p+1, "pred": pred, "gt_next": gt_next})
    return pd.DataFrame(outs) if outs else pd.DataFrame(columns=[PLATE_COL,"from_pass","to_pass","pred","gt_next"])

# --- rollforward R²: 교집합 Test 판만(엄격) ---
def rollforward_r2_on_test_only(all_featured_df: pd.DataFrame) -> float:
    e2x_bundle = _load_bundle("e2x")
    x2e_bundle = _load_bundle("x2e")
    e2x_test = set(joblib.load(MODELS_DIR/"e2x_test_plates_tplus1.pkl")) if (MODELS_DIR/"e2x_test_plates_tplus1.pkl").exists() else set()
    x2e_test = set(joblib.load(MODELS_DIR/"x2e_test_plates_tplus1.pkl")) if (MODELS_DIR/"x2e_test_plates_tplus1.pkl").exists() else set()
    strict_plates = e2x_test.intersection(x2e_test)
    if not strict_plates:
        print("[Strict RF] 교집합 Test 판이 없습니다.")
        return float("nan")
    rf = rollforward_predict_all_plates(all_featured_df, e2x_bundle, x2e_bundle, restrict_plates=strict_plates)
    mask = rf["gt_next"].notna()
    return float(r2_score(rf.loc[mask,"gt_next"], rf.loc[mask,"pred"])) if mask.any() else float("nan")

# ------------- 8) Δ-분석: 패스쌍별 분포/검정/리포트 ----------------
def analyze_transition_variability(final_df: pd.DataFrame) -> dict:
    df = final_df.copy()
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["pair"] = _make_pair_series(df)

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
    df = final_df.copy()
    # 쌍/전이/델타 생성
    df["pair"] = _make_pair_series(df)
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["transition"] = np.where(df[PASS_COL] % 2 == 1, "E2X", "X2E")

    g = df.groupby(["transition", "pair"])["delta"]

    # ★ 명명된 집계로 'n' 컬럼을 확정 생성 (KeyError 방지)
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
    stats.loc[stats["n"] <= 1, ["se"]] = np.nan
    stats["ci95_lo"] = stats["mean"] - 1.96 * stats["se"]
    stats["ci95_hi"] = stats["mean"] + 1.96 * stats["se"]

    # 샘플 수 적음 경고 플래그
    stats["n_flag"] = np.where(stats["n"] < 30, "small-n(<30)", "ok")

    out = REPORTS_DIR / "pair_delta_stats.csv"
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
    srf_r2 = summary_data.get("strict_rollforward_r2", None)
    if srf_r2 is not None and not np.isnan(srf_r2):
        lines.append(f"[Strict Test‑only Rollforward R²] {srf_r2:.4f}")

    # Δ-분석 요약
    v = summary_data.get("variability", {})
    for tag in ["e2x","x2e"]:
        vv = v.get(tag, {})
        lines.append(f"\n[Δ-분석 {tag.upper()}] std_max/min={vv.get('std_ratio', np.nan):.3f} | "
                     f"Kruskal p={vv.get('kw_p', np.nan)} | Levene p={vv.get('lev_p', np.nan)}")
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

def write_howto_txt_for_cross_models():
    path = REPORTS_DIR / "howto_cross_transition_models.txt"
    doc = []
    doc.append("교차 전이(t→t+1) 회귀 파이프라인 HOW-TO (TXT)")
    doc.append("="*72)
    doc.append("")
    doc.append("1) 목적")
    doc.append("- 같은 판 내부에서 Pass t의 정보로 Pass t+1의 warping_index를 예측한다.")
    doc.append("- 홀수→짝수(E→X)와 짝수→홀수(X→E) 전이를 분리 학습한다.")
    doc.append("")
    doc.append("2) 데이터 병합/타깃")
    doc.append("- GT(entry/exit csv 세로결합)에서 판번호(PB*), PASS를 추출한다.")
    doc.append("- FM(posco2_*.csv)와 판/패스로 inner-join 후 정렬한다.")
    doc.append("- 타깃: 같은 판에서 warping_index를 shift(-1) → warping_index_target.")
    doc.append("- 현재 패스의 warping_index는 warping_index_current_pass로 보존(피처로 사용).")
    doc.append("- RM(posco1_*.csv)는 판 단위로 수치컬럼 5통계(mean/max/min/std/median) 집계 후 FM에 left-join.")
    doc.append("")
    doc.append("3) 전이 데이터 분리")
    doc.append("- E→X: PASS 홀수 행(Entry t)만 사용하여 t→t+1(Exit) 타깃을 학습.")
    doc.append("- X→E: PASS 짝수 행(Exit t)만 사용하여 t→t+1(Entry) 타깃을 학습.")
    doc.append("")
    doc.append("4) Feature Engineering")
    doc.append("- FM_* 수치 전체의 행단위 요약 6종(FM_mean/std/max/min/range/cv).")
    doc.append("- PASS 파생(제곱/진행도/Stage/사인·코사인/원-핫) + 상호작용, prev/plate 상대 피처.")
    doc.append("- RM 판단위 집계 특성 다수 + warping_index_current_pass 포함.")
    doc.append("")
    doc.append("5) 이상치 제거")
    doc.append("- IsolationForest(5%) → KNN-타깃일관성(상위 5% 제거) 2단계.")
    doc.append("- 결측치는 중앙값 대체. 이후 스케일링은 모델 입력에 사용하지 않음.")
    doc.append("")
    doc.append("6) 학습/검증")
    doc.append("- GroupShuffleSplit(plate 기준) 8:2 분할로 누수 방지.")
    doc.append("- SelectKBest(f_regression) 상위 ≤120개 피처 선택.")
    doc.append("- 알고리즘: XGBoost/LightGBM/RandomForest + Optuna 튜닝.")
    doc.append("- 옵션: Δ-타깃, pair 타깃 인코딩, pair 가중치, 로버스트 손실, 잔차 바이어스 보정.")
    doc.append("- 출력: 모델/셀렉터/중앙값/열순서/테스트판목록/보정맵/TE맵 저장.")
    doc.append("")
    doc.append("7) 평가 지표")
    doc.append("- Test R²: 각 전이(E→X, X→E).")
    doc.append("- Rollforward R²: 모든 판의 모든 t→t+1 전이를 한 번에 예측하여 전역 R² 산출.")
    doc.append("- Strict Rollforward R²: 두 모델의 테스트 판 교집합만 사용.")
    path.write_text("\n".join(doc), encoding="utf-8")
    print(f"✓ HOW-TO TXT 저장: {path}")

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

        # 6) 학습/튜닝(+보정/메타 저장)
        e2x_ret = train_model(e2x_clean, "E2X", e2x_algo)
        x2e_ret = train_model(x2e_clean, "X2E", x2e_algo)

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

        # 7) 번갈아 예측 및 R2 (전체 판)
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
            "strict_rollforward_r2": strict_rollforward_r2,
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
            "strict_rollforward_r2": strict_rollforward_r2,
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
        print(top[["experiment_id","average_test_r2","e2x_test_r2","x2e_test_r2","rollforward_r2","strict_rollforward_r2"]].to_string(index=False))

        success = True

        write_howto_txt_for_cross_models()
        
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
