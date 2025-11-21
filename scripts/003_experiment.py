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
    from scipy.stats import kruskal, levene, spearmanr
except Exception:
    kruskal = levene = spearmanr = None

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

# ---------------- 3) Feature Engineering (동일 규칙) ---------------
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

    if PASS_COL in df.columns:
        df["FM_PASS_squared"]  = df[PASS_COL] ** 2
        df["FM_PASS_progress"] = df[PASS_COL] / df[PASS_COL].max()

    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path}")
    return df

# -------------------- 4) 이상치 제거 (동일 규칙) -------------------
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

# ------------------- 5) 알고리즘 비교 (동일 규칙) -------------------
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

# ------------ 6) 학습/튜닝 + 전처리 메타 저장(중앙값/열순서) -----------
def train_model(df: pd.DataFrame, dataset_name: str, algorithm: str, seed: int = 42):
    from sklearn.model_selection import GroupShuffleSplit, cross_val_score
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.metrics import mean_squared_error, r2_score
    from xgboost import XGBRegressor
    import numpy as np, joblib

    TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL = "warping_index_target", "FM_날판번호", "FM_PASS NO N", "FM_압연월"
    EXCLUDE_FROM_X = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}

    data = df.copy()
    X_raw = data.drop(columns=[c for c in EXCLUDE_FROM_X if c in data.columns], errors="ignore").select_dtypes(include=[np.number])
    y = data[TARGET_COL]
    med = X_raw.median(numeric_only=True); X_raw = X_raw.fillna(med)

    groups = data.get(PLATE_COL, pd.Series(np.arange(len(data)), index=data.index)).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gss.split(X_raw, y, groups))

    # test plate id 목록 확보
    test_plates = set(data.iloc[te_idx][PLATE_COL].astype(str).unique())

    X_tr_raw, X_te_raw = X_raw.iloc[tr_idx], X_raw.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    selector = SelectKBest(score_func=f_regression, k=min(120, X_tr_raw.shape[1]))
    selector.fit(X_tr_raw, y_tr)
    X_tr = selector.transform(X_tr_raw); X_te = selector.transform(X_te_raw)

    # 간단 Optuna (기존과 동일)
    import optuna
    from optuna.trial import TrialState
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
            }
            return XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", **params)
        elif algorithm == "LightGBM":
            import lightgbm as lgb
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
            }
            return lgb.LGBMRegressor(random_state=seed, n_jobs=1, **params)
        else:
            from sklearn.ensemble import RandomForestRegressor
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
    model = build_model(optuna.trial.FixedTrial(best_params)); model.fit(X_tr, y_tr)

    y_tr_pred, y_te_pred = model.predict(X_tr), model.predict(X_te)
    results = {
        "train_r2": float(r2_score(y_tr, y_tr_pred)),
        "test_r2":  float(r2_score(y_te, y_te_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_tr, y_tr_pred))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te, y_te_pred))),
        "train_samples": int(len(y_tr)), "test_samples": int(len(y_te)),
        "best_params": model.get_params(),
    }

    prefix = dataset_name.lower()
    joblib.dump(model,    Path("models")/f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(selector, Path("models")/f"{prefix}_selector_tplus1.pkl")
    joblib.dump(best_params, Path("models")/f"{prefix}_regressor_params_tplus1.pkl")
    joblib.dump(med,      Path("models")/f"{prefix}_feature_medians_tplus1.pkl")
    # ★ test plate 목록도 저장
    joblib.dump(sorted(list(test_plates)), Path("models")/f"{prefix}_test_plates_tplus1.pkl")

    return results, test_plates

# ----------------- 7) 번갈아 예측(roll-forward) -------------------
def _load_bundle(prefix: str):
    return {
        "model":    joblib.load(MODELS_DIR / f"{prefix}_regressor_tplus1.pkl"),
        "selector": joblib.load(MODELS_DIR / f"{prefix}_selector_tplus1.pkl"),
        "cols":     joblib.load(MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl"),
    }

def _predict_next_from_row(row: pd.Series, bundle: dict) -> float:
    X_row = pd.DataFrame([row.reindex(bundle["cols"]).to_dict()]).fillna(bundle["medians"])
    X_sel = bundle["selector"].transform(X_row)
    return float(bundle["model"].predict(X_sel)[0])

# --- rollforward 예측: '교집합 Test 판'만 평가하는 함수 추가 ---
def rollforward_r2_on_test_only(all_featured_df: pd.DataFrame) -> float:
    """
    교집합 Test 판만 대상으로 t→t+1 한 칸 예측을 수행하고 전역 R² 계산.
    Series에 select_dtypes를 호출하던 부분을 제거하고,
    훈련 때 저장해 둔 'feature_cols_tplus1.pkl' 과 'feature_medians_tplus1.pkl' 을 기준으로
    안전하게 1-행 DataFrame을 구성한다.
    """
    from sklearn.metrics import r2_score

    # 번들 로드
    e2x_bundle = {
        "model":    joblib.load(Path("models")/"e2x_regressor_tplus1.pkl"),
        "selector": joblib.load(Path("models")/"e2x_selector_tplus1.pkl"),
        "cols":     joblib.load(Path("models")/"e2x_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(Path("models")/"e2x_feature_medians_tplus1.pkl"),
    }
    x2e_bundle = {
        "model":    joblib.load(Path("models")/"x2e_regressor_tplus1.pkl"),
        "selector": joblib.load(Path("models")/"x2e_selector_tplus1.pkl"),
        "cols":     joblib.load(Path("models")/"x2e_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(Path("models")/"x2e_feature_medians_tplus1.pkl"),
    }
    # 교집합 Test 판
    e2x_test = set(joblib.load(Path("models")/"e2x_test_plates_tplus1.pkl"))
    x2e_test = set(joblib.load(Path("models")/"x2e_test_plates_tplus1.pkl"))
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

                # 현재 전이 방향에 맞는 번들 선택
                if p % 2 == 1:   # E→X
                    bundle = e2x_bundle
                else:            # X→E
                    bundle = x2e_bundle

                # ★ Series → 1행 DataFrame: 훈련 시 사용한 feature_cols 순서대로 재배열
                desired_cols = list(bundle["cols"])
                X_row = pd.DataFrame([row_t.reindex(desired_cols)], columns=desired_cols)
                # 결측은 훈련 중앙값으로 채움
                X_row = X_row.fillna(bundle["medians"])

                # SelectKBest 셀렉터와 동일한 변환 적용
                X_sel = bundle["selector"].transform(X_row)
                pred = float(bundle["model"].predict(X_sel)[0])

                gt_next = float(idx.loc[p+1].get("warping_index_current_pass"))
                rows.append({"plate": plate, "from": p, "to": p+1, "pred": pred, "gt_next": gt_next})
            except Exception as ex:
                # 행 단위 에러를 누적 기록(전체 실험 중단 방지)
                row_errors.append({"plate": plate, "from": p, "to": p+1, "error": repr(ex)})
                continue

    # 행 단위 에러 로그 저장(있을 때만)
    if row_errors:
        err_csv = REPORTS_DIR / "rollforward_test_only_row_errors.csv"
        pd.DataFrame(row_errors).to_csv(err_csv, index=False)
        print(f"[Strict RF] 행 단위 에러 {len(row_errors)}건 → {err_csv}")

    if not rows:
        print("[Strict RF] 유효한 예측 행이 없습니다.")
        return float("nan")

    df = pd.DataFrame(rows)
    r2 = float(r2_score(df["gt_next"], df["pred"]))
    return r2

def rollforward_predict_all_plates(all_featured_df: pd.DataFrame, e2x_bundle: dict, x2e_bundle: dict) -> pd.DataFrame:
    outs = []
    for plate in sorted(all_featured_df[PLATE_COL].dropna().unique()):
        sub = all_featured_df[all_featured_df[PLATE_COL] == plate].sort_values(PASS_COL)
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index:  # t+1 없으면 skip
                continue
            row_t = idx.loc[p]
            feat_row = row_t.drop(labels=[TARGET_COL], errors="ignore")
            bundle = e2x_bundle if (p % 2 == 1) else x2e_bundle
            pred = _predict_next_from_row(feat_row, bundle)
            gt_next = idx.loc[p+1].get(CUR_WARP, None)
            outs.append({PLATE_COL: plate, "from_pass": p, "to_pass": p+1, "pred": pred, "gt_next": gt_next})
    return pd.DataFrame(outs) if outs else pd.DataFrame(columns=[PLATE_COL,"from_pass","to_pass","pred","gt_next"])

# ------------- 8) Δ-분석: 패스쌍별 분포/검정/리포트 ----------------
def analyze_transition_variability(final_df: pd.DataFrame) -> dict:
    df = final_df.copy()
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["pair"] = df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

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
    import numpy as np, pandas as pd
    from pathlib import Path

    PLATE_COL, PASS_COL = "FM_날판번호", "FM_PASS NO N"
    TARGET_COL, CUR_WARP = "warping_index_target", "warping_index_current_pass"

    df = final_df.copy()
    # 쌍/전이/델타 생성
    df["pair"] = df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["transition"] = np.where(df[PASS_COL] % 2 == 1, "E2X", "X2E")

    g = df.groupby(["transition", "pair"])["delta"]

    # ★ 명명된 집계로 'n' 컬럼을 확정 생성
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
    # n<=1이면 CI 계산 불가 → NaN
    stats.loc[stats["n"] <= 1, ["se"]] = np.nan
    stats["ci95_lo"] = stats["mean"] - 1.96 * stats["se"]
    stats["ci95_hi"] = stats["mean"] + 1.96 * stats["se"]

    # 샘플 수 적음 경고 플래그
    stats["n_flag"] = np.where(stats["n"] < 30, "small-n(<30)", "ok")

    out = Path("reports") / "pair_delta_stats.csv"
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

    # Δ-분석 요약
    v = summary_data.get("variability", {})
    for tag in ["e2x","x2e"]:
        vv = v.get(tag, {})
        lines.append(f"\n[Δ-분석 {tag.upper()}] std_max/min={vv.get('std_ratio', np.nan):.3f} | "
                     f"Kruskal p={vv.get('kw_p', np.nan)} | Levene p={vv.get('lev_p', np.nan)}")
        # 상위 3개 |mean| 큰 패스쌍
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
    path = Path("reports") / "howto_cross_transition_models.txt"
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
    doc.append("- PASS 파생 2종(FM_PASS_squared, FM_PASS_progress).")
    doc.append("- RM 판단위 집계 특성 다수 + warping_index_current_pass 포함.")
    doc.append("")
    doc.append("5) 이상치 제거")
    doc.append("- IsolationForest(5%) → KNN-타깃일관성(상위 5% 제거) 2단계.")
    doc.append("- 결측치는 중앙값 대체. 이후 스케일링은 모델 입력에 사용하지 않음.")
    doc.append("")
    doc.append("6) 학습/검증")
    doc.append("- GroupShuffleSplit(plate 기준) 8:2 분할로 누수 방지.")
    doc.append("- SelectKBest(f_regression) 상위 ≤120개 피처 선택.")
    doc.append("- 알고리즘: XGBoost/LightGBM/RandomForest 중 비교 + Optuna 튜닝.")
    doc.append("- 출력: 모델/셀렉터/중앙값/테스트판목록을 저장.")
    doc.append("")
    doc.append("7) 평가 지표")
    doc.append("- Test R²: 각 전이(E→X, X→E) 홀드아웃 세트에서 개별 산출.")
    doc.append("- Rollforward R²: 모든 판의 모든 t→t+1 전이를 한 번에 예측하여 전역 R² 산출.")
    doc.append("- Strict Rollforward R²(권장): 두 모델의 테스트 판 교집합만 사용해 전역 R² 산출.")
    doc.append("")
    doc.append("8) Δ-분석")
    doc.append("- Δ := warping_index_target − warping_index_current_pass.")
    doc.append("- 전이(E2X/X2E)별로 쌍(pair = p→p+1) 그룹화하여 n, mean, std, median, 25%/75%, 95% CI 계산.")
    doc.append("- Kruskal/Levene으로 쌍 간 분포/분산 이질성 검정.")
    doc.append("")
    doc.append("9) 주의점")
    doc.append("- rollforward는 기본 설정상 Train+Test 모두 포함이므로 과대추정 가능 → Strict 버전 사용 권장.")
    doc.append("- Δ 이질성이 크면 전이/단계별 전략(아래 §10)을 병행.")
    doc.append("")
    doc.append("10) 권장 전략 요약")
    doc.append("- Δ-타깃 회귀(상세는 본 보고 참조), 쌍/단계 타깃인코딩, Stage-wise 소모델,")
    doc.append("- 쌍별 가중치, Huber/Quantile 손실, Residual Bias 보정 등.")
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

        # 6) 학습/튜닝
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

        # 7) 번갈아 예측 및 R2
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
        all_fe = feature_engineering(final_df, "All", PROCESSED_DIR / "all_featured_tplus1.csv")
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
        print(top[["experiment_id","average_test_r2","e2x_test_r2","x2e_test_r2","rollforward_r2"]].to_string(index=False))

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
