# 003_experiment.py 잠재 누수 수정 버전

import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import traceback
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
def train_model(df,
                dataset_name: str,           # "E2X" 또는 "X2E"
                algorithm: str = "XGBoost",  # "XGBoost" | "RandomForest" | "LightGBM"(있다면)
                seed: int = 42,
                test_size: float = 0.2,
                select_k: int = 120):
    """
    - 셀렉터 객체를 저장하지 않고, '선택된 컬럼 리스트'만 저장하도록 수정.
    - 추론 시 저장된 cols로 직접 reindex하여 모델 입력 구성.
    """
    import numpy as np, pandas as pd, joblib
    from pathlib import Path
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
    try:
        from xgboost import XGBRegressor
    except Exception:
        XGBRegressor = None
    try:
        import lightgbm as lgb
    except Exception:
        lgb = None

    TARGET_COL = "warping_index_target"
    CUR_WARP   = "warping_index_current_pass"
    PLATE_COL  = "FM_날판번호"
    PASS_COL   = "FM_PASS NO N"
    MONTH_COL  = "FM_압연월"
    EXCLUDE    = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}

    MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 유틸 ----------
    def _hard_leak_guard(cols: list[str]) -> list[str]:
        bad_kw = ("target", "label", "y_", "_y", "oof", "te_", "_te", "pred", "_hat")
        return [c for c in cols if not any(k in c.lower() for k in bad_kw)]

    def _base_fe_train_test(tr: pd.DataFrame, te: pd.DataFrame):
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
            pm = float(pass_max) if pass_max and pass_max > 0 else 1.0
            d["FM_PASS_progress"] = d[PASS_COL] / pm
            return d
        pass_max_train = float(tr[PASS_COL].max()) if len(tr) else 1.0
        tr = _fe(tr, pass_max_train)
        te = _fe(te, pass_max_train)
        return tr, te

    def _build_model(name: str):
        if name == "XGBoost":
            if XGBRegressor is None:
                raise RuntimeError("xgboost가 설치되어 있지 않습니다.")
            return XGBRegressor(
                objective="reg:squarederror",
                random_state=seed, tree_method="hist",
                n_estimators=800, learning_rate=0.05,
                max_depth=6, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.0,
                n_jobs=0, eval_metric="rmse"
            )
        elif name == "RandomForest":
            return RandomForestRegressor(
                random_state=seed, n_estimators=600, n_jobs=-1,
                max_depth=20, min_samples_leaf=5, max_features="sqrt"
            )
        elif name == "LightGBM" and lgb is not None:
            return lgb.LGBMRegressor(
                random_state=seed, n_estimators=800, learning_rate=0.05,
                num_leaves=64, subsample=0.8, colsample_bytree=0.8
            )
        else:
            raise ValueError(f"알 수 없는 알고리즘: {name}")

    # ---------- 데이터 준비 & 분할 ----------
    data = df.copy()
    if dataset_name.upper() == "E2X":
        data = data[data[PASS_COL] % 2 == 1].copy()
        prefix = "e2x"
    else:
        data = data[data[PASS_COL] % 2 == 0].copy()
        prefix = "x2e"

    anchor_path = Path("models") / f"{prefix}_test_plates_tplus1.pkl"
    X_all = data.drop(columns=[c for c in EXCLUDE if c in data.columns], errors="ignore")
    feature_cols_all = _hard_leak_guard(X_all.columns.tolist())
    X_all = X_all[feature_cols_all].select_dtypes(include=[np.number]).copy()
    y_all = data[TARGET_COL].values
    groups_all = data[PLATE_COL].astype(str).values

    if anchor_path.exists():
        test_plates = set(joblib.load(anchor_path))
        tr_mask = ~data[PLATE_COL].astype(str).isin(test_plates)
        te_mask =  data[PLATE_COL].astype(str).isin(test_plates)
        tr_idx = np.where(tr_mask.values)[0]; te_idx = np.where(te_mask.values)[0]
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(X_all, y_all, groups=groups_all))
        test_plates = sorted(data.iloc[te_idx][PLATE_COL].astype(str).unique())
        joblib.dump(test_plates, anchor_path)

    tr_df_raw = data.iloc[tr_idx].copy()
    te_df_raw = data.iloc[te_idx].copy()
    y_tr_raw  = y_all[tr_idx].copy()
    y_te      = y_all[te_idx].copy()

    # ---------- FE (train 기준 스케일) ----------
    tr_df_fe, te_df_fe = _base_fe_train_test(tr_df_raw, te_df_raw)

    feats_tr = tr_df_fe.drop(columns=[c for c in EXCLUDE if c in tr_df_fe.columns], errors="ignore")
    feats_te = te_df_fe.drop(columns=[c for c in EXCLUDE if c in te_df_fe.columns], errors="ignore")
    feat_cols = _hard_leak_guard(feats_tr.columns.tolist())
    feats_tr = feats_tr[feat_cols].select_dtypes(include=[np.number]).copy()
    feats_te = feats_te[feat_cols].select_dtypes(include=[np.number]).copy()
    med = feats_tr.median(numeric_only=True).to_dict()
    feats_tr = feats_tr.fillna(med); feats_te = feats_te.fillna(med)

    # ---------- 이상치 가드 (train만) ----------
    q_low, q_high = 0.01, 0.99
    bounds = {}
    for c in feats_tr.columns:
        lo, hi = np.quantile(feats_tr[c].values, [q_low, q_high])
        bounds[c] = (float(lo), float(hi))
    y_lo, y_hi = np.quantile(y_tr_raw, [0.005, 0.995])
    keep = (y_tr_raw >= y_lo) & (y_tr_raw <= y_hi)
    for c,(lo,hi) in bounds.items():
        keep &= (feats_tr[c].values >= lo) & (feats_tr[c].values <= hi)
    feats_tr = feats_tr.loc[keep].reset_index(drop=True)
    y_tr = y_tr_raw[keep]

    for c,(lo,hi) in bounds.items():
        if c in feats_te.columns:
            feats_te[c] = feats_te[c].clip(lower=lo, upper=hi)

    # ---------- 상관 기반 K-Best (간단 구현) ----------
    # 필수 컬럼: 현재 패스의 warping
    mandatory = ["warping_index_current_pass"] if "warping_index_current_pass" in feats_tr.columns else []
    # 상수/비유한 제외
    x = feats_tr.values
    finite_mask = np.isfinite(x).all(axis=0)
    var = feats_tr.var(numeric_only=True).reindex(feats_tr.columns).fillna(0.0).values
    keep_mask = finite_mask & (var > 1e-12)
    cols_kept = [c for c,m in zip(feats_tr.columns.tolist(), keep_mask) if m]
    # 상관 F-score
    Xc = feats_tr[cols_kept]
    y0 = y_tr - y_tr.mean()
    ys = y0.std() if y0.std() >= 1e-12 else 1e-12
    xm = Xc.values.mean(axis=0); xs = Xc.values.std(axis=0); xs[xs < 1e-12] = 1e-12
    xr = (Xc.values - xm)/xs; yr = y0/ys
    r = (xr.T @ yr) / (len(y_tr)-1)
    r2 = np.clip(r**2, 0.0, 1.0 - 1e-12)
    F = (r2 / (1.0 - r2)) * (len(y_tr) - 2)

    mand = [c for c in mandatory if c in cols_kept]
    others = [c for c in cols_kept if c not in mand]
    k_remain = max(1, min(int(select_k) - len(mand), len(others)))
    # others의 F값 정렬
    idx_map = {c:i for i,c in enumerate(cols_kept)}
    order = np.argsort([F[idx_map[c]] for c in others])[::-1] if others else []
    selected_cols = mand + [others[i] for i in order[:k_remain]]

    X_tr = feats_tr[selected_cols].values
    X_te = feats_te[selected_cols].values

    # ---------- 학습 (내부 plate-group 홀드아웃) ----------
    model = _build_model(algorithm)
    from sklearn.model_selection import GroupShuffleSplit
    grp_tr = tr_df_fe.loc[keep, "FM_날판번호"].astype(str).values
    tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X_tr, y_tr, groups=grp_tr))
    Xfit, yfit = X_tr[tr_i], y_tr[tr_i]
    Xval, yval = X_tr[va_i], y_tr[va_i]

    if algorithm == "XGBoost":
        try:
            model.fit(Xfit, yfit, eval_set=[(Xval, yval)], early_stopping_rounds=50, verbose=False)
        except TypeError:
            model.fit(Xfit, yfit)
    else:
        model.fit(Xfit, yfit)

    # ---------- 평가 ----------
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)
    res = {
        "train_r2":  float(r2_score(y_tr, y_tr_pred)),
        "test_r2":   float(r2_score(y_te, y_te_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_tr, y_tr_pred))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te, y_te_pred))),
        "train_samples": int(len(y_tr)), "test_samples": int(len(y_te)),
        "best_params": getattr(model, "get_params", lambda: {})(),
    }

    # ---------- 저장 (셀렉터 객체 저장 제거) ----------
    joblib.dump(model,               MODELS_DIR / f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(selected_cols,       MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl")
    joblib.dump(med,                 MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl")
    joblib.dump(sorted(list(test_plates)), MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl")

    return res, set(test_plates)

# ----------------- 7) 번갈아 예측(roll-forward) -------------------
def _load_bundle(prefix: str):
    return {
        "model":    joblib.load(MODELS_DIR / f"{prefix}_regressor_tplus1.pkl"),
        # "selector": 제거
        "cols":     joblib.load(MODELS_DIR / f"{prefix}_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(MODELS_DIR / f"{prefix}_feature_medians_tplus1.pkl"),
    }


def _predict_next_from_row(row: pd.Series, bundle: dict) -> float:
    cols = list(bundle["cols"])
    med  = bundle["medians"]
    X_row = pd.DataFrame([row.reindex(cols)], columns=cols).fillna(med)
    # 셀렉터 transform 제거 → 선택 컬럼 그대로 사용
    return float(bundle["model"].predict(X_row.values)[0])


# --- rollforward 예측: '교집합 Test 판'만 평가하는 함수 추가 ---
def rollforward_r2_on_test_only(all_featured_df: pd.DataFrame) -> float:
    from sklearn.metrics import r2_score
    e2x_bundle = {
        "model":    joblib.load(Path("models")/"e2x_regressor_tplus1.pkl"),
        "cols":     joblib.load(Path("models")/"e2x_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(Path("models")/"e2x_feature_medians_tplus1.pkl"),
    }
    x2e_bundle = {
        "model":    joblib.load(Path("models")/"x2e_regressor_tplus1.pkl"),
        "cols":     joblib.load(Path("models")/"x2e_feature_cols_tplus1.pkl"),
        "medians":  joblib.load(Path("models")/"x2e_feature_medians_tplus1.pkl"),
    }
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
                bundle = e2x_bundle if (p % 2 == 1) else x2e_bundle
                cols = list(bundle["cols"]); med = bundle["medians"]
                X_row = pd.DataFrame([row_t.reindex(cols)], columns=cols).fillna(med)
                pred = float(bundle["model"].predict(X_row.values)[0])
                gt_next = float(idx.loc[p+1].get("warping_index_current_pass"))
                rows.append({"plate": plate, "from": p, "to": p+1, "pred": pred, "gt_next": gt_next})
            except Exception as ex:
                row_errors.append({"plate": plate, "from": p, "to": p+1, "error": repr(ex)})
                continue

    if row_errors:
        err_csv = REPORTS_DIR / "rollforward_test_only_row_errors.csv"
        pd.DataFrame(row_errors).to_csv(err_csv, index=False)
        print(f"[Strict RF] 행 단위 에러 {len(row_errors)}건 → {err_csv}")

    if not rows:
        print("[Strict RF] 유효한 예측 행이 없습니다.")
        return float("nan")

    df = pd.DataFrame(rows)
    return float(r2_score(df["gt_next"], df["pred"]))

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
