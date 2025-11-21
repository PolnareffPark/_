import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
REPORTS_SUMMARY_DIR = REPORTS_DIR / "summaries"
REPORTS_CODE_DIR = REPORTS_DIR / "scripts"
EXPERIMENT_LOG = REPORTS_DIR / "experiments_log.csv"
MODELS_DIR = Path("models")
IMAGES_DIR = Path("images")
SCRIPT_PATH = Path(__file__).resolve()


def _ensure_dirs():
    for directory in [
        PROCESSED_DIR,
        REPORTS_DIR,
        REPORTS_SUMMARY_DIR,
        REPORTS_CODE_DIR,
        MODELS_DIR,
        IMAGES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def _next_experiment_id() -> str:
    existing_ids = []
    for summary_file in REPORTS_SUMMARY_DIR.glob("*.txt"):
        match = re.match(r"(\d{3})_", summary_file.name)
        if match:
            try:
                existing_ids.append(int(match.group(1)))
            except ValueError:
                continue
    next_idx = max(existing_ids) + 1 if existing_ids else 1
    return f"{next_idx:03d}"


def _update_experiment_log(row: dict) -> pd.DataFrame:
    columns = [
        "experiment_id",
        "timestamp",
        "entry_algorithm",
        "exit_algorithm",
        "entry_test_r2",
        "exit_test_r2",
        "average_test_r2",
        "entry_train_r2",
        "exit_train_r2",
    ]

    if EXPERIMENT_LOG.exists():
        log_df = pd.read_csv(EXPERIMENT_LOG)
    else:
        log_df = pd.DataFrame(columns=columns)

    log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    log_df = log_df.drop_duplicates(subset="experiment_id", keep="last")
    log_df["experiment_numeric"] = pd.to_numeric(log_df["experiment_id"], errors="coerce")
    log_df = log_df.sort_values("experiment_numeric", na_position="last").drop(columns="experiment_numeric")
    log_df.to_csv(EXPERIMENT_LOG, index=False)
    return log_df


def load_and_merge_data():
    print("\n" + "=" * 80)
    print("[Step 1] 데이터 병합 및 Target 변환 (Pass t → Pass t+2)")
    print("=" * 80)

    gt_entry = pd.read_csv(DATA_DIR / "entry_direction_results.csv")
    gt_exit = pd.read_csv(DATA_DIR / "exit_direction_results.csv")
    ground_truth = pd.concat([gt_entry, gt_exit], ignore_index=True)

    rm_data = pd.read_csv(DATA_DIR / "posco1_105190.csv")
    fm_data = pd.read_csv(DATA_DIR / "posco2_105190.csv")

    ground_truth = ground_truth.copy()
    ground_truth["extracted_plate"] = ground_truth["filename"].str.extract(r"(PB\d+)")
    ground_truth["extracted_pass"] = ground_truth["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    print(f"Ground Truth: {len(ground_truth)}개")
    print(f"날판번호: {ground_truth['extracted_plate'].nunique()}개")
    print(f"Pass 범위: {ground_truth['extracted_pass'].min()} ~ {ground_truth['extracted_pass'].max()}")

    merged_rm = pd.merge(
        ground_truth,
        rm_data,
        left_on=["extracted_plate", "extracted_pass"],
        right_on=["RM_날판번호", "RM_압연Pass번호"],
        how="inner",
    )

    numeric_cols = merged_rm.select_dtypes(include=[np.number]).columns.tolist()
    stat_cols = [col for col in numeric_cols if col not in [
        "RM_날판번호",
        "RM_압연Pass번호",
        "warping_index",
        "extracted_pass",
    ]]

    rm_stats = (
        merged_rm.groupby("RM_날판번호")[stat_cols]
        .agg(["mean", "max", "min", "std", "median"])
        .round(4)
    )
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    rm_stats = rm_stats.reset_index()
    print(f"RM 통계량: {rm_stats.shape[1] - 1}개 변수 생성")

    merged_fm = pd.merge(
        ground_truth,
        fm_data,
        left_on=["extracted_plate", "extracted_pass"],
        right_on=["FM_날판번호", "FM_PASS NO N"],
        how="inner",
    ).sort_values(["FM_날판번호", "FM_PASS NO N"], ascending=True)

    merged_fm["warping_index_target"] = (
        merged_fm.groupby("FM_날판번호")["warping_index"].shift(-2)
    )
    merged_fm = merged_fm.rename(
        columns={"warping_index": "warping_index_current_pass"}
    )

    before_drop = len(merged_fm)
    merged_fm = merged_fm.dropna(subset=["warping_index_target"]).reset_index(drop=True)
    print(f"Target shift: {before_drop} → {len(merged_fm)} (제거 {before_drop - len(merged_fm)})")

    final_df = pd.merge(
        merged_fm,
        rm_stats,
        left_on="FM_날판번호",
        right_on="RM_날판번호",
        how="left",
    ).drop(columns=["RM_날판번호"], errors="ignore")

    drop_cols = [
        "extracted_plate",
        "extracted_pass",
        "filename",
        "quality_grade",
        "quality_grade_current_pass",
    ]
    final_df = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns])

    print(f"최종 데이터 크기: {final_df.shape}")
    final_df.to_csv(PROCESSED_DIR / "final_merged_data_regression.csv", index=False)
    print(f"✓ 저장: {PROCESSED_DIR / 'final_merged_data_regression.csv'}")
    return final_df


def split_entry_exit(final_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("[Step 2] Entry / Exit 분리")
    print("=" * 80)

    pass_col = "FM_PASS NO N"
    if pass_col not in final_df.columns:
        raise KeyError(f"필수 컬럼 '{pass_col}' 이(가) 존재하지 않습니다.")

    entry_df = final_df[final_df[pass_col] % 2 == 1].copy()
    exit_df = final_df[final_df[pass_col] % 2 == 0].copy()

    print(f"Entry 데이터: {entry_df.shape} (홀수 pass)")
    print(f"Exit  데이터: {exit_df.shape} (짝수 pass)")

    entry_path = PROCESSED_DIR / "final_merged_data_entry_reg.csv"
    exit_path = PROCESSED_DIR / "final_merged_data_exit_reg.csv"
    entry_df.to_csv(entry_path, index=False)
    exit_df.to_csv(exit_path, index=False)
    print(f"✓ 저장: {entry_path}")
    print(f"✓ 저장: {exit_path}")
    return entry_df, exit_df


def feature_engineering(df: pd.DataFrame, dataset_name: str, output_path: Path):
    print(f"\n[Step 3] Feature Engineering - {dataset_name}")
    df = df.copy()
    fm_cols = [
        col for col in df.columns
        if col.startswith("FM_") and col not in {"FM_날판번호", "FM_PASS NO N", "FM_압연월"}
    ]

    if fm_cols:
        fm_block = df[fm_cols]
        df["FM_mean"] = fm_block.mean(axis=1)
        df["FM_std"] = fm_block.std(axis=1)
        df["FM_max"] = fm_block.max(axis=1)
        df["FM_min"] = fm_block.min(axis=1)
        df["FM_range"] = df["FM_max"] - df["FM_min"]
        df["FM_cv"] = df["FM_std"] / (df["FM_mean"].abs() + 1e-8)

    if "FM_PASS NO N" in df.columns:
        df["FM_PASS_squared"] = df["FM_PASS NO N"] ** 2
        df["FM_PASS_progress"] = df["FM_PASS NO N"] / df["FM_PASS NO N"].max()

    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path}")
    return df


def remove_outliers(input_path: Path, output_path: Path, dataset_name: str):
    print(f"\n[Step 4] 이상치 제거 - {dataset_name}")
    df = pd.read_csv(input_path)

    target_col = "warping_index_target"
    exclude_cols = {
        target_col,
        "FM_날판번호",
        "FM_PASS NO N",
        "FM_압연월",
    }
    features = [col for col in df.columns if col not in exclude_cols]

    X = df[features].fillna(df[features].median())
    y = df[target_col]

    print(f"  데이터: {df.shape[0]}개 샘플, {X.shape[1]}개 피처")

    iso = IsolationForest(contamination=0.05, random_state=42)
    mask = iso.fit_predict(X) == 1
    removed_iso = (~mask).sum()
    print(f"  IsolationForest 제거: {removed_iso}개")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(20, len(X_scaled) - 1))
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    neighbor_mean = y.values[indices[:, 1:]].mean(axis=1)
    inconsistency = np.abs(y.values - neighbor_mean)
    threshold = np.percentile(inconsistency, 95)
    mask &= inconsistency <= threshold
    removed_knn = (inconsistency > threshold).sum()
    print(f"  KNN 일관성 제거: {removed_knn}개")

    cleaned = df[mask].reset_index(drop=True)
    cleaned.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path} (최종 {cleaned.shape[0]}개)")
    return cleaned


def compare_algorithms(df: pd.DataFrame, dataset_name: str):
    print("\n" + "=" * 80)
    print(f"[Step 5] 알고리즘 비교 - {dataset_name}")
    print("=" * 80)

    target_col = "warping_index_target"
    exclude_cols = {
        target_col,
        "FM_날판번호",
        "FM_PASS NO N",
        "FM_압연월",
    }

    X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    y = df[target_col]
    X = X.fillna(X.median())

    algorithms = {
        "XGBoost": XGBRegressor(
            random_state=42,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
        ),
        "RandomForest": RandomForestRegressor(
            random_state=42,
            n_estimators=500,
            max_depth=None,
            n_jobs=-1,
        ),
    }

    if lgb is not None:
        algorithms["LightGBM"] = lgb.LGBMRegressor(
            random_state=42,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
        )

    best_algo = None
    best_score = -np.inf

    for name, model in algorithms.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"  {name:12s}: R2 = {mean_score:.4f} ± {std_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_algo = name

    print(f"\n  → 최적 알고리즘: {best_algo} (CV R2 = {best_score:.4f})")
    return best_algo, best_score


def train_model(df: pd.DataFrame, dataset_name: str, algorithm: str, seed: int = 42):
    print("\n" + "=" * 80)
    print(f"[Step 6] {dataset_name} 모델 학습 - {algorithm}")
    print("=" * 80)

    target_col = "warping_index_target"
    exclude_cols = {
        target_col,
        "FM_날판번호",
        "FM_PASS NO N",
        "FM_압연월",
    }

    data = df.copy()
    X = data.drop(columns=[c for c in exclude_cols if c in data.columns], errors="ignore")
    y = data[target_col]

    X = X.fillna(X.median(numeric_only=True))

    default_groups = pd.Series(np.arange(len(data)), index=data.index)
    groups = data.get("FM_날판번호", default_groups).values if "FM_날판번호" in data.columns else default_groups

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    selector = SelectKBest(score_func=f_regression, k=min(120, X_train_raw.shape[1]))
    selector.fit(X_train_raw, y_train)
    feature_mask = selector.get_support()
    X_train = X_train_raw.loc[:, feature_mask]
    X_test = X_test_raw.loc[:, feature_mask]

    print(f"  선택된 피처: {X_train.shape[1]}개")

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
        if algorithm == "LightGBM" and lgb is not None:
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
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 8, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        return RandomForestRegressor(random_state=seed, n_jobs=1, **params)

    def objective(trial):
        model = build_model(trial)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed:
        best_params = study.best_trial.params
        print(f"  Optuna 최적 R2: {study.best_value:.4f}")
    else:
        print("  경고: Optuna trial이 완료되지 않아 기본 하이퍼파라미터 사용")
        best_params = {}

    model = build_model(optuna.trial.FixedTrial(best_params)) if completed else build_model(optuna.trial.FixedTrial({}))
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "features": list(X_train.columns),
        "best_params": model.get_params(),
    }

    print(f"  Train R2: {results['train_r2']:.4f} | Test R2: {results['test_r2']:.4f}")
    print(f"  Train RMSE: {results['train_rmse']:.4f} | Test RMSE: {results['test_rmse']:.4f}")

    model_prefix = dataset_name.lower()
    joblib.dump(model, MODELS_DIR / f"{model_prefix}_regressor.pkl")
    joblib.dump(selector, MODELS_DIR / f"{model_prefix}_selector.pkl")
    joblib.dump(best_params, MODELS_DIR / f"{model_prefix}_regressor_params.pkl")
    print(f"  ✓ 모델 저장: {MODELS_DIR / f'{model_prefix}_regressor.pkl'}")

    return results


def write_summary(summary_data, summary_path: Path, timestamp: datetime, experiment_id: str, best_record: dict):
    lines = []
    lines.append("Pass t → Pass t+2 Regression Pipeline Summary")
    lines.append("=" * 70)
    lines.append(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} 작성됨")
    lines.append(f"실험 ID: {experiment_id}")

    lines.append("\n[데이터]")
    for item in summary_data["data_artifacts"]:
        lines.append(f"- {item}")

    lines.append("\n[모델 성능 - R2 중점]")
    for key in ["Entry", "Exit"]:
        res = summary_data["models"].get(key, {})
        lines.append(
            f"* {key} | Algorithm: {res.get('algorithm', 'N/A')} | Train R2: {res.get('train_r2', float('nan')):.4f} | Test R2: {res.get('test_r2', float('nan')):.4f}"
        )
        lines.append(
            f"  - Train RMSE: {res.get('train_rmse', float('nan')):.4f}, Test RMSE: {res.get('test_rmse', float('nan')):.4f}"
        )

    lines.append("\n[평균 R2 비교]")
    avg_r2 = float(summary_data.get("average_test_r2", float("nan")))
    lines.append(f"- 이번 실험 평균 Test R2: {avg_r2:.4f}")
    if summary_data.get("r2_gap") is not None:
        lines.append(f"- Entry 대비 Exit Test R2 차이: {summary_data['r2_gap']:+.4f}")

    best_id = best_record.get("experiment_id")
    best_avg = float(best_record.get("average_test_r2", float("nan")))
    if best_id:
        lines.append(f"- 현재 최고 평균 Test R2: {best_avg:.4f} (실험 ID: {best_id})")
        if best_id == experiment_id:
            lines.append("  → 이번 실험이 현재 최고 기록입니다.")
        else:
            delta = avg_r2 - best_avg
            lines.append(f"  → 최고 기록과의 차이: {delta:+.4f}")

    lines.append("\n[하이퍼파라미터]")
    for key, res in summary_data["models"].items():
        best_params = res.get("best_params", {})
        lines.append(f"* {key}")
        for p_key, p_val in best_params.items():
            lines.append(f"    - {p_key}: {p_val}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 요약 저장: {summary_path}")


def main():
    _ensure_dirs()

    experiment_id = _next_experiment_id()
    timestamp = datetime.now()
    summary_path = REPORTS_SUMMARY_DIR / f"{experiment_id}_pass_t_plus_2_summary.txt"
    archived_script_path = REPORTS_CODE_DIR / f"{experiment_id}_pass_t_plus_2_pipeline.py"

    final_df = load_and_merge_data()
    entry_df, exit_df = split_entry_exit(final_df)

    entry_fe_path = PROCESSED_DIR / "entry_regression_featured.csv"
    exit_fe_path = PROCESSED_DIR / "exit_regression_featured.csv"
    entry_eng = feature_engineering(entry_df, "Entry", entry_fe_path)
    exit_eng = feature_engineering(exit_df, "Exit", exit_fe_path)

    entry_clean_path = PROCESSED_DIR / "entry_regression_cleaned.csv"
    exit_clean_path = PROCESSED_DIR / "exit_regression_cleaned.csv"
    entry_clean = remove_outliers(entry_fe_path, entry_clean_path, "Entry")
    exit_clean = remove_outliers(exit_fe_path, exit_clean_path, "Exit")

    entry_algo, entry_cv = compare_algorithms(entry_clean, "Entry")
    exit_algo, exit_cv = compare_algorithms(exit_clean, "Exit")

    entry_results = train_model(entry_clean, "Entry", entry_algo)
    entry_results["algorithm"] = entry_algo
    entry_results["cv_r2"] = entry_cv

    exit_results = train_model(exit_clean, "Exit", exit_algo)
    exit_results["algorithm"] = exit_algo
    exit_results["cv_r2"] = exit_cv

    average_test_r2 = np.nanmean([entry_results["test_r2"], exit_results["test_r2"]])

    log_row = {
        "experiment_id": experiment_id,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "entry_algorithm": entry_algo,
        "exit_algorithm": exit_algo,
        "entry_test_r2": entry_results["test_r2"],
        "exit_test_r2": exit_results["test_r2"],
        "average_test_r2": average_test_r2,
        "entry_train_r2": entry_results["train_r2"],
        "exit_train_r2": exit_results["train_r2"],
    }

    log_df = _update_experiment_log(log_row)
    best_idx = log_df["average_test_r2"].astype(float).idxmax()
    best_record = log_df.loc[best_idx].to_dict()

    shutil.copy2(SCRIPT_PATH, archived_script_path)
    print(f"✓ 코드 아카이브 저장: {archived_script_path}")

    summary = {
        "data_artifacts": [
            f"Final merged: {PROCESSED_DIR / 'final_merged_data_regression.csv'}",
            f"Entry raw: {PROCESSED_DIR / 'final_merged_data_entry_reg.csv'}",
            f"Exit raw: {PROCESSED_DIR / 'final_merged_data_exit_reg.csv'}",
            f"Entry cleaned: {entry_clean_path}",
            f"Exit cleaned: {exit_clean_path}",
            f"Summary file: {summary_path}",
            f"Archived script: {archived_script_path}",
        ],
        "models": {
            "Entry": entry_results,
            "Exit": exit_results,
        },
        "average_test_r2": average_test_r2,
        "r2_gap": entry_results["test_r2"] - exit_results["test_r2"],
    }

    write_summary(summary, summary_path, timestamp, experiment_id, best_record)


if __name__ == "__main__":
    main()
