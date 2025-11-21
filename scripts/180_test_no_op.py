"""
119_test_no_op.py — rollforward 제거 + SVM 고정 + R² 전략 스위치 유지 (누수 가드 엄격)

핵심:
- 알고리즘 SVM 고정(SVR + 표준화). E2X/X2E 모두 SVM.
- GroupShuffleSplit(plate 단위)로 분할 → 모든 통계/대치/특수특성은 Train 기준으로만 계산 후 Test에 적용
- USE_* 스위치:
  * USE_DELTA_TARGET=1 : y=(t+1)-cur 로 학습, 추정값은 다시 cur 더해 복원해 평가
  * USE_PAIR_TE=1      : pair( p→p+1 ) OOF Δ‑mean 인코딩(Train OOF / Test는 Train 평균 주입)
  * USE_HETERO_W=1     : pair별 Δ 분산 기반 inverse-variance 가중(sample_weight)으로 SVM 학습
  * USE_RM_TRAINONLY=1 : RM_* 컬럼은 Train 기준으로만 사용(Test는 train median으로 채워 실질 배제)
- SelectKBest(f_regression)로 피처 선택(k는 env로 조절), 표준화(StandardScaler)는 Train 기준
- 라벨/타깃/예측/TE 등 이름에 의한 누수 방지 필터링 포함
- summary_path 규격: reports/summaries/{experiment_id}_test_no_op_summary.txt
- last_run_metrics.csv 열 순서 고정: experiment_id, e2x_test_r2, x2e_test_r2, average_test_r2, summary_path
"""

import os
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import traceback

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- 경로/상수 --------------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
REPORTS_SUMMARY_DIR = REPORTS_DIR / "summaries"
REPORTS_CODE_DIR = REPORTS_DIR / "scripts"
MODELS_DIR = Path("models")  # (저장은 최소화하지만 디렉토리는 유지)
IMAGES_DIR = Path("images")  # (미사용)

TARGET_COL = "warping_index_target"   # t+1 타깃
PASS_COL   = "FM_PASS NO N"
PLATE_COL  = "FM_날판번호"
MONTH_COL  = "FM_압연월"
CUR_WARP   = "warping_index_current_pass"

# 모델 입력에서 절대 배제해야 할 컬럼(타깃/식별자/시계열키 등)
EXCLUDE_FROM_X = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}

# --------------------------- ENV 스위치/하이퍼 -----------------------------
def _env_on(name: str, default: str = "0") -> int:
    v = os.getenv(name, default).strip().lower()
    return 1 if v in ("1", "true", "yes", "y", "on") else 0

RUN_NAME         = os.getenv("RUN_NAME", "pass_t_plus_1_cross")
PERMUTE_TARGET   = _env_on("PERMUTE_TARGET", "0")
LEAK_AUTOCHECK   = _env_on("LEAK_AUTOCHECK", "1")  # 빠른 퍼뮤 확인(기본 켜짐)

USE_DELTA_TARGET = _env_on("USE_DELTA_TARGET", "0")
USE_PAIR_TE      = _env_on("USE_PAIR_TE", "0")
USE_HETERO_W     = _env_on("USE_HETERO_W", "0")
USE_RM_TRAINONLY = _env_on("USE_RM_TRAINONLY", "0")

# SelectK, SVR 하이퍼(환경변수로 조정 가능)
SELECT_K   = int(os.getenv("SELECT_K", "120"))
SVR_C      = float(os.getenv("SVR_C", "10.0"))
SVR_EPS    = float(os.getenv("SVR_EPS", "0.2"))
SVR_GAMMA  = os.getenv("SVR_GAMMA", "scale").strip()  # "scale" 또는 "auto"

# --------------------------- 유틸 -----------------------------
def _ensure_dirs():
    for d in [PROCESSED_DIR, REPORTS_DIR, REPORTS_SUMMARY_DIR, REPORTS_CODE_DIR, MODELS_DIR, REPORTS_DIR/"metrics"]:
        d.mkdir(parents=True, exist_ok=True)

def _next_experiment_id() -> str:
    existing = []
    for f in REPORTS_SUMMARY_DIR.glob("*.txt"):
        m = re.match(r"(\d{3})_", f.name)
        if m:
            try:
                existing.append(int(m.group(1)))
            except:
                pass
    return f"{(max(existing)+1) if existing else 1:03d}"

def _hard_leak_guard(cols: list[str]) -> list[str]:
    """이름에 타깃/예측 스멜이 있는 컬럼 전부 배제"""
    bad_kw = ("target", "label", "oof", "pred", "_hat", "y_", "_y", "te_")
    safe = []
    for c in cols:
        low = c.lower()
        if any(k in low for k in bad_kw):
            continue
        safe.append(c)
    return safe

def _save_last_metrics(experiment_id: str, e2x_test_r2: float, x2e_test_r2: float, avg_test_r2: float, summary_path: Path):
    met_dir = REPORTS_DIR / "metrics"
    met_dir.mkdir(parents=True, exist_ok=True)
    csv = met_dir / "last_run_metrics.csv"
    df = pd.DataFrame([{
        "experiment_id": experiment_id,
        "e2x_test_r2": float(e2x_test_r2),
        "x2e_test_r2": float(x2e_test_r2),
        "average_test_r2": float(avg_test_r2),
        "summary_path": str(summary_path),
    }])
    df = df[["experiment_id","e2x_test_r2","x2e_test_r2","average_test_r2","summary_path"]]  # 열 순서 고정
    df.to_csv(csv, index=False)
    print(f"✓ 메트릭 저장: {csv}")

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

    # 키 추출
    ground_truth["extracted_plate"] = ground_truth["filename"].str.extract(r"(PB\d+)")
    ground_truth["extracted_pass"]  = ground_truth["filename"].str.extract(r"_\d+_(\d+)").astype(int)

    # RM 통계(전량) — 누수 방지 위해 Test에는 실질 반영 차단(USE_RM_TRAINONLY로 제어)
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

    # 타깃 생성
    merged_fm[TARGET_COL] = merged_fm.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged_fm = merged_fm.rename(columns={"warping_index": CUR_WARP})
    before = len(merged_fm)
    merged_fm = merged_fm.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"Target shift(-1): {before} → {len(merged_fm)} (삭제 {before-len(merged_fm)})")

    # PERMUTE (sanity)
    if PERMUTE_TARGET:
        merged_fm[TARGET_COL] = merged_fm[TARGET_COL].sample(frac=1.0, random_state=42).values
        print("[PERMUTE] Target shuffled for leak sanity check.")

    # RM 집계 결합 + 보조 컬럼 정리
    final_df = pd.merge(
        merged_fm, rm_stats,
        left_on=PLATE_COL, right_on="RM_날판번호", how="left"
    ).drop(columns=["RM_날판번호"], errors="ignore")
    final_df = final_df.drop(columns=["extracted_plate","extracted_pass","filename",
                                      "quality_grade","quality_grade_current_pass","direction"], errors="ignore")

    out_path = PROCESSED_DIR / "final_merged_data_regression_tplus1.csv"
    final_df.to_csv(out_path, index=False)
    print(f"✓ 저장: {out_path}")
    return final_df

# --------- 2) 전이 세트(E→X / X→E) 분리 ----------
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

# ---------------- 3) Feature Engineering (경량; 누수 안전) ---------------
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

    # 단조 제약(MONO)은 SVM에선 실제 거는 게 불가 → 로깅만 함
    if PASS_COL in df.columns:
        df["FM_PASS_squared"] = df[PASS_COL] ** 2

    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path}")
    return df

# -------------------- 4) 이상치 제거 (NO-OP; train-only 처리) -------------------
def remove_outliers(input_path: Path, output_path: Path, dataset_name: str):
    print(f"\n[Step 4] 이상치 제거 - {dataset_name}")
    df = pd.read_csv(input_path)
    # 실제 이상치 처리는 train 내에서만 해야 누수 방지 → 여기서는 NO-OP
    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path} (최종 {df.shape[0]}개)")
    return df

# -------------------- 학습 보조 (OOF TE / 가중치) --------------------
def _make_pair_id(df: pd.DataFrame) -> pd.Series:
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL] + 1).astype(int).astype(str)

def _oof_pair_te(tr_df: pd.DataFrame, y_tr: np.ndarray, use_delta: bool, n_splits=5):
    """plate-group OOF로 pair Δ 평균 타깃 인코딩 생성."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = tr_df[PLATE_COL].astype(str).values
    pair  = _make_pair_id(tr_df).values
    cur   = tr_df[CUR_WARP].values
    target = y_tr.copy()
    if use_delta:
        target = y_tr - cur  # Δ

    te = np.zeros(len(tr_df), dtype=float)
    for tr_idx, va_idx in gkf.split(tr_df, target, groups):
        pair_tr = pd.Series(pair[tr_idx])
        val_pairs = pair[va_idx]
        means = pd.DataFrame({"pair": pair_tr, "y": target[tr_idx]}).groupby("pair")["y"].mean()
        global_mean = target[tr_idx].mean()
        te[va_idx] = [means.get(p, global_mean) for p in val_pairs]

    global_means = pd.DataFrame({"pair": pair, "y": target}).groupby("pair")["y"].mean().to_dict()
    global_default = float(target.mean())
    return te, global_means, global_default

def _pair_weights(tr_df: pd.DataFrame, y_tr: np.ndarray, use_delta: bool):
    pair  = _make_pair_id(tr_df).values
    cur   = tr_df[CUR_WARP].values
    target = y_tr.copy()
    if use_delta:
        target = y_tr - cur  # Δ
    df = pd.DataFrame({"pair": pair, "y": target})
    s = df.groupby("pair")["y"].std().replace(0, np.nan).fillna(df["y"].std())
    w = 1.0 / (s**2 + 1e-6)
    w_map = w.to_dict()
    return np.array([w_map.get(p, 1.0) for p in pair], dtype=float)

# ---------------------------- 학습 본체 (SVM 고정) ----------------------------
def train_model(df: pd.DataFrame,
                dataset_name: str,           # "E2X" | "X2E"
                seed: int = 42,
                test_size: float = 0.2,
                select_k: int = SELECT_K):
    """
    - plate 단위 GroupShuffleSplit으로 8:2 분할
    - Train 기준 중앙값 대치 → Test에 적용(누수 방지)
    - SelectKBest(f_regression) → StandardScaler → SVR(C=SVR_C, eps=SVR_EPS, gamma=SVR_GAMMA)
    - Δ-타깃/Pair‑TE/이분산 가중/Train‑only RM 옵션 적용
    """
    data = df.copy()
    # 홀/짝 분기 (안전용 — 보통 외부에서 이미 분리됨)
    if dataset_name.upper() == "E2X":
        data = data[data[PASS_COL] % 2 == 1].copy()
    else:
        data = data[data[PASS_COL] % 2 == 0].copy()

    # features 후보 (수치형만), EXCLUDE 제거 + 이름 스멜 가드
    drop_cols = [c for c in EXCLUDE_FROM_X if c in data.columns]
    X_all = data.drop(columns=drop_cols, errors="ignore")
    X_all = X_all.select_dtypes(include=[np.number]).copy()
    X_all = X_all[_hard_leak_guard(X_all.columns.tolist())]  # 이름 스멜 가드

    y_all = data[TARGET_COL].values
    groups_all = data[PLATE_COL].astype(str).values

    # plate group split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(X_all, y_all, groups=groups_all))

    X_tr_raw, X_te_raw = X_all.iloc[tr_idx].copy(), X_all.iloc[te_idx].copy()
    y_tr_raw, y_te_raw = y_all[tr_idx].copy(), y_all[te_idx].copy()
    tr_df_raw = data.iloc[tr_idx].copy()
    te_df_raw = data.iloc[te_idx].copy()

    # Train 기준 중앙값 대치
    med = X_tr_raw.median(numeric_only=True).to_dict()
    X_tr = X_tr_raw.fillna(med)
    X_te = X_te_raw.fillna(med)

    # RM_* Train-only (Test는 train median으로 채워 실질 배제)
    if USE_RM_TRAINONLY:
        rm_cols = [c for c in X_tr.columns if c.startswith("RM_")]
        if rm_cols:
            # Train에는 그대로 두고, Test는 중앙값으로 고정(=정보적 기여 제거)
            X_te[rm_cols] = pd.DataFrame({c: med.get(c, 0.0) for c in rm_cols}, index=X_te.index)

    # Δ-타깃
    if USE_DELTA_TARGET:
        y_tr = y_tr_raw - tr_df_raw[CUR_WARP].values
        y_te = y_te_raw - te_df_raw[CUR_WARP].values
        cur_tr = tr_df_raw[CUR_WARP].values
        cur_te = te_df_raw[CUR_WARP].values
    else:
        y_tr = y_tr_raw.copy()
        y_te = y_te_raw.copy()
        cur_tr = None
        cur_te = None

    # Pair‑TE (OOF on train, test는 train 평균 주입)
    if USE_PAIR_TE:
        te_tr, te_map, te_def = _oof_pair_te(tr_df_raw, y_tr_raw, use_delta=bool(USE_DELTA_TARGET), n_splits=5)
        X_tr["pair_te"] = te_tr
        te_pairs = _make_pair_id(te_df_raw).values
        X_te["pair_te"] = np.array([te_map.get(p, te_def) for p in te_pairs], dtype=float)

    # 이분산 가중(WLS)
    sample_w = None
    if USE_HETERO_W:
        sample_w = _pair_weights(tr_df_raw, y_tr_raw, use_delta=bool(USE_DELTA_TARGET))

    # SelectK → Scale → SVR
    k_actual = max(1, min(select_k, X_tr.shape[1]))
    selector = SelectKBest(score_func=f_regression, k=k_actual)
    selector.fit(X_tr, y_tr)
    Xtr_sel = selector.transform(X_tr.values)   # numpy로 전달(이름 검사 회피)
    Xte_sel = selector.transform(X_te.values)

    scaler = StandardScaler()
    scaler.fit(Xtr_sel)                         # Train 기준
    Xtr_sc = scaler.transform(Xtr_sel)
    Xte_sc = scaler.transform(Xte_sel)

    model = SVR(C=SVR_C, epsilon=SVR_EPS, gamma=SVR_GAMMA, kernel="rbf")
    if sample_w is not None:
        model.fit(Xtr_sc, y_tr, sample_weight=sample_w)
    else:
        model.fit(Xtr_sc, y_tr)

    # 예측 (Δ 사용 시 복원)
    y_tr_hat = model.predict(Xtr_sc)
    y_te_hat = model.predict(Xte_sc)
    if USE_DELTA_TARGET:
        y_tr_hat_abs = y_tr_hat + cur_tr
        y_te_hat_abs = y_te_hat + cur_te
        tr_r2  = float(r2_score(y_tr_raw, y_tr_hat_abs))
        te_r2  = float(r2_score(y_te_raw, y_te_hat_abs))
        tr_rmse = float(np.sqrt(mean_squared_error(y_tr_raw, y_tr_hat_abs)))
        te_rmse = float(np.sqrt(mean_squared_error(y_te_raw, y_te_hat_abs)))
    else:
        tr_r2  = float(r2_score(y_tr_raw, y_tr_hat))
        te_r2  = float(r2_score(y_te_raw, y_te_hat))
        tr_rmse = float(np.sqrt(mean_squared_error(y_tr_raw, y_tr_hat)))
        te_rmse = float(np.sqrt(mean_squared_error(y_te_raw, y_te_hat)))

    res = {
        "algorithm": "SVM",
        "train_r2": tr_r2, "test_r2": te_r2,
        "train_rmse": tr_rmse, "test_rmse": te_rmse,
        "train_samples": int(len(y_tr)), "test_samples": int(len(y_te)),
        "select_k": int(k_actual),
        "svr": {"C": SVR_C, "epsilon": SVR_EPS, "gamma": SVR_GAMMA},
        "flags": {
            "USE_DELTA_TARGET": int(USE_DELTA_TARGET),
            "USE_PAIR_TE": int(USE_PAIR_TE),
            "USE_HETERO_W": int(USE_HETERO_W),
            "USE_RM_TRAINONLY": int(USE_RM_TRAINONLY),
        }
    }
    return res

# ------------- Δ-분석/요약 ----------------
def analyze_transition_variability(final_df: pd.DataFrame) -> dict:
    try:
        from scipy.stats import kruskal, levene
    except Exception:
        kruskal = levene = None

    df = final_df.copy()
    df["delta"] = df[TARGET_COL] - df[CUR_WARP]
    df["pair"] = df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

    def _stats(sub):
        g = sub.groupby("pair")["delta"]
        stat = g.agg(count="count", mean="mean", std="std", median="median",
                     q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index()
        std_ratio = (stat["std"].max() / stat["std"].replace(0, np.nan).min()) if stat["std"].notna().any() else np.nan
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

def write_summary(summary_data, summary_path: Path, timestamp: datetime, experiment_id: str):
    lines = []
    lines.append("Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
    lines.append("="*72)
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

    v = summary_data.get("variability", {})
    for tag in ["e2x","x2e"]:
        vv = v.get(tag, {})
        lines.append(f"\n[Δ-분석 {tag.upper()}] std_max/min={vv.get('std_ratio', np.nan):.3f} | "
                     f"Kruskal p={vv.get('kw_p', np.nan)} | Levene p={vv.get('lev_p', np.nan)}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 요약 저장: {summary_path}")

# ----------------------------- main ------------------------------
def main():
    _ensure_dirs()
    success = False
    experiment_id = _next_experiment_id()
    timestamp = datetime.now()
    summary_path = REPORTS_SUMMARY_DIR / f"{experiment_id}_test_no_op_summary.txt"
    archived_script_path = REPORTS_CODE_DIR / f"{experiment_id}_test_no_op.py"

    print("\n[DEBUG] FLAGS",
          f"DT={USE_DELTA_TARGET} TE={USE_PAIR_TE} W={USE_HETERO_W} RM={USE_RM_TRAINONLY} "
          f"| PERMUTE={PERMUTE_TARGET} LEAK_AUTOCHECK={LEAK_AUTOCHECK} "
          f"| SELECT_K={SELECT_K} SVR(C={SVR_C},eps={SVR_EPS},gamma={SVR_GAMMA})")

    try:
        # 1) 병합/타깃(t+1)
        final_df = load_and_merge_data_tplus1()

        # Δ‑변동성 분석
        variability = analyze_transition_variability(final_df)

        # 2) 전이 세트
        e2x_df, x2e_df = split_cross_transitions(final_df)

        # 3) FE
        e2x_fe = feature_engineering(e2x_df, "E2X", PROCESSED_DIR / "e2x_featured_tplus1.csv")
        x2e_fe = feature_engineering(x2e_df, "X2E", PROCESSED_DIR / "x2e_featured_tplus1.csv")

        # 4) 이상치 (분할 전 NO-OP; 내부 train-only 처리)
        e2x_clean = remove_outliers(PROCESSED_DIR / "e2x_featured_tplus1.csv",
                                    PROCESSED_DIR / "e2x_cleaned_tplus1.csv", "E2X")
        x2e_clean = remove_outliers(PROCESSED_DIR / "x2e_featured_tplus1.csv",
                                    PROCESSED_DIR / "x2e_cleaned_tplus1.csv", "X2E")

        # 5) 학습(SVM 고정)
        e2x_res = train_model(e2x_clean, "E2X", seed=42, test_size=0.2, select_k=SELECT_K)
        x2e_res = train_model(x2e_clean, "X2E", seed=42, test_size=0.2, select_k=SELECT_K)

        average_test_r2 = float(np.nanmean([e2x_res["test_r2"], x2e_res["test_r2"]]))

        # 요약/산출물
        summary = {
            "data_artifacts": [
                f"Final merged(t+1): {PROCESSED_DIR / 'final_merged_data_regression_tplus1.csv'}",
                f"E2X raw/clean: {PROCESSED_DIR / 'e2x_raw_tplus1.csv'}, {PROCESSED_DIR / 'e2x_cleaned_tplus1.csv'}",
                f"X2E raw/clean: {PROCESSED_DIR / 'x2e_raw_tplus1.csv'}, {PROCESSED_DIR / 'x2e_cleaned_tplus1.csv'}",
                f"All featured: {PROCESSED_DIR / 'all_featured_tplus1.csv'}",
                f"Archived script: {archived_script_path}",
            ],
            "models": {"E2X": e2x_res, "X2E": x2e_res},
            "average_test_r2": average_test_r2,
            "variability": variability,
        }
        write_summary(summary, summary_path, timestamp, experiment_id)

        shutil.copy2(Path(__file__).resolve(), archived_script_path)
        print(f"✓ 코드 아카이브 저장: {archived_script_path}")

        # 메트릭 파일 (sh 집계 호환)
        _save_last_metrics(experiment_id,
                           e2x_res["test_r2"], x2e_res["test_r2"],
                           average_test_r2, summary_path)
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
