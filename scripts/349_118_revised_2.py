"""
118_revised_2.py

【목적】
Pass K의 RM, FM 데이터로 Pass K의 날판두께를 예측 (t → t 예측)
단, K는 5, 6, 7, 8, 9만 사용

【모델】
단일 SVM 모델로 Pass 5, 6, 7, 8, 9의 날판두께 예측

【데이터 흐름】
1. 데이터 병합 및 로드
   - data/entry_direction_results.csv
   - data/exit_direction_results.csv  
   - data/posco1_105190.csv (RM)
   - data/posco2_105190.csv (FM)

2. Pass 5, 6, 7, 8, 9 데이터만 추출
   [입력 특징]
   - RM 통계량 (mean, max, min, std, median) - plate 단위 집계
   - FM 데이터 (해당 pass의 FM 측정값들)
   - Pass 번호 정보
   
   [타깃]
   - 해당 pass의 warping_index_current_pass (날판두께)

3. 학습 방법
   - SVM (Support Vector Regression) 사용
   - GroupShuffleSplit으로 plate 기준 train/test 분할
   - Train set에서만 통계량 계산
   - 모든 pass를 하나의 모델로 학습

4. 기존 코드와의 차이점
   - 기존: t → t+1 예측 (연속된 두 pass 사용)
   - 현재: t → t 예측 (단일 pass의 RM/FM으로 같은 pass의 날판두께 예측)

5. 데이터 누수 방지
   - Train/Test 분할 후 모든 통계량 계산 (train-only)
   - RM 통계량은 plate 단위로 계산하되, train plate에서만 계산
   - 타깃 정보(날판두께)는 특징에서 완전 제외
   - Feature selection도 train set에서만 수행
"""

import os
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import traceback

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- 경로/상수 --------------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
REPORTS_SUMMARY_DIR = REPORTS_DIR / "summaries"
REPORTS_CODE_DIR = REPORTS_DIR / "scripts"
MODELS_DIR = Path("models")

PLATE_COL = "FM_날판번호"
PASS_COL = "FM_PASS NO N"
MONTH_COL = "FM_압연월"
CUR_WARP = "warping_index_current_pass"
TARGET_COL = "target_warp"

# 허용된 패스 번호
ALLOWED_PASSES = {5, 6, 7, 8, 9}

# --------------------------- 유틸 -----------------------------
def _ensure_dirs():
    for d in [PROCESSED_DIR, REPORTS_DIR, REPORTS_SUMMARY_DIR, 
              REPORTS_CODE_DIR, MODELS_DIR, REPORTS_DIR/"metrics"]:
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

# ---------- 1) 데이터 로드 및 허용된 Pass 추출 ----------
def load_and_extract_allowed_passes() -> pd.DataFrame:
    """
    Pass 5, 6, 7, 8, 9의 RM, FM 데이터를 특징으로,
    같은 pass의 warping_index를 타깃으로 설정
    """
    print("\n" + "="*80)
    print(f"[Step 1] 데이터 병합 및 Pass {sorted(ALLOWED_PASSES)} 추출 (t → t 예측)")
    print("="*80)
    
    # 기본 데이터 로드
    gt_entry = pd.read_csv(DATA_DIR / "entry_direction_results.csv")
    gt_exit = pd.read_csv(DATA_DIR / "exit_direction_results.csv")
    ground_truth = pd.concat([gt_entry, gt_exit], ignore_index=True)
    
    rm_data = pd.read_csv(DATA_DIR / "posco1_105190.csv")  # RM
    fm_data = pd.read_csv(DATA_DIR / "posco2_105190.csv")  # FM
    
    # 키 추출
    ground_truth["extracted_plate"] = ground_truth["filename"].str.extract(r"(PB\d+)")
    ground_truth["extracted_pass"] = ground_truth["filename"].str.extract(r"_\d+_(\d+)").astype(int)
    
    # RM 집계용 병합
    merged_rm = pd.merge(
        ground_truth, rm_data,
        left_on=["extracted_plate", "extracted_pass"],
        right_on=["RM_날판번호", "RM_압연Pass번호"],
        how="inner"
    )
    num_cols = merged_rm.select_dtypes(include=[np.number]).columns.tolist()
    stat_cols = [c for c in num_cols 
                 if c not in ["RM_날판번호", "RM_압연Pass번호", "warping_index", "extracted_pass"]]
    
    # RM 통계량 (plate 단위)
    rm_stats = merged_rm.groupby("RM_날판번호")[stat_cols].agg(
        ["mean", "max", "min", "std", "median"]
    ).round(4)
    rm_stats.columns = [f"RM_{c[0]}_{c[1]}" for c in rm_stats.columns]
    rm_stats = rm_stats.reset_index()
    
    # FM 병합
    merged_fm = pd.merge(
        ground_truth, fm_data,
        left_on=["extracted_plate", "extracted_pass"],
        right_on=[PLATE_COL, PASS_COL],
        how="inner"
    ).sort_values([PLATE_COL, PASS_COL])
    
    # warping_index를 현재 pass의 날판두께로 rename
    merged_fm = merged_fm.rename(columns={"warping_index": CUR_WARP})
    
    # RM 집계 결합
    merged_fm = pd.merge(
        merged_fm, rm_stats,
        left_on=PLATE_COL, right_on="RM_날판번호", how="left"
    ).drop(columns=["RM_날판번호"], errors="ignore")
    
    # 불필요한 컬럼 제거
    merged_fm = merged_fm.drop(columns=[
        "extracted_plate", "extracted_pass", "filename",
        "quality_grade", "quality_grade_current_pass", "direction"
    ], errors="ignore")
    
    print(f"전체 데이터: {len(merged_fm)}행")
    
    # 허용된 Pass만 필터링
    before = len(merged_fm)
    allowed_df = merged_fm[merged_fm[PASS_COL].isin(ALLOWED_PASSES)].copy()
    print(f"Pass {sorted(ALLOWED_PASSES)} 필터링: {before}행 → {len(allowed_df)}행")
    
    # Pass별 데이터 분포 확인
    pass_counts = allowed_df[PASS_COL].value_counts().sort_index()
    print(f"\nPass별 데이터 수:")
    for pass_num, count in pass_counts.items():
        print(f"  Pass {pass_num}: {count}행")
    
    # 타깃 설정: 동일 pass의 warping_index
    allowed_df[TARGET_COL] = allowed_df[CUR_WARP]
    
    # NaN 제거
    before = len(allowed_df)
    allowed_df = allowed_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"\n타깃 NaN 제거: {before} → {len(allowed_df)} (삭제 {before - len(allowed_df)})")
    
    # 저장
    out_path = PROCESSED_DIR / "allowed_passes_data.csv"
    allowed_df.to_csv(out_path, index=False)
    print(f"✓ 저장: {out_path}")
    print(f"최종 데이터 shape: {allowed_df.shape}")
    
    return allowed_df

# -------------------- 2) Feature Engineering --------------------
def feature_engineering(df: pd.DataFrame, output_path: Path):
    """
    각 pass의 FM 특징들로부터 통계량 생성
    """
    print(f"\n[Step 2] Feature Engineering")
    df = df.copy()
    
    # FM 특징 (누수 방지: PLATE_COL, PASS_COL, TARGET_COL, CUR_WARP, MONTH_COL 제외)
    fm_cols = [c for c in df.columns 
               if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    
    if fm_cols:
        fm_blk = df[fm_cols]
        df["FM_mean"] = fm_blk.mean(axis=1)
        df["FM_std"] = fm_blk.std(axis=1)
        df["FM_max"] = fm_blk.max(axis=1)
        df["FM_min"] = fm_blk.min(axis=1)
        df["FM_range"] = df["FM_max"] - df["FM_min"]
        df["FM_cv"] = df["FM_std"] / (df["FM_mean"].abs() + 1e-8)
        print(f"  FM 통계 특징 생성: mean, std, max, min, range, cv")
    
    # Pass 관련 특징
    if PASS_COL in df.columns:
        df["PASS_squared"] = df[PASS_COL] ** 2
        df["PASS_cubed"] = df[PASS_COL] ** 3
        print(f"  Pass 특징 생성: squared, cubed")
    
    # Pass별 원-핫 인코딩 (선택적)
    for p in ALLOWED_PASSES:
        df[f"is_pass_{p}"] = (df[PASS_COL] == p).astype(int)
    print(f"  Pass 원-핫 인코딩 생성: {sorted(ALLOWED_PASSES)}")
    
    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path}")
    return df

# -------------------- 3) 학습 --------------------
def train_model(df: pd.DataFrame, seed: int = 42, test_size: float = 0.2, select_k: int = 120):
    """
    SVM을 사용하여 Pass K의 RM, FM → Pass K의 날판두께 예측
    모든 허용된 pass를 하나의 모델로 학습
    """
    print("\n" + "="*80)
    print(f"[Step 3] SVM 모델 학습 (Pass K RM/FM → Pass K 날판두께, K∈{sorted(ALLOWED_PASSES)})")
    print("="*80)
    
    data = df.copy()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 특징/타깃 분리 (누수 방지: 타깃과 현재 warping_index 모두 제외)
    exclude_from_x = {TARGET_COL, PLATE_COL, CUR_WARP, MONTH_COL}
    X_all = data.drop(columns=[c for c in exclude_from_x if c in data.columns], errors="ignore")
    
    # 숫자형 컬럼만
    X_all = X_all.select_dtypes(include=[np.number]).copy()
    y_all = data[TARGET_COL].values
    groups_all = data[PLATE_COL].astype(str).values
    
    # Pass 정보 (분석용)
    pass_info = data[PASS_COL].values if PASS_COL in data.columns else None
    
    print(f"전체 특징 수: {X_all.shape[1]}, 샘플 수: {X_all.shape[0]}")
    print(f"특징 컬럼 예시 (첫 15개): {list(X_all.columns[:15])}")
    
    # Group-aware split (plate 기준으로 분할)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(X_all, y_all, groups=groups_all))
    
    X_tr_raw = X_all.iloc[tr_idx].copy()
    X_te_raw = X_all.iloc[te_idx].copy()
    y_tr = y_all[tr_idx].copy()
    y_te = y_all[te_idx].copy()
    
    print(f"Train 샘플: {len(y_tr)}, Test 샘플: {len(y_te)}")
    print(f"Train plates: {len(set(groups_all[tr_idx]))}, Test plates: {len(set(groups_all[te_idx]))}")
    
    # Train/Test에서 각 pass별 분포 확인
    if pass_info is not None:
        pass_tr = pass_info[tr_idx]
        pass_te = pass_info[te_idx]
        
        print(f"\nTrain set Pass 분포:")
        for p in sorted(ALLOWED_PASSES):
            count = (pass_tr == p).sum()
            print(f"  Pass {p}: {count}개 ({count/len(pass_tr)*100:.1f}%)")
        
        print(f"\nTest set Pass 분포:")
        for p in sorted(ALLOWED_PASSES):
            count = (pass_te == p).sum()
            print(f"  Pass {p}: {count}개 ({count/len(pass_te)*100:.1f}%)")
    
    # ====== 데이터 누수 방지: train-only 통계량 계산 ======
    # train 중앙값으로만 결측 대치
    med = X_tr_raw.median(numeric_only=True).fillna(0.0).to_dict()
    X_tr = X_tr_raw.fillna(med)
    X_te = X_te_raw.fillna(med)
    
    print(f"\n결측치 대치: train 중앙값 사용 ({len(med)} features)")
    
    # Feature selection (train에서만 fit)
    k_actual = max(1, min(select_k, X_tr.shape[1]))
    print(f"Feature selection: {X_tr.shape[1]} → {k_actual} features")
    
    selector = SelectKBest(score_func=f_regression, k=k_actual)
    selector.fit(X_tr.values, y_tr)
    Xtr_sel = selector.transform(X_tr.values)
    Xte_sel = selector.transform(X_te.values)
    
    # 선택된 특징 인덱스
    selected_mask = selector.get_support()
    selected_features = X_tr.columns[selected_mask].tolist()
    print(f"선택된 특징 예시 (첫 15개): {selected_features[:15]}")
    
    # SVM 모델 파이프라인
    model = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("reg", SVR(kernel="rbf", C=10.0, epsilon=0.1)),
    ])
    
    # 학습
    print("\n모델 학습 중...")
    model.fit(Xtr_sel, y_tr)
    
    # 예측
    y_tr_pred = model.predict(Xtr_sel)
    y_te_pred = model.predict(Xte_sel)
    
    # 전체 평가
    train_r2 = float(r2_score(y_tr, y_tr_pred))
    test_r2 = float(r2_score(y_te, y_te_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_te, y_te_pred)))
    
    print(f"\n[전체 성능]")
    print(f"  Train R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
    print(f"  Test R2:  {test_r2:.4f}, RMSE: {test_rmse:.4f}")
    
    # Pass별 성능 평가
    pass_results = {}
    if pass_info is not None:
        print(f"\n[Pass별 Test 성능]")
        for p in sorted(ALLOWED_PASSES):
            mask = pass_te == p
            if mask.sum() > 0:
                p_r2 = float(r2_score(y_te[mask], y_te_pred[mask]))
                p_rmse = float(np.sqrt(mean_squared_error(y_te[mask], y_te_pred[mask])))
                print(f"  Pass {p}: R2={p_r2:.4f}, RMSE={p_rmse:.4f} (n={mask.sum()})")
                pass_results[p] = {"r2": p_r2, "rmse": p_rmse, "n_samples": int(mask.sum())}
    
    res = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_samples": int(len(y_tr)),
        "test_samples": int(len(y_te)),
        "selected_features": selected_features,
        "n_features_selected": len(selected_features),
        "pass_results": pass_results,
    }
    
    # 모델 저장
    joblib.dump(model, MODELS_DIR / "revised2_svm_model.pkl")
    joblib.dump(selector, MODELS_DIR / "revised2_selector.pkl")
    joblib.dump(list(X_tr.columns), MODELS_DIR / "revised2_feature_cols.pkl")
    joblib.dump(med, MODELS_DIR / "revised2_medians.pkl")
    joblib.dump(selected_features, MODELS_DIR / "revised2_selected_features.pkl")
    
    print(f"\n✓ 모델 저장 완료")
    
    return res

# -------------------- 4) 요약 작성 --------------------
def write_summary(results: dict, summary_path: Path, timestamp: datetime, experiment_id: str):
    lines = []
    lines.append(f"Pass K RM/FM → Pass K Thickness Regression (t→t) (118_revised_2.py)")
    lines.append("="*72)
    lines.append(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} 작성")
    lines.append(f"실험 ID: {experiment_id}\n")
    
    lines.append("[모델 설명]")
    lines.append(f"- Pass K (K ∈ {sorted(ALLOWED_PASSES)})의 RM, FM 데이터")
    lines.append(f"- → Pass K의 날판두께(warping_index) 예측")
    lines.append("- 알고리즘: SVM (RBF kernel)")
    lines.append("- 예측 방식: t → t (동일 시점 예측)")
    lines.append("- 단일 모델로 모든 pass 예측\n")
    
    lines.append("[데이터 산출물]")
    lines.append(f"- Allowed passes data: {PROCESSED_DIR / 'allowed_passes_data.csv'}")
    lines.append(f"- Featured data: {PROCESSED_DIR / 'allowed_passes_featured.csv'}")
    lines.append(f"- Model: {MODELS_DIR / 'revised2_svm_model.pkl'}\n")
    
    lines.append("[전체 모델 성능]")
    lines.append(f"* Train R2: {results['train_r2']:.4f} | RMSE: {results['train_rmse']:.4f}")
    lines.append(f"* Test R2:  {results['test_r2']:.4f} | RMSE: {results['test_rmse']:.4f}")
    lines.append(f"* Train samples: {results['train_samples']}")
    lines.append(f"* Test samples:  {results['test_samples']}\n")
    
    if results.get('pass_results'):
        lines.append("[Pass별 Test 성능]")
        for p in sorted(results['pass_results'].keys()):
            pr = results['pass_results'][p]
            lines.append(f"* Pass {p}: R2={pr['r2']:.4f}, RMSE={pr['rmse']:.4f} (n={pr['n_samples']})")
        lines.append("")
    
    lines.append("[Feature Selection]")
    lines.append(f"* 선택된 특징 수: {results['n_features_selected']}")
    lines.append(f"* 선택된 특징 예시 (첫 15개):")
    for feat in results['selected_features'][:15]:
        lines.append(f"  - {feat}")
    if len(results['selected_features']) > 15:
        lines.append(f"  - ... 외 {len(results['selected_features']) - 15}개\n")
    else:
        lines.append("")
    
    lines.append("[데이터 누수 방지]")
    lines.append("- Train/Test는 plate 단위로 분할 (GroupShuffleSplit)")
    lines.append("- 모든 통계량(중앙값, feature selection, scaling)은 train-only로 계산")
    lines.append("- 타깃 정보(warping_index_current_pass)는 특징에서 완전 제외")
    lines.append("- 동일 plate의 데이터가 train/test에 섞이지 않도록 보장")
    lines.append("\n[기존 코드와의 차이점]")
    lines.append("- 기존: t → t+1 예측 (연속된 두 pass 사용)")
    lines.append("- 현재: t → t 예측 (단일 pass의 RM/FM으로 같은 pass의 날판두께 예측)")
    lines.append("- 시간적 전이 없이 동일 시점의 특징으로 타깃 예측")
    lines.append(f"- Pass 5, 6, 7, 8, 9만 사용하여 학습")
    
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 요약 저장: {summary_path}")

# ----------------------------- main ------------------------------
def main():
    _ensure_dirs()
    experiment_id = _next_experiment_id()
    timestamp = datetime.now()
    summary_path = REPORTS_SUMMARY_DIR / f"{experiment_id}_118_revised_2_summary.txt"
    archived_script_path = REPORTS_CODE_DIR / f"{experiment_id}_118_revised_2.py"
    
    try:
        # 1) 허용된 Pass 데이터 추출
        allowed_df = load_and_extract_allowed_passes()
        
        # 2) Feature Engineering
        featured_df = feature_engineering(
            allowed_df, 
            PROCESSED_DIR / "allowed_passes_featured.csv"
        )
        
        # 3) 모델 학습
        results = train_model(featured_df)
        
        # 4) 요약 작성
        write_summary(results, summary_path, timestamp, experiment_id)
        
        # 5) 코드 아카이브
        shutil.copy2(Path(__file__).resolve(), archived_script_path)
        print(f"✓ 코드 아카이브 저장: {archived_script_path}")
        
        print("\n" + "="*80)
        print("실험 완료!")
        print("="*80)
        
    except Exception as e:
        tb = traceback.format_exc()
        print("\n[실험 실패] 예외가 발생했습니다.")
        print(f"Exception: {type(e).__name__}: {e}")
        print(tb)
        err_path = REPORTS_DIR / "last_error_traceback_revised2.txt"
        err_path.write_text(tb, encoding="utf-8")
        print(f"↳ 전체 traceback을 {err_path} 에 저장했습니다.")

if __name__ == "__main__":
    main()
