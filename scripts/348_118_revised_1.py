"""
118_revised_1.py

【목적】
Pass t와 Pass t+1의 정보를 함께 사용하여 Pass t+1의 날판두께를 예측

【모델】
1. E2X (Entry → Exit): Pass t(홀수) + Pass t+1(짝수) → Pass t+1 날판두께
   - 5+6 → 6, 7+8 → 8, 9+10 → 10
   
2. X2E (Exit → Entry): Pass t(짝수) + Pass t+1(홀수) → Pass t+1 날판두께  
   - 6+7 → 7, 8+9 → 9

【데이터 흐름】
1. 데이터 병합 및 로드
   - data/entry_direction_results.csv
   - data/exit_direction_results.csv  
   - data/posco1_105190.csv (RM)
   - data/posco2_105190.csv (FM)

2. Pass t → t+1 전이쌍 생성
   - 각 plate마다 연속된 pass 쌍 생성
   - Pass t의 특징: RM 통계량, FM 데이터, 날판두께
   - Pass t+1의 특징: RM 통계량, FM 데이터
   - 타깃: Pass t+1의 날판두께

3. 전이쌍 필터링
   - E2X: 5→6, 7→8, 9→10 (ALLOWED_E2X_SOURCE_PASSES = {5, 7, 9})
   - X2E: 6→7, 8→9 (ALLOWED_X2E_SOURCE_PASSES = {6, 8})

4. 학습 방법
   - SVM (Support Vector Regression) 사용
   - GroupShuffleSplit으로 plate 기준 train/test 분할
   - Train set에서만 통계량 계산
   - 각 방향별로 별도 모델 학습 (E2X, X2E)

5. 데이터 누수 방지
   - Train/Test 분할 후 모든 통계량 계산 (train-only)
   - Pass 연속성 검증 (t → t+1만 허용)
   - 타깃 정보 누수를 막기 위해 t+1의 날판두께는 타깃으로만 사용
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
TARGET_COL = "target_next_warp"

# 허용된 전이 소스 패스
ALLOWED_E2X_SOURCE_PASSES = {5, 7, 9}  # Entry → Exit: 5→6, 7→8, 9→10
ALLOWED_X2E_SOURCE_PASSES = {6, 8}     # Exit → Entry: 6→7, 8→9

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

# ---------- 1) 데이터 로드 및 Pass t → t+1 쌍 생성 ----------
def load_and_create_transition_pairs() -> pd.DataFrame:
    """
    각 plate의 연속된 pass 쌍(t, t+1)을 생성하고,
    Pass t 특징 + Pass t+1 특징 → Pass t+1 날판두께 구조를 만든다.
    """
    print("\n" + "="*80)
    print("[Step 1] 데이터 병합 및 Pass t → t+1 전이쌍 생성")
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
    
    # Pass t → t+1 쌍 생성
    pairs = []
    
    for plate, group in merged_fm.groupby(PLATE_COL):
        group = group.sort_values(PASS_COL).reset_index(drop=True)
        
        for i in range(len(group) - 1):
            pass_t = group.iloc[i]
            pass_t_plus_1 = group.iloc[i + 1]
            
            # 연속성 검증: t+1이 t보다 정확히 1 큰지 확인
            if pass_t_plus_1[PASS_COL] != pass_t[PASS_COL] + 1:
                continue
            
            # 특징 컬럼 선택
            exclude_cols = {PLATE_COL, PASS_COL, MONTH_COL, CUR_WARP}
            
            # Pass t 특징 (날판두께 포함)
            pass_t_feats = {f"passt_{k}": v for k, v in pass_t.items() 
                           if k not in exclude_cols or k == CUR_WARP}
            pass_t_feats[f"passt_{PASS_COL}"] = pass_t[PASS_COL]
            pass_t_feats[f"passt_{CUR_WARP}"] = pass_t[CUR_WARP]
            
            # Pass t+1 특징 (날판두께는 타깃으로만 사용)
            pass_t1_feats = {f"passt1_{k}": v for k, v in pass_t_plus_1.items() 
                            if k not in exclude_cols and k != CUR_WARP}
            pass_t1_feats[f"passt1_{PASS_COL}"] = pass_t_plus_1[PASS_COL]
            
            # 쌍 정보
            pair_data = {
                PLATE_COL: plate,
                "source_pass": pass_t[PASS_COL],
                "target_pass": pass_t_plus_1[PASS_COL],
                TARGET_COL: pass_t_plus_1[CUR_WARP],  # 타깃: t+1의 날판두께
            }
            pair_data.update(pass_t_feats)
            pair_data.update(pass_t1_feats)
            
            pairs.append(pair_data)
    
    pairs_df = pd.DataFrame(pairs)
    
    # NaN 제거
    before = len(pairs_df)
    pairs_df = pairs_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"전이쌍 생성: {before}개 → NaN 제거 후 {len(pairs_df)}개")
    
    # 저장
    out_path = PROCESSED_DIR / "transition_pairs_raw.csv"
    pairs_df.to_csv(out_path, index=False)
    print(f"✓ 저장: {out_path}")
    
    return pairs_df

# -------------------- 2) 전이쌍 필터링 --------------------
def filter_transitions(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    dataset_name('E2X'|'X2E')에 따라 허용된 source pass만 남긴다.
    """
    if dataset_name.upper() == "E2X":
        allowed = ALLOWED_E2X_SOURCE_PASSES
    else:
        allowed = ALLOWED_X2E_SOURCE_PASSES
    
    before = len(df)
    filtered = df[df["source_pass"].isin(allowed)].copy()
    print(f"[Filter {dataset_name}] source_pass in {sorted(allowed)}: {len(filtered)}/{before}")
    
    return filtered

# -------------------- 3) Feature Engineering --------------------
def feature_engineering(df: pd.DataFrame, output_path: Path):
    """
    Pass t와 Pass t+1 각각의 FM 특징들로부터 통계량 생성
    """
    print(f"\n[Step 2] Feature Engineering")
    df = df.copy()
    
    # Pass t FM 특징
    passt_fm_cols = [c for c in df.columns if c.startswith("passt_FM_") and "날판번호" not in c]
    if passt_fm_cols:
        fm_blk = df[passt_fm_cols]
        df["passt_FM_mean"] = fm_blk.mean(axis=1)
        df["passt_FM_std"] = fm_blk.std(axis=1)
        df["passt_FM_max"] = fm_blk.max(axis=1)
        df["passt_FM_min"] = fm_blk.min(axis=1)
        df["passt_FM_range"] = df["passt_FM_max"] - df["passt_FM_min"]
        df["passt_FM_cv"] = df["passt_FM_std"] / (df["passt_FM_mean"].abs() + 1e-8)
    
    # Pass t+1 FM 특징
    passt1_fm_cols = [c for c in df.columns if c.startswith("passt1_FM_") and "날판번호" not in c]
    if passt1_fm_cols:
        fm_blk = df[passt1_fm_cols]
        df["passt1_FM_mean"] = fm_blk.mean(axis=1)
        df["passt1_FM_std"] = fm_blk.std(axis=1)
        df["passt1_FM_max"] = fm_blk.max(axis=1)
        df["passt1_FM_min"] = fm_blk.min(axis=1)
        df["passt1_FM_range"] = df["passt1_FM_max"] - df["passt1_FM_min"]
        df["passt1_FM_cv"] = df["passt1_FM_std"] / (df["passt1_FM_mean"].abs() + 1e-8)
    
    # Pass 관련 특징
    if "source_pass" in df.columns:
        df["source_pass_squared"] = df["source_pass"] ** 2
    if "target_pass" in df.columns:
        df["target_pass_squared"] = df["target_pass"] ** 2
    
    df.to_csv(output_path, index=False)
    print(f"  ✓ 저장: {output_path}")
    return df

# -------------------- 4) 학습 --------------------
def train_model(df: pd.DataFrame, 
                dataset_name: str,
                seed: int = 42, 
                test_size: float = 0.2, 
                select_k: int = 120):
    """
    SVM을 사용하여 Pass t + Pass t+1 → Pass t+1 날판두께 예측
    """
    print("\n" + "="*80)
    print(f"[Step 3] SVM 모델 학습 - {dataset_name}")
    print("="*80)
    
    data = df.copy()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 특징/타깃 분리
    exclude_from_x = {TARGET_COL, PLATE_COL, "source_pass", "target_pass"}
    X_all = data.drop(columns=[c for c in exclude_from_x if c in data.columns], errors="ignore")
    X_all = X_all.select_dtypes(include=[np.number]).copy()
    y_all = data[TARGET_COL].values
    groups_all = data[PLATE_COL].astype(str).values
    
    print(f"전체 특징 수: {X_all.shape[1]}, 샘플 수: {X_all.shape[0]}")
    
    # Group-aware split (plate 기준)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(X_all, y_all, groups=groups_all))
    
    X_tr_raw = X_all.iloc[tr_idx].copy()
    X_te_raw = X_all.iloc[te_idx].copy()
    y_tr = y_all[tr_idx].copy()
    y_te = y_all[te_idx].copy()
    
    print(f"Train 샘플: {len(y_tr)}, Test 샘플: {len(y_te)}")
    
    # ====== 데이터 누수 방지: train-only 통계량 계산 ======
    # train 중앙값으로만 결측 대치
    med = X_tr_raw.median(numeric_only=True).fillna(0.0).to_dict()
    X_tr = X_tr_raw.fillna(med)
    X_te = X_te_raw.fillna(med)
    
    # Feature selection (train에서만 fit)
    k_actual = max(1, min(select_k, X_tr.shape[1]))
    print(f"Feature selection: {X_tr.shape[1]} → {k_actual} features")
    
    selector = SelectKBest(score_func=f_regression, k=k_actual)
    selector.fit(X_tr.values, y_tr)
    Xtr_sel = selector.transform(X_tr.values)
    Xte_sel = selector.transform(X_te.values)
    
    # SVM 모델 파이프라인
    model = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("reg", SVR(kernel="rbf", C=10.0, epsilon=0.1)),
    ])
    
    # 학습
    print("모델 학습 중...")
    model.fit(Xtr_sel, y_tr)
    
    # 예측
    y_tr_pred = model.predict(Xtr_sel)
    y_te_pred = model.predict(Xte_sel)
    
    # 평가
    train_r2 = float(r2_score(y_tr, y_tr_pred))
    test_r2 = float(r2_score(y_te, y_te_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_te, y_te_pred)))
    
    print(f"\n[{dataset_name} 성능]")
    print(f"  Train R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
    print(f"  Test R2:  {test_r2:.4f}, RMSE: {test_rmse:.4f}")
    
    res = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_samples": int(len(y_tr)),
        "test_samples": int(len(y_te)),
    }
    
    # 모델 저장
    prefix = dataset_name.lower()
    joblib.dump(model, MODELS_DIR / f"revised1_{prefix}_svm_model.pkl")
    joblib.dump(selector, MODELS_DIR / f"revised1_{prefix}_selector.pkl")
    joblib.dump(list(X_tr.columns), MODELS_DIR / f"revised1_{prefix}_feature_cols.pkl")
    joblib.dump(med, MODELS_DIR / f"revised1_{prefix}_medians.pkl")
    
    print(f"✓ 모델 저장 완료: {prefix}")
    
    return res

# -------------------- 5) 요약 작성 --------------------
def write_summary(e2x_res: dict, x2e_res: dict, 
                 summary_path: Path, timestamp: datetime, experiment_id: str):
    lines = []
    lines.append("Pass t + Pass t+1 → Pass t+1 Thickness Regression (118_revised_1.py)")
    lines.append("="*72)
    lines.append(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} 작성")
    lines.append(f"실험 ID: {experiment_id}\n")
    
    lines.append("[모델 설명]")
    lines.append("Pass t와 Pass t+1의 정보를 함께 사용하여 Pass t+1의 날판두께 예측")
    lines.append("알고리즘: SVM (RBF kernel)\n")
    
    lines.append("[모델 1: E2X (Entry → Exit)]")
    lines.append(f"- 전이: 5+6→6, 7+8→8, 9+10→10")
    lines.append(f"- Train R2: {e2x_res['train_r2']:.4f} | RMSE: {e2x_res['train_rmse']:.4f}")
    lines.append(f"- Test R2:  {e2x_res['test_r2']:.4f} | RMSE: {e2x_res['test_rmse']:.4f}")
    lines.append(f"- Train samples: {e2x_res['train_samples']}, Test samples: {e2x_res['test_samples']}\n")
    
    lines.append("[모델 2: X2E (Exit → Entry)]")
    lines.append(f"- 전이: 6+7→7, 8+9→9")
    lines.append(f"- Train R2: {x2e_res['train_r2']:.4f} | RMSE: {x2e_res['train_rmse']:.4f}")
    lines.append(f"- Test R2:  {x2e_res['test_r2']:.4f} | RMSE: {x2e_res['test_rmse']:.4f}")
    lines.append(f"- Train samples: {x2e_res['train_samples']}, Test samples: {x2e_res['test_samples']}\n")
    
    avg_test_r2 = (e2x_res['test_r2'] + x2e_res['test_r2']) / 2
    lines.append(f"[평균 Test R2] {avg_test_r2:.4f}\n")
    
    lines.append("[데이터 산출물]")
    lines.append(f"- Transition pairs: {PROCESSED_DIR / 'transition_pairs_raw.csv'}")
    lines.append(f"- E2X featured: {PROCESSED_DIR / 'e2x_transition_featured.csv'}")
    lines.append(f"- X2E featured: {PROCESSED_DIR / 'x2e_transition_featured.csv'}")
    lines.append(f"- E2X model: {MODELS_DIR / 'revised1_e2x_svm_model.pkl'}")
    lines.append(f"- X2E model: {MODELS_DIR / 'revised1_x2e_svm_model.pkl'}\n")
    
    lines.append("[데이터 누수 방지]")
    lines.append("- Train/Test는 plate 단위로 분할 (GroupShuffleSplit)")
    lines.append("- 모든 통계량(중앙값, feature selection, scaling)은 train-only로 계산")
    lines.append("- Pass 연속성 검증 (t+1이 t보다 정확히 1 큰 경우만)")
    lines.append("- Pass t+1의 날판두께는 타깃으로만 사용, 특징에서 제외")
    
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ 요약 저장: {summary_path}")

# ----------------------------- main ------------------------------
def main():
    _ensure_dirs()
    experiment_id = _next_experiment_id()
    timestamp = datetime.now()
    summary_path = REPORTS_SUMMARY_DIR / f"{experiment_id}_118_revised_1_summary.txt"
    archived_script_path = REPORTS_CODE_DIR / f"{experiment_id}_118_revised_1.py"
    
    try:
        # 1) 전이쌍 생성
        pairs_df = load_and_create_transition_pairs()
        
        # 2) E2X 데이터 처리
        e2x_df = filter_transitions(pairs_df, "E2X")
        e2x_featured = feature_engineering(
            e2x_df, 
            PROCESSED_DIR / "e2x_transition_featured.csv"
        )
        
        # 3) X2E 데이터 처리
        x2e_df = filter_transitions(pairs_df, "X2E")
        x2e_featured = feature_engineering(
            x2e_df, 
            PROCESSED_DIR / "x2e_transition_featured.csv"
        )
        
        # 4) 모델 학습
        e2x_res = train_model(e2x_featured, "E2X")
        x2e_res = train_model(x2e_featured, "X2E")
        
        # 5) 요약 작성
        write_summary(e2x_res, x2e_res, summary_path, timestamp, experiment_id)
        
        # 6) 코드 아카이브
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
        err_path = REPORTS_DIR / "last_error_traceback_revised1.txt"
        err_path.write_text(tb, encoding="utf-8")
        print(f"↳ 전체 traceback을 {err_path} 에 저장했습니다.")

if __name__ == "__main__":
    main()
