# -*- coding: utf-8 -*-
"""
003_based_experiment_Conly_fast.py
- 목적: AVG R² 개선 아이디어 C1/C2/C3만 빠르게 실행
  * C1: 시퀀스 lag/momentum/rolling (판 내부 과거만 → 누수 없음)
  * C2: 쌍별 Δ-통계(OOF, EB수축) 피처 + 이분산 가중 학습(WLS)
  * C3: C1+C2 + 직전 FM 요약 lag + (XGB는 pseudo-Huber)
- 알고리즘 고정: E2X=XGBoost, X2E=RandomForest (003 실험 근거)
- 데이터: data/processed/e2x_cleaned_tplus1.csv, x2e_cleaned_tplus1.csv (+all_featured_tplus1.csv 있으면 사용)
- 출력: reports/{ID}_{RUN_NAME}_steps.csv / summaries/{ID}_{RUN_NAME}_summary.txt
"""

# ========================= [ USER CONFIG ] =========================
RUN_NAME           = "003_based_Conly_fast_v1"
SEED               = 42
CV_FOLDS           = 2        # OOF/누수방지 최소화용
SELECTK_K          = 120
PAIR_EB_N0         = 50       # EB 수축 강도
WEIGHT_CLIP        = (0.25, 3.0)
USE_ROBUST_LOSS_C3 = True     # XGB robust loss(pseudo-Huber)
DO_STRICT_RF       = False    # 계약지표는 Avg R² → 기본 False

# 알고리즘 고정 (003 결과 근거)
FIX_ALGO = {"E2X":"XGBoost", "X2E":"RandomForest"}
XGB_N_EST = 400
# ==================================================================

import warnings, shutil, re, joblib, traceback
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path("data"); PRO_DIR = DATA_DIR/"processed"
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SUM_DIR = REPORTS_DIR/"summaries"; SUM_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR = REPORTS_DIR/"scripts"; SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "warping_index_target"
CUR_WARP   = "warping_index_current_pass"
PLATE_COL  = "FM_날판번호"
PASS_COL   = "FM_PASS NO N"
MONTH_COL  = "FM_압연월"
EXCLUDE    = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}

THIS_PATH = Path(__file__).resolve()

# ---------- utils ----------
def _next_experiment_id():
    nums=[]
    for f in SUM_DIR.glob("*.txt"):
        m=re.match(r"(\d{3})_", f.name)
        if m:
            try: nums.append(int(m.group(1)))
            except: pass
    return f"{(max(nums)+1) if nums else 1:03d}"

def _ensure_cols(df: pd.DataFrame):
    X_all = df.drop(columns=[c for c in EXCLUDE if c in df.columns], errors="ignore")
    X = X_all.select_dtypes(include=[np.number]).copy()
    med = X.median(numeric_only=True)
    return X.fillna(med), med, list(X.columns)

def _selectK_with_mandatory(Xtr_df, ytr, Xte_df, mandatory, k=SELECTK_K):
    mand = [c for c in mandatory if c in Xtr_df.columns]
    others = [c for c in Xtr_df.columns if c not in mand]
    if len(others)==0:
        chosen = mand
    else:
        k_remain = max(1, min(k-len(mand), len(others)))
        sel = SelectKBest(score_func=f_regression, k=k_remain)
        sel.fit(Xtr_df[others], ytr)
        chosen = mand + [others[i] for i in sel.get_support(indices=True)]
    return chosen, Xtr_df[chosen].values, Xte_df[chosen].values

def _make_pair_str(df): 
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

def _build_xgb(robust=False):
    params=dict(random_state=SEED, tree_method="hist", n_estimators=XGB_N_EST,
                learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                n_jobs=0)
    try:
        return XGBRegressor(objective=("reg:pseudohubererror" if robust else "reg:squarederror"), **params)
    except Exception:
        return XGBRegressor(objective=("reg:absoluteerror" if robust else "reg:squarederror"), **params)

def _build_rf():
    return RandomForestRegressor(random_state=SEED, n_estimators=600, n_jobs=-1)

def _split_by_saved_plates(df, prefix):
    pkl = MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl"
    X, med, cols = _ensure_cols(df); y = df[TARGET_COL].values
    if pkl.exists():
        test_plates = set(joblib.load(pkl))
        tr = df[~df[PLATE_COL].astype(str).isin(test_plates)].copy()
        te = df[df[PLATE_COL].astype(str).isin(test_plates)].copy()
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr_idx, te_idx = next(gss.split(X, y, groups=df[PLATE_COL].astype(str).values))
        test_plates = sorted(df.iloc[te_idx][PLATE_COL].astype(str).unique())
        joblib.dump(test_plates, pkl)
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
    return tr, te

# ---------- C1: seq lag ----------
def _add_seq_lag_features(all_df: pd.DataFrame, include_fm_lag=False):
    df = all_df.copy().sort_values([PLATE_COL, PASS_COL])
    g = df.groupby(PLATE_COL, sort=False)
    df["lag1_prev_warp"]  = g[CUR_WARP].shift(1)
    df["lag2_prev2_warp"] = g[CUR_WARP].shift(2)
    df["mom1_warp"]       = df[CUR_WARP] - df["lag1_prev_warp"]
    df["mom2_warp"]       = df["lag1_prev_warp"] - df["lag2_prev2_warp"]
    df["roll_std_warp2"]  = g[CUR_WARP].apply(lambda s: s.rolling(2).std()).reset_index(level=0, drop=True)
    if include_fm_lag:
        bases=["FM_mean","FM_std","FM_max","FM_min","FM_range","FM_cv"]
        if not any(b in df.columns for b in bases):
            fm_cols = [c for c in df.columns if c.startswith("FM_") and c not in {PLATE_COL,PASS_COL,MONTH_COL}]
            if fm_cols:
                blk=df[fm_cols]
                df["FM_mean"]=blk.mean(axis=1); df["FM_std"]=blk.std(axis=1)
                df["FM_max"]=blk.max(axis=1);   df["FM_min"]=blk.min(axis=1)
                df["FM_range"]=df["FM_max"]-df["FM_min"]; df["FM_cv"]=df["FM_std"]/(df["FM_mean"].abs()+1e-8)
        for b in bases:
            if b in df.columns:
                df[f"{b}_lag1"] = g[b].shift(1)
    return df

def step_C1_seq_lag(train_df, test_df, prefix):
    Xtr_df, med, cols = _ensure_cols(train_df); Xte_df = test_df[cols].fillna(med)
    ytr, yte = train_df[TARGET_COL].values, test_df[TARGET_COL].values
    mand = [c for c in ["lag1_prev_warp","mom1_warp",CUR_WARP] if c in Xtr_df.columns]
    chosen, Xtr, Xte = _selectK_with_mandatory(Xtr_df, ytr, Xte_df, mand, k=SELECTK_K)
    algo = FIX_ALGO.get(prefix.upper(), "XGBoost")
    mdl  = _build_xgb(False) if algo=="XGBoost" else _build_rf()
    mdl.fit(Xtr, ytr); pte = mdl.predict(Xte)
    return dict(r2_te=float(r2_score(yte, pte)), algo=algo, pred_te=pte)

# ---------- C2/C3: pair Δ-통계 + WLS ----------
def _pair_te_oof_and_weights(train_df, test_df, n0=PAIR_EB_N0, w_clip=WEIGHT_CLIP, eps=1e-6):
    tr=train_df.copy(); te=test_df.copy()
    tr["pair"]=_make_pair_str(tr); te["pair"]=_make_pair_str(te)
    tr["delta"]=tr[TARGET_COL]-tr[CUR_WARP]

    gkf=GroupKFold(n_splits=CV_FOLDS); groups=tr[PLATE_COL].astype(str).values
    oof_mean=np.zeros(len(tr)); oof_std=np.zeros(len(tr)); oof_n=np.zeros(len(tr))
    for tr_i, va_i in gkf.split(tr, tr["delta"].values, groups):
        sub=tr.iloc[tr_i]; gm=sub["delta"].mean(); gs=sub["delta"].std(ddof=1)
        stat=sub.groupby("pair")["delta"].agg(n="size", mean="mean", std="std").reset_index()
        w=stat["n"]/(stat["n"]+n0)
        stat["mean_sh"]=w*stat["mean"]+(1-w)*gm
        stat["std_sh"]= w*stat["std"].fillna(gs)+(1-w)*gs
        m_mean=dict(zip(stat["pair"], stat["mean_sh"])); m_std=dict(zip(stat["pair"], stat["std_sh"]))
        m_n=dict(zip(stat["pair"], stat["n"]))
        p_va=tr.iloc[va_i]["pair"].values
        oof_mean[va_i]=[m_mean.get(p,gm) for p in p_va]
        oof_std[va_i] =[m_std.get(p,gs)  for p in p_va]
        oof_n[va_i]   =[m_n.get(p,1)     for p in p_va]

    # test map(full-train)
    gm=tr["delta"].mean(); gs=tr["delta"].std(ddof=1)
    stat=tr.groupby("pair")["delta"].agg(n="size", mean="mean", std="std").reset_index()
    w=stat["n"]/(stat["n"]+n0)
    stat["mean_sh"]=w*stat["mean"]+(1-w)*gm
    stat["std_sh"]= w*stat["std"].fillna(gs)+(1-w)*gs
    m_mean=dict(zip(stat["pair"], stat["mean_sh"])); m_std=dict(zip(stat["pair"], stat["std_sh"]))
    te_mean=te["pair"].map(m_mean).fillna(gm).values
    te_std =te["pair"].map(m_std ).fillna(gs).values

    sigma0=float(tr["delta"].std(ddof=1))
    w_tr=(oof_n/(oof_n+n0))*(sigma0/(oof_std+eps))
    w_tr=np.clip(w_tr, w_clip[0], w_clip[1])

    cache = {
        "train_index": train_df.index,
        "oof_mean": oof_mean, "oof_std": oof_std, "w_tr": w_tr,
        "test_mean": te_mean, "test_std": te_std
    }
    return cache

def step_C2_pair_te_weighted(train_df, test_df, prefix, te_cache):
    ytr, yte = train_df[TARGET_COL].values, test_df[TARGET_COL].values
    Xtr_df, med, cols = _ensure_cols(train_df); Xte_df = test_df[cols].fillna(med)

    # ★ index 정렬 후 안전하게 부착 (길이/순서 보장)
    s_mean = pd.Series(te_cache["oof_mean"], index=te_cache["train_index"]).reindex(train_df.index)
    s_std  = pd.Series(te_cache["oof_std"],  index=te_cache["train_index"]).reindex(train_df.index)
    Xtr_df = Xtr_df.assign(pair_delta_mean_te=s_mean.values,
                           pair_delta_std_te=s_std.values)
    Xte_df = Xte_df.assign(pair_delta_mean_te=te_cache["test_mean"],
                           pair_delta_std_te=te_cache["test_std"])

    # 가중치도 동일하게 정렬
    w_tr = pd.Series(te_cache["w_tr"], index=te_cache["train_index"]).reindex(train_df.index)
    w_tr = w_tr.fillna(w_tr.median()).values  # 드물게 미스매치가 있을 경우 완충

    mand=[c for c in [CUR_WARP,"pair_delta_std_te"] if c in Xtr_df.columns]
    chosen, Xtr, Xte = _selectK_with_mandatory(Xtr_df, ytr, Xte_df, mandatory=mand, k=SELECTK_K)

    algo = FIX_ALGO.get(prefix.upper(), "XGBoost")
    robust = (USE_ROBUST_LOSS_C3 if algo=="XGBoost" else False)
    mdl = _build_xgb(robust) if algo=="XGBoost" else _build_rf()
    mdl.fit(Xtr, ytr, sample_weight=w_tr); pte=mdl.predict(Xte)
    return dict(r2_te=float(r2_score(yte, pte)), algo=algo, pred_te=pte)

def step_C3_seq_fm_te_weighted(train_df, test_df, prefix, te_cache):
    ytr, yte = train_df[TARGET_COL].values, test_df[TARGET_COL].values
    Xtr_df, med, cols = _ensure_cols(train_df); Xte_df = test_df[cols].fillna(med)

    # ★ Δ-통계 정렬 부착
    s_mean = pd.Series(te_cache["oof_mean"], index=te_cache["train_index"]).reindex(train_df.index)
    s_std  = pd.Series(te_cache["oof_std"],  index=te_cache["train_index"]).reindex(train_df.index)
    Xtr_df = Xtr_df.assign(pair_delta_mean_te=s_mean.values,
                           pair_delta_std_te=s_std.values)
    Xte_df = Xte_df.assign(pair_delta_mean_te=te_cache["test_mean"],
                           pair_delta_std_te=te_cache["test_std"])

    # 필수 피처 지정(lag/FM-lag가 실제로 존재하는 경우만)
    mand=[c for c in [CUR_WARP,"lag1_prev_warp","pair_delta_std_te","FM_mean_lag1"] if c in Xtr_df.columns]
    chosen, Xtr, Xte = _selectK_with_mandatory(Xtr_df, ytr, Xte_df, mandatory=mand, k=SELECTK_K)

    # 가중치 정렬
    w_tr = pd.Series(te_cache["w_tr"], index=te_cache["train_index"]).reindex(train_df.index)
    w_tr = w_tr.fillna(w_tr.median()).values

    algo = FIX_ALGO.get(prefix.upper(), "XGBoost")
    robust = (USE_ROBUST_LOSS_C3 if algo=="XGBoost" else False)
    mdl = _build_xgb(robust) if algo=="XGBoost" else _build_rf()
    mdl.fit(Xtr, ytr, sample_weight=w_tr); pte = mdl.predict(Xte)
    return dict(r2_te=float(r2_score(yte, pte)), algo=algo, pred_te=pte)

# ---------- runner ----------
def run_one_side_c_only(df_base, df_lag_only, df_lag_fm, prefix):
    # 공통 split 고정(저장된 test plates 재사용)
    tr_base, te_base = _split_by_saved_plates(df_base,     prefix.lower())
    tr_c1,   te_c1   = _split_by_saved_plates(df_lag_only, prefix.lower())
    tr_c3,   te_c3   = _split_by_saved_plates(df_lag_fm,   prefix.lower())

    # ★ Δ-통계(OOB) 캐시를 단계별 학습 DF로 각각 계산
    te_cache_c2 = _pair_te_oof_and_weights(tr_base, te_base, n0=PAIR_EB_N0, w_clip=WEIGHT_CLIP)
    te_cache_c3 = _pair_te_oof_and_weights(tr_c3,   te_c3,   n0=PAIR_EB_N0, w_clip=WEIGHT_CLIP)

    c1 = step_C1_seq_lag(tr_c1, te_c1, prefix)
    c2 = step_C2_pair_te_weighted(tr_base, te_base, prefix, te_cache_c2)
    c3 = step_C3_seq_fm_te_weighted(tr_c3, te_c3, prefix, te_cache_c3)
    return dict(C1=c1, C2=c2, C3=c3)


def main():
    e2x = pd.read_csv(PRO_DIR/"e2x_cleaned_tplus1.csv")
    x2e = pd.read_csv(PRO_DIR/"x2e_cleaned_tplus1.csv")
    allf = pd.read_csv(PRO_DIR/"all_featured_tplus1.csv") if (PRO_DIR/"all_featured_tplus1.csv").exists() else pd.concat([e2x,x2e], ignore_index=True)

    # C용 파생
    e2x_c1 = _add_seq_lag_features(e2x, include_fm_lag=False)
    e2x_c3 = _add_seq_lag_features(e2x, include_fm_lag=True)
    x2e_c1 = _add_seq_lag_features(x2e, include_fm_lag=False)
    x2e_c3 = _add_seq_lag_features(x2e, include_fm_lag=True)

    exp_id = _next_experiment_id()
    steps_csv   = REPORTS_DIR / f"{exp_id}_{RUN_NAME}_steps.csv"
    summary_txt = SUM_DIR     / f"{exp_id}_{RUN_NAME}_summary.txt"
    archived_py = SCRIPTS_DIR / f"{exp_id}_003_based_experiment_Conly_fast.py"

    try:
        print("\n=== [C-only] E2X ==="); e2x_res = run_one_side_c_only(e2x, e2x_c1, e2x_c3, "E2X")
        print("\n=== [C-only] X2E ==="); x2e_res = run_one_side_c_only(x2e, x2e_c1, x2e_c3, "X2E")

        avg = lambda a,b: float(np.mean([a,b]))
        rows = [
            {"step":"C1 +SeqLag",              "E2X":e2x_res["C1"]["r2_te"], "X2E":x2e_res["C1"]["r2_te"], "Avg":avg(e2x_res["C1"]["r2_te"], x2e_res["C1"]["r2_te"])},
            {"step":"C2 +pairΔ-TE+WLS",        "E2X":e2x_res["C2"]["r2_te"], "X2E":x2e_res["C2"]["r2_te"], "Avg":avg(e2x_res["C2"]["r2_te"], x2e_res["C2"]["r2_te"])},
            {"step":"C3 +SeqLag+FMlag+TE+WLS", "E2X":e2x_res["C3"]["r2_te"], "X2E":x2e_res["C3"]["r2_te"], "Avg":avg(e2x_res["C3"]["r2_te"], x2e_res["C3"]["r2_te"])},
        ]
        pd.DataFrame(rows).to_csv(steps_csv, index=False)
        print(f"\n✓ 단계별 R² 저장: {steps_csv}")

        # Summary
        now = datetime.now()
        lines=[]
        lines.append(f"003-based C-only Summary ({now.strftime('%Y-%m-%d %H:%M:%S')})")
        lines.append("="*72)
        lines.append(f"실험 ID: {exp_id} | RUN_NAME: {RUN_NAME}\n")
        lines.append("[모델 성능 - Test R²]")
        lines.append(f"* E2X | C1: {e2x_res['C1']['r2_te']:.4f} | C2: {e2x_res['C2']['r2_te']:.4f} | C3: {e2x_res['C3']['r2_te']:.4f}")
        lines.append(f"* X2E | C1: {x2e_res['C1']['r2_te']:.4f} | C2: {x2e_res['C2']['r2_te']:.4f} | C3: {x2e_res['C3']['r2_te']:.4f}")
        lines.append(f"\n[평균 Test R²] C1={rows[0]['Avg']:.4f} | C2={rows[1]['Avg']:.4f} | C3={rows[2]['Avg']:.4f}")
        if DO_STRICT_RF: lines.append("[Strict roll-forward R²] (생략 설정)")
        lines.append("\n[참고] 003 기준치: Avg Test R² 0.4174")  # 비교 기준(요약) 
        summary_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"✓ 요약 저장: {summary_txt}")

        shutil.copy2(THIS_PATH, archived_py); print(f"✓ 코드 아카이브: {archived_py}")

    except Exception as e:
        print(f"[실패] {type(e).__name__}: {e}")
        tb = traceback.format_exc()
        (REPORTS_DIR/"last_error_traceback.txt").write_text(tb, encoding="utf-8")
        raise

if __name__ == "__main__":
    main()
