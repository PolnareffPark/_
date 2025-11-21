# -*- coding: utf-8 -*-
"""
003_based_experiment.py
- 003 파이프라인을 기반으로, S0(직접 t+1) → B1(OOF Isotonic) → B2(pair-linear prior + Residual-Stacking)
  → B3(Top-N pair Specialist Blending) 순차 적용으로 R² 상승을 노린다.
- 모델: XGBoost / RandomForest only (Optuna 없음)
- 누수 방지: plate-GroupKFold OOF 전용
- 결과: reports/summaries/{ID}_003_based_summary.txt 한 장 + steps CSV + 코드 스냅샷
"""

# ========================= [ USER CONFIG - 상단만 수정 ] =========================
RUN_NAME                 = "003_based_v1"     # 보고/steps 접두사. 동시 다실험 시 서로 다른 이름으로 변경
SEED                     = 42
CV_FOLDS                 = 3                   # plate-GroupKFold
SELECTK_Baseline_K       = 120                 # S0에서만 K-best 적용
USE_SELECTK_Residual     = False               # Residual-head는 피처 최대 유지(신호 보존)
LAMBDA_GRID              = [0.0, 0.25, 0.5, 0.75, 1.0]   # ŷ_final = ŷ_base + λ·ŷ_resid
PAIR_EB_N0               = 40                  # pair (a,b) EB 수축 강도
SPECIALIST_MIN_N         = 200                 # 대형 쌍 최소 n
SPECIALIST_BLEND_N0      = 40                  # Specialist γ = n/(n+N0)
SAVE_BUNDLES_FOR_RF      = True                # Strict roll-forward용 번들 저장
DO_STRICT_ROLLFORWARD    = True                # 교집합 Test 판만 평가
# ===============================================================================

import warnings, shutil, re, joblib, traceback
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.isotonic import IsotonicRegression

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------- Paths / Consts --------------------------------
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

# --------------------------------- Utilities -----------------------------------
def _next_experiment_id():
    existing = []
    for f in SUM_DIR.glob("*.txt"):
        m = re.match(r"(\d{3})_", f.name)
        if m:
            try: existing.append(int(m.group(1)))
            except: pass
    return f"{(max(existing)+1) if existing else 1:03d}"

def _ensure_cols(df: pd.DataFrame):
    X_all = df.drop(columns=[c for c in EXCLUDE if c in df.columns], errors="ignore")
    X = X_all.select_dtypes(include=[np.number]).copy()
    med = X.median(numeric_only=True)
    return X.fillna(med), med, list(X.columns)

def _selectK_with_mandatory(Xtr_df, ytr, Xte_df, mandatory, use_select, k):
    """SelectKBest(f_regression), 단 mandatory 컬럼은 반드시 포함.
       (이슈 #4 대응: mandatory 처리 중 피처 수 불일치 방지)"""
    if not use_select:
        cols = list(Xtr_df.columns)
        return cols, Xtr_df.values, Xte_df.values

    mandatory_in = [c for c in mandatory if c in Xtr_df.columns]
    others = [c for c in Xtr_df.columns if c not in mandatory_in]
    if len(others) == 0:
        chosen = mandatory_in
    else:
        k_remain = max(1, min(k - len(mandatory_in), len(others)))
        sel = SelectKBest(score_func=f_regression, k=k_remain)
        sel.fit(Xtr_df[others], ytr)
        chosen_others = [others[i] for i in sel.get_support(indices=True)]
        chosen = mandatory_in + chosen_others
    return chosen, Xtr_df[chosen].values, Xte_df[chosen].values

def _make_pair_str(df: pd.DataFrame) -> pd.Series:
    """이슈 #5 대응: 함수명 명확화."""
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

def _build_xgb(robust: bool):
    params = dict(random_state=SEED, tree_method="hist",
                  n_estimators=600, learning_rate=0.05,
                  max_depth=6, subsample=0.8, colsample_bytree=0.8)
    try:
        return XGBRegressor(objective=("reg:pseudohubererror" if robust else "reg:squarederror"), **params)
    except Exception:
        return XGBRegressor(objective=("reg:absoluteerror" if robust else "reg:squarederror"), **params)

def _build_rf():
    return RandomForestRegressor(random_state=SEED, n_estimators=600, n_jobs=-1)

def _pick_algo(X, y, groups, robust=False):
    gkf = GroupKFold(n_splits=CV_FOLDS)
    cand = [("XGBoost", _build_xgb(robust)), ("RandomForest", _build_rf())]
    best_name, best = cand[0][0], -1e9
    for name, mdl in cand:
        scores=[]
        for tr, va in gkf.split(X, y, groups):
            mdl.fit(X[tr], y[tr]); scores.append(r2_score(y[va], mdl.predict(X[va])))
        m = float(np.mean(scores))
        if m > best: best, best_name = m, name
    return best_name

def _split_by_saved_plates(df: pd.DataFrame, prefix: str):
    """003과 동일: 저장된 테스트 판 재사용, 없으면 새로 생성(plate-group)."""
    pkl = MODELS_DIR / f"{prefix}_test_plates_tplus1.pkl"
    X_raw, med, cols = _ensure_cols(df)
    y = df[TARGET_COL].values
    if pkl.exists():
        test_plates = set(joblib.load(pkl))
        tr = df[~df[PLATE_COL].astype(str).isin(test_plates)].copy()
        te = df[df[PLATE_COL].astype(str).isin(test_plates)].copy()
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr_idx, te_idx = next(gss.split(X_raw, y, groups=df[PLATE_COL].astype(str).values))
        test_plates = sorted(df.iloc[te_idx][PLATE_COL].astype(str).unique())
        joblib.dump(test_plates, pkl)
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
    return tr, te

# --------------------------------- Steps ---------------------------------------
def step_S0_baseline(train_df, test_df):
    Xtr_df, med, cols = _ensure_cols(train_df)
    Xte_df = test_df[cols].fillna(med)
    ytr, yte = train_df[TARGET_COL].values, test_df[TARGET_COL].values

    # SelectKBest(120) only for baseline
    chosen, Xtr, Xte = _selectK_with_mandatory(Xtr_df, ytr, Xte_df, mandatory=[], use_select=True, k=SELECTK_Baseline_K)

    algo = _pick_algo(Xtr, ytr, train_df[PLATE_COL].astype(str).values, robust=False)
    mdl = _build_xgb(False) if algo=="XGBoost" else _build_rf()
    mdl.fit(Xtr, ytr)
    ptr, pte = mdl.predict(Xtr), mdl.predict(Xte)

    bundle = dict(algo=algo, med=med, cols=cols, chosen=chosen, selector="selectk",
                  model=mdl, kind="baseline")
    return dict(bundle=bundle,
                r2_tr=float(r2_score(ytr, ptr)), r2_te=float(r2_score(yte, pte)),
                rmse_tr=float(np.sqrt(mean_squared_error(ytr, ptr))),
                rmse_te=float(np.sqrt(mean_squared_error(yte, pte))),
                pred_tr=ptr, pred_te=pte)

def _oof_preds_baseline(train_df):
    """Baseline의 OOF 예측(누수 방지). N2/N3 반환값 일관화(#3 대응)."""
    X_df, _, _ = _ensure_cols(train_df)
    y = train_df[TARGET_COL].values
    groups = train_df[PLATE_COL].astype(str).values
    gkf = GroupKFold(n_splits=CV_FOLDS)
    oof = np.zeros_like(y, dtype=float)
    for tr, va in gkf.split(X_df.values, y, groups):
        Xtr_df, Xva_df = X_df.iloc[tr], X_df.iloc[va]; ytr = y[tr]
        chosen, Xtr, Xva = _selectK_with_mandatory(Xtr_df, ytr, Xva_df, mandatory=[], use_select=True, k=SELECTK_Baseline_K)
        algo = _pick_algo(Xtr, ytr, groups[tr], robust=False)
        mdl = _build_xgb(False) if algo=="XGBoost" else _build_rf()
        mdl.fit(Xtr, ytr); oof[va] = mdl.predict(Xva)
    return oof

def _pair_linear_prior_oof(train_df, test_df, n0=PAIR_EB_N0):
    """fold별로 학습 파트에서 pair별 a,b 추정 → EB 수축 → OOF prior_pred 생성.
       test는 full-train에서 추정한 (a,b)로 prior_pred_test 생성."""
    ytr = train_df[TARGET_COL].values; cur_tr = train_df[CUR_WARP].values
    pair_tr = _make_pair_str(train_df).astype(str).values

    gkf = GroupKFold(n_splits=CV_FOLDS)
    groups = train_df[PLATE_COL].astype(str).values
    prior_oof = np.zeros(len(train_df))

    def _fit_ab(df_sub):
        y = df_sub[TARGET_COL].values; t = df_sub[CUR_WARP].values; p = _make_pair_str(df_sub).astype(str).values
        # global OLS
        b0 = np.cov(t, y, ddof=1)[0,1] / (np.var(t, ddof=1) + 1e-8); a0 = y.mean() - b0*t.mean()
        stat=[]
        for k, sub in pd.DataFrame({"p":p,"y":y,"t":t}).groupby("p"):
            n=len(sub); 
            if n<2:
                stat.append((k,n,a0,b0)); continue
            tt=sub["t"].values; yy=sub["y"].values
            b = np.cov(tt, yy, ddof=1)[0,1] / (np.var(tt, ddof=1) + 1e-8)
            a = yy.mean() - b*tt.mean()
            w = n/(n+n0); a_s = w*a + (1-w)*a0; b_s = w*b + (1-w)*b0
            stat.append((k,n,a_s,b_s))
        df_ab = pd.DataFrame(stat, columns=["pair","n","a","b"]).set_index("pair")
        return (a0,b0), df_ab

    # OOF prior
    for tr, va in gkf.split(train_df, ytr, groups):
        sub = train_df.iloc[tr]
        (a0,b0), ab = _fit_ab(sub)
        p_va = _make_pair_str(train_df.iloc[va]).astype(str).values
        t_va = train_df.iloc[va][CUR_WARP].values
        prior_oof[va] = np.array([ (ab.loc[p,"a"] + ab.loc[p,"b"]*t) if p in ab.index else (a0 + b0*t) for p,t in zip(p_va,t_va) ])

    # test prior (full-train)
    (a0,b0), ab_full = _fit_ab(train_df)
    p_te = _make_pair_str(test_df).astype(str).values
    t_te = test_df[CUR_WARP].values
    prior_te = np.array([ (ab_full.loc[p,"a"] + ab_full.loc[p,"b"]*t) if p in ab_full.index else (a0 + b0*t) for p,t in zip(p_te,t_te) ])
    return prior_oof, prior_te, (a0,b0), ab_full

def step_B1_isotonic(train_df, test_df, base_oof, base_pred_te):
    """Isotonic OOF 보정. (이슈 #3: 반환 구조 통일)"""
    y = train_df[TARGET_COL].values; yte = test_df[TARGET_COL].values
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(base_oof, y)
    pte_cal = ir.predict(base_pred_te)
    return dict(model=ir, pred_te=pte_cal, r2_te=float(r2_score(yte, pte_cal)))

def step_B2_residual_stack(train_df, test_df, base_oof, base_pred_te):
    """pair-linear prior + residual-head + λ-blend (OOF 기반 튜닝, 이슈 #6 보완)"""
    # prior(OOB) 생성
    prior_oof, prior_te, (a0,b0), ab_full = _pair_linear_prior_oof(train_df, test_df, n0=PAIR_EB_N0)

    # residual-head 훈련 피처
    def _build_res_X(df, oof_pred, prior_vec):
        X = df.drop(columns=[c for c in EXCLUDE if c in df.columns], errors="ignore").select_dtypes(include=[np.number]).copy()
        X["prior_pred"] = prior_vec
        if CUR_WARP in X.columns:
            X["cur_x_prior"] = X[CUR_WARP] * X["prior_pred"]
        res = df[TARGET_COL].values - oof_pred
        return X, res

    Xtr_df, res_tr = _build_res_X(train_df, base_oof, prior_oof)
    Xte_df, _      = _build_res_X(test_df,  np.full(len(test_df), base_pred_te.mean()), prior_te)

    # 선택/알고리즘
    chosen, Xtr, Xte = _selectK_with_mandatory(Xtr_df, res_tr, Xte_df, mandatory=[CUR_WARP, "prior_pred"], use_select=USE_SELECTK_Residual, k=256)
    algo = _pick_algo(Xtr, res_tr, train_df[PLATE_COL].astype(str).values, robust=True)
    mdl = _build_xgb(True) if algo=="XGBoost" else _build_rf()
    mdl.fit(Xtr, res_tr)
    res_hat_tr = mdl.predict(Xtr); res_hat_te = mdl.predict(Xte)

    # λ 튜닝 (OOF 기준)
    best_lam, best_r2 = 0.0, -1e9
    y = train_df[TARGET_COL].values
    for lam in LAMBDA_GRID:
        r2 = r2_score(y, base_oof + lam*res_hat_tr)
        if r2 > best_r2: best_r2, best_lam = r2, lam

    # 테스트 결합
    pred_te_stack = base_pred_te + best_lam * res_hat_te
    return dict(model=mdl, algo=algo, chosen=chosen,
                lam=float(best_lam), r2_oof=float(best_r2),
                res_hat_tr=res_hat_tr, res_hat_te=res_hat_te,
                prior_full=dict(a0=float(a0), b0=float(b0), ab=ab_full.reset_index().to_dict("list")),
                pred_te=pred_te_stack, r2_te=float(r2_score(test_df[TARGET_COL].values, pred_te_stack)))

def step_B3_specialist_blend(train_df, test_df, pred_te_in):
    """Top-N 대형쌍 전용 선형전문가와 보수적 블렌딩."""
    pair = _make_pair_str(train_df).astype(str)
    y = train_df[TARGET_COL].values; t = train_df[CUR_WARP].values
    b0 = np.cov(t, y, ddof=1)[0,1] / (np.var(t, ddof=1)+1e-8); a0 = y.mean() - b0*t.mean()

    g = pd.DataFrame({"pair":pair, "y":y, "t":t}).groupby("pair")
    stat=[]
    for k, sub in g:
        n=len(sub)
        if n < SPECIALIST_MIN_N: continue
        tt=sub["t"].values; yy=sub["y"].values
        b=np.cov(tt,yy,ddof=1)[0,1]/(np.var(tt,ddof=1)+1e-8); a=yy.mean()-b*tt.mean()
        w = n/(n+SPECIALIST_BLEND_N0); a_s=w*a+(1-w)*a0; b_s=w*b+(1-w)*b0
        stat.append((k,n,a_s,b_s))
    if not stat:
        return dict(pred_te=pred_te_in, r2_te=float(r2_score(test_df[TARGET_COL].values, pred_te_in)), used_pairs=[])

    coef = pd.DataFrame(stat, columns=["pair","n","a","b"]).set_index("pair")
    pred = pred_te_in.copy()
    pair_te = _make_pair_str(test_df).astype(str).values
    cur_te  = test_df[CUR_WARP].values

    used=[]
    for i,(p,cur) in enumerate(zip(pair_te, cur_te)):
        if p in coef.index:
            n,a,b = coef.loc[p,["n","a","b"]]
            gamma = n/(n+SPECIALIST_BLEND_N0)
            y_spec = float(a + b*cur)
            pred[i] = (1-gamma)*pred[i] + gamma*y_spec
            used.append(p)
    return dict(pred_te=pred, r2_te=float(r2_score(test_df[TARGET_COL].values, pred)), used_pairs=sorted(set(used)))

# ------------------------------- Roll-forward -----------------------------------
def _save_bundle(prefix, base_bundle, b1_model, b2_bundle):
    """Strict roll-forward용 번들 저장."""
    joblib.dump({
        "baseline": base_bundle,
        "iso": b1_model,
        "residual": dict(model=b2_bundle["model"], algo=b2_bundle["algo"], chosen=b2_bundle["chosen"], lam=b2_bundle["lam"]),
        "prior": b2_bundle["prior_full"],
        "seed": SEED
    }, MODELS_DIR / f"{RUN_NAME}__{prefix}_bundle.pkl")

def _predict_row_strict(row: pd.Series, bundle: dict):
    """1-행 예측(roll-forward). 이슈 #1/#2 대응: NaN 반환 방지, 실제 모델 추론."""
    # 1) baseline
    base = bundle["baseline"]; cols, med, chosen = base["cols"], base["med"], base["chosen"]
    X_row = pd.DataFrame([row.reindex(cols)], columns=cols).fillna(med)
    if base["selector"] == "selectk":
        # 동일 SelectKBest(훈련 시점)와 같은 열 순서 적용
        sel_cols = chosen
        X_sel = X_row[sel_cols].values
    else:
        X_sel = X_row.values
    yb = float(base["model"].predict(X_sel)[0])

    # 2) prior (pair-linear): a,b 사전 + EB
    prior = bundle["prior"]; a0, b0 = prior["a0"], prior["b0"]
    ab_map = dict(zip(prior["ab"]["pair"], zip(prior["ab"]["a"], prior["ab"]["b"])))
    p = f"{int(row[PASS_COL])}→{int(row[PASS_COL])+1}"
    t = float(row[CUR_WARP])
    if p in ab_map:
        a,b = ab_map[p]
        prior_pred = float(a + b*t)
    else:
        prior_pred = float(a0 + b0*t)

    # 3) residual-head 입력 구성
    Xr = X_row.copy()
    Xr["prior_pred"] = prior_pred
    if CUR_WARP in Xr.columns:
        Xr["cur_x_prior"] = Xr[CUR_WARP] * Xr["prior_pred"]

    # residual selector는 USE_SELECTK_Residual=False 설계 → 그대로 values
    Xr_val = Xr[Xr.columns].values
    yr_hat = float(bundle["residual"]["model"].predict(Xr_val)[0])
    y_stack = yb + bundle["residual"]["lam"] * yr_hat

    # 4) isotonic
    y_cal = float(bundle["iso"].predict([y_stack])[0])
    return y_cal

def strict_rollforward_r2(all_df: pd.DataFrame):
    """교집합 Test 판만 대상 roll-forward R². (#1: 반드시 호출, #2: NaN 방지 로직)"""
    e2x_bundle = MODELS_DIR / f"{RUN_NAME}__E2X_bundle.pkl"
    x2e_bundle = MODELS_DIR / f"{RUN_NAME}__X2E_bundle.pkl"
    if not (e2x_bundle.exists() and x2e_bundle.exists()):
        return float("nan")
    e2x = joblib.load(e2x_bundle); x2e = joblib.load(x2e_bundle)

    # 교집합 test plates
    e2x_test = set(joblib.load(MODELS_DIR / "e2x_test_plates_tplus1.pkl")) if (MODELS_DIR / "e2x_test_plates_tplus1.pkl").exists() else set()
    x2e_test = set(joblib.load(MODELS_DIR / "x2e_test_plates_tplus1.pkl")) if (MODELS_DIR / "x2e_test_plates_tplus1.pkl").exists() else set()
    inter = {str(p) for p in e2x_test}.intersection({str(p) for p in x2e_test})
    if not inter: 
        print("[Strict RF] 교집합 Test 판이 없습니다."); 
        return float("nan")

    rows=[]
    for plate, sub in all_df.groupby(PLATE_COL):
        if str(plate) not in inter: continue
        sub = sub.sort_values(PASS_COL)
        idx = sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index: continue
            row_t = idx.loc[p].drop(labels=[TARGET_COL], errors="ignore")
            bun = e2x if (p % 2 == 1) else x2e
            try:
                pred = _predict_row_strict(row_t, bun)
                gt_next = float(idx.loc[p+1][CUR_WARP])
                rows.append({"pred": pred, "gt": gt_next})
            except Exception:
                continue
    if not rows: 
        print("[Strict RF] 유효 행이 없습니다.")
        return float("nan")
    df = pd.DataFrame(rows)
    return float(r2_score(df["gt"], df["pred"]))

# --------------------------------- Runner --------------------------------------
def run_one_side(df, prefix):
    # 분할(003 저장 테스트판 재사용)
    tr, te = _split_by_saved_plates(df, prefix.lower())

    # S0
    s0 = step_S0_baseline(tr, te)

    # OOF baseline (B1/B2에 공통 사용)
    base_oof = _oof_preds_baseline(tr)

    # B1
    b1 = step_B1_isotonic(tr, te, base_oof, s0["pred_te"])

    # B2
    b2 = step_B2_residual_stack(tr, te, base_oof, s0["pred_te"])

    # B1 on B2 (일관성: Residual-Stack 이후에도 isotonic 적용 가능하나, 여기선 S0→B1과 병렬 보고)
    # 최종 조합은 B2 결과를 사용하고, 이후 B3에서 전문가 블렌딩
    # B3
    b3 = step_B3_specialist_blend(tr, te, b2["pred_te"])

    # 번들 저장(Strict RF용)
    if SAVE_BUNDLES_FOR_RF:
        _save_bundle(prefix, s0["bundle"], b1["model"], b2)

    # 요약 반환
    return dict(
        S0=s0, B1=b1, B2=b2, B3=b3,
        test_df=te
    )

def main():
    # 데이터 로드(003 산출물 재사용)
    e2x = pd.read_csv(PRO_DIR/"e2x_cleaned_tplus1.csv")
    x2e = pd.read_csv(PRO_DIR/"x2e_cleaned_tplus1.csv")
    allf = pd.read_csv(PRO_DIR/"all_featured_tplus1.csv") if (PRO_DIR/"all_featured_tplus1.csv").exists() else pd.concat([e2x, x2e], ignore_index=True)

    exp_id = _next_experiment_id()
    steps_csv = REPORTS_DIR / f"{exp_id}_{RUN_NAME}_steps.csv"
    summary_txt = SUM_DIR / f"{exp_id}_{RUN_NAME}_003_based_summary.txt"
    archived_py = SCRIPTS_DIR / f"{exp_id}_003_based_experiment.py"

    try:
        print("\n=== E2X ===")
        e2x_res = run_one_side(e2x, "E2X")
        print("\n=== X2E ===")
        x2e_res = run_one_side(x2e, "X2E")

        def _avg(a,b): return float(np.mean([a,b]))

        rows = [
            {"step":"S0 Baseline", 
             "E2X":e2x_res["S0"]["r2_te"], "X2E":x2e_res["S0"]["r2_te"], "Avg":_avg(e2x_res["S0"]["r2_te"], x2e_res["S0"]["r2_te"])},
            {"step":"B1 +Isotonic(OOF)", 
             "E2X":e2x_res["B1"]["r2_te"], "X2E":x2e_res["B1"]["r2_te"], "Avg":_avg(e2x_res["B1"]["r2_te"], x2e_res["B1"]["r2_te"])},
            {"step":"B2 +PairLinearPrior+ResidualStack", 
             "E2X":e2x_res["B2"]["r2_te"], "X2E":x2e_res["B2"]["r2_te"], "Avg":_avg(e2x_res["B2"]["r2_te"], x2e_res["B2"]["r2_te"])},
            {"step":"B3 +SpecialistBlend(top-N pair)", 
             "E2X":e2x_res["B3"]["r2_te"], "X2E":x2e_res["B3"]["r2_te"], "Avg":_avg(e2x_res["B3"]["r2_te"], x2e_res["B3"]["r2_te"])},
        ]
        pd.DataFrame(rows).to_csv(steps_csv, index=False)
        print(f"\n✓ 단계별 R² 저장: {steps_csv}")

        # Strict roll-forward
        rf_r2 = strict_rollforward_r2(allf) if DO_STRICT_ROLLFORWARD else float("nan")
        print(f"[Strict roll-forward R²] {rf_r2:.4f}" if not np.isnan(rf_r2) else "[Strict roll-forward] NaN")

        # Summary(.txt) 구성
        now = datetime.now()
        lines=[]
        lines.append(f"003-based R² Booster Summary ({now.strftime('%Y-%m-%d %H:%M:%S')})")
        lines.append("="*72)
        lines.append(f"실험 ID: {exp_id} | RUN_NAME: {RUN_NAME}\n")
        lines.append("[모델 성능 - R²]")
        lines.append(f"* E2X | S0: {e2x_res['S0']['r2_te']:.4f} | B1: {e2x_res['B1']['r2_te']:.4f} | B2: {e2x_res['B2']['r2_te']:.4f} | B3: {e2x_res['B3']['r2_te']:.4f}")
        lines.append(f"* X2E | S0: {x2e_res['S0']['r2_te']:.4f} | B1: {x2e_res['B1']['r2_te']:.4f} | B2: {x2e_res['B2']['r2_te']:.4f} | B3: {x2e_res['B3']['r2_te']:.4f}")
        avg_s0 = float(np.mean([e2x_res['S0']['r2_te'], x2e_res['S0']['r2_te']]))
        avg_b3 = float(np.mean([e2x_res['B3']['r2_te'], x2e_res['B3']['r2_te']]))
        lines.append(f"\n[평균 Test R²] S0={avg_s0:.4f} → B3={avg_b3:.4f}")
        if not np.isnan(rf_r2):
            lines.append(f"[Strict roll-forward R²(교집합 Test 판)] {rf_r2:.4f}")
        lines.append("\n[참고] 003 기준치: Avg Test R² 0.4174 / Strict RF R² 0.4963")  # 비교 기준
        summary_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"✓ 요약 저장: {summary_txt}")

        # 코드 스냅샷
        shutil.copy2(THIS_PATH, archived_py)
        print(f"✓ 코드 아카이브: {archived_py}")

    except Exception as e:
        print(f"[실패] {type(e).__name__}: {e}")
        tb = traceback.format_exc()
        (REPORTS_DIR/"last_error_traceback.txt").write_text(tb, encoding="utf-8")
        raise

if __name__ == "__main__":
    main()
