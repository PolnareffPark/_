# -*- coding: utf-8 -*-
"""
090_test.py  (leak-hardening, feature-flag ready, anchor split, metrics csv)

- E2X/X2E 분리, Group anchor split 고정(plate)
- 하드 누수 가드: 이름 기반 + 값 기반(훈련셋에서 y와 동일/거의 동일한 열 drop)
- 분할 이후에만 중앙값 대치/스케일/통계 산출
- 이상치: train quantile 경계로 train 필터, test는 clip만
- 기능 플래그 반영:
  * USE_DELTA_TARGET (DT): y := y - CUR_WARP (CUR_WARP는 필수 피처)
  * USE_PAIR_TE (TE): pair Δ 타깃인코딩(OOF, EB)
  * USE_HETERO_W (W): pair Δ 분산 기반 가중(OOF, shrinkage)
  * USE_MONO_CUR (MONO): XGB monotone(+1 on CUR_WARP)
  * USE_RM_TRAINONLY (RM): RM 통계는 train-plate로만 집계 → train/test에 매핑
- X2E 알고리즘: XGBoost/RandomForest 선택 (env X2E_ALGO), E2X는 XGBoost 고정
- Permutation leak audit: permuted R² ≈ 0 기대, 기록만 함
- Strict Test-only Rollforward R²: 요약 .txt에 반드시 기록
- 메트릭: reports/metrics/last_run_metrics.csv 매 실행 덮어씀
"""

import os, re, shutil, warnings, traceback, joblib
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------ PATHS / CONST ------------------------------
DATA_DIR = Path("data")
PRO_DIR  = DATA_DIR / "processed"; PRO_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SUM_DIR = REPORTS_DIR / "summaries"; SUM_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR = REPORTS_DIR / "scripts"; SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR = REPORTS_DIR / "metrics"; MET_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "warping_index_target"
CUR_WARP   = "warping_index_current_pass"
PLATE_COL  = "FM_날판번호"
PASS_COL   = "FM_PASS NO N"
MONTH_COL  = "FM_압연월"
EXCLUDE_BASE = {TARGET_COL, PLATE_COL, PASS_COL, MONTH_COL}

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
SELECT_K = 120

# ------------------------------ ENV FLAGS ----------------------------------
def _env_on(name: str, default="0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1","true","yes","y"}

RUN_NAME          = os.environ.get("RUN_NAME", "pass_t_plus_1_cross")
ANCHOR_TAG        = os.environ.get("ANCHOR_TAG", RUN_NAME)
PERMUTE_TARGET    = _env_on("PERMUTE_TARGET","0")
LEAK_AUTOCHECK    = _env_on("LEAK_AUTOCHECK","1")
USE_DELTA_TARGET  = _env_on("USE_DELTA_TARGET","0")
USE_PAIR_TE       = _env_on("USE_PAIR_TE","0")
USE_HETERO_W      = _env_on("USE_HETERO_W","0")
USE_MONO_CUR      = _env_on("USE_MONO_CUR","0")
USE_RM_TRAINONLY  = _env_on("USE_RM_TRAINONLY","0")
X2E_ALGO          = os.environ.get("X2E_ALGO","XGBoost")  # "XGBoost" | "RandomForest"

# ------------------------------ UTIL: SimpleKBest ---------------------------
class SimpleKBest:
    """훈련셋에서만 상관 기반 K-best. transform(X_df) -> np.ndarray"""
    def __init__(self, k: int, mandatory=None):
        self.k = int(max(1, k))
        self.mandatory = list(mandatory or [])
        self.selected_cols_: list[str] = []

    def fit(self, X_df: pd.DataFrame, y: np.ndarray):
        X = X_df.select_dtypes(include=[np.number]).copy()
        cols_all = X.columns.tolist()
        # 사전 정리: 비유한/상수 제거
        finite_mask = np.isfinite(X.values).all(axis=0)
        var = X.var(numeric_only=True).reindex(cols_all).fillna(0.0).values
        keep = finite_mask & (var > 1e-12)
        cols = [c for c,m in zip(cols_all, keep) if m]
        Xc = X[cols].fillna(X[cols].median(numeric_only=True))
        y0 = y - y.mean()
        ys = y0.std(); ys = ys if ys >= 1e-12 else 1e-12
        xm = Xc.values.mean(axis=0); xs = Xc.values.std(axis=0); xs[xs<1e-12]=1e-12
        xr = (Xc.values - xm) / xs;  yr = y0 / ys
        r = (xr.T @ yr) / (len(y) - 1)
        r2 = np.clip(r**2, 0.0, 1.0 - 1e-12)
        F = (r2/(1.0-r2)) * (len(y)-2)
        mand = [c for c in self.mandatory if c in cols]
        others = [c for c in cols if c not in mand]
        k_rem = max(1, min(self.k - len(mand), len(others)))
        order = np.argsort([F[others.index(c)] for c in others])[::-1] if others else []
        self.selected_cols_ = mand + [others[i] for i in order[:k_rem]]
        return self

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        D = X_df.copy()
        for c in self.selected_cols_:
            if c not in D.columns: D[c] = np.nan
        return D[self.selected_cols_].values

# ------------------------------ I/O & MERGE --------------------------------
def _next_experiment_id() -> str:
    nums=[]
    for f in SUM_DIR.glob("*.txt"):
        m=re.match(r"(\d{3})_", f.name)
        if m:
            try: nums.append(int(m.group(1)))
            except: pass
    return f"{(max(nums)+1) if nums else 1:03d}"

def load_and_merge_tplus1() -> pd.DataFrame:
    gt_entry = pd.read_csv(DATA_DIR/"entry_direction_results.csv")
    gt_exit  = pd.read_csv(DATA_DIR/"exit_direction_results.csv")
    gt = pd.concat([gt_entry, gt_exit], ignore_index=True)
    gt["extracted_plate"] = gt["filename"].str.extract(r"(PB\d+)")
    gt["extracted_pass"]  = gt["filename"].str.extract(r"_\d+_(\d+)").astype(int)
    fm = pd.read_csv(DATA_DIR/"posco2_105190.csv")
    merged = (pd.merge(gt, fm, left_on=["extracted_plate","extracted_pass"],
                       right_on=[PLATE_COL, PASS_COL], how="inner")
                .sort_values([PLATE_COL, PASS_COL]).reset_index(drop=True))
    # t+1 target
    merged[TARGET_COL] = merged.groupby(PLATE_COL)["warping_index"].shift(-1)
    merged = merged.rename(columns={"warping_index": CUR_WARP})
    merged = merged.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    # 불필요 제거
    merged = merged.drop(columns=["extracted_plate","extracted_pass","filename",
                                  "quality_grade","quality_grade_current_pass","direction"],
                         errors="ignore")
    # 저장
    out = PRO_DIR/"final_merged_data_regression_tplus1.csv"
    merged.to_csv(out, index=False)
    return merged

def split_e2x_x2e(df: pd.DataFrame):
    e2x = df[df[PASS_COL] % 2 == 1].copy()
    x2e = df[df[PASS_COL] % 2 == 0].copy()
    (PRO_DIR/"e2x_raw_tplus1.csv").write_text(e2x.to_csv(index=False))
    (PRO_DIR/"x2e_raw_tplus1.csv").write_text(x2e.to_csv(index=False))
    return e2x, x2e

# ------------------------------ FE & GUARDS --------------------------------
LEAKY_SUBSTR = ("target","_target","label","_y","y_","pred","_hat","oof","perm","fold","valid","test")

def _hard_drop_leaky_names(cols: list[str]) -> list[str]:
    keep=[]
    for c in cols:
        low=c.lower()
        if any(k in low for k in LEAKY_SUBSTR):
            continue
        keep.append(c)
    return keep

def _base_fe_train_test(tr: pd.DataFrame, te: pd.DataFrame):
    tr=tr.copy(); te=te.copy()
    fm_cols = [c for c in tr.columns if c.startswith("FM_") and c not in {PLATE_COL, PASS_COL, MONTH_COL}]
    def _fe(d: pd.DataFrame, pass_max: float):
        if fm_cols:
            blk=d[fm_cols]
            d["FM_mean"]=blk.mean(axis=1); d["FM_std"]=blk.std(axis=1)
            d["FM_max"]=blk.max(axis=1); d["FM_min"]=blk.min(axis=1)
            d["FM_range"]=d["FM_max"]-d["FM_min"]
            d["FM_cv"]=d["FM_std"]/(d["FM_mean"].abs()+1e-8)
        d["FM_PASS_squared"]=d[PASS_COL]**2
        d["FM_PASS_progress"]=d[PASS_COL]/(pass_max if pass_max>0 else 1.0)
        return d
    pass_max=float(tr[PASS_COL].max()) if len(tr) else 1.0
    tr=_fe(tr, pass_max); te=_fe(te, pass_max)
    return tr, te

def _ensure_X_from(df: pd.DataFrame) -> pd.DataFrame:
    X_all = df.drop(columns=[c for c in EXCLUDE_BASE if c in df.columns], errors="ignore")
    cols = _hard_drop_leaky_names(X_all.columns.tolist())
    X = X_all[cols].select_dtypes(include=[np.number]).copy()
    return X

def _assert_no_y_in_X_train(Xtr: pd.DataFrame, ytr: np.ndarray, label: str):
    # 값 상 누수 탐지(학습 전 강제): y와 완전히 같거나 거의 같은 열 제거
    to_drop=[]
    for c in Xtr.columns:
        xv=Xtr[c].values
        if len(xv)==len(ytr):
            if np.allclose(xv, ytr, atol=1e-10, rtol=1e-10):
                to_drop.append(c); continue
            # 상관 기반 차단
            sx=np.std(xv); sy=np.std(ytr)
            if sx>0 and sy>0:
                r=np.corrcoef(xv, ytr)[0,1]
                if np.isfinite(r) and abs(r)>=0.9999:
                    to_drop.append(c)
    if to_drop:
        Xtr.drop(columns=to_drop, inplace=True)
    # 마지막 방어선: 타깃 컬럼이 feature에 존재하면 즉시 오류
    assert TARGET_COL not in Xtr.columns, f"[{label}] feature에 {TARGET_COL} 존재(누수)"

def _train_outlier_bounds(train_df: pd.DataFrame, y: np.ndarray, qx=(0.01,0.99), qy=(0.005,0.995)):
    bounds={}
    X=_ensure_X_from(train_df).fillna(_ensure_X_from(train_df).median(numeric_only=True))
    for c in X.columns:
        lo,hi=np.quantile(X[c].values, qx)
        bounds[c]=(float(lo),float(hi))
    ylo,yhi=np.quantile(y, qy)
    return bounds,float(ylo),float(yhi)

def _apply_bounds(train_df: pd.DataFrame, test_df: pd.DataFrame, bounds: dict, y: np.ndarray, ylo: float, yhi: float):
    # train filter
    tr=train_df.copy(); te=test_df.copy()
    keep=(y>=ylo)&(y<=yhi)
    Xtr=_ensure_X_from(tr).fillna(_ensure_X_from(tr).median(numeric_only=True))
    for c,(lo,hi) in bounds.items():
        if c in Xtr.columns:
            keep &= (Xtr[c].values>=lo)&(Xtr[c].values<=hi)
    tr=tr.loc[keep].reset_index(drop=True); y_tr=y[keep]
    # test clip
    Xte=_ensure_X_from(te)
    for c,(lo,hi) in bounds.items():
        if c in Xte.columns:
            Xte[c]=Xte[c].clip(lo,hi)
    for c in Xte.columns:
        if c not in te.columns: te[c]=Xte[c]
        else: te[c]=Xte[c]
    return tr,y_tr,te

# ------------------------------ TE / W (OOF) -------------------------------
def _pair_str(df: pd.DataFrame) -> pd.Series:
    return df[PASS_COL].astype(int).astype(str) + "→" + (df[PASS_COL]+1).astype(int).astype(str)

def oof_pair_delta_te(train_df: pd.DataFrame, test_df: pd.DataFrame, n0=50):
    tr=train_df.copy(); te=test_df.copy()
    tr["pair"]=_pair_str(tr)
    tr["delta"]=tr[TARGET_COL]-tr[CUR_WARP]
    plates=tr[PLATE_COL].astype(str).values
    gkf=GroupKFold(n_splits=CV_FOLDS)
    oof=np.zeros(len(tr))
    for tr_i,va_i in gkf.split(tr, tr["delta"].values, groups=plates):
        sub=tr.iloc[tr_i]
        gmean=sub["delta"].mean()
        stat=sub.groupby("pair")["delta"].agg(n="size", mean="mean").reset_index()
        stat["te"]=(stat["n"]/(stat["n"]+n0))*stat["mean"] + (n0/(stat["n"]+n0))*gmean
        m=dict(zip(stat["pair"], stat["te"]))
        oof[va_i]=tr.iloc[va_i]["pair"].map(m).fillna(gmean).values
    # test mapping(full train)
    gmean=tr["delta"].mean()
    stat=tr.groupby("pair")["delta"].agg(n="size", mean="mean").reset_index()
    stat["te"]=(stat["n"]/(stat["n"]+n0))*stat["mean"] + (n0/(stat["n"]+n0))*gmean
    m_full=dict(zip(stat["pair"], stat["te"]))
    te_feat=_pair_str(te).map(m_full).fillna(gmean).values
    tr=tr.drop(columns=["pair","delta"], errors="ignore")
    te=te.copy()
    tr["pair_delta_te"]=oof; te["pair_delta_te"]=te_feat
    return tr, te

def oof_pair_std_weights(train_df: pd.DataFrame, test_df: pd.DataFrame, n0=50, clip=(0.25,3.0)):
    tr=train_df.copy(); te=test_df.copy()
    tr["pair"]=_pair_str(tr); tr["delta"]=tr[TARGET_COL]-tr[CUR_WARP]
    plates=tr[PLATE_COL].astype(str).values
    gkf=GroupKFold(n_splits=CV_FOLDS)
    oof_std=np.zeros(len(tr))
    for tr_i,va_i in gkf.split(tr, tr["delta"].values, groups=plates):
        sub=tr.iloc[tr_i]
        gs=sub["delta"].std(ddof=1)
        stat=sub.groupby("pair")["delta"].agg(n="size", std="std").reset_index()
        w=stat["n"]/(stat["n"]+n0)
        stat["std_sh"]=w*stat["std"].fillna(gs)+(1-w)*gs
        m=dict(zip(stat["pair"], stat["std_sh"]))
        oof_std[va_i]=tr.iloc[va_i]["pair"].map(m).fillna(gs).values
    gs=tr["delta"].std(ddof=1)
    w_tr=(float(gs)/(oof_std+1e-6))
    w_tr=np.clip(w_tr, clip[0], clip[1])
    return w_tr/np.mean(w_tr)

# ------------------------------ RM(train-only) ------------------------------
def attach_train_only_rm_stats(train_df: pd.DataFrame, test_df: pd.DataFrame):
    rm = pd.read_csv(DATA_DIR/"posco1_105190.csv")
    plates=set(train_df[PLATE_COL].astype(str).unique().tolist())
    rm = rm[rm["RM_날판번호"].astype(str).isin(plates)].copy()
    if rm.empty: return train_df, test_df
    num_cols=rm.select_dtypes(include=[np.number]).columns.tolist()
    drop_like={"RM_날판번호","RM_압연Pass번호","warping_index"}
    stat_cols=[c for c in num_cols if c not in drop_like and not c.endswith("_target")]
    agg=rm.groupby("RM_날판번호")[stat_cols].agg(["mean","max","min","std","median"]).round(6)
    agg.columns=[f"RM_{c[0]}_{c[1]}" for c in agg.columns]
    agg=agg.reset_index()
    tr=pd.merge(train_df, agg, left_on=PLATE_COL, right_on="RM_날판번호", how="left").drop(columns=["RM_날판번호"], errors="ignore")
    te=pd.merge(test_df, agg, left_on=PLATE_COL, right_on="RM_날판번호",  how="left").drop(columns=["RM_날판번호"], errors="ignore")
    if not agg.empty:
        med=agg.drop(columns=["RM_날판번호"], errors="ignore").median(numeric_only=True)
        for c in med.index:
            if c in tr.columns: tr[c]=tr[c].fillna(float(med[c]))
            if c in te.columns: te[c]=te[c].fillna(float(med[c]))
    return tr, te

# ------------------------------ MODEL BUILD --------------------------------
def _build_model(name: str, n_features: int, mono_index: int | None):
    if name=="XGBoost":
        params=dict(
            objective="reg:squarederror", random_state=SEED, tree_method="hist",
            n_estimators=800, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
            n_jobs=0, eval_metric="rmse",
        )
        if mono_index is not None and 0<=mono_index<n_features:
            v=[0]*n_features; v[mono_index]=1
            params["monotone_constraints"]=tuple(v)
        return XGBRegressor(**params)
    elif name=="RandomForest":
        return RandomForestRegressor(random_state=SEED, n_estimators=600, n_jobs=-1,
                                     max_depth=20, min_samples_leaf=5, max_features="sqrt")
    else:
        raise ValueError(f"Unknown algo: {name}")

def _algo_for(direction: str) -> str:
    return "XGBoost" if direction=="E2X" else (X2E_ALGO if X2E_ALGO in {"XGBoost","RandomForest"} else "XGBoost")

# ------------------------------ TRAIN ONE SIDE ------------------------------
def train_one_side(raw_df: pd.DataFrame, direction: str, anchor_tag: str):
    # 1) 방향 필터
    df = raw_df[raw_df[PASS_COL] % 2 == (1 if direction=="E2X" else 0)].copy()
    prefix = "e2x" if direction=="E2X" else "x2e"

    # 2) anchor split (plate group)
    anchor_file = MODELS_DIR / f"{prefix}__{anchor_tag}_test_plates.pkl"
    X_all = _ensure_X_from(df)
    groups = df[PLATE_COL].astype(str).values
    y_all = df[TARGET_COL].values

    if anchor_file.exists():
        test_plates=set(joblib.load(anchor_file))
        tr_mask=~df[PLATE_COL].astype(str).isin(test_plates)
        te_mask= df[PLATE_COL].astype(str).isin(test_plates)
        tr_idx=np.where(tr_mask.values)[0]; te_idx=np.where(te_mask.values)[0]
    else:
        gss=GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        tr_idx, te_idx = next(gss.split(X_all, y_all, groups))
        test_plates=sorted(df.iloc[te_idx][PLATE_COL].astype(str).unique())
        joblib.dump(test_plates, anchor_file)

    tr_raw=df.iloc[tr_idx].copy(); te_raw=df.iloc[te_idx].copy()

    # 3) RM train-only (옵션)
    if USE_RM_TRAINONLY:
        tr_raw, te_raw = attach_train_only_rm_stats(tr_raw, te_raw)

    # 4) FE (train 기준 스케일)
    tr_fe, te_fe = _base_fe_train_test(tr_raw, te_raw)

    # 5) TE/W (OOF 안전)
    if USE_PAIR_TE:
        tr_fe, te_fe = oof_pair_delta_te(tr_fe, te_fe)
    sample_weight=None
    if USE_HETERO_W:
        sw = oof_pair_std_weights(tr_fe, te_fe)
        sample_weight = sw / (np.mean(sw)+1e-8)

    # 6) y 정의 (Δ‑타깃 옵션)
    if USE_DELTA_TARGET:
        tr_fe["_y"] = tr_fe[TARGET_COL] - tr_fe[CUR_WARP]
        te_fe["_y"] = te_fe[TARGET_COL] - te_fe[CUR_WARP]
        mandatory=[CUR_WARP] + (["pair_delta_te"] if USE_PAIR_TE else [])
    else:
        tr_fe["_y"] = tr_fe[TARGET_COL]
        te_fe["_y"] = te_fe[TARGET_COL]
        mandatory=[CUR_WARP] + (["pair_delta_te"] if USE_PAIR_TE else [])

    # 7) 이상치 경계(train) → train filter + test clip
    bounds, ylo, yhi = _train_outlier_bounds(tr_fe, tr_fe["_y"].values)
    tr_filt, y_tr, te_clip = _apply_bounds(tr_fe, te_fe, bounds, tr_fe["_y"].values, ylo, yhi)

    # 8) X 구성 + 누수 하드 가드
    Xtr_df = _ensure_X_from(tr_filt).fillna(_ensure_X_from(tr_filt).median(numeric_only=True))
    Xte_df = _ensure_X_from(te_clip).reindex(columns=Xtr_df.columns, fill_value=np.nan)
    Xte_df = Xte_df.fillna(Xtr_df.median(numeric_only=True))

    _assert_no_y_in_X_train(Xtr_df.copy(), y_tr, direction)

    # 9) 선택(훈련셋만) + 모델
    chosen = SimpleKBest(k=min(SELECT_K, Xtr_df.shape[1]), mandatory=mandatory).fit(Xtr_df, y_tr).selected_cols_
    Xtr = Xtr_df[chosen].values
    Xte = Xte_df[chosen].values
    mono_idx = (chosen.index(CUR_WARP) if (USE_MONO_CUR and CUR_WARP in chosen and _algo_for(direction)=="XGBoost") else None)
    model = _build_model(_algo_for(direction), n_features=len(chosen), mono_index=mono_idx)

    # 10) 내부 홀드아웃(plate-group)으로 조기종료 호환
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    grp_tr = tr_filt[PLATE_COL].astype(str).values
    tr_i, va_i = next(gss.split(Xtr, y_tr, groups=grp_tr))
    Xfit, yfit = Xtr[tr_i], y_tr[tr_i]
    Xval, yval = Xtr[va_i], y_tr[va_i]
    sw_fit = None if sample_weight is None else sample_weight[tr_i]

    if isinstance(model, XGBRegressor):
        try:
            model.fit(Xfit, yfit, sample_weight=sw_fit, eval_set=[(Xval, yval)],
                      early_stopping_rounds=50, verbose=False)
        except TypeError:
            model.fit(Xfit, yfit, sample_weight=sw_fit)
    else:
        model.fit(Xfit, yfit, sample_weight=sw_fit)

    # 11) 예측/복원
    y_tr_pred = model.predict(Xtr)
    y_te_pred = model.predict(Xte)
    if USE_DELTA_TARGET:
        # Δ-타깃이면 te 측은 CUR_WARP + Δ̂로 복원
        y_te_pred = te_clip[CUR_WARP].values + y_te_pred
        y_tr_pred = tr_filt[CUR_WARP].values + y_tr_pred

    r2_tr = float(r2_score(tr_filt[TARGET_COL].values, y_tr_pred))
    r2_te = float(r2_score(te_clip[TARGET_COL].values, y_te_pred))
    rmse_tr = float(np.sqrt(mean_squared_error(tr_filt[TARGET_COL].values, y_tr_pred)))
    rmse_te = float(np.sqrt(mean_squared_error(te_clip[TARGET_COL].values, y_te_pred)))

    # 저장(모델/선택열/중앙값/테스트판)
    joblib.dump(model,     MODELS_DIR/f"{prefix}_regressor_tplus1.pkl")
    joblib.dump(chosen,    MODELS_DIR/f"{prefix}_feature_cols_tplus1.pkl")
    joblib.dump(Xtr_df.median(numeric_only=True).to_dict(), MODELS_DIR/f"{prefix}_feature_medians_tplus1.pkl")
    joblib.dump(sorted(list(test_plates)), MODELS_DIR/f"{prefix}_test_plates_tplus1.pkl")

    return {
        "direction": direction, "algo": _algo_for(direction),
        "r2_tr": r2_tr, "r2_te": r2_te, "rmse_tr": rmse_tr, "rmse_te": rmse_te,
        "yte": te_clip[TARGET_COL].values, "yhat": y_te_pred,
        "test_plates": set(test_plates),
    }

# ------------------------------ PERM AUDIT ----------------------------------
def _quick_perm_r2(train_df: pd.DataFrame, test_df: pd.DataFrame, direction: str) -> float:
    # 매우 가벼운 누수 감사(훈련 y 섞기)
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df[df[PASS_COL] % 2 == (1 if direction=="E2X" else 0)].copy()
    gss=GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    Xall=_ensure_X_from(df).fillna(_ensure_X_from(df).median(numeric_only=True))
    y=df[TARGET_COL].values.copy()
    groups=df[PLATE_COL].astype(str).values
    tr_i, te_i = next(gss.split(Xall, y, groups))
    rng=np.random.RandomState(SEED+7)
    y_perm=y.copy(); rng.shuffle(y_perm[tr_i])
    # 간단 XGB
    m=XGBRegressor(objective="reg:squarederror", random_state=SEED, tree_method="hist",
                   n_estimators=200, learning_rate=0.08, max_depth=4,
                   subsample=0.8, colsample_bytree=0.8, n_jobs=0, eval_metric="rmse")
    m.fit(Xall.iloc[tr_i].values, y_perm[tr_i])
    yhat=m.predict(Xall.iloc[te_i].values)
    return float(r2_score(y[te_i], yhat))

# ------------------------------ ROLLFORWARD ---------------------------------
def _load_bundle(prefix: str):
    return {
        "model": joblib.load(MODELS_DIR/f"{prefix}_regressor_tplus1.pkl"),
        "cols":  joblib.load(MODELS_DIR/f"{prefix}_feature_cols_tplus1.pkl"),
        "med":   joblib.load(MODELS_DIR/f"{prefix}_feature_medians_tplus1.pkl"),
        "plates":set(joblib.load(MODELS_DIR/f"{prefix}_test_plates_tplus1.pkl")),
    }

def _predict_row(df_row: pd.Series, bundle: dict) -> float:
    X_row = pd.DataFrame([df_row.reindex(bundle["cols"])], columns=bundle["cols"])
    med=pd.Series(bundle["med"])
    X_row=X_row.fillna(med)
    return float(bundle["model"].predict(X_row.values)[0])

def strict_rollforward_r2(final_df: pd.DataFrame) -> float:
    e2x=_load_bundle("e2x"); x2e=_load_bundle("x2e")
    plates = e2x["plates"] & x2e["plates"]
    if not plates: return float("nan")
    rows=[]
    for plate, sub in final_df[final_df[PLATE_COL].astype(str).isin(plates)].groupby(PLATE_COL):
        sub=sub.sort_values(PASS_COL)
        idx=sub.set_index(PASS_COL, drop=False)
        for p in idx.index:
            if (p+1) not in idx.index: continue
            row_t = idx.loc[p].drop(labels=[TARGET_COL], errors="ignore")
            bundle = e2x if (p%2==1) else x2e
            pred=_predict_row(row_t, bundle)
            gt = float(idx.loc[p+1].get(CUR_WARP))
            rows.append((gt,pred))
    if not rows: return float("nan")
    gt = np.array([g for g,_ in rows]); pr=np.array([p for _,p in rows])
    return float(r2_score(gt, pr))

# ------------------------------ SUMMARY / METRICS ---------------------------
def write_summary(experiment_id: str, e2x_res: dict, x2e_res: dict,
                  avg_r2: float, rf_r2: float, strict_rf_r2: float,
                  perm_e2x: float | None, perm_x2e: float | None, summary_path: Path):
    lines=[]
    lines.append("Pass t → Pass t+1 (E→X / X→E) Cross-Transition Regression Summary")
    lines.append("="*72)
    lines.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 작성")
    lines.append(f"실험 ID: {experiment_id}\n")
    lines.append("[모델 성능 - R2]")
    lines.append(f"* E2X | Algo: {e2x_res['algo']} | Train R2: {e2x_res['r2_tr']:.4f} | Test R2: {e2x_res['r2_te']:.4f} (RMSE {e2x_res['rmse_tr']:.4f}/{e2x_res['rmse_te']:.4f})")
    lines.append(f"* X2E | Algo: {x2e_res['algo']} | Train R2: {x2e_res['r2_tr']:.4f} | Test R2: {x2e_res['r2_te']:.4f} (RMSE {x2e_res['rmse_tr']:.4f}/{x2e_res['rmse_te']:.4f})")
    lines.append(f"\n[평균 Test R2] {avg_r2:.4f}")
    lines.append(f"[Strict Test‑only Rollforward R²] {strict_rf_r2:.4f}")
    if LEAK_AUTOCHECK:
        lines.append("\n[Leak Audit — quick permute R² (expect ≈ 0)]")
        lines.append(f"- E2X permuted R²: {perm_e2x:.4f}")
        lines.append(f"- X2E permuted R²: {perm_x2e:.4f}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # metrics csv (덮어쓰기)
    met_path = MET_DIR / "last_run_metrics.csv"
    met_cols = ["timestamp","experiment_id","e2x_algo","x2e_algo",
                "e2x_train_r2","e2x_test_r2","x2e_train_r2","x2e_test_r2",
                "avg_test_r2","summary_path","rollforward_r2","strict_rollforward_r2",
                "perm_e2x","perm_x2e"]
    met_row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), experiment_id,
               e2x_res["algo"], x2e_res["algo"],
               f"{e2x_res['r2_tr']:.6f}", f"{e2x_res['r2_te']:.6f}",
               f"{x2e_res['r2_tr']:.6f}", f"{x2e_res['r2_te']:.6f}",
               f"{avg_r2:.6f}", str(summary_path), f"{rf_r2:.6f}", f"{strict_rf_r2:.6f}",
               (f"{perm_e2x:.6f}" if perm_e2x is not None else ""), (f"{perm_x2e:.6f}" if perm_x2e is not None else "")]
    pd.DataFrame([met_row], columns=met_cols).to_csv(met_path, index=False)

# ------------------------------ MAIN ---------------------------------------
def main():
    experiment_id = _next_experiment_id()
    final_df = load_and_merge_tplus1()

    if PERMUTE_TARGET:
        rng=np.random.RandomState(SEED)
        final_df[TARGET_COL] = rng.permutation(final_df[TARGET_COL].values)
        print("[PERMUTE] Target shuffled for leak sanity check.")

    e2x_raw, x2e_raw = split_e2x_x2e(final_df)

    # 학습
    e2x_res = train_one_side(final_df, "E2X", anchor_tag=ANCHOR_TAG)
    x2e_res = train_one_side(final_df, "X2E", anchor_tag=ANCHOR_TAG)

    avg_r2 = (e2x_res["r2_te"] + x2e_res["r2_te"]) / 2.0

    # 롤포워드
    rf_r2 = float("nan")  # (옵션) 전체 판 RF는 과대평가 소지 → 생략/보류
    strict_rf = strict_rollforward_r2(final_df)

    # 퍼뮤 감사
    perm_e2x = _quick_perm_r2(e2x_raw, x2e_raw, "E2X") if LEAK_AUTOCHECK else None
    perm_x2e = _quick_perm_r2(e2x_raw, x2e_raw, "X2E") if LEAK_AUTOCHECK else None

    # 요약/메트릭 저장
    summary_path = SUM_DIR / f"{experiment_id}_{RUN_NAME}_summary.txt"
    write_summary(experiment_id, e2x_res, x2e_res, avg_r2, rf_r2, strict_rf, perm_e2x, perm_x2e, summary_path)

    # 스크립트 아카이브
    try:
        src = Path(__file__).resolve()
        shutil.copy2(src, SCRIPTS_DIR/f"{experiment_id}_{src.name}")
    except Exception:
        pass

    print("\n" + "="*72)
    print(f"✓ 평균 Test R²: {avg_r2:.4f} | Strict RF R²: {strict_rf:.4f}")
    if LEAK_AUTOCHECK:
        print(f"✓ Perm audit: E2X {perm_e2x:.4f}, X2E {perm_x2e:.4f}")
    print(f"✓ 요약 저장: {summary_path}")
    print("="*72)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        (REPORTS_DIR/"last_error_traceback.txt").write_text(tb, encoding="utf-8")
        print(f"[실패] {type(e).__name__}: {e}")
        print(tb)
