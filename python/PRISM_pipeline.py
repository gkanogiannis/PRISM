#!/usr/bin/env python3
"""
PRISM_pipeline.py
NIRS Calibration Pipeline

All data-loading, preprocessing, modelling, and plotting functions.
Used as a module by PRISM.ipynb and as a standalone script.

Script usage:
    python python/PRISM_pipeline.py               # full run, all methods
    python python/PRISM_pipeline.py --quick       # skip CNN, RF, XGB
    python python/PRISM_pipeline.py --crops barley grasspea
    python python/PRISM_pipeline.py --out my_results/
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Python 3.12+ made warning filters thread-local, so main-thread filters do not
# suppress warnings raised inside joblib worker threads. Patch Parallel.__call__
# directly so suppression applies at the call site regardless of thread context.
try:
    import sklearn.utils.parallel as _skp
    _orig_parallel_call = _skp.Parallel.__call__

    def _parallel_call_quiet(self, iterable):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return _orig_parallel_call(self, iterable)

    _skp.Parallel.__call__ = _parallel_call_quiet
except Exception:
    pass

# Propagate suppression to spawned worker processes via the environment.
_pw = os.environ.get('PYTHONWARNINGS', '')
for _w in ('ignore::UserWarning', 'ignore::FutureWarning'):
    if _w not in _pw:
        _pw = f'{_pw},{_w}'.strip(',')
os.environ['PYTHONWARNINGS'] = _pw

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.decomposition import PCA
from sklearn.model_selection import (KFold, GridSearchCV, cross_val_predict,
                                      train_test_split)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

import xgboost as xgb
import lightgbm as lgb

# ── Optional dependencies ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    HAS_TORCH = False
    DEVICE = None

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# GPU availability flags that are checked once at import, used by XGBoost/LightGBM/CNN
_CUDA = HAS_TORCH and torch.cuda.is_available()
_WSL  = os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()
_XGB_DEVICE  = 'cuda' if _CUDA else 'cpu'          # XGBoost ≥ 2.0 'device' kwarg
_LGBM_DEVICE = 'cpu' if _WSL else ('gpu' if _CUDA else 'cpu')  # LightGBM: no OpenCL in WSL2

# ── Paths ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))

# ── Config-driven globals (populated by apply_config / load_config) ────────
# These start with safe defaults and are overridden by the config file.

SEED    = 42
N_FOLDS = 10
CV      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

CROP_CONFIG:           dict = {}   # crop_name → {'file': path}
AVAILABLE_TRAITS:      dict = {}   # crop_name → [trait_col, ...]  (auto-discovered)
TRAIT_INCLUDE:         dict = {}   # crop_name → [trait_col, ...]  (whitelist from config)
TRAIT_LABELS:          dict = {}   # col_name  → display label
DEFAULT_METHODS_FULL   = ['pls', 'lasso', 'enet', 'svr', 'rf', 'xgb', 'lgbm']
DEFAULT_METHODS_QUICK  = ['pls', 'lasso', 'enet', 'svr', 'lgbm']
DEFAULT_PREPROCS       = ['snv', 'msc', 'sg1', 'sg2']
MODEL_PARAMS:          dict = {}   # method → {param: value}; populated from config


import contextlib

@contextlib.contextmanager
def _quiet():
    """Suppress all warnings inside a block — robust across Python versions."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def _mp(method: str, key: str, default):
    """Return MODEL_PARAMS[method][key] if set, else default."""
    return MODEL_PARAMS.get(method, {}).get(key, default)


def _to_list(v):
    """Wrap scalar in list; leave lists unchanged (for GridSearchCV grids)."""
    return v if isinstance(v, list) else [v]

_DEFAULT_CONFIG_PATH   = os.path.join(BASE_DIR, 'PRISM_config.yaml')


# ── Config loading ─────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    """Parse a YAML config file and return the dict."""
    try:
        import yaml
    except ImportError:
        raise ImportError('pyyaml is required: pip install pyyaml')
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_config(cfg: dict) -> None:
    """Apply a config dict to all pipeline globals.

    Crops, trait labels, and pipeline defaults are all driven from here.
    Called automatically when the pipeline starts; can also be called from
    the notebook after importing the module.
    """
    global SEED, N_FOLDS, CV
    global CROP_CONFIG, TRAIT_LABELS
    global DEFAULT_METHODS_FULL, DEFAULT_METHODS_QUICK, DEFAULT_PREPROCS
    global MODEL_PARAMS

    pip = cfg.get('pipeline', {})
    SEED    = pip.get('seed',    SEED)
    N_FOLDS = pip.get('n_folds', N_FOLDS)
    CV      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    if 'full_methods'  in pip: DEFAULT_METHODS_FULL  = pip['full_methods']
    if 'quick_methods' in pip: DEFAULT_METHODS_QUICK = pip['quick_methods']
    if 'preprocs'      in pip: DEFAULT_PREPROCS      = pip['preprocs']

    if 'crops' in cfg:
        CROP_CONFIG.clear()
        for name, crop_cfg in cfg['crops'].items():
            path = crop_cfg['file']
            if not os.path.isabs(path):
                path = os.path.join(BASE_DIR, path)
            CROP_CONFIG[name] = {'file': path}

    if 'trait_labels' in cfg:
        TRAIT_LABELS.update(cfg['trait_labels'])

    # Optional per-crop trait whitelists
    if 'trait_include' in cfg:
        for crop, traits in cfg['trait_include'].items():
            TRAIT_INCLUDE[crop] = traits

    if 'model_params' in cfg:
        MODEL_PARAMS.clear()
        MODEL_PARAMS.update(cfg['model_params'] or {})


def _auto_discover_crops(csv_dir: str) -> None:
    """Populate CROP_CONFIG from *.nirs.csv files found in csv_dir."""
    if not os.path.isdir(csv_dir):
        return
    for fname in sorted(os.listdir(csv_dir)):
        if fname.endswith('.nirs.csv'):
            crop = fname[:-9]
            CROP_CONFIG[crop] = {'file': os.path.join(csv_dir, fname)}
    if CROP_CONFIG:
        print(f'Auto-discovered {len(CROP_CONFIG)} crop(s) from {csv_dir}')


# Load default config at import time so notebooks see populated globals immediately
if os.path.exists(_DEFAULT_CONFIG_PATH):
    apply_config(load_config(_DEFAULT_CONFIG_PATH))
else:
    _auto_discover_crops(os.path.join(BASE_DIR, 'data', 'csv'))

# Module-level data cache; populated by load_all_crops()
# Using a mutable dict so that `from nirs_pipeline import crops_data` keeps
# a live reference even after load_all_crops() fills it.
crops_data: dict = {}


# ── Data loading ───────────────────────────────────────────────────────────
def load_nirs_data(crop_name: str):
    """Return (traits_df, spectra_ndarray, wavelengths_array) for one crop.

    Reads the NIRS CSV format:
      - Non-numeric column headers → trait / metadata columns
      - Numeric column headers     → spectral data (wavelengths in nm)

    Also populates AVAILABLE_TRAITS[crop_name] with columns that have
    numeric values and at least 10 non-NaN rows.
    """
    path = CROP_CONFIG[crop_name]['file']
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'CSV not found for {crop_name!r}: {path}\n'
            f'Run:  python python/convert_xlsx_to_nirs_csv.py'
        )

    raw    = pd.read_csv(path)
    wl_raw = pd.to_numeric(pd.Series(raw.columns, dtype=object), errors='coerce')
    is_spec = ~wl_raw.isna()

    wl        = wl_raw[is_spec].values.astype(float)
    spec      = raw.loc[:, is_spec.values].values.astype(float)
    traits_df = raw.loc[:, ~is_spec.values].copy()

    # Auto-discover usable trait columns
    _id_cols = {'sample_id', 'sample_number', 'position', 'id', 'part',
                'id_lfm', 'tax_name', 'blank_col'}
    traits = [
        c for c in traits_df.columns
        if c.lower() not in _id_cols
        and pd.to_numeric(traits_df[c], errors='coerce').notna().sum() >= 10
    ]
    if crop_name in TRAIT_INCLUDE:
        whitelist = TRAIT_INCLUDE[crop_name]
        unknown_wl = [t for t in whitelist if t not in traits_df.columns]
        if unknown_wl:
            raise ValueError(
                f'trait_include for {crop_name!r} lists unknown column(s): {unknown_wl}\n'
                f'Available columns: {list(traits_df.columns)}'
            )
        traits = [t for t in traits if t in whitelist]
    AVAILABLE_TRAITS[crop_name] = traits

    print(f'Loaded {crop_name}: {spec.shape[0]} samples, '
          f'{spec.shape[1]} wavelengths ({wl.min():.0f}–{wl.max():.0f} nm), '
          f'{len(traits)} traits')
    return traits_df, spec, wl


def load_all_crops(crop_list=None):
    """Populate the module-level crops_data cache and return it."""
    crops_data.clear()
    crops_data.update(
        {c: load_nirs_data(c) for c in (crop_list or list(CROP_CONFIG))}
    )
    return crops_data


# ── Spectral preprocessing ─────────────────────────────────────────────────
def _msc(X, ref):
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        c, *_ = np.linalg.lstsq(np.c_[np.ones(len(ref)), ref], X[i], rcond=None)
        out[i] = (X[i] - c[0]) / c[1]
    return out


def preprocess_spectra(X, method: str, ref=None):
    """
    Returns (X_pp, ref_out, wl_trim).
    wl_trim: wavelengths trimmed from each side (SG only).
    """
    SG_W, SG_P = 11, 2
    trim = (SG_W - 1) // 2

    if method == 'raw':
        return X.copy(), None, 0
    if method == 'snv':
        mu  = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1,  keepdims=True)
        std[std == 0] = 1
        return (X - mu) / std, None, 0
    if method == 'msc':
        if ref is None:
            ref = X.mean(axis=0)
        return _msc(X, ref), ref, 0
    if method == 'sg1':
        return savgol_filter(X, SG_W, SG_P, deriv=1, axis=1), None, trim
    if method == 'sg2':
        return savgol_filter(X, SG_W, SG_P, deriv=2, axis=1), None, trim
    raise ValueError(f'Unknown preprocessing method: {method!r}')


# ── Metrics ────────────────────────────────────────────────────────────────
def _rpd(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(y_true.std() / rmse) if rmse > 0 else np.nan


def _metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse, _rpd(y_true, y_pred)


# ── Model helpers ──────────────────────────────────────────────────────────
def _fit_pls(X_tr, y_tr, max_comp=None):
    max_comp = min(_mp('pls', 'max_comp', 15) if max_comp is None else max_comp,
                   X_tr.shape[1], X_tr.shape[0] - 1)
    best_rmse, best_nc, best_preds = np.inf, 1, None
    for nc in range(1, max_comp + 1):
        preds = cross_val_predict(
            PLSRegression(n_components=nc, scale=True), X_tr, y_tr, cv=CV
        ).ravel()
        rmse = np.sqrt(mean_squared_error(y_tr, preds))
        if rmse < best_rmse:
            best_rmse, best_nc, best_preds = rmse, nc, preds
    cv_r2 = r2_score(y_tr, best_preds)
    model = PLSRegression(n_components=best_nc, scale=True).fit(X_tr, y_tr)
    return model, cv_r2, best_rmse


def _make_grid_estimator(method: str):
    lam = _mp(method, 'alpha',    np.logspace(-4, 0, 25).tolist())
    C   = _mp(method, 'C',        np.logspace(-2, 2, 10).tolist())
    if method == 'lasso':
        return (Lasso(max_iter=_mp('lasso', 'max_iter', 10000)),
                {'alpha': _to_list(lam)})
    if method == 'enet':
        return (ElasticNet(max_iter=_mp('enet', 'max_iter', 10000)),
                {'alpha':    _to_list(lam),
                 'l1_ratio': _to_list(_mp('enet', 'l1_ratio', [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]))})
    if method == 'svr':
        return (Pipeline([('sc', StandardScaler()),
                          ('svr', LinearSVR(max_iter=_mp('svr', 'max_iter', 5000)))]),
                {'svr__C': _to_list(C)})
    if method == 'rf':
        return (RandomForestRegressor(
                    n_estimators=_mp('rf', 'n_estimators', 300),
                    random_state=SEED, n_jobs=_mp('rf', 'n_jobs', 1)),
                {'max_features': _to_list(_mp('rf', 'max_features', ['sqrt', 0.1, 0.3]))})
    if method == 'xgb':
        return (xgb.XGBRegressor(n_jobs=1, random_state=SEED, verbosity=0,
                                  device=_XGB_DEVICE),
                {'n_estimators':  _to_list(_mp('xgb', 'n_estimators',  [100, 200])),
                 'max_depth':     _to_list(_mp('xgb', 'max_depth',     [3, 5])),
                 'learning_rate': _to_list(_mp('xgb', 'learning_rate', [0.05, 0.10]))})
    if method == 'lgbm':
        return (lgb.LGBMRegressor(n_jobs=1, random_state=SEED, verbose=-1,
                                   device=_LGBM_DEVICE),
                {'n_estimators':  _to_list(_mp('lgbm', 'n_estimators',  [100, 200])),
                 'max_depth':     _to_list(_mp('lgbm', 'max_depth',     [3, 5])),
                 'learning_rate': _to_list(_mp('lgbm', 'learning_rate', [0.05, 0.10]))})
    raise ValueError(f'Unknown method: {method!r}')


def run_analysis(crop_name, trait_name, method, prep_method,
                 test_size=0.2, seed=SEED, verbose=False):
    """Run one crop × trait × method × preprocessing combination."""
    traits_df, spectra, _ = crops_data[crop_name]

    if trait_name not in traits_df.columns:
        return None
    y  = pd.to_numeric(traits_df[trait_name], errors='coerce').values
    ok = ~np.isnan(y)
    if ok.sum() < 10:
        return None

    X_ok, y_ok = spectra[ok], y[ok]
    X_tr, X_te, y_tr, y_te = train_test_split(X_ok, y_ok,
                                               test_size=test_size, random_state=seed)
    X_tr_pp, ref, _ = preprocess_spectra(X_tr, prep_method)
    X_te_pp, _,  _  = preprocess_spectra(X_te, prep_method, ref=ref)

    try:
        with _quiet():
            if method == 'pls':
                model, r2_cv, rmse_cv = _fit_pls(X_tr_pp, y_tr)
                if verbose:
                    print(f'    params: n_components={model.n_components}')
            else:
                est, grid = _make_grid_estimator(method)
                gs = GridSearchCV(est, grid, cv=CV,
                                  scoring='neg_root_mean_squared_error',
                                  n_jobs=-1, refit=True)
                gs.fit(X_tr_pp, y_tr)
                model   = gs.best_estimator_
                cv_p    = cross_val_predict(model, X_tr_pp, y_tr, cv=CV)
                r2_cv   = r2_score(y_tr, cv_p)
                rmse_cv = np.sqrt(mean_squared_error(y_tr, cv_p))
                if verbose:
                    print(f'    params: {gs.best_params_}')
    except Exception as e:
        print(f'  [error] {crop_name}/{trait_name}/{method}/{prep_method}: {e}')
        return None

    cal_p = model.predict(X_tr_pp)
    val_p = model.predict(X_te_pp)
    if hasattr(cal_p, 'ravel'): cal_p = cal_p.ravel()
    if hasattr(val_p, 'ravel'): val_p = val_p.ravel()

    r2_cal, rmse_cal, _       = _metrics(y_tr, cal_p)
    r2_val, rmse_val, rpd_val = _metrics(y_te, val_p)

    return {
        'Crop': crop_name, 'Trait': trait_name,
        'Method': method,  'Preprocessing': prep_method,
        'N_train': len(y_tr), 'N_test': len(y_te),
        'R2_cal':  round(r2_cal,  4), 'RMSE_cal': round(rmse_cal, 4),
        'R2_cv':   round(r2_cv,   4), 'RMSE_cv':  round(rmse_cv,  4),
        'R2_val':  round(r2_val,  4), 'RMSE_val': round(rmse_val, 4),
        'RPD_val': round(rpd_val, 3),
    }


# ── Stacking ensemble ──────────────────────────────────────────────────────
def _make_stack():
    sp = MODEL_PARAMS.get('stack', {})
    return StackingRegressor(
        estimators=[
            ('pls',  PLSRegression(
                        n_components=sp.get('pls_n_components', 5), scale=True)),
            ('lasso', Lasso(
                        alpha=sp.get('lasso_alpha', 0.01),
                        max_iter=sp.get('lasso_max_iter', 10000))),
            ('svr',   Pipeline([('sc', StandardScaler()),
                                ('svr', LinearSVR(max_iter=sp.get('svr_max_iter', 5000)))])),
            ('xgb',  xgb.XGBRegressor(
                        n_estimators=sp.get('xgb_n_estimators', 100),
                        max_depth=sp.get('xgb_max_depth', 3),
                        learning_rate=sp.get('xgb_learning_rate', 0.1),
                        verbosity=0, random_state=SEED, device=_XGB_DEVICE)),
            ('lgbm', lgb.LGBMRegressor(
                        n_estimators=sp.get('lgbm_n_estimators', 100),
                        max_depth=sp.get('lgbm_max_depth', 3),
                        learning_rate=sp.get('lgbm_learning_rate', 0.1),
                        verbose=-1, random_state=SEED, device=_LGBM_DEVICE)),
        ],
        final_estimator=Ridge(alpha=sp.get('ridge_alpha', 1.0)),
        cv=sp.get('cv', 5), n_jobs=-1,
    )


def run_analysis_stack(crop_name, trait_name, prep_method='snv', test_size=0.2):
    """Stacking ensemble."""
    traits_df, spectra, _ = crops_data[crop_name]
    if trait_name not in traits_df.columns:
        return None
    y  = pd.to_numeric(traits_df[trait_name], errors='coerce').values
    ok = ~np.isnan(y)
    if ok.sum() < 10:
        return None

    X_ok, y_ok = spectra[ok], y[ok]
    X_tr, X_te, y_tr, y_te = train_test_split(X_ok, y_ok,
                                               test_size=test_size, random_state=SEED)
    X_tr_pp, ref, _ = preprocess_spectra(X_tr, prep_method)
    X_te_pp, _,  _  = preprocess_spectra(X_te, prep_method, ref=ref)

    try:
        with _quiet():
            model = _make_stack()
            model.fit(X_tr_pp, y_tr)
            cv_p    = cross_val_predict(_make_stack(), X_tr_pp, y_tr, cv=CV)
            r2_cv   = r2_score(y_tr, cv_p)
            rmse_cv = np.sqrt(mean_squared_error(y_tr, cv_p))
    except Exception as e:
        print(f'  [stack error] {crop_name}/{trait_name}: {e}')
        return None

    cal_p = model.predict(X_tr_pp)
    val_p = model.predict(X_te_pp)
    r2_cal, rmse_cal, _       = _metrics(y_tr, cal_p)
    r2_val, rmse_val, rpd_val = _metrics(y_te, val_p)

    return {
        'Crop': crop_name, 'Trait': trait_name,
        'Method': 'stack', 'Preprocessing': prep_method,
        'N_train': len(y_tr), 'N_test': len(y_te),
        'R2_cal':  round(r2_cal,  4), 'RMSE_cal': round(rmse_cal, 4),
        'R2_cv':   round(r2_cv,   4), 'RMSE_cv':  round(rmse_cv,  4),
        'R2_val':  round(r2_val,  4), 'RMSE_val': round(rmse_val, 4),
        'RPD_val': round(rpd_val, 3),
        '_model': model, '_X_te': X_te_pp, '_y_te': y_te,
    }


# ── Gaussian Process Regression ────────────────────────────────────────────
def _make_gpr():
    return Pipeline([
        ('pca', PCA(n_components=_mp('gpr', 'n_components', 20))),
        ('gpr', GaussianProcessRegressor(
            kernel=(RBF(length_scale=_mp('gpr', 'length_scale', 1.0)) +
                    WhiteKernel(noise_level=_mp('gpr', 'noise_level', 0.1))),
            n_restarts_optimizer=_mp('gpr', 'n_restarts_optimizer', 3),
            normalize_y=True, random_state=SEED,
        )),
    ])


def run_analysis_gpr(crop_name, trait_name, prep_method='snv', test_size=0.2):
    """GPR with PCA(20); returns metrics + per-sample σ."""
    traits_df, spectra, wavelengths = crops_data[crop_name]
    if trait_name not in traits_df.columns:
        return None
    y  = pd.to_numeric(traits_df[trait_name], errors='coerce').values
    ok = ~np.isnan(y)
    if ok.sum() < 10:
        return None

    X_ok, y_ok = spectra[ok], y[ok]
    X_tr, X_te, y_tr, y_te = train_test_split(X_ok, y_ok,
                                               test_size=test_size, random_state=SEED)
    X_tr_pp, ref, _ = preprocess_spectra(X_tr, prep_method)
    X_te_pp, _,  _  = preprocess_spectra(X_te, prep_method, ref=ref)

    with _quiet():
        try:
            gpr = _make_gpr()
            gpr.fit(X_tr_pp, y_tr)
        except Exception as e:
            print(f'  [gpr error] {crop_name}/{trait_name}: {e}')
            return None

        pca_te    = gpr['pca'].transform(X_te_pp)
        mu, sigma = gpr['gpr'].predict(pca_te, return_std=True)
        cv_p      = cross_val_predict(_make_gpr(), X_tr_pp, y_tr, cv=CV)
        r2_cv     = r2_score(y_tr, cv_p)
        rmse_cv   = np.sqrt(mean_squared_error(y_tr, cv_p))

    r2_cal, rmse_cal, _ = _metrics(y_tr, gpr.predict(X_tr_pp))
    r2_val, rmse_val, rpd_val = _metrics(y_te, mu)

    z90, z95 = 1.645, 1.960
    cov90 = float(np.mean(np.abs(y_te - mu) <= z90 * sigma))
    cov95 = float(np.mean(np.abs(y_te - mu) <= z95 * sigma))

    return {
        'Crop': crop_name, 'Trait': trait_name,
        'Method': 'gpr', 'Preprocessing': prep_method,
        'N_train': len(y_tr), 'N_test': len(y_te),
        'R2_cal':  round(r2_cal,  4), 'RMSE_cal': round(rmse_cal, 4),
        'R2_cv':   round(r2_cv,   4), 'RMSE_cv':  round(rmse_cv,  4),
        'R2_val':  round(r2_val,  4), 'RMSE_val': round(rmse_val, 4),
        'RPD_val': round(rpd_val, 3),
        'coverage_90': round(cov90, 3), 'coverage_95': round(cov95, 3),
        '_mu': mu, '_sigma': sigma, '_y_te': y_te,
    }


def run_analysis_cnn(crop_name, trait_name, prep_method='snv', test_size=0.2):
    """1D-CNN."""
    if not HAS_TORCH:
        return None
    traits_df, spectra, _ = crops_data[crop_name]
    if trait_name not in traits_df.columns:
        return None
    y  = pd.to_numeric(traits_df[trait_name], errors='coerce').values
    ok = ~np.isnan(y)
    if ok.sum() < 10:
        return None

    X_ok, y_ok = spectra[ok], y[ok]
    X_tr, X_te, y_tr, y_te = train_test_split(X_ok, y_ok,
                                               test_size=test_size, random_state=SEED)
    X_tr_pp, ref, _ = preprocess_spectra(X_tr, prep_method)
    X_te_pp, _,  _  = preprocess_spectra(X_te, prep_method, ref=ref)

    _cnn_epochs    = _mp('cnn', 'epochs',      200)
    _cnn_batch     = _mp('cnn', 'batch_size',  16)
    _cnn_patience  = _mp('cnn', 'patience',    30)
    _cnn_cv_epochs = _mp('cnn', 'cv_epochs',   max(50, _cnn_epochs - 50))
    _cnn_cv_pat    = _mp('cnn', 'cv_patience', max(10, _cnn_patience - 10))

    with _quiet():
        try:
            torch.manual_seed(SEED)
            model = CNN1DRegressor(epochs=_cnn_epochs, patience=_cnn_patience,
                                   batch_size=_cnn_batch)
            model.fit(X_tr_pp, y_tr)
        except Exception as e:
            print(f'  [cnn error] {crop_name}/{trait_name}: {e}')
            return None

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_preds = np.empty(len(y_tr))
        for tr_i, va_i in kf.split(X_tr_pp):
            torch.manual_seed(SEED)
            c_fold = CNN1DRegressor(epochs=_cnn_cv_epochs, patience=_cnn_cv_pat,
                                    batch_size=_cnn_batch)
            c_fold.fit(X_tr_pp[tr_i], y_tr[tr_i])
            cv_preds[va_i] = c_fold.predict(X_tr_pp[va_i]).ravel()
        r2_cv   = r2_score(y_tr, cv_preds)
        rmse_cv = np.sqrt(mean_squared_error(y_tr, cv_preds))

    cal_p = model.predict(X_tr_pp)
    val_p = model.predict(X_te_pp)
    if hasattr(cal_p, 'ravel'): cal_p = cal_p.ravel()
    if hasattr(val_p, 'ravel'): val_p = val_p.ravel()

    r2_cal, rmse_cal, _       = _metrics(y_tr, cal_p)
    r2_val, rmse_val, rpd_val = _metrics(y_te, val_p)

    return {
        'Crop': crop_name, 'Trait': trait_name,
        'Method': 'cnn', 'Preprocessing': prep_method,
        'N_train': len(y_tr), 'N_test': len(y_te),
        'R2_cal':  round(r2_cal,  4), 'RMSE_cal': round(rmse_cal, 4),
        'R2_cv':   round(r2_cv,   4), 'RMSE_cv':  round(rmse_cv,  4),
        'R2_val':  round(r2_val,  4), 'RMSE_val': round(rmse_val, 4),
        'RPD_val': round(rpd_val, 3),
        '_model': model, '_X_te': X_te_pp, '_y_te': y_te,
    }


# ── 1D-CNN (requires torch) ────────────────────────────────────────────────
if HAS_TORCH:
    class _CNN1D(nn.Module):
        def __init__(self, dropout=0.4):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 32,  kernel_size=11, padding=5)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=7,  padding=3)
            self.pool2 = nn.MaxPool1d(2)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.adapt = nn.AdaptiveAvgPool1d(32)
            self.drop  = nn.Dropout(dropout)
            self.fc1   = nn.Linear(128 * 32, 64)
            self.fc2   = nn.Linear(64, 1)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)

        def forward(self, x):
            x = x.unsqueeze(1)
            x = F.relu(self.conv1(x)); x = self.pool1(x)
            x = F.relu(self.conv2(x)); x = self.pool2(x)
            x = F.relu(self.conv3(x)); x = self.adapt(x)
            x = x.view(x.size(0), -1)
            x = self.drop(F.relu(self.fc1(x)))
            return self.fc2(x).squeeze(-1)

    class CNN1DRegressor(BaseEstimator, RegressorMixin):
        """Sklearn-compatible 1D-CNN with early stopping and noise augmentation."""
        def __init__(self, dropout=0.4, lr=1e-3, weight_decay=1e-4,
                     epochs=200, batch_size=16, patience=30,
                     val_frac=0.15, noise_std=0.001):
            self.dropout      = dropout
            self.lr           = lr
            self.weight_decay = weight_decay
            self.epochs       = epochs
            self.batch_size   = batch_size
            self.patience     = patience
            self.val_frac     = val_frac
            self.noise_std    = noise_std

        def fit(self, X, y):
            X = np.asarray(X, np.float32)
            y = np.asarray(y, np.float32)
            n_val = max(1, int(len(y) * self.val_frac))
            idx = np.random.permutation(len(y))
            vi, ti = idx[:n_val], idx[n_val:]

            self.model_ = _CNN1D(self.dropout).to(DEVICE)
            opt   = torch.optim.Adam(self.model_.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10,
                                                                factor=0.5)
            loader = DataLoader(
                TensorDataset(torch.tensor(X[ti]), torch.tensor(y[ti])),
                batch_size=self.batch_size, shuffle=True)

            best_val, wait, best_state = np.inf, 0, None
            for _ in range(self.epochs):
                self.model_.train()
                for Xb, yb in loader:
                    Xb = Xb + torch.randn_like(Xb) * self.noise_std
                    opt.zero_grad()
                    F.mse_loss(self.model_(Xb.to(DEVICE)),
                               yb.to(DEVICE)).backward()
                    opt.step()
                self.model_.eval()
                with torch.no_grad():
                    vl = F.mse_loss(
                        self.model_(torch.tensor(X[vi]).to(DEVICE)),
                        torch.tensor(y[vi]).to(DEVICE)).item()
                sched.step(vl)
                if vl < best_val - 1e-6:
                    best_val, wait = vl, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model_.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            if best_state:
                self.model_.load_state_dict(best_state)
            self.model_.eval()
            return self

        def predict(self, X):
            X = np.asarray(X, np.float32)
            self.model_.eval()
            with torch.no_grad():
                return self.model_(torch.tensor(X).to(DEVICE)).cpu().numpy()

        def saliency(self, X):
            """Mean |∂output/∂input| is the wavelength importance."""
            self.model_.eval()
            X_t = torch.tensor(np.asarray(X, np.float32),
                               device=DEVICE, requires_grad=True)
            with torch.enable_grad():
                self.model_(X_t).sum().backward()
            return X_t.grad.abs().mean(0).cpu().numpy()


# ── SHAP helpers (requires shap) ───────────────────────────────────────────
if HAS_SHAP:
    def _get_shap_values(model, X_bg, X_explain):
        inner = model
        if hasattr(model, 'named_steps'):
            inner = list(model.named_steps.values())[-1]
        mtype = type(inner).__name__

        if mtype in ('RandomForestRegressor', 'XGBRegressor',
                     'LGBMRegressor', 'GradientBoostingRegressor'):
            ex = shap.TreeExplainer(inner)
            if hasattr(model, 'named_steps') and len(model.named_steps) > 1:
                for step in list(model.named_steps.values())[:-1]:
                    X_explain = step.transform(X_explain)
            return ex.shap_values(X_explain)

        if mtype in ('Lasso', 'ElasticNet', 'Ridge', 'LinearRegression'):
            masker = shap.maskers.Independent(X_bg, max_samples=100)
            ex = shap.LinearExplainer(inner, masker=masker)
            if hasattr(model, 'named_steps') and len(model.named_steps) > 1:
                for step in list(model.named_steps.values())[:-1]:
                    X_explain = step.transform(X_explain)
            return ex.shap_values(X_explain)

        bg = shap.sample(X_bg, min(30, len(X_bg)))
        ex = shap.KernelExplainer(model.predict, bg)
        return ex.shap_values(X_explain[:min(20, len(X_explain))], silent=True)

    def plot_shap_wavelengths(shap_vals, wavelengths, title='', ax=None):
        mean_abs = np.abs(shap_vals).mean(0)
        if len(mean_abs) != len(wavelengths):
            n = min(len(mean_abs), len(wavelengths))
            mean_abs   = mean_abs[:n]
            wavelengths = wavelengths[:n]
        show = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3))
        ax.plot(wavelengths, mean_abs, color='#2c7bb6', lw=0.8)
        ax.fill_between(wavelengths, mean_abs, alpha=0.2, color='#2c7bb6')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Mean |SHAP|')
        ax.set_title(title, fontsize=9)
        if show:
            plt.tight_layout(); plt.show()
        return mean_abs


# ── Plotting helpers ───────────────────────────────────────────────────────
def plot_predictions(y_true, y_pred, title='', units='', ax=None):
    df  = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred}).dropna()
    r2, rmse, rpd = _metrics(df.Actual.values, df.Predicted.values)
    lim = [min(df.Actual.min(), df.Predicted.min()),
           max(df.Actual.max(), df.Predicted.max())]
    pad = (lim[1] - lim[0]) * 0.05
    lim = [lim[0] - pad, lim[1] + pad]
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df.Actual, df.Predicted, alpha=0.7, color='#2c7bb6', s=25)
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel(f'Actual ({units})' if units else 'Actual')
    ax.set_ylabel(f'Predicted ({units})' if units else 'Predicted')
    ax.set_title(title, fontsize=9)
    ax.text(0.05, 0.95,
            f'R²={r2:.3f}\nRMSE={rmse:.3f}\nRPD={rpd:.2f}',
            transform=ax.transAxes, va='top', fontsize=8,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    if show:
        plt.tight_layout(); plt.show()


def plot_importance(model, wl, title='', X_ref=None):
    inner = model
    if hasattr(model, 'named_steps'):
        inner = list(model.named_steps.values())[-1]
    if hasattr(inner, 'coef_'):
        imp, ylabel = np.abs(inner.coef_).ravel(), 'Abs. coefficient'
    elif hasattr(inner, 'feature_importances_'):
        imp, ylabel = inner.feature_importances_,  'Feature importance'
    elif hasattr(inner, 'x_weights_'):
        imp, ylabel = np.abs(inner.x_weights_[:, 0]), '|PLS weight comp.1|'
    elif hasattr(model, 'saliency') and X_ref is not None:
        imp, ylabel = model.saliency(X_ref), 'CNN saliency'
    else:
        return None
    if len(imp) != len(wl):
        n = min(len(imp), len(wl))
        imp = imp[:n]
        wl = wl[:n]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(wl, imp, color='#2c7bb6', lw=0.7)
    ax.fill_between(wl, imp, alpha=0.25, color='#2c7bb6')
    ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    return fig


# ── Command-line entry point ───────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description='NIRS Calibration Pipeline')
    p.add_argument('--config', default=None, metavar='YAML',
                   help='Path to YAML config file '
                        f'(default: {_DEFAULT_CONFIG_PATH})')
    p.add_argument('--quick', action='store_true',
                   help='Skip CNN, RF, XGB for a faster exploratory run')
    p.add_argument('--crops',   nargs='+', default=None, metavar='CROP',
                   help='Crops to run (default: all crops in config)')
    p.add_argument('--traits',  nargs='+', default=None, metavar='TRAIT',
                   help='Traits to run (default: all auto-discovered per crop)')
    p.add_argument('--methods', nargs='+', default=None, metavar='METHOD',
                   help='Methods to run (default: full_methods from config)')
    p.add_argument('--preprocs', nargs='+', default=None, metavar='PREP',
                   help='Preprocessing methods (default: preprocs from config)')
    p.add_argument('--out', default=os.path.join(BASE_DIR, 'results_python'),
                   metavar='DIR')
    p.add_argument('--no-advanced', action='store_true',
                   help='Skip stacking, CNN, GPR sections')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Print metrics and best hyperparameters after each run')
    return p.parse_args()


_BANNER = """\
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║    ╭─╮       ╭──╮    ╭╮      ╭────╮                   ║
║  ──╯  ╰──╭───╯  ╰────╯╰──────╯    ╰──  λ →            ║
║                                                       ║
║  ██████╗ ██████╗ ██╗███████╗███╗   ███╗               ║
║  ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║               ║
║  ██████╔╝██████╔╝██║███████╗██╔████╔██║               ║
║  ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║               ║
║  ██║     ██║  ██║██║███████║██║ ╚═╝ ██║               ║
║  ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝               ║
║  Predictive Regression via Infrared Spectral Models   ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝"""


if __name__ == '__main__':
    plt.switch_backend('Agg')   # non-interactive; all plots saved to files
    sns.set_style('whitegrid')
    print(_BANNER)
    print()

    args = _parse_args()

    # ── Load config (CLI flag overrides the default path) ──────────────────
    config_path = args.config or _DEFAULT_CONFIG_PATH
    if os.path.exists(config_path):
        apply_config(load_config(config_path))
        print(f'Config: {config_path}')
    elif args.config:
        raise FileNotFoundError(f'Config file not found: {args.config}')

    OUT_DIR   = args.out
    PLOT_DIR  = os.path.join(OUT_DIR, 'plots')
    MODEL_DIR = os.path.join(OUT_DIR, 'best_models')
    for d in [OUT_DIR, PLOT_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

    # CLI args override config defaults
    CROPS    = args.crops   or list(CROP_CONFIG)
    PREPROCS = args.preprocs or DEFAULT_PREPROCS
    _methods_default = DEFAULT_METHODS_QUICK if args.quick else DEFAULT_METHODS_FULL
    METHODS  = args.methods or _methods_default

    # Validate crops
    unknown_crops = [c for c in CROPS if c not in CROP_CONFIG]
    if unknown_crops:
        print(f'[error] Unknown crop(s): {unknown_crops}')
        print(f'        Available: {list(CROP_CONFIG)}')
        sys.exit(1)

    adv_fns = []
    if not args.no_advanced:
        adv_fns = [(run_analysis_stack, 'stack'), (run_analysis_gpr, 'gpr')]
        if not args.quick:
            adv_fns.append((run_analysis_cnn, 'cnn'))
    adv_labels = [label for _, label in adv_fns]
    all_methods = METHODS + adv_labels

    # Validate methods (grid only; advanced labels are derived, not user-typed)
    _known_grid = set(DEFAULT_METHODS_FULL)
    unknown_methods = [m for m in METHODS if m not in _known_grid]
    if unknown_methods:
        print(f'[error] Unknown method(s): {unknown_methods}')
        print(f'        Available: {sorted(_known_grid)}')
        sys.exit(1)

    # Validate preprocessing methods
    _known_preps = {'raw', 'snv', 'msc', 'sg1', 'sg2'}
    unknown_preps = [p for p in PREPROCS if p not in _known_preps]
    if unknown_preps:
        print(f'[error] Unknown preprocessing method(s): {unknown_preps}')
        print(f'        Available: {sorted(_known_preps)}')
        sys.exit(1)

    # ── Load data (needed to discover traits before printing summary) ─────────
    load_all_crops(CROPS)

    # Validate traits (after loading so AVAILABLE_TRAITS is populated)
    if args.traits:
        bad_traits = {}
        for crop in CROPS:
            unknown_traits = [t for t in args.traits if t not in AVAILABLE_TRAITS[crop]]
            if unknown_traits:
                bad_traits[crop] = unknown_traits
        if bad_traits:
            for crop, traits in bad_traits.items():
                print(f'[error] Unknown trait(s) for {crop!r}: {traits}')
                print(f'        Available: {AVAILABLE_TRAITS[crop]}')
            sys.exit(1)

    traits_by_crop = {c: args.traits or AVAILABLE_TRAITS[c] for c in CROPS}

    print('=' * 60)
    print('NIRS Calibration: a Python pipeline')
    print(f'Crops:         {CROPS}')
    for crop, traits in traits_by_crop.items():
        print(f'  {crop}: {traits}')
    print(f'Methods:       {all_methods}')
    print(f'  Grid:        {METHODS}')
    print(f'  Advanced:    {adv_labels if adv_labels else "(none)"}')
    print(f'Preprocessing: {PREPROCS}')
    print(f'Output:        {OUT_DIR}')
    if _CUDA:
        props = torch.cuda.get_device_properties(0)
        print(f'GPU:           {props.name}  ({props.total_memory/1e9:.1f} GB VRAM)')
        print(f'               XGBoost={_XGB_DEVICE}  LightGBM={_LGBM_DEVICE}  CNN=cuda')
    else:
        print('GPU:           not available so we are running on CPU')
    print('=' * 60)

    # ── Main grid run ──────────────────────────────────────────────────────
    records, done = [], 0

    total = 0
    for crop, traits in traits_by_crop.items():
        total += len(traits) * len(METHODS) * len(PREPROCS)
        total += len(traits) * len(adv_fns) * len(PREPROCS)

    for crop in CROPS:
        print(f'\n=== {crop.upper()} ===')
        traits = traits_by_crop[crop]
        for trait in traits:
            for method in METHODS:
                for prep in PREPROCS:
                    done += 1
                    print(f'  [{done}/{total}] {crop}/{trait}/{method}/{prep}')
                    res = run_analysis(crop, trait, method, prep,
                                       verbose=args.verbose)
                    if res:
                        records.append(res)
                        if args.verbose:
                            print(f'    -> R2_cal={res["R2_cal"]:.3f}  '
                                  f'R2_cv={res["R2_cv"]:.3f}  '
                                  f'R2_val={res["R2_val"]:.3f}  '
                                  f'RMSE_val={res["RMSE_val"]:.4f}  '
                                  f'RPD={res["RPD_val"]:.2f}')

    print(f'\nGrid complete: {len(records)} results.')

    # ── Advanced methods (integrated into main comparison) ─────────────────
    if adv_fns:
        print('\nRunning advanced methods...')
        for fn, label in adv_fns:
            for crop in CROPS:
                for trait in traits_by_crop[crop]:
                    for prep in PREPROCS:
                        done += 1
                        print(f'  [{done}/{total}] {crop}/{trait}/{label}/{prep}')
                        res = fn(crop, trait, prep_method=prep)
                        if res:
                            records.append({k: v for k, v in res.items()
                                            if not k.startswith('_')})
                            if args.verbose:
                                extra = (f'  cov90={res["coverage_90"]:.2f}'
                                         if 'coverage_90' in res else '')
                                print(f'    -> R2_cal={res["R2_cal"]:.3f}  '
                                      f'R2_cv={res["R2_cv"]:.3f}  '
                                      f'R2_val={res["R2_val"]:.3f}  '
                                      f'RMSE_val={res["RMSE_val"]:.4f}  '
                                      f'RPD={res["RPD_val"]:.2f}{extra}')
                            else:
                                extra = (f'  cov90={res["coverage_90"]:.2f}'
                                         if 'coverage_90' in res else '')
                                print(f'    -> R2_val={res["R2_val"]:.3f}{extra}')

    results_df = pd.DataFrame(records)
    results_df.to_csv(os.path.join(OUT_DIR, 'all_model_results.csv'), index=False)
    print(f'Total: {len(results_df)} results saved.')

    # ── Best configs ───────────────────────────────────────────────────────
    best = (results_df.dropna(subset=['R2_val'])
            .sort_values('R2_val', ascending=False)
            .groupby(['Crop', 'Trait'], as_index=False).first()
            .sort_values(['Crop', 'Trait']))
    best.to_csv(os.path.join(OUT_DIR, 'best_model_configs.csv'), index=False)
    print('\nBest model per crop × trait:')
    print(best[['Crop','Trait','Method','Preprocessing','R2_val','RPD_val']]
          .to_string(index=False))

    # ── Retrain and save best models ───────────────────────────────────────
    print('\nRetraining best models...')
    for _, row in best.iterrows():
        crop, trait, method, prep = row.Crop, row.Trait, row.Method, row.Preprocessing
        tdf, spec, wl = crops_data[crop]
        y  = pd.to_numeric(tdf[trait], errors='coerce').values
        ok = ~np.isnan(y)
        X_ok, y_ok = spec[ok], y[ok]
        X_tr, X_te, y_tr, y_te = train_test_split(X_ok, y_ok,
                                                   test_size=0.2, random_state=SEED)
        X_tr_pp, ref, wl_trim = preprocess_spectra(X_tr, prep)
        X_te_pp, _,  _        = preprocess_spectra(X_te, prep, ref=ref)
        wl_use = wl[wl_trim:len(wl)-wl_trim] if wl_trim > 0 else wl

        try:
            with _quiet():
                if method == 'pls':
                    model_obj, _, _ = _fit_pls(X_tr_pp, y_tr)
                elif method == 'stack':
                    model_obj = _make_stack()
                    model_obj.fit(X_tr_pp, y_tr)
                elif method == 'gpr':
                    model_obj = _make_gpr()
                    model_obj.fit(X_tr_pp, y_tr)
                elif method == 'cnn':
                    if not HAS_TORCH:
                        print(f'  [skip] {crop}/{trait}: torch not available'); continue
                    torch.manual_seed(SEED)
                    model_obj = CNN1DRegressor(epochs=200, patience=30)
                    model_obj.fit(X_tr_pp, y_tr)
                else:
                    est, grid = _make_grid_estimator(method)
                    gs = GridSearchCV(est, grid, cv=CV,
                                      scoring='neg_root_mean_squared_error',
                                      n_jobs=-1, refit=True)
                    gs.fit(X_tr_pp, y_tr)
                    model_obj = gs.best_estimator_
        except Exception as e:
            print(f'  [skip] {crop}/{trait}: {e}'); continue

        y_pred = model_obj.predict(X_te_pp)
        if hasattr(y_pred, 'ravel'): y_pred = y_pred.ravel()

        fname = f'{crop}_{trait}_{method}_{prep}.pkl'
        joblib.dump({'model': model_obj, 'ref': ref, 'preprocess': prep,
                     'crop': crop, 'trait': trait, 'wavelengths': wl_use,
                     'y_test': y_te, 'y_pred': y_pred, 'X_test_pp': X_te_pp},
                    os.path.join(MODEL_DIR, fname))

        fig = plot_importance(model_obj, wl_use,
                              title=f'{crop}/{trait} [{method}+{prep}]',
                              X_ref=X_te_pp if method == 'cnn' else None)
        if fig:
            fig.savefig(os.path.join(PLOT_DIR,
                                     f'importance_{crop}_{trait}.png'),
                        bbox_inches='tight')
            plt.close(fig)

    print(f'Models saved to {MODEL_DIR}')
    print('\nDone.')
