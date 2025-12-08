#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Richards & Double-Richards Curve Fitting - GUI Version
WITH PREPROCESSING (Outlier Filtering + Smoothing)

GUI application for fitting Richards curves to C. elegans speed data
with optional preprocessing steps.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import traceback
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning

warnings.filterwarnings("ignore", category=OptimizeWarning)
plt.rcParams["figure.dpi"] = 110


# =============================================================================
# Preprocessing functions
# =============================================================================

def apply_hybrid_outlier_filter(
    df: pd.DataFrame,
    value_col: str = "speed",
    group_cols: List[str] = ["assay_num", "track_num"],
    percentile_lower: float = 1.25,
    percentile_upper: float = 98.75,
    mad_threshold: float = 2.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply hybrid outlier detection: percentile + MAD"""
    df = df.copy()
    df['is_outlier'] = False
    df['flagged_by'] = ''
    
    outlier_stats = []
    
    for group_id, group in df.groupby(group_cols):
        values = group[value_col].dropna()
        if len(values) < 3:
            continue
            
        lower_pct = np.percentile(values, percentile_lower)
        upper_pct = np.percentile(values, percentile_upper)
        pct_out = (values < lower_pct) | (values > upper_pct)
        
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))
        if mad > 0:
            lower_mad = median_val - mad_threshold * mad
            upper_mad = median_val + mad_threshold * mad
            mad_out = (values < lower_mad) | (values > upper_mad)
        else:
            mad_out = pd.Series(False, index=values.index)
        
        inter = pct_out & mad_out
        out_idx = values[inter].index
        df.loc[out_idx, 'is_outlier'] = True
        
        for idx in out_idx:
            methods = []
            if pct_out.loc[idx]:
                methods.append('percentile')
            if mad_out.loc[idx]:
                methods.append('MAD')
            df.loc[idx, 'flagged_by'] = '+'.join(methods)
        
        n_total = len(values)
        n_outliers = len(out_idx)
        pct_outliers = (n_outliers / n_total * 100) if n_total > 0 else 0
        
        outlier_stats.append({
            **dict(zip(group_cols, group_id if isinstance(group_id, tuple) else [group_id])),
            'n_points': n_total,
            'n_outliers': n_outliers,
            'pct_outliers': pct_outliers,
        })
    
    df_filtered = df[~df['is_outlier']].copy()
    df_filtered.drop(columns=['is_outlier', 'flagged_by'], inplace=True)
    
    outlier_report = pd.DataFrame(outlier_stats)
    
    return df_filtered, outlier_report


def apply_smoothing(
    df: pd.DataFrame,
    value_col: str = "speed",
    group_cols: List[str] = ["assay_num", "track_num"],
    window: int = 5,
    method: str = "median",
) -> pd.DataFrame:
    """Apply rolling median or mean smoothing"""
    if window <= 1:
        return df
    
    df = df.copy()
    smoothed_groups = []
    
    for group_id, group in df.groupby(group_cols):
        group = group.sort_values('time').copy()
        
        if method == "median":
            group[value_col] = group[value_col].rolling(
                window=window, center=True, min_periods=1
            ).median()
        else:
            group[value_col] = group[value_col].rolling(
                window=window, center=True, min_periods=1
            ).mean()
        
        smoothed_groups.append(group)
    
    return pd.concat(smoothed_groups, ignore_index=True)


def convert_speed_to_mm_s(
    df: pd.DataFrame,
    pixels_per_mm: float = 104.0,
    speed_col: str = "speed",
    assume_pixels: bool = True,
) -> pd.DataFrame:
    """Convert speed from pixels/frame to mm/s"""
    df = df.copy()
    
    if speed_col in df.columns and assume_pixels:
        df[speed_col] = df[speed_col] / pixels_per_mm
    
    return df


def calculate_speed_from_xy(x, y, time, smoothing_window: int = 3, pixels_per_mm: float = 104.0):
    """Compute instantaneous speed from x,y coords in mm/s"""
    x = np.asarray(x); y = np.asarray(y); time = np.asarray(time)
    if len(x) < 2:
        return np.full_like(time, np.nan, dtype=float)

    dx = np.diff(x); dy = np.diff(y); dt = np.diff(time)
    dt = np.where(dt == 0, 1e-12, dt)
    dist_px = np.sqrt(dx**2 + dy**2)
    dist_mm = dist_px / pixels_per_mm
    v = dist_mm / dt
    v = np.r_[v[0], v]

    if smoothing_window and smoothing_window > 1:
        v = pd.Series(v).rolling(smoothing_window, center=True, min_periods=1).mean().to_numpy()
    return v


# =============================================================================
# Curve functions
# =============================================================================

def richards_curve(t, y_initial, y_final, B, M, nu):
    """Single Richards curve"""
    A = y_final - y_initial
    with np.errstate(over="ignore", invalid="ignore"):
        denom = (1.0 + np.exp(-B * (t - M))) ** (1.0 / np.maximum(nu, 1e-6))
    denom = np.clip(denom, 1e-10, 1e10)
    return y_initial + A / denom


def double_richards_curve(t, y1_i, y1_f, B1, M1, nu1, y2_f, B2, M2, nu2):
    """Double Richards curve"""
    with np.errstate(over="ignore", invalid="ignore"):
        term1 = (y1_f - y1_i) / (1.0 + np.exp(-B1 * (t - M1))) ** (1.0 / np.maximum(nu1, 1e-6))
        term2 = (y2_f - y1_f) / (1.0 + np.exp(-B2 * (t - M2))) ** (1.0 / np.maximum(nu2, 1e-6))
    term1 = np.clip(term1, -1e10, 1e10)
    term2 = np.clip(term2, -1e10, 1e10)
    return y1_i + term1 + term2


# =============================================================================
# Windowing
# =============================================================================

def find_first_food_encounter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Find first food encounter for each animal"""
    out = []
    for (assay, track), g in df.groupby(["assay_num", "track_num"]):
        mask = (g.get("food_encounter", pd.Series(False, index=g.index)) == "food")
        if mask.any():
            first_idx = g[mask].index[0]
            out.append({
                "assay_num": assay,
                "track_num": track,
                "encounter_time": g.loc[first_idx, "time"],
                "encounter_idx": int(first_idx),
            })
    return pd.DataFrame(out)


def extract_windows(df: pd.DataFrame, window_before: float = 30.0, window_after: float = 30.0,
                    min_points: int = 10) -> Dict[Tuple[int, int], pd.DataFrame]:
    """Extract time windows around first food encounter"""
    encounters = find_first_food_encounter_rows(df)
    if encounters.empty:
        return {}

    out: Dict[Tuple[int, int], pd.DataFrame] = {}
    for _, row in encounters.iterrows():
        assay = row["assay_num"]; track = row["track_num"]; t0 = row["encounter_time"]
        g = df[(df["assay_num"] == assay) & (df["track_num"] == track)].copy()
        mask = (g["time"] >= t0 - window_before) & (g["time"] <= t0 + window_after)
        w = g.loc[mask].copy()
        if len(w) < min_points:
            continue
        w["time_rel"] = w["time"] - t0
        out[(assay, track)] = w[["time_rel", "speed"]].reset_index(drop=True)
    return out


# =============================================================================
# Fitting functions
# =============================================================================

def _failed(model_type: str, reason: str) -> dict:
    base = {
        "model_type": model_type, "fit_type": "failed", "converged": False,
        "r_squared": np.nan, "rmse": np.nan, "aic": np.nan, "bic": np.nan,
        "fitted_curve": None, "residuals": None,
        "y_initial": np.nan, "y_final": np.nan, "A": np.nan,
    }
    if model_type == "single":
        base.update({"B": np.nan, "M": np.nan, "nu": np.nan})
    else:
        base.update({
            "y1_i": np.nan, "y1_f": np.nan, "y2_f": np.nan,
            "B1": np.nan, "M1": np.nan, "nu1": np.nan,
            "B2": np.nan, "M2": np.nan, "nu2": np.nan,
            "delta1": np.nan, "delta2": np.nan,
        })
    base["failure_reason"] = reason
    return base


def _metrics(y_true: np.ndarray, y_fit: np.ndarray, k_params: int) -> Tuple[float, float, float, float]:
    res = y_true - y_fit
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(res**2)))
    n = len(y_true)
    if n <= k_params or ss_res <= 0:
        return r2, rmse, np.nan, np.nan
    aic = n * np.log(ss_res / n) + 2 * k_params
    bic = n * np.log(ss_res / n) + k_params * np.log(n)
    return r2, rmse, aic, bic


def fit_single(time_rel: np.ndarray, speed: np.ndarray) -> dict:
    valid = ~(np.isnan(time_rel) | np.isnan(speed))
    t = time_rel[valid]; y = speed[valid]
    if len(t) < 8:
        return _failed("single", "insufficient_data")

    yi = np.nanpercentile(y, 10)
    yf = np.nanpercentile(y, 90)
    p0 = [yi, yf, 0.1, 0.0, 1.0]
    lower = [np.min(y) - 3*np.std(y), np.min(y) - 3*np.std(y), 1e-3, np.min(t) - 60, 0.1]
    upper = [np.max(y) + 3*np.std(y), np.max(y) + 3*np.std(y), 10.0, np.max(t) + 60, 10.0]

    try:
        params, _ = curve_fit(richards_curve, t, y, p0=p0, bounds=(lower, upper), maxfev=20000, method="trf")
        yhat = richards_curve(t, *params)
        r2, rmse, aic, bic = _metrics(y, yhat, k_params=5)
        yi, yf, B, M, nu = params
        return {
            "model_type": "single", "fit_type": "single", "converged": True,
            "r_squared": r2, "rmse": rmse, "aic": aic, "bic": bic,
            "y_initial": yi, "y_final": yf, "A": yf - yi,
            "B": B, "M": M, "nu": nu,
            "fitted_curve": yhat, "residuals": y - yhat,
        }
    except Exception as e:
        return _failed("single", str(e))


def fit_double(time_rel: np.ndarray, speed: np.ndarray) -> dict:
    valid = ~(np.isnan(time_rel) | np.isnan(speed))
    t = time_rel[valid]; y = speed[valid]
    if len(t) < 12:
        return _failed("double", "insufficient_data")

    tmin, tmax = float(t.min()), float(t.max())
    ymin, ymax = float(y.min()), float(y.max())
    ymean = float(y.mean())

    y1_i_guess = np.nanpercentile(y, 10)
    y1_f_guess = ymean
    y2_f_guess = np.nanpercentile(y, 90)
    M1_guess = tmin + 0.3 * (tmax - tmin)
    M2_guess = tmin + 0.7 * (tmax - tmin)

    p0 = [y1_i_guess, y1_f_guess, 0.2, M1_guess, 1.0, y2_f_guess, 0.2, M2_guess, 1.0]
    lower = [ymin - 3*np.std(y), ymin - 3*np.std(y), 1e-3, tmin - 30, 0.1,
             ymin - 3*np.std(y), 1e-3, tmin - 30, 0.1]
    upper = [ymax + 3*np.std(y), ymax + 3*np.std(y), 5.0, tmax + 30, 10.0,
             ymax + 3*np.std(y), 5.0, tmax + 30, 10.0]

    try:
        params, _ = curve_fit(double_richards_curve, t, y, p0=p0, bounds=(lower, upper),
                              maxfev=50000, method="trf")
        yhat = double_richards_curve(t, *params)
        r2, rmse, aic, bic = _metrics(y, yhat, k_params=9)
        y1_i, y1_f, B1, M1, nu1, y2_f, B2, M2, nu2 = params
        return {
            "model_type": "double", "fit_type": "double", "converged": True,
            "r_squared": r2, "rmse": rmse, "aic": aic, "bic": bic,
            "y_initial": y1_i, "y_final": y2_f, "A": y2_f - y1_i,
            "y1_i": y1_i, "y1_f": y1_f, "y2_f": y2_f,
            "delta1": y1_f - y1_i, "delta2": y2_f - y1_f,
            "B1": B1, "M1": M1, "nu1": nu1,
            "B2": B2, "M2": M2, "nu2": nu2,
            "fitted_curve": yhat, "residuals": y - yhat,
        }
    except Exception as e:
        return _failed("double", str(e))


# =============================================================================
# Plotting
# =============================================================================

def plot_summary_by_sex_strain_treatment(df_results, param, save_path):
    """Violin plot: param ~ (sex, strain, treatment)"""
    print(f"DEBUG: Plotting {param} to {save_path}")
    df = df_results[df_results["converged"] & df_results[param].notna()].copy()
    if df.empty:
        print(f"DEBUG: No data for {param}, returning early")
        return

    groups = []
    for (sex, strain, treat), g in df.groupby(["sex", "strain_genotype", "treatment"], dropna=False):
        s_str = str(sex) if pd.notna(sex) else "NA"
        strain_str = str(strain) if pd.notna(strain) else "NA"
        treat_str = str(treat) if pd.notna(treat) else "NA"
        groups.append((f"{s_str}|{strain_str}|{treat_str}", g[param].values))

    if not groups:
        print(f"DEBUG: No groups for {param}, returning early")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(groups)*0.8), 5))
    positions = list(range(1, len(groups) + 1))
    data_list = [g[1] for g in groups]
    labels = [g[0] for g in groups]

    parts = ax.violinplot(data_list, positions=positions, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(param)
    ax.set_title(f"{param} by Sex | Strain | Treatment")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# GUI Application
# =============================================================================

class RichardsCurveFitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Richards Curve Fitting - C. elegans Speed Analysis")
        self.root.geometry("1000x900")
        
        # Data storage
        self.data_path = None
        self.data = None
        self.results_single = None
        self.results_double = None
        self.outlier_report = None
        self.output_dir = None
        self.windowed_data = None
        
        # Parameters
        self.pixels_per_mm = 104.0
        self.window_before = 30.0
        self.window_after = 30.0
        self.percentile_lower = 1.25
        self.percentile_upper = 98.75
        self.mad_threshold = 2.5
        self.smooth_window = 3
        
        self.setup_gui()
    
    def setup_gui(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(7, weight=1)  # Make log section expandable
        
        ttk.Label(main, text="Richards Curve Fitting for Speed Analysis",
                  font=("Helvetica", 16, "bold")).grid(row=0, column=0, pady=10)
        
        self.create_file_section(main)
        self.create_parameters_section(main)
        self.create_preprocessing_section(main)
        self.create_r2_flagging_section(main)
        self.create_model_section(main)
        self.create_action_section(main)
        self.create_log_section(main)
    
    def create_file_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Data File", padding="10")
        frame.grid(row=1, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        ttk.Label(frame, text="CSV file:").grid(row=0, column=0, sticky="w", pady=5)
        self.file_label = ttk.Label(frame, text="No file loaded", foreground="gray")
        self.file_label.grid(row=0, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Load CSV", command=self.load_file).grid(row=0, column=2, padx=5)
        
        ttk.Label(frame, text="Output directory:").grid(row=1, column=0, sticky="w", pady=5)
        self.output_label = ttk.Label(frame, text="Not selected", foreground="gray")
        self.output_label.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Select Output Dir", command=self.select_output_dir).grid(row=1, column=2, padx=5)
    
    def create_parameters_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Analysis Parameters", padding="10")
        frame.grid(row=2, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Calibration
        ttk.Label(frame, text="Pixels per mm:").grid(row=0, column=0, sticky="w", pady=5)
        self.pixels_var = tk.DoubleVar(value=104.0)
        ttk.Entry(frame, textvariable=self.pixels_var, width=15).grid(row=0, column=1, sticky="w", padx=5)
        
        # Time windows
        ttk.Label(frame, text="Window before encounter (s):").grid(row=1, column=0, sticky="w", pady=5)
        self.before_var = tk.DoubleVar(value=30.0)
        ttk.Entry(frame, textvariable=self.before_var, width=15).grid(row=1, column=1, sticky="w", padx=5)
        
        ttk.Label(frame, text="Window after encounter (s):").grid(row=2, column=0, sticky="w", pady=5)
        self.after_var = tk.DoubleVar(value=30.0)
        ttk.Entry(frame, textvariable=self.after_var, width=15).grid(row=2, column=1, sticky="w", padx=5)
    
    def create_preprocessing_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Preprocessing Options", padding="10")
        frame.grid(row=3, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Outlier filtering
        self.enable_filtering_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Enable Outlier Filtering (Hybrid Percentile + MAD)",
                       variable=self.enable_filtering_var,
                       command=self.toggle_filtering).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        
        self.filter_frame = ttk.Frame(frame)
        self.filter_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20)
        
        ttk.Label(self.filter_frame, text="Lower percentile (%):").grid(row=0, column=0, sticky="w", pady=2)
        self.lower_percentile_var = tk.DoubleVar(value=1.25)
        self.lower_entry = ttk.Entry(self.filter_frame, textvariable=self.lower_percentile_var, width=10)
        self.lower_entry.grid(row=0, column=1, sticky="w", padx=5)
        self.lower_entry.config(state='disabled')
        
        ttk.Label(self.filter_frame, text="Upper percentile (%):").grid(row=1, column=0, sticky="w", pady=2)
        self.upper_percentile_var = tk.DoubleVar(value=98.75)
        self.upper_entry = ttk.Entry(self.filter_frame, textvariable=self.upper_percentile_var, width=10)
        self.upper_entry.grid(row=1, column=1, sticky="w", padx=5)
        self.upper_entry.config(state='disabled')
        
        ttk.Label(self.filter_frame, text="MAD threshold:").grid(row=2, column=0, sticky="w", pady=2)
        self.mad_threshold_var = tk.DoubleVar(value=2.5)
        self.mad_entry = ttk.Entry(self.filter_frame, textvariable=self.mad_threshold_var, width=10)
        self.mad_entry.grid(row=2, column=1, sticky="w", padx=5)
        self.mad_entry.config(state='disabled')
        
        # Smoothing
        ttk.Separator(frame, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.enable_smoothing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Enable Smoothing",
                       variable=self.enable_smoothing_var,
                       command=self.toggle_smoothing).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)
        
        self.smooth_frame = ttk.Frame(frame)
        self.smooth_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=20)
        
        ttk.Label(self.smooth_frame, text="Window size (frames):").grid(row=0, column=0, sticky="w", pady=2)
        self.smooth_window_var = tk.IntVar(value=3)
        self.smooth_window_entry = ttk.Entry(self.smooth_frame, textvariable=self.smooth_window_var, width=10)
        self.smooth_window_entry.grid(row=0, column=1, sticky="w", padx=5)
        self.smooth_window_entry.config(state='disabled')
        
        ttk.Label(self.smooth_frame, text="Method:").grid(row=1, column=0, sticky="w", pady=2)
        self.smooth_method_var = tk.StringVar(value="mean")
        method_frame = ttk.Frame(self.smooth_frame)
        method_frame.grid(row=1, column=1, sticky="w", padx=5)
        self.median_radio = ttk.Radiobutton(method_frame, text="Median", variable=self.smooth_method_var, 
                                           value="median", state='disabled')
        self.median_radio.pack(side="left", padx=5)
        self.mean_radio = ttk.Radiobutton(method_frame, text="Mean", variable=self.smooth_method_var, 
                                         value="mean", state='disabled')
        self.mean_radio.pack(side="left", padx=5)
    
    def create_r2_flagging_section(self, parent):
        frame = ttk.LabelFrame(parent, text="R² Quality Flagging", padding="10")
        frame.grid(row=4, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        
        # R² flagging checkbox
        self.enable_r2_flagging_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Flag Poor Fits Below R² Threshold",
                       variable=self.enable_r2_flagging_var,
                       command=self.toggle_r2_flagging).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        
        self.r2_frame = ttk.Frame(frame)
        self.r2_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20)
        
        ttk.Label(self.r2_frame, text="R² threshold:").grid(row=0, column=0, sticky="w", pady=2)
        self.r2_threshold_var = tk.DoubleVar(value=0.7)
        self.r2_entry = ttk.Entry(self.r2_frame, textvariable=self.r2_threshold_var, width=10)
        self.r2_entry.grid(row=0, column=1, sticky="w", padx=5)
        self.r2_entry.config(state='disabled')
        
        ttk.Label(self.r2_frame, text="(Fits below this will be flagged and saved separately)",
                 font=("Helvetica", 8, "italic"), foreground="gray").grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
    
    def toggle_r2_flagging(self):
        state = 'normal' if self.enable_r2_flagging_var.get() else 'disabled'
        self.r2_entry.config(state=state)
    
    def create_model_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Model Selection", padding="10")
        frame.grid(row=5, column=0, sticky="ew", pady=5)
        
        self.model_var = tk.StringVar(value="both")
        ttk.Radiobutton(frame, text="Single Richards", variable=self.model_var, 
                       value="single").pack(side="left", padx=10)
        ttk.Radiobutton(frame, text="Double Richards", variable=self.model_var, 
                       value="double").pack(side="left", padx=10)
        ttk.Radiobutton(frame, text="Both (with comparison)", variable=self.model_var, 
                       value="both").pack(side="left", padx=10)
    
    def create_action_section(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.grid(row=6, column=0, sticky="ew", pady=5)
        
        self.run_btn = ttk.Button(frame, text="Run Analysis", command=self.run_analysis,
                                  style='Accent.TButton')
        self.run_btn.pack(side="left", padx=5)
        self.run_btn.config(state='disabled')
        
        ttk.Button(frame, text="Save Results", command=self.save_results).pack(side="left", padx=5)
        ttk.Button(frame, text="Clear Log", command=self.clear_log).pack(side="left", padx=5)
        
        self.progress = ttk.Progressbar(frame, mode='indeterminate')
        self.progress.pack(side="left", fill="x", expand=True, padx=10)
    
    def create_log_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Analysis Log", padding="10")
        frame.grid(row=7, column=0, sticky="nsew", pady=5)
        
        self.log_text = scrolledtext.ScrolledText(frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)
    
    def toggle_filtering(self):
        state = 'normal' if self.enable_filtering_var.get() else 'disabled'
        self.lower_entry.config(state=state)
        self.upper_entry.config(state=state)
        self.mad_entry.config(state=state)
    
    def toggle_smoothing(self):
        state = 'normal' if self.enable_smoothing_var.get() else 'disabled'
        self.smooth_window_entry.config(state=state)
        self.median_radio.config(state=state)
        self.mean_radio.config(state=state)
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
    
    def load_file(self):
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.data = pd.read_csv(filename)
                self.data_path = Path(filename)
                self.file_label.config(text=self.data_path.name, foreground="black")
                self.update_run_button_state()
                self.log(f"Loaded: {filename}")
                self.log(f"  Rows: {len(self.data)}")
                self.log(f"  Columns: {', '.join(self.data.columns)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                self.log(f"ERROR: {str(e)}")
    
    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = Path(directory)
            self.output_label.config(text=self.output_dir.name, foreground="black")
            self.update_run_button_state()
            self.log(f"Output directory: {directory}")
    
    def update_run_button_state(self):
        if self.data is not None and self.output_dir is not None:
            self.run_btn.config(state='normal')
        else:
            self.run_btn.config(state='disabled')
    
    def run_analysis(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return
        
        if self.output_dir is None:
            messagebox.showwarning("Warning", "Please select an output directory first")
            return
        
        try:
            self.progress.start()
            self.run_btn.config(state='disabled')
            
            self.log("\n" + "="*70)
            self.log("STARTING RICHARDS CURVE FITTING ANALYSIS")
            self.log("="*70)
            
            # Get parameters
            self.pixels_per_mm = self.pixels_var.get()
            self.window_before = self.before_var.get()
            self.window_after = self.after_var.get()
            
            df = self.data.copy()
            
            # Preprocessing
            self.log("\n" + "="*70)
            self.log("PREPROCESSING")
            self.log("="*70)
            
            # Step 1: Ensure speed exists
            if "speed" not in df.columns and {"x","y","time"}.issubset(df.columns):
                self.log("Computing speed from x,y coordinates...")
                self.root.update_idletasks()  # Update GUI
                df["speed"] = calculate_speed_from_xy(
                    df["x"].to_numpy(), 
                    df["y"].to_numpy(),
                    df["time"].to_numpy(), 
                    smoothing_window=3,
                    pixels_per_mm=self.pixels_per_mm
                )
                self.log("  Speed computed in mm/s")
            elif "speed" in df.columns:
                self.log("Converting existing speed column to mm/s...")
                self.root.update_idletasks()  # Update GUI
                df = convert_speed_to_mm_s(df, pixels_per_mm=self.pixels_per_mm, assume_pixels=True)
                self.log("  Conversion complete")
            
            self.root.update_idletasks()  # Update GUI
            
            # Step 2: Outlier filtering
            if self.enable_filtering_var.get():
                self.log("\nApplying outlier filtering...")
                self.root.update_idletasks()  # Update GUI
                self.percentile_lower = self.lower_percentile_var.get()
                self.percentile_upper = self.upper_percentile_var.get()
                self.mad_threshold = self.mad_threshold_var.get()
                
                self.log(f"  Percentile bounds: {self.percentile_lower}% - {self.percentile_upper}%")
                self.log(f"  MAD threshold: {self.mad_threshold}x")
                
                n_before = len(df)
                df, self.outlier_report = apply_hybrid_outlier_filter(
                    df,
                    percentile_lower=self.percentile_lower,
                    percentile_upper=self.percentile_upper,
                    mad_threshold=self.mad_threshold,
                )
                n_after = len(df)
                n_removed = n_before - n_after
                pct_removed = (n_removed / n_before * 100) if n_before > 0 else 0
                
                self.log(f"  Removed {n_removed} outlier points ({pct_removed:.2f}%)")
                self.log(f"  Remaining: {n_after} points")
                self.root.update_idletasks()  # Update GUI
            
            # Step 3: Smoothing
            if self.enable_smoothing_var.get():
                self.log("\nApplying smoothing...")
                self.root.update_idletasks()  # Update GUI
                self.smooth_window = self.smooth_window_var.get()
                smooth_method = self.smooth_method_var.get()
                
                self.log(f"  Method: {smooth_method}")
                self.log(f"  Window: {self.smooth_window} frames")
                
                df = apply_smoothing(
                    df,
                    window=self.smooth_window,
                    method=smooth_method,
                )
                self.log("  Smoothing complete")
                self.root.update_idletasks()  # Update GUI
            
            # Extract windows
            self.log("\n" + "="*70)
            self.log("EXTRACTING WINDOWS AROUND FOOD ENCOUNTERS")
            self.log("="*70)
            self.root.update_idletasks()  # Update GUI
            
            windowed = extract_windows(df, 
                                      window_before=self.window_before,
                                      window_after=self.window_after,
                                      min_points=10)
            
            if not windowed:
                messagebox.showerror("Error", "No valid encounter windows found")
                self.log("ERROR: No encounter windows found")
                return
            
            self.log(f"Found {len(windowed)} animals with valid windows")
            self.root.update_idletasks()  # Update GUI
            
            # Fit models
            model_choice = self.model_var.get()
            
            if model_choice in ("single", "both"):
                self.log("\n" + "="*70)
                self.log("FITTING SINGLE RICHARDS MODEL")
                self.log("="*70)
                self.results_single = self.fit_model(df, windowed, "single")
                self.report_model_summary(self.results_single, "single")
            
            if model_choice in ("double", "both"):
                self.log("\n" + "="*70)
                self.log("FITTING DOUBLE RICHARDS MODEL")
                self.log("="*70)
                self.results_double = self.fit_model(df, windowed, "double")
                self.report_model_summary(self.results_double, "double")
            
            # Comparison
            if model_choice == "both" and self.results_single is not None and self.results_double is not None:
                self.log("\n" + "="*70)
                self.log("MODEL COMPARISON")
                self.log("="*70)
                self.compare_models()
            
            self.log("\n" + "="*70)
            self.log("ANALYSIS COMPLETE")
            self.log("="*70)
            
            messagebox.showinfo("Success", "Analysis completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            self.log(f"\nERROR: {str(e)}")
            self.log(traceback.format_exc())
        finally:
            self.progress.stop()
            self.run_btn.config(state='normal')
    
    def fit_model(self, df_master, windowed, model_type):
        """Fit model and save all outputs to disk"""
        # Create output directory structure
        model_dir = self.output_dir / f"model_{model_type}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        windowed_data_list = []
        
        total = len(windowed)
        for i, ((assay, track), wdf) in enumerate(windowed.items()):
            if i % 5 == 0:  # Update more frequently
                self.log(f"  Processing animal {i+1}/{total}...")
                self.root.update_idletasks()  # Update GUI
            
            t_rel = wdf["time_rel"].to_numpy()
            spd = wdf["speed"].to_numpy()
            
            if model_type == "single":
                res = fit_single(t_rel, spd)
            else:
                res = fit_double(t_rel, spd)
            
            # Add metadata
            animal_meta = df_master[(df_master["assay_num"] == assay) & 
                                   (df_master["track_num"] == track)].iloc[0]
            res["assay_num"] = assay
            res["track_num"] = track
            for col in ["sex", "strain_genotype", "treatment"]:
                res[col] = animal_meta.get(col, np.nan)
            
            results.append(res)
            
            # Store windowed data
            wdf_out = wdf.copy()
            wdf_out["assay_num"] = assay
            wdf_out["track_num"] = track
            windowed_data_list.append(wdf_out)
        
        self.root.update_idletasks()  # Update GUI
        
        df_results = pd.DataFrame(results)
        df_windowed = pd.concat(windowed_data_list, ignore_index=True)
        
        # Save CSV files with correct names for viewer compatibility
        df_results.to_csv(model_dir / "richards_fit_parameters.csv", index=False)
        df_windowed.to_csv(model_dir / "windowed_speed_data.csv", index=False)  # Correct filename!
        
        self.log(f"  Saved results to {model_dir}")
        self.root.update_idletasks()  # Update GUI
        
        # Generate parameter cheat sheet
        self.save_parameter_cheatsheet(model_dir, model_type)
        
        # Generate summary plots
        self.log(f"  Generating summary plots...")
        self.root.update_idletasks()  # Update GUI
        metrics = (["y_initial","y_final","A","B","M","nu"] if model_type=="single"
                   else ["y_initial","y_final","A","y1_i","y1_f","y2_f","delta1","delta2","B1","M1","nu1","B2","M2","nu2"])
        plot_count = 0
        for m in metrics:
            try:
                plot_summary_by_sex_strain_treatment(df_results, m, model_dir / f"{m}_by_sex_strain_treatment.png")
                plot_count += 1
            except Exception as e:
                self.log(f"    Warning: Could not generate plot for {m}: {str(e)}")
        
        self.log(f"  Generated {plot_count}/{len(metrics)} summary plots")
        
        # Generate R² distribution histogram with threshold line
        self.plot_r2_distribution(df_results, model_dir, model_type)
        
        self.root.update_idletasks()  # Update GUI
        return df_results
    
    def plot_r2_distribution(self, df_results, model_dir, model_type):
        """Generate R² distribution histogram with optional threshold line"""
        converged = df_results[df_results["converged"] == True]
        if len(converged) == 0:
            return
        
        r2_values = converged['r_squared'].dropna()
        
        plt.figure(figsize=(8, 6))
        plt.hist(r2_values, bins=30, alpha=0.7, edgecolor='black')
        
        title_str = f'{model_type.title()} Model: R² Distribution'
        
        # Only add threshold line if R² flagging is enabled
        if self.enable_r2_flagging_var.get():
            r2_threshold = self.r2_threshold_var.get()
            plt.axvline(r2_threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold = {r2_threshold}')
            
            # Count and annotate
            n_below = (r2_values < r2_threshold).sum()
            n_above = (r2_values >= r2_threshold).sum()
            
            title_str += f'\nBelow threshold: {n_below} ({n_below/len(r2_values)*100:.1f}%) | ' \
                        f'Above threshold: {n_above} ({n_above/len(r2_values)*100:.1f}%)'
            plt.legend()
        else:
            title_str += f'\nTotal converged fits: {len(r2_values)}'
        
        plt.xlabel('R²')
        plt.ylabel('Count')
        plt.title(title_str)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_dir / "r2_distribution.png", bbox_inches="tight", dpi=150)
        plt.close()
    
    def save_parameter_cheatsheet(self, model_dir, model_type):
        """Save parameter explanation cheat sheet"""
        cheat_path = model_dir / "PARAMETER_CHEATSHEET.txt"
        with open(cheat_path, "w") as f:
            f.write(f"Richards Curve Fitting - {model_type.upper()} Model\n")
            f.write("="*60 + "\n\n")
            if model_type == "single":
                f.write("Single Richards: y(t) = y_initial + A / (1 + exp(-B(t - M)))^(1/nu)\n\n")
                f.write("Parameters:\n")
                f.write("  y_initial: baseline speed before encounter\n")
                f.write("  y_final: asymptotic speed after encounter\n")
                f.write("  A = y_final - y_initial: total speed change\n")
                f.write("  B: steepness of transition (higher = faster change)\n")
                f.write("  M: midpoint time of transition (seconds)\n")
                f.write("  nu: asymmetry parameter (nu=1 => symmetric logistic)\n\n")
            else:
                f.write("Double Richards: two sequential transitions\n")
                f.write("  Phase 1: y1_i -> y1_f (B1, M1, nu1)\n")
                f.write("  Phase 2: y1_f -> y2_f (B2, M2, nu2)\n\n")
                f.write("Parameters:\n")
                f.write("  y_initial = y1_i: initial speed\n")
                f.write("  y_final = y2_f: final asymptotic speed\n")
                f.write("  A = y2_f - y1_i: total speed change\n")
                f.write("  delta1 = y1_f - y1_i: phase 1 change\n")
                f.write("  delta2 = y2_f - y1_f: phase 2 change\n")
                f.write("  B1, M1, nu1: phase 1 steepness, midpoint, asymmetry\n")
                f.write("  B2, M2, nu2: phase 2 steepness, midpoint, asymmetry\n\n")
            f.write("Fit Quality:\n")
            f.write("  R²: proportion of variance explained (0–1, higher is better)\n")
            f.write("  RMSE: root mean squared error (lower is better)\n")
            f.write("  AIC: Akaike Information Criterion (lower is better)\n")
            f.write("  BIC: Bayesian Information Criterion (lower is better, penalizes complexity)\n")
    
    def report_model_summary(self, results_df, model):
        converged = results_df[results_df["converged"] == True]
        self.log(f"\n{model.upper()} MODEL SUMMARY:")
        self.log(f"  Total animals: {len(results_df)}")
        self.log(f"  Converged fits: {len(converged)} ({len(converged)/len(results_df)*100:.1f}%)")
        if len(converged) > 0:
            self.log(f"  Mean R²: {converged['r_squared'].mean():.3f}")
            self.log(f"  Median R²: {converged['r_squared'].median():.3f}")
            
            # R² threshold analysis - only if enabled
            if self.enable_r2_flagging_var.get():
                r2_threshold = self.r2_threshold_var.get()
                poor_fits = converged[converged['r_squared'] < r2_threshold]
                n_poor = len(poor_fits)
                pct_poor = (n_poor / len(converged) * 100) if len(converged) > 0 else 0
                
                self.log(f"\n  R² THRESHOLD ANALYSIS (threshold = {r2_threshold}):")
                self.log(f"  Fits below threshold: {n_poor} ({pct_poor:.1f}%)")
                self.log(f"  Fits above threshold: {len(converged) - n_poor} ({100-pct_poor:.1f}%)")
                
                if n_poor > 0:
                    self.log(f"  Mean R² of poor fits: {poor_fits['r_squared'].mean():.3f}")
                    self.log(f"  Range of poor fits: {poor_fits['r_squared'].min():.3f} - {poor_fits['r_squared'].max():.3f}")
                    
                    # Save poor fits report
                    model_dir = self.output_dir / f"model_{model}"
                    poor_fits_path = model_dir / f"poor_fits_below_r2_{r2_threshold}.csv"
                    poor_fits.to_csv(poor_fits_path, index=False)
                    self.log(f"  Saved poor fits report: {poor_fits_path.name}")
    
    def compare_models(self):
        """Compare single vs double models and save comparison files"""
        m = pd.merge(
            self.results_single[["assay_num","track_num","r_squared","aic","bic"]],
            self.results_double[["assay_num","track_num","r_squared","aic","bic"]],
            on=["assay_num","track_num"], 
            suffixes=("_single","_double"), 
            how="inner"
        )
        
        if m.empty:
            self.log("No overlapping animals for comparison")
            return
        
        m["ΔR²"] = m["r_squared_double"] - m["r_squared_single"]
        m["ΔAIC"] = m["aic_single"] - m["aic_double"]
        m["ΔBIC"] = m["bic_single"] - m["bic_double"]
        
        # Add preferred model based on BIC
        def pref_from_delta_bic(delta_bic):
            if pd.isna(delta_bic): return "Unknown"
            if delta_bic > 10: return "Double"
            if delta_bic < -10: return "Single"
            return "Similar"
        m["preferred_model"] = m["ΔBIC"].apply(pref_from_delta_bic)
        
        # Save comparison summary
        summary_path = self.output_dir / "model_comparison_summary.csv"
        m.to_csv(summary_path, index=False)
        self.log(f"  Saved comparison summary: {summary_path.name}")
        
        # Annotate individual model CSVs with comparison data
        single_path = self.output_dir / "model_single" / "richards_fit_parameters.csv"
        double_path = self.output_dir / "model_double" / "richards_fit_parameters.csv"
        
        if single_path.exists():
            ds = pd.read_csv(single_path)
            ds_aug = pd.merge(ds, m[["assay_num","track_num","r_squared_double","aic_double","bic_double","ΔR²","ΔAIC","ΔBIC","preferred_model"]],
                            on=["assay_num","track_num"], how="left")
            ds_aug.rename(columns={"r_squared_double":"r_squared_other","aic_double":"aic_other","bic_double":"bic_other",
                                  "ΔR²":"delta_r2","ΔAIC":"delta_aic","ΔBIC":"delta_bic"}, inplace=True)
            ds_aug.to_csv(single_path, index=False)
        
        if double_path.exists():
            dd = pd.read_csv(double_path)
            dd_aug = pd.merge(dd, m[["assay_num","track_num","r_squared_single","aic_single","bic_single","ΔR²","ΔAIC","ΔBIC","preferred_model"]],
                            on=["assay_num","track_num"], how="left")
            dd_aug.rename(columns={"r_squared_single":"r_squared_other","aic_single":"aic_other","bic_single":"bic_other",
                                  "ΔR²":"delta_r2","ΔAIC":"delta_aic","ΔBIC":"delta_bic"}, inplace=True)
            dd_aug.to_csv(double_path, index=False)
        
        # Generate comparison plots
        self.log(f"  Generating comparison plots...")
        self.save_comparison_plots(m)
        
        # Log statistics
        self.log(f"  Compared {len(m)} animals")
        self.log(f"  Mean ΔR² (Double - Single): {m['ΔR²'].mean():.4f}")
        self.log(f"  Mean ΔBIC (Single - Double): {m['ΔBIC'].mean():.2f}")
        
        # Preference based on BIC
        double_better = (m["ΔBIC"] > 10).sum()
        single_better = (m["ΔBIC"] < -10).sum()
        similar = len(m) - double_better - single_better
        
        self.log(f"\nModel Preference (by BIC):")
        self.log(f"  Double better: {double_better} ({double_better/len(m)*100:.1f}%)")
        self.log(f"  Single better: {single_better} ({single_better/len(m)*100:.1f}%)")
        self.log(f"  Similar: {similar} ({similar/len(m)*100:.1f}%)")
    
    def save_comparison_plots(self, m):
        """Generate and save model comparison plots"""
        # R² scatter
        self.root.update_idletasks()  # Update GUI
        plt.figure(figsize=(6,6))
        plt.scatter(m["r_squared_single"], m["r_squared_double"], alpha=0.55, edgecolors="black", linewidths=0.4)
        plt.plot([0,1],[0,1], "--", color="gray")
        plt.xlim(0,1); plt.ylim(0,1)
        plt.xlabel("R² (Single)"); plt.ylabel("R² (Double)")
        plt.title("R² Comparison Per Animal")
        plt.tight_layout()
        plt.savefig(self.output_dir / "r2_scatter_single_vs_double.png", bbox_inches="tight", dpi=150)
        plt.close()
        
        # R² violin
        self.root.update_idletasks()  # Update GUI
        plt.figure(figsize=(6,5))
        r2s = m["r_squared_single"].dropna().values
        r2d = m["r_squared_double"].dropna().values
        parts = plt.violinplot([r2s, r2d], positions=[1,2], showmeans=True, showmedians=True)
        for body in parts["bodies"]:
            body.set_alpha(0.75)
        plt.xticks([1,2], ["Single","Double"])
        plt.ylabel("R²"); plt.title("R² Distribution by Model")
        plt.tight_layout()
        plt.savefig(self.output_dir / "r2_violin_single_vs_double.png", bbox_inches="tight", dpi=150)
        plt.close()
        
        # ΔAIC / ΔBIC histograms
        for col, fname in [("ΔAIC","delta_AIC_hist.png"), ("ΔBIC","delta_BIC_hist.png")]:
            self.root.update_idletasks()  # Update GUI
            plt.figure(figsize=(7,5))
            vals = m[col].dropna().values
            if len(vals): 
                plt.hist(vals, bins=30)
            plt.axvline(0, color="black", linestyle="--")
            plt.xlabel(f"{col} (Single − Double)")
            plt.ylabel("Count")
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(self.output_dir / fname, bbox_inches="tight", dpi=150)
            plt.close()
    
    def save_results(self):
        """Save preprocessing summary (main results already saved during analysis)"""
        if self.output_dir is None:
            messagebox.showwarning("Warning", "No output directory selected")
            return
        
        if self.results_single is None and self.results_double is None:
            messagebox.showwarning("Warning", "No results to save. Run analysis first.")
            return
        
        try:
            # Save outlier report if filtering was used
            if self.outlier_report is not None and not self.outlier_report.empty:
                outlier_path = self.output_dir / "outlier_filtering_report.csv"
                self.outlier_report.to_csv(outlier_path, index=False)
                self.log(f"Saved outlier report: {outlier_path}")
            
            # Save preprocessing summary
            n_removed = 0
            pct_removed = 0
            if self.outlier_report is not None and not self.outlier_report.empty:
                n_removed = self.outlier_report['n_outliers'].sum()
                n_total = self.outlier_report['n_points'].sum()
                pct_removed = (n_removed / n_total * 100) if n_total > 0 else 0
            
            preproc_summary = {
                "filtering_enabled": self.enable_filtering_var.get(),
                "smoothing_enabled": self.enable_smoothing_var.get(),
                "pixels_per_mm": self.pixels_per_mm,
                "window_before": self.window_before,
                "window_after": self.window_after,
            }
            if self.enable_filtering_var.get():
                preproc_summary.update({
                    "percentile_lower": self.percentile_lower,
                    "percentile_upper": self.percentile_upper,
                    "mad_threshold": self.mad_threshold,
                    "points_removed": n_removed,
                    "pct_removed": pct_removed,
                })
            if self.enable_smoothing_var.get():
                preproc_summary.update({
                    "smooth_method": self.smooth_method_var.get(),
                    "smooth_window": self.smooth_window,
                })
            
            pd.DataFrame([preproc_summary]).to_csv(self.output_dir / "preprocessing_summary.csv", index=False)
            self.log(f"Saved preprocessing summary: preprocessing_summary.csv")
            
            messagebox.showinfo("Success", f"Additional files saved to:\n{self.output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save files:\n{str(e)}")
            self.log(f"ERROR saving files: {str(e)}")


def main():
    root = tk.Tk()
    app = RichardsCurveFitterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()