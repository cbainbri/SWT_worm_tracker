#!/usr/bin/env python3
"""
C. elegans Behavioral Context Analyzer
Calculates Food Detection and Food Encounter contexts with outlier detection
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
from typing import Dict, Tuple, List, Optional
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
warnings.filterwarnings('ignore')


class BehavioralContextAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("C. elegans Behavioral Context Analyzer")
        self.root.geometry("900x800")

        # Data storage
        self.off_food_data = None
        self.on_food_data = None
        self.off_food_path = None
        self.on_food_path = None

        # Results storage
        self.results_df = None
        self.group_summary_df = None
        self.outlier_report_df = None
        self.speed_traces_df = None  # Store full traces for plotting

        # Analysis mode
        self.analysis_mode = tk.StringVar(value="both")

        # Parameters
        self.pixels_per_mm = 104
        self.before_window = (-15, -5)
        self.after_window = (5, 15)
        self.percentile_lower = 1.25
        self.percentile_upper = 98.75
        self.mad_threshold = 2.5

        self.setup_gui()

    # ---------- GUI ----------
    def setup_gui(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        ttk.Label(main, text="Behavioral Context Analyzer",
                  font=("Helvetica", 16, "bold")).grid(row=0, column=0, pady=10)

        self.create_file_section(main)
        self.create_parameters_section(main)
        self.create_analysis_section(main)
        self.create_log_section(main)

    def create_file_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Data Files", padding="10")
        frame.grid(row=1, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)

        mode = ttk.Frame(frame)
        mode.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(mode, text="Analysis Mode:",
                  font=("Helvetica", 10, "bold")).pack(side="left", padx=5)
        ttk.Radiobutton(mode, text="Both Contexts (Food Detection + Encounter)",
                        variable=self.analysis_mode, value="both",
                        command=self.on_mode_change).pack(side="left", padx=10)
        ttk.Radiobutton(mode, text="Food Encounter Only (ON-food only)",
                        variable=self.analysis_mode, value="encounter_only",
                        command=self.on_mode_change).pack(side="left", padx=10)
        ttk.Separator(frame, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=5)

        ttk.Label(frame, text="OFF-food data:").grid(
            row=2, column=0, sticky="w", pady=5)
        self.off_food_label = ttk.Label(
            frame, text="No file loaded", foreground="gray")
        self.off_food_label.grid(row=2, column=1, sticky="w", padx=5)
        self.off_food_btn = ttk.Button(
            frame, text="Load OFF-food CSV", command=self.load_off_food)
        self.off_food_btn.grid(row=2, column=2, padx=5)

        ttk.Label(frame, text="ON-food data:").grid(
            row=3, column=0, sticky="w", pady=5)
        self.on_food_label = ttk.Label(
            frame, text="No file loaded", foreground="gray")
        self.on_food_label.grid(row=3, column=1, sticky="w", padx=5)
        ttk.Button(frame, text="Load ON-food CSV",
                   command=self.load_on_food).grid(row=3, column=2, padx=5)

        self.data_info_label = ttk.Label(frame, text="", foreground="blue")
        self.data_info_label.grid(row=4, column=0, columnspan=3, pady=5)

    def on_mode_change(self):
        mode = self.analysis_mode.get()
        if mode == "encounter_only":
            self.off_food_btn.config(state="disabled")
            self.off_food_label.config(
                text="Not required for Encounter Only mode", foreground="gray")
            self.log("Mode: Encounter only — OFF-food data not required")
        else:
            self.off_food_btn.config(state="normal")
            if self.off_food_data is None:
                self.off_food_label.config(text="No file loaded",
                                           foreground="gray")
            self.log("Mode: Both Contexts — need OFF- and ON-food data")
        self.check_ready_for_analysis()

    def create_parameters_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Analysis Parameters", padding="10")
        frame.grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Label(frame, text="Outlier Detection:",
                  font=("Helvetica", 10, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(frame, text="Lower percentile (%):").grid(
            row=1, column=0, sticky="w")
        self.lower_percentile_var = tk.DoubleVar(value=1.25)
        ttk.Entry(frame, textvariable=self.lower_percentile_var,
                  width=10).grid(row=1, column=1)
        ttk.Label(frame, text="Upper percentile (%):").grid(
            row=1, column=2, sticky="w", padx=(20, 0))
        self.upper_percentile_var = tk.DoubleVar(value=98.75)
        ttk.Entry(frame, textvariable=self.upper_percentile_var,
                  width=10).grid(row=1, column=3)
        ttk.Label(frame, text="MAD threshold:").grid(row=2, column=0, sticky="w")
        self.mad_threshold_var = tk.DoubleVar(value=2.5)
        ttk.Entry(frame, textvariable=self.mad_threshold_var,
                  width=10).grid(row=2, column=1)

        ttk.Separator(frame, orient="horizontal").grid(
            row=3, column=0, columnspan=4, sticky="ew", pady=10)

        ttk.Label(frame, text="Speed Smoothing:",
                  font=("Helvetica", 10, "bold")).grid(row=4, column=0, columnspan=4, sticky="w")
        sm = ttk.Frame(frame)
        sm.grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Label(sm, text="Method:").pack(side="left", padx=(0, 10))
        self.smooth_method_var = tk.StringVar(value="mean")
        ttk.Radiobutton(sm, text="Mean", variable=self.smooth_method_var,
                        value="mean").pack(side="left")
        ttk.Radiobutton(sm, text="Median", variable=self.smooth_method_var,
                        value="median").pack(side="left")
        ttk.Label(frame, text="Window (frames):").grid(
            row=5, column=2, sticky="w", padx=(20, 0))
        self.smooth_window_var = tk.IntVar(value=1)
        ttk.Entry(frame, textvariable=self.smooth_window_var,
                  width=10).grid(row=5, column=3)
        ttk.Label(frame, text="(1 = no smoothing)", font=("Helvetica", 8, "italic"),
                  foreground="gray").grid(row=6, column=2, columnspan=2, sticky="w", padx=(20, 0))

    def create_analysis_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Analysis", padding="10")
        frame.grid(row=3, column=0, sticky="ew", pady=5)
        frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew", pady=5)
        bf = ttk.Frame(frame)
        bf.grid(row=1, column=0)
        self.analyze_btn = ttk.Button(
            bf, text="Run Analysis", command=self.run_analysis, state="disabled")
        self.analyze_btn.grid(row=0, column=0, padx=5)
        self.export_btn = ttk.Button(
            bf, text="Export Results", command=self.export_results, state="disabled")
        self.export_btn.grid(row=0, column=1, padx=5)
        self.plot_btn = ttk.Button(
            bf, text="Generate Plots", command=self.generate_plots, state="disabled")
        self.plot_btn.grid(row=0, column=2, padx=5)

    def create_log_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Analysis Log", padding="10")
        frame.grid(row=4, column=0, sticky="nsew", pady=5)
        parent.rowconfigure(4, weight=1)
        self.log_text = scrolledtext.ScrolledText(frame, height=15, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    # ---------- DATA LOADING ----------
    def load_off_food(self):
        """Load OFF-food CSV file"""
        filepath = filedialog.askopenfilename(
            title="Select OFF-food CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.off_food_data = pd.read_csv(filepath)
                self.off_food_path = filepath
                filename = Path(filepath).name
                self.off_food_label.config(text=filename, foreground="green")
                self.log(f"Loaded OFF-food data: {filename}")
                self.log(f"  Shape: {self.off_food_data.shape}")
                self.update_data_info()
                self.check_ready_for_analysis()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load OFF-food data:\n{str(e)}")
                self.log(f"ERROR loading OFF-food data: {str(e)}")

    def load_on_food(self):
        """Load ON-food CSV file"""
        filepath = filedialog.askopenfilename(
            title="Select ON-food CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.on_food_data = pd.read_csv(filepath)
                self.on_food_path = filepath
                filename = Path(filepath).name
                self.on_food_label.config(text=filename, foreground="green")
                self.log(f"Loaded ON-food data: {filename}")
                self.log(f"  Shape: {self.on_food_data.shape}")
                self.update_data_info()
                self.check_ready_for_analysis()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ON-food data:\n{str(e)}")
                self.log(f"ERROR loading ON-food data: {str(e)}")

    def update_data_info(self):
        """Show counts of animals if both datasets loaded"""
        if self.off_food_data is not None and self.on_food_data is not None:
            off_animals = self.count_animals(self.off_food_data)
            on_animals = self.count_animals(self.on_food_data)
            info = f"OFF-food: {off_animals} animals | ON-food: {on_animals} animals"
            self.data_info_label.config(text=info)

    def count_animals(self, df):
        """Count unique animals by assay, track, pc_number"""
        return df.groupby(['assay_num', 'track_num', 'pc_number']).ngroups

    def check_ready_for_analysis(self):
        """Enable analysis when correct datasets loaded"""
        mode = self.analysis_mode.get()
        if (mode == "encounter_only" and self.on_food_data is not None) or \
           (mode == "both" and self.off_food_data is not None and self.on_food_data is not None):
            self.analyze_btn.config(state='normal')
        else:
            self.analyze_btn.config(state='disabled')

    # ---------- CORE CALCULATIONS ----------
    def calculate_speed(self, df):
        """Calculate instantaneous speed in mm/s (IDs now globally unique with condition)"""
        self.log("  Calculating speeds...")

        # Ensure condition is present
        if 'condition' not in df.columns:
            df['condition'] = 'unknown'

        # Include condition in the ID to make it globally unique
        df['animal_id'] = (
            df['condition'].astype(str) + '_' +
            df['assay_num'].astype(str) + '_' +
            df['track_num'].astype(str) + '_' +
            df['pc_number'].astype(str)
        )

        # Smoothing parameters
        smooth_window = self.smooth_window_var.get()
        smooth_method = self.smooth_method_var.get()

        speeds = []
        for animal_id, group in df.groupby('animal_id'):
            group = group.sort_values('time').reset_index(drop=True)
            dx = group['x'].diff()
            dy = group['y'].diff()
            dist_pixels = np.sqrt(dx**2 + dy**2)
            speed_mm_s = dist_pixels / self.pixels_per_mm

            if smooth_window > 1:
                if smooth_method == "mean":
                    speed_mm_s = speed_mm_s.rolling(window=smooth_window, center=True, min_periods=1).mean()
                else:
                    speed_mm_s = speed_mm_s.rolling(window=smooth_window, center=True, min_periods=1).median()

            group['speed'] = speed_mm_s
            speeds.append(group)
        return pd.concat(speeds, ignore_index=True)

    # ---------- MAIN ANALYSIS ----------
    def run_analysis(self):
        """Run full analysis pipeline"""
        try:
            self.progress.start()
            self.analyze_btn.config(state='disabled')
            self.log("\n" + "="*60)
            self.log("Starting analysis...")
            self.log("="*60)

            # Update parameters
            self.percentile_lower = self.lower_percentile_var.get()
            self.percentile_upper = self.upper_percentile_var.get()
            self.mad_threshold = self.mad_threshold_var.get()
            mode = self.analysis_mode.get()
            self.log(f"\nAnalysis Mode: {mode}")

            # ---------------- OFF-food ----------------
            if mode == "both":
                self.log("\nProcessing OFF-food data...")
                off_food_df = self.off_food_data.copy()
                off_food_df['condition'] = 'OFF_food'
                off_food_df = self.calculate_speed(off_food_df)
                off_encounter = self.find_food_encounter(off_food_df)
                off_windows = self.extract_time_windows(off_food_df, off_encounter)
                off_windows['condition'] = 'OFF_food'
                off_traces = self.extract_speed_traces(off_food_df, off_encounter)
                off_traces['condition'] = 'OFF_food'
                self.log(f"  Found {len(off_windows)} OFF-food animals with valid data")

                self.log("\nDetecting outliers in OFF-food 'before' speeds...")
                off_windows = self.detect_outliers_parallel(off_windows, 'before_mean_speed',
                                                           ['treatment', 'sex', 'genotype'])
                off_outliers = off_windows[off_windows['is_outlier']].copy()
                off_outliers['outlier_context'] = 'before'
                self.log(f"  Flagged {len(off_outliers)} OFF-food outliers")

                off_windows_clean = off_windows[~off_windows['is_outlier']].copy()
                self.log(f"  OFF-food clean: {len(off_windows_clean)} animals")

            # ---------------- ON-food ----------------
            self.log("\nProcessing ON-food data...")
            on_food_df = self.on_food_data.copy()
            on_food_df['condition'] = 'ON_food'
            on_food_df = self.calculate_speed(on_food_df)
            on_encounter = self.find_food_encounter(on_food_df)
            on_windows = self.extract_time_windows(on_food_df, on_encounter)
            on_windows['condition'] = 'ON_food'
            on_traces = self.extract_speed_traces(on_food_df, on_encounter)
            on_traces['condition'] = 'ON_food'
            self.log(f"  Found {len(on_windows)} ON-food animals with valid data")

            # ---------- Uniqueness Check ----------
            if mode == "both":
                duplicate_ids = set(off_food_df['animal_id']).intersection(set(on_food_df['animal_id']))
                if duplicate_ids:
                    self.log(f"⚠️ WARNING: {len(duplicate_ids)} overlapping animal IDs found!")
                else:
                    self.log("✅ All animal IDs are unique across ON and OFF food datasets.")
            # ---------- ON-food Outliers ----------
            self.log("\nDetecting outliers in ON-food 'before' speeds...")
            on_windows_before = self.detect_outliers_parallel(
                on_windows, 'before_mean_speed', ['treatment', 'sex', 'genotype']
            )
            before_outliers = on_windows_before[on_windows_before['is_outlier']]['animal_id'].tolist()
            self.log(f"  Flagged {len(before_outliers)} ON-food animals in 'before' context")

            self.log("\nDetecting outliers in ON-food 'after' speeds...")
            on_windows_after = self.detect_outliers_parallel(
                on_windows, 'after_mean_speed', ['treatment', 'sex', 'genotype']
            )
            after_outliers = on_windows_after[on_windows_after['is_outlier']]['animal_id'].tolist()
            self.log(f"  Flagged {len(after_outliers)} ON-food animals in 'after' context")

            all_on_outliers = list(set(before_outliers + after_outliers))
            self.log(f"  Total ON-food outliers (union): {len(all_on_outliers)}")

            # Mark outlier context for ON-food
            on_windows['is_outlier'] = on_windows['animal_id'].isin(all_on_outliers)
            on_windows['outlier_context'] = 'none'
            on_windows.loc[
                on_windows['animal_id'].isin(before_outliers) & on_windows['animal_id'].isin(after_outliers),
                'outlier_context'
            ] = 'both'
            on_windows.loc[
                on_windows['animal_id'].isin(before_outliers) & ~on_windows['animal_id'].isin(after_outliers),
                'outlier_context'
            ] = 'before'
            on_windows.loc[
                ~on_windows['animal_id'].isin(before_outliers) & on_windows['animal_id'].isin(after_outliers),
                'outlier_context'
            ] = 'after'

            # Outlier report & cleaning
            on_outliers = on_windows[on_windows['is_outlier']].copy()
            if mode == "both":
                self.outlier_report_df = pd.concat([off_outliers, on_outliers], ignore_index=True)
            else:
                self.outlier_report_df = on_outliers

            self.log("\nRemoving outliers from datasets...")
            on_windows_clean = on_windows[~on_windows['is_outlier']].copy()
            self.log(f"  ON-food clean: {len(on_windows_clean)} animals")

            # Clean traces too
            on_traces_clean = on_traces[~on_traces['animal_id'].isin(all_on_outliers)].copy()
            if mode == "both":
                off_outlier_ids = off_outliers['animal_id'].tolist()
                off_traces_clean = off_traces[~off_traces['animal_id'].isin(off_outlier_ids)].copy()
                self.speed_traces_df = pd.concat([off_traces_clean, on_traces_clean], ignore_index=True)
            else:
                self.speed_traces_df = on_traces_clean
            self.log(f"  Speed traces stored for {self.speed_traces_df['animal_id'].nunique()} animals")

            # ---------- Food Detection Context (requires OFF + ON) ----------
            if mode == "both":
                self.log("\n" + "-"*60)
                self.log("Calculating Food Detection Context...")
                self.log("-"*60)

                # OFF-food baselines per (treatment, sex, genotype)
                off_baselines = off_windows_clean.groupby(
                    ['treatment', 'sex', 'genotype']
                )['before_mean_speed'].mean().reset_index()
                off_baselines.columns = ['treatment', 'sex', 'genotype', 'off_food_baseline']

                self.log("\nOFF-food baselines by group:")
                for _, r in off_baselines.iterrows():
                    self.log(f"  {r['treatment']}, {r['sex']}, {r['genotype']}: {r['off_food_baseline']:.4f} mm/s")

                # Attach baselines to ON-food animals
                on_windows_clean = on_windows_clean.merge(
                    off_baselines, on=['treatment', 'sex', 'genotype'], how='left'
                )
                # Detection score = ON before / OFF baseline
                on_windows_clean['food_detection_score'] = on_windows_clean['before_mean_speed'] / on_windows_clean['off_food_baseline']
                self.log(f"\nCalculated food detection scores for {len(on_windows_clean)} ON-food animals")
            else:
                on_windows_clean['off_food_baseline'] = np.nan
                on_windows_clean['food_detection_score'] = np.nan
                self.log("\nSkipping Food Detection Context (Encounter Only mode)")

            # ---------- Food Encounter Context (always) ----------
            self.log("\n" + "-"*60)
            self.log("Calculating Food Encounter Context (ON-food only)...")
            self.log("-"*60)

            # ON-food before baselines per group
            on_baselines = on_windows_clean.groupby(
                ['treatment', 'sex', 'genotype']
            )['before_mean_speed'].mean().reset_index()
            on_baselines.columns = ['treatment', 'sex', 'genotype', 'on_food_before_baseline']

            self.log("\nON-food before baselines by group:")
            for _, r in on_baselines.iterrows():
                self.log(f"  {r['treatment']}, {r['sex']}, {r['genotype']}: {r['on_food_before_baseline']:.4f} mm/s")

            on_windows_clean = on_windows_clean.merge(
                on_baselines, on=['treatment', 'sex', 'genotype'], how='left'
            )
            # Encounter score = ON after / ON before baseline
            on_windows_clean['food_encounter_score'] = on_windows_clean['after_mean_speed'] / on_windows_clean['on_food_before_baseline']
            self.log(f"\nCalculated food encounter scores for {len(on_windows_clean)} ON-food animals")

            # ---------- Combine per-animal results ----------
            if mode == "both":
                # Pad OFF data with NaNs for context fields
                off_windows_clean['food_detection_score'] = np.nan
                off_windows_clean['food_encounter_score'] = np.nan
                off_windows_clean['off_food_baseline'] = np.nan
                off_windows_clean['on_food_before_baseline'] = np.nan
                self.results_df = pd.concat([off_windows_clean, on_windows_clean], ignore_index=True)
            else:
                self.results_df = on_windows_clean

            # ---------- Group Summaries ----------
            self.log("\n" + "-"*60)
            self.log("Calculating group summaries...")
            self.log("-"*60)
            self.calculate_group_summaries()

            self.log("\n" + "="*60)
            self.log("ANALYSIS COMPLETE!")
            self.log("="*60)
            self.log("\nResults ready for export.")
            self.export_btn.config(state='normal')
            self.plot_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}\n\n{traceback.format_exc()}")
            self.log(f"\nERROR: {str(e)}")
            self.log(traceback.format_exc())
        finally:
            self.progress.stop()
            self.analyze_btn.config(state='normal')

    # ---------- HELPERS / STATS ----------
    def sort_treatments(self, treatments):
        order_map = {'fed': 0, '30min': 1, '3hr': 2}
        return sorted(treatments, key=lambda x: (order_map.get(x, 100), x))

    def find_food_encounter(self, df):
        """Find first timepoint where 'food_encounter'=='food' or 'nose_on_food'==1 per animal"""
        self.log("  Finding food encounter timepoints...")
        encounter_times = []
        for animal_id, group in df.groupby('animal_id'):
            food_idx = group[group['food_encounter'] == 'food'].index
            if len(food_idx) > 0:
                encounter_time = group.loc[food_idx[0], 'time']
            else:
                nose_idx = group[group['nose_on_food'] == 1].index
                encounter_time = group.loc[nose_idx[0], 'time'] if len(nose_idx) > 0 else None
            encounter_times.append({'animal_id': animal_id, 'encounter_time': encounter_time})
        return pd.DataFrame(encounter_times)

    def extract_time_windows(self, df, encounter_df):
        """Compute mean speeds in pre/post windows around encounter per animal"""
        self.log("  Extracting time windows...")
        results = []
        for _, row in encounter_df.iterrows():
            animal_id = row['animal_id']
            t0 = row['encounter_time']
            if pd.isna(t0):
                continue
            animal = df[df['animal_id'] == animal_id].copy()
            animal['time_relative'] = animal['time'] - t0

            before = animal[(animal['time_relative'] >= self.before_window[0]) &
                            (animal['time_relative'] <= self.before_window[1])]
            after = animal[(animal['time_relative'] >= self.after_window[0]) &
                           (animal['time_relative'] <= self.after_window[1])]

            before_mean = before['speed'].mean() if len(before) > 0 else np.nan
            after_mean = after['speed'].mean() if len(after) > 0 else np.nan

            sample = animal.iloc[0]
            results.append({
                'animal_id': animal_id,
                'assay_num': sample['assay_num'],
                'track_num': sample['track_num'],
                'pc_number': sample['pc_number'],
                'treatment': sample['treatment'],
                'sex': sample['sex'],
                'genotype': sample['strain_genotype'],
                'before_mean_speed': before_mean,
                'after_mean_speed': after_mean
            })
        return pd.DataFrame(results)

    def extract_speed_traces(self, df, encounter_df, trace_window=(-30, 30)):
        """Extract full speed traces per animal for plotting around encounter"""
        self.log(f"  Extracting speed traces from {trace_window[0]} to {trace_window[1]} seconds...")
        all_traces = []
        for _, row in encounter_df.iterrows():
            animal_id = row['animal_id']
            t0 = row['encounter_time']
            if pd.isna(t0):
                continue
            animal = df[df['animal_id'] == animal_id].copy()
            animal['time_relative'] = animal['time'] - t0
            trace = animal[(animal['time_relative'] >= trace_window[0]) &
                           (animal['time_relative'] <= trace_window[1])].copy()
            if len(trace) == 0:
                continue
            sample = animal.iloc[0]
            trace['animal_id'] = animal_id
            trace['treatment'] = sample['treatment']
            trace['sex'] = sample['sex']
            trace['genotype'] = sample['strain_genotype']
            all_traces.append(trace[['animal_id', 'treatment', 'sex', 'genotype', 'time_relative', 'speed']])
        return pd.concat(all_traces, ignore_index=True) if len(all_traces) > 0 else pd.DataFrame()

    def detect_outliers_parallel(self, df, value_col, group_vars):
        """Flag outliers that are extreme by BOTH percentile and MAD within each group"""
        df = df.copy()
        df['is_outlier'] = False
        df['flagged_by'] = ''
        for _, group in df.groupby(group_vars):
            values = group[value_col].dropna()
            if len(values) < 3:
                continue
            lower_pct = np.percentile(values, self.percentile_lower)
            upper_pct = np.percentile(values, self.percentile_upper)
            pct_out = (values < lower_pct) | (values > upper_pct)

            median_val = np.median(values)
            mad = np.median(np.abs(values - median_val))
            if mad > 0:
                lower_mad = median_val - self.mad_threshold * mad
                upper_mad = median_val + self.mad_threshold * mad
                mad_out = (values < lower_mad) | (values > upper_mad)
            else:
                mad_out = pd.Series(False, index=values.index)

            inter = pct_out & mad_out
            out_idx = values[inter].index
            df.loc[out_idx, 'is_outlier'] = True
            for idx in out_idx:
                methods = []
                if pct_out.loc[idx]: methods.append('percentile')
                if mad_out.loc[idx]: methods.append('MAD')
                df.loc[idx, 'flagged_by'] = '+'.join(methods)
        return df

    def calculate_group_summaries(self):
        """Summaries by condition × (treatment, sex, genotype)"""
        summaries = []
        mode = self.analysis_mode.get()
        conditions = ['OFF_food', 'ON_food'] if mode == "both" else ['ON_food']
        for condition in conditions:
            sub = self.results_df[self.results_df['condition'] == condition]
            for (treatment, sex, genotype), group in sub.groupby(['treatment', 'sex', 'genotype']):
                n_animals = len(group)
                n_outliers = len(self.outlier_report_df[
                    (self.outlier_report_df['condition'] == condition) &
                    (self.outlier_report_df['treatment'] == treatment) &
                    (self.outlier_report_df['sex'] == sex) &
                    (self.outlier_report_df['genotype'] == genotype)
                ]) if self.outlier_report_df is not None else 0
                summary = {
                    'condition': condition,
                    'treatment': treatment,
                    'sex': sex,
                    'genotype': genotype,
                    'n_animals': n_animals,
                    'n_outliers_removed': n_outliers,
                    'mean_before_speed': group['before_mean_speed'].mean(),
                    'sem_before_speed': group['before_mean_speed'].sem(),
                }
                if condition == 'ON_food':
                    summary.update({
                        'mean_after_speed': group['after_mean_speed'].mean(),
                        'sem_after_speed': group['after_mean_speed'].sem(),
                        'mean_food_detection': group['food_detection_score'].mean(),
                        'sem_food_detection': group['food_detection_score'].sem(),
                        'mean_food_encounter': group['food_encounter_score'].mean(),
                        'sem_food_encounter': group['food_encounter_score'].sem(),
                    })
                else:
                    summary.update({
                        'mean_after_speed': np.nan,
                        'sem_after_speed': np.nan,
                        'mean_food_detection': np.nan,
                        'sem_food_detection': np.nan,
                        'mean_food_encounter': np.nan,
                        'sem_food_encounter': np.nan,
                    })
                summaries.append(summary)
        self.group_summary_df = pd.DataFrame(summaries)

        self.log("\nGroup Summaries:")
        for _, row in self.group_summary_df.iterrows():
            self.log(f"\n  {row['condition']} - {row['treatment']}, {row['sex']}, {row['genotype']}")
            self.log(f"    N = {row['n_animals']} (removed {row['n_outliers_removed']} outliers)")
            self.log(f"    Before speed: {row['mean_before_speed']:.4f} ± {row['sem_before_speed']:.4f} mm/s")
            if row['condition'] == 'ON_food':
                self.log(f"    After speed: {row['mean_after_speed']:.4f} ± {row['sem_after_speed']:.4f} mm/s")
                if mode == "both":
                    self.log(f"    Food detection: {row['mean_food_detection']:.4f} ± {row['sem_food_detection']:.4f}")
                self.log(f"    Food encounter: {row['mean_food_encounter']:.4f} ± {row['sem_food_encounter']:.4f}")

    # ---------- PLOTTING ----------
    def generate_plots(self):
        if self.results_df is None or self.group_summary_df is None:
            messagebox.showwarning("Warning", "No results to plot. Run analysis first.")
            return
        if not hasattr(self, 'last_output_dir') or self.last_output_dir is None:
            output_dir = filedialog.askdirectory(title="Select Output Directory for Plots")
            if not output_dir:
                self.log("\nPlot generation cancelled - no output directory selected")
                return
            self.last_output_dir = Path(output_dir)
        output_path = self.last_output_dir
        self.log(f"\nSaving plots to: {output_path}")
        try:
            self.log("\n" + "="*60)
            self.log("Generating summary plots...")
            self.log("="*60)

            mode = self.analysis_mode.get()
            if mode == "both":
                metrics = [
                    ('food_detection_score', 'Food Detection Score', 'Food Detection'),
                    ('food_encounter_score', 'Food Encounter Score', 'Food Encounter')
                ]
            else:
                metrics = [('food_encounter_score', 'Food Encounter Score', 'Food Encounter')]

            on_food_data = self.results_df[self.results_df['condition'] == 'ON_food'].copy()
            sexes = sorted(on_food_data['sex'].unique())
            genotypes = sorted(on_food_data['genotype'].unique())
            self.log(f"\nDetected sexes: {sexes}")
            self.log(f"Detected genotypes: {genotypes}")

            for metric_col, metric_label, metric_name in metrics:
                self.log(f"\nGenerating {metric_name} plot...")
                self.create_summary_plot(on_food_data, metric_col, metric_label, metric_name, sexes, genotypes, output_path)

            self.log(f"\nGenerating raw speed distribution plots...")
            self.create_raw_speed_plots(sexes, genotypes, mode, output_path)

            if self.speed_traces_df is not None and len(self.speed_traces_df) > 0:
                self.log(f"\nGenerating speed trace plots...")
                self.create_speed_trace_plots(sexes, genotypes, mode, output_path)
            else:
                self.log(f"\nWarning: No speed trace data available for plotting")

            self.log("\n" + "="*60)
            self.log("Plot generation complete!")
            self.log("="*60)
            messagebox.showinfo("Success", f"Plots generated successfully!\nSaved to: {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plots:\n{str(e)}\n\n{traceback.format_exc()}")
            self.log(f"\nERROR generating plots: {str(e)}")
            self.log(traceback.format_exc())

    def create_summary_plot(self, data, metric_col, metric_label, metric_name, sexes, genotypes, output_path):
        n_sexes = len(sexes)
        n_genotypes = len(genotypes)
        fig, axes = plt.subplots(n_sexes, n_genotypes, figsize=(6*n_genotypes, 5*n_sexes), squeeze=False)
        fig.suptitle(f'{metric_name} by Sex and Genotype', fontsize=16, fontweight='bold', y=0.995)
        treatments = self.sort_treatments(data['treatment'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(treatments)))
        color_map = dict(zip(treatments, colors))
        self.log(f"  Treatments detected (sorted): {treatments}")

        for i, sex in enumerate(sexes):
            for j, genotype in enumerate(genotypes):
                ax = axes[i, j]
                panel = data[(data['sex'] == sex) & (data['genotype'] == genotype)].copy()
                if len(panel) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
                    ax.set_title(f'{sex.upper()} - {genotype}', fontweight='bold')
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                summary = panel.groupby('treatment')[metric_col].agg(['mean', 'sem', 'count']).reset_index()
                summary['treatment'] = pd.Categorical(summary['treatment'], categories=treatments, ordered=True)
                summary = summary.sort_values('treatment').reset_index(drop=True)
                x = np.arange(len(summary))
                bars = ax.bar(x, summary['mean'], yerr=summary['sem'], capsize=5,
                              color=[color_map[t] for t in summary['treatment']],
                              edgecolor='black', linewidth=1.5, alpha=0.8)
                ax.set_title(f'{sex.upper()} - {genotype}', fontweight='bold', fontsize=12)
                ax.set_xticks(x); ax.set_xticklabels(summary['treatment'], rotation=45, ha='right')
                ax.set_ylabel(metric_label, fontsize=10)
                for idx, (bar, n) in enumerate(zip(bars, summary['count'])):
                    h = bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2., h + (summary['sem'].iloc[idx] if not np.isnan(summary['sem'].iloc[idx]) else 0),
                            f'n={int(n)}', ha='center', va='bottom', fontsize=8)
                if metric_col in ['food_detection_score', 'food_encounter_score']:
                    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                y_min, y_max = ax.get_ylim()
                ax.set_ylim(0, y_max * 1.15)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.grid(axis='y', alpha=0.3, linestyle=':')
        plt.tight_layout()
        out = output_path / f"{metric_name.lower().replace(' ', '_')}_summary.png"
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
        self.log(f"  Saved: {out}")

    def create_raw_speed_plots(self, sexes, genotypes, mode, output_path):
        n_sexes = len(sexes); n_genotypes = len(genotypes)
        treatments = self.sort_treatments(self.results_df['treatment'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(treatments)))
        color_map = dict(zip(treatments, colors))
        if mode == "both":
            speed_types = [
                ('OFF_food', 'before_mean_speed', 'OFF-food Before'),
                ('ON_food', 'before_mean_speed', 'ON-food Before'),
                ('ON_food', 'after_mean_speed', 'ON-food After')
            ]
        else:
            speed_types = [
                ('ON_food', 'before_mean_speed', 'ON-food Before'),
                ('ON_food', 'after_mean_speed', 'ON-food After')
            ]
        for condition, speed_col, label in speed_types:
            fig, axes = plt.subplots(n_sexes, n_genotypes, figsize=(6*n_genotypes, 5*n_sexes), squeeze=False)
            fig.suptitle(f'Raw Speed: {label} (mm/s)', fontsize=16, fontweight='bold', y=0.995)
            for i, sex in enumerate(sexes):
                for j, genotype in enumerate(genotypes):
                    ax = axes[i, j]
                    panel = self.results_df[
                        (self.results_df['sex'] == sex) &
                        (self.results_df['genotype'] == genotype) &
                        (self.results_df['condition'] == condition)
                    ].copy()
                    if len(panel) == 0:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
                        ax.set_title(f'{sex.upper()} - {genotype}', fontweight='bold')
                        ax.set_xticks([]); ax.set_yticks([])
                        continue
                    summary = panel.groupby('treatment')[speed_col].agg(['mean', 'sem', 'count']).reset_index()
                    summary['treatment'] = pd.Categorical(summary['treatment'], categories=treatments, ordered=True)
                    summary = summary.sort_values('treatment').reset_index(drop=True)
                    x = np.arange(len(summary))
                    bars = ax.bar(x, summary['mean'], yerr=summary['sem'], capsize=5,
                                  color=[color_map[t] for t in summary['treatment']],
                                  edgecolor='black', linewidth=1.5, alpha=0.8)
                    ax.set_title(f'{sex.upper()} - {genotype}', fontweight='bold', fontsize=12)
                    ax.set_xticks(x); ax.set_xticklabels(summary['treatment'], rotation=45, ha='right')
                    ax.set_ylabel('Speed (mm/s)', fontsize=10)
                    for idx, (bar, n) in enumerate(zip(bars, summary['count'])):
                        h = bar.get_height()
                        ax.text(bar.get_x()+bar.get_width()/2., h + (summary['sem'].iloc[idx] if not np.isnan(summary['sem'].iloc[idx]) else 0),
                                f'n={int(n)}', ha='center', va='bottom', fontsize=8)
                    y_min, y_max = ax.get_ylim()
                    ax.set_ylim(0, y_max * 1.15)
                    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                    ax.grid(axis='y', alpha=0.3, linestyle=':')
            plt.tight_layout()
            fname = f"raw_speed_{condition.lower()}_{speed_col.replace('_mean_speed', '')}.png"
            out = output_path / fname
            plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
            self.log(f"  Saved: {out}")

    def create_speed_trace_plots(self, sexes, genotypes, mode, output_path):
        treatments = self.sort_treatments(self.speed_traces_df['treatment'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(treatments)))
        color_map = dict(zip(treatments, colors))
        n_sexes = len(sexes); n_genotypes = len(genotypes)
        fig, axes = plt.subplots(n_sexes, n_genotypes, figsize=(8*n_genotypes, 6*n_sexes), squeeze=False)
        fig.suptitle('Speed Traces Around Food Encounter', fontsize=16, fontweight='bold', y=0.995)
        all_trace_summaries = []

        for i, sex in enumerate(sexes):
            for j, genotype in enumerate(genotypes):
                ax = axes[i, j]
                panel = self.speed_traces_df[
                    (self.speed_traces_df['sex'] == sex) &
                    (self.speed_traces_df['genotype'] == genotype) &
                    (self.speed_traces_df['condition'] == 'ON_food')
                ].copy()
                if len(panel) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
                    ax.set_title(f'{sex.upper()} - {genotype}', fontweight='bold')
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                for treatment in treatments:
                    tdata = panel[panel['treatment'] == treatment].copy()
                    if len(tdata) == 0:
                        continue
                    ts = tdata.groupby('time_relative')['speed'].agg(['mean', 'sem', 'std', 'count']).reset_index()
                    ts['sex'] = sex; ts['genotype'] = genotype; ts['treatment'] = treatment; ts['condition'] = 'ON_food'
                    all_trace_summaries.append(ts)
                    ax.plot(ts['time_relative'], ts['mean'], linewidth=2, label=treatment, color=color_map[treatment], alpha=0.8)
                    ax.fill_between(ts['time_relative'], ts['mean']-ts['sem'], ts['mean']+ts['sem'],
                                    color=color_map[treatment], alpha=0.2)
                ax.set_title(f'{sex.upper()} - {genotype}', fontweight='bold', fontsize=12)
                ax.set_xlabel('Time Relative to Food Encounter (s)', fontsize=10)
                ax.set_ylabel('Speed (mm/s)', fontsize=10)
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Food Encounter')
                ax.legend(loc='best', framealpha=0.9, fontsize=9)
                ax.grid(True, alpha=0.3, linestyle=':')
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                y_min, y_max = ax.get_ylim(); ax.set_ylim(0, y_max * 1.05)

        plt.tight_layout()
        out = output_path / "speed_trace_food_encounter.png"
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
        self.log(f"  Saved: {out}")

        if len(all_trace_summaries) > 0:
            trace_summary_df = pd.concat(all_trace_summaries, ignore_index=True)
            cols = ['condition', 'treatment', 'sex', 'genotype', 'time_relative', 'mean', 'std', 'sem', 'count']
            trace_summary_df = trace_summary_df[cols]
            csv_file = output_path / "speed_trace_summary.csv"
            trace_summary_df.to_csv(csv_file, index=False)
            self.log(f"  Saved trace summary CSV: {csv_file}")

    # ---------- EXPORT ----------
    def export_results(self):
        if self.results_df is None:
            messagebox.showwarning("Warning", "No results to export. Run analysis first.")
            return
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        try:
            output_path = Path(output_dir)
            self.last_output_dir = output_path
            individual_file = output_path / "individual_animal_results.csv"
            self.results_df.to_csv(individual_file, index=False)
            self.log(f"\nExported individual results to: {individual_file}")

            summary_file = output_path / "group_summaries.csv"
            self.group_summary_df.to_csv(summary_file, index=False)
            self.log(f"Exported group summaries to: {summary_file}")

            outlier_file = output_path / "outlier_report.csv"
            self.outlier_report_df.to_csv(outlier_file, index=False)
            self.log(f"Exported outlier report to: {outlier_file}")

            # Export speed traces separated by condition
            if self.speed_traces_df is not None and len(self.speed_traces_df) > 0:
                # Export ON-food traces
                on_traces = self.speed_traces_df[self.speed_traces_df['condition'] == 'ON_food']
                if len(on_traces) > 0:
                    on_traces_file = output_path / "speed_traces_on_food.csv"
                    on_traces.to_csv(on_traces_file, index=False)
                    self.log(f"Exported ON-food speed traces to: {on_traces_file}")
                
                # Export OFF-food traces if they exist
                off_traces = self.speed_traces_df[self.speed_traces_df['condition'] == 'OFF_food']
                if len(off_traces) > 0:
                    off_traces_file = output_path / "speed_traces_off_food.csv"
                    off_traces.to_csv(off_traces_file, index=False)
                    self.log(f"Exported OFF-food speed traces to: {off_traces_file}")

            messagebox.showinfo("Success", f"Results exported to:\n{output_dir}\n\nYou can now generate plots to the same directory.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
            self.log(f"\nERROR exporting results: {str(e)}")


def main():
    root = tk.Tk()
    app = BehavioralContextAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()