#!/usr/bin/env python3
"""
Interactive Richards/Double-Richards Fit Viewer

Browse individual animal speed traces with fitted curves.
Features:
- Load results from CSV files interactively
- Navigate animal-by-animal
- Filter by genotype and fit type
- Export individual or batch plots
- Keyboard shortcuts for navigation

Works with BOTH:
- results/.../richards_single/
- results/.../richards_double/

Usage:
    python view_richards_fits.py [results_dir/]
    
Requirements:
    pandas, numpy, matplotlib, tkinter
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


def richards_curve(t, y_initial, y_final, B, M, nu):
    """Richards curve function."""
    A = y_final - y_initial
    with np.errstate(over='ignore', invalid='ignore'):
        denominator = (1 + np.exp(-B * (t - M)))**(1/nu)
        denominator = np.clip(denominator, 1e-10, 1e10)
    return y_initial + A / denominator


def double_richards_curve(t,
                          y1_i, y1_f, B1, M1, nu1,
                          y2_f, B2, M2, nu2):
    """Double Richards (two-phase) function."""
    with np.errstate(over='ignore', invalid='ignore'):
        term1 = (y1_f - y1_i) / (1 + np.exp(-B1 * (t - M1)))**(1/nu1)
        term2 = (y2_f - y1_f) / (1 + np.exp(-B2 * (t - M2)))**(1/nu2)
    term1 = np.clip(term1, -1e10, 1e10)
    term2 = np.clip(term2, -1e10, 1e10)
    return y1_i + term1 + term2


class RichardsFitViewer:
    """Interactive viewer for Richards/Double-Richards fits."""
    
    def __init__(self, results_dir=None):
        self.root = tk.Tk()
        self.root.title("Richards Fit Viewer")
        self.root.geometry("1400x900")

        self.results_dir = None
        self.df_params = None
        self.df_windowed = None
        self.animal_ids = []
        self.filtered_animal_ids = []

        self._setup_gui()

        if results_dir is not None:
            self.results_dir = Path(results_dir)
            if self._load_data():
                self._populate_animal_list()
                if len(self.animal_ids) > 0:
                    self.animal_listbox.selection_set(0)
                    self._on_animal_select(None)
        else:
            self.root.after(100, self._prompt_load_files)
    
    def _load_data(self):
        if self.results_dir is None:
            return False

        params_path = self.results_dir / 'richards_fit_parameters.csv'
        windowed_path = self.results_dir / 'windowed_speed_data.csv'
        
        if not params_path.exists():
            messagebox.showerror("File Not Found", f"Parameters file not found:\n{params_path}")
            return False
        if not windowed_path.exists():
            messagebox.showerror("File Not Found", f"Windowed data file not found:\n{windowed_path}")
            return False

        try:
            print(f"Loading parameters from: {params_path}")
            self.df_params = pd.read_csv(params_path)

            print(f"Loading windowed data from: {windowed_path}")
            self.df_windowed = pd.read_csv(windowed_path)

            self.animal_ids = list(zip(self.df_params['assay_num'], self.df_params['track_num']))

            self.status_label.config(
                text=f"Loaded: {len(self.animal_ids)} animals from {self.results_dir.name}"
            )
            self._update_filter_options()
            return True

        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading data:\n{e}")
            return False
    
    def _setup_gui(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Results Directory...", command=self._load_directory)
        file_menu.add_command(label="Load Parameters CSV...", command=self._load_params_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_command(label="About", command=self._show_about)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_label = ttk.Label(
            status_frame, 
            text="No data loaded. Use File → Load Results Directory...",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)

        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        filter_frame = ttk.LabelFrame(left_frame, text="Filters", padding=10)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(filter_frame, text="Genotype:").grid(row=0, column=0, sticky="w")
        self.genotype_var = tk.StringVar(value="All")
        self.genotype_combo = ttk.Combobox(
            filter_frame, textvariable=self.genotype_var,
            values=["All"], state="readonly", width=15
        )
        self.genotype_combo.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.genotype_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        ttk.Label(filter_frame, text="Fit Type:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.fittype_var = tk.StringVar(value="All")
        self.fittype_combo = ttk.Combobox(
            filter_frame, textvariable=self.fittype_var,
            values=["All"], state="readonly", width=15
        )
        self.fittype_combo.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))
        self.fittype_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        ttk.Label(filter_frame, text="Model:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.model_var = tk.StringVar(value="All")
        self.model_combo = ttk.Combobox(
            filter_frame, textvariable=self.model_var,
            values=["All"], state="readonly", width=15
        )
        self.model_combo.grid(row=2, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        filter_frame.columnconfigure(1, weight=1)

        list_frame = ttk.LabelFrame(left_frame, text="Animals (↑↓ or click to select)", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.animal_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE, font=('Courier', 9)
        )
        self.animal_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.animal_listbox.yview)

        self.animal_listbox.bind("<<ListboxSelect>>", self._on_animal_select)

        self.root.bind('<Up>', lambda e: self._prev_animal())
        self.root.bind('<Down>', lambda e: self._next_animal())
        self.root.bind('<Left>', lambda e: self._prev_animal())
        self.root.bind('<Right>', lambda e: self._next_animal())
        self.root.bind('<space>', lambda e: self._next_animal())
        self.root.bind('e', lambda e: self._export_plot())
        self.root.bind('E', lambda e: self._export_plot())

        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X)
        ttk.Button(nav_frame, text="◄ Previous (←)", command=self._prev_animal).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(nav_frame, text="Next (→) ►", command=self._next_animal).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.info_text = tk.Text(right_frame, height=7, wrap=tk.WORD, font=('Courier', 9))
        self.info_text.pack(fill=tk.X, pady=(0, 10))

        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        export_frame = ttk.Frame(right_frame)
        export_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(export_frame, text="Export Current Plot (E)", command=self._export_plot).pack(side=tk.LEFT)
        ttk.Button(export_frame, text="Export All Filtered Plots", command=self._export_all_plots).pack(side=tk.LEFT, padx=(10, 0))

        self._show_welcome_message()
    
    def _populate_animal_list(self):
        if self.df_params is None:
            return

        self.animal_listbox.delete(0, tk.END)

        df_filtered = self.df_params.copy()
        if self.genotype_var.get() != "All":
            df_filtered = df_filtered[df_filtered['strain_genotype'] == self.genotype_var.get()]
        if self.fittype_var.get() != "All":
            df_filtered = df_filtered[df_filtered['fit_type'] == self.fittype_var.get()]
        if self.model_var.get() != "All" and 'model_type' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['model_type'] == self.model_var.get()]

        self.filtered_animal_ids = list(zip(df_filtered['assay_num'], df_filtered['track_num']))

        for assay, track in self.filtered_animal_ids:
            row = df_filtered[(df_filtered['assay_num'] == assay) & (df_filtered['track_num'] == track)].iloc[0]
            model_label = row['model_type'] if 'model_type' in row else 'single'
            label = (f"A{assay:02d}_T{track:02d} | "
                     f"{row['strain_genotype']:10s} | "
                     f"{row['fit_type']:12s} | "
                     f"R²={row['r_squared']:.2f} | {model_label}")
            self.animal_listbox.insert(tk.END, label)

    def _show_welcome_message(self):
        self.ax.clear()
        self.ax.text(
            0.5, 0.5,
            'Welcome to Richards Fit Viewer!\n\n'
            'Use File → Load Results Directory...\n'
            'to get started',
            transform=self.ax.transAxes,
            ha='center', va='center',
            fontsize=14, color='gray'
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
    
    def _prompt_load_files(self):
        if messagebox.askyesno("Load Data", "No data loaded.\n\nLoad results now?"):
            self._load_directory()
    
    def _load_directory(self):
        directory = filedialog.askdirectory(title="Select Results Directory", initialdir=Path.cwd())
        if directory:
            self.results_dir = Path(directory)
            if self._load_data():
                self._populate_animal_list()
                if len(self.animal_ids) > 0:
                    self.animal_listbox.selection_set(0)
                    self._on_animal_select(None)
    
    def _load_params_csv(self):
        params_file = filedialog.askopenfilename(
            title="Select Parameters CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=Path.cwd()
        )
        if not params_file:
            return
        windowed_file = filedialog.askopenfilename(
            title="Select Windowed Speed Data CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=Path(params_file).parent
        )
        if not windowed_file:
            return

        try:
            print(f"Loading parameters from: {params_file}")
            self.df_params = pd.read_csv(params_file)
            print(f"Loading windowed data from: {windowed_file}")
            self.df_windowed = pd.read_csv(windowed_file)
            self.animal_ids = list(zip(self.df_params['assay_num'], self.df_params['track_num']))

            self.results_dir = Path(params_file).parent
            self.status_label.config(text=f"Loaded: {len(self.animal_ids)} animals from individual CSV files")
            self._update_filter_options()
            self._populate_animal_list()
            if len(self.animal_ids) > 0:
                self.animal_listbox.selection_set(0)
                self._on_animal_select(None)
        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading CSV files:\n{e}")

    def _update_filter_options(self):
        if self.df_params is None:
            return
        genotypes = ["All"] + sorted(self.df_params['strain_genotype'].dropna().unique().tolist())
        self.genotype_combo['values'] = genotypes
        self.genotype_var.set("All")

        fit_types = ["All"] + sorted(self.df_params['fit_type'].dropna().unique().tolist())
        self.fittype_combo['values'] = fit_types
        self.fittype_var.set("All")

        models = ["All"]
        if 'model_type' in self.df_params.columns:
            models += sorted(self.df_params['model_type'].dropna().unique().tolist())
        self.model_combo['values'] = models
        self.model_var.set("All")

    def _show_shortcuts(self):
        shortcuts = """
Keyboard Shortcuts:

Navigation:
  ↑ / ↓  or  ← / →   Previous / Next animal
  Space               Next animal

Actions:
  E                   Export current plot

Mouse:
  Click               Select animal from list
  Scroll              Scroll through animal list
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)
    
    def _show_about(self):
        about_text = """
Richards Fit Viewer
Version 1.1

Interactive browser for viewing SINGLE and DOUBLE Richards curve fits
to individual animal speed traces.

Part of the Richards Curve Analysis Toolkit
for Worm Locomotion Studies.
        """
        messagebox.showinfo("About", about_text)
    
    def _on_filter_change(self, event):
        self._populate_animal_list()
        if self.animal_listbox.size() > 0:
            self.animal_listbox.selection_set(0)
            self._on_animal_select(None)
    
    def _on_animal_select(self, event):
        selection = self.animal_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        assay, track = self.filtered_animal_ids[idx]
        self._plot_animal(assay, track)
    
    def _plot_animal(self, assay_num, track_num):
        if self.df_params is None or self.df_windowed is None:
            messagebox.showwarning("No Data", "Please load data first using File → Load Results Directory")
            return

        row = self.df_params[(self.df_params['assay_num'] == assay_num) &
                             (self.df_params['track_num'] == track_num)]
        if len(row) == 0:
            messagebox.showwarning("No Data", f"No parameters found for animal {assay_num}, {track_num}")
            return
        row = row.iloc[0]

        data = self.df_windowed[(self.df_windowed['assay_num'] == assay_num) &
                                (self.df_windowed['track_num'] == track_num)].copy()
        if len(data) == 0:
            messagebox.showwarning("No Data", f"No windowed data found for animal {assay_num}, {track_num}")
            return

        self.ax.clear()

        time_rel = data['time_rel'].values
        speed = data['speed'].values
        self.ax.plot(time_rel, speed, 'o', alpha=0.5, markersize=4, label='Raw speed', color='gray')

        model_type = row['model_type'] if 'model_type' in row else 'single'
        converged = bool(row.get('converged', True))

        if converged:
            if model_type == 'double':
                # Need double parameters present in CSV
                needed = ['y1_i', 'y1_f', 'B1', 'M1', 'nu1', 'y2_f', 'B2', 'M2', 'nu2']
                if all(k in row.index and pd.notnull(row[k]) for k in needed):
                    fitted = double_richards_curve(
                        time_rel,
                        row['y1_i'], row['y1_f'], row['B1'], row['M1'], row['nu1'],
                        row['y2_f'], row['B2'], row['M2'], row['nu2']
                    )
                    self.ax.plot(time_rel, fitted, '-', linewidth=2, label='Double-Richards fit')
                else:
                    self.ax.text(0.5, 0.5, 'PARAMS MISSING (double)', transform=self.ax.transAxes,
                                 ha='center', va='center', fontsize=16, color='red', alpha=0.5)
            else:
                # single
                needed = ['y_initial', 'y_final', 'B', 'M', 'nu']
                if all(k in row.index and pd.notnull(row[k]) for k in needed):
                    fitted = richards_curve(time_rel, row['y_initial'], row['y_final'], row['B'], row['M'], row['nu'])
                    self.ax.plot(time_rel, fitted, '-', linewidth=2, label='Richards fit')
                else:
                    self.ax.text(0.5, 0.5, 'PARAMS MISSING (single)', transform=self.ax.transAxes,
                                 ha='center', va='center', fontsize=16, color='red', alpha=0.5)
        else:
            self.ax.text(0.5, 0.5, 'FIT FAILED', transform=self.ax.transAxes,
                         ha='center', va='center', fontsize=20, color='red', alpha=0.3)

        self.ax.axvline(0, color='green', linestyle='--', alpha=0.7, label='Food encounter')

        self.ax.set_xlabel('Time relative to food encounter (s)', fontsize=11)
        self.ax.set_ylabel('Speed (mm/s)', fontsize=11)
        subtitle = f"{row.get('strain_genotype','')} | {row.get('fit_type','')} | model={model_type}"
        self.ax.set_title(f'Animal: Assay {assay_num}, Track {track_num} | {subtitle}', fontsize=12)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

        # info panel
        self.info_text.delete(1.0, tk.END)
        info = (
            f"Assay: {assay_num}  Track: {track_num}  Source: {row.get('source_file','')}\n"
            f"Genotype: {row.get('strain_genotype','')}  Sex: {row.get('sex','')}  Treatment: {row.get('treatment','')}\n\n"
            f"Model: {model_type}  Converged: {row.get('converged', True)}  "
            f"R² = {row.get('r_squared', float('nan')):.4f}  RMSE = {row.get('rmse', float('nan')):.4f}\n\n"
        )
        if model_type == 'double':
            info += (
                f"y1_i = {row.get('y1_i', float('nan')):.4f}  |  "
                f"y1_f = {row.get('y1_f', float('nan')):.4f}  |  "
                f"y2_f = {row.get('y2_f', float('nan')):.4f}\n"
                f"B1 = {row.get('B1', float('nan')):.4f}  |  "
                f"M1 = {row.get('M1', float('nan')):.2f}s  |  "
                f"nu1 = {row.get('nu1', float('nan')):.4f}\n"
                f"B2 = {row.get('B2', float('nan')):.4f}  |  "
                f"M2 = {row.get('M2', float('nan')):.2f}s  |  "
                f"nu2 = {row.get('nu2', float('nan')):.4f}\n"
                f"Δ1 = {row.get('delta1', float('nan')):.4f}  |  "
                f"Δ2 = {row.get('delta2', float('nan')):.4f}  |  "
                f"A = {row.get('A', float('nan')):.4f}\n"
            )
        else:
            info += (
                f"y_initial = {row.get('y_initial', float('nan')):.4f}  |  "
                f"y_final = {row.get('y_final', float('nan')):.4f}  |  "
                f"A = {row.get('A', float('nan')):.4f}\n"
                f"B = {row.get('B', float('nan')):.4f}  |  "
                f"M = {row.get('M', float('nan')):.2f}s  |  "
                f"nu = {row.get('nu', float('nan')):.4f}\n"
            )
        self.info_text.insert(1.0, info)
    
    def _prev_animal(self):
        sel = self.animal_listbox.curselection()
        if not sel: return
        idx = sel[0]
        if idx > 0:
            self.animal_listbox.selection_clear(0, tk.END)
            self.animal_listbox.selection_set(idx - 1)
            self.animal_listbox.see(idx - 1)
            self._on_animal_select(None)
    
    def _next_animal(self):
        sel = self.animal_listbox.curselection()
        if not sel: return
        idx = sel[0]
        if idx < self.animal_listbox.size() - 1:
            self.animal_listbox.selection_clear(0, tk.END)
            self.animal_listbox.selection_set(idx + 1)
            self.animal_listbox.see(idx + 1)
            self._on_animal_select(None)
    
    def _export_plot(self):
        sel = self.animal_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        assay, track = self.filtered_animal_ids[idx]
        filename = f"animal_{assay:02d}_{track:02d}.png"
        save_path = self.results_dir / filename
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        messagebox.showinfo("Exported", f"Saved plot to:\n{save_path}")
    
    def _export_all_plots(self):
        if len(self.filtered_animal_ids) == 0:
            messagebox.showwarning("No Animals", "No animals match current filters")
            return
        if not messagebox.askyesno("Export All", f"Export plots for {len(self.filtered_animal_ids)} animals?\nThis may take a while."):
            return

        export_dir = self.results_dir / 'individual_plots'
        export_dir.mkdir(exist_ok=True)
        for i, (assay, track) in enumerate(self.filtered_animal_ids):
            if i % 10 == 0:
                print(f"Exporting {i+1}/{len(self.filtered_animal_ids)}...")
            self._plot_animal(assay, track)
            filename = f"animal_{assay:02d}_{track:02d}.png"
            save_path = export_dir / filename
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        messagebox.showinfo("Export Complete", f"Exported {len(self.filtered_animal_ids)} plots to:\n{export_dir}")
    
    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Interactive viewer for Richards and Double-Richards curve fits')
    parser.add_argument('results_dir', nargs='?', type=str, default=None,
                        help='Directory containing richards_fit_parameters.csv and windowed_speed_data.csv (optional)')
    args = parser.parse_args()

    try:
        viewer = RichardsFitViewer(args.results_dir)
        viewer.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
