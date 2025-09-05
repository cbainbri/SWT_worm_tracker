#!/usr/bin/env python3
"""
Interactive Food Encounter Velocity Analysis Pipeline

Analyzes animal velocity around food encounters using composite CSV data.
- Creates unique animal IDs from assay_num + track_num
- Calculates velocity from centroid positions (x, y coordinates)
- Aligns food encounters to time=0 for comparison
- Provides statistical summaries for different time windows
- Groups analysis by treatment, sex, and strain/genotype

Requirements: pandas, numpy, matplotlib, seaborn, scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Try to import seaborn, set flag if available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Using matplotlib defaults for plotting.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

class FoodEncounterAnalyzer:
    """Main analysis class for food encounter velocity analysis"""
    
    def __init__(self, pixels_per_mm: float = 104.0):
        self.data = None
        self.velocity_data = None
        self.encounter_aligned_data = None
        self.summary_stats = None
        self.pixels_per_mm = pixels_per_mm
        
    def load_data(self, csv_path: str) -> bool:
        """Load composite CSV data"""
        try:
            self.data = pd.read_csv(csv_path)
            print(f"Loaded data: {len(self.data)} rows, {len(self.data.columns)} columns")
            
            # Validate required columns
            required_cols = ['assay_num', 'track_num', 'time', 'x', 'y', 'food_encounter', 
                           'treatment', 'sex', 'strain_genotype']
            missing = [col for col in required_cols if col not in self.data.columns]
            if missing:
                print(f"ERROR: Missing required columns: {missing}")
                return False
                
            # Create unique animal IDs
            self.data['animal_id'] = self.data['assay_num'].astype(str) + '_' + self.data['track_num'].astype(str)
            
            print(f"Unique animals: {self.data['animal_id'].nunique()}")
            print(f"Animals with food encounters: {self.data[self.data['food_encounter'] == 1]['animal_id'].nunique()}")
            
            return True
            
        except Exception as e:
            print(f"ERROR loading data: {e}")
            return False
    
    def calculate_velocities(self, time_window: float = 1.0) -> None:
        """Calculate instantaneous velocity from centroid positions"""
        print(f"Calculating velocities with {time_window}s time window...")
        
        velocity_data = []
        
        for animal_id in self.data['animal_id'].unique():
            animal_data = self.data[self.data['animal_id'] == animal_id].sort_values('time').copy()
            
            if len(animal_data) < 2:
                continue
                
            # Calculate time differences
            dt = animal_data['time'].diff()
            
            # Calculate position differences
            dx = animal_data['x'].diff()
            dy = animal_data['y'].diff()
            
            # Calculate distance and velocity
            distance = np.sqrt(dx**2 + dy**2)
            velocity = distance / dt
            
            # Convert to mm/s if pixels_per_mm is provided
            if self.pixels_per_mm > 0:
                velocity = velocity / self.pixels_per_mm
                distance = distance / self.pixels_per_mm
            
            # Apply smoothing window if specified
            if time_window > 0 and len(velocity) > 3:
                # Simple rolling mean for smoothing
                window_size = max(3, int(time_window))
                velocity = velocity.rolling(window=window_size, center=True, min_periods=1).mean()
            
            # Add velocity data
            animal_data['velocity'] = velocity
            animal_data['dt'] = dt
            animal_data['distance'] = distance
            
            # Remove first row (NaN velocity)
            animal_data = animal_data.iloc[1:].copy()
            
            velocity_data.append(animal_data)
        
        self.velocity_data = pd.concat(velocity_data, ignore_index=True)
        
        print(f"Calculated velocities for {len(velocity_data)} animals")
        print(f"Mean velocity: {self.velocity_data['velocity'].mean():.3f} ± {self.velocity_data['velocity'].std():.3f}")
    
    def align_to_food_encounters(self, time_before: float = 120, time_after: float = 120) -> None:
        """Align data to food encounter events (time = 0)"""
        print(f"Aligning data to food encounters (-{time_before}s to +{time_after}s)...")
        
        aligned_data = []
        encounter_count = 0
        
        for animal_id in self.velocity_data['animal_id'].unique():
            animal_data = self.velocity_data[self.velocity_data['animal_id'] == animal_id].copy()
            
            # Find food encounter events
            encounters = animal_data[animal_data['food_encounter'] == 1]
            
            if len(encounters) == 0:
                continue
                
            # Use first encounter for each animal
            encounter_time = encounters.iloc[0]['time']
            encounter_count += 1
            
            # Create relative time (encounter = 0)
            animal_data['relative_time'] = animal_data['time'] - encounter_time
            
            # Filter to time window around encounter
            window_data = animal_data[
                (animal_data['relative_time'] >= -time_before) & 
                (animal_data['relative_time'] <= time_after)
            ].copy()
            
            if len(window_data) > 0:
                aligned_data.append(window_data)
        
        if aligned_data:
            self.encounter_aligned_data = pd.concat(aligned_data, ignore_index=True)
            print(f"Aligned data for {encounter_count} food encounters")
            print(f"Time range: {self.encounter_aligned_data['relative_time'].min():.1f} to {self.encounter_aligned_data['relative_time'].max():.1f} seconds")
        else:
            print("WARNING: No food encounter data could be aligned")
    
    def calculate_summary_statistics(self) -> pd.DataFrame:
        """Calculate summary statistics for different time windows"""
        if self.encounter_aligned_data is None:
            print("ERROR: No aligned data available")
            return None
            
        print("Calculating summary statistics...")
        
        # Define time windows
        time_windows = {
            'short_before': (-15, -5),
            'short_after': (5, 15), 
            'long_before': (-120, -15),
            'long_after': (15, 120)
        }
        
        summary_data = []
        
        # Group by treatment, sex, strain/genotype
        groups = ['treatment', 'sex', 'strain_genotype']
        
        for group_vals, group_data in self.encounter_aligned_data.groupby(groups):
            treatment, sex, strain = group_vals
            
            # Calculate statistics for each time window
            row = {
                'treatment': treatment,
                'sex': sex, 
                'strain_genotype': strain,
                'n_animals': group_data['animal_id'].nunique()
            }
            
            for window_name, (t_start, t_end) in time_windows.items():
                window_data = group_data[
                    (group_data['relative_time'] >= t_start) & 
                    (group_data['relative_time'] <= t_end)
                ]
                
                if len(window_data) > 0:
                    velocities = window_data['velocity'].dropna()
                    
                    if len(velocities) > 0:
                        row[f'{window_name}_mean'] = velocities.mean()
                        row[f'{window_name}_std'] = velocities.std()
                        row[f'{window_name}_sem'] = velocities.std() / np.sqrt(len(velocities))
                        row[f'{window_name}_n_points'] = len(velocities)
                    else:
                        row[f'{window_name}_mean'] = np.nan
                        row[f'{window_name}_std'] = np.nan
                        row[f'{window_name}_sem'] = np.nan
                        row[f'{window_name}_n_points'] = 0
                else:
                    row[f'{window_name}_mean'] = np.nan
                    row[f'{window_name}_std'] = np.nan
                    row[f'{window_name}_sem'] = np.nan
                    row[f'{window_name}_n_points'] = 0
            
            summary_data.append(row)
        
        self.summary_stats = pd.DataFrame(summary_data)
        return self.summary_stats
    
    def analyze_food_leaving_behavior(self) -> Optional[pd.DataFrame]:
        """Analyze velocity patterns when animals leave food areas"""
        if self.velocity_data is None:
            print("No velocity data available for food leaving analysis")
            return None
            
        print("Analyzing food leaving behavior...")
        
        leaving_data = []
        
        for animal_id in self.velocity_data['animal_id'].unique():
            animal_data = self.velocity_data[self.velocity_data['animal_id'] == animal_id].sort_values('time').copy()
            
            # Find transitions from food encounter to no encounter
            food_shifts = animal_data['food_encounter'].diff()
            leaving_times = animal_data[food_shifts == -1]['time'].values
            
            for leave_time in leaving_times:
                # Get data around leaving event
                window_data = animal_data[
                    (animal_data['time'] >= leave_time - 60) & 
                    (animal_data['time'] <= leave_time + 60)
                ].copy()
                
                if len(window_data) > 10:  # Minimum data points
                    window_data['relative_time'] = window_data['time'] - leave_time
                    window_data['event_type'] = 'food_leaving'
                    leaving_data.append(window_data)
        
        if leaving_data:
            result = pd.concat(leaving_data, ignore_index=True)
            print(f"Found {len(leaving_data)} food leaving events")
            return result
        else:
            print("No food leaving events found")
            return None
    
    def print_summary_report(self) -> None:
        """Print human-readable summary report"""
        if self.summary_stats is None:
            print("No summary statistics available")
            return
            
        print("\n" + "="*80)
        print("FOOD ENCOUNTER VELOCITY ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        # Overall statistics
        if self.encounter_aligned_data is not None:
            total_animals = self.encounter_aligned_data['animal_id'].nunique()
            total_encounters = len(self.encounter_aligned_data[self.encounter_aligned_data['relative_time'].abs() < 0.1])
            print(f"Total animals analyzed: {total_animals}")
            print(f"Total food encounters: {total_encounters}")
        
        print(f"\nTime Windows:")
        print(f"  Short-term before: -15 to -5 seconds")
        print(f"  Short-term after:   5 to 15 seconds") 
        print(f"  Long-term before: -120 to -15 seconds")
        print(f"  Long-term after:   15 to 120 seconds")
        
        velocity_unit = "mm/s" if self.pixels_per_mm > 0 else "pixels/time"
        print(f"\nVelocity units: {velocity_unit}")
        
        # Group results
        for idx, row in self.summary_stats.iterrows():
            print(f"\n{'-'*60}")
            print(f"Group: {row['treatment']} | {row['sex']} | {row['strain_genotype']}")
            print(f"Animals (n): {row['n_animals']}")
            print(f"{'-'*60}")
            
            # Format velocity data
            windows = ['short_before', 'short_after', 'long_before', 'long_after']
            labels = ['Short Before', 'Short After', 'Long Before', 'Long After']
            
            for window, label in zip(windows, labels):
                mean_val = row[f'{window}_mean']
                std_val = row[f'{window}_std'] 
                sem_val = row[f'{window}_sem']
                n_points = row[f'{window}_n_points']
                
                if pd.notna(mean_val):
                    print(f"{label:12}: {mean_val:8.3f} ± {std_val:6.3f} (SEM: {sem_val:6.3f}) [n={n_points:4.0f}]")
                else:
                    print(f"{label:12}: No data available")
        
        print("\n" + "="*80)
    
    def plot_velocity_profiles(self, save_plots: bool = True, output_dir: str = "plots", 
                             show_individual_traces: bool = False) -> None:
        """Create velocity profile plots"""
        if self.encounter_aligned_data is None:
            print("No aligned data available for plotting")
            return
            
        print("Creating velocity profile plots...")
        
        # Create output directory
        if save_plots:
            Path(output_dir).mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
        else:
            # Use matplotlib color cycle if seaborn not available
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        
        velocity_unit = "mm/s" if self.pixels_per_mm > 0 else "pixels/time"
        
        # 1. Overall velocity profile plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Velocity Profiles Around Food Encounters', fontsize=16)
        
        # Group by treatment, sex, strain
        groups = ['treatment', 'sex', 'strain_genotype']
        
        # Plot 1: All groups overlaid
        ax = axes[0, 0]
        for group_vals, group_data in self.encounter_aligned_data.groupby(groups):
            treatment, sex, strain = group_vals
            label = f"{treatment}_{sex}_{strain}"
            
            # Bin data for smoother curves
            time_bins = np.arange(-120, 121, 5)
            binned_velocity = []
            binned_time = []
            
            for i in range(len(time_bins)-1):
                t_start, t_end = time_bins[i], time_bins[i+1]
                bin_data = group_data[
                    (group_data['relative_time'] >= t_start) & 
                    (group_data['relative_time'] < t_end)
                ]
                if len(bin_data) > 0:
                    binned_velocity.append(bin_data['velocity'].mean())
                    binned_time.append((t_start + t_end) / 2)
            
            if binned_velocity:
                ax.plot(binned_time, binned_velocity, label=label, alpha=0.7, linewidth=2)
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Food Encounter')
        ax.axvline(x=-15, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=15, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time Relative to Food Encounter (seconds)')
        ax.set_ylabel(f'Velocity ({velocity_unit})')
        ax.set_title('All Groups')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: By treatment
        ax = axes[0, 1]
        for treatment, treat_data in self.encounter_aligned_data.groupby('treatment'):
            time_bins = np.arange(-120, 121, 5)
            binned_velocity = []
            binned_time = []
            
            for i in range(len(time_bins)-1):
                t_start, t_end = time_bins[i], time_bins[i+1]
                bin_data = treat_data[
                    (treat_data['relative_time'] >= t_start) & 
                    (treat_data['relative_time'] < t_end)
                ]
                if len(bin_data) > 0:
                    binned_velocity.append(bin_data['velocity'].mean())
                    binned_time.append((t_start + t_end) / 2)
            
            if binned_velocity:
                ax.plot(binned_time, binned_velocity, label=treatment, linewidth=2)
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax.set_xlabel('Time Relative to Food Encounter (seconds)')
        ax.set_ylabel(f'Velocity ({velocity_unit})')
        ax.set_title('By Treatment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: By sex
        ax = axes[1, 0]
        for sex, sex_data in self.encounter_aligned_data.groupby('sex'):
            time_bins = np.arange(-120, 121, 5)
            binned_velocity = []
            binned_time = []
            
            for i in range(len(time_bins)-1):
                t_start, t_end = time_bins[i], time_bins[i+1]
                bin_data = sex_data[
                    (sex_data['relative_time'] >= t_start) & 
                    (sex_data['relative_time'] < t_end)
                ]
                if len(bin_data) > 0:
                    binned_velocity.append(bin_data['velocity'].mean())
                    binned_time.append((t_start + t_end) / 2)
            
            if binned_velocity:
                ax.plot(binned_time, binned_velocity, label=sex, linewidth=2)
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax.set_xlabel('Time Relative to Food Encounter (seconds)')
        ax.set_ylabel(f'Velocity ({velocity_unit})')
        ax.set_title('By Sex')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: By strain/genotype
        ax = axes[1, 1]
        for strain, strain_data in self.encounter_aligned_data.groupby('strain_genotype'):
            time_bins = np.arange(-120, 121, 5)
            binned_velocity = []
            binned_time = []
            
            for i in range(len(time_bins)-1):
                t_start, t_end = time_bins[i], time_bins[i+1]
                bin_data = strain_data[
                    (strain_data['relative_time'] >= t_start) & 
                    (strain_data['relative_time'] < t_end)
                ]
                if len(bin_data) > 0:
                    binned_velocity.append(bin_data['velocity'].mean())
                    binned_time.append((t_start + t_end) / 2)
            
            if binned_velocity:
                ax.plot(binned_time, binned_velocity, label=strain, linewidth=2)
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax.set_xlabel('Time Relative to Food Encounter (seconds)')
        ax.set_ylabel(f'Velocity ({velocity_unit})')
        ax.set_title('By Strain/Genotype')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/velocity_profiles.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{output_dir}/velocity_profiles.pdf", bbox_inches='tight')
        
        plt.show()
        
        # 2. Bar plot of summary statistics
        if self.summary_stats is not None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Velocity Summary Statistics by Time Windows', fontsize=16)
            
            windows = ['short_before', 'short_after', 'long_before', 'long_after']
            titles = ['Short Before (-15 to -5s)', 'Short After (5 to 15s)', 
                     'Long Before (-120 to -15s)', 'Long After (15 to 120s)']
            
            for i, (window, title) in enumerate(zip(windows, titles)):
                ax = axes[i//2, i%2]
                
                # Create group labels
                stats_data = self.summary_stats.copy()
                stats_data['group_label'] = (stats_data['treatment'] + '_' + 
                                           stats_data['sex'] + '_' + 
                                           stats_data['strain_genotype'])
                
                means = stats_data[f'{window}_mean'].values
                errors = stats_data[f'{window}_sem'].values
                labels = stats_data['group_label'].values
                
                # Remove NaN values
                valid_idx = ~pd.isna(means)
                means = means[valid_idx]
                errors = errors[valid_idx] 
                labels = labels[valid_idx]
                
                if len(means) > 0:
                    x_pos = np.arange(len(labels))
                    bars = ax.bar(x_pos, means, yerr=errors, capsize=5, alpha=0.7)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylabel(f'Velocity ({velocity_unit}) (mean ± SEM)')
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add N values as text on bars
                    n_animals = stats_data[valid_idx][f'{window}_n_animals'].values
                    for i, (bar, n) in enumerate(zip(bars, n_animals)):
                        if not pd.isna(n):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + errors[i],
                                   f'n={int(n)}', ha='center', va='bottom', fontsize=8)
                    
                    # Color bars by treatment
                    treatments = stats_data[valid_idx]['treatment'].values
                    colors = plt.cm.Set1(np.linspace(0, 1, len(set(treatments))))
                    treat_colors = {treat: colors[i] for i, treat in enumerate(set(treatments))}
                    for bar, treat in zip(bars, treatments):
                        bar.set_color(treat_colors[treat])
                
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f"{output_dir}/summary_statistics.png", dpi=300, bbox_inches='tight')
                plt.savefig(f"{output_dir}/summary_statistics.pdf", bbox_inches='tight')
            
            plt.show()
    
    def save_results(self, output_dir: str = "analysis_results") -> None:
        """Save analysis results to CSV files"""
        Path(output_dir).mkdir(exist_ok=True)
        
        if self.velocity_data is not None:
            self.velocity_data.to_csv(f"{output_dir}/velocity_data.csv", index=False)
            print(f"Saved velocity data to {output_dir}/velocity_data.csv")
        
        if self.encounter_aligned_data is not None:
            self.encounter_aligned_data.to_csv(f"{output_dir}/aligned_data.csv", index=False)
            print(f"Saved aligned data to {output_dir}/aligned_data.csv")
        
        if self.summary_stats is not None:
            self.summary_stats.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
            print(f"Saved summary statistics to {output_dir}/summary_statistics.csv")
            
        # Save food leaving analysis if available
        if hasattr(self, 'food_leaving_data') and self.food_leaving_data is not None:
            self.food_leaving_data.to_csv(f"{output_dir}/food_leaving_analysis.csv", index=False)
            print(f"Saved food leaving analysis to {output_dir}/food_leaving_analysis.csv")

# GUI Application
class AnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Food Encounter Velocity Analysis")
        self.root.geometry("800x600")
        
        self.analyzer = FoodEncounterAnalyzer()
        self.setup_gui()
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # File selection
        ttk.Label(main_frame, text="Composite CSV File:").grid(row=0, column=0, sticky="w", pady=5)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var, width=60).pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side="right", padx=(5,0))
        
        # Parameters
        params_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding=10)
        params_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Label(params_frame, text="Time before encounter (s):").grid(row=0, column=0, sticky="w")
        self.time_before_var = tk.StringVar(value="120")
        ttk.Entry(params_frame, textvariable=self.time_before_var, width=10).grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(params_frame, text="Time after encounter (s):").grid(row=1, column=0, sticky="w")
        self.time_after_var = tk.StringVar(value="120") 
        ttk.Entry(params_frame, textvariable=self.time_after_var, width=10).grid(row=1, column=1, sticky="w", padx=5)
        
        ttk.Label(params_frame, text="Velocity smoothing window (s):").grid(row=2, column=0, sticky="w")
        self.smooth_var = tk.StringVar(value="1.0")
        ttk.Entry(params_frame, textvariable=self.smooth_var, width=10).grid(row=2, column=1, sticky="w", padx=5)
        
        ttk.Label(params_frame, text="Pixels per mm:").grid(row=3, column=0, sticky="w")
        self.pixels_per_mm_var = tk.StringVar(value="104.0")
        ttk.Entry(params_frame, textvariable=self.pixels_per_mm_var, width=10).grid(row=3, column=1, sticky="w", padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Show Plots", command=self.show_plots).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side="left", padx=5)
        
        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Analysis Log", padding=5)
        log_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=10)
        
        self.log_text = tk.Text(log_frame, height=15, wrap="word")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Redirect print statements to log
        import sys
        
        class TextRedirector:
            def __init__(self, widget):
                self.widget = widget
                
            def write(self, text):
                self.widget.insert("end", text)
                self.widget.see("end")
                self.widget.update_idletasks()
                
            def flush(self):
                pass
        
        sys.stdout = TextRedirector(self.log_text)
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Composite CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_var.set(filename)
    
    def run_analysis(self):
        try:
            csv_file = self.file_var.get()
            if not csv_file:
                messagebox.showerror("Error", "Please select a CSV file")
                return
                
            # Clear log
            self.log_text.delete(1.0, "end")
            
            # Get parameters
            time_before = float(self.time_before_var.get())
            time_after = float(self.time_after_var.get())
            smooth_window = float(self.smooth_var.get())
            pixels_per_mm = float(self.pixels_per_mm_var.get())
            
            # Update analyzer with new pixels_per_mm
            self.analyzer.pixels_per_mm = pixels_per_mm
            
            # Load data
            if not self.analyzer.load_data(csv_file):
                return
            
            # Run analysis
            self.analyzer.calculate_velocities(smooth_window)
            self.analyzer.align_to_food_encounters(time_before, time_after)
            self.analyzer.calculate_summary_statistics()
            self.analyzer.print_summary_report()
            
            messagebox.showinfo("Success", "Analysis completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            print(f"ERROR: {e}")
    
    def show_plots(self):
        if self.analyzer.encounter_aligned_data is None:
            messagebox.showwarning("Warning", "No analysis data available. Run analysis first.")
            return
        
        try:
            # Check if show_individual_var exists (GUI fully initialized)
            show_individual = getattr(self, 'show_individual_var', tk.BooleanVar(value=False)).get()
            self.analyzer.plot_velocity_profiles(save_plots=True, show_individual_traces=show_individual)
        except Exception as e:
            messagebox.showerror("Error", f"Plot generation failed: {e}")
    
    def save_results(self):
        if self.analyzer.velocity_data is None:
            messagebox.showwarning("Warning", "No analysis data available. Run analysis first.")
            return
            
        try:
            self.analyzer.save_results()
            messagebox.showinfo("Success", "Results saved to 'analysis_results' directory")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")
    
    def run(self):
        self.root.mainloop()

# Command line interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Food Encounter Velocity Analysis")
    parser.add_argument("--csv", help="Path to composite CSV file")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("--time-before", type=float, default=120, help="Time before encounter (s)")
    parser.add_argument("--time-after", type=float, default=120, help="Time after encounter (s)")
    parser.add_argument("--smooth", type=float, default=1.0, help="Velocity smoothing window (s)")
    parser.add_argument("--pixels-per-mm", type=float, default=104.0, 
                        help="Pixel to mm conversion factor (default: 104 pix/mm)")
    parser.add_argument("--show-individual", action="store_true", 
                        help="Show individual animal traces on plots")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    if args.gui or not args.csv:
        # Launch GUI
        app = AnalysisApp()
        app.run()
    else:
        # Command line analysis
        analyzer = FoodEncounterAnalyzer(pixels_per_mm=args.pixels_per_mm)
        
        if analyzer.load_data(args.csv):
            analyzer.calculate_velocities(args.smooth)
            analyzer.align_to_food_encounters(args.time_before, args.time_after)
            analyzer.calculate_summary_statistics()
            
            # Add food leaving analysis
            leaving_results = analyzer.analyze_food_leaving_behavior()
            if leaving_results is not None:
                analyzer.food_leaving_data = leaving_results
            
            analyzer.print_summary_report()
            
            if not args.no_plots:
                analyzer.plot_velocity_profiles(show_individual_traces=args.show_individual)
            
            analyzer.save_results()

if __name__ == "__main__":
    main()