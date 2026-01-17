#!/usr/bin/env python3
"""
Food Encounter Population Analysis Script - GMM Version

Analyzes C. elegans behavioral data around food encounters using Gaussian Mixture Models
to DISCOVER if distinct behavioral populations exist and how many.

Uses BIC model selection to test k=1 through k=6 components:
- k=1 → No distinct populations (continuous variation)
- k=2+ → Multiple distinct responder types

Features analyzed:
- Mean speed before encounter (-15 to -5 frames)
- Mean speed after encounter (+5 to +15 frames)
- Delta speed (after - before)
- Acceleration (slope during transition)

Input: CSV from tracking with food_encounter column
Output: BIC comparison, cluster assignments, validation plots
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

class FoodEncounterAnalyzer:
    """Analyze food encounter behavioral data using GMM"""
    
    # Conversion factor (adjust based on your setup)
    PIXELS_PER_MM = 104.0
    MAX_COMPONENTS = 6  # Test up to 6 populations
    
    def __init__(self, csv_path: str, output_dir: str, 
                 sex_filter=None, genotype_filter=None, treatment_filter=None):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.sex_filter = sex_filter
        self.genotype_filter = genotype_filter
        self.treatment_filter = treatment_filter
        
        self.df = None
        self.animal_ids = []
        self.encounter_data = {}
        self.cluster_labels = None
        self.cluster_animals = []
        self.progress_callback = None
        
        # GMM results
        self.gmm_models = {}
        self.bic_scores = {}
        self.silhouette_scores = {}
        self.best_k = None
        self.best_gmm = None
        
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self.progress_callback = callback
        
    def update_progress(self, message: str):
        """Update progress via callback"""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)
    
    def load_data(self):
        """Load and validate CSV data"""
        self.update_progress(f"Loading {self.csv_path.name}...")
        self.df = pd.read_csv(self.csv_path)
        
        # Check required columns
        required = ['time', 'x', 'y', 'food_encounter']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for ID columns
        id_cols = ['pc_number', 'assay_num', 'track_num']
        missing_id = [col for col in id_cols if col not in self.df.columns]
        if missing_id:
            raise ValueError(f"Missing ID columns: {missing_id}")
        
        self.update_progress(f"Loaded {len(self.df)} rows")
        
    def apply_filters(self):
        """Apply sex, genotype, and treatment filters"""
        initial_count = len(self.df)
        
        if self.sex_filter and self.sex_filter != "All":
            self.df = self.df[self.df['sex'] == self.sex_filter]
            self.update_progress(f"Filtered by sex={self.sex_filter}: {len(self.df)} rows")
        
        if self.genotype_filter and self.genotype_filter != "All":
            self.df = self.df[self.df['strain_genotype'] == self.genotype_filter]
            self.update_progress(f"Filtered by genotype={self.genotype_filter}: {len(self.df)} rows")
        
        if self.treatment_filter and self.treatment_filter != "All":
            self.df = self.df[self.df['treatment'] == self.treatment_filter]
            self.update_progress(f"Filtered by treatment={self.treatment_filter}: {len(self.df)} rows")
        
        if len(self.df) == 0:
            raise ValueError("No data remaining after filtering!")
        
        self.update_progress(f"Applied filters: {initial_count} → {len(self.df)} rows")
    
    def create_animal_ids(self):
        """Create unique animal IDs"""
        self.update_progress("Creating unique animal IDs...")
        
        self.df['animal_id'] = (
            'PC' + self.df['pc_number'].astype(str) + 
            '_A' + self.df['assay_num'].astype(str) + 
            '_T' + self.df['track_num'].astype(str)
        )
        
        self.animal_ids = sorted(self.df['animal_id'].unique())
        self.update_progress(f"Found {len(self.animal_ids)} unique animals")
    
    def calculate_speed(self):
        """Calculate speed for each animal (in mm/s)"""
        self.update_progress("Calculating speeds...")
        
        self.df['speed'] = np.nan
        
        for i, animal_id in enumerate(self.animal_ids):
            if i % 10 == 0:
                self.update_progress(f"Calculating speed: {i+1}/{len(self.animal_ids)}")
            
            mask = self.df['animal_id'] == animal_id
            animal_df = self.df[mask].copy()
            animal_df = animal_df.sort_values('time').reset_index(drop=True)
            
            # Calculate displacement
            dx = animal_df['x'].diff()
            dy = animal_df['y'].diff()
            dt = animal_df['time'].diff()
            
            # Speed = distance / time
            distance = np.sqrt(dx**2 + dy**2)
            speed_pixels = distance / dt
            
            # Convert to mm/s
            speed_mm = speed_pixels / self.PIXELS_PER_MM
            speed_mm = speed_mm.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Store in main dataframe
            original_indices = self.df[mask].sort_values('time').index
            self.df.loc[original_indices, 'speed'] = speed_mm.values
        
        self.update_progress("Speed calculation complete")
    
    def analyze_food_encounters(self):
        """Analyze speed around food encounters for each animal"""
        self.update_progress("Analyzing food encounters...")
        
        for animal_id in self.animal_ids:
            animal_df = self.df[self.df['animal_id'] == animal_id].copy()
            animal_df = animal_df.sort_values('time').reset_index(drop=True)
            
            # Find food encounter events
            food_events = animal_df[animal_df['food_encounter'] == 'food']
            
            if len(food_events) == 0:
                continue
            
            # For each food encounter
            for idx, event_row in food_events.iterrows():
                event_frame = event_row['time']
                event_idx = animal_df[animal_df['time'] == event_frame].index[0]
                
                # Define windows: -15 to -5 (before) and +5 to +15 (after)
                before_start = max(0, event_idx - 15)
                before_end = max(0, event_idx - 5)
                after_start = min(len(animal_df), event_idx + 5)
                after_end = min(len(animal_df), event_idx + 15)
                
                # Extract speed data
                before_speeds = animal_df.iloc[before_start:before_end]['speed'].values
                after_speeds = animal_df.iloc[after_start:after_end]['speed'].values
                
                # Calculate transition period for slope (from -5 to +5)
                transition_start = max(0, event_idx - 5)
                transition_end = min(len(animal_df), event_idx + 5)
                transition_speeds = animal_df.iloc[transition_start:transition_end]['speed'].values
                
                if len(before_speeds) < 3 or len(after_speeds) < 3:
                    continue
                
                # Calculate metrics
                mean_before = np.mean(before_speeds)
                mean_after = np.mean(after_speeds)
                delta_speed = mean_after - mean_before
                
                # Calculate acceleration (slope during transition)
                if len(transition_speeds) >= 3:
                    x_trans = np.arange(len(transition_speeds))
                    slope, intercept, r_val, p_val, std_err = stats.linregress(
                        x_trans, transition_speeds
                    )
                else:
                    slope = np.nan
                
                # Store encounter data
                encounter_key = f"{animal_id}_t{event_frame}"
                self.encounter_data[encounter_key] = {
                    'animal_id': animal_id,
                    'event_time': event_frame,
                    'event_idx': event_idx,
                    'mean_before': mean_before,
                    'mean_after': mean_after,
                    'delta_speed': delta_speed,
                    'slope': slope,
                    'before_speeds': before_speeds,
                    'after_speeds': after_speeds,
                    'full_trace': animal_df['speed'].values,
                    'time_trace': animal_df['time'].values,
                }
        
        self.update_progress(f"Found {len(self.encounter_data)} food encounter events")
    
    def prepare_clustering_data(self):
        """Prepare feature matrix for GMM"""
        self.update_progress("Preparing data for clustering...")
        
        cluster_data = []
        encounter_keys = []
        
        for key, data in self.encounter_data.items():
            mean_before = data['mean_before']
            mean_after = data['mean_after']
            delta = data['delta_speed']
            slope = data['slope']
            
            if np.isnan(mean_before) or np.isnan(mean_after) or np.isnan(delta) or np.isnan(slope):
                continue
            
            cluster_data.append([mean_before, mean_after, delta, slope])
            encounter_keys.append(key)
        
        if len(cluster_data) < 2:
            raise ValueError("Not enough valid data for clustering")
        
        self.cluster_data = np.array(cluster_data)
        self.encounter_keys = encounter_keys
        
        # Standardize features
        self.scaler = StandardScaler()
        self.cluster_data_scaled = self.scaler.fit_transform(self.cluster_data)
        
        self.update_progress(f"Prepared {len(self.cluster_data)} encounters for clustering")
    
    def fit_gmm_models(self):
        """Fit GMM for k=1 through MAX_COMPONENTS and compare using BIC"""
        self.update_progress("Fitting Gaussian Mixture Models...")
        self.update_progress("=" * 60)
        
        for k in range(1, self.MAX_COMPONENTS + 1):
            self.update_progress(f"Fitting GMM with {k} component(s)...")
            
            # Fit GMM
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=42,
                n_init=10
            )
            gmm.fit(self.cluster_data_scaled)
            
            # Calculate BIC
            bic = gmm.bic(self.cluster_data_scaled)
            
            # Calculate silhouette score (only for k>1)
            if k > 1:
                labels = gmm.predict(self.cluster_data_scaled)
                silhouette = silhouette_score(self.cluster_data_scaled, labels)
            else:
                silhouette = np.nan
            
            # Store results
            self.gmm_models[k] = gmm
            self.bic_scores[k] = bic
            self.silhouette_scores[k] = silhouette
            
            # Format silhouette score
            sil_str = f"{silhouette:.3f}" if not np.isnan(silhouette) else "N/A"
            self.update_progress(f"  k={k}: BIC={bic:.2f}, Silhouette={sil_str}")
        
        # Find best model (lowest BIC)
        self.best_k = min(self.bic_scores, key=self.bic_scores.get)
        self.best_gmm = self.gmm_models[self.best_k]
        
        self.update_progress("=" * 60)
        self.update_progress(f"BEST MODEL: k={self.best_k} components (lowest BIC={self.bic_scores[self.best_k]:.2f})")
        
        # Calculate delta BIC from best model
        self.update_progress("\nΔBIC from best model (lower is better):")
        for k in sorted(self.bic_scores.keys()):
            delta_bic = self.bic_scores[k] - self.bic_scores[self.best_k]
            stars = ""
            if delta_bic == 0:
                stars = " ★ BEST"
            elif delta_bic > 10:
                stars = " (very strong evidence against)"
            elif delta_bic > 6:
                stars = " (strong evidence against)"
            elif delta_bic > 2:
                stars = " (positive evidence against)"
            
            self.update_progress(f"  k={k}: ΔBIC={delta_bic:+.2f}{stars}")
        
        self.update_progress("=" * 60)
        
        # Interpret results
        if self.best_k == 1:
            self.update_progress("\n⚠️  INTERPRETATION: No distinct populations detected!")
            self.update_progress("    Animals show CONTINUOUS VARIATION in food responses.")
            self.update_progress("    There are no discrete 'responder types'.")
        else:
            self.update_progress(f"\n✓  INTERPRETATION: {self.best_k} distinct behavioral populations detected!")
            self.update_progress(f"    Animals naturally fall into {self.best_k} response categories.")
            if self.best_k > 1 and not np.isnan(self.silhouette_scores[self.best_k]):
                sil = self.silhouette_scores[self.best_k]
                if sil > 0.5:
                    self.update_progress(f"    Silhouette={sil:.3f} indicates well-separated clusters.")
                elif sil > 0.25:
                    self.update_progress(f"    Silhouette={sil:.3f} indicates moderate separation.")
                else:
                    self.update_progress(f"    Silhouette={sil:.3f} indicates weak separation (consider k=1).")
    
    def assign_clusters(self):
        """Assign cluster labels using best GMM"""
        if self.best_k == 1:
            self.update_progress("\nNo clustering performed (k=1 is best model)")
            # Assign all to single cluster
            for key in self.encounter_keys:
                self.encounter_data[key]['cluster'] = 'Single_Population'
            self.cluster_labels = ['Single_Population'] * len(self.encounter_keys)
        else:
            self.update_progress(f"\nAssigning encounters to {self.best_k} clusters...")
            
            # Get cluster assignments
            labels = self.best_gmm.predict(self.cluster_data_scaled)
            
            # Get probabilities (soft clustering)
            probabilities = self.best_gmm.predict_proba(self.cluster_data_scaled)
            
            # Sort clusters by mean delta_speed (most negative = Cluster 0)
            cluster_deltas = {}
            for k in range(self.best_k):
                mask = labels == k
                cluster_deltas[k] = self.cluster_data[mask, 2].mean()  # delta is column 2
            
            # Create mapping: sorted by delta_speed (most negative first)
            sorted_clusters = sorted(cluster_deltas.items(), key=lambda x: x[1])
            cluster_mapping = {old_k: new_k for new_k, (old_k, _) in enumerate(sorted_clusters)}
            
            # Remap labels
            labels = np.array([cluster_mapping[l] for l in labels])
            
            # Assign to encounter data
            self.cluster_labels = []
            for i, key in enumerate(self.encounter_keys):
                cluster_id = labels[i]
                cluster_name = f"Population_{cluster_id + 1}"
                self.encounter_data[key]['cluster'] = cluster_name
                self.encounter_data[key]['cluster_prob'] = probabilities[i, cluster_id]
                self.cluster_labels.append(cluster_name)
            
            # Report cluster sizes and characteristics
            for k in range(self.best_k):
                cluster_name = f"Population_{k + 1}"
                count = sum(1 for l in self.cluster_labels if l == cluster_name)
                mean_delta = np.mean([d['delta_speed'] for d in self.encounter_data.values() 
                                     if d.get('cluster') == cluster_name])
                self.update_progress(f"  {cluster_name}: n={count}, mean Δspeed={mean_delta:.3f} mm/s")
        
        # Save cluster assignments
        cluster_df = pd.DataFrame([
            {
                'encounter_key': key,
                'animal_id': data['animal_id'],
                'event_time': data['event_time'],
                'cluster': data.get('cluster', 'Unknown'),
                'cluster_probability': data.get('cluster_prob', 1.0),
                'mean_before': data['mean_before'],
                'mean_after': data['mean_after'],
                'delta_speed': data['delta_speed'],
                'slope': data['slope'],
            }
            for key, data in self.encounter_data.items() if key in self.encounter_keys
        ])
        
        output_path = self.output_dir / 'cluster_assignments.csv'
        cluster_df.to_csv(output_path, index=False)
        self.update_progress(f"\nSaved: {output_path.name}")
    
    def plot_bic_comparison(self):
        """Plot BIC scores across different k values"""
        self.update_progress("Creating BIC comparison plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: BIC scores
        k_values = sorted(self.bic_scores.keys())
        bic_values = [self.bic_scores[k] for k in k_values]
        
        ax1.plot(k_values, bic_values, 'o-', linewidth=2, markersize=8)
        ax1.axvline(self.best_k, color='red', linestyle='--', linewidth=2, 
                   label=f'Best k={self.best_k}', alpha=0.7)
        ax1.set_xlabel('Number of Components (k)', fontsize=12)
        ax1.set_ylabel('BIC Score (lower is better)', fontsize=12)
        ax1.set_title('BIC Model Selection', fontsize=14, fontweight='bold')
        ax1.set_xticks(k_values)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Silhouette scores (for k>1)
        k_values_sil = [k for k in k_values if k > 1]
        sil_values = [self.silhouette_scores[k] for k in k_values_sil]
        
        ax2.plot(k_values_sil, sil_values, 's-', linewidth=2, markersize=8, color='green')
        if self.best_k > 1:
            ax2.axvline(self.best_k, color='red', linestyle='--', linewidth=2, 
                       label=f'Best k={self.best_k}', alpha=0.7)
        ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Good separation (>0.5)')
        ax2.axhline(0.25, color='gray', linestyle=':', alpha=0.5, label='Weak separation (>0.25)')
        ax2.set_xlabel('Number of Components (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score (higher is better)', fontsize=12)
        ax2.set_title('Cluster Separation Quality', fontsize=14, fontweight='bold')
        ax2.set_xticks(k_values_sil)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, 1.0)
        
        plt.tight_layout()
        output_path = self.output_dir / 'bic_model_selection.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.update_progress(f"Saved: {output_path.name}")
        plt.close()
    
    def plot_population_analysis(self):
        """Create comprehensive visualization of populations"""
        self.update_progress("Creating population analysis plots...")
        
        if self.best_k == 1:
            self.update_progress("Skipping cluster plots (k=1, single population)")
            return
        
        # Collect data by cluster
        clusters = {}
        for key, data in self.encounter_data.items():
            if 'cluster' not in data or key not in self.encounter_keys:
                continue
            cluster_name = data['cluster']
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(data)
        
        # Define colors for populations
        colors = plt.cm.tab10(np.linspace(0, 1, self.best_k))
        cluster_colors = {f"Population_{i+1}": colors[i] for i in range(self.best_k)}
        
        # Create figure
        n_rows = 2 + (self.best_k + 1) // 2  # Dynamic rows based on number of clusters
        fig = plt.figure(figsize=(16, 5 * n_rows))
        gs = fig.add_gridspec(n_rows, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Delta speed comparison
        ax1 = fig.add_subplot(gs[0, 0])
        deltas_by_cluster = [
            [d['delta_speed'] for d in cluster_data]
            for cluster_data in clusters.values()
        ]
        bp = ax1.boxplot(deltas_by_cluster, labels=list(clusters.keys()), patch_artist=True)
        for patch, cluster_name in zip(bp['boxes'], clusters.keys()):
            patch.set_facecolor(cluster_colors[cluster_name])
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Δ Speed (mm/s)', fontsize=12)
        ax1.set_title('Speed Change by Population', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Panel 2: PCA visualization
        ax2 = fig.add_subplot(gs[0, 1])
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.cluster_data_scaled)
        
        for cluster_name, cluster_data in clusters.items():
            indices = [i for i, key in enumerate(self.encounter_keys) 
                      if self.encounter_data[key].get('cluster') == cluster_name]
            ax2.scatter(data_pca[indices, 0], data_pca[indices, 1],
                       c=[cluster_colors[cluster_name]], alpha=0.6, s=50,
                       edgecolors='black', linewidth=0.5, label=cluster_name)
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax2.set_title('PCA of Behavioral Features', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Delta speed histogram
        ax3 = fig.add_subplot(gs[1, 0])
        for cluster_name, cluster_data in clusters.items():
            deltas = [d['delta_speed'] for d in cluster_data]
            ax3.hist(deltas, bins=20, alpha=0.5, color=cluster_colors[cluster_name],
                    label=cluster_name, density=True)
        ax3.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Δ Speed (mm/s)', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Distribution of Speed Changes', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Before vs After speeds
        ax4 = fig.add_subplot(gs[1, 1])
        for cluster_name, cluster_data in clusters.items():
            for d in cluster_data:
                ax4.plot([0, 1], [d['mean_before'], d['mean_after']],
                        color=cluster_colors[cluster_name], alpha=0.3, linewidth=1)
        
        # Add cluster means
        for cluster_name, cluster_data in clusters.items():
            mean_before = np.mean([d['mean_before'] for d in cluster_data])
            mean_after = np.mean([d['mean_after'] for d in cluster_data])
            ax4.plot([0, 1], [mean_before, mean_after],
                    color=cluster_colors[cluster_name], linewidth=3,
                    marker='o', markersize=10, label=cluster_name)
        
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Before', 'After'])
        ax4.set_ylabel('Speed (mm/s)', fontsize=12)
        ax4.set_title('Speed Before vs After by Population', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panels 5+: Individual traces for each population
        row = 2
        col = 0
        for cluster_name, cluster_data in clusters.items():
            ax = fig.add_subplot(gs[row, col])
            
            # Plot individual traces
            for d in cluster_data[:30]:  # Show up to 30 traces
                event_idx = d['event_idx']
                speeds = d['full_trace']
                relative_frames = np.arange(len(speeds)) - event_idx
                ax.plot(relative_frames, speeds,
                       color=cluster_colors[cluster_name], alpha=0.3, linewidth=1)
            
            # Add average trace
            max_len = 60
            traces = []
            for d in cluster_data:
                event_idx = d['event_idx']
                speeds = d['full_trace']
                start_idx = max(0, event_idx - 30)
                end_idx = min(len(speeds), event_idx + 30)
                aligned_speeds = speeds[start_idx:end_idx]
                if len(aligned_speeds) < max_len:
                    aligned_speeds = np.pad(aligned_speeds, (0, max_len - len(aligned_speeds)),
                                          'constant', constant_values=np.nan)
                traces.append(aligned_speeds[:max_len])
            
            traces = np.array(traces)
            frames = np.arange(-30, 30)
            mean_trace = np.nanmean(traces, axis=0)
            sem_trace = np.nanstd(traces, axis=0) / np.sqrt(np.sum(~np.isnan(traces), axis=0))
            
            ax.plot(frames, mean_trace, color=cluster_colors[cluster_name],
                   linewidth=3, label='Mean')
            ax.fill_between(frames, mean_trace - sem_trace, mean_trace + sem_trace,
                           color=cluster_colors[cluster_name], alpha=0.2)
            
            ax.axvline(0, color='black', linewidth=2, linestyle='--', alpha=0.8)
            ax.axvline(-10, color='green', linewidth=1, linestyle=':', alpha=0.5)
            ax.axvline(10, color='green', linewidth=1, linestyle=':', alpha=0.5)
            ax.set_xlabel('Frames relative to food encounter', fontsize=11)
            ax.set_ylabel('Speed (mm/s)', fontsize=11)
            ax.set_title(f'{cluster_name} (n={len(cluster_data)})',
                        fontsize=12, fontweight='bold',
                        color=cluster_colors[cluster_name])
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-30, 30)
            
            # Move to next subplot position
            col += 1
            if col >= 2:
                col = 0
                row += 1
        
        plt.tight_layout()
        output_path = self.output_dir / 'population_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.update_progress(f"Saved: {output_path.name}")
        plt.close()
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        try:
            self.load_data()
            self.apply_filters()
            self.create_animal_ids()
            self.calculate_speed()
            self.analyze_food_encounters()
            
            if len(self.encounter_data) == 0:
                self.update_progress("ERROR: No food encounters found in filtered data!")
                return False
            
            self.prepare_clustering_data()
            self.fit_gmm_models()
            self.assign_clusters()
            self.plot_bic_comparison()
            self.plot_population_analysis()
            
            self.update_progress("\n✓ Analysis complete!")
            return True
        except Exception as e:
            self.update_progress(f"ERROR: {str(e)}")
            import traceback
            self.update_progress(traceback.format_exc())
            return False


class FoodEncounterGUI:
    """GUI for food encounter analysis"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Food Encounter Population Analysis (GMM)")
        self.root.geometry("700x550")
        
        self.input_file = None
        self.output_dir = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup GUI components"""
        # Title
        title_frame = ttk.Frame(self.root, padding=10)
        title_frame.pack(fill='x')
        ttk.Label(title_frame, text="Food Encounter Population Analysis", 
                 font=('TkDefaultFont', 16, 'bold')).pack()
        ttk.Label(title_frame, text="Discovers natural behavioral populations using GMM + BIC",
                 font=('TkDefaultFont', 10, 'italic')).pack()
        
        # Input file
        input_frame = ttk.LabelFrame(self.root, text="Input File", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        self.input_label = ttk.Label(input_frame, text="No file selected")
        self.input_label.pack(side='left', expand=True, fill='x')
        
        ttk.Button(input_frame, text="Browse...", 
                  command=self.select_input_file).pack(side='right')
        
        # Filters
        filter_frame = ttk.LabelFrame(self.root, text="Filters", padding=10)
        filter_frame.pack(fill='x', padx=10, pady=5)
        
        # Sex filter
        sex_row = ttk.Frame(filter_frame)
        sex_row.pack(fill='x', pady=2)
        ttk.Label(sex_row, text="Sex:", width=15).pack(side='left')
        self.sex_var = tk.StringVar(value="All")
        self.sex_combo = ttk.Combobox(sex_row, textvariable=self.sex_var, 
                                      state='readonly', width=20)
        self.sex_combo['values'] = ['All']
        self.sex_combo.pack(side='left', padx=5)
        
        # Genotype filter
        geno_row = ttk.Frame(filter_frame)
        geno_row.pack(fill='x', pady=2)
        ttk.Label(geno_row, text="Genotype:", width=15).pack(side='left')
        self.genotype_var = tk.StringVar(value="All")
        self.genotype_combo = ttk.Combobox(geno_row, textvariable=self.genotype_var, 
                                           state='readonly', width=20)
        self.genotype_combo['values'] = ['All']
        self.genotype_combo.pack(side='left', padx=5)
        
        # Treatment filter
        treat_row = ttk.Frame(filter_frame)
        treat_row.pack(fill='x', pady=2)
        ttk.Label(treat_row, text="Treatment:", width=15).pack(side='left')
        self.treatment_var = tk.StringVar(value="All")
        self.treatment_combo = ttk.Combobox(treat_row, textvariable=self.treatment_var, 
                                            state='readonly', width=20)
        self.treatment_combo['values'] = ['All']
        self.treatment_combo.pack(side='left', padx=5)
        
        # Output directory
        output_frame = ttk.LabelFrame(self.root, text="Output Directory", padding=10)
        output_frame.pack(fill='x', padx=10, pady=5)
        
        self.output_label = ttk.Label(output_frame, text="No directory selected")
        self.output_label.pack(side='left', expand=True, fill='x')
        
        ttk.Button(output_frame, text="Browse...", 
                  command=self.select_output_dir).pack(side='right')
        
        # Run button
        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.pack(fill='x')
        
        self.run_button = ttk.Button(button_frame, text="Run Analysis", 
                                     command=self.run_analysis, state='disabled')
        self.run_button.pack()
        
        # Progress
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=10)
        progress_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Text widget with scrollbar
        text_scroll = ttk.Scrollbar(progress_frame)
        text_scroll.pack(side='right', fill='y')
        
        self.progress_text = tk.Text(progress_frame, height=10, 
                                     yscrollcommand=text_scroll.set)
        self.progress_text.pack(fill='both', expand=True)
        text_scroll.config(command=self.progress_text.yview)
    
    def select_input_file(self):
        """Select input CSV file and detect available filters"""
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.input_file = file_path
            self.input_label.config(text=Path(file_path).name)
            
            # Read file to get available filter values
            try:
                df = pd.read_csv(file_path)
                
                if 'sex' in df.columns:
                    sex_values = ['All'] + sorted(df['sex'].unique().tolist())
                    self.sex_combo['values'] = sex_values
                
                if 'strain_genotype' in df.columns:
                    geno_values = ['All'] + sorted(df['strain_genotype'].unique().tolist())
                    self.genotype_combo['values'] = geno_values
                
                if 'treatment' in df.columns:
                    treat_values = ['All'] + sorted(df['treatment'].unique().tolist())
                    self.treatment_combo['values'] = treat_values
                
                self.update_progress(f"Loaded filter options from {Path(file_path).name}")
            except Exception as e:
                self.update_progress(f"Warning: Could not read filter options: {e}")
            
            self.check_ready()
    
    def select_output_dir(self):
        """Select output directory"""
        dir_path = filedialog.askdirectory(
            title="Select output directory"
        )
        
        if dir_path:
            self.output_dir = dir_path
            self.output_label.config(text=Path(dir_path).name)
            self.check_ready()
    
    def check_ready(self):
        """Check if ready to run analysis"""
        if self.input_file and self.output_dir:
            self.run_button.config(state='normal')
        else:
            self.run_button.config(state='disabled')
    
    def update_progress(self, message: str):
        """Update progress text"""
        self.progress_text.insert('end', message + '\n')
        self.progress_text.see('end')
        self.root.update()
    
    def run_analysis_thread(self):
        """Run analysis in background thread"""
        sex_filter = self.sex_var.get()
        genotype_filter = self.genotype_var.get()
        treatment_filter = self.treatment_var.get()
        
        analyzer = FoodEncounterAnalyzer(
            self.input_file, 
            self.output_dir,
            sex_filter=sex_filter,
            genotype_filter=genotype_filter,
            treatment_filter=treatment_filter
        )
        analyzer.set_progress_callback(self.update_progress)
        
        success = analyzer.run_analysis()
        
        if success:
            self.root.after(0, lambda: messagebox.showinfo(
                "Complete", 
                f"Analysis complete!\n\nResults saved to:\n{self.output_dir}"
            ))
        else:
            self.root.after(0, lambda: messagebox.showerror(
                "Error", 
                "Analysis failed. Check the progress log for details."
            ))
        
        # Re-enable button
        self.root.after(0, lambda: self.run_button.config(state='normal'))
    
    def run_analysis(self):
        """Run analysis"""
        if not self.input_file or not self.output_dir:
            messagebox.showwarning("Missing Input", 
                                  "Please select both input file and output directory")
            return
        
        # Clear progress
        self.progress_text.delete('1.0', 'end')
        
        # Show selected filters
        self.update_progress(f"Filters: Sex={self.sex_var.get()}, " +
                           f"Genotype={self.genotype_var.get()}, " +
                           f"Treatment={self.treatment_var.get()}\n")
        
        # Disable button
        self.run_button.config(state='disabled')
        
        # Run in thread
        thread = threading.Thread(target=self.run_analysis_thread, daemon=True)
        thread.start()
    
    def run(self):
        """Start GUI"""
        self.root.mainloop()


def main():
    app = FoodEncounterGUI()
    app.run()


if __name__ == "__main__":
    main()
