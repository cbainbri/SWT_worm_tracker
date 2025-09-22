#!/usr/bin/env python3
"""
Interactive Food Encounter Velocity Analysis Pipeline with Contextual Behavioral Analysis

Analyzes animal velocity around food encounters using composite CSV data with enhanced
contextual behavioral measurements and statistical analysis capabilities.

New Features:
- Food Detection context: nearby_food_velocity / off_food_velocity ratios
- Food Encounter context: on_food_velocity / nearby_food_velocity ratios  
- Statistical analysis with mixed-effects models, transformations, and diagnostics
- Proper handling of unpaired (detection) vs paired (encounter) experimental designs

- Creates unique animal IDs from assay_num + track_num
- Calculates velocity from centroid positions (x, y coordinates)
- Aligns food encounters to time=0 for comparison
- Provides statistical summaries for different time windows
- Groups analysis by treatment, sex, and strain/genotype

Requirements: pandas, numpy, matplotlib, seaborn, scipy, statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from scipy.stats import boxcox, normaltest, levene
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Try to import advanced stats packages
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    from statsmodels.stats.diagnostic import het_breuschpagan
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Mixed-effects models will be unavailable.")

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
    """Main analysis class for food encounter velocity analysis with contextual behavioral measurements"""
    
    def __init__(self, pixels_per_mm: float = 104.0):
        self.data = None
        self.velocity_data = None
        self.encounter_aligned_data = None
        self.summary_stats = None
        self.contextual_velocities = None
        self.behavioral_ratios = None
        self.statistical_results = None
        self.pixels_per_mm = pixels_per_mm
        
    def load_data(self, csv_path: str) -> bool:
        """Load composite CSV data with enhanced validation for merge script output"""
        try:
            self.data = pd.read_csv(csv_path)
            print(f"Loaded data: {len(self.data)} rows, {len(self.data.columns)} columns")
            
            # Validate required columns for merge script output
            required_cols = ['assay_num', 'track_num', 'time', 'x', 'y', 
                           'treatment', 'sex', 'strain_genotype']
            optional_cols = ['food_encounter', 'nose_on_food', 'centroid_on_food', 
                           'pc_number', 'source_file']
            
            missing = [col for col in required_cols if col not in self.data.columns]
            if missing:
                print(f"ERROR: Missing required columns: {missing}")
                return False
            
            # Create missing optional columns if needed
            for col in optional_cols:
                if col not in self.data.columns:
                    if col == 'food_encounter':
                        self.data[col] = ''
                    elif col in ['nose_on_food', 'centroid_on_food']:
                        self.data[col] = 0
                    else:
                        self.data[col] = ''
                        
            # Create unique animal IDs (compatible with merge script output)
            if 'animal_id' not in self.data.columns:
                self.data['animal_id'] = self.data['assay_num'].astype(str) + '_' + self.data['track_num'].astype(str)
            
            print(f"Unique animals: {self.data['animal_id'].nunique()}")
            print(f"Treatments: {sorted(self.data['treatment'].unique())}")
            print(f"Animals with food encounters: {self.data[self.data['food_encounter'] == 'food']['animal_id'].nunique()}")
            
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
    
    def extract_contextual_velocities(self) -> Dict[str, pd.DataFrame]:
        """Extract velocities for different behavioral contexts"""
        if self.velocity_data is None:
            print("ERROR: No velocity data available")
            return {}
            
        print("Extracting contextual velocities...")
        
        contexts = {}
        
        # Off food velocities (from treatments ending in '_off')
        off_food_mask = self.velocity_data['treatment'].str.endswith('_off', na=False)
        contexts['off_food'] = self.velocity_data[off_food_mask].copy()
        
        # Food assay data (treatments NOT ending in '_off')
        food_assay_mask = ~self.velocity_data['treatment'].str.endswith('_off', na=False)
        food_data = self.velocity_data[food_assay_mask].copy()
        
        if len(food_data) > 0:
            # Nearby food velocities (nose_on_food=0 AND centroid_on_food=0)
            nearby_mask = (food_data['nose_on_food'] == 0) & (food_data['centroid_on_food'] == 0)
            contexts['nearby_food'] = food_data[nearby_mask].copy()
            
            # On food velocities (nose_on_food=1)
            on_food_mask = (food_data['nose_on_food'] == 1)
            contexts['on_food'] = food_data[on_food_mask].copy()
        else:
            contexts['nearby_food'] = pd.DataFrame()
            contexts['on_food'] = pd.DataFrame()
        
        self.contextual_velocities = contexts
        
        # Print summary
        for context, data in contexts.items():
            if len(data) > 0:
                print(f"{context}: {len(data)} data points, {data['animal_id'].nunique()} animals, "
                      f"mean velocity = {data['velocity'].mean():.3f}")
            else:
                print(f"{context}: No data")
                
        return contexts
    
    def calculate_behavioral_ratios(self) -> pd.DataFrame:
        """Calculate food detection and food encounter ratios per animal"""
        if self.contextual_velocities is None:
            self.extract_contextual_velocities()
            
        print("Calculating behavioral ratios...")
        
        ratios_data = []
        
        # Get all unique animals across all contexts
        all_animals = set()
        for context_data in self.contextual_velocities.values():
            if len(context_data) > 0:
                all_animals.update(context_data['animal_id'].unique())
        
        for animal_id in all_animals:
            # Get animal metadata (assuming consistent across timepoints)
            animal_meta = None
            for context_data in self.contextual_velocities.values():
                if len(context_data) > 0:
                    animal_subset = context_data[context_data['animal_id'] == animal_id]
                    if len(animal_subset) > 0:
                        animal_meta = animal_subset.iloc[0]
                        break
            
            if animal_meta is None:
                continue
                
            # Extract mean velocities for each context
            off_food_vel = self._get_mean_velocity(animal_id, 'off_food')
            nearby_food_vel = self._get_mean_velocity(animal_id, 'nearby_food')
            on_food_vel = self._get_mean_velocity(animal_id, 'on_food')
            
            # Calculate ratios
            food_detection_ratio = np.nan
            food_encounter_ratio = np.nan
            
            if not pd.isna(nearby_food_vel) and not pd.isna(off_food_vel) and off_food_vel > 0:
                food_detection_ratio = nearby_food_vel / off_food_vel
                
            if not pd.isna(on_food_vel) and not pd.isna(nearby_food_vel) and nearby_food_vel > 0:
                food_encounter_ratio = on_food_vel / nearby_food_vel
            
            ratios_data.append({
                'animal_id': animal_id,
                'assay_num': animal_meta['assay_num'],
                'track_num': animal_meta['track_num'],
                'treatment': animal_meta['treatment'],
                'sex': animal_meta['sex'],
                'strain_genotype': animal_meta['strain_genotype'],
                'pc_number': animal_meta.get('pc_number', ''),
                'off_food_velocity': off_food_vel,
                'nearby_food_velocity': nearby_food_vel,
                'on_food_velocity': on_food_vel,
                'food_detection_ratio': food_detection_ratio,
                'food_encounter_ratio': food_encounter_ratio,
                'has_detection_data': not pd.isna(food_detection_ratio),
                'has_encounter_data': not pd.isna(food_encounter_ratio)
            })
        
        self.behavioral_ratios = pd.DataFrame(ratios_data)
        
        print(f"Calculated ratios for {len(self.behavioral_ratios)} animals")
        print(f"Animals with food detection data: {self.behavioral_ratios['has_detection_data'].sum()}")
        print(f"Animals with food encounter data: {self.behavioral_ratios['has_encounter_data'].sum()}")
        
        return self.behavioral_ratios
    
    def _get_mean_velocity(self, animal_id: str, context: str) -> float:
        """Helper function to get mean velocity for an animal in a specific context"""
        if context not in self.contextual_velocities:
            return np.nan
            
        context_data = self.contextual_velocities[context]
        animal_data = context_data[context_data['animal_id'] == animal_id]
        
        if len(animal_data) == 0:
            return np.nan
            
        return animal_data['velocity'].mean()
    
    def perform_statistical_analysis(self, transformation: str = 'auto') -> Dict[str, any]:
        """Perform comprehensive statistical analysis of behavioral ratios"""
        if self.behavioral_ratios is None:
            self.calculate_behavioral_ratios()
            
        print("Performing statistical analysis...")
        
        results = {}
        
        # Analyze food detection ratios (unpaired design)
        detection_data = self.behavioral_ratios[self.behavioral_ratios['has_detection_data']].copy()
        if len(detection_data) > 0:
            results['food_detection'] = self._analyze_context(
                detection_data, 'food_detection_ratio', 'Food Detection', 
                transformation, paired=False
            )
        
        # Analyze food encounter ratios (paired design)
        encounter_data = self.behavioral_ratios[self.behavioral_ratios['has_encounter_data']].copy()
        if len(encounter_data) > 0:
            results['food_encounter'] = self._analyze_context(
                encounter_data, 'food_encounter_ratio', 'Food Encounter',
                transformation, paired=True
            )
        
        self.statistical_results = results
        return results
    
    def _analyze_context(self, data: pd.DataFrame, response_col: str, context_name: str, 
                        transformation: str, paired: bool) -> Dict[str, any]:
        """Analyze a specific behavioral context with appropriate statistical methods"""
        
        print(f"\nAnalyzing {context_name} context (paired={paired})...")
        
        result = {
            'context': context_name,
            'paired': paired,
            'n_animals': len(data),
            'raw_data': data.copy(),
            'transformation': None,
            'transformed_data': None,
            'normality_tests': {},
            'statistical_tests': {},
            'effect_sizes': {},
            'diagnostics': {}
        }
        
        # Extract response variable
        y = data[response_col].dropna()
        
        if len(y) < 3:
            print(f"Insufficient data for {context_name} analysis")
            return result
        
        # Apply transformation
        y_transformed, transform_info = self._apply_transformation(y, transformation)
        result['transformation'] = transform_info
        result['transformed_data'] = data.copy()
        result['transformed_data'][f'{response_col}_transformed'] = np.nan
        valid_idx = data[response_col].notna()
        result['transformed_data'].loc[valid_idx, f'{response_col}_transformed'] = y_transformed
        
        # Test normality
        result['normality_tests'] = self._test_normality(y, y_transformed, context_name)
        
        # Choose analysis based on data distribution and design
        if len(set(data['treatment'])) > 1 or len(set(data['sex'])) > 1:
            # Multi-factor analysis
            result['statistical_tests'] = self._perform_multifactor_analysis(
                result['transformed_data'], f'{response_col}_transformed', paired
            )
        else:
            # Single-factor analysis
            result['statistical_tests'] = self._perform_single_factor_analysis(
                y_transformed, context_name
            )
        
        return result
    
    def _apply_transformation(self, y: np.ndarray, transformation: str) -> Tuple[np.ndarray, Dict]:
        """Apply data transformation and return transformed data with info"""
        
        y_clean = y[np.isfinite(y) & (y > 0)]  # Remove inf, nan, and non-positive values
        
        if len(y_clean) < 3:
            return y, {'method': 'none', 'reason': 'insufficient_data'}
        
        transform_info = {'method': transformation, 'lambda': None, 'success': False}
        
        try:
            if transformation == 'auto':
                # Test if log transformation improves normality
                _, p_original = normaltest(y_clean)
                
                try:
                    y_log = np.log(y_clean)
                    _, p_log = normaltest(y_log)
                    
                    if p_log > p_original:
                        transformation = 'log'
                    else:
                        transformation = 'none'
                except:
                    transformation = 'none'
            
            if transformation == 'log':
                y_transformed = np.log(y_clean)
                transform_info.update({'method': 'log', 'success': True})
                
            elif transformation == 'sqrt':
                y_transformed = np.sqrt(y_clean)
                transform_info.update({'method': 'sqrt', 'success': True})
                
            elif transformation == 'boxcox':
                y_transformed, lambda_val = boxcox(y_clean)
                transform_info.update({'method': 'boxcox', 'lambda': lambda_val, 'success': True})
                
            else:  # no transformation
                y_transformed = y_clean
                transform_info.update({'method': 'none', 'success': True})
                
        except Exception as e:
            print(f"Transformation failed: {e}")
            y_transformed = y_clean
            transform_info.update({'method': 'none', 'success': False, 'error': str(e)})
        
        return y_transformed, transform_info
    
    def _test_normality(self, y_original: np.ndarray, y_transformed: np.ndarray, 
                       context_name: str) -> Dict[str, any]:
        """Test normality of original and transformed data"""
        
        results = {}
        
        for name, data in [('original', y_original), ('transformed', y_transformed)]:
            clean_data = data[np.isfinite(data)]
            
            if len(clean_data) < 3:
                results[name] = {'test': 'insufficient_data'}
                continue
                
            try:
                # Shapiro-Wilk test (better for small samples)
                if len(clean_data) <= 5000:
                    stat, p_value = stats.shapiro(clean_data)
                    test_name = 'shapiro'
                else:
                    # D'Agostino and Pearson's test for larger samples
                    stat, p_value = normaltest(clean_data)
                    test_name = 'dagostino'
                
                results[name] = {
                    'test': test_name,
                    'statistic': stat,
                    'p_value': p_value,
                    'normal': p_value > 0.05,
                    'mean': np.mean(clean_data),
                    'std': np.std(clean_data),
                    'skewness': stats.skew(clean_data),
                    'kurtosis': stats.kurtosis(clean_data)
                }
                
            except Exception as e:
                results[name] = {'test': 'failed', 'error': str(e)}
        
        return results
    
    def _perform_multifactor_analysis(self, data: pd.DataFrame, response_col: str, 
                                    paired: bool) -> Dict[str, any]:
        """Perform multi-factor statistical analysis"""
        
        results = {}
        
        # Clean data
        clean_data = data.dropna(subset=[response_col, 'treatment', 'sex', 'strain_genotype'])
        
        if len(clean_data) < 5:
            return {'error': 'insufficient_data'}
        
        # Mixed-effects model (if statsmodels available)
        if HAS_STATSMODELS:
            try:
                # Create formula for mixed-effects model
                if paired:
                    # For paired design, include animal as random effect
                    formula = f"{response_col} ~ treatment + sex + strain_genotype + treatment:sex"
                    model = mixedlm(formula, clean_data, groups=clean_data['animal_id'])
                else:
                    # For unpaired design, include assay as random effect
                    formula = f"{response_col} ~ treatment + sex + strain_genotype + treatment:sex"
                    model = mixedlm(formula, clean_data, groups=clean_data['assay_num'])
                
                fitted_model = model.fit()
                
                results['mixed_effects'] = {
                    'model': fitted_model,
                    'summary': str(fitted_model.summary()),
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'formula': formula,
                    'converged': fitted_model.converged
                }
                
            except Exception as e:
                results['mixed_effects'] = {'error': str(e)}
        
        # Traditional ANOVA (as backup)
        try:
            from scipy.stats import f_oneway
            
            # Group by treatment
            groups = []
            group_names = []
            for treatment in clean_data['treatment'].unique():
                group_data = clean_data[clean_data['treatment'] == treatment][response_col]
                if len(group_data) > 0:
                    groups.append(group_data)
                    group_names.append(treatment)
            
            if len(groups) > 1:
                f_stat, p_value = f_oneway(*groups)
                results['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'groups': group_names,
                    'group_sizes': [len(g) for g in groups]
                }
        
        except Exception as e:
            results['anova'] = {'error': str(e)}
        
        return results
    
    def _perform_single_factor_analysis(self, y: np.ndarray, context_name: str) -> Dict[str, any]:
        """Perform single-factor analysis (descriptive statistics)"""
        
        if len(y) < 3:
            return {'error': 'insufficient_data'}
        
        return {
            'descriptive': {
                'n': len(y),
                'mean': np.mean(y),
                'std': np.std(y),
                'sem': np.std(y) / np.sqrt(len(y)),
                'median': np.median(y),
                'q25': np.percentile(y, 25),
                'q75': np.percentile(y, 75),
                'min': np.min(y),
                'max': np.max(y)
            }
        }
    
    def align_to_food_encounters(self, time_before: float = 120, time_after: float = 120) -> None:
        """Align data to food encounter events (time = 0) - maintains existing functionality"""
        print(f"Aligning data to food encounters (-{time_before}s to +{time_after}s)...")
        
        aligned_data = []
        encounter_count = 0
        
        for animal_id in self.velocity_data['animal_id'].unique():
            animal_data = self.velocity_data[self.velocity_data['animal_id'] == animal_id].copy()
            
            # Find food encounter events (including fictive encounters in off food conditions)
            encounters = animal_data[animal_data['food_encounter'] == 'food']
            
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
        """Calculate summary statistics for different time windows - maintains existing functionality"""
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
    
    def print_behavioral_analysis_report(self) -> None:
        """Print comprehensive behavioral analysis report"""
        if self.behavioral_ratios is None:
            print("No behavioral ratio data available")
            return
            
        print("\n" + "="*80)
        print("BEHAVIORAL CONTEXT ANALYSIS REPORT")
        print("="*80)
        
        # Overall statistics
        total_animals = len(self.behavioral_ratios)
        detection_animals = self.behavioral_ratios['has_detection_data'].sum()
        encounter_animals = self.behavioral_ratios['has_encounter_data'].sum()
        
        print(f"Total animals analyzed: {total_animals}")
        print(f"Animals with food detection data: {detection_animals}")
        print(f"Animals with food encounter data: {encounter_animals}")
        
        velocity_unit = "mm/s" if self.pixels_per_mm > 0 else "pixels/time"
        print(f"\nVelocity units: {velocity_unit}")
        
        # Contextual velocity summary
        print(f"\nContextual Velocity Summary:")
        print(f"{'Context':<15} {'N Animals':<10} {'Mean Velocity':<15} {'Std Velocity':<15}")
        print("-" * 60)
        
        if self.contextual_velocities:
            for context, data in self.contextual_velocities.items():
                if len(data) > 0:
                    mean_vel = data['velocity'].mean()
                    std_vel = data['velocity'].std()
                    n_animals = data['animal_id'].nunique()
                    print(f"{context:<15} {n_animals:<10} {mean_vel:<15.3f} {std_vel:<15.3f}")
        
        # Behavioral ratios by group
        print(f"\nBehavioral Ratios by Group:")
        print("-" * 80)
        
        groups = ['treatment', 'sex', 'strain_genotype']
        for group_vals, group_data in self.behavioral_ratios.groupby(groups):
            treatment, sex, strain = group_vals
            
            print(f"\nGroup: {treatment} | {sex} | {strain}")
            print(f"Animals (n): {len(group_data)}")
            
            # Food detection ratios
            detection_data = group_data[group_data['has_detection_data']]
            if len(detection_data) > 0:
                det_mean = detection_data['food_detection_ratio'].mean()
                det_std = detection_data['food_detection_ratio'].std()
                det_sem = det_std / np.sqrt(len(detection_data))
                print(f"Food Detection Ratio: {det_mean:.3f} ± {det_std:.3f} (SEM: {det_sem:.3f}) [n={len(detection_data)}]")
            else:
                print("Food Detection Ratio: No data")
            
            # Food encounter ratios  
            encounter_data = group_data[group_data['has_encounter_data']]
            if len(encounter_data) > 0:
                enc_mean = encounter_data['food_encounter_ratio'].mean()
                enc_std = encounter_data['food_encounter_ratio'].std()
                enc_sem = enc_std / np.sqrt(len(encounter_data))
                print(f"Food Encounter Ratio: {enc_mean:.3f} ± {enc_std:.3f} (SEM: {enc_sem:.3f}) [n={len(encounter_data)}]")
            else:
                print("Food Encounter Ratio: No data")
        
        # Statistical results summary
        if self.statistical_results:
            print(f"\nStatistical Analysis Summary:")
            print("-" * 80)
            
            for context, results in self.statistical_results.items():
                print(f"\n{context.upper()} CONTEXT:")
                
                if 'mixed_effects' in results:
                    me_results = results['mixed_effects']
                    if 'error' not in me_results:
                        print(f"Mixed-Effects Model: Converged = {me_results.get('converged', 'Unknown')}")
                        print(f"AIC = {me_results.get('aic', 'N/A'):.2f}, BIC = {me_results.get('bic', 'N/A'):.2f}")
                
                if 'anova' in results:
                    anova_results = results['anova']
                    if 'error' not in anova_results:
                        print(f"ANOVA: F = {anova_results['f_statistic']:.3f}, p = {anova_results['p_value']:.3f}")
                
                if 'normality_tests' in results:
                    norm_results = results['normality_tests']
                    if 'transformed' in norm_results and 'error' not in norm_results['transformed']:
                        p_val = norm_results['transformed']['p_value']
                        normal = "Yes" if p_val > 0.05 else "No"
                        print(f"Data normality (transformed): {normal} (p = {p_val:.3f})")
        
        print("\n" + "="*80)
    
    def print_summary_report(self) -> None:
        """Print human-readable summary report - maintains existing functionality"""
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
    
    def analyze_food_leaving_behavior(self) -> Optional[pd.DataFrame]:
        """Analyze velocity patterns when animals leave food areas - maintains existing functionality"""
        if self.velocity_data is None:
            print("No velocity data available for food leaving analysis")
            return None
            
        print("Analyzing food leaving behavior...")
        
        leaving_data = []
        
        for animal_id in self.velocity_data['animal_id'].unique():
            animal_data = self.velocity_data[self.velocity_data['animal_id'] == animal_id].sort_values('time').copy()
            
            # Find transitions from food encounter to no encounter
            food_shifts = animal_data['nose_on_food'].diff()
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
    
    def plot_behavioral_ratios(self, save_plots: bool = True, output_dir: str = "plots") -> None:
        """Create plots for behavioral ratio analysis"""
        if self.behavioral_ratios is None:
            print("No behavioral ratio data available for plotting")
            return
            
        print("Creating behavioral ratio plots...")
        
        # Create output directory
        if save_plots:
            Path(output_dir).mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Behavioral Context Analysis', fontsize=16)
        
        # Plot 1: Food Detection Ratios by Treatment
        ax = axes[0, 0]
        detection_data = self.behavioral_ratios[self.behavioral_ratios['has_detection_data']]
        if len(detection_data) > 0:
            treatment_groups = detection_data.groupby('treatment')['food_detection_ratio']
            treatments = list(treatment_groups.groups.keys())
            ratios = [treatment_groups.get_group(t).values for t in treatments]
            
            bp = ax.boxplot(ratios, labels=treatments, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_title('Food Detection Ratios by Treatment')
            ax.set_ylabel('Nearby Food / Off Food Velocity')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Food Encounter Ratios by Treatment
        ax = axes[0, 1]
        encounter_data = self.behavioral_ratios[self.behavioral_ratios['has_encounter_data']]
        if len(encounter_data) > 0:
            treatment_groups = encounter_data.groupby('treatment')['food_encounter_ratio']
            treatments = list(treatment_groups.groups.keys())
            ratios = [treatment_groups.get_group(t).values for t in treatments]
            
            bp = ax.boxplot(ratios, labels=treatments, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')
                patch.set_alpha(0.7)
            
            ax.set_title('Food Encounter Ratios by Treatment')
            ax.set_ylabel('On Food / Nearby Food Velocity')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Food Detection Ratios by Sex
        ax = axes[0, 2]
        if len(detection_data) > 0:
            sex_groups = detection_data.groupby('sex')['food_detection_ratio']
            sexes = list(sex_groups.groups.keys())
            ratios = [sex_groups.get_group(s).values for s in sexes]
            
            bp = ax.boxplot(ratios, labels=sexes, patch_artist=True)
            colors = ['lightpink', 'lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title('Food Detection Ratios by Sex')
            ax.set_ylabel('Nearby Food / Off Food Velocity')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Food Encounter Ratios by Sex
        ax = axes[1, 0]
        if len(encounter_data) > 0:
            sex_groups = encounter_data.groupby('sex')['food_encounter_ratio']
            sexes = list(sex_groups.groups.keys())
            ratios = [sex_groups.get_group(s).values for s in sexes]
            
            bp = ax.boxplot(ratios, labels=sexes, patch_artist=True)
            colors = ['lightpink', 'lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title('Food Encounter Ratios by Sex')
            ax.set_ylabel('On Food / Nearby Food Velocity')
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Correlation between Detection and Encounter Ratios
        ax = axes[1, 1]
        correlation_data = self.behavioral_ratios[
            self.behavioral_ratios['has_detection_data'] & 
            self.behavioral_ratios['has_encounter_data']
        ]
        if len(correlation_data) > 5:
            x = correlation_data['food_detection_ratio']
            y = correlation_data['food_encounter_ratio']
            
            ax.scatter(x, y, alpha=0.6)
            
            # Calculate correlation
            corr_coef = np.corrcoef(x, y)[0, 1]
            ax.set_title(f'Detection vs Encounter Ratios\n(r = {corr_coef:.3f})')
            ax.set_xlabel('Food Detection Ratio')
            ax.set_ylabel('Food Encounter Ratio')
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Raw velocity distributions
        ax = axes[1, 2]
        if self.contextual_velocities:
            for i, (context, data) in enumerate(self.contextual_velocities.items()):
                if len(data) > 0:
                    velocities = data['velocity'].dropna()
                    if len(velocities) > 0:
                        ax.hist(velocities, bins=30, alpha=0.5, label=context, density=True)
            
            ax.set_title('Velocity Distributions by Context')
            ax.set_xlabel('Velocity (mm/s)' if self.pixels_per_mm > 0 else 'Velocity (pixels/time)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/behavioral_ratios.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{output_dir}/behavioral_ratios.pdf", bbox_inches='tight')
        
        plt.show()
    
    def plot_velocity_profiles(self, save_plots: bool = True, output_dir: str = "plots", 
                             show_individual_traces: bool = False) -> None:
        """Create velocity profile plots - maintains existing functionality"""
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
                    n_animals = stats_data[valid_idx]['n_animals'].values
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
        
        if self.behavioral_ratios is not None:
            self.behavioral_ratios.to_csv(f"{output_dir}/behavioral_ratios.csv", index=False)
            print(f"Saved behavioral ratios to {output_dir}/behavioral_ratios.csv")
        
        if self.contextual_velocities is not None:
            for context, data in self.contextual_velocities.items():
                if len(data) > 0:
                    data.to_csv(f"{output_dir}/contextual_velocities_{context}.csv", index=False)
                    print(f"Saved {context} velocities to {output_dir}/contextual_velocities_{context}.csv")
        
        if self.statistical_results is not None:
            # Save statistical results summary
            stat_summary = []
            for context, results in self.statistical_results.items():
                row = {'context': context}
                
                if 'mixed_effects' in results and 'error' not in results['mixed_effects']:
                    me = results['mixed_effects']
                    row.update({
                        'mixed_effects_aic': me.get('aic'),
                        'mixed_effects_bic': me.get('bic'),
                        'mixed_effects_converged': me.get('converged')
                    })
                
                if 'anova' in results and 'error' not in results['anova']:
                    anova = results['anova']
                    row.update({
                        'anova_f_stat': anova.get('f_statistic'),
                        'anova_p_value': anova.get('p_value')
                    })
                
                if 'normality_tests' in results:
                    norm = results['normality_tests']
                    if 'transformed' in norm and 'error' not in norm['transformed']:
                        row.update({
                            'normality_test': norm['transformed'].get('test'),
                            'normality_p_value': norm['transformed'].get('p_value'),
                            'data_normal': norm['transformed'].get('normal')
                        })
                
                stat_summary.append(row)
            
            if stat_summary:
                pd.DataFrame(stat_summary).to_csv(f"{output_dir}/statistical_summary.csv", index=False)
                print(f"Saved statistical summary to {output_dir}/statistical_summary.csv")
            
        # Save food leaving analysis if available
        if hasattr(self, 'food_leaving_data') and self.food_leaving_data is not None:
            self.food_leaving_data.to_csv(f"{output_dir}/food_leaving_analysis.csv", index=False)
            print(f"Saved food leaving analysis to {output_dir}/food_leaving_analysis.csv")

# GUI Application
class AnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Food Encounter Velocity Analysis")
        self.root.geometry("900x700")
        
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
        
        # Statistical analysis options
        ttk.Label(params_frame, text="Data transformation:").grid(row=0, column=2, sticky="w", padx=(20,5))
        self.transform_var = tk.StringVar(value="auto")
        transform_combo = ttk.Combobox(params_frame, textvariable=self.transform_var, 
                                      values=["auto", "log", "sqrt", "boxcox", "none"], width=10)
        transform_combo.grid(row=0, column=3, sticky="w", padx=5)
        
        # Analysis options
        self.do_behavioral_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Behavioral context analysis", 
                       variable=self.do_behavioral_var).grid(row=1, column=2, columnspan=2, sticky="w", padx=(20,0))
        
        self.do_stats_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Statistical analysis", 
                       variable=self.do_stats_var).grid(row=2, column=2, columnspan=2, sticky="w", padx=(20,0))
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Show Traditional Plots", command=self.show_plots).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Show Behavioral Plots", command=self.show_behavioral_plots).pack(side="left", padx=5)
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
            transformation = self.transform_var.get()
            do_behavioral = self.do_behavioral_var.get()
            do_stats = self.do_stats_var.get()
            
            # Update analyzer with new pixels_per_mm
            self.analyzer.pixels_per_mm = pixels_per_mm
            
            # Load data
            if not self.analyzer.load_data(csv_file):
                return
            
            # Run basic analysis
            self.analyzer.calculate_velocities(smooth_window)
            self.analyzer.align_to_food_encounters(time_before, time_after)
            self.analyzer.calculate_summary_statistics()
            
            # Run behavioral analysis if requested
            if do_behavioral:
                self.analyzer.extract_contextual_velocities()
                self.analyzer.calculate_behavioral_ratios()
                
                if do_stats and HAS_STATSMODELS:
                    self.analyzer.perform_statistical_analysis(transformation)
                elif do_stats:
                    print("Statistical analysis requested but statsmodels not available")
            
            # Print reports
            self.analyzer.print_summary_report()
            if do_behavioral:
                self.analyzer.print_behavioral_analysis_report()
            
            messagebox.showinfo("Success", "Analysis completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            print(f"ERROR: {e}")
    
    def show_plots(self):
        if self.analyzer.encounter_aligned_data is None:
            messagebox.showwarning("Warning", "No analysis data available. Run analysis first.")
            return
        
        try:
            self.analyzer.plot_velocity_profiles(save_plots=True)
        except Exception as e:
            messagebox.showerror("Error", f"Plot generation failed: {e}")
    
    def show_behavioral_plots(self):
        if self.analyzer.behavioral_ratios is None:
            messagebox.showwarning("Warning", "No behavioral ratio data available. Run behavioral analysis first.")
            return
        
        try:
            self.analyzer.plot_behavioral_ratios(save_plots=True)
        except Exception as e:
            messagebox.showerror("Error", f"Behavioral plot generation failed: {e}")
    
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
    
    parser = argparse.ArgumentParser(description="Food Encounter Velocity Analysis with Behavioral Context Analysis")
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
    parser.add_argument("--no-behavioral", action="store_true", help="Skip behavioral context analysis")
    parser.add_argument("--no-stats", action="store_true", help="Skip statistical analysis")
    parser.add_argument("--transformation", choices=["auto", "log", "sqrt", "boxcox", "none"], 
                        default="auto", help="Data transformation method")
    
    args = parser.parse_args()
    
    if args.gui or not args.csv:
        # Launch GUI
        app = AnalysisApp()
        app.run()
    else:
        # Command line analysis
        analyzer = FoodEncounterAnalyzer(pixels_per_mm=args.pixels_per_mm)
        
        if analyzer.load_data(args.csv):
            # Basic analysis
            analyzer.calculate_velocities(args.smooth)
            analyzer.align_to_food_encounters(args.time_before, args.time_after)
            analyzer.calculate_summary_statistics()
            
            # Behavioral analysis
            if not args.no_behavioral:
                analyzer.extract_contextual_velocities()
                analyzer.calculate_behavioral_ratios()
                
                # Statistical analysis
                if not args.no_stats and HAS_STATSMODELS:
                    analyzer.perform_statistical_analysis(args.transformation)
                elif not args.no_stats:
                    print("Statistical analysis requested but statsmodels not available")
            
            # Add food leaving analysis
            leaving_results = analyzer.analyze_food_leaving_behavior()
            if leaving_results is not None:
                analyzer.food_leaving_data = leaving_results
            
            # Print reports
            analyzer.print_summary_report()
            if not args.no_behavioral:
                analyzer.print_behavioral_analysis_report()
            
            # Generate plots
            if not args.no_plots:
                analyzer.plot_velocity_profiles(show_individual_traces=args.show_individual)
                if not args.no_behavioral:
                    analyzer.plot_behavioral_ratios()
            
            analyzer.save_results()

if __name__ == "__main__":
    main()