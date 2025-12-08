#!/usr/bin/env python3
"""
Food Encounter Label Validator - COMPLETE VERSION

Validates existing food_encounter labels by checking if there's actually
a slowdown at the labeled time.

For each labeled food encounter:
- Measures mean speed before label (-10 to -5 sec)
- Measures mean speed after label (+5 to +10 sec)
- Calculates deceleration magnitude
- Flags tracks with insufficient or unusual slowdown

Usage:
    python food_encounter_qc_validator.py composite.csv
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from scipy import stats

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Time windows for measuring speed (seconds)
BEFORE_WINDOW_START = -10  # Start of "before" window
BEFORE_WINDOW_END = -5     # End of "before" window
AFTER_WINDOW_START = 5     # Start of "after" window
AFTER_WINDOW_END = 10      # End of "after" window

# Minimum points needed in each window
MIN_POINTS_PER_WINDOW = 3

# ==============================================================================
# CORE VALIDATION FUNCTIONS
# ==============================================================================

def calculate_kinematics(track_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate speed for a single track."""
    track_df = track_df.copy().sort_values('time').reset_index(drop=True)
    
    # Calculate dx, dy, dt
    track_df['dx'] = track_df['x'].diff()
    track_df['dy'] = track_df['y'].diff()
    track_df['dt'] = track_df['time'].diff()
    
    # Calculate instantaneous speed
    track_df['speed'] = np.sqrt(track_df['dx']**2 + track_df['dy']**2) / track_df['dt']
    
    return track_df


def validate_labeled_encounter(track_df: pd.DataFrame) -> Dict:
    """
    Validate a food encounter label by checking for slowdown.
    
    Measures:
    - Speed before label (mean from -10 to -5 sec)
    - Speed after label (mean from +5 to +10 sec)
    - Deceleration magnitude
    - Statistical significance of change
    
    Returns validation metrics for this track.
    """
    
    # Find labeled food encounter time
    if 'food_encounter' not in track_df.columns:
        return {
            'status': 'NO_FOOD_COLUMN',
            'label_time': np.nan,
            'speed_before': np.nan,
            'speed_after': np.nan,
            'speed_change': np.nan,
            'percent_change': np.nan,
            'decel_rate': np.nan,
            'n_before': 0,
            'n_after': 0,
            'std_before': np.nan,
            'std_after': np.nan,
            'track_start': track_df['time'].min() if len(track_df) > 0 else np.nan,
            'track_end': track_df['time'].max() if len(track_df) > 0 else np.nan
        }
    
    food_events = track_df[track_df['food_encounter'] == 'food']
    
    if food_events.empty:
        return {
            'status': 'NOT_LABELED',
            'label_time': np.nan,
            'speed_before': np.nan,
            'speed_after': np.nan,
            'speed_change': np.nan,
            'percent_change': np.nan,
            'decel_rate': np.nan,
            'n_before': 0,
            'n_after': 0,
            'std_before': np.nan,
            'std_after': np.nan,
            'track_start': track_df['time'].min(),
            'track_end': track_df['time'].max()
        }
    
    # Get first food encounter time
    label_time = food_events.iloc[0]['time']
    
    # Define time windows RELATIVE to label
    before_start = label_time + BEFORE_WINDOW_START
    before_end = label_time + BEFORE_WINDOW_END
    after_start = label_time + AFTER_WINDOW_START
    after_end = label_time + AFTER_WINDOW_END
    
    # Extract data from each window
    before_data = track_df[
        (track_df['time'] >= before_start) & 
        (track_df['time'] <= before_end) &
        (track_df['speed'].notna())  # Only non-NaN speeds
    ]
    
    after_data = track_df[
        (track_df['time'] >= after_start) & 
        (track_df['time'] <= after_end) &
        (track_df['speed'].notna())  # Only non-NaN speeds
    ]
    
    # Check if we have enough data
    if len(before_data) < MIN_POINTS_PER_WINDOW or len(after_data) < MIN_POINTS_PER_WINDOW:
        return {
            'status': 'INSUFFICIENT_DATA',
            'label_time': label_time,
            'speed_before': np.nan,
            'speed_after': np.nan,
            'speed_change': np.nan,
            'percent_change': np.nan,
            'decel_rate': np.nan,
            'n_before': len(before_data),
            'n_after': len(after_data),
            'std_before': np.nan,
            'std_after': np.nan,
            'track_start': track_df['time'].min(),
            'track_end': track_df['time'].max(),
            'before_window': f"{before_start:.1f} to {before_end:.1f}",
            'after_window': f"{after_start:.1f} to {after_end:.1f}"
        }
    
    # Calculate speed statistics
    speed_before_mean = before_data['speed'].mean()
    speed_before_std = before_data['speed'].std()
    
    speed_after_mean = after_data['speed'].mean()
    speed_after_std = after_data['speed'].std()
    
    # Calculate deceleration metrics
    speed_change = speed_before_mean - speed_after_mean  # Positive = slowing down
    
    if speed_before_mean > 0:
        percent_change = (speed_change / speed_before_mean) * 100
    else:
        percent_change = 0
    
    # Deceleration rate (speed change per second)
    time_span = abs(BEFORE_WINDOW_END) + AFTER_WINDOW_START  # Time from end of before to start of after
    if time_span > 0:
        decel_rate = speed_change / time_span
    else:
        decel_rate = np.nan
    
    # Determine status based on whether there was a slowdown
    if speed_change > 0:
        status = 'VALID_SLOWDOWN'
    elif speed_change < 0:
        status = 'SPEED_INCREASED'  # Animal sped up!
    else:
        status = 'NO_CHANGE'
    
    return {
        'status': status,
        'label_time': label_time,
        'speed_before': speed_before_mean,
        'speed_after': speed_after_mean,
        'speed_change': speed_change,
        'percent_change': percent_change,
        'decel_rate': decel_rate,
        'n_before': len(before_data),
        'n_after': len(after_data),
        'std_before': speed_before_std,
        'std_after': speed_after_std,
        'track_start': track_df['time'].min(),
        'track_end': track_df['time'].max()
    }


def validate_single_track(track_df: pd.DataFrame) -> Dict:
    """Validate food encounter labeling for a single track."""
    
    # Calculate kinematics
    track_df = calculate_kinematics(track_df)
    
    # Validate the labeled encounter
    validation = validate_labeled_encounter(track_df)
    
    # Add track metadata
    validation['track_duration'] = track_df['time'].max() - track_df['time'].min()
    validation['n_timepoints'] = len(track_df)
    
    return validation


# ==============================================================================
# BATCH VALIDATION
# ==============================================================================

def validate_composite(composite_df: pd.DataFrame) -> pd.DataFrame:
    """Validate all tracks in a composite dataframe."""
    print("\n" + "="*70)
    print("STARTING VALIDATION - LABELED ENCOUNTER VERIFICATION")
    print("="*70)
    
    # Check required columns
    required_cols = ['source_file', 'assay_num', 'track_num', 'time', 'x', 'y']
    missing_cols = [col for col in required_cols if col not in composite_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get metadata columns if they exist
    metadata_cols = ['sex', 'strain_genotype', 'treatment', 'pc_number']
    available_metadata = [col for col in metadata_cols if col in composite_df.columns]
    
    results = []
    
    # Group by assay and track
    grouped = composite_df.groupby(['source_file', 'assay_num', 'track_num'])
    total_tracks = len(grouped)
    
    print(f"\nValidating {total_tracks} tracks...")
    print(f"Method: Verify slowdown at labeled food encounters")
    print(f"Before window: {BEFORE_WINDOW_START} to {BEFORE_WINDOW_END} sec (relative to label)")
    print(f"After window: {AFTER_WINDOW_START} to {AFTER_WINDOW_END} sec (relative to label)")
    print()
    
    for i, ((source_file, assay_num, track_num), track_df) in enumerate(grouped, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{total_tracks} tracks...")
        
        # Validate this track
        validation = validate_single_track(track_df)
        
        # Compile results
        result = {
            'source_file': source_file,
            'assay_num': assay_num,
            'track_num': track_num,
        }
        
        # Add metadata if available
        for col in available_metadata:
            result[col] = track_df[col].iloc[0]
        
        # Add validation results
        result.update(validation)
        
        results.append(result)
    
    print(f"  Completed: {total_tracks}/{total_tracks} tracks\n")
    
    return pd.DataFrame(results)


# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================

def flag_outliers(validation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag tracks that are statistical outliers.
    Uses z-score to identify unusual speed changes.
    """
    
    # Only analyze tracks with valid slowdown data
    valid_data = validation_df[validation_df['status'] == 'VALID_SLOWDOWN'].copy()
    
    if len(valid_data) < 10:
        print("Warning: Too few valid tracks for outlier detection")
        validation_df['is_outlier'] = False
        validation_df['z_score'] = np.nan
        return validation_df
    
    # Calculate z-scores for speed change
    mean_change = valid_data['speed_change'].mean()
    std_change = valid_data['speed_change'].std()
    
    if std_change == 0:
        validation_df['is_outlier'] = False
        validation_df['z_score'] = 0
        return validation_df
    
    # Calculate z-score for ALL tracks
    validation_df['z_score'] = (validation_df['speed_change'] - mean_change) / std_change
    
    # Flag outliers (|z| > 2 = more than 2 std devs from mean)
    validation_df['is_outlier'] = validation_df['z_score'].abs() > 2
    
    return validation_df


def generate_summary_stats(validation_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics by group."""
    group_cols = ['source_file', 'assay_num']
    
    for col in ['sex', 'strain_genotype', 'treatment']:
        if col in validation_df.columns:
            group_cols.append(col)
    
    summary = validation_df.groupby(group_cols).agg({
        'track_num': 'count',
    }).reset_index()
    
    summary.columns = list(group_cols) + ['total_tracks']
    
    # Count by status
    for status_type in ['VALID_SLOWDOWN', 'SPEED_INCREASED', 'NO_CHANGE', 
                       'INSUFFICIENT_DATA', 'NOT_LABELED', 'NO_FOOD_COLUMN']:
        counts = validation_df[validation_df['status'] == status_type].groupby(group_cols).size()
        summary[f'{status_type.lower()}_count'] = summary[group_cols].apply(
            lambda row: counts.get(tuple(row), 0), axis=1
        )
    
    # Calculate mean speed change for valid tracks
    valid_tracks = validation_df[validation_df['status'] == 'VALID_SLOWDOWN']
    if len(valid_tracks) > 0:
        mean_speed_change = valid_tracks.groupby(group_cols)['speed_change'].mean().round(3)
        mean_percent_change = valid_tracks.groupby(group_cols)['percent_change'].mean().round(1)
        
        summary['mean_speed_change'] = summary[group_cols].apply(
            lambda row: mean_speed_change.get(tuple(row), np.nan), axis=1
        )
        summary['mean_percent_change'] = summary[group_cols].apply(
            lambda row: mean_percent_change.get(tuple(row), np.nan), axis=1
        )
    else:
        summary['mean_speed_change'] = np.nan
        summary['mean_percent_change'] = np.nan
    
    # Count outliers
    if 'is_outlier' in validation_df.columns:
        outlier_counts = validation_df[validation_df['is_outlier']].groupby(group_cols).size()
        summary['outlier_count'] = summary[group_cols].apply(
            lambda row: outlier_counts.get(tuple(row), 0), axis=1
        )
    
    return summary


def print_validation_report(validation_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Print a formatted validation report to console."""
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)
    
    total_tracks = len(validation_df)
    
    # Count by status
    status_counts = validation_df['status'].value_counts()
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total tracks: {total_tracks}")
    for status, count in status_counts.items():
        pct = count / total_tracks * 100
        print(f"  {status}: {count} ({pct:.1f}%)")
    
    # Statistics for valid slowdown tracks
    valid_tracks = validation_df[validation_df['status'] == 'VALID_SLOWDOWN']
    if len(valid_tracks) > 0:
        print(f"\nVALID SLOWDOWN TRACKS ({len(valid_tracks)}):")
        print(f"  Mean speed change: {valid_tracks['speed_change'].mean():.3f} ± {valid_tracks['speed_change'].std():.3f}")
        print(f"  Mean percent change: {valid_tracks['percent_change'].mean():.1f}%")
        print(f"  Median speed change: {valid_tracks['speed_change'].median():.3f}")
        print(f"  Range: [{valid_tracks['speed_change'].min():.3f}, {valid_tracks['speed_change'].max():.3f}]")
    
    # Outlier summary
    if 'is_outlier' in validation_df.columns:
        outliers = validation_df[validation_df['is_outlier']]
        print(f"\nOUTLIERS (|z-score| > 2):")
        print(f"  Total outliers: {len(outliers)} ({len(outliers)/total_tracks*100:.1f}%)")
    
    # Problem tracks
    problem_tracks = validation_df[validation_df['status'].isin(['SPEED_INCREASED', 'NO_CHANGE', 'INSUFFICIENT_DATA'])]
    if len(problem_tracks) > 0:
        print(f"\nPROBLEM TRACKS:")
        for status in ['SPEED_INCREASED', 'NO_CHANGE', 'INSUFFICIENT_DATA']:
            count = (problem_tracks['status'] == status).sum()
            if count > 0:
                print(f"  {status}: {count}")
    
    # Group-level statistics
    print(f"\n{'='*70}")
    print("SUMMARY BY GROUP:")
    print(f"{'='*70}\n")
    print(summary_df.to_string(index=False))
    
    # Flagged tracks for investigation
    flagged = validation_df[
        (validation_df['status'].isin(['SPEED_INCREASED', 'NO_CHANGE', 'INSUFFICIENT_DATA'])) |
        (validation_df.get('is_outlier', False))
    ]
    
    if len(flagged) > 0:
        print(f"\n{'='*70}")
        print(f"FLAGGED TRACKS FOR INVESTIGATION: {len(flagged)}")
        print(f"{'='*70}\n")
        
        cols_to_show = ['source_file', 'assay_num', 'track_num', 'status', 
                       'speed_change', 'percent_change', 'n_before', 'n_after']
        if 'sex' in flagged.columns:
            cols_to_show.insert(3, 'sex')
        if 'strain_genotype' in flagged.columns:
            cols_to_show.insert(4, 'strain_genotype')
        if 'is_outlier' in flagged.columns:
            cols_to_show.append('z_score')
        
        flagged_sorted = flagged.sort_values('speed_change', ascending=True)
        print(flagged_sorted[cols_to_show].head(20).to_string(index=False))
        
        if len(flagged) > 20:
            print(f"\n... and {len(flagged) - 20} more (see qc_report_flagged_tracks.csv)")
        
        print(f"\n  Investigate these tracks for:")
        print(f"    - Camera shifts / motion artifacts")
        print(f"    - Tracking errors")
        print(f"    - Animals already on food (no transition)")


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_validation_plots(validation_df: pd.DataFrame, output_path: Path):
    """Create visualization plots for validation results."""
    print(f"\nGenerating validation plots...")
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
    
    # Plot 1: Status distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    status_counts = validation_df['status'].value_counts()
    colors = {
        'VALID_SLOWDOWN': 'green',
        'SPEED_INCREASED': 'red',
        'NO_CHANGE': 'orange',
        'INSUFFICIENT_DATA': 'gray',
        'NOT_LABELED': 'lightgray',
        'NO_FOOD_COLUMN': 'lightgray'
    }
    bar_colors = [colors.get(status, 'blue') for status in status_counts.index]
    ax.bar(range(len(status_counts)), status_counts.values, color=bar_colors, alpha=0.7)
    ax.set_xticks(range(len(status_counts)))
    ax.set_xticklabels(status_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Tracks', fontsize=12)
    ax.set_title('Label Validation Results', fontsize=14, fontweight='bold')
    for i, (status, count) in enumerate(status_counts.items()):
        pct = count / len(validation_df) * 100
        ax.text(i, count + 5, f'{count}\n({pct:.1f}%)', ha='center', fontsize=10)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # Plot 2: Speed change distribution
    valid_tracks = validation_df[validation_df['status'] == 'VALID_SLOWDOWN']
    if len(valid_tracks) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Speed change histogram
        ax1.hist(valid_tracks['speed_change'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=valid_tracks['speed_change'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {valid_tracks["speed_change"].mean():.2f}')
        ax1.set_xlabel('Speed Change (units/s)', fontsize=12)
        ax1.set_ylabel('Number of Tracks', fontsize=12)
        ax1.set_title('Speed Change at Food Encounter', fontsize=13, fontweight='bold')
        ax1.legend()
        
        # Percent change histogram
        ax2.hist(valid_tracks['percent_change'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(x=valid_tracks['percent_change'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {valid_tracks["percent_change"].mean():.1f}%')
        ax2.set_xlabel('Percent Speed Change (%)', fontsize=12)
        ax2.set_ylabel('Number of Tracks', fontsize=12)
        ax2.set_title('Percent Speed Change at Food Encounter', fontsize=13, fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    # Plot 3: Speed before vs after
    if len(valid_tracks) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(valid_tracks['speed_before'], valid_tracks['speed_after'], 
                  alpha=0.5, s=50, edgecolors='black', linewidths=0.5)
        
        # Add diagonal line (no change)
        max_speed = max(valid_tracks['speed_before'].max(), valid_tracks['speed_after'].max())
        ax.plot([0, max_speed], [0, max_speed], 'r--', linewidth=2, label='No change')
        
        ax.set_xlabel('Speed Before (units/s)', fontsize=12)
        ax.set_ylabel('Speed After (units/s)', fontsize=12)
        ax.set_title('Speed Before vs After Food Encounter', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    # Plot 4: Outlier detection
    if 'z_score' in validation_df.columns and len(valid_tracks) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        outliers = valid_tracks[valid_tracks['is_outlier']]
        normal = valid_tracks[~valid_tracks['is_outlier']]
        
        ax.scatter(range(len(normal)), normal['speed_change'], 
                  alpha=0.5, s=50, label='Normal', color='blue')
        if len(outliers) > 0:
            outlier_indices = list(range(len(normal), len(normal) + len(outliers)))
            ax.scatter(outlier_indices, outliers['speed_change'],
                      alpha=0.7, s=80, label='Outlier', color='red', marker='x')
        
        mean_change = valid_tracks['speed_change'].mean()
        std_change = valid_tracks['speed_change'].std()
        ax.axhline(y=mean_change, color='green', linestyle='--', linewidth=2, label='Mean')
        ax.axhline(y=mean_change + 2*std_change, color='orange', linestyle=':', linewidth=1.5, label='±2σ')
        ax.axhline(y=mean_change - 2*std_change, color='orange', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel('Track Index', fontsize=12)
        ax.set_ylabel('Speed Change (units/s)', fontsize=12)
        ax.set_title('Outlier Detection (Z-score)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    pdf.close()
    print(f"  Saved plots to: {output_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description='Validate food encounter labels by checking for slowdown'
    )
    parser.add_argument('composite_file', nargs='?', help='Path to composite CSV file')
    parser.add_argument('--output-dir', '-o', default='qc_output', help='Output directory')
    parser.add_argument('--before-start', type=float, default=-10,
                       help='Start of before window (default: -10 sec)')
    parser.add_argument('--before-end', type=float, default=-5,
                       help='End of before window (default: -5 sec)')
    parser.add_argument('--after-start', type=float, default=5,
                       help='Start of after window (default: 5 sec)')
    parser.add_argument('--after-end', type=float, default=10,
                       help='End of after window (default: 10 sec)')
    
    args = parser.parse_args()
    
    # Update global parameters
    global BEFORE_WINDOW_START, BEFORE_WINDOW_END, AFTER_WINDOW_START, AFTER_WINDOW_END
    BEFORE_WINDOW_START = args.before_start
    BEFORE_WINDOW_END = args.before_end
    AFTER_WINDOW_START = args.after_start
    AFTER_WINDOW_END = args.after_end
    
    # Get input file
    if args.composite_file:
        composite_path = Path(args.composite_file)
    else:
        print("No file specified. Please enter the path to your composite CSV file:")
        composite_path = Path(input("> ").strip())
    
    if not composite_path.exists():
        print(f"ERROR: File not found: {composite_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load composite data
    print(f"\nLoading data from: {composite_path}")
    try:
        composite_df = pd.read_csv(composite_path)
        print(f"  Loaded {len(composite_df)} rows")
        print(f"  Columns: {', '.join(composite_df.columns)}")
    except Exception as e:
        print(f"ERROR loading file: {e}")
        sys.exit(1)
    
    # Validate all tracks
    validation_df = validate_composite(composite_df)
    
    # Flag outliers
    validation_df = flag_outliers(validation_df)
    
    # Generate summary statistics
    summary_df = generate_summary_stats(validation_df)
    
    # Print report to console
    print_validation_report(validation_df, summary_df)
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    summary_path = output_dir / 'qc_report_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary: {summary_path}")
    
    all_tracks_path = output_dir / 'qc_report_all_tracks.csv'
    validation_df.to_csv(all_tracks_path, index=False)
    print(f"  All tracks: {all_tracks_path}")
    
    flagged_df = validation_df[
        (validation_df['status'].isin(['SPEED_INCREASED', 'NO_CHANGE', 'INSUFFICIENT_DATA'])) |
        (validation_df.get('is_outlier', False))
    ]
    if len(flagged_df) > 0:
        flagged_path = output_dir / 'qc_report_flagged_tracks.csv'
        flagged_df.to_csv(flagged_path, index=False)
        print(f"  Flagged tracks: {flagged_path}")
    
    # Create visualizations
    plots_path = output_dir / 'qc_report_plots.pdf'
    create_validation_plots(validation_df, plots_path)
    
    print(f"\n{'='*70}")
    print("QC VALIDATION COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Method: Label verification by speed comparison")
    print(f"  - Measures speed before label ({BEFORE_WINDOW_START} to {BEFORE_WINDOW_END} sec)")
    print(f"  - Measures speed after label ({AFTER_WINDOW_START} to {AFTER_WINDOW_END} sec)")
    print(f"  - Flags tracks with no slowdown or unusual patterns")
    print(f"  - Identifies statistical outliers (|z-score| > 2)")
    print(f"\nResults in: {output_dir}/")
    print()


if __name__ == '__main__':
    main()