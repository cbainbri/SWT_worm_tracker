# Quick Start Guide

## Installation

```bash
pip install pandas numpy scipy matplotlib --break-system-packages
```

## Basic Usage

### 1. Run Analysis on Your Data

```bash
python richards_speed_analysis.py composite.csv --output results/
```

This will:
- Calculate speed from x,y coordinates  
- Extract ±120s windows around food encounters
- Fit Richards curves to each animal
- Save parameters to `results/richards_fit_parameters.csv`

### 2. Browse Individual Fits (INTERACTIVE!)

```bash
# Option 1: Load directory automatically
python view_richards_fits.py results/

# Option 2: Start GUI then load files interactively
python view_richards_fits.py
```

**Interactive Features:**
- ✅ **Load CSV files via File menu** - no command line needed!
- ✅ **Navigate with arrow keys** or click animals
- ✅ **Filter by genotype and fit type**
- ✅ **See raw speed trace with fitted curve overlaid**
- ✅ **View all parameters and fit quality for each animal**
- ✅ **Export individual or batch plots**
- ✅ **Keyboard shortcuts** (↑↓ or ←→ to navigate, E to export, Space for next)

The viewer shows:
- Raw speed data (gray dots)
- Richards curve fit (red line)
- Food encounter marker (green dashed line)
- Fit parameters and quality metrics
- Animal metadata (genotype, sex, treatment)

### 3. Test with Synthetic Data

```bash
python test_richards_analysis.py
python richards_speed_analysis.py synthetic_composite.csv --output synthetic_results/
python view_richards_fits.py synthetic_results/
```

## Output Files

**`richards_fit_parameters.csv`** - One row per animal with:
- Metadata: assay_num, track_num, strain_genotype, sex, treatment
- Fit type: decrease, increase, flat, no_transition, failed
- Parameters: y_initial, y_final, A, B, M, nu (with standard errors)
- Quality: r_squared, rmse, aic, bic

**`windowed_speed_data.csv`** - Speed traces for all animals:
- assay_num, track_num
- time_rel (seconds relative to food encounter)
- speed (mm/s)

## Key Parameters

**y_initial**: Speed before food encounter  
**y_final**: Speed after food encounter  
**A = y_final - y_initial**: Change in speed (negative = slowing)  
**B**: Transition rate (higher = steeper)  
**M**: Timing of transition (seconds from food encounter)  
**nu**: Asymmetry (1 = symmetric, <1 = slow saturation, >1 = rapid onset)

## Fit Types

- **decrease**: Significant slowing (A < -3×SE) - normal behavior
- **increase**: Significant speedup (A > 3×SE) - rare
- **no_transition**: No significant change (|A| < 3×SE)
- **flat**: Completely flat line (B, M, nu are undefined)
- **failed**: Optimization failed

**Important**: For `flat` and `no_transition` fits, only y_initial, y_final, and A are meaningful. B, M, and nu are poorly constrained.

## Statistical Analysis

See README_richards_analysis.md for detailed statistical comparison strategies:
- Tier 1: Compare proportions showing transitions (chi-square)
- Tier 2: Compare A across all animals (bootstrap)  
- Tier 3: Compare B, M, nu for transitioners only (bootstrap/Mann-Whitney)

## Example Analysis

```python
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('results/richards_fit_parameters.csv')

# Filter to successful fits
df_good = df[df['converged'] & (df['r_squared'] > 0.7)]

# Compare genotypes
for genotype in df_good['strain_genotype'].unique():
    subset = df_good[df_good['strain_genotype'] == genotype]
    print(f"\n{genotype}:")
    print(f"  N = {len(subset)}")
    print(f"  Transitioners: {sum(subset['fit_type'] == 'decrease')}/{len(subset)}")
    print(f"  Mean A = {subset['A'].mean():.3f} ± {subset['A'].std():.3f} mm/s")
    
    # Only for animals with transitions
    trans = subset[subset['fit_type'] == 'decrease']
    if len(trans) > 0:
        print(f"  Mean B = {trans['B'].mean():.3f} ± {trans['B'].std():.3f} s⁻¹")
        print(f"  Mean nu = {trans['nu'].mean():.3f} ± {trans['nu'].std():.3f}")
```

## Troubleshooting

**Many failed fits?** Try:
- `--window-before 180 --window-after 180` (larger window)
- `--min-points 10` (fewer required points)
- `--smoothing 5` (more smoothing)

**Want to plot specific animal?**
```bash
python richards_speed_analysis.py composite.csv --output results/ --plot-animal "1,5"
```

**Need help?** See full documentation in README_richards_analysis.md
