# Richards Curve Analysis for Worm Locomotion

This toolkit fits **Richards (generalized logistic) curves** to individual animal speed traces around food encounters, enabling rigorous comparison of locomotor dynamics across genotypes.

## Overview

The Richards function captures sigmoid transitions from initial to final speed:

```
y(t) = y_initial + (y_final - y_initial) / (1 + exp(-B(t - M)))^(1/ν)
```

**Parameters:**
- `y_initial`: Starting speed (first asymptote)
- `y_final`: Ending speed (second asymptote)
- `A = y_final - y_initial`: Magnitude of change
- `B`: Transition rate (steepness)
- `M`: Midpoint time (inflection point, relative to food encounter)
- `ν` (nu): Asymmetry parameter
  - ν = 1: Symmetric sigmoid (standard logistic)
  - ν < 1: Slow saturation (left-skewed)
  - ν > 1: Rapid onset (right-skewed)

## Installation

```bash
# Required packages
pip install pandas numpy scipy matplotlib --break-system-packages
```

## Workflow

### 1. Prepare Data

Start with a composite CSV from `merge_files2.py` containing:
- `assay_num`, `track_num`: Animal identifiers
- `time`: Time in seconds
- `x`, `y`: Position coordinates
- `food_encounter`: Marks when nose enters food (value = 'food')
- Metadata: `strain_genotype`, `sex`, `treatment`, etc.

### 2. Run Analysis

```bash
python richards_speed_analysis.py composite.csv --output results/
```

**Options:**
- `--window-before 120`: Seconds before food encounter (default: 120)
- `--window-after 120`: Seconds after food encounter (default: 120)
- `--smoothing 3`: Smoothing window for speed calculation (default: 3)
- `--min-points 20`: Minimum data points required per animal (default: 20)
- `--plot-animal "1,5"`: Plot specific animal (assay 1, track 5)

**Output files:**
- `richards_fit_parameters.csv`: One row per animal with all fit parameters
- `windowed_speed_data.csv`: Speed traces centered on food encounter
- `fit_type_distribution.png`: Summary of fit types
- `A_by_genotype.png`, `B_by_genotype.png`, etc.: Parameter distributions

### 3. Browse Individual Fits

```bash
python view_richards_fits.py results/
```

**Features:**
- Filter by genotype and fit type
- Navigate through animals with arrow buttons
- View fit quality (R², RMSE) and parameters
- Export individual or all plots

## Understanding the Output

### Fit Types

The analysis automatically classifies each animal:

1. **`decrease`**: Significant speed decrease (A < -3×SE)
   - Normal slowing behavior upon food encounter
   
2. **`increase`**: Significant speed increase (A > 3×SE)
   - Rare; may indicate unusual behavior
   
3. **`no_transition`**: No significant change (|A| < 3×SE)
   - Animal maintained relatively constant speed
   - B, M, ν parameters are poorly constrained (high uncertainty)
   
4. **`flat`**: Completely flat line detected
   - Speed variance extremely low
   - Only y_initial is meaningful
   - B, M, ν are NaN (undefined)
   
5. **`failed`**: Optimization failed
   - Insufficient data or numerical issues

### Key Metrics

**Fit Quality:**
- `r_squared`: Coefficient of determination (0-1, higher = better)
- `rmse`: Root mean squared error (lower = better)
- `aic`, `bic`: Information criteria for model comparison

**Primary Parameters** (compare across genotypes):
- `y_initial`: Speed before food encounter
- `y_final`: Speed after food encounter
- `A`: Change in speed (negative = slowing)

**Transition Dynamics** (only for animals with significant transitions):
- `B`: Rate of transition (higher = steeper)
- `M`: Timing of transition relative to food encounter
- `nu`: Asymmetry of transition

### Parameter Uncertainties

Each parameter has a standard error (`_se` suffix):
- Small SE → well-determined parameter
- Large SE (>50% of parameter value) → poorly constrained

**Important:** For `no_transition` and `flat` fits, B, M, and ν have large uncertainties because the data doesn't constrain them. **Do not interpret these parameters for flat animals.**

## Statistical Analysis Strategy

### Tier 1: Proportion Analysis

Compare the fraction of animals showing transitions:

```python
import pandas as pd

df = pd.read_csv('results/richards_fit_parameters.csv')

# Calculate proportions
for genotype in df['strain_genotype'].unique():
    subset = df[df['strain_genotype'] == genotype]
    n_decrease = sum(subset['fit_type'] == 'decrease')
    total = len(subset)
    print(f"{genotype}: {n_decrease}/{total} ({100*n_decrease/total:.1f}%) show slowing")

# Chi-square test
from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(df['strain_genotype'], df['fit_type'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
```

### Tier 2: Magnitude Comparison (All Animals)

Compare `A` (speed change) including flat animals:

```python
import numpy as np
from scipy import stats

wt = df[df['strain_genotype'] == 'wt']['A'].dropna()
mutant = df[df['strain_genotype'] == 'mutant']['A'].dropna()

# Bootstrap comparison
def bootstrap_diff(a, b, n_boot=10000):
    diffs = []
    for _ in range(n_boot):
        a_sample = np.random.choice(a, len(a), replace=True)
        b_sample = np.random.choice(b, len(b), replace=True)
        diffs.append(np.mean(a_sample) - np.mean(b_sample))
    
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    return np.mean(diffs), ci_low, ci_high

diff, ci_low, ci_high = bootstrap_diff(wt, mutant)
print(f"Difference: {diff:.3f} mm/s, 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
```

### Tier 3: Dynamics Comparison (Transitioners Only)

For animals with clear transitions, compare B, M, ν:

```python
# Filter to only transitioners
df_trans = df[df['fit_type'].isin(['decrease', 'increase'])].copy()

# Compare transition rate B
wt_B = df_trans[df_trans['strain_genotype'] == 'wt']['B'].dropna()
mut_B = df_trans[df_trans['strain_genotype'] == 'mutant']['B'].dropna()

# Mann-Whitney U test (non-parametric)
u_stat, p_value = stats.mannwhitneyu(wt_B, mut_B)

# Or bootstrap as above
```

### Multivariate Analysis

Test if the full parameter vector differs:

```python
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

# Standardize parameters
params_to_test = ['A', 'B', 'M', 'nu']
df_valid = df_trans[params_to_test].dropna()

# PCA to visualize
pca = PCA(n_components=2)
coords = pca.fit_transform(df_valid)

# Plot
import matplotlib.pyplot as plt
for genotype in df_trans['strain_genotype'].unique():
    mask = df_trans['strain_genotype'] == genotype
    subset_coords = coords[mask[df_valid.index]]
    plt.scatter(subset_coords[:, 0], subset_coords[:, 1], label=genotype, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
plt.legend()
plt.show()
```

## Example Results Section

> **Methods**: Individual animal speed traces were extracted in a ±120s window around food encounter. We fit generalized logistic (Richards) curves to characterize the transition from pre-encounter to post-encounter speeds. Animals were classified as showing significant speed transitions (|A| > 3×SE) or maintaining constant speed. For transitioners, we analyzed transition magnitude (A), rate (B), timing (M), and asymmetry (ν).

> **Results**: WT animals predominantly showed speed decreases upon food encounter (18/20 animals, 90%), slowing from 0.65 ± 0.08 mm/s to 0.31 ± 0.05 mm/s (A = -0.34 ± 0.07 mm/s) with transition rate B = 0.15 ± 0.04 s⁻¹ and symmetric dynamics (ν = 1.1 ± 0.3). The transition occurred near the food encounter (M = 2.3 ± 5.1 s).
>
> In contrast, mutant animals showed heterogeneous phenotypes: 12/20 (60%) were non-transitioners, maintaining constant speeds of 0.52 ± 0.09 mm/s (χ² = 10.1, p = 0.001 vs WT). Among the 8 mutants that did transition, magnitude was reduced (A = -0.19 ± 0.06 mm/s, p = 0.008 vs WT) but rate was preserved (B = 0.13 ± 0.05 s⁻¹, p = 0.43). Notably, mutants showed increased asymmetry (ν = 2.2 ± 0.8, p = 0.02), indicating more rapid initial slowing followed by slower saturation compared to WT's symmetric transition.

## Troubleshooting

**Problem**: Many animals classified as "failed"
- **Solution**: Decrease `--min-points` threshold or increase time window

**Problem**: Strange parameter values (B > 10, ν > 10)
- **Solution**: Check if data actually shows transitions; may need stricter quality filters

**Problem**: High RMSE values
- **Solution**: Increase `--smoothing` to reduce noise in speed calculation

**Problem**: Want to exclude poor fits
```python
# Filter by R²
df_good = df[df['r_squared'] > 0.8]

# Or by relative parameter uncertainty
df_good = df[df['B_se'] / df['B'] < 0.5]  # SE < 50% of parameter value
```

## Advanced Usage

### Custom Analysis Scripts

```python
import pandas as pd
import numpy as np
from richards_speed_analysis import (
    fit_richards_to_animal, 
    richards_curve,
    extract_windowed_data,
    calculate_speed
)

# Load your composite
df = pd.read_csv('composite.csv')

# Calculate speeds
# ... (see richards_speed_analysis.py for full code)

# Fit with custom bounds
windowed_data = extract_windowed_data(df)

for (assay, track), data in windowed_data.items():
    time_rel = data['time_rel'].values
    speed = data['speed'].values
    
    # Custom fitting logic here
    result = fit_richards_to_animal(time_rel, speed, bounds_factor=3.0)
    
    # Your analysis...
```

### Batch Plotting

```python
from pathlib import Path
import matplotlib.pyplot as plt
from richards_speed_analysis import plot_individual_fit

results_dir = Path('results')

# Load results
df = pd.read_csv(results_dir / 'richards_fit_parameters.csv')
# ... load windowed_data and results dicts

# Plot all WT animals
wt_animals = df[df['strain_genotype'] == 'wt']

for _, row in wt_animals.iterrows():
    plot_individual_fit(
        assay_num=row['assay_num'],
        track_num=row['track_num'],
        windowed_data=windowed_data,
        results=results,
        save_path=results_dir / f"wt_animal_{row['assay_num']}_{row['track_num']}.png",
        show=False
    )
```

## Citation

If you use this analysis toolkit, please cite the Richards curve:

> Richards, F. J. (1959). A flexible growth function for empirical use. Journal of Experimental Botany, 10(2), 290-301.

## Files

- `richards_speed_analysis.py`: Main analysis pipeline
- `view_richards_fits.py`: Interactive visualization tool
- `README.md`: This file

## Contact

For questions or issues, please contact [your contact info].
