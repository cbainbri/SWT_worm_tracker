# Interactive Viewer Guide

## Overview

The Richards Fit Viewer is a GUI application for browsing individual animal speed traces with fitted curves overlaid. Perfect for quality control, exploring data, and creating publication figures.

## Starting the Viewer

```bash
# Option 1: Load results directory automatically
python view_richards_fits.py results/

# Option 2: Start GUI first, load files via menu
python view_richards_fits.py
```

## Interface Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ File  Help                                    Status: 30 animals... │
├───────────────┬─────────────────────────────────────────────────────┤
│  FILTERS      │                                                     │
│  Genotype: ▼  │  Animal Info:                                       │
│  Fit Type: ▼  │  Assay: 1  Track: 5  Genotype: wt                   │
│               │  y_initial = 0.650 ± 0.021  R² = 0.94              │
│  ANIMAL LIST  │                                                     │
│               │                                                     │
│  A01_T01 | wt │         Raw speed (gray dots)                      │
│  A01_T02 | wt │                      •                              │
│  A01_T03 | wt │      • •           • •                Richards      │
│ ▶A01_T04 | wt │    •     •       •     •              fit (red)     │
│  A01_T05 | wt │  •         •   •         •                          │
│  A02_T01 | mut│            •                                        │
│               │         • •                                         │
│               │     Food│encounter (t=0)                           │
│  [◄ Prev]     │         │                                           │
│  [Next ►]     │─────────┼─────────────────────────────────────────│
│               │  -120s  0s  Time relative to encounter  +120s      │
│               │                                                     │
│               │  [Export Current]  [Export All Filtered]            │
└───────────────┴─────────────────────────────────────────────────────┘
```

## Loading Data

### Method 1: File → Load Results Directory

1. Click `File` → `Load Results Directory...`
2. Navigate to your results folder (e.g., `results/`)
3. Select the folder containing:
   - `richards_fit_parameters.csv`
   - `windowed_speed_data.csv`
4. Click "Select Folder"

The viewer automatically finds and loads both CSV files.

### Method 2: File → Load Parameters CSV

1. Click `File` → `Load Parameters CSV...`
2. Select `richards_fit_parameters.csv`
3. When prompted, select `windowed_speed_data.csv`

This method is useful if files are in different locations.

## Navigation

### Mouse
- **Click** an animal in the list to view its trace
- **Scroll** through the animal list

### Keyboard Shortcuts
- **↑ / ↓** or **← / →** - Previous/Next animal
- **Space** - Next animal
- **E** - Export current plot
- **Ctrl+O** - Load directory (File menu)

### Buttons
- **◄ Previous** - Show previous animal
- **Next ►** - Show next animal

## Filtering

Use the dropdown menus to filter animals:

### Genotype Filter
- Shows all unique genotypes in your data
- Select specific genotype to view only those animals
- Select "All" to show everything

### Fit Type Filter
Options:
- **decrease** - Animals that slowed down significantly
- **increase** - Animals that sped up (rare)
- **flat** - Completely flat traces (no change)
- **no_transition** - No significant change detected
- **failed** - Fitting failed

## Understanding the Plot

### Raw Data (Gray Dots)
- Individual speed measurements
- Centered on food encounter (t=0)
- Shows natural variability

### Fitted Curve (Red Line)
- Richards curve fit through the data
- Smooth representation of the transition
- Should follow the general trend

### Food Encounter (Green Dashed Line)
- Vertical line at t=0
- Marks when nose entered food
- Transition dynamics measured relative to this

### Parameter Text Box
Shows fitted parameters with uncertainties:
```
Fit type: decrease
y_initial = 0.650 ± 0.021
y_final = 0.310 ± 0.015
A = -0.340 ± 0.026
B = 0.095 ± 0.018
M = 5.2 ± 3.1 s
ν = 1.12 ± 0.21
R² = 0.943
```

**Interpretation:**
- Small SE (standard error) = well-determined parameter
- R² close to 1.0 = good fit
- For flat/no_transition: B, M, ν have large SE (ignore them)

## Quality Control Workflow

Use the viewer to spot-check fits:

1. **Load your results**
   ```bash
   python view_richards_fits.py results/
   ```

2. **Check overall fit quality**
   - Look at R² values in the list
   - Animals with R² < 0.7 may have poor fits

3. **Filter to potential problems**
   - Set "Fit Type" → "failed" to see failed fits
   - Or filter to low R² animals

4. **Browse visually**
   - Use arrow keys to quickly scan through animals
   - Watch for:
     - Red curve not following gray dots
     - Extreme parameter values
     - Noisy/choppy traces

5. **Export good examples**
   - Find representative animals from each genotype
   - Press 'E' to export
   - Use for presentations/papers

## Exporting Plots

### Single Animal
1. Navigate to the animal you want
2. Click "Export Current Plot" or press 'E'
3. File saved as `animal_XX_YY.png` in results directory

### Batch Export
1. Apply filters (e.g., genotype = "wt", fit_type = "decrease")
2. Click "Export All Filtered Plots"
3. Confirm the number of plots
4. Files saved to `results/individual_plots/`
5. Naming: `animal_01_05.png` (assay_track)

Export settings:
- Format: PNG
- Resolution: 150 DPI (publication quality)
- Size: ~10x6 inches

## Typical Use Cases

### Use Case 1: Quality Control
**Goal:** Check that fits look reasonable

1. Load results
2. Set filter: Fit Type → "decrease" (normal animals)
3. Use arrow keys to quickly scan through
4. Note any with R² < 0.8 or weird-looking fits
5. Decide if those animals should be excluded

### Use Case 2: Compare Genotypes
**Goal:** Visual comparison of WT vs mutant

1. Filter: Genotype → "wt"
2. Browse a few representative animals, export good ones
3. Filter: Genotype → "mutant"
4. Browse and export comparable examples
5. Create side-by-side figure in your paper

### Use Case 3: Find Interesting Animals
**Goal:** Identify animals with unusual dynamics

1. Sort list mentally by fit type
2. Look for animals with extreme parameter values
3. Check if these are artifacts or real biology
4. Export for further investigation

### Use Case 4: Presentation Figures
**Goal:** Get clean plots for a talk

1. Find the "best" example of each phenotype
   - High R²
   - Clear transition
   - Minimal noise
2. Export with 'E'
3. Import into PowerPoint/Keynote

## Tips and Tricks

### Fast Navigation
- Hold down arrow key to scan quickly
- Or press Space repeatedly for one-handed browsing

### Filter Combinations
- Combine genotype + fit type filters
- Example: "wt" + "decrease" = normal WT animals only

### Check Specific Animals
- If you know assay/track numbers from other analysis
- Scan the list for "A01_T05" format
- Click to jump directly to that animal

### Keyboard-Only Workflow
1. Start: `python view_richards_fits.py results/`
2. Navigate: arrow keys
3. Export: E key
4. No mouse needed!

## Troubleshooting

### "No data loaded" message
**Solution:** Use File → Load Results Directory to load data first

### Empty animal list
**Check:**
- Are filters too restrictive? Set both to "All"
- Were any animals successfully fit? Check console output

### Plot looks wrong
**Possible causes:**
- Failed fit (check fit type)
- Insufficient data in window
- Try loading with longer time windows in analysis

### Can't see all parameters
**For flat/no_transition animals:**
- B, M, ν are poorly determined (very large SE)
- This is normal - only y_initial and A are meaningful

### Slow loading
- Large datasets (>100 animals) may take a few seconds
- Windowed data CSV contains all raw speed points
- Be patient during initial load

## Advanced: Batch Processing

For automated figure generation:

```python
# Load data programmatically
import pandas as pd
from view_richards_fits import RichardsFitViewer

# Initialize viewer without GUI
df_params = pd.read_csv('results/richards_fit_parameters.csv')

# Filter to animals of interest
wt_good = df_params[
    (df_params['strain_genotype'] == 'wt') &
    (df_params['r_squared'] > 0.85) &
    (df_params['fit_type'] == 'decrease')
]

# Use the plotting functions directly
from richards_speed_analysis import plot_individual_fit

for _, row in wt_good.iterrows():
    plot_individual_fit(
        assay_num=row['assay_num'],
        track_num=row['track_num'],
        windowed_data=windowed_data,
        results=results,
        save_path=f"figures/wt_{row['assay_num']}_{row['track_num']}.png",
        show=False
    )
```

## Summary

The interactive viewer lets you:
- ✅ Load CSV files with point-and-click
- ✅ Browse animals one-by-one with arrow keys
- ✅ Filter by genotype and fit type
- ✅ See raw data with fitted curves overlaid
- ✅ Check fit quality visually
- ✅ Export individual or batch plots

**Start exploring your data:**
```bash
python view_richards_fits.py
```

Then use File → Load Results Directory to begin!
