#!/usr/bin/env python3
"""
CSV/TXT Organizer UI (Wide → Long) with sequential metadata dialogs, smart re-import, and filename inference.

What's new in this version
- Column order: `source_file` is now the FIRST column in the composite.
- The last columns are coordinates/flags in this exact order: `x`, `y`, `centroid_on_food`, `nose_on_food`, `food_encounter`.
- Supports both older wide format (x,y,flag) and NEW wide format from the masking pipeline that emits
  per-worm `*_x`, `*_y`, `*_centroid_on_food` (centroid vs mask) and `*__nose_on_food` (nose vs mask).
- Gracefully handles composite/long CSVs produced by this script: reorders columns and assigns
  `assay_num` for the current import batch (does not break if long files are added).
- Remembers last-entered metadata per session and pre-fills dialogs.
- `nose_on_food`: binary flag for when nose is on food (previously called food_encounter)
- `food_encounter`: marks "food" only at the FIRST transition when nose hits food
- NEW: Filename inference mode - automatically parse metadata from standardized filenames
  Format: PC1_5.28.2025_m_wt_3hr# (PC#_date_sex_strain_treatment#)

Output columns (final order):
  1) source_file
  2) assay_num
  3) track_num
  4) pc_number
  5) sex
  6) strain_genotype
  7) treatment
  8) time
  9...) (any extra columns carried through, if present)
  last-4) x, y, centroid_on_food, nose_on_food, food_encounter

Run: python csv_analysis_ui.py
Requires: Python 3.9+, pandas
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "CSV/TXT Organizer - Worm Tracks (Wide → Long)"
DEFAULT_ANALYZE_SUBDIR = "analyze"

# ------------------------------
# Utility: delimiter inference
# ------------------------------
def _infer_delimiter(path: Path) -> str:
    delimiters = [',', '\t', ';', ' ']
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    sample = line
                    break
            else:
                return ','
    except Exception:
        return ','
    best, best_count = ',', -1
    for d in delimiters:
        c = sample.count(d)
        if c > best_count:
            best, best_count = d, c
    return best

# ------------------------------
# Filename parsing utilities
# ------------------------------
def normalize_date(date_str: str) -> str:
    """Normalize various date formats to a consistent format."""
    # Remove any extra characters and normalize separators
    date_str = re.sub(r'[^\d\.\-\/]', '', date_str)
    # Convert - or / to .
    date_str = re.sub(r'[\-\/]', '.', date_str)
    return date_str

def normalize_sex(sex_str: str) -> str:
    """Normalize sex field to standard lowercase format."""
    sex_lower = sex_str.lower().strip()
    sex_mapping = {
        'm': 'm', 'male': 'm',
        'f': 'f', 'female': 'f', 
        'h': 'h', 'hermaphrodite': 'h', 'herm': 'h'
    }
    return sex_mapping.get(sex_lower, sex_lower)

def parse_filename_metadata(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse metadata from filename with format: PC1_5.28.2025_m_wt_3hr#
    Expected format: PC#_date_sex_strain_treatment#
    
    Returns dict with keys: pc_number, date, sex, strain_genotype, treatment
    Returns None if parsing fails.
    """
    # Remove file extension
    basename = Path(filename).stem
    
    # Support both _ and - as delimiters
    parts = re.split(r'[_\-]', basename)
    
    if len(parts) < 5:
        return None
    
    try:
        # Extract components
        pc_part = parts[0].strip()
        date_part = parts[1].strip()
        sex_part = parts[2].strip()
        strain_part = parts[3].strip()
        treatment_part = parts[4].strip()
        
        # Validate PC number format
        if not re.match(r'^PC\d+$', pc_part, re.IGNORECASE):
            return None
        
        # Normalize components
        pc_number = pc_part.upper()  # Keep PC uppercase
        date = normalize_date(date_part)
        sex = normalize_sex(sex_part)
        strain_genotype = strain_part.lower()
        # Remove trailing # from treatment and normalize
        treatment = re.sub(r'#*$', '', treatment_part).lower()
        
        return {
            'pc_number': pc_number,
            'date': date,
            'sex': sex,
            'strain_genotype': strain_genotype,
            'treatment': treatment
        }
        
    except Exception:
        return None

def validate_parsed_metadata(metadata: Dict[str, str]) -> Tuple[bool, str]:
    """Validate parsed metadata. Returns (is_valid, error_message)."""
    if not metadata['pc_number']:
        return False, "PC number is required"
    
    if not re.match(r'^PC\d+$', metadata['pc_number'], re.IGNORECASE):
        return False, "PC number must be in format PC# (e.g., PC1)"
    
    valid_sex = {'m', 'f', 'h', 'male', 'female', 'hermaphrodite'}
    if metadata['sex'].lower() not in valid_sex:
        return False, f"Sex must be one of: {', '.join(valid_sex)}"
    
    return True, ""

# ------------------------------
# Robust file reading
# ------------------------------
def read_table(path: Path) -> pd.DataFrame:
    delim = _infer_delimiter(path)
    for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            df = pd.read_csv(path, sep=delim, engine='python', encoding=enc)
            if df.shape[1] == 1 and delim != '\t':  # try tab if single col
                df = pd.read_csv(path, sep='\t', engine='python', encoding=enc)
            return df
        except Exception:
            continue
    # last resort: whitespace
    try:
        return pd.read_csv(path, sep=r"\s+", engine='python')
    except Exception as e:
        raise RuntimeError(f"Failed to read {path.name}: {e}")

# ------------------------------
# Helpers to detect long format and normalize columns
# ------------------------------
LONG_REQUIRED = {'time'}
LONG_ANY_COORD = {'x', 'y'}
LONG_ANY_FLAGS = {'nose_on_food', 'centroid_on_food', 'food_encounter'}

def is_long_format(df: pd.DataFrame) -> bool:
    cols = {c.strip().lower() for c in df.columns}
    # Minimal long-format check: must have time and at least one of x/y OR one of the flags
    return ('time' in cols) and (len(LONG_ANY_COORD & cols) > 0 or len(LONG_ANY_FLAGS & cols) > 0)

def enforce_column_order(df: pd.DataFrame) -> pd.DataFrame:
    # Desired final order
    first = ['source_file', 'assay_num', 'track_num', 'pc_number', 'sex', 'strain_genotype', 'treatment', 'time']
    last = ['x', 'y', 'centroid_on_food', 'nose_on_food', 'food_encounter']
    existing = list(df.columns)

    out = []
    for c in first:
        if c in df.columns:
            out.append(c)

    # Middle: anything not in first/last
    for c in existing:
        if c not in out and c not in last:
            out.append(c)

    # End: coords/flags
    for c in last:
        if c in df.columns:
            out.append(c)

    return df[out]

# ------------------------------
# Function to create food_encounter column from nose_on_food
# ------------------------------
def create_food_encounter_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates food_encounter column that marks 'food' only at the first transition
    from 0 to 1 in nose_on_food for each track.
    """
    if 'nose_on_food' not in df.columns:
        df['food_encounter'] = ''
        return df
    
    df = df.copy()
    df['food_encounter'] = ''
    
    # Group by track to handle each worm separately
    if 'track_num' in df.columns:
        for track_id in df['track_num'].unique():
            mask = df['track_num'] == track_id
            track_data = df.loc[mask, 'nose_on_food'].reset_index(drop=True)
            
            # Find first transition from 0 to 1
            transitions = (track_data.diff() == 1) & (track_data == 1)
            if transitions.any():
                first_encounter_idx = transitions.idxmax()
                # Get the original index in the full dataframe
                original_idx = df.loc[mask].iloc[first_encounter_idx].name
                df.loc[original_idx, 'food_encounter'] = 'food'
    else:
        # If no track_num, treat entire dataset as one track
        nose_data = df['nose_on_food'].reset_index(drop=True)
        transitions = (nose_data.diff() == 1) & (nose_data == 1)
        if transitions.any():
            first_encounter_idx = transitions.idxmax()
            df.iloc[first_encounter_idx, df.columns.get_loc('food_encounter')] = 'food'
    
    return df

# ------------------------------
# Column detection / parsing for wide format
# ------------------------------
_TIME_CANDIDATES = ['time', 'frame', 't', 'frames']

def detect_time_column(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    for name in _TIME_CANDIDATES:
        for c in cols:
            if c.strip().lower() == name:
                return c
    return cols[0]

# classify columns and catch centroid_on_food separately
_CENTROID_FLAG_PAT = re.compile(r'centroid[_ ]?on[_ ]?food', re.I)
_X_PAT = re.compile(r'(?:^|[^a-z])(x|xpos|x_coord|xcoordinate|xposition)\b', re.I)
_Y_PAT = re.compile(r'(?:^|[^a-z])(y|ypos|y_coord|ycoordinate|yposition)\b', re.I)
_FLAG_PAT = re.compile(r'(?:flag|food|enc|onfood|nose_on_food|on_food|flag\d*)', re.I)

def extract_id_token(colname: str) -> Optional[str]:
    m = re.search(r'(\d+)\s*$', colname)
    if m:
        return m.group(1)
    parts = re.split(r'[_\s]+', colname.strip())
    if len(parts) > 1:
        return parts[-1].lower()
    return None

def classify_col(colname: str) -> str:
    name = colname.strip().lower()
    if _CENTROID_FLAG_PAT.search(name):
        return 'centroid'   # specific centroid_on_food signal
    if _X_PAT.search(name): return 'x'
    if _Y_PAT.search(name): return 'y'
    if _FLAG_PAT.search(name): return 'flag'  # nose-based food encounter or generic flags
    if name.endswith('x'): return 'x'
    if name.endswith('y'): return 'y'
    if 'flag' in name or 'food' in name or 'enc' in name: return 'flag'
    return 'other'

def find_worm_sets(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping: worm_id -> {'x': col, 'y': col, 'flag': col?, 'centroid': col?}
    At least x and y must exist to create a track.
    """
    cols = [str(c) for c in df.columns]
    buckets: Dict[str, Dict[str, str]] = {}
    for c in cols:
        ctype = classify_col(c)
        if ctype == 'other':
            continue
        tok = extract_id_token(c) or c
        if tok not in buckets:
            buckets[tok] = {}
        buckets[tok][ctype] = c
    sets_ = {wid: sub for wid, sub in buckets.items() if 'x' in sub and 'y' in sub}
    return sets_

def normalize_flag_series(series: pd.Series) -> pd.Series:
    def to01(v):
        if pd.isna(v): return 0
        s = str(v).strip().lower()
        if s in ('1', 'true', 't', 'yes', 'y', '*', 'food', 'on', 'enc', 'encounter'):
            return 1
        try:
            return 1 if float(s) != 0.0 else 0
        except Exception:
            return 0
    return series.map(to01)

def parse_wide_file_to_long(path: Path,
                            assay_num: int,
                            pc_number: str,
                            sex: str,
                            strain_genotype: str,
                            treatment: str) -> pd.DataFrame:
    df = read_table(path)
    if df.empty or df.shape[1] < 2:
        raise RuntimeError(f"{path.name}: not enough columns to parse.")

    # If the file is already long format, just standardize/order columns and override assay_num
    if is_long_format(df):
        # Handle legacy food_encounter column (rename to nose_on_food if needed)
        if 'food_encounter' in df.columns and 'nose_on_food' not in df.columns:
            df = df.rename(columns={'food_encounter': 'nose_on_food'})
        
        # Ensure required columns exist / fill if missing
        for col in ['source_file', 'track_num', 'pc_number', 'sex', 'strain_genotype', 'treatment']:
            if col not in df.columns:
                if col == 'source_file':
                    df[col] = path.name
                elif col == 'track_num':
                    # If track_num missing, assign 1 per unique x/y/centroid group or per file
                    df[col] = 1
                else:
                    df[col] = ''
        
        # Ensure nose_on_food exists
        if 'nose_on_food' not in df.columns:
            df['nose_on_food'] = 0
            
        df['assay_num'] = assay_num
        
        # Create the new food_encounter column
        df = create_food_encounter_column(df)
        
        # Order columns and return
        df = enforce_column_order(df)
        return df

    # Otherwise, wide format parsing
    time_col = detect_time_column(df)
    time_series = df[time_col]
    sets_ = find_worm_sets(df)

    if not sets_:
        # attempt fallback: repeating groups (x,y,flag,centroid?) after time
        other_cols = [c for c in df.columns if c != time_col]
        if len(other_cols) >= 2:
            # heuristic: try groups of 4 first (x,y,centroid,flag), else groups of 3 (x,y,flag)
            group_size = 4 if len(other_cols) % 4 == 0 else (3 if len(other_cols) % 3 == 0 else None)
            if group_size is None:
                raise RuntimeError(f"{path.name}: could not detect worm column groups.")
            sets_ = {}
            n = len(other_cols) // group_size
            for i in range(n):
                grp = other_cols[group_size*i:group_size*i+group_size]
                entry = {'x': grp[0], 'y': grp[1]}
                if group_size == 4:
                    entry['centroid'] = grp[2]
                    entry['flag'] = grp[3]
                elif group_size == 3:
                    entry['flag'] = grp[2]
                sets_[str(i+1)] = entry
        else:
            raise RuntimeError(f"{path.name}: could not detect worm columns.")

    rows: List[pd.DataFrame] = []
    track_counter = 0

    def worm_sort_key(k: str):
        try:
            return int(re.sub(r'\D+', '', k) or '0')
        except Exception:
            return 0

    for worm_id in sorted(sets_.keys(), key=worm_sort_key):
        cols = sets_[worm_id]
        xcol, ycol = cols.get('x'), cols.get('y')
        flagcol = cols.get('flag')     # expected to be nose-based food encounter
        centcol = cols.get('centroid') # centroid_on_food

        if xcol is None or ycol is None:
            continue

        track_counter += 1
        x = pd.to_numeric(df[xcol], errors='coerce')
        y = pd.to_numeric(df[ycol], errors='coerce')

        if centcol and centcol in df.columns:
            centroid_flag = normalize_flag_series(df[centcol])
        else:
            centroid_flag = pd.Series(0, index=df.index)

        if flagcol and flagcol in df.columns:
            nose_flag = normalize_flag_series(df[flagcol])
        else:
            nose_flag = pd.Series(0, index=df.index)

        part = pd.DataFrame({
            'source_file': path.name,
            'assay_num': assay_num,
            'track_num': track_counter,
            'pc_number': pc_number,
            'sex': sex,
            'strain_genotype': strain_genotype,
            'treatment': treatment,
            'time': pd.to_numeric(time_series, errors='coerce'),
            'x': x,
            'y': y,
            'centroid_on_food': centroid_flag,
            'nose_on_food': nose_flag,
        })
        # Drop rows with NaN time
        part = part[~part['time'].isna()].reset_index(drop=True)
        
        # Create food_encounter column for this track
        part = create_food_encounter_column(part)
        
        # Enforce final order now
        part = enforce_column_order(part)
        rows.append(part)

    if not rows:
        raise RuntimeError(f"{path.name}: no valid worm x/y columns found.")
    return pd.concat(rows, ignore_index=True)

# ------------------------------
# UI components
# ------------------------------
class InputModeDialog(tk.Toplevel):
    """Dialog to choose between manual input and filename inference."""
    def __init__(self, master):
        super().__init__(master)
        self.title("Select Input Mode")
        self.resizable(False, False)
        self.result = None

        frm = ttk.Frame(self, padding=15)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Choose how to enter experimental metadata:", 
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0,15))

        # Manual mode
        manual_frame = ttk.LabelFrame(frm, text="Manual Input", padding=10)
        manual_frame.grid(row=1, column=0, padx=(0,10), pady=5, sticky="nsew")
        
        ttk.Label(manual_frame, text="Enter metadata for each file\nthrough dialog boxes").grid(row=0, column=0, pady=5)
        ttk.Button(manual_frame, text="Use Manual Input", 
                   command=lambda: self._set_result('manual')).grid(row=1, column=0, pady=5)

        # Filename inference mode
        inference_frame = ttk.LabelFrame(frm, text="Filename Inference", padding=10)
        inference_frame.grid(row=1, column=1, padx=(10,0), pady=5, sticky="nsew")
        
        ttk.Label(inference_frame, text="Parse metadata from filenames\n\nExpected format:").grid(row=0, column=0, pady=2)
        ttk.Label(inference_frame, text="PC1_5.28.2025_m_wt_3hr#", 
                  font=('TkDefaultFont', 9, 'bold'), foreground='blue').grid(row=1, column=0, pady=2)
        ttk.Label(inference_frame, text="(PC#_date_sex_strain_treatment#)", 
                  font=('TkDefaultFont', 8), foreground='gray').grid(row=2, column=0, pady=2)
        ttk.Button(inference_frame, text="Use Filename Inference", 
                   command=lambda: self._set_result('inference')).grid(row=3, column=0, pady=5)

        # Cancel button
        ttk.Button(frm, text="Cancel", command=self._cancel).grid(row=2, column=0, columnspan=2, pady=(15,0))

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_visibility()
        self.focus_set()

    def _set_result(self, mode):
        self.result = mode
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

class FilenamePreviewDialog(tk.Toplevel):
    """Dialog to preview and confirm parsed metadata from filenames."""
    def __init__(self, master, parsed_data: List[Tuple[Path, Dict[str, str]]]):
        super().__init__(master)
        self.title("Preview Parsed Metadata")
        self.geometry("800x500")
        self.result = None
        self.parsed_data = parsed_data

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        ttk.Label(frm, text="Preview of parsed metadata (click rows to edit):", 
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, sticky="w", pady=(0,10))

        # Create treeview for preview
        columns = ('filename', 'pc_number', 'sex', 'strain_genotype', 'treatment', 'date')
        self.tree = ttk.Treeview(frm, columns=columns, show='headings', height=15)
        
        # Define headings
        self.tree.heading('filename', text='Filename')
        self.tree.heading('pc_number', text='PC Number')
        self.tree.heading('sex', text='Sex')
        self.tree.heading('strain_genotype', text='Strain/Genotype')
        self.tree.heading('treatment', text='Treatment')
        self.tree.heading('date', text='Date')

        # Set column widths
        self.tree.column('filename', width=150)
        self.tree.column('pc_number', width=80)
        self.tree.column('sex', width=60)
        self.tree.column('strain_genotype', width=120)
        self.tree.column('treatment', width=100)
        self.tree.column('date', width=100)

        self.tree.grid(row=1, column=0, sticky="nsew", pady=(0,10))
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(frm, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Populate tree
        for i, (path, metadata) in enumerate(parsed_data):
            self.tree.insert('', 'end', iid=i, values=(
                path.name,
                metadata['pc_number'],
                metadata['sex'],
                metadata['strain_genotype'],
                metadata['treatment'],
                metadata['date']
            ))

        # Bind double-click for editing
        self.tree.bind('<Double-1>', self._edit_item)

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Edit Selected", command=self._edit_selected).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Proceed", command=self._proceed).grid(row=0, column=2, padx=5)

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _edit_item(self, event):
        selection = self.tree.selection()
        if selection:
            self._edit_selected()

    def _edit_selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a row to edit.")
            return
        
        item_id = int(selection[0])
        path, metadata = self.parsed_data[item_id]
        
        # Open edit dialog
        edit_dialog = FileMetaWizard(self, path.name, defaults=metadata)
        self.wait_window(edit_dialog)
        
        if edit_dialog.result:
            # Update the stored metadata and tree display
            self.parsed_data[item_id] = (path, edit_dialog.result)
            self.tree.item(selection[0], values=(
                path.name,
                edit_dialog.result['pc_number'],
                edit_dialog.result['sex'],
                edit_dialog.result['strain_genotype'],
                edit_dialog.result['treatment'],
                edit_dialog.result.get('date', metadata.get('date', ''))
            ))

    def _proceed(self):
        self.result = self.parsed_data
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

class FileMetaWizard(tk.Toplevel):
    """Modal dialog to collect metadata for a single file (prefills with last values)."""
    def __init__(self, master, fname: str, defaults: Optional[dict] = None):
        super().__init__(master)
        self.title(f"Metadata for file: {fname}")
        self.resizable(False, False)
        self.result = None

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        defaults = defaults or {}
        self.pc_var = tk.StringVar(value=defaults.get('pc_number', ''))
        self.sex_var = tk.StringVar(value=defaults.get('sex', ''))
        self.strain_var = tk.StringVar(value=defaults.get('strain_genotype', ''))
        self.treat_var = tk.StringVar(value=defaults.get('treatment', ''))

        def add_row(r, label, var, tip):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", padx=(0,8), pady=3)
            e = ttk.Entry(frm, textvariable=var, width=36)
            e.grid(row=r, column=1, sticky="ew", pady=3)
            ttk.Label(frm, text=tip, foreground="#666").grid(row=r, column=2, sticky="w", padx=(8,0))

        add_row(0, "PC #", self.pc_var, "(e.g., PC1)")
        add_row(1, "Sex", self.sex_var, "(e.g., M / H)")
        add_row(2, "Strain/Genotype", self.strain_var, "(e.g., N2 or tph-1)")
        add_row(3, "Treatment", self.treat_var, "(e.g., fed / 30min starved)")

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=3, pady=(10,0), sticky="e")
        ttk.Button(btns, text="Cancel", command=self._cancel).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="OK", command=self._ok).grid(row=0, column=1, padx=5)

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_visibility()
        self.focus_set()

    def _ok(self):
        self.result = {
            'pc_number': self.pc_var.get().strip(),
            'sex': self.sex_var.get().strip(),
            'strain_genotype': self.strain_var.get().strip(),
            'treatment': self.treat_var.get().strip(),
        }
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x680")
        self.minsize(900, 600)

        # Remember last-entered metadata across files in a session
        self.last_meta = {
            'pc_number': '',
            'sex': '',
            'strain_genotype': '',
            'treatment': '',
        }

        style = ttk.Style(self)
        try:
            self.tk.call('tk', 'scaling', 1.25)
        except Exception:
            pass
        style.theme_use('clam')

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.new_tab = ttk.Frame(nb, padding=10)
        self.cont_tab = ttk.Frame(nb, padding=10)
        nb.add(self.new_tab, text="New Analysis")
        nb.add(self.cont_tab, text="Continue Analysis")
        self._build_new_tab()
        self._build_continue_tab()

    def _get_input_mode(self) -> Optional[str]:
        """Show dialog to choose input mode. Returns 'manual', 'inference', or None."""
        mode_dialog = InputModeDialog(self)
        self.wait_window(mode_dialog)
        return mode_dialog.result

    def _process_files_with_inference(self, files: List[Path]) -> Optional[List[Tuple[Path, Dict[str, str]]]]:
        """
        Process files using filename inference.
        Returns list of (path, metadata) tuples or None if cancelled.
        """
        parsed_data = []
        failed_files = []

        for path in files:
            metadata = parse_filename_metadata(path.name)
            if metadata is None:
                failed_files.append(path.name)
                continue
            
            # Validate parsed metadata
            is_valid, error_msg = validate_parsed_metadata(metadata)
            if not is_valid:
                failed_files.append(f"{path.name}: {error_msg}")
                continue
                
            parsed_data.append((path, metadata))

        # Show any failed files
        if failed_files:
            failed_msg = "The following files could not be parsed:\n\n" + "\n".join(failed_files)
            failed_msg += "\n\nExpected format: PC1_5.28.2025_m_wt_3hr#"
            messagebox.showwarning("Parsing Failures", failed_msg)
            
            if not parsed_data:
                return None

        # Show preview dialog
        if parsed_data:
            preview_dialog = FilenamePreviewDialog(self, parsed_data)
            self.wait_window(preview_dialog)
            return preview_dialog.result
        
        return None

    def _process_files_manual(self, files: List[Path]) -> Optional[List[Tuple[Path, Dict[str, str]]]]:
        """
        Process files using manual input dialogs.
        Returns list of (path, metadata) tuples or None if cancelled.
        """
        manual_data = []
        
        for path in files:
            md = FileMetaWizard(self, path.name, defaults=self.last_meta)
            self.wait_window(md)
            if md.result is None:
                # User cancelled, ask if they want to skip this file or abort
                response = messagebox.askyesnocancel(
                    "File Skipped", 
                    f"Skip {path.name} and continue with remaining files?\n\n"
                    "Yes = Skip this file\n"
                    "No = Abort entire process\n"
                    "Cancel = Go back to enter metadata"
                )
                if response is True:  # Skip this file
                    continue
                elif response is False:  # Abort
                    return None
                else:  # Cancel - go back (this shouldn't happen with our dialog)
                    continue
            
            # Update defaults for next file
            self.last_meta = md.result.copy()
            manual_data.append((path, md.result))
            
        return manual_data if manual_data else None

    # -------- New Analysis Tab --------
    def _build_new_tab(self):
        frm = self.new_tab
        desc = ttk.Label(frm, text=(
            "Reads all .csv/.txt files from a directory (default: ./analyze),\n"
            "asks for metadata per file (manual or filename inference), and saves a composite long-format CSV.\n"
            "Supports new wide files with centroid_on_food and nose_on_food, and re-import of long CSVs.\n"
            "Creates food_encounter column marking only the FIRST nose-food contact per track.\n"
            "Filename format for inference: PC1_5.28.2025_m_wt_3hr# (PC#_date_sex_strain_treatment#)"
        ))
        desc.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,8))

        ttk.Label(frm, text="Data directory:").grid(row=1, column=0, sticky="e")
        self.new_dir_var = tk.StringVar(value=str((Path(sys.argv[0]).resolve().parent / DEFAULT_ANALYZE_SUBDIR)))
        ttk.Entry(frm, textvariable=self.new_dir_var, width=60).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Browse...", command=self._browse_new_dir).grid(row=1, column=2, sticky="w")

        ttk.Label(frm, text="Save composite as:").grid(row=2, column=0, sticky="e")
        self.new_save_var = tk.StringVar(value=str(Path.cwd() / "composite.csv"))
        ttk.Entry(frm, textvariable=self.new_save_var, width=60).grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Choose...", command=self._choose_new_save).grid(row=2, column=2, sticky="w")

        ttk.Button(frm, text="Start Import", command=self._run_new_analysis).grid(row=3, column=1, pady=(10,0))

        self.new_log = tk.Text(frm, height=20, wrap="word")
        self.new_log.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(10,0))
        frm.rowconfigure(4, weight=1)
        frm.columnconfigure(1, weight=1)

    def _browse_new_dir(self):
        d = filedialog.askdirectory(title="Choose data directory")
        if d:
            self.new_dir_var.set(d)

    def _choose_new_save(self):
        f = filedialog.asksaveasfilename(title="Save composite as",
                                         defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if f:
            self.new_save_var.set(f)

    def _run_new_analysis(self):
        data_dir = Path(self.new_dir_var.get().strip())
        save_path = Path(self.new_save_var.get().strip())
        if not data_dir.exists() or not data_dir.is_dir():
            messagebox.showerror("Error", f"Data directory does not exist:\n{data_dir}")
            return

        files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in ('.csv', '.txt')])
        if not files:
            messagebox.showwarning("No files", f"No .csv/.txt files found in:\n{data_dir}")
            return

        # Get input mode
        input_mode = self._get_input_mode()
        if input_mode is None:
            return

        # Process files based on mode
        if input_mode == 'inference':
            file_data = self._process_files_with_inference(files)
        else:  # manual
            file_data = self._process_files_manual(files)

        if not file_data:
            self._log_new("Import cancelled or no valid files processed.")
            return

        # Process the files
        composite_rows: List[pd.DataFrame] = []
        assay_num = 1

        for path, metadata in file_data:
            try:
                part = parse_wide_file_to_long(
                    path=path,
                    assay_num=assay_num,
                    pc_number=metadata['pc_number'],
                    sex=metadata['sex'],
                    strain_genotype=metadata['strain_genotype'],
                    treatment=metadata['treatment'],
                )
                composite_rows.append(part)
                self._log_new(f"Parsed {path.name}: assay_num={assay_num}, rows={len(part)}")
                assay_num += 1
            except Exception as e:
                self._log_new(f"ERROR parsing {path.name}: {e}")

        if not composite_rows:
            messagebox.showwarning("Nothing saved", "No data imported.")
            return

        composite = pd.concat(composite_rows, ignore_index=True)
        try:
            composite = enforce_column_order(composite)
            composite.to_csv(save_path, index=False)
            self._log_new(f"Saved composite to: {save_path}")
            messagebox.showinfo("Done", f"Composite saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save composite:\n{e}")

    def _log_new(self, msg: str):
        self.new_log.insert("end", msg + "\n")
        self.new_log.see("end")
        self.new_log.update_idletasks()

    # -------- Continue Analysis Tab --------
    def _build_continue_tab(self):
        frm = self.cont_tab
        desc = ttk.Label(frm, text=(
            "Open an existing composite CSV, select a new data directory, and append new assays.\n"
            "Assay numbering continues from the last assay_num (supports re-import of long CSVs).\n"
            "Handles legacy food_encounter columns and creates new food_encounter logic.\n"
            "Supports both manual input and filename inference for new files."
        ))
        desc.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,8))

        ttk.Label(frm, text="Existing composite:").grid(row=1, column=0, sticky="e")
        self.cont_comp_var = tk.StringVar(value=str(Path.cwd() / "composite.csv"))
        ttk.Entry(frm, textvariable=self.cont_comp_var, width=60).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Browse...", command=self._browse_composite).grid(row=1, column=2, sticky="w")

        ttk.Label(frm, text="New data directory:").grid(row=2, column=0, sticky="e")
        self.cont_data_dir_var = tk.StringVar(value=str((Path(sys.argv[0]).resolve().parent / DEFAULT_ANALYZE_SUBDIR)))
        ttk.Entry(frm, textvariable=self.cont_data_dir_var, width=60).grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Choose...", command=self._browse_cont_dir).grid(row=2, column=2, sticky="w")

        ttk.Label(frm, text="Save updated composite as:").grid(row=3, column=0, sticky="e")
        self.cont_save_var = tk.StringVar(value=str(Path.cwd() / "composite_updated.csv"))
        ttk.Entry(frm, textvariable=self.cont_save_var, width=60).grid(row=3, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Choose...", command=self._choose_cont_save).grid(row=3, column=2, sticky="w")

        ttk.Button(frm, text="Append New Data", command=self._run_continue_analysis).grid(row=4, column=1, pady=(10,0))

        self.cont_log = tk.Text(frm, height=20, wrap="word")
        self.cont_log.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(10,0))
        frm.rowconfigure(5, weight=1)
        frm.columnconfigure(1, weight=1)

    def _browse_composite(self):
        f = filedialog.askopenfilename(title="Open existing composite CSV",
                                       filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if f:
            self.cont_comp_var.set(f)

    def _browse_cont_dir(self):
        d = filedialog.askdirectory(title="Choose new data directory")
        if d:
            self.cont_data_dir_var.set(d)

    def _choose_cont_save(self):
        f = filedialog.asksaveasfilename(title="Save updated composite as",
                                         defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if f:
            self.cont_save_var.set(f)

    def _run_continue_analysis(self):
        comp_path = Path(self.cont_comp_var.get().strip())
        if not comp_path.exists():
            messagebox.showerror("Error", f"Composite not found:\n{comp_path}")
            return
        try:
            composite = pd.read_csv(comp_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read composite:\n{e}")
            return

        # Handle legacy composite files - rename old food_encounter to nose_on_food if needed
        if 'food_encounter' in composite.columns and 'nose_on_food' not in composite.columns:
            # Check if food_encounter contains binary values (legacy nose_on_food data)
            unique_vals = set(composite['food_encounter'].dropna().astype(str).str.lower())
            if unique_vals.issubset({'0', '1', '0.0', '1.0', 'true', 'false', 'nan'}):
                composite = composite.rename(columns={'food_encounter': 'nose_on_food'})
                composite['nose_on_food'] = normalize_flag_series(composite['nose_on_food'])
                self._log_cont("Renamed legacy food_encounter column to nose_on_food")
        
        # Ensure nose_on_food exists
        if 'nose_on_food' not in composite.columns:
            composite['nose_on_food'] = 0
            
        # Create new food_encounter column if it doesn't exist or is the old binary format
        if 'food_encounter' not in composite.columns:
            composite = create_food_encounter_column(composite)
            self._log_cont("Created new food_encounter column marking first nose-food contacts")
        
        # Reorder columns
        composite = enforce_column_order(composite)

        if 'assay_num' not in composite.columns:
            messagebox.showerror("Error", "Composite missing 'assay_num' column.")
            return

        last_assay = int(pd.to_numeric(composite['assay_num'], errors='coerce').fillna(0).max())
        next_assay = last_assay + 1

        data_dir = Path(self.cont_data_dir_var.get().strip())
        if not data_dir.exists() or not data_dir.is_dir():
           messagebox.showerror("Error", f"New data directory does not exist:\n{data_dir}")
           return

        files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in ('.csv', '.txt')])
        if not files:
           messagebox.showwarning("No files", f"No .csv/.txt files found in:\n{data_dir}")
           return

        # Get input mode
        input_mode = self._get_input_mode()
        if input_mode is None:
            return

        # Process files based on mode
        if input_mode == 'inference':
            file_data = self._process_files_with_inference(files)
        else:  # manual
            file_data = self._process_files_manual(files)

        if not file_data:
            self._log_cont("Import cancelled or no valid files processed.")
            return

        # Process the files
        new_rows: List[pd.DataFrame] = []
        assay_num = next_assay

        for path, metadata in file_data:
           try:
               part = parse_wide_file_to_long(
                   path=path,
                   assay_num=assay_num,
                   pc_number=metadata['pc_number'],
                   sex=metadata['sex'],
                   strain_genotype=metadata['strain_genotype'],
                   treatment=metadata['treatment'],
               )
               new_rows.append(part)
               self._log_cont(f"Parsed {path.name}: assay_num={assay_num}, rows={len(part)}")
               assay_num += 1
           except Exception as e:
               self._log_cont(f"ERROR parsing {path.name}: {e}")

        if not new_rows:
           messagebox.showwarning("Nothing appended", "No new data imported.")
           return

        updated = pd.concat([composite] + new_rows, ignore_index=True)
        try:
           updated = enforce_column_order(updated)
           save_path = Path(self.cont_save_var.get().strip())
           updated.to_csv(save_path, index=False)
           self._log_cont(f"Saved updated composite to: {save_path}")
           messagebox.showinfo("Done", f"Updated composite saved to:\n{save_path}")
        except Exception as e:
           messagebox.showerror("Save error", f"Failed to save updated composite:\n{e}")

    def _log_cont(self, msg: str):
        self.cont_log.insert("end", msg + "\n")
        self.cont_log.see("end")
        self.cont_log.update_idletasks()

def main():
   app = App()
   app.mainloop()

if __name__ == "__main__":
   main()