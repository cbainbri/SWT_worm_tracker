# SWT_worm_tracker
A simple C. elegans multi worm tracker and masking for border encounters.

# Worm Tracking Toolkit

This repository contains Python tools for worm tracking, mask generation (manual or automatic with **Segment Anything**), batch processing, and track editing.

---

## Installation

This toolkit requires Python 3.9–3.12 (recommended: 3.11 for best compatibility), git, and git-lfs for managing SWT_worm_tracker and SAM model file.

## 1. git install

#### Windows
```bash
#Download git here 
https://git-scm.com/downloads/win
#or Powershell
winget install Git.Git
```
#### MacOS
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"   #installs homebrew if not already installed
brew install git
```
#### Linux
```bash
#package manager specific
sudo apt install git 
```

## 2. Clone the Repository 
```bash
git clone https://github.com/cbainbri/SWT_worm_tracker.git
cd SWT_worm_tracker
```
## 3. Installing required dependencies

Run following commands inside SWT_worm_tracker root directory

#### Windows 10/11 Powershell
```bash
winget install GitHub.GitLFS
git lfs install
pip install --upgrade pip
pip install -r requirements.txt
git lfs pull
```
####  MacOS
```bash
brew install python git-lfs tcl-tk
# IF using conda to manage environments
conda create -n wormtracker python=3.11
conda activate wormtracker
#
git lfs install
pip install --upgrade pip
pip install -r requirements.txt
git lfs pull
```
#### Linux
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-tk git-lfs libgl1   #python3-tk necessary for tkinter packaged separately on linux
# IF using conda to manage environments
conda create -n wormtracker python=3.11
conda activate wormtracker
#
git lfs install
pip install --upgrade pip
pip install -r requirements.txt
git lfs pull
```

⚠️ By default, requirements.txt installs the CPU-only PyTorch build.
If you need GPU acceleration (CUDA or ROCm), uninstall torch and reinstall the appropriate wheel from PyTorch.org


# RUNNING SCRIPTS
## Run following commands inside SWT_worm_tracker root directory


### Tracking GUI
```bash
python tracking.py
```
### Batch processing
```bash
python batch_tracking.py
```
### Track editor (with images + CSV)
```bash
python track_editor.py 
```
### Automatic mask (with SAM or PNG fallback)
```bash
python automatic_mask.py
```
### Manual mask creation
```bash
python manual_mask.py
```

### Or for MacOS and Linux - make executable 
```bash
cd SWT_worm_tracker 
chmod +x *.py

#then you can just run ./<script of choice>.py or make .desktop/.app launcher
```



# General workflow: 
## tracking.py or batch_tracking.py -> track_editor.py ->track_mask.py

This workflow is designed for separated image files for tracking analysis. Each independent script outputs sequentially modified csv files (saved back to selected input directories) which include worm ID, x,y centroid positions,nose positions, and masking logic to determine when an animal encounters food borders (or other geometric space in the environment).

## **tracking.py** or **batch_tracking.py**

Setup and background - Script performs background generation from 75 images sampled from image data. 

Threshold and QC - After background generation, this tab allows for quality control during thresholding. It takes a random 5 background-subtracted images for quality control for thresholding worms. We offer two tracking algorithms, Greedy or Hungarian depending on needs. Tracking parameters are set for worm detection optimized for our recording, but can be adjusted as needed. Blob size thresholding is most useful for filtering debris and small non moving particles that are still thresholded. Search radius is the limiting distance for the next detected frame during tracking. Track weightings are to account for animal "momentum" and correct for worm intersections and ID switching. In particular it prevents tracks from "ricocheting" after intersection. For example a 0.7 track weight means that worm behavior is 70% predictable by past behavior). Minimum track length filters tracks below a certain frame length. Hitting track will track using these parameters. 

Tracking Results- will summarize statistics and allow you to save the initial tracking result CSV. Export CSV (export simple csv for debugging) will provide initial tracking results. 


## **batch_tracking.py**

This script will launch a GUI that will allow you to select multiple experimental directories. It will then analyze each selected directory sequentially, saving tracks.csv files back to the input directories. 


## **track_editor.py**

This script is designed to view and edit tracks generated from tracking.py or batch_tracking.py. This is done by loading tracks.csv and the image directory. This will allow you to view and clean up tracks from the initial CSV. By entereing track selection mode you can delete tracks selectively or select keep tracks to delete all but the ones you have selected. Additionally you can selectively merge tracks that might have dropped do to thresholding errors, or contrast issues. When complete you can export these tracks. You will now have a finalized CSV relevant only to animal locomotion. 

## **automatic_mask.py**
This tool allows you to load an image from the image directory, and click to select features of interest for automated masking. This is usually quite robust, and draws from the sam-vit-L checkpoint file. -NOTE This is tracked via github and git pull LFS should pull the necessary checkpoint file (1-2gb). If running automatic mask gives a no SAM model error, re-run git lfs pull. The track mask function will use the current mask being generated for analysis of a csv. Alternatively it can be used post-hoc by loading a mask and appropriate csv then analyzing and storing the results.

## **track_mask.py**

This tool allows you to load a single image to manually annotate static background features by tracing. This is done by loading an image and tracing the mask. Hitting generate mask will then provide a binary mask jpg. Importantly, there is a checkbox for what the mask is detecting. In our case again it is on/off food, but essentially this checkbox just switches the binary mask logic. This tool also allows you to analyze traced masks against your cleaned up track coordinates to analyze on/off food status (or other positional status). The track mask function will use the current mask being generated for analysis of a csv. Alternatively it can be used post-hoc by loading a mask and appropriate csv then analyzing and storing the results.
This is done by loading the generated binary mask and the csv file and hitting process CSV. The new CSV will have a new column next to each worm that shows boolean logic defined by the user for on/off food. 


