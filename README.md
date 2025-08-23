# SWF_worm_tracker
A simple C. elegans multi worm tracker and masking for border encounters.

General workflow: tracking.py -> track_editor.py ->track_mask.py

This workflow is designed for separated image files for tracking analysis. Each independent script outputs sequentially modified csv files (saved back to selected image input directory) which include x,y centroid positions, worm ID, and masked logic to determine when an animal is on/off food (or other geometric space in the environment).

**tracking.py**
Setup and background - Script performs background generation from 75 images sampled from image data. 

Threshold and QC - After background generation, this tab allows for quality control during thresholding. It takes a random 5 background-subtracted images for quality control for thresholding worms. We offer two tracking algorithms, Greedy or Hungarian depending on needs. Tracking parameters are set for worm detection optimized for our recording, but can be adjusted as needed. Blob size thresholding is most useful for filtering debris and small non moving particles that are still thresholded. Search radius is the limiting distance for the next detected frame during tracking. Track weightings are to account for animal "momentum" and correct for worm intersections and ID switching. In particular it prevents tracks from "ricocheting" after intersection. For example a 0.7 track weight means that worm behavior is 70% predictable by past behavior). Minimum track length filters tracks below a certain frame length. Hitting track will track using these parameters. 

Tracking Results- will summarize statistics and allow you to save the initial tracking result CSV. Export CSV (export simple csv for debugging) will provide initial tracking results. 

**track_editor.py**
This script is designed to view and edit the initial track CSV. this is done by loading the initial CSV from tracking.py, and the image directory. This will allow you to view and clean up tracks from the initial CSV. By entereing track selection mode you can delete tracks selectively or select keep tracks to delete all but the ones you have selected. Additionally you can selectively merge tracks that might have dropped do to thresholding errors, or contrast issues. When complete you can export these tracks. You will now have a finalized CSV relevant only to animal locomotion. 

 **track_mask.py**
This tool allows you to load a single image to annotate static background features, in our case food border, by tracing. This is done by loading an image and tracing the mask. Hitting generate mask will then provide a binary mask jpg.Importantly, there is a checkbox for what the mask is detecting. In our case again it is on/off food, but essentially this checkbox just switches the binary mask logic. This tool also allows you to analyze traced masks against your cleaned up track coordinates to analyze on/off food status (or other positional status). This is done by loading the generated binary mask and the csv file and hitting process CSV. The new CSV will have a new column next to each worm that shows boolean logic defined by the user for on/off food. 

The track mask function is independent of other functions. For example just load a new mask and your CSV file from the track editor to re-process the CSV. Alternatively you can load a corrected CSV and process with an already created mask. This function can work post-hoc or in-line with the rest of data processing.
