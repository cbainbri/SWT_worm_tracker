Merge_files.py allows you to take multiple CSV outputs from the analysis pipeline and make a composite analysis file. 

New analysis- This tab will initialize a new composite file to aggregate experiments. It will ask for meta-data about the experiment and cycle through CSVsplaced in the analyze folder. It will output a composite file in the Analysis root folder where the merge_files.py is kept.

Continue Analysis- This tab will load the existing composite file and continue to add additional experimental files to the pre-existing composite data file. 
useful when you need to load and add additional csvs to particular experiments. 

Proposed workflow:

New analysis - place csvs from a particular experiment into analysis folder - initilize the new composite file and fill out meta data. 

If adding experiments- clear analysis folder and add new experiment CSVs - Select Continue analysis - load existing composite file (file path can be selected) -
pull new CSVs to add.
