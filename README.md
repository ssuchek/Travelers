# Travelers

This is a prototype of a Python module to process claim data from Travelers.

## Quickstart

Main running script is **categories.py**. Before run one need to install required Python packages using:

`pip install -r requirements.txt`

or using virtual environment

`.venv/bin/python -m pip install -r requirements.txt`

After installation, run the Python code:

`python categories.py`

or using virtual environment

`.venv/bin/python categories.py`

Or alternatevily one can run bash script which do all the work:

`source run.sh`

**NOTE**: before running, if one wants to update *Pentatonic_Xact_Categories_5yr_categorized* or *Pentatonic_Xact_Categories_5yr_raw.csv* files, these files should be deleted first otherwise data will be loaded directly from these files.

## Module structure

### Data 

All data files, figures and logs are stored in *data/*, *figs/* and *output/logs* folders correspondingly. 
*data/* folder has the following structure:

*data/raw/*: contains raw source files and output CSV file with data collected from raw files (*Pentatonic_Xact_Categories_5yr_raw*).
*data/output/*: contains output files:
    **categories_schema.json**: JSON schema of the category structure and key words used to categorize each claim item 
    **categories_stats**: number of items in each category and subcategory
    **most_frequent_words**: collection of the most frequent words in the claim descriptions
    **Pentatonic_Xact_Categories_5yr_categorized**: collection data categorized according to the structure defined in schema file *categories_schema.json*.

### category.py

1. Loads data from source Excel file and apply basic transformations to it (remove stop words, unnecessary claims, punctuation, fills NaN values, etc.). The output is written to CSV file.

2. Perform word frequency analysis. 

3. Distribute each claim item among different categories and subcategories according to the structure defined in schema file *data/categories_schema.json*.

3. Calculates number of items in each category and subcategory according to the structure defined in schema file *data/categories_schema.json*.

### config ini and config.py

Contain main global constants and define input & output file names.

### requirements.txt

List of Python dependencies required to run the scripts.

### utils

Folder that contain all helper Python modules:

**loader**: module used to define *ClaimDataLoader* class that includes all necessary scripts to load, transform and save transformed data.

**preprocess**: module used to define standardized preprocess operations (transformations)  on a data Python DataFrame and corresponding calculations. *preprocessor.py* defines all required classes and transformation functions while *transformations.py* contain lists of specific transformations to be applied to input data.

**logging**: module used to configure logs and define log structure. Logs are stored in *output/logs*
