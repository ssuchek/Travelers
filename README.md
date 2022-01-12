# Travelers

Python module to process claim data from Travelers. Includes data cleansing, basic transformations, categorisation, weights matching.

## Quickstart

First install required Python packages using:

```pip install -r requirements.txt```

or using virtual environment as described down below.

### Virtual environment

#### Create virtual environment

The following commands create virtual environment in `.venv` folder in the current folder for Python 3.8 and 3.9 respectively:

```
virtualenv .venv -p python3.8
virtualenv .venv -p python3.9
```

#### Activate, deactivate and manage virtual environment
In order to activate virtual environment, use the following command:

```
source .venv/bin/activate
```

If no specific options provided during installation, virtual environment uses global Python packages on your machine. In order to isolate virtual environment from global, modify the activate script:

- Isolate virtual environment from global: unset global `PYTHONPATH`

    ```
        ...
        VIRTUAL_ENV='/Users/user/Documents/Pentatonic/Travelers/.venv'
        if ([ "$OSTYPE" = "cygwin" ] || [ "$OSTYPE" = "msys" ]) && $(command -v cygpath &> /dev/null) ; then
            VIRTUAL_ENV=$(cygpath -u "$VIRTUAL_ENV")
        fi
        export VIRTUAL_ENV

        unset PYTHONPATH

        _OLD_VIRTUAL_PATH="$PATH"
        PATH="$VIRTUAL_ENV/bin:$PATH"
        export PATH
        ...
    ```

- Deactivate isolation during environment deactivation: add back `export PYTHONPATH` (`PYTHONPATH` depends on the Python version and OS)

    ```
    ...
    deactivate () {
        unset -f pydoc >/dev/null 2>&1 || true

        export PYTHONPATH=$ROOTSYS/lib/python3.9/site-packages/:$PYTHONPATH
    ...
    ```

Use `deactivate` command to deactivate virtual environment:

```
deactivate
```

#### Install packages inside venv

To install a single Python package, use `pip` command from the `venv` Python executor:

```
    .venv/bin/python -m pip install <package>
```

or if packages are provided in configuration file `requirements.txt`:

```
    .venv/bin/python -m pip install -r requirements.txt
```

#### Run scripts in virtual environment

```
    .venv/bin/python <script>
```

### Deleting data files before installation

Majority of output data files are NOT recreated when running Travelers module. 

Check if certain file exists and delete it:

```
FILENAME="path_to_file"
if [[ -f $FILENAME ]]; then
    rm $FILENAME
fi
```

Check if multiple files exist and delete:

```
FILES="<path_to_folder>/<pattern>"
if ls $FILES* &> /dev/null; then
    rm $FILES*
fi
```

### Running all together using BASH script

Alternatevily run bash script to run all steps of file deletion and script running at once:

`source run.sh`

## Module structure

### Data 

All data files, figures and logs are stored in *data/*, *figs/* and *output/logs* folders correspondingly. 
*data/* folder has the following structure:

#### categories_schema.json
JSON schema of the category structure and key words used to categorize each claim item 

#### zip_code_database_enterprise.csv 
CSV database of all ZIP code in United States

#### data/raw/
Contains raw source files and output CSV file with data combined from raw source files (**Pentatonic_Xact_Categories_5yr_raw.csv**).

#### data/output/
Contains output files:
**most_frequent_words**: collection of the most frequent words in the claim descriptions
**Pentatonic_Xact_Categories_5yr_preprocessed**: preprocessed raw data (removed stop words, unnecessary claims and fill NaNs)
**Pentatonic_Xact_Categories_5yr_categorized**: collection data categorized according to the structure defined in schema file *categories_schema.json*.
**categories_stats**: number of items in each category and subcategory after categorization
**category_zipcode_map**: file containing map of all ZIP codes across different categories
**subcategory_zipcode_map**: file containing map of all ZIP codes across different subcategories

### category.py

1. Loads data from source Excel file and apply basic transformations to it (remove stop words, unnecessary claims, punctuation, fills NaN values, etc.). The output is written to CSV file.

2. Perform word frequency analysis. 

3. Distribute each claim item among different categories and subcategories according to the structure defined in schema file *data/categories_schema.json*.

3. Calculates number of items in each category and subcategory according to the structure defined in schema file *data/categories_schema.json*.

4. Perform mapping of ZIP codes over different categories and subcategories.

### config ini and config.py

Contain main global constants and define input & output file names.

### requirements.txt

List of Python dependencies required to run the scripts.

### utils

Folder that contain all helper Python modules:

**loader**: module used to define *ClaimDataLoader* class that includes all necessary scripts to load, transform and save transformed data.

**preprocess**: module used to define standardized preprocess operations (transformations)  on a data Python DataFrame and corresponding calculations. *preprocessor.py* defines all required classes and transformation functions while *transformations.py* contain lists of specific transformations to be applied to input data.

**logging**: module used to configure logs and define log structure. Logs are stored in *output/logs*
