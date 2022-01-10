# Melanoma Tumor Pathway Mutation Analysis
The code provided in this repository is the code used to develop and analyze algorithms to predict melanoma response to ICI therapy.
This readme consists of two parts. The first part is a description of how to reproduce the figures from saved results produced during the project.
The second part is a description of how to run the algorithms on new datasets.

Machine and Python Version: The code was tested and run on a Windows 10 machine version 10.0.19042 build 19042, Python 3.8.8

Part 1: Reproducing Figures
This section details how to produce the raw figures which were used in the paper.

Raw data can be found in the 00_Data folder.

Step 1: Prepare the data on your local machine or server.
In order to run the code, pickled files must be created using the raw data.

Run this code:
```python
from handle_data.py import *
process_data_main()
```
This will create pickled files of the data that are used in later steps.

Step 2: Import the python file containing the figure code.

```python
from make_plots.py import *
```

Step 3: Run the main function to repopulate plots into 30_Figures and Figures.

```python
make_plots_main()
```

This will recreate the raw figures used in the paper.


Part 2: Reproducing Results

The following sections detail how to use and rerun the code, either to reproduce the results in part 1 or to use the algorithms on new datasets

Data and Data preparation: Datasets should consist of two dataframes. The first file contains the patient clinical information, with individual patients as rows and including a BR score column. BR scores should be converted into a 1 for response or 0 for non-response to ICI treatment. The second file contains a file of patient mutation data. Patients are rows, and each gene is a column. Each entry in the dataframe is a boolean indicating presence or absence of a mutation in that gene (python can also handle this information as 0 or 1).

When preparing the data, if running these algorithms across multiple datasets, you must first make sure the genes in the second dataframe containing patient mutation data are the same across all datasets. The algorithms will return an error if you attempt to train on a dataset and validate/test on a separate dataset if the genes are not shared.

Preparing Data:

This section is for preparing the data used in this study. Run steps 1 and 2 once when setting up on your local machine or server, even if you are not planning on using the original data in your research.

Step 1: import the python file with the data handling functions
```python
from handle_data.py import *
```

Step 2: run the main function which will prepare and pickle the data for later use.
```python
process_data_main()
```

Step 3: Preparing your own data
All datasets should have the exact same genes if they are trained and validated as a set, or else the algorithm code will run into errors. There is no specific function to do this, but you can use the load_save_data() function in handle_data.py as a reference point. Once you have the data processed, you can pickle the data using this function found in make_plots.py:

```python
save_pickle()
```
You can then load the data back into CSV or dictionaries using this function:

```python
load_pickle()
```
See these functions in handle_data.py for details on use.

Two other potentially useful functions for processing data:

The first is a simple function that can read in a CSV generated in certain steps of the algorithms that contains a saved dictionary as a CSV.

```python
read_csv_dictionar()
```
This is used to read in dictionaries saved during algorithm use.

The second function is used to create pathway dictionaries (dictionaries of pathways as keys and genes as entries) which have been selected based on two different datasets. This is used when comparing a new dataset to an old dataset's pathway genes.

```python
new_pathway_test_intersection()
```

Once the data is prepared, load in your data and you can proceed to the algorithm section.

Feature Selection Algorithms:

This section details how to run the feature selection algorithms.

Step 1: import the python file with the feature selection algorithms
```python
from feature_selection_pathway_mutation_load.py import *
```
Step 2: Run the desired algorithm. For the inputs, please see function documentation in the above .py file

For Greedy Forward Selection:
```python
multiple_pathway_forward_selector()
```
Random Forward Selection:
```python
multiple_pathway_forward_selector(probability = True)
```

Genetic Algorithm:
```python
genetic_algorithm()
```

Outputs for all feature selection based algorithms can be a csv or dataframe (see documentation in the python file for details). Outputs will be scores for the pathway and a separate dictionary or csv of the genes used to produce that score.

Tree Based Algorithms:

Step 1: Load in tree based algorithms:
```python
from decision_trees.py import *
```

Step 2: Run tree based methods:

Gradient Boosting:
```python
gradient_boosting_multiple_pathway()
```

Random Forest:
```python
random_forest_multiple_pathway
```

Outputs can be a dataframe or CSV file (see documentation in the .py file for details)

Neural Network Based Methods:

Step 1: Import the Neural Network functions
```python
neural_network.py
```

Step 2: Run Neural Network based methods

Forward Neural Network:
```python
fnn_multiple_pathways()
```

Long Short-Term Memory Neural Network:
```python
lstm_multiple_pathways()
```

Outputs a dataframe of the scores for the neural networks. If run using only a single pathway, can also save models. See documentation for details.

Step 3: Survival Analysis

To calculate the proportional hazards, use this function:

```python
calc_proportional_hazard()
```

in the file testing_survival_robustness_analysis.py. See the file for details on the function.
