# NLU final project
This repository contains all the code and the data necessary to run an extractive
summarization framework which exploits several scoring strategies to achieve a 
good summary of the documents in the input dataset.

# Files in the repository
The files contains in this repository are:
- The *.py modules which contain the code for the framework implemented.
- The Extractive_summarization_2.ipynb which containing the code generating the results in the report.
- The Extractive_summarization_2.pdf which is the report of the project.

# Dataset
The dataset used is the _CNN\_dailymail 3.0.0_ downloaded from
[Huggingface](https://huggingface.co/datasets/cnn_dailymail).

# Necessary Libraries
In order to correctly run the code provided, few python modules must exist
on the device. These modules are:
- [Datasets](https://huggingface.co/datasets/cnn_dailymail): Used for downloading the CNN\_dailymail dataset
- [Pytextrank](https://pypi.org/project/pytextrank/): Used for sentence ranking.
- [Tqdm](https://pypi.org/project/tqdm/): Used for the progress bars.
- [Optuna](https://optuna.org/): Used for the optimisation part.
- [Pandas](https://pandas.pydata.org/): Used for representing the results in a compact way.
- [Spacy](https://spacy.io/): Used for tokenization, sentence similarity and sentence ranking.

It is possible to install all of them by running the following command:  
`pip3 install datasets pytextrank tqdm optuna pandas spacy`

Furthermore, in order to exploit the sentence similarity computation of spacy a suitable spacy model must be loaded.
During the development of the code the "en_core_web_md" spacy pipeline had been used. To install it run the command:  
`python -m spacy download en_core_web_md`

# Example of Usage
## Load necessary libraries
```python
from datasets import load_dataset
from Dataset import Dataset  # Import the local class
```
## Download the dataset
```python
chosen_dataset = load_dataset(huggingface_dataset_name, dataset_version)
```
Now the dataset is loaded and a dictionary is returned. The training part of the dataset will be used from now on.
## Build the data structure
```python
myDataset = Dataset()
myDataset.process_dataset(chosen_dataset['train'])
```
In this case the scores used for the summarization task will be computed immediately, to avoid
this behaviour and compute the scores in the future or in a cycle
it is possible to use this syntax:
```python
myDataset.build_dataset(chosen_dataset['train'])
```
This will populate the class structure and collect some relevant data of the provided documents.
To run the score computation it is sufficient to call the previous method without arguments:  
```python
myDataset.process_dataset()  # Performs only the sentence scoring computation
```
## Rouge score computation
```python
myDataset.rouge_computation()  # Automatically computes Rouge-2 score and prints them
```
This method will compute the Rouge-N score for each document summarized and will print the results.  
It will return a pandas dataframe with the results of the rouge-N scores for each document in the
dataset. In the dataframe the last row is reserved for the mean values.
If the flag: `getSummary=True` is passed as an argument, this method also returns the computed
summary.

## Show summaries
To observe the resulting summarization for each document it is enough to run the command:  
```python
myDataset.summarization(show=True)
```
It is possible to add also the argument `weights=myWeights` where _myWeights_ is a list of float scaling factors that 
will be applied to the different sentence scoring strategies in order to obtain a customized summary.
Use `myDataset.get_num_weights()` to retrieve the number of available scoring strategies and understand the length of the
weights list.

## Changing spacy pipeline
The _Dataset_ class allows the user to change the spacy model used during the computation. To do so, just call the method:  
`myDataset.set_spacy_pipeline('en_core_web_md')`
In this example the "en_core_web_md" is the name of the spacy model to use.
Every downloaded spacy pipeline can be used, but just some of them have the sentence similarity feature.
To check which spacy pipelines are downloaded in the system just run the command `spacy info`.  
It is important to notice that **the pytextrank pipeline will be added to the chosen spacy pipeline**
