# NLU final project
This repository contains all the code and the data necessary to run an extractive
summarization framework which exploits several scoring strategies to achieve a 
good summary of the documents in the input dataset.

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
To run the score computation it's enough to call the previous method without arguments:  
```python
myDataset.process_dataset()  # Performs only the sentence scoring computation
```
## Rouge score computation
```python
myDataset.rouge_computation(show=True)  # Automatically computes Rouge-2 score and prints them
```
This method will compute the Rouge-N score for each document summarized and will print the results.

## Show summaries
To observe the resultingsummarization for each document it is enough to run the command:  
```python
myDataset.summarization(show=True)
```
It is possible to add also the argument `weights=myWeights` where _myWeights_ is a list of float scaling factors that 
will be applied to the different sentence scoring strategies in order to obtain a customized summary.
Use `myDataset.get_num_weights()` to retrieve the number of available scoring strategies and understand the length of the
weights list.

## Changing spacy pipeline
The _Dataset_ class allows the user to change the spacy pipeline used during the computation. To do so, just call the method:  
`myDataset.set_spacy_pipeline('en_core_web_md')`
Every downloaded spacy pipeline can be used, but just some of them have the sentence similarity feature.
To check which spacy pipelines are downloaded in the system just run the command `spacy info`.  
It is important to notice that **the pytextrank pipeline will be added to the chosen spacy pipeline**
