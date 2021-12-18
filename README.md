# NLU final project
This repository contains all the code and the data necessary to run an extractive
summarization framework which exploits several scoring strategies to achieve a 
good summary of the documents in the input dataset.

# Dataset
The dataset used is the _CNN\_dailymail 3.0.0_ downloaded from [Huggingface](https://huggingface.co/datasets/cnn_dailymail).

# Necessary Libraries
In order to correctly run the code provided, few python modules must exist
on the device. These modules are:
- [Datasets](https://huggingface.co/datasets/cnn_dailymail): Used for downloading the CNN\_dailymail dataset
- [Pytextrank](https://pypi.org/project/pytextrank/): Used for sentence ranking.
- [Tqdm](https://pypi.org/project/tqdm/): Used for the progress bars.
- [Optuna](https://optuna.org/): Used for the optimisation part.
- [Pandas](https://pandas.pydata.org/): Used for representing the results in a compact way.

It is possible to install all of them by running the following command:  
`pip3 install datasets pytextrank tqdm optuna pandas`

# Usage
## Download the dataset
```python
from datasets import load_dataset

chosen_dataset = load_dataset(_<dataset_name>_, _<dataset_version>_)
```
