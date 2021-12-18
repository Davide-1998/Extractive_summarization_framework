from datasets import load_dataset
import pandas as pd
import optuna

from Dataset import Dataset

# References for dataset:
# https://huggingface.co/datasets/cnn_dailymail
# https://huggingface.co/datasets/viewer/?dataset=cnn_dailymail&config=3.0.0


if __name__ == '__main__':

    scores = []

    # Load dataset into a variable
    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')

    # Create a new instance of the Dataset class with a custom name
    CNN_processed = Dataset(name='CNN_processed.json')
    weights = [1 for x in range(CNN_processed.get_num_weights())]

    # Populate the Dataset class with a custom number of document
    CNN_processed.process_dataset(CNN_dataset['train'], doc_th=100,
                                  scoreList=scores)
    # CNN_processed.print_scores(onlyTotal=False)

    # Compute the summarization scores and apply the available weights
    rouge_result = CNN_processed.rouge_computation(show=True,
                                                   weights=weights,
                                                   sentences=False,
                                                   n=2)
    # Meena & Gopalani environment
    MG_scores = {
            'comb1': ['TF_IDF', 'Co_occurrence', 'Sentence_length'],
            'comb2': ['Co_occurrence', 'Sentence_length', 'Sentence_location'],
            'comb3': ['TF_IDF', 'Co_occurrence', 'Sentence_length',
                      'Sentence_location'],
            'comb4': ['Sentence_length', 'Sentence_location', 'Named_entities',
                      'Positive', 'Proper_noun'],
            'comb5': ['Co_occurrence', 'Sentence_length', 'Sentence_location',
                      'Named_entities', 'Positive', 'Proper_noun'],
            'comb6': ['TF_IDF', 'Co_occurrence', 'Sentence_length',
                      'Sentence_location', 'Named_entities', 'Positive',
                      'Negative', 'Sentence_rank'],
            'comb7': ['TF_IDF', 'Co_occurrence', 'Sentence_length',
                      'Sentence_location', 'Named_entities', 'Positive',
                      'Negative']}

    '''  # Meena & Golapani initial test
    MG_test = Dataset(name='MG_test_dataset.json')
    MG_num_docs = 100
    MG_test.build_dataset(CNN_dataset['train'], MG_num_docs)
    MG_results = pd.DataFrame()
    for comb, scores in MG_scores.items():
        MG_test.process_dataset(scoreList=scores)
        MG_rouge = MG_test.rouge_computation()
        MG_results = pd.concat([MG_results, MG_rouge.loc['Mean']], axis=1,
                               ignore_index=True)
    print(MG_results.T)
    '''

    # MG optimisation
    MG_test = Dataset(name='MG_test_dataset.json')
    MG_num_docs = 100
    MG_test.build_dataset(CNN_dataset['train'], MG_num_docs)
    MG_results = pd.DataFrame()


    # Produce the available summarization
    # summarization = CNN_processed.summarization(weights, False)
