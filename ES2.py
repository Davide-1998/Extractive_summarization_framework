from datasets import load_dataset
import pandas as pd
import optuna

from Dataset import Dataset

# References for dataset:
# https://huggingface.co/datasets/cnn_dailymail
# https://huggingface.co/datasets/viewer/?dataset=cnn_dailymail&config=3.0.0


def MG_optimisation(trial, MG_dataset, comb):
    _weights = [0 for x in range(MG_dataset.get_num_weights())]

    # Trials
    co_occ = trial.suggest_float(comb[0], 1.0, 10.0, step=0.1)
    sent_len = trial.suggest_float(comb[1], 1.0, 10.0, step=0.1)
    sent_loc = trial.suggest_float(comb[2], 1.0, 10.0, step=0.1)
    n_ent = trial.suggest_float(comb[3], 1.0, 10.0, step=0.1)
    pos = trial.suggest_float(comb[4], 1.0, 10.0, step=0.1)
    prop_n = trial.suggest_float(comb[5], 1.0, 10.0, step=0.1)
    _loc = trial.suggest_categorical('location_filter',
                                     ['ED', 'NB1', 'NB2', 'NB3', 'FaR'])
    # support dict:
    loc_dict = {'ED': [1, 0, 0, 0, 0],
                'NB1': [0, 1, 0, 0, 0],
                'NB2': [0, 0, 1, 0, 0],
                'NB3': [0, 0, 0, 1, 0],
                'FaR': [0, 0, 0, 0, 1]}

    _weights[3] = co_occ
    _weights[8] = sent_len
    _weights[1] = sent_loc
    loc = loc_dict.get(_loc, [1, 0, 0, 0, 0])
    _weights[12] = n_ent
    _weights[9] = pos
    _weights[2] = prop_n

    MG_dataset.process_dataset(doc_th=100, scoreList=comb, locFilter=loc)
    results = MG_dataset.rouge_computation(weights=_weights)
    return results.loc['Mean']['Precision']


def CNN_scores_optimisation(trial, CNN_processed):
    scores = set()
    available_scores = CNN_processed.get_num_weights(True)
    i = trial.suggest_int('Number_of_scores', 1, len(available_scores))
    for x in range(i):
        score = trial.suggest_categorical('Score-%d' % x, available_scores)
        scores.add(score)

    CNN_processed.process_dataset(scoreList=scores)
    results = CNN_processed.rouge_computation()
    return results.loc['Mean']['F1-score']


if __name__ == '__main__':

    # Load dataset into a variable
    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')

    # Create a new instance of the Dataset class with a custom name
    CNN_processed = Dataset(name='CNN_processed.json')
    weights = [1 for x in range(CNN_processed.get_num_weights())]

    # Populate the Dataset class with a custom number of document
    CNN_processed.build_dataset(CNN_dataset['train'], doc_th=100)
    CNN_processed.process_dataset(scoreList=['pos_keywords',
                                             'named_entities',
                                             'co_occur'])
    # Compute the summarization scores and apply the available weights
    rouge_result = CNN_processed.rouge_computation(show=True,
                                                   sentences=False,
                                                   n=2)
    # Meena & Gopalani environment
    MG_scores = {
            'comb1': ['TF_ISF_IDF', 'co_occur', 'sent_length'],
            'comb2': ['co_occur', 'sent_length', 'sent_location'],
            'comb3': ['TF_ISF_IDF', 'co_occur', 'sent_length',
                      'sent_location'],
            'comb4': ['sent_length', 'sent_location', 'named_entities',
                      'pos_keywords', 'proper_noun'],
            'comb5': ['co_occur', 'sent_length', 'sent_location',
                      'named_entities', 'pos_keywords', 'proper_noun'],
            'comb6': ['TF_ISF_IDF', 'co_occur', 'sent_length',
                      'sent_location', 'named_entities', 'pos_keywords',
                      'neg_keywords', 'sent_rank'],
            'comb7': ['TF_ISF_IDF', 'co_occur', 'sent_length',
                      'sent_location', 'named_entities', 'pos_keywords',
                      'neg_keywords']}

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
    '''
    # MG optimisation using optuna
    MG_test = Dataset(name='MG_test_dataset.json')
    MG_num_docs = 100
    MG_test.build_dataset(CNN_dataset['train'], MG_num_docs)

    study_comb = MG_scores['comb5']
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: MG_optimisation(trial, MG_test, study_comb),
                   n_trials=300)
    print(study.best_params)
    '''

    # All scores optimizations
    '''
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: CNN_scores_optimisation(trial, CNN_processed),
                   n_trials=300)
    print(study.best_params)
    '''
    # Produce the available summarization
    # summarization = CNN_processed.summarization(weights, False)
