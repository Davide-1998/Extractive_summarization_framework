from datasets import load_dataset
import pandas as pd
import optuna

from Dataset import Dataset

# References for dataset:
# https://huggingface.co/datasets/cnn_dailymail
# https://huggingface.co/datasets/viewer/?dataset=cnn_dailymail&config=3.0.0


def CNN_scores_best_representation(trial, CNN_dataset, value, pipeline=None,
                                   lemma=False):
    scores = set()
    available_scores = CNN_dataset.get_num_weights(True)
    i = trial.suggest_int('Number_of_scores', 1, len(available_scores))
    for x in range(i):
        score = trial.suggest_categorical('Score-%d' % x, available_scores)
        scores.add(score)
    '''
    _loc = trial.suggest_categorical('location_filter',
                                     ['ED', 'NB1', 'NB2', 'NB3', 'FaR'])
    # support dict:
    loc_dict = {'ED': [1, 0, 0, 0, 0],
                'NB1': [0, 1, 0, 0, 0],
                'NB2': [0, 0, 1, 0, 0],
                'NB3': [0, 0, 0, 1, 0],
                'FaR': [0, 0, 0, 0, 1]}
    locFilter = loc_dict.get(_loc, [1, 0, 0, 0, 0])
    '''
    CNN_dataset.process_dataset(scoreList=scores,
                                all_loc_scores=True,
                                lemma=lemma,
                                nlp=pipeline)
    results = CNN_dataset.rouge_computation(show=False)
    return results.loc['Mean'][value]


def CNN_scores_weights_task(trial, CNN_dataset, value, pipeline=None,
                            lemma=False):
    scores = [x for x in CNN_dataset.get_num_weights(True)]
    weights = [1 for x in range(len(scores))]

    for score in scores:
        idx = scores.index(score)
        weights[idx] = trial.suggest_float(score, -10.0, 10.0, step=0.5)
    '''
    _loc = trial.suggest_categorical('location_filter',
                                     ['ED', 'NB1', 'NB2', 'NB3', 'FaR'])
    # support dict:
    loc_dict = {'ED': [1, 0, 0, 0, 0],
                'NB1': [0, 1, 0, 0, 0],
                'NB2': [0, 0, 1, 0, 0],
                'NB3': [0, 0, 0, 1, 0],
                'FaR': [0, 0, 0, 0, 1]}
    locFilter = loc_dict.get(_loc, [1, 0, 0, 0, 0])
    '''
    CNN_dataset.process_dataset(scoreList=scores, all_loc_scores=True,
                                nlp=pipeline, lemma=lemma)
    results = CNN_dataset.rouge_computation(weights=weights, show=False)
    return results.loc['Mean'][value]


if __name__ == '__main__':

    # Load dataset into a variable
    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')

    # Create a new instance of the Dataset class with a custom name
    CNN_processed = Dataset(name='CNN_processed.json')
    # weights = [1 for x in range(CNN_processed.get_num_weights())]
    # weights = [8.5, 3.0, -7.0, 9.0, 1.0, 4.5, 0.0,
    #            -4.0, -4.0, 9.0, 8.5, 8.0, -7.0]

    # Populate the Dataset class with a custom number of document
    # Lemma test
    '''
    _lemma = True
    _doc_num = 100
    CNN_processed.process_dataset(CNN_dataset['train'], doc_th=_doc_num,
                                  lemma=_lemma, all_loc_scores=True,
                                  scoreList=['pos_keywords', 'co_occur'])
    CNN_processed.rouge_computation()
    '''
    # CNN_new = Dataset('New')
    # CNN_processed.process_dataset(CNN_dataset['train'], doc_th=_doc_num,
    #                               lemma=True, suppress_warnings=True)
    # CNN_processed.rouge_computation()

    '''
    # Sum of Location scoring strategies
    CNN_processed = Dataset('CNN_dataset')
    CNN_processed.process_dataset(CNN_dataset['train'], doc_th=100,
                                  all_loc_scores=True)
    CNN_processed.rouge_computation()
    '''
    '''
    # Nobata Location Treshold analysis
    CNN_processed = Dataset('CNN_dataset')
    CNN_processed.build_dataset(CNN_dataset['train'], doc_th=100)
    loc_task = pd.DataFrame(columns=['Rouge-2', 'Precision', 'F1-score'])
    for x in range(1, 21):
        CNN_processed.process_dataset(loc_th=x, all_loc_scores=True)
        loc_task.loc[x] = CNN_processed.rouge_computation().loc['Mean']
    print(loc_task)
    '''
    # CNN_processed.process_dataset(scoreList=['pos_keywords',
    #                                          'named_entities',
    #                                          'co_occur'])
    # Compute the summarization scores and apply the available weights
    # rouge_result = CNN_processed.rouge_computation(show=True,
    #                                                sentences=False,
    #                                                n=2)
    '''
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
    # Meena & Golapani initial test
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
    # '''
    # Scores weights optimization task
    _lemma = True
    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')
    CNN_processed = Dataset('Weights_finding_optimisation_task.json')
    pipe = CNN_processed.build_dataset(CNN_dataset['train'], doc_th=100,
                                       return_pipe=True, lemma=_lemma)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: CNN_scores_weights_task(trial,
                                                         CNN_processed,
                                                         'Precision',
                                                         pipe,
                                                         lemma=_lemma),
                   n_trials=300)
    print(study.best_params)
    # '''
    '''
    # Best representation optimization task
    _lemma = False
    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')
    CNN_processed = Dataset('Weights_finding_optimisation_task.json')
    pipe = CNN_processed.build_dataset(CNN_dataset['train'], doc_th=100,
                                       return_pipe=True, lemma=_lemma)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: CNN_scores_best_representation(trial,
                                                                CNN_processed,
                                                                'Precision',
                                                                pipe,
                                                                lemma=_lemma),
                   n_trials=500)
    print(study.best_params)
    '''
    # Produce the available summarization
    # summarization = CNN_processed.summarization(weights, False)
