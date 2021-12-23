from datasets import load_dataset
import pandas as pd
import optuna

from Dataset import Dataset


def CNN_scores_best_representation(trial, CNN_dataset, value, pipeline=None, lemma=False):
    scores = set()
    available_scores = CNN_dataset.get_num_weights(True)

    i = trial.suggest_int('Number_of_scores', 1, len(available_scores))
    for x in range(i):
        score = trial.suggest_categorical('Score-%d' % x, available_scores)
        scores.add(score)

    CNN_dataset.process_dataset(
        scoreList=scores, all_loc_scores=True, lemma=lemma, nlp=pipeline)
    results = CNN_dataset.rouge_computation(show=False)
    return results.loc['Mean'][value]


CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')
CNN_processed = Dataset('Weights_finding_optimisation_task.json')

subset_rouge = pd.DataFrame(columns=['Rouge-2', 'Precision', 'F1-score'])
subsets = {}
_num_doc = 10

for _lemma in [False, True]:
    pipe = CNN_processed.build_dataset(CNN_dataset['train'], doc_th=_num_doc,
                                       return_pipe=True, lemma=_lemma, suppress_warnings=True)
    study = optuna.create_study(direction='maximize')
    for _value in ['Precision', 'Rouge-2', 'F1-score']:
        study.optimize(lambda trial: CNN_scores_best_representation(trial,
                                                                    CNN_processed,
                                                                    _value,
                                                                    pipe,
                                                                    lemma=_lemma),
                       n_trials=1)

        subset = set()
        for key, value in study.best_params.items():
            if key != 'Number_of_scores':
                subset.add(value)

        CNN_processed.process_dataset(lemma=_lemma, scoreList=subset)
        index = _value + '_Lemma_{}'.format(_lemma)
        subsets[index] = subset
        subset_rouge.loc[index] = CNN_processed.rouge_computation(
            show=False).loc['Mean'].T

print(subset_rouge, '\n')

for key, value in subsets.items():
    print(key, '\t', value)
