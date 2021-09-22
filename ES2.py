from datasets import load_dataset
import spacy
import json
import os
from tqdm import tqdm


# Dataset is dictionary made of three key values:
# train, test, validation.
# Each of these keys lead to a dictionary having as keys:
# id, article, highlights
# Article is unsummarized, hilights is the target

# Refs for dataset:
# https://huggingface.co/datasets/cnn_dailymail
# https://huggingface.co/datasets/viewer/?dataset=cnn_dailymail&config=3.0.0


class Scores():
    def __init__(self):
        self.TF = 0
        self.sent_location = 0
        self.IDF = 0
        self.cue = 0
        # self.title = 0 Dropped
        self.proper_noun = 0
        self.co_occour = 0
        self.sent_similarity = 0
        self.num_val = 0
        # self.font_style = 0 Dropped
        self.laxycal_similarity = 0
        self.TF_ISF_IDF = 0
        self.text_rank = 0
        self.sent_length = 0
        self.pos_keywords = 0
        self.neg_keywords = 0
        self.busy_path = 0
        self.aggregate_simm = 0
        self.word_simm_sents = 0
        self.word_simm_par = 0
        self.IQS = 0
        self.thematic_features = 0
        self.named_entities = 0

    def Print(self):
        for key in self.__dict__:
            print(key, '\n')


def saveToDisk_dict_dataset(dataset_dict, pathToFile=None):
    if pathToFile is None:
        pathToFile = os.getcwd() + os.sep + 'Processed_dataset.json'
    if '.json' not in pathToFile:
        pathToFile += '.json'
    if os.path.isfile(pathToFile):
        print('File {} will be overwritten', os.path.basename(pathToFile))

    out_stream = open(pathToFile, 'w')
    json.dump(dataset_dict, out_stream)
    out_stream.close()


def loadFromDisk_dict_dataset(pathToFile=None):
    if pathToFile is None:
        pathToFile = os.getcwd() + os.sep + 'Processed_dataset.json'
    if '.json' not in pathToFile:
        pathToFile += '.json'
    if not os.path.isfile(pathToFile):
        print('File {} not found', os.path.basename(pathToFile))
        return None

    in_stream = open(pathToFile, 'r')
    loaded_dict = json.load(in_stream)
    in_stream.close()
    return loaded_dict


def process_dataset(dataset_name=None, dataset_ver=None, save=True):
    if dataset_name is None:
        print('No dataset specified')
        return

    CNN_dataset = load_dataset(dataset_name, dataset_ver)
    # train_CNN = CNN_dataset['train'].to_dict()
    # print(train_CNN[0]['highlights'])

    nlp = spacy.load('en_core_web_sm')  # Loads pipeline for english language

    # Generating tokenized structure for evaluation of some features

    # print('Tokenizing dataset:')
    processed_CNN = {}
    i = 0
    with tqdm(total=len(CNN_dataset['train'])) as pbar:
        for key in CNN_dataset['train']:
            pbar.set_description('processing dataset: ')
            id = key['id']
            processed_CNN[id] = {'sentences': {},
                                 'highlights': key['highlights']}

            tokenized_article = nlp(key['article'])
            for sentence in tokenized_article.sents:
                temp = []
                for token in sentence:
                    if not token.is_punct:
                        temp.append(token.text)
                idx = id + '_{}'.format(len(processed_CNN[id]['sentences']))
                processed_CNN[id]['sentences'][idx] = temp

                # Sentences feeded to article list are spacy sents classes

            pbar.update(1)
            if i == 2:
                break
            i += 1
    pbar.close()

    if save:
        saveToDisk_dict_dataset(processed_CNN)

    for el in processed_CNN:
        for pair in processed_CNN[el]['sentences']:
            print(pair, '\n', processed_CNN[el]['sentences'][pair], '\n')
        print(processed_CNN[el]['highlights'])


if __name__ == '__main__':
    process_dataset('cnn_dailymail', '3.0.0')
