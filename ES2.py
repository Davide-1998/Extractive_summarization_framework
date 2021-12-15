from datasets import load_dataset
import spacy
import json
import os
from tqdm import tqdm
import pytextrank
from Document import Document
from rouge import Rouge
import pandas as pd
import numpy as np
import time


# References for dataset:
# https://huggingface.co/datasets/cnn_dailymail
# https://huggingface.co/datasets/viewer/?dataset=cnn_dailymail&config=3.0.0


class Dataset():
    def __init__(self, name='Processed_dataset'):
        self.documents = {}
        self.proper_nouns = set()       # For all dataset to avoid duplicates
        self.named_entities = set()     # For all dataset to avoid duplicates
        self.cue_words = set()          # For all dataset to avoid duplicates
        self.DF = {}                    # Dataset-wise word frequency
        self.name = name                # Name of the dataset file

    def add_document(self, doc, doc_id, summary):
        if doc_id not in self.documents:
            self.documents[doc_id] = Document(doc, doc_id, summary)
        else:
            print('Key already exist, run stopped to preserve consistency')
            return

    def rename(self, name):
        if name is not None:
            self.name = name
        else:
            print('NoneType cannot be used as a name for the Dataset class')

    def save(self, pathToFile=None):
        if pathToFile is None:
            pathToFile = os.getcwd() + os.sep + self.name
        if '.json' not in pathToFile:
            pathToFile += '.json'
        if os.path.isfile(pathToFile):
            filename = os.path.basename(pathToFile)
            print('File \"{}\" will be overwritten'.format(filename))

        data = self.__dict__
        docs = {}
        for doc in self.documents:
            docs.update({doc: self.documents[doc].toJson()})
        data['documents'] = docs

        out_stream = open(pathToFile, 'w')
        json.dump(data, out_stream, indent=4)
        out_stream.close()

    def load(self, pathToFile=None):
        if pathToFile is None:
            pathToFile = os.getcwd() + os.sep + self.name
        if '.json' not in pathToFile:
            pathToFile += '.json'
        if not os.path.isfile(pathToFile):
            filename = os.path.basename(pathToFile)
            print('File \"{}\" not found'.format(filename))
            return None

        in_stream = open(pathToFile, 'r')
        loaded_dataset = json.load(in_stream)
        in_stream.close()

        for key in loaded_dataset:
            if key in self.__dict__ and key != 'documents':
                self.__dict__[key] = loaded_dataset[key]

        for doc_id in loaded_dataset['documents']:
            loaded_document = loaded_dataset['documents'][doc_id]
            temp_doc = Document()
            temp_doc.from_dict(loaded_document)
            self.documents[doc_id] = temp_doc

    def process_dataset(self, dataset_in, doc_th=3, save=True, scoreList=[]):
        start_time = time.time()
        # Medium dataset for spacy to allow sentence similarity computation
        nlp = spacy.load('en_core_web_md')

        # Making textrank pipe
        nlp.add_pipe('textrank', last=True)

        # Generating tokenized structure for feature evaluation

        i = 0
        with tqdm(total=len(dataset_in)) as pbar_load:
            for key in dataset_in:
                pbar_load.set_description('processing dataset: ')
                doc_id = str(key['id'])
                summary = key['highlights']

                tokenized_article = nlp(key['article'])  # Spacy object

                segmented_document = []
                num_tokens = []

                for sentence in tokenized_article.sents:
                    tokenized_sent = []
                    for token in sentence:
                        # if not token.is_punct:  # Do not consider punctuature
                        norm_token = token.text.casefold()  # Try .lemma_
                        tokenized_sent.append(token.text)

                        if token.pos_ == 'PROPN':
                            self.proper_nouns.add(norm_token)
                        if token.like_num:  # Record numerical token
                            num_tokens.append(norm_token)

                        # Frequency among documents
                        if doc_id not in self.DF:
                            self.DF[doc_id] = {}
                        if norm_token not in self.DF[doc_id]:
                            self.DF[doc_id][norm_token] = 1

                    segmented_document.append(tokenized_sent)  # Text object
                self.add_document(segmented_document, doc_id, summary)
                self.documents[doc_id].add_nums(num_tokens)
                self.documents[doc_id].compute_meanLength()

                # Record sentence ranking
                for phrase in tokenized_article._.phrases:
                    norm_text = phrase.text.casefold()
                    self.add_sentenceRank(doc_id, norm_text, phrase.rank)

                # Record named entities
                self.add_namedEntities(tokenized_article)

                # Similarity among sentences in same document
                sent_sim = {}
                idx = 0
                for sentence in tokenized_article.sents:
                    sent_str = sentence.text.casefold()
                    sent_id = str(doc_id) + '_{}'.format(idx)

                    sent2_idx = 0
                    for sent2 in tokenized_article.sents:
                        if sent2.text.casefold() != sent_str:
                            index = '{}:{}'.format(sent_id, sent2_idx)
                            # char-based length
                            mlen = max(len(sentence), len(sent2))
                            sent_sim[index] = sentence.similarity(sent2)/mlen
                        sent2_idx += 1
                    idx += 1
                self.documents[doc_id].add_sentSimm(sent_sim)

                pbar_load.update(1)
                if i == doc_th-1:
                    break
                i += 1
        pbar_load.close()

        with tqdm(total=len(self.documents)) as pbar_proc:
            for doc in self.documents:
                pbar_proc.set_description('computing scores: ')
                self.documents[doc].compute_scores(self.proper_nouns,
                                                   self.DF,
                                                   self.named_entities,
                                                   scoreList)
                pbar_proc.update(1)
        pbar_proc.close()

        print('Total Processing Time: {}'.format(time.time()-start_time))

    def add_proper_noun(self, obj):
        if isinstance(obj, spacy.tokens.Token):
            if obj.pos_ == 'PROPN':
                self.proper_nouns.add(obj.text.casefold())
            else:
                return
        elif isinstance(obj, str):
            self.proper_nouns.add(obj.casefold())
        else:
            print('A spacy.tokens.Token or a string must be passed as input')

    def add_namedEntities(self, spacyObject):
        for ent in spacyObject.ents:
            norm_ent = ent.text.casefold()
            self.named_entities.add(norm_ent)

    def add_sentenceRank(self, doc_id, sentenceText, sentenceRank):
        if sentenceRank > 0:
            self.documents[doc_id].add_sentRank(sentenceText, sentenceRank)

    def info(self, verbose=True):
        if verbose:
            print('Dataset name: {}\n'
                  'Documents in dataset: {}\n'
                  .format(self.name, len(self.documents)))
            for doc_id in self.documents:
                print('-'*80, '\nDocument ID: {}'.format(doc_id))
                self.documents[doc_id].info()

    def print_scores(self, text=False, onlyTotal=True):
        if len(self.documents) > 0:
            print(self.documents)
            for doc in self.documents.values():
                doc.print_scores(_text=text, _onlyTotal=onlyTotal)
        else:
            print('No documents from which print scores')

    def summarization(self, weights=[]):
        summarized_dataset = {}
        for doc in self.documents.values():
            if len(weights) == 0:
                ordered_scores = doc.get_total_scores()
            else:
                ordered_scores = doc.get_weighted_total_scores(weights)
            ordered_doc = ''
            document = self.documents[str(doc.id)]
            count = 0
            for sent_id in ordered_scores:
                # Take same number of sentences as the reference
                if count < len(doc.summary.split('\n')):
                    sentence = document.get_sentence(sent_id, True)
                    ordered_doc += '{}\n'.format(sentence)
                    count += 1
            summarized_dataset[doc.id] = ordered_doc
        return summarized_dataset

    def rouge_computation(self, n=2, weights=[], show=False, sentences=False):
        if len(weights) == 0:
            summarization = self.summarization()
        else:
            summarization = self.summarization(weights)
        rouge_results = {}

        for doc_id, doc in summarization.items():
            # Split summaries in sentences
            hyp_rouge = doc
            ref_rouge = self.documents[doc_id].summary

            metric = 'rouge-%d' % n
            rouge = Rouge(metrics=[metric])
            scores = rouge.get_scores(ref_rouge, hyp_rouge)[0][metric]
            rouge_results[doc_id] = [scores['r'], scores['p'], scores['f']]

        if show:
            for doc_id, value in rouge_results.items():
                print('Doc ID: {}'.format(doc_id))
                print('\tRouge-{}: {:0.4f}'.format(n, value[0]),
                      '\n\tPrecision: {:0.4f}'.format(value[1]),
                      '\n\tF-score: {:0.4f}'.format(value[2]))

        pd_results = pd.DataFrame.from_dict(rouge_results, orient='index',
                                            columns=['Rouge', 'Precision',
                                                     'F-score'])
        return pd_results


if __name__ == '__main__':

    test_train = [{'id': 0,
                   'article': 'Two cats where found near the old '
                              'tree beside the river.\nLocal authorities '
                              'are ashamed: "Not even in the films" said '
                              'the chief officer.\n'
                              'New developments soon',
                   'highlights': 'Two cats where found near the old tree.\n'
                                 'New developments soon.'
                   }]

    weights = np.ones(14)

    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')
    CNN_processed = Dataset(name='CNN_processed.json')
    CNN_processed.process_dataset(CNN_dataset['train'], doc_th=1)
    rouge_result = CNN_processed.rouge_computation(show=False, weights=weights)

    # summaryzation = CNN_processed.summarization(weights)
    # for key in summaryzation:
    #     print(CNN_processed.documents[key].summary)
    #     print('* {} summary:'.format(key), '*'*(70-len(key)))
    #     print(summaryzation[key])

    mean_rouge = rouge_result['Rouge'].mean()
    mean_precision = rouge_result['Precision'].mean()
    mean_fscore = rouge_result['F-score'].mean()

    print('\n')
    print('Average stats: Rouge {:0.4f}, Precision {:0.4f}, F-Score {:0.4f}'
          .format(mean_rouge, mean_precision, mean_fscore))
