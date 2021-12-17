from datasets import load_dataset, load_metric
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

import threading

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
                            sentence_similarity = sentence.similarity(sent2)
                            if sentence_similarity is not None:
                                sent_sim[index] = sentence_similarity/mlen
                        sent2_idx += 1
                    idx += 1
                self.documents[doc_id].add_sentSimm(sent_sim)

                pbar_load.update(1)
                if i == doc_th-1:
                    break
                i += 1
        pbar_load.close()

        self.process_documents(self.documents.keys(),
                               scoreList,
                               spacyPipe=nlp)

        print('Total Processing Time: {:0.4f}[sec]'
              .format(time.time()-start_time))

    def process_documents(self, docs_id, scoreList, spacyPipe=None):
        with tqdm(total=len(docs_id)) as pbar_proc:
            for doc in docs_id:
                pbar_proc.set_description('computing scores: ')
                self.documents[doc].compute_scores(self.proper_nouns,
                                                   self.DF,
                                                   self.named_entities,
                                                   scoreList,
                                                   spacy_pipeline=spacyPipe)
                pbar_proc.update(1)
        pbar_proc.close()

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

    def spacy_to_listOfLists(self, spacy_object, lemma=False):
        list_of_sentences = []
        for sentence in spacy_object.sents:
            tokenized_sent = []
            for token in sentence:
                if not lemma:
                    token = token.text.casefold()
                else:
                    token = token.lemma_.casefold()
                tokenized_sent.append(token)
            list_of_sentences.append(tokenized_sent)
        return list_of_sentences

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

    def summarization(self, weights=[], show_scores=False):
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
                    ordered_doc += '{}'.format(sentence)
                    count += 1
                    if count != len(doc.summary.split('\n')):
                        ordered_doc += '\n'
                    if show_scores:
                        print(sentence)
                        scores = document.get_sentence(sent_id).scores
                        scores.print_total_scores()
                        print('\n')
            summarized_dataset[doc.id] = ordered_doc
        return summarized_dataset

    def rouge_computation(self, n=2, weights=[], show=False, sentences=False):
        if len(weights) == 0:
            summarization = self.summarization()
        else:
            summarization = self.summarization(weights)
        rouge_results = {}
        student_rouge = {}

        for doc_id, doc in summarization.items():
            # Split summaries in casefolded sentences
            hyp = doc.casefold()
            ref = self.documents[doc_id].summary.casefold()

            hyp_flat = hyp.split(' ')
            ref_flat = ref.split(' ')

            hyp = hyp.split('\n')
            ref = ref.split('\n')

            if len(hyp) != len(ref):
                print('Reference and hypotesys must be of same length.'
                      ' Got: Hyp {} and Ref {}'.format(len(hyp), len(ref)))
                return

            if sentences:
                print('-'*80)
                print(hyp, '\n', '-'*39, 'VS', '-'*39, '\n', ref)
                print('-'*80)

            # n-gram merging
            for summary in [ref, hyp, ref_flat, hyp_flat]:
                summary = [x.split(' ') for x in summary]
                for sentence in summary:
                    temp = []
                    i = 0
                    while i+n < len(sentence):
                        temp.append(sentence[i:i+n])
                        i += 1
                    summary[summary.index(sentence)] = temp

            # Student Rouge-N
            ref_flat_copy = ref_flat.copy()
            for ngram in hyp_flat:
                if ngram in ref_flat_copy:
                    ref_flat_copy.remove(ngram)
            match_ngram = (len(ref_flat) - len(ref_flat_copy))

            # Rouge-N
            ref_ngram_count = sum(len(i) for i in ref)
            rouge_n = match_ngram / ref_ngram_count

            # Precision -> how much of the summarization is useful
            hyp_ngram_count = sum(len(i) for i in hyp)
            rouge_precision = match_ngram / hyp_ngram_count

            # F1 score
            F1 = 2 * ((rouge_precision * rouge_n)/(rouge_precision + rouge_n))
            student_rouge[doc_id] = {'r': rouge_n, 'p': rouge_precision,
                                     'f': F1}

            # Module Rouge-N
            metric = 'rouge-%d' % n
            rouge = Rouge(metrics=[metric])
            scores = rouge.get_scores(ref, hyp, avg=True)

            # scores_l = rouge.get_scores(ref, hyp)[0]['rouge-l']
            rouge_results[doc_id] = scores[metric]

            # Count ngram matching
            '''
            summary_match_count = 0
            for i in range(len(ref)):
                ref_sentence = ref[i]
                hyp_sentence = hyp[i]
                max_sent_co_occor = 0
                for ref_ngram in ref_sentence:
                    original_ngram = ref_ngram
                    co_occor = 0
                    for hyp_ngram in hyp_sentence:
                        if ref_ngram == hyp_ngram:
                            next_idx = ref_sentence.index(ref_ngram) + 1
                            if next_idx < len(ref_sentence):
                                ref_ngram = ref_sentence[next_idx]
                                co_occor += 1
                            else:
                                ref_ngram = original_ngram
                                if co_occor+1 > max_sent_co_occor:
                                    max_sent_co_occor += co_occor+1
                                    co_occor = 0
                        else:
                            if co_occor > max_sent_co_occor:
                                max_sent_co_occor += co_occor
                            co_occor = 0
                            ref_ngram = original_ngram

                summary_match_count += max_sent_co_occor

            # Rouge-N
            ref_ngram_count = sum(len(i) for i in ref)
            rouge_n = summary_match_count/ref_ngram_count

            # Precision -> how much of the summarization is useful
            hyp_ngram_count = sum(len(i) for i in hyp)
            rouge_precision = summary_match_count / hyp_ngram_count

            # F1 score
            F1 = 2 * ((rouge_precision * rouge_n) / rouge_precision + rouge_n)

            rouge_results[doc_id] = {'Rouge-%d' % n: rouge_n,
                                     'Precision': rouge_precision,
                                     'F1-score': F1}
        pd_results = pd.DataFrame.from_dict(rouge_results, orient='index')
        '''

        if show:
            for doc_id, value in rouge_results.items():
                print('Doc ID: {}'.format(doc_id))
                print('\tRouge-{}: {:0.4f}'.format(n, value['Rouge-%d' % n]),
                      '\n\tPrecision: {:0.4f}'.format(value['Precision']),
                      '\n\tF1 Score: {:0.4f}'.format(value['F1-score']))

        pd_student = pd.DataFrame.from_dict(student_rouge, orient='index')
        pd_results = pd.DataFrame.from_dict(rouge_results, orient='index')
        print(pd_student, '\n\nModule')
        print(pd_results)
        return  # pd_results


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
    CNN_processed.process_dataset(CNN_dataset['train'], doc_th=3)
    rouge_result = CNN_processed.rouge_computation(show=False,
                                                   weights=weights,
                                                   sentences=False,
                                                   n=1)

    summaryzation = CNN_processed.summarization(weights, False)
    # for key in summaryzation:
    #     print(CNN_processed.documents[key].summary)
    #     print('* {} summary:'.format(key), '*'*(70-len(key)))
    #     print(summaryzation[key].score)

    # mean_rouge = rouge_result['Rouge'].mean()
    # mean_precision = rouge_result['Precision'].mean()
    # mean_fscore = rouge_result['F-score'].mean()

    # print('\n')
    # print('Average stats: Rouge {:0.4f}, Precision {:0.4f}, F-Score {:0.4f}'
    #      .format(mean_rouge, mean_precision, mean_fscore))
