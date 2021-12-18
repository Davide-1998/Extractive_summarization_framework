import spacy
import json
import os
from tqdm import tqdm
import pytextrank
from Document import Document
from Scores import Scores
import pandas as pd
import time


def ngram_extraction(n, plain_text):
    plain_text = plain_text.casefold()
    plain_text = plain_text.split(' ')
    tokenized_text = []
    i = 0
    while i+n < len(plain_text):
        ngram = ''
        for token in plain_text[i:i+n]:
            ngram += '{} '.format(token)
        tokenized_text.append(ngram.strip())
        i += 1
    return tokenized_text


class Dataset():
    def __init__(self, name='Processed_dataset'):
        self.documents = {}
        self.proper_nouns = set()       # For all dataset to avoid duplicates
        self.named_entities = set()     # For all dataset to avoid duplicates
        # self.cue_words = set()          # For all dataset to avoid duplicates
        self.DF = {}                    # Dataset-wise word frequency
        self.name = name                # Name of the dataset file
        self.numerical_tokens = set()   # Enforce shared knowledge among docs
        self.spacy_pipeline_name = 'en_core_web_md'  # Minimum medium!

    def add_document(self, doc, doc_id, summary, suppress_warning=False):
        if doc_id not in self.documents:
            self.documents[doc_id] = Document(doc, doc_id, summary)
        else:
            if not suppress_warning:
                print('Key already exist, document not overwritten to preserve'
                      ' consistency')
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

    def build_dataset(self, dataset_in, doc_th=3, suppress_warnings=False,
                      save=False, savePath=None, return_pipe=False):
        start_time = time.time()
        # Medium dataset for spacy to allow sentence similarity computation
        nlp = spacy.load(self.spacy_pipeline_name)

        # Adding textrank pipe
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

                for sentence in tokenized_article.sents:
                    tokenized_sent = []
                    for token in sentence:
                        norm_token = token.text.casefold()  # Try .lemma_
                        tokenized_sent.append(token.text)

                        # Record proper nouns
                        self.add_proper_noun(token)

                        # Record numerical tokens
                        self.add_numerical_token(token)

                        # Frequency among documents
                        if doc_id not in self.DF:
                            self.DF[doc_id] = {}
                        if norm_token not in self.DF[doc_id]:
                            self.DF[doc_id][norm_token] = 1

                    segmented_document.append(tokenized_sent)  # Text object
                self.add_document(segmented_document, doc_id,
                                  summary, suppress_warnings)
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

        if return_pipe:
            return nlp

        print('Dataset built in {}[sec]'.format(time.time()-start_time))

        if save:
            self.save(savePath)

    def process_dataset(self, dataset_in=None, doc_th=3, save=False,
                        scoreList=[], suppress_warnings=False, savePath=None):

        if dataset_in is not None:
            nlp = self.build_dataset(dataset_in, doc_th,
                                     suppress_warnings,
                                     return_pipe=True)  # Saves time
        else:
            nlp = spacy.load(self.spacy_pipeline_name)

        start_time = time.time()
        self.process_documents(self.documents.keys(),
                               scoreList,
                               spacyPipe=nlp)

        print('Total Processing Time: {:0.4f}[sec]'
              .format(time.time()-start_time))
        if save:
            self.save(savePath)

    def process_documents(self, docs_id, scoreList, spacyPipe=None,
                          reset=True):
        with tqdm(total=len(docs_id)) as pbar_proc:
            for doc in docs_id:
                pbar_proc.set_description('computing scores: ')
                document = self.documents[doc]
                document.compute_scores(self.proper_nouns, self.DF,
                                        self.named_entities, scoreList,
                                        self.numerical_tokens,
                                        spacy_pipeline=spacyPipe,
                                        _reset=reset)
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
            print('A spacy.tokens.Token or a string must be given as input')

    def add_numerical_token(self, obj):
        if isinstance(obj, spacy.tokens.Token):
            if obj.like_num:
                self.numerical_tokens.add(obj.text.casefold())
        elif isinstance(obj, str):
            self.numerical_tokens.add(obj.casefold())
        else:
            print('A spacy.tokens.Token or a string must be given as input')

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

    def get_num_weights(self):
        return len(Scores().__dict__)

    def summarization(self, weights=[], show_scores=False, show=False):
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
            if show:
                for key, summary in summarized_dataset.items():
                    print('***{}***:\n{}'.format(key, summary))
        return summarized_dataset

    def rouge_computation(self, n=2, weights=[], show=False, sentences=False):
        if len(weights) == 0:
            summarization = self.summarization()
        else:
            summarization = self.summarization(weights)
        rouge_results = {}

        for doc_id, doc in summarization.items():
            # Split summaries in casefolded sentences
            hyp = doc.casefold()
            ref = self.documents[doc_id].summary.casefold()

            hyp = ngram_extraction(n, hyp)
            ref = ngram_extraction(n, ref)

            if sentences:
                print('-'*80)
                print(hyp, '\n', '-'*39, 'VS', '-'*39, '\n', ref)
                print('-'*80)

            # Ngram grouping
            match_ngram = 0
            hyp_copy = hyp.copy()
            for ngram in ref:
                if ngram in hyp_copy:
                    match_ngram += 1
                    hyp_copy.remove(ngram)

            # Rouge-N
            ref_ngram_count = len(ref)
            rouge_n = match_ngram / ref_ngram_count

            # Precision -> how much of the summarization is useful
            hyp_ngram_count = len(hyp)
            rouge_precision = match_ngram / hyp_ngram_count

            # F1 score
            numerator = rouge_precision * rouge_n
            denominator = rouge_precision + rouge_n
            if denominator != 0:
                F1 = 2 * (numerator / denominator)
            else:
                F1 = 0
            rouge_results[doc_id] = {'Rouge-%d' % n: rouge_n,
                                     'Precision': rouge_precision,
                                     'F1-score': F1}

        pd_results = pd.DataFrame.from_dict(rouge_results, orient='index')
        pd_results.loc['Mean'] = pd_results.mean()
        if show:
            print(pd_results)
        return pd_results
