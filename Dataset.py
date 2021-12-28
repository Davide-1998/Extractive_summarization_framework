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
    '''
    This is the class responsible to store the input documents.
    Attributes
    ----------
    documents: dict
        A dictionary of Document classes, addressed by the document ID
    proper_nouns: set
        A set of tokens identified as proper nouns. It has been filled
        using tokens coming from all the input doucuments.
    named_entities: set
        Is a set containing all the tokens identified as named entities
        while parsing the input documents.
    DF: dict
        Is a dictionary containing the document frequency of the tokens
        found in the documents. For each document, if the document
        contains the token under analysis a dictionary having as key
        the token and as value 1 is added to the DF dictionary indexed
        by the document ID.
    name: str
        Is the filename under which the dataset will be saved in the
        device. May be a path, if only the name is given, the local
        path is assumed (i.e. os.path.getcwd() ).
    numerical_tokens: set
        Is the set containing all the tokens identified as numerical
        while processing the input documents.
    spacy_pipeline_name: str
        Is the name of the spacy pipeline to use. By default it is set
        as 'en_core_web_md'. If necessary download it through:
        python -m spacy download en_core_web_md
    '''

    def __init__(self, name='Processed_dataset'):
        self.documents = {}
        self.proper_nouns = set()       # For all dataset to avoid duplicates
        self.named_entities = set()     # For all dataset to avoid duplicates
        self.DF = {}                    # Dataset-wise word frequency
        self.name = name                # Name of the dataset file
        self.numerical_tokens = set()   # Enforce shared knowledge among docs
        self.spacy_pipeline_name = 'en_core_web_md'  # Minimum medium!

    def rename(self, name):
        if name is not None:
            self.name = name
        else:
            print('NoneType cannot be used as a name for the Dataset class')

    def set_spacy_pipeline(self, spacyPipeName):
        self.spacy_pipeline_name = spacyPipeName

    def add_document(self, doc, doc_id, summary, suppress_warning=False):
        '''
        This method adds the tokenized document to the Dataset class,
        in order to work the tokenized document, its plain text summary
        and the document ID must be given.

        Parameters
        ----------
        doc: list
            Is a list of tokens representing the tokenized version of
            the document.
        doc_id: str
            Is a string representing the unique identifier of the
            document to add.
        summary: str
            Is the string representing the plain text of the summary
            of the document.
        suppress_warning: bool
            Is used to wether or not notify the user that the document
            given in input already exist and for this reason it won't
            be overwritten.
        '''

        if doc_id not in self.documents:
            self.documents[doc_id] = Document(doc, doc_id, summary)
        else:
            if not suppress_warning:
                print('Key already exist, document not overwritten to preserve'
                      ' consistency')

    def add_proper_noun(self, obj):
        '''
        This method adds the given object to the proper noun set
        if it is identified as a Proper Noun by spacy.

        obj: spacy.tokens.Token
            Is the token that is requested to add to the set. A string
            may also be passed, but its truth value cannot be proven.
        '''

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
        '''
        This method allows to add a token to the numerical tokens set.

        obj: spacy.tokens.Token
            Is the token, processed by spacy, which is requested to add
            to the set. A string may also be passed, but its truth
            value cannot be checked.
        '''

        if isinstance(obj, spacy.tokens.Token):
            if obj.like_num:
                self.numerical_tokens.add(obj.text.casefold())
        elif isinstance(obj, str):
            self.numerical_tokens.add(obj.casefold())
        else:
            print('A spacy.tokens.Token or a string must be given as input')

    def add_namedEntities(self, spacyObject):
        '''
        This method allows to add the given spacy object to the
        named entities set.

        spacyObject: spacy.tokens.Span
            Is the span object from which the named entities are taken
            and added to the set.
        '''

        for ent in spacyObject.ents:
            norm_ent = ent.text.casefold()
            self.named_entities.add(norm_ent)

    def add_sentenceRank(self, doc_id, sentenceText, sentenceRank):
        '''
        This method allows to add the ranking, performed by the
        pytextrank module, to the correct dictionary in the Document
        class.

        doc_id:str
            Is the unique identifier of the document to which the
            sentenceText belongs to.
        sentenceText: str
            Is the plain text of the sentence that had been ranked.
        sentenceRank: float
            Is the result of the textrank pipeline.
        '''

        if sentenceRank > 0:
            self.documents[doc_id].add_sentRank(sentenceText, sentenceRank)

    def compute_sent_simm(self, doc_id, spacy_doc, lemma=False):
        '''
        This module exploits the sentence similarity method to get an
        evaluation of the similarity among the sentences in the document.

        doc_id: str
            Is the unique identifier of the document from which the
            sentences to score are taken.
        spacy_doc: spacy.tokens.doc.Doc
            Is the spacy object of the document.
        lemma: bool
            Is a flag used to determine whether to use the tokens or
            their lemmatized version.
        '''

        if not isinstance(spacy_doc, spacy.tokens.doc.Doc):
            print('A spacy Doc object must be given')
            return
        sent_sim = {}
        idx = 0
        for sentence in spacy_doc.sents:
            if lemma:
                sent_str = sentence.lemma_.casefold()
            else:
                sent_str = sentence.text.casefold()
            sent_id = str(doc_id) + '_{}'.format(idx)

            sent2_idx = 0
            for sent2 in spacy_doc.sents:
                if lemma:
                    sent2_text = sent2.lemma_.casefold()
                else:
                    sent2_text = sent2.text.casefold()
                if sent2_text != sent_str:
                    index = '{}:{}'.format(sent_id, sent2_idx)
                    # char-based length
                    mlen = max(len(sentence), len(sent2))
                    sentence_similarity = sentence.similarity(sent2)
                    if sentence_similarity is not None:
                        sent_sim[index] = sentence_similarity/mlen
                sent2_idx += 1
            idx += 1
        return sent_sim

    def build_dataset(self, dataset_in, doc_th=3, suppress_warnings=False,
                      save=False, savePath=None, return_pipe=False,
                      lemma=False):
        '''
        This method is used to fill the Dataset structure and store
        the meaningful tokens.

        dataset_in: dict
            Is the dictionary of the loaded dataset
            (e.g. CNN_dailymail 3.0.0).
            The dictionary must have 3 keys: id which is the unique
            identifier, article which is the document plain text and
            highlights which is the plain text of the document summary.

        doc_th: int
            Threshold of documents to process. Default to 3.
        suppress_warnings: bool
            Flags telling whether or not the user must be notified of
            incoming duplicated documents.
        save: bool
            Flags telling to save the filled dataset.
        savePath: str
            Path to the location in which the dataset must be saved.
            Used together with \'save\' parameter.
        return_pipe: bool
            Flags telling whether or not to return the spacy pipeline
            that has been loaded.
        lemma: bool
            Flags telling whether or not using the lemma of each token
            in the document.
        '''

        if len(self.documents) > 0:
            self.__init__(self.name)
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
                self.DF[doc_id] = {}

                for sentence in tokenized_article.sents:
                    tokenized_sent = []
                    for token in sentence:
                        norm_token = token.text.casefold()
                        if lemma:
                            norm_token = token.lemma_.casefold()
                        tokenized_sent.append(norm_token)

                        # Record proper nouns
                        self.add_proper_noun(norm_token)

                        # Record numerical tokens
                        self.add_numerical_token(norm_token)

                        # Frequency among documents
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
                sent_sim = self.compute_sent_simm(doc_id, tokenized_article,
                                                  lemma)
                self.documents[doc_id].add_sentSimm(sent_sim)

                pbar_load.update(1)
                if i == doc_th-1:
                    break
                i += 1
        pbar_load.close()

        if return_pipe:
            nlp.remove_pipe('textrank')
            return nlp

        print('Dataset built in {}[sec]'.format(time.time()-start_time))

        if save:
            self.save(savePath)

    def process_dataset(self, dataset_in=None, doc_th=3, save=False, loc_th=5,
                        all_loc_scores=False, locFilter=[0, 0, 0, 1, 0],
                        scoreList=[], suppress_warnings=False, savePath=None,
                        nlp=None, lemma=False, reset=True):
        '''
        This method builds and processes the dataset, meaning that it
        adds data to the Dataset class and computes the sentences'
        scoring strategies.

        dataset_in: list
            List of dictionaries containing the documents to add.
            Each entry must have: id, article and highlights.
        doc_th: int
            Maximum number of document to load. Default is 3.
        save: bool
            Flags telling whether to save or not. Path is decided using
            savePath.
        loc_th: int
            Maximum number of sentences to consider during the
            computation of Nobata et al. scoring strategy 1.
            Further informations are given in the Score.py file.
        all_loc_scores: bool
            Flags whether to take the maximum among all the sentence
            location scoring strategies or not.
        locFilter: list
            Vector of size 5 representing the Sentence location
            scoring strategies available. If used with all_loc_scores
            set to True, it doesn't have any effect.
        scoreList: list
            List of scoring strategies to compute for the sentences.
            Each element in the list is a string representing a
            scoring method. Available values are:
                "TF" -> Term Frequency
                "sent_location" -> Sentence Location
                "proper_noun" -> Proper Nouns
                "co_occur" -> Co Occurrence
                "sent_similarity" -> Sentence Similarity
                "num_val" -> Numerical Tokens
                "TF_ISF_IDF" -> Term Frequency/inverse document frequency
                "sent_rank" -> Sentence Ranking
                "sent_length" -> Sentence Length
                "pos_keywords" -> Positive Keywords
                "neg_keywords" -> Negative Keywords
                "thematic_features" -> Thematic Words
                "named_entities" -> Named Entities
            Further informations can be retrieved by the Score class
            or in the Score.py file
        suppress_warnings: bool
            Flags whether to notify or not the user that duplicated
            documents have been discarded.
        savePath: str
            Path in which save the processed Dataset.
            Default one is local one. (i.e. os.path.getcwd() )
        nlp: spacy.lang
            Is the loaded spacy language model.
            (i.e result of spacy.load( <model> ) )
        lemma: bool
            Flags whether to use the tokens or their lemma.
        reset: bool
            Flags whether to reset the scores before computing them.
        '''

        if dataset_in is not None:
            self.__init__()  # Avoids errors in successive computations
            nlp = self.build_dataset(dataset_in, doc_th,
                                     suppress_warnings,
                                     save, savePath,
                                     return_pipe=True,
                                     lemma=lemma)
        elif nlp is None:
            nlp = spacy.load(self.spacy_pipeline_name)

        start_time = time.time()
        self.process_documents(docs_id=self.documents.keys(),
                               scoreList=scoreList,
                               spacyPipe=nlp, reset=reset,
                               loc_th=loc_th,
                               loc=locFilter,
                               all_loc_scores=all_loc_scores,
                               lemma=lemma)

        print('Dataset processed in: {:0.4f}[sec]'
              .format(time.time()-start_time))
        if save:
            self.save(savePath)

    def process_documents(self, docs_id, scoreList, spacyPipe=None,
                          reset=True, loc_th=5, loc=[0, 0, 0, 1, 0],
                          all_loc_scores=False, lemma=False):
        '''
        This method computes the sentence-wise scoring strategies.

        docs_id: list of string
            List of documents' IDs to process that exist in the dataset.
        scoreList: list of string
            List of scoring strategies to use.
            Available values are:
                "TF" -> Term Frequency
                "sent_location" -> Sentence Location
                "proper_noun" -> Proper Nouns
                "co_occur" -> Co Occurrence
                "sent_similarity" -> Sentence Similarity
                "num_val" -> Numerical Tokens
                "TF_ISF_IDF" -> Term Frequency/inverse document frequency
                "sent_rank" -> Sentence Ranking
                "sent_length" -> Sentence Length
                "pos_keywords" -> Positive Keywords
                "neg_keywords" -> Negative Keywords
                "thematic_features" -> Thematic Words
                "named_entities" -> Named Entities
            Further informations can be retrieved by the Score class
            or in the Score.py file
        spacyPipe: spacy.lang
            Is the loaded spacy language model.
            (i.e result of spacy.load( <model> ) )
        reset: bool
            Flags whether to reset the scores before computing them.
        loc_th: int
            Maximum number of sentences to consider during the
            computation of Nobata et al. scoring strategy 1.
            Further informations are given in the Score.py file.
        loc: list of bool
            Vector of size 5 representing the Sentence location
            scoring strategies available. If used with all_loc_scores
            set to True, it doesn't have any effect.
        all_loc_scores: bool
            Flags whether to take the maximum among all the sentence
            location scoring strategies or not.
        lemma: bool
            Flags whether to use the tokens or their lemma.
        '''

        with tqdm(total=len(docs_id)) as pbar_proc:
            for doc in docs_id:
                pbar_proc.set_description('computing scores: ')
                document = self.documents[doc]
                document.compute_scores(self.proper_nouns, self.DF,
                                        self.named_entities, scoreList,
                                        self.numerical_tokens,
                                        spacy_pipeline=spacyPipe,
                                        _reset=reset,
                                        loc_threshold=loc_th,
                                        _all_loc=all_loc_scores,
                                        locFilter=loc,
                                        lemma=lemma)
                pbar_proc.update(1)
        pbar_proc.close()

    def spacy_to_listOfLists(self, spacy_object, lemma=False):
        '''
        This method is used to transform a spacy.token.doc.Doc into a
        list of lists representing the processed document.

        spacy_object: spacy.token.doc.Doc
            Result of processing the document using the loaded
            spacy language model.
        lemma: bool
            Flags whether to use or not the lemma of the tokens
        '''
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

    def get_num_weights(self, names=False):
        '''
        Returns the number of available scoring strategies. This
        is also the compulsory length of the list representing the
        weights for the rouge computation.

        names: bool
        Flags whether to return also the name of the scoring methods or
        not. If set to True the method returns a dictionary, else a
        list is given.
        '''

        if not names:
            return len(Scores().__dict__)
        else:
            return list(Scores().__dict__.keys())

    def summarization(self, weights=[], show_scores=False, show=False):
        '''
        Computes the summary of the document using the sentence-wise
        scores given by the process_document method.

        weights: list of float
            Weights to apply over the scoring strategies during the
            summary computation. Number of weights can be retrieved
            by using: Dataset.get_num_weights()
        show_scores: bool
            Flags whether to print the total score of each sentence.
        show: bool
            Flags whether to print the summary.
        '''

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

    def rouge_computation(self, n=2, weights=[], show=True, sentences=False):
        '''
        Computes the rouge score for each sentence in the document.

        n: int
            Chose the ngram dimension for the rouge.
            n=1 -> unigram
            n=2 -> bigram
            ...
        weights: list of float
            List of weights to apply to the scoring strategies.
            Number of weights to provide can be retrieved by using:
            Dataset.get_num_weights()
        show: bool
            Flags whether to print the results.
        sentences: bool
            Flags wheter to print the rouge score for each sentence.
        '''

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

    def save(self, pathToFile=None):
        '''
        Saves the dataset in a .json file.

        pathToFile: str
            Path in which to save the dataset.
            If None, the dataset is saved in the local one.
            (i.e. os.path.getcwd())
        '''

        if pathToFile is None:
            pathToFile = os.getcwd() + os.sep + self.name
        if '.json' not in pathToFile:
            pathToFile += '.json'
        if os.path.isfile(pathToFile):
            filename = os.path.basename(pathToFile)
            print('File \"{}\" will be overwritten'.format(filename))

        data = self.__dict__.copy()
        docs = {}
        for doc in self.documents:
            docs.update({doc: self.documents[doc].toJson()})
        data['documents'] = docs
        data['proper_nouns'] = [x for x in self.proper_nouns]
        data['named_entities'] = [x for x in self.named_entities]
        data['numerical_tokens'] = [x for x in self.numerical_tokens]

        out_stream = open(pathToFile, 'w')
        json.dump(data, out_stream, indent=4)
        out_stream.close()

    def load(self, pathToFile=None):
        '''
        Load a dataset from a .json file.

        pathToFile: str
            Path from which the .json file will be loaded. If None it
            relies on the local one. (i.e. os.path.getcwd() )
        '''

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
        filter_keys = ['documents', 'proper_nouns', 'named_entities',
                       'numerical_tokens']

        for key in loaded_dataset:
            if key in self.__dict__ and key not in filter_keys:
                self.__dict__[key] = loaded_dataset[key]

        for key in filter_keys[1:]:
            self.__dict__[key] = set(loaded_dataset[key])

        for doc_id in loaded_dataset['documents']:
            loaded_document = loaded_dataset['documents'][doc_id]
            temp_doc = Document()
            temp_doc.from_dict(loaded_document)
            self.documents[doc_id] = temp_doc

    def info(self, verbose=True):
        '''
        Prints the number of documents in the dataset and
        the file name of the dataset.

        verbose: bool
            Flags wether to print each document information.
        '''

        print('Dataset name: {}\n'
              'Documents in dataset: {}\n'
              .format(self.name, len(self.documents)))
        if verbose:
            for doc_id in self.documents:
                print('-'*80, '\nDocument ID: {}'.format(doc_id))
                self.documents[doc_id].info()

    def print_scores(self, text=False, onlyTotal=True):
        '''
        Print the document-wise scores.
        text: bool
            Flags whether or not to print the text in the sentences.
        onlyTotal: bool
            Flags whether or not to print only the total value of the
            rouge score. Set to False to have all the scores.
        '''

        if len(self.documents) > 0:
            print(self.documents)
            for doc in self.documents.values():
                doc.print_scores(_text=text, _onlyTotal=onlyTotal)
        else:
            print('No documents from which print scores')
