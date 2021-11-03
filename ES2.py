from datasets import load_dataset
import spacy
import json
import os
from tqdm import tqdm
from math import log
import pytextrank


# Dataset is dictionary made of three key values:
# train, test, validation.
# Each of these keys lead to a dictionary having as keys:
# id, article, highlights
# Article is unmarized, hilights is the target

# Refs for dataset:
# https://huggingface.co/datasets/cnn_dailymail
# https://huggingface.co/datasets/viewer/?dataset=cnn_dailymail&config=3.0.0


class Scores():
    def __init__(self):
        self.TF = 0
        self.sent_location = 0
        self.cue = 0
        # self.title = 0 Dropped
        self.proper_noun = 0
        self.co_occour = 0
        self.sent_similarity = 0  # Semantic similarity
        self.num_val = 0
        # self.font_style = 0 Dropped
        # self.lexycal_similarity = 0 Dropped due to redundancy
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

    def zero(self):  # Sets all values to 0
        for el in self.__dict__:
            if self.__dict__ != 0:  # Adds control, may make process long
                self.__dict__[el] = 0

    def set_TF(self, token_list, term_freq_dict):
        if self.TF != 0:
            print('TF value will be overwritten')
            self.TF = 0
        for token in token_list:
            self.TF += term_freq_dict[token.casefold()]

    def set_sent_location(self, sent_id, ED=False, NB1=False,
                          NB2=False, NB3=False, FaR=False):
        if [ED, NB1, NB2, NB3, FaR].count(True) != 1:
            print('Multiple or none options setted, score of 0 attributed')
            self.sent_location = 0
        else:
            # ED -> Edmundson -> mostly works for news data
            if ED:
                if sent_id < 2:
                    self.sent_location = 1
                else:
                    self.sent_location = 0

            # NB1 -> Nobata method 1
            if NB1:
                if sent_id == 0:  # First sentence
                    self.sent_location = 1
                else:
                    self.sent_location = 0

            # NB2 -> Nobata method 2
            if NB2:
                if sent_id == 0:
                    self.sent_location = 1
                else:
                    self.sent_location = 1-(sent_id * 0.01)

            # NB3 -> Nobata method 3

            # FaR -> Fattah and Ren
            if FaR:
                if sent_id < 5:
                    self.sent_location = 1 - (sent_id*1/5)
                else:
                    self.sent_location = 0

    def set_proper_noun(self, sent, prop_noun_list, term_freq_dict):
        if self.proper_noun != 0:
            print('Proper noun will be overwritten')
            self.proper_noun = 0
        for token in sent:
            if token.casefold() in prop_noun_list:
                self.proper_noun += term_freq_dict[token.casefold()]

    def set_similarity_score(self, score):
        self.sent_similarity = score

    def set_numScore(self, sent, freqDict, numList):
        for token in sent:
            if token.casefold() in numList:
                self.num_val += freqDict[token.casefold()]

    def set_TF_ISF_IDF(self, sent, TF, DF):
        num_doc = len(DF)
        TF_ISF_IDF = 0
        for token in sent:
            token = token.casefold()
            doc_has_token = 0
            for doc in DF:
                doc_has_token += DF[doc].get(token, 0)
            IDF = log(num_doc/(1 + doc_has_token))  # Avoids division by 0
            TF_ISF_IDF += TF[token]*IDF
        self.TF_ISF_IDF = TF_ISF_IDF
        return

    def total_score(self):
        tot = 0
        for el in self.__dict__:
            tot += self.__dict__[el]
        self.Print()
        print('Tot Score = %0.4f' % tot)
        return tot

    def Print(self):
        for key in self.__dict__:
            print(key, '\n')


class Sentence():
    def __init__(self, sent):
        self.loc = 0
        self.tokenized = sent
        self.scores = Scores()

    def print_Sentence(self):  # Only for debug purposes -> too much verbose
        print([self.tokenized])

    def print_Scores(self):
        self.scores.Print()

    def compute_Scores(self, doc_term_freq, prop_nouns, simScore, nums,
                       DF_dict, reset=True):
        if reset:
            self.scores.zero()  # Reset to avoid update of loaded values

        # TF score
        self.scores.set_TF(self.tokenized, doc_term_freq)

        # Sentence location score
        self.scores.set_sent_location(self.loc, FaR=True)

        # Proper noun score
        self.scores.set_proper_noun(self.tokenized, prop_nouns, doc_term_freq)

        # Similarity Score
        self.scores.set_similarity_score(simScore)

        # Numerical Score -> if number exist in sentence
        self.scores.set_numScore(self.tokenized, doc_term_freq, nums)

        # TF-IDF score
        self.scores.set_TF_ISF_IDF(self.tokenized, doc_term_freq, DF_dict)

    def info(self, verbose=True):
        if verbose:
            print('Tokens in sentence: {}'.format(len(self.tokenized)))
        return len(self.tokenized)


class Document():
    def __init__(self, doc, doc_id, high, load=False):
        self.sentences = {}
        self.highlights = None
        self.add_highlights(high)
        self.termFrequencies = {}
        self.sentSimilarities = {}
        self.nums = []
        self.tot_tokens = 0

        if not load:
            for sentence in doc:  # Must be pre-processed by spacy pipeline
                if len(sentence) > 0:
                    self.add_sentence(sentence, doc_id)
                    self.tot_tokens += len(sentence)

            for key in self.termFrequencies:
                self.termFrequencies[key] /= self.tot_tokens

    def add_sentence(self, sent, doc_id):
        if not isinstance(sent, list):
            print('Format type of input must be \'List\'')
            return
        if len(sent) == 1:
            print('Warning: singleton {} detected!'.format(sent))
        elif len(sent) == 0:  # Enforced in mathod above
            print('Input sentence not eligible for adding operation, Ignored')
            return
        else:
            # Push sentence into dictionary and organize by internal ID
            sent_id = doc_id + '_{}'.format(len(self.sentences))
            self.sentences[sent_id] = Sentence(sent)

            # Save sentence location
            self.sentences[sent_id].loc = len(self.sentences) - 1

            # Update term frequency
            for token in sent:
                token = token.casefold()
                if token not in self.termFrequencies:
                    self.termFrequencies[token] = 1
                else:
                    self.termFrequencies[token] += 1

    def add_highlights(self, high):
        if not isinstance(high, str):
            print('Input type must be \'string\'')
            return
        else:
            self.highlights = high

    def compute_scores(self, properNouns, DF_dict):
        for sentence in self.sentences:
            simScore = self.sentSimilarities[sentence]
            self.sentences[sentence].compute_Scores(self.termFrequencies,
                                                    properNouns,
                                                    simScore,
                                                    self.nums,
                                                    DF_dict)

    def add_sentSimm(self, simmDict):
        self.sentSimilarities.update(simmDict)

    def add_nums(self, numList):
        self.nums.append(numList)

    def info(self, verbose=True):
        num_sents = len(self.sentences)
        num_high = len(self.highlights.split('.'))
        av_tokens = self.tot_tokens/num_sents

        if verbose:
            print('Total tokens in document: {}\n'
                  'Average tokens per sentence: {:0.2f}\n'
                  'Total sentences: {}\n'
                  'Sentences in highlights: {}\n'
                  .format(self.tot_tokens, av_tokens, num_sents, num_high))
        return {'tot_tokens': self.tot_tokens, 'av_tokens': av_tokens,
                'num_sents': num_sents, 'num_high': num_high}


class Dataset():
    def __init__(self, name='Processed_dataset.json'):
        self.documents = {}
        self.proper_nouns = []  # For all dataset to avoid duplicates
        self.named_entities = []  # For all dataset to avoid duplicates
        self.cue_words = []
        self.DF = {}
        self.name = self.rename(name)

    def add_document(self, doc, doc_id, high):
        if doc_id not in self.documents:
            self.documents[doc_id] = Document(doc, doc_id, high)
        else:
            print('Key already exist, run stopped to preserve consistency')
            return

    def rename(self, name):
        if name is not None:
            if '.json' not in name:
                name += '.json'
            self.name = name

    '''def saveToDisk(self, pathToFile=None):
        if pathToFile is None:
            pathToFile = os.getcwd() + os.sep + self.name
        if '.json' not in pathToFile:
            pathToFile += '.json'
        if os.path.isfile(pathToFile):
            print('File {} will be overwritten', os.path.basename(pathToFile))

        out_stream = open(pathToFile, 'w')
        json.dump(self.documents, out_stream)
        out_stream.close()

    def loadFromDisk(self, pathToFile=None):
        if pathToFile is None:
            pathToFile = os.getcwd() + os.sep + self.name
        if '.json' not in pathToFile:
            pathToFile += '.json'
        if not os.path.isfile(pathToFile):
            print('File {} not found', os.path.basename(pathToFile))
            return None

        in_stream = open(pathToFile, 'r')
        loaded_dataset = json.load(in_stream)
        in_stream.close()
        self.documents = loaded_dataset'''

    def process_dataset(self, dataset_in, save=True):

        # nlp = spacy.load('en_core_web_sm')  # Loads pipeline for english
        nlp = spacy.load('en_core_web_md')  # Try this for having vectors

        # Making textrank pipe
        nlp.add_pipe('textrank', last=True)

        # Generating tokenized structure for feature evaluation

        i = 0
        with tqdm(total=len(dataset_in)) as pbar_load:
            for key in dataset_in:
                pbar_load.set_description('processing dataset: ')
                doc_id = key['id']
                high = key['highlights']

                tokenized_article = nlp(key['article'])  # Spacy object
                for phrase in tokenized_article._.phrases:
                    print(phrase.text)
                    print(phrase.rank, phrase.count)
                    print(phrase.chunks)
                segmented_document = []
                num_tokens = []

                for sentence in tokenized_article.sents:
                    tokenized_sent = []
                    for token in sentence:
                        if not token.is_punct:
                            norm_token = token.text.casefold()
                            tokenized_sent.append(token.text)  # Try with lemma
                            if token.pos_ == 'PROPN' and \
                               token.text not in self.proper_nouns:
                                self.proper_nouns.append(token.text.casefold())
                            if token.like_num:  # Record numerical token
                                num_tokens.append(token.text.casefold())

                            # Frequency among documents
                            if doc_id not in self.DF:
                                self.DF[doc_id] = {}
                            if norm_token not in self.DF[doc_id]:
                                self.DF[doc_id][norm_token] = 1

                    segmented_document.append(tokenized_sent)  # Text object
                self.add_document(segmented_document, doc_id, high)
                self.documents[doc_id].add_nums(num_tokens)

                # Record named entities
                for ent in tokenized_article.ents:
                    if ent.text not in self.named_entities:
                        self.named_entities.append(ent.text.casefold())

                # Similarity among sentences
                sent_sim = {}
                idx = 0
                for sentence in tokenized_article.sents:
                    sent_str = sentence.text.casefold()
                    sent_id = doc_id + '_{}'.format(idx)
                    sent_sim[sent_id] = 0
                    for sent2 in tokenized_article.sents:
                        if sent2.text.casefold() != sent_str:
                            sent_sim[sent_id] += sentence.similarity(sent2)
                    idx += 1
                tot_sim = sum(sent_sim.values())
                for sent in sent_sim:
                    sent_sim[sent] /= tot_sim
                self.documents[doc_id].add_sentSimm(sent_sim)

                pbar_load.update(1)
                if i == 2:
                    break
                i += 1
        pbar_load.close()
        print('\n'*3)

        with tqdm(total=len(self.documents)) as pbar_proc:
            for doc in self.documents:
                pbar_proc.set_description('Computing scores: ')
                self.documents[doc].compute_scores(self.proper_nouns, self.DF)
                pbar_proc.update(1)
        pbar_proc.close()

    def info(self, verbose=True):
        if verbose:
            print('Dataset name: {}\n'
                  'Documents in dataset: {}\n'
                  .format(self.name, len(self.documents)))
            for doc_id in self.documents:
                print('-'*80, '\nDocument ID: {}'.format(doc_id))
                self.documents[doc_id].info()


if __name__ == '__main__':
    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')
    CNN_processed = Dataset(name='CNN_processed.json')
    CNN_processed.process_dataset(CNN_dataset['train'])
    # CNN_processed.info()
    # CNN_processed.saveToDisk()
