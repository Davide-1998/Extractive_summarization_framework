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
        self.proper_noun = 0
        self.co_occour = 0
        self.sent_similarity = 0  # Semantic similarity
        self.num_val = 0
        self.TF_ISF_IDF = 0
        self.sent_rank = 0  # Text rank
        self.sent_length = 0
        self.pos_keywords = 0
        self.neg_keywords = 0
        # self.busy_path = 0
        # self.aggregate_simm = 0
        # self.word_simm_sents = 0
        # self.word_simm_par = 0
        # self.IQS = 0
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

    def set_sentRank(self, sentence, rankings):
        reconstructed_sentence = ''
        for token in sentence:
            reconstructed_sentence += token + ' '
        reconstructed_sentence = reconstructed_sentence.casefold()
        for chunk in rankings:
            if chunk in reconstructed_sentence:
                self.sent_rank += rankings[chunk]

    def set_sentLength(self, sentence, mean_length):
        # a parabola is used as an implicit treshold
        sent_len = len(sentence)
        if sent_len < 2*mean_length:  # otherwise 0
            self.sent_length = (-(1/mean_length**2))*(sent_len**2) + \
                               (2/mean_length)*sent_len

    def set_posnegScore(self, sentence, summary_tf):
        mean = sum(summary_tf.values())/len(summary_tf)
        for token in sentence:
            norm_token = token.casefold()
            if norm_token in summary_tf:
                token_freq = summary_tf[norm_token]
                if token_freq > 2*mean:
                    self.pos_keywords += 1.1*token_freq
                elif token_freq < 0.25*mean:
                    self.neg_keywords += -10*token_freq
                else:
                    self.pos_keywords += token_freq
        return

    def set_thematicWordsScore(self, sentence, sent_id, doc_tf):
        mean = sum(doc_tf.values())/len(doc_tf)
        for token in sentence:
            norm_token = token.casefold()
            if doc_tf[norm_token] > 2*mean:  # Is thematic
                self.thematic_features += doc_tf[norm_token]
        return

    def set_namedEntitiesScore(self, sentence, termFreqDict, neList):
        for token in sentence:
            norm_token = token.casefold()
            if norm_token in neList:
                self.named_entities += termFreqDict[norm_token]
        return

    def total_score(self, show=False):
        tot = 0
        for el in self.__dict__:
            tot += self.__dict__[el]
        if show:
            print('Tot Score = %0.4f' % tot)
        return tot

    def print_scores(self, total=True):
        for key in self.__dict__:
            print(key, '->', self.__dict__[key])
        if total:
            print('Tot Score = %0.4f' % self.total_score())


class Sentence():
    def __init__(self, sent, sent_id):
        self.id = sent_id
        self.loc = int(sent_id.split('_')[1])
        self.tokenized = sent
        self.scores = Scores()

    def print_Sentence(self):  # Only for debug purposes -> too much verbose
        print([self.tokenized])

    def print_Scores(self):
        self.scores.Print()

    def compute_Scores(self, attributes, reset=True):
        if reset:
            self.scores.zero()  # Reset to avoid update of loaded values

        doc_term_freq = attributes['termFrequencies']
        prop_nouns = attributes['properNouns']
        simScore = attributes['similarityScore']
        nums = attributes['numbers']
        DF_dict = attributes['documentsFrequencies']
        nameEnt = attributes['namedEntities']

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

        # Sentence Rank
        self.scores.set_sentRank(self.tokenized, attributes['sentenceRanks'])

        # Sentence length Score
        self.scores.set_sentLength(self.tokenized,
                                   attributes['meanSentenceLength'])

        # Positive-Negative keywords scoring
        self.scores.set_posnegScore(self.tokenized, attributes['highlightsTF'])

        # Thematic words
        self.scores.set_thematicWordsScore(self.tokenized, self.id,
                                           doc_term_freq)

        # Named entities score
        self.scores.set_namedEntitiesScore(self.tokenized, doc_term_freq,
                                           nameEnt)

    def get_total_score(self):
        return self.scores.total_score()

    def info(self, verbose=True):
        if verbose:
            print('Tokens in sentence: {}'.format(len(self.tokenized)))
        return len(self.tokenized)

    def text(self):
        reconstructed_sentence = ''
        for token in self.tokenized:
            reconstructed_sentence += '{} '.format(token)
        return reconstructed_sentence

    def print_scores(self, text=False):
        print('\nSentence id {}'.format(self.id))
        if text:
            reconstructed_sent = ''
            for token in self.tokenized:
                reconstructed_sent += '{} '.format(token)
            print(reconstructed_sent)
        self.scores.print_scores()


class Document():
    def __init__(self, doc, doc_id, high, load=False):
        self.id = doc_id
        self.sentences = {}
        self.highlights = None
        self.termFrequencies = {}
        self.highlightsTF = {}
        self.sentSimilarities = {}
        self.sentRanks = {}
        self.nums = []
        self.mean_length = 0
        self.tot_tokens = 0

        self.add_highlights(high)

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
            self.sentences[sent_id] = Sentence(sent, sent_id)

            # Save sentence location
            # self.sentences[sent_id].loc = len(self.sentences) - 1

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
            for token in high:
                norm_token = token.casefold()
                if norm_token not in self.highlightsTF:
                    self.highlightsTF[norm_token] = 1
                else:
                    self.highlightsTF[norm_token] += 1
            self.highlightsTF = dict(sorted(self.highlightsTF.items(),
                                     key=lambda x: x[1]))

    def add_sentRank(self, text, rank):
        if isinstance(rank, float) and isinstance(text, str):
            if text not in self.sentRanks:
                self.sentRanks[text] = rank
            else:
                print('Entry \"{}\" already exist in record, skipping.'
                      .format(text))
                # Maybe try with accumulating them and see if rouge score up
        else:
            print('Expected text and rank to be of type string and float, but '
                  'got input of type {} and {}'.format(type(text), type(rank)))
            return

    def compute_scores(self, properNouns, DF_dict, namedEntities):
        attributes = {'termFrequencies': self.termFrequencies,
                      'properNouns': properNouns,
                      'numbers': self.nums,
                      'documentsFrequencies': DF_dict,
                      'sentenceRanks': self.sentRanks,
                      'meanSentenceLength': self.mean_length,
                      'highlightsTF': self.highlightsTF,
                      'namedEntities': namedEntities}
        for sentence in self.sentences:
            attributes['similarityScore'] = self.sentSimilarities[sentence]
            self.sentences[sentence].compute_Scores(attributes)

    def add_sentSimm(self, simmDict):
        self.sentSimilarities.update(simmDict)

    def add_nums(self, numList):
        self.nums.append(numList)

    def compute_meanLength(self):
        for sent in self.sentences:
            self.mean_length += len(self.sentences[sent].tokenized)
        self.mean_length = self.mean_length/len(self.sentences)

    def get_total_scores(self, show=False):
        scores = {}
        for sentence in self.sentences.values():
            scores[sentence.id] = sentence.get_total_score()
        ordered_scores = dict(sorted(scores.items(),
                                     key=lambda x: x[1],
                                     reverse=True))
        if show:
            for el in ordered_scores:
                print(el, ' -> ', ordered_scores[el])
        return ordered_scores

    def get_sentence(self, sentence_id, text=False):
        if sentence_id not in self.sentences.keys():
            print('No sentence {} in dictionary'.format(sentence_id))
            return None
        if text:
            return self.sentences.get(sentence_id).text()
        else:
            return self.sentences.get(sentence_id)

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

    def print_scores(self):
        print('\nDocument {}'.format(self.id), '-'*(80-len(self.id)))
        for sentence in self.sentences.values():
            sentence.print_scores()


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

                '''
                for phrase in tokenized_article._.phrases:
                    print(phrase.text)
                    print(phrase.rank, '\n', phrase.text)
                    print(phrase.chunks)
                '''

                segmented_document = []
                num_tokens = []

                for sentence in tokenized_article.sents:
                    tokenized_sent = []
                    for token in sentence:
                        if not token.is_punct:  # Do not consider punctuature
                            norm_token = token.text.casefold()
                            tokenized_sent.append(token.text)  # Try with lemma
                            if token.pos_ == 'PROPN' and \
                               norm_token not in self.proper_nouns:
                                self.proper_nouns.append(norm_token)
                            if token.like_num:  # Record numerical token
                                num_tokens.append(norm_token)

                            # Frequency among documents
                            if doc_id not in self.DF:
                                self.DF[doc_id] = {}
                            if norm_token not in self.DF[doc_id]:
                                self.DF[doc_id][norm_token] = 1

                    segmented_document.append(tokenized_sent)  # Text object
                self.add_document(segmented_document, doc_id, high)
                self.documents[doc_id].add_nums(num_tokens)
                self.documents[doc_id].compute_meanLength()

                # Record sentence ranking
                for phrase in tokenized_article._.phrases:
                    if phrase.rank > 0:
                        norm_text = phrase.text.casefold()
                        self.documents[doc_id].add_sentRank(norm_text,
                                                            phrase.rank)

                # Record named entities
                for ent in tokenized_article.ents:
                    norm_ent = ent.text.casefold()
                    if norm_ent not in self.named_entities:
                        self.named_entities.append(norm_ent)

                # Similarity among sentences in same document
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
                self.documents[doc].compute_scores(self.proper_nouns,
                                                   self.DF,
                                                   self.named_entities)
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

    def print_scores(self):
        for doc in self.documents.values():
            doc.print_scores()

    def summarization(self, th=0):
        summarized_dataset = {}
        for doc in self.documents.values():
            ordered_scores = doc.get_total_scores()
            ordered_doc = ''
            document = self.documents[doc.id]
            for sent_id in ordered_scores:
                if ordered_scores[sent_id] > th:
                    sentence = document.get_sentence(sent_id, True)
                    ordered_doc += '{}\n'.format(sentence)
            summarized_dataset[doc.id] = ordered_doc
        return summarized_dataset

    def rouge_computation(self, n, th=0):
        summarization = self.summarization(th)
        for doc_id, doc in summarization.items():
            hyp = doc.split('\n')
            ref = self.documents[doc_id].highlights.split('\n')
            print(hyp, '\n*', ref, '\n\n')
        return


if __name__ == '__main__':
    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')
    CNN_processed = Dataset(name='CNN_processed.json')
    CNN_processed.process_dataset(CNN_dataset['train'])

    # CNN_summarized = CNN_processed.summarization()
    CNN_processed.rouge_computation(2)
    # CNN_processed.print_scores()
    # CNN_processed.print_scores()
    # CNN_processed.info()
    # CNN_processed.saveToDisk()
