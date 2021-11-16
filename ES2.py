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
# Article is unsummarized, hilights is the target

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
            self.TF += term_freq_dict.get(token.casefold(), 0)
        # self.TF /= len(token_list)

    def set_sent_location(self, sent_id, sent_len, th=5,
                          loc_scores=[0, 0, 0, 0, 1]):
        [ED, NB1, NB2, NB3, FaR] = loc_scores
        if [ED, NB1, NB2, NB3, FaR].count(1) != 1:
            print('Multiple or none options setted, score of 0 attributed')
            self.sent_location = 0
        else:
            sent_id = int(sent_id.split('_')[1])
            # ED -> Edmundson -> mostly works for news data
            if ED:
                if sent_id < 2:
                    self.sent_location = 1
                else:
                    self.sent_location = 0

            # NB1 -> Nobata method 1
            if NB1:
                if sent_id < th:  # First sentence
                    self.sent_location = 1

            # NB2 -> Nobata method 2
            if NB2:
                self.sent_location = 1/(sent_id + 1)

            # NB3 -> Nobata method 3
            if NB3:
                sent_id += 1  # Avoids division by 0
                self.sent_location = max(1/sent_id, 1/(sent_len-sent_id+1))

            # FaR -> Fattah and Ren
            if FaR:
                if sent_id < 5:
                    self.sent_location = 1 - (sent_id*1/5)
                else:
                    self.sent_location = 0

    def set_proper_noun(self, sentence, prop_noun_list, tf_dict):
        # Nobata et al 2001
        if self.proper_noun != 0:
            print('Proper noun will be overwritten')
            self.proper_noun = 0
        for token in sentence:
            if token.casefold() in prop_noun_list:
                self.proper_noun += tf_dict[token.casefold()]

    def set_co_occour(self, tokenized_sentence, summary, tf_dict):
        co_occurrence = 0
        summary = summary.casefold()
        for token in tokenized_sentence:
            norm_token = token.casefold()
            co_occurrence = summary.count(norm_token)
            co_occurrence *= tf_dict[norm_token]
            self.co_occour += co_occurrence/len(tokenized_sentence)
            # Division avoids bias given by sentence length

    def set_similarity_score(self, sent_id, score):
        # Fattah & Ren 2009
        for key in score.keys():
            if sent_id == key.split(':')[0]:
                # Cumulative sum
                self.sent_similarity += score[key]

    def set_numScore(self, sent, freqDict, numList):
        # Fattah & Ren 2009
        count = 0
        for token in sent:
            if token.casefold() in numList:
                count += 1
        self.num_val = count/len(sent)  # Token-wise length

    def set_TF_ISF_IDF(self, sent, TF, DF):
        # Nobata et al 2001
        num_doc = len(DF)
        TF_ISF_IDF = 0
        for token in sent:
            token = token.casefold()
            doc_has_token = 0
            for doc in DF:
                doc_has_token += DF[doc].get(token, 0)
            IDF = log(num_doc/(doc_has_token))
            TF_ISF_IDF += (TF[token]/1+TF[token])*IDF  # Token-wise
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
            self.sent_length = ((-1/mean_length**2))*(sent_len**2) + \
                               ((2/mean_length)*sent_len)

    def set_posnegScore(self, sentence, summary_tf, highlightsOC):
        for token in sentence:
            if token in highlightsOC:
                self.pos_keywords += sentence.count(token)*highlightsOC[token]
        self.pos_keywords /= len(sentence)

    def set_thematicWordsScore(self, sentence, sent_id, doc_tf):
        mean = sum(doc_tf.values())/len(doc_tf)
        for token in sentence:
            norm_token = token.casefold()
            if doc_tf[norm_token] > 2*mean:  # Is thematic
                self.thematic_features += doc_tf[norm_token]

    def set_namedEntitiesScore(self, sentence, termFreqDict, neList):
        for token in sentence:
            norm_token = token.casefold()
            if norm_token in neList:
                self.named_entities += termFreqDict[norm_token]
        return

    def get_total(self, show=False, getVal=True):
        if show:
            print('Tot Score = %0.4f' % self.get_total())
        if getVal:
            return sum(self.__dict__.values())

    def print_total_scores(self, detail=True, total=True):
        if detail:
            for key in self.__dict__:
                print(key, '->', self.__dict__[key])
        if total:
            self.get_total(show=True, getVal=False)

    def toJson(self):
        return self.__dict__


class Sentence():
    def __init__(self, sent, sent_id):
        self.id = sent_id
        self.tokenized = sent
        self.scores = Scores()

    def toJson(self):
        data = self.__dict__
        data['scores'] = self.scores.toJson()
        return data

    def print_Sentence(self):  # Only for debug purposes -> too much verbose
        print([self.tokenized])

    def compute_Scores(self, attributes, loc_th=5, loc=[0, 0, 0, 1, 0],
                       reset=True):
        if reset:
            self.scores.zero()  # Reset to avoid update of loaded values

        doc_term_freq = attributes['termFrequencies']
        prop_nouns = attributes['properNouns']
        simScore = attributes['similarityScores']
        nums = attributes['numbers']
        DF_dict = attributes['documentsFrequencies']
        nameEnt = attributes['namedEntities']

        # TF score
        self.scores.set_TF(self.tokenized, doc_term_freq)

        # Sentence location score
        sents_num = attributes['sents_num']
        self.scores.set_sent_location(self.id, sent_len=sents_num, th=loc_th,
                                      loc_scores=loc)

        # Proper noun score
        self.scores.set_proper_noun(self.tokenized, prop_nouns, doc_term_freq)

        # Word Co-occurence
        self.scores.set_co_occour(self.tokenized, attributes['summary'],
                                  doc_term_freq)

        # Similarity Score
        self.scores.set_similarity_score(self.id, simScore)

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
        self.scores.set_posnegScore(self.tokenized,
                                    attributes['highlightsTF'],
                                    attributes['highlightsOC'])

        # Thematic words
        self.scores.set_thematicWordsScore(self.tokenized, self.id,
                                           doc_term_freq)

        # Named entities score
        self.scores.set_namedEntitiesScore(self.tokenized, doc_term_freq,
                                           nameEnt)

    def get_total_score(self):
        return self.scores.get_total()

    def info(self, verbose=True):
        if verbose:
            print('Tokens in sentence: {}'.format(len(self.tokenized)))
        return len(self.tokenized)

    def text(self):
        reconstructed_sentence = ''
        for token in self.tokenized:
            reconstructed_sentence += '{} '.format(token)
        return reconstructed_sentence

    def print_scores(self, text=False, onlyTotal=True):
        print('\nSentence id {}'.format(self.id))
        if text:
            reconstructed_sent = ''
            for token in self.tokenized:
                reconstructed_sent += '{} '.format(token)
            print(reconstructed_sent)
        if onlyTotal:
            self.scores.print_total_scores(detail=False, total=True)
        else:
            self.scores.print_total_scores(detail=True, total=True)


class Document():
    def __init__(self, doc, doc_id, summary):
        self.id = str(doc_id)
        self.sentences = {}
        self.summary = None
        self.termFrequencies = {}
        self.highlightsTF = {}
        self.HF = {}
        self.sentSimilarities = {}
        self.sentRanks = {}
        self.nums = []
        self.mean_length = 0
        self.tot_tokens = 0

        self.add_summary(summary)

        for sentence in doc:  # Must be pre-processed by spacy pipeline
            if len(sentence) > 0:
                self.add_sentence(sentence, doc_id)
                self.tot_tokens += len(sentence)

        # Normalize term frequency
        for key in self.termFrequencies:
            self.termFrequencies[key] /= self.tot_tokens

    def toJson(self):
        data = self.__dict__
        sents = {}
        for sent in self.sentences:
            sents.update({sent: self.sentences[sent].toJson()})
        data['sentences'] = sents
        return data

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
            sent_id = str(doc_id) + '_{}'.format(len(self.sentences))
            self.sentences[sent_id] = Sentence(sent, sent_id)

            # Update term frequency
            for token in sent:
                token = token.casefold()
                if token not in self.termFrequencies:
                    self.termFrequencies[token] = 1
                else:
                    self.termFrequencies[token] += 1

    def add_summary(self, summary):
        if not isinstance(summary, str):
            print('Input type must be \'string\'')
            return
        else:
            self.summary = summary
            # highlights term frequencies
            for token in summary:
                norm_token = token.casefold()
                if norm_token not in self.highlightsTF:
                    self.highlightsTF[norm_token] = 1
                else:
                    self.highlightsTF[norm_token] += 1
            self.highlightsTF = dict(sorted(self.highlightsTF.items(),
                                     key=lambda x: x[1]))
            # highlights word occurrence
            sum_sentences = summary.splitlines()
            updates = {}
            for sentence in sum_sentences:
                for token in summary.split(' '):
                    updates[token] = False
                    if token in sentence and not updates[token]:
                        if token not in self.HF:
                            self.HF[token] = 1
                        else:
                            self.HF[token] += 1
                        updates[token] = True
                updates[token] = False  # Ensures each token is counted once

            # Normalization of HF
            for key in self.HF:
                self.HF[key] /= len(self.HF)

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
                      'sents_num': len(self.sentences),
                      'properNouns': properNouns,
                      'similarityScores': self.sentSimilarities,
                      'summary': self.summary,
                      'numbers': self.nums,
                      'documentsFrequencies': DF_dict,
                      'sentenceRanks': self.sentRanks,
                      'meanSentenceLength': self.mean_length,
                      'highlightsTF': self.highlightsTF,
                      'highlightsOC': self.HF,
                      'namedEntities': namedEntities}
        for sentence in self.sentences:
            # attributes['similarityScore'] = self.sentSimilarities[sentence]
            self.sentences[sentence].compute_Scores(attributes)

    def add_sentSimm(self, simmDict):
        self.sentSimilarities.update(simmDict)

    def add_nums(self, numList):
        self.nums.append(numList)

    def compute_meanLength(self):
        '''
        summary = self.summary
        summary = summary.split('\n')
        summary = [x.split(' ') for x in summary]
        for sent in summary:  # Word-wise
            self.mean_length += len(sent)
        self.mean_length /= len(summary)
        '''

        # Token - wise
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

    def print_scores(self, _text=False, _onlyTotal=True):
        print('\nDocument {} {}'.format(self.id, '-'*(79-len(self.id))))
        for sentence in self.sentences.values():
            sentence.print_scores(text=_text, onlyTotal=_onlyTotal)


class Dataset():
    def __init__(self, name='Processed_dataset'):
        self.documents = {}
        self.proper_nouns = []  # For all dataset to avoid duplicates
        self.named_entities = []  # For all dataset to avoid duplicates
        self.cue_words = []
        self.DF = {}
        self.name = name

    def add_document(self, doc, doc_id, high):
        if doc_id not in self.documents:
            self.documents[doc_id] = Document(doc, doc_id, high)
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
        json.dump(data, out_stream)
        out_stream.close()

    def load(self, pathToFile=None):
        if pathToFile is None:
            pathToFile = os.getcwd() + os.sep + self.name
        if '.json' not in pathToFile:
            pathToFile += '.json'
        if not os.path.isfile(pathToFile):
            print('File \"{}\" not found'.format(os.path.basename(pathToFile)))
            return None

        in_stream = open(pathToFile, 'r')
        loaded_dataset = json.load(in_stream)
        in_stream.close()

        self.proper_nouns = loaded_dataset['proper_nouns']
        self.named_entities = loaded_dataset['named_entities']
        self.cue_words = loaded_dataset['cue_words']
        self.DF = loaded_dataset['DF']
        self.name = loaded_dataset['name']

    def process_dataset(self, dataset_in, doc_th=3, save=True):

        # nlp = spacy.load('en_core_web_sm')  # Loads pipeline for english
        nlp = spacy.load('en_core_web_md')  # Try this for having vectors

        # Making textrank pipe
        nlp.add_pipe('textrank', last=True)

        # Generating tokenized structure for feature evaluation

        i = 0
        with tqdm(total=len(dataset_in)) as pbar_load:
            for key in dataset_in:
                pbar_load.set_description('processing dataset: ')
                doc_id = str(key['id'])
                high = key['highlights']

                tokenized_article = nlp(key['article'])  # Spacy object

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

    def print_scores(self, text=False, onlyTotal=True):
        for doc in self.documents.values():
            doc.print_scores(_text=text, _onlyTotal=onlyTotal)

    def summarization(self, th=0):
        summarized_dataset = {}
        for doc in self.documents.values():
            ordered_scores = doc.get_total_scores()
            ordered_doc = ''
            document = self.documents[str(doc.id)]
            for sent_id in ordered_scores:
                if ordered_scores[sent_id] > th:
                    sentence = document.get_sentence(sent_id, True)
                    ordered_doc += '{}\n'.format(sentence)
            summarized_dataset[doc.id] = ordered_doc
        return summarized_dataset

    def rouge_computation(self, n, th=0, show=False, sentences=False):
        summarization = self.summarization(th)
        for doc_id, doc in summarization.items():
            # Split summaries in sentences
            hyp = doc.split('\n')
            ref = self.documents[doc_id].summary.split('\n')

            # Split sentences in tokens and retain same number of sentences
            ref = [sent.split(' ') for sent in ref]
            hyp = [sent.split(' ') for sent in hyp][:len(ref)]  # to compare
            if sentences:
                print('-'*80)
                print(hyp, '\n', '-'*39, 'VS', '-'*39, '\n', ref)
                print('-'*80)

            # n-gram merging
            for summary in [ref, hyp]:
                for sentence in summary:
                    temp = []
                    i = 0
                    while i+n < len(sentence):
                        temp.append(sentence[i:i+n])
                        i += 1
                    summary[summary.index(sentence)] = temp

            # Count ngram matching
            summary_match_count = 0
            for sentence in hyp:
                ref_ngram_list = ref[hyp.index(sentence)]
                for ngram in sentence:
                    if ngram in ref_ngram_list:
                        summary_match_count += 1

            # Rouge-N
            ref_ngram_count = sum(len(i) for i in ref)
            rouge_n = summary_match_count/ref_ngram_count

            # Recall -> number of words captured by summary wrt reference
            '''
            overlap_ngrams = 0
            for sentence in hyp:
                ref_sentence = ref[hyp.index(sentence)]
                for token in sentence:
                    if token in ref_sentence:
                        overlap_ngrams += 1
            rouge_recall = overlap_ngrams / ref_ngram_count
            '''

            # Precision -> how much of the summarization is useful
            hyp_ngram_count = sum(len(i) for i in hyp)
            rouge_precision = summary_match_count / hyp_ngram_count
            if show:
                print('\nDocument id: {}'.format(doc_id))
                print(' Rouge-{}: {:0.4f}'.format(n, rouge_n))
                # print(' Rouge Recall: {:0.4f}'.format(rouge_recall))
                print(' Rouge Precision: {:0.4f}\n'.format(rouge_precision))
        return rouge_n


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

    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')
    CNN_processed = Dataset(name='CNN_processed.json')
    CNN_processed.process_dataset(CNN_dataset['train'])
    CNN_processed.save()

    '''
    test_dataset = Dataset(name='Cats_dataset')
    test_dataset.process_dataset(test_train)
    test_dataset.save()


    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')
    CNN_processed = Dataset(name='CNN_processed.json')
    CNN_processed.process_dataset(CNN_dataset['train'])

    CNN_processed.print_scores(onlyTotal=False)
    CNN_processed.rouge_computation(2, show=True)

    summary = CNN_processed.summarization()
    for key in summary:
        print(summary[key])
    '''
