from math import log
import numpy as np


# Available scoring methods = 14

class Scores():
    '''
    This class contains the results of the scoring strategies
    available and the methods to compute them.
    By default, each score is set to 0.
    The attributes are the available scoring strategies.

    Attributes
    ---------
    TF: float
        Term Frequency score
    sent_location: float
        Sentence Location score
    proper_noun: float
        Proper Nouns score
    co_occur: float
        Co Occurrence score
    sent_similarity: float
        Sentence Similarity score
    num_val: float
        Numerical Tokens score
    "TF_ISF_IDF: float
        Term Frequency/inverse document frequency score
    sent_rank: float
        Sentence Ranking score
    sent_length: float
        Sentence Length score
    pos_keywords: float
        Positive Keywords score
    neg_keywords: float
        Negative Keywords score
    thematic_features: float
        Thematic Words score
    named_entities: float
        Named Entities score
    '''

    def __init__(self):
        self.TF = 0                 # Term Frequency Score
        self.sent_location = 0      # Sentence Location Score
        self.proper_noun = 0        # Proper Nouns Score
        self.co_occur = 0           # Co-Occurrence Score
        self.sent_similarity = 0    # Semantic Similarity Score
        self.num_val = 0            # Numerical Value Score
        self.TF_ISF_IDF = 0         # TF-IDF Score
        self.sent_rank = 0          # Text Rank Score
        self.sent_length = 0        # Sentence Length Score
        self.pos_keywords = 0       # Positive Keywords Score
        self.neg_keywords = 0       # Negative Keywords Score
        self.thematic_features = 0  # Thematic Features Score
        self.named_entities = 0     # Named Entities Score

    def zero(self):  # Sets all values to 0
        '''
        This method is used to set all the available scores to 0
        '''

        for el in self.__dict__:
            if self.__dict__ != 0:
                self.__dict__[el] = 0

    def set_TF(self, attributes):
        '''
        This method is used to compute the term frequency score.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        token_list = attributes['tokenized']
        term_freq_dict = attributes['termFrequencies']
        if self.TF != 0:
            print('TF value will be overwritten')
            self.TF = 0
        for token in token_list:
            self.TF += term_freq_dict.get(token.casefold(), 0)
        self.TF /= len(token_list)  # Added for normalization -> scores go up

    def set_sent_location(self, attributes):
        '''
        This method is used to compute the sentence location score.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        sent_id = attributes['sent_id']
        sent_len = len(attributes['sentences'])
        [ED, NB1, NB2, NB3, FaR] = attributes['location_score_filter']
        th = attributes['location_threshold']
        ALL = attributes['all_location_scores']

        if not ALL:
            if [ED, NB1, NB2, NB3, FaR].count(1) != 1:
                print('Multiple or none options setted'
                      'score of 0 attributed')
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
                    lower_b = 1/sent_id
                    higher_b = 1/(sent_len-sent_id+1)
                    self.sent_location = max(lower_b, higher_b)

                # FaR -> Fattah and Ren
                if FaR:
                    if sent_id < 5:
                        self.sent_location = 1 - (sent_id*1/5)
                    else:
                        self.sent_location = 0
        else:
            sent_id = int(sent_id.split('_')[1])
            # ED computation
            ED_loc = 0
            if sent_id < 2:
                ED_loc = 1

            # NB1 computation
            NB1_loc = 0
            if sent_id < th:
                NB1_loc = 1

            # NB2 computation
            NB2_loc = 1/(sent_id + 1)

            # NB3 cmputation
            NB3_loc = 0
            lower_b = 1 / (sent_id + 1)
            upper_b = 1 / (sent_len - sent_id + 1)
            NB3 = max(lower_b, upper_b)

            # Far computation
            FaR_loc = 0
            if sent_id < 5:
                FaR_loc = 1 - (sent_id*(1/5))

            self.sent_location = max(ED_loc, NB1_loc, NB2_loc,
                                     NB3_loc, FaR_loc)

    def set_proper_noun(self, attributes):
        '''
        This method is used to compute the proper noun score.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        sentence = attributes['tokenized']
        prop_noun_list = attributes['properNouns']
        tf_dict = attributes['termFrequencies']

        # Nobata et al 2001
        if self.proper_noun != 0:
            print('Proper noun will be overwritten')
            self.proper_noun = 0
        for token in sentence:
            if token.casefold() in prop_noun_list:
                self.proper_noun += tf_dict[token.casefold()]

    def set_co_occour(self, attributes):
        '''
        This method is used to compute the Co Occurrence score.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        tokenized_sentence = attributes['tokenized']
        summary = attributes['summary']
        tf_dict = attributes['termFrequencies']

        co_occurrence = 0
        summary = summary.casefold()
        for token in tokenized_sentence:
            norm_token = token.casefold()
            co_occurrence = summary.count(norm_token)
            co_occurrence *= tf_dict[norm_token]
            self.co_occur += co_occurrence/len(tokenized_sentence)
            # Division avoids bias given by sentence length

    def set_similarity_score(self, attributes):
        '''
        This method is used to compute the similarity score as in
        Fattah & Ren 2009 paper.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        sent_id = attributes['sent_id']
        score = attributes['similarityScores']

        for key in score.keys():
            if sent_id == key.split(':')[0]:
                # Cumulative sum
                self.sent_similarity += score[key]

    def set_numScore(self, attributes):
        '''
        This method is used to compute the numerical tokens score as
        in Fattah & Ren 2009 paper.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        sent = attributes['tokenized']
        numList = attributes['numbers']

        count = 0
        for token in sent:
            if token.casefold() in numList:
                count += 1
        self.num_val = count/len(sent)  # Token-wise length

    def set_TF_ISF_IDF(self, attributes):
        '''
        This method is used to compute the term frequency / inverse
        document frequency score as in Nobata et al. 2001 paper.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        sentence = attributes['tokenized']
        TF = attributes['termFrequencies']
        DF = attributes['documentsFrequencies']

        num_doc = len(DF)
        TF_ISF_IDF = 0
        for token in sentence:
            token = token.casefold()
            doc_has_token = 0
            for doc in DF:
                doc_has_token += DF[doc].get(token, 0)
            IDF = log((num_doc/doc_has_token))
            TF_ISF_IDF += (TF[token]/1+TF[token])*IDF  # Token-wise
            self.TF_ISF_IDF += TF_ISF_IDF

    def set_sentRank(self, attributes):
        '''
        This method is used to compute the sentence rank score.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        sentence = attributes['tokenized']
        rankings = attributes['sentenceRanks']

        reconstructed_sentence = ''
        for token in sentence:
            reconstructed_sentence += token + ' '
        reconstructed_sentence = reconstructed_sentence.casefold()
        for chunk in rankings:
            if chunk in reconstructed_sentence:
                self.sent_rank += rankings[chunk]

    def set_sentLength(self, attributes):
        '''
        This method is used to compute the sentence length score.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        sentence = attributes['tokenized']
        mean_length = attributes['meanSentenceLength']

        # a parabola is used as an implicit treshold
        sent_len = len(sentence)
        if sent_len < 2*mean_length:  # otherwise 0 -> implicit
            self.sent_length = ((-1/mean_length**2)) * \
                               (sent_len**2) + \
                               ((2/mean_length)*sent_len)

    def set_positiveScore(self, attributes):
        '''
        This method is used to compute the positive keywords score.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        # Use tokenized version for summary and document's sentences
        summary = attributes['tokenized_summary']
        sentences = [x.tokenized for x in attributes['sentences'].values()]

        # Make both case-insensitive
        casefold_sents = []
        for sentence in sentences:
            casefold_tokens = []
            for token in sentence:
                casefold_tokens.append(token.casefold())
            casefold_sents.append(casefold_tokens)
        sentences = casefold_sents

        # Prior Probability
        prior = len(summary) / len(sentences)

        # Conditional and Event Probability
        reduced_sent = set()
        for token in attributes['tokenized']:
            reduced_sent.add(token.casefold())

        for token in reduced_sent:
            count_summary = 0
            count_dataset = 0
            for sentence in summary:
                if token in sentence:
                    count_summary += 1
            for sentence in sentences:
                if token in sentence:
                    count_dataset += 1

            conditional_prob = count_summary / len(summary)
            event_prob = count_dataset / len(sentences)

            # Scores computing
            occurring_frequency = attributes['tokenized'].count(token)
            probability = (conditional_prob * prior) / event_prob

            self.pos_keywords += occurring_frequency * probability

        self.pos_keywords /= len(attributes['tokenized'])

    def set_negativeScore(self, attributes):
        '''
        This method is used to compute the sentence negative keywords
        score.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        # Use tokenized version for summary and document's sentences
        summary = attributes['tokenized_summary']
        sentences = [x.tokenized for x in attributes['sentences'].values()]
        sents_not_in_summary = len(sentences) - len(summary)

        # Make both case-insensitive
        casefold_sents = []
        for sentence in sentences:
            casefold_tokens = []
            for token in sentence:
                casefold_tokens.append(token.casefold())
            casefold_sents.append(casefold_tokens)
        sentences = casefold_sents

        # Prior Probability
        prior = len(summary) / len(sentences)

        # Conditional and Event Probability
        reduced_sent = set()
        for token in attributes['tokenized']:
            reduced_sent.add(token.casefold())

        for token in reduced_sent:
            count_summary = 0
            count_dataset = 0
            for sentence in summary:
                if token in sentence:
                    count_summary += 1
            for sentence in sentences:
                if token in sentence:
                    count_dataset += 1

            inverse_summary_count = len(summary)-count_summary
            if sents_not_in_summary != 0:
                conditional_prob_neg = inverse_summary_count / \
                                       sents_not_in_summary
            else:
                conditional_prob_neg = 0
            event_prob = count_dataset / len(sentences)

            # Scores computing
            occurring_frequency = attributes['tokenized'].count(token)
            if event_prob != 0:
                probability_neg = (conditional_prob_neg * 1-prior) / \
                                   event_prob
            else:
                probability_neg = 0
            self.neg_keywords -= occurring_frequency * probability_neg

        self.neg_keywords /= len(attributes['tokenized'])

    def set_thematicWordsScore(self, attributes):
        '''
        This method is used to compute the thematic words score.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        sentence = attributes['tokenized']
        doc_tf = attributes['termFrequencies']

        mean = sum(doc_tf.values())/len(doc_tf)
        for token in sentence:
            norm_token = token.casefold()
            if doc_tf[norm_token] > 2*mean:  # Is thematic
                self.thematic_features += doc_tf[norm_token]

    def set_namedEntitiesScore(self, attributes):
        '''
        This method is used to compute the named entities score.

        attributes: dict
            Dictionary of document and sentence features.
        '''

        sentence = attributes['tokenized']
        termFreqDict = attributes['termFrequencies']
        neList = attributes['namedEntities']

        for token in sentence:
            norm_token = token.casefold()
            if norm_token in neList:
                self.named_entities += termFreqDict[norm_token]

    def get_total(self, show=False, weights=[]):
        '''
        This method returns the weighted version of the scoring
        strategies computed.

        show: bool
            Flags whether or not to print the total score.

        weights: list of floats
            Is the list of scaling vectors to apply.
            The number of weights to provide can be retrieved using
            Dataset.get_num_weights()

        Return
        ------
        float:
            Sum of the weighted scores.
        '''

        scores = list(self.__dict__.values())

        if len(weights) == 0:  # No weights case
            tot_score = sum(scores)
            if show:
                print('Tot Score = %0.4f' % tot_score)
            return tot_score
        else:  # Weighted case
            if len(weights) != len(scores):
                print('Incompatible shapes among input weights'
                      'and available scores')

            values = np.array(scores)
            weights = np.array(weights)
            weighted_sum = sum(values * weights)
            if show:
                print('Tot Score = %0.4f' % weighted_sum)
            return weighted_sum

    def toJson(self):
        '''
        This method is used to return a .json serializable version
        of the class.

        Returns
        -------
        dict
        '''

        return self.__dict__.copy()

    def from_dict(self, loaded_dict):
        '''
        This method is used to load the data of a loaded .json
        file into the score class.

        loaded_dict:
            Dictionary of data having the class attributes
        '''

        for key in loaded_dict:
            if key in self.__dict__:
                self.__dict__[key] = loaded_dict[key]

    def print_total_scores(self, detail=True, total=True):
        '''
        This method is used to print the detailed scores contained in
        the class.

        detail: bool
            Flags whether or not to print all the available scores.
        total: bool
            Flags whether or not to print the sum of all the scoring
            strategies.
        '''

        if detail:
            for key in self.__dict__:
                print(key, '->', self.__dict__[key])
        if total:
            self.get_total(show=True, getVal=False)
