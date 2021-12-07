from math import log


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
            if self.__dict__ != 0:
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
            self.sent_length = ((-1/mean_length**2)) * \
                               (sent_len**2) + \
                               ((2/mean_length)*sent_len)

    def set_posnegScore(self, sentence, summary_tf, highlightsOC):
        for token in sentence:
            if token in highlightsOC:
                self.pos_keywords += sentence.count(token) * \
                                     highlightsOC[token]
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

    def from_dict(self, loaded_dict):
        for key in loaded_dict:
            if key in self.__dict__:
                self.__dict__[key] = loaded_dict[key]
