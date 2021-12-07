from Scores import Scores


class Sentence():
    def __init__(self, sent=None, sent_id=0):
        self.id = str(sent_id)
        self.tokenized = sent
        self.scores = Scores()

    def toJson(self):
        data = self.__dict__
        data['scores'] = self.scores.toJson()
        return data

    def from_dict(self, loadedDict):
        # self.set_sentence(loadedDict['tokenized'], loadedDict['id'])
        for key in loadedDict:
            if key in self.__dict__ and key != 'scores':
                self.__dict__[key] = loadedDict[key]
        self.scores.from_dict(loadedDict['scores'])

    def set_sentence(self, list_of_tokens, _id='0'):
        if not isinstance(list_of_tokens, list):
            print('A list of token must be given in order to crete a Sentence')
            return
        else:
            self.tokenized = list_of_tokens
            self.id = _id

    def print_Sentence(self):  # Only for debug -> too much verbose
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
