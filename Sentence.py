from Scores import Scores


class Sentence():
    def __init__(self, sent=None, sent_id=0):
        self.id = str(sent_id)      # Unique identifier of the sentence
        self.tokenized = sent       # Tokenized Sentence
        self.scores = Scores()      # Scores of the sentence

    def set_sentence(self, list_of_tokens, _id='0'):
        if not isinstance(list_of_tokens, list):
            print('A list of token must be given in order to crete a Sentence')
            return
        else:
            self.tokenized = list_of_tokens
            self.id = _id

    def compute_Scores(self, attributes, score_list=[], loc_th=5,
                       loc=[0, 0, 0, 1, 0], reset=True):
        if reset:
            self.scores.zero()  # Reset to avoid update of loaded values

        attributes['sent_id'] = self.id
        attributes['tokenized'] = self.tokenized
        attributes['location_score_filter'] = loc
        attributes['location_threshold'] = loc_th

        functions = {'TF': self.scores.set_TF,
                     'sent_location': self.scores.set_sent_location,
                     'proper_noun': self.scores.set_proper_noun,
                     'co_occur': self.scores.set_co_occour,
                     'sent_similarity': self.scores.set_similarity_score,
                     'num_val': self.scores.set_numScore,
                     'TF_ISF_IDF': self.scores.set_TF_ISF_IDF,
                     'sent_rank': self.scores.set_sentRank,
                     'sent_length': self.scores.set_sentLength,
                     'pos_keywords': self.scores.set_positiveScore,
                     'neg_keywords': self.scores.set_negativeScore,
                     'thematic_features': self.scores.set_thematicWordsScore,
                     'named_entities': self.scores.set_namedEntitiesScore}

        if len(score_list) > 0:
            for key in score_list:
                if key in functions:
                    functions.get(key)(attributes)
                else:
                    print('Invalid key {}.\nAvailable ones are: {}'
                          .format(key, functions.keys()))
        else:
            for key in functions:  # If none in input, all scorings will run
                functions.get(key)(attributes)

    def get_total_score(self):
        return self.scores.get_total()

    def get_weighted_total_score(self, weights):
        return self.scores.get_weighted_total(weights)

    def text(self):
        reconstructed_sentence = ''
        for token in self.tokenized:
            reconstructed_sentence += '{} '.format(token)
        return reconstructed_sentence

    def toJson(self):
        data = self.__dict__
        data['scores'] = self.scores.toJson()
        return data

    def from_dict(self, loadedDict):
        for key in loadedDict:
            if key in self.__dict__ and key != 'scores':
                self.__dict__[key] = loadedDict[key]
        self.scores.from_dict(loadedDict['scores'])

    def info(self, verbose=True):
        if verbose:
            print('Tokens in sentence: {}'.format(len(self.tokenized)))
        return len(self.tokenized)

    def print_Sentence(self):  # Only for debug -> too much verbose
        print([self.tokenized])

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
