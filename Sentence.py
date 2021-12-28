from Scores import Scores


class Sentence():
    '''
    Is the class used to represent each sentence in the input documents.

    Attributes
    ----------
    id: str
        Is the unique identifier of the sentence. It is made of the
        document ID concatenated with the sentence location.
        (e.g. <document ID>_<sentence location> )
    tokenized: list of str
        Is the list containing the tokens of the sentence.
    scores: Scores
        Is a Scores class instantiation containing the summarization
        scores for the sentence.
    '''

    def __init__(self, sent=None, sent_id=0):
        self.id = str(sent_id)      # Unique identifier of the sentence
        self.tokenized = sent       # Tokenized Sentence
        self.scores = Scores()      # Scores of the sentence

    def set_sentence(self, list_of_tokens, _id='0'):
        '''
        Fills or changes the sentence contained in the class.

        list_of_tokens: list of str
            Is the list of tokens representing the sentence.
        _id: str
            Is the unique identifier of the sentence.
        '''

        if not isinstance(list_of_tokens, list):
            print('A list of token must be given in order to crete a Sentence')
            return
        else:
            self.tokenized = list_of_tokens
            self.id = _id

    def compute_Scores(self, attributes, score_list=[], loc_th=5,
                       all_loc_scores=False, loc=[0, 0, 0, 1, 0], reset=True):
        '''
        Is the method performing the computation of the sentence scores.
        It calls the methods implemented in the Scores class.

        attributes: dict
            Is a dictionary of document features that may be used for
            the computation of the scoring strategies.
            At this level it is comprehensive of:
                'termFrequencies' -> term frequencies of the tokens in
                                     the document.
                'sentences' -> Sentences in the document class.
                'properNouns' -> Proper nouns in the document class.
                'similarityScores' -> Similarity scores available in
                                      the document class
                'summary' -> Plain text of the summary.
                'tokenized_summary' -> Tokenized version of the summary.
                'numbers' -> Numerical tokens in the document class.
                'documentsFrequencies' -> Document frequencies for the
                                          tokens in the document.
                'sentenceRanks' -> Dictionary of Sentence ranks of each
                                   sentence in the document class.
                'meanSentenceLength' -> Mean length of the sentences in
                                        the document.
                'namedEntities' -> Named entities in the dataset.
        score_list: list of str
            List of scores that is requested to compute. The other
            scores not present in the list will be kept to 0.
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
        loc_th: int
            Maximum number of sentences to consider during the
            computation of Nobata et al. scoring strategy 1.
        all_loc_scores: bool
            Flag whether to take the maximum among all the sentence
            location scoring strategies or not.
        loc: list of bool
            Vector of size 5 representing the Sentence location
            scoring strategies available. If used with all_loc_scores
            set to True, it doesn't have any effect.
        reset: bool
            Flags whether to reset the scores before computing them.
        '''

        if reset:
            self.scores.zero()  # Reset to avoid update of loaded values

        attributes['sent_id'] = self.id
        attributes['tokenized'] = self.tokenized
        attributes['location_score_filter'] = loc
        attributes['location_threshold'] = loc_th
        attributes['all_location_scores'] = all_loc_scores

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
        '''
        Returns the summed total of the scores computed.

        Returns
        -------
        float
        '''

        return self.scores.get_total()

    def get_weighted_total_score(self, weights):
        '''
        Returns the weighted sum of the scoring strategies computed
        for the sentence.

        weights: list of floats
            Is the list of scaling factors to apply to the scores.

        Returns
        -------
        float
        '''

        return self.scores.get_total(weights=weights)

    def text(self):
        '''
        Returns the text of the sentence.

        Returns
        ------
        str
        '''

        reconstructed_sentence = ''
        for token in self.tokenized:
            reconstructed_sentence += '{} '.format(token)
        return reconstructed_sentence

    def toJson(self):
        '''
        This method is used to return a json serializable representation
        of the class.

        Returns
        -------
        dict
        '''

        data = self.__dict__.copy()
        data['scores'] = self.scores.toJson()
        return data

    def from_dict(self, loadedDict):
        '''
        This method allows to make an istance of the class with the
        data coming from a loaded dataset .json file.

        loadedDict: dict
            Is the loaded dictionary containing all the attributes
            of the class.
        '''

        for key in loadedDict:
            if key in self.__dict__ and key != 'scores':
                self.__dict__[key] = loadedDict[key]
        self.scores.from_dict(loadedDict['scores'])

    def info(self, verbose=True):
        '''
        Returns the length of the sentence.

        verbose: bool
            If set to True prints the length of the sentence.

        Returns:
        --------
        int
        '''

        if verbose:
            print('Tokens in sentence: {}'.format(len(self.tokenized)))
        return len(self.tokenized)

    def print_Sentence(self):  # Only for debug -> too much verbose
        '''
        Used to print each token in the sentence.
        '''

        print([self.tokenized])

    def print_scores(self, text=False, onlyTotal=True):
        '''
        Method used to print the scores of the sentence.

        text: bool
            Flags whether or not to print also the plain text of the
            sentence.
        onlyTotal: bool
            FLags whether or not to print just the sum of the scores.
            Set to False to have a detailed print.
        '''

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
