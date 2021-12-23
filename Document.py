from Sentence import Sentence
import spacy


class Document():
    '''
    This class has the role to represent each document processed
    by the spacy language model and initiate the
    Sentence class able to compute the summarization scores.

    Attributes
    ----------

    id: str
        Unique identifier of the document.
    sentences: dict
        Dictionary of Sentence classes representing each sentence
        in the document. Each element is addressed by the sentence ID
        which is the document id concatenated with the sentence
        position in the document.
    summary: str
        Is the plain text summary provided with the document.
    termFrequencies: dict
        Is the dictionary of term frequencies. The keys are the
        casefolded text of the tokens and the values are a
        float describing each token frequency.
    sentSimilarities: dict
        Is the dictionary containing the sentence similarity scores
        computed by the .simm method of spacy object.
        Each element in the dictionary is addressed by the
        concatenation of the IDs of the sentences involved
        in the computation.
    sentRanks: dict
        Is the dicitonary containing the sentence rankings
        computed by the textrank pipeline.
        Each element is addressed by the sentence plain text.
    mean_length: float
        Is the mean sentence length computed over the sentences in the
        document.
    tot_tokens: float
        Is the total number of tokens in the document.
    '''

    def __init__(self, doc=None, doc_id=0, summary=None):
        self.id = str(doc_id)           # Document Identifier
        self.sentences = {}             # Sentences of the Document
        self.summary = None             # Summary of the Document
        self.termFrequencies = {}       # Document-wise Term Frequencies
        self.sentSimilarities = {}      # Similarity computed for all sentences
        self.sentRanks = {}             # Rankings of the Sentences
        self.mean_length = 0            # Mean length of sentences in document
        self.tot_tokens = 0             # Total tokens in the document

        if summary is not None:
            self.add_summary(summary)

        if doc is not None:
            for sentence in doc:
                if len(sentence) > 0:
                    self.add_sentence(sentence, doc_id)
                    self.tot_tokens += len(sentence)

            # Normalize term frequency
            for key in self.termFrequencies:
                self.termFrequencies[key] /= self.tot_tokens

    def set_docID(self, _id):
        self.id = str(_id)

    def add_sentence(self, sent, doc_id, load=False):
        '''
        This method adds a sentence to the document.

        sent: list of str
            List of tokens representing the sentence.
        doc_id: str
            Unique identifier of the document.
        load: bool
            Flag whether the document has been loaded from a .json file.
            Avoids to cause errors in menaging the data.
        '''
        if not isinstance(sent, list):
            print('Format type of input must be \'List\'')
            return
        # if len(sent) == 1:
        #     print('Warning: singleton {} detected!'.format(sent))
        elif len(sent) == 0:  # Enforced in mathod above
            print('Input sentence not eligible for adding operation, Ignored')
            return
        else:
            # Push sentence into dictionary and organize by internal ID
            sent_id = str(doc_id) + '_{}'.format(len(self.sentences))
            self.sentences[sent_id] = Sentence(sent, sent_id)

            if not load:
                # Update term frequency
                for token in sent:
                    token = token.casefold()
                    if token not in self.termFrequencies:
                        self.termFrequencies[token] = 1
                    else:
                        self.termFrequencies[token] += 1

    def add_sentSimm(self, simmDict):
        '''
        Adds the provided sentence similarity scores to the already
        existing ones.

        simmDict: dict {str: float}
            Dictionary of sentence similarity scores.
        '''
        self.sentSimilarities.update(simmDict)

    def add_summary(self, summary):
        '''
        Adds a plain text summary to the document in use.

        summary: str
            Is the summary of the document.
        '''

        if not isinstance(summary, str):
            print('Input type must be \'string\'')
            return
        else:
            self.summary = summary

    def add_sentRank(self, text, rank):
        '''
        Adds a sentence rank measure to the dictionary in the class.

        text: str
            Is the plain text of the sentence.
        rank: float
            Is the score given to the input sentence.
        '''

        if isinstance(rank, float) and isinstance(text, str):
            if text not in self.sentRanks:
                self.sentRanks[text] = rank
            # else:
            #     self.sentRanks[text] += rank
            #     print('Entry \"{}\" already exist in record, skipping.'
            #           .format(text))
                # Maybe try with accumulating them and see if rouge score up
        else:
            print('Expected text and rank to be of type string and float, but '
                  'got input of type {} and {}'.format(type(text), type(rank)))
            return

    def compute_scores(self, properNouns=[], DF_dict={}, namedEntities=[],
                       scores=[], numerical_tokens=[], spacy_pipeline=None,
                       _reset=True, loc_threshold=5, _all_loc=False,
                       locFilter=[0, 0, 0, 1, 0], lemma=False):
        '''
        Computes the sentence-wise scores of each sentence in the
        document.

        properNouns: list of str
            Is the list of token recognised as proper nouns.
        DF_dict: dictionary {str: bool}
            Is the dictionary containing the document frequency of
            each token. The key is the casefolded text of the token.
        namedEntities: list of str
            Is the list of tokens recognised as named entities.
            Each key is in casefold.
        scores: list of str
            Is the list of scoring strategies to compute. If empty all
            wil be used.
        numerical_tokens: list of str
            List of tokens identified as numerical. Each entry is in
            casefold.
        spacy_pipeline: spacy.lang
            Is the spacy pipeline to use in order to tokenize the
            summary. If None a default one is created.
        _reset: bool
            Flag whether or not to set all the scoring strategies to 0
            before starting the computations.
        loc_threshold: int
            Is the maximum location used in computing the Nobata et al.
            Sentence Location strategy.
        _all_loc: bool
            Flag whether or not to take the maximum among all the
            location scores available.
        locFilter: list of bool
            Flags whether or not to use the specified Sentence Location
            method. If _all_loc is set to True, this flag has no effect.
        lemma: bool
            Flags whether or not to use the lemma of each token.
        '''

        # Computed here to avoid multiple recomputations in sentences
        if spacy_pipeline is None:
            nlp = spacy.load('en_core_web_md')
        else:
            nlp = spacy_pipeline
        summary = nlp(self.summary)
        tokenized_summary = []
        lemma_summary = ''
        for sentence in summary.sents:
            tokenized_sent = []
            for token in sentence:
                norm_token = token.text.casefold()
                if lemma:
                    norm_token = token.lemma_.casefold()
                    lemma_summary += norm_token + ' '
                tokenized_sent.append(norm_token)
            tokenized_summary.append(tokenized_sent)

        if lemma:
            self.summary = lemma_summary.strip()

        attributes = {'termFrequencies': self.termFrequencies,
                      'sentences': self.sentences,
                      'properNouns': properNouns,
                      'similarityScores': self.sentSimilarities,
                      'summary': self.summary,
                      'tokenized_summary': tokenized_summary,
                      'numbers': numerical_tokens,
                      'documentsFrequencies': DF_dict,
                      'sentenceRanks': self.sentRanks,
                      'meanSentenceLength': self.mean_length,
                      'namedEntities': namedEntities}
        for sentence in self.sentences:
            self.sentences[sentence].compute_Scores(attributes,
                                                    score_list=scores,
                                                    reset=_reset,
                                                    loc_th=loc_threshold,
                                                    loc=locFilter,
                                                    all_loc_scores=_all_loc)

    def compute_meanLength(self):
        '''
        Computes the mean length of the sentences in the document and
        stores it in the self.mean_length class attribute.
        '''

        # Token - wise
        for sent in self.sentences:
            self.mean_length += len(self.sentences[sent].tokenized)
        self.mean_length = self.mean_length/len(self.sentences)

    def get_total_scores(self, show=False):
        '''
        Returns a dict of sums of the available summarization scores.

        show: bool
            Flag whether to print or not the score for each
            summarization method.

        Returns
        -------

        dict {str: float}
        '''

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

    def get_weighted_total_scores(self, weights, show=False):
        '''
        This method returns the weighted sum of the available
        scoring strategies.

        weights: list of float
            List of scaling factors to apply at each scoring method.
        show: bool
            Flags whether or not to print the summed scores for each
            sentence.

        Returns
        -------

        dict{str: float}
        '''

        scores = {}
        for sentence in self.sentences.values():
            scores[sentence.id] = sentence.get_weighted_total_score(weights)
        ordered_scores = dict(sorted(scores.items(),
                                     key=lambda x: x[1],
                                     reverse=True))
        if show:
            for el in ordered_scores:
                print(el, ' -> ', ordered_scores[el])
        return ordered_scores

    def get_sentence(self, sentence_id, text=False):
        '''
        Returns a specific sentence giving its ID.

        sentence_id: str
            Is the unique identifier of each sentence.
        text: bool
            Flags whether to return the plain text of the sentence.
        '''

        if sentence_id not in self.sentences.keys():
            print('No sentence {} in dictionary'.format(sentence_id))
            return None
        if text:
            return self.sentences.get(sentence_id).text()
        else:
            return self.sentences.get(sentence_id)

    def toJson(self):
        '''
        Returns the dictionary of the class to be able to dump
        it in a .json file.
        '''

        data = self.__dict__.copy()
        sents = {}
        for sent in self.sentences:
            sents.update({sent: self.sentences[sent].toJson()})
        data['sentences'] = sents
        return data

    def from_dict(self, loadedDict):
        '''
        Fills the class using a dictionary coming from a loaded
        .json file.

        loadedDict:
            dictionary of a loaded .json file
        '''

        for key in loadedDict:
            if key in self.__dict__ and key != 'sentences':
                self.__dict__[key] = loadedDict[key]
        for sentence in loadedDict['sentences']:
            sent_in = loadedDict['sentences'][sentence]
            temp_sent = Sentence()
            temp_sent.from_dict(sent_in)
            self.sentences[sentence] = temp_sent

    def info(self, verbose=True):
        '''
        Returns informations regarding the document class.
        verbose: bool
            Flags whether or not to print all the informations of
            the document.
        '''

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
        '''
        Prints the scores of each sentence in the document.

        _text: bool
            Flags whether or not to print the text of the sentence.
        _onlyTotal: bool
            Flags whether or not to print the sum of the scores or
            a detailed version of them.
        '''

        print('\nDocument {} {}'.format(self.id, '-'*(79-len(self.id))))
        for sentence in self.sentences.values():
            sentence.print_scores(text=_text, onlyTotal=_onlyTotal)
