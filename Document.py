from Sentence import Sentence
import spacy


class Document():
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
        self.sentSimilarities.update(simmDict)

    def add_summary(self, summary):
        if not isinstance(summary, str):
            print('Input type must be \'string\'')
            return
        else:
            self.summary = summary

    def add_sentRank(self, text, rank):
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
                       locFilter=[0, 0, 0, 1, 0]):
        # Computed here to avoid multiple recomputations in sentences
        if spacy_pipeline is None:
            nlp = spacy.load('en_core_web_md')
        else:
            nlp = spacy_pipeline
        summary = nlp(self.summary)
        tokenized_summary = []
        for sentence in summary.sents:
            tokenized_sent = []
            for token in sentence:
                if not token.is_punct:
                    norm_token = token.text.casefold()
                    tokenized_sent.append(norm_token)
            tokenized_summary.append(tokenized_sent)

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

    def get_weighted_total_scores(self, weights, show=False):
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
        if sentence_id not in self.sentences.keys():
            print('No sentence {} in dictionary'.format(sentence_id))
            return None
        if text:
            return self.sentences.get(sentence_id).text()
        else:
            return self.sentences.get(sentence_id)

    def toJson(self):
        data = self.__dict__.copy()
        sents = {}
        for sent in self.sentences:
            sents.update({sent: self.sentences[sent].toJson()})
        data['sentences'] = sents
        return data

    def from_dict(self, loadedDict):
        for key in loadedDict:
            if key in self.__dict__ and key != 'sentences':
                self.__dict__[key] = loadedDict[key]
        for sentence in loadedDict['sentences']:
            sent_in = loadedDict['sentences'][sentence]
            temp_sent = Sentence()
            temp_sent.from_dict(sent_in)
            self.sentences[sentence] = temp_sent

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
