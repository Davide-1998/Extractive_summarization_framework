from Sentence import Sentence


class Document():
    def __init__(self, doc=None, doc_id=0, summary=None):
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

    def toJson(self):
        data = self.__dict__
        sents = {}
        for sent in self.sentences:
            sents.update({sent: self.sentences[sent].toJson()})
        data['sentences'] = sents
        return data

    def set_docID(self, id):
        self.id = str(id)

    def from_dict(self, loadedDict):
        for key in loadedDict:
            if key in self.__dict__ and key != 'sentences':
                self.__dict__[key] = loadedDict[key]
        for sentence in loadedDict['sentences']:
            sent_in = loadedDict['sentences'][sentence]
            temp_sent = Sentence()
            temp_sent.from_dict(sent_in)
            self.sentences[sentence] = temp_sent

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

    def compute_scores(self, properNouns, DF_dict, namedEntities, scores=[]):
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
            self.sentences[sentence].compute_Scores(attributes,
                                                    score_list=scores,
                                                    reset=True)

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
