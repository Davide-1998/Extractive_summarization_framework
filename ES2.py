from datasets import load_dataset
import spacy
import json
import os
from tqdm import tqdm


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
        self.IDF = 0
        self.cue = 0
        # self.title = 0 Dropped
        self.proper_noun = 0
        self.co_occour = 0
        self.sent_similarity = 0
        self.num_val = 0
        # self.font_style = 0 Dropped
        self.laxycal_similarity = 0
        self.TF_ISF_IDF = 0
        self.text_rank = 0
        self.sent_length = 0
        self.pos_keywords = 0
        self.neg_keywords = 0
        self.busy_path = 0
        self.aggregate_simm = 0
        self.word_simm_sents = 0
        self.word_simm_par = 0
        self.IQS = 0
        self.thematic_features = 0
        self.named_entities = 0

    def Print(self):
        for key in self.__dict__:
            print(key, '\n')


class Sentence():
    def __init__(self, sent):
        self.tokenized = sent
        self.scores = Scores()

    def Print_Sentence(self):  # Only for debug purposes -> too much verbose
        print([self.tokenized])

    def Print_Scores(self):
        self.scores.Print()

    def info(self, verbose=True):
        if verbose:
            print('Tokens in sentence: {}'.format(len(self.tokenized)))
        return len(self.tokenized)


class Document():
    def __init__(self, doc, doc_id, high):
        self.sentences = {}
        self.highlights = None

        for sentence in doc:  # Must be pre-processed by spacy pipeline
            if len(sentence) > 0:  # Removes punctuation only lines
                self.add_sentence(sentence, doc_id)
                self.add_highlights(high)

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
            sent_id = doc_id + '_{}'.format(len(self.sentences))
            self.sentences[sent_id] = Sentence(sent)

    def add_highlights(self, high):
        if not isinstance(high, str):
            print('Input type must be \'string\'')
            return
        else:
            self.highlights = high

    def info(self, verbose=True):
        tot_tokens = 0
        num_sents = len(self.sentences)
        num_high = len(self.highlights.split('.'))
        for key in self.sentences:
            tot_tokens += self.sentences[key].info(verbose=False)
        av_tokens = tot_tokens/num_sents

        if verbose:
            print('Total tokens in document: {}\n'
                  'Average token per document: {:0.2f}\n'
                  'Total sentences: {}\n'
                  'Sentences in highlights: {}\n'
                  .format(tot_tokens, av_tokens, num_sents, num_high))
        return {'tot_tokens': tot_tokens, 'av_tokens': av_tokens,
                'num_sents': num_sents, 'num_high': num_high}


class Dataset():
    def __init__(self, name='Processed_dataset.json'):
        self.documents = {}
        self.name = name
        if '.json' not in name:
            name += '.json'

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

    def saveToDisk(self, pathToFile=None):
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
        self.documents = loaded_dataset

    def process_dataset(self, dataset_in, save=True):

        nlp = spacy.load('en_core_web_sm')  # Loads pipeline for english

        # Generating tokenized structure for feature evaluation

        i = 0
        with tqdm(total=len(dataset_in)) as pbar:
            for key in dataset_in:
                pbar.set_description('processing dataset: ')
                doc_id = key['id']
                high = key['highlights']

                tokenized_article = nlp(key['article'])
                segmented_document = []
                for sentence in tokenized_article.sents:
                    tokenized_sent = []
                    for token in sentence:
                        if not token.is_punct:
                            tokenized_sent.append(token.text)  # Try with lemma
                    segmented_document.append(tokenized_sent)
                self.add_document(segmented_document, doc_id, high)

                pbar.update(1)
                if i == 2:
                    break
                i += 1
        pbar.close()
        print('\n'*3)

    def info(self, verbose=True):
        if verbose:
            print('Dataset name: {}\n'
                  'Documents in dataset: {}\n'
                  .format(self.name, len(self.documents)))
            for doc_id in self.documents:
                print('-'*80, '\nDocument ID: {}'.format(doc_id))
                self.documents[doc_id].info()


if __name__ == '__main__':
    CNN_dataset = load_dataset('cnn_dailymail', '3.0.0')
    CNN_processed = Dataset(name='CNN_processed.json')
    CNN_processed.process_dataset(CNN_dataset['train'])
    CNN_processed.info()
