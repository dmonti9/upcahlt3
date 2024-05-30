import string
import re
import torch

from dataset import *
from transformers import BertTokenizer, BertModel  # BertModel is needed if you decide to use BioBERT embeddings directly

class Codemaps:
    def __init__(self, data, maxlen=None, pretrained_model_name='dmis-lab/biobert-v1.1'):
        # Initialize the tokenizer from BioBERT
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        if isinstance(data, Dataset) and maxlen is not None:
            self.__create_indexs(data, maxlen)
        elif type(data) == str and maxlen is None:
            self.__load(data)
        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()

    def __create_indexs(self, data, maxlen):
        # Index creation as before
        self.maxlen = maxlen
        words = set([])
        lc_words = set([])
        lems = set([])
        pos = set([])
        labels = set([])
        
        for s in data.sentences():
            for t in s['sent']:
                words.add(t['form'])
                lc_words.add(t['lc_form'])
                lems.add(t['lemma'])
                pos.add(t['pos'])
            labels.add(s['type'])

        # Create indexes for each feature type
        self.word_index = {w: i+2 for i, w in enumerate(sorted(list(words)))}
        self.word_index['PAD'] = 0
        self.word_index['UNK'] = 1

        self.lc_word_index = {w: i+2 for i, w in enumerate(sorted(list(lc_words)))}
        self.lc_word_index['PAD'] = 0
        self.lc_word_index['UNK'] = 1

        self.lemma_index = {s: i+2 for i, s in enumerate(sorted(list(lems)))}
        self.lemma_index['PAD'] = 0
        self.lemma_index['UNK'] = 1

        self.pos_index = {s: i+2 for i, s in enumerate(sorted(list(pos)))}
        self.pos_index['PAD'] = 0
        self.pos_index['UNK'] = 1

        self.label_index = {t: i for i, t in enumerate(sorted(list(labels)))}

    def __tokenize_for_bert(self, sentences):
        # Tokenize with BioBERT
        inputs = self.tokenizer(sentences, padding='max_length', truncation=True, max_length=self.maxlen, return_tensors='pt')
        return inputs['input_ids'], inputs['attention_mask']

    def encode_words(self, data):
        # Convert sentences to strings
        sentences = [" ".join([t['form'] for t in s['sent']]) for s in data.sentences()]
        input_ids, attention_mask = self.__tokenize_for_bert(sentences)
        
        # Use existing encode and pad methods
        Xw = self.__encode_and_pad(data, self.word_index, 'form')
        Xlw = self.__encode_and_pad(data, self.lc_word_index, 'lc_form')
        Xl = self.__encode_and_pad(data, self.lemma_index, 'lemma')
        Xp = self.__encode_and_pad(data, self.pos_index, 'pos')  
        
        return [Xw, Xp, Xlw, Xl, input_ids, attention_mask]

    def encode_labels(self, data):
        labels = [[1 if i == self.label_index[s['type']] else 0 for i in range(len(self.label_index))] for s in data.sentences()]
        Y = torch.Tensor(labels)
        return Y

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get word index size ---------
    def get_n_lc_words(self) :
        return len(self.lc_word_index)
    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)
    ## -------- get label index size ---------
    def get_n_lemmas(self) :
        return len(self.lemma_index)
    ## -------- get label index size ---------
    def get_n_pos(self) :
        return len(self.pos_index)

    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given word ---------
    def lcword2idx(self, w) :
        return self.lc_word_index[w]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError
