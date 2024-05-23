import os
import string
import re
import torch

from dataset import *


class Codemaps:
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None):

        if isinstance(data, Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen)

            # new external list
            # self.external_list = self.create_external_list()

        elif type(data) == str and maxlen is None and suflen is None:
            self.__load(data)

        else:
            print("codemaps: Invalid or missing parameters in constructor")
            exit()

    # NEW
    def create_external_list(self):
        external = {}
        with open("../resources/HSDB.txt", encoding="utf-8") as h:
            for x in h.readlines():
                external[x.strip().lower()] = "drug"
        with open("../resources/DrugBank.txt", encoding="utf-8") as h:
            for x in h.readlines():
                (n, t) = x.strip().lower().split("|")
                external[n] = t
        return external

    # NEW

    # GO BACK TO PROVIDED VERSION; ADJUST LESS !!!

    def encode_features(self, data):
        # Extract features for each word in each sentence
        enc_features = []
        for s in data.sentences():
            if not s:  # Check if the sentence is empty
                sentence_features = torch.zeros(
                    (1, 1), dtype=torch.int64
                )  # Create a placeholder zero vector
            else:
                sentence_features = torch.tensor(
                    [
                        [
                            int(word["form"][0].isupper()),  # Capitalization
                            # int("-" in word["form"]),  # Dash presence
                            # int(
                            #    any(char.isdigit() for char in word["form"])
                            #  ),  # Digit presence
                            # int(
                            #    word["form"].lower() in self.external_list
                            # ),  # External list presence
                        ]
                        for word in s
                    ],
                    dtype=torch.int64,
                )
            enc_features.append(sentence_features)

        # Create a tensor full of zeros for padding, with an extra dimension for features
        num_sentences = len(enc_features)
        num_features = 1  # Update this based on the number of features you are using
        Xf = torch.zeros((num_sentences, self.maxlen, num_features), dtype=torch.int64)

        # Fill the padding tensor with sentence feature data
        for i, features in enumerate(enc_features):
            actual_length = features.size(0)
            if actual_length > self.maxlen:
                features = features[
                    : self.maxlen
                ]  # Truncate features if longer than maxlen
            Xf[i, :actual_length] = features

        return Xf

    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen):

        self.maxlen = maxlen
        self.suflen = suflen
        words = set([])
        sufs = set([])
        labels = set([])

        for s in data.sentences():
            for t in s:
                words.add(t["form"])
                sufs.add(t["lc_form"][-self.suflen :])
                labels.add(t["tag"])

        self.word_index = {w: i + 2 for i, w in enumerate(list(words))}
        self.word_index["PAD"] = 0  # Padding
        self.word_index["UNK"] = 1  # Unknown words

        self.suf_index = {s: i + 2 for i, s in enumerate(list(sufs))}
        self.suf_index["PAD"] = 0  # Padding
        self.suf_index["UNK"] = 1  # Unknown suffixes

        self.label_index = {t: i + 1 for i, t in enumerate(list(labels))}
        self.label_index["PAD"] = 0  # Padding

    ## --------- load indexs -----------
    def __load(self, name):
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.label_index = {}

        with open(name + ".idx") as f:
            for line in f.readlines():
                (t, k, i) = line.split()
                if t == "MAXLEN":
                    self.maxlen = int(k)
                elif t == "SUFLEN":
                    self.suflen = int(k)
                elif t == "WORD":
                    self.word_index[k] = int(i)
                elif t == "SUF":
                    self.suf_index[k] = int(i)
                elif t == "LABEL":
                    self.label_index[k] = int(i)

    ## ---------- Save model and indexs ---------------
    def save(self, name):
        # save indexes
        with open(name + ".idx", "w") as f:
            print("MAXLEN", self.maxlen, "-", file=f)
            print("SUFLEN", self.suflen, "-", file=f)
            for key in self.label_index:
                print("LABEL", key, self.label_index[key], file=f)
            for key in self.word_index:
                print("WORD", key, self.word_index[key], file=f)
            for key in self.suf_index:
                print("SUF", key, self.suf_index[key], file=f)

    ## --------- encode X from given data -----------
    def encode_words(self, data):

        # encode sentence words
        enc = [
            torch.Tensor(
                [
                    (
                        self.word_index[w["form"]]
                        if w["form"] in self.word_index
                        else self.word_index["UNK"]
                    )
                    for w in s
                ]
            )
            for s in data.sentences()
        ]
        # cut sentences longer than maxlen
        enc = [s[0 : self.maxlen] for s in enc]
        # create a tensor full of padding
        tsr = torch.Tensor([])
        Xw = tsr.new_full(
            (len(enc), self.maxlen), self.word_index["PAD"], dtype=torch.int64
        )
        # fill padding tensor with sentence data
        for i, s in enumerate(enc):
            Xw[i, 0 : s.size()[0]] = s

        # encode sentence suffixes
        enc = [
            torch.Tensor(
                [
                    (
                        self.suf_index[w["lc_form"][-self.suflen :]]
                        if w["lc_form"][-self.suflen :] in self.suf_index
                        else self.suf_index["UNK"]
                    )
                    for w in s
                ]
            )
            for s in data.sentences()
        ]
        # cut sentences longer than maxlen
        enc = [s[0 : self.maxlen] for s in enc]
        # create a tensor full of padding
        tsr = torch.Tensor([])
        Xs = tsr.new_full(
            (len(enc), self.maxlen), self.suf_index["PAD"], dtype=torch.int64
        )
        # fill padding tensor with sentence data
        for i, s in enumerate(enc):
            Xs[i, 0 : s.size()[0]] = s

        # cut sentences longer than maxlen
        enc = [s[0 : self.maxlen] for s in enc]
        # create a tensor full of zeros
        # Xf = torch.zeros((len(enc), self.maxlen, 11), dtype=torch.int64)
        # fill padding tensor with sentence data
        # for i, s in enumerate(enc):
        #   for j, f in enumerate(enc[i]) :
        #        Xf[i, j] = f

        Xf = self.encode_features(data)
        # return encoded sequences
        # return [Xlw,Xw,Xs,Xf]
        return [Xw, Xs, Xf]

    ## --------- encode Y from given data -----------
    def encode_labels(self, data):
        # encode and pad sentence labels
        enc = [
            torch.Tensor([self.label_index[w["tag"]] for w in s])
            for s in data.sentences()
        ]
        # cut sentences longer than maxlen
        enc = [s[0 : self.maxlen] for s in enc]
        # create a tensor full of padding
        tsr = torch.Tensor([])
        Y = tsr.new_full(
            (len(enc), self.maxlen), self.label_index["PAD"], dtype=torch.int64
        )
        # fill padding tensor with sentence data
        for i, s in enumerate(enc):
            Y[i, 0 : s.size()[0]] = s

        return Y

    ## -------- get word index size ---------
    def get_n_words(self):
        return len(self.word_index)

    ## -------- get suf index size ---------
    def get_n_sufs(self):
        return len(self.suf_index)

    ## -------- get label index size ---------
    def get_n_labels(self):
        return len(self.label_index)

    ## -------- get index for given word ---------
    def word2idx(self, w):
        return self.word_index[w]

    ## -------- get index for given suffix --------
    def suff2idx(self, s):
        return self.suff_index[s]

    ## -------- get index for given label --------
    def label2idx(self, l):
        return self.label_index[l]

    ## -------- get label name for given index --------
    def idx2label(self, i):
        for l in self.label_index:
            if self.label_index[l] == i:
                return l
        raise KeyError
