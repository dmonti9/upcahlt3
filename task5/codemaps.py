import os
import string
import re
import torch

from dataset import *


class Codemaps:
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None):
        self.get_external()
        if isinstance(data, Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen)

            # new external list
            # self.external_list = self.create_external_list()

        elif type(data) == str and maxlen is None and suflen is None:
            self.__load(data)

        else:
            print("codemaps: Invalid or missing parameters in constructor")
            exit()

    def get_external(self):
        self.external = {}
        with open("../resources/HSDB.txt") as h:
            for x in h.readlines():
                self.external[x.strip().lower()] = "drug"
        with open("../resources/DrugBank.txt") as h:
            for x in h.readlines():
                (n, t) = x.strip().lower().split("|")
                self.external[n] = t

    def encode_features(self, sentence, maxlen, external_index, external):
        features = torch.zeros(
            (maxlen, 7), dtype=torch.int64
        )  # Adjust number of features!!!!

        for i, word in enumerate(sentence):
            if i >= maxlen:
                break
            # Existing features
            is_capitalized = 1 if word["form"][0].isupper() else 0
            contains_dash = 1 if "-" in word["form"] else 0
            contains_numbers = 1 if any(char.isdigit() for char in word["form"]) else 0
            word_length = len(word["form"])

            # External classification feature
            lc_form = word["lc_form"]
            external_feature = (
                1
                if lc_form in external_index and external[lc_form] == "drug"
                else (
                    2
                    if lc_form in external_index and external[lc_form] == "brand"
                    else (
                        3
                        if lc_form in external_index and external[lc_form] == "drug_n"
                        else (
                            4
                            if lc_form in external_index
                            and external[lc_form] == "group"
                            else 0
                        )
                    )
                )
            )

            # New features
            position_in_sentence = i
            distance_from_start = word["start"]
            is_start = 1 if i == 0 else 0
            is_end = 1 if i == len(sentence) - 1 else 0
            percent_uppercase = sum(1 for c in word["form"] if c.isupper()) / len(
                word["form"]
            )

            # Collect all features into a tensor
            features[i] = torch.tensor(
                [
                    is_capitalized,
                    contains_dash,
                    contains_numbers,
                    word_length,
                    external_feature,
                    position_in_sentence,
                    percent_uppercase,
                ],
                dtype=torch.int64,
            )

        return features

    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen):

        self.maxlen = maxlen
        self.suflen = suflen
        words = set([])
        sufs = set([])
        labels = set([])
        lwords = set([])

        for s in data.sentences():
            for t in s:
                words.add(t["form"])
                sufs.add(t["lc_form"][-self.suflen :])
                labels.add(t["tag"])
                lwords.add(t["lc_form"])

        self.word_index = {w: i + 2 for i, w in enumerate(list(words))}
        self.word_index["PAD"] = 0  # Padding
        self.word_index["UNK"] = 1  # Unknown words

        self.suf_index = {s: i + 2 for i, s in enumerate(list(sufs))}
        self.suf_index["PAD"] = 0  # Padding
        self.suf_index["UNK"] = 1  # Unknown suffixes

        self.label_index = {t: i + 1 for i, t in enumerate(list(labels))}
        self.label_index["PAD"] = 0  # Padding

        self.lword_index = {w: i + 2 for i, w in enumerate(list(lwords))}
        self.lword_index["PAD"] = 0  # Padding
        self.lword_index["UNK"] = 1  # Unknown words

        externals = set(self.external.keys())
        self.external_index = {e: i + 2 for i, e in enumerate(list(externals))}
        self.external_index["PAD"] = 0  # Padding
        self.external_index["UNK"] = 1  # Unknown external entries

    ## --------- load indexs -----------
    def __load(self, name):
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.label_index = {}
        self.lword_index = {}
        self.external_index = {}

        with open(name + ".idx") as f:
            for line in f.readlines():

                parts = line.split()
                t = parts[0]  # The tag is always the first part
                i = parts[-1]  # The integer is always the last part
                k = " ".join(
                    parts[1:-1]
                )  # The key is everything between the tag and the integer

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
                elif t == "LWORD":
                    self.lword_index[k] = int(i)

                elif t == "EXT":
                    self.external_index[k] = int(i)

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

            for key in self.lword_index:
                print("LWORD", key, self.lword_index[key], file=f)

            for key in self.external_index:
                print("EXT", key, self.external_index[key], file=f)

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

        # encode lowercase words
        enc = [
            torch.Tensor(
                [
                    (
                        self.lword_index[w["lc_form"]]
                        if w["lc_form"] in self.lword_index
                        else self.lword_index["UNK"]
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
        Xlw = tsr.new_full(
            (len(enc), self.maxlen), self.lword_index["PAD"], dtype=torch.int64
        )
        # fill padding tensor with sentence data
        for i, s in enumerate(enc):
            Xlw[i, 0 : s.size()[0]] = s

        ################## Features

        # Encode additional features
        Xf = [
            self.encode_features(s, self.maxlen, self.external_index, self.external)
            for s in data.sentences()
        ]

        # Return encoded sequences
        return [Xw, Xs, Xlw, torch.stack(Xf)]

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

    ## -------- get lc word index size ---------

    def get_n_lcwords(self):
        return len(self.lword_index)

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

    def get_n_exts(self):
        return len(self.external_index)
