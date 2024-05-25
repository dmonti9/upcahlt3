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

    def encode_external(self, word):
        return self.external_index.get(word.lower(), self.external_index["UNK"])

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

        # Features, TODOOO

        enc = []
        for s in data.sentences():
            cap = torch.Tensor([1 if w["form"].istitle() else 0 for w in s]).unsqueeze(
                1
            )
            dash = torch.Tensor([1 if "-" in w["form"] else 0 for w in s]).unsqueeze(1)
            num = torch.Tensor(
                [1 if any(char.isdigit() for char in w["form"]) else 0 for w in s]
            ).unsqueeze(1)
            ext = torch.Tensor(
                [1 if w["lc_form"] in self.external_index else 0 for w in s]
            ).unsqueeze(1)
            features = torch.cat((cap, dash, num, ext), dim=1)
            enc.append(features)

        # Cut sentences longer than maxlen
        enc = [e[: self.maxlen] for e in enc]
        # Create a tensor full of zeros
        Xf = torch.zeros((len(enc), self.maxlen, 4), dtype=torch.int64)

        # Fill the padding tensor with sentence data
        for i, s in enumerate(enc):
            end_idx = s.shape[0]  # Find the actual length of the sentence features
            Xf[i, :end_idx, :] = s  # Assign the sentence's features to the Xf tensor

        # Return encoded sequences
        return [Xw, Xs, Xlw, Xf]

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

    def get_n_exts(self):
        return len(self.external_index)
