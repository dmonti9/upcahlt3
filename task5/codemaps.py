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

    def encode_features(self, sentence, maxlen, external_index, external):
        features = torch.zeros(
            (maxlen, 7), dtype=torch.int64
        )  # Prepare tensor for features

        sentence_length = len(sentence)

        for i, word in enumerate(sentence):
            if i >= maxlen:
                break
            # Feature 1: Is capitalized
            is_capitalized = 1 if word["form"][0].isupper() else 0

            # Feature 2: Contains dash
            contains_dash = 1 if "-" in word["form"] else 0

            # Feature 3: Contains numbers
            contains_numbers = 1 if any(char.isdigit() for char in word["form"]) else 0

            # Feature 4: Word length
            word_length = len(word["form"])
            
            # Feature 5: Contains special characters
            contains_special = 1 if any(not char.isalnum() for char in word["form"]) else 0

            # Feature 6: External classification
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

            # Feature 7: Relative position
            relative_position = i / sentence_length

            # Collect all features into a tensor
            features[i] = torch.tensor(
                [
                    is_capitalized,
                    contains_dash,
                    contains_special,
                    contains_numbers,
                    word_length,
                    external_feature,
                    relative_position,
                ],
                dtype=torch.float32,  # Use float32 to accommodate relative_position
            )

        return features

    # --------- Create indexes from training data
    # Extract all words and labels in given sentences and
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen):
        self.maxlen = maxlen
        self.suflen = suflen
        self.preflen = 3  # Assuming preflen is intended to be 3
        words = set([])
        sufs = set([])
        prefs = set([])  # This should be for prefixes
        labels = set([])
        lwords = set([])
        bigrams = set([])
        trigrams = set([])
        preceding_bigrams = set([])

        for s in data.sentences():
            for t in s:
                words.add(t["form"])
                sufs.add(t["lc_form"][-self.suflen:])
                prefs.add(t["lc_form"][:self.preflen])  # Collecting prefixes
                labels.add(t["tag"])
                lwords.add(t["lc_form"])

            for i in range(len(s) - 1):
                bigrams.add(s[i + 1]["form"])
                if i > 0:
                    preceding_bigrams.add(s[i - 1]["form"])
            for i in range(len(s) - 2):
                trigrams.add((s[i]["form"], s[i + 1]["form"], s[i + 2]["form"]))

        self.bigram_index = {bg: i + 2 for i, bg in enumerate(list(bigrams))}
        self.bigram_index["PAD"] = 0
        self.bigram_index["UNK"] = 1

        self.trigram_index = {tg: i + 2 for i, tg in enumerate(list(trigrams))}
        self.trigram_index["PAD"] = 0
        self.trigram_index["UNK"] = 1

        self.preceding_bigram_index = {pbg: i + 2 for i, pbg in enumerate(list(preceding_bigrams))}
        self.preceding_bigram_index["PAD"] = 0
        self.preceding_bigram_index["UNK"] = 1

        self.word_index = {w: i + 2 for i, w in enumerate(list(words))}
        self.word_index["PAD"] = 0  # Padding
        self.word_index["UNK"] = 1  # Unknown words

        self.suf_index = {s: i + 2 for i, s in enumerate(list(sufs))}
        self.suf_index["PAD"] = 0  # Padding
        self.suf_index["UNK"] = 1  # Unknown suffixes

        self.pref_index = {p: i + 2 for i, p in enumerate(list(prefs))}  # Correctly using prefixes
        self.pref_index["PAD"] = 0  # Padding
        self.pref_index["UNK"] = 1  # Unknown prefixes  # Ensuring UNK is added

        self.label_index = {t: i + 1 for i, t in enumerate(list(labels))}
        self.label_index["PAD"] = 0  # Padding

        self.lword_index = {w: i + 2 for i, w in enumerate(list(lwords))}
        self.lword_index["PAD"] = 0  # Padding
        self.lword_index["UNK"] = 1  # Unknown words

        externals = set(self.external.keys())
        self.external_index = {e: i + 2 for i, e in enumerate(list(externals))}
        self.external_index["PAD"] = 0  # Padding
        self.external_index["UNK"] = 1  # Unknown external entries

    ## --------- load indexes -----------
    def __load(self, name):
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.pref_index = {}
        self.label_index = {}
        self.lword_index = {}
        self.external_index = {}
        self.bigram_index = {}
        self.trigram_index = {}
        self.preceding_bigram_index = {}

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
                elif t == "PREFLEN":
                    self.preflen = int(k)
                elif t == "WORD":
                    self.word_index[k] = int(i)
                elif t == "SUF":
                    self.suf_index[k] = int(i)
                elif t == "LABEL":
                    self.label_index[k] = int(i)
                elif t == "LWORD":
                    self.lword_index[k] = int(i)
                elif t == "PREF":
                    self.pref_index[k] = int(i)
                elif t == "BIGRAM":
                    self.bigram_index[k] = int(i)
                elif t == "TRIGRAM":
                    self.trigram_index[k] = int(i)
                elif t == "PBG":
                    self.preceding_bigram_index[k] = int(i)
                elif t == "EXT":
                    self.external_index[k] = int(i)

    ## ---------- Save model and indexes ---------------
    def save(self, name):
        # save indexes
        with open(name + ".idx", "w") as f:
            print("MAXLEN", self.maxlen, "-", file=f)
            print("SUFLEN", self.suflen, "-", file=f)
            print("PREFLEN", self.preflen, "-", file=f)
            for key in self.label_index:
                print("LABEL", key, self.label_index[key], file=f)
            for key in self.word_index:
                print("WORD", key, self.word_index[key], file=f)
            for key in self.suf_index:
                print("SUF", key, self.suf_index[key], file=f)
            for key in self.pref_index:
                print("PREF", key, self.pref_index[key], file=f)
            for key in self.lword_index:
                print("LWORD", key, self.lword_index[key], file=f)
            for key in self.bigram_index:
                print("BIGRAM", key, self.bigram_index[key], file=f)
            for key in self.trigram_index:
                print("TRIGRAM", key, self.trigram_index[key], file=f)
            for key in self.preceding_bigram_index:
                print("PBG", key, self.preceding_bigram_index[key], file=f)
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

        # encode bigrams
        bigrams = [
            torch.Tensor(
                [
                    (
                        self.bigram_index[s[i + 1]["form"]]
                        if (s[i + 1]["form"]) in self.bigram_index
                        else self.bigram_index["UNK"]
                    )
                    for i in range(len(s) - 1)
                ]
            )
            for s in data.sentences()
        ]
        bigrams = [s[0 : self.maxlen] for s in bigrams]
        tsr = torch.Tensor([])
        Xb = tsr.new_full(
            (len(bigrams), self.maxlen), self.bigram_index["PAD"], dtype=torch.int64
        )
        for i, s in enumerate(bigrams):
            Xb[i, 0 : s.size()[0]] = s

        # encode preceding bigrams
        preceding_bigrams = [
            torch.Tensor(
                [
                    (
                        self.preceding_bigram_index[s[i - 1]["form"]]
                        if s[i - 1]["form"] in self.preceding_bigram_index and i > 0
                        else self.preceding_bigram_index["UNK"]
                    )
                    for i in range(len(s))
                ]
            )
            for s in data.sentences()
        ]
        preceding_bigrams = [s[0 : self.maxlen] for s in preceding_bigrams]
        tsr = torch.Tensor([])
        Xpb = tsr.new_full(
            (len(preceding_bigrams), self.maxlen), self.preceding_bigram_index["PAD"], dtype=torch.int64
        )
        for i, s in enumerate(preceding_bigrams):
            Xpb[i, 0 : s.size()[0]] = s

        # encode sentence prefixes
        enc = [
            torch.Tensor(
                [
                    (
                        self.pref_index[w["lc_form"][:self.preflen]]
                        if w["lc_form"][:self.preflen] in self.pref_index
                        else self.pref_index["UNK"]
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
        Xp = tsr.new_full(
            (len(enc), self.maxlen), self.pref_index["PAD"], dtype=torch.int64
        )
        # fill padding tensor with sentence data
        for i, s in enumerate(enc):
            Xp[i, 0 : s.size()[0]] = s

        # encode trigrams
        trigrams = [
            torch.Tensor(
                [
                    (
                        self.trigram_index[(s[i]["form"], s[i + 1]["form"], s[i + 2]["form"])]
                        if (s[i]["form"], s[i + 1]["form"], s[i + 2]["form"]) in self.trigram_index
                        else self.trigram_index["UNK"]
                    )
                    for i in range(len(s) - 2)
                ]
            )
            for s in data.sentences()
        ]
        trigrams = [s[0 : self.maxlen] for s in trigrams]
        tsr = torch.Tensor([])
        Xt = tsr.new_full(
            (len(trigrams), self.maxlen), self.trigram_index["PAD"], dtype=torch.int64
        )
        for i, s in enumerate(trigrams):
            Xt[i, 0 : s.size()[0]] = s

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
        return [Xw, Xs, Xlw, torch.stack(Xf), Xp, Xb, Xt, Xpb]

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

    ## -------- get pref index size ---------
    def get_n_prefs(self):
        return len(self.pref_index)
    
    ## -------- get bigram index size ---------
    def get_n_bigrams(self):
        return len(self.bigram_index)

    ## -------- get trigram index size ---------
    def get_n_trigrams(self):
        return len(self.trigram_index)

    ## -------- get preceding bigram index size ---------
    def get_n_preceding_bigrams(self):
        return len(self.preceding_bigram_index)
    
    ## -------- get label index size ---------
    def get_n_labels(self):
        return len(self.label_index)

    ## -------- get index for given word ---------
    def word2idx(self, w):
        return self.word_index[w]

    ## -------- get index for given suffix --------
    def suff2idx(self, s):
        return self.suf_index[s]

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
