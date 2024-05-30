import torch
import torch.nn as nn
import torch.nn.functional as func


criterion = nn.CrossEntropyLoss()


class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_lcwords = codes.get_n_lcwords()
        n_labels = codes.get_n_labels()
        # Embeddings
        emb_words=100
        emb_sufs=50
        emb_lw=50
        emb_features=4
        total_dim=emb_words+emb_sufs+emb_lw+emb_features
        self.embW = nn.Embedding(n_words, emb_words)  # Embeddings for words
        self.embS = nn.Embedding(n_sufs, emb_sufs)  # Embeddings for suffixes
        self.embLW = nn.Embedding(n_lcwords, emb_lw)  # Embeddings for lowercased words

        # LSTM layer
        self.lstm = nn.LSTM(total_dim, total_dim//2,2, bidirectional=True, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(total_dim, total_dim)  # Process LSTM outputs
        self.fc2 = nn.Linear(total_dim, n_labels)  # Final output layer

    def forward(self, words, suffixes, lcwords, features):
        # Embeddings
        features=features.float()
        emb_words = self.embW(words)
        emb_sufs = self.embS(suffixes)
        emb_lcwords = self.embLW(lcwords)
        # Concatenate all embeddings and transformed features
        x = torch.cat((emb_words, emb_sufs, emb_lcwords, features), dim=2)

        # LSTM processing
        x, _ = self.lstm(x)

        # Fully connected layers
        x = func.relu(self.fc1(x))
        x = self.fc2(x)

        return x
