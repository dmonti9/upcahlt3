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
        self.embW = nn.Embedding(n_words, 100)  # Embeddings for words
        self.embS = nn.Embedding(n_sufs, 50)  # Embeddings for suffixes
        self.embLW = nn.Embedding(n_lcwords, 50)  # Embeddings for lowercased words

        # Dropout layers
        self.dropW = nn.Dropout(0.1)
        self.dropS = nn.Dropout(0.1)
        self.dropLW = nn.Dropout(0.1)

        # Feature transformation
        self.feature_transform = nn.Linear(
            4, 50
        )  # Transform features to have the same dimensionality

        # LSTM layer
        self.lstm = nn.LSTM(250, 200, bidirectional=True, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(400, 300)  # Process LSTM outputs
        self.fc2 = nn.Linear(300, n_labels)  # Final output layer

    def forward(self, words, suffixes, lcwords, features):
        # Embeddings
        emb_words = self.embW(words)
        emb_sufs = self.embS(suffixes)
        emb_lcwords = self.embLW(lcwords)

        # Apply dropout
        emb_words = self.dropW(emb_words)
        emb_sufs = self.dropS(emb_sufs)
        emb_lcwords = self.dropLW(emb_lcwords)

        # Transform features
        transformed_features = func.relu(self.feature_transform(features))

        # Concatenate all embeddings and transformed features
        x = torch.cat((emb_words, emb_sufs, emb_lcwords, transformed_features), dim=2)

        # LSTM processing
        x, _ = self.lstm(x)

        # Fully connected layers
        x = func.relu(self.fc1(x))
        x = self.fc2(x)

        return x
