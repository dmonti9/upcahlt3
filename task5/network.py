import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn import LayerNorm


criterion = nn.CrossEntropyLoss()


class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_lcwords = codes.get_n_lcwords()
        n_labels = codes.get_n_labels()

        # Embeddings
        emb_words = 100
        emb_sufs = 100
        emb_lw = 50
        emb_features = 7

        # Calculate total dimension for LSTM input: word embeddings + suffix embeddings + lcwords embeddings + features
        total_dim = emb_words + emb_sufs + emb_lw + emb_features

        self.embW = nn.Embedding(n_words, emb_words)  # Embeddings for words
        self.embS = nn.Embedding(n_sufs, emb_sufs)  # Embeddings for suffixes
        self.embLW = nn.Embedding(n_lcwords, emb_lw)  # Embeddings for lowercased words

        # Dropout layers for embeddings
        self.dropoutW = nn.Dropout(0.3)  # Dropout for word embeddings
        self.dropoutS = nn.Dropout(0.3)  # Dropout for suffix embeddings
        self.dropoutLW = nn.Dropout(0.3)  # Dropout for lowercase word embeddings

        # LSTM layer configuration
        hidden_size = total_dim // 2
        self.lstm = nn.LSTM(
            total_dim,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        # ADJUSTMENTS
        self.lstm_norm = LayerNorm(hidden_size * 2)  # After LSTM and before FC

        self.selu = nn.SELU()

        # Output layer dimensions adjustment due to bidirectional LSTM
        lstm_output_dim = hidden_size * 2
        self.fc1 = nn.Linear(lstm_output_dim, n_labels)  # Processing layer
        self.fc2 = nn.Linear(
            n_labels, n_labels
        )  # Assuming classification task, output size matches label count

    def forward(self, words, suffixes, lcwords, features):
        # Get embeddings from input indices
        emb_words = self.dropoutW(self.embW(words))
        emb_sufs = self.dropoutS(self.embS(suffixes))
        emb_lcwords = self.dropoutLW(self.embLW(lcwords))

        # Check feature dimension and concatenate all inputs
        x = torch.cat((emb_words, emb_sufs, emb_lcwords, features), dim=2)

        x, _ = self.lstm(x)
        x = self.lstm_norm(x)  # Normalize before passing to the fully connected layer

        # Process LSTM output through fully connected layers
        x = self.selu(self.fc1(x))
        x = self.fc2(x)  # Pass through the second FC layer to get final predictions

        return x
