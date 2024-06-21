import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

criterion = nn.CrossEntropyLoss()


class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        # Dimensions
        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_lcwords = codes.get_n_lcwords()
        n_labels = codes.get_n_labels()

        # Embeddings / Features
        emb_words = 100
        emb_sufs = 100
        emb_lw = 50
        emb_features = 7

        # Total dimension for LSTM input
        total_dim = emb_words + emb_sufs + emb_lw + emb_features

        # Embedding layers
        self.embW = nn.Embedding(n_words, emb_words)
        self.embS = nn.Embedding(n_sufs, emb_sufs)
        self.embLW = nn.Embedding(n_lcwords, emb_lw)

        # Dropout layers for embeddings
        self.dropoutW = nn.Dropout(0.3)
        self.dropoutS = nn.Dropout(0.3)
        self.dropoutLW = nn.Dropout(0.3)

        # CNN layer on concatenated embeddings
        self.cnn = nn.Conv1d(
            in_channels=total_dim, out_channels=total_dim // 2, kernel_size=3, padding=1
        )
        self.cnn_activation = nn.ReLU()

        # LSTM configuration
        hidden_size = total_dim // 4  # Adjusted due to dimension change after CNN
        self.lstm = nn.LSTM(
            total_dim // 2,  # Adjusted input size
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        # LSTM Normalization
        self.lstm_norm = LayerNorm(hidden_size * 2)  # After LSTM and before FC

        # Activation
        self.selu = nn.SELU()

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, n_labels)
        self.fc2 = nn.Linear(n_labels, n_labels)

    def forward(self, words, suffixes, lcwords, features):
        # Get embeddings from input indices
        emb_words = self.dropoutW(self.embW(words))
        emb_sufs = self.dropoutS(self.embS(suffixes))
        emb_lcwords = self.dropoutLW(self.embLW(lcwords))

        # Concatenate all inputs
        x = torch.cat((emb_words, emb_sufs, emb_lcwords, features), dim=2)
        x = x.permute(
            0, 2, 1
        )  # Rearrange dimensions for CNN (batch_size, channels, sequence_length)

        # Apply CNN and activation
        x = self.cnn_activation(self.cnn(x))
        x = x.permute(0, 2, 1)  # Switch back dimensions to match LSTM input requirement

        # Feed into LSTM
        x, _ = self.lstm(x)
        x = self.lstm_norm(x)

        # Process LSTM output through fully connected layers
        x = self.selu(self.fc1(x))
        x = self.fc2(x)

        return x
