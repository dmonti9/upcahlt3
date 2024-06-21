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
        emb_lw = 100
        emb_features = 5

        total_dim = emb_words + emb_sufs + emb_lw + emb_features

        self.embW = nn.Embedding(n_words, emb_words)
        self.embS = nn.Embedding(n_sufs, emb_sufs)
        self.embLW = nn.Embedding(n_lcwords, emb_lw)

        self.dropoutW = nn.Dropout(0.2)
        self.dropoutS = nn.Dropout(0.2)
        self.dropoutLW = nn.Dropout(0.2)

        hidden_size = total_dim // 2
        self.lstm = nn.LSTM(
            total_dim, hidden_size, num_layers=2, bidirectional=True, batch_first=True
        )

        # Multi-head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            batch_first=True,
        )

        self.selu = nn.SELU()

        lstm_output_dim = hidden_size * 2
        self.fc1 = nn.Linear(lstm_output_dim, n_labels)
        self.fc2 = nn.Linear(n_labels, n_labels)

    def forward(self, words, suffixes, lcwords, features):

        emb_words = self.dropoutW(self.embW(words))
        emb_sufs = self.dropoutS(self.embS(suffixes))
        emb_lcwords = self.dropoutLW(self.embLW(lcwords))

        x = torch.cat((emb_words, emb_sufs, emb_lcwords, features), dim=2)

        x, _ = self.lstm(x)

        attn_output, attn_output_weights = self.attention(x, x, x)
        x = attn_output

        x = self.selu(self.fc1(x))
        x = self.fc2(x)

        return x
