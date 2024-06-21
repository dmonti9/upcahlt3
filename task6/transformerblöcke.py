import torch
import torch.nn as nn
import torch.nn.functional as F
criterion = nn.CrossEntropyLoss()

class ddiCNN(nn.Module):
    def __init__(self, codes):
        super(ddiCNN, self).__init__()
        # Initialisierung wie zuvor
        n_words = codes.get_n_words()
        n_pos = codes.get_n_pos()
        n_labels = codes.get_n_labels()
        n_lc_word = codes.get_n_lc_words()
        n_lemmas = codes.get_n_lemmas()
        max_len = codes.maxlen

        # Embedding dimensions
        emb_dim_word = 64
        emb_dim_pos = 16
        emb_dim_lc = 16
        emb_dim_lemmas = 16

        # Create layers
        self.embW = nn.Embedding(n_words, emb_dim_word, padding_idx=0)
        self.embPOS = nn.Embedding(n_pos, emb_dim_pos, padding_idx=0)
        self.embLC = nn.Embedding(n_lc_word, emb_dim_lc, padding_idx=0)
        self.embLemmas = nn.Embedding(n_lemmas, emb_dim_lemmas, padding_idx=0)

        # Concatenated embedding dimension
        total_emb_dim = emb_dim_word + emb_dim_pos + emb_dim_lc + emb_dim_lemmas
        self.bilstm = nn.LSTM(total_emb_dim, total_emb_dim // 2, num_layers=2,
                              bidirectional=True, batch_first=True,dropout=0.5)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=total_emb_dim, nhead=2, dim_feedforward=48, dropout=0.2  )
            for _ in range(1)
        ])

        # Output layer setup
        self.out = nn.Linear(total_emb_dim, codes.get_n_labels())

    def forward(self, w, p, lc, lemmas):
        x = torch.cat((self.embW(w), self.embPOS(p), self.embLC(lc), self.embLemmas(lemmas)), dim=2)
        lstm_out, _ = self.bilstm(x)
        x = lstm_out

        for block in self.transformer_blocks:
            x = block(x)

        # Global Average Pooling
        x = x.mean(dim=1)  # Take the mean across the sequence dimension

        x = self.out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        # Multihead Attention and Add & Norm
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feedforward and Add & Norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x