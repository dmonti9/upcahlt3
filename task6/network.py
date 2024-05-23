import torch
import torch.nn as nn
import torch.nn.functional as F
criterion = nn.CrossEntropyLoss()

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == self.embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with batched matrix multiplication across the head dimension
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class ddiCNN(nn.Module):
    def __init__(self, codes):
        super(ddiCNN, self).__init__()
        # Get sizes from codes
        n_words = codes.get_n_words()
        n_pos = codes.get_n_pos()
        n_labels = codes.get_n_labels()
        n_lc_word = codes.get_n_lc_words()
        n_lemmas = codes.get_n_lemmas()
        max_len = codes.maxlen

        # Embedding dimensions
        emb_dim_word = 100
        emb_dim_pos = 50
        emb_dim_lc = 50
        emb_dim_lemmas = 50

        # LSTM configuration
        hidden_dim = 128
        num_layers = 2
        bidirectional = True

        # Create layers
        self.embW = nn.Embedding(n_words, emb_dim_word, padding_idx=0)
        self.embPOS = nn.Embedding(n_pos, emb_dim_pos, padding_idx=0)
        self.embLC = nn.Embedding(n_lc_word, emb_dim_lc, padding_idx=0)
        self.embLemmas = nn.Embedding(n_lemmas, emb_dim_lemmas, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim_word + emb_dim_pos+emb_dim_lc+emb_dim_lemmas, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=0.5)
        
        # CNN layer with Batch Normalization
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.cnn = nn.Conv1d(lstm_output_dim, 32, kernel_size=3, stride=1, padding='same')
        self.out = nn.Linear(32 * max_len, n_labels)

    def forward(self, w, p,lc,lemmas):
        x = self.embW(w)
        y = self.embPOS(p)
        u = self.embLC(lc)
        v = self.embLemmas(lemmas)
        x = torch.cat((x, y,u,v), dim=2)

        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)

        x = self.cnn(x)
        x = F.relu(x)
        x = x.flatten(start_dim=1)

        x = self.out(x)
        return x
