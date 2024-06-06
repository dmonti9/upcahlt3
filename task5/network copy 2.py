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
        n_prefs = codes.get_n_prefs()
        n_bigrams = codes.get_n_bigrams()
        n_pbigrams = codes.get_n_preceding_bigrams()
        n_lcwords = codes.get_n_lcwords()
        #trigrams
        n_trigrams = codes.get_n_trigrams()
        n_labels = codes.get_n_labels()

        # Embeddings
        emb_words = 128
        emb_sufs = 100
        emb_prefs = 100
        emb_lw = 50
        emb_features = 7  # Assuming the features tensor has 5 features per word
        emb_bigrams=12
        emb_trigrams=8
        emb_pbigrams=12

        # Calculate total dimension for LSTM input: word embeddings + suffix embeddings + lcwords embeddings + features
        total_dim = emb_words + emb_sufs + emb_lw + emb_features + emb_prefs+emb_bigrams+emb_pbigrams

        self.embW = nn.Embedding(n_words, emb_words)  # Embeddings for words
        self.embS = nn.Embedding(n_sufs, emb_sufs)  # Embeddings for suffixes
        self.embP = nn.Embedding(n_prefs, emb_prefs)  # Embeddings for suffixes
        self.embLW = nn.Embedding(n_lcwords, emb_lw)  # Embeddings for lowercased words
        #preceding bigrams
        self.embBG = nn.Embedding(n_bigrams, emb_bigrams)  # Embeddings for lowercased words
        self.embPBG = nn.Embedding(n_pbigrams, emb_pbigrams)  # Embeddings for lowercased words
        
        self.dropout=nn.Dropout(0.2)

        # LSTM layer configuration
        hidden_size = total_dim // 2
        self.lstm = nn.LSTM(
            total_dim, hidden_size, num_layers=2, bidirectional=True, batch_first=True,dropout=0.5
        )

        # ADJUSTMENTS
        self.lstm_norm = LayerNorm(hidden_size * 2)  # After LSTM and before FC

        # Output layer dimensions adjustment due to bidirectional LSTM
        lstm_output_dim = hidden_size * 2
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim)  # Processing layer
        self.fc2 = nn.Linear(lstm_output_dim, lstm_output_dim//2)  # Processing layer
        self.fc3 = nn.Linear(lstm_output_dim//2, n_labels)  # Processing layer
        self.fc4 = nn.Linear(
            n_labels, n_labels
        )  # Assuming classification task, output size matches label count

    def forward(self, words, suffixes, lcwords, features,prefixes,bigrams,trigrams,pBigrams):
        # Get embeddings from input indices
        # Get embeddings from input indices
        emb_words = self.embW(words)
        emb_sufs = self.embS(suffixes)
        emb_lcwords =self.embLW(lcwords)
        emb_prefs = self.embP(prefixes)
        emb_bigrams = self.embBG(bigrams)
        #preceding bigrams
        emb_pbigrams = self.embPBG(pBigrams)
        # Check feature dimension and concatenate all inputs
        # Assuming features is already of the correct shape: [batch_size, sequence_length, 5]
        x = torch.cat((emb_words, emb_sufs, emb_lcwords, features,emb_prefs,emb_bigrams,emb_pbigrams), dim=2)
        
        x=self.dropout(x)
        x, _ = self.lstm(x)
        x = self.lstm_norm(x)  # Normalize before passing to the fully connected layer

        # Process LSTM output through fully connected layers
        x = self.fc2(x)  # Pass through the second FC layer to get final predictions

        return x
