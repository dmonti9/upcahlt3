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
        emb_words = 100
        emb_sufs = 50
        emb_lw = 50
        emb_features = 5  # Assuming the features tensor has 5 features per word

        # Calculate total dimension for LSTM input: word embeddings + suffix embeddings + lcwords embeddings + features
        total_dim = emb_words + emb_sufs + emb_lw + emb_features

        self.embW = nn.Embedding(n_words, emb_words)  # Embeddings for words
        self.embS = nn.Embedding(n_sufs, emb_sufs)  # Embeddings for suffixes
        self.embLW = nn.Embedding(n_lcwords, emb_lw)  # Embeddings for lowercased words

        # LSTM layer configuration
        hidden_size = total_dim // 2
        self.lstm = nn.LSTM(
            total_dim, hidden_size, num_layers=2, bidirectional=True, batch_first=True
        )

        # Output layer dimensions adjustment due to bidirectional LSTM
        lstm_output_dim = hidden_size * 2
        self.fc1 = nn.Linear(lstm_output_dim, n_labels)  # Processing layer
        self.fc2 = nn.Linear(
            n_labels, n_labels
        )  # Assuming classification task, output size matches label count

    def forward(self, words, suffixes, lcwords, features):
        # Get embeddings from input indices
        emb_words = self.embW(words)
        emb_sufs = self.embS(suffixes)
        emb_lcwords = self.embLW(lcwords)

        # Check feature dimension and concatenate all inputs
        # Assuming features is already of the correct shape: [batch_size, sequence_length, 5]
        x = torch.cat((emb_words, emb_sufs, emb_lcwords, features), dim=2)

        # Pass concatenated input through LSTM
        x, _ = self.lstm(x)

        # Process LSTM output through fully connected layers
        x = func.relu(
            self.fc1(x)
        )  # Apply ReLU activation function to the output of the first FC layer
        x = self.fc2(x)  # Pass through the second FC layer to get final predictions

        return x
