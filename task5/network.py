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
        emb_features = 5
        total_dim = (
            emb_words + emb_sufs + emb_lw + (emb_features * 4)
        )  # each feature is an additional dimension
        self.embW = nn.Embedding(n_words, emb_words)  # Embeddings for words
        self.embS = nn.Embedding(n_sufs, emb_sufs)  # Embeddings for suffixes
        self.embLW = nn.Embedding(n_lcwords, emb_lw)  # Embeddings for lowercased words

        # LSTM layer
        hidden_size = total_dim // 2
        self.lstm = nn.LSTM(
            total_dim, hidden_size, 2, bidirectional=True, batch_first=True
        )

        # Fully connected layers
        lstm_output_dim = hidden_size * 2  # due to bidirectional
        self.fc1 = nn.Linear(lstm_output_dim, n_labels)  # Adjusted to correct size
        self.fc2 = nn.Linear(
            n_labels, n_labels
        )  # Final output layer, assuming a classification task

    def forward(self, words, suffixes, lcwords, cap, dash, num, ext, ext_multi):
        # Embeddings
        cap = cap.float()

        emb_words = self.embW(words)
        emb_sufs = self.embS(suffixes)
        emb_lcwords = self.embLW(lcwords)

        # Unsqueeze feature tensors
        cap = cap.unsqueeze(2)
        dash = dash.unsqueeze(2)
        num = num.unsqueeze(2)
        ext = ext.unsqueeze(2)
        ext_multi = ext_multi.unsqueeze(2)

        # Concatenate all embeddings and transformed features
        x = torch.cat(
            (emb_words, emb_sufs, emb_lcwords, cap, dash, num, ext, ext_multi), dim=2
        )

        # Debugging: Print sizes before LSTM
        # print(f"Shape of input to LSTM: {x.shape}")

        # LSTM processing
        x, _ = self.lstm(x)

        # Debugging: Print sizes after LSTM
        # print(f"Shape of output from LSTM: {x.shape}")

        # Fully connected layers
        x = func.relu(self.fc1(x))
        x = self.fc2(x)

        # Debugging: Print sizes after FC layers
        # print(f"Shape of output from FC1: {x.shape}")

        return x
