import torch
import torch.nn as nn
import torch.nn.functional as func

criterion = nn.CrossEntropyLoss()

class ddiCNN(nn.Module):

    def __init__(self, codes):
        super(ddiCNN, self).__init__()
        # get sizes
        n_words = codes.get_n_words()
        n_labels = codes.get_n_labels()
        max_len = codes.maxlen
        embedding_dim = 100
        hidden_dim = 64  # LSTM hidden dimension size
        num_layers = 1   # number of LSTM layers

        # create layers
        self.embW = nn.Embedding(n_words, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.cnn = nn.Conv1d(hidden_dim * 2, 32, kernel_size=3, stride=1, padding="same")  # Adjust input channels for bidirectional

        self.out = nn.Linear(32 * max_len, n_labels)

    def forward(self, w):
        # run layers on given data
        x = self.embW(w)  # [batch_size, seq_length, embedding_dim]
        x, _ = self.lstm(x)  # LSTM output: [batch_size, seq_length, hidden_dim]
        x = x.permute(0, 2, 1)  # [batch_size, hidden_dim, seq_length]
        x = self.cnn(x)  # [batch_size, 32, seq_length]
        x = func.relu(x)
        x = x.flatten(start_dim=1)  # [batch_size, 32 * seq_length]
        x = self.out(x)  # [batch_size, n_labels]
        return x
