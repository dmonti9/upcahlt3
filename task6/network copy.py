import torch
import torch.nn as nn
import torch.nn.functional as F
criterion = nn.CrossEntropyLoss()


class ddiCNN(nn.Module):
    def __init__(self, codes):
        super(ddiCNN, self).__init__()
        # Utilize sizes from Codemaps
        n_words = codes.get_n_words()
        n_pos = codes.get_n_pos()
        n_labels = codes.get_n_labels()
        max_len = codes.maxlen

        # Embedding dimensions
        emb_dim_word = 100
        emb_dim_pos = 50

        # LSTM configuration
        self.hidden_dim = 128  # Number of LSTM units
        num_layers = 2
        bidirectional = True

        # Create layers
        self.embW = nn.Embedding(n_words, emb_dim_word, padding_idx=0)
        self.embPOS = nn.Embedding(n_pos, emb_dim_pos, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim_word + emb_dim_pos, self.hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=0.5)
        self.attention = nn.Linear(self.hidden_dim * 2, 1)
        self.cnn = nn.Conv1d(self.hidden_dim * 2, 32, kernel_size=3, stride=1, padding='same')
        self.out = nn.Linear(32, n_labels)  # Now using max_len correctly

    def forward(self, w, p):
        x = self.embW(w)
        y = self.embPOS(p)
        x = torch.cat((x, y), dim=2)

        x, _ = self.lstm(x)

        attn_weights = F.softmax(self.attention(x), dim=1)
        x = torch.bmm(attn_weights.transpose(1, 2), x)
        x = x.squeeze(1)

        x = x.view(x.size(0), self.hidden_dim * 2, -1)

        x = self.cnn(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # Ensure the output matches the expected size for self.out

        x = self.out(x)
        return x
