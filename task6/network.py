import torch
import torch.nn as nn
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()

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
        num_layers = 2
        bidirectional = True
        total_dim = emb_dim_word + emb_dim_pos + emb_dim_lc + emb_dim_lemmas
        
        # Create layers
        self.embW = nn.Embedding(n_words, emb_dim_word, padding_idx=0)
        self.embPOS = nn.Embedding(n_pos, emb_dim_pos, padding_idx=0)
        self.embLC = nn.Embedding(n_lc_word, emb_dim_lc, padding_idx=0)
        self.embLemmas = nn.Embedding(n_lemmas, emb_dim_lemmas, padding_idx=0)
        self.lstm = nn.LSTM(total_dim, total_dim // 2, num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=0.5)
        
        lstm_output_dim = total_dim if bidirectional else total_dim // 2

       
        self.cnn1 = nn.Conv1d(lstm_output_dim, 32, kernel_size=3, stride=1, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.out = nn.Linear(32 * (max_len // 2), n_labels) 

    def forward(self, w, p, lc, lemmas):
        x = self.embW(w)
        y = self.embPOS(p)
        u = self.embLC(lc)
        v = self.embLemmas(lemmas)
        x = torch.cat((x, y, u, v), dim=2)

        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)

        x = self.cnn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = x.flatten(start_dim=1)
        x = self.out(x)
        return x