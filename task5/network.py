import torch
import torch.nn as nn
import torch.nn.functional as func


criterion = nn.CrossEntropyLoss()


class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_labels = codes.get_n_labels()

        self.embW = nn.Embedding(n_words, 100)
        self.embS = nn.Embedding(n_sufs, 50)
        self.embWl = nn.Embedding(n_words, 100)
        self.embE = nn.Embedding(2, 50)  # Embedding layer for external features

        self.dropW = nn.Dropout(0.1)
        self.dropS = nn.Dropout(0.1)
        self.dropWl = nn.Dropout(0.1)
        self.dropE = nn.Dropout(0.1)  # Dropout for the external features

        # Update LSTM input size to 300 (100 + 100 + 50 + 200)
        self.lstm = nn.LSTM(450, 200, bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, n_labels)

    def forward(self, w, s, lw, e):
        x = self.embW(w)
        y = self.embS(s)
        z = self.embWl(lw)
        e = self.embE(e)  # Apply the new embedding layer

        # Debugging: Print shapes
        # print(f"x shape: {x.shape}")
        # print(f"y shape: {y.shape}")
        # print(f"z shape: {z.shape}")
        # print(f"e shape: {e.shape}")

        # Reshape e by merging the last two dimensions
        e = e.view(e.size(0), e.size(1), -1)

        x = self.dropW(x)
        y = self.dropS(y)
        z = self.dropWl(z)
        e = self.dropE(e)  # Apply dropout to the new embedding layer

        # Debugging: Print shapes after dropout
        # print(f"x shape after dropout: {x.shape}")
        # print(f"y shape after dropout: {y.shape}")
        # print(f"z shape after dropout: {z.shape}")
        # print(f"e shape after dropout: {e.shape}")

        # Concatenate the four embeddings along the third dimension
        x = torch.cat((x, z, y, e), dim=2)

        x, _ = self.lstm(x)

        x = self.fc1(x)
        x = self.fc2(x)
        return x
