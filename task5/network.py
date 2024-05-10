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
        self.dropW = nn.Dropout(0.1)
        self.dropS = nn.Dropout(0.1)
        self.lstm = nn.LSTM(150, 256, bidirectional=True, batch_first=True)

        # Adding additional FC layers
        self.fc1 = nn.Linear(512, 256)  # First additional layer
        self.dropFC1 = nn.Dropout(0.1)  # Dropout for the first additional layer
        self.fc2 = nn.Linear(256, 128)  # Second additional layer
        self.dropFC2 = nn.Dropout(0.1)  # Dropout for the second additional layer

        self.out = nn.Linear(
            128, n_labels
        )  # Final output layer with size adjusted to 128

    def forward(self, w, s, f):
        x = self.embW(w)
        y = self.embS(s)
        x = self.dropW(x)
        y = self.dropS(y)

        f = f.type_as(x)

        x = torch.cat((x, y, f), dim=2)
        x = self.lstm(x)[0]  # Get the outputs of the LSTM

        x = self.fc1(x)
        x = func.relu(x)  # Apply ReLU activation function
        x = self.dropFC1(x)  # Apply dropout

        x = self.fc2(x)
        x = func.relu(x)  # Apply ReLU activation function
        x = self.dropFC2(x)  # Apply dropout

        x = self.out(x)  # Final output layer
        return x
