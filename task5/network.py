import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

label_to_index = {
    "B-group": 1,
    "I-drug": 2,
    "B-drug_n": 3,
    "I-group": 4,
    "I-drug_n": 5,
    "B-brand": 6,
    "I-brand": 7,
    "B-drug": 8,
    "O": 9,
    "PAD": 0,
}

# Number of classes is the length of this dictionary
num_classes = len(label_to_index)

# Create a tensor for weights initialized to one (no emphasis)
weights = torch.zeros(num_classes)

# Set higher weights for B-drug_n and I-drug_n classes
higher_weight = 70.0
weights[label_to_index["B-drug_n"]] = higher_weight
weights[label_to_index["I-drug_n"]] = higher_weight

# Create the loss function with these weights
criterion = nn.CrossEntropyLoss(weight=weights)


class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        # Dimensions from 'codes' object, presumably provided elsewhere
        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_lcwords = codes.get_n_lcwords()
        n_labels = codes.get_n_labels()

        # Embedding dimensions
        emb_words = 100  # Must match GloVe vector size
        emb_sufs = 100
        emb_lw = 50
        emb_features = 7

        glove_path = "../data/glove.6B.100d.txt"
        vocab_path = "./vocabulary.txt"  # Path to your vocabulary file

        # Load GloVe Embeddings
        self.glove_embeddings = self.load_glove_embeddings(
            glove_path, emb_words, 12026, vocab_path
        )

        # Total dimension for LSTM input
        total_dim = emb_words + emb_sufs + emb_lw + emb_features

        # Embedding layers
        self.embW = nn.Embedding.from_pretrained(self.glove_embeddings, freeze=True)
        self.embS = nn.Embedding(n_sufs, emb_sufs)
        self.embLW = nn.Embedding(n_lcwords, emb_lw)

        # Dropout layers for embeddings
        self.dropoutW = nn.Dropout(0.4)
        self.dropoutS = nn.Dropout(0.4)
        self.dropoutLW = nn.Dropout(0.4)

        # LSTM configuration
        hidden_size = total_dim // 2
        self.lstm = nn.LSTM(
            total_dim,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        # LSTM Normalization
        self.lstm_norm = LayerNorm(hidden_size * 2)  # After LSTM and before FC

        # Activation
        self.selu = nn.SELU()

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, n_labels)
        self.fc2 = nn.Linear(n_labels, n_labels)

    def load_vocabulary(self, file_path):
        vocab = {}
        with open(file_path, "r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                word = line.strip()
                vocab[word] = index
        return vocab

    def load_glove_embeddings(self, glove_path, embedding_dim, vocab_size, vocab_path):
        glove_dict = {}
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = torch.tensor(
                    [float(val) for val in values[1:]], dtype=torch.float32
                )
                glove_dict[word] = vector

        vocab_dictionary = self.load_vocabulary(vocab_path)

        # Create embedding tensor and initialize for all words in your model's vocabulary
        embedding_matrix = torch.zeros((vocab_size, embedding_dim))
        for word, idx in vocab_dictionary.items():
            vector = glove_dict.get(word)
            if vector is not None:
                embedding_matrix[idx] = vector

        return embedding_matrix

    def forward(self, words, suffixes, lcwords, features):
        emb_words = self.dropoutW(self.embW(words))
        emb_sufs = self.dropoutS(self.embS(suffixes))
        emb_lcwords = self.dropoutLW(self.embLW(lcwords))
        x = torch.cat((emb_words, emb_sufs, emb_lcwords, features), dim=2)
        x, _ = self.lstm(x)
        x = self.lstm_norm(x)
        x = self.selu(self.fc1(x))
        x = self.fc2(x)
        return x
