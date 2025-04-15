import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_p=0.1):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        

        self.embedding = nn.Embedding(input_size, embedding_size)

        self.lstm = nn.LSTM(input_size=embedding_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers,
                          dropout=dropout_p if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_p)

        
            
        
    def forward(self, x):

        embedding = self.dropout(self.embedding(x))

        out, (hidden, cell) = self.lstm(embedding)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_p=0.1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)

        self.lstm = nn.LSTM(input_size=embedding_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            dropout=dropout_p if num_layers > 1 else 0)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):

        # shape of x want it to be (1, N)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))

        predictions = self.fc(outputs)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
