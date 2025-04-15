import torch
import torch.nn as nn
import torch.nn.functional as F
import random

SRC_VOCAB_PATH = './models/src_vocab.pth'
TGT_VOCAB_PATH = './models/tgt_vocab.pth'

SRC_VOCAB = torch.load(SRC_VOCAB_PATH)
TGT_VOCAB = torch.load(TGT_VOCAB_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]

        target_vocab_size = len(TGT_VOCAB)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size, device=self.device)

        hidden, cell = self.encoder(source)
         
        # Grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output

            # (N, tgt vocab size)
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs