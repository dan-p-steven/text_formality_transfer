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
                          bidirectional=True, 
                          dropout=dropout_p if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_p)

        
            
        
    def forward(self, x):

        embedding = self.dropout(self.embedding(x))

        out, (h, c) = self.lstm(embedding)

        return out, (h, c)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h, encoder_out, mask=None):
        batch_size = encoder_out.shape[0]
        src_len = encoder_out.shape[1]

            # Repeat decoder hidden state src_len times
        h = h.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hidden_size]
        
        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((h, encoder_out), dim=2)))  # [batch_size, src_len, hidden_size]
        
        # Calculate attention
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=1)  # [batch_size, src_len]
        
        # Get context vector
        context = torch.bmm(attention.unsqueeze(1), encoder_out).squeeze(1)  # [batch_size, hidden_size]
        
        return attention, context


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, attention, num_layers, dropout_p=0.1):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.attention = attention
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, embedding_size)
        
        self.lstm = nn.LSTM(input_size=embedding_size + hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True, 
                            dropout=dropout_p if num_layers > 1 else 0)
        
        self.out = nn.Linear(hidden_size*2, output_size)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, h, encoder_out, mask=None):
        embedding = self.dropout(self.embedding(x))

        attention_weights, context = self.attention(h[0][-1], encoder_out, mask)

        lstm_in = torch.cat((embedding, context.unsqueeze(1)), dim=2)

        out, h = self.lstm(lstm_in, h)

        out = torch.cat((out.squeeze(1), context), dim=1)  # [batch_size, hidden_size * 2]

        prediction = self.out(out)  # [batch_size, output_size]

        return prediction, h, attention_weights

