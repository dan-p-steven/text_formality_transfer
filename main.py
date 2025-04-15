from src.datasets import FormalityDataset
from src.vocab import _generate_vocab, _tokenize, get_transform
from src.model import Encoder, Decoder, Seq2Seq

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

SRC_PATH = './data/train.src'
TGT_PATH = './data/train.tgt'

SRC_VOCAB_PATH = './models/src_vocab.pth'
TGT_VOCAB_PATH = './models/tgt_vocab.pth'

SRC_VOCAB = torch.load(SRC_VOCAB_PATH)
TGT_VOCAB = torch.load(TGT_VOCAB_PATH)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    # Get the data transforms
    src_transform = get_transform(SRC_VOCAB)
    tgt_transform = get_transform(TGT_VOCAB)

    src_tensors = [src_transform(_tokenize(x)) for x in src_batch]
    tgt_tensors = [tgt_transform(_tokenize(x)) for x in tgt_batch]

    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=SRC_VOCAB['<pad>'])
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=TGT_VOCAB['<pad>'])
    
    # Transpose vectors for seq2seq model
    src_padded = src_padded.transpose(0, 1)
    tgt_padded = tgt_padded.transpose(0, 1)
    
    return src_padded, tgt_padded




def main():
    formality_dataset = FormalityDataset(SRC_PATH, TGT_PATH)
    print (len(formality_dataset))


if __name__ == "__main__":

    # Training hyperparameters
    num_epochs = 1
    learning_rate = 0.001
    batch_size = 64

    # Model Hyperparameters
    input_size_encoder = len(SRC_VOCAB)
    input_size_decoder = len(TGT_VOCAB)

    output_size = len(TGT_VOCAB)
    encoder_embedding_size = 128
    decoder_embedding_size = 128

    hidden_size = 1024
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5



    # Instantiate encoder
    encoder = Encoder(
        input_size=input_size_encoder,
        embedding_size=encoder_embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_p=enc_dropout
    )
    
    # Instantiate decoder
    decoder = Decoder(
        input_size=input_size_decoder,
        embedding_size=decoder_embedding_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout_p=dec_dropout
    )

    # Create seq2seq model
    model = Seq2Seq(encoder=encoder,
                    decoder=decoder)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define loss function, tell it to ignore the <pad> index when calculating loss
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_VOCAB["<pad>"])

    # Prepare training data
    train_dataset = FormalityDataset(SRC_PATH, TGT_PATH)
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)

    for epoch in range(num_epochs):

        print (f'Epoch [{epoch+1}/{num_epochs}]')
        # Training loop?
        for b, (src, tgt) in enumerate(train_loader):

            print (f'\t\rbatch {b}/', end='', flush=True)
            output = model(src, tgt)

            output_dim = output.shape[-1]
            
            # Ignore the <sos> token
            output = output[1:].reshape(-1, output.shape[2])
            tgt = tgt[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, tgt)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()