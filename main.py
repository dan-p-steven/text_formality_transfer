from src.datasets import FormalityDataset
from src.vocab import _generate_vocab, _tokenize, get_transform
from src.model import Encoder, Decoder, Seq2Seq

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from random import randrange

import time

TRAIN_SRC_PATH = './data/train.src'
TRAIN_TGT_PATH = './data/train.tgt'

VAL_SRC_PATH = './data/valid.src'
VAL_TGT_PATH = './data/valid.tgt'

SRC_VOCAB_PATH = './models/src_vocab.pth'
TGT_VOCAB_PATH = './models/tgt_vocab.pth'

SRC_VOCAB = torch.load(SRC_VOCAB_PATH)
TGT_VOCAB = torch.load(TGT_VOCAB_PATH)

import warnings
warnings.filterwarnings(
    "ignore", 
    message="The inner type of a container is lost when calling torch.jit.isinstance in eager mode.*",
    category=UserWarning
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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

def val_collate_fn(batch):

    src_batch, tgt_batch = zip(*batch)
    
    # Get the data transforms
    src_transform = get_transform(SRC_VOCAB)
    tgt_transform = get_transform(TGT_VOCAB)

    src_tensors = [src_transform(_tokenize(x)) for x in src_batch]

    tgt_tensors = []

    # get a random variant from translations
    for translations in tgt_batch:
        # generate a random index from 1 to len translations
        idx = randrange(len(translations))
        # pick that translation
        x = translations[idx]
        # transform it and add it to the tensor
        tgt_tensors.append(tgt_transform(_tokenize(x)))

    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=SRC_VOCAB['<pad>'])
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=TGT_VOCAB['<pad>'])

    src_padded = src_padded.transpose(0, 1)
    tgt_padded = tgt_padded.transpose(0, 1)

    return src_padded, tgt_padded

def evaluate(model, loader, criterion):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for b, batch in enumerate(loader):
            src, tgt = batch

            src = src.to(device)
            tgt = tgt.to(device)

            output = model(src, tgt, 0)

            output = output[1:].reshape(-1, output.shape[2])
            tgt = tgt[1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss/len(loader)

def train(model, loader, num_epochs, optimizer, criterion):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        model.train()
        epoch_loss = 0

        print (f'\nEpoch [{epoch+1}/{num_epochs}]')
        for b, batch in enumerate(loader):
            src, tgt = batch

            # move to gpu
            src = src.to(device)
            tgt = tgt.to(device)

            start = time.time()
            
            output = model(src, tgt)
        
            # Ignore the <sos> token
            output = output[1:].reshape(-1, output.shape[2])
            tgt = tgt[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, tgt)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            epoch_loss += loss.item()
            end = time.time()
            print (f'\r\tbatch {b+1}/{len(loader)}: {end-start:.2f}s', end='', flush=True)

        train_losses.append(epoch_loss/len(loader))
        val_losses.append(evaluate(model, val_loader, criterion))

        # Model Checkpoint
        checkpoint_path = f'./models/checkpoints/model_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, checkpoint_path)
        print(f'\tSaved checkpoint: {checkpoint_path}')

        # Save best model
        if val_losses[-1] < best_val_loss:
            best_path = './models/best.pt'
            best_val_loss = val_losses[-1]
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }, best_path)
            print (f'New best model (loss={best_val_loss:.3f}) saved to {best_path}.')



        # x Do model evaluation
        # x val loader
        #  x src, tgt
        #  generate predictions with this current model on val 
        #  compare predictions with tgt (calculate bleu, rouge perplexity)
        # save 



if __name__ == "__main__":

    # Generate vocabs
    with open(TRAIN_SRC_PATH, 'r', encoding='utf-8') as f:
        src = [line.strip() for line in f]

    with open(TRAIN_TGT_PATH, 'r', encoding='utf-8') as f:
        tgt = [line.strip() for line in f]

    _generate_vocab(src, SRC_VOCAB_PATH)
    _generate_vocab(tgt, TGT_VOCAB_PATH)

#     # Training hyperparameters
#     num_epochs = 10
#     learning_rate = 0.001
#     batch_size = 64

#     # Model Hyperparameters
#     input_size_encoder = len(SRC_VOCAB)
#     input_size_decoder = len(TGT_VOCAB)

#     output_size = len(TGT_VOCAB)
#     encoder_embedding_size = 128
#     decoder_embedding_size = 128

#     hidden_size = 256
#     num_layers = 2
#     enc_dropout = 0.5
#     dec_dropout = 0.5



#     # Instantiate encoder
#     encoder = Encoder(
#         input_size=input_size_encoder,
#         embedding_size=encoder_embedding_size,
#         hidden_size=hidden_size,
#         num_layers=num_layers,
#         dropout_p=enc_dropout
#     )
    
#     # Instantiate decoder
#     decoder = Decoder(
#         input_size=input_size_decoder,
#         embedding_size=decoder_embedding_size,
#         hidden_size=hidden_size,
#         output_size=output_size,
#         num_layers=num_layers,
#         dropout_p=dec_dropout
#     )

#     # Create seq2seq model
#     model = Seq2Seq(encoder=encoder,
#                     decoder=decoder)
#     model.to(device)
    
#     # Define optimizer
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Define loss function, tell it to ignore the <pad> index when calculating loss
#     criterion = nn.CrossEntropyLoss(ignore_index=TGT_VOCAB["<pad>"])

#     # Prepare training data
#     train_dataset = FormalityDataset(TRAIN_SRC_PATH, TRAIN_TGT_PATH)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

#     val_dataset = FormalityDataset(VAL_SRC_PATH, VAL_TGT_PATH)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_collate_fn)

#     train(model, train_loader, num_epochs, optimizer, criterion)

# '''
# TODO: 
#     * track certain metrics during training and validation
#         * bleu score
#         * rouge score
#         * loss
#         * perplexity
# '''