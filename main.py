from src.datasets import FormalityDataset
from src.vocab import _generate_vocab, _tokenize, get_transform
from src.model import Encoder, Decoder, Seq2Seq

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import gc

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

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge

rouge = Rouge()

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

def get_predictions(model, src, tgt_vocab):
    """Generate predictions from the model for a batch of source sentences"""
    model.eval()
    with torch.no_grad():
        # Get encoder outputs
        hidden, cell = model.encoder(src)
        
        # Tensor to store predictions [seq_len, batch_size]
        batch_size = src.shape[1]
        predictions = []
        
        # Start with <sos> token (index 1 usually)
        tgt_idx = tgt_vocab['<sos>']
        tgt_tensor = torch.LongTensor([tgt_idx] * batch_size).to(device)
        
        # Maximum length of output sentence
        max_len = 100  # Adjust as needed
        
        # Generate until <eos> or max_len
        for _ in range(max_len):
            output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)
            
            # Get next word prediction (greedy decoding)
            tgt_tensor = output.argmax(1)
            
            # Add to predictions
            predictions.append(tgt_tensor.cpu().numpy())
            
            # Stop if all sequences have generated <eos>
            if (tgt_tensor == tgt_vocab['<eos>']).all():
                break
                
    # Convert predictions to batch_size x seq_len
    return torch.tensor(predictions).transpose(0, 1).cpu().numpy()

def decode_predictions(predictions, tgt_vocab):
    """Convert prediction indices to tokens"""
    # Create reverse vocabulary (index -> token)
    idx_to_token = {idx: token for token, idx in tgt_vocab.get_stoi().items()}
    
    # Convert each sequence
    decoded_predictions = []
    for pred_seq in predictions:
        # Convert indices to tokens, stopping at <eos> if present
        tokens = []
        for idx in pred_seq:
            if idx == tgt_vocab['<eos>']:
                break
            tokens.append(idx_to_token.get(idx, '<unk>'))
        decoded_predictions.append(' '.join(tokens))
    
    return decoded_predictions

def calculate_metrics(predictions, references):
    """Calculate BLEU and ROUGE scores"""
    # Make sure we have the same number of predictions and references
    if len(predictions) != len(references):
        # Trim to the shorter length
        min_length = min(len(predictions), len(references))
        predictions = predictions[:min_length]
        references = references[:min_length]
    
    # Prepare references for BLEU (list of list of references)
    # NLTK expects: list of references for each hypothesis
    # Each reference should be a list of tokens, not a string
    bleu_references = [[ref.split()] for ref in references]  # List of lists of tokenized references
    bleu_predictions = [pred.split() for pred in predictions]  # List of tokenized predictions
    
    # Calculate BLEU score
    smoothie = SmoothingFunction().method1
    bleu_score = corpus_bleu(bleu_references, bleu_predictions, smoothing_function=smoothie)
    
    # Calculate ROUGE scores
    rouge = Rouge()
    try:
        # ROUGE requires non-empty predictions and references
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
        if valid_pairs:
            preds, refs = zip(*valid_pairs)
            rouge_scores = rouge.get_scores(preds, refs, avg=True)
            rouge_1 = rouge_scores['rouge-1']['f']
            rouge_2 = rouge_scores['rouge-2']['f']
            rouge_l = rouge_scores['rouge-l']['f']
        else:
            rouge_1 = rouge_2 = rouge_l = 0.0
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        rouge_1 = rouge_2 = rouge_l = 0.0
    
    return {
        'bleu': bleu_score,
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l
    }

def evaluate(model, loader, criterion):

    all_predictions = []
    all_references = []

    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for b, batch in enumerate(loader):
            src, tgt = batch

            src = src.to(device)
            tgt = tgt.to(device)

            tgt_original = tgt.clone()

            output = model(src, tgt, 0)

            output = output[1:].reshape(-1, output.shape[2])
            tgt = tgt[1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)
            epoch_loss += loss.item()

            # Get model predictions
            batch_predictions = get_predictions(model, src, TGT_VOCAB)
            
            # Decode predictions and references to text
            decoded_predictions = decode_predictions(batch_predictions, TGT_VOCAB)
            
            # Convert target tensors to text
            tgt_cpu = tgt_original.cpu().numpy()
            decoded_references = []
            
            for tgt_seq in tgt_cpu.transpose(0, 1):  # Convert to batch_size x seq_len
                # Skip <sos> and convert until <eos> or end
                tokens = []
                for idx in tgt_seq[1:]:  # Skip <sos>
                    if idx == TGT_VOCAB['<eos>'] or idx == TGT_VOCAB['<pad>']:
                        break
                    token = next((t for t, i in TGT_VOCAB.get_stoi().items() if i == idx), '<unk>')
                    tokens.append(token)
                decoded_references.append(' '.join(tokens))
            
            all_predictions.extend(decoded_predictions)
            all_references.extend(decoded_references)

    avg_loss =  epoch_loss/len(loader)
    metrics = {}
    if all_predictions and all_references:
        metrics = calculate_metrics(all_predictions, all_references)
    return avg_loss, metrics

def train(model, loader, num_epochs, optimizer, criterion):
    train_losses = []
    val_losses = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
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
        val_loss, metrics = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)

        # Store metrics
        bleu = metrics.get('bleu', 0.0)
        rouge1 = metrics.get('rouge-1', 0.0)
        rouge2 = metrics.get('rouge-2', 0.0)
        rougeL = metrics.get('rouge-l', 0.0)
        
        bleu_scores.append(bleu)
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougeL_scores.append(rougeL)

        # Model Checkpoint
        checkpoint_path = f'./models/checkpoints/model_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'bleu_scores': bleu_scores,
            'rouge1_scores': rouge1_scores,
            'rouge2_scores': rouge2_scores,
            'rougeL_scores': rougeL_scores
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
                'val_losses': val_losses,
                'bleu_scores': bleu_scores,
                'rouge1_scores': rouge1_scores,
                'rouge2_scores': rouge2_scores,
                'rougeL_scores': rougeL_scores
            }, best_path)
            print (f'New best model (loss={best_val_loss:.3f}) saved to {best_path}.')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'bleu_scores': bleu_scores,
        'rouge1_scores': rouge1_scores,
        'rouge2_scores': rouge2_scores,
        'rougeL_scores': rougeL_scores,
    }

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load a saved checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    metrics = {
        'train_losses': checkpoint.get('train_losses'),
        'val_losses': checkpoint.get('val_losses'),
        'bleu_scores': checkpoint.get('bleu_scores'),
        'rouge1_scores': checkpoint.get('rouge1_scores'),
        'rouge2_scores': checkpoint.get('rouge2_scores'),
        'rougeL_scores': checkpoint.get('rougeL_scores')
    }
    
    print(f"Loaded checkpoint from epoch {epoch}")   
    return model, optimizer, epoch, metrics



if __name__ == "__main__":

    import nltk
    nltk.download('punkt')


    # # Generate vocabs
    # with open(TRAIN_SRC_PATH, 'r', encoding='utf-8') as f:
    #     src = [line.strip() for line in f]

    # with open(TRAIN_TGT_PATH, 'r', encoding='utf-8') as f:
    #     tgt = [line.strip() for line in f]

    # _generate_vocab(src, SRC_VOCAB_PATH)
    # _generate_vocab(tgt, TGT_VOCAB_PATH)

    # Training hyperparameters
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 16

    # Model Hyperparameters
    input_size_encoder = len(SRC_VOCAB)
    input_size_decoder = len(TGT_VOCAB)

    output_size = len(TGT_VOCAB)
    encoder_embedding_size = 300
    decoder_embedding_size = 300

    hidden_size = 512
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
    model.to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define loss function, tell it to ignore the <pad> index when calculating loss
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_VOCAB["<pad>"])

    # Prepare training data
    train_dataset = FormalityDataset(TRAIN_SRC_PATH, TRAIN_TGT_PATH)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

    val_dataset = FormalityDataset(VAL_SRC_PATH, VAL_TGT_PATH)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_collate_fn)
    
    gc.collect()
    torch.cuda.empty_cache()
    train(model, train_loader, num_epochs, optimizer, criterion)

# '''
# TODO: 
#     * track certain metrics during training and validation
#         * bleu score
#         * rouge score
#         * loss
#         * perplexity
# '''

   #model, optimizer, epoch, metrics = load_checkpoint(model, optimizer, './models/checkpoints/model_epoch_15.pt')
    #print (metrics['bleu_scores'])