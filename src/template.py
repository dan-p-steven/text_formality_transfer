import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import Truncate, PadTransform, VocabTransform, Sequential
from torch.utils.data import Dataset, DataLoader
import io

# Step 1: Create a custom dataset for parallel text files
class TranslationDataset(Dataset):
    def __init__(self, src_file_path, tgt_file_path):
        self.src_sentences = []
        self.tgt_sentences = []
        
        # Read source sentences
        with open(src_file_path, 'r', encoding='utf-8') as f:
            self.src_sentences = [line.strip() for line in f]
            
        # Read target sentences
        with open(tgt_file_path, 'r', encoding='utf-8') as f:
            self.tgt_sentences = [line.strip() for line in f]
        
        # Ensure equal number of sentences
        assert len(self.src_sentences) == len(self.tgt_sentences), "Source and target files must have the same number of lines"
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        return self.src_sentences[idx], self.tgt_sentences[idx]

# Step 2: Define tokenizers for source and target languages
src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')  # For English
tgt_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')  # For German (example)

# Step 3: Create dataset instances
train_dataset = TranslationDataset('train.src', 'train.tgt')
valid_dataset = TranslationDataset('valid.src', 'valid.tgt')

# Step 4: Build vocabularies for source and target languages
def yield_tokens(data_iter, tokenizer, language):
    for src, tgt in data_iter:
        if language == 'src':
            yield tokenizer(src)
        else:
            yield tokenizer(tgt)

# Special tokens
special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

# Source vocabulary
src_vocab = build_vocab_from_iterator(
    yield_tokens([(src, tgt) for src, tgt in train_dataset], src_tokenizer, 'src'),
    min_freq=2,  # Only include words that appear at least twice
    specials=special_tokens
)
src_vocab.set_default_index(src_vocab['<unk>'])

# Target vocabulary
tgt_vocab = build_vocab_from_iterator(
    yield_tokens([(src, tgt) for src, tgt in train_dataset], tgt_tokenizer, 'tgt'),
    min_freq=2,
    specials=special_tokens
)
tgt_vocab.set_default_index(tgt_vocab['<unk>'])

# Step 5: Define the transforms for source and target sequences
max_seq_len = 128  # Set appropriate max sequence length

# Source transforms
src_vocab_transform = VocabTransform(src_vocab)
src_truncate_transform = Truncate(max_seq_len - 2)  # Leave space for BOS/EOS tokens
src_pad_transform = PadTransform(max_seq_len, pad_val=src_vocab['<pad>'])

# Target transforms
tgt_vocab_transform = VocabTransform(tgt_vocab)
tgt_truncate_transform = Truncate(max_seq_len - 2)  # Leave space for BOS/EOS tokens
tgt_pad_transform = PadTransform(max_seq_len, pad_val=tgt_vocab['<pad>'])

# Step 6: Create sequential transforms
src_transform = Sequential(
    src_vocab_transform,
    src_truncate_transform,
    src_pad_transform
)

tgt_transform = Sequential(
    tgt_vocab_transform,
    tgt_truncate_transform,
    tgt_pad_transform
)

# Step 7: Create a collation function for DataLoader
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        # Tokenize
        src_tokens = src_tokenizer(src_sample)
        tgt_tokens = tgt_tokenizer(tgt_sample)
        
        # Add BOS and EOS tokens to target sequence (common in seq2seq models)
        processed_src = src_transform(src_tokens)
        
        # For the target, we'll prepare both the input (with BOS) and output (with EOS)
        # for teacher forcing during training
        tgt_input = ['<bos>'] + tgt_tokens
        tgt_output = tgt_tokens + ['<eos>']
        
        # Apply transforms
        processed_tgt_input = tgt_transform(tgt_input)
        processed_tgt_output = tgt_transform(tgt_output)
        
        src_batch.append(processed_src)
        tgt_batch.append((processed_tgt_input, processed_tgt_output))
    
    # Stack all sequences into tensors
    src_batch = torch.stack(src_batch)
    tgt_input_batch = torch.stack([x[0] for x in tgt_batch])
    tgt_output_batch = torch.stack([x[1] for x in tgt_batch])
    
    return {
        'src': src_batch,
        'tgt_input': tgt_input_batch,
        'tgt_output': tgt_output_batch
    }

# Step 8: Create DataLoaders
batch_size = 64
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# Step 9: Example usage in training loop
def train_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Move tensors to device
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # Reshape output and target for loss calculation
        # (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        output = output.view(-1, output.size(-1))
        tgt_output = tgt_output.view(-1)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            output, tgt_output, ignore_index=tgt_vocab['<pad>']
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
