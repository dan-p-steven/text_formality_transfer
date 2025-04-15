from torchtext.transforms import Sequential, VocabTransform, ToTensor, AddToken
from torchtext.vocab import build_vocab_from_iterator

import torchtext.transforms as transforms

import spacy
import torch

# Load spacy tokenizer
NLP = spacy.load('en_core_web_sm')

# List of special tokens
SPECIAL_TOKS = ["<unk>", "<pad>", "<sos>", "<eos>"]
    
def _tokenize(text):
    return [token.text for token in NLP(text)]

def _yield_tokens(data):

    count = 0
    total = len(data)
    for line in data:

        count += 1

        if count % 50 == 0:
            print (f'\rProcessed {count}/{total} lines.', end='', flush=True)

        yield _tokenize(line)


def _generate_vocab(data, vocab_save_path=None, token_limit=5000):

    vocab = build_vocab_from_iterator(_yield_tokens(data), 
                                      specials=SPECIAL_TOKS, 
                                      min_freq=2,
                                      max_tokens=token_limit)

    vocab.set_default_index(vocab["<unk>"])

    if vocab_save_path:
        torch.save(vocab, vocab_save_path)
    else: 
        return vocab

def get_transform(vocab):

    transform = Sequential(
        VocabTransform(vocab=vocab),
        AddToken(vocab["<sos>"], begin=True),
        AddToken(vocab["<eos>"], begin=False),
        ToTensor(padding_value=vocab["<pad>"])
    )

    return transform
