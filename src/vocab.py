from torchtext.transforms import Sequential, VocabTransform, ToTensor, Truncate, PadTransform
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import spacy
import torch

NLP = spacy.load('en_core_web_sm')

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

def _generate_vocab(data, vocab_save_path=None):

    vocab = build_vocab_from_iterator(_yield_tokens(data), specials=SPECIAL_TOKS)

    vocab.set_default_index(vocab["<unk>"])

    if vocab_save_path:
        torch.save(vocab, vocab_save_path)
    else: 
        return vocab





    


def _generate_target_vocab():
    pass
