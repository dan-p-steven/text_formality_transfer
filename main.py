from src.datasets import FormalityDataset
from src.vocab import _generate_vocab, _tokenize, get_transform
import torch
import torchtext.transforms as T

SRC_PATH = './data/train.src'
TGT_PATH = './data/train.tgt'

SRC_VOCAB_PATH = './models/src_vocab.pth'
TGT_VOCAB_PATH = './models/tgt_vocab.pth'

SRC_VOCAB = torch.load(SRC_VOCAB_PATH)
TGT_VOCAB = torch.load(TGT_VOCAB_PATH)




def main():

    formality_dataset = FormalityDataset(SRC_PATH, TGT_PATH)
    print (len(formality_dataset))


if __name__ == "__main__":

    src_transform = get_transform(SRC_VOCAB)
    sentence = 'Yo wuss good this is a sentence.'
    transformed_sentence = src_transform(_tokenize(sentence))



