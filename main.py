from src.datasets import FormalityDataset
from src.vocab import _generate_vocab

SRC_PATH = './data/train.src'
TGT_PATH = './data/train.tgt'

SRC_VOCAB_PATH = './models/src_vocab.pth'
TGT_VOCAB_PATH = './models/tgt_vocab.pth'

def main():

    formality_dataset = FormalityDataset(SRC_PATH, TGT_PATH)
    print (len(formality_dataset))


if __name__ == "__main__":

    # Read source sentences
    with open(SRC_PATH, 'r', encoding='utf-8') as f:
        src = [line.strip() for line in f]

    print (f'[source vocab]')
    _generate_vocab(src, vocab_save_path=SRC_VOCAB_PATH)

    with open(TGT_PATH, 'r', encoding='utf-8') as f:
        tgt = [line.strip() for line in f]

    print (f'[target vocab]')
    _generate_vocab(tgt, vocab_save_path=TGT_VOCAB_PATH)

