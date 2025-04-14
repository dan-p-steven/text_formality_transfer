from torchtext.transforms import Sequential, VocabTransform, ToTensor, Truncate, PadTransform
#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader


class FormalityDataset(Dataset):
    def __init__(self, src_path, tgt_path):

        self.src = []
        self.tgt = []
        
        # Read source sentences
        with open(src_path, 'r', encoding='utf-8') as f:
            self.src = [line.strip() for line in f]
            
        # Read target sentences
        with open(tgt_path, 'r', encoding='utf-8') as f:
            self.tgt = [line.strip() for line in f]
        
        assert len(self.src) == len(self.tgt), "Error: \
                Source and target files must have the same number of lines"

    def __len__(self):
        return len(self.src)
