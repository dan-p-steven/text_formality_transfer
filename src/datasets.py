from torch.utils.data import Dataset


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
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]
