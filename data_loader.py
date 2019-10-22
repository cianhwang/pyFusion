import torch
from torch.utils import data
from torchvision import transforms

class train_dataset(data.Dataset):
    
    def __init__(self, data_dir, gross, seq = 5, transform = None):
        self.data_dir = data_dir
        self.gross = grosss
        self.seq = seq
        self.transform = transform
        
    def __len__(self):
        return self.gross
    
    def __getitem__(self, index):
        
        # shape: (Seq, C, H, W)

        X = torch.rand(self.seq , 3, 512, 896)
        dpt = torch.rand(self.seq, 1, 512, 896)
        
        return (X, dpt)

def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])
    
    train_data = train_dataset(
    
