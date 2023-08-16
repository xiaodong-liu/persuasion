import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class dataset(Dataset):
    def __init__(self, data, s_dim):
        tmp = torch.tensor(data, dtype = torch.float)
        self.x = tmp[:, 0:s_dim]
        self.y = tmp[:, s_dim: ]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def split_train_val(collect_data, val_size):
    tmp = np.array(collect_data)
    total_sample = len(tmp)
    np.random.shuffle(tmp)
    val_size = int(total_sample * val_size)
    train_data = tmp[val_size : , :]
    val_data = tmp[:val_size, :]
    return train_data, val_data
    
def build_dataloader(train_data, val_data, batch_size, s_dim):
    trainset = dataset(train_data, s_dim)
    valset = dataset(val_data, s_dim)
    train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size = batch_size, shuffle=False)
    return train_loader, val_loader
    

