import torch
from torch.utils import data
from glob import glob
import numpy as np
from torchvision import transforms

class FPRED_datasets(data.Dataset):
    def __init__(self, positive_dir, negative_dir, ratio, len_):
        self.positive_files=glob(positive_dir+'/*.npy')
        self.negative_files=glob(negative_dir+'/*.npy')
        self.ratio=ratio+1
        self.len_=len_
        self.pos_len=len(self.positive_files)

    def __getitem__(self, index):
        if index%self.ratio==0:
            sample=np.load(self.positive_files[int(index/self.ratio)%self.pos_len]).astype(np.float32)
            label=1
        else:
            sample=np.load(self.negative_files[np.random.randint(0, len(self.negative_files))])
            label=0
        return torch.from_numpy(sample).contiguous().view(1,40,40,40).float(), label

    def __len__(self):
        return self.len_
