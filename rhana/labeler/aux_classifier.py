from pathlib import Path

import mmengine

import pandas as pd

import torch
import torch.nn as nn
from torchvision.ops import MLP
from torch.utils.data import Dataset, DataLoader

from typing import Dict, List, Union, Optional


def transform_backbone_features(backbone_features, drop_batch_dim=False):
    feature = nn.functional.adaptive_avg_pool2d(backbone_features[-1], output_size=(1,1))
    feature = feature.reshape(1, -1)
    if drop_batch_dim: feature = feature.reshape(-1)
    return feature
    

class BackBoneFeatureDataset(Dataset):
    def __init__(self, index, label_mapper, root_folder):
        if isinstance(index, (str,Path)):
            self.index = pd.read_csv(str(index))
        else:
            self.index = index
            
        if isinstance(label_mapper, (str,Path)):
            self.label_mapper = mmengine.load(label_mapper)
        else:
            self.label_mapper = label_mapper
        
        self.rev_label_mapper = {v:k for k, v in self.label_mapper.items()}
        self.n_label = len(self.label_mapper)
        self.label_columns = [str(i) for i in range(self.n_label)]
        self.label_names = [self.rev_label_mapper[i] for i in range(self.n_label)]
        self.root_folder = Path(root_folder)
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        path = self.root_folder / row['feature']
        img_path = self.root_folder / row['image']
        label = torch.from_numpy( row[self.label_columns].values.astype(float) )
        feature = torch.load(path)
        
        sample = {
            "feature" : feature,
            "label" : label,
            "img_path" : str(img_path),
            "feature_path" : str(path),
        }
        
        return sample


class ResnetClassifierHeadConfig(Dict):
    def __init__(
            self, 
            in_channels, 
            hidden_channels, 
            bias,
            classes,
        ):
        super().__init__()
        self['in_channels'] = in_channels
        self['hidden_channels'] = hidden_channels
        self['bias'] = bias
        self['classes'] = classes

    def save(self, path):
        mmengine.dump(self, path)

    @classmethod
    def load(cls, path):
        config = mmengine.load(path)
        return cls(**config)

class ResnetClassifierHead(nn.Module):
    def __init__(
            self, 
            config
        ):
        super().__init__()
        self.mlp = MLP(
            in_channels = config['in_channels'],
            hidden_channels = config['hidden_channels'],
            inplace = True,
            bias = config['bias'],
            dropout = 0.0,
        )
        
        self.config = config
    
    def forward(
            self, 
            feature
        ):
        return self.mlp(feature)