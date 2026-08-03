from ast import List
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_v2_m, efficientnet_v2_s, efficientnet_v2_l
from torchvision.ops import MLP
from torchvision.transforms import ToTensor,Compose, CenterCrop, Normalize, RandomResizedCrop, RandomHorizontalFlip, Resize, RandomRotation
from torchvision.transforms import ColorJitter, GaussianBlur, RandomAdjustSharpness, RandomAutocontrast, RandomEqualize

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, Literal, Union, List
import pandas as pd
import torch
from pathlib import Path
from PIL import Image

from rhana.pattern import Rheed


def get_train_transforms(input_size:int=480, model_name:str='efficientnet_v2_m'): 
    if model_name == 'efficientnet_v2_m':

        # hard version2
        return Compose([
            # RandomEqualize(),
            # for gray scale image, only brightness and contrast are effective
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0),
            # RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            # RandomAutocontrast(p=0.2), # conflicts with ColorJitter
            RandomRotation(degrees=1),

            ToTensor(),
            Resize(input_size, antialias = True),
            # RandomResizedCrop(input_size), # some rheed features is very small, avoid crop
            GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5.)),
            # GaussianNoise(mean=0.0, std=0.1), # does not have this one at this pytorch version
            RandomHorizontalFlip(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        ])

        # # hard version
        # return Compose([
        #     # RandomEqualize(),
        #     # for gray scale image, only brightness and contrast are effective
        #     ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
        #     RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        #     # RandomAutocontrast(p=0.2), # conflicts with ColorJitter
        #     RandomRotation(degrees=1),

        #     ToTensor(),
        #     Resize(input_size, antialias = True),
        #     # RandomResizedCrop(input_size), # some rheed features is very small, avoid crop
        #     # GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5.)),
        #     # GaussianNoise(mean=0.0, std=0.1), # does not have this one at this pytorch version
        #     RandomHorizontalFlip(),
        #     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        # ])

        # easy version
        # return Compose([
        #     RandomRotation(degrees=1),
        #     ToTensor(),
        #     Resize(input_size, antialias = True),
        #     RandomHorizontalFlip(),
        #     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        # ])


    elif model_name == 'efficientnet_v2_s':
        # hard version2
        return Compose([
            # RandomEqualize(),
            # for gray scale image, only brightness and contrast are effective
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
            # RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            # RandomAutocontrast(p=0.2), # conflicts with ColorJitter
            RandomRotation(degrees=2),

            ToTensor(),
            Resize(input_size, antialias = True),
            # RandomResizedCrop(input_size), # some rheed features is very small, avoid crop
            GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5.)),
            # GaussianNoise(mean=0.0, std=0.1), # does not have this one at this pytorch version
            RandomHorizontalFlip(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        ])

        # return Compose([
        #     RandomRotation(degrees=1),
        #     ToTensor(),
        #     Resize(input_size, antialias = True),
        #     RandomHorizontalFlip(),
        #     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        # ])
    elif model_name == 'efficientnet_v2_l':
        return Compose([
            RandomRotation(degrees=1),
            ToTensor(),
            Resize(input_size, antialias = True),
            RandomHorizontalFlip(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])        
        ])
    else:
        raise ValueError(f"Invalid model name: {model_name}")   


def get_test_transforms(input_size:int=480, model_name:str='efficientnet_v2_m'):
    if model_name == 'efficientnet_v2_m':
        return Compose([
            ToTensor(),
            Resize(input_size, antialias = True),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif model_name == 'efficientnet_v2_s':
        return Compose([
            ToTensor(),
            Resize(input_size, antialias = True),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        ])
    elif model_name == 'efficientnet_v2_l':
        return Compose([
            ToTensor(),
            Resize(input_size, antialias = True),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])        
        ])
    else:
        raise ValueError(f"Invalid model name: {model_name}")   


class EfficientNetClassifierConfig(Dict):
    def __init__(
            self, 
            bias,
            dropout,
            n_classes,
            model_name : Literal['efficientnet_v2_m', 'efficientnet_v2_s', 'efficientnet_v2_l'] = 'efficientnet_v2_m',
        ):
        super().__init__()
        self['bias'] = bias
        self['dropout'] = dropout
        self['n_classes'] = n_classes
        self['model_name'] = model_name

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(**config)


class EfficientNetMultiClassBinaryClassifier(nn.Module):
    def __init__(
            self, 
            config:EfficientNetClassifierConfig,
        ):
        super().__init__()
        self.config = config
        if self.config['model_name'] == 'efficientnet_v2_m':
            self.efficientnet = efficientnet_v2_m(weights="DEFAULT")
        elif self.config['model_name'] == 'efficientnet_v2_s':
            self.efficientnet = efficientnet_v2_s(weights="DEFAULT")
        elif self.config['model_name'] == 'efficientnet_v2_l':
            self.efficientnet = efficientnet_v2_l(weights="DEFAULT")
        else:
            raise ValueError(f"Invalid model name: {self.config['model_name']}")
        classifier = nn.Sequential(
            nn.Dropout(p=self.config['dropout'], inplace=True),
            nn.Linear(self.efficientnet.classifier[-1].in_features, self.config['n_classes'], bias=self.config['bias']),
        )
        # classifier = MLP(
        #     in_channels=self.efficientnet.classifier[-1].in_features,
        #     hidden_channels=[self.efficientnet.classifier[-1].in_features//2, self.config['n_classes']],
        #     bias=self.config['bias'],
        #     dropout=self.config['dropout'],
        # )
        self.efficientnet.classifier = classifier

    @property
    def classifier(self):
        return self.efficientnet.classifier

    def forward(self, x):
        x = self.efficientnet(x)
        return x

class MultiClassBinaryClassifierDataset(Dataset):
    def __init__(self, index, label_mapper, root_folder, transforms=None):
        if isinstance(index, (str,Path)):
            self.index = pd.read_csv(str(index))
        else:
            self.index = index
            
        if isinstance(label_mapper, (str,Path)):
            with open(label_mapper, 'r') as f:
                self.label_mapper = json.load(f)
        else:
            self.label_mapper = label_mapper
        
        self.rev_label_mapper = {v:k for k, v in self.label_mapper.items()}
        self.n_label = len(self.label_mapper)
        self.label_columns = ["label_" + str(i) for i in range(self.n_label)]
        self.label_names = [self.rev_label_mapper[i] for i in range(self.n_label)]
        self.root_folder = Path(root_folder)
        self.transforms = transforms

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        img_path = self.root_folder / row['image']
        label = torch.from_numpy( row[self.label_columns].values.astype(float) )
        image = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)
        
        sample = {
            "label" : label,
            "image" : image,
            "img_path" : str(img_path),
        }
        
        return sample

class EfficientNetMultiClassBinaryClassifierInference:
    def __init__(
            self,
            model_path:Union[str, Path],
            device:str=None,
            transforms:Union[Compose, str]=None,
            label_mapper:Union[Dict, str]=None,
        ):
        """
        Args:
            model_path: path to the model file
            device: device to use for the model. It can be a string like "cuda:0" or "cpu"
            transforms: transforms to apply to the image. It can be a Compose object or a path to a pth file
            label_mapper: label mapper to use for the model. It can be a dictionary or a path to a json file
        """

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        if isinstance(transforms, (str, Path)):
            self.transforms = torch.load(transforms)
        else:
            self.transforms = transforms

        if isinstance(label_mapper, (str, Path)):
            with open(label_mapper, 'r') as f:
                self.label_mapper = json.load(f)
        else:
            self.label_mapper = label_mapper

        self.rev_label_mapper = {v:k for k, v in self.label_mapper.items()}
        self.label_names = [self.rev_label_mapper[i] for i in range(len(self.rev_label_mapper))]

    def preprocess_rd(self, rd : Rheed):
        """
        Preprocess the rheed pattern to the input of the model.

        Args:
            rd (rhana.pattern.Rheed): an input rheed pattern which is scaled to the range of (0, 1)

        Returns:
            np.array: a preprocessed rheed pattern
        """
        if rd.pattern.ndim == 2:
            inp = np.repeat(rd.pattern[..., None], 3, axis=-1)
        else:
            inp = rd.pattern

        inp = inp * 255 # no inplace operation

        # transform only works on uint8 images
        return inp.astype(np.uint8)
        # return inp

    @property
    def classes(self):
        return self.label_names


    def predict(self, rd: Rheed):
        image = self.preprocess_rd(rd)
        if self.transforms is not None:
            image = self.transforms(image)

        with torch.inference_mode():
            image = image.to(device=self.device, dtype=torch.float32)
            if image.ndim == 3: image = image.unsqueeze(0)

            logits = self.model(image)
            probs = F.sigmoid(logits)
        return probs

    def predict_batch(self, rds: List[Rheed]):
        batch = np.stack( [ self.preprocess_rd(rd) for rd in rds ], axis=0 )

        if self.transforms is not None:
            batch = self.transforms(batch)
        with torch.inference_mode:
            batch = batch.to(device=self.device, dtype=torch.float32)
            logits = self.model(batch)
            probs = F.sigmoid(logits)
        return probs
    
    def plot_prediction(self, rd, result):
        fig, ax = rd.plot_pattern(show_axes=True)
        if result.ndim == 1:
            result = result[None, ...]
        classes = self.classes
        text = "\n".join([ f"{classes[i]}:{p:.2f}" for i, p in enumerate(result[0])])
        ax.text(0.5, 0.1, text, ha="center", va="bottom", color="white")
        return fig, ax