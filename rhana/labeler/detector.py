"""
update (6/10/2023)
The below mentioned model has been developed. milestone achieved. (yeh)

The author wishes to upgrade the model to a object detector. If you are interest in extending the 
current work, please email this guy (Haotong Liang, Email: hliang16@umd.edu). He is very willing to talk 
with you and excite about the possible things that could be done in the future.
"""

import os.path as osp
from pathlib import Path
import json
from tqdm.notebook import tqdm

import numpy as np

import torch, torchvision
# print(torch.__version__, torch.cuda.is_available())

import cv2
import mmcv
from mmcv.transforms import Compose
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

import mmdet
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

import mmengine
from mmengine import Config, runner
from mmengine.utils import track_iter_progress

from .hook import FeatureExtractor
from .aux_classifier import ResnetClassifierHead, ResnetClassifierHeadConfig, transform_backbone_features
from ..utils import to_img

from typing import Union, Optional, List, Dict, Tuple

class CascadeMaskRCNNDetector:
    def __init__(
            self, 
            config_path:Union[str, Path], 
            checkpoint_path:Union[str, Path], 
            device:str=None
        ):
        """Initializer of the UnetMasker

        Args:
            learner_path (_type_): the pickle directory that store the FastAI Unet learner
            cpu (bool, optional): use CPU if set to True else use GPU. Defaults to False.
        """
        if device is None: 
            device = torch.device("cpu")
        else: 
            device = torch.device(device)

        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.device = next(iter(self.model.parameters())).device

        self.visualizer = None

    def preprocess_rd(self, rd):
        if rd.pattern.ndim == 2:
            inp = np.repeat(rd.pattern[..., None], 3, axis=-1)
        else:
            inp = rd.pattern

        inp *= 255

        # return torch.from_numpy(inp)
        return inp

    def predict(self, rd):
        """Predict instance segmentation masks for both streak and spot features.
        Each instance mask contains feature detected by the model.

        Args:
            rd (rhana.pattern.Rheed): an input rheed pattern which is scaled to the range of (0, 1)

        Returns:
            dict: a dictionary contains { feature_name : feature_mask }
        """
        # shape = rd.pattern.shape
        inp = self.preprocess_rd(rd)
        with torch.inference_mode():
            result = inference_detector(self.model, inp)

        return result
    
    def predict_batch(self, rds):
        batch = np.stack( [ self.preprocess_rd(rd) for rd in rds ], axis=0 )

        with torch.inference_mode:
            result = inference_detector(self.model, batch)

        return result
    
    def plot_prediction(self, rd, result, name="result", out_path=None):
        if self.visualizer is None:
            self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
            self.visualizer.dataset_meta = self.model.dataset_meta

        inp = to_img( self.preprocess_rd(rd) )
        
        self.visualizer.add_datasample(
            name,
            inp,
            data_sample=result,
            draw_gt = None,
            wait_time = 0,
            out_file = out_path
        )

        return self.visualizer.show()


class CascadeMaskRCNNDetectorAndClassifier:
    def __init__(
            self, 
            config_path:Union[str, Path], 
            checkpoint_path:Union[str, Path],
            classifier_config_path:Union[str, Path],
            classifier_checkpoint_path:Union[str, Path],
            device:str=None
        ):
        """Initializer of the UnetMasker

        Args:
            learner_path (_type_): the pickle directory that store the FastAI Unet learner
            cpu (bool, optional): use CPU if set to True else use GPU. Defaults to False.
        """
        if device is None: 
            device = torch.device("cpu")
        else: 
            device = torch.device(device)

        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.device = next(iter(self.model.parameters())).device

        self.fe = FeatureExtractor(self.model, layers=["backbone"])

        self.classifier_config = ResnetClassifierHeadConfig.load(classifier_config_path)
        self.classifier = ResnetClassifierHead(config=self.classifier_config)
        self.classifier.load_state_dict( torch.load(classifier_checkpoint_path) )
        self.classifier.to(self.device)

        self.visualizer = None

    @property
    def classifier_classes(self):
        return self.classifier_config['classes']

    @property
    def detector_classes(self):
        return self.model.cfg['metainfo']['classes']

    def preprocess_rd(self, rd):
        if rd.pattern.ndim == 2:
            inp = np.repeat(rd.pattern[..., None], 3, axis=-1)
        else:
            inp = rd.pattern

        inp *= 255

        return inp

    def predict(self, rd):
        """Predict instance segmentation masks for both streak and spot features.
        Each instance mask contains feature detected by the model.

        Args:
            rd (rhana.pattern.Rheed): an input rheed pattern which is scaled to the range of (0, 1)

        Returns:
            dict: a dictionary contains { feature_name : feature_mask }
        """
        # shape = rd.pattern.shape
        inp = self.preprocess_rd(rd)
        with torch.inference_mode():
            result = inference_detector(self.model, inp)
            backbone_features = self.fe._features['backbone']
            classification_result = self.classifier( transform_backbone_features(backbone_features) )
            classification_result = torch.sigmoid(classification_result)

        return result, classification_result
    
    def predict_batch(self, rds):
        batch = np.stack( [ self.preprocess_rd(rd) for rd in rds ], axis=0 )

        with torch.inference_mode:
            result = inference_detector(self.model, batch)
            backbone_features = self.fe._features['backbone']
            classification_result = self.classifier( transform_backbone_features(backbone_features) )
            classification_result = torch.sigmoid(classification_result)

        return result, classification_result
    
    def plot_prediction(self, rd, result, classification_result, name="result", out_path=None):
        # TODO: plot the classification result on the figure

        if self.visualizer is None:
            self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
            self.visualizer.dataset_meta = self.model.dataset_meta

        inp = to_img( self.preprocess_rd(rd) )
        
        self.visualizer.add_datasample(
            name,
            inp,
            data_sample=result,
            draw_gt = None,
            wait_time = 0,
            out_file = out_path
        )

        fig = self.visualizer.show()
        ax = fig.axes[0]
        classes = self.classifier.config['classes']
        ax.text(
            0.5, 0.1, 
            "\n".join([ f"{classes[i]}:{p:.2f}" for i, p in enumerate(classification_result[0])]), 
            transform=ax.transAxes, 
            fontdict={"color":"white"}, 
            ha="center",
            va="center"
        )

        return fig