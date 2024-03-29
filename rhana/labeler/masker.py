import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

import torch
from fastai.learner import load_learner
from rhana.labeler.unet import RHEEDTensorImage, RHEEDTensorMask, rle_encode, rle_decode

_default_learner = Path(__file__).parent.parent.parent / "learner" / "UNet_May6_2021.pkl"

class UnetMasker:
    """A wrapper class that convert the input to Fastai format and pass it
    to the UNet model. 
    """
    def __init__(self, learner_path:Union[str, Path], cpu:bool=False, device:str=None):
        """Initializer of the UnetMasker

        Args:
            learner_path (_type_): the pickle directory that store the FastAI Unet learner
            cpu (bool, optional): use CPU if set to True else use GPU. Defaults to False.
        """
        if device is None: 
            device = torch.device("cpu")
        else: 
            device = torch.device(device)

        self.learn = load_learner(learner_path, cpu=True)
        self.learn.to(device)
        self.learn.dls.to(device) # patch dls do not convert to corrent dtype

        if "Normalize" in str(self.learn.dls.after_batch.fs[0].__class__):
            self.normalize = self.learn.dls.after_batch.fs[0]
        else:
            print("Check learner data transform normalization term")

        self.device = next(iter(self.learn.model.parameters())).device

    def predict_batch(self, rds, threshold:bool=0.5, return_raw=False):
        """Run batch prediction that maximize the usage of gpu. 

        Args:
        rds (np.array or list of RHEED): multiple input rheed pattern which are scaled to the range of (0, 1)
        threshold (bool, optional): Output probability above this value the is classicfied as yes in.
            the binary mask. Range: (0, 1). Defaults to 0.5.

        Returns:
            - dict: a dictionary contains { feature_name : feature_mask } if return_raw is False
            - floattensor: a raw tensor if return_raw is True
        """
        if isinstance(rds, list):
            if len(rds) > 1:
                inp = np.stack([rd.pattern for rd in rds], axis=0)
            else:
                inp = rds[0].pattern[None, ...]
            assert len(inp.shape)==3, f"current shape: {inp.shape}"

            inp = np.repeat(inp[:, None, :, :], 3, axis=1)
        elif isinstance(rds, np.array):
            inp = rds
        else:
            inp = np.array(rds)
        
        with torch.inference_mode():
            inp = torch.from_numpy(inp).to(self.device, torch.float32)
            inp = (inp - self.normalize.mean) / self.normalize.std

            scores = torch.sigmoid(self.learn.model(inp))
            masks = scores > threshold

            # let's test if the model need to resize the output to match the pattern shape or not
            # shape is not change, no resize is needed.
        masks = masks.detach().cpu().numpy()
        classes = self.learn.classes # classes variable store which label is predicted channel-wise

        if return_raw:
            return masks
        else:
            return { c: masks[:, i, :, :] for i, c in enumerate(classes) }


    def predict(self, rd, do_rle:bool=False, threshold:bool=0.5):
        """Predict masks for both streak and spot features. The original output from the UNet is
        2 x H x W tensor with each element contains a probability value of either this pixel contains
        the features or not. Use threshold to adjust your confidence level over the probability output.

        Args:
            rd (rhana.pattern.Rheed): an input rheed pattern which is scaled to the range of (0, 1)
            do_rle (bool, optional): do run line encoding if set to True else return the mask in Array. 
                Defaults to False.
            threshold (bool, optional): Output probability above this value the is classicfied as yes in.
                the binary mask. Range: (0, 1). Defaults to 0.5.

        Returns:
            dict: a dictionary contains { feature_name : feature_mask }
        """
        # shape = rd.pattern.shape

        inp = np.tile(rd.pattern, (3, 1, 1))[None, ...]
        # inp = RHEEDTensorImage.create(np.tile(rd.pattern, (3, 1, 1)))
        # inp = inp.to(self.device)
        # with self.learn.no_bar():
        #     outputs = self.learn.predict(inp)
        # scores = outputs[2]
        with torch.inference_mode():
            inp = torch.from_numpy(inp).to(self.device, torch.float32)
            inp = (inp - self.normalize.mean)/self.normalize.std
            scores = torch.sigmoid(self.learn.model(inp))
            masks = scores > threshold
            masks = masks[0].detach().cpu().numpy()

        # masks = scores > threshold
        classes = self.learn.classes # classes variable store which label is predicted channel-wise

        # let's test if the model need to resize the output to match the pattern shape or not
        # shape is not change, no resize is needed.

        if do_rle:
            return { c: rle_encode(masks[i, ...]) for i, c in enumerate(classes) }
        else:
            return { c: masks[i, ...] for i, c in enumerate(classes) }