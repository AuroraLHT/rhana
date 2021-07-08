from fastai.learner import load_learner
import numpy as np
from pathlib import Path

from rhana.labeler.unet import RHEEDTensorImage, RHEEDTensorMask, rle_encode, rle_decode

_default_learner = Path(__file__).parent.parent.parent / "learner" / "UNet_May6_2021.pkl"

class UnetMasker:
    def __init__(self, learner_path, cpu=False):
        self.learn = load_learner(learner_path, cpu=cpu)

    def predict(self, rd, do_rle=False):
        # shape = rd.pattern.shape

        inp = RHEEDTensorImage.create(np.tile(rd.pattern, (3, 1, 1)))
        outputs = self.learn.predict(inp)
        scores = outputs[2] 
        masks = scores > 0.5
        classes = self.learn.classes # classes variable store which label is predicted channel-wise

        # let's test if the model need to resize the output to match the pattern shape or not
        # shape is not change, no resize is needed.

        if do_rle:
            return { c: rle_encode(masks[i, ...]) for i, c in enumerate(classes)}
        else:
            return { c: masks[i, ...] for i, c in enumerate(classes)}

