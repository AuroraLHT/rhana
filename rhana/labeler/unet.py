from typing import Tuple, List, Union
import numpy as np

from fastai.data.all import *
from fastai.vision.all import *

from skimage import io as skm_io

def bce_logits_floatify(input, target, reduction='mean'):
    """Loss function for the model. See torch.nn.Functional.binary_cross_entropy_with_logits
    for more details.

    Args:
        input (torch.Tensor): model predictions values
        target (torch.Tensor): target values
        reduction (str, optional): how to reduce the dimension. Defaults to 'mean'.

    Returns:
        torch.Tensor: the BCE loss value
    """
    return F.binary_cross_entropy_with_logits(input, target.float(), reduction=reduction)


# can use sigmoid on the input too, in this case the threshold would be 0.5
def dice_metric(pred, targs, threshold=0.5):
    """The dice metric for evaluating the performance of the model

    Args:
        pred (torch.Tensor): model predictions values
        targs (torch.Tensor): target values
        threshold (float, optional): threshold for converting the probablistic mask 
            into binary mask. Defaults to 0.5.

    Returns:
        float: the metric value
    """
    pred = (pred > threshold).float()
    targs = targs.float()  # make sure target is float too
    return 2.0 * (pred*targs).sum() / ((pred+targs).sum() + 1.0)


def open_rle_from_df(fn, shape=None, cls=torch.Tensor, df=None):
    row = df.loc[fn]
    masks = []
    for c in df.columns:
        rle = row[c]
        if isinstance(rle, str):
            masks.append(rle_decode(rle, shape))
        else:
            masks.append(np.zeros(shape))
    return cls(np.stack(masks, axis=0).astype(int))


def rle_encode(img)->str:
    """"Return run-length encoding string from `img`."

    Args:
        img (Array): the mask array

    Returns:
        str: rle enecoded mask
    """
    
    pixels = np.concatenate([[0], img.flatten() , [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle:str, shape:Union[Tuple, List]):
    """Return an image array from run-length encoded string `mask_rle` with `shape`.

    Args:
        mask_rle (str): the rle encoded mask
        shape (Union[Tuple, List]): the (h, w) of the binary mask
    Returns:
        Array: a mask with the shape of (h, w)
    """    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint)
    for low, up in zip(starts, ends): img[low:up] = 1
    return img.reshape(shape)



def open_rheed(fn, cls=torch.Tensor):
    """Open rheed from a image

    Args:
        fn (str): image path
        cls (class, optional): the final convertion of the image data. Defaults to torch.Tensor.

    Returns:
        cls: image data
    """
    im = skm_io.imread(fn)
    im = np.tile(im, (3, 1, 1)) / 255 # add 255 to rescale it to 1

    return cls(im)

class RHEEDTensorImage(TensorImage):
    """This Class is a data class that allow the pipeline to
    recognize the RHEED image.
    """
    
    def __init__(self, x, chnls_first=True):
        self.chnls_first = chnls_first

    @classmethod
    def create(cls, data:Union[Path, str, ndarray], chnls_first=True):
        """Create a RHEEDTensorImage object from multiple different sources

        Args:
            data (Path,str,ndarray): source of the image. Could be an Array,
            a file path, or an pandas series. The name of series should store
            the file path.
            chnls_first (bool, optional): see torch.Tensor for more details. Defaults to True.

        Returns:
            RHEEDTensorImage: a data object
        """

        if isinstance(data, Path) or isinstance(data, str):
            im = open_rheed(fn=data, cls=torch.Tensor)
        elif isinstance(data, pd.Series):
            im = open_rheed(fn=data.name, cls=torch.Tensor)
        elif isinstance(data, ndarray): 
            im = torch.from_numpy(data)
        else:
            im = data
        
        return cls(im, chnls_first=chnls_first)

    
    def show(self, ctx=None):
        """plot the image

        Args:
            ctx (Matplotlib.pyplot.Axes, optional): plot's axes. Defaults to None.

        Returns:
            Matplotlib.pyplot.Axes: plot's axes
        """
        visu_img = self[0, ...]
                
        plt.imshow(visu_img) if ctx is None else ctx.imshow(visu_img)
        
        return ctx
    
    def __repr__(self):
        
        return (f'RHEEDTensorImage: {self.shape}')

def open_rle_from_row(fn, shape=None, cls=torch.Tensor):
    row = fn
    masks = []
    for c in row.index:
        rle = row[c]
        if isinstance(rle, str):
            masks.append(rle_decode(rle, shape))
        else:
            masks.append(np.zeros(shape))
    return cls(np.stack(masks, axis=0).astype(int))

class RHEEDTensorMask(TensorMask):
    """A data clas that store masks with more than one channel
    """
    def __init__(self, x, chnls_first=True):
        self.chnls_first = chnls_first

    @classmethod
    def create(cls, data:Union[Path, str, ndarray, pd.Series], chnls_first:bool=True, shape:Union[List, Tuple]=None):
        """Create a RHEEDTensorMask object from multiple different sources

        Args:
            data (Path,str,ndarray,pd.Series): data source could be a file path (not recommanded), 
            a pd.series with a format that is digestable by open_rle_from_row, or a Array
            chnls_first (bool, optional): see torch.Tensor for more details. Defaults to True.
            shape (Union[List, Tuple], optional): the shape of the mask. Need to be specified if the data
                is not an Array. Defaults to None.

        Returns:
            RHEEDTensorMask: a data object
        """
        if isinstance(data, Path) or isinstance(data, str):
            # this implementation is a little bit hacky.
            global df
            im = open_rle_from_df(fn=data, shape=shape, cls=torch.Tensor, df=df)
        elif isinstance(data, pd.Series):
            im = open_rle_from_row(fn=data, shape=shape, cls=torch.Tensor)
        elif isinstance(data, ndarray): 
            im = torch.from_numpy(data)
        else:
            im = data
            
        return cls(im, chnls_first=chnls_first)

    def show(self, chn=0, ctx=None):
        """Plot one channel of the mask.

        Args:
            chn (int, optional): the channel you want to plot. Defaults to 0.
            ctx (matplotlib.pyplot.Axes, optional): plot's axes. Defaults to None.

        Returns:
            matplotlib.pyplot.Axes: plot's axes
        """
        visu_mask = self[chn, ...]                
        plt.imshow(visu_mask) if ctx is None else ctx.imshow(visu_mask)
        return ctx
    
    def __repr__(self):
        
        return (f'RHEEDTensorMask: {self.shape}')