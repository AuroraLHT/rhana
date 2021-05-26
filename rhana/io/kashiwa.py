import re
from pathlib import Path
from PIL import Image
import struct
from functools import partial

import numpy as np

# the path where I put the mask.
# I uploaded the mask file alongside the with notebok so the path is needed to be adjusted also
_maskpath = Path(__file__).parent / Path("mask.png")
_pimg = Image.open(_maskpath)
_APMASK = np.array(_pimg)[:,:, 0] == 255

_MAX_INT_KASHIWA = 2**14 - 1

kashiwa_config = dict(
    sub_ccd_dist = 225,
    pixel_real = 65 / 800,
    ccd_cam_width = 65,
    ccd_cam_height = 48.75,
    max_intensity = 16383,
    wave_length = 0.08 * 1e-7
)

def get_rheed_time(x, IT):
    if IT == "IT003":
        return int(re.findall("_([0-9]+)", str(x))[0])
    elif IT == "IT005":
        return int(re.findall("([0-9]+)p", str(x))[0])
    else:
        return int(Path(x).stem.split("-")[1])

def deaparture(img, sub=None, copy=False):
    if copy: img = img.copy()
    if sub is None: sub = img.min()
    img[_APMASK] = sub
    return img

# do not work for IT004
# reference could be found at numpy data type object page
def decode_rheed(file):
    """
        the rheed file is encoded by big endian unsigned int16 (2 byte).
        the first 8 byte (64bit) tells the size of the matrix: (height (int32), width (int32))
    """
    file = Path(file)
    with file.open(mode="rb") as f:
        c = f.read()
        dims = struct.unpack(">II", c[:8])
        h, w = dims
        arr = np.frombuffer(c, dtype=">H", offset=8)
        
        reshaped = arr.reshape(h, w)
        return reshaped.copy()

def decode_rheed2(file, h, w):
    # for IT004, small indience, with no h and w specification at the beginning of the file
    file = Path(file)
    with file.open(mode="rb") as f:
        c = f.read()
        arr = np.frombuffer(c, dtype=np.int16)
        reshaped = arr.reshape(h, w)
        return reshaped.copy()

def find_raw_rheed(path):
    """
        Find all the binary rheed data from a given folder

        Args:
            path : folder's directory
    """
    rheed_all = {}

    for p in list(path.glob("IT[0-9]*")):
        if p.stem in ['IT004', ] : continue 
        s = sorted(p.glob("*-*-*.bin"), key=partial(get_rheed_time, IT=p.name) )
        rheed_all[p.name] = s
    return rheed_all

exp_cons = {
    "IT003": (1e-5, 700),
    "IT004": (1e-5, 500),
    "IT005": (1e-5, 800),
    "IT006": (1e-3, 800),
    "IT007": (1e-3, 800),
    "IT008": (1e-3, 500),
    "IT009": (1e-3, 700),
    "IT010": (1e-2, 700),
    "IT011": (1e-3, 600),
    "IT012": (1e-2, 600),
    "IT013": (1e-1, 600),
    "IT014": (1e-3, 550),
    "IT015": (1e-5, 550),
    "IT016": (1e-5, 550),
    "IT017": (1e-5, 600),
    "IT018": (1e-2, 550),
    "IT020": (1e-5, 400),
    "IT021": (1e-3, 400),
    "IT022": (1e-5, 450),
    "IT023": (1e-3, 450),
    "IT024": (1e-2, 500),
    "IT025": (1e-2, 450),
    "IT027": (1e-2, 400),
    "IT028": (1e-3, 300),
    "IT029": (1e-2, 300),
    "IT030": (1e-5, 300),
}
