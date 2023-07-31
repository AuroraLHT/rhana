import re
from pathlib import Path
from PIL import Image
import struct
from functools import partial

import numpy as np

# the path where I put the mask.
# I uploaded the mask file alongside the with notebok so the path is needed to be adjusted also
_maskpath = Path(__file__).parent / Path("asset/mask.png")
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

    rheed_all['IT003'] = sorted( list( (path/ "IT003").iterdir() ), key=partial(get_rheed_time, IT="IT003") )
    rheed_all['IT005'] = sorted( list( (path/ "IT005").glob("*p.png") ), key=partial(get_rheed_time, IT="IT005") )
    rheed_all['IT004'] = sorted( list( (path/ "IT004").glob("*-*.bin") ), key=lambda x: float(x.stem.split("-")[1]) )

    return rheed_all

def hv2wl(voltage):
    """
    Convert accelerating voltage to electron wavelength
    Args:
        voltage (float): acceleration voltage of electron in kV

    Returns:
        wavelength in mm
    """
    # me = 510998.946 # in electron volts / (the speed of light^2)
    me = 9.10938356e-31 # in kilograms
    # h = 4.1357e-15 # in eV s
    h = 6.62607004e-34 #in m2 kg / s    
    
    E = voltage * 1e3 * 1.6022e-19 # convert keV to J
    wave_length = h / (me * np.sqrt( 2 * E / me ) ) * 1e3 # use matter wave equation to convert kinetic energy to momentum and wavelength
    
    return wave_length

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

# kV
# from Mikk slide
exp_voltage = {
    "IT003": 25, #
    "IT004": 25, 
    "IT005": 23, #
    "IT006": 24, #
    "IT007": 24, #
    "IT008": 24, #
    "IT009": 24, #
    "IT010": 24, #
    "IT011": 22, #
    "IT012": 23, #
    "IT013": 23, #
    "IT014": 25, #
    "IT015": 25, #
    "IT016": 23, #
    "IT017": 23, #
    "IT018": 23, #
    "IT020": 24, 
    "IT021": 24,
    "IT022": 24,
    "IT023": 25,
    "IT024": 25,
    "IT025": 25,
    "IT027": 24,
    "IT028": 25,
    "IT029": 25,
    "IT030": 25,
}

# keV
# calibrate by the first pattern
# using a fix Al2O3 
# cali_exp_voltage = {
#     'IT003': 32.113,
#     'IT004': 32.113,
#     'IT005': 32.821,
#     'IT006': 27.701,
#     'IT007': 24.656,
#     'IT008': 25.127,
#     'IT009': 26.127,
#     'IT010': 26.127,
#     'IT011': 25.623,
#     'IT012': 29.493,
#     'IT013': 26.127,
#     'IT014': 25.623,
#     'IT015': 28.256,
#     'IT016': 27.183,
#     'IT017': 23.741,
#     'IT018': 25.623,
#     'IT020': 19.524,
#     'IT021': 21.139,
#     'IT022': 19.746,
#     'IT023': 19.86,
#     'IT024': 20.207,
#     'IT025': 20.556,
#     'IT027': 19.86,
#     'IT028': 22.881,
#     'IT029': 24.011,
#     'IT030': 25.229
# }

# in mm
# calibrate by the first pattern
# using a fix Al2O3 
# not 
# cali_exp_wave_length = {
#     'IT003': 6.843785072628894e-09,
#     'IT004': 6.843785072628894e-09,
#     'IT005': 6.769552539467009e-09,
#     'IT006': 7.368715128559365e-09,
#     'IT007': 7.810512322096808e-09,
#     'IT008': 7.736847895056062e-09,
#     'IT009': 7.587435985197064e-09,
#     'IT010': 7.587435985197064e-09,
#     'IT011': 7.661668518358947e-09,
#     'IT012': 7.141283311397571e-09,
#     'IT013': 7.587435985197064e-09,
#     'IT014': 7.661668518358947e-09,
#     'IT015': 7.295997545053845e-09,
#     'IT016': 7.4385921814592e-09,
#     'IT017': 7.959545494541715e-09,
#     'IT018': 7.661668518358947e-09,
#     'IT020': 8.777239571564727e-09,
#     'IT021': 8.435239686640326e-09,
#     'IT022': 8.727624970318771e-09,
#     'IT023': 8.702628300988748e-09,
#     'IT024': 8.627638292998682e-09,
#     'IT025': 8.553973865957933e-09,
#     'IT027': 8.702628300988748e-09,
#     'IT028': 8.10782119215844e-09,
#     'IT029': 7.914665110971902e-09,
#     'IT030': 7.721319661078321e-09
# }

cali_exp_pixel_real = {
    # 'IT003': 65 / 800 * 346 / 604,
    # 'IT003': 65 / 800 * 110 / 210,
    # IT003 camera focus might be chaned
    # based on the XRD, AFM and RHEED from near by sample
    # we know the distance is refer to Fe3O4
    # thus we backcalculated the correct pixel real value
    'IT003': 65 / 800 * 0.8984615384615384,
    'IT004': 65 / 800 * 110 / 97.5,
    'IT005': 65 / 800 * 110 / 97.5,
    'IT006': 65 / 800 * 1,
    'IT007': 65 / 800 * 1,
    'IT008': 65 / 800 * 1,
    'IT009': 65 / 800 * 1,
    'IT010': 65 / 800 * 1,
    'IT011': 65 / 800 * 1,
    'IT012': 65 / 800 * 1 ,
    'IT013': 65 / 800 * 1,
    'IT014': 65 / 800 * 1,  
    'IT015': 65 / 800 * 1 ,
    'IT016': 65 / 800 * 1,
    'IT017': 65 / 800 * 1,
    'IT018': 65 / 800 * 1,
    'IT020': 65 / 800 * 110 / 130,
    'IT021': 65 / 800 * 110 / 130,
    'IT022': 65 / 800 * 110 / 130,
    'IT023': 65 / 800 * 110 / 130,
    'IT024': 65 / 800 * 110 / 130,
    'IT025': 65 / 800 * 110 / 130,
    'IT027': 65 / 800 * 110 / 130,
    'IT028': 65 / 800 * 110 / 114,
    'IT029': 65 / 800 * 110 / 114,
    'IT030': 65 / 800 * 110 / 114
}

pyroT2caliT = {
    300:383,
    400:524,
    450:600,
    500:666,
    550:734,
    600:805,
    700:927,
    800:1048,
}

cali_exp_cons= {}
for name, con in exp_cons.items():
    cali_exp_cons[name] = (con[0], pyroT2caliT[con[1]])

 