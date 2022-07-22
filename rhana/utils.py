import time
import pickle
import yaml
import signal
from pathlib import Path
import threading
import multiprocessing

import colorlover as cl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_CM_rgb = cl.scales['10']['qual']['Paired']
_CM = cl.to_numeric( cl.scales['10']['qual']['Paired'] )
_CM = list( map(lambda x: "#{:02x}{:02x}{:02x}".format(*tuple(map(int, x)) ), _CM ) ) # transfer "rgb(0,0,0)" into #FFFFFF format

def save_pickle(obj, file):
    with Path(file).open("wb") as f:
        pickle.dump(obj, f)

def load_pickle(file):
    with Path(file).open("rb") as f:
        return pickle.load(f)

def load_yaml(file):
    with Path(file).open("r") as f:
        return yaml.load_all(f)

def gaussian(x, A, x0, sig):
    return A*np.exp(-(x-x0)**2/(2*sig**2))

def multi_gaussian(x, *pars):
    offset = pars[-1]
    summation = offset
    for i in range(len(pars)//3):
        g = gaussian(x, pars[i*3], pars[i*3+1], pars[i*3+2])
        summation += g
    return summation

def show_circle(ax, xy, radius, color="black", **kargs):
    """
        plot a circle on the matplotlib axes
    """
    
    cir = mpatches.Circle(xy, radius=radius, fill=False, color=color, **kargs)
    ax.add_patch(cir)
    return ax

def _create_figure(ax=None, **subplots_kargs):
    if ax is None:
        return plt.subplots(**subplots_kargs)
    else:
        return None, ax

def crop(img, sx, sy, ex, ey):
    return img[sx:ex, sy:ey]


class Timeout:
    """
    Ref: https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
    A timeout class for killing code pieces that run too long
    
    Example:
        import time
        with timeout(seconds=3):
            time.sleep(4)
    """
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
        
def timeout_MP(func, args=(), ):
    l = threading.Lock()

    time_thread = threading.Timer(1.0, lambda l: l.acquire(), args=(l,))
    # work_thread = threading.Thread(target=_fit, daemon=True)
    my_proc = multiprocessing.Process(target=func, args=args)

    time_thread.start()
    my_proc.start()

    while not l.locked():
        time.sleep(.1)
    # print("main thread start to terminate")
    if my_proc.is_alive:
        # print("kill still running child process")
        my_proc.terminate()
    time_thread.join()