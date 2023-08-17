import warnings
import itertools
from pathlib import Path
import pickle
import copy

import numpy as np
import pandas as pd

from PIL import Image

from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy import ndimage

# from skimage import data
# from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage.filters import gaussian as skim_gaussian
from skimage.transform import rotate as skim_rotate
from skimage.restoration import estimate_sigma
from skimage.measure._regionprops import RegionProperties

from skimage.transform import resize
from skimage import restoration

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import plotly.graph_objects as go

from skimage.morphology import reconstruction
from skimage.measure import label, regionprops
from skimage.measure import moments

from typing import List, Dict, Union

from rhana.utils import _CM_rgb, _CM, _CM_np, multi_gaussian, gaussian, show_circle, _create_figure, crop
from rhana.io import kashiwa as ksw
from rhana.spectrum.spectrum import CollapseSpectrum, analyze_peaks_distance_cent, get_center_peak_idxs, get_center_peak_idx, get_peaks_distance
from rhana.utils import *
from rhana.periodicity import PeriodicityAnalyzer

from dataclasses import dataclass, field

from warnings import warn

def image_bg_sub_dilation(image, seed_bias=.1):
    im = image.astype(float)
    #im = gaussian_filter(im, 1)
    seed = np.copy(im)
    seed[1:-1, 1:-1] = im.min()
    mask = im

    h = seed_bias
    seed = im - h
    dilated = reconstruction(seed, mask, method='dilation')
    hdome = im - dilated
    return im, dilated, hdome


def correct_zero_laue(xy, db, ndb):
    (nx, ny, _) = ndb
    (x, y, _) = db
    return np.array( [xy[0] + (nx-x), xy[1] + (ny-y)] )


@dataclass 
class RheedConfig:
    """A storage class for experimental configuration of the RHEED
    """
    sub_ccd_dist : int = field(metadata={"unit" : "mm"})
    pixel_real : int = field(metadata={"unit" : "mm/pixel"})
    ccd_cam_width : int = field(metadata={"unit" : "mm"})
    ccd_cam_height : int = field(metadata={"unit" : "mm"})
    max_intensity : int = field(metadata={"unit": "unitless"})
    wave_length : float = field(metadata={"unit": "mm"})

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)
    
    def hdist2G(self, dist):
        """
        convert horizontal distance (d) in pixel space to reciprocal space (dG)
        
        Args:        
            dist (float): horizontal distance is in pixel
            
        Returns
            float : dG is in nm^-1
        """
        real_dist = self.pixel_real * dist
        # print(real_dist)
        k0 = 2*np.pi/self.wave_length
        # print(k0)
        r = (self.sub_ccd_dist/real_dist)**2
        dG = k0 / np.sqrt( r + 1 ) * 1e-6
        return dG


class Rheed:
    """The main class that allow you to:
     1. load rheed from image or binary form (need more supports)
     2. simple manipulation of the rheed pattern
     3. extract simple features from the rheed pattern
     4. visualize the rheed pattern along with other extracted features
    """
    
    _CMAP = "cividis"

    def __init__(self, pattern:np.array, min_max_scale:bool=False, standard_norm:bool=False, AOI:np.array=None, config:RheedConfig=None):
        """Rheed Class Initializer

        Args:
            pattern (np.array): a 2d rheed pattern stored in a numpy array
            min_max_scale (bool, optional): scale pattern to 0 and 1. Defaults to False.
            standard_norm (bool, optional): scale pattern by mean and std. Defaults to False.
            AOI (np.array, optional): a 2d mask that cover the area of interest. Defaults to None.
            config (RheedConfig, optional): Rheed experimental configuration. Defaults to None.
        """
        self.pattern = pattern.copy()
        self.AOI = AOI
        if AOI is not None:
            assert AOI.shape == pattern.shape, f"Shape of the Area of Interest {AOI.shape} is different than the pattern {pattern.shape}"
            self.pattern[~self.AOI] = self.pattern.min()

        if standard_norm:
            self.standard_norm()
        if min_max_scale:
            self.min_max_scale()
        self.config = config
  
    @classmethod
    def from_kashiwa(cls, path, contain_hw=True, min_max_scale=False, standard_norm=False, log=False, use_mask=True, rotate=0, config:RheedConfig=None):
        """Load Rheed pattern form Mikk's Labs formation. Format name unknown. The detail decoding method see io.kashiwa 

        Args:
            path (str): rheed binary file path
            contain_hw (bool, optional): Does the file contain height and width info. Defaults to True.
            min_max_scale (bool, optional): scale pattern to 0 and 1. Defaults to False.
            standard_norm (bool, optional): scale pattern by mean and std. Defaults to False.
            log (bool, optional): _description_. Defaults to False.
            use_mask (bool, optional): use a predefined mask as AOI. Defaults to True.
            rotate (int, optional): fix the slight tilting in RHEED. Defaults to 0.
            config (RheedConfig, optional): Rheed experimental configuration.. Defaults to None.

        Returns:
            Rheed: an instantiated rheed obj
        """
        if contain_hw:
            pattern = ksw.decode_rheed(path)
        else:
            pattern = ksw.decode_rheed2(path, 600, 800)

        AOI = ~ksw._APMASK if use_mask else None
        pattern = np.log10(pattern) if log else pattern / ksw._MAX_INT_KASHIWA
        if rotate != 0: pattern = skim_rotate(pattern, rotate)

        return cls(pattern, min_max_scale=min_max_scale, standard_norm=standard_norm, AOI=AOI, config=config)

    @classmethod
    def from_multi(cls, patterns, min_max_scale=False, standard_norm=False, AOI=None):
        """
        Convert multiple patterns into one rheed pattern

        Args:
            patterns (List): a list of patterns in np.array
            min_max_scale (bool, optional): scale pattern to 0 and 1. Defaults to False.
            standard_norm (bool, optional): scale pattern by mean and std. Defaults to False.
            AOI (np.array, optional): a 2d mask that cover the area of interest. Defaults to None.
        Returns:
            Rheed: an instantiated rheed obj
        """
        return cls(np.mean(patterns, axis=0), min_max_scale=min_max_scale, standard_norm=standard_norm, AOI=AOI)

    @classmethod 
    def from_multi_kashiwa(cls, paths, contain_hw=True, min_max_scale=False, standard_norm=False, log=False):
        """
        Convert multiple patterns stored in kashiwa's binary format into one rheed pattern

        Args:
            path (List): a list of binary paths
            min_max_scale (bool, optional): scale pattern to 0 and 1. Defaults to False.
            standard_norm (bool, optional): scale pattern by mean and std. Defaults to False.
            log (bool): convert the intensity to log scale. Defaults to False.
        Returns:
            Rheed: an instantiated rheed obj
        """        
        
        patterns = [ cls.from_kashiwa(path,contain_hw=contain_hw, min_max_scale=min_max_scale, standard_norm=standard_norm, log=log).pattern for path in paths ]
        return cls.from_multi(patterns, AOI=~ksw._APMASK)

    @classmethod
    def from_image(cls, path:str, rotate:float=0, crop_box:np.array=None, min_max_scale:bool=False, standard_norm:bool=False,  AOI:np.array=None, config:RheedConfig=None):
        """Create a rheed obj from an image.

        Args:
            path (str): the image path
            rotate (int, optional): fix the slight tilting in RHEED. Defaults to 0.
            crop_box (array, optional): a box that define the area to crop out. see PIL.Image.crop. Defaults to None.
            min_max_scale (bool, optional): scale pattern to 0 and 1. Defaults to False.
            standard_norm (bool, optional): scale pattern by mean and std. Defaults to False.
            AOI (np.array, optional): a 2d mask that cover the area of interest. Defaults to None.
            config (RheedConfig, optional): Rheed experimental configuration.. Defaults to None..

        Returns:
            Rheed: an instantiated rheed obj
        """
        img = Image.open(path)
        if rotate != 0 :
            img = img.rotate(rotate)
        if crop_box is not None:
            img = img.crop(crop_box)

        img = np.array(img)

        if len(img.shape) == 3:
            img = np.mean(img[:, :, :3], axis=2)

        if img.max() > 1:
            img = img / 255
        
        return cls(np.array(img), min_max_scale=min_max_scale, standard_norm=standard_norm, AOI=AOI, config=config)

    def rotate(self, angle, center=None, inplace=False):
        """
            rotate the rheed pattern wrt to the given center position
            center is in image coordinate meaning x is row id and y is col id
        """
        pattern = skim_rotate(self.pattern, angle, center=center[::-1])
        return self._update_pattern(pattern=pattern, inplace=inplace)

    def clip(self, min_v=0, max_v=1, inplace=True):
        """ Clip the pattern by the given min and max value

        Args:
            min_v (float): the lowest intensity value before clip
            max_v (float): the largest intensity value before clip
            inplace (bool, optional): the operated result would overwrite the stored pattern if True. Defaults to True.

        Returns:
            Rheed: either itself or a newly created rheed obj
        """
        pattern = np.clip(self.pattern, min_v, max_v)
        return self._update_pattern(pattern, inplace=inplace)
        
    def mean_clip(self, inplace:bool=True):
        """Clip the value that below the average intensity to average intensity and shift the
        average intensity to 0. An effective way to remove noise.

        Args:
            inplace (bool, optional): the operated result would overwrite the stored pattern if True. Defaults to True.
            
        Returns:
            Rheed: either itself or a newly created rheed obj
        """
        if self.AOI is not None:
            mean = self.pattern[self.AOI].mean()
        else:
            mean = self.pattern.mean()
        
        return self._update_pattern(np.clip(self.pattern-mean, 0, 1), inplace=inplace)

    def rolling_ball(self, resize_shape=(10, 10), inplace=False, **rollingball_kargs):
        _pattern = resize(self.pattern, resize_shape)
        _background = restoration.rolling_ball(_pattern, **rollingball_kargs)
        background = resize(_background, self.pattern.shape)
        pattern =  np.clip( self.pattern - background, 0, None)
        return self._update_pattern(pattern, inplace=inplace)

    def _update_pattern(self, pattern, inplace):
        if inplace:
            self.pattern = pattern
            return self
        else:
            rd = copy.deepcopy(self)
            rd.pattern = pattern
            return rd
            # return Rheed(pattern, config=self.config)

    def crop(self, sx:int, sy:int, ex:int, ey:int, inplace=False):
        """Crop the pattern by the given corner points. 

        Args:
            sx (int): crop starting x position
            sy (int): crop starting y position
            ex (int): crop end x position
            ey (int): crop end y position
            inplace (bool, optional): the operated result would overwrite the stored pattern if True. Defaults to True.

        Returns:
            Rheed: either itself or a newly created rheed obj
        """
        sx = max(0, sx)
        sy = max(0, sy)
        ex = min(self.pattern.shape[0], ex)
        ey = min(self.pattern.shape[1], ey)

        return self._update_pattern(crop(self.pattern, sx, sy, ex, ey), inplace=inplace)

    def remove_bg(self, dilation_bias=0.5, inplace=True):
        """Remove background by dilation methods. Not very intuitive to use.
        see skimage.morphology.reconstruction for more detail
        Args:
            dilation_bias (float, optional): background level. Defaults to 0.5.
            inplace (bool, optional): the operated result would overwrite the stored pattern if True. Defaults to True.

        Returns:
            Rheed: either itself or a newly created rheed obj
        """
        im, dilated, hdome0 = image_bg_sub_dilation(self.pattern, dilation_bias)
        dilated = hdome0 / hdome0.max()

        return self._update_pattern(dilated, inplace=inplace)

    def smooth(self, inplace=True, **gaussian_kargs):
        """Apply a 2d gaussian kernel to smooth the pattern
        see skimage.filters.gaussian for more details
        Args:
            inplace (bool, optional): the operated result would overwrite the stored pattern if True.

        Returns:
            Rheed: either itself or a newly created rheed obj
        """
        smoothed = skim_gaussian(self.pattern, **gaussian_kargs)
        return self._update_pattern(smoothed, inplace=inplace)

    def gamma_correction(self, gamma=1, coeff=1, inplace=True):
        """Apply gamma correction to enhance image contrast
        see https://en.wikipedia.org/wiki/Gamma_correction for description
        The fomular is simple O = coeff * (I**gamma)
        Args:
            gamma (float): the gamma value
            coeff (float): the coefficient value, default to 1 for no change
            inplace (bool, optional): the operated result would overwrite the stored pattern if True.

        Returns:
            Rheed: either itself or a newly created rheed obj
        """
        if gamma == 1 and coeff == 1: 
            return self
        else:
            pattern = coeff * (self.pattern**gamma)
            
            return self._update_pattern(pattern, inplace=inplace)

    def min_max_scale(self, min_v=None, max_v=None, inplace=True):
        """Scale the pattern to min==0 and max==1

        Args:
            min
            inplace (bool, optional): the operated result would overwrite the stored pattern if True.

        Returns:
            Rheed: either itself or a newly created rheed obj
        """
        pattern = ( self.pattern - self.pattern.min() ) / (self.pattern.max() - self.pattern.min() + 1e-5)
        return self._update_pattern(pattern, inplace=inplace)

    def standard_norm(self, inplace=True):
        """Normalized the pattern by mean and std.

        Args:
            inplace (bool, optional): the operated result would overwrite the stored pattern if True.

        Returns:
            Rheed: either itself or a newly created rheed obj
        """
        if self.AOI is not None:
            pattern = ( self.pattern - self.pattern[self.AOI].mean() ) / (self.pattern[self.AOI].std() + 1e-5)
        else:
            pattern = ( self.pattern - self.pattern.mean() ) / (self.pattern.std() + 1e-5)
        return self._update_pattern(pattern, inplace=inplace)

    def noise_based_correction(self, max_value, prev_noise=None, ratio=100, squash_ratio=0.8, accumulate_rate=0.9, noise_region=None, subtract_median=False, inplace=False):
        frame = self.pattern
        if noise_region is None:
            noise = estimate_sigma(frame, channel_axis=None)
        else:
            sx, sy, ex, ey = noise_region
            noise = estimate_sigma(frame[sx:ex, sy:ey], channel_axis=None)
        if prev_noise is None:
            pass
        else:
            noise = noise * (1-accumulate_rate) + prev_noise * accumulate_rate
        
        if subtract_median:
            frame = frame - np.median(frame.ravel())
            
        enhanced = frame.copy()

        spliter = min( noise * ratio, max_value )
        if spliter == max_value: 
            pass
        else:
            # spliter = squash_ratio * max_value

            low_intensity_mask = enhanced < spliter
            high_intensity_mask = ~low_intensity_mask

            alpha_low = max_value * squash_ratio / enhanced[low_intensity_mask].max()

            alpha_high = (max_value - max_value * squash_ratio) / (max_value - enhanced[high_intensity_mask].min() )
            beta_high =  max_value - alpha_high * max_value

            enhanced[low_intensity_mask] = enhanced[low_intensity_mask] * alpha_low
            enhanced[high_intensity_mask] = enhanced[high_intensity_mask] * alpha_high + beta_high
        
        rd = self._update_pattern(enhanced, inplace=inplace)
        rd.noise = noise
        return rd

    def get_blobs(self, max_sigma=30, num_sigma=10, threshold=.1, **blob_kargs):
        """
        Find bright spots in the RHEED pattern image using the blob detection algorithm
        with Laplacian of Gaussian method. See skimage.feature.blob_log for more details

        Args:
            max_sigma (float, optional):
                The maximum standard deviation for Gaussian Kernel. Keep this high to detect larger blobs.
            num_sigma (int, optional):
                The number of intermediate values of standard deviations to consider between min_sigma and max_sigma.
            threshold (float, optional):
                The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored. Reduce this to detect blobs with less intensities.

        Returns
            Array: an array of blogs

        """
        blobs_log = blob_log(self.pattern, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, **blob_kargs)

        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

        self.blobs = blobs_log
        return blobs_log

    def plot_blobs(self, blob_color="red", ax=None, **fig_kargs):
        """
        plot the blobs location on the RHEED pattern
        
        Args:
            blob_color (str, optional): the color of the blob
            ax (matplotlib.pyplot.Axes): plot's figure
            
        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's figure
        """
        
        fig, ax = _create_figure(ax=ax, **fig_kargs)
        self.plot_pattern(ax=ax)
        for blob in self.blobs:
            x, y, r = blob
            show_circle(ax, (x,y), r, color=blob_color)
        ax.set_axis_off()
        
        return fig, ax

    def get_direct_beam(self, rmin=3):
        """
        Find the blob that contain direct beam information. Assuming the direct beam is the top one.

        Args:
            rmin : minimum radius for a spot to be selected
            
        Returns:
            Array: blob x and y
            int: blob id
        """
        def get_top_blob(blobs):
            top = np.inf
            tb = None
            tb_i = -1
            for i, (x, y, r) in enumerate(blobs):
                if x<top and r>=rmin:
                    tb = (x, y, r)
                    top = x
                    tb_i = i
            return tb, tb_i

        if hasattr(self, "blobs"):
            self.db, self.db_i = get_top_blob(self.blobs)
            return self.db, self.db_i
        else:
            raise Exception("Blobs is not stored in the object. Run .get_blobs() first")

    def get_candidate_specular_spot(self, rmin=3):
        """
        Find all the possible blob that contain specular spot information. Return [] if no specular spot
        spot is found. 

        Args:
            rmin (float): minimum radius for a spot to be selected
        
        Returns:
            List of blob
        """
        
        dbx, dby, dbr = self.blobs[self.db_i]
        i = -1
        candidates = []
        condidates_id = []
        for i, (x,y,r) in enumerate(self.blobs):
            if (y==dby and x==dbx and r==dbr): continue

            if (r>=rmin and y<=dby+dbr and y>=dby-dbr):
                ss = (x,y,r)
                candidates.append(ss)
                condidates_id.append(i)

        return candidates, condidates_id


    def get_specular_spot(self, rmin=3):
        """
        Find the blob that contain specular spot information. Return None, -1 if no specular spot
        spot is found. The basic idea is to find the first blob that is right below the direct
        beam blob. Need to call get_direct_beam method first.

        Args:
            rmin (float): minimum radius for a spot to be selected
        
        Returns:
            Array: blob x and y
            int: blob id
        """

        warn("This method would be removed. please use get_candidate_specular_spot", DeprecationWarning, stacklevel=2)        
        dbx, dby, dbr = self.blobs[self.db_i]
        ss = None
        top = np.inf
        i = -1
        for i, (x,y,r) in enumerate(self.blobs):
            if (y==dby and x==dbx and r==dbr): continue

            if (r>=rmin and y<=dby+dbr and y>=dby-dbr and top>x):
                ss = (x,y,r)
                top = x
        self.ss = ss
        return ss, i

    def laue_circle_analyse(self, rmin=3):
        """
            use db to guess where the specular spot and the 0-order laue circle is
        """

        if hasattr(self, "db") and self.db is not None:
            db_x, db_y, db_r = self.db

            candidates, candidates_id = self.get_candidate_specular_spot(rmin=rmin)            
            selected_id = candidates_id + [self.db_i]

            other_blobs = [ blob for i, blob in enumerate( self.blobs ) if i not in selected_id ]
            other_blobs_xy = np.stack(other_blobs, axis=0)[:, :2]

            best_dist = np.inf
            best_xy, best_r = None, None
            best_ss, best_ss_i = None, None

            for ss, ss_i in zip(candidates, candidates_id):
                ss_x, ss_y, ss_r = ss
                dbc = np.array([db_x, db_y])
                ssc = np.array([ss_x, ss_y])
                xy = np.stack( [dbc, ssc], axis=0 ).mean(axis=0)
                r = np.mean( (np.linalg.norm(xy-dbc), np.linalg.norm(xy-ssc)) )
                
                dist = np.mean( [ np.abs(r-np.linalg.norm(xy-o_xy)) for o_xy in other_blobs_xy] )

                if best_dist > dist:
                    best_xy = xy
                    best_r = r
                    best_ss = ss
                    best_ss_i = ss_i
                    best_dist = dist

            self.xy = best_xy
            self.ss = best_ss
            self.r = best_r
            
            # return xy location and radius
            return self.xy, self.r
        else:
            raise Exception("either direct beam or specular beam is not detected")

    def get_0laue(self):
        """
        Get 0-Laue Circle location and radius when the direct beam and specular spot blobs is located
        Need to run get_direct_beam and get_specular_spot methods first.
        
        Returns:
            Array: 0-zero order Laue circle center's x and y
            float: 0-zero order Laue circle radius
        """

        warn("This method would be removed. please use get_candidate_specular_spot", DeprecationWarning, stacklevel=2)        
        if hasattr(self, "db") and hasattr(self, "ss") and self.db is not None and self.ss is not None:
            dbx, dby, dbr = self.blobs[self.db_i]
            dbc = np.array([dbx, dby])
            ssx, ssy, ssr, = self.ss
            ssc = np.array([ssx, ssy])

            xy = np.stack( [dbc, ssc], axis=0 ).mean(axis=0)
            self.r = np.mean( (np.linalg.norm(xy-dbc), np.linalg.norm(xy-ssc)) )        
            self.xy = xy
            
            # return xy location and radius
            return self.xy, self.r
        else:
            raise Exception("either direct beam or specular beam is not detected")

    def plot_0laue(self, ax=None, plot_aux=False, **fig_kargs):
        """Plot the 0 order laue circle
        
        Args:
            ax (matplotlib.pyplot.Axes): figure's axes. Defaults to None
                
        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's figure
        """    
    
        if hasattr(self, "laue_xy") and hasattr(self, "laue_r"):
            fig, ax = self.plot_nlaue(self.laue_xy, [self.laue_r], ax=ax, **fig_kargs)
        if plot_aux:
            ax.vlines([self.laue_xy[1], self.laue_xy[1] - self.laue_r, self.laue_xy[1] + self.laue_r], 0, self.pattern.shape[0]-1)
            # plt.vlines(, 0, rd_.pattern.shape[0]-1)
            ax.scatter(self.laue_xy[1], self.laue_xy[0])
        
        return fig, ax
        
    def plot_nlaue(self, xy, rs, ax=None, **fig_kargs):
        """Plot multiple laue circles. 

        Args:
            xy (Array): x and y locations of the circle center 
            rs (Array): radius of different Laue circle
            ax (matplotlib.pyplot.Axes, optional): figure's axes. Defaults to None.
            
        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's figure
        """
        
        fig, ax = _create_figure(ax=ax, **fig_kargs)
        self.plot_pattern(ax=ax)
        for r in rs:
            show_circle(ax, xy, r )
        ax.set_axis_off()
        
        return fig, ax

    def plotly_pattern(self,  **fig_kargs):
        """Plot pattern with interactive plotly backend

        Returns:
            plotly.graph_objects.Figure: Plotly graph
        """

        fig = go.Figure(**fig_kargs)
        fig = fig.add_trace(go.Heatmap(z=self.pattern,  colorscale="Cividis"))
        fig.update_layout(height=600, width=800, yaxis=dict(autorange='reversed'), showlegend=True)

        fig.update_layout(
            legend=go.layout.Legend(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="white"
                ),
                # bgcolor="LightSteelBlue",
                bgcolor="Black",
                bordercolor="Black",
                borderwidth=2
            )
        )
        return fig

    def plot_pattern(self, ax=None, show_axes=False, cmap=None, **fig_kargs):
        """Plot pattern with matplotlib

        Args:
            ax (matplotlib.pyplot.Axes, optional): Figure's Axes. Defaults to None.
            show_axes (bool, optional): show the x y label and ticks. Defaults to False.
            cmap (str, optional): Matplotlib continueous colormap name. Defaults to None.

        Returns:
            matplotlib.pyplot.Figure : plot's figure
            matplotlib.pyplot.Axes : plot's axes
        """
        fig, ax = _create_figure(ax, **fig_kargs)
        ax.imshow(self.pattern, cmap=cmap if cmap is not None else self._CMAP)

        if show_axes:
            ax.set_xlabel("x (pixel)")
            ax.set_ylabel("y (pixel)")
        else:
            ax.set_axis_off()

        return fig, ax

    def get_fft(self, center=True):
        """Compute the fast fourier transform of the RHEED pattern

        Args:
            center (bool, optional): Use image center as origin. Defaults to True.

        Returns:
            complex ndarray: fourier transform of the pattern
            real ndaray: Magnitude or |c| of the fourier transform
        """
        
        img = self.pattern
        f = np.fft.fft2(img)
        if center: f = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(f)
        self.fft = f
        self.fft_center = center
        self.fft_mag = magnitude_spectrum
        return f, magnitude_spectrum
    
    def fft_reconstruct(self, window_x, window_y, inplace=True):
        """Reconstruct the RHEED pattern from the whole or partial fft.
        A window with a width of 2*window_x and height of 2*window_y could
        be specified to limit the fourier component that would be used to
        do the reconstruction (only the one within the window would be used)        

        Args:
            window_x (int): window x
            window_y (int): window y
            inplace (bool, optional): the operated result would overwrite the stored pattern if True.

        Returns:
            Rheed: return a new object or update the current object
        """
        fft = self.fft.copy()
        rows, cols = self.fft.shape
        crow,ccol = int(rows/2) , int(cols/2)
        fft[crow-window_y:crow+window_y+1, ccol-window_x:ccol+window_x+1] = 0
        if self.fft_center: fft = np.fft.ifftshift(fft)
        recon = np.fft.ifft2(fft)
        return self._update_pattern(np.abs(recon), inplace)

    def plot_fft(self, ax=None, **fig_kargs):
        """Plot the fast fourier transform of the pattern with matplotlib

        Args:
            ax (matplotlib.pyplot.Axes, optional): Figure's Axes. Defaults to None.

        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's axes
        """        
        
        fig, ax = _create_figure(ax, **fig_kargs)
        ax.imshow(self.fft_mag)
        ax.set_title('Magnitude Spectrum')
        ax.set_xlabel("Frequency X")
        ax.set_ylabel("Frequency y")
        ax.set_yticklabels(ax.get_yticks().astype(int) - int(self.fft_mag.shape[0] // 2))
        ax.set_xticklabels(ax.get_xticks().astype(int) - int(self.fft_mag.shape[1] // 2))
        return fig, ax


class RheedMask():
    """A Class that store a rheed object with a mask that label out the user
    interest region. These mask could be generated from human or ML models.
    It also provides tools for analyze and visualize not only the rheed pattern 
    within the masks but also features that derived from the mask.
    
    """
    def __init__(self, rd:Rheed, mask:np.ndarray):
        """Initializer of the RheedMask

        Args:
            rd (Rheed): a Rheed Object
            mask (np.ndarray): the bindary mask
        """
        self.rd = rd
        self.mask = mask


    def crop(self, sx, sy, ex, ey, inplace=False):        
        """crop the rheed object and the mask by the given corner points. 

        Args:
            sx (int): crop starting x position
            sy (int): crop starting y position
            ex (int): crop end x position
            ey (int): crop end y position
            inplace (bool, optional): the operated result would overwrite the stored pattern and mask if True. Defaults to True.

        Returns:
            RheedMask: either itself or a newly created RheedMask obj       
        """
        if inplace:
            self.rd = self.rd.crop(sx, sy, ex, ey, inplace=inplace)
            self.mask = crop(self.mask, sx, sy, ex, ey)
            return self
        else:
            rd = self.rd.crop(sx, sy, ex, ey, inplace=inplace)
            mask = crop(self.mask, sx, sy, ex, ey)
            return RheedMask(rd, mask)


    def get_regions(self, with_intensity=False):
        """Get all the connected regions in the binary mask.
        See skimage.measure.regionprops and skimage.measure.label for more details

        Args:
            with_intensity (bool, optional): keep region's intensity value in the out. Defaults to False.

        Returns:
            list: list of RegionProperties
        """
        # labeling:dict -> store
        # regions:dict -> store
        # regions primary key -> (name, id)
        self.label = label(self.mask)
        if with_intensity:
            self.regions = regionprops(self.label, self.rd.pattern)
        else:
            self.regions = regionprops(self.label)
        return self.regions


    def filter_regions(self, min_area, inplace=True):
        """remove regions that has very small areas.

        Args:
            min_area (float): minimum value of a region's area 
            inplace (bool, optional): inplace (bool, optional): the operated result would overwrite the stored regions if True. Defaults to True.

        Returns:
            list: list of RegionProperties
        """
        filtered = [ r for r in self.regions if r.area >= min_area ]
        if inplace: self.regions = filtered
        return [ r for r in self.regions if r.area >= min_area ]


    def get_region_collapse(self, region, direction="h"):
        """Compute integral spectrum from a given region

        Args:
            region (RegionProperties): a region with bounding box information
            direction (str, optional): direction of integration.
                'h' integrate over row direction and 'v' integrate
                over column direction. Defaults to "h".

        Returns:
            CollapseSpectrum: the 1d integration of pattern within the region
        """
        sx, sy, ex, ey = region.bbox
        if direction == "h":
            cs = CollapseSpectrum.from_rheed_horizontal(self.rd, sx, sy, ex, ey)
        elif direction == "v":
            cs = CollapseSpectrum.from_rheed_vertical(self.rd, sx, sy, ex, ey)
        else:
            raise ValueError(f"Unknown direction : {direction}")
        return cs


    def get_regions_collapse(self, direction="h"):
        """Compute integral spectrum from every region that stored in the objects

        Args:
            direction (str, optional): direction of integration.
                'h' integrate over row direction and 'v' integrate
                over column direction. Defaults to "h".

        Returns:
            List[CollapseSpectrum]: list of the 1d integration of pattern
        """        
        
        # collapse :list[1d spectrums] -> store
        self.collapses = []
        for region in self.regions:
            cs = self.get_region_collapse(region, direction)
            self.collapses.append(cs)
        return self.collapses


    def clean_collapse(self, smooth:bool=True, rm_bg:bool=True, scale:bool=True):
        """A warp method that perform normalization, gaussian smoothing, and 
        remove background to all the extracted integrated region spectrums.

        Args:
            smooth (bool, optional): perform gaussian smoothing. Defaults to True.
            rm_bg (bool, optional): remove background. Defaults to True.
            scale (bool, optional): use mean and std to scale the spectrum. Defaults to True.

        Returns:
            RheedMask: return the object itself
        """
        for cs in self.collapses:
            if rm_bg:
                try:
                    cs.remove_background()
                except np.linalg.LinAlgError as e:
                    warnings.warn(f"Encounter LinAlgError when doing background removal! {e}")
            if scale: cs.normalize()
            if smooth: cs.smooth()
        return self


    def fit_collapse_peaks(self, height:float, threshold:float, prominence:float):
        """Use the peak finding algorihm to get every peaks in every integrated region spectrums. 
        See scipy.signal.find_peaks for more details.

        Args:
            height (float): Required height of peaks. Either a number, None, an array matching 
                x or a 2-element sequence of the former. The first element is always interpreted 
                as the minimal and the second, if supplied, as the maximal required height.
            threshold (float): Required threshold of peaks, the vertical distance to its neighboring samples. 
                Either a number, None, an array matching x or a 2-element sequence of the former. The first 
                element is always interpreted as the minimal and the second, if supplied, as the maximal 
                required threshold.
            prominence (float): Required prominence of peaks. Either a number, None, an array matching x 
                or a 2-element sequence of the former. The first element is always interpreted as the 
                minimal and the second, if supplied, as the maximal required prominence.
        Returns:
            list: a list of all spectrums peaks's index w.r.t spectrums
            list: a list of all spectrums peaks position in image x coordinate
            list: a list of all spectrums peaks properties
        """
        # peaks dict : list->
        self.collapses_peaks = []
        self.collapses_peaks_ws = []
        self.collapses_peaks_info = []
        for cs in self.collapses:
            if cs is not None:
                peaks, peaks_info = cs.find_spectrum_peaks(height=height, threshold=threshold, prominence=prominence)
                self.collapses_peaks.append( peaks )
                self.collapses_peaks_ws.append( cs.ws[peaks] )
                self.collapses_peaks_info.append( peaks_info )
            else:
                self.collapses_peaks.append([])
                self.collapses_peaks_ws.append([])
                self.collapses_peaks_info.append([])

        self.collapses_peaks_regions = [ [i]*len(ps) for i, ps in enumerate(self.collapses_peaks) ]
        return self.collapses_peaks, self.collapses_peaks_ws, self.collapses_peaks_info


    def get_top_region(self):
        """Get the region that is cloest to the top of the RHEED pattern.

        Returns:
            RegionProperties: the top region
            int: region id
        """
        topx = self.rd.pattern.shape[0]
        topr = None
        topr_i  = -1
        for i, r in enumerate(self.regions):
            if r.centroid[0] < topx:
                topr = r
                topr_i = i
                topx = r.centroid[0]
        return topr, topr_i


    def _get_region_centroid(self, region):
        centroid = region.weighted_centroid if hasattr(region, "weighted_centroid") else region.centroid
        xy = centroid
        return xy


    def get_close_region(self, x, y):
        """Find the one among the whole extracted regions that has the smallest distance between
        the input coordinates and its centroid.

        Args:
            x (float): x coordinate
            y (float): y coordinate

        Returns:
            RegionProperties: the region
            int: the region's id
        """
        centroids = np.stack([ self._get_region_centroid(region) for region in self.regions ], axis=0)
        dists = np.linalg.norm(centroids - np.array([x,y]), axis=1)
        i = np.argmin(dists)
        return self.regions[i], i


    def get_region_within(self, x:int, y:int):
        """Get the first region that its mask contain (x, y)

        Args:
            x (int): x coordinate 
            y (int): y coordinate

        Returns:
            RegionProperties: the top region
            int: region id
        """
        for i, r in enumerate(self.regions):
            sx, sy, ex, ey = r.bbox
            within_box_x = x >= sx and x <= ex
            within_box_y = y >= sy and y <= ey
            within_box = within_box_x and within_box_y
            if within_box:
                within_mask = r.image[int(x - sx), int(y - sy)]
                if within_mask: return r, i
        else:
            return None, None


    def get_direct_beam(self, method="top", tracker=None, track=None):
        """Find the direct beam of the rheed pattern using heuristic or a
        iou tracker. When method is set to 'top', the method would select the
        region that is cloest to the top as the direct beam. 'top+tracker' is 
        similar but it also register the discovered region to the tracker. 
        'tracker', on the other hand, would completely rely on tracker to find
        the region in the current obj that follow the historic movement of the 
        direct beam. The tracker we demonstrated here is an simply IOU tracker
        but it could be extended to other fancy methods.

        Args:
            method (str, optional): _description_. Defaults to "top".
            tracker (_type_, optional): _description_. Defaults to None.
            track (_type_, optional): _description_. Defaults to None.
        """
        def _get_centroid(r_i):
            r = self.regions[r_i]
            return self._get_region_centroid(r)

        def _get_top():
            r, r_i = self.get_top_region()
            xy = self._get_region_centroid(r)
            return xy, r_i

        if method=="top":
            xy, r_i = _get_top()
            return xy, r_i, None
        elif method=="top+tracker":
            xy, r_i = _get_top()
            track = tracker.region2track[r_i]
            return xy, r_i, track
        elif method=="tracker":
            r_i = tracker.track2region[track]
            xy = self._get_region_centroid(self.regions[r_i])
            return xy, r_i, track
        else:
            raise Exception(f"method of {method} is unknown, allow top, top+tracker, and tracker")

        # csh = self.get_region_collapse(topr, "h")
        # csv = self.get_region_collapse(topr, "v")

        # xy = []
        # for d, cs in {"horizontal":csh, "vertical":csv}.items():
        #     cs.remove_background()
        #     cs.smooth(sigma=sigma)
        #     peaks, _ = cs.fit_spectrum_peaks(height=height, threshold=threshold, prominence=prominence, **peak_args)
        #     peaks_w = cs.ws[peaks]
        #     assert len(peaks_w) == 1, f"found {len(peaks)} peaks in {d} direction"
        #     xy.append(peaks_w[0])


    def _flatten_peaks(self):
        self.collapses_peaks_ws_flatten = np.array(list(itertools.chain.from_iterable(self.collapses_peaks_ws)))
        self.collapses_peaks_flatten = np.array(list(itertools.chain.from_iterable(self.collapses_peaks)))
        self.collapses_peaks_flatten_regions = np.array(list(itertools.chain.from_iterable(self.collapses_peaks_regions)))

        sortidxs = np.argsort(self.collapses_peaks_ws_flatten)

        self.collapses_peaks_ws_flatten = self.collapses_peaks_ws_flatten[sortidxs]
        self.collapses_peaks_flatten =  self.collapses_peaks_flatten[sortidxs]
        self.collapses_peaks_flatten_pids = sortidxs
        self.collapses_peaks_flatten_regions = self.collapses_peaks_flatten_regions[sortidxs]
        return self.collapses_peaks_ws_flatten


    def analyze_peaks_distance_cent(self, tolerant=0.01, abs_tolerant=10, allow_discontinue=1):
        """Apply periodic analysis to all intergrated region spectrum. 
        See "spectrum.spectrum.analyze_peaks_distance_cent" for more details
        
        The actual tolerant which define the maximum allowed deviation between computed next peak in 
        the family and the actual peaks locaitons
            act_tolerant = min(tolerant * dist, abs_tolerant)

        Args:
            tolerant (float, optional): relative tolerant. Defaults to 0.01.
            abs_tolerant (int, optional): absolute tolerant. Defaults to 10.
            allow_discontinue (int, optional): _description_. Defaults to 1.
            allow_discontinue (int, optional): allowed discontinity when searching peaks. Defaults to 1.
        Returns:
            list: a list of spectrum.spectrum.PeakAnalysisDetail
        """
        allpeaks = self._flatten_peaks()
        if len(allpeaks) == 0:
            self.collapses_peaks_pgs = []
        
        else:
            ci = get_center_peak_idx(allpeaks, self.rd.pattern.shape[1]//2, abs_tolerant)
            cis = get_center_peak_idxs(allpeaks, self.rd.pattern.shape[1]//2, abs_tolerant)
            ciw = allpeaks[ci]

            inter_dist = get_peaks_distance(
                np.arange(len(allpeaks)),
                np.array(allpeaks),
                full=True,
                polar=True
            )

            self.collapses_peaks_flatten_nbr_dist = inter_dist[ci, :]
            self.collapses_peaks_flatten_ci = ci
            self.collapses_peaks_flatten_cis = cis
            self.collapses_peaks_flatten_ciw = ciw

            analyzer = PeriodicityAnalyzer(
                tolerant=tolerant, 
                abs_tolerant=abs_tolerant, 
                allow_discontinue=allow_discontinue
            )

            self.collapses_peaks_pgs = analyzer.analyze(
                arr= self.collapses_peaks_flatten,
                center=self.rd.pattern.shape[1]//0,
                grid_min=0,
                grid_max= self.rd.pattern.shape[1],
            )
            # self.collapses_peaks_pgs = analyze_peaks_distance_cent(
            #     self.collapses_peaks_flatten,
            #     self.collapses_peaks_flatten_nbr_dist,
            #     self.collapses_peaks_flatten_ciw,
            #     self.collapses_peaks_flatten_ci,
            #     grid_min_w= 0,
            #     grid_max_w= self.rd.pattern.shape[1],
            #     tolerant= tolerant,
            #     abs_tolerant= abs_tolerant,
            #     allow_discontinue= allow_discontinue
            # )

        return self.collapses_peaks_pgs


    def plot_pattern_masks(self, ax=None):
        """Plot the pattern with the mask overlay

        Args:
            ax (matplotlib.pyplot.Axes, optional): plot's axes. Defaults to None.

        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's axes
        """
        
        # plot pattern and mask
        fig, ax = _create_figure(ax=ax)

        self.rd.plot_pattern(ax)
        ax.imshow(self.mask, alpha=0.7)
        return fig, ax


    def plot_region(self, region_id:int, zoom:bool=True, ax=None, **fig_kargs):
        """Plot one region in the form of bounding box. An inset plot of the pattern within the region
        is also inserted if zoom is set to true.

        Args:
            region_id (int): the region id you want to plot
            zoom (bool, optional): Plot the pattern bounded by the region if set to True. Defaults to True.
            ax (matplotlib.pyplot.Axes, optional): The plot's axes. Defaults to None.

        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's axes
        """
        fig, ax = _create_figure(ax=ax, **fig_kargs)
        self.rd.plot_pattern(ax=ax)

        region = self.regions[region_id]

        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)

        # ax.scatter(region.centroid[1], region.centroid[0], c="white", s=1)
        ax.add_patch(rect)
                
        if zoom:
            axins = ax.inset_axes([0.9, 0, 0.1, 1 ])
            axins.set_xticklabels('')
            axins.set_yticklabels('')

            self.rd.crop(minr-1, minc-1, maxr+1, maxc+1, inplace=False).plot_pattern(axins, cmap=self.rd._CMAP)

        return fig, ax


    def plot_regions(self, ax=None, min_area:float=0.0, centroid=False,**fig_kargs):
        """Plot all extracted regions from the mask.

        Args:
            ax (matplotlib.pyplot.Axes, optional): _description_. Defaults to None.
            min_area (float, optional): Regions with area higher than this value would be shown. Defaults to 0.
            centroid (bool, optional): Plot the centroid location if True. Defaults to False.

        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's axes
        """
        
        fig, ax = _create_figure(ax=ax, **fig_kargs)
        # image_label_overlay = label2rgb(self.label, image=self.rd.pattern, bg_label=0)

        # ax.imshow(image_label_overlay)
        self.rd.plot_pattern(ax=ax)

        for rid, region in enumerate(self.regions):
            # take regions with large enough areas
            if region.area >= min_area:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)

                if centroid: ax.scatter(region.centroid[1], region.centroid[0], c="white", s=1)
                ax.add_patch(rect)

                ax.text(x= min(minc - self.rd.pattern.shape[1]*0.01, 0), y=minr, s=f"{rid}", color="white", fontsize="xx-small", va='top', ha='right')
        
        return fig, ax


    def plot_peak_dist(self, ax=None, dist_text_color="white", show_text=True):
        """Plot the extracted peak family periodicity. Call this method when finish the 
        periodicity analysis first.

        Args:
            ax (matplotlib.pyplot.Axes, optional): plot's axes. Defaults to None.
            dist_text_color (str, optional): text color of the periodicity. Defaults to "white".
            show_text (bool, optional): show the exact periodicity on the figure if set to True. Defaults to True.

        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's axes
        """
        fig, ax = _create_figure(ax)
        
        self.rd.plot_pattern(ax=ax)
        
        for i, res in enumerate(self.collapses_peaks_pgs):            
            ax.vlines(
                x= self.collapses_peaks_ws_flatten[res.peaks_family.astype(int)],
                ymin=0.05*self.rd.pattern.shape[0], ymax=0.95*self.rd.pattern.shape[0],
                alpha=0.5, 
                color=plt.cm.Set1.colors[i]
            )
            if show_text:
                for p in self.collapses_peaks_ws_flatten[res.peaks_family.astype(int)]:
                    ax.text(x=p, y=20*(i+1), s=f"{res.avg_dist:.1f}", color=dist_text_color)
        
        return fig, ax


    def get_group_intensity(self):
        """Get the total intensity and the relative intensity of each periodicity. If the region corresponding 
        to only one peak family, the sum of the intensity within a region is the group intensity. If there are more than one 
        peak families, a gaussian mixture model would used to fit the spectrum and the width of the gaussian is used to
        estimate the corresponding region of its corresponding family. The group intensity is the sum of that region instead.

        Relative intensity = Group Intensity / Sum(Group Intensity)

        Returns:
            Array: group intensity
            Array: relative group intensity
        """

        # for the case where no cluster labels is presented for this pattern
        if self.cluster_labels_unique is None or len(self.cluster_labels_unique) == 0:
            return [], []

        gauss_fit = {}
        group_intensity = np.zeros(len(self.cluster_labels_unique))

        if len(self.collapses_peaks_pgs) > 0:
            cidxs =self.collapses_peaks_flatten_cis

            for i, ul in enumerate(self.cluster_labels_unique):
                group_mask = np.zeros(self.rd.pattern.shape, dtype=bool)
                selected = [self.collapses_peaks_pgs[i] for i, cl in enumerate(self.cluster_labels) if cl == ul]
                
                for j, ana in enumerate(selected):
                    for p in ana.peaks_family:
                        if p in cidxs: continue 
                        pid = self.collapses_peaks_flatten_pids[p]
                        rid = self.collapses_peaks_flatten_regions[p]
                        region = self.regions[rid]
                        minr, minc , maxr, maxc = region.bbox

                        if len(self.collapses_peaks[rid]) == 1:
                            #regions only has one peak
                            group_mask[minr:maxr, minc:maxc] = group_mask[minr:maxr, minc:maxc] | region.image
                        else:
                            #regions has 2 or more peaks
                            p_neighbor = self.collapses_peaks_flatten_pids[self.collapses_peaks_flatten_regions == rid]
                            pid = pid - np.min(p_neighbor)

                            if rid not in gauss_fit:
                                cs = self.collapses[rid]
                                ps = self.collapses_peaks[rid]
                                xs = cs.ws[ps]
                                hs = cs.spec[ps]
                                width = (max(xs) - min(xs)) / (2*len(ps))

                                guess = []
                                for k in range(len(xs)):
                                    guess+=[hs[k], xs[k], width]
                                guess += [0]
                                try:
                                    popt, pcov = curve_fit(multi_gaussian, cs.ws, cs.spec, guess)
                                    gauss_fit[rid] = (popt, pcov)
                                except Exception as e:
                                    # raise e
                                    # print(self)
                                    # print(guess)
                                    # print(rid)
                                    # print(ps)
                                    warnings.warn(f"Gaussian Fail at {rid}, split regions equally")
                                    popt, pcov = (guess, None)
                                    gauss_fit[rid] = popt, pcov
                            else:
                                popt, pcov = gauss_fit[rid]

                            # the absolute here is to filp the variance all to positive, some time it would be negative!
                            old_minc = minc
                            minc = max(minc, int(popt[pid*3+1]-abs(popt[pid*3+2])))
                            maxc = min(maxc, int(popt[pid*3+1]+abs(popt[pid*3+2])))
                            
                            try:
                                group_mask[minr:maxr, minc:maxc] = group_mask[minr:maxr, minc:maxc] | region.image[:, minc-old_minc:maxc-old_minc]
                            except:
                                warnings.warn(f"Fail to register to the mask {rid}")
                            
                group_intensity[i] = np.sum( group_mask * self.rd.pattern )
                group_percent = group_intensity / np.sum(group_intensity)
        else:
            group_percent = np.zeros(len(self.cluster_labels_unique))

        self.group_intensity = group_intensity
        self.group_percent = group_percent

        return group_intensity, group_percent
    

class InstanceSegmentationItem:
    def __init__(self, bbox, mask, score, label, label_name) -> None:
        """
            the bounding box is in xyxy format following the same definition to skimage regions

        """
        self.bbox = bbox
        self.mask = mask 
        self.score = score
        self.label = label
        self.label_name = label_name
        self.region = None # for compatibility with the existing workflow

    def _update(self, inplace, **kargs):
        if inplace:
            this = self
        else:
            this = copy.deepcopy(self)

        for k, v in kargs.items():
            # TODO: maybe add a check of the attribute before update
            # make this functionality a abstract thing?
            setattr(this, k, v)

        return this


    def crop(self, sx, sy, ex, ey, inplace=False):
        bbox = self.bbox
        bbox = (bbox[0] - sx, bbox[1] - sy, bbox[2] - sx, bbox[3] - sy)
        bbox[2] = min(bbox[2], ex)
        bbox[3] = min(bbox[3], ey)
        
        mask = crop(self.mask, sx=sx, sy=sy, ex=ex, ey=ey)

        self._update(inplace, bbox=bbox, mask=mask)


class RheedInstanceSegmentation:
    """A Class that store a rheed object and the instance segmentation results.
    
    """
    def __init__(self, rd:Rheed, inst_segs:List[InstanceSegmentationItem], auto_compute_regions=True):
        """Initializer of the RheedMask

        Args:
            rd (Rheed): a Rheed Object
            inst_segs (List[InstanceSegmentationItem]): a list of segmentation items
        """
        self.rd = rd
        self.inst_segs = inst_segs

        if auto_compute_regions:
            self.compute_regions(with_intensity=True)

    def _update(self, inplace, **kargs):
        if inplace:
            this = self
        else:
            this = copy.deepcopy(self)

        for k, v in kargs.items():
            # TODO: maybe add a check of the attribute before update
            # make this functionality a abstract thing?
            setattr(this, k, v)

        return this
    
    @classmethod
    def from_mmdet(cls, rd, det_result, detector_model=None, auto_compute_regions=True):
        det_result = det_result.detach().cpu().pred_instances
        bboxes = det_result.bboxes.numpy()
        labels = det_result.labels.numpy()
        scores = det_result.scores.numpy()
        masks = det_result.masks.numpy()
        
        inst_segs = []
        for i in range(len(labels)):
            label = int(labels[i])
            try:
                label_name = detector_model.cfg['metainfo']['classes'][label]
            except Exception as e:
                print(e)
                print("Fail to extrace label name from the detector")
            item = InstanceSegmentationItem(
                bbox = bboxes[i],
                mask = masks[i],
                score = scores[i],
                label = label,
                label_name = label_name
            )
            inst_segs.append(item)
        
        return cls(rd=rd, inst_segs=inst_segs, auto_compute_regions=auto_compute_regions)

    def from_detectron2(cls, rd, detections_result):
        # TODO: 
        # return cls()
        pass


    def crop(self, sx, sy, ex, ey, inplace=False):        
        """crop the rheed object and the mask by the given corner points. 

        Args:
            sx (int): crop starting x position
            sy (int): crop starting y position
            ex (int): crop end x position
            ey (int): crop end y position
            inplace (bool, optional): the operated result would overwrite the stored pattern and mask if True. Defaults to True.

        Returns:
            RheedMask: either itself or a newly created RheedMask obj       
        """
        rd = self.rd.crop(sx, sy, ex, ey, inplace=inplace)
        for item in self.inst_segs:
            item.crop( sx, sy, ex, ey, inplace=inplace)
        inst_segs = self.inst_segs if inplace else list(inst_segs)
        
        return self._update(rd=rd, inst_segs=self.inst_segs)

    @property
    def regions(self):
        return [ item.region for item in self.inst_segs ]
    
    @property
    def regions_label(self):
        return [ item.label for item in self.inst_segs ]

    def compute_regions(self, with_intensity=False):
        """Get all the connected regions in the binary mask.
        See skimage.measure.regionprops and skimage.measure.label for more details

        Args:
            with_intensity (bool, optional): keep region's intensity value in the out. Defaults to False.

        Returns:
            list: list of RegionProperties
        """
        # labeling:dict -> store
        # regions:dict -> store
        # regions primary key -> (name, id)
        intensity_image = self.rd.pattern if with_intensity else None
        for i, item in enumerate(self.inst_segs):
            slice = ndimage.find_objects(item.mask)[0]

            if with_intensity:
                region = RegionProperties(slice, 1, item.mask, intensity_image, cache_active=True)
            else:
                region = RegionProperties(slice, 1, item.mask, intensity_image, cache_active=True)
            item.region = region

        return self.regions


    def filter_detections(self, min_area, inplace=True):
        """remove detection that has very small areas.

        Args:
            min_area (float): minimum value of a region's area 
            inplace (bool, optional): inplace (bool, optional): the operated result would overwrite the stored regions if True. Defaults to True.

        """
        filtered = [ item for item in self.inst_segs if item.region.area >= min_area ]
        return self._update(inplace, inst_segs=filtered)


    def get_region_collapse(self, region, direction="h"):
        """Compute integral spectrum from a given region

        Args:
            region (RegionProperties): a region with bounding box information
            direction (str, optional): direction of integration.
                'h' integrate over row direction and 'v' integrate
                over column direction. Defaults to "h".

        Returns:
            CollapseSpectrum: the 1d integration of pattern within the region
        """
        sx, sy, ex, ey = region.bbox
        if direction == "h":
            cs = CollapseSpectrum.from_rheed_horizontal(self.rd, sx, sy, ex, ey)
        elif direction == "v":
            cs = CollapseSpectrum.from_rheed_vertical(self.rd, sx, sy, ex, ey)
        else:
            raise ValueError(f"Unknown direction : {direction}")
        return cs


    def get_regions_collapse(self, direction="h"):
        """Compute integral spectrum from every region that stored in the objects

        Args:
            direction (str, optional): direction of integration.
                'h' integrate over row direction and 'v' integrate
                over column direction. Defaults to "h".

        Returns:
            List[CollapseSpectrum]: list of the 1d integration of pattern
        """        
        
        # collapse :list[1d spectrums] -> store
        self.collapses = []
        for region in self.regions:
            cs = self.get_region_collapse(region, direction)
            self.collapses.append(cs)
        return self.collapses


    def clean_collapse(self, smooth:bool=True, rm_bg:bool=True, scale:bool=True):
        """A warp method that perform normalization, gaussian smoothing, and 
        remove background to all the extracted integrated region spectrums.

        Args:
            smooth (bool, optional): perform gaussian smoothing. Defaults to True.
            rm_bg (bool, optional): remove background. Defaults to True.
            scale (bool, optional): use mean and std to scale the spectrum. Defaults to True.

        Returns:
            RheedMask: return the object itself
        """
        for cs in self.collapses:
            if rm_bg:
                try:
                    cs.remove_background()
                except np.linalg.LinAlgError as e:
                    warnings.warn(f"Encounter LinAlgError when doing background removal! {e}")
            if scale: cs.normalize()
            if smooth: cs.smooth()
        return self


    def fit_collapse_peaks(self, height:float, threshold:float, prominence:float):
        """Use the peak finding algorihm to get every peaks in every integrated region spectrums. 
        See scipy.signal.find_peaks for more details.

        Args:
            height (float): Required height of peaks. Either a number, None, an array matching 
                x or a 2-element sequence of the former. The first element is always interpreted 
                as the minimal and the second, if supplied, as the maximal required height.
            threshold (float): Required threshold of peaks, the vertical distance to its neighboring samples. 
                Either a number, None, an array matching x or a 2-element sequence of the former. The first 
                element is always interpreted as the minimal and the second, if supplied, as the maximal 
                required threshold.
            prominence (float): Required prominence of peaks. Either a number, None, an array matching x 
                or a 2-element sequence of the former. The first element is always interpreted as the 
                minimal and the second, if supplied, as the maximal required prominence.
        Returns:
            list: a list of all spectrums peaks's index w.r.t spectrums
            list: a list of all spectrums peaks position in image x coordinate
            list: a list of all spectrums peaks properties
        """
        # peaks dict : list->
        self.collapses_peaks = []
        self.collapses_peaks_ws = []
        self.collapses_peaks_info = []
        for cs in self.collapses:
            if cs is not None:
                peaks, peaks_info = cs.find_spectrum_peaks(height=height, threshold=threshold, prominence=prominence)
                self.collapses_peaks.append( peaks )
                self.collapses_peaks_ws.append( cs.ws[peaks] )
                self.collapses_peaks_info.append( peaks_info )
            else:
                self.collapses_peaks.append([])
                self.collapses_peaks_ws.append([])
                self.collapses_peaks_info.append([])

        self.collapses_peaks_regions = [ [i]*len(ps) for i, ps in enumerate(self.collapses_peaks) ]
        return self.collapses_peaks, self.collapses_peaks_ws, self.collapses_peaks_info


    def get_top_region(self, region_label=None):
        """Get the region that is cloest to the top of the RHEED pattern.

        Returns:
            RegionProperties: the top region
            int: region id
        """
        topx = self.rd.pattern.shape[0]
        topr = None
        topr_i  = -1
        labels = self.regions_label
        for i, r in enumerate(self.regions):
            label_condition = region_label == labels[i] if region_label != -1 else True
            if r.centroid[0] < topx and label_condition:
                topr = r
                topr_i = i
                topx = r.centroid[0]
        return topr, topr_i


    def _get_region_centroid(self, region):
        centroid = region.weighted_centroid if hasattr(region, "weighted_centroid") else region.centroid
        xy = centroid
        return xy


    def get_close_region(self, x, y):
        """Find the one among the whole extracted regions that has the smallest distance between
        the input coordinates and its centroid.

        Args:
            x (float): x coordinate
            y (float): y coordinate

        Returns:
            RegionProperties: the region
            int: the region's id
        """
        centroids = np.stack([ self._get_region_centroid(region) for region in self.regions ], axis=0)
        dists = np.linalg.norm(centroids - np.array([x,y]), axis=1)
        i = np.argmin(dists)
        return self.regions[i], i


    def get_region_within(self, x:int, y:int):
        """Get the first region that its mask contain (x, y)

        Args:
            x (int): x coordinate 
            y (int): y coordinate

        Returns:
            RegionProperties: the top region
            int: region id
        """
        for i, r in enumerate(self.regions):
            sx, sy, ex, ey = r.bbox
            within_box_x = x >= sx and x <= ex
            within_box_y = y >= sy and y <= ey
            within_box = within_box_x and within_box_y
            if within_box:
                within_mask = r.image[int(x - sx), int(y - sy)]
                if within_mask: return r, i
        else:
            return None, None


    def get_direct_beam(self, method="top", tracker=None, track=None, direct_beam_label=3):
        """Find the direct beam of the rheed pattern using heuristic or a
        iou tracker. When method is set to 'top', the method would select the
        region that is cloest to the top as the direct beam. 'top+tracker' is 
        similar but it also register the discovered region to the tracker. 
        'tracker', on the other hand, would completely rely on tracker to find
        the region in the current obj that follow the historic movement of the 
        direct beam. The tracker we demonstrated here is an simply IOU tracker
        but it could be extended to other fancy methods.

        Args:
            method (str, optional): _description_. Defaults to "top".
            tracker (_type_, optional): _description_. Defaults to None.
            track (_type_, optional): _description_. Defaults to None.
        """
        def _get_centroid(r_i):
            r = self.regions[r_i]
            return self._get_region_centroid(r)

        def _get_top():
            r, r_i = self.get_top_region(region_label=direct_beam_label)
            xy = self._get_region_centroid(r)
            return xy, r_i

        if method=="top":
            xy, r_i = _get_top()
        elif method=="top+tracker":
            xy, r_i = _get_top()
            track = tracker.det2track[r_i]
        elif method=="tracker":
            r_i = tracker.track2det[track]
            xy = self._get_region_centroid(self.regions[r_i])
        else:
            raise Exception(f"method of {method} is unknown, allow top, top+trackes, and tracker")

        self.db_i = r_i
        self.db = xy

        return xy, r_i, track


    def get_candidate_specular_spot(self, area_min=3):
        """
        Find all the possible blob that contain specular spot information. Return [] if no specular spot
        spot is found. 

        Args:
            rmin (float): minimum radius for a spot to be selected
        
        Returns:
            List of blob
        """
        
        db_inst = self.inst_segs[self.db_i]
        dby_s, dbx_s, dby_e, dbx_e = db_inst.bbox # the coordination is flipped
        # print(db_inst.bbox)

        i = -1
        candidates = []
        condidates_id = []
        for i, (inst, region) in enumerate(zip(self.inst_segs, self.regions)):
            if inst.label_name == "spot":
                x, y = self._get_region_centroid(region)
                area = inst.mask.sum()
                # print(x, y)

                if (area>=area_min and y<=dby_e and y>=dby_s):
                    ss = (x,y)
                    candidates.append(ss)
                    condidates_id.append(i)

        return candidates, condidates_id


    def laue_circle_analyse(self, area_min=3):
        """
            use db to guess where the specular spot and the 0-order laue circle is
        """

        if hasattr(self, "db") and self.db is not None:
            db_x, db_y = self.db

            candidates, candidates_id = self.get_candidate_specular_spot(area_min=area_min)            
            selected_id = candidates_id + [self.db_i]

            other_xy =  np.stack([ self._get_region_centroid(r) for i, r in enumerate( self.regions ) if i not in selected_id ], axis=0)

            best_dist = np.inf
            best_xy, best_r = None, None
            best_ss, best_ss_i = None, None

            for ss, ss_i in zip(candidates, candidates_id):
                ss_x, ss_y = ss
                dbc = np.array([db_x, db_y])
                ssc = np.array([ss_x, ss_y])
                xy = np.stack( [dbc, ssc], axis=0 ).mean(axis=0)
                r = np.mean( (np.linalg.norm(xy-dbc), np.linalg.norm(xy-ssc)) )
                
                dist = np.mean( [ np.abs(r-np.linalg.norm(xy-o_xy)) for o_xy in other_xy] )

                if best_dist > dist:
                    best_xy = xy
                    best_r = r
                    best_ss = ss
                    best_ss_i = ss_i
                    best_dist = dist

            self.laue_xy = best_xy
            self.ss = best_ss
            self.laue_r = best_r
            
            # return xy location and radius
            return self.laue_xy, self.laue_r
        else:
            raise Exception("either direct beam or specular beam is not detected")


    def plot_0laue(self, ax=None, plot_aux=False, **fig_kargs):
        """Plot the 0 order laue circle
        
        Args:
            ax (matplotlib.pyplot.Axes): figure's axes. Defaults to None
                
        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's figure
        """    
    
        if hasattr(self, "laue_xy") and hasattr(self, "laue_r"):
            fig, ax = self.rd.plot_nlaue(self.laue_xy, [self.laue_r], ax=ax, **fig_kargs)
        if plot_aux:
            ax.vlines([self.laue_xy[1], self.laue_xy[1] - self.laue_r, self.laue_xy[1] + self.laue_r], 0, self.rd.pattern.shape[0]-1)
            # plt.vlines(, 0, rd_.pattern.shape[0]-1)
            ax.scatter(self.laue_xy[1], self.laue_xy[0])

        return fig, ax


    def _flatten_peaks(self):
        self.collapses_peaks_ws_flatten = np.array(list(itertools.chain.from_iterable(self.collapses_peaks_ws)))
        self.collapses_peaks_flatten = np.array(list(itertools.chain.from_iterable(self.collapses_peaks)))
        self.collapses_peaks_flatten_regions = np.array(list(itertools.chain.from_iterable(self.collapses_peaks_regions)))

        sortidxs = np.argsort(self.collapses_peaks_ws_flatten)

        self.collapses_peaks_ws_flatten = self.collapses_peaks_ws_flatten[sortidxs]
        self.collapses_peaks_flatten =  self.collapses_peaks_flatten[sortidxs]
        self.collapses_peaks_flatten_pids = sortidxs
        self.collapses_peaks_flatten_regions = self.collapses_peaks_flatten_regions[sortidxs]
        return self.collapses_peaks_ws_flatten


    # def analyze_peaks_distance_cent(self, tolerant=0.01, abs_tolerant=10, allow_discontinue=1, center=None, method="single", tracker=None):
    def analyze_peaks_periodicity(self, center=None, method="single", analyzer=None, tracker=None, track_frame_num=None):

        """
        Apply periodic analysis to all intergrated region spectrum. 
        See "spectrum.spectrum.analyze_peaks_distance_cent" for more details
        
        The actual tolerant which define the maximum allowed deviation between computed next peak in 
        the family and the actual peaks locaitons
            act_tolerant = min(tolerant * dist, abs_tolerant)

        Args:
            center (float, optional): rheed pattern center location
            method (str, optional): method used to perform periodicity analysis. could be "single or track"
            analyzer (PeriodicityAnalyzer, optional): the analyzer used to peform periodicty analysis. not required if tracker is provide            
            tracker (PeriodicityTracker, optional): the tracker used to link previous periodicty to current peaks
            tracker_frame_num (int, optional): the frame number of the rheed. required by the tracker
        Returns:
            list: a list of spectrum.spectrum.PeakAnalysisDetail
        """
        method = method.lower()
        allpeaks = self._flatten_peaks()
        if len(allpeaks) == 0:
            self.collapses_peaks_pgs = []
        
        else:
            if center is None:
                center = self.rd.pattern.shape[1]//2

            if method == "single":
                self.collapses_peaks_pgs = analyzer.analyze(
                    arr=allpeaks,
                    center=center,
                    grid_min=0,
                    grid_max=self.rd.pattern.shape[1],
                )
            elif method == "track":
                self.collapses_peaks_pgs = tracker.update(
                    arr=allpeaks, 
                    center=center,
                    grid_min=0,
                    grid_max= self.rd.pattern.shape[1],
                    frame_num=track_frame_num
                )
            else:
                raise ValueError(f"Unknown Method Name: {method}")
            
        return self.collapses_peaks_pgs
        #     ci = get_center_peak_idx(allpeaks, center, abs_tolerant)
        #     cis = get_center_peak_idxs(allpeaks, center, abs_tolerant)
        #     ciw = allpeaks[ci]

        #     inter_dist = get_peaks_distance(
        #         np.arange(len(allpeaks)),
        #         np.array(allpeaks),
        #         full=True,
        #         polar=True
        #     )

        #     self.collapses_peaks_flatten_nbr_dist = inter_dist[ci, :]
        #     self.collapses_peaks_flatten_ci = ci
        #     self.collapses_peaks_flatten_cis = cis
        #     self.collapses_peaks_flatten_ciw = ciw

        #     self.collapses_peaks_pgs = analyze_peaks_distance_cent(
        #         self.collapses_peaks_flatten,
        #         self.collapses_peaks_flatten_nbr_dist,
        #         self.collapses_peaks_flatten_ciw,
        #         self.collapses_peaks_flatten_ci,
        #         grid_min_w= 0,
        #         grid_max_w= self.rd.pattern.shape[1],
        #         tolerant= tolerant,
        #         abs_tolerant= abs_tolerant,
        #         allow_discontinue= allow_discontinue
        #     )

        # return self.collapses_peaks_pgs


    def plot_pattern_segs(self, ax=None, dpi=150, color_by="order"):
        """Plot the pattern with the mask overlay

        Args:
            ax (matplotlib.pyplot.Axes, optional): plot's axes. Defaults to None.
            dpi (int) : dpi of the plot
            color_by (str) : Options: "order", "class"
        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's axes
        """
        
        # plot pattern and mask
        fig, ax = _create_figure(ax=ax, dpi=dpi)
        self.rd.plot_pattern(ax)
        h, w = self.rd.pattern.shape
        for i, item in enumerate(self.inst_segs):
            final_mask = np.zeros( (*self.rd.pattern.shape, 4), dtype=np.uint8 )
            _mask = item.mask
            if color_by == "class":
                color = _CM_np[item.label%len(_CM_np)]
            elif color_by == "order":
                color = _CM_np[i%len(_CM_np)]
            else:
                raise ValueError(f"Unrecognize color_by value: {color_by}")
            color = (*color, 255*0.7)
            final_mask[_mask] = np.array( color, dtype=np.uint8)
            ax.text( item.bbox[0] + h*0.01, item.bbox[1], item.label_name, fontdict={"color":"white", "size":5})
            ax.imshow(final_mask, zorder=item.label)        

        return fig, ax


    def plot_region(self, region_id:int, zoom:bool=True, ax=None, **fig_kargs):
        """Plot one region in the form of bounding box. An inset plot of the pattern within the region
        is also inserted if zoom is set to true.

        Args:
            region_id (int): the region id you want to plot
            zoom (bool, optional): Plot the pattern bounded by the region if set to True. Defaults to True.
            ax (matplotlib.pyplot.Axes, optional): The plot's axes. Defaults to None.

        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's axes
        """
        fig, ax = _create_figure(ax=ax, **fig_kargs)
        self.rd.plot_pattern(ax=ax)

        region = self.regions[region_id]

        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)

        # ax.scatter(region.centroid[1], region.centroid[0], c="white", s=1)
        ax.add_patch(rect)
                
        if zoom:
            axins = ax.inset_axes([0.9, 0, 0.1, 1 ])
            axins.set_xticklabels('')
            axins.set_yticklabels('')

            self.rd.crop(minr-1, minc-1, maxr+1, maxc+1, inplace=False).plot_pattern(axins, cmap=self.rd._CMAP)

        return fig, ax


    def plot_regions(self, ax=None, min_area:float=0.0, centroid=False,**fig_kargs):
        """Plot all extracted regions from the mask.

        Args:
            ax (matplotlib.pyplot.Axes, optional): _description_. Defaults to None.
            min_area (float, optional): Regions with area higher than this value would be shown. Defaults to 0.
            centroid (bool, optional): Plot the centroid location if True. Defaults to False.

        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's axes
        """
        
        fig, ax = _create_figure(ax=ax, **fig_kargs)
        # image_label_overlay = label2rgb(self.label, image=self.rd.pattern, bg_label=0)

        # ax.imshow(image_label_overlay)
        self.rd.plot_pattern(ax=ax)

        for rid, region in enumerate(self.regions):
            # take regions with large enough areas
            if region.area >= min_area:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)

                if centroid: ax.scatter(region.centroid[1], region.centroid[0], c="white", s=1)
                ax.add_patch(rect)

                ax.text(x= min(minc - self.rd.pattern.shape[1]*0.01, 0), y=minr, s=f"{rid}", color="white", fontsize="xx-small", va='top', ha='right')
        
        return fig, ax


    def plot_peak_dist(self, ax=None, dist_text_color="white", show_text=True):
        """Plot the extracted peak family periodicity. Call this method when finish the 
        periodicity analysis first.

        Args:
            ax (matplotlib.pyplot.Axes, optional): plot's axes. Defaults to None.
            dist_text_color (str, optional): text color of the periodicity. Defaults to "white".
            show_text (bool, optional): show the exact periodicity on the figure if set to True. Defaults to True.

        Returns:
            matplotlib.pyplot.Figure: plot's figure
            matplotlib.pyplot.Axes: plot's axes
        """
        fig, ax = _create_figure(ax)
        
        self.rd.plot_pattern(ax=ax)
        
        for i, res in enumerate(self.collapses_peaks_pgs):            
            ax.vlines(
                x= self.collapses_peaks_ws_flatten[res.family_idx],
                ymin=0.05*self.rd.pattern.shape[0], ymax=0.95*self.rd.pattern.shape[0],
                alpha=0.5, 
                color=plt.cm.Set1.colors[i%len(plt.cm.Set1.colors)]
            )
            if show_text:
                h, w = self.rd.pattern.shape
                offset = int(h * 0.05)
                for p in self.collapses_peaks_ws_flatten[res.family_idx]:
                    ax.text(x=p, y=offset*(i+1), s=f"{res.avg_dist:.1f}", color=dist_text_color)
        
        return fig, ax

    def get_pgs_intensity(self):
        gauss_fit = {}
        total_intensity = 0
        
        if len(self.collapses_peaks_pgs) > 0:

            for j, pg in enumerate(self.collapses_peaks_pgs):
                group_mask = np.zeros_like(self.rd.pattern, dtype=bool)

                for p, m in zip(pg.family_idx, pg.family_multis):
                    if m == 0: continue 
                    pid = self.collapses_peaks_flatten_pids[p]
                    rid = self.collapses_peaks_flatten_regions[p]
                    region = self.regions[rid]
                    minr, minc , maxr, maxc = region.bbox

                    if len(self.collapses_peaks[rid]) == 1:
                        # regions only has one peak
                        group_mask[minr:maxr, minc:maxc] = group_mask[minr:maxr, minc:maxc] | region.image
                    else:
                        # regions has 2 or more peaks
                        p_neighbor = self.collapses_peaks_flatten_pids[self.collapses_peaks_flatten_regions == rid]
                        pid = pid - np.min(p_neighbor)

                        if rid not in gauss_fit:
                            cs = self.collapses[rid]
                            ps = self.collapses_peaks[rid]
                            xs = cs.ws[ps]
                            hs = cs.spec[ps]
                            width = (max(xs) - min(xs)) / (2*len(ps))

                            guess = []
                            for k in range(len(xs)):
                                guess+=[hs[k], xs[k], width]
                            guess += [0]
                            try:
                                popt, pcov = curve_fit(multi_gaussian, cs.ws, cs.spec, guess)
                                gauss_fit[rid] = (popt, pcov)
                            except Exception as e:
                                # raise e
                                # print(self)
                                # print(guess)
                                # print(rid)
                                # print(ps)
                                warnings.warn(f"Gaussian Fail at {rid}, split regions equally")
                                popt, pcov = (guess, None)
                                gauss_fit[rid] = popt, pcov
                        else:
                            popt, pcov = gauss_fit[rid]

                        # the absolute here is to filp the variance all to positive, some time it would be negative!
                        old_minc = minc
                        minc = max(minc, int(popt[pid*3+1]-abs(popt[pid*3+2])))
                        maxc = min(maxc, int(popt[pid*3+1]+abs(popt[pid*3+2])))
                        
                        try:
                            group_mask[minr:maxr, minc:maxc] = group_mask[minr:maxr, minc:maxc] | region.image[:, minc-old_minc:maxc-old_minc]
                        except:
                            warnings.warn(f"Fail to register to the mask {rid}")
                            
                        pg.intensity = np.sum( group_mask * self.rd.pattern )
                        total_intensity += pg.intensity
                for j, pg in enumerate(self.collapses_peaks_pgs):            
                    pg.ratio = pg.intensity / total_intensity

            pg_intensities = np.array([pg.intensities for pg in self.collapses_peaks_pgs])
            pg_ratios = np.array([pg.ratio for pg in self.collapses_peaks_pgs])

            self.pg_intensities = pg_intensities
            self.pg_ratios = pg_ratios

            return pg_intensities, pg_ratios

    def get_group_intensity(self):
        """Get the total intensity and the relative intensity of each periodicity. If the region corresponding 
        to only one peak family, the sum of the intensity within a region is the group intensity. If there are more than one 
        peak families, a gaussian mixture model would used to fit the spectrum and the width of the gaussian is used to
        estimate the corresponding region of its corresponding family. The group intensity is the sum of that region instead.

        Relative intensity = Group Intensity / Sum(Group Intensity)

        Returns:
            Array: group intensity
            Array: relative group intensity
        """

        # for the case where no cluster labels is presented for this pattern
        if self.cluster_labels_unique is None or len(self.cluster_labels_unique) == 0:
            return [], []

        # gauss_fit = {}
        group_intensity = np.zeros(len(self.cluster_labels_unique))
        pg_intensities, pg_ratio = self.get_pgs_intensity()

        if len(self.collapses_peaks_pgs) > 0:
            for i, ul in enumerate(self.cluster_labels_unique):
                selected = [self.collapses_peaks_pgs[i] for i, cl in enumerate(self.cluster_labels) if cl == ul]
                for j, pg in enumerate(selected):
                    group_intensity[i] = pg.intensity

            group_percent = group_intensity / np.sum(group_intensity)

        self.group_intensity = group_intensity
        self.group_percent = group_percent

        #     cidxs =self.collapses_peaks_flatten_cis

        #     for i, ul in enumerate(self.cluster_labels_unique):
        #         group_mask = np.zeros(self.rd.pattern.shape, dtype=bool)
        #         selected = [self.collapses_peaks_pgs[i] for i, cl in enumerate(self.cluster_labels) if cl == ul]
                
        #         for j, ana in enumerate(selected):
        #             for p in ana.peaks_family:
        #                 if p in cidxs: continue 
        #                 pid = self.collapses_peaks_flatten_pids[p]
        #                 rid = self.collapses_peaks_flatten_regions[p]
        #                 region = self.regions[rid]
        #                 minr, minc , maxr, maxc = region.bbox

        #                 if len(self.collapses_peaks[rid]) == 1:
        #                     # regions only has one peak
        #                     group_mask[minr:maxr, minc:maxc] = group_mask[minr:maxr, minc:maxc] | region.image
        #                 else:
        #                     # regions has 2 or more peaks
        #                     p_neighbor = self.collapses_peaks_flatten_pids[self.collapses_peaks_flatten_regions == rid]
        #                     pid = pid - np.min(p_neighbor)

        #                     if rid not in gauss_fit:
        #                         cs = self.collapses[rid]
        #                         ps = self.collapses_peaks[rid]
        #                         xs = cs.ws[ps]
        #                         hs = cs.spec[ps]
        #                         width = (max(xs) - min(xs)) / (2*len(ps))

        #                         guess = []
        #                         for k in range(len(xs)):
        #                             guess+=[hs[k], xs[k], width]
        #                         guess += [0]
        #                         try:
        #                             popt, pcov = curve_fit(multi_gaussian, cs.ws, cs.spec, guess)
        #                             gauss_fit[rid] = (popt, pcov)
        #                         except Exception as e:
        #                             # raise e
        #                             # print(self)
        #                             # print(guess)
        #                             # print(rid)
        #                             # print(ps)
        #                             warnings.warn(f"Gaussian Fail at {rid}, split regions equally")
        #                             popt, pcov = (guess, None)
        #                             gauss_fit[rid] = popt, pcov
        #                     else:
        #                         popt, pcov = gauss_fit[rid]

        #                     # the absolute here is to filp the variance all to positive, some time it would be negative!
        #                     old_minc = minc
        #                     minc = max(minc, int(popt[pid*3+1]-abs(popt[pid*3+2])))
        #                     maxc = min(maxc, int(popt[pid*3+1]+abs(popt[pid*3+2])))
                            
        #                     try:
        #                         group_mask[minr:maxr, minc:maxc] = group_mask[minr:maxr, minc:maxc] | region.image[:, minc-old_minc:maxc-old_minc]
        #                     except:
        #                         warnings.warn(f"Fail to register to the mask {rid}")
                            
        #         group_intensity[i] = np.sum( group_mask * self.rd.pattern )
        #         group_percent = group_intensity / np.sum(group_intensity)
        # else:
        #     group_percent = np.zeros(len(self.cluster_labels_unique))

        # self.group_intensity = group_intensity
        # self.group_percent = group_percent

        return group_intensity, group_percent
    