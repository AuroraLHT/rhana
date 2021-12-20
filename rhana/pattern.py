import warnings
import itertools
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


from PIL import Image

from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import curve_fit

# from skimage import data
# from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.transform import rotate as skim_rotate

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import plotly.graph_objects as go

from skimage.morphology import reconstruction
from skimage.measure import label, regionprops
from skimage.measure import moments

from typing import List, Dict, Union

from rhana.utils import _CM_rgb, _CM, multi_gaussian, gaussian, show_circle, _create_figure, crop
from rhana.io import kashiwa as ksw
from rhana.spectrum.spectrum import CollapseSpectrum, analyze_peaks_distance_cent, get_center_peak_idxs, get_center_peak_idx, get_peaks_distance
from rhana.utils import *

from dataclasses import dataclass, field

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
            convert horizontal distance 2 G
            dist : horizontal distance is in pixel
            return dG is in nm^-1
        """
        real_dist = self.pixel_real * dist
        # print(real_dist)
        k0 = 2*np.pi/self.wave_length
        # print(k0)
        r = (self.sub_ccd_dist/real_dist)**2
        dG = k0 / np.sqrt( r + 1 ) * 1e-6
        return dG

class Rheed:
    _CMAP = "cividis"

    def __init__(self, pattern, min_max_scale=False, standard_norm=False, AOI=None, config:RheedConfig=None):
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
        return cls(np.mean(patterns, axis=0), min_max_scale=min_max_scale, standard_norm=standard_norm, AOI=AOI)

    @classmethod 
    def from_multi_kashiwa(cls, paths, contain_hw=True, min_max_scale=False, standard_norm=False, log=False):
        patterns = [ cls.from_kashiwa(path,contain_hw=contain_hw, min_max_scale=min_max_scale, standard_norm=standard_norm, log=log).pattern for path in paths ]
        return cls.from_multi(patterns, AOI=~ksw._APMASK)

    @classmethod
    def from_image(cls, path, rotate=0, crop_box=None, min_max_scale=False, standard_norm=False,  AOI=None, config=None):
        img = Image.open(path)
        if rotate != 0 :
            img = img.rotate(rotate)
        if crop_box is not None:
            img = img.crop(crop_box)
        return cls(np.array(img)/255, min_max_scale=min_max_scale, standard_norm=standard_norm, AOI=AOI, config=config)

    def mean_clip(self, alpha=1, inplace=True):
        if self.AOI is not None:
            mean = self.pattern[self.AOI].mean()
        else:
            mean = self.pattern.mean()
        
        return self._update_pattern(np.clip(self.pattern-mean, 0, 1), inplace=inplace)
            
    def _update_pattern(self, pattern, inplace):
        if inplace:
            self.pattern = pattern
            return self
        else:
            return Rheed(pattern)

    def crop(self, sx, sy, ex, ey, inplace=False):
        return self._update_pattern(crop(self.pattern, sx, sy, ex, ey), inplace=inplace)

    def remove_bg(self, dilation_bias=0.5, inplace=True):
        im, dilated, hdome0 = image_bg_sub_dilation(self.pattern, dilation_bias)
        dilated = hdome0 / hdome0.max()

        return self._update_pattern(dilated, inplace=inplace)

    def smooth(self, inplace=True, **gaussian_kargs):
        smoothed = gaussian(self.pattern, **gaussian_kargs)
        return self._update_pattern(smoothed, inplace=inplace)

    def min_max_scale(self, inplace=True):
        pattern = ( self.pattern - self.pattern.min() ) / (self.pattern.max() - self.pattern.min() + 1e-5)
        return self._update_pattern(pattern, inplace=inplace)

    def standard_norm(self, inplace=True):
        
        if self.AOI is not None:
            pattern = ( self.pattern - self.pattern[self.AOI].mean() ) / (self.pattern[self.AOI].std() + 1e-5)
        else:
            pattern = ( self.pattern - self.pattern.mean() ) / (self.pattern.std() + 1e-5)
        return self._update_pattern(pattern, inplace=inplace)

    def get_blobs(self, max_sigma=30, num_sigma=10, threshold=.1, **blob_kargs):
        """
            Find bright spots in the RHEED pattern image using the blob detection algorithm
            with Laplacian of Gaussian method.

            Arguments:
                min_sigma : float, optional
                    The minimum standard deviation for Gaussian Kernel. Keep this low to detect smaller blobs.
                max_sigma : float, optional
                    The maximum standard deviation for Gaussian Kernel. Keep this high to detect larger blobs.
                num_sigma : int, optional
                    The number of intermediate values of standard deviations to consider between min_sigma and max_sigma.
                threshold : float, optional.
                    The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored. Reduce this to detect blobs with less intensities.
                overlap : float, optional
                    A value between 0 and 1. If the area of two blobs overlaps by a fraction greater than threshold, the smaller blob is eliminated.
                log_scale : bool, optional
                    If set intermediate values of standard deviations are interpolated using a logarithmic scale to the base 10. If not, linear interpolation is used
        """
        blobs_log = blob_log(self.pattern, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, **blob_kargs)

        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

        self.blobs = blobs_log
        return blobs_log

    def plot_blobs(self, blob_color="red", ax=None, **fig_kargs):
        """
            plot the blobs location on the RHEED pattern
            
            Arguments:
                img : RHEED pattern image
                blobs : output from blobs detection algorithm or get_blobs()
        """
        
        fig, ax = _create_figure(ax=ax, **fig_kargs)
        self.plot_pattern(ax=ax)
        for blob in self.blobs:
            x, y, r = blob
            show_circle(ax, (x,y), r, color=blob_color)
        ax.set_axis_off()

    def plot_0laue(self, ax=None, **fig_kargs):
        if hasattr(self, "xy") and hasattr(self, "r"):
            return self.plot_nlaue(self.xy, [self.r], ax=ax, **fig_kargs)

    def get_direct_beam(self, rmin=3):
        """
            Find the blob that contain direct beam information. Assuming the direct beam is the top one.

            Arguments:
                rmin : minimum radius for a spot to be selected
        """
        def get_top_blob(blobs):
            top = np.inf
            tb = None
            for i, (x, y, r) in enumerate(blobs):
                if x<top and r>=rmin:
                    tb = (x, y)
                    top = x
            return tb, i

        if hasattr(self, "blobs"):
            self.db, self.db_i = get_top_blob(self.blobs)
            return self.db, self.db_i
        else:
            raise Exception("Blobs is not stored in the object. Run .get_blobs() first")

    def get_specular_spot(self, rmin=3):
        """
            Find the blob that contain specular spot information. Return None, -1 if no specular spot
            spot is found.

            Arguments:
                rmin : minimum radius for a spot to be selected
        """
        
        dbx, dby, dbr = self.blobs[self.db_i]
        ss = None
        top = np.inf
        i = -1
        for i, (x,y,r) in enumerate(self.blobs):
            if (y==dby and x==dbx and r==dbr): continue

            if (r>=rmin and x<=dbx+dbr and x>=dbx-dbr and top>x):
                ss = (x, y,r)
                top = x
        self.ss = ss
        return ss, i

    def get_Laue0(self):
        """
            get 0-Laue Circle location and radius given the direct beam and specular spot blobs            
        """
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

    def plot_nlaue(self, xy, rs, ax=None, **fig_kargs):
        fig, ax = _create_figure(ax=ax, **fig_kargs)
        self.plot_pattern(ax=ax)
        for r in rs:
            show_circle(ax, xy, r )
        ax.set_axis_off()

    def plotly_pattern(self, mp_spectrums=None, spectrums=None, cluster_labels=None, clustered_locations=None, dist_flat=None, **fig_kargs):

        fig = go.Figure(**fig_kargs)
        fig = fig.add_trace(go.Heatmap(z=self.pattern,  colorscale="Cividis"))
        fig.update_layout(height=600, width=800, yaxis=dict(autorange='reversed'), showlegend=True)

        if cluster_labels is not None and clustered_locations is not None and dist_flat is not None:
            xs_list = clustered_locations[0]
            ys_list = clustered_locations[1]

            for label in np.unique(cluster_labels):
                if label == -1 : continue

                indexs = np.where(cluster_labels==label)[0]
                cluster_center = np.mean(dist_flat[indexs])

                xs = np.concatenate( [ xs_list[i] for i in indexs ] )
                ys = np.concatenate( [ ys_list[i] for i in indexs ] )

                fig.add_trace( 
                    go.Scatter(
                        x=xs, y=ys,
                        marker=go.scatter.Marker(
                            symbol="x-open",
                            color=_CM[label%len(_CM)],
                            opacity=1,
                            size=10,
                        ),
                        name=f"c{label} = {cluster_center :.2f}",
                        mode="markers"
                    )
                )

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
            #         bgcolor="LightSteelBlue",
                    bgcolor="Black",
                    bordercolor="Black",
                    borderwidth=2
                )
            )
        return fig

    def plot_pattern(self, ax=None, mp_spectrums=None, cluster_labels=None, clustered_locations=None, dist_flat=None, cmap=None, **fig_kargs):
        fig, ax = _create_figure(ax, **fig_kargs)
        ax.imshow(self.pattern, cmap=cmap if cmap is not None else self._CMAP)

        if mp_spectrums is not None and mp_spectrums.spectrums is not None:
            for i, spec in enumerate(mp_spectrums.spectrums):
                show_circle(ax, spec.xy, spec.pr, alpha=0.3, color="red")

        if mp_spectrums is not None and hasattr(mp_spectrums,"cluster_labels") and hasattr(mp_spectrums, "clustered_locations") and hasattr(mp_spectrums, "dist_flat"):

            xs_list = mp_spectrums.clustered_locations[0]
            ys_list = mp_spectrums.clustered_locations[1]

            for label in np.unique(mp_spectrums.cluster_labels):
                if label == -1 : continue
                
                indexs = np.where(mp_spectrums.cluster_labels==label)[0]
                cluster_center = np.mean(mp_spectrums.dist_flat[indexs])
                
                xs = np.concatenate( [ xs_list[i] for i in indexs ] )
                ys = np.concatenate( [ ys_list[i] for i in indexs ] )
                    
                ax.scatter(xs, ys, marker="x", c=_CM[label%len(_CM)], alpha=0.8, label=f"c{label} = {cluster_center :.2f}")
            ax.legend()
            # the plotting mode would change from image to scatter so we need to fip the y-axis and constraint the x, y range
            ax.set_ylim(self.pattern.shape[0], 0)
            ax.set_xlim(0, self.pattern.shape[1])

            ax.set_xlabel("x (pixel)")
            ax.set_ylabel("y (pixel)")

        return fig, ax

    def get_fft(self, center=True):
        img = self.pattern
        f = np.fft.fft2(img)
        if center: f = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(f)
        self.fft = f
        self.fft_center = center
        self.fft_mag = magnitude_spectrum
        return f, magnitude_spectrum
    
    def fft_reconstruct(self, window_x, window_y, inplace=True):
        fft = self.fft.copy()
        rows, cols = self.fft.shape
        crow,ccol = int(rows/2) , int(cols/2)
        fft[crow-window_y:crow+window_y+1, ccol-window_x:ccol+window_x+1] = 0
        if self.fft_center: fft = np.fft.ifftshift(fft)
        recon = np.fft.ifft2(fft)
        return self._update_pattern(np.abs(recon), inplace)

    def plot_fft(self, ax=None, **fig_kargs):
        fig, ax = _create_figure(ax, **fig_kargs)
        ax.imshow(self.fft_mag)
        ax.set_title('Magnitude Spectrum')
        ax.set_xlabel("Frequency X")
        ax.set_ylabel("Frequency y")
        ax.set_yticklabels(ax.get_yticks().astype(int) - int(self.fft_mag.shape[0] // 2))
        ax.set_xticklabels(ax.get_xticks().astype(int) - int(self.fft_mag.shape[1] // 2))
        return fig, ax

class RheedMask():
    def __init__(self, rd:Rheed, mask:np.ndarray):
        self.rd = rd
        self.mask = mask

    def crop(self, sx, sy, ex, ey, inplace=False):
        if inplace:
            self.rd = self.rd.crop(sx, sy, ex, ey, inplace=inplace)
            self.mask = crop(self.mask, sx, sy, ex, ey)
            return self
        else:
            rd = self.rd.crop(sx, sy, ex, ey, inplace=inplace)
            mask = crop(self.mask, sx, sy, ex, ey)
            return RheedMask(rd, mask)

    def get_regions(self, with_intensity=False):
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
        filtered = [ r for r in self.regions if r.area >= min_area ]
        if inplace: self.regions = filtered
        return [ r for r in self.regions if r.area >= min_area ]

    def get_region_collapse(self, region, direction="h"):
        sx, sy, ex, ey = region.bbox
        if direction == "h":
            cs = CollapseSpectrum.from_rheed_horizontal(self.rd, sx, sy, ex, ey)
        else:
            cs = CollapseSpectrum.from_rheed_vertical(self.rd, sx, sy, ex, ey)
        return cs

    def get_regions_collapse(self, direction="h"):
        # collapse :list[1d spectrums] -> store
        self.collapses = []
        for region in self.regions:
            cs = self.get_region_collapse(region, direction)
            self.collapses.append(cs)
        return self.collapses
    
    def clean_collapse(self, smooth=True, rm_bg=True, scale=True):
        for cs in self.collapses:
            if rm_bg:
                try:
                    cs.remove_background()
                except np.linalg.LinAlgError as e:
                    warnings.warn(f"Encounter LinAlgError when doing background removal! {e}")
            if scale: cs.normalize()
            if smooth: cs.smooth()
        return self

    def fit_collapse_peaks(self, height, threshold, prominence):
        # peaks dict : list->
        self.collapses_peaks = []
        self.collapses_peaks_ws = []
        self.collapses_peaks_info = []
        for cs in self.collapses:
            if cs is not None:
                peaks, peaks_info = cs.fit_spectrum_peaks(height=height, threshold=threshold, prominence=prominence)
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
        centroids = np.stack([ self._get_region_centroid(region) for region in self.regions ], axis=0)
        dists = np.linalg.norm(centroids - np.array([x,y]), axis=1)
        i = np.argmin(dists)
        return self.regions[i], i
    
    def get_region_within(self, x, y):
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
            raise Exception(f"method of {method} is unknown, allow top, top+trackes, and tracker")

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
        allpeaks = self._flatten_peaks()

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

        self.collapses_peaks_flatten_ana_res = analyze_peaks_distance_cent(
            self.collapses_peaks_flatten,
            self.collapses_peaks_flatten_nbr_dist,
            self.collapses_peaks_flatten_ciw,
            self.collapses_peaks_flatten_ci,
            grid_min_w= 0,
            grid_max_w= self.rd.pattern.shape[1],
            tolerant= tolerant,
            abs_tolerant= abs_tolerant,
            allow_discontinue= allow_discontinue
        )

        return self.collapses_peaks_flatten_ana_res

    def plot_pattern_masks(self, ax=None, name=None, regions=True, split=False):
        # plot pattern and mask
        fig, ax = _create_figure(ax=ax)

        self.rd.plot_pattern(ax)
        ax.imshow(self.mask, alpha=0.7)
        return fig, ax

    def plot_region(self, region_id, zoom=True, ax=None, **fig_kargs):
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

    def plot_regions(self, ax=None, zoom=False, min_area=0, centroid=False,**fig_kargs):
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
        fig, ax = _create_figure(ax)
        
        self.rd.plot_pattern(ax=ax)
        
        for i, res in enumerate(self.collapses_peaks_flatten_ana_res):            
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
        # max_width??
        gauss_fit = {}
        group_intensity = np.zeros(len(self.cluster_labels_unique))

        cidxs =self.collapses_peaks_flatten_cis

        for i, ul in enumerate(self.cluster_labels_unique):
            group_mask = np.zeros(self.rd.pattern.shape, dtype=bool)
            selected = [ self.collapses_peaks_flatten_ana_res[i] for i, cl in enumerate(self.cluster_labels) if cl == ul ]
            
            for j, ana in enumerate(selected):
                for p in ana.peaks_family:
                    if p in cidxs: continue 
                    pid = self.collapses_peaks_flatten_pids[p]
                    rid = self.collapses_peaks_flatten_regions[p]
                    region = self.regions[rid]
                    minr, minc , maxr, maxc = region.bbox

                    if len(self.collapses_peaks[rid]) == 1:
                        # region only has one peak
                        group_mask[minr:maxr, minc:maxc] = group_mask[minr:maxr, minc:maxc] | region.image
                    else:
                        # region has 2 or more peaks
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

            self.group_intensity = group_intensity
            self.group_percent = group_percent

        return group_intensity, group_percent