from pathlib import Path
from typing import List, Dict, Union, Optional
from dataclasses import dataclass, field, fields
from collections import namedtuple
# from copy import deepcopy

import numpy as np

from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import curve_fit

from scipy.signal import find_peaks
from scipy.spatial import distance_matrix

from scipy.interpolate import interp1d

from lmfit import models as lm_models

from rhana.spectrum.model import SpectrumModel
from rhana.utils import _create_figure, crop, save_pickle, load_pickle


def gaussian(x, A, x0, sig):
    return A*np.exp(-(x-x0)**2/(2*sig**2))


def multi_gaussian(x, *pars):
    offset = pars[-1]
    summation = offset
    for i in range(len(pars)//3):
        g = gaussian(x, pars[i*3], pars[i*3+1], pars[i*3+2])
        summation += g
    return summation


def get_peaks_distance(peaks_idx:List[int], peaks_w:List[float], full=False, polar=True):
    """
    Given peaks of a list of spectrum, this function calculate the
    inter-peak distance for all peaks in one spectrum
    
    Args:
        peaks_idx (list) : list of peak indexes
        peaks_w (list): list of peak locations
        full (bool): return the full distance matrix if set True
        polar (bool): turn the lower triangle of the matrix negative
    
    Returns:
        np.array : a len(peaks_w) x len(peaks_w) distance matrix
    """
    if len(peaks_w.shape) == 1:
        peaks_w = peaks_w[:, None]
    if not full:
        dm = distance_matrix(peaks_w, peaks_w, p=1)
        interdist = dm[np.tril_indices_from(dm, -1)]
    else:
        interdist = distance_matrix(peaks_w, peaks_w)

        # make the distance matrix polarize
        if polar: interdist[np.tril_indices_from(interdist)] = -interdist[np.tril_indices_from(interdist)]

    return interdist


def create_grid(start:float, end:float, center:float, dist:float):
    """ 
    Generate a grid with spaceing 'dist', that center at 'center', range from 'start - center' to 'end - center' 
    
    Args:
        start (float) : grid starting position
        end (float) : grid end position
        center (float) : the grid origin
        dist (float) : the grid spacing
        
    Returns:
        np.array : a grid 
    """
    rn = int((end - center) // dist)
    ln = int((start - center) // dist)
    return np.arange(ln+1, rn+1) * dist


def get_center_peak_idxs(peaks:List[float], spec_center_loc:float, tolerant:float):
    """
    Extract all the center peaks index from a list of peak

    Args:
        peaks (list): a list of peaks
        spec_center_loc (float): center peak location that want to find
        tolerant (float): the maximum allow different between peak center and target center value

    Returns:
        list: center peak indexs
    """
    
    return [i for i, p in enumerate(peaks) if abs(p - spec_center_loc) < tolerant]


def get_center_peak_idx(peaks:List[float], spec_center_loc:float, tolerant:float):
    """
    Extract the center peaks index from a list of peak

    Args:
        peaks (list): a list of peaks
        spec_center_loc (float): center peak location that want to find
        tolerant (float): the maximum allow different between peak center and target center value

    Returns:
        int: center peak index, return -1 if not found
    """    
    
    ci = np.argmin( abs(peaks - spec_center_loc) )
    if peaks[ci] - spec_center_loc < tolerant : return ci
    else: return -1


def get_all_nbr_idxs(center_i:int, idxs:List[int]):
    """
    A generator of all neighbors indexs given a starting index
    Example:

    center_i = 3
    idx = 0, 1, 2, 3, 4, 

    what it yields in order:
        2, 4, 1, 0
        
    Args:
        center_i : the center indexes
        idxs : all the allowed indexes
        
    Returns:
        generator : a generator of indexes
    """
    level = 0
    endleft = False
    endright = False
    maxidx = max(idxs)

    while not (endleft and endright):
        level = level + 1
        if center_i - level >= 0:
            yield center_i - level
        else:
            endleft = True
        if center_i + level <=maxidx:
            yield center_i + level
        else:
            endright = True


def analyze_peaks_distance_cent(
    peaks:List[float],
    center_nbr_dists:List[float],
    center_peak:float, 
    center_peak_i:int,
    grid_min_w:float,
    grid_max_w:float,
    tolerant:float=0.01,
    abs_tolerant:float=10,
    allow_discontinue:float=1
    ): 
    """
    Extract peak families from a list of peaks using periodic analysis. The analysis would start from center peak and iteratively contruct families that has different
    periodicity.  
    
    The actual tolerant which define the maximum allowed deviation between computed next peak in the family and the actual peaks locaitons
        act_tolerant = min(tolerant * dist, abs_tolerant)
        
    Example of how it works:
        peaks : 1 5 7 9 ->|13|<- 17  21 25
        center is 13
        peaks families:
            periodicity of 4: 1 5 9 13 17 21
            periodicity of 6: 7 13 25 (this one should have 21 since we set allow discountine to 1, it is accepted)

    Args:
        peaks (List[float]): list of peaks we analyze
        center_nbr_dists (List[float]): all computed distance between nbrs peaks and the center peaks
        center_peak (float): center peak location
        center_peak_i (int): center peak index
        grid_min_w (float): grid starting position, see 'create_grid'
        grid_max_w (float): grid end position, see 'create_grid'
        tolerant (float, optional): relative tolerant. Defaults to 0.01.
        abs_tolerant (int, optional): absolute tolerant. Defaults to 10.
        allow_discontinue (int, optional): allowed discontinity when searching peaks. Defaults to 1.

    Returns:
        List[PeakAnalysisDetail]: all the discovered peak family
    """
    mask = np.zeros((len(peaks), len(peaks)), dtype=bool)
    out = []

    cp = center_peak
    ci = center_peak_i

    for j in get_all_nbr_idxs(ci, np.arange(len(peaks))):
        if ci == j : continue
        if mask[ci, j] : continue

        dist = abs(center_nbr_dists[j])

        if dist <= abs_tolerant: continue # avoid picking up point near the center points
        grid = create_grid(grid_min_w, grid_max_w, cp, dist)

        nbr_grid_dm = abs(distance_matrix(center_nbr_dists[:, None], grid[:, None])) # this step could be optimize to O(n)
        close = nbr_grid_dm.argmin(axis=1) # a array of peak id that has the shortest distance to one of the tick in the grid 
        
        match_error = nbr_grid_dm[np.arange(len(center_nbr_dists)), close] # get the distance of those peaks to the grid

        select = np.abs(match_error) < min(tolerant * dist, abs_tolerant) # (binary) pick those with acceptable distance

        idx = np.where(select)[0] # (center neighbor integar index) 

        gidx = close[ select ] # (grid integar index)
        uni_gidx = np.unique(gidx)
        uni_gidx.sort()

        continuity = np.diff(uni_gidx) # (ses if there are disconnection)
        allowed = np.all(continuity <= allow_discontinue+1) # (selected if only the pattern has no/less disconnection)

        if allowed and len(uni_gidx) > 1:
            x, y = np.meshgrid(idx, idx)
            mask[x, y] = True # label those selected pick-pick distance as processed, won't touch if again

            selected_multi = grid[gidx] / dist
            
            nonzero_multi = selected_multi != 0

            avg_dist = np.sum( center_nbr_dists[idx][nonzero_multi] / selected_multi[nonzero_multi] ) / (sum(nonzero_multi))
            avg_err = (np.sum( abs(center_nbr_dists[idx][nonzero_multi] / selected_multi[nonzero_multi] - avg_dist) ) ) / (sum(nonzero_multi))

            out.append( PeakAnalysisResult(peaks_family=idx, avg_dist=avg_dist, avg_err=avg_err, detail=None) )
        
    return out


@dataclass
class PeakAnalysisDetail:
    """deprecated"""
    tpd: float
    matched: list


@dataclass
class PeakAnalysisResult:
    """
    Storage class of a peak family and its properties
    """
    peaks_family: np.ndarray
    avg_dist: float
    avg_err: float
    detail: PeakAnalysisDetail


@dataclass
class Spectrum:
    """The Spectrum class is built to abstractalize all spectral data.
    we use ws to store spectrum x-axes values instead xs since xs is 
    first used as the index in our code.
        
    Args:
        spec (np.array): The spectrum values
        ws (np.array): The spectrum x coordinate values        
        
    """

    spec: np.ndarray = field(repr=False, metadata="the intensity of the spectrum stored in an array form")
    ws: np.ndarray = field(metadata="the location of the intensity stored in an array form")

    def copy(self):
        """copy the spectrum object
        
        Returns:
            Spectrum : a new spectrum
        """
        
        return self._update_spectrum(self.spec[...], self.ws[...], inplace=False)

    def _update_spectrum(self, spec, ws, inplace=True, **kargs):
        """do an update in the current spectrum's field or create a
        new spectrum and do update over that object.
        """

        if inplace:
            self.spec = spec
            self.ws = ws
            spectrum = self
        else:
            # copy rest of the fields
            stored = { field.name : getattr(self, field.name) for field in fields(self) if field.name not in ["spec", "ws"] }
            spectrum = self.__class__(spec, ws, **stored)

        for k, v in kargs.items():
            setattr(self, k, v)

        return spectrum

    def crop(self, sw, ew, inplace=True):
        mask = np.logical_and(self.ws >= sw, self.ws <= ew)
        return self._update_spectrum(self.spec[mask], self.ws[mask], inplace=inplace)

    def interpolate(self, ws, fill_value=0, kind='linear', assume_sorted=True, inplace=True):
        """Perform interplaction on the spectrum given a new location ws

        Args:
            ws : a new location to interpolation on
            fill_value : the constant to fill when location is outside the range of the spectrum
            kind : interpolation method. Could be 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.
            assume_sorted : assume x is monotonically increasing value
            inplace : update the current spectrum or create a new spectrum

        Returns: 
            Spectrum : the original or a new spectrum
        """

        f = interp1d(self.ws, self.spec, bounds_error=False, fill_value=fill_value, kind=kind, assume_sorted=assume_sorted)
        new_ws = ws
        new_spec = f(new_ws)

        return self._update_spectrum(spec=new_spec, ws=new_ws, inplace=inplace)


    def normalize(self, min_v=None, max_v=None, inplace=True):
        """
            normalize the spectrum by min, max value
            if min max is not given then it would be computed from the given spectrum

            Args:
                min_v : minimum value
                max_v : maximum value
                inplace : if true update current object else return a new object
                
            Returns:
                Spectrum : the original or a new spectrum
        """
        
        _min = self.spec.min() if min_v is None else min_v
        _max = self.spec.max() if max_v is None else max_v
        
        nspec = (self.spec - _min) / (_max - _min + 1e-5)

        return self._update_spectrum(nspec, self.ws, inplace=inplace)

    def denormalize(self, min_v, max_v, inplace=True):
        """
        Transform the normalized from into the ususal form

        Args:
            min_v (float): minimum value
            max_v ([float): maximum value
            inplace (bool, optional): if true, update the current object else reuturn a new object. Defaults to True.
            
        Returns:
            Spectrum : the original or a new spectrum                        
        """
        nspec = self.spec * (max_v - min_v) + min_v
        return self._update_spectrum(nspec, self.ws, inplace=inplace)

    def scale(self, scaler, inplace=True):
        """
        Scale the signal intensity by the scaler value

        Args:
            scaler : scaling value 
            inplace : if true update current object else return new object
            
        Returns:
            Spectrum : the original or a new spectrum        
        """

        return self._update_spectrum(self.spec*scaler, self.ws, inplace=inplace)

    def savgol(self, window_length=15, polyorder=2, deriv=0, delta=1, mode="wrap", cval=0, inplace=True):
        """
        Savitzky-Golay filter for noise reduction. Parameters see scipy.signal.savgol_filter
        
        Args:
            window_length : The length of the filter window (i.e., the number of coefficients). Must be odd
            polyorder : The order of the polynomial used to fit the samples. polyorder must be less than window_length.
            deriv : The order of the derivative to compute. This must be a nonnegative integer. 
                The default is 0, which means to filter the data without differentiating.
            delta: The spacing of the samples to which the filter will be applied. This is only used if deriv > 0. Default is 1.0.
            mode : Must be ‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ or ‘interp’. 
                This determines the type of extension to use for the padded signal to which the filter is applied. 
                When mode is ‘constant’, the padding value is given by cval. See the Notes for more details on ‘mirror’,
                    ‘constant’, ‘wrap’, and ‘nearest’. When the ‘interp’ mode is selected (the default), no extension is used. 
                Instead, a degree polyorder polynomial is fit to the last window_length values of the edges, 
                    and this polynomial is used to evaluate the last window_length // 2 output values.

            cval : Value to fill past the edges of the input if mode is ‘constant’. Default is 0.0.

        Returns:
            Spectrum : the original or a new spectrum
        """
        filtered = savgol_filter(
            self.spec, 
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta, 
            mode=mode,
            cval= 0, # if mode is not constant then whatever
        )
        return self._update_spectrum(filtered, self.ws, inplace=inplace)

    def smooth(self, sigma:float=1, inplace:bool=True, **kargs):
        """
        Apply gaussian smooth to the spectrum.

        Args:
            sigma (float, optional): Gaussian kernel's sigma. Defaults to 1.
            inplace (bool, optional): if true update current object else return new object. Defaults to True.
            **kargs : see scipy.ndimage.filters.gaussian_filter1d for more parameters

        Returns:
            Spectrum : the original or a new spectrum
        """
                
        nspec = gaussian_filter1d(self.spec, sigma=sigma, **kargs)
        return self._update_spectrum(nspec, self.ws, inplace=inplace)

    def clip(self, clip_min, clip_max, inplace=True):
        return self._update_spectrum( np.clip( self.spec, clip_min, clip_max ), self.ws, inplace=inplace )

    # Backgroun subtraction
    def remove_background(self, n=5, split=[], inplace=True):
        """ 
        Assuming the background error is in linear form.
        Fit a linear line from n data points at the beginning and the end of the spectrum.
        Subtract the spacetrum by the fitted linear intensity.

        Args:
            n : number of entries from front and tail to be consider
            split: keypoints that split the spectrum into different small regions
            inplace (bool, optional): if true update current object else return new object. Defaults to True.
            
        Returns:
            Spectrum : the original or a new spectrum
        """
        x = self.ws
        y = self.spec

        def _remove_background(x, y):
            if n > 1:
                X = np.concatenate( (x[0:n], x[-n:]) )
                Y = np.concatenate( (y[0:n], y[-n:]) )
            else:
                X = np.array([x[0], x[-1]])
                Y = np.array([y[0], y[-1]])
            X = np.stack((X, np.ones_like(X)), axis=1 )
            Y = Y.T
            # pdb.set_trace()
            A = np.linalg.inv(X.T@X)@(X.T@Y)
            # pdb.set_trace()
            pX = np.stack( (x, np.ones_like(x)), axis=1 )
            # pdb.set_trace()
            background = (pX @ A)
            nspec = y - background
            # pdb.set_trace()
            return nspec, background
        
        if split:
            ys = []
            bgs = []
            split = [np.min(x)] + split + [np.max(x)+1e-10]
            for i in range(len(split)-1):
                mask = np.logical_and( x >= split[i], x < split[i+1] )
                # pdb.set_trace()
                ny, bg = _remove_background(x[mask], y[mask])
                ys.append(ny)
                bgs.append(bg)

            new_spec, background = np.concatenate(ys), np.concatenate(bgs)
        else:
            new_spec, background = _remove_background(x, y)

        return self._update_spectrum(new_spec, self.ws, background=background, inplace=inplace)

    # def remove_background(self, n=2, inplace=True):
    #     """ 
    #         Assuming the background error is in linear form.
    #         Fit a linear line from n data points at the beginning and the end of the spectrum.
    #         Subtract the spacetrum by the fitted linear intensity.

    #         Argument :
    #             n : number of entries from front and tail to be consider
    #     """
    #     if n > 1:
    #         X = np.concatenate( (self.ws[0:n], self.ws[-n:]) )
    #         Y = np.concatenate( (self.spec[0:n], self.spec[-n:]) )
    #     else:
    #         X = np.array([self.ws[0], self.ws[-1]])
    #         Y = np.array([self.spec[0], self.spec[-1]])
    #     X = np.stack( (X, np.ones_like(X)), axis=1 )
    #     Y = Y.T

    #     A = np.linalg.inv(X.T@X)@(X.T@Y)
        
    #     pX = np.stack( (self.ws, np.ones_like(self.ws)), axis=1 )
    #     nspec = self.spec - (pX @ A)
    #     return self._update_spectrum( nspec, self.ws, inplace=inplace)


    def filling_flat(self, trunc:float=0.99, inplace:bool=True):
        """
        Filling truncated area with quadratic spline. Return a new 
        PseudoLaueCircleSpectrum if there are area to fill else return 
        the original object.
        
        Args:
            trunc (float) : maximum value where signal is truncated
            inplace (bool, optional): if true update current object else return new object. Defaults to True.
                
        Returns:
            Spectrum : the original or a new spectrum            
        """
        
        # we fill it by the quadratic curve
        ws, spec = self.ws, self.spec
        smask = spec <= trunc
        if smask.sum() < len(spec) and smask.sum() > 2:
            f = interp1d(ws[smask], spec[smask], kind="quadratic")
            spec = f(ws)

        if inplace:
            self.spec = spec
            return self
        else:
            return Spectrum(spec, self.ws)

    def find_spectrum_peaks(self, height:float=0.001, threshold:float=0.001, prominence:float=0.10, **peak_args):
        """
        Find Peaks over the list of spectrum using scipy.signal.find_peaks. The wavelet transform version would be 
        implemented in another method used here.
        
        Arguments:
            height : minimum height of the peak, see find_peaks ref to more detail 
            thres : minimum vertical distance to its neighbor peak, see find_peaks ref to more detail 
            prominence : peak prominence, see find_peaks ref to more detail 
        Returns:
            Array : the peak index
            Array : the peak properties
        """
        peaks, peaks_info = find_peaks(self.spec, height=height, threshold=threshold, prominence=prominence, **peak_args)

        self.peaks = peaks
        self.peaks_info = peaks_info

        return peaks, peaks_info

    def get_peaks_distance(self, full:bool=False, polar:bool=True):
        """
        Given peaks of a list of spectrum, this function calculate the
        inter-peak distance for all peaks in one spectrum
        
        Args:
            full : get the full inter peak distance matrix or only get the upper triangle
            polar : make d[i, j] = -d[j, i]
            
        Returns:
            Array : a matrix of inter-peaks distance
        """
        interdist = get_peaks_distance(self.peaks, self.ws[self.peaks], full, polar)

        self.interdist = interdist
        
        return interdist

    def analyze_peaks_distance_cent(self, tolerant=0.01, abs_tolerant=10, allow_discontinue=1):
        """
        Wrapped method of the analyze_peaks_distance_cent function
        
        Extract peak families from a list of peaks using periodic analysis. 
        The analysis would start from center peak and iteratively contruct families that has differentperiodicity.  
        
        The actual tolerant which define the maximum allowed deviation between computed next 
        peak in the family and the actual peaks locaitons
            act_tolerant = min(tolerant * dist, abs_tolerant)
            
        Example of how it works:
            peaks : 1 5 7 9 ->|13|<- 17  21 25
            center is 13
            peaks families:
                periodicity of 4: 1 5 9 13 17 21
                periodicity of 6: 7 13 25 (this one should have 21 since we set allow discountine to 1, it is accepted)

        Args:

            tolerant (float, optional): relative tolerant. Defaults to 0.01.
            abs_tolerant (int, optional): absolute tolerant. Defaults to 10.
            allow_discontinue (int, optional): allowed discontinity when searching peaks. Defaults to 1.

        Returns:
            List[PeakAnalysisDetail]: all the discovered peak family
        """        
        
        
        grid_max_w = max(self.ws)
        grid_min_w = 0

        interdist = self.get_peaks_distance(full=True, polar=True) # a peak-peak distance matrix
        interdist[np.tril_indices_from(interdist)] = -interdist[np.tril_indices_from(interdist)] # make the distance matrix polarize

        center_peak_i = get_center_peak_idx(self.peaks, int(len(self.ws)//2) , abs_tolerant)
        if center_peak_i == -1 : return []
        center_peak = self.peaks[center_peak_i]
        
        center_nbr_dists = interdist[center_peak_i, :]
        
        return analyze_peaks_distance_cent(self.peaks, center_nbr_dists, center_peak, center_peak_i, grid_min_w, grid_max_w, tolerant, abs_tolerant, allow_discontinue)

    def plot_spectrum(
            self, 
            ax=None, 
            peaks=None, 
            peakgroups=None, 
            offset=0, 
            peak_offset=0, 
            showlegend=False, 
            **fig_kargs
        ):
        """
        Plot the spectrum using matplotlib
        
        Args:
            ax : the matplotlib Axes to plot onto, if None then a new figure is created
            peaks : the peaks index in array form 
            peakgroups : a group index of peak's index telling how peak index is group togethered
            offset : a offset that life the spectrum line 
            peak_offset : a offset that lift the peak symbol away from the spectrum line
            showlegend : show legend
            
        Returns:
            fig : matplotlib Figure
            ax : matplotlib Axes
        """

        # peaks, peaksinfo = peaks
        fig, ax = _create_figure(ax=ax, **fig_kargs)
        peak_offset_arr = np.zeros_like(peaks, dtype=float)

        ax.plot(self.ws, self.spec+offset)

        if peaks is not None:
            if peakgroups is not None:
                for g, dist in peakgroups[:-1]:
                    ax.plot(self.ws[peaks[g]], self.spec[peaks[g]]+offset+peak_offset_arr[g], "x", label=f"dist={dist:.1f}")
                    peak_offset_arr[g] += peak_offset

                g, _ = peakgroups[-1]
                ax.plot(self.ws[peaks[g]], self.spec[peaks[g]]+offset+offset+peak_offset_arr[g], "o")
            else:
                ax.plot(self.ws[peaks], self.spec[peaks]+offset+peak_offset, "x")
        
        if showlegend: ax.legend()
        return fig, ax

    @staticmethod        
    def get_peaks_group(ana_res, peaks, exclusive=True):
        """
        Get all peaks with similar peaks distance for each spectrum in the spectrum list.
        This method run on the ana_res where the peaks with similar inter-peak distances are 
        identified already. The method here just to make sure one peak is only presented in one group,
        which is constraint by picking peak from the group with lower peak distance then ignore any
        group has overlapping peaks but with higher peak distance.

        Return a list of the grouped peaks and their average peak distance 
        
        Args:
            ana_res (list): list of PeakAnalysisResult
            peaks (list): fitted peaks
            exclusive (bool, optional): not allow one peak position to be occupy many time if True
            
        Returns:
            List : a list of peakgroups
        """
        
        allpeaks_unfilterd = [ (res.peaks_family, res.avg_dist) for res in ana_res ]
        selected = np.repeat(False, len(peaks))
        allpeaks_unfilterd = sorted( allpeaks_unfilterd, key= lambda x: x[1] )

        out = []
        for family, avg_dist in allpeaks_unfilterd:
            if avg_dist < np.inf :
                if (not exclusive or np.all( selected[family] == False ) ):
                    out.append((family, avg_dist))
                    selected[family] = True
        out.append( ((np.arange(len(peaks))[~selected]).tolist(), -1 ) )
        return out


    def save(self, path:Union[str, Path], name:str="spectrum"):
        """
        Save the spectrum class into a pickle file
        The file directory would be 
            path/f"{name}.pkl"

        Args:
            path (Union[str, Path]): folder to save
            name (str, optional): filename to save. Defaults to "spectrum".

        Returns:
            Spectrum : the original or a new spectrum               
            
        """
        
        _exclude = ['sm']
        path = Path(path)
        temps = {k:v for k, v in self.__dict__.items() if k in _exclude }
        
        for k in _exclude: setattr(self, k, None)

        save_pickle(self, path/f"{name}.pkl")

        if "sm" in temps: temps['sm'].save(path)

        for k, v in temps.items(): setattr(self, k, v)
        
        return self
    
    @classmethod
    def load(cls, path, name="spectrum", ignore_err=True):
        """
        Load the spectrum class from a pickle file
        The file directory would be 
            path/f"{name}.pkl"

        Args:
            path (Union[str, Path]): folder to save
            name (str, optional): filename to save. Defaults to "spectrum".
            ignore_err (bool): If set true then error would only be printed. Default to True

        Returns:
            Spectrum : the original or a new spectrum 
        """        
        
        self = load_pickle(path/f"{name}.pkl")
        if hasattr(self, "sm"):
            try:
                self.sm = SpectrumModel.load(path)
            except Exception as e:
                if ignore_err:
                    print(e)
                else:
                    raise e
        return self


@dataclass
class CollapseSpectrum(Spectrum):
    """
    An intergrated 1D spectrum from a region of a 2D diffraction image

    sx (int) : cropped starting point - x
    sy (int) : cropped starting point - y
    ex (int) : cropped end point - x
    ey (int) : cropped end point - y
    """

    sx: int # cropped starting point - x
    sy : int #  cropped starting point - y
    ex : int # cropped end point - x
    ey : int # cropped end point - y

    @classmethod
    def from_rheed_horizontal(cls, rd, sx, sy, ex, ey):
        """
        Get horizontal collapse spectrum which is basically the integrated spectrum over rows
        
        rd (Rheed): rheed pattern
        sx (int): cropped starting point - x
        sy (int): cropped starting point - y
        ex (int): cropped end point - x
        ey (int): cropped end point - x
        
        Returns:
            CollapseSpectrum : the integrated spectrum
        """
        pattern = crop(rd.pattern, sx, sy, ex, ey)
        return cls(pattern.sum(axis=0), np.arange(sy, ey), sx, sy, ex, ey)

    @classmethod
    def from_rheed_vertical(cls, rd, sx, sy, ex, ey):
        """
        Get vertical collapse spectrum which is basically the integrated spectrum over columns
            
        rd (Rheed): rheed pattern
        sx (int): cropped starting point - x
        sy (int): cropped starting point - y
        ex (int): cropped end point - x
        ey (int): cropped end point - x
        
        Returns:
            CollapseSpectrum : the integrated spectrum            
        """

        pattern = crop(rd.pattern, sx, sy, ex, ey)
        return cls(pattern.sum(axis=1), np.arange(sx, ex), sx, sy, ex, ey)

