from typing import List, Dict, Union, Optional
from collections import defaultdict
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
from dataclasses import dataclass


@dataclass
class PeriodicityAnalysisDetail:
    """deprecated"""
    tpd: float
    matched: list


@dataclass
class PeriodicityGroup:
    """
    Storage class of a peak family and its properties
    """
    family_idx: np.ndarray
    family_elements: np.ndarray
    family_multis: np.array 
    avg_dist: float
    avg_err: float
    intensity: float = None
    ratio: float = None
    # detail: PeriodicityAnalysisDetail


class PeriodicityAnalyzer:
    def __init__(self, tolerant=0.01, abs_tolerant=10, allow_discontinue=1, center_tolerant=None) -> None:

        self.tolerant = tolerant
        self.abs_tolerant = abs_tolerant
        self.allow_discontinue = allow_discontinue
        self.center_tolerant = center_tolerant if center_tolerant is not None else abs_tolerant

    def _match_grid(
        self,
        grid,
        grid_dist,
        center_nbr_dists,
        tolerant,
        abs_tolerant,
        allow_discontinue
    ):
        nbr_grid_dm = abs(distance_matrix(center_nbr_dists[:, None], grid[:, None])) # this step could be optimize to O(n)
        close = nbr_grid_dm.argmin(axis=1) # a array of peak id that has the shortest distance to one of the tick in the grid 
        
        match_error = nbr_grid_dm[np.arange(len(center_nbr_dists)), close] # get the distance of those arr to the grid

        select_mask = np.abs(match_error) < min(tolerant * grid_dist, abs_tolerant) # (binary) pick those with acceptable distance

        select_idx = np.where(select_mask)[0] # (center neighbor integar index) 

        gidx = close[ select_mask ] # (grid integar index)
        uni_gidx = np.unique(gidx)
        uni_gidx.sort()

        continuity = np.diff(uni_gidx) # (ses if there are disconnection)
        allowed = np.all(continuity <= allow_discontinue+1) # (selected if only the pattern has no/less disconnection)

        if allowed and len(uni_gidx) > 1:
            selected_multi = grid[gidx] / grid_dist            
            # nonzero_multi = selected_multi != 0
            # avg_dist = np.sum( center_nbr_dists[select_idx][nonzero_multi] / selected_multi[nonzero_multi] ) / (sum(nonzero_multi))
            # avg_err = (np.sum( abs(center_nbr_dists[select_idx][nonzero_multi] / selected_multi[nonzero_multi] - avg_dist) ) ) / (sum(nonzero_multi))
        else:
            selected_multi = []            
        return allowed, select_idx, gidx, selected_multi, match_error[select_mask]

    def _get_group(self, arr, center_nbr_dists, select_idx, selected_multi):
        # print(arr)
        # if 22 in select_idx: 
        #     print(arr)
        #     print(select_idx)
        nonzero_multi = selected_multi != 0
        avg_dist = np.sum( 
            center_nbr_dists[select_idx][nonzero_multi] / selected_multi[nonzero_multi] ) / (sum(nonzero_multi)
        )
        avg_err = (np.sum( abs(center_nbr_dists[select_idx][nonzero_multi] / selected_multi[nonzero_multi] - avg_dist) ) ) / (sum(nonzero_multi))

        return PeriodicityGroup(
            family_idx=select_idx,
            family_elements=arr[select_idx],
            family_multis=selected_multi,
            avg_dist=avg_dist,
            avg_err=avg_err,
            # detail=None
        )


    def match_periodicity(
        self,
        arr:List[float],
        periodicities:List[PeriodicityGroup],
        center:float, 
        grid_min:float,
        grid_max:float,
        arr_mask:Optional[List[bool]]=None,
    ):
        arr = np.array(arr)
        if arr_mask is None: arr_mask = np.zeros_like(arr, dtype=bool)
        ci, cp, _ = get_cloest_element(arr, center, self.center_tolerant)
        center_nbr_dists = np.abs( arr - cp ).astype(float)
        if np.any(arr_mask): center_nbr_dists[arr_mask] = np.inf
        center_peaks_mask = center_nbr_dists < self.center_tolerant
 
        match_order = np.argsort( [ p.avg_dist for p in periodicities ] )[::-1] # match from big to small
        targets = [ (create_grid(grid_min, grid_max, cp, p.avg_dist), p.avg_dist) for p in periodicities]
        
        max_value= np.inf

        matched = np.zeros(len(arr), dtype=bool)
        match_res = defaultdict(lambda:[])

        match_periodicities = [None] * len(periodicities)

        for i in match_order:
        # for i, (grid, grid_dist) in enumerate(targets):
            grid, grid_dist = targets[i]
            allowed, select_idx, gidx, selected_multi, match_error = self._match_grid(
                grid=grid,
                grid_dist=grid_dist,
                center_nbr_dists=center_nbr_dists,
                tolerant=self.tolerant,
                abs_tolerant=self.abs_tolerant,
                allow_discontinue=self.allow_discontinue
            )

            if allowed and len(np.unique(gidx))>1:
                match_periodicities[i] = self._get_group(
                    arr, 
                    center_nbr_dists, 
                    select_idx, 
                    selected_multi= selected_multi
                )
                
                match_res[i] = select_idx

                matched[select_idx] = True
                matched[center_peaks_mask] = False
                center_nbr_dists[matched] = max_value


        return match_periodicities, match_res, matched


    def match_periodicity2(
        self,
        arr:List[float],
        periodicities:List[PeriodicityGroup],
        center:float, 
        grid_min:float,
        grid_max:float,
        arr_mask:Optional[List[bool]]=None,
    ):
        arr = np.array(arr)
        if arr_mask is None: arr_mask = np.zeros_like(arr, dtype=bool)
        ci, cp, _ = get_cloest_element(arr, center, self.center_tolerant)
        center_nbr_dists = np.abs( arr - cp )
        if np.any(arr_mask): center_nbr_dists[arr_mask] = np.inf
        center_peaks_mask = center_nbr_dists < self.center_tolerant
 
        # match_order = np.argsort( [ p.avg_dist for p in periodicities ] )[::-1] # match from big to small

        targets = [ (create_grid(grid_min, grid_max, cp, p.avg_dist), p.avg_dist) for p in periodicities]
        
        match_matrix = np.zeros((len(arr), len(targets)), dtype=float)
        max_value= np.inf
        match_matrix.fill(np.inf)

        # for i in match_order:
        for i, (grid, grid_dist) in enumerate(targets):
            grid, grid_dist = targets[i]
            allowed, select_idx, gidx, selected_multi, match_error = self._match_grid(
                grid=grid,
                grid_dist=grid_dist,
                center_nbr_dists=center_nbr_dists,
                tolerant=self.tolerant,
                abs_tolerant=self.abs_tolerant,
                allow_discontinue=self.allow_discontinue
            )

            if allowed and len(np.unique(gidx))>1:
                match_matrix[select_idx, i] = match_error
            match_matrix[center_peaks_mask, i] = max_value

        # print(match_matrix)
        match_res = defaultdict(lambda:[])
        # row is the arr index, c is the pg index
        # row_ind, col_ind = linear_sum_assignment(match_matrix)
        row_ind = np.arange(len(match_matrix))
        col_ind = np.argmin(match_matrix, axis=1)

        matched = np.zeros(len(arr), dtype=bool)
        for r, c in zip(row_ind, col_ind):
            if match_matrix[r, c] != max_value:
                match_res[c].append(r)
                matched[r] = True

        # since we filtered out the center, we should put it back
        # but we don't register the center element since we still need it for
        # further analysis
        for k, v in match_res.items():
            v.append(ci)
        
        match_periodicities = [None] * len(periodicities)
        for k, v in match_res.items():
            match_periodicities[k] = self._get_group(
                arr, 
                center_nbr_dists, 
                v, 
                selected_multi= np.round(center_nbr_dists[v]/periodicities[k].avg_dist)
            )

        return match_periodicities, match_res, matched


    def analyze(
        self,
        arr:List[float],
        center:float, 
        grid_min:float,
        grid_max:float,
        arr_mask:Optional[List[bool]]=None,
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
        arr = np.array(arr)
        if arr_mask is None: arr_mask = np.zeros_like(arr, dtype=bool)

        mask = np.zeros((len(arr), len(arr)), dtype=bool)
        out = []
        ci, cp, _ = get_cloest_element(arr, center, self.center_tolerant)
        center_nbr_dists = np.abs( arr - cp ).astype(np.float64)
        if np.any(arr_mask): center_nbr_dists[arr_mask] = np.inf

        for j in get_all_nbr_idxs(ci, np.arange(len(arr))):
            if ci == j : continue
            if mask[ci, j] : continue

            dist = abs(center_nbr_dists[j])

            if dist <= self.center_tolerant: continue # avoid picking up point near the center points
            if dist == np.inf: continue

            grid = create_grid(grid_min, grid_max, cp, dist)

            allowed, select_idx, gidx, selected_multi, match_error = self._match_grid(
                grid=grid,
                grid_dist=dist,
                center_nbr_dists=center_nbr_dists,
                tolerant=self.tolerant,
                abs_tolerant=self.abs_tolerant,
                allow_discontinue=self.allow_discontinue
            )

            x, y = np.meshgrid(select_idx, select_idx)
            mask[x, y] = True # label those selected pick-pick distance as processed, won't touch if again

            if allowed and len(np.unique(gidx)) > 1:
                x, y = np.meshgrid(select_idx, select_idx)
                mask[x, y] = True # label those selected pick-pick distance as processed, won't touch if again
                
                out.append( self._get_group(arr, center_nbr_dists, select_idx, selected_multi) )

                # nonzero_multi = selected_multi != 0
                # avg_dist = np.sum( center_nbr_dists[select_idx][nonzero_multi] / selected_multi[nonzero_multi] ) / (sum(nonzero_multi))
                # avg_err = (np.sum( abs(center_nbr_dists[select_idx][nonzero_multi] / selected_multi[nonzero_multi] - avg_dist) ) ) / (sum(nonzero_multi))

                # out.append( 
                #     PeriodicityGroup(
                #         family_idx=select_idx,
                #         family_elements=arr[select_idx],
                #         family_multis=selected_multi,
                #         avg_dist=avg_dist, 
                #         avg_err=avg_err, 
                #         detail=None
                #     )
                # )
            
        return out

    def is_sub_family(self, group1, group2):
        """
            check if group 1 or 2 average distance is some integer of
            gourp 2 or 1. Very important when checking the epitaxial 
            phase. Some growth actually make 1/n streaks between the 
            original substrate streak.
        """
        base = min(group1.avg_dist, group2.avg_dist)
        multi = max(group1.avg_dist, group2.avg_dist)

        abs_diff = np.abs( np.round( multi / base ) * base - multi )
        diff = abs_diff / base

        return diff <= self.tolerant and abs_diff <= self.abs_tolerant

        #     nbr_grid_dm = abs(distance_matrix(center_nbr_dists[:, None], grid[:, None])) # this step could be optimize to O(n)
        #     close = nbr_grid_dm.argmin(axis=1) # a array of peak id that has the shortest distance to one of the tick in the grid 
            
        #     match_error = nbr_grid_dm[np.arange(len(center_nbr_dists)), close] # get the distance of those arr to the grid

        #     select = np.abs(match_error) < min(tolerant * dist, abs_tolerant) # (binary) pick those with acceptable distance

        #     idx = np.where(select)[0] # (center neighbor integar index) 

        #     gidx = close[ select ] # (grid integar index)
        #     uni_gidx = np.unique(gidx)
        #     uni_gidx.sort()

        #     continuity = np.diff(uni_gidx) # (ses if there are disconnection)
        #     allowed = np.all(continuity <= allow_discontinue+1) # (selected if only the pattern has no/less disconnection)

        #     if allowed and len(uni_gidx) > 1:
        #         x, y = np.meshgrid(idx, idx)
        #         mask[x, y] = True # label those selected pick-pick distance as processed, won't touch if again

        #         selected_multi = grid[gidx] / dist
                
        #         nonzero_multi = selected_multi != 0

        #         avg_dist = np.sum( center_nbr_dists[idx][nonzero_multi] / selected_multi[nonzero_multi] ) / (sum(nonzero_multi))
        #         avg_err = (np.sum( abs(center_nbr_dists[idx][nonzero_multi] / selected_multi[nonzero_multi] - avg_dist) ) ) / (sum(nonzero_multi))

        #         out.append( 
        #             PeriodicityAnalizer(
        #             family=idx,
        #             family_elements=arr[idx],
        #             family_multis=selected_multi,
        #             avg_dist=avg_dist, 
        #             avg_err=avg_err, 
        #             detail=None) 
        #         )
            
        # return out


def get_pair_distance(arr:List[float],  full=False, polar=True):
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


    if len(arr.shape) == 1:
        arr = arr[:, None]

    if not full:
        dm = distance_matrix(arr, arr, p=1)
        interdist = dm[np.tril_indices_from(dm, -1)]
    else:
        interdist = distance_matrix(arr, arr)

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
    # print(start, end, ln, rn, dist)
    return np.arange(ln+1, rn+1) * dist


def get_elements_within_tolerant(arr:List[float], search_target:float, tolerant:float):
    """
    Extract all the center peaks index from a list of peak

    Args:
        peaks (list): a list of peaks
        spec_center_loc (float): center peak location that want to find
        tolerant (float): the maximum allow different between peak center and target center value

    Returns:
        list: center peak indexs
    """
    elements = []
    for i, e in enumerate(arr):
        err = abs(e - search_target)
        if err < tolerant:
            elements.append( (i, e, err) )
    return elements

def get_cloest_element(arr:List[float], search_target:float, tolerant:float):
    elements = get_elements_within_tolerant(arr, search_target, tolerant)
    return elements[ np.argmin( [e[2] for e in elements] ) ]

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
