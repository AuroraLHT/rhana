import copy
import numpy as np
from dataclasses import dataclass

from typing import List, Dict


def regions2detections(regions, regions_label=None):
    """Convert region from skimage to detections format used object detection

    Args:
        regions (List [RegionProperties]): list of RegionProperties that needed to be converted

    Returns:
        List: a list of detections 
    """
    if regions_label is None: regions_label = [0] * len(regions)
    return [ {'bbox':r.bbox, 'image':r.image, 'area':r.area, 'id':i, 'label':rl} for i, (r, rl) in enumerate(zip(regions, regions_label)) ]

def overlap_rect(bbox1, bbox2):
    """Compute overlapped corner between two bounding boxes
    
    Bounding box style:
        (Upper Left x, Upper Left y, Lower right x, Lower right y)

    Args:
        bbox1 (Array): bounding box 1
        bbox2 (Array): bounding box 1

    Returns:
        float: Upper Left overlap x
        float: Upper Left overlap y
        float: Lower Right overlap x
        float: Lower Right overlap y
        
    """
    try:
        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2
    except Exception as e:
        raise ValueError(f"Could not unpack bbox1 and bbox2. bbox1: {bbox1}. bbox2: {bbox2}" )

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    return overlap_x0, overlap_y0, overlap_x1, overlap_y1

def iou_bbox(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    
    Args:
        bbox1 (Array): bounding box in the format of x1,y1,x2,y2.
        bbox2 (Array): bounding box in the format of x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    overlap_x0, overlap_y0, overlap_x1, overlap_y1 = overlap_rect(bbox1, bbox2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    size_1 = float(x1_1 - x0_1) * float(y1_1 - y0_1)
    size_2 = float(x1_2 - x0_2) * float(y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def iou(bbox1, bbox2, image1, image2, area1, area2):
    """Compute the intersection-over-union of two instance segmentation mask.
    To clarify, image1 and image2 is the two bindary masks. area1 and area2 are the
    areas of mask bounded by the bboxs. 

    Args:
        bbox1 (Array): bounding box in the format of x1,y1,x2,y2.
        bbox2 (Array): bounding box in the format of x1,y1,x2,y2.
        image1 (Array): a bindary mask
        image2 (Array): a bindary mask
        area1 (float): area of the mask enclosed by the bbox
        area2 (float): area of the mask enclosed by the bbox

    Returns:
        float: iou value
    """
    bbox_overlap = iou_bbox(bbox1, bbox2)
    if bbox_overlap > 0 :
        overlap_x0, overlap_y0, overlap_x1, overlap_y1 = overlap_rect(bbox1, bbox2)
        sx_1 = int(overlap_x0 - bbox1[0])
        sx_2 = int(overlap_x0 - bbox2[0])
        sy_1 = int(overlap_y0 - bbox1[1])
        sy_2 = int(overlap_y0 - bbox2[1])

        ex_1 = int(overlap_x1 - bbox1[0])
        ex_2 = int(overlap_x1 - bbox2[0])
        ey_1 = int(overlap_y1 - bbox1[1])
        ey_2 = int(overlap_y1 - bbox2[1])

        overlap_area = np.sum(np.logical_and(image1[sx_1:ex_1, sy_1:ey_1], image2[sx_2:ex_2, sy_2:ey_2]))
        return overlap_area / (area1 + area2 - overlap_area)
    else:
        return 0


@dataclass
class Track:
    id : int
    det_bboxes: List
    det_ids: List
    det_framenums : List    
    det_label : int
    start_frame : int

    @classmethod
    def from_detection(cls, id, detection, start_frame):
        return cls(
            id = id,
            det_bboxes = [detection['bbox']],
            det_ids = [detection['id']],
            det_framenums = [start_frame],
            det_label = detection['label'],
            start_frame = start_frame
        )
    
    def add(self, detection, frame_num):
        self.det_bboxes.append(detection['bbox'])
        self.det_ids.append(detection['id'])
        self.det_framenums.append(frame_num)


class IOUTracker:
    """A simple tracker that trace the movement of a particular object by connective the
    bounding boxes with the highest IOU value between two consecutive frames.
    """
    def __init__(self, t_min=2, sigma_iou=0.5):
        """IOUTracker initializer

        Args:
            t_min (int, optional): Number of frames that required to set the trace to be terminate. Defaults to 2.
            sigma_iou (float, optional): the minimum iou value for two bounding boxes to be treated as from the same trace. Defaults to 0.5.
        """
        self._tracks_active = []
        self._tracks_finished = []
        self.t_min= t_min
        self.sigma_iou= sigma_iou
        self._track_id = 0
        self.det2track = {}
        self.track2det = {}
    
    def next_track_id(self):
        """Generate a new track_id

        Returns:
            int: a new track id
        """
        old_id = self._track_id
        self._track_id += 1
        return old_id
    
    @property
    def tracks(self):
        """Get both active and terminated traces.

        Returns:
            list: all the discovered trace
        """
        tracks = self._tracks_finished + self._tracks_active
        return tracks
    
    def update(self, detections, frame_num):
        """Use the detections extracted from the current frame to update
        the tracker. Use 'regions2detections' to convert region to this format.

        Args:
            detections (dict): object detection style.
            frame_num (int): frame number of the current frame

        Returns:
            dict: a mapping between regions in the current and trace stored in the tracker.
        """
        detections = copy.deepcopy(detections)
        updated_tracks = []
        det2track = {}
        
        # TODO change this to linear assignment
        for track in self._tracks_active:
            if len(detections) > 0:
                # get det with highest iou and also has the same class label
                best_match = max(detections, key=lambda x: iou_bbox(track.det_bboxes[-1], x['bbox']) if x['label'] == track.det_label else 0)

                if iou_bbox(track.det_bboxes[-1], best_match['bbox']) >= self.sigma_iou:
                    # track.det_bboxes.append(best_match['bbox'])
                    # track.det_ids.append(best_match['id'])
                    # track.det_frames.append(frame_num)
                    # we do not have score here
                    # track['max_score'] = max(track['max_score'], best_match['score'])
                    track.add(best_match, frame_num)

                    updated_tracks.append(track)
                    
                    det2track[best_match['id']] = track.id
                    
                    # remove from best matching detection from detections
                    del detections[detections.index(best_match)]
                    

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                # if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                                
                if frame_num - track.start_frame >= self.t_min:
                    # we don't have score again and wait long until the t_min frames pass
                    self._tracks_finished.append(track)
                else:
                    # still keep this track updating
                    updated_tracks.append(track)

        # create new tracks from unselected region
        # new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num} for det in dets]
        new_tracks = []
        for det in detections:
            new_track = Track.from_detection(self.next_track_id(), detection=det, start_frame=frame_num)
            new_tracks.append(new_track)
            # new_tracks.append({'bboxes': [det['bbox']], 'region_ids':[det['region_id']], 'start_frame': frame_num, 'track_id' : track_id})
            det2track[det['id']] = new_track.id
            
        self._tracks_active = updated_tracks + new_tracks

        self.det2track = det2track
        self.track2det = {v:k for k, v in self.det2track.items()}
        return det2track


class IOUMaskTracker:
    """ A simple tracker that trace the movement of a particular object by connective the
    bounding boxes with the highest IOU value between two consecutive frames.
    Similar to IOUTracker but use mask with bounding box to compute the IOU value.
    """
    def __init__(self, t_min=2, sigma_iou=0.5):
        """IOUMaskTracker initializer

        Args:
            t_min (int, optional): Number of frames that required to set the trace to be terminate. Defaults to 2.
            sigma_iou (float, optional): the minimum iou value for two bounding boxes to be treated as from the same trace. Defaults to 0.5.
        """        
        
        self._tracks_active = []
        self._tracks_finished = []
        self.t_min= t_min
        self.sigma_iou= sigma_iou
        self._track_id = 0
        self.region2track = {}
        self.track2region = {}
    
    def track_id(self):
        """Get both active and terminated traces.

        Returns:
            list: all the discovered trace
        """       

        old_id = self._track_id
        self._track_id += 1
        return old_id
    
    @property
    def tracks(self):
        """Get both active and terminated traces.

        Returns:
            list: all the discovered trace
        """        
        tracks = self._tracks_finished + self._tracks_active
        return tracks
    
    def update(self, detections, frame_num):
        """Use the detections extracted from the current frame to update
        the tracker. Use 'regions2detections' to convert region to this format.

        Args:
            detections (dict): object detection style.
            frame_num (int): frame number of the current frame

        Returns:
            dict: a mapping between regions in the current and trace stored in the tracker.
        """        
        
        detections = copy.copy(detections)
        updated_tracks = []
        region2track = {}
        
        for track in self._tracks_active:
            if len(detections) > 0:
                # get det with highest iou
                best_match = max(detections, key=lambda x: iou(track['bboxes'][-1], x['bbox'], track['image'], x['image'], track['area'], x['area']))
                if iou(track['bboxes'][-1], best_match['bbox'], track['image'], best_match['image'], track['area'], best_match['area']) >= self.sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['region_ids'].append(best_match['region_id'])
                    track['image'] = best_match['image']
                    track['area'] = best_match['area']
                    # we do not have score here
                    # track['max_score'] = max(track['max_score'], best_match['score'])

                    updated_tracks.append(track)
                    
                    region2track[best_match['region_id']] = track['track_id']
                    
                    # remove from best matching detection from detections
                    del detections[detections.index(best_match)]
                    

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                # if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                                
                if frame_num - track['start_frame'] >= self.t_min:
                    # we don't have score again and wait long until the t_min frames pass
                    self._tracks_finished.append(track)
                else:
                    # still keep this track updating
                    updated_tracks.append(track)

        # create new tracks from unselected region
        # new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num} for det in dets]
        new_tracks = []
        for det in detections:
            track_id = self.track_id()
            new_tracks.append({'bboxes': [det['bbox']], 'region_ids':[det['region_id']], 'image':det['image'], 'area':det['area'], 'start_frame': frame_num, 'track_id' : track_id})
            region2track[det['region_id']] = track_id
            
        self._tracks_active = updated_tracks + new_tracks

        self.region2track = region2track
        self.track2region = {v:k for k, v in self.region2track.items()}
        return region2track