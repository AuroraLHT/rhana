import copy
import numpy as np

def regions2detections(regions):    
     return [ {'bbox':r.bbox, 'image':r.image, 'area':r.area, 'region_id':i} for i, r in enumerate(regions) ]

def overlap_rect(bbox1, bbox2):
    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

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
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
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

class IOUTracker:
    def __init__(self, t_min=2, sigma_iou=0.5):
        self._tracks_active = []
        self._tracks_finished = []
        self.t_min= t_min
        self.sigma_iou= sigma_iou
        self._track_id = 0
        self.region2track = {}
        self.track2region = {}
    
    def track_id(self):
        old_id = self._track_id
        self._track_id += 1
        return old_id
    
    @property
    def tracks(self):
        tracks = self._tracks_finished + self._tracks_active
        return tracks
    
    def update(self, detections, frame_num):
        detections = copy.copy(detections)
        updated_tracks = []
        region2track = {}
        
        for track in self._tracks_active:
            if len(detections) > 0:
                # get det with highest iou
                best_match = max(detections, key=lambda x: iou_bbox(track['bboxes'][-1], x['bbox']))
                if iou_bbox(track['bboxes'][-1], best_match['bbox']) >= self.sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['region_ids'].append(best_match['region_id'])
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
            new_tracks.append({'bboxes': [det['bbox']], 'region_ids':[det['region_id']], 'start_frame': frame_num, 'track_id' : track_id})
            region2track[det['region_id']] = track_id
            
        self._tracks_active = updated_tracks + new_tracks

        self.region2track = region2track
        self.track2region = {v:k for k, v in self.region2track.items()}
        return region2track
        
class IOUMaskTracker:
    def __init__(self, t_min=2, sigma_iou=0.5):
        self._tracks_active = []
        self._tracks_finished = []
        self.t_min= t_min
        self.sigma_iou= sigma_iou
        self._track_id = 0
        self.region2track = {}
        self.track2region = {}
    
    def track_id(self):
        old_id = self._track_id
        self._track_id += 1
        return old_id
    
    @property
    def tracks(self):
        tracks = self._tracks_finished + self._tracks_active
        return tracks
    
    def update(self, detections, frame_num):
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
        
    