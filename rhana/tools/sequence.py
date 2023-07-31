from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import numpy as np
import PIL
import json
from collections import defaultdict

import cv2
import skimage.exposure as exposure

from rhana.io.tokyo_u import RHEEDStreamReader
from rhana.labeler.unet import rle_decode_arr
from rhana.pattern import Rheed

from rhana.labeler.detector import CascadeMaskRCNNDetectorAndClassifier
from rhana.tracker.iou_tracker import IOUTracker, regions2detections, IOUMaskTracker
from rhana.pattern import RheedInstanceSegmentation
from rhana.periodicity import PeriodicityAnalyzer
from rhana.tracker.periodicity_tracker import PeriodicityTracker

import mmcv
from mmcv.transforms import Compose

import mmengine
from mmengine import Config
from mmengine.utils import track_iter_progress
# this import take a lot of time
# from mmengine import runner`

from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

from .restoration import suggest_restoration

model_folder = Path("../nn/cascade_maskrcnn_exps")

aux_detector = CascadeMaskRCNNDetectorAndClassifier(
    config_path = str(model_folder / "config_predict.py"),
    checkpoint_path = str(model_folder / "epoch_12.pth"),
    classifier_config_path = str(model_folder / "classifier/config.json"),
    classifier_checkpoint_path = str(model_folder / "classifier/model.pth"),
    device = "cuda:0",
)

def analyze_sequence(reader, beam_id, timestep=None, pred_folder="./pred", plot_fig=True, laue_xy=None, laue_angle=None):

    pred_folder = Path(pred_folder)
    pred_folder.mkdir(exist_ok=True)

    beams = reader.get_beams()
    frames = beams[beam_id]['frames']

    i = 0
    frame_id = frames[i]    

    result_collector = defaultdict(lambda : [])
    result_collector['cls_label'] = aux_detector.classifier_config
    db_track = None

    frame, frame_header = reader.read_frame(frame_id)
    rd = Rheed(frame)
    laue_xy, angle = suggest_restoration(rd)

    tracker = IOUTracker(t_min=100000, sigma_iou=0.4)

    analyzer = PeriodicityAnalyzer(
        tolerant=0.1, 
        abs_tolerant=20, 
        allow_discontinue=1
    )

    periodicity_tracker = PeriodicityTracker(
        analyzer = analyzer,
        disconnect_time= 10,
    )

    d_frame = timestep // beams[beam_id]['exposure']

    for i, frame_id in tqdm(enumerate(frames[::d_frame])):
        frame, frame_header = reader.read_frame(frame_id)
        rd = Rheed(frame)
        if laue_xy is not None and laue_angle is not None:
            rd = rd.rotate(
                angle=laue_angle,
                center=laue_xy, 
                inplace=False
            )
        rd = rd.min_max_scale(min_v=0, max_v=2**14)
        
        rd = rd.noise_based_correction(max_value=1, ratio=300, squash_ratio=0.8,)
        
        result, cls_result = aux_detector.predict(rd, )
        rdints = RheedInstanceSegmentation.from_mmdet(rd, result, aux_detector.model, auto_compute_regions=True)
        
        detections = regions2detections(rdints.regions, rdints.regions_label)
        region2tracks = tracker.update(detections, i)
        db_xy, db_r_i, db_track = rdints.get_direct_beam(method='top+tracker', direct_beam_label=3, tracker=tracker, track=db_track)
        
        rdints.get_regions_collapse()
        rdints.clean_collapse()
        rdints.fit_collapse_peaks(height=0.001, threshold=0.000, prominence=0.001)
        res = rdints.analyze_peaks_periodicity(center=db_xy[1], method='track', tracker=periodicity_tracker, track_frame_num=i)
        # print( len( periodicity_tracker.active_traces ), len( periodicity_tracker.traces ) )
        rdints.plot_peak_dist()
        
        result_collector['id'].append(frame_id)
        result_collector['header'].append(frame_header)
        result_collector['det'].append(result.detach().cpu())
        result_collector['cls'].append(cls_result.detach().cpu())
            
        if plot_fig:
            fig = aux_detector.plot_prediction(rd, result, cls_result)
            fig.savefig( pred_folder / f"{frame_id}.png", dpi=150 )
            
        # if i == 10: break

        return result_collector, tracker, periodicity_tracker

    
def plot_periodicity_dynamic(periodicity_tracker, result_collector):
    st = result_collector['header'][0].tstamp
    fig, ax = plt.subplots(dpi=150)
    for track in periodicity_tracker.traces:
        dists = []
        times = []
        for frame_id, pg in zip(track.frame_nums, track.periodicitygroups):
            dists.append(pg.avg_dist)
            times.append(  result_collector['header'][frame_id].tstamp )
        times= (np.array(times) - st) * 1e-9
        ax.plot(times, dists, label=f"{np.mean(dists):.2f}")
        ax.text(times[0], dists[0]+10, f"{np.mean(dists):.2f}", ha="right")
    # plt.legend()
    return fig, ax


def plot_classification(result_collector):
    fig, ax = plt.subplots(dpi=150)

    ts = []
    for h in result_collector['header']:
        ts.append(h.tstamp)
    ts = np.array(ts)
    ts = ts - ts[0]

    cls_all = np.concatenate(result_collector['cls'])
    for i in range(cls_all.shape[1]):
        plt.plot(ts, cls_all[:, i], label=aux_detector.classifier_classes[i])
    plt.legend()

    return fig, ax
