import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

@dataclass
class PeriodicityTrace:
    id : int
    frame_nums : List
    periodicitygroups : List

    def get_last_element(self):
        return self.periodicitygroups[-1], self.frame_nums[-1]

    def add(self, periodicity, frame_num):
        self.frame_nums.append(frame_num)
        self.periodicitygroups.append(periodicity)


class PeriodicityTracker:
    def __init__(self, analyzer, disconnect_time=10, curr_frame_num=-1):
        self.analyzer = analyzer
        self.disconnect_time = disconnect_time
        self.curr_frame_num = curr_frame_num
        self.clear_history()

    def clear_history(self):
        self.finished_traces = []
        self.active_traces = []
        self.periodicitygroups = []
        self._next_trace_id = 0

    @property
    def traces(self):
        return self.active_traces + self.finished_traces

    @property
    def next_trace_id(self):
        self._next_trace_id += 1
        return self._next_trace_id
    
    def initiate_new_traces(self, periodicitygroups, frame_num):
        for p in periodicitygroups:
            trace = PeriodicityTrace(id=self.next_trace_id, frame_nums=[frame_num], periodicitygroups=[p])
            self.active_traces.append(trace)

    def track_with_traces(self, arr, center, grid_min, grid_max, frame_num):
        if len(self.active_traces) == 0:
            matched_mask = np.zeros_like(arr, dtype=bool)
        else:        
            periodicities = [t.get_last_element()[0] for t in self.active_traces]
            try:
                mpgs, mres, matched_mask = self.analyzer.match_periodicity(
                    arr = arr, 
                    periodicities = periodicities,
                    center = center,
                    grid_min = grid_min,
                    grid_max = grid_max,
                )

                active_traces = []
                for i, pg in enumerate(mpgs):
                    trace = self.active_traces[i]
                    if pg is None:
                        pass                    
                    else:
                        trace.add(pg, frame_num)
                        
                    if (frame_num - trace.frame_nums[-1] > self.disconnect_time):                    
                        self.finished_traces.append(trace)
                    else:
                        active_traces.append(trace)

                self.active_traces = active_traces

            except Exception as e:
                print(f"Periodicity Match Error: {e}")
                matched_mask = np.zeros_like(arr, dtype=bool)
       
        return matched_mask

    def get_current_periodicities(self, frame_num):
        pgs = []
        for trace in self.active_traces:
            t_pg, t_fn = trace.get_last_element()
            if t_fn == frame_num: pgs.append(t_pg)
        return pgs
    
    def update(self, arr, center, grid_min, grid_max, frame_num=None):
        # predict and match with previous traces
        if frame_num is None:
            frame_num = self.curr_frame_num + 1
            
        matched_mask = self.track_with_traces(arr, center, grid_min, grid_max, frame_num)
      
        # generate new traces
        pgs = self.analyzer.analyze(
            arr=arr,
            center=center,
            grid_min=grid_min,
            grid_max=grid_max,
            arr_mask = matched_mask
        )
        
        self.initiate_new_traces(pgs, frame_num)

        # update tracker state
        self.curr_frame_num = frame_num
        
        return self.get_current_periodicities(frame_num)