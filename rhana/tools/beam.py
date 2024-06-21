import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_metas(reader, frames):
    metas = []
    for fid in frames:
        header = reader.read_frame_header(fid)
        metas.append(header.meta)
    metas = pd.DataFrame(metas)
    return metas

def plot_metas(metas):
    nrows = int(np.ceil(len(metas.columns)/3))
    fig, axs = plt.subplots(ncols=3, nrows=nrows, figsize=(15, 4*nrows))
    axs = axs.flatten()
    for i, c in enumerate(metas.columns):
        axs[i].plot( metas[c], label=c)
        axs[i].set_title(c)
        # axs[i].legend(loc="upper left")
    plt.show()


def get_deposition_window(metas, verbose):
    dep_total = metas['Deposition-Total (Count)'].values
    dep_fire = metas['Deposition-Fired (Count)'].values
    dep_rate = metas['Deposition-Rate (Count)'].values
    dep_req = metas['Deposition-Requested (Count)'].values

    total_zero_mask = dep_total == 0
    
    total_end = dep_total[dep_total > 0][-1]
    total_start = total_end - dep_req[-1]
    
    fire_start = max(dep_fire[-1] - dep_req[-1], dep_fire.min())
    fire_end = dep_fire[-1]

    if verbose:
        print("dep total start, end:", total_start, total_end)
        print("dep fire start, end:", fire_start, fire_end)
    
    total_starts = np.where( (total_start<dep_total) & ~total_zero_mask )[0]
    fire_starts = np.where(fire_start<dep_fire)[0]

    if len(total_starts):
        total_start_pos = total_starts[dep_total[total_starts].argmin()] - 1
        total_start_pos = max(0, total_start_pos)
    else:
        total_start_pos = -1

    if len(fire_starts):
        fire_start_pos = fire_starts[dep_fire[fire_starts].argmin()] - 1
        fire_start_pos = max(0, total_start_pos)
        
    else:
        fire_start_pos = -1
    if total_start_pos != -1 and fire_start_pos != -1:
        start_pos = int( min(total_start_pos, fire_start_pos) )
    else:
        start_pos = None

    total_ends = np.where(total_end==dep_total)[0]
    total_ends = total_ends[total_ends>start_pos]

    fire_ends = np.where( fire_end==dep_fire )[0]
    fire_ends = fire_ends[fire_ends>start_pos]
    
    if len(total_ends):
        total_end_pos = int(total_ends[0])
    else:
        total_end_pos = -1
    
    if len(fire_ends):
        fire_end_pos = int(fire_ends[0])
    else:
        fire_end_pos = -1

    end_pos = max(fire_end_pos, total_end_pos)
    end_pos = None if end_pos == -1 else end_pos

    return start_pos, end_pos