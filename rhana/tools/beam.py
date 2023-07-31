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

def get_deposition_window(metas):
    dep_total = metas['Deposition-Total (Count)'].values
    dep_fire = metas['Deposition-Fired (Count)'].values
    dep_rate = metas['Deposition-Rate (Count)'].values
    dep_req = metas['Deposition-Requested (Count)'].values

    total_start = dep_total[-1] - dep_req[-1]
    total_end = dep_total[-1]

    fire_start = dep_fire[-1] - dep_req[-1] + 1

    total_starts = np.where(total_start==dep_total)[0]
    fire_starts = np.where(fire_start==dep_fire)[0]

    if len(total_starts):
        total_start_pos = total_starts[-1]
    else:
        total_start_pos = -1

    if len(fire_starts):
        fire_start_pos = fire_starts[-1]
    else:
        fire_start_pos = -1

    if total_start_pos != -1 and fire_start_pos != -1 and total_start_pos < fire_start_pos:
        start_pos = total_start_pos
    else:
        start_pos = None

    total_ends = np.where(total_end==dep_total)[0]
    if len(total_ends):
        end_pos = total_ends[0]
    else:
        end_pos = None

    return start_pos, end_pos