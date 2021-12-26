import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .proc import to_chw, to_hwc

def show_stats(image, cfirst=False):
    """prints statistic for an image
    --------
    image: np.array
    cfirst: bool, use cfirst=True if image format is [c,w,h]
    """
    print(f'shape: {image.shape}')
    if not cfirst:
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        else:
            image = to_chw(image)
    props = pd.DataFrame({
        'min': np.nanmin(image, (1,2)),
        'max': np.nanmax(image, (1,2)),
        'mean': np.nanmean(image, (1,2)),
        'std': np.nanstd(image, (1,2)),
        'pos': np.count_nonzero(np.nan_to_num(image, nan=-1.)>0, (1,2)),
        'zero': np.count_nonzero(image==0, (1,2)),
        'neg': np.count_nonzero(np.nan_to_num(image, nan=1.)<0, (1,2)),
        'nan': np.count_nonzero(np.isnan(image), (1,2)),
    })
    print(props)


def show_hist(image, ax=None, cfirst=False, num_bins=256, start=None, end=None):
    """creates histogram for 1-4 channel image
    --------
    image: np.array with [h,w,c]
    ax: plt.axis. Use f,ax = plt.subplots() and feed the ax here
    cfirst: bool. Use cfirst=1 if image is [c,h,w] format
    num_bins: int. default=256
    start: int or float (match image type). If not provided then np.min(image)
    end: default: np.max(image)
    """
    color = ['r','g','b','k']
    if start == None: start = np.min(image)
    if end ==None: end = np.max(image)
    
    # if channel first, make it channel last
    if cfirst: image = to_hwc(image)
    # if ax not specified, create one
    if not ax: f,ax = plt.subplots(1,1)

    # 1ch image usually have only [w,h] or [w,h,1]
    if len(image.shape)==2 or image.shape[-1]==1:
        ax.hist(image.ravel(), num_bins, [start,end])
    else:
        for i in range(image.shape[-1]):
            ax.hist(image[:,:,i].ravel(), num_bins, [start,end],
                    color=color[i], histtype='step', alpha=0.6)