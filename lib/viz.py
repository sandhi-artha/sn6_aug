import numpy as np
import pandas as pd
import seaborn as sns
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


### PLOT ###
def show_example(img, mask):
    f,ax = plt.subplots(1,2,figsize=(10,5))
    if img.shape[-1] == 1:
        cmap = 'gray'
    else:
        cmap = None
    ax[0].imshow(img, cmap=cmap)
    if len(mask.shape)==3:
        mask = np.squeeze(mask, axis=-1)
    ax[1].imshow(mask, cmap='gray')
    plt.show()

def plot_metrics(history):
    sns.set(style='whitegrid')
    metric_list = [m for m in list(history.keys()) if m is not 'lr']
    size = len(metric_list)//2  # adjust vertical space to fit all metrics
    fig, axes = plt.subplots(size, 1, sharex='col', figsize=(20, size * 4))
    if size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for index in range(len(metric_list)//2):
        metric_name = metric_list[index]
        val_metric_name = metric_list[index+size]
        axes[index].plot(history[metric_name], label='Train %s' % metric_name)
        axes[index].plot(history[val_metric_name], label='Validation %s' % metric_name)
        axes[index].legend(loc='best', fontsize=16)
        axes[index].set_title(metric_name)
        if 'loss' in metric_name:
            axes[index].axvline(np.argmin(history[metric_name]), linestyle='dashed')
            axes[index].axvline(np.argmin(history[val_metric_name]), linestyle='dashed', color='orange')
        else:
            axes[index].axvline(np.argmax(history[metric_name]), linestyle='dashed')
            axes[index].axvline(np.argmax(history[val_metric_name]), linestyle='dashed', color='orange')

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.show()
    sns.set_theme()
    
    # print model performance
    bm_idx = np.argmin(history['val_loss'])
    print(f'best model at epoch: {bm_idx}')
    v_loss = history['val_loss'][bm_idx]
    v_iou = history["val_iou_score"][bm_idx]
    v_f1 = history["val_f1-score"][bm_idx]

    loss = history['loss'][bm_idx]
    iou = history["iou_score"][bm_idx]
    f1 = history["f1-score"][bm_idx]


    print(f'val loss: {v_loss:.4f}, val iou: {v_iou:.4f}, val f1: {v_f1:.4f}')
    print(f'loss: {loss:.4f}, iou: {iou:.4f}, f1: {f1:.4f}')
    print(f'best val IoU: {np.max(history["val_iou_score"])}')