import numpy as np

def to_hwc(image):
    """swaps channel/band to last [height, width, channel]
    --------
    image: np.array of [c,h,w] format
    """
    return np.transpose(image,[1,2,0])


def to_chw(image):
    """swaps channel/band to first [channel, height, width]
    --------
    image: np.array of [h,w,c] format
    """
    return np.transpose(image,[2,0,1])

def _norm_plane(plane):
    """normalization only on 1 plane
    plane: np.array in format [h,w,c]
    """
    plane =  plane - np.nanmin(plane)
    return plane / np.nanmax(plane)

def normalize(plane, cfirst=False):
    """Scales pixel value to 0.0 - 1.0
    does not change image shape
    --------
    plane: np.array in format [h,w,c]
    cfirst: bool, use cfirst=1 if image format [c,h,w]
    """
    if cfirst: plane = to_hwc(plane)

    # 1ch image usually have only [w,h] or [w,h,1]
    if len(plane.shape) == 2 or plane.shape[-1] == 1:
        norm_image = _norm_plane(plane)
    else:
        norm_image = np.zeros_like(plane, dtype=np.float32)
        for i in range(plane.shape[-1]):
            norm_image[:,:,i] = _norm_plane(plane[:,:,i])

    # swap back to original ch first
    if cfirst: norm_image = to_chw(norm_image)

    return norm_image

def _stan_plane(plane):
    plane = plane - np.mean(plane)
    return plane / np.std(plane)

def standardize(plane, cfirst=False):
    """Scales pixel value to have mean=0.0 and std=1.0
    does not change image shape
    --------
    plane: np.array in format [h,w,c]
    cfirst: bool, use cfirst=1 if image format [c,h,w]
    """
    if cfirst: plane = to_hwc(plane)

    # 1ch image usually have only [w,h] or [w,h,1]
    if len(plane.shape) == 2 or plane.shape[-1] == 1:
        stan_image = _stan_plane(plane)
    else:
        stan_image = np.zeros(plane.shape)
        for i in range(plane.shape[-1]): 
            stan_image[:,:,i] = _stan_plane(plane[:,:,i])

    # swap back to original ch first
    if cfirst: stan_image = to_chw(stan_image)

    return stan_image

def hist_clip(image, thresh):
    """removes extreme low/high pixel values that are rare (low count)
    to better stretch the image.
    image: np.array
        will clip same pixel values for all channels
    thresh: int
        pixel count threshold. pixel values lower that have count < thresh will
        be clipped 
        first idx of hist where pixel counts exceeds thresh -> low_idx
        last idx of hist where pixel counts below thresh -> high_idx
    returns: clipped image 
    """
    bins = 256
    hist, bin = np.histogram(image, bins=bins,
                             range=(np.nanmin(image),np.nanmax(image)))
    # returns index where condition is true
    idx_limit = np.argwhere(hist>thresh)
    # how far away the lowest hist_idx > thresh is from idx of hist_peak
    peak = np.argmax(hist)
    dif = peak-idx_limit[0][0]
    if dif>int(bins/2):
        return hist_clip(image, thresh+80)
    else:
        low_idx = idx_limit[0][0]
        high_idx = idx_limit[-1][0]
        # get low and high pixel values
        low = bin[low_idx]
        high = bin[high_idx]
        # clip based on low and high limit
        return np.clip(image, a_min=low, a_max=high)