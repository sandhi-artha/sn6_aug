import numpy as np
import rasterio as rs
from shapely.geometry import box
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from rasterio import plot
"""TODO : create functions with less dependencies
"""


def get_tile_bounds(fns):
    """read raster's bounds with rasterio
    -------
    return: list of tile_bounds
    """
    return [list(rs.open(fn).bounds) for fn in fns]

def show_tile_bounds(bound_list, return_gdf=False):
    """plot or return gdf of boundaries
    hint: can also use gpd.GesSeries(bound) for a single bound
    """
    geometry = [box(*bound) for bound in bound_list]
    # create geodataframe
    d = {'geometry': geometry}
    gdf = gpd.GeoDataFrame(d, crs='epsg:32631')
    if return_gdf:
        return gdf
    else:
        gdf.plot(color='grey', edgecolor='black', alpha=0.8)

def create_patch(bounds, lw=1, c='r', ax=None):
    """create a rectangle with given bounds
    bounds = list
        [left, bot, right, top], default bounds used in rasterio
    lw = int
        linewidth, thickness of rectangle
    c = str
        color of line
    ax = matplotlib axis
        if not provided, returns the patch object
    """
    left, bot, right, top = bounds
    width = right-left
    height = top-bot
    rect = patches.Rectangle((left, bot), width, height,
                             linewidth=lw, edgecolor=c, facecolor='none')
    if ax:
        ax.add_patch(rect)
    else:
        return rect

def get_data_region_idx(raster, thresh=300):
    """
    leaves nodata trails in the edges
    np.argwhere() returns list of index. [[row,col],[row,col]]
        of every element that gets true condition
    thresh is used to prevent taking an edge which have ones in the middle, e.g.:
        [0 0 0 0 1 1 1 1 1 0 0 0 0]
        these usually have num of pixels less than 300
    """
    bin_mask = raster.read_masks(1)
    coords = np.argwhere(bin_mask==255)
    r0_cr,c0_cr = coords.min(axis=0)  # find lowest row and col
    r1_cr,c1_cr = coords.max(axis=0)  # find highest row and col

    # REFINING - first, crop to initial corners
    bin_mask = bin_mask[r0_cr:r1_cr, c0_cr:c1_cr]

    h_left = np.where(bin_mask[:,0]==255)  # left edge
    r0_left = np.min(h_left)
    r1_left = np.max(h_left)

    h_right = np.where(bin_mask[:,-1]==255)  # right edge
    r0_right = np.min(h_right)
    r1_right = np.max(h_right)

    h_left = r1_left-r0_left
    h_right = r1_right-r0_right

    # compare: take which is > thresh and minimum than the other
    if h_left<thresh:
        r0 = r0_right + r0_cr
        r1 = r1_right + r0_cr
    elif h_right<thresh:
        r0 = r0_left + r0_cr
        r1 = r1_left + r0_cr
    else:  # if both are > thresh, take the least one
        if h_left<h_right:
            r0 = r0_left + r0_cr
            r1 = r1_left + r0_cr
        else:
            r0 = r0_right + r0_cr
            r1 = r1_right + r0_cr

    # crop image
    bin_mask = bin_mask[r0:r1,:]

    w_top = np.where(bin_mask[0,:]==255)  # top edge
    c0_top = np.min(w_top)
    c1_top = np.max(w_top)

    w_bot = np.where(bin_mask[-1,:]==255)  # bot edge
    c0_bot = np.min(w_bot)
    c1_bot = np.max(w_bot)

    w_top = c1_top-c0_top
    w_bot = c1_bot-c0_bot

    # compare: take which is > thresh and minimum than the other
    if w_top<thresh:
        c0 = c0_bot + c0_cr
        c1 = c1_bot + c0_cr
    elif w_bot<thresh:
        c0 = c0_top + c0_cr
        c1 = c1_top + c0_cr
    else:  # if both are > thresh, take the least one
        if w_top<w_bot:
            c0 = c0_top + c0_cr
            c1 = c1_top + c0_cr
        else:
            c0 = c0_bot + c0_cr
            c1 = c1_bot + c0_cr

    assert r1-r0 > thresh, 'row less than thresh'
    assert c1-c0 > thresh, 'col less than thresh'

    return r0,r1,c0,c1