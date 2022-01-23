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

def get_data_region_idx(raster):
    """
    gets 4 corners of a raster, cutting out large portion of nodata region
        imperfect, as it still leaves tiny nodata edges due to angle
    np.argwhere() returns list of index. [[row,col],[row,col]]
        of every element that gets true condition
    """
    bin_mask = raster.read_masks(1)
    coords = np.argwhere(bin_mask==255)
    row0,col0 = coords.min(axis=0)  # find lowest row and col
    row1,col1 = coords.max(axis=0)  # find highest row and col
    return row0,row1,col0,col1