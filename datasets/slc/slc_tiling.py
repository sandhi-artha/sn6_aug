import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from timebudget import timebudget
import geopandas as gpd
from shapely.geometry import box
import rasterio as rs

from datasets.datagen import get_ts_orient
from datasets.slc.slc_cfg import slc_cfg
import solaris.raster_tile as raster_tile
import solaris.vector_tile as vector_tile
from solaris.vector_mask import mask_to_poly_geojson
from solaris.core import save_empty_geojson


def get_label_gdf(split, in_dir):
    """split: str. 'train', 'val', 'test'
    in_dir: str. root path to folder containing .geojson
    """
    fn = f'SN6_AOI_11_Rotterdam_Buildings_GT20sqm-{split.capitalize()}.geojson'
    return gpd.read_file(os.path.join(in_dir, fn))

def get_labels_bounds(label_dir):
    """returns [labels, bounds]
    each is dictionary containing 'train','val','test' split
    labels : geodataframe of building footprints
    bounds : shapely. boundary/extent where labels are available
    """
    labels = {
        'train' : get_label_gdf('train', label_dir),
        'val'   : get_label_gdf('val', label_dir),
        'test'  : get_label_gdf('test', label_dir)
    }

    # give paddings to ensure same raster and vector coverage
    # when using total_bounds, the train boundaries take square coverage
    #   so raster at bot right corner has image but with labels cutoff
    #   +- 100 for val is not enough (blame the tiling algo)
    train_bounds = labels['train'].total_bounds
    train_bounds[2] -= 400

    val_bounds = labels['val'].total_bounds
    val_bounds[0] += 400
    val_bounds[2] -= 400

    test_bounds = labels['test'].total_bounds
    test_bounds[0] += 400
    test_bounds[2] -= 400

    bounds = {
        'train' : box(*train_bounds),
        'val'   : box(*val_bounds),
        'test'  : box(*test_bounds)
    }

    return labels, bounds


def save_clip_vector_mask(raster_path, vector_path):
    """reads the mask from raster, converts to polygon and clips the vector using it
        when input vector is empty, will return an empty gdf
    raster_path : str
        path to .tif raster that have mask
    vector : geopandas gdf
    """
    raster = rs.open(raster_path)
    mask = raster.read_masks([1])
    mask_gdf = mask_to_poly_geojson(
        mask[0], reference_im=raster_path,
        do_transform=True
    )

    # read vector and clip it
    vector = gpd.read_file(vector_path)
    vector_fix = vector.clip(mask_gdf)

    if vector_fix.shape[0] == 0:
        save_empty_geojson(vector_path, crs=raster.crs)
    else:
        vector_fix.to_file(vector_path, driver='GeoJSON')

    raster.close()  # close the opened dataset

def raster_vector_tiling(cfg, timestamp, orient, slc_path):
    """create raster and vector tiles using Tiler from solaris
    """
    labels, bounds = get_labels_bounds(cfg["label_dir"])

    raster_save_path = os.path.join(cfg['out_dir'],'raster')
    vector_save_path = os.path.join(cfg['out_dir'],'vector')
    
    for split in cfg['splits']:
        fn = '{}_{}_o{}_{}_{}_s{}'.format(
            cfg["project"],
            timestamp,
            orient,
            cfg["name"],
            split,
            cfg["stride"]
        )

        # tile the raster
        raster_tiler = raster_tile.RasterTiler(dest_dir=raster_save_path, 
                                       src_tile_size=(900, 900),
                                       aoi_boundary=bounds[split],
                                       verbose=cfg["verbose"],
                                       stride=(cfg["stride"],cfg["stride"]))
        
        raster_tiler.tile(slc_path, dest_fname_base=fn, nodata_threshold=0.5)

        # use created tiles for vector tiling
        vector_tiler = vector_tile.VectorTiler(dest_dir=vector_save_path,
                                               super_verbose=cfg["verbose"])
        
        vector_tiler.tile(labels[split], tile_bounds=raster_tiler.tile_bounds,
                          split_multi_geoms=False, dest_fname_base=fn)
        
        # add vector fix
        for i in range(len(raster_tiler.tile_paths)):
            save_clip_vector_mask(
                raster_tiler.tile_paths[i], vector_tiler.tile_paths[i])






if __name__ == '__main__':
    timestamps = get_ts_orient(slc_cfg['base_dir'], slc_cfg['orient'])

    # tile raster and vector
    with timebudget('TILING'):
        for i,timestamp in enumerate(timestamps):
            print(f'tiling raster {timestamp}.. {i} of {len(timestamps)}')

            out_fn = f'HH_{timestamp}.tif'
            proc_slc_path = os.path.join(slc_cfg["out_dir"], out_fn)  # slc.tif path

            raster_vector_tiling(slc_cfg, timestamp, slc_cfg['orient'], proc_slc_path)
    