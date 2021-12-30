# vizualize batched dataset, plot history, plot prediction results
import os
import json
import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import rasterio as rs
from rasterio import features
from shapely.geometry import shape
import geopandas as gpd
import wandb
from affine import Affine


from lib.proc import to_hwc, normalize
from model import load_pretrained_model
from dataloader import get_config_wandb
from cfg import ev_cfg






### EVALUATION ###
# for binary you need a threshold value to differ what's category 0 and what's 1
def create_binary_mask(pred_mask):
    thresh = 0.5
    return tf.where(pred_mask>=thresh, 1, 0)

def display_img(display_list):
    title = ['Input Tile', 'True Maks', 'Predicted Mask']
    plt.figure(figsize=(18,8))
    
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 1:
            cmap = 'gray'
        else:
            cmap = None
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i], fontsize=24)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap=cmap)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def show_predictions(model, ds_fn, n_show=4, shuffle=False, num_pass=0):
    """ create new dataset, shuffle(250), batch(n_show), predict and display with fn
    """
    dataset = get_preview_dataset(ds_fn, n_show, shuffle)
    
    for img,mask,fn in dataset.skip(num_pass).take(1):
        pred_mask = model(img)
        pred_mask = create_binary_mask(pred_mask)
        for i in range(n_show):
            print(fn[i].numpy().decode())
            display_img([img[i], mask[i], pred_mask[i]])


def _get_raster_fn(raster_fp):
    bn = os.path.basename(raster_fp)
    return bn.split('.')[0]

def _get_ts_tile_id(fp):
    bn = os.path.basename(fp)
    bn_list = bn.split('_')
    return ('_').join(bn_list[2:4]), bn_list[-1].split('.')[0]

def load_val_image(raster_fp):
    """loads a single SAR raster and perform same ops as if loading from tfrec
    output is [1,640,640,1]
    """
    raster = rs.open(raster_fp)
    image = raster.read(indexes=[1], masked=True)
    image = to_hwc(image)
    image = normalize(image)
    raster.close()

    image = tf.cast(image*(2**8 - 1), tf.uint8)
    image = tf.cast(image, tf.float32)/255.0
    image = tf.image.resize(image, (640,640))
    image = tf.expand_dims(image, axis=0)
    return image

def save_pred_vector(preds, raster_fp, out_dir=None):
    """create polygon from prediction mask
    if out_dir==None, geojson won't be saved
    """
    preds = create_binary_mask(preds)
    preds = tf.image.resize(preds, (900,900))
    fn = _get_raster_fn(raster_fp)
    if out_dir is None:
        pred_path = None
    else:
        pred_path = f'{out_dir}/{fn}_pred.geojson'
    pred_gdf = mask_to_poly_geojson(
        preds.numpy()[0], reference_im=raster_fp,
        output_path=pred_path, simplify=True,
        do_transform=True)
    
    return pred_gdf

def save_combine_gdf(raster_fps, save_path):
    """from raster file_paths, read all vector file_paths
    combine into a single geojson and save
    """
    gdf_list = []
    for raster_fp in raster_fps:
        vector_fp = raster_fp.replace('raster','vector')
        vector_fp = vector_fp.replace('.tif', '.geojson')
        gdf_list.append(gpd.read_file(vector_fp))

    crs = gdf_list[0].crs
    gdf_comb = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=crs)

    gdf_comb.to_file(save_path, driver='GeoJSON')


def save_empty_geojson(path, crs):
    empty_geojson_dict = {
        "type": "FeatureCollection",
        "crs":
        {
            "type": "name",
            "properties":
            {
                "name": "urn:ogc:def:crs:EPSG:{}".format(crs.to_epsg())
            }
        },
        "features":
        []
    }
    with open(path, 'w') as f:
        json.dump(empty_geojson_dict, f)
        f.close()


def mask_to_poly_geojson(bin_arr, reference_im=None,
                         output_path=None, output_type='geojson', min_area=40,
                         bg_threshold=0, do_transform=None, simplify=False,
                         tolerance=0.5, **kwargs):
    """Get polygons from an image mask.

    Arguments
    ---------
    pred_arr : :class:`numpy.ndarray`
        
    reference_im : str, optional
        The path to a reference geotiff to use for georeferencing the polygons
        in the mask. Required if saving to a GeoJSON (see the ``output_type``
        argument), otherwise only required if ``do_transform=True``.
    output_path : str, optional
        Path to save the output file to. If not provided, no file is saved.
    output_type : ``'csv'`` or ``'geojson'``, optional
        If ``output_path`` is provided, this argument defines what type of file
        will be generated - a CSV (``output_type='csv'``) or a geojson
        (``output_type='geojson'``).
    min_area : int, optional
        The minimum area of a polygon to retain. Filtering is done AFTER
        any coordinate transformation, and therefore will be in destination
        units.
    bg_threshold : int, optional
        The cutoff in ``bin_arr`` that denotes background (non-object).
        Defaults to ``0``.
    simplify : bool, optional
        If ``True``, will use the Douglas-Peucker algorithm to simplify edges,
        saving memory and processing time later. Defaults to ``False``.
    tolerance : float, optional
        The tolerance value to use for simplification with the Douglas-Peucker
        algorithm. Defaults to ``0.5``. Only has an effect if
        ``simplify=True``.

    Returns
    -------
    gdf : :class:`geopandas.GeoDataFrame`
        A GeoDataFrame of polygons.

    """

    if do_transform and reference_im is None:
        raise ValueError(
            'Coordinate transformation requires a reference image.')

    if do_transform:
        with rs.open(reference_im) as ref:
            transform = ref.transform
            crs = ref.crs
            ref.close()
    else:
        transform = Affine(1, 0, 0, 0, 1, 0)  # identity transform
        crs = rs.crs.CRS()

    mask = bin_arr > bg_threshold
    mask = mask.astype('uint8')

    polygon_generator = features.shapes(bin_arr,
                                        transform=transform,
                                        mask=mask)
    polygons = []
    values = []  # pixel values for the polygon in mask_arr
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(0.0)
        if p.area >= min_area:
            polygons.append(shape(polygon).buffer(0.0))
            values.append(value)

    polygon_gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values},
                                   crs=crs.to_wkt())
    if simplify:
        polygon_gdf['geometry'] = polygon_gdf['geometry'].apply(
            lambda x: x.simplify(tolerance=tolerance)
        )
    # save output files
    if output_path is not None:
        if output_type.lower() == 'geojson':
            if len(polygon_gdf) > 0:
                polygon_gdf.to_file(output_path, driver='GeoJSON')
            else:
                save_empty_geojson(output_path, polygon_gdf.crs.to_epsg())
        elif output_type.lower() == 'csv':
            polygon_gdf.to_csv(output_path, index=False)

    return polygon_gdf



if __name__ =='__main__':
    if os.path.isfile('ev_cfg.json'):
        print('using Kaggle evaluation config')
        with open('ev_cfg.json', 'r') as fp:
            ev_cfg = json.load(fp)
    else:
        print('using saved config')

    # read all raster paths
    raster_fps = glob.glob(f'{ev_cfg["base_dir"]}/raster/*.tif')

    # load pre-trained model and its config from wandb
    cfg = get_config_wandb(ev_cfg['run_path'])
    print(f'loading model {cfg["NAME"]}')
    model_file = wandb.restore('model-best.h5', run_path=ev_cfg['run_path'])
    model_path = model_file.name
    model_file.close()
    model = load_pretrained_model(model_path)

    pred_gdf_path = os.path.join(ev_cfg['save_dir'], f'{cfg["NAME"]}_pred.geojson')

    # for debug
    # model = load_pretrained_model('model-best.h5')
    # pred_gdf_path = 'comb_pred.geojson'
    raster_fps = raster_fps[:15]
    
    pred_gdf_list = []
    tot_raster = len(raster_fps)

    t_str = time.time()
    for i,raster_fp in enumerate(raster_fps):
        print(f'{i} out of {tot_raster}')
        # load image and predict a mask
        image = load_val_image(raster_fp)
        pred = model(image)

        # convert as polygon
        pred_gdf = save_pred_vector(pred, raster_fp)
        
        # add tile_id
        timestamp, tile_id = _get_ts_tile_id(raster_fp)
        pred_gdf['timestamp'] = timestamp
        pred_gdf['tile_id'] = tile_id

        pred_gdf_list.append(pred_gdf)
    print(f'finish in {time.time() - t_str}')

    # combine all pred_gdf and save as geojson
    print(f'saving to {pred_gdf_path}')
    crs = pred_gdf.crs
    gdf_comb = gpd.GeoDataFrame(pd.concat(pred_gdf_list, ignore_index=True), crs=crs)
    gdf_comb.to_file(pred_gdf_path, driver='GeoJSON')

        

        



