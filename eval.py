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
import geopandas as gpd
import wandb

from solaris.vector_mask import mask_to_poly_geojson
from solaris.base import Evaluator
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

    if ev_cfg['show_res']:
        # combine all pred_gdf and save as geojson
        true_gdf_path = os.path.join(ev_cfg['save_dir'], 'true.geojson')
        save_combine_gdf(raster_fps, true_gdf_path)

        evaluator = Evaluator(true_gdf_path)
        evaluator.load_proposal(pred_gdf_path, proposalCSV=False, conf_field_list=[])

        res = evaluator.eval_iou_return_GDFs(calculate_class_scores=False)
        print(res[0])
        # if res[1] is not None and res[2] is not None:
        #     tp_from_gt_gdf = get_tp_from_gt(res[2], true_gdf_path)
        #     show_eval_res(res, tp_from_gt_gdf)

        

        



