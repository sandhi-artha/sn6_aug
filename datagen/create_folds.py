import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import glob
import pickle
import numpy as np
from sklearn.cluster import KMeans

from lib.raster import get_tile_bounds
from datagen.dg_cfg import dg_cfg



### TILE SELECTOR ###
def get_ts_orient(base_dir, orient):
    """reads SAR_orientations
    return list of timestamps with orientation 0 or 1
    """
    orient_fp = os.path.join(base_dir, 'SummaryData/SAR_orientations.txt')
    with open(orient_fp, 'rb') as f:
        timsestamps = []
        for i,ts in enumerate(f):
            ts_split = ts.decode().split(' ')
            if str(orient) in ts_split[1]:
                timsestamps.append(ts_split[0])
    print(f'total timestamps: {i+1}')
    print(f'timestamps with orientation {orient}: {len(timsestamps)}')
    return timsestamps

def get_fp_orient(base_dir, orient):
    """reads SAR_orientations
    return list of sar file paths with orientation 0 or 1
    """
    timestamps = get_ts_orient(base_dir, orient)
    fps = []
    for ts in timestamps:
        # get list of file_paths for every timestamps
        _fps = glob.glob(f'{base_dir}/SAR-Intensity/*{ts}*.tif')
        for fp in _fps:
            fps.append(fp)

    print(f'total SAR images with orientation {orient}: {len(fps)}')
    return fps

def get_fps_folds(fps, folds=4):
    """fps : list
        filepaths of rasters
    folds: int
        how many splits to create
    returns: list of fps for every fold
    """
    sar_bounds = get_tile_bounds(fps)

    # mid = top-bot/2 + bot
    mid = [((b[3]-b[1])/2) + b[1] for b in sar_bounds]
    # left for 2d clustering
    # left = [b[0] for b in sar_bounds]

    mid_arr = np.array(mid)
    mid_arr = np.expand_dims(mid_arr, axis=-1)

    # create n folds dataset
    kmeans = KMeans(n_clusters=folds)
    y_kmeans = kmeans.fit_predict(mid_arr)
    
    print('Cluster distribution:')
    print(np.unique(y_kmeans, return_counts=1))

    idxs_folds = []
    # get indexes belonging to a cluster
    for cluster in range(folds):
        s = np.argwhere(y_kmeans == cluster)
        idxs_folds.append(s)

    return idxs_folds

def cluster(fps, n_fold):
    """cluster center tile of each image using KMeans
    output from sar_bounds is [left,bot,right,top]"""
    sar_bounds = get_tile_bounds(fps)

    # mid = top-bot/2 + bot
    midy = [((b[3]-b[1])/2) + b[1] for b in sar_bounds]
    # left for 2d clustering
    midx = [((b[2]-b[0])/2) + b[0] for b in sar_bounds]

    mid_arr = np.array(midx)
    mid_arr = np.expand_dims(mid_arr, axis=-1)

    # create n folds dataset
    kmeans = KMeans(n_clusters=n_fold)
    y_kmeans = kmeans.fit_predict(mid_arr)

    return midx, midy, y_kmeans

def get_cluster_fp(n_fold, y_kmeans, fps):
    print('Cluster distribution :')
    print(np.unique(y_kmeans, return_counts=1))
    fps_folds = []
    # get indexes belonging to a cluster
    for cluster in range(n_fold):
        s_list = np.argwhere(y_kmeans == cluster)
        fps_fold = [os.path.basename(fps[s[0]]) for s in s_list]
        fps_folds.append(fps_fold)    
    return fps_folds

if __name__ == '__main__':
    folds = dg_cfg['folds']
    fps0 = get_fp_orient(dg_cfg['base_dir'], 0)
    fps1 = get_fp_orient(dg_cfg['base_dir'], 1)

    comb_fps = []
    for fp in fps0: comb_fps.append(fp)
    for fp in fps1: comb_fps.append(fp)
    print(f'total tiles: {len(comb_fps)}')

    midx, midy, y_kmeans = cluster(comb_fps, n_fold=folds)
    print('Cluster distribution :')
    print(np.unique(y_kmeans, return_counts=1))

    # splitting the folds based on orient
    fps0_folds = get_cluster_fp(folds, y_kmeans[:1749], fps0)
    fps1_folds = get_cluster_fp(folds, y_kmeans[1749:], fps1)

    # save pickle in this directory
    with open(f'fps0_{folds}folds.pickle', 'wb') as f:
        pickle.dump(fps0_folds, f)

    with open(f'fps1_{folds}folds.pickle', 'wb') as f:
        pickle.dump(fps1_folds, f)