import os
import json
from multiprocessing import Pool

from timebudget import timebudget

from datagen import get_ts_orient
from slc_preproc import SarPreproc
from slc_tiling import raster_vector_tiling
from slc_cfg import slc_cfg

def preproc_and_tiling(timestamp):
    out_fn = f'HH_{timestamp}.tif'
    sar_preproc = SarPreproc(slc_cfg, timestamp, slc_cfg["in_dir"], slc_cfg["out_dir"], out_fn)
    sar_preproc()

    proc_slc_path = os.path.join(slc_cfg["out_dir"], out_fn)  # slc.tif path
    raster_vector_tiling(slc_cfg, timestamp, slc_cfg['orient'], proc_slc_path)

if __name__=='__main__':
    timestamps = get_ts_orient(slc_cfg['base_dir'], slc_cfg['orient'])
    if not os.path.isdir(slc_cfg['out_dir']):
        os.makedirs(slc_cfg['out_dir'])

    processes_pool = Pool(4)
    processes_pool.map(preproc_and_tiling, timestamps)