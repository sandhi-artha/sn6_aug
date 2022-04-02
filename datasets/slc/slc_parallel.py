import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from multiprocessing import Pool

from datasets.datagen import get_ts_orient
from datasets.slc.slc_cfg import slc_cfg
from datasets.slc.slc_preproc import SarPreproc
from datasets.slc.slc_tiling import raster_vector_tiling

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

    NUM_OF_PROCESS = 4
    processes_pool = Pool(NUM_OF_PROCESS)
    processes_pool.map(preproc_and_tiling, timestamps)