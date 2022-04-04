import os
import json
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from timebudget import timebudget

from datasets.create_folds import get_ts_orient
from datasets.slc.slc_cfg import slc_cfg

import solaris.pipesegment as pipesegment
import solaris.image as image
import solaris.sar as sar

class SarPreproc(pipesegment.PipeSegment):
    """Process 1 SLC SAR stripe
    """
    def __init__(self,
                 cfg,
                 timestamp='20190823162315_20190823162606',
                 input_dir='expanded',
                 out_dir='results',
                 out_fn='output.tif'
                 ):
        super().__init__()
        out_path = os.path.join(out_dir, out_fn)
        fn = f'CAPELLA_ARL_SM_SLC_{cfg["pol"]}_{timestamp}.tif'
        quads = image.LoadImage(os.path.join(input_dir, fn))

        self.feeder = (
            quads
            * sar.CapellaScaleFactor()
            * sar.Intensity()
            * sar.Multilook(kernel_size=cfg['ml_size'], method=cfg['ml_filter'])
            * sar.Decibels()
            * sar.Orthorectify(projection=32631, row_res=.5, col_res=.5)
            * image.SaveImage(out_path, return_image=False, no_data_value='nan')
        )

    

if __name__ == '__main__':
    timestamps = get_ts_orient(slc_cfg['base_dir'], slc_cfg['orient'])
    if not os.path.isdir(slc_cfg['out_dir']):
        os.makedirs(slc_cfg['out_dir'])

    with timebudget('SAR PRE-PROC'):
        for i,timestamp in enumerate(timestamps):
            print(f'processing raster {timestamp}.. {i} of {len(timestamps)}')

            out_fn = f'HH_{timestamp}.tif'

            # process SAR
            sar_preproc = SarPreproc(slc_cfg, timestamp, slc_cfg["in_dir"], slc_cfg["out_dir"], out_fn)
            sar_preproc()


    cfg_fn = os.path.join(slc_cfg['out_dir'], 'slc_cfg.json')
    with open(cfg_fn, 'w') as f: json.dump(slc_cfg, f)
    print('cfg saved!')