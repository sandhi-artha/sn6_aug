dg_cfg = {
    'base_dir'      : '../dataset/spacenet6-challenge/AOI_11_Rotterdam',
    'orient'        : 0,
    'resize'        : None,
    'folds'         : 4,
    'channel'       : [1,4,3],
    'tfrec_size'    : 100,
    'out_precision' : 8,
}

tr_cfg = {

}


"""
%cd sn6_aug

import json
with open('dg_cfg.json', 'w') as fp:
    json.dump(dg_cfg, fp)
"""