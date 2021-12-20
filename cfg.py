dg_cfg = {
    'base_dir'      : '../dataset/spacenet6-challenge/AOI_11_Rotterdam',
    'out_dir'       : '../',
    'orient'        : 0,
    'resize'        : None,
    'folds'         : 4,
    'channel'       : [1,4,3],
    'tfrec_size'    : 100,
    'out_precision' : 8,
}

tr_cfg = {
    # log cfg
    'run'        : 4,   # REMEMBER TO INCREMENT!!
    'COMMENTS'   : 'external validation',
    
    # dataset cfg
    'IMAGE_DIM'  : 900,  # must be the same as tfrecord res
    'IMAGE_RS'   : 640,
    'TRAIN_PATH' : 'sn6-900-uint8',  # kaggle ds name
    'VAL_PATH'   : 'base-val-8',
    'IS_FP32'    : 0,       # 1 if train or validating on fp32
    'SAR_CH'     : [1,4,3], # [0,3,2],      # HH, VV, VH. 0 = all channel
    
    # training cfg
    'DEVICE'     : 'tpu',
    'SEED'       : 17,
    'BACKBONE'   : 'effb4',            # 'effb4', 'res50'
    'ARC'        : 'fpn',              # 'unet', 'fpn'
    'WEIGHT'     : 'imagenet',         # 'imagenet', 'pre-trained from:..', None
    'LF'         : 'bce',    # 'bce', 'jaccard_distance', 'focal', 'giou'
    'EPOCHS'     : 60,
    'L_RATE'     : 5e-5,       # 5e-5
    'IS_3_FOLD'  : 0,   # do same training 3x to get an average value
    'IS_CB_ES'   : 0,   # early stopping
    'IS_CB_LRS'  : 0,   # learning rate scheduler, if false uses lr_ramp
}


"""
%cd sn6_aug

import json
with open('dg_cfg.json', 'w') as fp:
    json.dump(dg_cfg, fp)
"""