dg_cfg = {
    'base_dir'      : '../dataset/spacenet6-challenge/AOI_11_Rotterdam',
    'out_dir'       : '../dataset/sn6_aug/filter_aug_tfrec',
    'orient'        : 1,
    'resize'        : None,
    'folds'         : 4,
    'channel'       : [1],
    'tfrec_size'    : 50,
    'out_precision' : 8,
}

tr_cfg = {
    # log cfg
    'RUN'           : 6,   # REMEMBER TO INCREMENT!!
    'COMMENTS'      : 'test loss function from scratch',
    
    # dataset cfg
    'IMAGE_DIM'     : 900,  # must be the same as tfrecord res
    'IMAGE_RS'      : 640,
    'TRAIN_PATH'    : 'sn6-900-uint8-o1', #'sn6-900-uint8',  # kaggle ds name
    'TRAIN_SPLITS'  : ['fold0', 'fold1', 'fold2', 'fold3'],
    'VAL_PATH'      : None, #'base-val-8',  # if None, don't validate
    'VAL_SPLITS'    : 'val',  # will only be considered if val path exist and IS_CV=0
    'IS_FP32'       : 0,       # 1 if train or validating on fp32
    'SAR_CH'        : [1], # [0,3,2],      # HH, VV, VH. 0 = all channel
    
    # training cfg
    'DEVICE'        : 'gpu',
    'SEED'          : 17,
    'BACKBONE'      : 'effb4',            # 'effb4', 'res50'
    'ARC'           : 'fpn',              # 'unet', 'fpn'
    'WEIGHT'        : 'imagenet',         # 'imagenet', 'pre-trained from:..', None
    'LF'            : 'dice',    # 'bce', 'jaccard_distance', 'focal', 'giou'
    'EPOCHS'        : 60,
    'L_RATE'        : 4e-4,       # 32e-4, 4e-4, 5e-5
    'IS_CV'         : 0,   # cross validation
    'IS_3_FOLD'     : 0,   # do same training 3x to get an average value
    'IS_CB_ES'      : 0,   # early stopping
    'IS_CB_LRS'     : 0,   # learning rate scheduler, if false uses lr_ramp
    
    # augmentations
    'IS_RESIZE'     : 1,       
    'IS_CROP'       : 0,
    'IS_HFLIP'      : 0,
    'IS_VFLIP'      : 0,
    'IS_ROT'        : 1,
    'IS_FINE_ROT'   : 0,
    'IS_SHEAR_X'    : 0,
    'IS_SHEAR_Y'    : 0,

    'IS_F_GAUS'     : 1,
    'IS_F_MED'      : 0,

    # aug magnitude
    'ROT_RANGE'     : [-10, 10],  # None, activates fine rotate with min and max ANGLE
    'SHEAR_RANGE'   : [-10, 10],
    'GAUS_SIGMA'    : 3.0,

    # offline augs
    'OFF_AUG'       : 0,
    'OFF_FILTER'    : 'elee',
}


# inferred config
tr_cfg['BATCH_SIZE'] = 4
tr_cfg['SHUFFLE_BUFFER'] = 50


ev_cfg = {
    'run_path'      : '',
    'base_dir'      : '../dataset/sn6_aug/val',  # where raster and vector dir are located
    'save_dir'      : '../dataset/sn6_aug',
    'chart'         : 'tp',     # 'tp','recall'
}

"""
%cd sn6_aug

import json
with open('dg_cfg.json', 'w') as fp:
    json.dump(dg_cfg, fp)
"""