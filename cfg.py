dg_cfg = {
    'base_dir'      : '../dataset/spacenet6-challenge/AOI_11_Rotterdam',
    'out_dir'       : '../dataset/sn6_aug/gpu_tfrec',
    'orient'        : 1,
    'resize'        : None,
    'folds'         : 5,
    'channel'       : [1],
    'tfrec_size'    : 50,
    'out_precision' : 32,
}

tr_cfg = {
    # log cfg
    'RUN'           : 1,   # REMEMBER TO INCREMENT!!
    'COMMENTS'      : 'gpu experiments',
    
    # dataset cfg
    'IMAGE_DIM'     : 900,  # must be the same as tfrecord res
    'IMAGE_RS'      : 320,
    'TRAIN_PATH'    : '../dataset/sn6_aug/gpu_tfrec', #'sn6-900-uint8',  # kaggle ds name
    'TRAIN_SPLITS'  : ['fold0', 'fold1', 'fold2'],
    'VAL_PATH'      : '../dataset/sn6_aug/gpu_tfrec', #'base-val-8',  # if None, don't validate
    'VAL_SPLITS'    : ['fold4'],  # will only be considered if val path exist and IS_CV=0
    'SAR_CH'        : [1], # [0,3,2],      # HH, VV, VH. 0 = all channel
    'ORIENT'        : 1,
    
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
    
    # spatial transformations
    'IS_RESIZE'     : 1,       
    'IS_CROP'       : 0,
    'IS_HFLIP'      : 0,
    'IS_VFLIP'      : 0,
    'IS_ROT90'      : 0,
    'IS_FINE_ROT'   : 0,

    # pixel transformations
    'IS_F_GAUS'     : 0,
    'IS_F_MED'      : 0,
    'IS_GAUS_NOISE' : 0,
    'IS_MOT_BLUR'   : 0,

    # aug magnitude
    'ROT_RANGE'     : [-10, 10],  # None, activates fine rotate with min and max ANGLE
    'GAUS_SIGMA'    : 3.0,
    'GAUS_NOISE_VAR': [10.0, 50.0],

    # offline augs
    'OFF_AUG'       : 0,
    'OFF_FILTER'    : 'elee',

    # albu augs
    'AL_AUG'        : 0,  # toggle for albu augs
    'IS_MOT_BLUR'   : 0,  # call get_transform() each time changing this val
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