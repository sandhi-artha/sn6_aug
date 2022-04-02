tr_cfg = {
    # log cfg
    'RUN'           : 1,   # REMEMBER TO INCREMENT!!
    'COMMENTS'      : 'gpu experiments',
    
    # dataset cfg
    'IMAGE_DIM'     : 900,  # must be the same as tfrecord res
    'IMAGE_RS'      : 320,
    'TRAIN_PATH'    : '../dataset/sn6_aug/gpu_tfrec', #'sn6-900-uint8',  # kaggle ds name
    'OFF_AUG_PATH'  : 'sn6-crop-off/elee',
    'TRAIN_SPLITS'  : ['fold0', 'fold1'],
    'VAL_PATH'      : '../dataset/sn6_aug/gpu_tfrec', #'base-val-8',  # if None, don't validate
    'VAL_SPLITS'    : ['fold4'],  # will only be considered if val path exist and IS_CV=0
    'SAR_CH'        : [1], # [0,3,2],      # HH, VV, VH. 0 = all channel
    'ORIENT'        : 1,
    
    # training cfg
    'DEVICE'        : 'gpu',
    'SEED'          : 17,
    'BACKBONE'      : 'effb4',            # 'effb4', 'res50'
    'ARC'           : 'fpn',              # 'unet', 'fpn'
    'WEIGHT'        : None,         # 'imagenet', 'pre-trained from:..', None
    'LF'            : 'dice',    # 'bce', 'jaccard_distance', 'focal', 'giou'
    'L_RATE'        : 32e-4,       # 32e-4, 4e-4, 5e-5
    'IS_CV'         : 0,   # cross validation
    'IS_3_FOLD'     : 0,   # do same training 3x to get an average value
    'IS_CB_ES'      : 0,   # early stopping
    'IS_CB_LRS'     : 0,   # learning rate scheduler, if false uses lr_ramp
    
    # reduce_resize
    'REDUCE_RES'    : 'pad_resize',
    'VAL_REDUCE_RES': 'pad_resize',

    # spatial transformations
    'IS_HFLIP'      : 0,
    'IS_VFLIP'      : 0,
    'IS_ROT90'      : 0,
    'IS_FINE_ROT'   : 0,
    'IS_SHEAR_X'    : 0,
    'IS_SHEAR_Y'    : 0,

    # aug magnitude
    'ROT_RANGE'     : [-10, 10],
    'SHEAR_RANGE'   : [-10, 10],

    # pixel transformations
    'IS_MOT_BLUR'   : 0,
    'IS_SHARPEN'    : 0,
    'IS_CLAHE'      : 0,
    'IS_GAUS_NOISE' : 0,
    'IS_SPECKLE_NOISE': 0,
    'IS_COARSE_DO'  : 0,

    # offline augs
    'OFF_DS'        : '',
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