# returns model compiled
# mod by cfg
import os
import json
import math  # augmentation and LR_ramp

import tensorflow as tf
from tensorflow.keras import backend as K
# import tensorflow_addons as tfa
import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau


# print(f'tensorflow_addons version: {tfa.__version__}')






### LOSSS ###
# https://gist.github.com/robinvvinod/60d61a3ca642ddd826cf1e6c207cb421#file-ftllosses-py
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

LF_dict = {
    'bce'           : tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # 'focal'         : tfa.losses.SigmoidFocalCrossEntropy(),  # loss very low, but iou and f1 don't improve
    'dice'          : dice_coef_loss,
    'tversky'       : tversky_loss,
    # 'giou'          : tfa.losses.GIoULoss(),  # error when training
    'jacc'          : sm.losses.JaccardLoss(),  # iou loss
    'dice_sm'       : sm.losses.DiceLoss(),
    'focal_sm'      : sm.losses.BinaryFocalLoss(),
    'focal_tversky' : focal_tversky_loss
}

### MODEL ###
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        progress = (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) / (tr_cfg['EPOCHS'] - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)
        lr = LR_MAX * (0.5 * (1.0 + tf.math.cos(math.pi * N_CYCLES * 2.0 * progress)))
        if LR_MIN is not None:
            lr = tf.math.maximum(LR_MIN, lr)
            
    return lr


# load pre-trained models
def load_pretrained_model(model_path):

    custom_objects = {
        'dice_coef_loss': dice_coef_loss,
        'iou_score': sm.metrics.IOUScore(threshold=0.5),
        'f1-score': sm.metrics.FScore(threshold=0.5)
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def load_new_model():
    print(f'loading the {tr_cfg["ARC"]}_{tr_cfg["BACKBONE"]} model..')
    kwargs = {}
    
    # determine backbone
    if tr_cfg['BACKBONE'] == 'effb4':
        kwargs['backbone_name'] = 'efficientnetb4'
    elif tr_cfg['BACKBONE'] == 'res50':
        kwargs['backbone_name'] = 'resnet50'
    else:
        raise ValueError(f'backbone: {tr_cfg["BACKBONE"]} is not recognized')
    
    # model params
    kwargs['input_shape'] = (None, None, IMAGE_CH)
    kwargs['classes'] = 1
    kwargs['activation'] = 'sigmoid'
    kwargs['encoder_weights'] = tr_cfg['WEIGHT']
    

    if tr_cfg['ARC'] == 'unet':
        model = sm.Unet(**kwargs)
    elif tr_cfg['ARC'] == 'fpn':
        model = sm.FPN(**kwargs)
    else:
        raise ValueError(f'architecture: {tr_cfg["ARC"]} is not recognized')
        
    # compile keras model with defined optimizer, loss and metrics
    LF =  tr_cfg['LF']
    try:
        loss = LF_dict[LF]
    except KeyError:
        raise KeyError(f'loss function: {LF} not recognized')

    optim = tf.keras.optimizers.Adam(tr_cfg['L_RATE'])
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model.compile(optim, loss, metrics)
    return model

def load_model():
    """either download new model, or use pt ones to continue training"""
    weight = tr_cfg['WEIGHT']
    if weight == 'imagenet' or weight is None:
        print('loading new model')
        model = load_new_model()
    else:
        # download .h5 file
        cfg = get_config_wandb(weight)
        print(f'loading model {cfg["NAME"]}')
        model_file = wandb.restore('model-best.h5', run_path=weight)
        model_path = model_file.name
        model_file.close()
        model = load_pretrained_model(model_path)
    
    print(f'Total params: {model.count_params():,}')
    return model




### CALLBACKS ###
def get_cb_list():
    # stop training when no improvements in val_loss
    cb_early_stopping = EarlyStopping(patience = 5, verbose = 0, restore_best_weights = True)

    # allows changing the learning rate based on val_auc
    cb_lr_reduce = ReduceLROnPlateau(monitor="val_loss",
                                    factor=0.75, patience=5,
                                    verbose=1, min_lr = 1e-8)
    
    callbacks = []
    if tr_cfg['IS_CB_ES']: callbacks.append(cb_early_stopping)
    if tr_cfg['IS_CB_LRS']:
        callbacks.append(cb_lr_reduce)
    else:
        callbacks.append(LearningRateScheduler(lrfn))

    return callbacks

