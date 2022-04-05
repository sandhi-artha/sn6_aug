# returns model compiled
# mod by cfg
import os
import json
import math  # augmentation and LR_ramp
import random
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
import wandb
from wandb.keras import WandbCallback
import yaml

from sn6.cfg import lrfn
from sn6.dataloader import DataLoader
from aug.aug_albu import AugAlbu
from aug.aug_tf import AugTF, get_reduce_fun

# IS_TPU = 1 if tr_cfg['DEVICE'] == 'tpu' else 0

class Dict2Obj(object):
    """Turns a dictionary into a class"""
    def __init__(self, dictionary):
        for key in dictionary: setattr(self, key, dictionary[key])

class BFE():
    """Building Footprint Extractor"""
    def __init__(self, cfg, is_log, project_name):
        self.cfg = Dict2Obj(cfg)
        self.image_ch = len(self.cfg.SAR_CH)
        self.is_log = is_log
        if is_log:
            self.run = wandb.init(
                project=project_name, name=cfg['NAME'],
                config=cfg, reinit=True)
        else:
            self.run = None
        seed_everything(self.cfg.SEED)  # set seed
    
    def load_data(self):
        """create dataloader object with selected reduce and aug function"""
        train_reduce = get_reduce_fun(self.cfg.REDUCE_RES, self.cfg)
        val_reduce = get_reduce_fun(self.cfg.VAL_REDUCE_RES, self.cfg)

        aug_albu = AugAlbu(self.cfg)
        if aug_albu.IS_AUG: aug_albu_fun = aug_albu.transform
        else: aug_albu_fun = None
        
        aug_tf = AugTF(self.cfg)
        if aug_tf.IS_AUG: aug_tf_fun = aug_tf.transform
        else: aug_tf_fun = None
            
        self.dataloader = DataLoader(
            self.cfg, train_reduce, val_reduce, aug_albu_fun, aug_tf_fun)

    def load_model(self, model_path=None, wandb_path=None):
        """create model object by either download new model,
            or use pt model from local,
            or download from wandb"""
        if model_path:
            self.model = load_pretrained_model(model_path)
        elif wandb_path:
            self.model = load_wandb_model(wandb_path)
        else:
            self.model = load_new_model(self.cfg)
        
        print(f'Total params: {self.model.count_params():,}')

    def train(self):
        train_ds, train_steps, val_ds, val_steps = self.dataloader.load_data()
        callbacks = get_cb_list(self.cfg, self.is_log)
        print('starting training..')
        self.history = self.model.fit(
            train_ds, 
            steps_per_epoch=train_steps, 
            epochs=self.cfg.EPOCHS,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=2)

        if self.is_log:  # finish run session in wandb
            self.run.finish()

    def view_results(self, num_images):
        self.dataloader.show_predictions(self.model, num_images)



    
# self.TRAIN_FN, self.TRAIN_STEPS = get_filenames(
#     self.cfg.TRAIN_SPLITS, self.cfg.TRAIN_PATH)
# self.train_ds = self.get_training_dataset(self.TRAIN_FN)

# self.VAL_FN, self.VAL_STEPS = get_filenames(
#     self.cfg.VAL_SPLITS, self.cfg.VAL_PATH)
# self.val_ds = self.get_validation_dataset(self.VAL_FN)





        
















def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

def get_config_wandb(run_path):
    """restore model and read yaml as dict"""
    cfg_file = wandb.restore('config.yaml', run_path=run_path)
    cfg_y = yaml.load(cfg_file, Loader=yaml.FullLoader)
    cfg_file.close()
    
    cfg = {}  # create new dictionary
    
    # get only capital keys, cfg that you wrote
    for key in cfg_y.keys():
        if key.isupper():
            cfg[key] = cfg_y[key]['value']
    
    return cfg


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


### MODEL ###
def load_pretrained_model(model_path):

    custom_objects = {
        'dice_coef_loss': dice_coef_loss,
        'iou_score': sm.metrics.IOUScore(threshold=0.5),
        'f1-score': sm.metrics.FScore(threshold=0.5)
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def load_new_model(cfg):
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

    IMAGE_CH = len(cfg.SAR_CH)
    print(f'loading the {cfg.ARC}_{cfg.BACKBONE} model..')
    kwargs = {}
    
    # determine backbone
    if cfg.BACKBONE == 'effb4':
        kwargs['backbone_name'] = 'efficientnetb4'
    elif cfg.BACKBONE == 'res50':
        kwargs['backbone_name'] = 'resnet50'
    else:
        raise ValueError(f'backbone: {cfg.BACKBONE} is not recognized')
    
    # model params
    kwargs['input_shape'] = (None, None, IMAGE_CH)
    kwargs['classes'] = 1
    kwargs['activation'] = 'sigmoid'
    kwargs['encoder_weights'] = cfg.WEIGHT

    if cfg.ARC == 'unet':
        model = sm.Unet(**kwargs)
    elif cfg.ARC == 'fpn':
        model = sm.FPN(**kwargs)
    else:
        raise ValueError(f'architecture: {cfg.ARC} is not recognized')
        
    # compile keras model with defined optimizer, loss and metrics
    try:
        loss = LF_dict[cfg.LF]
    except KeyError:
        raise KeyError(f'loss function: {cfg.LF} not recognized')

    optim = tf.keras.optimizers.Adam(cfg.L_RATE)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model.compile(optim, loss, metrics)
    return model

def load_wandb_model(wandb_path):
    # download .h5 file
    wandb_cfg = get_config_wandb(wandb_path)
    print(f'loading model {wandb_cfg["NAME"]}')
    model_file = wandb.restore('model-best.h5', run_path=wandb_path)
    model_path = model_file.name
    model_file.close()
    return load_pretrained_model(model_path)

### CALLBACKS ###
def get_cb_list(cfg, is_log):
    # stop training when no improvements in val_loss
    cb_early_stopping = EarlyStopping(
        patience=5, verbose=0, restore_best_weights=True)

    # allows changing the learning rate based on val_auc
    cb_lr_reduce = ReduceLROnPlateau(
        monitor="val_loss", factor=0.75, patience=5, verbose=1, min_lr = 1e-8)
    
    callbacks = []
    if cfg.IS_CB_ES: callbacks.append(cb_early_stopping)
    if cfg.IS_CB_LRS:
        callbacks.append(LearningRateScheduler(lrfn))
    else:
        callbacks.append(cb_lr_reduce)

    if is_log:  # use wandb only when log is True
        callbacks.append(WandbCallback(monitor='val_iou_score', mode='max'))

    return callbacks

