import os
import re    # count tfrec
import math  # augmentation and LR_ramp
import gc    # deleting stuff
import json

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from functools import partial # used when parsing tfrecords
import matplotlib.pyplot as plt
import seaborn as sns

import segmentation_models as sm
import wandb
from wandb.keras import WandbCallback

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau



### KEEP TRACK - VOLATILE CHANGES
print(tf.__version__)
print(f'tensorflow_addons version: {tfa.__version__}')
print(f'Wandb Version: {wandb.__version__}')
os.environ["KMP_SETTINGS"] = "false"  # disable logs from Intel MKL



### LOAD CONFIG ###
if os.path.isfile('tr_cfg.json'):
    print('using Kaggle config')
    with open('tr_cfg.json', 'r') as fp:
        tr_cfg = json.load(fp)
else:
    from cfg import tr_cfg
    print('using saved config')






### GLOBAL VARIABLES ###
AUTOTUNE = tf.data.experimental.AUTOTUNE

# convert string and other types to bool for faster change
IS_EXT_VAL = 1 if tr_cfg['VAL_PATH'] == 'base-val-8' else 0
IMAGE_CH = len(tr_cfg['SAR_CH'])


def seed_everything(seed):
#     random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

seed_everything(tr_cfg['SEED'])







### AUGMENTATIONS
def resize_example(image, label, ext_val):
    """image must be type float"""
    if ext_val:
        image = tf.reshape(image, [640, 640, 3])
        label = tf.reshape(label, [640, 640, 1])
    else:
        image = tf.reshape(image, [tr_cfg['IMAGE_DIM'], tr_cfg['IMAGE_DIM'], IMAGE_CH])
        label = tf.reshape(label, [tr_cfg['IMAGE_DIM'], tr_cfg['IMAGE_DIM'], 1])

    image = tf.image.resize(image, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']])
    label = tf.image.resize(label, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']])
    return image, label

def random_crop(image, label):
    """image and mask must be same dtype"""
    # make [900,900,4]
    size = [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']]
    stacked_image = tf.concat([image, label], axis=-1)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[*size, 4])
    
    image_crop = tf.reshape(image, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS'], IMAGE_CH])
    label_crop = tf.reshape(label, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS'], 1])
    # for label, if you want [w,h,1] shape, use -1:
    return cropped_image[:,:,:3], cropped_image[:,:,-1:]




### DATALOADER ###
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, 
    # i.e. test10-687.tfrec = 687 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    
    return np.sum(n)

def get_filenames(split, ds_path, sub_path='', out=False):
    if isinstance(split, list):
        fns = []
        for fold in split:
            fol_path = os.path.join(ds_path, sub_path, f'{fold}*.tfrec')
            fold_fns  = tf.io.gfile.glob(fol_path)
            for fn in fold_fns:
                fns.append(fn)
    else:
        fol_path = os.path.join(ds_path, sub_path, f'{split}*.tfrec')
        fns  = tf.io.gfile.glob(fol_path)
    
    num_img = count_data_items(fns)
    steps = num_img//tr_cfg['BATCH_SIZE']

    if out:
        print(f'{split} files: {len(fns)} with {num_img} images')
    
    return fns, steps

TFREC_FORMAT = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'fn': tf.io.FixedLenFeature([], tf.string)
}

def read_tfrecord(feature, load_fn):
    features = tf.io.parse_single_example(feature, TFREC_FORMAT)

    image = tf.io.parse_tensor(features["image"], tf.uint8)
    image = tf.cast(image, tf.float32)/255.0
    label = tf.io.parse_tensor(features["label"], tf.bool)
    label = tf.cast(label, tf.float32)

    if load_fn:
        fn = features["fn"]
        return image, label, fn
    else:
        return image, label


def load_dataset(filenames, load_fn=False, ext_val=False, ordered=False):
    """
    takes list of .tfrec files, read using TFRecordDataset,
    parse and decode using read_tfrecord func, returns image&label/image_name(test)
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda x: read_tfrecord(x, load_fn),
                          num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda img,label: resize_example(img, label, ext_val),
                          num_parallel_calls=AUTOTUNE)
    return dataset


    

def get_training_dataset(files, augment=False, shuffle=True, load_fn=False):
    dataset = load_dataset(files, load_fn=load_fn)
#     if augment: dataset = dataset.map()
    dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(2000)  # 2000
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def get_validation_dataset(files, load_fn=False):
    dataset = load_dataset(files, load_fn=load_fn, ext_val=IS_EXT_VAL)
    dataset = dataset.cache()
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset






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

if not tr_cfg['IS_CB_LRS']:
    LR_START = 1e-8 # 1e-8
    LR_MIN = 3e-8  # 1e-8
    LR_MAX = tr_cfg['L_RATE']
    LR_RAMPUP_EPOCHS = 5  # 5
    LR_SUSTAIN_EPOCHS = 2  # 2
    N_CYCLES = .5

    print(f'Learning rate schedule: {LR_START} to {LR_MAX} to {LR_MIN}')

# load pre-trained models
def load_pretrained_model():
    model_path = '../input/sn6-models/Unet_b4_512_imagenet_rgb_crop.h5'

    custom_objects = {
        'iou_score': sm.metrics.IOUScore(threshold=0.5),
        'f1-score': sm.metrics.FScore(threshold=0.5)
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def load_model():
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
    if tr_cfg['LF'] == 'bce':
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # elif tr_cfg['LF'] == 'jaccard_distance':
    #     loss = jaccard_distance
    elif tr_cfg['LF'] == 'focal':
        loss = tfa.losses.SigmoidFocalCrossEntropy()
    elif tr_cfg['LF'] == 'giou':
        loss = tfa.losses.GIoULoss()
    else:
        raise ValueError(f'loss function: {tr_cfg["LF"]} not recognized')

    
    optim = tf.keras.optimizers.Adam(tr_cfg['L_RATE'])
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model.compile(optim, loss, metrics)
    return model




### CALLBACKS ###
# stop training when no improvements in val_loss
cb_early_stopping = EarlyStopping(patience = 5, verbose = 0, restore_best_weights = True)

# allows changing the learning rate based on val_auc
cb_lr_reduce = ReduceLROnPlateau(monitor="val_loss",
                                 factor=0.75, patience=5,
                                 verbose=1, min_lr = 1e-8)




### PLOT ###
def show_example(img, mask):
    f,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(img)
    if len(mask.shape)==3:
        mask = np.squeeze(mask, axis=-1)
    ax[1].imshow(mask, cmap='gray')
    plt.show()

def plot_metrics(history):
    metric_list = [m for m in list(history.keys()) if m is not 'lr']
    size = len(metric_list)//2  # adjust vertical space to fit all metrics
    fig, axes = plt.subplots(size, 1, sharex='col', figsize=(20, size * 4))
    if size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for index in range(len(metric_list)//2):
        metric_name = metric_list[index]
        val_metric_name = metric_list[index+size]
        axes[index].plot(history[metric_name], label='Train %s' % metric_name)
        axes[index].plot(history[val_metric_name], label='Validation %s' % metric_name)
        axes[index].legend(loc='best', fontsize=16)
        axes[index].set_title(metric_name)
        if 'loss' in metric_name:
            axes[index].axvline(np.argmin(history[metric_name]), linestyle='dashed')
            axes[index].axvline(np.argmin(history[val_metric_name]), linestyle='dashed', color='orange')
        else:
            axes[index].axvline(np.argmax(history[metric_name]), linestyle='dashed')
            axes[index].axvline(np.argmax(history[val_metric_name]), linestyle='dashed', color='orange')

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.show()
    
    # print model performance
    bm_idx = np.argmin(history['val_loss'])
    print(f'best model at epoch: {bm_idx}')
    v_loss = history['val_loss'][bm_idx]
    v_iou = history["val_iou_score"][bm_idx]
    v_f1 = history["val_f1-score"][bm_idx]

    loss = history['loss'][bm_idx]
    iou = history["iou_score"][bm_idx]
    f1 = history["f1-score"][bm_idx]


    print(f'val loss: {v_loss:.4f}, val iou: {v_iou:.4f}, val f1: {v_f1:.4f}')
    print(f'loss: {loss:.4f}, iou: {iou:.4f}, f1: {f1:.4f}')
    print(f'best val IoU: {np.max(history["val_iou_score"])}')




### EVALUATION ###
# for binary you need a threshold value to differ what's category 0 and what's 1
def create_binary_mask(pred_mask):
    thresh = 0.5
    return tf.where(pred_mask>=thresh, 1, 0)

def display_img(display_list):
    title = ['Input Tile', 'True Maks', 'Predicted Mask']
    plt.figure(figsize=(18,8))
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    plt.show()

def show_predictions(model, dataset, num=1, n_show=4, num_pass=0):
    """Take n images (according to BATCH_SIZE), predict and display results
    """
    for img,mask in dataset.skip(num_pass).take(num):
        pred_mask = model.predict(img)
        pred_mask = create_binary_mask(pred_mask)
        for i in range(n_show):
            display_img([img[i], mask[i], pred_mask[i]])
