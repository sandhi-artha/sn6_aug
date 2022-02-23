import seaborn as sns
import matplotlib.pyplot as plt

def preview_data():
    train_paths = KaggleDatasets().get_gcs_path(tr_cfg['TRAIN_PATH'])
    if tr_cfg['OFF_AUG_PATH']:
        off_ds_paths = KaggleDatasets().get_gcs_path(tr_cfg['OFF_AUG_PATH'])
    TRAIN_FN, TRAIN_STEPS = get_filenames(tr_cfg['TRAIN_SPLITS'],
                                          train_paths, off_ds_paths, out=1)
    train_dataset = get_training_dataset(TRAIN_FN, shuffle=False, ordered=True)

    f, ax = plt.subplots(2,4,figsize=(20,10))
    for i, [img,label] in enumerate(train_dataset.take(4)):
        ax[0,i].imshow(img[0], cmap='gray')
        ax[1,i].imshow(label[0], cmap='gray')
    plt.show()

    val_paths = KaggleDatasets().get_gcs_path(tr_cfg['VAL_PATH'])
    VAL_FN, VAL_STEPS = get_filenames(tr_cfg['VAL_SPLITS'], val_paths, out=1)
    valid_dataset = get_validation_dataset(VAL_FN)

    for img, label in valid_dataset.take(1):
        f,[ax1,ax2] = plt.subplots(1,2)
        ax1.imshow(img[0], cmap='gray')
        ax2.imshow(label[0], cmap='gray')
        plt.show()
    
    # delete everything to start fresh
    del train_dataset, valid_dataset
    gc.collect()


def do_train():
    for val_splits in cv_folds:  # alternate val fold if CV
        model_list = []
        for i in range(REP):  # repeat 3x
            print(f'TRAIN RUN: {tr_cfg["RUN"]}, REP: {i}..')

            if tr_cfg['IS_CV']:
                # train_splits is the rest of splits other than used by val_splits
                train_splits = [split for split in cv_folds if split != val_splits]
                print(f'Training on: {train_splits}, Validating on: {val_splits}')
                # update splits in config
                tr_cfg['TRAIN_SPLITS'] = train_splits
                tr_cfg['VAL_SPLITS'] = val_splits
            else:
                # use all splits for training, use external for validation
                train_splits = tr_cfg['TRAIN_SPLITS']
                val_splits = tr_cfg['VAL_SPLITS']

            model_list.append(train(train_splits, val_splits))

            # increment RUN, append to name,
            tr_cfg['RUN'] += 1
            tr_cfg['NAME'] = f"{tr_cfg['NAME'].split('_')[0]}_{tr_cfg['RUN']}"

            if REP>1:  # increase seed if doing >1 REP with same config
                tr_cfg['SEED'] += 1
                seed_everything(tr_cfg['SEED'])

        # averaging predictions
        if tr_cfg['IS_3_FOLD'] and tr_cfg['VAL_PATH']:
            get_mean_pred(tr_cfg['VAL_SPLITS'])

        if not tr_cfg['IS_CV']:
            break  # prevent repeating if not using CV


def train(train_splits, val_splits):
    # clear tensorflow state
    K.clear_session()
    
    # get filenames and load dataset
    train_paths = KaggleDatasets().get_gcs_path(tr_cfg['TRAIN_PATH'])
    if tr_cfg['OFF_AUG_PATH']:
        off_ds_paths = KaggleDatasets().get_gcs_path(tr_cfg['OFF_AUG_PATH'])
    TRAIN_FN, TRAIN_STEPS = get_filenames(train_splits,
                                        train_paths, off_ds_paths)
    train_dataset = get_training_dataset(TRAIN_FN)                                                         

    if tr_cfg['VAL_PATH']:
        val_paths = KaggleDatasets().get_gcs_path(tr_cfg['VAL_PATH'])
        VAL_FN, VAL_STEPS = get_filenames(val_splits, val_paths)
        valid_dataset = get_validation_dataset(VAL_FN)
    else:
        valid_dataset = VAL_STEPS = None

    # load model
    if tr_cfg['DEVICE'] == 'tpu':
        # this is to distribute same weights of models to each 8 tpu cores
        with strategy.scope(): model = load_model()
    else:
        model = load_model()

    # load callbacks
    callbacks = get_cb_list()

    if LOG:  # initialize wandb
        run = wandb.init(project=PROJECT_NAME,
                         name= tr_cfg['NAME'],
                         config=tr_cfg,
                         reinit=True,
                        )
        callbacks.append(WandbCallback())


    print('starting training..')
    history = model.fit(train_dataset, 
                        steps_per_epoch=TRAIN_STEPS, 
                        epochs=tr_cfg['EPOCHS'],
                        validation_data=valid_dataset,
                        validation_steps=VAL_STEPS,
                        callbacks=callbacks,
                        verbose=1)

    # finish run session in wandb
    if LOG: run.finish()

    # preview results
    if valid_dataset is not None: plot_metrics(history.history)
    if IMAGE_CH<4:
        if N_VAL_PREV>0:
            if tr_cfg['VAL_PATH']:
                print('\n'*4)
                show_predictions(model, VAL_FN, N_VAL_PREV)  # on gpu, shuffle is expensive
            print('\n'*6, 'training results')
            show_predictions(model, TRAIN_FN, N_VAL_PREV)

    # delete everything to start fresh
    del train_dataset, valid_dataset#, model
    gc.collect()

    return model


# TRAINING VALIDATION PLOT
# for binary you need a threshold value to differ what's category 0 and what's 1
def create_binary_mask(pred_mask):
    thresh = 0.5
    return tf.where(pred_mask>=thresh, 1, 0)

def display_img(display_list):
    title = ['Input Tile', 'True Maks', 'Predicted Mask']
    plt.figure(figsize=(18,8))
    
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 1:
            cmap = 'gray'
        else:
            cmap = None
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i], fontsize=24)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap=cmap)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def show_predictions(model, ds_fn, n_show=4, num_pass=0):
    """ create new dataset, shuffle(250), batch(n_show), predict and display with fn
    """
    dataset = tf.data.TFRecordDataset(ds_fn, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    # dataset = get_preview_dataset(ds_fn, n_show, shuffle)
    
    for img,mask,fn in dataset.skip(num_pass).take(n_show):
        img = tf.expand_dims(img, axis=0)
        mask = tf.expand_dims(mask, axis=0)
        img, mask = VAL_REDUCE_RES(img, mask)
        pred_mask = model(img)
        pred_mask = create_binary_mask(pred_mask)
        print(fn.numpy().decode())
        display_img([img[0], mask[0], pred_mask[0]])