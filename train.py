import seaborn as sns
import matplotlib.pyplot as plt

def train(train_splits, val_splits):
    # clear tensorflow state
    K.clear_session()
    
    # get filenames and load dataset
    train_paths = KaggleDatasets().get_gcs_path(tr_cfg['TRAIN_PATH'])
    TRAIN_FN, TRAIN_STEPS = get_filenames(train_splits,
                                          train_paths, SUB_PATH_TRAIN)
    train_dataset = get_training_dataset(TRAIN_FN)                                                         

    if tr_cfg['VAL_PATH']:
        val_paths = KaggleDatasets().get_gcs_path(tr_cfg['VAL_PATH'])
        VAL_FN, VAL_STEPS = get_filenames(val_splits, val_paths, SUB_PATH_VAL)
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
        if tr_cfg['VAL_PATH']:
            print('\n'*4)
            show_predictions(model, VAL_FN, N_VAL_PREV, shuffle=IS_TPU)  # on gpu, shuffle is expensive
        print('\n'*6, 'training results')
        show_predictions(model, TRAIN_FN, N_VAL_PREV, shuffle=IS_TPU)

    # delete everything to start fresh
    del train_dataset, valid_dataset, model
    gc.collect()



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

def show_predictions(model, ds_fn, n_show=4, shuffle=False, num_pass=0):
    """ create new dataset, shuffle(250), batch(n_show), predict and display with fn
    """
    dataset = get_preview_dataset(ds_fn, n_show, shuffle)
    
    for img,mask,fn in dataset.skip(num_pass).take(1):
        pred_mask = model(img)
        pred_mask = create_binary_mask(pred_mask)
        for i in range(n_show):
            print(fn[i].numpy().decode())
            display_img([img[i], mask[i], pred_mask[i]])