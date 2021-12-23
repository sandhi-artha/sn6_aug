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
            show_predictions(model, VAL_FN, 24, True)
        print('\n'*6, 'training results')
        show_predictions(model, TRAIN_FN, 24, True)

    # delete everything to start fresh
    del train_dataset, valid_dataset, model
    gc.collect()