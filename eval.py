# vizualize batched dataset, plot history, plot prediction results
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf



### PLOT ###
def show_example(img, mask):
    f,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(img)
    if len(mask.shape)==3:
        mask = np.squeeze(mask, axis=-1)
    ax[1].imshow(mask, cmap='gray')
    plt.show()

def plot_metrics(history):
    sns.set(style='whitegrid')
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
    sns.set_theme()
    
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