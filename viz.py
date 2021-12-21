# viz batched dataset, plot history, plot prediction results
import matplotlib.pyplot as plt
import numpy as np

def show_example(img, mask):
    f,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(img)
    if len(mask.shape)==3:
        mask = np.squeeze(mask, axis=-1)
    ax[1].imshow(mask, cmap='gray')
    plt.show()

a = np.array([1,2,3])
print(a)