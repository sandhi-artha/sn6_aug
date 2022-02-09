import pickle
from scipy.io import savemat


if __name__=='__main__':
    orient = 1
    with open(f'../fps{orient}_5folds.pickle', 'rb') as f:
        fns_folds = pickle.load(f)

    fns_folds_dict = {}
    for i in range(len(fns_folds)):
        # remove initial p
        fns_folds_dict[f'fold{i}'] = [
            fn.replace('SN6_Train_AOI_11_Rotterdam_SAR-Intensity_', '')
            for fn in fns_folds[i]
        ]

    savemat(f'fps{orient}_5folds.mat', fns_folds_dict)