# Dataset Processing Scripts

## SpaceNet6

### Train data
Requires SpaceNet6 Competition Training Data, [download here](https://spacenet.ai/sn6-challenge/). Alternatively, here's the same version uploaded in [Kaggle](https://www.kaggle.com/datasets/sandhiwangiyana/spacenet-6-multisensor-allweather-mapping)
1. Create N folds, divided based on sensor orientation (orient 0 == South Facing, orient 1 == North Facing) see `notebooks/create_folds.ipynb`
2. Create tfrecords using config modified in `dg_cfg.py`
```
%cd datasets
python datagen.py
```

### Test data
Requires Spacenet6 Expanded Dataset, can also be [found here](https://spacenet.ai/sn6-challenge/)
1. set configuration in `datasets/slc/slc_cfg.py`
2. run `slc_preproc.py` to preprocess including: Calibration, Convert pixel values to Intensity, Multilook, Convert pixel values to Log-Intensity, and orthorectification
3. run `slc_tiling.py` to split the large processed SAR raster into smaller 900x900 pixel raster. This process also create each tiles' labels by reading the annotation data included in the Expanded Dataset. optionally run `slc_parallel.py` to perform preprocessing and tiling in multiple parallel process (needs 4-5GB of RAM per process)
4. run `slc_tfrec.py` to create tfrecord files


### Offline Augmentation
Requires filtered images from MATLAB. In the future, will migrate MATLAB scripts (including implementation of each speckle filter) to Python.


## Inria

