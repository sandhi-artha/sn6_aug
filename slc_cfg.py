slc_cfg = {
    # slc-selector
    'base_dir'      : '../dataset/spacenet6-challenge/AOI_11_Rotterdam', # path to SummaryData/SAR_orientations.txt
    'orient'        : 1,        # (bool) 0 or 1
    
    # sar-preproc
    'pol'           : 'HH',
    'ml_filter'     : 'avg',        # 'avg', 'med'
    'ml_size'       : 2,            # (int) multi-look kernel: 2, 4, 8
    'in_dir'        : '../dataset/sn6-expanded', # '../../expanded-dataset', # where the SLC stripes located
    
    # tiling
    'splits'        : ['val'],  # (list) what split to generate

    # if load_tile is 1, below data is where to load tiles from. Else, it's where to save data
    'project'       : 'sn6_aug',
    'name'          : 'base',       # (str) DO NOT USE 'train', 'val' or 'test' in the name
    'stride'        : 0,            # (int) 0, 80 unit in px
    'out_dir'       : '../dataset/sn6_aug/val', # '../../sensor',  # where the tiles will be stored
    'label_dir'     : '../dataset/spacenet6-challenge/expanded/exp_geojson_buildings', # '../../expanded/geojson_buildings',  
    'verbose'       : 0,            # (int) 0 minimum logs, 1 for all info, 2 for necessary tiling

    # tfrec
    'resize'        : None,
    'folds'         : 1,
    'channel'       : [1],
    'tfrec_size'    : 100,
    'out_precision' : 8,
}


