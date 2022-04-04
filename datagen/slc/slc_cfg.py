slc_cfg = {
    # slc-selector
    'base_dir'      : '../../../image_folder/AOI_11_Rotterdam', # path to SummaryData/SAR_orientations.txt
    'orient'        : 0,            # (bool) 0 or 1
    
    # sar-preproc
    'pol'           : 'HH',
    'ml_filter'     : 'avg',        # 'avg', 'med'
    'ml_size'       : 2,            # (int) multi-look kernel: 2, 4, 8
    'in_dir'        : '../../../image_folder/sn6-expanded',    # where the SLC files located
    
    # tiling
    'splits'        : ['val'],      # (list) what split to generate
    'project'       : 'sn6_aug',    # folder name
    'name'          : 'base',       # (str) DO NOT USE 'train', 'val' or 'test' in the name
    'stride'        : 0,            # (int) 0, 80 unit in px
    'out_dir'       : '../../../image_folder/processed/val', # where processed raster and the tiles will be stored
    'label_dir'     : '../../../image_folder/exp_geojson_buildings', # where geojson labels for expanded dataset are located,  
    'verbose'       : 0,            # (int) 0 minimum logs, 1 for all info, 2 for necessary tiling

    # tfrec
    'resize'        : None,
    'folds'         : 1,
    'channel'       : [1],
    'tfrec_size'    : 50,
    'out_precision' : 32,
}


