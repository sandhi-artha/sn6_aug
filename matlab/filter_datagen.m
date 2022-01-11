% read an example
DATA_DIR = '';
ex_fn = '';
% image contianing buildings with speckle
ex_fp = 'D:\Projects\python\dataset\spacenet6-challenge\AOI_11_Rotterdam\SAR-Intensity\SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190804135332_20190804135556_tile_7159.tif';
% image of water (homogenous), for true homogenous, std = 0
% ex_fp = 'D:\Projects\python\dataset\spacenet6-challenge\AOI_11_Rotterdam\SAR-Intensity\SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190804124749_20190804125033_tile_3671.tif';

[sar,r] = readgeoraster(ex_fp);
% read just the HH of SAR
sar_hh = db2mag(sar(:,:,1));
% sar_hh = sar(301:end,1:600,1);  % crop the nodata part
% sar_ori = rescale(sar_hh);

% 'mean', 'mmse', 'kuan', 'ekuan', 'lee', 'elee', 'frost', 'efrost', 'gmap', 'av2d', 'ml2d', 'bm3d'
f_list = ["mean" "mmse" "elee" "frost" "gmap"];
% f_list = ["ml2d" "bm3d"];
k_list = [3 5 7 9];

% f = tiledlayout(length(f_list), length(k_list), 'TileSpacing', 'compact', 'Padding', 'compact');

% matlab ask to regenerate the P-file, I simply turn off warning
for f_idx = 1:length(f_list)
    fig = figure();
    f = tiledlayout(1, length(k_list), 'TileSpacing', 'tight', 'Padding', 'tight');
    for k_idx = 1:length(k_list)
        filt = f_list(f_idx);
        size = k_list(k_idx);
        sar_res = sarimg_despeckling_filter(filt, sar_hh, 1, size, 2);
        warning('off','last');  % only need to do once per session
        
        im_std = std(sar_res, 0, 'all', 'omitnan');
        
        nexttile
        imshow(rescale(mag2db(sar_res))); title(filt + "-" + size + "std: " + im_std)
        
    end
    exportgraphics(fig, "result_linear\"+filt+".png", 'Resolution', 700);
    close(fig)
end
