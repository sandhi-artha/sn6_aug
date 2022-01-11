% read rasters from orientation 1
DATA_DIR = 'D:\Projects\python\dataset\spacenet6-challenge\AOI_11_Rotterdam\SAR-Intensity';
BASE_SAVE_DIR = 'D:\Projects\python\dataset\sn6_aug\filter';
filenames1 = readlines("orient1_fp.txt", "EmptyLineRule","skip");


filt_list = ["elee" "frost" "gmap"];
win_list = [3 5 7];

for i = 1:length(filenames1)
    fprintf('tile %d from %d\n', i, length(filenames1))
    fp = append(DATA_DIR, '\', filenames1(i));

    % read raster
    [sar,r] = readgeoraster(fp);
    % read just HH and convert to linear scale
    sar_hh = to_lin(sar(:,:,1));

    % loop through all selected filters
    for j = 1:length(filt_list)
        %loop through all window_size
        for k = 1:length(win_list)
            filt = filt_list(j);
            win = win_list(k);
            sar_res = sarimg_despeckling_filter(filt, sar_hh, 1, win, 2);
            
            % silence p-regen warning, only need to do once per session
            warning('off','last');
            
            % handle when output is below 1 (nodata value in linear)
            % prevents inf when converting back to dB if there's value 0
            sar_res(sar_res < 1) = 1;
            
            % convert to dB again and rescale to range of uint8
            sar_res = uint8(rescale(to_db(sar_res), 0, 255));

            % save to filter folder with same filename as original
            save_fn = strrep(filenames1(i), '.tif','.png');
            save_dir = append(BASE_SAVE_DIR, '\', filt, '\', num2str(win));
            status = mkdir(save_dir);
            save_fp = append(save_dir, '\', save_fn);
            imwrite(sar_res, save_fp)
        end
    end
end


function y_db = to_db(y)
    y_db = 10*log10(y);
end

function y = to_lin(y_db)
    y = 10.^(y_db/10);
end