% read rasters from orientation 1
load('fps1_5folds.mat')

DATA_DIR = 'D:\Projects\python\dataset\sn6_aug\hh_crop';
BASE_SAVE_DIR = 'D:\Projects\python\dataset\sn6_aug\filter_crop';
% filenames1 = readlines("orient1_fp.txt", "EmptyLineRule","skip");

folds = ["fold0", "fold1", "fold4"];
folds_cell = {fold0; fold1; fold4};

filt_list = ["elee" "frost" "gmap"];
win_list = [3 5 7];

for f = 1:length(folds)
    fold = folds_cell{f};  % to access cell's content use {}
    fold_name = folds(f);
    fprintf('fold: %s\n', fold_name)

    for i = 1:length(fold)
        fprintf('tile %d from %d\n', i, length(fold))
        % to equal space, some filenames have trailing space -> strtrim
        fp = append(DATA_DIR, '\', fold_name, '\', strtrim(fold(i,:)));
        % read cropped sar_hh tif and convert to linear
        sar_hh = to_lin(imread(fp));

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
                sar_res = single(to_db(sar_res));
%                 sar_res = uint8(rescale(to_db(sar_res), 0, 255));

                % save to filter folder with same filename as original
                save_fn = strrep(strtrim(fold(i,:)), '.tif','.mat');
                save_dir = append(BASE_SAVE_DIR, '\', filt, '\', num2str(win));
                status = mkdir(save_dir);
                save_fp = append(save_dir, '\', save_fn);
                save(save_fp, 'sar_res')
            end
        end
    end
end





function y_db = to_db(y)
    y_db = 10*log10(y);
end

function y = to_lin(y_db)
    y = 10.^(y_db/10);
end