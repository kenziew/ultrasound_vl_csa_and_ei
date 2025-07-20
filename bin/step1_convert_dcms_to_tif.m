%% Options 
clear; clc; close all

% Set Study Name and Data Paths (parent directory to each subject folder containing
% dicoms)
study_name = 'demo';
main_data_path = 'C:\Users\Kenzie\Documents\MATLAB_2023b\AI_Ultrasound\demo_dcms_to_tif';
path_to_dcms = strcat(main_data_path, '\raw_dcms');
all_subjects = 1;
if ~all_subjects, subject_ids = {'DCM_UM05', 'SUB03'}; end

% Get subject list
if ~all_subjects && (~exist('subject_ids','var') || isempty(subject_ids))
    error('Error: Must choose subject IDs or enable all_subjects.');
end

if all_subjects
    d = dir(path_to_dcms); d = d([d.isdir] & ~ismember({d.name}, {'.','..'}));
    subject_paths = fullfile(path_to_dcms, {d.name});
else
    paths = fullfile(path_to_dcms, subject_ids);
    exists = cellfun(@(p) exist(p,'dir')==7, paths);
    subject_paths = paths(exists);
end

if isempty(subject_paths)
    disp('No subjects found. Quitting.');
    return
else
    fprintf('Found %d subject(s).\n', numel(subject_paths));
end

%% Main
hSub = waitbar(0, 'Processing subjects...');
for i = 1:numel(subject_paths)
    % Subject ID
    [~, subject_id] = fileparts(subject_paths{i});

    % Get valid files
    files = dir2(subject_paths{i}, '/s');
    files = files(~contains(files, {'DICOMDIR', '.csv', 'HTM', 'SVG', 'TXT', 'JPG'}));

    % Output directory
    out_path = fullfile(main_data_path, 'img_files', subject_id);
    if ~exist(out_path, 'dir'), mkdir(out_path); end

    % File-level progress bar
    hFile = waitbar(0, sprintf('Processing files for %s...', subject_id));
    
    for ii = 1:numel(files)
        waitbar(ii / numel(files), hFile, sprintf('Subject %s: File %d of %d', subject_id, ii, numel(files)));

        [~, fname] = fileparts(files{ii});
        info = dicominfo(files{ii});
        img = dicomread(files{ii});
        sub_table = table({subject_id}, {fname}, 'VariableNames', {'subjectID', 'filename'});
        im_empty = isempty(img);
        img_table = detect_resolution(info);

        if isempty(img_table)
            fprintf('file %s is empty; skipping\n', fname);
            continue;
        end

        if ~im_empty
            count = 1;
            out_fname = sprintf('%s_img%d', subject_id, count);
            out_file = fullfile(out_path, [out_fname, '.tif']);
            while exist(out_file, 'file')
                count = count + 1;
                out_fname = sprintf('%s_img%d', subject_id, count);
                out_file = fullfile(out_path, [out_fname, '.tif']);
            end

            imwrite(img, out_file);

            matfile_name = strrep(out_file, '.tif', '.mat');
            f = matfile(matfile_name, "Writable", true);
            if ~isempty(img_table.resx{1}) && ~isempty(img_table.resy{1})
                f.dicom_file_name = fname;
                f.resx = cell2mat(img_table.resx);
                f.resy = cell2mat(img_table.resy);
            end
        else
            out_dir = {};
            out_fname = {};
            notes = 'Error. Image is empty. No data was copied. Skipped';
        end

        % Log
        new_folder = table({out_path}, 'VariableNames', {'new_folder'});
        new_name = table({out_fname}, 'VariableNames', {'new_filename'});
        log = [sub_table, new_folder, new_name, img_table];

        if exist('log_out', 'var')
            log_out = [log_out; log];
        else
            log_out = log;
        end
    end
    close(hFile);
    waitbar(i / numel(subject_paths), hSub, sprintf('Processed %d of %d subjects', i, numel(subject_paths)));
end
close(hSub);

writetable(log_out, fullfile(main_data_path,  strcat(study_name, '_data_log.csv')));
fprintf('Done!\n')