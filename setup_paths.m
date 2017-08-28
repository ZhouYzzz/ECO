function setup_paths()

% Add the neccesary paths

[pathstr, name, ext] = fileparts(mfilename('fullpath'));

% Tracker implementation
addpath(genpath([pathstr '/implementation/']));

% Runfiles
addpath([pathstr '/runfiles/']);

% Utilities
addpath([pathstr '/utils/']);

% The feature extraction
addpath(genpath([pathstr '/feature_extraction/']));

% Matconvnet
addpath([pathstr '/external_libs/matconvnet/matlab']);
addpath([pathstr '/external_libs/matconvnet/matlab/mex/']);
addpath([pathstr '/external_libs/matconvnet/matlab/simplenn']);

% PDollar toolbox
addpath(genpath([pathstr '/external_libs/pdollar_toolbox/channels']));

% Mtimesx
addpath([pathstr '/external_libs/mtimesx/']);

% mexResize
addpath([pathstr '/external_libs/mexResize/']);

% Setup global variables
set_global_variable('ECO_root_path', pathstr);
set_global_variable('ECO_dataset_path', fullfile(pathstr,'..','..','Data'));
