
% This demo script runs the ECO tracker with hand-crafted features on the
% included "Crossing" video.

% Add paths
setup_paths();

% Load video information
%video_path = 'sequences/Crossing';
video_name = 'MotorRolling';
video_path = fullfile(get_global_variable('ECO_dataset_path'),'OTB50',video_name);

[seq, ground_truth] = load_video_info(video_path);

% Run ECO
results = testing_ECO_HC_RS(seq);

% perform better : Surfer (final 10 frames), MotorRolling

% same : ClifBar, CarScale

% perform worse : Tiger2 

%% Surfer
% params.RS = 1;
% params.RS_debug = 0;
% params.n_angs = 9;                     % num of angles for searching, even prefered
% params.ang_interval = 5;
% params.rotate_alpha = 1.05;
% params.dynamic_angles = 1;
% params.use_rotated_sample = 0;


% Singer2, Matrix