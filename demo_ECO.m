
% This demo script runs the ECO tracker with deep features on the
% included "Crossing" video.

% Add paths
setup_paths();

% Load video information
video_path = 'sequences/Crossing';
video_path = '/home/zhouyz/Development/OTB/sequences/Singer2';

% video_path = '/home/zhouyz/Development/OTB/sequences/MotorRolling';
video_path = '/home/zhouyz/Development/OTB/sequences/Skiing';
% video_path = '/home/zhouyz/Development/OTB/sequences/Jogging.1';
% video_path = '/home/zhouyz/Development/OTB/sequences/Girl2';
% video_path = '/home/zhouyz/Development/OTB/sequences/Coupon';
% video_path = '/home/zhouyz/Development/OTB/sequences/Basketball';
% video_path = '/home/zhouyz/Development/OTB/sequences/Ironman';
% video_path = '/home/zhouyz/Development/OTB/sequences/Suv';
% video_path = '/home/zhouyz/Development/OTB/sequences/Soccer';
% video_path = '/home/zhouyz/Development/OTB/sequences/Box';
video_path = '/home/zhouyz/Development/OTB/sequences/Biker';

% video_path = '/home/zhouyz/Development/vot-workspace-vot2016-final/sequences/bmx';
% video_path = '/home/zhouyz/Development/vot-workspace-vot2016-final/sequences/gymnastics3';
% video_path = '/home/zhouyz/Development/vot-workspace-vot2016-final/sequences/gymnastics2';

% video_path = '/home/zhouyz/Development/vot-workspace-vot2016-final/sequences/leaves';
% video_path = '/home/zhouyz/Development/vot-workspace-vot2016-final/sequences/motocross2';
% video_path = '/home/zhouyz/Development/vot-workspace-vot2016-final/sequences/nature';
% video_path = '/home/zhouyz/Development/vot-workspace-vot2016-final/sequences/book';
% video_path = '/home/zhouyz/Development/vot-workspace-vot2016-final/sequences/soccer2';
% video_path = '/home/zhouyz/Development/vot-workspace-vot2016-final/sequences/fish2';

[seq, ground_truth] = load_video_info(video_path);
% [seq, ground_truth] = load_video_info_vot(video_path);

% Run ECO
%results = testing_ECO(seq);
% results = OTB_DEEP_tune_29OCT_M4(seq);
results = VOT2016_DEEP_tune_29OCT(seq);
% results = testing_ECO_gpu(seq);