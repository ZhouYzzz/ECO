
% This demo script runs the ECO tracker with deep features on the
% included "Crossing" video.

% Add paths
setup_paths();

% Load video information
video_path = 'sequences/Crossing';
video_path = '/home/zhouyz/Development/OTB/sequences/MotorRolling';
[seq, ground_truth] = load_video_info(video_path);

% Run ECO
figure;
results = ANA_DEEP_norm1_aug(seq);
plot(results.angles, results.response, 'r-', 'lineWidth', 2); hold on;
results = ANA_DEEP_norm1_45(seq);
plot(results.angles, results.response, 'g-', 'lineWidth', 2); hold on;
