% Copy this template configuration file to your VOT workspace.
% Enter the full path to the ECO repository root folder.

RACF_repo_path = ########

tracker_label = 'RACF';
% VOT-2016
tracker_command = generate_matlab_command('benchmark_tracker_wrapper(''RACF'', ''VOT2016_DEEP_settings'', true)', {[RACF_repo_path '/VOT_integration/benchmark_wrapper']});
% VOT-2017
%tracker_command = generate_matlab_command('benchmark_tracker_wrapper(''RACF'', ''VOT2017_DEEP_settings'', true)', {[RACF_repo_path '/VOT_integration/benchmark_wrapper']});

tracker_interpreter = 'matlab';

% both ECO and RACF are deterministic
tracker_metadata.deterministic = true;
