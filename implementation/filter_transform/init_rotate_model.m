%% init_rotate_model: function description
function [model] = init_rotate_model(params)
    model.author = 'Yizhuang Zhou';
    model.n_angs = params.n_angs;
    model.ang_interval = params.ang_interval;
    model.current_ang = 0;
    model.rotate_alpha = params.rotate_alpha;
    model.dynamic_angles = params.dynamic_angles;


    model.rotated_filters_cell = cell(1,1,params.n_angs);
    model.scores_fs_cell = cell(1,1,params.n_angs);
    model.angs = ((1:params.n_angs) - round(params.n_angs/2)).*params.ang_interval;
    % model.angs = ((1:params.n_angs)-1).*params.ang_interval;
    model.zero_ind = ceil(model.n_angs/2);

    model.transfer_alpha = 1 + params.transfer_alpha * (1-hann(params.n_angs)');
    % model.zero_ind = 1;
    % assert(model.angs(model.zero_ind)==0);
end