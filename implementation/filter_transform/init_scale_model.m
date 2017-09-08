%% init_scale_model: function description
function [model] = init_scale_model(filter_sz, params)
	filter_sz = filter_sz{1}(1);
    model.author = 'Yizhuang Zhou';
    model.n_scales = params.n_scales;
    % model.scales = (filter_sz + ((1:params.n_scales)-ceil(params.n_scales/2))*2) / filter_sz;
    model.scales = (filter_sz - (1:params.n_scales)*2 + 2) / filter_sz;

    model.scaled_filters_cell = cell(1,1,params.n_scales);
end