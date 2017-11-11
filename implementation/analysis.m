function results = analysis(params)
% assert(params.augment == 0);
%% prepare results
results = struct();
results.angles = 0:2:180; % range from -90 to 90 degress
results.robustness = zeros(size(results.angles));
results.represness = zeros(size(results.angles));
results.response = zeros(size(results.angles));
% results.augmresp = zeros(size(results.angles));

%% Initialization just like tracker.m
% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [seq, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;
% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Correct max number of samples
params.nSamples = min(params.nSamples, seq.num_frames);

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

assert(is_color_image==true);

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    warning('mexResize error');
    % warning('ECO:tracker', 'Error when using the mexResize function. Using Matlab''s interpolation function instead, which is slower.\nTry to run the compile script in "external_libs/mexResize/".\n\nThe error was:\n%s', getReport(err));
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2]; % for testing
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');

assert(length(features)==1);

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
feature_dim = feature_info.dim;
num_feature_blocks = length(feature_dim);

% Get feature specific parameters
feature_params = init_feature_params(features, feature_info);
feature_extract_info = get_feature_extract_info(features);

% Set the sample feature dimension
if params.use_projection_matrix
    sample_dim = feature_params.compressed_dim;
else
    sample_dim = feature_dim;
end

sample_dim_cell = reshape(mat2cell(sample_dim, ones(1,num_feature_blocks)),1,1,num_feature_blocks);

% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% Number of Fourier coefficients to save for each filter layer. This will
% be an odd number.
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size.
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% How much each feature block has to be padded to the obtain output_sz
pad_sz = cellfun(@(filter_sz) (output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false);

% Compute the Fourier series indices and their transposes
ky = cellfun(@(sz) (-ceil((sz(1) - 1)/2) : floor((sz(1) - 1)/2))', filter_sz_cell, 'uniformoutput', false);
kx = cellfun(@(sz) -ceil((sz(2) - 1)/2) : 0, filter_sz_cell, 'uniformoutput', false);

% construct the Gaussian label function using Poisson formula
sig_y = sqrt(prod(floor(base_target_sz))) * params.output_sigma_factor * (output_sz ./ img_support_sz);
yf_y = cellfun(@(ky) single(sqrt(2*pi) * sig_y(1) / output_sz(1) * exp(-2 * (pi * sig_y(1) * ky / output_sz(1)).^2)), ky, 'uniformoutput', false);
yf_x = cellfun(@(kx) single(sqrt(2*pi) * sig_y(2) / output_sz(2) * exp(-2 * (pi * sig_y(2) * kx / output_sz(2)).^2)), kx, 'uniformoutput', false);
yf = cellfun(@(yf_y, yf_x) cast(yf_y * yf_x, 'like', params.data_type), yf_y, yf_x, 'uniformoutput', false);

% construct cosine window
cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

% Compute Fourier series of interpolation function
[interp1_fs, interp2_fs] = cellfun(@(sz) get_interp_fourier(sz, params), filter_sz_cell, 'uniformoutput', false);

% Get the reg_window_edge parameter
reg_window_edge = {};
for k = 1:length(features)
    if isfield(features{k}.fparams, 'reg_window_edge')
        reg_window_edge = cat(3, reg_window_edge, permute(num2cell(features{k}.fparams.reg_window_edge(:)), [2 3 1]));
    else
        reg_window_edge = cat(3, reg_window_edge, cell(1, 1, length(features{k}.fparams.nDim)));
    end
end

% Construct spatial regularization filter
reg_filter = cellfun(@(reg_window_edge) get_reg_filter(img_support_sz, base_target_sz, params, reg_window_edge), reg_window_edge, 'uniformoutput', false);

% Compute the energy of the filter (used for preconditioner)
reg_energy = cellfun(@(reg_filter) real(reg_filter(:)' * reg_filter(:)), reg_filter, 'uniformoutput', false);

if params.use_scale_filter
    [nScales, scale_step, scaleFactors, scale_filter, params] = init_scale_filter(params);
else
    % Use the translation filter to estimate the scale.
    nScales = params.number_of_scales;
    scale_step = params.scale_step;
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
end

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

% Set conjugate gradient uptions
init_CG_opts.CG_use_FR = true;
init_CG_opts.tol = 1e-6;
init_CG_opts.CG_standard_alpha = true;
init_CG_opts.debug = params.debug;
CG_opts.CG_use_FR = params.CG_use_FR;
CG_opts.tol = 1e-6;
CG_opts.CG_standard_alpha = params.CG_standard_alpha;
CG_opts.debug = params.debug;
if params.CG_forgetting_rate == Inf || params.learning_rate >= 1
    CG_opts.init_forget_factor = 0;
else
    CG_opts.init_forget_factor = (1-params.learning_rate)^params.CG_forgetting_rate;
end

seq.time = 0;

% Initialize and allocate
prior_weights = zeros(params.nSamples,1, 'single');
sample_weights = cast(prior_weights, 'like', params.data_type);
samplesf = cell(1, 1, num_feature_blocks);

augment_factor = 1;
if params.augment, augment_factor = params.augment_factor; end;

if params.use_gpu
    % In the GPU version, the data is stored in a more normal way since we
    % dont have to use mtimesx.
    for k = 1:num_feature_blocks
        samplesf{k} = zeros(filter_sz(k,1),(filter_sz(k,2)+1)/2,sample_dim(k)*augment_factor,params.nSamples, 'like', params.data_type_complex);
    end
else
    for k = 1:num_feature_blocks
        samplesf{k} = zeros(params.nSamples,sample_dim(k)*augment_factor,filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
    end
end

% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);

% Distance matrix stores the square of the euclidean distance between each pair of
% samples. Initialise it to inf
distance_matrix = inf(params.nSamples, 'single');

% Kernel matrix, used to update distance matrix
gram_matrix = inf(params.nSamples, 'single');

latest_ind = [];
frames_since_last_train = inf;
num_training_samples = 0;

% Find the minimum allowed sample weight. Samples are discarded if their weights become lower 
params.minimum_sample_weight = params.learning_rate*(1-params.learning_rate)^(2*params.nSamples);

res_norms = [];
residuals_pcg = [];

%% tracking loop
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end

    %% TRAINING
    % =========
    % Extract image region for training sample
    sample_pos = round(pos);
    sample_scale = currentScaleFactor;
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
    
    % Do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);

    %% We do robost abd repres analysis here
    if ~params.augment
    norm = inner_product_filter(xlw,xlw);
    for i = 1:length(results.angles)
        ang = results.angles(i);
        xlr = extract_rotated_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info, ang);
        xlrw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xlr, cos_window, 'uniformoutput', false);
        xlrw2 = cellfun(@(x) imrotate(x, ang,'bicubic','crop'), xlw, 'uniformoutput', false);

%         diff = cellfun(@minus, xlrw, xlw, 'uniformoutput', false);
        diff2 = cellfun(@minus, xlrw2, xlrw, 'uniformoutput', false);
%         ip = inner_product_filter(diff,diff) / norm;
        ip2 = inner_product_filter(diff2,diff2) / norm;
%         results.robustness(i) = ip;
        results.represness(i) = ip2;
        %fprintf('%d,%f,%f\n',ang,ip,ip2);
    end
    end
    fprintf('ANA: 1/2\n');
    return;

    % Shift feature to [-T/2, T/2] domain
    xlw = cellfun(@(x) fftshift(fftshift(x,1),2), xlw, 'uniformoutput', false);
    
    % Compute the fourier series
    xlf = cellfun(@cfft2, xlw, 'uniformoutput', false);
    
    % Interpolate features to the continuous domain
    xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
    
    if params.augment
        % cpu:[H,W,C], cat along channel(3rd) dim -> [H,W,3C]
        xlf = augment_sample(xlf, params);
        % xlf = cellfun(@(xf) cat(3,xf,rotatef(xf,-params.augment_angle),rotatef(xf,params.augment_angle)),xlf,'uniformoutput',false);
    end
    
    % New sample to be added
    xlf = compact_fourier_coeff(xlf);
    
    % Shift sample
    shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
    xlf = shift_sample(xlf, shift_samp, kx, ky);
    
    % Init the projection matrix
    projection_matrix = init_projection_matrix(xl, sample_dim, params);
    
    % Project sample
    if params.augment
        xlf = cellfun(@(xf) reshape(xf,size(xf,1),size(xf,2),[],augment_factor),xlf,'uniformoutput',false);
        xlf_proj = project_sample(xlf, projection_matrix);
        xlf_proj = cellfun(@(xf) reshape(xf,size(xf,1),size(xf,2),[]),xlf_proj,'uniformoutput',false);
    else
        xlf_proj = project_sample(xlf, projection_matrix);
    end
    
    clear xlw
    % The permuted sample is only needed for the CPU implementation
    if ~params.use_gpu
        xlf_proj_perm = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj, 'uniformoutput', false);
    end
        
    if params.use_sample_merge
        % Update the samplesf to include the new sample. The distance
        % matrix, kernel matrix and prior weight are also updated
        if params.use_gpu
            [merged_sample, new_sample, merged_sample_id, new_sample_id, distance_matrix, gram_matrix, prior_weights] = ...
                update_sample_space_model_gpu(samplesf, xlf_proj, distance_matrix, gram_matrix, prior_weights,...
                num_training_samples,params);
        else
            [merged_sample, new_sample, merged_sample_id, new_sample_id, distance_matrix, gram_matrix, prior_weights] = ...
                update_sample_space_model(samplesf, xlf_proj_perm, distance_matrix, gram_matrix, prior_weights,...
                num_training_samples,params);
        end
        
        if num_training_samples < params.nSamples
            num_training_samples = num_training_samples + 1;
        end
    else
        % Do the traditional adding of a training sample and weight update
        % of C-COT
        [prior_weights, replace_ind] = update_prior_weights(prior_weights, gather(sample_weights), latest_ind, seq.frame, params);
        latest_ind = replace_ind;
        
        merged_sample_id = 0;
        new_sample_id = replace_ind;
        if params.use_gpu
            new_sample = xlf_proj;
        else
            new_sample = xlf_proj_perm;
        end
    end
    
    if seq.frame > 1 && params.learning_rate > 0 || seq.frame == 1 && ~params.update_projection_matrix
        % Insert the new training sample
        for k = 1:num_feature_blocks
            if params.use_gpu
                if merged_sample_id > 0
                    samplesf{k}(:,:,:,merged_sample_id) = merged_sample{k};
                end
                if new_sample_id > 0
                    samplesf{k}(:,:,:,new_sample_id) = new_sample{k};
                end
            else
                if merged_sample_id > 0
                    samplesf{k}(merged_sample_id,:,:,:) = merged_sample{k};
                end
                if new_sample_id > 0
                    samplesf{k}(new_sample_id,:,:,:) = new_sample{k};
                end
            end
        end
    end

    sample_weights = cast(prior_weights, 'like', params.data_type);
           
    train_tracker = (seq.frame < params.skip_after_frame) || (frames_since_last_train >= params.train_gap);
    
    if train_tracker     
        % Used for preconditioning
        if params.augment
            % use only the 1st silce (of 3rd dim) as xlf_proj, since we augment before
            new_sample_energy = cellfun(@(xlf,sd) abs(xlf(:,:,1:sd) .* conj(xlf(:,:,1:sd))),xlf_proj,sample_dim_cell,'uniformoutput',false);
        else
            new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf_proj, 'uniformoutput', false);
        end
        
        if seq.frame == 1
            % Initialize stuff for the filter learning
            
            % Initialize Conjugate Gradient parameters
            sample_energy = new_sample_energy;
            CG_state = [];
            
            if params.update_projection_matrix
                % Number of CG iterations per GN iteration 
                init_CG_opts.maxit = ceil(params.init_CG_iter / params.init_GN_iter);
            
                hf = cell(2,1,num_feature_blocks);
                proj_energy = cellfun(@(P, yf) 2*sum(abs(yf(:)).^2) / sum(feature_dim) * ones(size(P), 'like', params.data_type), projection_matrix, yf, 'uniformoutput', false);
            else
                CG_opts.maxit = params.init_CG_iter; % Number of initial iterations if projection matrix is not updated
            
                hf = cell(1,1,num_feature_blocks);
            end
            
            % Initialize the filter with zeros
            for k = 1:num_feature_blocks
                hf{1,1,k} = zeros([filter_sz(k,1) (filter_sz(k,2)+1)/2 sample_dim(k)], 'like', params.data_type_complex);
            end
        else
            CG_opts.maxit = params.CG_iter;
            
            % Update the approximate average sample energy using the learning
            % rate. This is only used to construct the preconditioner.
            sample_energy = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, sample_energy, new_sample_energy, 'uniformoutput', false);
        end
        
        % Do training
        if seq.frame == 1 && params.update_projection_matrix
            if params.debug
                projection_matrix_init = projection_matrix;
            end
            
            % Initial Gauss-Newton optimization of the filter and
            % projection matrix.
            if params.use_gpu
                [hf, projection_matrix, res_norms] = train_joint_gpu(hf, projection_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, params, init_CG_opts);
            else
                [hf, projection_matrix, res_norms] = train_joint(hf, projection_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, params, init_CG_opts);
            end
            
            % Re-project and insert training sample
            xlf_proj = project_sample(xlf, projection_matrix);
            if params.augment
                xlf_proj = cellfun(@(xf) reshape(xf,size(xf,1),size(xf,2),[]), xlf_proj, 'uniformoutput', false);
            end
            for k = 1:num_feature_blocks
                if params.use_gpu
                    samplesf{k}(:,:,:,1) = xlf_proj{k};
                else
                    samplesf{k}(1,:,:,:) = permute(xlf_proj{k}, [4 3 1 2]);
                end
            end
            
            % Update the gram matrix since the sample has changed
            if strcmp(params.distance_matrix_update_type, 'exact')
                % Find the norm of the reprojected sample
                new_train_sample_norm =  0;
                
                for k = 1:num_feature_blocks
                    new_train_sample_norm = new_train_sample_norm + real(gather(2*(xlf_proj{k}(:)' * xlf_proj{k}(:))));% - reshape(xlf_proj{k}(:,end,:,:), [], 1, 1)' * reshape(xlf_proj{k}(:,end,:,:), [], 1, 1));
                end
                
                gram_matrix(1,1) = new_train_sample_norm;
            end
            
            if params.debug
                norm_proj_mat_init = sqrt(sum(cellfun(@(P) gather(norm(P(:))^2), projection_matrix_init)));
                norm_proj_mat = sqrt(sum(cellfun(@(P) gather(norm(P(:))^2), projection_matrix)));
                norm_proj_mat_change = sqrt(sum(cellfun(@(P,P2) gather(norm(P(:) - P2(:))^2), projection_matrix_init, projection_matrix)));
                fprintf('Norm init: %f, Norm final: %f, Matrix change: %f\n', norm_proj_mat_init, norm_proj_mat, norm_proj_mat_change / norm_proj_mat_init);
            end
        else
            % Do Conjugate gradient optimization of the filter
            if params.use_gpu
                [hf, res_norms, CG_state] = train_filter_gpu(hf, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, params, CG_opts, CG_state);
            else
                [hf, res_norms, CG_state] = train_filter(hf, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, params, CG_opts, CG_state);
            end
        end
        
        % Reconstruct the full Fourier series
        hf_full = full_fourier_coeff(hf);
        frames_since_last_train = 0;
    else
        frames_since_last_train = frames_since_last_train+1;
    end

    %% analysis here
    sample_pos = round(pos);
    det_sample_pos = sample_pos;
    sample_scale = currentScaleFactor*scaleFactors;
    for i = 1:length(results.angles)
        break;
        ang = results.angles(i);

        % Extract features at multiple resolutions
        xt = extract_rotated_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info, ang);
                    
        % Project sample
        xt_proj = project_sample(xt, projection_matrix);
        
        % Do windowing of features
        xt_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj, cos_window, 'uniformoutput', false);
        
        % Shift feature to [-T/2, T/2] domain
        xt_proj = cellfun(@(x) fftshift(fftshift(x,1),2), xt_proj, 'uniformoutput', false);
        
        % Compute the fourier series
        xtf_proj = cellfun(@cfft2, xt_proj, 'uniformoutput', false);
        
        % Interpolate features to the continuous domain
        xtf_proj = interpolate_dft(xtf_proj, interp1_fs, interp2_fs);
        
        % Compute convolution for each feature block in the Fourier domain
        % and the sum over all blocks.
        scores_fs_feat{k1} = sum(bsxfun(@times, hf_full{k1}, xtf_proj{k1}), 3);
        scores_fs_sum = scores_fs_feat{k1};
        for k = block_inds
            scores_fs_feat{k} = sum(bsxfun(@times, hf_full{k}, xtf_proj{k}), 3);
            scores_fs_sum(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) = ...
                scores_fs_sum(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) + ...
                scores_fs_feat{k};
        end
        
        % Also sum over all feature blocks.
        % Gives the fourier coefficients of the convolution response.
        scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
        response = sample_fs(scores_fs);
        results.response(i) = response(1,1);
        %fprintf('%d, %f\n', ang, response(1,1));
    end
    fprintf('ANA: 2/2\n');
    return;
end

end
