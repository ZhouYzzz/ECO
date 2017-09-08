%% track_rotate_model: function description
function [model, scores_fs_cell] = track_scale_model(xf,hf,model,k1,block_inds,pad_sz,params)
    if params.debug,tic;end;
    % construct rotated filters
    for i = 1:model.n_scales
        model.scaled_filters_cell{i} = cellfun(@(hf) scale_filter(hf, model.scales(i)), hf, 'uniformoutput', false);
    end
    % compute scores of different angles
    scores_fs_cell = cellfun(@(hf_scl) compute_scoref(hf_scl,xf,k1,block_inds,pad_sz), model.scaled_filters_cell, 'uniformoutput', false);
    if params.debug,t=toc;fprintf('track_scale_model: takes %.2f ms\n',t*1e3);end;
end

%% compute_scoref: function description
function [scores_fs] = compute_scoref(hf_full,xtf_proj,k1,block_inds,pad_sz)
    persistent scores_fs_feat; % declear as local variable
    scores_fs_feat{k1} = sum(bsxfun(@times, hf_full{k1}, xtf_proj{k1}), 3);
    scores_fs_sum = scores_fs_feat{k1};
    for k = block_inds
        scores_fs_feat{k} = sum(bsxfun(@times, hf_full{k}, xtf_proj{k}), 3);
        scores_fs_sum(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) = ...
            scores_fs_sum(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) + ...
            scores_fs_feat{k};
    end
    scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
end
