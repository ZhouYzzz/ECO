%% update_rotate_model: function description
function [model, scores_fs] = update_rotate_model(scores_fs_base, scores_fs_rotated, model, params)
    assert(size(scores_fs_base,3)==1); % no scale estimation
    scores_fs = scores_fs_base;
    scores_base = sample_fs(scores_fs_base);
    max_score_base = max(scores_base(:)); % base ECO score

    scores_fs_rotated = cell2mat(scores_fs_rotated);
    scores_rotated = sample_fs(scores_fs_rotated); % cifft2
    max_scores = max(reshape(scores_rotated,[],model.n_angs),[],1);

    max_scores = max_scores .* model.transfer_alpha;

	max_score_zero_rotate = max_scores(model.zero_ind);

    %%%%%%%%%%% !!!!!!!!!!!!
    max_score_zero_rotate = max_score_base;
    
    [max_score, max_ind] = max(max_scores);

	if max_score > model.rotate_alpha * max_score_zero_rotate % we find a better angle
		if model.dynamic_angles
			% update current_ang
			model.current_ang = model.current_ang + model.angs(max_ind);
            fprintf('ROTATE (DYNAMIC): CURRENT ANGLE %d\n', model.current_ang);
		else
			% use fixed angle
            model.current_ang = model.angs(max_ind);
			fprintf('ROTATE (FIXED): CURRENT ANGLE %d\n', model.angs(max_ind));% update scores_fs
		end
		scores_fs = scores_fs_rotated(:,:,max_ind);
	else % base response is large enough
		if model.dynamic_angles
			scores_fs = scores_fs_rotated(:,:,model.zero_ind);
		else
			scores_fs = scores_fs_base;
		end
	end

    if params.RS_debug
        % figure(86);
        % SR = ceil(sqrt(model.n_angs));
        % SC = ceil(model.n_angs/SR);
        % for i = 1:model.n_angs
        %     subplot(SR,SC,i);
        %     imagesc(ifftshift(scores_rotated(:,:,i))); colorbar;
        %     title(sprintf('%d, %.4f',model.angs(i),max_scores(i)));
        % end
        % figure(87);
        % plot(model.current_ang + model.angs,max_scores);
        % pause;
    end
end