%% update_scale_model: function description
function [model, scores_fs] = update_scale_model(scores_fs_base, scores_fs_scaled, model, params)
    assert(size(scores_fs_base,3)==1); % no scale estimation
    scores_fs = scores_fs_base;
    scores_base = sample_fs(scores_fs_base);
    max_score_base = max(scores_base(:)) % base ECO score

    scores_fs_scaled = cell2mat(scores_fs_scaled);
    scores_scaled = sample_fs(scores_fs_scaled); % cifft2
    max_scores = max(reshape(scores_scaled,[],model.n_scales),[],1);
	% max_score_zero_rotate = max_scores(model.zero_ind);
    [max_score, max_ind] = max(max_scores);

	% if max_score > model.rotate_alpha * max_score_zero_rotate % we find a better angle
	% 	if model.dynamic_angles
	% 		% update current_ang
	% 		model.current_ang = model.current_ang + model.angs(max_ind);
 %            fprintf('ROTATE (DYNAMIC): CURRENT ANGLE %d\n', model.current_ang);
	% 	else
	% 		% use fixed angle
 %            model.current_ang = model.angs(max_ind);
	% 		fprintf('ROTATE (FIXED): CURRENT ANGLE %d\n', model.angs(max_ind));% update scores_fs
	% 	end
	% 	scores_fs = scores_fs_scaled(:,:,max_ind);
	% else % base response is large enough
	% 	if model.dynamic_angles
	% 		scores_fs = scores_fs_scaled(:,:,model.zero_ind);
	% 	else
	% 		scores_fs = scores_fs_base;
	% 	end
	% end

    if 1
        figure(88); plot(model.scales, max_scores); ylim([0,max_score]);
        figure(89); 
        SR = ceil(sqrt(model.n_scales));
        SC = ceil(model.n_scales/SR);
        for i = 1:model.n_scales
            subplot(SR,SC,i);
            imagesc(ifftshift(scores_scaled(:,:,i))); colorbar;
            title(sprintf('%.3f, %.4f',model.scales(i),max_scores(i)));
        end
        pause;
    end

    % if params.RS_debug
    %     figure(86);
    %     SR = ceil(sqrt(model.n_angs));
    %     SC = ceil(model.n_angs/SR);
    %     for i = 1:model.n_angs
    %         subplot(SR,SC,i);
    %         imagesc(ifftshift(scores_scaled(:,:,i))); colorbar;
    %         title(sprintf('%d, %.4f',model.angs(i),max_scores(i)));
    %     end
    %     figure(87);
    %     plot(model.angs,max_scores);
    %     pause;
    % end
end