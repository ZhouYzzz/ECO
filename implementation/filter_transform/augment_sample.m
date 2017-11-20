function [ xlf_augmented ] = augment_sample( xlf, params )
%AUGMENT_SAMPLE Summary of this function goes here
%   Detailed explanation goes here
augment_angles_cell = mat2cell(params.augment_angle', ones(1,params.augment_factor));
augment_flags_cell = mat2cell(params.augment_flags', ones(1,length(xlf)));
fun_augment_sample = @(xf, flag) cellfun(@(angle) rotatef(xf, angle, flag), augment_angles_cell,'uniformoutput', false);
xlf_augmented = cell(size(xlf));

for k = 1:length(xlf)
	sample_cell = fun_augment_sample(xlf{k}, augment_flags_cell{k});
	xlf_augmented{k} = cat(3, sample_cell{:});
end

end
