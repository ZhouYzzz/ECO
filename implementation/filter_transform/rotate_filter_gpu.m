function [ hf_rotated ] = rotate_filter( hf, ang )
	% hf = imrotate(hf, ang, 'bicubic', 'crop'); % bicubic, bilinear, nearest
	if ~mod(ang,360),hf_rotated = hf;return;end;
	% directly use mex function to accelarate
	outputSize = [size(hf,1) size(hf,2)];
	method = 'bicubic';
	hf_rotated = complex(imrotategpumex(real(hf),ang,method,outputSize),...
						 imrotategpumex(imag(hf),ang,method,outputSize));
end
