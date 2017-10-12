function [ hf_rotated ] = rotatef( hf, ang )
%ROTATEF Rotate in Fourier domain
%   directly use mex functions for acceleration, gpu supported
%	`hf` should be single precision when using gpu mode
	if ~mod(ang,360),hf_rotated = hf;return;end;
	outputSize = [size(hf,1) size(hf,2)];
	method = 'bicubic';
	if isa(hf,'gpuArray')
		hf_rotated = complex(imrotategpumex(real(hf),ang,method,outputSize),...
							 imrotategpumex(imag(hf),ang,method,outputSize));
	else
		hf_rotated = complex(imrotatemex(real(hf),ang,outputSize,method),...
							 imrotatemex(imag(hf),ang,outputSize,method));
	end
end
