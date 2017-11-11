function [ hf_rotated ] = rotatef( hf, ang, flag )
%ROTATEF Rotate in Fourier domain
%   directly use mex functions for acceleration, gpu supported
%	`hf` should be single precision when using gpu mode
	if ~mod(ang,360)||~flag,hf_rotated = hf;return;end;
	outputSize = [size(hf,1) size(hf,2)];
	method = 'bicubic';
	if isa(hf,'gpuArray')
% 		hf_rotated = complex(imrotategpumex(real(hf),ang,method,outputSize),...
% 							 imrotategpumex(imag(hf),ang,method,outputSize));
        hf_rotated = complex(images.internal.gpu.imrotate(real(hf), ang, method, outputSize),...
                    images.internal.gpu.imrotate(imag(hf), ang, method, outputSize)); % R2017a
	else
		hf_rotated = complex(imrotatemex(real(hf),ang,outputSize,method),...
							 imrotatemex(imag(hf),ang,outputSize,method));
	end
end
