function [ hf_rotated ] = rotatef( hf, ang )
%ROTATEF Rotate in Fourier domain
%   directly use mex functions for acceleration, gpu supported
%   `hf` should be single precision when using gpu mode
%
%   NOTE: some higher matlab version may not have the internal mex files
%   Here are some possible solutions
%
%   1. we provide some mexfiles (marked with `.backup`) in `private`, however they may not compatible with your matlab version.
%
%   2. search for your matlab folders for the required mexfiles, you may not find them if the image processing toolbox is not installed.
%
%   3. directly use a unified wrapped function `imrotate` for both gpu and cpu versions


if ~mod(ang,360), hf_rotated = hf; return; end;
outputSize = [size(hf,1) size(hf,2)];
method = 'bicubic';

% Use internal rotate functions for accelerations
if isa(hf,'gpuArray')

    % MATLAB lower versions use imrotategpumex
    hf_rotated = complex(imrotategpumex(real(hf),ang,method,outputSize),...
                         imrotategpumex(imag(hf),ang,method,outputSize));

    % MATLAB higher than R2017a use different routine, uncomment the following lines.
    %
    % hf_rotated = complex(images.internal.gpu.imrotate(real(hf), ang, method, outputSize),...
    %                      images.internal.gpu.imrotate(imag(hf), ang, method, outputSize)); % R2017a

else
    % the cpu interfaces are the same
    hf_rotated = complex(imrotatemex(real(hf),ang,outputSize,method),...
                         imrotatemex(imag(hf),ang,outputSize,method));

end

% Uncomment the following lines to use a higher level `imrotate` function instead, which is slower
%
% hf_rotated = imrotate(hf, 'crop', 'bicubic');

end
