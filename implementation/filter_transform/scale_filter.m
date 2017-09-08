function [ hf_scaled ] = scale_filter( hf, scale )
	if (scale==1),hf_scaled = hf;return;end;
    sz = size(hf,1);
	sz_o = sz*scale; % should be integer
	% directly use mex function to accelarate
    hf_scaled = complex(imResampleMex(real(hf),sz_o,sz_o,1 / scale^2),...
                        imResampleMex(imag(hf),sz_o,sz_o,1 / scale^2));
    dif = abs(size(hf_scaled,1)-sz)/2;
    if scale > 1
        hf_scaled = hf_scaled(1+dif:end-dif,1+dif:end-dif,:);
    else
        hf_scaled = padarray(hf_scaled,[dif,dif]);
    end
    % assert(size(hf_scaled,1)==sz);
end
