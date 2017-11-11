function [seq, ground_truth] = load_video_info_vot(video_path)

ground_truth = dlmread([video_path '/groundtruth.txt']);

seq.format = 'otb';
seq.len = size(ground_truth, 1);
sf = 61;
seq.init_rect = ground_truth(sf,:);
seq.init_rect = poly2rect(seq.init_rect);

img_path = [video_path '/'];

if exist([img_path num2str(1, '%08i.png')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%08i.png']);
elseif exist([img_path num2str(1, '%08i.jpg')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%08i.jpg']);
elseif exist([img_path num2str(1, '%08i.bmp')], 'file'),
    img_files = num2str((1:seq.len)', [img_path '%08i.bmp']);
else
    error('No image files to load.')
end

seq.s_frames = cellstr(img_files);
seq.s_frames = seq.s_frames(sf:end);
end

function [init_rect] = poly2rect(init_region)
    factor = 3;
    if numel(init_region) > 4
        % Init with an axis aligned bounding box with correct area and center
        % coordinate
        cx = mean(init_region(1:2:end));
        cy = mean(init_region(2:2:end));
        x1 = min(init_region(1:2:end));
        x2 = max(init_region(1:2:end));
        y1 = min(init_region(2:2:end));
        y2 = max(init_region(2:2:end));
        A1 = norm(init_region(1:2) - init_region(3:4)) * norm(init_region(3:4) - init_region(5:6));
%         A1 = norm(cross([init_region(1:2) - init_region(3:4), 0], [init_region(3:4) - init_region(5:6), 0]));
        A2 = (x2 - x1) * (y2 - y1);
        s = sqrt(A1/A2);
        w = s * (x2 - x1) + 1;
        h = s * (y2 - y1) + 1;
        if h/w > factor, w = h/factor;end;
        if w/h > factor, h = w/factor;end;
    else
        cx = init_region(1) + (init_region(3) - 0)/2;
        cy = init_region(2) + (init_region(4) - 0)/2;
        w = init_region(3);
        h = init_region(4);
    end
%     init_c = [cy cx];
%     init_sz = 1 * [h w];
    init_rect = [cx - w/2, cy - h/2, w, h];
end

% function rect = poly2rect(poly)
% xs = poly(1:2:end);
% ys = poly(2:2:end);
% rect = zeros(1,4);
% lx = min(xs,[],2);
% uy = min(ys,[],2);
% rx = max(xs,[],2);
% dy = max(ys,[],2);
% w = rx - lx;
% h = dy - uy;
% cx = (lx + rx)/2;
% cy = (uy + dy)/2;
% w = w * 1.;
% h = h * 1.;
% rect(:,1) = cx - w/2;
% rect(:,2) = cy - h/2;
% rect(:,3) = w;
% rect(:,4) = h;
% end