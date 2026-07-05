function out = imrotate(im, angle, varargin)
% Minimal imrotate shim for curated Codex parity cases.

method = 'linear';
shape = 'loose';
if ~isempty(varargin)
    if numel(varargin) >= 1 && ~isempty(varargin{1})
        candidate = lower(char(varargin{1}));
        if strcmp(candidate, 'bilinear')
            method = 'linear';
        elseif strcmp(candidate, 'nearest')
            method = 'nearest';
        else
            method = candidate;
        end
    end
    if numel(varargin) >= 2 && ~isempty(varargin{2})
        shape = lower(char(varargin{2}));
    end
end

if abs(angle) < eps
    out = im;
    return;
end

[rows, cols] = size(im);
cx = (cols + 1) / 2;
cy = (rows + 1) / 2;
theta = angle * pi / 180;

corners = [
    1, 1;
    cols, 1;
    cols, rows;
    1, rows
];

rot = [
    cos(theta), -sin(theta);
    sin(theta),  cos(theta)
];

centered = corners - [cx, cy];
rotated = centered * rot';
rotated = rotated + [cx, cy];

xMin = floor(min(rotated(:, 1)));
xMax = ceil(max(rotated(:, 1)));
yMin = floor(min(rotated(:, 2)));
yMax = ceil(max(rotated(:, 2)));

[xOut, yOut] = meshgrid(xMin:xMax, yMin:yMax);
xShift = xOut - cx;
yShift = yOut - cy;

xIn =  cos(theta) * xShift + sin(theta) * yShift + cx;
yIn = -sin(theta) * xShift + cos(theta) * yShift + cy;

out = interp2(1:cols, 1:rows, double(im), xIn, yIn, method, 0);
if strcmp(shape, 'crop')
    rowStart = floor((size(out, 1) - rows) / 2) + 1;
    colStart = floor((size(out, 2) - cols) / 2) + 1;
    rowStop = rowStart + rows - 1;
    colStop = colStart + cols - 1;
    out = out(max(rowStart, 1):min(rowStop, size(out, 1)), max(colStart, 1):min(colStop, size(out, 2)));
end
if isa(im, 'single')
    out = single(out);
end
end
