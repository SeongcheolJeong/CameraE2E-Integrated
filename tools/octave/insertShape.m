function out = insertShape(image, shape, params, varargin)
% Minimal insertShape shim for headless Octave parity cases.
%
% Supported shapes:
%   insertShape(image, 'filled-circle', [x y radius], 'Color', color, 'Opacity', alpha)
%   insertShape(image, 'filled-rectangle', [x y width height], 'Color', color, 'Opacity', alpha)
%   insertShape(image, 'line', [x1 y1 x2 y2 ...], 'Color', color, 'LineWidth', width)

out = local_to_rgb(double(image));
shape_name = lower(strrep(strrep(strtrim(shape), ' ', ''), '-', ''));
color = [1, 1, 1];
opacity = 1.0;
line_width = 1.0;

ii = 1;
while ii <= numel(varargin)
    key = lower(strtrim(varargin{ii}));
    if strcmp(key, 'color')
        color = local_parse_color(varargin{ii + 1});
        ii = ii + 2;
    elseif strcmp(key, 'opacity')
        opacity = double(varargin{ii + 1});
        ii = ii + 2;
    elseif strcmp(key, 'linewidth')
        line_width = double(varargin{ii + 1});
        ii = ii + 2;
    else
        error('Unsupported insertShape option: %s', varargin{ii});
    end
end

if strcmp(shape_name, 'filledcircle')
    if numel(params) ~= 3
        error('filled-circle expects [x y radius].');
    end
    out = local_draw_circle(out, params(1), params(2), params(3), color, opacity);
elseif strcmp(shape_name, 'filledrectangle')
    if numel(params) ~= 4
        error('filled-rectangle expects [x y width height].');
    end
    out = local_draw_rectangle(out, params(1), params(2), params(3), params(4), color, opacity);
elseif strcmp(shape_name, 'line')
    if mod(numel(params), 2) ~= 0 || numel(params) < 4
        error('line expects [x1 y1 x2 y2 ...].');
    end
    out = local_draw_line(out, params, line_width, color);
else
    error('Unsupported insertShape shape: %s', shape);
end
end

function out = local_to_rgb(image)
if ndims(image) == 2
    out = repmat(image, [1, 1, 3]);
elseif ndims(image) == 3 && size(image, 3) == 3
    out = image;
else
    error('insertShape shim only supports grayscale or RGB images.');
end
end

function color = local_parse_color(value)
if isnumeric(value)
    color = double(value(:)');
    if numel(color) == 1
        color = repmat(color, 1, 3);
    end
    if numel(color) ~= 3
        error('Numeric insertShape color must be scalar or RGB triplet.');
    end
    return;
end

switch lower(strtrim(char(value)))
    case 'white'
        color = [1, 1, 1];
    case 'green'
        color = [0, 1, 0];
    case 'blue'
        color = [0, 0, 1];
    case 'yellow'
        color = [1, 1, 0];
    case 'magenta'
        color = [1, 0, 1];
    case 'red'
        color = [1, 0, 0];
    case 'cyan'
        color = [0, 1, 1];
    case 'black'
        color = [0, 0, 0];
    otherwise
        error('Unsupported insertShape color: %s', char(value));
end
end

function out = local_draw_circle(image, x, y, radius, color, opacity)
[rows, cols, ~] = size(image);
[xx, yy] = meshgrid(1:cols, 1:rows);
mask = (xx - double(x)).^2 + (yy - double(y)).^2 <= double(radius).^2;
out = local_apply_mask(image, mask, color, opacity);
end

function out = local_draw_rectangle(image, x, y, width, height, color, opacity)
[rows, cols, ~] = size(image);
x0 = max(round(double(x)), 1);
y0 = max(round(double(y)), 1);
x1 = min(round(double(x + width)), cols);
y1 = min(round(double(y + height)), rows);
if x1 < x0
    x1 = min(x0, cols);
end
if y1 < y0
    y1 = min(y0, rows);
end

mask = false(rows, cols);
mask(y0:y1, x0:x1) = true;
out = local_apply_mask(image, mask, color, opacity);
end

function out = local_draw_line(image, params, line_width, color)
points = reshape(double(params), 2, []).';
radius = max(double(line_width), 1) / 2.0;
out = image;
for idx = 1:(size(points, 1) - 1)
    p0 = points(idx, :);
    p1 = points(idx + 1, :);
    steps = max(ceil(max(abs(p1 - p0)) * 2.0), 1);
    xs = linspace(p0(1), p1(1), steps + 1);
    ys = linspace(p0(2), p1(2), steps + 1);
    for sample_idx = 1:numel(xs)
        out = local_draw_circle(out, xs(sample_idx), ys(sample_idx), radius, color, 1.0);
    end
end
end

function out = local_apply_mask(image, mask, color, opacity)
out = image;
alpha = min(max(double(opacity), 0.0), 1.0);
for channel = 1:3
    plane = out(:, :, channel);
    plane(mask) = (1.0 - alpha) * plane(mask) + alpha * color(channel);
    out(:, :, channel) = plane;
end
end
