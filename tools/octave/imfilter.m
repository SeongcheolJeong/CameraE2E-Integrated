function out = imfilter(image, kernel, varargin)
% Minimal imfilter shim for headless Octave parity cases.
%
% Supported call patterns:
%   imfilter(image, kernel)
%   imfilter(image, kernel, 'replicate')
%   imfilter(image, kernel, 0, 'same', 'conv')

image = double(image);
kernel = double(kernel);

boundary = 0;
shape = 'same';

for ii = 1:numel(varargin)
    arg = varargin{ii};
    if ischar(arg)
        normalized = lower(strtrim(arg));
        if strcmp(normalized, 'replicate')
            boundary = 'replicate';
        elseif any(strcmp(normalized, {'same', 'full', 'valid'}))
            shape = normalized;
        elseif any(strcmp(normalized, {'conv', 'corr'}))
            % Only convolution mode is needed for the parity cases.
        else
            error('Unsupported imfilter option: %s', arg);
        end
    elseif isnumeric(arg) && isscalar(arg)
        boundary = double(arg);
    else
        error('Unsupported imfilter argument.');
    end
end

if ~strcmp(shape, 'same')
    error('This shim only supports ''same'' output size.');
end

kernel = rot90(kernel, 2);
pad_rows = floor(size(kernel, 1) / 2);
pad_cols = floor(size(kernel, 2) / 2);

if ischar(boundary) && strcmp(boundary, 'replicate')
    padded = local_pad_replicate(image, pad_rows, pad_cols);
else
    padded = local_pad_constant(image, pad_rows, pad_cols, boundary);
end

out = conv2(padded, kernel, 'valid');
end

function padded = local_pad_constant(image, pad_rows, pad_cols, value)
[n_rows, n_cols] = size(image);
padded = value * ones(n_rows + 2 * pad_rows, n_cols + 2 * pad_cols);
padded(pad_rows + 1:pad_rows + n_rows, pad_cols + 1:pad_cols + n_cols) = image;
end

function padded = local_pad_replicate(image, pad_rows, pad_cols)
[n_rows, n_cols] = size(image);
padded = zeros(n_rows + 2 * pad_rows, n_cols + 2 * pad_cols);
padded(pad_rows + 1:pad_rows + n_rows, pad_cols + 1:pad_cols + n_cols) = image;

if pad_rows > 0
    padded(1:pad_rows, pad_cols + 1:pad_cols + n_cols) = repmat(image(1, :), pad_rows, 1);
    padded(pad_rows + n_rows + 1:end, pad_cols + 1:pad_cols + n_cols) = repmat(image(end, :), pad_rows, 1);
end
if pad_cols > 0
    padded(:, 1:pad_cols) = repmat(padded(:, pad_cols + 1), 1, pad_cols);
    padded(:, pad_cols + n_cols + 1:end) = repmat(padded(:, pad_cols + n_cols), 1, pad_cols);
end
end
