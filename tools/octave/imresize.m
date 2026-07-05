function out = imresize(in, target, method)
% Minimal imresize shim for Octave parity runs.
%
% Supports:
%   imresize(A, [rows cols])
%   imresize(A, [rows cols], 'nearest'|'linear'|'bilinear')

if nargin < 2
    error('imresize requires at least two arguments');
end
if nargin < 3 || isempty(method)
    method = 'linear';
end
method = lower(method);
if strcmp(method, 'bilinear')
    method = 'linear';
end

if isscalar(target)
    target = max(round(size(in, 1:2) * target), 1);
end
target = double(target(:)');
if numel(target) ~= 2
    error('Unsupported imresize target specification');
end

src_rows = size(in, 1);
src_cols = size(in, 2);
dst_rows = max(round(target(1)), 1);
dst_cols = max(round(target(2)), 1);

if src_rows == dst_rows && src_cols == dst_cols
    out = in;
    return;
end

row_positions = linspace(1, src_rows, dst_rows);
col_positions = linspace(1, src_cols, dst_cols);
[src_x, src_y] = meshgrid(1:src_cols, 1:src_rows);
[dst_x, dst_y] = meshgrid(col_positions, row_positions);

if ndims(in) == 2
    out = interp2(src_x, src_y, double(in), dst_x, dst_y, method);
else
    out = zeros(dst_rows, dst_cols, size(in, 3), class(in));
    for ii = 1:size(in, 3)
        out(:, :, ii) = cast(interp2(src_x, src_y, double(in(:, :, ii)), dst_x, dst_y, method), class(in));
    end
    return;
end

out = cast(out, class(in));
end
