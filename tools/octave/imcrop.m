function out = imcrop(img, rect)
% Minimal imcrop shim for numeric arrays using [x y width height].

if nargin < 2
    error('imcrop requires an image and a crop rect');
end

rect = round(rect(:)');
if numel(rect) ~= 4
    error('Crop rect must be [x y width height]');
end

x = rect(1);
y = rect(2);
w = rect(3);
h = rect(4);

row_start = max(y, 1);
col_start = max(x, 1);
row_end = min(y + h, size(img, 1));
col_end = min(x + w, size(img, 2));

subs = repmat({':'}, 1, ndims(img));
subs{1} = row_start:row_end;
subs{2} = col_start:col_end;
out = img(subs{:});

end
