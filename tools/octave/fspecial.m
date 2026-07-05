function h = fspecial(type, varargin)
% Minimal fspecial shim for Octave parity runs.
%
% Supports:
%   fspecial('gaussian', hsize, sigma)
%   fspecial('gauss', hsize, sigma)
%   fspecial('average')
%   fspecial('average', hsize)

if nargin < 1
    error('fspecial requires a filter type');
end

type = lower(type);
switch type
    case {'gaussian', 'gauss'}
        if numel(varargin) < 1 || isempty(varargin{1})
            hsize = [3 3];
        else
            hsize = varargin{1};
        end
        if isscalar(hsize)
            hsize = [hsize hsize];
        end
        if numel(varargin) < 2 || isempty(varargin{2})
            sigma = 0.5;
        else
            sigma = varargin{2};
        end
        rows = double(hsize(1));
        cols = double(hsize(2));
        [x, y] = meshgrid(1:cols, 1:rows);
        cx = (cols + 1) / 2;
        cy = (rows + 1) / 2;
        h = exp(-((x - cx).^2 + (y - cy).^2) / (2 * sigma^2));
        h = h / sum(h(:));

    case 'average'
        if numel(varargin) < 1 || isempty(varargin{1})
            hsize = [3 3];
        else
            hsize = varargin{1};
        end
        if isscalar(hsize)
            hsize = [hsize hsize];
        end
        h = ones(double(hsize(1)), double(hsize(2)));
        h = h / sum(h(:));

    otherwise
        error('Unsupported fspecial type: %s', type);
end
end
