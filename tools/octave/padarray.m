function out = padarray(in, padsize, padval, direction)
% Minimal padarray shim for Octave parity runs.
%
% Supports the constant-padding subset used by the curated ISETCam cases:
%   padarray(A, padsize)
%   padarray(A, padsize, padval)
%   padarray(A, padsize, padval, 'both'|'pre'|'post')

if nargin < 2
    error('padarray requires at least two arguments');
end
if nargin < 3 || isempty(padval)
    padval = 0;
end
if nargin < 4 || isempty(direction)
    direction = 'both';
end

padsize = double(padsize(:)');
nd = ndims(in);
if numel(padsize) < nd
    padsize = [padsize, zeros(1, nd - numel(padsize))];
else
    padsize = padsize(1:nd);
end

switch lower(direction)
    case 'both'
        pre = padsize;
        post = padsize;
    case 'pre'
        pre = padsize;
        post = zeros(size(padsize));
    case 'post'
        pre = zeros(size(padsize));
        post = padsize;
    otherwise
        error('Unsupported padarray direction: %s', direction);
end

sz = size(in);
if numel(sz) < nd
    sz = [sz, ones(1, nd - numel(sz))];
end
out_sz = sz + pre + post;
out = repmat(cast(padval, class(in)), out_sz);

subs = cell(1, nd);
for ii = 1:nd
    start_idx = pre(ii) + 1;
    stop_idx = pre(ii) + sz(ii);
    subs{ii} = start_idx:stop_idx;
end
out(subs{:}) = in;
end
