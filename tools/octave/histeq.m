function out = histeq(im, n)
% Minimal histeq shim for Octave parity runs.

if nargin < 1
    error('histeq requires an input image');
end
if nargin < 2 || isempty(n)
    n = 64;
end

im = double(im);
imin = min(im(:));
imax = max(im(:));
if !(imax > imin)
    out = zeros(size(im));
    return;
end

scaled = (im - imin) / (imax - imin);
edges = linspace(0, 1, double(n) + 1);
counts = histc(scaled(:), edges);
counts(end - 1) = counts(end - 1) + counts(end);
counts = counts(1:end - 1);
cdf = cumsum(counts);
if cdf(end) <= 0
    out = zeros(size(im));
    return;
end
cdf = cdf / cdf(end);
centers = (edges(1:end - 1) + edges(2:end)) / 2;
out = interp1(centers, cdf, scaled, 'linear', 'extrap');
out = min(max(out, 0), 1);
end
