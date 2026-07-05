function r = poissrnd(lambda, varargin)
% Minimal poissrnd shim for headless Octave parity exports.

if nargin < 1
    error('poissrnd requires at least the lambda input');
end

if ~isempty(varargin)
    sz = cellfun(@double, varargin);
    if isscalar(lambda)
        lambda = lambda .* ones(sz);
    else
        lambda = reshape(lambda, sz);
    end
end

if exist('randp', 'builtin') || exist('randp', 'file')
    r = randp(lambda);
    return;
end

lambda = double(lambda);
r = zeros(size(lambda));

largeMask = lambda > 30;
if any(largeMask(:))
    approx = round(lambda(largeMask) + sqrt(lambda(largeMask)) .* randn(sum(largeMask(:)), 1));
    approx(approx < 0) = 0;
    r(largeMask) = approx;
end

smallIdx = find(~largeMask);
for ii = 1:numel(smallIdx)
    lam = lambda(smallIdx(ii));
    if lam <= 0
        r(smallIdx(ii)) = 0;
        continue;
    end
    limit = exp(-lam);
    k = 0;
    p = 1;
    while p > limit
        k = k + 1;
        p = p * rand();
    end
    r(smallIdx(ii)) = k - 1;
end

end
